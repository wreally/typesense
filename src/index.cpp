#include "index.h"

#include <numeric>
#include <chrono>
#include <set>
#include <unordered_map>
#include <random>
#include <array_utils.h>
#include <match_score.h>
#include <string_utils.h>
#include <art.h>
#include <tokenizer.h>
#include <s2/s2point.h>
#include <s2/s2latlng.h>
#include <s2/s2region_term_indexer.h>
#include <s2/s2cap.h>
#include <s2/s2earth.h>
#include <s2/s2loop.h>
#include <s2/s2builder.h>
#include <posting.h>
#include <thread_local_vars.h>
#include <unordered_set>
#include <or_iterator.h>
#include <timsort.hpp>
#include "logger.h"
#include <collection_manager.h>

#define RETURN_CIRCUIT_BREAKER if((std::chrono::duration_cast<std::chrono::microseconds>( \
                  std::chrono::system_clock::now().time_since_epoch()).count() - search_begin_us) > search_stop_us) { \
                    search_cutoff = true; \
                    return ;\
            }

#define BREAK_CIRCUIT_BREAKER if((std::chrono::duration_cast<std::chrono::microseconds>( \
                 std::chrono::system_clock::now().time_since_epoch()).count() - search_begin_us) > search_stop_us) { \
                    search_cutoff = true; \
                    break;\
                }

spp::sparse_hash_map<uint32_t, int64_t> Index::text_match_sentinel_value;
spp::sparse_hash_map<uint32_t, int64_t> Index::seq_id_sentinel_value;
spp::sparse_hash_map<uint32_t, int64_t> Index::eval_sentinel_value;
spp::sparse_hash_map<uint32_t, int64_t> Index::geo_sentinel_value;
spp::sparse_hash_map<uint32_t, int64_t> Index::str_sentinel_value;
spp::sparse_hash_map<uint32_t, int64_t> Index::vector_distance_sentinel_value;

struct token_posting_t {
    uint32_t token_id;
    const posting_list_t::iterator_t& posting;

    token_posting_t(uint32_t token_id, const posting_list_t::iterator_t& posting):
            token_id(token_id), posting(posting) {

    }
};

Index::Index(const std::string& name, const uint32_t collection_id, const Store* store,
             SynonymIndex* synonym_index, ThreadPool* thread_pool,
             const tsl::htrie_map<char, field> & search_schema,
             const std::vector<char>& symbols_to_index, const std::vector<char>& token_separators):
        name(name), collection_id(collection_id), store(store), synonym_index(synonym_index), thread_pool(thread_pool),
        search_schema(search_schema),
        seq_ids(new id_list_t(256)), symbols_to_index(symbols_to_index), token_separators(token_separators) {

    for(const auto& a_field: search_schema) {
        if(!a_field.index) {
            continue;
        }

        if(a_field.num_dim > 0) {
            auto hnsw_index = new hnsw_index_t(a_field.num_dim, 1024, a_field.vec_dist);
            vector_index.emplace(a_field.name, hnsw_index);
            continue;
        }

        if(a_field.is_string()) {
            art_tree *t = new art_tree;
            art_tree_init(t);
            search_index.emplace(a_field.name, t);
        } else if(a_field.is_geopoint()) {
            auto field_geo_index = new spp::sparse_hash_map<std::string, std::vector<uint32_t>>();
            geopoint_index.emplace(a_field.name, field_geo_index);

            if(!a_field.is_single_geopoint()) {
                spp::sparse_hash_map<uint32_t, int64_t*> * doc_to_geos = new spp::sparse_hash_map<uint32_t, int64_t*>();
                geo_array_index.emplace(a_field.name, doc_to_geos);
            }
        } else {
            num_tree_t* num_tree = new num_tree_t;
            numerical_index.emplace(a_field.name, num_tree);
        }

        if(a_field.sort) {
            if(a_field.type == field_types::STRING) {
                adi_tree_t* tree = new adi_tree_t();
                str_sort_index.emplace(a_field.name, tree);
            } else if(a_field.type != field_types::GEOPOINT_ARRAY) {
                spp::sparse_hash_map<uint32_t, int64_t> * doc_to_score = new spp::sparse_hash_map<uint32_t, int64_t>();
                sort_index.emplace(a_field.name, doc_to_score);
            }
        }

        if(a_field.facet) {
            initialize_facet_indexes(a_field);
        }

        // initialize for non-string facet fields
        if(a_field.facet && !a_field.is_string()) {
            art_tree *ft = new art_tree;
            art_tree_init(ft);
            search_index.emplace(a_field.faceted_name(), ft);
        }

        if(a_field.infix) {
            array_mapped_infix_t infix_sets(ARRAY_INFIX_DIM);

            for(auto& infix_set: infix_sets) {
                infix_set = new tsl::htrie_set<char>();
            }

            infix_index.emplace(a_field.name, infix_sets);
        }
    }

    num_documents = 0;
}

Index::~Index() {
    std::unique_lock lock(mutex);

    for(auto & name_tree: search_index) {
        art_tree_destroy(name_tree.second);
        delete name_tree.second;
        name_tree.second = nullptr;
    }

    search_index.clear();

    for(auto & name_index: geopoint_index) {
        delete name_index.second;
        name_index.second = nullptr;
    }

    geopoint_index.clear();

    for(auto& name_index: geo_array_index) {
        for(auto& kv: *name_index.second) {
            delete [] kv.second;
        }

        delete name_index.second;
        name_index.second = nullptr;
    }

    geo_array_index.clear();

    for(auto & name_tree: numerical_index) {
        delete name_tree.second;
        name_tree.second = nullptr;
    }

    numerical_index.clear();

    for(auto & name_map: sort_index) {
        delete name_map.second;
        name_map.second = nullptr;
    }

    sort_index.clear();

    for(auto& kv: infix_index) {
        for(auto& infix_set: kv.second) {
            delete infix_set;
            infix_set = nullptr;
        }
    }

    infix_index.clear();

    for(auto& name_tree: str_sort_index) {
        delete name_tree.second;
        name_tree.second = nullptr;
    }

    str_sort_index.clear();

    for(auto& field_name_facet_map_array: facet_index_v3) {
        for(auto& facet_map: field_name_facet_map_array.second) {
            delete facet_map;
            facet_map = nullptr;
        }
    }

    facet_index_v3.clear();

    for(auto& field_name_facet_map_array: single_val_facet_index_v3) {
        for(auto& facet_map: field_name_facet_map_array.second) {
            delete facet_map;
            facet_map = nullptr;
        }
    }

    single_val_facet_index_v3.clear();

    delete seq_ids;

    for(auto& vec_index_kv: vector_index) {
        delete vec_index_kv.second;
    }
}

int64_t Index::get_points_from_doc(const nlohmann::json &document, const std::string & default_sorting_field) {
    int64_t points = 0;

    if(document[default_sorting_field].is_number_float()) {
        // serialize float to an integer and reverse the inverted range
        float n = document[default_sorting_field];
        memcpy(&points, &n, sizeof(int32_t));
        points ^= ((points >> (std::numeric_limits<int32_t>::digits - 1)) | INT32_MIN);
        points = -1 * (INT32_MAX - points);
    } else if(document[default_sorting_field].is_string()) {
        // not much value in supporting default sorting field as string, so we will just dummy it out
        points = 0;
    } else {
        points = document[default_sorting_field].is_boolean() ? int64_t(document[default_sorting_field].get<bool>()) :
                 document[default_sorting_field].get<int64_t>();
    }

    return points;
}

int64_t Index::float_to_int64_t(float f) {
    // https://stackoverflow.com/questions/60530255/convert-float-to-int64-t-while-preserving-ordering
    int32_t i;
    memcpy(&i, &f, sizeof i);
    if (i < 0) {
        i ^= INT32_MAX;
    }
    return i;
}

float Index::int64_t_to_float(int64_t n) {
    int32_t i = (int32_t) n;

    if(i < 0) {
        i ^= INT32_MAX;
    }

    float f;
    memcpy(&f, &i, sizeof f);
    return f;
}

void Index::compute_token_offsets_facets(index_record& record,
                                         const tsl::htrie_map<char, field>& search_schema,
                                         const std::vector<char>& local_token_separators,
                                         const std::vector<char>& local_symbols_to_index) {

    const auto& document = record.doc;

    for(const auto& the_field: search_schema) {
        const std::string& field_name = the_field.name;
        if(document.count(field_name) == 0 || !the_field.index) {
            continue;
        }

        offsets_facet_hashes_t offset_facet_hashes;

        bool is_facet = search_schema.at(field_name).facet;

        // non-string, non-geo faceted field should be indexed as faceted string field as well
        if(the_field.facet && !the_field.is_string() && !the_field.is_geopoint()) {
            if(the_field.is_array()) {
                std::vector<std::string> strings;

                if(the_field.type == field_types::INT32_ARRAY) {
                    for(int32_t value: document[field_name]){
                        strings.push_back(std::to_string(value));
                    }
                } else if(the_field.type == field_types::INT64_ARRAY) {
                    for(int64_t value: document[field_name]){
                        strings.push_back(std::to_string(value));
                    }
                } else if(the_field.type == field_types::FLOAT_ARRAY) {
                    for(float value: document[field_name]){
                        strings.push_back(StringUtils::float_to_str(value));
                    }
                } else if(the_field.type == field_types::BOOL_ARRAY) {
                    for(bool value: document[field_name]){
                        strings.push_back(std::to_string(value));
                    }
                }

                tokenize_string_array_with_facets(strings, is_facet, the_field,
                                                  local_symbols_to_index, local_token_separators,
                                                  offset_facet_hashes.offsets, offset_facet_hashes.facet_hashes);
            } else {
                std::string text;

                if(the_field.type == field_types::INT32) {
                    text = std::to_string(document[field_name].get<int32_t>());
                } else if(the_field.type == field_types::INT64) {
                    text = std::to_string(document[field_name].get<int64_t>());
                } else if(the_field.type == field_types::FLOAT) {
                    text = StringUtils::float_to_str(document[field_name].get<float>());
                } else if(the_field.type == field_types::BOOL) {
                    text = std::to_string(document[field_name].get<bool>());
                }

                tokenize_string_with_facets(text, is_facet, the_field,
                                            local_symbols_to_index, local_token_separators,
                                            offset_facet_hashes.offsets, offset_facet_hashes.facet_hashes);
            }
        }

        if(the_field.is_string()) {

            if(the_field.type == field_types::STRING) {
                tokenize_string_with_facets(document[field_name], is_facet, the_field,
                                            local_symbols_to_index, local_token_separators,
                                            offset_facet_hashes.offsets, offset_facet_hashes.facet_hashes);
            } else {
                tokenize_string_array_with_facets(document[field_name], is_facet, the_field,
                                                  local_symbols_to_index, local_token_separators,
                                                  offset_facet_hashes.offsets, offset_facet_hashes.facet_hashes);
            }
        }

        if(!offset_facet_hashes.offsets.empty() || !offset_facet_hashes.facet_hashes.empty()) {
            record.field_index.emplace(field_name, std::move(offset_facet_hashes));
        }
    }
}

bool doc_contains_field(const nlohmann::json& doc, const field& a_field,
                        const tsl::htrie_map<char, field> & search_schema) {

    if(doc.count(a_field.name)) {
        return true;
    }

    // check for a nested field, e.g. `foo.bar.baz` indexed but `foo.bar` present in schema
    if(a_field.is_object()) {
        auto prefix_it = search_schema.equal_prefix_range(a_field.name);
        std::string nested_field_name;
        for(auto kv = prefix_it.first; kv != prefix_it.second; kv++) {
            kv.key(nested_field_name);
            bool is_child_field = (nested_field_name.size() > a_field.name.size() &&
                                   nested_field_name[a_field.name.size()] == '.');
            if(is_child_field && doc.count(nested_field_name) != 0) {
                return true;
            }
        }
    }

    return false;
}

bool validate_object_field(nlohmann::json& doc, const field& a_field) {
    auto field_it = doc.find(a_field.name);
    if(field_it != doc.end()) {
        if(a_field.type == field_types::OBJECT && doc[a_field.name].is_object()) {
            return true;
        } else if(a_field.type == field_types::OBJECT_ARRAY && doc[a_field.name].is_array()) {
            return true;
        }

        return false;
    }

    std::vector<std::string> field_parts;
    StringUtils::split(a_field.name, field_parts, ".");

    nlohmann::json* obj = &doc;
    bool has_array = false;

    for(auto& field_part: field_parts) {
        if(obj->is_array()) {
            has_array = true;

            if(obj->empty()) {
                return false;
            }

            obj = &obj->at(0);
            if(!obj->is_object()) {
                return false;
            }
        }

        auto obj_it = obj->find(field_part);
        if(obj_it == obj->end()) {
            return false;
        }

        obj = &obj_it.value();
    }

    LOG(INFO) << "obj: " << *obj;
    LOG(INFO) << "doc: " << doc;

    if(a_field.type == field_types::OBJECT && obj->is_object()) {
        return true;
    } else if(a_field.type == field_types::OBJECT_ARRAY && (obj->is_array() || (has_array && obj->is_object()))) {
        return true;
    }

    return false;
}

void Index::validate_and_preprocess(Index *index, std::vector<index_record>& iter_batch,
                                    const size_t batch_start_index, const size_t batch_size,
                                    const std::string& default_sorting_field,
                                    const tsl::htrie_map<char, field>& search_schema,
                                    const tsl::htrie_map<char, field>& embedding_fields,
                                    const std::string& fallback_field_type,
                                    const std::vector<char>& token_separators,
                                    const std::vector<char>& symbols_to_index,
                                    const bool do_validation, const bool generate_embeddings) {

    // runs in a partitioned thread
    std::vector<index_record*> records_to_embed;

    for(size_t i = 0; i < batch_size; i++) {
        index_record& index_rec = iter_batch[batch_start_index + i];

        try {
            if(!index_rec.indexed.ok()) {
                // some records could have been invalidated upstream
                continue;
            }

            if(index_rec.operation == DELETE) {
                continue;
            }

            if(do_validation) {
                Option<uint32_t> validation_op = validator_t::validate_index_in_memory(index_rec.doc, index_rec.seq_id,
                                                                          default_sorting_field,
                                                                          search_schema,
                                                                          embedding_fields,
                                                                          index_rec.operation,
                                                                          index_rec.is_update,
                                                                          fallback_field_type,
                                                                          index_rec.dirty_values, generate_embeddings);

                if(!validation_op.ok()) {
                    index_rec.index_failure(validation_op.code(), validation_op.error());
                    continue;
                }
            }

            if(index_rec.is_update) {
                // scrub string fields to reduce delete ops
                get_doc_changes(index_rec.operation, search_schema, index_rec.doc, index_rec.old_doc,
                                index_rec.new_doc, index_rec.del_doc);

                if(generate_embeddings) {
                    for(auto& field: index_rec.doc.items()) {
                        for(auto& embedding_field : embedding_fields) {
                            if(!embedding_field.embed[fields::from].is_null()) {
                                auto embed_from_vector = embedding_field.embed[fields::from].get<std::vector<std::string>>();
                                for(auto& embed_from: embed_from_vector) {
                                    if(embed_from == field.key()) {
                                        records_to_embed.push_back(&index_rec);
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                handle_doc_ops(search_schema, index_rec.doc, index_rec.old_doc);
                if(generate_embeddings) {
                    records_to_embed.push_back(&index_rec);
                }
            }

            compute_token_offsets_facets(index_rec, search_schema, token_separators, symbols_to_index);

            int64_t points = 0;

            if(index_rec.doc.count(default_sorting_field) == 0) {
                auto default_sorting_field_it = index->sort_index.find(default_sorting_field);
                if(default_sorting_field_it != index->sort_index.end()) {
                    auto seq_id_it = default_sorting_field_it->second->find(index_rec.seq_id);
                    if(seq_id_it != default_sorting_field_it->second->end()) {
                        points = seq_id_it->second;
                    } else {
                        points = INT64_MIN;
                    }
                } else {
                    points = INT64_MIN;
                }
            } else {
                points = get_points_from_doc(index_rec.doc, default_sorting_field);
            }

            index_rec.points = points;
            index_rec.index_success();
        } catch(const std::exception &e) {
            LOG(INFO) << "Error while validating document: " << e.what();
            index_rec.index_failure(400, e.what());
        }
    }
    if(generate_embeddings) {
        batch_embed_fields(records_to_embed, embedding_fields, search_schema);
    }
}

size_t Index::batch_memory_index(Index *index, std::vector<index_record>& iter_batch,
                                 const std::string & default_sorting_field,
                                 const tsl::htrie_map<char, field> & search_schema,
                                 const tsl::htrie_map<char, field> & embedding_fields,
                                 const std::string& fallback_field_type,
                                 const std::vector<char>& token_separators,
                                 const std::vector<char>& symbols_to_index,
                                 const bool do_validation, const bool generate_embeddings) {

    const size_t concurrency = 4;
    const size_t num_threads = std::min(concurrency, iter_batch.size());
    const size_t window_size = (num_threads == 0) ? 0 :
                               (iter_batch.size() + num_threads - 1) / num_threads;  // rounds up

    size_t num_indexed = 0;
    size_t num_processed = 0;
    std::mutex m_process;
    std::condition_variable cv_process;

    size_t num_queued = 0;
    size_t batch_index = 0;

    // local is need to propogate the thread local inside threads launched below
    auto local_write_log_index = write_log_index;

    for(size_t thread_id = 0; thread_id < num_threads && batch_index < iter_batch.size(); thread_id++) {
        size_t batch_len = window_size;

        if(batch_index + window_size > iter_batch.size()) {
            batch_len = iter_batch.size() - batch_index;
        }

        num_queued++;

        index->thread_pool->enqueue([&, batch_index, batch_len]() {
            write_log_index = local_write_log_index;
            validate_and_preprocess(index, iter_batch, batch_index, batch_len, default_sorting_field, search_schema,
                                    embedding_fields, fallback_field_type, token_separators, symbols_to_index, do_validation, generate_embeddings);

            std::unique_lock<std::mutex> lock(m_process);
            num_processed++;
            cv_process.notify_one();
        });

        batch_index += batch_len;
    }

    {
        std::unique_lock<std::mutex> lock_process(m_process);
        cv_process.wait(lock_process, [&](){ return num_processed == num_queued; });
    }

    std::unordered_set<std::string> found_fields;

    for(size_t i = 0; i < iter_batch.size(); i++) {
        auto& index_rec = iter_batch[i];

        if(!index_rec.indexed.ok()) {
            // some records could have been invalidated upstream
            continue;
        }

        if(index_rec.is_update) {
            index->remove(index_rec.seq_id, index_rec.del_doc, {}, index_rec.is_update);
        } else if(index_rec.indexed.ok()) {
            num_indexed++;
        }

        for(const auto& kv: index_rec.doc.items()) {
            found_fields.insert(kv.key());
        }
    }

    num_queued = num_processed = 0;

    for(const auto& field_name: found_fields) {
        //LOG(INFO) << "field name: " << field_name;
        if(field_name != "id" && search_schema.count(field_name) == 0) {
            continue;
        }

        num_queued++;

        index->thread_pool->enqueue([&]() {
            write_log_index = local_write_log_index;

            const field& f = (field_name == "id") ?
                             field("id", field_types::STRING, false) : search_schema.at(field_name);
            try {
                index->index_field_in_memory(f, iter_batch);
            } catch(std::exception& e) {
                LOG(ERROR) << "Unhandled Typesense error: " << e.what();
                for(auto& record: iter_batch) {
                    record.index_failure(500, "Unhandled Typesense error in index batch, check logs for details.");
                }
            }

            std::unique_lock<std::mutex> lock(m_process);
            num_processed++;
            cv_process.notify_one();
        });
    }

    {
        std::unique_lock<std::mutex> lock_process(m_process);
        cv_process.wait(lock_process, [&](){ return num_processed == num_queued; });
    }

    return num_indexed;
}

void Index::index_field_in_memory(const field& afield, std::vector<index_record>& iter_batch) {
    // indexes a given field of all documents in the batch

    if(afield.name == "id") {
        for(const auto& record: iter_batch) {
            if(!record.indexed.ok()) {
                // some records could have been invalidated upstream
                continue;
            }
            if(!record.is_update && record.indexed.ok()) {
                // for updates, the seq_id will already exist
                seq_ids->upsert(record.seq_id);
            }
        }

        return;
    }

    if(!afield.index) {
        return;
    }

    // We have to handle both these edge cases:
    // a) `afield` might not exist in the document (optional field)
    // b) `afield` value could be empty

    // non-geo faceted field should be indexed as faceted string field as well
    bool non_string_facet_field = (afield.facet && !afield.is_geopoint());

    if(afield.is_string() || non_string_facet_field) {
        std::unordered_map<std::string, std::vector<art_document>> token_to_doc_offsets;
        int64_t max_score = INT64_MIN;

        for(const auto& record: iter_batch) {
            if(!record.indexed.ok()) {
                // some records could have been invalidated upstream
                continue;
            }

            const auto& document = record.doc;
            const auto seq_id = record.seq_id;

            if(document.count(afield.name) == 0 || !record.indexed.ok()) {
                continue;
            }

            auto field_index_it = record.field_index.find(afield.name);
            if(field_index_it == record.field_index.end()) {
                continue;
            }

            if(afield.facet) {
                if(afield.is_array()) {
                    facet_hash_values_t fhashvalues;
                    fhashvalues.length = field_index_it->second.facet_hashes.size();
                    fhashvalues.hashes = new uint64_t[field_index_it->second.facet_hashes.size()];

                    for(size_t i  = 0; i < field_index_it->second.facet_hashes.size(); i++) {
                        fhashvalues.hashes[i] = field_index_it->second.facet_hashes[i];
                    }

                    auto& facet_dim_index = facet_index_v3[afield.name][seq_id % ARRAY_FACET_DIM];
                    if(facet_dim_index == nullptr) {
                        LOG(ERROR) << "Error, facet index not initialized for field " << afield.name;
                    } else {
                        facet_dim_index->emplace(seq_id, std::move(fhashvalues));
                    }
                } else {
                    uint64_t fhash;
                    fhash = field_index_it->second.facet_hashes[0];

                    auto& facet_dim_index = single_val_facet_index_v3[afield.name][seq_id % ARRAY_FACET_DIM];
                    if(facet_dim_index == nullptr) {
                        LOG(ERROR) << "Error, facet index not initialized for field " << afield.name;
                    } else {
                        facet_dim_index->emplace(seq_id, std::move(fhash));
                    }
                }
            }

            if(record.points > max_score) {
                max_score = record.points;
            }

            for(auto &token_offsets: field_index_it->second.offsets) {
                token_to_doc_offsets[token_offsets.first].emplace_back(seq_id, record.points, token_offsets.second);

                if(afield.infix) {
                    auto strhash = StringUtils::hash_wy(token_offsets.first.c_str(), token_offsets.first.size());
                    const auto& infix_sets = infix_index.at(afield.name);
                    infix_sets[strhash % 4]->insert(token_offsets.first);
                }
            }
        }

        auto tree_it = search_index.find(afield.faceted_name());
        if(tree_it == search_index.end()) {
            return;
        }

        art_tree *t = tree_it->second;

        for(auto& token_to_doc: token_to_doc_offsets) {
            const std::string& token = token_to_doc.first;
            std::vector<art_document>& documents = token_to_doc.second;

            const auto *key = (const unsigned char *) token.c_str();
            int key_len = (int) token.length() + 1;  // for the terminating \0 char

            //LOG(INFO) << "key: " << key << ", art_doc.id: " << art_doc.id;
            art_inserts(t, key, key_len, max_score, documents);
        }
    }

    if(!afield.is_string()) {
        if (afield.type == field_types::INT32) {
            auto num_tree = numerical_index.at(afield.name);
            iterate_and_index_numerical_field(iter_batch, afield, [&afield, num_tree]
                    (const index_record& record, uint32_t seq_id) {
                int32_t value = record.doc[afield.name].get<int32_t>();
                num_tree->insert(value, seq_id);
            });
        }

        else if(afield.type == field_types::INT64) {
            auto num_tree = numerical_index.at(afield.name);
            iterate_and_index_numerical_field(iter_batch, afield, [&afield, num_tree]
                    (const index_record& record, uint32_t seq_id) {
                int64_t value = record.doc[afield.name].get<int64_t>();
                num_tree->insert(value, seq_id);
            });
        }

        else if(afield.type == field_types::FLOAT) {
            auto num_tree = numerical_index.at(afield.name);
            iterate_and_index_numerical_field(iter_batch, afield, [&afield, num_tree]
                    (const index_record& record, uint32_t seq_id) {
                float fvalue = record.doc[afield.name].get<float>();
                int64_t value = float_to_int64_t(fvalue);
                num_tree->insert(value, seq_id);
            });
        } else if(afield.type == field_types::BOOL) {
            auto num_tree = numerical_index.at(afield.name);
            iterate_and_index_numerical_field(iter_batch, afield, [&afield, num_tree]
                    (const index_record& record, uint32_t seq_id) {
                bool value = record.doc[afield.name].get<bool>();
                num_tree->insert(value, seq_id);
            });
        } else if(afield.type == field_types::GEOPOINT || afield.type == field_types::GEOPOINT_ARRAY) {
            auto geo_index = geopoint_index.at(afield.name);

            iterate_and_index_numerical_field(iter_batch, afield,
            [&afield, &geo_array_index=geo_array_index, geo_index](const index_record& record, uint32_t seq_id) {
                // nested geopoint value inside an array of object will be a simple array so must be treated as geopoint
                bool nested_obj_arr_geopoint = (afield.nested && afield.type == field_types::GEOPOINT_ARRAY &&
                                    !record.doc[afield.name].empty() && record.doc[afield.name][0].is_number());

                if(afield.type == field_types::GEOPOINT || nested_obj_arr_geopoint) {
                    // this could be a nested gepoint array so can have more than 2 array values
                    const std::vector<double>& latlongs = record.doc[afield.name];
                    for(size_t li = 0; li < latlongs.size(); li+=2) {
                        S2RegionTermIndexer::Options options;
                        options.set_index_contains_points_only(true);
                        S2RegionTermIndexer indexer(options);
                        S2Point point = S2LatLng::FromDegrees(latlongs[li], latlongs[li+1]).ToPoint();

                        for(const auto& term: indexer.GetIndexTerms(point, "")) {
                            (*geo_index)[term].push_back(seq_id);
                        }
                    }

                    if(nested_obj_arr_geopoint) {
                        int64_t* packed_latlongs = new int64_t[(latlongs.size()/2) + 1];
                        packed_latlongs[0] = latlongs.size()/2;
                        size_t j_packed_latlongs = 0;

                        for(size_t li = 0; li < latlongs.size(); li+=2) {
                            int64_t packed_latlong = GeoPoint::pack_lat_lng(latlongs[li], latlongs[li+1]);
                            packed_latlongs[j_packed_latlongs + 1] = packed_latlong;
                            j_packed_latlongs++;
                        }

                        geo_array_index.at(afield.name)->emplace(seq_id, packed_latlongs);
                    }
                } else {
                    const std::vector<std::vector<double>>& latlongs = record.doc[afield.name];
                    S2RegionTermIndexer::Options options;
                    options.set_index_contains_points_only(true);
                    S2RegionTermIndexer indexer(options);

                    int64_t* packed_latlongs = new int64_t[latlongs.size() + 1];
                    packed_latlongs[0] = latlongs.size();

                    for(size_t li = 0; li < latlongs.size(); li++) {
                        auto& latlong = latlongs[li];
                        S2Point point = S2LatLng::FromDegrees(latlong[0], latlong[1]).ToPoint();
                        for(const auto& term: indexer.GetIndexTerms(point, "")) {
                            (*geo_index)[term].push_back(seq_id);
                        }

                        int64_t packed_latlong = GeoPoint::pack_lat_lng(latlong[0], latlong[1]);
                        packed_latlongs[li + 1] = packed_latlong;
                    }

                    geo_array_index.at(afield.name)->emplace(seq_id, packed_latlongs);
                }
            });
        } else if(afield.is_array()) {
            // handle vector index first
            if(afield.type == field_types::FLOAT_ARRAY && afield.num_dim > 0) {
                auto vec_index = vector_index[afield.name]->vecdex;
                size_t curr_ele_count = vec_index->getCurrentElementCount();
                if(curr_ele_count + iter_batch.size() > vec_index->getMaxElements()) {
                    vec_index->resizeIndex((curr_ele_count + iter_batch.size()) * 1.3);
                }

                const size_t num_threads = std::min<size_t>(4, iter_batch.size());
                const size_t window_size = (num_threads == 0) ? 0 :
                                           (iter_batch.size() + num_threads - 1) / num_threads;  // rounds up
                size_t num_processed = 0;
                std::mutex m_process;
                std::condition_variable cv_process;

                size_t num_queued = 0;
                size_t result_index = 0;

                for(size_t thread_id = 0; thread_id < num_threads && result_index < iter_batch.size(); thread_id++) {
                    size_t batch_len = window_size;

                    if(result_index + window_size > iter_batch.size()) {
                        batch_len = iter_batch.size() - result_index;
                    }

                    num_queued++;

                    thread_pool->enqueue([thread_id, &afield, &vec_index, &records = iter_batch,
                                          result_index, batch_len, &num_processed, &m_process, &cv_process]() {

                        size_t batch_counter = 0;
                        while(batch_counter < batch_len) {
                            auto& record = records[result_index + batch_counter];
                            if(record.doc.count(afield.name) == 0 || !record.indexed.ok()) {
                                batch_counter++;
                                continue;
                            }

                            const std::vector<float>& float_vals = record.doc[afield.name].get<std::vector<float>>();

                            try {
                                if(afield.vec_dist == cosine) {
                                    std::vector<float> normalized_vals(afield.num_dim);
                                    hnsw_index_t::normalize_vector(float_vals, normalized_vals);
                                    vec_index->addPoint(normalized_vals.data(), (size_t)record.seq_id, true);
                                } else {
                                    vec_index->addPoint(float_vals.data(), (size_t)record.seq_id, true);
                                }
                            } catch(const std::exception &e) {
                                record.index_failure(400, e.what());
                            }

                            batch_counter++;
                        }

                        std::unique_lock<std::mutex> lock(m_process);
                        num_processed++;
                        cv_process.notify_one();
                    });

                    result_index += batch_len;
                }

                std::unique_lock<std::mutex> lock_process(m_process);
                cv_process.wait(lock_process, [&](){ return num_processed == num_queued; });
                return;
            }

            // all other numerical arrays
            auto num_tree = numerical_index.at(afield.name);
            iterate_and_index_numerical_field(iter_batch, afield, [&afield, num_tree]
                    (const index_record& record, uint32_t seq_id) {
                for(size_t arr_i = 0; arr_i < record.doc[afield.name].size(); arr_i++) {
                    const auto& arr_value = record.doc[afield.name][arr_i];

                    if(afield.type == field_types::INT32_ARRAY) {
                        const int32_t value = arr_value;
                        num_tree->insert(value, seq_id);
                    }

                    else if(afield.type == field_types::INT64_ARRAY) {
                        const int64_t value = arr_value;
                        num_tree->insert(value, seq_id);
                    }

                    else if(afield.type == field_types::FLOAT_ARRAY) {
                        const float fvalue = arr_value;
                        int64_t value = float_to_int64_t(fvalue);
                        num_tree->insert(value, seq_id);
                    }

                    else if(afield.type == field_types::BOOL_ARRAY) {
                        const bool value = record.doc[afield.name][arr_i];
                        num_tree->insert(int64_t(value), seq_id);
                    }
                }
            });
        }

        // add numerical values automatically into sort index if sorting is enabled
        if(afield.is_num_sortable() && afield.type != field_types::GEOPOINT_ARRAY) {
            spp::sparse_hash_map<uint32_t, int64_t> *doc_to_score = sort_index.at(afield.name);

            bool is_integer = afield.is_integer();
            bool is_float = afield.is_float();
            bool is_bool = afield.is_bool();
            bool is_geopoint = afield.is_geopoint();

            for(const auto& record: iter_batch) {
                if(!record.indexed.ok()) {
                    continue;
                }

                const auto& document = record.doc;
                const auto seq_id = record.seq_id;

                if (document.count(afield.name) == 0 || !afield.index) {
                    continue;
                }

                if(is_integer) {
                    doc_to_score->emplace(seq_id, document[afield.name].get<int64_t>());
                } else if(is_float) {
                    int64_t ifloat = float_to_int64_t(document[afield.name].get<float>());
                    doc_to_score->emplace(seq_id, ifloat);
                } else if(is_bool) {
                    doc_to_score->emplace(seq_id, (int64_t) document[afield.name].get<bool>());
                } else if(is_geopoint) {
                    const std::vector<double>& latlong = document[afield.name];
                    int64_t lat_lng = GeoPoint::pack_lat_lng(latlong[0], latlong[1]);
                    doc_to_score->emplace(seq_id, lat_lng);
                }
            }
        }
    } else if(afield.is_str_sortable()) {
        adi_tree_t* str_tree = str_sort_index.at(afield.name);

        for(const auto& record: iter_batch) {
            if(!record.indexed.ok()) {
                continue;
            }

            const auto& document = record.doc;
            const auto seq_id = record.seq_id;

            if (document.count(afield.name) == 0 || !afield.index) {
                continue;
            }

            std::string raw_str = document[afield.name].get<std::string>();
            Tokenizer str_tokenizer("", true, false, "", {' '});
            str_tokenizer.tokenize(raw_str);

            if(!raw_str.empty()) {
                str_tree->index(seq_id, raw_str.substr(0, 2000));
            }
        }
    }
}

uint64_t Index::facet_token_hash(const field & a_field, const std::string &token) {
    // for integer/float use their native values
    uint64_t hash = 0;

    if(a_field.is_float()) {
        float f = std::stof(token);
        reinterpret_cast<float&>(hash) = f;  // store as int without loss of precision
    } else if(a_field.is_integer() || a_field.is_bool()) {
        hash = atoll(token.c_str());
    } else {
        // string field
        hash = StringUtils::hash_wy(token.c_str(), token.size());
    }

    return hash;
}

void Index::tokenize_string_with_facets(const std::string& text, bool is_facet, const field& a_field,
                                        const std::vector<char>& symbols_to_index,
                                        const std::vector<char>& token_separators,
                                        std::unordered_map<std::string, std::vector<uint32_t>>& token_to_offsets,
                                        std::vector<uint64_t>& facet_hashes) {

    Tokenizer tokenizer(text, true, !a_field.is_string(), a_field.locale, symbols_to_index, token_separators);
    std::string token;
    std::string last_token;
    size_t token_index = 0;
    uint64_t facet_hash = 1;

    while(tokenizer.next(token, token_index)) {
        if(token.empty()) {
            continue;
        }

        if(token.size() > 100) {
            token.erase(100);
        }

        token_to_offsets[token].push_back(token_index + 1);
        last_token = token;

        if(is_facet) {
            uint64_t token_hash = Index::facet_token_hash(a_field, token);
            if(token_index == 0) {
                facet_hash = token_hash;
            } else {
                facet_hash = StringUtils::hash_combine(facet_hash, token_hash);
            }
        }
    }

    if(!token_to_offsets.empty()) {
        // push 0 for the last occurring token (used for exact match ranking)
        token_to_offsets[last_token].push_back(0);
    }

    if(is_facet) {
        facet_hashes.push_back(facet_hash);
    }
}

void Index::tokenize_string_array_with_facets(const std::vector<std::string>& strings, bool is_facet,
                                              const field& a_field,
                                              const std::vector<char>& symbols_to_index,
                                              const std::vector<char>& token_separators,
                                              std::unordered_map<std::string, std::vector<uint32_t>>& token_to_offsets,
                                              std::vector<uint64_t>& facet_hashes) {

    for(size_t array_index = 0; array_index < strings.size(); array_index++) {
        const std::string& str = strings[array_index];
        std::set<std::string> token_set;  // required to deal with repeating tokens

        Tokenizer tokenizer(str, true, !a_field.is_string(), a_field.locale, symbols_to_index, token_separators);
        std::string token, last_token;
        size_t token_index = 0;
        uint64_t facet_hash = 1;

        // iterate and append offset positions
        while(tokenizer.next(token, token_index)) {
            if(token.empty()) {
                continue;
            }

            if(token.size() > 100) {
                token.erase(100);
            }

            token_to_offsets[token].push_back(token_index + 1);
            token_set.insert(token);
            last_token = token;

            if(is_facet) {
                uint64_t token_hash = Index::facet_token_hash(a_field, token);
                if(token_index == 0) {
                    facet_hash = token_hash;
                } else {
                    facet_hash = StringUtils::hash_combine(facet_hash, token_hash);
                }
            }
        }

        if(is_facet) {
            facet_hashes.push_back(facet_hash);
        }

        if(token_set.empty()) {
            continue;
        }

        for(auto& the_token: token_set) {
            // repeat last element to indicate end of offsets for this array index
            token_to_offsets[the_token].push_back(token_to_offsets[the_token].back());

            // iterate and append this array index to all tokens
            token_to_offsets[the_token].push_back(array_index);
        }

        // push 0 for the last occurring token (used for exact match ranking)
        token_to_offsets[last_token].push_back(0);
    }
}
void Index::initialize_facet_indexes(const field& facet_field) {
    if(facet_field.is_array()) {
        array_mapped_facet_t facet_array;
        for(size_t i = 0; i < ARRAY_FACET_DIM; i++) {
            facet_array[i] = new facet_map_t();
        }
        facet_index_v3.emplace(facet_field.name, facet_array);
    } else {
        array_mapped_single_val_facet_t facet_array;
        for(size_t i = 0; i < ARRAY_FACET_DIM; i++) {
            facet_array[i] = new single_val_facet_map_t();
        }
        single_val_facet_index_v3.emplace(facet_field.name, facet_array);
    }
}

void Index::compute_facet_stats(facet &a_facet, uint64_t raw_value, const std::string & field_type) {
    if(field_type == field_types::INT32 || field_type == field_types::INT32_ARRAY) {
        int32_t val = raw_value;
        if (val < a_facet.stats.fvmin) {
            a_facet.stats.fvmin = val;
        }
        if (val > a_facet.stats.fvmax) {
            a_facet.stats.fvmax = val;
        }
        a_facet.stats.fvsum += val;
        a_facet.stats.fvcount++;
    } else if(field_type == field_types::INT64 || field_type == field_types::INT64_ARRAY) {
        int64_t val = raw_value;
        if(val < a_facet.stats.fvmin) {
            a_facet.stats.fvmin = val;
        }
        if(val > a_facet.stats.fvmax) {
            a_facet.stats.fvmax = val;
        }
        a_facet.stats.fvsum += val;
        a_facet.stats.fvcount++;
    } else if(field_type == field_types::FLOAT || field_type == field_types::FLOAT_ARRAY) {
        float val = reinterpret_cast<float&>(raw_value);
        if(val < a_facet.stats.fvmin) {
            a_facet.stats.fvmin = val;
        }
        if(val > a_facet.stats.fvmax) {
            a_facet.stats.fvmax = val;
        }
        a_facet.stats.fvsum += val;
        a_facet.stats.fvcount++;
    }
}

void Index::do_facets(std::vector<facet> & facets, facet_query_t & facet_query,
                      bool estimate_facets, size_t facet_sample_percent,
                      const std::vector<facet_info_t>& facet_infos,
                      const size_t group_limit, const std::vector<std::string>& group_by_fields,
                      const uint32_t* result_ids, size_t results_size) const {

    // assumed that facet fields have already been validated upstream
    for(size_t findex=0; findex < facets.size(); findex++) {
        auto& a_facet = facets[findex];
        const auto& facet_field = facet_infos[findex].facet_field;
        const bool use_facet_query = facet_infos[findex].use_facet_query;
        const auto& fquery_hashes = facet_infos[findex].hashes;
        const bool should_compute_stats = facet_infos[findex].should_compute_stats;

        auto sort_index_it = sort_index.find(a_facet.field_name);

        size_t mod_value = 100 / facet_sample_percent;

        facet_map_t::iterator facet_map_it;
        single_val_facet_map_t::iterator single_facet_map_it;
        uint64_t fhash = 0;
        size_t facet_hash_count = 1;
        const auto& field_facet_mapping_it = facet_index_v3.find(a_facet.field_name);
        const auto& field_single_val_facet_mapping_it = single_val_facet_index_v3.find(a_facet.field_name);

        if((field_facet_mapping_it == facet_index_v3.end()) 
            && (field_single_val_facet_mapping_it == single_val_facet_index_v3.end())) {
                continue;
        }

        for(size_t i = 0; i < results_size; i++) {
            // if sampling is enabled, we will skip a portion of the results to speed up things
            if(estimate_facets) {
                if(i % mod_value != 0) {
                    continue;
                }
            }
    
            uint32_t doc_seq_id = result_ids[i];
            if(facet_field.is_array()) {
                const auto& field_facet_mapping = field_facet_mapping_it->second;
                facet_map_it = field_facet_mapping[doc_seq_id % ARRAY_FACET_DIM]->find(doc_seq_id);
    
                if(facet_map_it == field_facet_mapping[doc_seq_id % ARRAY_FACET_DIM]->end()) {
                    continue;
                }
                facet_hash_count = facet_map_it->second.size();
            } else {
                const auto& field_facet_mapping = field_single_val_facet_mapping_it->second;
                single_facet_map_it = field_facet_mapping[doc_seq_id % ARRAY_FACET_DIM]->find(doc_seq_id);
    
                if(single_facet_map_it == field_facet_mapping[doc_seq_id % ARRAY_FACET_DIM]->end()) {
                    continue;
                }
                facet_hash_count = 1;
                fhash = single_facet_map_it->second;
            }
            
            const uint64_t distinct_id = group_limit ? get_distinct_id(group_by_fields, doc_seq_id) : 0;

            if(((i + 1) % 16384) == 0) {
                RETURN_CIRCUIT_BREAKER
            }

            for(size_t j = 0; j < facet_hash_count; j++) {
                
                if(facet_field.is_array()) {
                    fhash = facet_map_it->second.hashes[j];
                }
        
                if(should_compute_stats) {
                    compute_facet_stats(a_facet, fhash, facet_field.type);
                }
                if(a_facet.is_range_query) {
                    if(sort_index_it != sort_index.end()){
                        auto doc_id_val_map = sort_index_it->second;
                        auto doc_seq_id_it = doc_id_val_map->find(doc_seq_id);
                
                        if(doc_seq_id_it != doc_id_val_map->end()){
                            int64_t doc_val = doc_seq_id_it->second;
                            std::pair<int64_t, std::string> range_pair {};
                            if(a_facet.get_range(doc_val, range_pair)) {
                                int64_t range_id = range_pair.first;
                                facet_count_t& facet_count = a_facet.result_map[range_id];
                                facet_count.count += 1;
                            }
                        }
                    }
                } else if(!use_facet_query || fquery_hashes.find(fhash) != fquery_hashes.end()) {
                    facet_count_t& facet_count = a_facet.result_map[fhash];
                    //LOG(INFO) << "field: " << a_facet.field_name << ", doc id: " << doc_seq_id << ", hash: " <<  fhash;
                    facet_count.doc_id = doc_seq_id;
                    facet_count.array_pos = j;
                    if(group_limit) {
                        a_facet.hash_groups[fhash].emplace(distinct_id);
                    } else {
                        facet_count.count += 1;
                    }
                    if(use_facet_query) {
                        a_facet.hash_tokens[fhash] = fquery_hashes.at(fhash);
                    }
                }
            }
        }
    }
}

void Index::aggregate_topster(Topster* agg_topster, Topster* index_topster) {
    if(index_topster->distinct) {
        for(auto &group_topster_entry: index_topster->group_kv_map) {
            Topster* group_topster = group_topster_entry.second;
            for(const auto& map_kv: group_topster->kv_map) {
                agg_topster->add(map_kv.second);
            }
        }
    } else {
        for(const auto& map_kv: index_topster->kv_map) {
            agg_topster->add(map_kv.second);
        }
    }
}

void Index::search_all_candidates(const size_t num_search_fields,
                                  const text_match_type_t match_type,
                                  const std::vector<search_field_t>& the_fields,
                                  const uint32_t* filter_ids, size_t filter_ids_length,
                                  const uint32_t* exclude_token_ids, size_t exclude_token_ids_size,
                                  const std::unordered_set<uint32_t>& excluded_group_ids,
                                  const std::vector<sort_by>& sort_fields,
                                  std::vector<tok_candidates>& token_candidates_vec,
                                  std::vector<std::vector<art_leaf*>>& searched_queries,
                                  tsl::htrie_map<char, token_leaf>& qtoken_set,
                                  const std::vector<token_t>& dropped_tokens,
                                  Topster* topster,
                                  spp::sparse_hash_map<uint64_t, uint32_t>& groups_processed,
                                  uint32_t*& all_result_ids, size_t& all_result_ids_len,
                                  const size_t typo_tokens_threshold,
                                  const size_t group_limit,
                                  const std::vector<std::string>& group_by_fields,
                                  const std::vector<token_t>& query_tokens,
                                  const std::vector<uint32_t>& num_typos,
                                  const std::vector<bool>& prefixes,
                                  bool prioritize_exact_match,
                                  const bool prioritize_token_position,
                                  const bool exhaustive_search,
                                  const size_t max_candidates,
                                  int syn_orig_num_tokens,
                                  const int* sort_order,
                                  std::array<spp::sparse_hash_map<uint32_t, int64_t>*, 3>& field_values,
                                  const std::vector<size_t>& geopoint_indices,
                                  std::set<uint64>& query_hashes,
                                  std::vector<uint32_t>& id_buff) const {

    /*if(!token_candidates_vec.empty()) {
        LOG(INFO) << "Prefix candidates size: " << token_candidates_vec.back().candidates.size();
        LOG(INFO) << "max_candidates: " << max_candidates;
        LOG(INFO) << "token_candidates_vec.size(): " << token_candidates_vec.size();
    }*/

    auto product = []( long long a, tok_candidates & b ) { return a*b.candidates.size(); };
    long long int N = std::accumulate(token_candidates_vec.begin(), token_candidates_vec.end(), 1LL, product);

    // escape hatch to prevent too much looping but subject to being overriden explicitly via `max_candidates`
    long long combination_limit = (num_search_fields == 1 && prefixes[0]) ? max_candidates :
                                  std::max<size_t>(Index::COMBINATION_MIN_LIMIT, max_candidates);

    for(long long n = 0; n < N && n < combination_limit; ++n) {
        RETURN_CIRCUIT_BREAKER

        std::vector<token_t> query_suggestion(token_candidates_vec.size());

        uint64 qhash;
        uint32_t total_cost = next_suggestion2(token_candidates_vec, n, query_suggestion, qhash);

        /*LOG(INFO) << "n: " << n;
        std::stringstream fullq;
        for(const auto& qtok : query_suggestion) {
            fullq << qtok.value << " ";
        }
        LOG(INFO) << "query: " << fullq.str() << ", total_cost: " << total_cost
                  << ", all_result_ids_len: " << all_result_ids_len << ", bufsiz: " << id_buff.size();*/

        if(query_hashes.find(qhash) != query_hashes.end()) {
            // skip this query since it has already been processed before
            //LOG(INFO) << "Skipping qhash " << qhash;
            continue;
        }

        //LOG(INFO) << "field_num_results: " << field_num_results << ", typo_tokens_threshold: " << typo_tokens_threshold;

        search_across_fields(query_suggestion, num_typos, prefixes, the_fields, num_search_fields, match_type,
                             sort_fields, topster,groups_processed,
                             searched_queries, qtoken_set, dropped_tokens, group_limit, group_by_fields,
                             prioritize_exact_match, prioritize_token_position,
                             filter_ids, filter_ids_length, total_cost, syn_orig_num_tokens,
                             exclude_token_ids, exclude_token_ids_size, excluded_group_ids,
                             sort_order, field_values, geopoint_indices,
                             id_buff, all_result_ids, all_result_ids_len);

        query_hashes.insert(qhash);
    }
}

void Index::search_candidates(const uint8_t & field_id, bool field_is_array,
                              const uint32_t* filter_ids, size_t filter_ids_length,
                              const uint32_t* exclude_token_ids, size_t exclude_token_ids_size,
                              const std::vector<uint32_t>& curated_ids,
                              std::vector<sort_by> & sort_fields,
                              std::vector<token_candidates> & token_candidates_vec,
                              std::vector<std::vector<art_leaf*>> & searched_queries,
                              Topster* topster,
                              spp::sparse_hash_map<uint64_t, uint32_t>& groups_processed,
                              uint32_t** all_result_ids, size_t & all_result_ids_len,
                              size_t& field_num_results,
                              const size_t typo_tokens_threshold,
                              const size_t group_limit,
                              const std::vector<std::string>& group_by_fields,
                              const std::vector<token_t>& query_tokens,
                              bool prioritize_exact_match,
                              const bool exhaustive_search,
                              int syn_orig_num_tokens,
                              const size_t concurrency,
                              std::set<uint64>& query_hashes,
                              std::vector<uint32_t>& id_buff) const {

    auto product = []( long long a, token_candidates & b ) { return a*b.candidates.size(); };
    long long int N = std::accumulate(token_candidates_vec.begin(), token_candidates_vec.end(), 1LL, product);

    int sort_order[3]; // 1 or -1 based on DESC or ASC respectively
    std::array<spp::sparse_hash_map<uint32_t, int64_t>*, 3> field_values;
    std::vector<size_t> geopoint_indices;

    populate_sort_mapping(sort_order, geopoint_indices, sort_fields, field_values);

    // escape hatch to prevent too much looping
    size_t combination_limit = exhaustive_search ? Index::COMBINATION_MAX_LIMIT : Index::COMBINATION_MIN_LIMIT;

    for (long long n = 0; n < N && n < combination_limit; ++n) {
        RETURN_CIRCUIT_BREAKER

        // every element in `query_suggestion` contains a token and its associated hits
        std::vector<art_leaf*> query_suggestion(token_candidates_vec.size());

        // actual query suggestion preserves original order of tokens in query
        std::vector<art_leaf*> actual_query_suggestion(token_candidates_vec.size());
        uint64 qhash;

        uint32_t token_bits = 0;
        uint32_t total_cost = next_suggestion(token_candidates_vec, n, actual_query_suggestion,
                                              query_suggestion, syn_orig_num_tokens, token_bits, qhash);

        if(query_hashes.find(qhash) != query_hashes.end()) {
            // skip this query since it has already been processed before
            continue;
        }

        query_hashes.insert(qhash);

        //LOG(INFO) << "field_num_results: " << field_num_results << ", typo_tokens_threshold: " << typo_tokens_threshold;
        //LOG(INFO) << "n: " << n;

        /*std::stringstream fullq;
        for(const auto& qleaf : actual_query_suggestion) {
            std::string qtok(reinterpret_cast<char*>(qleaf->key),qleaf->key_len - 1);
            fullq << qtok << " ";
        }
        LOG(INFO) << "field: " << size_t(field_id) << ", query: " << fullq.str() << ", total_cost: " << total_cost;*/

        // Prepare excluded document IDs that we can later remove from the result set
        uint32_t* excluded_result_ids = nullptr;
        size_t excluded_result_ids_size = ArrayUtils::or_scalar(exclude_token_ids, exclude_token_ids_size,
                                                                &curated_ids[0], curated_ids.size(), &excluded_result_ids);

        std::vector<void*> posting_lists;

        for(auto& query_leaf : query_suggestion) {
            posting_lists.push_back(query_leaf->values);
        }

        result_iter_state_t iter_state(
                excluded_result_ids, excluded_result_ids_size, filter_ids, filter_ids_length
        );

        // We fetch offset positions only for multi token query
        bool fetch_offsets = (query_suggestion.size() > 1);
        bool single_exact_query_token = false;

        if(total_cost == 0 && query_suggestion.size() == query_tokens.size() == 1) {
            // does this candidate suggestion token match query token exactly?
            single_exact_query_token = true;
        }

        if(topster == nullptr) {
            posting_t::block_intersector_t(posting_lists, iter_state)
            .intersect([&](uint32_t seq_id, std::vector<posting_list_t::iterator_t>& its) {
                id_buff.push_back(seq_id);
            });
        } else {
            posting_t::block_intersector_t(posting_lists, iter_state)
            .intersect([&](uint32_t seq_id, std::vector<posting_list_t::iterator_t>& its) {
                score_results(sort_fields, searched_queries.size(), field_id, field_is_array,
                              total_cost, topster, query_suggestion, groups_processed,
                              seq_id, sort_order, field_values, geopoint_indices,
                              group_limit, group_by_fields, token_bits,
                              prioritize_exact_match, single_exact_query_token, syn_orig_num_tokens, its);

                id_buff.push_back(seq_id);
            });
        }

        delete [] excluded_result_ids;
        const size_t num_result_ids = id_buff.size();

        if(id_buff.size() > 100000) {
            // prevents too many ORs during exhaustive searching
            std::sort(id_buff.begin(), id_buff.end());
            id_buff.erase(std::unique( id_buff.begin(), id_buff.end() ), id_buff.end());

            uint32_t* new_all_result_ids = nullptr;
            all_result_ids_len = ArrayUtils::or_scalar(*all_result_ids, all_result_ids_len, &id_buff[0],
                                                       id_buff.size(), &new_all_result_ids);
            delete[] *all_result_ids;
            *all_result_ids = new_all_result_ids;
            id_buff.clear();
        }

        if(num_result_ids == 0) {
            continue;
        }

        field_num_results += num_result_ids;
        searched_queries.push_back(actual_query_suggestion);
    }
}

void Index::numeric_not_equals_filter(num_tree_t* const num_tree,
                                      const int64_t value,
                                      const uint32_t& context_ids_length,
                                      uint32_t* const& context_ids,
                                      size_t& ids_len,
                                      uint32_t*& ids) const {
    uint32_t* to_exclude_ids = nullptr;
    size_t to_exclude_ids_len = 0;

    if (context_ids_length != 0) {
        num_tree->contains(EQUALS, value, context_ids_length, context_ids, to_exclude_ids_len, to_exclude_ids);
    } else {
        num_tree->search(EQUALS, value, &to_exclude_ids, to_exclude_ids_len);
    }

    auto all_ids = seq_ids->uncompress();
    auto all_ids_size = seq_ids->num_ids();

    uint32_t* to_include_ids = nullptr;
    size_t to_include_ids_len = 0;

    to_include_ids_len = ArrayUtils::exclude_scalar(all_ids, all_ids_size, to_exclude_ids,
                                                    to_exclude_ids_len, &to_include_ids);

    delete[] all_ids;
    delete[] to_exclude_ids;

    uint32_t* out = nullptr;
    ids_len = ArrayUtils::or_scalar(ids, ids_len, to_include_ids, to_include_ids_len, &out);

    delete[] ids;
    delete[] to_include_ids;

    ids = out;
}

bool Index::field_is_indexed(const std::string& field_name) const {
    return search_index.count(field_name) != 0 ||
    numerical_index.count(field_name) != 0 ||
    geopoint_index.count(field_name) != 0;
}

Option<bool> Index::do_filtering(filter_node_t* const root,
                                 filter_result_t& result,
                                 const std::string& collection_name,
                                 const uint32_t& context_ids_length,
                                 uint32_t* const& context_ids) const {
    // auto begin = std::chrono::high_resolution_clock::now();
    const filter a_filter = root->filter_exp;

    bool is_referenced_filter = !a_filter.referenced_collection_name.empty();
    if (is_referenced_filter) {
        // Apply filter on referenced collection and get the sequence ids of current collection from the filtered documents.
        auto& cm = CollectionManager::get_instance();
        auto collection = cm.get_collection(a_filter.referenced_collection_name);
        if (collection == nullptr) {
            return Option<bool>(400, "Referenced collection `" + a_filter.referenced_collection_name + "` not found.");
        }

        filter_result_t reference_filter_result;
        auto reference_filter_op = collection->get_reference_filter_ids(a_filter.field_name,
                                                                        reference_filter_result,
                                                                        collection_name);
        if (!reference_filter_op.ok()) {
            return Option<bool>(400, "Failed to apply reference filter on `" + a_filter.referenced_collection_name
                                        + "` collection: " + reference_filter_op.error());
        }

        if (context_ids_length != 0) {
            std::vector<uint32_t> include_indexes;
            include_indexes.reserve(std::min(context_ids_length, reference_filter_result.count));

            size_t context_index = 0, reference_result_index = 0;
            while (context_index < context_ids_length && reference_result_index < reference_filter_result.count) {
                if (context_ids[context_index] == reference_filter_result.docs[reference_result_index]) {
                    include_indexes.push_back(reference_result_index);
                    context_index++;
                    reference_result_index++;
                } else if (context_ids[context_index] < reference_filter_result.docs[reference_result_index]) {
                    context_index++;
                } else {
                    reference_result_index++;
                }
            }

            result.count = include_indexes.size();
            result.docs = new uint32_t[include_indexes.size()];
            auto& result_references = result.reference_filter_results[a_filter.referenced_collection_name];
            result_references = new reference_filter_result_t[include_indexes.size()];

            for (uint32_t i = 0; i < include_indexes.size(); i++) {
                result.docs[i] = reference_filter_result.docs[include_indexes[i]];
                result_references[i] = reference_filter_result.reference_filter_results[a_filter.referenced_collection_name][include_indexes[i]];
            }

            return Option(true);
        }

        result = std::move(reference_filter_result);
        return Option(true);
    }

    if (a_filter.field_name == "id") {
        // we handle `ids` separately
        std::vector<uint32> result_ids;
        for (const auto& id_str : a_filter.values) {
            result_ids.push_back(std::stoul(id_str));
        }

        std::sort(result_ids.begin(), result_ids.end());

        auto result_array = new uint32[result_ids.size()];
        std::copy(result_ids.begin(), result_ids.end(), result_array);

        if (context_ids_length != 0) {
            uint32_t* out = nullptr;
            result.count = ArrayUtils::and_scalar(context_ids, context_ids_length,
                                                  result_array, result_ids.size(), &out);

            delete[] result_array;

            result.docs = out;
            return Option(true);
        }

        result.docs = result_array;
        result.count = result_ids.size();
        return Option(true);
    }

    if (!field_is_indexed(a_filter.field_name)) {
        return Option(true);
    }

    field f = search_schema.at(a_filter.field_name);

    uint32_t* result_ids = nullptr;
    size_t result_ids_len = 0;

    if (f.is_integer()) {
        auto num_tree = numerical_index.at(a_filter.field_name);

        for (size_t fi = 0; fi < a_filter.values.size(); fi++) {
            const std::string& filter_value = a_filter.values[fi];
            int64_t value = (int64_t)std::stol(filter_value);

            if(a_filter.comparators[fi] == RANGE_INCLUSIVE && fi+1 < a_filter.values.size()) {
                const std::string& next_filter_value = a_filter.values[fi + 1];
                auto const range_end_value = (int64_t)std::stol(next_filter_value);

                if (context_ids_length != 0) {
                    num_tree->range_inclusive_contains(value, range_end_value, context_ids_length, context_ids,
                                                       result_ids_len, result_ids);
                } else {
                    num_tree->range_inclusive_search(value, range_end_value, &result_ids, result_ids_len);
                }

                fi++;
            } else if (a_filter.comparators[fi] == NOT_EQUALS) {
                numeric_not_equals_filter(num_tree, value, context_ids_length, context_ids, result_ids_len, result_ids);
            } else {
                if (context_ids_length != 0) {
                    num_tree->contains(a_filter.comparators[fi], value,
                                       context_ids_length, context_ids, result_ids_len, result_ids);
                } else {
                    num_tree->search(a_filter.comparators[fi], value, &result_ids, result_ids_len);
                }
            }
        }
    } else if (f.is_float()) {
        auto num_tree = numerical_index.at(a_filter.field_name);

        for (size_t fi = 0; fi < a_filter.values.size(); fi++) {
            const std::string& filter_value = a_filter.values[fi];
            float value = (float)std::atof(filter_value.c_str());
            int64_t float_int64 = float_to_int64_t(value);

            if(a_filter.comparators[fi] == RANGE_INCLUSIVE && fi+1 < a_filter.values.size()) {
                const std::string& next_filter_value = a_filter.values[fi+1];
                int64_t range_end_value = float_to_int64_t((float) std::atof(next_filter_value.c_str()));

                if (context_ids_length != 0) {
                    num_tree->range_inclusive_contains(float_int64, range_end_value, context_ids_length, context_ids,
                                                       result_ids_len, result_ids);
                } else {
                    num_tree->range_inclusive_search(float_int64, range_end_value, &result_ids, result_ids_len);
                }

                fi++;
            } else if (a_filter.comparators[fi] == NOT_EQUALS) {
                numeric_not_equals_filter(num_tree, float_int64,
                                          context_ids_length, context_ids, result_ids_len, result_ids);
            } else {
                if (context_ids_length != 0) {
                    num_tree->contains(a_filter.comparators[fi], float_int64,
                                       context_ids_length, context_ids, result_ids_len, result_ids);
                } else {
                    num_tree->search(a_filter.comparators[fi], float_int64, &result_ids, result_ids_len);
                }
            }
        }
    } else if (f.is_bool()) {
        auto num_tree = numerical_index.at(a_filter.field_name);

        size_t value_index = 0;
        for (const std::string& filter_value : a_filter.values) {
            int64_t bool_int64 = (filter_value == "1") ? 1 : 0;
            if (a_filter.comparators[value_index] == NOT_EQUALS) {
                numeric_not_equals_filter(num_tree, bool_int64,
                                          context_ids_length, context_ids, result_ids_len, result_ids);
            } else {
                if (context_ids_length != 0) {
                    num_tree->contains(a_filter.comparators[value_index], bool_int64,
                                       context_ids_length, context_ids, result_ids_len, result_ids);
                } else {
                    num_tree->search(a_filter.comparators[value_index], bool_int64, &result_ids, result_ids_len);
                }
            }

            value_index++;
        }
    } else if (f.is_geopoint()) {
        for (const std::string& filter_value : a_filter.values) {
            std::vector<uint32_t> geo_result_ids;

            std::vector<std::string> filter_value_parts;
            StringUtils::split(filter_value, filter_value_parts, ",");  // x, y, 2, km (or) list of points

            bool is_polygon = StringUtils::is_float(filter_value_parts.back());
            S2Region* query_region;

            if (is_polygon) {
                const int num_verts = int(filter_value_parts.size()) / 2;
                std::vector<S2Point> vertices;
                double sum = 0.0;

                for (size_t point_index = 0; point_index < size_t(num_verts);
                     point_index++) {
                    double lat = std::stod(filter_value_parts[point_index * 2]);
                    double lon = std::stod(filter_value_parts[point_index * 2 + 1]);
                    S2Point vertex = S2LatLng::FromDegrees(lat, lon).ToPoint();
                    vertices.emplace_back(vertex);
                }

                auto loop = new S2Loop(vertices, S2Debug::DISABLE);
                loop->Normalize();  // if loop is not CCW but CW, change to CCW.

                S2Error error;
                if (loop->FindValidationError(&error)) {
                    LOG(ERROR) << "Query vertex is bad, skipping. Error: " << error;
                    delete loop;
                    continue;
                } else {
                    query_region = loop;
                }
            } else {
                double radius = std::stof(filter_value_parts[2]);
                const auto& unit = filter_value_parts[3];

                if (unit == "km") {
                    radius *= 1000;
                } else {
                    // assume "mi" (validated upstream)
                    radius *= 1609.34;
                }

                S1Angle query_radius = S1Angle::Radians(S2Earth::MetersToRadians(radius));
                double query_lat = std::stod(filter_value_parts[0]);
                double query_lng = std::stod(filter_value_parts[1]);
                S2Point center = S2LatLng::FromDegrees(query_lat, query_lng).ToPoint();
                query_region = new S2Cap(center, query_radius);
            }

            S2RegionTermIndexer::Options options;
            options.set_index_contains_points_only(true);
            S2RegionTermIndexer indexer(options);

            for (const auto& term : indexer.GetQueryTerms(*query_region, "")) {
                auto geo_index = geopoint_index.at(a_filter.field_name);
                const auto& ids_it = geo_index->find(term);
                if(ids_it != geo_index->end()) {
                    geo_result_ids.insert(geo_result_ids.end(), ids_it->second.begin(), ids_it->second.end());
                }
            }

            gfx::timsort(geo_result_ids.begin(), geo_result_ids.end());
            geo_result_ids.erase(std::unique( geo_result_ids.begin(), geo_result_ids.end() ), geo_result_ids.end());

            // `geo_result_ids` will contain all IDs that are within approximately within query radius
            // we still need to do another round of exact filtering on them

            if (context_ids_length != 0) {
                uint32_t *out = nullptr;
                uint32_t count = ArrayUtils::and_scalar(context_ids, context_ids_length,
                                                        &geo_result_ids[0], geo_result_ids.size(), &out);

                geo_result_ids = std::vector<uint32_t>(out, out + count);
            }

            std::vector<uint32_t> exact_geo_result_ids;

            if (f.is_single_geopoint()) {
                spp::sparse_hash_map<uint32_t, int64_t>* sort_field_index = sort_index.at(f.name);

                for (auto result_id : geo_result_ids) {
                    // no need to check for existence of `result_id` because of indexer based pre-filtering above
                    int64_t lat_lng = sort_field_index->at(result_id);
                    S2LatLng s2_lat_lng;
                    GeoPoint::unpack_lat_lng(lat_lng, s2_lat_lng);
                    if (query_region->Contains(s2_lat_lng.ToPoint())) {
                        exact_geo_result_ids.push_back(result_id);
                    }
                }
            } else {
                spp::sparse_hash_map<uint32_t, int64_t*>* geo_field_index = geo_array_index.at(f.name);

                for (auto result_id : geo_result_ids) {
                    int64_t* lat_lngs = geo_field_index->at(result_id);

                    bool point_found = false;

                    // any one point should exist
                    for (size_t li = 0; li < lat_lngs[0]; li++) {
                        int64_t lat_lng = lat_lngs[li + 1];
                        S2LatLng s2_lat_lng;
                        GeoPoint::unpack_lat_lng(lat_lng, s2_lat_lng);
                        if (query_region->Contains(s2_lat_lng.ToPoint())) {
                            point_found = true;
                            break;
                        }
                    }

                    if (point_found) {
                        exact_geo_result_ids.push_back(result_id);
                    }
                }
            }

            uint32_t* out = nullptr;
            result_ids_len = ArrayUtils::or_scalar(&exact_geo_result_ids[0], exact_geo_result_ids.size(),
                                                   result_ids, result_ids_len, &out);

            delete[] result_ids;
            result_ids = out;

            delete query_region;
        }
    } else if (f.is_string()) {
        art_tree* t = search_index.at(a_filter.field_name);

        uint32_t* or_ids = nullptr;
        size_t or_ids_size = 0;

        // aggregates IDs across array of filter values and reduces excessive ORing
        std::vector<uint32_t> f_id_buff;

        for (const std::string& filter_value : a_filter.values) {
            std::vector<void*> posting_lists;

            // there could be multiple tokens in a filter value, which we have to treat as ANDs
            // e.g. country: South Africa
            Tokenizer tokenizer(filter_value, true, false, f.locale, symbols_to_index, token_separators);

            std::string str_token;
            size_t token_index = 0;
            std::vector<std::string> str_tokens;

            while (tokenizer.next(str_token, token_index)) {
                str_tokens.push_back(str_token);

                art_leaf* leaf = (art_leaf *) art_search(t, (const unsigned char*) str_token.c_str(),
                                                         str_token.length()+1);
                if (leaf == nullptr) {
                    continue;
                }

                posting_lists.push_back(leaf->values);
            }

            if (posting_lists.size() != str_tokens.size()) {
                continue;
            }

            if(a_filter.comparators[0] == EQUALS || a_filter.comparators[0] == NOT_EQUALS) {
                // needs intersection + exact matching (unlike CONTAINS)
                std::vector<uint32_t> result_id_vec;
                posting_t::intersect(posting_lists, result_id_vec, context_ids_length, context_ids);

                if (result_id_vec.empty()) {
                    continue;
                }

                // need to do exact match
                uint32_t* exact_str_ids = new uint32_t[result_id_vec.size()];
                size_t exact_str_ids_size = 0;
                std::unique_ptr<uint32_t[]> exact_str_ids_guard(exact_str_ids);

                posting_t::get_exact_matches(posting_lists, f.is_array(), result_id_vec.data(), result_id_vec.size(),
                                             exact_str_ids, exact_str_ids_size);

                if (exact_str_ids_size == 0) {
                    continue;
                }

                for (size_t ei = 0; ei < exact_str_ids_size; ei++) {
                    f_id_buff.push_back(exact_str_ids[ei]);
                }
            } else {
                // CONTAINS
                size_t before_size = f_id_buff.size();
                posting_t::intersect(posting_lists, f_id_buff, context_ids_length, context_ids);
                if (f_id_buff.size() == before_size) {
                    continue;
                }
            }

            if (f_id_buff.size() > 100000 || a_filter.values.size() == 1) {
                gfx::timsort(f_id_buff.begin(), f_id_buff.end());
                f_id_buff.erase(std::unique( f_id_buff.begin(), f_id_buff.end() ), f_id_buff.end());

                uint32_t* out = nullptr;
                or_ids_size = ArrayUtils::or_scalar(or_ids, or_ids_size, f_id_buff.data(), f_id_buff.size(), &out);
                delete[] or_ids;
                or_ids = out;
                std::vector<uint32_t>().swap(f_id_buff);  // clears out memory
            }
        }

        if (!f_id_buff.empty()) {
            gfx::timsort(f_id_buff.begin(), f_id_buff.end());
            f_id_buff.erase(std::unique( f_id_buff.begin(), f_id_buff.end() ), f_id_buff.end());

            uint32_t* out = nullptr;
            or_ids_size = ArrayUtils::or_scalar(or_ids, or_ids_size, f_id_buff.data(), f_id_buff.size(), &out);
            delete[] or_ids;
            or_ids = out;
            std::vector<uint32_t>().swap(f_id_buff);  // clears out memory
        }

        result_ids = or_ids;
        result_ids_len = or_ids_size;
    }

    if (a_filter.apply_not_equals) {
        auto all_ids = seq_ids->uncompress();
        auto all_ids_size = seq_ids->num_ids();

        uint32_t* to_include_ids = nullptr;
        size_t to_include_ids_len = 0;

        to_include_ids_len = ArrayUtils::exclude_scalar(all_ids, all_ids_size, result_ids,
                                                        result_ids_len, &to_include_ids);

        delete[] all_ids;
        delete[] result_ids;

        result_ids = to_include_ids;
        result_ids_len = to_include_ids_len;

        if (context_ids_length != 0) {
            uint32_t *out = nullptr;
            result.count = ArrayUtils::and_scalar(context_ids, context_ids_length,
                                                  result_ids, result_ids_len, &out);

            delete[] result_ids;

            result.docs = out;
            return Option(true);
        }
    }

    result.docs = result_ids;
    result.count = result_ids_len;

    return Option(true);
    /*long long int timeMillis =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now()
    - begin).count();

    LOG(INFO) << "Time taken for filtering: " << timeMillis << "ms";*/
}

void Index::aproximate_numerical_match(num_tree_t* const num_tree,
                                       const NUM_COMPARATOR& comparator,
                                       const int64_t& value,
                                       const int64_t& range_end_value,
                                       uint32_t& filter_ids_length) const {
    if (comparator == RANGE_INCLUSIVE) {
        num_tree->approx_range_inclusive_search_count(value, range_end_value, filter_ids_length);
        return;
    }

    if (comparator == NOT_EQUALS) {
        uint32_t to_exclude_ids_len = 0;
        num_tree->approx_search_count(EQUALS, value, to_exclude_ids_len);

        auto all_ids_size = seq_ids->num_ids();
        filter_ids_length += (all_ids_size - to_exclude_ids_len);
        return;
    }

    num_tree->approx_search_count(comparator, value, filter_ids_length);
}

Option<bool> Index::_approximate_filter_ids(const filter& a_filter,
                                            uint32_t& filter_ids_length,
                                            const std::string& collection_name) const {
    if (!a_filter.referenced_collection_name.empty()) {
        auto& cm = CollectionManager::get_instance();
        auto collection = cm.get_collection(a_filter.referenced_collection_name);
        if (collection == nullptr) {
            return Option<bool>(400, "Referenced collection `" + a_filter.referenced_collection_name + "` not found.");
        }

        return collection->get_approximate_reference_filter_ids(a_filter.field_name, filter_ids_length);
    }

    if (a_filter.field_name == "id") {
        filter_ids_length = a_filter.values.size();
        return Option(true);
    }

    if (!field_is_indexed(a_filter.field_name)) {
        return Option(true);
    }

    field f = search_schema.at(a_filter.field_name);

    if (f.is_integer()) {
        auto num_tree = numerical_index.at(f.name);

        for (size_t fi = 0; fi < a_filter.values.size(); fi++) {
            const std::string& filter_value = a_filter.values[fi];
            auto const value = (int64_t)std::stol(filter_value);

            if (a_filter.comparators[fi] == RANGE_INCLUSIVE && fi+1 < a_filter.values.size()) {
                const std::string& next_filter_value = a_filter.values[fi + 1];
                auto const range_end_value = (int64_t)std::stol(next_filter_value);

                aproximate_numerical_match(num_tree, a_filter.comparators[fi], value, range_end_value,
                                           filter_ids_length);
                fi++;
            } else {
                aproximate_numerical_match(num_tree, a_filter.comparators[fi], value, 0, filter_ids_length);
            }
        }
    } else if (f.is_float()) {
        auto num_tree = numerical_index.at(a_filter.field_name);

        for (size_t fi = 0; fi < a_filter.values.size(); fi++) {
            const std::string& filter_value = a_filter.values[fi];
            float value = (float)std::atof(filter_value.c_str());
            int64_t float_int64 = float_to_int64_t(value);

            if (a_filter.comparators[fi] == RANGE_INCLUSIVE && fi+1 < a_filter.values.size()) {
                const std::string& next_filter_value = a_filter.values[fi + 1];
                auto const range_end_value = float_to_int64_t((float) std::atof(next_filter_value.c_str()));

                aproximate_numerical_match(num_tree, a_filter.comparators[fi], float_int64, range_end_value,
                                           filter_ids_length);
                fi++;
            } else {
                aproximate_numerical_match(num_tree, a_filter.comparators[fi], float_int64, 0, filter_ids_length);
            }
        }
    } else if (f.is_bool()) {
        auto num_tree = numerical_index.at(a_filter.field_name);

        size_t value_index = 0;
        for (const std::string& filter_value : a_filter.values) {
            int64_t bool_int64 = (filter_value == "1") ? 1 : 0;

            aproximate_numerical_match(num_tree, a_filter.comparators[value_index], bool_int64, 0, filter_ids_length);
            value_index++;
        }
    } else if (f.is_geopoint()) {
        filter_ids_length = 100;
    } else if (f.is_string()) {
        art_tree* t = search_index.at(a_filter.field_name);

        for (const std::string& filter_value : a_filter.values) {
            Tokenizer tokenizer(filter_value, true, false, f.locale, symbols_to_index, token_separators);

            std::string str_token;
            size_t token_index = 0;

            while (tokenizer.next(str_token, token_index)) {
                auto const leaf = (art_leaf *) art_search(t, (const unsigned char*) str_token.c_str(),
                                                          str_token.length()+1);
                if (leaf == nullptr) {
                    continue;
                }

                filter_ids_length += posting_t::num_ids(leaf->values);
            }
        }
    }

    if (a_filter.apply_not_equals) {
        auto all_ids_size = seq_ids->num_ids();
        filter_ids_length = (all_ids_size - filter_ids_length);
    }

    return Option(true);
}

Option<bool> Index::rearrange_filter_tree(filter_node_t* const root,
                                          uint32_t& approx_filter_ids_length,
                                          const std::string& collection_name) const {
    if (root == nullptr) {
        return Option(true);
    }

    if (root->isOperator) {
        uint32_t l_filter_ids_length = 0;
        if (root->left != nullptr) {
            auto rearrange_op = rearrange_filter_tree(root->left, l_filter_ids_length, collection_name);
            if (!rearrange_op.ok()) {
                return rearrange_op;
            }
        }

        uint32_t r_filter_ids_length = 0;
        if (root->right != nullptr) {
            auto rearrange_op = rearrange_filter_tree(root->right, r_filter_ids_length, collection_name);
            if (!rearrange_op.ok()) {
                return rearrange_op;
            }
        }

        if (root->filter_operator == AND) {
            approx_filter_ids_length = std::min(l_filter_ids_length, r_filter_ids_length);
        } else {
            approx_filter_ids_length = l_filter_ids_length + r_filter_ids_length;
        }

        if (l_filter_ids_length > r_filter_ids_length) {
            std::swap(root->left, root->right);
        }

        return Option(true);
    }

    _approximate_filter_ids(root->filter_exp, approx_filter_ids_length, collection_name);
    return Option(true);
}

Option<bool> Index::recursive_filter(filter_node_t* const root,
                                     filter_result_t& result,
                                     const std::string& collection_name,
                                     const uint32_t& context_ids_length,
                                     uint32_t* const& context_ids) const {
    if (root == nullptr) {
        return Option(true);
    }

    if (root->isOperator) {
        filter_result_t l_result;
        if (root->left != nullptr) {
            auto filter_op = recursive_filter(root->left, l_result , collection_name, context_ids_length, context_ids);
            if (!filter_op.ok()) {
                return filter_op;
            }
        }

        filter_result_t r_result;
        if (root->right != nullptr) {
            auto filter_op = recursive_filter(root->right, r_result , collection_name, context_ids_length, context_ids);
            if (!filter_op.ok()) {
                return filter_op;
            }
        }

        if (root->filter_operator == AND) {
            filter_result_t::and_filter_results(l_result, r_result, result);
        } else {
            filter_result_t::or_filter_results(l_result, r_result, result);
        }

        return Option(true);
    }

    return do_filtering(root, result, collection_name, context_ids_length, context_ids);
}

Option<bool> Index::do_filtering_with_lock(filter_node_t* const filter_tree_root,
                                           filter_result_t& filter_result,
                                           const std::string& collection_name) const {
    std::shared_lock lock(mutex);

    auto filter_op = recursive_filter(filter_tree_root, filter_result, collection_name);
    if (!filter_op.ok()) {
        return filter_op;
    }

    return Option(true);
}

Option<bool> Index::do_reference_filtering_with_lock(filter_node_t* const filter_tree_root,
                                                     filter_result_t& filter_result,
                                                     const std::string& collection_name,
                                                     const std::string& reference_helper_field_name) const {
    std::shared_lock lock(mutex);

    filter_result_t reference_filter_result;
    auto filter_op = recursive_filter(filter_tree_root, reference_filter_result);
    if (!filter_op.ok()) {
        return filter_op;
    }

    // doc id -> reference doc ids
    std::map<uint32_t, std::vector<uint32_t>> reference_map;
    for (uint32_t i = 0; i < reference_filter_result.count; i++) {
        auto reference_doc_id = reference_filter_result.docs[i];
        auto doc_id = sort_index.at(reference_helper_field_name)->at(reference_doc_id);

        reference_map[doc_id].push_back(reference_doc_id);
    }

    filter_result.count = reference_map.size();
    filter_result.docs = new uint32_t[reference_map.size()];
    filter_result.reference_filter_results[collection_name] = new reference_filter_result_t[reference_map.size()];

    size_t doc_index = 0;
    for (auto &item: reference_map) {
        filter_result.docs[doc_index] = item.first;

        auto& reference_result = filter_result.reference_filter_results[collection_name][doc_index];
        reference_result.count = item.second.size();
        reference_result.docs = new uint32_t[item.second.size()];
        std::copy(item.second.begin(), item.second.end(), reference_result.docs);

        doc_index++;
    }

    return Option(true);
}

Option<bool> Index::get_approximate_reference_filter_ids_with_lock(filter_node_t* const filter_tree_root,
                                                                   uint32_t& filter_ids_length) const {
    std::shared_lock lock(mutex);

    return rearrange_filter_tree(filter_tree_root, filter_ids_length);
}

Option<bool> Index::run_search(search_args* search_params, const std::string& collection_name) {
    return search(search_params->field_query_tokens,
           search_params->search_fields,
           search_params->match_type,
           search_params->filter_tree_root, search_params->facets, search_params->facet_query,
           search_params->included_ids, search_params->excluded_ids,
           search_params->sort_fields_std, search_params->num_typos,
           search_params->topster, search_params->curated_topster,
           search_params->per_page, search_params->offset, search_params->token_order,
           search_params->prefixes, search_params->drop_tokens_threshold,
           search_params->all_result_ids_len, search_params->groups_processed,
           search_params->searched_queries,
           search_params->qtoken_set,
           search_params->raw_result_kvs, search_params->override_result_kvs,
           search_params->typo_tokens_threshold,
           search_params->group_limit, search_params->group_by_fields,
           search_params->default_sorting_field,
           search_params->prioritize_exact_match,
           search_params->prioritize_token_position,
           search_params->exhaustive_search,
           search_params->concurrency,
           search_params->search_cutoff_ms,
           search_params->min_len_1typo,
           search_params->min_len_2typo,
           search_params->max_candidates,
           search_params->infixes,
           search_params->max_extra_prefix,
           search_params->max_extra_suffix,
           search_params->facet_query_num_typos,
           search_params->filter_curated_hits,
           search_params->split_join_tokens,
           search_params->vector_query,
           search_params->facet_sample_percent,
           search_params->facet_sample_threshold,
           collection_name);
}

void Index::collate_included_ids(const std::vector<token_t>& q_included_tokens,
                                 const std::map<size_t, std::map<size_t, uint32_t>> & included_ids_map,
                                 Topster* curated_topster,
                                 std::vector<std::vector<art_leaf*>> & searched_queries) const {

    if(included_ids_map.empty()) {
        return;
    }

    for(const auto& pos_ids: included_ids_map) {
        const size_t outer_pos = pos_ids.first;

        for(const auto& index_seq_id: pos_ids.second) {
            uint32_t inner_pos = index_seq_id.first;
            uint32_t seq_id = index_seq_id.second;

            uint64_t distinct_id = outer_pos;                           // outer pos is the group distinct key
            uint64_t match_score = (64000 - outer_pos - inner_pos);    // both outer pos and inner pos inside group

            // LOG(INFO) << "seq_id: " << seq_id << " - " << match_score;

            int64_t scores[3];
            scores[0] = match_score;
            scores[1] = int64_t(1);
            scores[2] = int64_t(1);

            KV kv(searched_queries.size(), seq_id, distinct_id, 0, scores, nullptr);
            curated_topster->add(&kv);
        }
    }
}

void Index::concat_topster_ids(Topster* topster, spp::sparse_hash_map<uint64_t, std::vector<KV*>>& topster_ids) {
    if(topster->distinct) {
        for(auto &group_topster_entry: topster->group_kv_map) {
            Topster* group_topster = group_topster_entry.second;
            for(const auto& map_kv: group_topster->kv_map) {
                topster_ids[map_kv.first].push_back(map_kv.second);
            }
        }
    } else {
        for(const auto& map_kv: topster->kv_map) {
            //LOG(INFO) << "map_kv.second.key: " << map_kv.second->key;
            //LOG(INFO) << "map_kv.first: " << map_kv.first;
            topster_ids[map_kv.first].push_back(map_kv.second);
        }
    }
}

bool Index::static_filter_query_eval(const override_t* override,
                                     std::vector<std::string>& tokens,
                                     filter_node_t*& filter_tree_root) const {
    std::string query = StringUtils::join(tokens, " ");

    if ((override->rule.match == override_t::MATCH_EXACT && override->rule.normalized_query == query) ||
        (override->rule.match == override_t::MATCH_CONTAINS &&
         StringUtils::contains_word(query, override->rule.normalized_query))) {
        filter_node_t* new_filter_tree_root = nullptr;
        Option<bool> filter_op = filter::parse_filter_query(override->filter_by, search_schema,
                                                            store, "", new_filter_tree_root);
        if (filter_op.ok()) {
            if (filter_tree_root == nullptr) {
                filter_tree_root = new_filter_tree_root;
            } else {
                filter_node_t* root = new filter_node_t(AND, filter_tree_root,
                                                        new_filter_tree_root);
                filter_tree_root = root;
            }
            return true;
        } else {
            delete new_filter_tree_root;
        }
    }

    return false;
}

bool Index::resolve_override(const std::vector<std::string>& rule_tokens, const bool exact_rule_match,
                             const std::vector<std::string>& query_tokens,
                             token_ordering token_order, std::set<std::string>& absorbed_tokens,
                             std::string& filter_by_clause) const {

    bool resolved_override = false;
    size_t i = 0, j = 0;

    std::unordered_map<std::string, std::vector<std::string>> field_placeholder_tokens;

    while(i < rule_tokens.size()) {
        if(rule_tokens[i].front() == '{' && rule_tokens[i].back() == '}') {
            // found a field placeholder
            std::vector<std::string> field_names;
            std::string rule_part = rule_tokens[i];
            field_names.emplace_back(rule_part.erase(0, 1).erase(rule_part.size() - 1));

            // skip until we find a non-placeholder token
            i++;

            while(i < rule_tokens.size() && (rule_tokens[i].front() == '{' && rule_tokens[i].back() == '}')) {
                rule_part = rule_tokens[i];
                field_names.emplace_back(rule_part.erase(0, 1).erase(rule_part.size() - 1));
                i++;
            }

            std::vector<std::string> matched_tokens;

            // `i` now points to either end of array or at a non-placeholder rule token
            // end of array: add remaining query tokens as matched tokens
            // non-placeholder: skip query tokens until it matches a rule token

            while(j < query_tokens.size() && (i == rule_tokens.size() || rule_tokens[i] != query_tokens[j])) {
                matched_tokens.emplace_back(query_tokens[j]);
                j++;
            }

            resolved_override = true;

            // we try to map `field_names` against `matched_tokens` now
            for(size_t findex = 0; findex < field_names.size(); findex++) {
                const auto& field_name = field_names[findex];
                bool slide_window = (findex == 0);  // fields following another field should match exactly
                std::vector<std::string> field_absorbed_tokens;
                resolved_override &= check_for_overrides(token_order, field_name, slide_window,
                                                         exact_rule_match, matched_tokens, absorbed_tokens,
                                                         field_absorbed_tokens);

                if(!resolved_override) {
                    goto RETURN_EARLY;
                }

                field_placeholder_tokens[field_name] = field_absorbed_tokens;
            }
        } else {
            // rule token is not a placeholder, so we have to skip the query tokens until it matches rule token
            while(j < query_tokens.size() && query_tokens[j] != rule_tokens[i]) {
                if(exact_rule_match) {
                    // a single mismatch is enough to fail exact match
                    return false;
                }
                j++;
            }

            // either we have exhausted all query tokens
            if(j == query_tokens.size()) {
                return false;
            }

            //  or query token matches rule token, so we can proceed

            i++;
            j++;
        }
    }

    RETURN_EARLY:

    if(!resolved_override || (exact_rule_match && query_tokens.size() != absorbed_tokens.size())) {
        return false;
    }

    // replace placeholder with field_absorbed_tokens in rule_tokens
    for(const auto& kv: field_placeholder_tokens) {
        std::string pattern = "{" + kv.first + "}";
        std::string replacement = StringUtils::join(kv.second, " ");
        StringUtils::replace_all(filter_by_clause, pattern, replacement);
    }

    return true;
}

void Index::process_filter_overrides(const std::vector<const override_t*>& filter_overrides,
                                     std::vector<std::string>& query_tokens,
                                     token_ordering token_order,
                                     filter_node_t*& filter_tree_root,
                                     std::vector<const override_t*>& matched_dynamic_overrides) const {
    std::shared_lock lock(mutex);

    for (auto& override : filter_overrides) {
        if (!override->rule.dynamic_query) {
            // Simple static filtering: add to filter_by and rewrite query if needed.
            // Check the original query and then the synonym variants until a rule matches.
            bool resolved_override = static_filter_query_eval(override, query_tokens, filter_tree_root);

            if (resolved_override) {
                if (override->remove_matched_tokens) {
                    std::vector<std::string> rule_tokens;
                    Tokenizer(override->rule.query, true).tokenize(rule_tokens);
                    std::set<std::string> rule_token_set(rule_tokens.begin(), rule_tokens.end());
                    remove_matched_tokens(query_tokens, rule_token_set);
                }

                if (override->stop_processing) {
                    return;
                }
            }
        } else {
            // need to extract placeholder field names from the search query, filter on them and rewrite query
            // we will cover both original query and synonyms

            std::vector<std::string> rule_parts;
            StringUtils::split(override->rule.normalized_query, rule_parts, " ");

            uint32_t* field_override_ids = nullptr;
            size_t field_override_ids_len = 0;

            bool exact_rule_match = override->rule.match == override_t::MATCH_EXACT;
            std::string filter_by_clause = override->filter_by;

            std::set<std::string> absorbed_tokens;
            bool resolved_override = resolve_override(rule_parts, exact_rule_match, query_tokens,
                                                      token_order, absorbed_tokens, filter_by_clause);

            if (resolved_override) {
                filter_node_t* new_filter_tree_root = nullptr;
                Option<bool> filter_op = filter::parse_filter_query(filter_by_clause, search_schema,
                                                                    store, "", new_filter_tree_root);
                if (filter_op.ok()) {
                    // have to ensure that dropped hits take precedence over added hits
                    matched_dynamic_overrides.push_back(override);

                    if (override->remove_matched_tokens) {
                        std::vector<std::string>& tokens = query_tokens;
                        remove_matched_tokens(tokens, absorbed_tokens);
                    }

                    if (filter_tree_root == nullptr) {
                        filter_tree_root = new_filter_tree_root;
                    } else {
                        filter_node_t* root = new filter_node_t(AND, filter_tree_root,
                                                                new_filter_tree_root);
                        filter_tree_root = root;
                    }
                } else {
                    delete new_filter_tree_root;
                }

                if (override->stop_processing) {
                    return;
                }
            }
        }
    }
}

void Index::remove_matched_tokens(std::vector<std::string>& tokens, const std::set<std::string>& rule_token_set) {
    std::vector<std::string> new_tokens;

    for(std::string& token: tokens) {
        if(rule_token_set.count(token) == 0) {
            new_tokens.push_back(token);
        }
    }

    if(new_tokens.empty()) {
        tokens = {"*"};
    } else {
        tokens = new_tokens;
    }
}

bool Index::check_for_overrides(const token_ordering& token_order, const string& field_name, const bool slide_window,
                                bool exact_rule_match, std::vector<std::string>& tokens,
                                std::set<std::string>& absorbed_tokens,
                                std::vector<std::string>& field_absorbed_tokens) const {

    for(size_t window_len = tokens.size(); window_len > 0; window_len--) {
        for(size_t start_index = 0; start_index+window_len-1 < tokens.size(); start_index++) {
            std::vector<token_t> window_tokens;
            std::set<std::string> window_tokens_set;
            for (size_t i = start_index; i < start_index + window_len; i++) {
                bool is_prefix = (i == (start_index + window_len - 1));
                window_tokens.emplace_back(i, tokens[i], is_prefix, tokens[i].size(), 0);
                window_tokens_set.emplace(tokens[i]);
            }

            std::vector<facet> facets;
            std::vector<std::vector<art_leaf*>> searched_queries;
            Topster* topster = nullptr;
            spp::sparse_hash_map<uint64_t, uint32_t> groups_processed;
            uint32_t* result_ids = nullptr;
            size_t result_ids_len = 0;
            size_t field_num_results = 0;
            std::vector<std::string> group_by_fields;
            std::set<uint64> query_hashes;

            size_t num_toks_dropped = 0;

            auto field_it = search_schema.find(field_name);
            if(field_it == search_schema.end()) {
                continue;
            }

            std::vector<sort_by> sort_fields;
            search_field(0, window_tokens, nullptr, 0, num_toks_dropped, field_it.value(), field_name,
                         nullptr, 0, {}, sort_fields, -1, 0, searched_queries, topster, groups_processed,
                         &result_ids, result_ids_len, field_num_results, 0, group_by_fields,
                         false, 4, query_hashes, token_order, false, 0, 0, false, -1, 3, 7, 4);

            if(result_ids_len != 0) {
                // we need to narraw onto the exact matches
                std::vector<void*> posting_lists;
                art_tree* t = search_index.at(field_name);

                for(auto& w_token: window_tokens) {
                    art_leaf* leaf = (art_leaf *) art_search(t, (const unsigned char*) w_token.value.c_str(),
                                                             w_token.value.length()+1);
                    if(leaf == nullptr) {
                        continue;
                    }

                    posting_lists.push_back(leaf->values);
                }

                uint32_t* exact_strt_ids = new uint32_t[result_ids_len];
                size_t exact_strt_size = 0;

                posting_t::get_exact_matches(posting_lists, field_it.value().is_array(), result_ids, result_ids_len,
                                             exact_strt_ids, exact_strt_size);

                delete [] result_ids;
                delete [] exact_strt_ids;

                if(exact_strt_size != 0) {
                    // remove window_tokens from `tokens`
                    std::vector<std::string> new_tokens;
                    for(size_t new_i = start_index; new_i < tokens.size(); new_i++) {
                        const auto& token = tokens[new_i];
                        if(window_tokens_set.count(token) == 0) {
                            new_tokens.emplace_back(token);
                        } else {
                            absorbed_tokens.insert(token);
                            field_absorbed_tokens.emplace_back(token);
                        }
                    }

                    tokens = new_tokens;
                    return true;
                }
            }

            if(!slide_window) {
                break;
            }
        }
    }

    return false;
}

void Index::search_infix(const std::string& query, const std::string& field_name,
                         std::vector<uint32_t>& ids, const size_t max_extra_prefix, const size_t max_extra_suffix) const {

    auto infix_maps_it = infix_index.find(field_name);

    if(infix_maps_it == infix_index.end()) {
        return ;
    }

    auto infix_sets = infix_maps_it->second;
    std::vector<art_leaf*> leaves;

    size_t num_processed = 0;
    std::mutex m_process;
    std::condition_variable cv_process;

    auto search_tree = search_index.at(field_name);

    const auto parent_search_begin = search_begin_us;
    const auto parent_search_stop_ms = search_stop_us;
    auto parent_search_cutoff = search_cutoff;

    for(auto infix_set: infix_sets) {
        thread_pool->enqueue([infix_set, &leaves, search_tree, &query, max_extra_prefix, max_extra_suffix,
                                     &num_processed, &m_process, &cv_process,
                                     &parent_search_begin, &parent_search_stop_ms, &parent_search_cutoff]() {

            search_begin_us = parent_search_begin;
            search_cutoff = parent_search_cutoff;
            auto op_search_stop_ms = parent_search_stop_ms/2;

            std::vector<art_leaf*> this_leaves;
            std::string key_buffer;
            size_t num_iterated = 0;

            for(auto it = infix_set->begin(); it != infix_set->end(); it++) {
                it.key(key_buffer);
                num_iterated++;

                auto start_index = key_buffer.find(query);
                if(start_index != std::string::npos && start_index <= max_extra_prefix &&
                   (key_buffer.size() - (start_index + query.size())) <= max_extra_suffix) {
                    art_leaf* l = (art_leaf *) art_search(search_tree,
                                                          (const unsigned char *) key_buffer.c_str(),
                                                          key_buffer.size()+1);
                    if(l != nullptr) {
                        this_leaves.push_back(l);
                    }
                }

                // check for search cutoff but only once every 2^10 docs to reduce overhead
                if(((num_iterated + 1) % (1 << 12)) == 0) {
                    if ((std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().
                        time_since_epoch()).count() - search_begin_us) > op_search_stop_ms) {
                        search_cutoff = true;
                        break;
                    }
                }
            }

            std::unique_lock<std::mutex> lock(m_process);
            leaves.insert(leaves.end(), this_leaves.begin(), this_leaves.end());
            num_processed++;
            parent_search_cutoff = parent_search_cutoff || search_cutoff;
            cv_process.notify_one();
        });
    }

    std::unique_lock<std::mutex> lock_process(m_process);
    cv_process.wait(lock_process, [&](){ return num_processed == infix_sets.size(); });
    search_cutoff = parent_search_cutoff;

    for(auto leaf: leaves) {
        posting_t::merge({leaf->values}, ids);
    }
}

Option<bool> Index::search(std::vector<query_tokens_t>& field_query_tokens, const std::vector<search_field_t>& the_fields,
                   const text_match_type_t match_type,
                   filter_node_t* filter_tree_root, std::vector<facet>& facets, facet_query_t& facet_query,
                   const std::vector<std::pair<uint32_t, uint32_t>>& included_ids,
                   const std::vector<uint32_t>& excluded_ids, std::vector<sort_by>& sort_fields_std,
                   const std::vector<uint32_t>& num_typos, Topster* topster, Topster* curated_topster,
                   const size_t per_page,
                   const size_t offset, const token_ordering token_order, const std::vector<bool>& prefixes,
                   const size_t drop_tokens_threshold, size_t& all_result_ids_len,
                   spp::sparse_hash_map<uint64_t, uint32_t>& groups_processed,
                   std::vector<std::vector<art_leaf*>>& searched_queries,
                   tsl::htrie_map<char, token_leaf>& qtoken_set,
                   std::vector<std::vector<KV*>>& raw_result_kvs, std::vector<std::vector<KV*>>& override_result_kvs,
                   const size_t typo_tokens_threshold, const size_t group_limit,
                   const std::vector<std::string>& group_by_fields,
                   const string& default_sorting_field, bool prioritize_exact_match,
                   const bool prioritize_token_position, bool exhaustive_search,
                   size_t concurrency, size_t search_cutoff_ms, size_t min_len_1typo, size_t min_len_2typo,
                   size_t max_candidates, const std::vector<enable_t>& infixes, const size_t max_extra_prefix,
                   const size_t max_extra_suffix, const size_t facet_query_num_typos,
                   const bool filter_curated_hits, const enable_t split_join_tokens,
                   const vector_query_t& vector_query,
                   size_t facet_sample_percent, size_t facet_sample_threshold,
                   const std::string& collection_name) const {
    std::shared_lock lock(mutex);

    uint32_t filter_ids_length = 0;
    auto rearrange_op = rearrange_filter_tree(filter_tree_root, filter_ids_length, collection_name);
    if (!rearrange_op.ok()) {
        return rearrange_op;
    }

    filter_result_t filter_result;
    auto filter_op = recursive_filter(filter_tree_root, filter_result, collection_name);
    if (!filter_op.ok()) {
        return filter_op;
    }

    if (filter_tree_root != nullptr && filter_result.count == 0) {
        return Option(true);
    }

    size_t fetch_size = offset + per_page;

    std::set<uint32_t> curated_ids;
    std::map<size_t, std::map<size_t, uint32_t>> included_ids_map;  // outer pos => inner pos => list of IDs
    std::vector<uint32_t> included_ids_vec;
    std::unordered_set<uint32_t> excluded_group_ids;

    process_curated_ids(included_ids, excluded_ids, group_by_fields, group_limit, filter_curated_hits,
                        filter_result.docs, filter_result.count, curated_ids, included_ids_map,
                        included_ids_vec, excluded_group_ids);

    std::vector<uint32_t> curated_ids_sorted(curated_ids.begin(), curated_ids.end());
    std::sort(curated_ids_sorted.begin(), curated_ids_sorted.end());

    // Order of `fields` are used to sort results
    // auto begin = std::chrono::high_resolution_clock::now();
    uint32_t* all_result_ids = nullptr;

    const size_t num_search_fields = std::min(the_fields.size(), (size_t) FIELD_LIMIT_NUM);

    // handle exclusion of tokens/phrases
    uint32_t* exclude_token_ids = nullptr;
    size_t exclude_token_ids_size = 0;
    handle_exclusion(num_search_fields, field_query_tokens, the_fields, exclude_token_ids, exclude_token_ids_size);

    int sort_order[3];  // 1 or -1 based on DESC or ASC respectively
    std::array<spp::sparse_hash_map<uint32_t, int64_t>*, 3> field_values;
    std::vector<size_t> geopoint_indices;
    populate_sort_mapping(sort_order, geopoint_indices, sort_fields_std, field_values);

    // Prepare excluded document IDs that we can later remove from the result set
    uint32_t* excluded_result_ids = nullptr;
    size_t excluded_result_ids_size = ArrayUtils::or_scalar(exclude_token_ids, exclude_token_ids_size,
                                                            &curated_ids_sorted[0], curated_ids_sorted.size(),
                                                            &excluded_result_ids);

    auto is_wildcard_query = !field_query_tokens.empty() && !field_query_tokens[0].q_include_tokens.empty() &&
                             field_query_tokens[0].q_include_tokens[0].value == "*";


    // handle phrase searches
    if (!field_query_tokens[0].q_phrases.empty()) {
        do_phrase_search(num_search_fields, the_fields, field_query_tokens,
                         sort_fields_std, searched_queries, group_limit, group_by_fields,
                         topster, sort_order, field_values, geopoint_indices, curated_ids_sorted,
                         all_result_ids, all_result_ids_len, groups_processed, curated_ids,
                         excluded_result_ids, excluded_result_ids_size, excluded_group_ids, curated_topster,
                         included_ids_map, is_wildcard_query,
                         filter_result.docs, filter_result.count);
        if (filter_result.count == 0) {
            goto process_search_results;
        }
    }

    // for phrase query, parser will set field_query_tokens to "*", need to handle that
    if (is_wildcard_query && field_query_tokens[0].q_phrases.empty()) {
        const uint8_t field_id = (uint8_t)(FIELD_LIMIT_NUM - 0);
        bool no_filters_provided = (filter_tree_root == nullptr && filter_result.count == 0);

        if(no_filters_provided && facets.empty() && curated_ids.empty() && vector_query.field_name.empty() &&
           sort_fields_std.size() == 1 && sort_fields_std[0].name == sort_field_const::seq_id &&
           sort_fields_std[0].order == sort_field_const::desc) {
            // optimize for this path specifically
            std::vector<uint32_t> result_ids;
            auto it = seq_ids->new_rev_iterator();
            while (it.valid()) {
                uint32_t seq_id = it.id();
                uint64_t distinct_id = seq_id;
                if (group_limit != 0) {
                    distinct_id = get_distinct_id(group_by_fields, seq_id);
                    if(excluded_group_ids.count(distinct_id) != 0) {
                        continue;
                    }
                }

                int64_t scores[3] = {0};
                scores[0] = seq_id;
                int64_t match_score_index = -1;

                result_ids.push_back(seq_id);

                KV kv(searched_queries.size(), seq_id, distinct_id, match_score_index, scores, nullptr);
                int ret = topster->add(&kv);

                if(group_limit != 0 && ret < 2) {
                    groups_processed[distinct_id]++;
                }

                if (result_ids.size() == fetch_size) {
                    break;
                }

                it.previous();
            }

            all_result_ids_len = seq_ids->num_ids();
            goto process_search_results;
        }

        // if filters were not provided, use the seq_ids index to generate the
        // list of all document ids
        if (no_filters_provided) {
            filter_result.count = seq_ids->num_ids();
            filter_result.docs = seq_ids->uncompress();
        }

        curate_filtered_ids(curated_ids, excluded_result_ids,
                            excluded_result_ids_size, filter_result.docs, filter_result.count, curated_ids_sorted);
        collate_included_ids({}, included_ids_map, curated_topster, searched_queries);

        if (!vector_query.field_name.empty()) {
            auto k = std::max<size_t>(vector_query.k, fetch_size);
            if(vector_query.query_doc_given) {
                // since we will omit the query doc from results
                k++;
            }

            VectorFilterFunctor filterFunctor(filter_result.docs, filter_result.count);
            auto& field_vector_index = vector_index.at(vector_query.field_name);

            std::vector<std::pair<float, size_t>> dist_labels;

            if(!no_filters_provided && filter_result.count < vector_query.flat_search_cutoff) {
                for(size_t i = 0; i < filter_result.count; i++) {
                    auto seq_id = filter_result.docs[i];
                    std::vector<float> values;

                    try {
                        values = field_vector_index->vecdex->getDataByLabel<float>(seq_id);
                    } catch(...) {
                        // likely not found
                        continue;
                    }

                    float dist;
                    if(field_vector_index->distance_type == cosine) {
                        std::vector<float> normalized_q(vector_query.values.size());
                        hnsw_index_t::normalize_vector(vector_query.values, normalized_q);
                        dist = field_vector_index->space->get_dist_func()(normalized_q.data(), values.data(),
                                                                           &field_vector_index->num_dim);
                    } else {
                        dist = field_vector_index->space->get_dist_func()(vector_query.values.data(), values.data(),
                                                                           &field_vector_index->num_dim);
                    }

                    dist_labels.emplace_back(dist, seq_id);
                }
            } else {
                if(field_vector_index->distance_type == cosine) {
                    std::vector<float> normalized_q(vector_query.values.size());
                    hnsw_index_t::normalize_vector(vector_query.values, normalized_q);
                    dist_labels = field_vector_index->vecdex->searchKnnCloserFirst(normalized_q.data(), k, &filterFunctor);
                } else {
                    dist_labels = field_vector_index->vecdex->searchKnnCloserFirst(vector_query.values.data(), k, &filterFunctor);
                }
            }

            std::vector<uint32_t> nearest_ids;

            for (const auto& dist_label : dist_labels) {
                uint32 seq_id = dist_label.second;

                if(vector_query.query_doc_given && vector_query.seq_id == seq_id) {
                    continue;
                }

                uint64_t distinct_id = seq_id;
                if (group_limit != 0) {
                    distinct_id = get_distinct_id(group_by_fields, seq_id);
                    if(excluded_group_ids.count(distinct_id) != 0) {
                        continue;
                    }
                }

                auto vec_dist_score = (field_vector_index->distance_type == cosine) ? std::abs(dist_label.first) :
                                      dist_label.first;
                                      
                if(vec_dist_score > vector_query.distance_threshold) {
                    continue;
                }

                int64_t scores[3] = {0};
                int64_t match_score_index = -1;

                compute_sort_scores(sort_fields_std, sort_order, field_values, geopoint_indices, seq_id, 0, 0, scores, match_score_index, vec_dist_score);

                KV kv(searched_queries.size(), seq_id, distinct_id, match_score_index, scores, nullptr);
                kv.vector_distance = vec_dist_score;
                int ret = topster->add(&kv);

                if(group_limit != 0 && ret < 2) {
                    groups_processed[distinct_id]++;
                }
                nearest_ids.push_back(seq_id);
            }

            if(!nearest_ids.empty()) {
                std::sort(nearest_ids.begin(), nearest_ids.end());  // seq_ids should be in ascending order
                all_result_ids = new uint32[nearest_ids.size()];
                std::copy(nearest_ids.begin(), nearest_ids.end(), all_result_ids);
                all_result_ids_len = nearest_ids.size();
            }
        } else {
            search_wildcard(filter_tree_root, included_ids_map, sort_fields_std, topster,
                            curated_topster, groups_processed, searched_queries, group_limit, group_by_fields,
                            curated_ids, curated_ids_sorted,
                            excluded_result_ids, excluded_result_ids_size, excluded_group_ids,
                            all_result_ids, all_result_ids_len,
                            filter_result.docs, filter_result.count, concurrency,
                            sort_order, field_values, geopoint_indices);
        }
    } else {
        // Non-wildcard
        // In multi-field searches, a record can be matched across different fields, so we use this for aggregation
        //begin = std::chrono::high_resolution_clock::now();

        // FIXME: needed?
        std::set<uint64> query_hashes;

        // resolve synonyms so that we can compute `syn_orig_num_tokens`
        std::vector<std::vector<token_t>> all_queries = {field_query_tokens[0].q_include_tokens};
        std::vector<std::vector<token_t>> q_pos_synonyms;
        std::vector<std::string> q_include_tokens;
        int syn_orig_num_tokens = -1;

        for(size_t j = 0; j < field_query_tokens[0].q_include_tokens.size(); j++) {
            q_include_tokens.push_back(field_query_tokens[0].q_include_tokens[j].value);
        }
        synonym_index->synonym_reduction(q_include_tokens, field_query_tokens[0].q_synonyms);

        if(!field_query_tokens[0].q_synonyms.empty()) {
            syn_orig_num_tokens = field_query_tokens[0].q_include_tokens.size();
        }

        for(const auto& q_syn_vec: field_query_tokens[0].q_synonyms) {
            std::vector<token_t> q_pos_syn;
            for(size_t j=0; j < q_syn_vec.size(); j++) {
                bool is_prefix = (j == q_syn_vec.size()-1);
                q_pos_syn.emplace_back(j, q_syn_vec[j], is_prefix, q_syn_vec[j].size(), 0);
            }

            q_pos_synonyms.push_back(q_pos_syn);
            all_queries.push_back(q_pos_syn);

            if((int)q_syn_vec.size() > syn_orig_num_tokens) {
                syn_orig_num_tokens = (int) q_syn_vec.size();
            }
        }

        fuzzy_search_fields(the_fields, field_query_tokens[0].q_include_tokens, {}, match_type, excluded_result_ids,
                            excluded_result_ids_size, filter_result.docs, filter_result.count, curated_ids_sorted,
                            excluded_group_ids,
                            sort_fields_std, num_typos, searched_queries, qtoken_set, topster, groups_processed,
                            all_result_ids, all_result_ids_len, group_limit, group_by_fields, prioritize_exact_match,
                            prioritize_token_position, query_hashes, token_order, prefixes,
                            typo_tokens_threshold, exhaustive_search,
                            max_candidates, min_len_1typo, min_len_2typo, syn_orig_num_tokens, sort_order,
                            field_values, geopoint_indices);

        // try split/joining tokens if no results are found
        if(split_join_tokens == always || (all_result_ids_len == 0 && split_join_tokens == fallback)) {
            std::vector<std::vector<std::string>> space_resolved_queries;

            for (size_t i = 0; i < num_search_fields; i++) {
                std::vector<std::string> orig_q_include_tokens;
                for(auto& q_include_token: field_query_tokens[i].q_include_tokens) {
                    orig_q_include_tokens.push_back(q_include_token.value);
                }

                resolve_space_as_typos(orig_q_include_tokens, the_fields[i].name,space_resolved_queries);

                if (!space_resolved_queries.empty()) {
                    break;
                }
            }

            // only one query is resolved for now, so just use that
            if (!space_resolved_queries.empty()) {
                const auto& resolved_query = space_resolved_queries[0];
                std::vector<token_t> resolved_tokens;

                for(size_t j=0; j < resolved_query.size(); j++) {
                    bool is_prefix = (j == resolved_query.size()-1);
                    resolved_tokens.emplace_back(j, space_resolved_queries[0][j], is_prefix,
                                                 space_resolved_queries[0][j].size(), 0);
                }

                fuzzy_search_fields(the_fields, resolved_tokens, {}, match_type, excluded_result_ids,
                                    excluded_result_ids_size, filter_result.docs, filter_result.count, curated_ids_sorted,
                                    excluded_group_ids,
                                    sort_fields_std, num_typos, searched_queries, qtoken_set, topster, groups_processed,
                                    all_result_ids, all_result_ids_len, group_limit, group_by_fields, prioritize_exact_match,
                                    prioritize_token_position, query_hashes, token_order, prefixes, typo_tokens_threshold, exhaustive_search,
                                    max_candidates, min_len_1typo, min_len_2typo, syn_orig_num_tokens, sort_order, field_values, geopoint_indices);
            }
        }

        // do synonym based searches
        do_synonym_search(the_fields, match_type, filter_tree_root, included_ids_map, sort_fields_std,
                          curated_topster, token_order,
                          0, group_limit, group_by_fields, prioritize_exact_match, prioritize_token_position,
                          exhaustive_search, concurrency, prefixes,
                          min_len_1typo, min_len_2typo, max_candidates, curated_ids, curated_ids_sorted,
                          excluded_result_ids, excluded_result_ids_size, excluded_group_ids,
                          topster, q_pos_synonyms, syn_orig_num_tokens,
                          groups_processed, searched_queries, all_result_ids, all_result_ids_len,
                          filter_result.docs, filter_result.count, query_hashes,
                          sort_order, field_values, geopoint_indices,
                          qtoken_set);

        // gather up both original query and synonym queries and do drop tokens

        if (exhaustive_search || all_result_ids_len < drop_tokens_threshold) {
            for (size_t qi = 0; qi < all_queries.size(); qi++) {
                auto& orig_tokens = all_queries[qi];
                size_t num_tokens_dropped = 0;

                while(exhaustive_search || all_result_ids_len < drop_tokens_threshold) {
                    // When atleast two tokens from the query are available we can drop one
                    std::vector<token_t> truncated_tokens;
                    std::vector<token_t> dropped_tokens;

                    if(orig_tokens.size() > 1 && num_tokens_dropped < 2*(orig_tokens.size()-1)) {
                        bool prefix_search = false;

                        if (num_tokens_dropped < orig_tokens.size() - 1) {
                            // drop from right
                            size_t truncated_len = orig_tokens.size() - num_tokens_dropped - 1;
                            for (size_t i = 0; i < orig_tokens.size(); i++) {
                                if(i < truncated_len) {
                                    truncated_tokens.emplace_back(orig_tokens[i]);
                                } else {
                                    dropped_tokens.emplace_back(orig_tokens[i]);
                                }
                            }
                        } else {
                            // drop from left
                            prefix_search = true;
                            size_t start_index = (num_tokens_dropped + 1) - orig_tokens.size() + 1;
                            for(size_t i = 0; i < orig_tokens.size(); i++) {
                                if(i >= start_index) {
                                    truncated_tokens.emplace_back(orig_tokens[i]);
                                } else {
                                    dropped_tokens.emplace_back(orig_tokens[i]);
                                }
                            }
                        }

                        num_tokens_dropped++;
                        std::vector<bool> drop_token_prefixes;

                        for (const auto p : prefixes) {
                            drop_token_prefixes.push_back(p && prefix_search);
                        }

                        fuzzy_search_fields(the_fields, truncated_tokens, dropped_tokens, match_type,
                                            excluded_result_ids, excluded_result_ids_size,
                                            filter_result.docs, filter_result.count,
                                            curated_ids_sorted, excluded_group_ids, sort_fields_std, num_typos, searched_queries,
                                            qtoken_set, topster, groups_processed,
                                            all_result_ids, all_result_ids_len, group_limit, group_by_fields,
                                            prioritize_exact_match, prioritize_token_position, query_hashes,
                                            token_order, prefixes, typo_tokens_threshold,
                                            exhaustive_search, max_candidates, min_len_1typo,
                                            min_len_2typo, -1, sort_order, field_values, geopoint_indices);

                    } else {
                        break;
                    }
                }
            }
        }

        do_infix_search(num_search_fields, the_fields, infixes, sort_fields_std, searched_queries,
                        group_limit, group_by_fields,
                        max_extra_prefix, max_extra_suffix,
                        field_query_tokens[0].q_include_tokens,
                        topster, filter_result.docs, filter_result.count,
                        sort_order, field_values, geopoint_indices,
                        curated_ids_sorted, excluded_group_ids, all_result_ids, all_result_ids_len, groups_processed);

        if(!vector_query.field_name.empty()) {
            // check at least one of sort fields is text match
            bool has_text_match = false;
            for(auto& sort_field : sort_fields_std) {
                if(sort_field.name == sort_field_const::text_match) {
                    has_text_match = true;
                    break;
                }
            }

            if(has_text_match) {
                // For hybrid search, we need to give weight to text match and vector search
                constexpr float TEXT_MATCH_WEIGHT = 0.7;
                constexpr float VECTOR_SEARCH_WEIGHT = 1.0 - TEXT_MATCH_WEIGHT;

                VectorFilterFunctor filterFunctor(filter_result.docs, filter_result.count);
                auto& field_vector_index = vector_index.at(vector_query.field_name);
                std::vector<std::pair<float, size_t>> dist_labels;
                // use k as 100 by default for ensuring results stability in pagination
                size_t default_k = 100;
                auto k = std::max<size_t>(vector_query.k, default_k);

                if(field_vector_index->distance_type == cosine) {
                    std::vector<float> normalized_q(vector_query.values.size());
                    hnsw_index_t::normalize_vector(vector_query.values, normalized_q);
                    dist_labels = field_vector_index->vecdex->searchKnnCloserFirst(normalized_q.data(), k, &filterFunctor);
                } else {
                    dist_labels = field_vector_index->vecdex->searchKnnCloserFirst(vector_query.values.data(), k, &filterFunctor);
                }

                std::vector<std::pair<uint32_t,float>> vec_results;
                for (const auto& dist_label : dist_labels) {
                    uint32_t seq_id = dist_label.second;

                    auto vec_dist_score = (field_vector_index->distance_type == cosine) ? std::abs(dist_label.first) :
                                            dist_label.first;
                    if(vec_dist_score > vector_query.distance_threshold) {
                        continue;
                    }
                    vec_results.emplace_back(seq_id, vec_dist_score);
                }
                
                std::sort(vec_results.begin(), vec_results.end(), [](const auto& a, const auto& b) {
                    return a.second < b.second;
                });

                topster->sort();
                // Reciprocal rank fusion
                // Score is  sum of (1 / rank_of_document) * WEIGHT from each list (text match and vector search)
                for(uint32_t i = 0; i < topster->size; i++) {
                    auto result = topster->getKV(i);
                    if(result->match_score_index < 0 || result->match_score_index > 2) {
                        continue;
                    }
                    // (1 / rank_of_document) * WEIGHT)
                    result->text_match_score = result->scores[result->match_score_index];   
                    result->scores[result->match_score_index] = float_to_int64_t((1.0 / (i + 1)) * TEXT_MATCH_WEIGHT);
                }

                std::vector<uint32_t> vec_search_ids;  // list of IDs found only in vector search

                for(size_t res_index = 0; res_index < vec_results.size(); res_index++) {
                    auto& vec_result = vec_results[res_index];
                    auto doc_id = vec_result.first;
                    auto result_it = topster->kv_map.find(doc_id);

                    if(result_it != topster->kv_map.end()) {
                        if(result_it->second->match_score_index < 0 || result_it->second->match_score_index > 2) {
                            continue;
                        }

                        // result overlaps with keyword search: we have to combine the scores

                        auto result = result_it->second;
                        // old_score + (1 / rank_of_document) * WEIGHT)
                        result->vector_distance = vec_result.second;
                        result->scores[result->match_score_index] = float_to_int64_t(
                                (int64_t_to_float(result->scores[result->match_score_index])) +
                                ((1.0 / (res_index + 1)) * VECTOR_SEARCH_WEIGHT));

                        for(size_t i = 0;i < 3; i++) {
                            if(field_values[i] == &vector_distance_sentinel_value) {
                                result->scores[i] = float_to_int64_t(vec_result.second);
                            }

                            if(sort_order[i] == -1) {
                                result->scores[i] = -result->scores[i];
                            }
                        }

                    } else {
                        // Result has been found only in vector search: we have to add it to both KV and result_ids
                        // (1 / rank_of_document) * WEIGHT)
                        int64_t scores[3] = {0};
                        int64_t match_score = float_to_int64_t((1.0 / (res_index + 1)) * VECTOR_SEARCH_WEIGHT);
                        int64_t match_score_index = -1;
                        compute_sort_scores(sort_fields_std, sort_order, field_values, geopoint_indices, doc_id, 0, match_score, scores, match_score_index, vec_result.second);
                        KV kv(searched_queries.size(), doc_id, doc_id, match_score_index, scores);
                        kv.vector_distance = vec_result.second;
                        topster->add(&kv);
                        vec_search_ids.push_back(doc_id);
                    }
                }

                if(!vec_search_ids.empty()) {
                    uint32_t* new_all_result_ids = nullptr;
                    all_result_ids_len = ArrayUtils::or_scalar(all_result_ids, all_result_ids_len, &vec_search_ids[0],
                                                               vec_search_ids.size(), &new_all_result_ids);
                    delete[] all_result_ids;
                    all_result_ids = new_all_result_ids;
                }
            }
        }

        /*auto timeMillis0 = std::chrono::duration_cast<std::chrono::milliseconds>(
                 std::chrono::high_resolution_clock::now() - begin0).count();
         LOG(INFO) << "Time taken for multi-field aggregation: " << timeMillis0 << "ms";*/

    }

    //LOG(INFO) << "topster size: " << topster->size;

    process_search_results:

    delete [] exclude_token_ids;
    delete [] excluded_result_ids;

    bool estimate_facets = (facet_sample_percent < 100 && all_result_ids_len > facet_sample_threshold);

    if(!facets.empty()) {
        const size_t num_threads = std::min(concurrency, all_result_ids_len);
        const size_t window_size = (num_threads == 0) ? 0 :
                                   (all_result_ids_len + num_threads - 1) / num_threads;  // rounds up
        size_t num_processed = 0;
        std::mutex m_process;
        std::condition_variable cv_process;

        std::vector<facet_info_t> facet_infos(facets.size());
        compute_facet_infos(facets, facet_query, facet_query_num_typos, all_result_ids, all_result_ids_len,
                            group_by_fields, max_candidates, facet_infos);

        std::vector<std::vector<facet>> facet_batches(num_threads);
        for(size_t i = 0; i < num_threads; i++) {
            for(const auto& this_facet: facets) {
                facet_batches[i].emplace_back(facet(this_facet.field_name, this_facet.facet_range_map, this_facet.is_range_query));
            }
        }

        size_t num_queued = 0;
        size_t result_index = 0;

        const auto parent_search_begin = search_begin_us;
        const auto parent_search_stop_ms = search_stop_us;
        auto parent_search_cutoff = search_cutoff;

        //auto beginF = std::chrono::high_resolution_clock::now();

        for(size_t thread_id = 0; thread_id < num_threads && result_index < all_result_ids_len; thread_id++) {
            size_t batch_res_len = window_size;

            if(result_index + window_size > all_result_ids_len) {
                batch_res_len = all_result_ids_len - result_index;
            }

            uint32_t* batch_result_ids = all_result_ids + result_index;
            num_queued++;

            thread_pool->enqueue([this, thread_id, &facet_batches, &facet_query, group_limit, group_by_fields,
                                         batch_result_ids, batch_res_len, &facet_infos,
                                         estimate_facets, facet_sample_percent,
                                         &parent_search_begin, &parent_search_stop_ms, &parent_search_cutoff,
                                         &num_processed, &m_process, &cv_process]() {
                search_begin_us = parent_search_begin;
                search_stop_us = parent_search_stop_ms;
                search_cutoff = parent_search_cutoff;

                auto fq = facet_query;
                do_facets(facet_batches[thread_id], fq, estimate_facets, facet_sample_percent,
                          facet_infos, group_limit, group_by_fields,
                          batch_result_ids, batch_res_len);
                std::unique_lock<std::mutex> lock(m_process);
                num_processed++;
                parent_search_cutoff = parent_search_cutoff || search_cutoff;
                cv_process.notify_one();
            });

            result_index += batch_res_len;
        }

        std::unique_lock<std::mutex> lock_process(m_process);
        cv_process.wait(lock_process, [&](){ return num_processed == num_queued; });
        search_cutoff = parent_search_cutoff;

        for(auto& facet_batch: facet_batches) {
            for(size_t fi = 0; fi < facet_batch.size(); fi++) {
                auto& this_facet = facet_batch[fi];
                auto& acc_facet = facets[fi];

                for(auto & facet_kv: this_facet.result_map) {
                    if(group_limit) {
                        // we have to add all group sets
                        acc_facet.hash_groups[facet_kv.first].insert(
                            this_facet.hash_groups[facet_kv.first].begin(),
                            this_facet.hash_groups[facet_kv.first].end()
                        );
                    } else {
                        size_t count = 0;
                        if(acc_facet.result_map.count(facet_kv.first) == 0) {
                            // not found, so set it
                            count = facet_kv.second.count;
                        } else {
                            count = acc_facet.result_map[facet_kv.first].count + facet_kv.second.count;
                        }
                        acc_facet.result_map[facet_kv.first].count = count;
                    }

                    acc_facet.result_map[facet_kv.first].doc_id = facet_kv.second.doc_id;
                    acc_facet.result_map[facet_kv.first].array_pos = facet_kv.second.array_pos;
                    acc_facet.hash_tokens[facet_kv.first] = this_facet.hash_tokens[facet_kv.first];
                }

                if(this_facet.stats.fvcount != 0) {
                    acc_facet.stats.fvcount += this_facet.stats.fvcount;
                    acc_facet.stats.fvsum += this_facet.stats.fvsum;
                    acc_facet.stats.fvmax = std::max(acc_facet.stats.fvmax, this_facet.stats.fvmax);
                    acc_facet.stats.fvmin = std::min(acc_facet.stats.fvmin, this_facet.stats.fvmin);
                }
            }
        }

        for(auto & acc_facet: facets) {
            for(auto& facet_kv: acc_facet.result_map) {
                if(group_limit) {
                    facet_kv.second.count = acc_facet.hash_groups[facet_kv.first].size();
                }

                if(estimate_facets) {
                    facet_kv.second.count = size_t(double(facet_kv.second.count) * (100.0f / facet_sample_percent));
                }
            }

            if(estimate_facets) {
                acc_facet.sampled = true;
            }
        }

        /*long long int timeMillisF = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - beginF).count();
        LOG(INFO) << "Time for faceting: " << timeMillisF;*/
    }

    std::vector<facet_info_t> facet_infos(facets.size());
    compute_facet_infos(facets, facet_query, facet_query_num_typos,
                        &included_ids_vec[0], included_ids_vec.size(), group_by_fields, max_candidates, facet_infos);
    do_facets(facets, facet_query, estimate_facets, facet_sample_percent,
              facet_infos, group_limit, group_by_fields, &included_ids_vec[0], included_ids_vec.size());

    all_result_ids_len += curated_topster->size;

    delete [] all_result_ids;

    //LOG(INFO) << "all_result_ids_len " << all_result_ids_len << " for index " << name;
    //long long int timeMillis = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - begin).count();
    //LOG(INFO) << "Time taken for result calc: " << timeMillis << "ms";

    return Option(true);
}

void Index::process_curated_ids(const std::vector<std::pair<uint32_t, uint32_t>>& included_ids,
                                const std::vector<uint32_t>& excluded_ids,
                                const std::vector<std::string>& group_by_fields, const size_t group_limit,
                                const bool filter_curated_hits, const uint32_t* filter_ids, uint32_t filter_ids_length,
                                std::set<uint32_t>& curated_ids,
                                std::map<size_t, std::map<size_t, uint32_t>>& included_ids_map,
                                std::vector<uint32_t>& included_ids_vec,
                                std::unordered_set<uint32_t>& excluded_group_ids) const {

    for(const auto& seq_id_pos: included_ids) {
        included_ids_vec.push_back(seq_id_pos.first);
    }

    if(group_limit != 0) {
        // if one `id` of a group is present in curated hits, we have to exclude that entire group from results
        for(auto seq_id: included_ids_vec) {
            uint64_t distinct_id = get_distinct_id(group_by_fields, seq_id);
            excluded_group_ids.emplace(distinct_id);
        }
    }

    std::sort(included_ids_vec.begin(), included_ids_vec.end());

    // if `filter_curated_hits` is enabled, we will remove curated hits that don't match filter condition
    std::set<uint32_t> included_ids_set;

    if(filter_ids_length != 0 && filter_curated_hits) {
        uint32_t* included_ids_arr = nullptr;
        size_t included_ids_len = ArrayUtils::and_scalar(&included_ids_vec[0], included_ids_vec.size(), filter_ids,
                                                         filter_ids_length, &included_ids_arr);

        included_ids_vec.clear();

        for(size_t i = 0; i < included_ids_len; i++) {
            included_ids_set.insert(included_ids_arr[i]);
            included_ids_vec.push_back(included_ids_arr[i]);
        }

        delete [] included_ids_arr;
    } else {
        included_ids_set.insert(included_ids_vec.begin(), included_ids_vec.end());
    }

    std::map<size_t, std::vector<uint32_t>> included_ids_grouped;  // pos -> seq_ids
    std::vector<uint32_t> all_positions;

    for(const auto& seq_id_pos: included_ids) {
        all_positions.push_back(seq_id_pos.second);
        if(included_ids_set.count(seq_id_pos.first) == 0) {
            continue;
        }
        included_ids_grouped[seq_id_pos.second].push_back(seq_id_pos.first);
    }


    for(const auto& pos_ids: included_ids_grouped) {
        size_t outer_pos = pos_ids.first;
        size_t ids_per_pos = std::max(size_t(1), group_limit);
        auto num_inner_ids = std::min(ids_per_pos, pos_ids.second.size());

        for(size_t inner_pos = 0; inner_pos < num_inner_ids; inner_pos++) {
            auto seq_id = pos_ids.second[inner_pos];
            included_ids_map[outer_pos][inner_pos] = seq_id;
            curated_ids.insert(seq_id);
        }
    }

    curated_ids.insert(excluded_ids.begin(), excluded_ids.end());

    if(all_positions.size() > included_ids_map.size()) {
        // Some curated IDs may have been removed via filtering or simply don't exist.
        // We have to shift lower placed hits upwards to fill those positions.
        std::sort(all_positions.begin(), all_positions.end());
        all_positions.erase(unique(all_positions.begin(), all_positions.end()), all_positions.end());

        size_t pos_count = 0;
        std::map<size_t, std::map<size_t, uint32_t>> new_included_ids_map;
        auto included_id_it = included_ids_map.begin();
        auto all_pos_it = all_positions.begin();

        while(included_id_it != included_ids_map.end()) {
            new_included_ids_map[*all_pos_it] = included_id_it->second;
            all_pos_it++;
            included_id_it++;
        }

        included_ids_map = new_included_ids_map;
    }
}

void Index::fuzzy_search_fields(const std::vector<search_field_t>& the_fields,
                                const std::vector<token_t>& query_tokens,
                                const std::vector<token_t>& dropped_tokens,
                                const text_match_type_t match_type,
                                const uint32_t* exclude_token_ids,
                                size_t exclude_token_ids_size,
                                const uint32_t* filter_ids, size_t filter_ids_length,
                                const std::vector<uint32_t>& curated_ids,
                                const std::unordered_set<uint32_t>& excluded_group_ids,
                                const std::vector<sort_by> & sort_fields,
                                const std::vector<uint32_t>& num_typos,
                                std::vector<std::vector<art_leaf*>> & searched_queries,
                                tsl::htrie_map<char, token_leaf>& qtoken_set,
                                Topster* topster, spp::sparse_hash_map<uint64_t, uint32_t>& groups_processed,
                                uint32_t*& all_result_ids, size_t & all_result_ids_len,
                                const size_t group_limit, const std::vector<std::string>& group_by_fields,
                                bool prioritize_exact_match,
                                const bool prioritize_token_position,
                                std::set<uint64>& query_hashes,
                                const token_ordering token_order,
                                const std::vector<bool>& prefixes,
                                const size_t typo_tokens_threshold,
                                const bool exhaustive_search,
                                const size_t max_candidates,
                                size_t min_len_1typo,
                                size_t min_len_2typo,
                                int syn_orig_num_tokens,
                                const int* sort_order,
                                std::array<spp::sparse_hash_map<uint32_t, int64_t>*, 3>& field_values,
                                const std::vector<size_t>& geopoint_indices) const {

    // NOTE: `query_tokens` preserve original tokens, while `search_tokens` could be a result of dropped tokens

    // To prevent us from doing ART search repeatedly as we iterate through possible corrections
    spp::sparse_hash_map<std::string, std::vector<std::string>> token_cost_cache;

    std::vector<std::vector<int>> token_to_costs;

    for(size_t stoken_index=0; stoken_index < query_tokens.size(); stoken_index++) {
        const std::string& token = query_tokens[stoken_index].value;

        std::vector<int> all_costs;
        // This ensures that we don't end up doing a cost of 1 for a single char etc.
        int bounded_cost = get_bounded_typo_cost(2, token.length(), min_len_1typo, min_len_2typo);

        for(int cost = 0; cost <= bounded_cost; cost++) {
            all_costs.push_back(cost);
        }

        token_to_costs.push_back(all_costs);
    }

    // stores candidates for each token, i.e. i-th index would have all possible tokens with a cost of "c"
    std::vector<tok_candidates> token_candidates_vec;
    std::set<std::string> unique_tokens;

    const size_t num_search_fields = std::min(the_fields.size(), (size_t) FIELD_LIMIT_NUM);

    auto product = []( long long a, std::vector<int>& b ) { return a*b.size(); };
    long long n = 0;
    long long int N = token_to_costs.size() > 30 ? 1 :
                      std::accumulate(token_to_costs.begin(), token_to_costs.end(), 1LL, product);

    const long long combination_limit = exhaustive_search ? Index::COMBINATION_MAX_LIMIT : Index::COMBINATION_MIN_LIMIT;

    while(n < N && n < combination_limit) {
        RETURN_CIRCUIT_BREAKER

        //LOG(INFO) << "fuzzy_search_fields, n: " << n;

        // Outerloop generates combinations of [cost to max_cost] for each token
        // For e.g. for a 3-token query: [0, 0, 0], [0, 0, 1], [0, 1, 1] etc.
        std::vector<uint32_t> costs(token_to_costs.size());
        ldiv_t q { n, 0 };
        for(long long i = (token_to_costs.size() - 1); 0 <= i ; --i ) {
            q = ldiv(q.quot, token_to_costs[i].size());
            costs[i] = token_to_costs[i][q.rem];
        }

        unique_tokens.clear();
        token_candidates_vec.clear();
        size_t token_index = 0;

        while(token_index < query_tokens.size()) {
            // For each token, look up the generated cost for this iteration and search using that cost
            const std::string& token = query_tokens[token_index].value;
            const std::string token_cost_hash = token + std::to_string(costs[token_index]);

            std::vector<std::string> leaf_tokens;

            if(token_cost_cache.count(token_cost_hash) != 0) {
                leaf_tokens = token_cost_cache[token_cost_hash];
            } else {
                //auto begin = std::chrono::high_resolution_clock::now();

                // Prefix query with a preceding token should be handled in such a way that we give preference to
                // possible phrase continuation. Example: "steve j" for "steve jobs" name field query. To do this,
                // we will first attempt to match the prefix with the most "popular" fields of the preceding token.
                // Tokens matched from popular fields will also be searched across other query fields.
                // Only when we find *no results* for such an expansion, we will attempt cross field matching.
                bool last_token = query_tokens.size() > 1 && dropped_tokens.empty() &&
                                  (token_index == (query_tokens.size() - 1));

                std::vector<size_t> query_field_ids(num_search_fields);
                for(size_t field_id = 0; field_id < num_search_fields; field_id++) {
                    query_field_ids[field_id] = the_fields[field_id].orig_index;
                }

                std::vector<size_t> popular_field_ids; // fields containing the token most across documents

                if(last_token) {
                    popular_fields_of_token(search_index,
                                            token_candidates_vec.back().candidates[0],
                                            the_fields, num_search_fields, popular_field_ids);

                    if(popular_field_ids.empty()) {
                        break;
                    }
                }

                const std::vector<size_t>& field_ids = last_token ? popular_field_ids : query_field_ids;

                for(size_t field_id: field_ids) {
                    // NOTE: when accessing other field ordered properties like prefixes or num_typos we have to index
                    // them by `the_field.orig_index` since the original fields could be reordered on their weights.
                    auto& the_field = the_fields[field_id];
                    const bool field_prefix = (the_field.orig_index < prefixes.size()) ? prefixes[the_field.orig_index] : prefixes[0];;
                    const bool prefix_search = field_prefix && query_tokens[token_index].is_prefix_searched;
                    const size_t token_len = prefix_search ? (int) token.length() : (int) token.length() + 1;

                    /*LOG(INFO) << "Searching for field: " << the_field.name << ", token:"
                              << token << " - cost: " << costs[token_index] << ", prefix_search: " << prefix_search;*/

                    int64_t field_num_typos = (the_field.orig_index < num_typos.size()) ? num_typos[the_field.orig_index] : num_typos[0];

                    auto& locale = search_schema.at(the_field.name).locale;
                    if(locale != "" && (locale == "zh" || locale == "ko" || locale == "ja")) {
                        // disable fuzzy trie traversal for CJK locales
                        field_num_typos = 0;
                    }

                    if(costs[token_index] > field_num_typos) {
                        continue;
                    }

                    //LOG(INFO) << "Searching for field: " << the_field.name << ", found token:" << token;
                    const auto& prev_token = last_token ? token_candidates_vec.back().candidates[0] : "";

                    std::vector<art_leaf*> field_leaves;
                    art_fuzzy_search(search_index.at(the_field.name), (const unsigned char *) token.c_str(), token_len,
                                     costs[token_index], costs[token_index], max_candidates, token_order, prefix_search,
                                     last_token, prev_token, filter_ids, filter_ids_length, field_leaves, unique_tokens);

                    /*auto timeMillis = std::chrono::duration_cast<std::chrono::milliseconds>(
                                    std::chrono::high_resolution_clock::now() - begin).count();
                    LOG(INFO) << "Time taken for fuzzy search: " << timeMillis << "ms";*/

                    if(field_leaves.empty()) {
                        // look at the next field
                        continue;
                    }

                    for(size_t i = 0; i < field_leaves.size(); i++) {
                        auto leaf = field_leaves[i];
                        std::string tok(reinterpret_cast<char*>(leaf->key), leaf->key_len - 1);
                        leaf_tokens.push_back(tok);
                    }

                    token_cost_cache.emplace(token_cost_hash, leaf_tokens);

                    if(leaf_tokens.size() >= max_candidates) {
                        goto token_done;
                    }
                }

                if(last_token && leaf_tokens.size() < max_candidates) {
                    // field-wise matching with previous token has failed, have to look at cross fields matching docs
                    std::vector<uint32_t> prev_token_doc_ids;
                    find_across_fields(token_candidates_vec.back().token,
                                       token_candidates_vec.back().candidates[0],
                                       the_fields, num_search_fields, filter_ids, filter_ids_length, exclude_token_ids,
                                       exclude_token_ids_size, prev_token_doc_ids, popular_field_ids);

                    for(size_t field_id: query_field_ids) {
                        auto& the_field = the_fields[field_id];
                        const bool field_prefix = (the_field.orig_index < prefixes.size()) ? prefixes[the_field.orig_index] : prefixes[0];;
                        const bool prefix_search = field_prefix && query_tokens[token_index].is_prefix_searched;
                        const size_t token_len = prefix_search ? (int) token.length() : (int) token.length() + 1;
                        int64_t field_num_typos = (the_field.orig_index < num_typos.size()) ? num_typos[the_field.orig_index] : num_typos[0];

                        auto& locale = search_schema.at(the_field.name).locale;
                        if(locale != "" && locale != "en" && locale != "th" && !Tokenizer::is_cyrillic(locale)) {
                            // disable fuzzy trie traversal for non-english locales
                            field_num_typos = 0;
                        }

                        if(costs[token_index] > field_num_typos) {
                            continue;
                        }

                        std::vector<art_leaf*> field_leaves;
                        art_fuzzy_search(search_index.at(the_field.name), (const unsigned char *) token.c_str(), token_len,
                                         costs[token_index], costs[token_index], max_candidates, token_order, prefix_search,
                                         false, "", filter_ids, filter_ids_length, field_leaves, unique_tokens);

                        if(field_leaves.empty()) {
                            // look at the next field
                            continue;
                        }

                        for(size_t i = 0; i < field_leaves.size(); i++) {
                            auto leaf = field_leaves[i];
                            std::string tok(reinterpret_cast<char*>(leaf->key), leaf->key_len - 1);
                            leaf_tokens.push_back(tok);
                        }

                        token_cost_cache.emplace(token_cost_hash, leaf_tokens);

                        if(leaf_tokens.size() >= max_candidates) {
                            goto token_done;
                        }
                    }
                }
            }

            token_done:

            if(!leaf_tokens.empty()) {
                //log_leaves(costs[token_index], token, leaves);
                token_candidates_vec.push_back(tok_candidates{query_tokens[token_index], costs[token_index],
                                                              query_tokens[token_index].is_prefix_searched, leaf_tokens});
            } else {
                // No result at `cost = costs[token_index]`. Remove `cost` for token and re-do combinations
                auto it = std::find(token_to_costs[token_index].begin(), token_to_costs[token_index].end(), costs[token_index]);
                if(it != token_to_costs[token_index].end()) {
                    token_to_costs[token_index].erase(it);

                    // when no more costs are left for this token
                    if(token_to_costs[token_index].empty()) {
                        // we cannot proceed further, as this token is not found within cost limits
                        // and, dropping of tokens are done elsewhere.
                        return ;
                    }
                }

                // Continue outerloop on new cost combination
                n = -1;
                N = std::accumulate(token_to_costs.begin(), token_to_costs.end(), 1LL, product);
                goto resume_typo_loop;
            }

            token_index++;
        }

        if(token_candidates_vec.size() == query_tokens.size()) {
            std::vector<uint32_t> id_buff;
            search_all_candidates(num_search_fields, match_type, the_fields, filter_ids, filter_ids_length,
                                  exclude_token_ids, exclude_token_ids_size, excluded_group_ids,
                                  sort_fields, token_candidates_vec, searched_queries, qtoken_set,
                                  dropped_tokens, topster,
                                  groups_processed, all_result_ids, all_result_ids_len,
                                  typo_tokens_threshold, group_limit, group_by_fields, query_tokens,
                                  num_typos, prefixes, prioritize_exact_match, prioritize_token_position,
                                  exhaustive_search, max_candidates,
                                  syn_orig_num_tokens, sort_order, field_values, geopoint_indices,
                                  query_hashes, id_buff);

            if(id_buff.size() > 1) {
                gfx::timsort(id_buff.begin(), id_buff.end());
                id_buff.erase(std::unique( id_buff.begin(), id_buff.end() ), id_buff.end());
            }

            uint32_t* new_all_result_ids = nullptr;
            all_result_ids_len = ArrayUtils::or_scalar(all_result_ids, all_result_ids_len, &id_buff[0],
                                                       id_buff.size(), &new_all_result_ids);
            delete[] all_result_ids;
            all_result_ids = new_all_result_ids;
        }

        resume_typo_loop:

        if(!exhaustive_search && all_result_ids_len >= typo_tokens_threshold) {
            // if typo threshold is breached, we are done
            return ;
        }

        n++;
    }
}

void Index::popular_fields_of_token(const spp::sparse_hash_map<std::string, art_tree*>& search_index,
                                    const std::string& previous_token,
                                    const std::vector<search_field_t>& the_fields,
                                    const size_t num_search_fields,
                                    std::vector<size_t>& popular_field_ids) {

    const auto token_c_str = (const unsigned char*) previous_token.c_str();
    const int token_len = (int) previous_token.size() + 1;

    std::vector<std::pair<size_t, size_t>> field_id_doc_counts;

    for(size_t i = 0; i < num_search_fields; i++) {
        const std::string& field_name = the_fields[i].name;
        auto leaf = static_cast<art_leaf*>(art_search(search_index.at(field_name), token_c_str, token_len));

        if(!leaf) {
            continue;
        }

        auto num_docs = posting_t::num_ids(leaf->values);
        field_id_doc_counts.emplace_back(i, num_docs);
    }

    std::sort(field_id_doc_counts.begin(), field_id_doc_counts.end(), [](const auto& p1, const auto& p2) {
        return p1.second > p2.second;
    });

    for(const auto& field_id_doc_count: field_id_doc_counts) {
        popular_field_ids.push_back(field_id_doc_count.first);
    }
}

void Index::find_across_fields(const token_t& previous_token,
                               const std::string& previous_token_str,
                               const std::vector<search_field_t>& the_fields,
                               const size_t num_search_fields,
                               const uint32_t* filter_ids, uint32_t filter_ids_length,
                               const uint32_t* exclude_token_ids, size_t exclude_token_ids_size,
                               std::vector<uint32_t>& prev_token_doc_ids,
                               std::vector<size_t>& top_prefix_field_ids) const {

    // one iterator for each token, each underlying iterator contains results of token across multiple fields
    std::vector<or_iterator_t> token_its;

    // used to track plists that must be destructed once done
    std::vector<posting_list_t*> expanded_plists;

    result_iter_state_t istate(exclude_token_ids, exclude_token_ids_size, filter_ids, filter_ids_length);

    const bool prefix_search = previous_token.is_prefix_searched;
    const uint32_t token_num_typos = previous_token.num_typos;
    const bool token_prefix = previous_token.is_prefix_searched;

    auto& token_str = previous_token_str;
    auto token_c_str = (const unsigned char*) token_str.c_str();
    const size_t token_len = token_str.size() + 1;
    std::vector<posting_list_t::iterator_t> its;

    std::vector<std::pair<size_t, size_t>> field_id_doc_counts;

    for(size_t i = 0; i < num_search_fields; i++) {
        const std::string& field_name = the_fields[i].name;

        art_tree* tree = search_index.at(field_name);
        art_leaf* leaf = static_cast<art_leaf*>(art_search(tree, token_c_str, token_len));

        if(!leaf) {
            continue;
        }

        /*LOG(INFO) << "Token: " << token_str << ", field_name: " << field_name
                  << ", num_ids: " << posting_t::num_ids(leaf->values);*/

        if(IS_COMPACT_POSTING(leaf->values)) {
            auto compact_posting_list = COMPACT_POSTING_PTR(leaf->values);
            posting_list_t* full_posting_list = compact_posting_list->to_full_posting_list();
            expanded_plists.push_back(full_posting_list);
            its.push_back(full_posting_list->new_iterator(nullptr, nullptr, i)); // moved, not copied
        } else {
            posting_list_t* full_posting_list = (posting_list_t*)(leaf->values);
            its.push_back(full_posting_list->new_iterator(nullptr, nullptr, i)); // moved, not copied
        }

        field_id_doc_counts.emplace_back(i, posting_t::num_ids(leaf->values));
    }

    if(its.empty()) {
        // this token does not have any match across *any* field: probably a typo
        LOG(INFO) << "No matching field found for token: " << token_str;
        return;
    }

    std::sort(field_id_doc_counts.begin(), field_id_doc_counts.end(), [](const auto& p1, const auto& p2) {
        return p1.second > p2.second;
    });

    for(auto& field_id_doc_count: field_id_doc_counts) {
        top_prefix_field_ids.push_back(field_id_doc_count.first);
    }

    or_iterator_t token_fields(its);
    token_its.push_back(std::move(token_fields));

    or_iterator_t::intersect(token_its, istate, [&](uint32_t seq_id, const std::vector<or_iterator_t>& its) {
        prev_token_doc_ids.push_back(seq_id);
    });

    for(posting_list_t* plist: expanded_plists) {
        delete plist;
    }
}

void Index::search_across_fields(const std::vector<token_t>& query_tokens,
                                 const std::vector<uint32_t>& num_typos,
                                 const std::vector<bool>& prefixes,
                                 const std::vector<search_field_t>& the_fields,
                                 const size_t num_search_fields,
                                 const text_match_type_t match_type,
                                 const std::vector<sort_by>& sort_fields,
                                 Topster* topster,
                                 spp::sparse_hash_map<uint64_t, uint32_t>& groups_processed,
                                 std::vector<std::vector<art_leaf*>>& searched_queries,
                                 tsl::htrie_map<char, token_leaf>& qtoken_set,
                                 const std::vector<token_t>& dropped_tokens,
                                 const size_t group_limit,
                                 const std::vector<std::string>& group_by_fields,
                                 const bool prioritize_exact_match,
                                 const bool prioritize_token_position,
                                 const uint32_t* filter_ids, uint32_t filter_ids_length,
                                 const uint32_t total_cost, const int syn_orig_num_tokens,
                                 const uint32_t* exclude_token_ids, size_t exclude_token_ids_size,
                                 const std::unordered_set<uint32_t>& excluded_group_ids,
                                 const int* sort_order,
                                 std::array<spp::sparse_hash_map<uint32_t, int64_t>*, 3>& field_values,
                                 const std::vector<size_t>& geopoint_indices,
                                 std::vector<uint32_t>& id_buff,
                                 uint32_t*& all_result_ids, size_t& all_result_ids_len) const {

    std::vector<art_leaf*> query_suggestion;

    // one or_iterator for each token (across multiple fields)
    std::vector<or_iterator_t> dropped_token_its;

    // used to track plists that must be destructed once done
    std::vector<posting_list_t*> expanded_dropped_plists;

    for(auto& dropped_token: dropped_tokens) {
        auto& token = dropped_token.value;
        auto token_c_str = (const unsigned char*) token.c_str();

        // convert token from each field into an or_iterator
        std::vector<posting_list_t::iterator_t> its;

        for(size_t i = 0; i < the_fields.size(); i++) {
            const std::string& field_name = the_fields[i].name;

            art_tree* tree = search_index.at(field_name);
            art_leaf* leaf = static_cast<art_leaf*>(art_search(tree, token_c_str, token.size()+1));

            if(!leaf) {
                continue;
            }

            /*LOG(INFO) << "Token: " << token << ", field_name: " << field_name
                        << ", num_ids: " << posting_t::num_ids(leaf->values);*/

            if(IS_COMPACT_POSTING(leaf->values)) {
                auto compact_posting_list = COMPACT_POSTING_PTR(leaf->values);
                posting_list_t* full_posting_list = compact_posting_list->to_full_posting_list();
                expanded_dropped_plists.push_back(full_posting_list);
                its.push_back(full_posting_list->new_iterator(nullptr, nullptr, i)); // moved, not copied
            } else {
                posting_list_t* full_posting_list = (posting_list_t*)(leaf->values);
                its.push_back(full_posting_list->new_iterator(nullptr, nullptr, i)); // moved, not copied
            }
        }

        or_iterator_t token_fields(its);
        dropped_token_its.push_back(std::move(token_fields));
    }



    // one iterator for each token, each underlying iterator contains results of token across multiple fields
    std::vector<or_iterator_t> token_its;

    // used to track plists that must be destructed once done
    std::vector<posting_list_t*> expanded_plists;

    result_iter_state_t istate(exclude_token_ids, exclude_token_ids_size, filter_ids, filter_ids_length);

    // for each token, find the posting lists across all query_by fields
    for(size_t ti = 0; ti < query_tokens.size(); ti++) {
        const bool prefix_search = query_tokens[ti].is_prefix_searched;
        const uint32_t token_num_typos = query_tokens[ti].num_typos;
        const bool token_prefix = query_tokens[ti].is_prefix_searched;

        auto& token_str = query_tokens[ti].value;
        auto token_c_str = (const unsigned char*) token_str.c_str();
        const size_t token_len = token_str.size() + 1;
        std::vector<posting_list_t::iterator_t> its;

        for(size_t i = 0; i < num_search_fields; i++) {
            const std::string& field_name = the_fields[i].name;
            const uint32_t field_num_typos = (the_fields[i].orig_index < num_typos.size())
                                             ? num_typos[the_fields[i].orig_index] : num_typos[0];
            const bool field_prefix = (the_fields[i].orig_index < prefixes.size()) ? prefixes[the_fields[i].orig_index]
                                                                                   : prefixes[0];

            if(token_num_typos > field_num_typos) {
                // since the token can come from any field, we still have to respect per-field num_typos
                continue;
            }

            if(token_prefix && !field_prefix) {
                // even though this token is an outcome of prefix search, we can't use it for this field, since
                // this field has prefix search disabled.
                continue;
            }

            art_tree* tree = search_index.at(field_name);
            art_leaf* leaf = static_cast<art_leaf*>(art_search(tree, token_c_str, token_len));

            if(!leaf) {
                continue;
            }

            query_suggestion.push_back(leaf);

            /*LOG(INFO) << "Token: " << token_str << ", field_name: " << field_name
                      << ", num_ids: " << posting_t::num_ids(leaf->values);*/

            if(IS_COMPACT_POSTING(leaf->values)) {
                auto compact_posting_list = COMPACT_POSTING_PTR(leaf->values);
                posting_list_t* full_posting_list = compact_posting_list->to_full_posting_list();
                expanded_plists.push_back(full_posting_list);
                its.push_back(full_posting_list->new_iterator(nullptr, nullptr, i)); // moved, not copied
            } else {
                posting_list_t* full_posting_list = (posting_list_t*)(leaf->values);
                its.push_back(full_posting_list->new_iterator(nullptr, nullptr, i)); // moved, not copied
            }
        }

        if(its.empty()) {
            // this token does not have any match across *any* field: probably a typo
            LOG(INFO) << "No matching field found for token: " << token_str;
            continue;
        }

        or_iterator_t token_fields(its);
        token_its.push_back(std::move(token_fields));
    }

    std::vector<uint32_t> result_ids;
    size_t filter_index = 0;

    or_iterator_t::intersect(token_its, istate, [&](uint32_t seq_id, const std::vector<or_iterator_t>& its) {
        //LOG(INFO) << "seq_id: " << seq_id;
        // Convert [token -> fields] orientation to [field -> tokens] orientation
        std::vector<std::vector<posting_list_t::iterator_t>> field_to_tokens(num_search_fields);

        for(size_t ti = 0; ti < its.size(); ti++) {
            const or_iterator_t& token_fields_iters = its[ti];
            const std::vector<posting_list_t::iterator_t>& field_iters = token_fields_iters.get_its();

            for(size_t fi = 0; fi < field_iters.size(); fi++) {
                const posting_list_t::iterator_t& field_iter = field_iters[fi];
                if(field_iter.id() == seq_id) {
                    // not all fields might contain a given token
                    field_to_tokens[field_iter.get_field_id()].push_back(field_iter.clone());
                }
            }
        }

        int64_t best_field_match_score = 0, best_field_weight = 0;
        uint32_t num_matching_fields = 0;

        for(size_t fi = 0; fi < field_to_tokens.size(); fi++) {
            const std::vector<posting_list_t::iterator_t>& token_postings = field_to_tokens[fi];
            if(token_postings.empty()) {
                continue;
            }

            const int64_t field_weight = the_fields[fi].weight;
            const bool field_is_array = search_schema.at(the_fields[fi].name).is_array();

            int64_t field_match_score = 0;
            bool single_exact_query_token = false;

            if(total_cost == 0 && query_tokens.size() == 1) {
                // does this candidate suggestion token match query token exactly?
                single_exact_query_token = true;
            }

            score_results2(sort_fields, searched_queries.size(), fi, field_is_array,
                           total_cost, field_match_score,
                           seq_id, sort_order,
                           prioritize_exact_match, single_exact_query_token, prioritize_token_position,
                           query_tokens.size(), syn_orig_num_tokens, token_postings);

            if(match_type == max_score && field_match_score > best_field_match_score) {
                best_field_match_score = field_match_score;
                best_field_weight = field_weight;
            }

            if(match_type == max_weight && field_weight > best_field_weight) {
                best_field_weight = field_weight;
                best_field_match_score = field_match_score;
            }

            num_matching_fields++;
        }

        uint64_t distinct_id = seq_id;
        if(group_limit != 0) {
            distinct_id = get_distinct_id(group_by_fields, seq_id);
            if(excluded_group_ids.count(distinct_id) != 0) {
                return;
            }
        }

        int64_t scores[3] = {0};
        int64_t match_score_index = -1;

        compute_sort_scores(sort_fields, sort_order, field_values, geopoint_indices, seq_id, filter_index,
                            best_field_match_score, scores, match_score_index);

        size_t query_len = query_tokens.size();

        // check if seq_id exists in any of the dropped_token iters and increment matching fields accordingly
        for(auto& dropped_token_it: dropped_token_its) {
            if(dropped_token_it.skip_to(seq_id) && dropped_token_it.id() == seq_id) {
                query_len++;
            }
        }

        if(syn_orig_num_tokens != -1) {
            query_len = syn_orig_num_tokens;
        }
        query_len = std::min<size_t>(15, query_len);

        // NOTE: `query_len` is total tokens matched across fields.
        // Within a field, only a subset can match

        // MAX_SCORE
        // [ sign | tokens_matched | max_field_score | max_field_weight | num_matching_fields ]
        // [   1  |        4       |        48       |       8          |         3           ]  (64 bits)

        // MAX_WEIGHT
        // [ sign | tokens_matched | max_field_weight | max_field_score  | num_matching_fields ]
        // [   1  |        4       |        8         |      48          |         3           ]  (64 bits)

        auto max_field_weight = std::min<size_t>(FIELD_MAX_WEIGHT, best_field_weight);
        num_matching_fields = std::min<size_t>(7, num_matching_fields);

        uint64_t aggregated_score = match_type == max_score ?
                                    ((int64_t(query_len) << 59) |
                                    (int64_t(best_field_match_score) << 11) |
                                    (int64_t(max_field_weight) << 3) |
                                    (int64_t(num_matching_fields) << 0))

                                    :

                                    ((int64_t(query_len) << 59) |
                                     (int64_t(max_field_weight) << 51) |
                                     (int64_t(best_field_match_score) << 3) |
                                     (int64_t(num_matching_fields) << 0))
                                    ;

        /*LOG(INFO) << "seq_id: " << seq_id << ", query_len: " << query_len
                  << ", syn_orig_num_tokens: " << syn_orig_num_tokens
                  << ", best_field_match_score: " << best_field_match_score
                  << ", max_field_weight: " << max_field_weight
                  << ", num_matching_fields: " << num_matching_fields
                  << ", aggregated_score: " << aggregated_score;*/

        KV kv(searched_queries.size(), seq_id, distinct_id, match_score_index, scores);
        if(match_score_index != -1) {
            kv.scores[match_score_index] = aggregated_score;
        }

        int ret = topster->add(&kv);
        if(group_limit != 0 && ret < 2) {
            groups_processed[distinct_id]++;
        }
        result_ids.push_back(seq_id);
    });

    id_buff.insert(id_buff.end(), result_ids.begin(), result_ids.end());

    if(id_buff.size() > 100000) {
        // prevents too many ORs during exhaustive searching
        gfx::timsort(id_buff.begin(), id_buff.end());
        id_buff.erase(std::unique( id_buff.begin(), id_buff.end() ), id_buff.end());

        uint32_t* new_all_result_ids = nullptr;
        all_result_ids_len = ArrayUtils::or_scalar(all_result_ids, all_result_ids_len, &id_buff[0],
                                                   id_buff.size(), &new_all_result_ids);
        delete[] all_result_ids;
        all_result_ids = new_all_result_ids;
        id_buff.clear();
    }

    if(!result_ids.empty()) {
        searched_queries.push_back(query_suggestion);
        for(const auto& qtoken: query_tokens) {
            qtoken_set.insert(qtoken.value, token_leaf(nullptr, qtoken.root_len, qtoken.num_typos, qtoken.is_prefix_searched));
        }
    }

    for(posting_list_t* plist: expanded_plists) {
        delete plist;
    }

    for(posting_list_t* plist: expanded_dropped_plists) {
        delete plist;
    }
}

void Index::compute_sort_scores(const std::vector<sort_by>& sort_fields, const int* sort_order,
                                std::array<spp::sparse_hash_map<uint32_t, int64_t>*, 3> field_values,
                                const std::vector<size_t>& geopoint_indices,
                                uint32_t seq_id, size_t filter_index, int64_t max_field_match_score,
                                int64_t* scores, int64_t& match_score_index, float vector_distance) const {

    int64_t geopoint_distances[3];

    for(auto& i: geopoint_indices) {
        spp::sparse_hash_map<uint32_t, int64_t>* geopoints = field_values[i];
        int64_t dist = INT32_MAX;

        S2LatLng reference_lat_lng;
        GeoPoint::unpack_lat_lng(sort_fields[i].geopoint, reference_lat_lng);

        if(geopoints != nullptr) {
            auto it = geopoints->find(seq_id);

            if(it != geopoints->end()) {
                int64_t packed_latlng = it->second;
                S2LatLng s2_lat_lng;
                GeoPoint::unpack_lat_lng(packed_latlng, s2_lat_lng);
                dist = GeoPoint::distance(s2_lat_lng, reference_lat_lng);
            }
        } else {
            // indicates geo point array
            auto field_it = geo_array_index.at(sort_fields[i].name);
            auto it = field_it->find(seq_id);

            if(it != field_it->end()) {
                int64_t* latlngs = it->second;
                for(size_t li = 0; li < latlngs[0]; li++) {
                    S2LatLng s2_lat_lng;
                    int64_t packed_latlng = latlngs[li + 1];
                    GeoPoint::unpack_lat_lng(packed_latlng, s2_lat_lng);
                    int64_t this_dist = GeoPoint::distance(s2_lat_lng, reference_lat_lng);
                    if(this_dist < dist) {
                        dist = this_dist;
                    }
                }
            }
        }

        if(dist < sort_fields[i].exclude_radius) {
            dist = 0;
        }

        if(sort_fields[i].geo_precision > 0) {
            dist = dist + sort_fields[i].geo_precision - 1 -
                   (dist + sort_fields[i].geo_precision - 1) % sort_fields[i].geo_precision;
        }

        geopoint_distances[i] = dist;

        // Swap (id -> latlong) index to (id -> distance) index
        field_values[i] = &geo_sentinel_value;
    }

    const int64_t default_score = INT64_MIN;  // to handle field that doesn't exist in document (e.g. optional)

    // avoiding loop
    if (sort_fields.size() > 0) {
        if (field_values[0] == &text_match_sentinel_value) {
            scores[0] = int64_t(max_field_match_score);
            match_score_index = 0;
        } else if (field_values[0] == &seq_id_sentinel_value) {
            scores[0] = seq_id;
        } else if(field_values[0] == &geo_sentinel_value) {
            scores[0] = geopoint_distances[0];
        } else if(field_values[0] == &str_sentinel_value) {
            scores[0] = str_sort_index.at(sort_fields[0].name)->rank(seq_id);
            if(scores[0] == adi_tree_t::NOT_FOUND) {
                if(sort_fields[0].order == sort_field_const::asc &&
                   sort_fields[0].missing_values == sort_by::missing_values_t::first) {
                    scores[0] = -scores[0];
                }

                else if(sort_fields[0].order == sort_field_const::desc &&
                        sort_fields[0].missing_values == sort_by::missing_values_t::last) {
                    scores[0] = -scores[0];
                }
            }
        } else if(field_values[0] == &eval_sentinel_value) {
            // Returns iterator to the first element that is >= to value or last if no such element is found.
            bool found = false;
            if (filter_index == 0 || filter_index < sort_fields[0].eval.size) {
                size_t found_index = std::lower_bound(sort_fields[0].eval.ids + filter_index,
                                                      sort_fields[0].eval.ids + sort_fields[0].eval.size, seq_id) -
                                     sort_fields[0].eval.ids;

                if (found_index != sort_fields[0].eval.size && sort_fields[0].eval.ids[found_index] == seq_id) {
                    filter_index = found_index + 1;
                    found = true;
                }

                filter_index = found_index;
            }

            scores[0] = int64_t(found);
        } else if(field_values[0] == &vector_distance_sentinel_value) {
            scores[0] = float_to_int64_t(vector_distance);
        } else {
            auto it = field_values[0]->find(seq_id);
            scores[0] = (it == field_values[0]->end()) ? default_score : it->second;

            if(scores[0] == INT64_MIN && sort_fields[0].missing_values == sort_by::missing_values_t::first) {
                // By default, missing numerical value are always going to be sorted to be at the end
                // because: -INT64_MIN == INT64_MIN. To account for missing values config, we will have to change
                // the default for missing value based on whether it's asc or desc sort.
                bool is_asc = (sort_order[0] == -1);
                scores[0] = is_asc ? (INT64_MIN + 1) : INT64_MAX;
            }
        }

        if (sort_order[0] == -1) {
            scores[0] = -scores[0];
        }
    }

    if(sort_fields.size() > 1) {
        if (field_values[1] == &text_match_sentinel_value) {
            scores[1] = int64_t(max_field_match_score);
            match_score_index = 1;
        } else if (field_values[1] == &seq_id_sentinel_value) {
            scores[1] = seq_id;
        } else if(field_values[1] == &geo_sentinel_value) {
            scores[1] = geopoint_distances[1];
        } else if(field_values[1] == &str_sentinel_value) {
            scores[1] = str_sort_index.at(sort_fields[1].name)->rank(seq_id);
            if(scores[1] == adi_tree_t::NOT_FOUND) {
                if(sort_fields[1].order == sort_field_const::asc &&
                   sort_fields[1].missing_values == sort_by::missing_values_t::first) {
                    scores[1] = -scores[1];
                }

                else if(sort_fields[1].order == sort_field_const::desc &&
                        sort_fields[1].missing_values == sort_by::missing_values_t::last) {
                    scores[1] = -scores[1];
                }
            }
        } else if(field_values[1] == &eval_sentinel_value) {
            // Returns iterator to the first element that is >= to value or last if no such element is found.
            bool found = false;
            if (filter_index == 0 || filter_index < sort_fields[1].eval.size) {
                size_t found_index = std::lower_bound(sort_fields[1].eval.ids + filter_index,
                                                      sort_fields[1].eval.ids + sort_fields[1].eval.size, seq_id) -
                                     sort_fields[1].eval.ids;

                if (found_index != sort_fields[1].eval.size && sort_fields[1].eval.ids[found_index] == seq_id) {
                    filter_index = found_index + 1;
                    found = true;
                }

                filter_index = found_index;
            }

            scores[1] = int64_t(found);
        }  else if(field_values[1] == &vector_distance_sentinel_value) {
            scores[1] = float_to_int64_t(vector_distance);
        } else {
            auto it = field_values[1]->find(seq_id);
            scores[1] = (it == field_values[1]->end()) ? default_score : it->second;
            if(scores[1] == INT64_MIN && sort_fields[1].missing_values == sort_by::missing_values_t::first) {
                bool is_asc = (sort_order[1] == -1);
                scores[1] = is_asc ? (INT64_MIN + 1) : INT64_MAX;
            }
        }

        if (sort_order[1] == -1) {
            scores[1] = -scores[1];
        }
    }

    if(sort_fields.size() > 2) {
        if (field_values[2] == &text_match_sentinel_value) {
            scores[2] = int64_t(max_field_match_score);
            match_score_index = 2;
        } else if (field_values[2] == &seq_id_sentinel_value) {
            scores[2] = seq_id;
        } else if(field_values[2] == &geo_sentinel_value) {
            scores[2] = geopoint_distances[2];
        } else if(field_values[2] == &str_sentinel_value) {
            scores[2] = str_sort_index.at(sort_fields[2].name)->rank(seq_id);
            if(scores[2] == adi_tree_t::NOT_FOUND) {
                if(sort_fields[2].order == sort_field_const::asc &&
                   sort_fields[2].missing_values == sort_by::missing_values_t::first) {
                    scores[2] = -scores[2];
                }

                else if(sort_fields[2].order == sort_field_const::desc &&
                        sort_fields[2].missing_values == sort_by::missing_values_t::last) {
                    scores[2] = -scores[2];
                }
            }
        } else if(field_values[2] == &eval_sentinel_value) {
            // Returns iterator to the first element that is >= to value or last if no such element is found.
            bool found = false;
            if (filter_index == 0 || filter_index < sort_fields[2].eval.size) {
                size_t found_index = std::lower_bound(sort_fields[2].eval.ids + filter_index,
                                                      sort_fields[2].eval.ids + sort_fields[2].eval.size, seq_id) -
                                     sort_fields[2].eval.ids;

                if (found_index != sort_fields[2].eval.size && sort_fields[2].eval.ids[found_index] == seq_id) {
                    filter_index = found_index + 1;
                    found = true;
                }

                filter_index = found_index;
            }

            scores[2] = int64_t(found);
        } else if(field_values[2] == &vector_distance_sentinel_value) {
            scores[2] = float_to_int64_t(vector_distance);
        } else {
            auto it = field_values[2]->find(seq_id);
            scores[2] = (it == field_values[2]->end()) ? default_score : it->second;
            if(scores[2] == INT64_MIN && sort_fields[2].missing_values == sort_by::missing_values_t::first) {
                bool is_asc = (sort_order[2] == -1);
                scores[2] = is_asc ? (INT64_MIN + 1) : INT64_MAX;
            }
        }

        if (sort_order[2] == -1) {
            scores[2] = -scores[2];
        }
    }
}

void Index::do_phrase_search(const size_t num_search_fields, const std::vector<search_field_t>& search_fields,
                             std::vector<query_tokens_t>& field_query_tokens,
                             const std::vector<sort_by>& sort_fields,
                             std::vector<std::vector<art_leaf*>>& searched_queries, const size_t group_limit,
                             const std::vector<std::string>& group_by_fields,
                             Topster* actual_topster,
                             const int sort_order[3],
                             std::array<spp::sparse_hash_map<uint32_t, int64_t>*, 3> field_values,
                             const std::vector<size_t>& geopoint_indices,
                             const std::vector<uint32_t>& curated_ids_sorted,
                             uint32_t*& all_result_ids, size_t& all_result_ids_len,
                             spp::sparse_hash_map<uint64_t, uint32_t>& groups_processed,
                             const std::set<uint32_t>& curated_ids,
                             const uint32_t* excluded_result_ids, size_t excluded_result_ids_size,
                             const std::unordered_set<uint32_t>& excluded_group_ids,
                             Topster* curated_topster,
                             const std::map<size_t, std::map<size_t, uint32_t>>& included_ids_map,
                             bool is_wildcard_query,
                             uint32_t*& filter_ids, uint32_t& filter_ids_length) const {

    std::map<uint32_t, size_t> phrase_match_id_scores;

    uint32_t* phrase_match_ids = nullptr;
    size_t phrase_match_ids_size = 0;

    for(size_t i = 0; i < num_search_fields; i++) {
        const std::string& field_name = search_fields[i].name;
        const size_t field_weight = search_fields[i].weight;
        bool is_array = search_schema.at(field_name).is_array();

        uint32_t* field_phrase_match_ids = nullptr;
        size_t field_phrase_match_ids_size = 0;

        for(const auto& phrase: field_query_tokens[i].q_phrases) {
            std::vector<void*> posting_lists;

            for(const std::string& token: phrase) {
                art_leaf* leaf = (art_leaf *) art_search(search_index.at(field_name),
                                                         (const unsigned char *) token.c_str(),
                                                         token.size() + 1);
                if(leaf) {
                    posting_lists.push_back(leaf->values);
                }
            }

            if(posting_lists.size() != phrase.size()) {
                // unmatched length means no matches will be found for this phrase, so skip to next phrase
                continue;
            }

            std::vector<uint32_t> contains_ids;
            posting_t::intersect(posting_lists, contains_ids);

            uint32_t* this_phrase_ids = new uint32_t[contains_ids.size()];
            size_t this_phrase_ids_size = 0;
            posting_t::get_phrase_matches(posting_lists, is_array, &contains_ids[0], contains_ids.size(),
                                          this_phrase_ids, this_phrase_ids_size);

            if(this_phrase_ids_size == 0) {
                // no results found for this phrase, but other phrases can find results
                delete [] this_phrase_ids;
                continue;
            }

            // results of multiple phrases must be ANDed

            if(field_phrase_match_ids_size == 0) {
                field_phrase_match_ids_size = this_phrase_ids_size;
                field_phrase_match_ids = this_phrase_ids;
            } else {
                uint32_t* phrase_ids_merged = nullptr;
                field_phrase_match_ids_size = ArrayUtils::and_scalar(this_phrase_ids, this_phrase_ids_size, field_phrase_match_ids,
                                                                     field_phrase_match_ids_size, &phrase_ids_merged);
                delete [] field_phrase_match_ids;
                delete [] this_phrase_ids;
                field_phrase_match_ids = phrase_ids_merged;
            }
        }

        if(field_phrase_match_ids_size == 0) {
            continue;
        }

        // upto 10K phrase match IDs per field will be weighted so that phrase match against a higher weighted field
        // is returned earlier in the results
        const size_t weight_score_base = 100000;  // just to make score be a large number
        for(size_t pi = 0; pi < std::min<size_t>(10000, field_phrase_match_ids_size); pi++) {
            auto this_field_score = (weight_score_base + field_weight);
            auto existing_score = phrase_match_id_scores[field_phrase_match_ids[pi]];
            phrase_match_id_scores[field_phrase_match_ids[pi]] = std::max(this_field_score, existing_score);
        }

        // across fields, we have to OR phrase match ids
        if(phrase_match_ids_size == 0) {
            phrase_match_ids = field_phrase_match_ids;
            phrase_match_ids_size = field_phrase_match_ids_size;
        } else {
            uint32_t* phrase_ids_merged = nullptr;
            phrase_match_ids_size = ArrayUtils::or_scalar(phrase_match_ids, phrase_match_ids_size, field_phrase_match_ids,
                                                          field_phrase_match_ids_size, &phrase_ids_merged);

            delete [] phrase_match_ids;
            delete [] field_phrase_match_ids;
            phrase_match_ids = phrase_ids_merged;
        }
    }

    // AND phrase id matches with filter ids
    if(filter_ids_length == 0) {
        filter_ids = phrase_match_ids;
        filter_ids_length = phrase_match_ids_size;
    } else {
        uint32_t* filter_ids_merged = nullptr;
        filter_ids_length = ArrayUtils::and_scalar(filter_ids, filter_ids_length, phrase_match_ids,
                                                   phrase_match_ids_size, &filter_ids_merged);

        delete [] filter_ids;
        filter_ids = filter_ids_merged;

        delete [] phrase_match_ids;
    }

    if(filter_ids_length == 0) {
        delete [] filter_ids;
        filter_ids = nullptr;
    }

    curate_filtered_ids(curated_ids, excluded_result_ids,
                        excluded_result_ids_size, filter_ids, filter_ids_length, curated_ids_sorted);
    collate_included_ids({}, included_ids_map, curated_topster, searched_queries);

    size_t filter_index = 0;

    if(is_wildcard_query) {
        all_result_ids = new uint32_t[filter_ids_length];
        std::copy(filter_ids, filter_ids + filter_ids_length, all_result_ids);
        all_result_ids_len = filter_ids_length;
    } else {
        // this means that the there are non-phrase tokens in the query
        // so we cannot directly copy to the all_result_ids array
        return ;
    }

    // populate topster
    for(size_t i = 0; i < std::min<size_t>(10000, filter_ids_length); i++) {
        auto seq_id = filter_ids[i];

        int64_t match_score = phrase_match_id_scores[seq_id];
        int64_t scores[3] = {0};
        int64_t match_score_index = -1;

        compute_sort_scores(sort_fields, sort_order, field_values, geopoint_indices, seq_id, filter_index,
                            match_score, scores, match_score_index);

        uint64_t distinct_id = seq_id;
        if(group_limit != 0) {
            distinct_id = get_distinct_id(group_by_fields, seq_id);
            if(excluded_group_ids.count(distinct_id) != 0) {
                continue;
            }
        }

        KV kv(searched_queries.size(), seq_id, distinct_id, match_score_index, scores);
        int ret = actual_topster->add(&kv);
        if(group_limit != 0 && ret < 2) {
            groups_processed[distinct_id]++;
        }

        if(((i + 1) % (1 << 12)) == 0) {
            BREAK_CIRCUIT_BREAKER
        }
    }

    searched_queries.push_back({});
}

void Index::do_synonym_search(const std::vector<search_field_t>& the_fields,
                              const text_match_type_t match_type,
                              filter_node_t const* const& filter_tree_root,
                              const std::map<size_t, std::map<size_t, uint32_t>>& included_ids_map,
                              const std::vector<sort_by>& sort_fields_std, Topster* curated_topster,
                              const token_ordering& token_order,
                              const size_t typo_tokens_threshold, const size_t group_limit,
                              const std::vector<std::string>& group_by_fields, bool prioritize_exact_match,
                              const bool prioritize_token_position,
                              const bool exhaustive_search, const size_t concurrency,
                              const std::vector<bool>& prefixes,
                              size_t min_len_1typo,
                              size_t min_len_2typo, const size_t max_candidates, const std::set<uint32_t>& curated_ids,
                              const std::vector<uint32_t>& curated_ids_sorted, const uint32_t* exclude_token_ids,
                              size_t exclude_token_ids_size,
                              const std::unordered_set<uint32_t>& excluded_group_ids,
                              Topster* actual_topster,
                              std::vector<std::vector<token_t>>& q_pos_synonyms,
                              int syn_orig_num_tokens,
                              spp::sparse_hash_map<uint64_t, uint32_t>& groups_processed,
                              std::vector<std::vector<art_leaf*>>& searched_queries,
                              uint32_t*& all_result_ids, size_t& all_result_ids_len,
                              const uint32_t* filter_ids, const uint32_t filter_ids_length,
                              std::set<uint64>& query_hashes,
                              const int* sort_order,
                              std::array<spp::sparse_hash_map<uint32_t, int64_t>*, 3>& field_values,
                              const std::vector<size_t>& geopoint_indices,
                              tsl::htrie_map<char, token_leaf>& qtoken_set) const {

    for (const auto& syn_tokens : q_pos_synonyms) {
        query_hashes.clear();
        fuzzy_search_fields(the_fields, syn_tokens, {}, match_type, exclude_token_ids,
                            exclude_token_ids_size, filter_ids, filter_ids_length, curated_ids_sorted, excluded_group_ids,
                            sort_fields_std, {0}, searched_queries, qtoken_set, actual_topster, groups_processed,
                            all_result_ids, all_result_ids_len, group_limit, group_by_fields, prioritize_exact_match,
                            prioritize_token_position, query_hashes, token_order, prefixes, typo_tokens_threshold,
                            exhaustive_search, max_candidates, min_len_1typo,
                            min_len_2typo, syn_orig_num_tokens, sort_order, field_values, geopoint_indices);
    }

    collate_included_ids({}, included_ids_map, curated_topster, searched_queries);
}

void Index::do_infix_search(const size_t num_search_fields, const std::vector<search_field_t>& the_fields,
                            const std::vector<enable_t>& infixes,
                            const std::vector<sort_by>& sort_fields,
                            std::vector<std::vector<art_leaf*>>& searched_queries, const size_t group_limit,
                            const std::vector<std::string>& group_by_fields, const size_t max_extra_prefix,
                            const size_t max_extra_suffix,
                            const std::vector<token_t>& query_tokens, Topster* actual_topster,
                            const uint32_t *filter_ids, size_t filter_ids_length,
                            const int sort_order[3],
                            std::array<spp::sparse_hash_map<uint32_t, int64_t>*, 3> field_values,
                            const std::vector<size_t>& geopoint_indices,
                            const std::vector<uint32_t>& curated_ids_sorted,
                            const std::unordered_set<uint32_t>& excluded_group_ids,
                            uint32_t*& all_result_ids, size_t& all_result_ids_len,
                            spp::sparse_hash_map<uint64_t, uint32_t>& groups_processed) const {

    for(size_t field_id = 0; field_id < num_search_fields; field_id++) {
        auto& field_name = the_fields[field_id].name;
        enable_t field_infix = (the_fields[field_id].orig_index < infixes.size())
                               ? infixes[the_fields[field_id].orig_index] : infixes[0];

        if(field_infix == always || (field_infix == fallback && all_result_ids_len == 0)) {
            std::vector<uint32_t> infix_ids;
            search_infix(query_tokens[0].value, field_name, infix_ids, max_extra_prefix, max_extra_suffix);

            if(!infix_ids.empty()) {
                gfx::timsort(infix_ids.begin(), infix_ids.end());
                infix_ids.erase(std::unique( infix_ids.begin(), infix_ids.end() ), infix_ids.end());

                uint32_t *raw_infix_ids = nullptr;
                size_t raw_infix_ids_length = 0;

                if(curated_ids_sorted.size() != 0) {
                    raw_infix_ids_length = ArrayUtils::exclude_scalar(&infix_ids[0], infix_ids.size(), &curated_ids_sorted[0],
                                                                      curated_ids_sorted.size(), &raw_infix_ids);
                    infix_ids.clear();
                } else {
                    raw_infix_ids = &infix_ids[0];
                    raw_infix_ids_length = infix_ids.size();
                }

                if(filter_ids_length != 0) {
                    uint32_t *filtered_raw_infix_ids = nullptr;
                    raw_infix_ids_length = ArrayUtils::and_scalar(filter_ids, filter_ids_length, raw_infix_ids,
                                                                  raw_infix_ids_length, &filtered_raw_infix_ids);
                    if(raw_infix_ids != &infix_ids[0]) {
                        delete [] raw_infix_ids;
                    }

                    raw_infix_ids = filtered_raw_infix_ids;
                }

                bool field_is_array = search_schema.at(the_fields[field_id].name).is_array();
                size_t filter_index = 0;

                for(size_t i = 0; i < raw_infix_ids_length; i++) {
                    auto seq_id = raw_infix_ids[i];

                    int64_t match_score = 0;
                    score_results2(sort_fields, searched_queries.size(), field_id, field_is_array,
                                   0, match_score, seq_id, sort_order, false, false, false, 1, -1, {});

                    int64_t scores[3] = {0};
                    int64_t match_score_index = -1;

                    compute_sort_scores(sort_fields, sort_order, field_values, geopoint_indices, seq_id, filter_index,
                                        100, scores, match_score_index);

                    uint64_t distinct_id = seq_id;
                    if(group_limit != 0) {
                        distinct_id = get_distinct_id(group_by_fields, seq_id);
                        if(excluded_group_ids.count(distinct_id) != 0) {
                            continue;
                        }
                    }

                    KV kv(searched_queries.size(), seq_id, distinct_id, match_score_index, scores);
                    int ret = actual_topster->add(&kv);
                    if(group_limit != 0 && ret < 2) {
                        groups_processed[distinct_id]++;
                    }

                    if(((i + 1) % (1 << 12)) == 0) {
                        BREAK_CIRCUIT_BREAKER
                    }
                }

                uint32_t* new_all_result_ids = nullptr;
                all_result_ids_len = ArrayUtils::or_scalar(all_result_ids, all_result_ids_len, raw_infix_ids,
                                                           raw_infix_ids_length, &new_all_result_ids);
                delete[] all_result_ids;
                all_result_ids = new_all_result_ids;

                if(raw_infix_ids != &infix_ids[0]) {
                    delete [] raw_infix_ids;
                }

                searched_queries.push_back({});
            }
        }
    }
}

void Index::handle_exclusion(const size_t num_search_fields, std::vector<query_tokens_t>& field_query_tokens,
                             const std::vector<search_field_t>& search_fields, uint32_t*& exclude_token_ids,
                             size_t& exclude_token_ids_size) const {
    for(size_t i = 0; i < num_search_fields; i++) {
        const std::string & field_name = search_fields[i].name;
        bool is_array = search_schema.at(field_name).is_array();

        for(const auto& q_exclude_phrase: field_query_tokens[i].q_exclude_tokens) {
            // if phrase has multiple words, then we have to do exclusion of phrase match results
            std::vector<void*> posting_lists;
            for(const std::string& exclude_token: q_exclude_phrase) {
                art_leaf* leaf = (art_leaf *) art_search(search_index.at(field_name),
                                                         (const unsigned char *) exclude_token.c_str(),
                                                         exclude_token.size() + 1);
                if(leaf) {
                    posting_lists.push_back(leaf->values);
                }
            }

            if(posting_lists.size() != q_exclude_phrase.size()) {
                continue;
            }

            std::vector<uint32_t> contains_ids;
            posting_t::intersect(posting_lists, contains_ids);

            if(posting_lists.size() == 1) {
                uint32_t *exclude_token_ids_merged = nullptr;
                exclude_token_ids_size = ArrayUtils::or_scalar(exclude_token_ids, exclude_token_ids_size,
                                                               &contains_ids[0], contains_ids.size(),
                                                               &exclude_token_ids_merged);
                delete [] exclude_token_ids;
                exclude_token_ids = exclude_token_ids_merged;
            } else {
                uint32_t* phrase_ids = new uint32_t[contains_ids.size()];
                size_t phrase_ids_size = 0;

                posting_t::get_phrase_matches(posting_lists, is_array, &contains_ids[0], contains_ids.size(),
                                              phrase_ids, phrase_ids_size);

                uint32_t *exclude_token_ids_merged = nullptr;
                exclude_token_ids_size = ArrayUtils::or_scalar(exclude_token_ids, exclude_token_ids_size,
                                                               phrase_ids, phrase_ids_size,
                                                               &exclude_token_ids_merged);
                delete [] phrase_ids;
                delete [] exclude_token_ids;
                exclude_token_ids = exclude_token_ids_merged;
            }
        }
    }
}

void Index::compute_facet_infos(const std::vector<facet>& facets, facet_query_t& facet_query,
                                const size_t facet_query_num_typos,
                                const uint32_t* all_result_ids, const size_t& all_result_ids_len,
                                const std::vector<std::string>& group_by_fields,
                                const size_t max_candidates,
                                std::vector<facet_info_t>& facet_infos) const {

    if(all_result_ids_len == 0) {
        return;
    }

    for(size_t findex=0; findex < facets.size(); findex++) {
        const auto& a_facet = facets[findex];

        const auto field_facet_mapping_it = facet_index_v3.find(a_facet.field_name);
        const auto field_single_val_facet_mapping_it = single_val_facet_index_v3.find(a_facet.field_name);
        
        if((field_facet_mapping_it == facet_index_v3.end()) 
            && (field_single_val_facet_mapping_it == single_val_facet_index_v3.end())) {
            continue;
        }

        facet_infos[findex].use_facet_query = false;

        const field &facet_field = search_schema.at(a_facet.field_name);
        facet_infos[findex].facet_field = facet_field;

        facet_infos[findex].should_compute_stats = (facet_field.type != field_types::STRING &&
                                                    facet_field.type != field_types::BOOL &&
                                                    facet_field.type != field_types::STRING_ARRAY &&
                                                    facet_field.type != field_types::BOOL_ARRAY);

        if(a_facet.field_name == facet_query.field_name && !facet_query.query.empty()) {
            facet_infos[findex].use_facet_query = true;

            if (facet_field.is_bool()) {
                if (facet_query.query == "true") {
                    facet_query.query = "1";
                } else if (facet_query.query == "false") {
                    facet_query.query = "0";
                }
            }

            //LOG(INFO) << "facet_query.query: " << facet_query.query;

            std::vector<std::string> query_tokens;
            Tokenizer(facet_query.query, true, !facet_field.is_string(),
                      facet_field.locale, symbols_to_index, token_separators).tokenize(query_tokens);

            std::vector<token_t> qtokens;

            for (size_t qtoken_index = 0; qtoken_index < query_tokens.size(); qtoken_index++) {
                bool is_prefix = (qtoken_index == query_tokens.size()-1);
                qtokens.emplace_back(qtoken_index, query_tokens[qtoken_index], is_prefix,
                                     query_tokens[qtoken_index].size(), 0);
            }

            std::vector<std::vector<art_leaf*>> searched_queries;
            Topster* topster = nullptr;
            spp::sparse_hash_map<uint64_t, uint32_t> groups_processed;
            uint32_t* field_result_ids = nullptr;
            size_t field_result_ids_len = 0;
            size_t field_num_results = 0;
            std::set<uint64> query_hashes;
            size_t num_toks_dropped = 0;
            std::vector<sort_by> sort_fields;

            search_field(0, qtokens, nullptr, 0, num_toks_dropped,
                         facet_field, facet_field.faceted_name(),
                         all_result_ids, all_result_ids_len, {}, sort_fields, -1, facet_query_num_typos, searched_queries, topster,
                         groups_processed, &field_result_ids, field_result_ids_len, field_num_results, 0, group_by_fields,
                         false, 4, query_hashes, MAX_SCORE, true, 0, 1, false, -1, 3, 1000, max_candidates);

            //LOG(INFO) << "searched_queries.size: " << searched_queries.size();

            // NOTE: `field_result_ids` will consists of IDs across ALL queries in searched_queries

            for(size_t si = 0; si < searched_queries.size(); si++) {
                const auto& searched_query = searched_queries[si];
                std::vector<std::string> searched_tokens;

                std::vector<void*> posting_lists;
                for(auto leaf: searched_query) {
                    posting_lists.push_back(leaf->values);
                    std::string tok(reinterpret_cast<char*>(leaf->key), leaf->key_len - 1);
                    searched_tokens.push_back(tok);
                    //LOG(INFO) << "tok: " << tok;
                }

                //LOG(INFO) << "si: " << si << ", field_result_ids_len: " << field_result_ids_len;

                for(size_t i = 0; i < field_result_ids_len; i++) {
                    uint32_t seq_id = field_result_ids[i];

                    
                        

                    bool id_matched = true;

                    for(auto pl: posting_lists) {
                        if(!posting_t::contains(pl, seq_id)) {
                            // need to ensure that document ID actually contains searched_query tokens
                            // since `field_result_ids` contains documents matched across all queries
                            id_matched = false;
                            break;
                        }
                    }

                    if(!id_matched) {
                        continue;
                    }

                    if(facet_field.is_array()) {
                        const auto doc_fvalues_it = field_facet_mapping_it->second[seq_id % ARRAY_FACET_DIM]->find(seq_id);
                        if(doc_fvalues_it == field_facet_mapping_it->second[seq_id % ARRAY_FACET_DIM]->end()) {
                            continue;
                        }

                        std::vector<size_t> array_indices;
                        posting_t::get_matching_array_indices(posting_lists, seq_id, array_indices);

                        for(size_t array_index: array_indices) {
                            if(array_index < doc_fvalues_it->second.length) {
                                uint64_t hash = doc_fvalues_it->second.hashes[array_index];

                                /*LOG(INFO) << "seq_id: " << seq_id << ", hash: " << hash << ", array index: "
                                          << array_index;*/

                                if(facet_infos[findex].hashes.count(hash) == 0) {
                                    facet_infos[findex].hashes.emplace(hash, searched_tokens);
                                }
                            }
                        }
                    } else {
                        const auto doc_single_fvalues_it = field_single_val_facet_mapping_it->second[seq_id % ARRAY_FACET_DIM]->find(seq_id);
                        if(doc_single_fvalues_it == field_single_val_facet_mapping_it->second[seq_id % ARRAY_FACET_DIM]->end()) {
                            continue;
                        }
                        
                        uint64_t hash = doc_single_fvalues_it->second;
                        if(facet_infos[findex].hashes.count(hash) == 0) {
                            facet_infos[findex].hashes.emplace(hash, searched_tokens);
                        }
                    }
                }
            }

            delete [] field_result_ids;
        }
    }
}

void Index::curate_filtered_ids(const std::set<uint32_t>& curated_ids,
                                const uint32_t* exclude_token_ids, size_t exclude_token_ids_size,
                                uint32_t*& filter_ids, uint32_t& filter_ids_length,
                                const std::vector<uint32_t>& curated_ids_sorted) const {
    if(!curated_ids.empty()) {
        uint32_t *excluded_result_ids = nullptr;
        filter_ids_length = ArrayUtils::exclude_scalar(filter_ids, filter_ids_length, &curated_ids_sorted[0],
                                                       curated_ids_sorted.size(), &excluded_result_ids);
        delete [] filter_ids;
        filter_ids = excluded_result_ids;
    }

    // Exclude document IDs associated with excluded tokens from the result set
    if(exclude_token_ids_size != 0) {
        uint32_t *excluded_result_ids = nullptr;
        filter_ids_length = ArrayUtils::exclude_scalar(filter_ids, filter_ids_length, exclude_token_ids,
                                                       exclude_token_ids_size, &excluded_result_ids);
        delete[] filter_ids;
        filter_ids = excluded_result_ids;
    }
}

void Index::search_wildcard(filter_node_t const* const& filter_tree_root,
                            const std::map<size_t, std::map<size_t, uint32_t>>& included_ids_map,
                            const std::vector<sort_by>& sort_fields, Topster* topster, Topster* curated_topster,
                            spp::sparse_hash_map<uint64_t, uint32_t>& groups_processed,
                            std::vector<std::vector<art_leaf*>>& searched_queries, const size_t group_limit,
                            const std::vector<std::string>& group_by_fields, const std::set<uint32_t>& curated_ids,
                            const std::vector<uint32_t>& curated_ids_sorted, const uint32_t* exclude_token_ids,
                            size_t exclude_token_ids_size, const std::unordered_set<uint32_t>& excluded_group_ids,
                            uint32_t*& all_result_ids, size_t& all_result_ids_len, const uint32_t* filter_ids,
                            uint32_t filter_ids_length, const size_t concurrency,
                            const int* sort_order,
                            std::array<spp::sparse_hash_map<uint32_t, int64_t>*, 3>& field_values,
                            const std::vector<size_t>& geopoint_indices) const {

    uint32_t token_bits = 0;
    const bool check_for_circuit_break = (filter_ids_length > 1000000);

    //auto beginF = std::chrono::high_resolution_clock::now();

    const size_t num_threads = std::min<size_t>(concurrency, filter_ids_length);
    const size_t window_size = (num_threads == 0) ? 0 :
                               (filter_ids_length + num_threads - 1) / num_threads;  // rounds up

    spp::sparse_hash_map<uint64_t, uint64_t> tgroups_processed[num_threads];
    Topster* topsters[num_threads];
    std::vector<posting_list_t::iterator_t> plists;

    size_t num_processed = 0;
    std::mutex m_process;
    std::condition_variable cv_process;

    size_t num_queued = 0;
    size_t filter_index = 0;

    const auto parent_search_begin = search_begin_us;
    const auto parent_search_stop_ms = search_stop_us;
    auto parent_search_cutoff = search_cutoff;

    for(size_t thread_id = 0; thread_id < num_threads && filter_index < filter_ids_length; thread_id++) {
        size_t batch_res_len = window_size;

        if(filter_index + window_size > filter_ids_length) {
            batch_res_len = filter_ids_length - filter_index;
        }

        const uint32_t* batch_result_ids = filter_ids + filter_index;
        num_queued++;

        searched_queries.push_back({});

        topsters[thread_id] = new Topster(topster->MAX_SIZE, topster->distinct);

        thread_pool->enqueue([this, &parent_search_begin, &parent_search_stop_ms, &parent_search_cutoff,
                                     thread_id, &sort_fields, &searched_queries,
                                     &group_limit, &group_by_fields, &topsters, &tgroups_processed, &excluded_group_ids,
                                     &sort_order, field_values, &geopoint_indices, &plists,
                                     check_for_circuit_break,
                                     batch_result_ids, batch_res_len,
                                     &num_processed, &m_process, &cv_process]() {

            search_begin_us = parent_search_begin;
            search_stop_us = parent_search_stop_ms;
            search_cutoff = parent_search_cutoff;

            size_t filter_index = 0;

            for(size_t i = 0; i < batch_res_len; i++) {
                const uint32_t seq_id = batch_result_ids[i];
                int64_t match_score = 0;

                score_results2(sort_fields, (uint16_t) searched_queries.size(), 0, false, 0,
                               match_score, seq_id, sort_order, false, false, false, 1, -1, plists);

                int64_t scores[3] = {0};
                int64_t match_score_index = -1;

                compute_sort_scores(sort_fields, sort_order, field_values, geopoint_indices, seq_id, filter_index,
                                    100, scores, match_score_index);

                uint64_t distinct_id = seq_id;
                if(group_limit != 0) {
                    distinct_id = get_distinct_id(group_by_fields, seq_id);
                    if(excluded_group_ids.count(distinct_id) != 0) {
                        continue;
                    }
                }

                KV kv(searched_queries.size(), seq_id, distinct_id, match_score_index, scores);
                int ret = topsters[thread_id]->add(&kv);

                if(group_limit != 0 && ret < 2) {
                    tgroups_processed[thread_id][distinct_id]++;
                }

                if(check_for_circuit_break && ((i + 1) % (1 << 15)) == 0) {
                    // check only once every 2^15 docs to reduce overhead
                    BREAK_CIRCUIT_BREAKER
                }
            }

            std::unique_lock<std::mutex> lock(m_process);
            num_processed++;
            parent_search_cutoff = parent_search_cutoff || search_cutoff;
            cv_process.notify_one();
        });

        filter_index += batch_res_len;
    }

    std::unique_lock<std::mutex> lock_process(m_process);
    cv_process.wait(lock_process, [&](){ return num_processed == num_queued; });

    search_cutoff = parent_search_cutoff;

    for(size_t thread_id = 0; thread_id < num_processed; thread_id++) {
        //groups_processed.insert(tgroups_processed[thread_id].begin(), tgroups_processed[thread_id].end());
        for(const auto& it : tgroups_processed[thread_id]) {
            groups_processed[it.first]+= it.second;
        } 
        aggregate_topster(topster, topsters[thread_id]);
        delete topsters[thread_id];
    }

    /*long long int timeMillisF = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - beginF).count();
    LOG(INFO) << "Time for raw scoring: " << timeMillisF;*/

    uint32_t* new_all_result_ids = nullptr;
    all_result_ids_len = ArrayUtils::or_scalar(all_result_ids, all_result_ids_len, filter_ids,
                                               filter_ids_length, &new_all_result_ids);
    delete [] all_result_ids;
    all_result_ids = new_all_result_ids;
}

void Index::populate_sort_mapping(int* sort_order, std::vector<size_t>& geopoint_indices,
                                  std::vector<sort_by>& sort_fields_std,
                                  std::array<spp::sparse_hash_map<uint32_t, int64_t>*, 3>& field_values) const {
    for (size_t i = 0; i < sort_fields_std.size(); i++) {
        sort_order[i] = 1;
        if (sort_fields_std[i].order == sort_field_const::asc) {
            sort_order[i] = -1;
        }

        if (sort_fields_std[i].name == sort_field_const::text_match) {
            field_values[i] = &text_match_sentinel_value;
        } else if (sort_fields_std[i].name == sort_field_const::seq_id || 
            sort_fields_std[i].name == sort_field_const::group_found) {
            field_values[i] = &seq_id_sentinel_value;
        } else if (sort_fields_std[i].name == sort_field_const::eval) {
            field_values[i] = &eval_sentinel_value;
            filter_result_t result;
            recursive_filter(sort_fields_std[i].eval.filter_tree_root, result);
            sort_fields_std[i].eval.ids = result.docs;
            sort_fields_std[i].eval.size = result.count;
            result.docs = nullptr;
        } else if(sort_fields_std[i].name == sort_field_const::vector_distance) {
            field_values[i] = &vector_distance_sentinel_value;
        } else if (search_schema.count(sort_fields_std[i].name) != 0 && search_schema.at(sort_fields_std[i].name).sort) {
            if (search_schema.at(sort_fields_std[i].name).type == field_types::GEOPOINT_ARRAY) {
                geopoint_indices.push_back(i);
                field_values[i] = nullptr; // GEOPOINT_ARRAY uses a multi-valued index
            } else if(search_schema.at(sort_fields_std[i].name).type == field_types::STRING) {
                field_values[i] = &str_sentinel_value;
            } else {
                field_values[i] = sort_index.at(sort_fields_std[i].name);

                if (search_schema.at(sort_fields_std[i].name).is_geopoint()) {
                    geopoint_indices.push_back(i);
                }
            }
        }
    }
}

void Index::search_field(const uint8_t & field_id,
                         const std::vector<token_t>& query_tokens,
                         const uint32_t* exclude_token_ids,
                         size_t exclude_token_ids_size,
                         size_t& num_tokens_dropped,
                         const field& the_field, const std::string& field_name, // to handle faceted index
                         const uint32_t *filter_ids, size_t filter_ids_length,
                         const std::vector<uint32_t>& curated_ids,
                         std::vector<sort_by> & sort_fields,
                         const int last_typo,
                         const int max_typos,
                         std::vector<std::vector<art_leaf*>> & searched_queries,
                         Topster* topster, spp::sparse_hash_map<uint64_t, uint32_t>& groups_processed,
                         uint32_t** all_result_ids, size_t & all_result_ids_len, size_t& field_num_results,
                         const size_t group_limit, const std::vector<std::string>& group_by_fields,
                         bool prioritize_exact_match,
                         const size_t concurrency,
                         std::set<uint64>& query_hashes,
                         const token_ordering token_order, const bool prefix,
                         const size_t drop_tokens_threshold,
                         const size_t typo_tokens_threshold,
                         const bool exhaustive_search,
                         int syn_orig_num_tokens,
                         size_t min_len_1typo,
                         size_t min_len_2typo,
                         const size_t max_candidates) const {

    // NOTE: `query_tokens` preserve original tokens, while `search_tokens` could be a result of dropped tokens

    size_t max_cost = (max_typos < 0 || max_typos > 2) ? 2 : max_typos;

    if(the_field.locale != "" && the_field.locale != "en" && !Tokenizer::is_cyrillic(the_field.locale)) {
        // disable fuzzy trie traversal for certain non-english locales
        max_cost = 0;
    }

    // To prevent us from doing ART search repeatedly as we iterate through possible corrections
    spp::sparse_hash_map<std::string, std::vector<art_leaf*>> token_cost_cache;

    std::vector<std::vector<int>> token_to_costs;

    for(size_t stoken_index=0; stoken_index < query_tokens.size(); stoken_index++) {
        const std::string& token = query_tokens[stoken_index].value;

        std::vector<int> all_costs;
        // This ensures that we don't end up doing a cost of 1 for a single char etc.
        int bounded_cost = get_bounded_typo_cost(max_cost, token.length(), min_len_1typo, min_len_2typo);

        for(int cost = 0; cost <= bounded_cost; cost++) {
            all_costs.push_back(cost);
        }

        token_to_costs.push_back(all_costs);
    }

    // stores candidates for each token, i.e. i-th index would have all possible tokens with a cost of "c"
    std::vector<token_candidates> token_candidates_vec;

    std::set<std::string> unique_tokens;

    auto product = []( long long a, std::vector<int>& b ) { return a*b.size(); };
    long long n = 0;
    long long int N = std::accumulate(token_to_costs.begin(), token_to_costs.end(), 1LL, product);

    const size_t combination_limit = exhaustive_search ? Index::COMBINATION_MAX_LIMIT : Index::COMBINATION_MIN_LIMIT;

    while(n < N && n < combination_limit) {
        RETURN_CIRCUIT_BREAKER

        // Outerloop generates combinations of [cost to max_cost] for each token
        // For e.g. for a 3-token query: [0, 0, 0], [0, 0, 1], [0, 1, 1] etc.
        std::vector<uint32_t> costs(token_to_costs.size());
        ldiv_t q { n, 0 };
        bool valid_combo = false;

        for(long long i = (token_to_costs.size() - 1); 0 <= i ; --i ) {
            q = ldiv(q.quot, token_to_costs[i].size());
            costs[i] = token_to_costs[i][q.rem];
            if(costs[i] == uint32_t(last_typo+1)) {
                // to support progressive typo searching, there must be atleast one typo that's greater than last_typo
                valid_combo = true;
            }
        }

        if(last_typo != -1 && !valid_combo) {
            n++;
            continue;
        }

        unique_tokens.clear();
        token_candidates_vec.clear();
        size_t token_index = 0;

        while(token_index < query_tokens.size()) {
            // For each token, look up the generated cost for this iteration and search using that cost
            const std::string& token = query_tokens[token_index].value;
            const std::string token_cost_hash = token + std::to_string(costs[token_index]);

            std::vector<art_leaf*> leaves;
            const bool prefix_search = prefix && query_tokens[token_index].is_prefix_searched;

            /*LOG(INFO) << "Searching for field: " << the_field.name << ", token:"
                      << token << " - cost: " << costs[token_index] << ", prefix_search: " << prefix_search;*/

            if(token_cost_cache.count(token_cost_hash) != 0) {
                leaves = token_cost_cache[token_cost_hash];
            } else {
                const size_t token_len = prefix_search ? (int) token.length() : (int) token.length() + 1;

                //auto begin = std::chrono::high_resolution_clock::now();

                // need less candidates for filtered searches since we already only pick tokens with results
                art_fuzzy_search(search_index.at(field_name), (const unsigned char *) token.c_str(), token_len,
                                 costs[token_index], costs[token_index], max_candidates, token_order, prefix_search,
                                 false, "", filter_ids, filter_ids_length, leaves, unique_tokens);

                /*auto timeMillis = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::high_resolution_clock::now() - begin).count();
                LOG(INFO) << "Time taken for fuzzy search: " << timeMillis << "ms";*/

                if(!leaves.empty()) {
                    token_cost_cache.emplace(token_cost_hash, leaves);
                    for(auto leaf: leaves) {
                        std::string tok(reinterpret_cast<char*>(leaf->key), leaf->key_len - 1);
                        unique_tokens.emplace(tok);
                    }
                }
            }

            if(!leaves.empty()) {
                //log_leaves(costs[token_index], token, leaves);
                token_candidates_vec.push_back(
                        token_candidates{query_tokens[token_index], costs[token_index], prefix_search, leaves});
            }

            token_index++;
        }

        if(token_candidates_vec.size() == query_tokens.size()) {
            std::vector<uint32_t> id_buff;

            // If all tokens are, go ahead and search for candidates
            search_candidates(field_id, the_field.is_array(), filter_ids, filter_ids_length,
                              exclude_token_ids, exclude_token_ids_size,
                              curated_ids, sort_fields, token_candidates_vec, searched_queries, topster,
                              groups_processed, all_result_ids, all_result_ids_len, field_num_results,
                              typo_tokens_threshold, group_limit, group_by_fields, query_tokens,
                              prioritize_exact_match, exhaustive_search, syn_orig_num_tokens,
                              concurrency, query_hashes, id_buff);

            if(id_buff.size() > 1) {
                std::sort(id_buff.begin(), id_buff.end());
                id_buff.erase(std::unique( id_buff.begin(), id_buff.end() ), id_buff.end());
            }

            uint32_t* new_all_result_ids = nullptr;
            all_result_ids_len = ArrayUtils::or_scalar(*all_result_ids, all_result_ids_len, &id_buff[0],
                                                       id_buff.size(), &new_all_result_ids);
            delete[] *all_result_ids;
            *all_result_ids = new_all_result_ids;
        }

        if(!exhaustive_search && field_num_results >= typo_tokens_threshold) {
            // if typo threshold is breached, we are done
            return ;
        }

        n++;
    }
}

int Index::get_bounded_typo_cost(const size_t max_cost, const size_t token_len,
                                 const size_t min_len_1typo, const size_t min_len_2typo) {
    if(token_len < min_len_1typo) {
        // typo correction is disabled for small tokens
        return 0;
    }

    if(token_len < min_len_2typo) {
        // 2-typos are enabled only at token length of 7 chars
        return std::min<int>(max_cost, 1);
    }

    return std::min<int>(max_cost, 2);
}

void Index::log_leaves(const int cost, const std::string &token, const std::vector<art_leaf *> &leaves) const {
    LOG(INFO) << "Index: " << name << ", token: " << token << ", cost: " << cost;

    for(size_t i=0; i < leaves.size(); i++) {
        std::string key((char*)leaves[i]->key, leaves[i]->key_len);
        LOG(INFO) << key << " - " << posting_t::num_ids(leaves[i]->values);
        LOG(INFO) << "frequency: " << posting_t::num_ids(leaves[i]->values) << ", max_score: " << leaves[i]->max_score;
        /*for(auto j=0; j<leaves[i]->values->ids.getLength(); j++) {
            LOG(INFO) << "id: " << leaves[i]->values->ids.at(j);
        }*/
    }
}

int64_t Index::score_results2(const std::vector<sort_by> & sort_fields, const uint16_t & query_index,
                              const size_t field_id,
                              const bool field_is_array,
                              const uint32_t total_cost,
                              int64_t& match_score,
                              const uint32_t seq_id, const int sort_order[3],
                              const bool prioritize_exact_match,
                              const bool single_exact_query_token,
                              const bool prioritize_token_position,
                              size_t num_query_tokens,
                              int syn_orig_num_tokens,
                              const std::vector<posting_list_t::iterator_t>& posting_lists) const {

    //auto begin = std::chrono::high_resolution_clock::now();
    //const std::string first_token((const char*)query_suggestion[0]->key, query_suggestion[0]->key_len-1);

    if (posting_lists.size() <= 1) {
        const uint8_t is_verbatim_match = uint8_t(
                prioritize_exact_match && single_exact_query_token &&
                posting_list_t::is_single_token_verbatim_match(posting_lists[0], field_is_array)
        );
        size_t words_present = (num_query_tokens == 1 && syn_orig_num_tokens != -1) ? syn_orig_num_tokens : 1;
        size_t distance = (num_query_tokens == 1 && syn_orig_num_tokens != -1) ? syn_orig_num_tokens-1 : 0;
        size_t max_offset = prioritize_token_position ? posting_list_t::get_last_offset(posting_lists[0],
                                                                                        field_is_array) : 255;
        Match single_token_match = Match(words_present, distance, max_offset, is_verbatim_match);
        match_score = single_token_match.get_match_score(total_cost, words_present);

        /*auto this_words_present = ((match_score >> 32) & 0xFF);
        auto unique_words = ((match_score >> 40) & 0xFF);
        auto typo_score = ((match_score >> 24) & 0xFF);
        auto proximity = ((match_score >> 16) & 0xFF);
        auto verbatim = ((match_score >> 8) & 0xFF);
        auto offset_score = ((match_score >> 0) & 0xFF);
        LOG(INFO) << "seq_id: " << seq_id
                  << ", words_present: " << this_words_present
                  << ", unique_words: " << unique_words
                  << ", typo_score: " << typo_score
                  << ", proximity: " << proximity
                  << ", verbatim: " << verbatim
                  << ", offset_score: " << offset_score
                  << ", match_score: " << match_score;*/

    } else {
        std::map<size_t, std::vector<token_positions_t>> array_token_positions;
        posting_list_t::get_offsets(posting_lists, array_token_positions);

        for (const auto& kv: array_token_positions) {
            const std::vector<token_positions_t>& token_positions = kv.second;
            if (token_positions.empty()) {
                continue;
            }

            const Match &match = Match(seq_id, token_positions, false, prioritize_exact_match);
            uint64_t this_match_score = match.get_match_score(total_cost, posting_lists.size());

            // Within a field, only a subset of query tokens can match (unique_words), but even a smaller set
            // might be available within the window used for proximity calculation (this_words_present)

            auto this_words_present = ((this_match_score >> 32) & 0xFF);
            auto unique_words = field_is_array ? this_words_present : ((this_match_score >> 40) & 0xFF);
            auto typo_score = ((this_match_score >> 24) & 0xFF);
            auto proximity = ((this_match_score >> 16) & 0xFF);
            auto verbatim = ((this_match_score >> 8) & 0xFF);
            auto offset_score = prioritize_token_position ? ((this_match_score >> 0) & 0xFF) : 0;

            if(syn_orig_num_tokens != -1 && num_query_tokens == posting_lists.size()) {
                unique_words = syn_orig_num_tokens;
                this_words_present = syn_orig_num_tokens;
                proximity = 100 - (syn_orig_num_tokens - 1);
            }

            uint64_t mod_match_score = (
                    (int64_t(this_words_present) << 40) |
                    (int64_t(unique_words) << 32) |
                    (int64_t(typo_score) << 24) |
                    (int64_t(proximity) << 16) |
                    (int64_t(verbatim) << 8) |
                    (int64_t(offset_score) << 0)
            );

            if(mod_match_score > match_score) {
                match_score = mod_match_score;
            }

            /*std::ostringstream os;
            os << "seq_id: " << seq_id << ", field_id: " << field_id
               << ", this_words_present: " << this_words_present
               << ", unique_words: " << unique_words
               << ", typo_score: " << typo_score
               << ", proximity: " << proximity
               << ", verbatim: " << verbatim
               << ", offset_score: " << offset_score
               << ", mod_match_score: " << mod_match_score
               << ", token_positions: " << token_positions.size()
               << ", num_query_tokens: " << num_query_tokens
               << ", posting_lists.size: " << posting_lists.size()
               << ", array_index: " << kv.first
               << std::endl;
            LOG(INFO) << os.str();*/
        }
    }

    //long long int timeNanos = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - begin).count();
    //LOG(INFO) << "Time taken for results iteration: " << timeNanos << "ms";

    return 0;
}

void Index::score_results(const std::vector<sort_by> & sort_fields, const uint16_t & query_index,
                          const uint8_t & field_id, const bool field_is_array, const uint32_t total_cost,
                          Topster* topster,
                          const std::vector<art_leaf *> &query_suggestion,
                          spp::sparse_hash_map<uint64_t, uint32_t>& groups_processed,
                          const uint32_t seq_id, const int sort_order[3],
                          std::array<spp::sparse_hash_map<uint32_t, int64_t>*, 3> field_values,
                          const std::vector<size_t>& geopoint_indices,
                          const size_t group_limit, const std::vector<std::string>& group_by_fields,
                          const uint32_t token_bits,
                          const bool prioritize_exact_match,
                          const bool single_exact_query_token,
                          int syn_orig_num_tokens,
                          const std::vector<posting_list_t::iterator_t>& posting_lists) const {

    int64_t geopoint_distances[3];

    for(auto& i: geopoint_indices) {
        spp::sparse_hash_map<uint32_t, int64_t>* geopoints = field_values[i];
        int64_t dist = INT32_MAX;

        S2LatLng reference_lat_lng;
        GeoPoint::unpack_lat_lng(sort_fields[i].geopoint, reference_lat_lng);

        if(geopoints != nullptr) {
            auto it = geopoints->find(seq_id);

            if(it != geopoints->end()) {
                int64_t packed_latlng = it->second;
                S2LatLng s2_lat_lng;
                GeoPoint::unpack_lat_lng(packed_latlng, s2_lat_lng);
                dist = GeoPoint::distance(s2_lat_lng, reference_lat_lng);
            }
        } else {
            // indicates geo point array
            auto field_it = geo_array_index.at(sort_fields[i].name);
            auto it = field_it->find(seq_id);

            if(it != field_it->end()) {
                int64_t* latlngs = it->second;
                for(size_t li = 0; li < latlngs[0]; li++) {
                    S2LatLng s2_lat_lng;
                    int64_t packed_latlng = latlngs[li + 1];
                    GeoPoint::unpack_lat_lng(packed_latlng, s2_lat_lng);
                    int64_t this_dist = GeoPoint::distance(s2_lat_lng, reference_lat_lng);
                    if(this_dist < dist) {
                        dist = this_dist;
                    }
                }
            }
        }

        if(dist < sort_fields[i].exclude_radius) {
            dist = 0;
        }

        if(sort_fields[i].geo_precision > 0) {
            dist = dist + sort_fields[i].geo_precision - 1 -
                   (dist + sort_fields[i].geo_precision - 1) % sort_fields[i].geo_precision;
        }

        geopoint_distances[i] = dist;

        // Swap (id -> latlong) index to (id -> distance) index
        field_values[i] = &geo_sentinel_value;
    }

    //auto begin = std::chrono::high_resolution_clock::now();
    //const std::string first_token((const char*)query_suggestion[0]->key, query_suggestion[0]->key_len-1);

    uint64_t match_score = 0;

    if (posting_lists.size() <= 1) {
        const uint8_t is_verbatim_match = uint8_t(
                prioritize_exact_match && single_exact_query_token &&
                posting_list_t::is_single_token_verbatim_match(posting_lists[0], field_is_array)
        );
        size_t words_present = (syn_orig_num_tokens == -1) ? 1 : syn_orig_num_tokens;
        size_t distance = (syn_orig_num_tokens == -1) ? 0 : syn_orig_num_tokens-1;
        Match single_token_match = Match(words_present, distance, is_verbatim_match);
        match_score = single_token_match.get_match_score(total_cost, words_present);
    } else {
        std::map<size_t, std::vector<token_positions_t>> array_token_positions;
        posting_list_t::get_offsets(posting_lists, array_token_positions);

        // NOTE: tokens found returned by matcher is only within the best matched window, so we have to still consider
        // unique tokens found if they are spread across the text.
        uint32_t unique_tokens_found = __builtin_popcount(token_bits);
        if(syn_orig_num_tokens != -1) {
            unique_tokens_found = syn_orig_num_tokens;
        }

        for (const auto& kv: array_token_positions) {
            const std::vector<token_positions_t>& token_positions = kv.second;
            if (token_positions.empty()) {
                continue;
            }

            const Match &match = Match(seq_id, token_positions, false, prioritize_exact_match);
            uint64_t this_match_score = match.get_match_score(total_cost, unique_tokens_found);

            auto this_words_present = ((this_match_score >> 24) & 0xFF);
            auto typo_score = ((this_match_score >> 16) & 0xFF);
            auto proximity = ((this_match_score >> 8) & 0xFF);
            auto verbatim = (this_match_score & 0xFF);

            if(syn_orig_num_tokens != -1) {
                this_words_present = syn_orig_num_tokens;
                proximity = 100 - (syn_orig_num_tokens - 1);
            }

            uint64_t mod_match_score = (
                    (int64_t(unique_tokens_found) << 32) |
                    (int64_t(this_words_present) << 24) |
                    (int64_t(typo_score) << 16) |
                    (int64_t(proximity) << 8) |
                    (int64_t(verbatim) << 0)
            );

            if(mod_match_score > match_score) {
                match_score = mod_match_score;
            }

            /*std::ostringstream os;
            os << name << ", total_cost: " << (255 - total_cost)
               << ", words_present: " << match.words_present
               << ", match_score: " << match_score
               << ", match.distance: " << match.distance
               << ", seq_id: " << seq_id << std::endl;
            LOG(INFO) << os.str();*/
        }
    }

    const int64_t default_score = INT64_MIN;  // to handle field that doesn't exist in document (e.g. optional)
    int64_t scores[3] = {0};
    size_t match_score_index = 0;

    // avoiding loop
    if (sort_fields.size() > 0) {
        if (field_values[0] == &text_match_sentinel_value) {
            scores[0] = int64_t(match_score);
            match_score_index = 0;
        } else if (field_values[0] == &seq_id_sentinel_value) {
            scores[0] = seq_id;
        } else if(field_values[0] == &geo_sentinel_value) {
            scores[0] = geopoint_distances[0];
        } else if(field_values[0] == &str_sentinel_value) {
            scores[0] = str_sort_index.at(sort_fields[0].name)->rank(seq_id);
        } else {
            auto it = field_values[0]->find(seq_id);
            scores[0] = (it == field_values[0]->end()) ? default_score : it->second;
        }

        if (sort_order[0] == -1) {
            scores[0] = -scores[0];
        }
    }

    if(sort_fields.size() > 1) {
        if (field_values[1] == &text_match_sentinel_value) {
            scores[1] = int64_t(match_score);
            match_score_index = 1;
        } else if (field_values[1] == &seq_id_sentinel_value) {
            scores[1] = seq_id;
        } else if(field_values[1] == &geo_sentinel_value) {
            scores[1] = geopoint_distances[1];
        } else if(field_values[1] == &str_sentinel_value) {
            scores[1] = str_sort_index.at(sort_fields[1].name)->rank(seq_id);
        } else {
            auto it = field_values[1]->find(seq_id);
            scores[1] = (it == field_values[1]->end()) ? default_score : it->second;
        }

        if (sort_order[1] == -1) {
            scores[1] = -scores[1];
        }
    }

    if(sort_fields.size() > 2) {
        if (field_values[2] == &text_match_sentinel_value) {
            scores[2] = int64_t(match_score);
            match_score_index = 2;
        } else if (field_values[2] == &seq_id_sentinel_value) {
            scores[2] = seq_id;
        } else if(field_values[2] == &geo_sentinel_value) {
            scores[2] = geopoint_distances[2];
        } else if(field_values[2] == &str_sentinel_value) {
            scores[2] = str_sort_index.at(sort_fields[2].name)->rank(seq_id);
        } else {
            auto it = field_values[2]->find(seq_id);
            scores[2] = (it == field_values[2]->end()) ? default_score : it->second;
        }

        if (sort_order[2] == -1) {
            scores[2] = -scores[2];
        }
    }

    uint64_t distinct_id = seq_id;

    if(group_limit != 0) {
        distinct_id = get_distinct_id(group_by_fields, seq_id);
    }

    //LOG(INFO) << "Seq id: " << seq_id << ", match_score: " << match_score;
    KV kv(query_index, seq_id, distinct_id, match_score_index, scores);
    int ret = topster->add(&kv);
    if(group_limit != 0 && ret < 2) {
        groups_processed[distinct_id]++;
    }

    //long long int timeNanos = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - begin).count();
    //LOG(INFO) << "Time taken for results iteration: " << timeNanos << "ms";
}

// pre-filter group_by_fields such that we can avoid the find() check
uint64_t Index::get_distinct_id(const std::vector<std::string>& group_by_fields,
                                const uint32_t seq_id) const {
    uint64_t distinct_id = 1; // some constant initial value

    // calculate hash from group_by_fields
    for(const auto& field: group_by_fields) {
        const auto& field_facet_mapping_it = facet_index_v3.find(field);
        const auto& field_single_val_facet_mapping_it = single_val_facet_index_v3.find(field);
        if((field_facet_mapping_it == facet_index_v3.end()) 
            && (field_single_val_facet_mapping_it == single_val_facet_index_v3.end())) {
            continue;
        }

        if(search_schema.at(field).is_array()) {
            const auto& field_facet_mapping = field_facet_mapping_it->second;
            const auto& facet_hashes_it = field_facet_mapping[seq_id % ARRAY_FACET_DIM]->find(seq_id);

            if(facet_hashes_it == field_facet_mapping[seq_id % ARRAY_FACET_DIM]->end()) {
                continue;
            }

            const auto& facet_hashes = facet_hashes_it->second;

            for(size_t i = 0; i < facet_hashes.size(); i++) {
                distinct_id = StringUtils::hash_combine(distinct_id, facet_hashes.hashes[i]);
            }
        } else {
            const auto& field_facet_mapping = field_single_val_facet_mapping_it->second;
            const auto& facet_hashes_it = field_facet_mapping[seq_id % ARRAY_FACET_DIM]->find(seq_id);

            if(facet_hashes_it == field_facet_mapping[seq_id % ARRAY_FACET_DIM]->end()) {
                continue;
            }

            const auto& facet_hash = facet_hashes_it->second;

            distinct_id = StringUtils::hash_combine(distinct_id, facet_hash);
        }
    }

    return distinct_id;
}

inline uint32_t Index::next_suggestion2(const std::vector<tok_candidates>& token_candidates_vec,
                                        long long int n,
                                        std::vector<token_t>& query_suggestion,
                                        uint64& qhash) {
    uint32_t total_cost = 0;
    qhash = 1;

    // generate the next combination from `token_leaves` and store it in `query_suggestion`
    ldiv_t q { n, 0 };
    for(size_t i = 0 ; i < token_candidates_vec.size(); i++) {
        size_t token_size = token_candidates_vec[i].token.value.size();
        q = ldiv(q.quot, token_candidates_vec[i].candidates.size());
        const auto& candidate = token_candidates_vec[i].candidates[q.rem];
        size_t typo_cost = token_candidates_vec[i].cost;

        if (candidate.size() > 1 && !Tokenizer::is_ascii_char(candidate[0])) {
            icu::UnicodeString ustr = icu::UnicodeString::fromUTF8(candidate);
            auto code_point = ustr.char32At(0);
            if(code_point >= 0x600 && code_point <= 0x6ff) {
                // adjust typo cost for Arabic strings, since 1 byte difference makes no sense
                if(typo_cost == 1) {
                    typo_cost = 2;
                }
            }
        }

        // we assume that toke was found via prefix search if candidate is longer than token's typo tolerance
        bool is_prefix_searched = token_candidates_vec[i].prefix_search &&
                                  (candidate.size() > (token_size + typo_cost));

        size_t actual_cost = (2 * typo_cost) + uint32_t(is_prefix_searched);
        total_cost += actual_cost;

        query_suggestion[i] = token_t(i, candidate, is_prefix_searched, token_size, typo_cost);

        uint64_t this_hash = StringUtils::hash_wy(query_suggestion[i].value.c_str(), query_suggestion[i].value.size());
        qhash = StringUtils::hash_combine(qhash, this_hash);

        /*LOG(INFO) << "suggestion key: " << actual_query_suggestion[i]->key << ", token: "
                  << token_candidates_vec[i].token.value << ", actual_cost: " << actual_cost;
        LOG(INFO) << ".";*/
    }

    return total_cost;
}

inline uint32_t Index::next_suggestion(const std::vector<token_candidates> &token_candidates_vec,
                                       long long int n,
                                       std::vector<art_leaf *>& actual_query_suggestion,
                                       std::vector<art_leaf *>& query_suggestion,
                                       const int syn_orig_num_tokens,
                                       uint32_t& token_bits,
                                       uint64& qhash) {
    uint32_t total_cost = 0;
    qhash = 1;

    // generate the next combination from `token_leaves` and store it in `query_suggestion`
    ldiv_t q { n, 0 };
    for(long long i = 0 ; i < (long long) token_candidates_vec.size(); i++) {
        size_t token_size = token_candidates_vec[i].token.value.size();
        q = ldiv(q.quot, token_candidates_vec[i].candidates.size());
        actual_query_suggestion[i] = token_candidates_vec[i].candidates[q.rem];
        query_suggestion[i] = token_candidates_vec[i].candidates[q.rem];

        bool exact_match = token_candidates_vec[i].cost == 0 && token_size == actual_query_suggestion[i]->key_len-1;
        bool incr_for_prefix_search = token_candidates_vec[i].prefix_search && !exact_match;

        size_t actual_cost = (2 * token_candidates_vec[i].cost) + uint32_t(incr_for_prefix_search);

        total_cost += actual_cost;

        token_bits |= 1UL << token_candidates_vec[i].token.position; // sets n-th bit

        uintptr_t addr_val = (uintptr_t) query_suggestion[i];
        qhash = StringUtils::hash_combine(qhash, addr_val);

        /*LOG(INFO) << "suggestion key: " << actual_query_suggestion[i]->key << ", token: "
                  << token_candidates_vec[i].token.value << ", actual_cost: " << actual_cost;
        LOG(INFO) << ".";*/
    }

    if(syn_orig_num_tokens != -1) {
        token_bits = 0;
        for(size_t i = 0; i < size_t(syn_orig_num_tokens); i++) {
            token_bits |= 1UL << i;
        }
    }

    return total_cost;
}

void Index::remove_facet_token(const field& search_field, spp::sparse_hash_map<std::string, art_tree*>& search_index,
                               const std::string& token, uint32_t seq_id) {
    const unsigned char *key = (const unsigned char *) token.c_str();
    int key_len = (int) (token.length() + 1);
    const std::string& field_name = search_field.faceted_name();

    art_leaf* leaf = (art_leaf *) art_search(search_index.at(field_name), key, key_len);
    if(leaf != nullptr) {
        posting_t::erase(leaf->values, seq_id);
        if (posting_t::num_ids(leaf->values) == 0) {
            void* values = art_delete(search_index.at(field_name), key, key_len);
            posting_t::destroy_list(values);
        }
    }
}

void Index::remove_field(uint32_t seq_id, const nlohmann::json& document, const std::string& field_name) {
    const auto& search_field_it = search_schema.find(field_name);
    if(search_field_it == search_schema.end()) {
        return;
    }

    const auto& search_field = search_field_it.value();

    if(!search_field.index) {
        return;
    }

    // Go through all the field names and find the keys+values so that they can be removed from in-memory index
    if(search_field.type == field_types::STRING_ARRAY || search_field.type == field_types::STRING) {
        std::vector<std::string> tokens;
        tokenize_string_field(document, search_field, tokens, search_field.locale, symbols_to_index, token_separators);

        for(size_t i = 0; i < tokens.size(); i++) {
            const auto& token = tokens[i];
            const unsigned char *key = (const unsigned char *) token.c_str();
            int key_len = (int) (token.length() + 1);

            art_leaf* leaf = (art_leaf *) art_search(search_index.at(field_name), key, key_len);
            if(leaf != nullptr) {
                posting_t::erase(leaf->values, seq_id);
                if (posting_t::num_ids(leaf->values) == 0) {
                    void* values = art_delete(search_index.at(field_name), key, key_len);
                    posting_t::destroy_list(values);
                }
            }

            if(search_field.infix) {
                auto strhash = StringUtils::hash_wy(key, token.size());
                const auto& infix_sets = infix_index.at(search_field.name);
                infix_sets[strhash % 4]->erase(token);
            }
        }
    } else if(search_field.is_int32()) {
        const std::vector<int32_t>& values = search_field.is_single_integer() ?
                                             std::vector<int32_t>{document[field_name].get<int32_t>()} :
                                             document[field_name].get<std::vector<int32_t>>();
        for(int32_t value: values) {
            num_tree_t* num_tree = numerical_index.at(field_name);
            num_tree->remove(value, seq_id);
            if(search_field.facet) {
                remove_facet_token(search_field, search_index, std::to_string(value), seq_id);
            }
        }
    } else if(search_field.is_int64()) {
        const std::vector<int64_t>& values = search_field.is_single_integer() ?
                                             std::vector<int64_t>{document[field_name].get<int64_t>()} :
                                             document[field_name].get<std::vector<int64_t>>();
        for(int64_t value: values) {
            num_tree_t* num_tree = numerical_index.at(field_name);
            num_tree->remove(value, seq_id);
            if(search_field.facet) {
                remove_facet_token(search_field, search_index, std::to_string(value), seq_id);
            }
        }
    } else if(search_field.num_dim) {
        vector_index[search_field.name]->vecdex->markDelete(seq_id);
    } else if(search_field.is_float()) {
        const std::vector<float>& values = search_field.is_single_float() ?
                                           std::vector<float>{document[field_name].get<float>()} :
                                           document[field_name].get<std::vector<float>>();

        for(float value: values) {
            num_tree_t* num_tree = numerical_index.at(field_name);
            int64_t fintval = float_to_int64_t(value);
            num_tree->remove(fintval, seq_id);
            if(search_field.facet) {
                remove_facet_token(search_field, search_index, StringUtils::float_to_str(value), seq_id);
            }
        }
    } else if(search_field.is_bool()) {

        const std::vector<bool>& values = search_field.is_single_bool() ?
                                          std::vector<bool>{document[field_name].get<bool>()} :
                                          document[field_name].get<std::vector<bool>>();
        for(bool value: values) {
            num_tree_t* num_tree = numerical_index.at(field_name);
            int64_t bool_int64 = value ? 1 : 0;
            num_tree->remove(bool_int64, seq_id);
            if(search_field.facet) {
                remove_facet_token(search_field, search_index, std::to_string(value), seq_id);
            }
        }
    } else if(search_field.is_geopoint()) {
        auto geo_index = geopoint_index[field_name];
        S2RegionTermIndexer::Options options;
        options.set_index_contains_points_only(true);
        S2RegionTermIndexer indexer(options);

        const std::vector<std::vector<double>>& latlongs = search_field.is_single_geopoint() ?
                                                           std::vector<std::vector<double>>{document[field_name].get<std::vector<double>>()} :
                                                           document[field_name].get<std::vector<std::vector<double>>>();

        for(const std::vector<double>& latlong: latlongs) {
            S2Point point = S2LatLng::FromDegrees(latlong[0], latlong[1]).ToPoint();
            for(const auto& term: indexer.GetIndexTerms(point, "")) {
                auto term_it = geo_index->find(term);
                if(term_it == geo_index->end()) {
                    continue;
                }
                std::vector<uint32_t>& ids = term_it->second;
                ids.erase(std::remove(ids.begin(), ids.end(), seq_id), ids.end());
                if(ids.empty()) {
                    geo_index->erase(term);
                }
            }
        }

        if(!search_field.is_single_geopoint()) {
            spp::sparse_hash_map<uint32_t, int64_t*>*& field_geo_array_map = geo_array_index.at(field_name);
            auto geo_array_it = field_geo_array_map->find(seq_id);
            if(geo_array_it != field_geo_array_map->end()) {
                delete [] geo_array_it->second;
                field_geo_array_map->erase(seq_id);
            }
        }
    }

    // remove facets
    const auto& field_facets_it = facet_index_v3.find(field_name);
    if(field_facets_it != facet_index_v3.end()) {
        const auto& fvalues_it = field_facets_it->second[seq_id % ARRAY_FACET_DIM]->find(seq_id);
        if(fvalues_it != field_facets_it->second[seq_id % ARRAY_FACET_DIM]->end()) {
            field_facets_it->second[seq_id % ARRAY_FACET_DIM]->erase(fvalues_it);
        }
    }

    const auto& field_single_val_facets_it = single_val_facet_index_v3.find(field_name);
    if(field_single_val_facets_it != single_val_facet_index_v3.end()) {
        const auto& fvalues_it = field_single_val_facets_it->second[seq_id % ARRAY_FACET_DIM]->find(seq_id);
        if(fvalues_it != field_single_val_facets_it->second[seq_id % ARRAY_FACET_DIM]->end()) {
            field_single_val_facets_it->second[seq_id % ARRAY_FACET_DIM]->erase(fvalues_it);
        }
    }

    // remove sort field
    if(sort_index.count(field_name) != 0) {
        sort_index[field_name]->erase(seq_id);
    }

    if(str_sort_index.count(field_name) != 0) {
        str_sort_index[field_name]->remove(seq_id);
    }
}

Option<uint32_t> Index::remove(const uint32_t seq_id, const nlohmann::json & document,
                               const std::vector<field>& del_fields, const bool is_update) {
    std::unique_lock lock(mutex);

    // The exception during removal is mostly because of an edge case with auto schema detection:
    // Value indexed as Type T but later if field is dropped and reindexed in another type X,
    // the on-disk data will differ from the newly detected type on schema. We've to log the error,
    // but have to ignore the field and proceed because there's no leak caused here.

    if(!del_fields.empty()) {
        for(auto& the_field: del_fields) {
            if(!document.contains(the_field.name)) {
                // could be an optional field
                continue;
            }

            try {
                remove_field(seq_id, document, the_field.name);
            } catch(const std::exception& e) {
                LOG(WARNING) << "Error while removing field `" << the_field.name << "` from document, message: "
                             << e.what();
            }
        }
    } else {
        for(auto it = document.begin(); it != document.end(); ++it) {
            const std::string& field_name = it.key();
            try {
                remove_field(seq_id, document, field_name);
            } catch(const std::exception& e) {
                LOG(WARNING) << "Error while removing field `" << field_name << "` from document, message: "
                             << e.what();
            }
        }
    }

    if(!is_update) {
        seq_ids->erase(seq_id);
    }

    return Option<uint32_t>(seq_id);
}

void Index::tokenize_string_field(const nlohmann::json& document, const field& search_field,
                                  std::vector<std::string>& tokens, const std::string& locale,
                                  const std::vector<char>& symbols_to_index,
                                  const std::vector<char>& token_separators) {

    const std::string& field_name = search_field.name;

    if(search_field.type == field_types::STRING) {
        Tokenizer(document[field_name], true, false, locale, symbols_to_index, token_separators).tokenize(tokens);
    } else if(search_field.type == field_types::STRING_ARRAY) {
        const std::vector<std::string>& values = document[field_name].get<std::vector<std::string>>();
        for(const std::string & value: values) {
            Tokenizer(value, true, false, locale, symbols_to_index, token_separators).tokenize(tokens);
        }
    }
}

art_leaf* Index::get_token_leaf(const std::string & field_name, const unsigned char* token, uint32_t token_len) {
    std::shared_lock lock(mutex);
    const art_tree *t = search_index.at(field_name);
    return (art_leaf*) art_search(t, token, (int) token_len);
}

const spp::sparse_hash_map<std::string, art_tree *> &Index::_get_search_index() const {
    return search_index;
}

const spp::sparse_hash_map<std::string, num_tree_t*>& Index::_get_numerical_index() const {
    return numerical_index;
}

const spp::sparse_hash_map<std::string, array_mapped_infix_t>& Index::_get_infix_index() const {
    return infix_index;
};

const spp::sparse_hash_map<std::string, hnsw_index_t*>& Index::_get_vector_index() const {
    return vector_index;
}

void Index::refresh_schemas(const std::vector<field>& new_fields, const std::vector<field>& del_fields) {
    std::unique_lock lock(mutex);

    for(const auto & new_field: new_fields) {
        if(!new_field.index || new_field.is_dynamic()) {
            continue;
        }

        search_schema.emplace(new_field.name, new_field);

        if(new_field.type == field_types::FLOAT_ARRAY && new_field.num_dim > 0) {
            auto hnsw_index = new hnsw_index_t(new_field.num_dim, 1024, new_field.vec_dist);
            vector_index.emplace(new_field.name, hnsw_index);
            continue;
        }

        if(new_field.is_sortable()) {
            if(new_field.is_num_sortable()) {
                spp::sparse_hash_map<uint32_t, int64_t> * doc_to_score = new spp::sparse_hash_map<uint32_t, int64_t>();
                sort_index.emplace(new_field.name, doc_to_score);
            } else if(new_field.is_str_sortable()) {
                str_sort_index.emplace(new_field.name, new adi_tree_t);
            }
        }

        if(search_index.count(new_field.name) == 0) {
            if(new_field.is_string() || field_types::is_string_or_array(new_field.type)) {
                art_tree *t = new art_tree;
                art_tree_init(t);
                search_index.emplace(new_field.name, t);
            } else if(new_field.is_geopoint()) {
                auto field_geo_index = new spp::sparse_hash_map<std::string, std::vector<uint32_t>>();
                geopoint_index.emplace(new_field.name, field_geo_index);
                if(!new_field.is_single_geopoint()) {
                    auto geo_array_map = new spp::sparse_hash_map<uint32_t, int64_t*>();
                    geo_array_index.emplace(new_field.name, geo_array_map);
                }
            } else {
                num_tree_t* num_tree = new num_tree_t;
                numerical_index.emplace(new_field.name, num_tree);
            }
        }

        if(new_field.is_facet()) {

            initialize_facet_indexes(new_field);

            // initialize for non-string facet fields
            if(!new_field.is_string()) {
                art_tree *ft = new art_tree;
                art_tree_init(ft);
                search_index.emplace(new_field.faceted_name(), ft);
            }
        }

        if(new_field.infix) {
            array_mapped_infix_t infix_sets(ARRAY_INFIX_DIM);
            for(auto& infix_set: infix_sets) {
                infix_set = new tsl::htrie_set<char>();
            }

            infix_index.emplace(new_field.name, infix_sets);
        }
    }

    for(const auto & del_field: del_fields) {
        if(search_schema.count(del_field.name) == 0) {
            // could be a dynamic field
            continue;
        }

        search_schema.erase(del_field.name);

        if(!del_field.index) {
            continue;
        }

        if(del_field.is_string() || field_types::is_string_or_array(del_field.type)) {
            art_tree_destroy(search_index[del_field.name]);
            delete search_index[del_field.name];
            search_index.erase(del_field.name);
        } else if(del_field.is_geopoint()) {
            delete geopoint_index[del_field.name];
            geopoint_index.erase(del_field.name);

            if(!del_field.is_single_geopoint()) {
                spp::sparse_hash_map<uint32_t, int64_t*>* geo_array_map = geo_array_index[del_field.name];
                for(auto& kv: *geo_array_map) {
                    delete kv.second;
                }
                delete geo_array_map;
                geo_array_index.erase(del_field.name);
            }
        } else {
            delete numerical_index[del_field.name];
            numerical_index.erase(del_field.name);
        }

        if(del_field.is_sortable()) {
            if(del_field.is_num_sortable()) {
                delete sort_index[del_field.name];
                sort_index.erase(del_field.name);
            } else if(del_field.is_str_sortable()) {
                delete str_sort_index[del_field.name];
                str_sort_index.erase(del_field.name);
            }
        }

        if(del_field.is_facet()) {
            if(del_field.is_array()) {
                auto& arr = facet_index_v3[del_field.name];
                for(size_t i = 0; i < ARRAY_FACET_DIM; i++) {
                    delete arr[i];
                }
                
                facet_index_v3.erase(del_field.name);
            } else {
                auto& arr = single_val_facet_index_v3[del_field.name];
                for(size_t i = 0; i < ARRAY_FACET_DIM; i++) {
                    delete arr[i];
                }

                single_val_facet_index_v3.erase(del_field.name);
            }

            if(!del_field.is_string()) {
                art_tree_destroy(search_index[del_field.faceted_name()]);
                delete search_index[del_field.faceted_name()];
                search_index.erase(del_field.faceted_name());
            }
        }

        if(del_field.infix) {
            auto& infix_set = infix_index[del_field.name];
            for(size_t i = 0; i < infix_set.size(); i++) {
                delete infix_set[i];
            }

            infix_index.erase(del_field.name);
        }

        if(del_field.num_dim) {
            auto hnsw_index = vector_index[del_field.name];
            delete hnsw_index;
            vector_index.erase(del_field.name);
        }
    }
}

void Index::handle_doc_ops(const tsl::htrie_map<char, field>& search_schema,
                           nlohmann::json& update_doc, const nlohmann::json& old_doc) {

    /*
        {
           "$operations": {
              "increment": {"likes": 1, "views": 20}
           }
        }
    */

    auto ops_it = update_doc.find("$operations");
    if(ops_it != update_doc.end()) {
        const auto& operations = ops_it.value();
        if(operations.contains("increment") && operations["increment"].is_object()) {
            for(const auto& item: operations["increment"].items()) {
                auto field_it = search_schema.find(item.key());
                if(field_it != search_schema.end()) {
                    if(field_it->type == field_types::INT32 && item.value().is_number_integer()) {
                        int32_t existing_value = 0;
                        if(old_doc.contains(item.key())) {
                            existing_value = old_doc[item.key()].get<int32_t>();
                        }

                        auto updated_value = existing_value + item.value().get<int32>();
                        update_doc[item.key()] = updated_value;
                    }
                }
            }
        }

        update_doc.erase("$operations");
    }
}

void Index::get_doc_changes(const index_operation_t op, const tsl::htrie_map<char, field>& search_schema,
                            nlohmann::json& update_doc, const nlohmann::json& old_doc, nlohmann::json& new_doc,
                            nlohmann::json& del_doc) {

    if(op == UPSERT) {
        new_doc = update_doc;
        // since UPSERT could replace a doc with lesser fields, we have to add those missing fields to del_doc
        for(auto it = old_doc.begin(); it != old_doc.end(); ++it) {
            if(it.value().is_object() || (it.value().is_array() && (it.value().empty() || it.value()[0].is_object()))) {
                continue;
            }

            if(!update_doc.contains(it.key())) {
                del_doc[it.key()] = it.value();
            }
        }
    } else {
        handle_doc_ops(search_schema, update_doc, old_doc);
        new_doc = old_doc;
        new_doc.merge_patch(update_doc);

        if(old_doc.contains(".flat")) {
            new_doc[".flat"] = old_doc[".flat"];
            for(auto& fl: update_doc[".flat"]) {
                new_doc[".flat"].push_back(fl);
            }
        }
    }

    auto it = update_doc.begin();
    while(it != update_doc.end()) {
        if(it.value().is_object() || (it.value().is_array() && !it.value().empty() && it.value()[0].is_object())) {
            ++it;
            continue;
        }

        if(it.value().is_null()) {
            // null values should not be indexed
            new_doc.erase(it.key());
            if(old_doc.contains(it.key())) {
                del_doc[it.key()] = old_doc[it.key()];
            }
            it = update_doc.erase(it);
            continue;
        }

        if(old_doc.contains(it.key())) {
            if(old_doc[it.key()] == it.value()) {
                // unchanged so should not be part of update doc
                it = update_doc.erase(it);
                continue;
            } else {
                // delete this old value from index
                del_doc[it.key()] = old_doc[it.key()];
            }
        }

        it++;
    }
}

size_t Index::num_seq_ids() const {
    std::shared_lock lock(mutex);
    return seq_ids->num_ids();
}

Option<bool> Index::seq_ids_outside_top_k(const std::string& field_name, size_t k,
                                          std::vector<uint32_t>& outside_seq_ids) {
    auto field_it = numerical_index.find(field_name);

    if(field_it == sort_index.end()) {
        return Option<bool>(400, "Field not found in numerical index.");
    }

    field_it->second->seq_ids_outside_top_k(k, outside_seq_ids);
    return Option<bool>(true);
}

void Index::resolve_space_as_typos(std::vector<std::string>& qtokens, const string& field_name,
                                   std::vector<std::vector<std::string>>& resolved_queries) const {

    auto tree_it = search_index.find(field_name);

    if(tree_it == search_index.end()) {
        return ;
    }

    // we will try to find a verbatim match first

    art_tree* t = tree_it->second;
    std::vector<art_leaf*> leaves;

    for(const std::string& token: qtokens) {
        art_leaf* leaf = (art_leaf *) art_search(t, (const unsigned char*) token.c_str(),
                                                 token.length()+1);
        if(leaf == nullptr) {
            break;
        }

        leaves.push_back(leaf);
    }

    // When we cannot find verbatim match, we can try concatting and splitting query tokens for alternatives.

    // Concatenation:

    size_t qtokens_size = std::min<size_t>(5, qtokens.size());  // only first 5 tokens will be considered

    if(qtokens.size() > 1) {
        // a) join all tokens to form a single string
        const string& all_tokens_query = StringUtils::join(qtokens, "");
        if(art_search(t, (const unsigned char*) all_tokens_query.c_str(), all_tokens_query.length()+1) != nullptr) {
            resolved_queries.push_back({all_tokens_query});
            return;
        }

        // b) join 2 adjacent tokens in a sliding window (provided they are atleast 2 tokens in size)

        for(size_t i = 0; i < qtokens_size-1 && qtokens_size > 2; i++) {
            std::vector<std::string> candidate_tokens;

            for(size_t j = 0; j < i; j++) {
                candidate_tokens.push_back(qtokens[j]);
            }

            std::string joined_tokens = qtokens[i] + qtokens[i+1];
            candidate_tokens.push_back(joined_tokens);

            for(size_t j = i+2; j < qtokens.size(); j++) {
                candidate_tokens.push_back(qtokens[j]);
            }

            leaves.clear();

            for(auto& token: candidate_tokens) {
                art_leaf* leaf = static_cast<art_leaf*>(art_search(t, (const unsigned char*) token.c_str(),
                                                                   token.length() + 1));
                if(leaf == nullptr) {
                    break;
                }

                leaves.push_back(leaf);
            }

            if(candidate_tokens.size() == leaves.size() && common_results_exist(leaves, false)) {
                resolved_queries.push_back(candidate_tokens);
                return;
            }
        }
    }

    // concats did not work, we will try splitting individual tokens
    for(size_t i = 0; i < qtokens_size; i++) {
        std::vector<std::string> candidate_tokens;

        for(size_t j = 0; j < i; j++) {
            candidate_tokens.push_back(qtokens[j]);
        }

        const std::string& token = qtokens[i];
        bool found_split = false;

        for(size_t ci = 1; ci < token.size(); ci++) {
            std::string first_part = token.substr(0, token.size()-ci);
            art_leaf* first_leaf = static_cast<art_leaf*>(art_search(t, (const unsigned char*) first_part.c_str(),
                                                                     first_part.length() + 1));

            if(first_leaf != nullptr) {
                // check if rest of the string is also a valid token
                std::string second_part = token.substr(token.size()-ci, ci);
                art_leaf* second_leaf = static_cast<art_leaf*>(art_search(t, (const unsigned char*) second_part.c_str(),
                                                                          second_part.length() + 1));

                std::vector<art_leaf*> part_leaves = {first_leaf, second_leaf};
                if(second_leaf != nullptr && common_results_exist(part_leaves, true)) {
                    candidate_tokens.push_back(first_part);
                    candidate_tokens.push_back(second_part);
                    found_split = true;
                    break;
                }
            }
        }

        if(!found_split) {
            continue;
        }

        for(size_t j = i+1; j < qtokens.size(); j++) {
            candidate_tokens.push_back(qtokens[j]);
        }

        leaves.clear();

        for(auto& candidate_token: candidate_tokens) {
            art_leaf* leaf = static_cast<art_leaf*>(art_search(t, (const unsigned char*) candidate_token.c_str(),
                                                               candidate_token.length() + 1));
            if(leaf == nullptr) {
                break;
            }

            leaves.push_back(leaf);
        }

        if(common_results_exist(leaves, false)) {
            resolved_queries.push_back(candidate_tokens);
            return;
        }
    }
}

bool Index::common_results_exist(std::vector<art_leaf*>& leaves, bool must_match_phrase) const {
    std::vector<uint32_t> result_ids;
    std::vector<void*> leaf_vals;

    for(auto leaf: leaves) {
        leaf_vals.push_back(leaf->values);
    }

    posting_t::intersect(leaf_vals, result_ids);

    if(result_ids.empty()) {
        return false;
    }

    if(!must_match_phrase) {
        return !result_ids.empty();
    }

    uint32_t* phrase_ids = new uint32_t[result_ids.size()];
    size_t num_phrase_ids;

    posting_t::get_phrase_matches(leaf_vals, false, &result_ids[0], result_ids.size(),
                                  phrase_ids, num_phrase_ids);
    bool phrase_exists = (num_phrase_ids != 0);
    delete [] phrase_ids;
    return phrase_exists;
}


void Index::batch_embed_fields(std::vector<index_record*>& records, 
                                       const tsl::htrie_map<char, field>& embedding_fields,
                                       const tsl::htrie_map<char, field> & search_schema) {
    for(const auto& field : embedding_fields) {
        std::vector<std::pair<index_record*, std::string>> texts_to_embed;
        auto indexing_prefix = TextEmbedderManager::get_instance().get_indexing_prefix(field.embed[fields::model_config]);
        for(auto& record : records) {
            if(!record->indexed.ok()) {
                continue;
            }
            nlohmann::json* document;
            if(record->is_update) {
                document = &record->new_doc;
            } else {
                document = &record->doc;
            }

            if(document == nullptr) {
                continue;
            }
            std::string text = indexing_prefix;
            auto embed_from = field.embed[fields::from].get<std::vector<std::string>>();
            for(const auto& field_name : embed_from) {
                auto field_it = search_schema.find(field_name);
                if(field_it.value().type == field_types::STRING) {
                    text += (*document)[field_name].get<std::string>() + " ";
                } else if(field_it.value().type == field_types::STRING_ARRAY) {
                    for(const auto& val : (*document)[field_name]) {
                        text += val.get<std::string>() + " ";
                    }
                }
            }
            if(text != indexing_prefix) {
                texts_to_embed.push_back(std::make_pair(record, text));
            }
        }

        if(texts_to_embed.empty()) {
            continue;
        }
        
        TextEmbedderManager& embedder_manager = TextEmbedderManager::get_instance();
        auto embedder_op = embedder_manager.get_text_embedder(field.embed[fields::model_config]);

        if(!embedder_op.ok()) {
            LOG(ERROR) << "Error while getting embedder for model: " << field.embed[fields::model_config];
            LOG(ERROR) << "Error: " << embedder_op.error();
            return;
        }

        // sort texts by length
        std::sort(texts_to_embed.begin(), texts_to_embed.end(),
                  [](const std::pair<index_record*, std::string>& a,
                     const std::pair<index_record*, std::string>& b) {
                      return a.second.size() < b.second.size();
                  });
        
        // get vector of texts
        std::vector<std::string> texts;
        for(const auto& text_to_embed : texts_to_embed) {
            texts.push_back(text_to_embed.second);
        }

        auto embeddings = embedder_op.get()->batch_embed(texts);

        for(size_t i = 0; i < embeddings.size(); i++) {
            auto& embedding_res = embeddings[i];
            if(!embedding_res.success) {
                texts_to_embed[i].first->embedding_res = embedding_res.error;
                texts_to_embed[i].first->index_failure(embedding_res.status_code, "");
                continue;
            }
            nlohmann::json* document;
            if(texts_to_embed[i].first->is_update) {
                document = &texts_to_embed[i].first->new_doc;
            } else {
                document = &texts_to_embed[i].first->doc;
            }
            (*document)[field.name] = embedding_res.embedding;
        }
    }
}

/*
// https://stackoverflow.com/questions/924171/geo-fencing-point-inside-outside-polygon
// NOTE: polygon and point should have been transformed with `transform_for_180th_meridian`
bool Index::is_point_in_polygon(const Geofence& poly, const GeoCoord &point) {
    int i, j;
    bool c = false;
    for (i = 0, j = poly.numVerts - 1; i < poly.numVerts; j = i++) {
        if ((((poly.verts[i].lat <= point.lat) && (point.lat < poly.verts[j].lat))
             || ((poly.verts[j].lat <= point.lat) && (point.lat < poly.verts[i].lat)))
            && (point.lon < (poly.verts[j].lon - poly.verts[i].lon) * (point.lat - poly.verts[i].lat)
                            / (poly.verts[j].lat - poly.verts[i].lat) + poly.verts[i].lon)) {
            c = !c;
        }
    }
    return c;
}
double Index::transform_for_180th_meridian(Geofence &poly) {
    double offset = 0.0;
    double maxLon = -1000, minLon = 1000;
    for(int v=0; v < poly.numVerts; v++) {
        if(poly.verts[v].lon < minLon) {
            minLon = poly.verts[v].lon;
        }
        if(poly.verts[v].lon > maxLon) {
            maxLon = poly.verts[v].lon;
        }
        if(std::abs(minLon - maxLon) > 180) {
            offset = 360.0;
        }
    }
    int i, j;
    for (i = 0, j = poly.numVerts - 1; i < poly.numVerts; j = i++) {
        if (poly.verts[i].lon < 0.0) {
            poly.verts[i].lon += offset;
        }
        if (poly.verts[j].lon < 0.0) {
            poly.verts[j].lon += offset;
        }
    }
    return offset;
}
void Index::transform_for_180th_meridian(GeoCoord &point, double offset) {
    point.lon = point.lon < 0.0 ? point.lon + offset : point.lon;
}
*/

