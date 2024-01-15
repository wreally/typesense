#include "collection.h"

#include <numeric>
#include <chrono>
#include <match_score.h>
#include <string_utils.h>
#include <art.h>
#include <rocksdb/write_batch.h>
#include <system_metrics.h>
#include <tokenizer.h>
#include <collection_manager.h>
#include <regex>
#include <list>
#include <posting.h>
#include <timsort.hpp>
#include "validator.h"
#include "topster.h"
#include "logger.h"
#include "thread_local_vars.h"
#include "vector_query_ops.h"
#include "embedder_manager.h"
#include "stopwords_manager.h"
#include "conversation_model.h"
#include "conversation_manager.h"
#include "conversation_model_manager.h"
#include "field.h"

const std::string override_t::MATCH_EXACT = "exact";
const std::string override_t::MATCH_CONTAINS = "contains";

struct sort_fields_guard_t {
    std::vector<sort_by> sort_fields_std;

    ~sort_fields_guard_t() {
        for(auto& sort_by_clause: sort_fields_std) {
            for (auto& eval_ids: sort_by_clause.eval.eval_ids_vec) {
                delete [] eval_ids;
            }
            delete [] sort_by_clause.eval.filter_trees;
        }
    }
};

Collection::Collection(const std::string& name, const uint32_t collection_id, const uint64_t created_at,
                       const uint32_t next_seq_id, Store *store, const std::vector<field> &fields,
                       const std::string& default_sorting_field,
                       const float max_memory_ratio, const std::string& fallback_field_type,
                       const std::vector<std::string>& symbols_to_index,
                       const std::vector<std::string>& token_separators,
                       const bool enable_nested_fields, std::shared_ptr<VQModel> vq_model) :
        name(name), collection_id(collection_id), created_at(created_at),
        next_seq_id(next_seq_id), store(store),
        fields(fields), default_sorting_field(default_sorting_field), enable_nested_fields(enable_nested_fields),
        max_memory_ratio(max_memory_ratio),
        fallback_field_type(fallback_field_type), dynamic_fields({}),
        symbols_to_index(to_char_array(symbols_to_index)), token_separators(to_char_array(token_separators)),
        index(init_index()), vq_model(vq_model) {
    
    if (vq_model) {
        vq_model->inc_collection_ref_count();
    }
    this->num_documents = 0;
}

Collection::~Collection() {
    std::unique_lock lifecycle_lock(lifecycle_mutex);
    std::unique_lock lock(mutex);
    delete index;
    delete synonym_index;   

    if (vq_model) {
        vq_model->dec_collection_ref_count();
        if (vq_model->get_collection_ref_count() == 0) {
            LOG(INFO) << "Unloading voice query model " << vq_model->get_model_name();
            VQModelManager::get_instance().delete_model(vq_model->get_model_name());
        }
    }
}

uint32_t Collection::get_next_seq_id() {
    std::shared_lock lock(mutex);
    store->increment(get_next_seq_id_key(name), 1);
    return next_seq_id++;
}

Option<bool> single_value_filter_query(nlohmann::json& document, const std::string& field_name,
                                       const std::string& ref_field_type, std::string& filter_query) {
    auto const& value = document[field_name];
    if (value.is_string() && ref_field_type == field_types::STRING) {
        filter_query[filter_query.size() - 1] = '=';
        filter_query += (" " + value.get<std::string>());
    } else if (value.is_number_integer() && (ref_field_type == field_types::INT64 ||
                                             (ref_field_type == field_types::INT32 && StringUtils::is_int32_t(std::to_string(value.get<int64_t>()))))) {
        filter_query += std::to_string(value.get<int64_t>());
    } else {
        return Option<bool>(400, "Field `" + field_name + "` must have `" + ref_field_type + "` value.");
    }

    return Option<bool>(true);
}

Option<bool> Collection::add_reference_helper_fields(nlohmann::json& document, const tsl::htrie_map<char, field>& schema,
                                                     const spp::sparse_hash_map<std::string, reference_pair>& reference_fields,
                                                     tsl::htrie_set<char>& object_reference_helper_fields) {
    tsl::htrie_set<char> flat_fields;
    if (!reference_fields.empty() && document.contains(".flat")) {
        for (const auto &item: document[".flat"].get<std::vector<std::string>>()) {
            flat_fields.insert(item);
        }
    }

    // Add reference helper fields in the document.
    for (auto const& pair: reference_fields) {
        auto field_name = pair.first;
        auto optional = schema.at(field_name).optional;
        if (!optional && document.count(field_name) != 1) {
            return Option<bool>(400, "Missing the required reference field `" + field_name
                                             + "` in the document.");
        } else if (document.count(field_name) != 1) {
            continue;
        }

        if (document.count(fields::reference_helper_fields) == 0) {
            document[fields::reference_helper_fields] = nlohmann::json::array();
        }

        auto reference_pair = pair.second;
        auto reference_collection_name = reference_pair.collection;
        auto reference_field_name = reference_pair.field;
        auto& cm = CollectionManager::get_instance();
        auto ref_collection = cm.get_collection(reference_collection_name);
        if (ref_collection == nullptr) {
            return Option<bool>(400, "Referenced collection `" + reference_collection_name
                                             + "` not found.");
        }

        auto const reference_helper_field = field_name + fields::REFERENCE_HELPER_FIELD_SUFFIX;
        document[fields::reference_helper_fields] += reference_helper_field;

        if (reference_field_name == "id") {
            auto id_field_type_error_op =  Option<bool>(400, "Field `" + field_name + "` must have string value.");
            if (document[field_name].is_array()) {
                document[reference_helper_field] = nlohmann::json::array();
                for (const auto &item: document[field_name].items()) {
                    if (!item.value().is_string()) {
                        return id_field_type_error_op;
                    }

                    auto id = item.value().get<std::string>();
                    auto ref_doc_id_op = ref_collection->doc_id_to_seq_id_with_lock(id);
                    if (!ref_doc_id_op.ok()) {
                        return Option<bool>(400, "Referenced document having `id: " + id +
                                                 "` not found in the collection `" +
                                                 reference_collection_name + "`." );
                    }

                    document[reference_helper_field] += ref_doc_id_op.get();
                }
            } else if (document[field_name].is_string()) {
                auto id = document[field_name].get<std::string>();
                auto ref_doc_id_op = ref_collection->doc_id_to_seq_id_with_lock(id);
                if (!ref_doc_id_op.ok()) {
                    return Option<bool>(400, "Referenced document having `id: " + id +
                                             "` not found in the collection `" +
                                             reference_collection_name + "`." );
                }

                document[reference_helper_field] = ref_doc_id_op.get();
            } else {
                return id_field_type_error_op;
            }

            continue;
        }

        if (ref_collection->get_schema().count(reference_field_name) == 0) {
            return Option<bool>(400, "Referenced field `" + reference_field_name +
                                             "` not found in the collection `" + reference_collection_name + "`.");
        }

        auto const ref_field = ref_collection->get_schema().at(reference_field_name);
        if (!ref_field.index) {
            return Option<bool>(400, "Referenced field `" + reference_field_name +
                                             "` in the collection `" + reference_collection_name + "` must be indexed.");
        }

        // Create filter query from the value(s) in the reference field and get the reference doc id(s).
        std::string filter_query = reference_field_name + ": ";
        std::string ref_field_type = ref_field.is_string() ? field_types::STRING :
                                    ref_field.is_int32() ? field_types::INT32 :
                                    ref_field.is_int64() ? field_types::INT64 : field_types::NIL;

        if (ref_field_type == field_types::NIL) {
            return Option<bool>(400, "Cannot add a reference to `" + reference_collection_name + "." + reference_field_name +
                                "` of type `" + ref_field.type + "`.");
        }

        bool is_object_reference_field = flat_fields.count(field_name) != 0;
        std::string object_key;
        bool is_object_array;
        if (is_object_reference_field) {
            object_reference_helper_fields.insert(reference_helper_field);

            std::vector<std::string> tokens;
            StringUtils::split(field_name, tokens, ".");
            if (schema.count(tokens[0]) == 0) {
                return Option<bool>(400, "Could not find `" + tokens[0] + "` object/object[] field in the schema.");
            }
            object_key = tokens[0];
            is_object_array = schema.at(object_key).is_array();
        }

        if (is_object_reference_field && is_object_array) {
            if (!document[field_name].is_array()) {
                return Option<bool>(400, "Expected `" + field_name + "` to be an array.");
            }

            document[reference_helper_field] = nlohmann::json::array();
            nlohmann::json temp_doc; // To store singular values of `field_name` field.

            std::vector<std::string> keys;
            StringUtils::split(field_name, keys, ".");
            auto const& object_array = document[keys[0]];

            for (uint32_t i = 0; i < object_array.size(); i++) {
                if (optional && object_array[i].count(keys[1]) == 0) {
                    continue;
                } else if (object_array[i].count(keys[1]) == 0) {
                    return Option<bool>(400, "Object at index `" + std::to_string(i) + "` is missing `" + field_name + "`.");
                }

                temp_doc[field_name] = object_array[i].at(keys[1]);
                auto single_value_filter_query_op = single_value_filter_query(temp_doc, field_name, ref_field_type,
                                                                              filter_query);
                if (!single_value_filter_query_op.ok()) {
                    return single_value_filter_query_op;
                }

                filter_result_t filter_result;
                auto filter_ids_op = ref_collection->get_filter_ids(filter_query, filter_result);
                if (!filter_ids_op.ok()) {
                    return filter_ids_op;
                }

                if (filter_result.count != 1) {
                    // Constraints similar to foreign key apply here. The reference match must be unique and not null.
                    return  Option<bool>(400, filter_result.count < 1 ?
                                              "Reference document having `" + filter_query + "` not found in the collection `"
                                              + reference_collection_name + "`." :
                                              "Multiple documents having `" + filter_query + "` found in the collection `" +
                                              reference_collection_name + "`.");
                }

                // Adding the index of the object along with referenced doc id to account for the scenario where a
                // reference field of an object array might be optional and missing.
                document[reference_helper_field] += nlohmann::json::array({i, filter_result.docs[0]});
                filter_query = reference_field_name + ": ";
            }
            continue;
        }

        if (document[field_name].is_array()) {
            if (ref_field_type == field_types::STRING) {
                filter_query[filter_query.size() - 1] = '=';
                filter_query += " [";
            } else {
                filter_query += "[";
            }
            for (const auto &item: document[field_name].items()) {
                auto const& item_value = item.value();
                if (item_value.is_string() && ref_field_type == field_types::STRING) {
                    filter_query += item_value.get<std::string>();
                } else if (item_value.is_number_integer() && (ref_field_type == field_types::INT64 ||
                    (ref_field_type == field_types::INT32 && StringUtils::is_int32_t(std::to_string(item_value.get<int64_t>()))))) {
                    filter_query += std::to_string(item_value.get<int64_t>());
                } else {
                    return Option<bool>(400, "Field `" + field_name + "` must only have `" + ref_field_type + "` values.");
                }
                filter_query += ",";
            }
            filter_query[filter_query.size() - 1] = ']';
        } else {
            auto single_value_filter_query_op = single_value_filter_query(document, field_name, ref_field_type,
                                                                          filter_query);
            if (!single_value_filter_query_op.ok()) {
                return single_value_filter_query_op;
            }
        }

        filter_result_t filter_result;
        auto filter_ids_op = ref_collection->get_filter_ids(filter_query, filter_result);
        if (!filter_ids_op.ok()) {
            return filter_ids_op;
        }

        if (document[field_name].is_array()) {
            document[reference_helper_field] = nlohmann::json::array();
            for (uint32_t i = 0; i < filter_result.count; i++) {
                document[reference_helper_field] += filter_result.docs[i];
            }
        } else {
            if (filter_result.count != 1) {
                // Constraints similar to foreign key apply here. The reference match must be unique and not null.
                return  Option<bool>(400, filter_result.count < 1 ?
                                          "Reference document having `" + filter_query + "` not found in the collection `"
                                          + reference_collection_name + "`." :
                                          "Multiple documents having `" + filter_query + "` found in the collection `" +
                                          reference_collection_name + "`.");
            }

            document[reference_helper_field] = filter_result.docs[0];
        }
    }

    return Option<bool>(true);
}

Option<doc_seq_id_t> Collection::to_doc(const std::string & json_str, nlohmann::json& document,
                                        const index_operation_t& operation,
                                        const DIRTY_VALUES dirty_values,
                                        const std::string& id) {
    try {
        document = nlohmann::json::parse(json_str);
    } catch(const std::exception& e) {
        LOG(ERROR) << "JSON error: " << e.what();
        return Option<doc_seq_id_t>(400, std::string("Bad JSON: ") + e.what());
    }

    if(!document.is_object()) {
        return Option<doc_seq_id_t>(400, "Bad JSON: not a properly formed document.");
    }

    if(document.count("id") != 0 && id != "" && document["id"] != id) {
        return Option<doc_seq_id_t>(400, "The `id` of the resource does not match the `id` in the JSON body.");
    }

    if(document.count("id") == 0 && !id.empty()) {
        // use the explicit ID (usually from a PUT request) if document body does not have it
        document["id"] = id;
    }

    if(document.count("id") != 0 && document["id"] == "") {
        return Option<doc_seq_id_t>(400, "The `id` should not be empty.");
    }

    if(document.count("id") == 0) {
        if(operation == UPDATE) {
            return Option<doc_seq_id_t>(400, "For update, the `id` key must be provided.");
        }
        // for UPSERT, EMPLACE or CREATE, if a document does not have an ID, we will treat it as a new doc
        uint32_t seq_id = get_next_seq_id();
        document["id"] = std::to_string(seq_id);

        return Option<doc_seq_id_t>(doc_seq_id_t{seq_id, true});
    } else {
        if(!document["id"].is_string()) {
            return Option<doc_seq_id_t>(400, "Document's `id` field should be a string.");
        }

        const std::string& doc_id = document["id"];

        // try to get the corresponding sequence id from disk if present
        std::string seq_id_str;
        StoreStatus seq_id_status = store->get(get_doc_id_key(doc_id), seq_id_str);

        if(seq_id_status == StoreStatus::ERROR) {
            return Option<doc_seq_id_t>(500, "Error fetching the sequence key for document with id: " + doc_id);
        }

        if(seq_id_status == StoreStatus::FOUND) {
            if(operation == CREATE) {
                return Option<doc_seq_id_t>(409, std::string("A document with id ") + doc_id + " already exists.");
            }
            


            // UPSERT, EMPLACE or UPDATE
            uint32_t seq_id = (uint32_t) std::stoul(seq_id_str);

            return Option<doc_seq_id_t>(doc_seq_id_t{seq_id, false});

        } else {
            if(operation == UPDATE) {
                // for UPDATE, a document with given ID must be found
                return Option<doc_seq_id_t>(404, "Could not find a document with id: " + doc_id);
            } else {
                // for UPSERT, EMPLACE or CREATE, if a document with given ID is not found, we will treat it as a new doc
                uint32_t seq_id = get_next_seq_id();

                return Option<doc_seq_id_t>(doc_seq_id_t{seq_id, true});
            }
        }
    }
}

nlohmann::json Collection::get_summary_json() const {
    std::shared_lock lock(mutex);

    nlohmann::json json_response;

    json_response["name"] = name;
    json_response["num_documents"] = num_documents.load();
    json_response["created_at"] = created_at.load();
    json_response["enable_nested_fields"] = enable_nested_fields;
    json_response["token_separators"] = nlohmann::json::array();
    json_response["symbols_to_index"] = nlohmann::json::array();

    for(auto c: symbols_to_index) {
        json_response["symbols_to_index"].push_back(std::string(1, c));
    }

    for(auto c: token_separators) {
        json_response["token_separators"].push_back(std::string(1, c));
    }

    nlohmann::json fields_arr;
    const std::regex sequence_id_pattern(".*_sequence_id$");

    for(const field & coll_field: fields) {
        if (std::regex_match(coll_field.name, sequence_id_pattern)) {
            // Don't add foo_sequence_id field.
            continue;
        }

        nlohmann::json field_json;
        field_json[fields::name] = coll_field.name;
        field_json[fields::type] = coll_field.type;
        field_json[fields::facet] = coll_field.facet;
        field_json[fields::optional] = coll_field.optional;
        field_json[fields::index] = coll_field.index;
        field_json[fields::sort] = coll_field.sort;
        field_json[fields::infix] = coll_field.infix;
        field_json[fields::locale] = coll_field.locale;
        if(coll_field.embed.count(fields::from) != 0) {
            field_json[fields::embed] = coll_field.embed;

            if(field_json[fields::embed].count(fields::model_config) != 0) {
                hide_credential(field_json[fields::embed][fields::model_config], "api_key");
                hide_credential(field_json[fields::embed][fields::model_config], "access_token");
                hide_credential(field_json[fields::embed][fields::model_config], "refresh_token");
                hide_credential(field_json[fields::embed][fields::model_config], "client_id");
                hide_credential(field_json[fields::embed][fields::model_config], "client_secret");
                hide_credential(field_json[fields::embed][fields::model_config], "project_id");
            }
        }

        if(coll_field.num_dim > 0) {
            field_json[fields::num_dim] = coll_field.num_dim;
            field_json[fields::vec_dist] = magic_enum::enum_name(coll_field.vec_dist);
        }

        if (!coll_field.reference.empty()) {
            field_json[fields::reference] = coll_field.reference;
        }

        fields_arr.push_back(field_json);
    }

    json_response["fields"] = fields_arr;
    json_response["default_sorting_field"] = default_sorting_field;
    
    if(vq_model) {
        json_response["voice_query_model"] = nlohmann::json::object();
        json_response["voice_query_model"]["model_name"] = vq_model->get_model_name();
    }

    return json_response;
}

Option<nlohmann::json> Collection::add(const std::string & json_str,
                                       const index_operation_t& operation, const std::string& id,
                                       const DIRTY_VALUES& dirty_values) {
    nlohmann::json document;
    std::vector<std::string> json_lines = {json_str};
    const nlohmann::json& res = add_many(json_lines, document, operation, id, dirty_values, false, false);

    if(!res["success"].get<bool>()) {
        nlohmann::json res_doc;

        try {
            res_doc = nlohmann::json::parse(json_lines[0]);
        } catch(const std::exception& e) {
            LOG(ERROR) << "JSON error: " << e.what();
            return Option<nlohmann::json>(400, std::string("Bad JSON: ") + e.what());
        }

        return Option<nlohmann::json>(res_doc["code"].get<size_t>(), res_doc["error"].get<std::string>());
    }

    return Option<nlohmann::json>(document);
}

nlohmann::json Collection::add_many(std::vector<std::string>& json_lines, nlohmann::json& document,
                                    const index_operation_t& operation, const std::string& id,
                                    const DIRTY_VALUES& dirty_values, const bool& return_doc, const bool& return_id,
                                    const size_t remote_embedding_batch_size,
                                    const size_t remote_embedding_timeout_ms,
                                    const size_t remote_embedding_num_tries) {
    //LOG(INFO) << "Memory ratio. Max = " << max_memory_ratio << ", Used = " << SystemMetrics::used_memory_ratio();
    std::vector<index_record> index_records;

    const size_t index_batch_size = 1000;
    size_t num_indexed = 0;
    //bool exceeds_memory_limit = false;

    // ensures that document IDs are not repeated within the same batch
    std::set<std::string> batch_doc_ids;

    for(size_t i=0; i < json_lines.size(); i++) {
        const std::string & json_line = json_lines[i];
        Option<doc_seq_id_t> doc_seq_id_op = to_doc(json_line, document, operation, dirty_values, id);

        const uint32_t seq_id = doc_seq_id_op.ok() ? doc_seq_id_op.get().seq_id : 0;
        index_record record(i, seq_id, document, operation, dirty_values);

        // NOTE: we overwrite the input json_lines with result to avoid memory pressure

        record.is_update = false;
        bool repeated_doc = false;

        if(!doc_seq_id_op.ok()) {
            record.index_failure(doc_seq_id_op.code(), doc_seq_id_op.error());
        } else {
            const std::string& doc_id = record.doc["id"].get<std::string>();
            repeated_doc = (batch_doc_ids.find(doc_id) != batch_doc_ids.end());

            if(repeated_doc) {
                // when a document repeats, we send the batch until this document so that we can deal with conflicts
                i--;
                goto do_batched_index;
            }

            record.is_update = !doc_seq_id_op.get().is_new;

            if(record.is_update) {
                get_document_from_store(get_seq_id_key(seq_id), record.old_doc);
            }

            batch_doc_ids.insert(doc_id);

            // if `fallback_field_type` or `dynamic_fields` is enabled, update schema first before indexing
            if(!fallback_field_type.empty() || !dynamic_fields.empty() || !nested_fields.empty() || !reference_fields.empty()) {
                std::vector<field> new_fields;
                std::unique_lock lock(mutex);

                Option<bool> new_fields_op = detect_new_fields(record.doc, dirty_values,
                                                               search_schema, dynamic_fields,
                                                               nested_fields,
                                                               fallback_field_type,
                                                               record.is_update,
                                                               new_fields,
                                                               enable_nested_fields,
                                                               reference_fields, object_reference_helper_fields);
                if(!new_fields_op.ok()) {
                    record.index_failure(new_fields_op.code(), new_fields_op.error());
                }

                else if(!new_fields.empty()) {
                    bool found_new_field = false;
                    for(auto& new_field: new_fields) {
                        if(search_schema.find(new_field.name) == search_schema.end()) {
                            found_new_field = true;
                            search_schema.emplace(new_field.name, new_field);
                            fields.emplace_back(new_field);
                            if(new_field.nested) {
                                nested_fields.emplace(new_field.name, new_field);
                            }
                        }
                    }

                    if(found_new_field) {
                        auto persist_op = persist_collection_meta();
                        if(!persist_op.ok()) {
                            record.index_failure(persist_op.code(), persist_op.error());
                        } else {
                            index->refresh_schemas(new_fields, {});
                        }
                    }
                }
            }
        }

        index_records.emplace_back(std::move(record));

        do_batched_index:


        if((i+1) % index_batch_size == 0 || i == json_lines.size()-1 || repeated_doc) {
            batch_index(index_records, json_lines, num_indexed, return_doc, return_id, remote_embedding_batch_size, remote_embedding_timeout_ms, remote_embedding_num_tries);

            // to return the document for the single doc add cases
            if(index_records.size() == 1) {
                const auto& rec = index_records[0];
                document = rec.is_update ? rec.new_doc : rec.doc;
                remove_flat_fields(document);
                remove_reference_helper_fields(document);
            }
            index_records.clear();
            batch_doc_ids.clear();
        }
    }

    nlohmann::json resp_summary;
    resp_summary["num_imported"] = num_indexed;
    resp_summary["success"] = (num_indexed == json_lines.size());

    return resp_summary;
}

Option<nlohmann::json> Collection::update_matching_filter(const std::string& filter_query,
                                                          const std::string & json_str,
                                                          std::string& req_dirty_values,
                                                          const int batch_size) {
    auto _filter_query = filter_query;
    StringUtils::trim(_filter_query);

    if (_filter_query.empty()) {
        nlohmann::json resp_summary;
        resp_summary["num_updated"] = 0;
        return Option(resp_summary);
    }

    const auto& dirty_values = parse_dirty_values_option(req_dirty_values);
    size_t docs_updated_count = 0;
    nlohmann::json update_document, dummy;

    try {
        update_document = nlohmann::json::parse(json_str);
    } catch(const std::exception& e) {
        LOG(ERROR) << "JSON error: " << e.what();
        return Option<nlohmann::json>(400, std::string("Bad JSON: ") + e.what());
    }

    std::vector<std::string> buffer;
    buffer.reserve(batch_size);

    if (_filter_query == "*") {
        // Get an iterator from rocksdb and iterate over all the documents present in the collection.
        std::string iter_upper_bound_key = get_seq_id_collection_prefix() + "`";
        auto iter_upper_bound = new rocksdb::Slice(iter_upper_bound_key);
        CollectionManager & collectionManager = CollectionManager::get_instance();
        const std::string seq_id_prefix = get_seq_id_collection_prefix();
        rocksdb::Iterator* it = collectionManager.get_store()->scan(seq_id_prefix, iter_upper_bound);

        while(it->Valid()) {
            // Generate a batch of documents to be ingested by add_many.
            for (int buffer_counter = 0; buffer_counter < batch_size && it->Valid();) {
                auto json_doc_str = it->value().ToString();
                it->Next();
                nlohmann::json existing_document;
                try {
                    existing_document = nlohmann::json::parse(json_doc_str);
                } catch(...) {
                    continue; // Don't add into buffer.
                }

                update_document["id"] = existing_document["id"].get<std::string>();
                buffer.push_back(update_document.dump());
                buffer_counter++;
            }

            auto res = add_many(buffer, dummy, index_operation_t::UPDATE, "", dirty_values);
            docs_updated_count += res["num_imported"].get<size_t>();
            buffer.clear();
        }

        delete iter_upper_bound;
        delete it;
    } else {
        filter_result_t filter_result;
        auto filter_ids_op = get_filter_ids(_filter_query, filter_result);
        if(!filter_ids_op.ok()) {
            return Option<nlohmann::json>(filter_ids_op.code(), filter_ids_op.error());
        }

        for (size_t i = 0; i < filter_result.count;) {
            for (int buffer_counter = 0; buffer_counter < batch_size && i < filter_result.count;) {
                uint32_t seq_id = filter_result.docs[i++];
                nlohmann::json existing_document;

                auto get_doc_op = get_document_from_store(get_seq_id_key(seq_id), existing_document);
                if (!get_doc_op.ok()) {
                    continue;
                }

                update_document["id"] = existing_document["id"].get<std::string>();
                buffer.push_back(update_document.dump());
                buffer_counter++;
            }

            auto res = add_many(buffer, dummy, index_operation_t::UPDATE, "", dirty_values);
            docs_updated_count += res["num_imported"].get<size_t>();
            buffer.clear();
        }
    }

    nlohmann::json resp_summary;
    resp_summary["num_updated"] = docs_updated_count;
    return Option(resp_summary);
}

bool Collection::is_exceeding_memory_threshold() const {
    return SystemMetrics::used_memory_ratio() > max_memory_ratio;
}

void Collection::batch_index(std::vector<index_record>& index_records, std::vector<std::string>& json_out,
                             size_t &num_indexed, const bool& return_doc, const bool& return_id, const size_t remote_embedding_batch_size,
                             const size_t remote_embedding_timeout_ms, const size_t remote_embedding_num_tries) {

    batch_index_in_memory(index_records, remote_embedding_batch_size, remote_embedding_timeout_ms, remote_embedding_num_tries, true);

    // store only documents that were indexed in-memory successfully
    for(auto& index_record: index_records) {
        nlohmann::json res;

        if(index_record.indexed.ok()) {
            if(index_record.is_update) {
                remove_flat_fields(index_record.new_doc);
                for(auto& field: fields) {
                    if(!field.store) {
                        index_record.new_doc.erase(field.name);
                    }
                }
                const std::string& serialized_json = index_record.new_doc.dump(-1, ' ', false, nlohmann::detail::error_handler_t::ignore);

                bool write_ok = store->insert(get_seq_id_key(index_record.seq_id), serialized_json);

                if(!write_ok) {
                    // we will attempt to reindex the old doc on a best-effort basis
                    LOG(ERROR) << "Update to disk failed. Will restore old document";
                    remove_document(index_record.new_doc, index_record.seq_id, false);
                    index_in_memory(index_record.old_doc, index_record.seq_id, index_record.operation, index_record.dirty_values);
                    index_record.index_failure(500, "Could not write to on-disk storage.");
                } else {
                    num_indexed++;
                    index_record.index_success();
                }

            } else {
                // remove flattened field values before storing on disk
                remove_flat_fields(index_record.doc);

                const std::string& seq_id_str = std::to_string(index_record.seq_id);
                const std::string& serialized_json = index_record.doc.dump(-1, ' ', false,
                                                                           nlohmann::detail::error_handler_t::ignore);

                rocksdb::WriteBatch batch;
                batch.Put(get_doc_id_key(index_record.doc["id"]), seq_id_str);
                batch.Put(get_seq_id_key(index_record.seq_id), serialized_json);
                bool write_ok = store->batch_write(batch);

                if(!write_ok) {
                    // remove from in-memory store to keep the state synced
                    LOG(ERROR) << "Write to disk failed. Will restore old document";
                    remove_document(index_record.doc, index_record.seq_id, false);
                    index_record.index_failure(500, "Could not write to on-disk storage.");
                } else {
                    num_indexed++;
                    index_record.index_success();
                }
            }
            res["success"] = index_record.indexed.ok();

            if (return_doc & index_record.indexed.ok()) {
                res["document"] = index_record.is_update ? index_record.new_doc : index_record.doc;
            }
            if (return_id & index_record.indexed.ok()) {
                res["id"] = index_record.is_update ? index_record.new_doc["id"] : index_record.doc["id"];
            }
            if(!index_record.indexed.ok()) {
                res["document"] = json_out[index_record.position];
                res["error"] = index_record.indexed.error();
                if (!index_record.embedding_res.empty()) {
                    res["embedding_error"] = nlohmann::json::object();
                    res["embedding_error"] = index_record.embedding_res;
                    res["error"] = index_record.embedding_res["error"];
                }
                res["code"] = index_record.indexed.code();
            }
        } else {
            res["success"] = false;
            res["document"] = json_out[index_record.position];
            res["error"] = index_record.indexed.error();
            if (!index_record.embedding_res.empty()) {
                res["embedding_error"] = nlohmann::json::object();
                res["error"] = index_record.embedding_res["error"];
                res["embedding_error"] = index_record.embedding_res;
            }
            res["code"] = index_record.indexed.code();
        }

        json_out[index_record.position] = res.dump(-1, ' ', false,
                                                   nlohmann::detail::error_handler_t::ignore);
    }
}

Option<uint32_t> Collection::index_in_memory(nlohmann::json &document, uint32_t seq_id,
                                             const index_operation_t op, const DIRTY_VALUES& dirty_values) {
    std::unique_lock lock(mutex);

    Option<uint32_t> validation_op = validator_t::validate_index_in_memory(document, seq_id, default_sorting_field,
                                                                     search_schema, embedding_fields, op, false,
                                                                     fallback_field_type, dirty_values);

    if(!validation_op.ok()) {
        return validation_op;
    }

    index_record rec(0, seq_id, document, op, dirty_values);

    std::vector<index_record> index_batch;
    index_batch.emplace_back(std::move(rec));
    Index::batch_memory_index(index, index_batch, default_sorting_field, search_schema, embedding_fields,
                              fallback_field_type, token_separators, symbols_to_index, true);

    num_documents += 1;
    return Option<>(200);
}

size_t Collection::batch_index_in_memory(std::vector<index_record>& index_records, const size_t remote_embedding_batch_size,
                                         const size_t remote_embedding_timeout_ms, const size_t remote_embedding_num_tries, const bool generate_embeddings) {
    std::unique_lock lock(mutex);
    size_t num_indexed = Index::batch_memory_index(index, index_records, default_sorting_field,
                                                   search_schema, embedding_fields, fallback_field_type,
                                                   token_separators, symbols_to_index, true, remote_embedding_batch_size, remote_embedding_timeout_ms, remote_embedding_num_tries, generate_embeddings);
    num_documents += num_indexed;
    return num_indexed;
}

bool Collection::does_override_match(const override_t& override, std::string& query,
                                     std::set<uint32_t>& excluded_set,
                                     string& actual_query, const string& filter_query,
                                     bool already_segmented,
                                     const bool tags_matched,
                                     const bool wildcard_tag_matched,
                                     const std::map<size_t, std::vector<std::string>>& pinned_hits,
                                     const std::vector<std::string>& hidden_hits,
                                     std::vector<std::pair<uint32_t, uint32_t>>& included_ids,
                                     std::vector<uint32_t>& excluded_ids,
                                     std::vector<const override_t*>& filter_overrides,
                                     bool& filter_curated_hits,
                                     std::string& curated_sort_by,
                                     nlohmann::json& override_metadata) const {

    if(!wildcard_tag_matched && !tags_matched && !override.rule.tags.empty()) {
        // only untagged overrides must be considered when no tags are given in the query
        return false;
    }

    auto now_epoch = int64_t(std::time(0));
    if(override.effective_from_ts != -1 && now_epoch < override.effective_from_ts) {
        return false;
    }

    if(override.effective_to_ts != -1 && now_epoch > override.effective_to_ts) {
        return false;
    }

    // ID-based overrides are applied first as they take precedence over filter-based overrides
    if(!override.filter_by.empty()) {
        filter_overrides.push_back(&override);
    }

    if((wildcard_tag_matched || tags_matched) && override.rule.query.empty() && override.rule.filter_by.empty()) {
        // allowed
    } else {
        bool filter_by_match = (override.rule.query.empty() && override.rule.match.empty() &&
                                !override.rule.filter_by.empty() && override.rule.filter_by == filter_query);

        bool query_match = (override.rule.match == override_t::MATCH_EXACT && override.rule.normalized_query == query) ||
                           (override.rule.match == override_t::MATCH_CONTAINS &&
                            StringUtils::contains_word(query, override.rule.normalized_query));

        if(!filter_by_match && !query_match) {
            return false;
        }

        if(!override.rule.filter_by.empty() && override.rule.filter_by != filter_query) {
            return false;
        }
    }

    // have to ensure that dropped hits take precedence over added hits
    for(const auto & hit: override.drop_hits) {
        Option<uint32_t> seq_id_op = doc_id_to_seq_id(hit.doc_id);
        if(seq_id_op.ok()) {
            excluded_ids.push_back(seq_id_op.get());
            excluded_set.insert(seq_id_op.get());
        }
    }

    for(const auto & hit: override.add_hits) {
        Option<uint32_t> seq_id_op = doc_id_to_seq_id(hit.doc_id);
        if(!seq_id_op.ok()) {
            continue;
        }
        uint32_t seq_id = seq_id_op.get();
        bool excluded = (excluded_set.count(seq_id) != 0);
        if(!excluded) {
            included_ids.emplace_back(seq_id, hit.position);
        }
    }

    if(!override.replace_query.empty()) {
        actual_query = override.replace_query;
    } else if(override.remove_matched_tokens && override.filter_by.empty()) {
        // don't prematurely remove tokens from query because dynamic filtering will require them
        StringUtils::replace_all(query, override.rule.normalized_query, "");
        StringUtils::trim(query);
        if(query.empty()) {
            query = "*";
        }

        actual_query = query;
    }

    filter_curated_hits = override.filter_curated_hits;
    curated_sort_by = override.sort_by;
    if(override_metadata.empty()) {
        override_metadata = override.metadata;
    }
    return true;
}

void Collection::curate_results(string& actual_query, const string& filter_query,
                                bool enable_overrides, bool already_segmented,
                                const std::set<std::string>& tags,
                                const std::map<size_t, std::vector<std::string>>& pinned_hits,
                                const std::vector<std::string>& hidden_hits,
                                std::vector<std::pair<uint32_t, uint32_t>>& included_ids,
                                std::vector<uint32_t>& excluded_ids,
                                std::vector<const override_t*>& filter_overrides,
                                bool& filter_curated_hits,
                                std::string& curated_sort_by,
                                nlohmann::json& override_metadata) const {

    std::set<uint32_t> excluded_set;

    // If pinned or hidden hits are provided, they take precedence over overrides

    // have to ensure that hidden hits take precedence over included hits
    if(!hidden_hits.empty()) {
        for(const auto & hit: hidden_hits) {
            Option<uint32_t> seq_id_op = doc_id_to_seq_id(hit);
            if(seq_id_op.ok()) {
                excluded_ids.push_back(seq_id_op.get());
                excluded_set.insert(seq_id_op.get());
            }
        }
    }

    if(enable_overrides && !overrides.empty()) {
        std::string query;

        if(actual_query == "*") {
            query = "*";
        } else {
            std::vector<std::string> tokens;
            Tokenizer tokenizer(actual_query, true, false, "", symbols_to_index, token_separators);
            tokenizer.tokenize(tokens);
            query = StringUtils::join(tokens, " ");
        }

        if(!tags.empty()) {
            bool all_tags_found = false;
            std::set<std::string> found_overrides;
            if(tags.size() > 1) {
                // check for AND match only when multiple tags are sent
                const auto& tag = *tags.begin();
                auto override_ids_it = override_tags.find(tag);
                if (override_ids_it != override_tags.end()) {
                    const auto &override_ids = override_ids_it->second;
                    for(const auto& id: override_ids) {
                        auto override_it = overrides.find(id);
                        if(override_it == overrides.end()) {
                            continue;
                        }

                        const auto& override = override_it->second;

                        if(override.rule.tags == tags) {
                            bool match_found = does_override_match(override, query, excluded_set, actual_query,
                                                                   filter_query, already_segmented, true, false,
                                                                   pinned_hits, hidden_hits, included_ids,
                                                                   excluded_ids, filter_overrides, filter_curated_hits,
                                                                   curated_sort_by, override_metadata);

                            if(match_found) {
                                all_tags_found = true;
                                found_overrides.insert(id);
                                if(override.stop_processing) {
                                    break;
                                }
                            }
                        }
                    }
                }
            }

            if(!all_tags_found) {
                // check for partial tag matches
                for(const auto& tag: tags) {
                    auto override_ids_it = override_tags.find(tag);
                    if (override_ids_it == override_tags.end()) {
                        continue;
                    }

                    const auto &override_ids = override_ids_it->second;

                    for(const auto& id: override_ids) {
                        if(found_overrides.count(id) != 0) {
                            continue;
                        }
                        auto override_it = overrides.find(id);
                        if (override_it == overrides.end()) {
                            continue;
                        }

                        const auto& override = override_it->second;
                        std::set<std::string> matching_tags;
                        std::set_intersection(override.rule.tags.begin(), override.rule.tags.end(),
                                              tags.begin(), tags.end(),
                                              std::inserter(matching_tags, matching_tags.begin()));

                        if(matching_tags.empty()) {
                            continue;
                        }

                        bool match_found = does_override_match(override, query, excluded_set, actual_query,
                                                               filter_query, already_segmented, true, false,
                                                               pinned_hits, hidden_hits, included_ids,
                                                               excluded_ids, filter_overrides, filter_curated_hits,
                                                               curated_sort_by, override_metadata);

                        if(match_found) {
                            found_overrides.insert(id);
                            if(override.stop_processing) {
                                break;
                            }
                        }
                    }
                }
            }
        } else {
            // no override tags given
            for(const auto& override_kv: overrides) {
                const auto& override = override_kv.second;
                bool wildcard_tag = override.rule.tags.size() == 1 && *override.rule.tags.begin() == "*";
                bool match_found = does_override_match(override, query, excluded_set, actual_query, filter_query,
                                                       already_segmented, false, wildcard_tag,
                                                       pinned_hits, hidden_hits, included_ids,
                                                       excluded_ids, filter_overrides, filter_curated_hits,
                                                       curated_sort_by, override_metadata);
                if(match_found && override.stop_processing) {
                    break;
                }
            }
        }
    }

    if(!pinned_hits.empty()) {
        for(const auto& pos_ids: pinned_hits) {
            size_t pos = pos_ids.first;
            for(const std::string& id: pos_ids.second) {
                Option<uint32_t> seq_id_op = doc_id_to_seq_id(id);
                if(!seq_id_op.ok()) {
                    continue;
                }
                uint32_t seq_id = seq_id_op.get();
                bool excluded = (excluded_set.count(seq_id) != 0);
                if(!excluded) {
                    included_ids.emplace_back(seq_id, pos);
                }
            }
        }
    }
}

Option<bool> Collection::validate_and_standardize_sort_fields_with_lock(const std::vector<sort_by> & sort_fields,
                                                                        std::vector<sort_by>& sort_fields_std,
                                                                        const bool is_wildcard_query,
                                                                        const bool is_vector_query,
                                                                        const std::string& query, const bool is_group_by_query,
                                                                        const size_t remote_embedding_timeout_ms,
                                                                        const size_t remote_embedding_num_tries) const {
    std::shared_lock lock(mutex);
    return validate_and_standardize_sort_fields(sort_fields, sort_fields_std, is_wildcard_query, is_vector_query,
                                                query, is_group_by_query, remote_embedding_timeout_ms, true);
}

Option<bool> Collection::validate_and_standardize_sort_fields(const std::vector<sort_by> & sort_fields,
                                                              std::vector<sort_by>& sort_fields_std,
                                                              const bool is_wildcard_query,
                                                              const bool is_vector_query,
                                                              const std::string& query, const bool is_group_by_query, 
                                                              const size_t remote_embedding_timeout_ms,
                                                              const size_t remote_embedding_num_tries,
                                                              const bool is_reference_sort) const {

    uint32_t eval_sort_count = 0;
    size_t num_sort_expressions = 0;

    for(size_t i = 0; i < sort_fields.size(); i++) {
        const sort_by& _sort_field = sort_fields[i];

        if (_sort_field.name[0] == '$') {
            // Reference sort_by
            auto parenthesis_index = _sort_field.name.find('(');
            std::string ref_collection_name = _sort_field.name.substr(1, parenthesis_index - 1);
            auto& cm = CollectionManager::get_instance();
            auto ref_collection = cm.get_collection(ref_collection_name);
            if (ref_collection == nullptr) {
                return Option<bool>(400, "Referenced collection `" + ref_collection_name + "` in `sort_by` not found.");
            }

            auto sort_by_str = _sort_field.name.substr(parenthesis_index + 1,
                                                       _sort_field.name.size() - parenthesis_index - 2);
            std::vector<sort_by> ref_sort_fields;
            bool parsed_sort_by = CollectionManager::parse_sort_by_str(sort_by_str, ref_sort_fields);
            if (!parsed_sort_by) {
                return Option<bool>(400, "Reference `sort_by` is malformed.");
            }

            std::vector<sort_by> ref_sort_fields_std;
            auto sort_validation_op = ref_collection->validate_and_standardize_sort_fields_with_lock(ref_sort_fields,
                                                                                                     ref_sort_fields_std,
                                                                                                     is_wildcard_query,
                                                                                                     is_vector_query,
                                                                                                     query,
                                                                                                     is_group_by_query,
                                                                                                     remote_embedding_timeout_ms,
                                                                                                     remote_embedding_num_tries);
            if (!sort_validation_op.ok()) {
                return Option<bool>(sort_validation_op.code(), "Referenced collection `" + ref_collection_name + "`: " +
                                                                sort_validation_op.error());
            }

            for (auto& ref_sort_field_std: ref_sort_fields_std) {
                ref_sort_field_std.reference_collection_name = ref_collection_name;
                sort_fields_std.emplace_back(ref_sort_field_std);
            }

            continue;
        } else if (_sort_field.name == sort_field_const::eval) {
            sort_by sort_field_std(sort_field_const::eval, _sort_field.order);

            auto const& count = _sort_field.eval_expressions.size();
            sort_field_std.eval.filter_trees = new filter_node_t[count];
            std::unique_ptr<filter_node_t []> filter_trees_guard(sort_field_std.eval.filter_trees);

            for (uint32_t j = 0; j < count; j++) {
                auto const& filter_exp = _sort_field.eval_expressions[j];
                if (filter_exp.empty()) {
                    return Option<bool>(400, "The eval expression in sort_by is empty.");
                }

                filter_node_t* filter_tree_root = nullptr;
                Option<bool> parse_filter_op = filter::parse_filter_query(filter_exp, search_schema,
                                                                          store, "", filter_tree_root);
                std::unique_ptr<filter_node_t> filter_tree_root_guard(filter_tree_root);

                if (!parse_filter_op.ok()) {
                    return Option<bool>(parse_filter_op.code(), "Error parsing eval expression in sort_by clause.");
                }

                sort_field_std.eval.filter_trees[j] = std::move(*filter_tree_root);
            }

            eval_sort_count++;
            sort_field_std.eval_expressions = _sort_field.eval_expressions;
            sort_field_std.eval.scores = _sort_field.eval.scores;
            sort_fields_std.emplace_back(sort_field_std);
            filter_trees_guard.release();
            continue;
        }

        sort_by sort_field_std(_sort_field.name, _sort_field.order);

        if(sort_field_std.name.back() == ')') {
            // check if this is a geo field or text match field
            size_t paran_start = 0;
            while(paran_start < sort_field_std.name.size() && sort_field_std.name[paran_start] != '(') {
                paran_start++;
            }

            const std::string& actual_field_name = sort_field_std.name.substr(0, paran_start);
            const auto field_it = search_schema.find(actual_field_name);

            if(actual_field_name == sort_field_const::text_match) {
                std::vector<std::string> match_parts;
                const std::string& match_config = sort_field_std.name.substr(paran_start+1, sort_field_std.name.size() - paran_start - 2);
                StringUtils::split(match_config, match_parts, ":");
                if(match_parts.size() != 2 || match_parts[0] != "buckets") {
                    return Option<bool>(400, "Invalid sorting parameter passed for _text_match.");
                }

                if(!StringUtils::is_uint32_t(match_parts[1])) {
                    return Option<bool>(400, "Invalid value passed for _text_match `buckets` configuration.");
                }

                sort_field_std.name = actual_field_name;
                sort_field_std.text_match_buckets = std::stoll(match_parts[1]);

            } else if(actual_field_name == sort_field_const::eval) {
                const std::string& filter_exp = sort_field_std.name.substr(paran_start + 1,
                                                                           sort_field_std.name.size() - paran_start -
                                                                           2);
                if(filter_exp.empty()) {
                    return Option<bool>(400, "The eval expression in sort_by is empty.");
                }

                Option<bool> parse_filter_op = filter::parse_filter_query(filter_exp, search_schema,
                                                                          store, "", sort_field_std.eval.filter_trees);
                if(!parse_filter_op.ok()) {
                    return Option<bool>(parse_filter_op.code(), "Error parsing eval expression in sort_by clause.");
                }

                sort_field_std.name = actual_field_name;
                num_sort_expressions++;
            } else if(actual_field_name == sort_field_const::vector_query) {
                const std::string& vector_query_str = sort_field_std.name.substr(paran_start + 1,
                                                                              sort_field_std.name.size() - paran_start -
                                                                              2);
                if(vector_query_str.empty()) {
                    return Option<bool>(400, "The vector query in sort_by is empty.");
                }


                auto parse_vector_op = VectorQueryOps::parse_vector_query_str(vector_query_str, sort_field_std.vector_query.query,
                                                                            is_wildcard_query, this, true);
                if(!parse_vector_op.ok()) {
                    return Option<bool>(400, parse_vector_op.error());
                }

                auto vector_field_it = search_schema.find(sort_field_std.vector_query.query.field_name);
                if(vector_field_it == search_schema.end() || vector_field_it.value().num_dim == 0) {
                    return Option<bool>(400, "Could not find a field named `" + sort_field_std.vector_query.query.field_name + "` in vector index.");
                }

                if(!sort_field_std.vector_query.query.queries.empty()) {
                    if(embedding_fields.find(sort_field_std.vector_query.query.field_name) == embedding_fields.end()) {
                        return Option<bool>(400, "`queries` parameter is only supported for auto-embedding fields.");
                    }

                    std::vector<std::vector<float>> embeddings;
                    for(const auto& q: sort_field_std.vector_query.query.queries) {
                        EmbedderManager& embedder_manager = EmbedderManager::get_instance();
                        auto embedder_op = embedder_manager.get_text_embedder(vector_field_it.value().embed[fields::model_config]);
                        if(!embedder_op.ok()) {
                            return Option<bool>(400, embedder_op.error());
                        }

                        auto remote_embedding_timeout_us = remote_embedding_timeout_ms * 1000;
                        if((std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::system_clock::now().time_since_epoch()).count() - search_begin_us) > remote_embedding_timeout_us) {
                            std::string error = "Request timed out.";
                            return Option<bool>(500, error);
                        }

                        auto embedder = embedder_op.get();

                        if(embedder->is_remote()) {
                            if(remote_embedding_num_tries == 0) {
                                std::string error = "`remote_embedding_num_tries` must be greater than 0.";
                                return Option<bool>(400, error);
                            }
                        }

                        std::string embed_query = embedder_manager.get_query_prefix(vector_field_it.value().embed[fields::model_config]) + q;
                        auto embedding_op = embedder->Embed(embed_query, remote_embedding_timeout_ms, remote_embedding_num_tries);

                        if(!embedding_op.success) {
                            if(!embedding_op.error["error"].get<std::string>().empty()) {
                                return Option<bool>(400, embedding_op.error["error"].get<std::string>());
                            } else {
                                return Option<bool>(400, embedding_op.error.dump());
                            }
                        }

                        embeddings.emplace_back(embedding_op.embedding);
                    }

                    if(sort_field_std.vector_query.query.query_weights.empty()) {
                        // get average of all embeddings
                        std::vector<float> avg_embedding(vector_field_it.value().num_dim, 0);
                        for(const auto& embedding: embeddings) {
                            for(size_t i = 0; i < embedding.size(); i++) {
                                avg_embedding[i] += embedding[i];
                            }
                        }
                        for(size_t i = 0; i < avg_embedding.size(); i++) {
                            avg_embedding[i] /= embeddings.size();
                        }

                        sort_field_std.vector_query.query.values = avg_embedding;
                    } else {
                        std::vector<float> weighted_embeddings(vector_field_it.value().num_dim, 0);
                        for(size_t i = 0; i < embeddings.size(); i++) {
                            for(size_t j = 0; j < embeddings[i].size(); j++) {
                                weighted_embeddings[j] += embeddings[i][j] * sort_field_std.vector_query.query.query_weights[i];
                            }
                        }

                        sort_field_std.vector_query.query.values = weighted_embeddings;
                    }
                }
                
                if(sort_field_std.vector_query.query.values.empty() && embedding_fields.find(sort_field_std.vector_query.query.field_name) != embedding_fields.end()) {
                    // generate embeddings for the query

                    EmbedderManager& embedder_manager = EmbedderManager::get_instance();
                    auto embedder_op = embedder_manager.get_text_embedder(vector_field_it.value().embed[fields::model_config]);
                    if(!embedder_op.ok()) {
                        return Option<bool>(embedder_op.code(), embedder_op.error());
                    }

                    auto embedder = embedder_op.get();

                    if(embedder->is_remote() && remote_embedding_num_tries == 0) {
                        std::string error = "`remote_embedding_num_tries` must be greater than 0.";
                        return Option<bool>(400, error);
                    }

                    std::string embed_query = embedder_manager.get_query_prefix(vector_field_it.value().embed[fields::model_config]) + query;
                    auto embedding_op = embedder->Embed(embed_query, remote_embedding_timeout_ms, remote_embedding_num_tries);

                    if(!embedding_op.success) {
                        if(!embedding_op.error["error"].get<std::string>().empty()) {
                            return Option<bool>(400, embedding_op.error["error"].get<std::string>());
                        } else {
                            return Option<bool>(400, embedding_op.error.dump());
                        }
                    }

                    sort_field_std.vector_query.query.values = embedding_op.embedding;
                }

                const auto& vector_index_map = index->_get_vector_index();
                if(vector_index_map.find(sort_field_std.vector_query.query.field_name) == vector_index_map.end()) {
                    return Option<bool>(400, "Field `" + sort_field_std.vector_query.query.field_name + "` does not have a vector index.");
                }


                if(vector_field_it.value().num_dim != sort_field_std.vector_query.query.values.size()) {
                    return Option<bool>(400, "Query field `" + sort_field_std.vector_query.query.field_name + "` must have " +
                                                    std::to_string(vector_field_it.value().num_dim) + " dimensions.");
                }

                sort_field_std.vector_query.vector_index = vector_index_map.at(sort_field_std.vector_query.query.field_name);

                if(sort_field_std.vector_query.vector_index->distance_type == cosine) {
                    std::vector<float> normalized_values(sort_field_std.vector_query.query.values.size());
                    hnsw_index_t::normalize_vector(sort_field_std.vector_query.query.values, normalized_values);
                    sort_field_std.vector_query.query.values = normalized_values;
                }

                sort_field_std.name = actual_field_name;

            } else {
                if(field_it == search_schema.end()) {
                    std::string error = "Could not find a field named `" + actual_field_name + "` in the schema for sorting.";
                    return Option<bool>(404, error);
                }

                std::string error = "Bad syntax for sorting field `" + actual_field_name + "`";

                if(!field_it.value().is_geopoint()) {
                    // check for null value order
                    const std::string& sort_params_str = sort_field_std.name.substr(paran_start + 1,
                                                                                    sort_field_std.name.size() -
                                                                                    paran_start - 2);

                    std::vector<std::string> param_parts;
                    StringUtils::split(sort_params_str, param_parts, ":");

                    if(param_parts.size() != 2) {
                        return Option<bool>(400, error);
                    }

                    if(param_parts[0] != sort_field_const::missing_values) {
                        return Option<bool>(400, error);
                    }

                    auto missing_values_op = magic_enum::enum_cast<sort_by::missing_values_t>(param_parts[1]);
                    if(missing_values_op.has_value()) {
                        sort_field_std.missing_values = missing_values_op.value();
                    } else {
                        return Option<bool>(400, error);
                    }
                }

                else {
                    const std::string& geo_coordstr = sort_field_std.name.substr(paran_start+1, sort_field_std.name.size() - paran_start - 2);

                    // e.g. geopoint_field(lat1, lng1, exclude_radius: 10 miles)

                    std::vector<std::string> geo_parts;
                    StringUtils::split(geo_coordstr, geo_parts, ",");

                    if(geo_parts.size() != 2 && geo_parts.size() != 3) {
                        return Option<bool>(400, error);
                    }

                    if(!StringUtils::is_float(geo_parts[0]) || !StringUtils::is_float(geo_parts[1])) {
                        return Option<bool>(400, error);
                    }

                    if(geo_parts.size() == 3) {
                        // try to parse the exclude radius option
                        bool is_exclude_option = false;

                        if(StringUtils::begins_with(geo_parts[2], sort_field_const::exclude_radius)) {
                            is_exclude_option = true;
                        } else if(StringUtils::begins_with(geo_parts[2], sort_field_const::precision)) {
                            is_exclude_option = false;
                        } else {
                            return Option<bool>(400, error);
                        }

                        std::vector<std::string> param_parts;
                        StringUtils::split(geo_parts[2], param_parts, ":");

                        if(param_parts.size() != 2) {
                            return Option<bool>(400, error);
                        }

                        // param_parts[1] is the value, in either "20km" or "20 km" format

                        if(param_parts[1].size() < 2) {
                            return Option<bool>(400, error);
                        }

                        std::string unit = param_parts[1].substr(param_parts[1].size()-2, 2);

                        if(unit != "km" && unit != "mi") {
                            return Option<bool>(400, "Sort field's parameter unit must be either `km` or `mi`.");
                        }

                        std::vector<std::string> dist_values;
                        StringUtils::split(param_parts[1], dist_values, unit);

                        if(dist_values.size() != 1) {
                            return Option<bool>(400, error);
                        }

                        if(!StringUtils::is_float(dist_values[0])) {
                            return Option<bool>(400, error);
                        }

                        int32_t value_meters;

                        if(unit == "km") {
                            value_meters = std::stof(dist_values[0]) * 1000;
                        } else if(unit == "mi") {
                            value_meters = std::stof(dist_values[0]) * 1609.34;
                        } else {
                            return Option<bool>(400, "Sort field's parameter "
                                                     "unit must be either `km` or `mi`.");
                        }

                        if(value_meters <= 0) {
                            return Option<bool>(400, "Sort field's parameter must be a positive number.");
                        }

                        if(is_exclude_option) {
                            sort_field_std.exclude_radius = value_meters;
                        } else {
                            sort_field_std.geo_precision = value_meters;
                        }
                    }

                    double lat = std::stod(geo_parts[0]);
                    double lng = std::stod(geo_parts[1]);
                    int64_t lat_lng = GeoPoint::pack_lat_lng(lat, lng);
                    sort_field_std.geopoint = lat_lng;
                }

                sort_field_std.name = actual_field_name;
            }
        }

        if (sort_field_std.name != sort_field_const::text_match && sort_field_std.name != sort_field_const::eval &&
            sort_field_std.name != sort_field_const::seq_id && sort_field_std.name != sort_field_const::group_found && sort_field_std.name != sort_field_const::vector_distance &&
            sort_field_std.name != sort_field_const::vector_query) {
            const auto field_it = search_schema.find(sort_field_std.name);
            if(field_it == search_schema.end() || !field_it.value().sort || !field_it.value().index) {
                std::string error = "Could not find a field named `" + sort_field_std.name +
                                    "` in the schema for sorting.";
                return Option<bool>(404, error);
            }
        }

        if(sort_field_std.name == sort_field_const::group_found && is_group_by_query == false) {
            std::string error = "group_by parameters should not be empty when using sort_by group_found";
            return Option<bool>(404, error);
        }

        if(sort_field_std.name == sort_field_const::vector_distance && !is_vector_query) {
            std::string error = "sort_by vector_distance is only supported for vector queries, semantic search and hybrid search.";
            return Option<bool>(404, error);
        }

        StringUtils::toupper(sort_field_std.order);

        if(sort_field_std.order != sort_field_const::asc && sort_field_std.order != sort_field_const::desc) {
            std::string error = "Order for field` " + sort_field_std.name + "` should be either ASC or DESC.";
            return Option<bool>(400, error);
        }

        sort_fields_std.emplace_back(sort_field_std);
    }

    if (is_reference_sort) {
        if (eval_sort_count > 1) {
            std::string message = "Only one sorting eval expression is allowed.";
            return Option<bool>(422, message);
        }
        return Option<bool>(true);
    }

    /*
      1. Empty: [match_score, dsf] upstream
      2. ONE  : [usf, match_score]
      3. TWO  : [usf1, usf2, match_score]
      4. THREE: do nothing
    */
    if(sort_fields_std.empty()) {
        if(!is_wildcard_query) {
            sort_fields_std.emplace_back(sort_field_const::text_match, sort_field_const::desc);
        }

        if(is_vector_query) {
            sort_fields_std.emplace_back(sort_field_const::vector_distance, sort_field_const::asc);
        }

        if(!default_sorting_field.empty()) {
            sort_fields_std.emplace_back(default_sorting_field, sort_field_const::desc);
        } else {
            sort_fields_std.emplace_back(sort_field_const::seq_id, sort_field_const::desc);
        }
    }

    bool found_match_score = false;
    bool found_vector_distance = false;
    for(const auto & sort_field : sort_fields_std) {
        if(sort_field.name == sort_field_const::text_match) {
            found_match_score = true;
        }
        if(sort_field.name == sort_field_const::vector_distance) {
            found_vector_distance = true;
        }
        if(found_match_score && found_vector_distance) {
            break;
        }
    }

    if(!found_match_score && !is_wildcard_query && sort_fields.size() < 3) {
        sort_fields_std.emplace_back(sort_field_const::text_match, sort_field_const::desc);
    }

    if(!found_vector_distance && is_vector_query && sort_fields.size() < 3) {
        sort_fields_std.emplace_back(sort_field_const::vector_distance, sort_field_const::asc);
    }

    if(sort_fields_std.size() > 3) {
        std::string message = "Only upto 3 sort_by fields can be specified.";
        return Option<bool>(422, message);
    }

    if(eval_sort_count > 1) {
        std::string message = "Only one sorting eval expression is allowed.";
        return Option<bool>(422, message);
    }

    return Option<bool>(true);
}

Option<bool> Collection::extract_field_name(const std::string& field_name,
                                            const tsl::htrie_map<char, field>& search_schema,
                                            std::vector<std::string>& processed_search_fields,
                                            const bool extract_only_string_fields,
                                            const bool enable_nested_fields,
                                            const bool handle_wildcard,
                                            const bool& include_id) {
    // Reference to other collection
    if (field_name[0] == '$') {
        processed_search_fields.push_back(field_name);
        return Option<bool>(true);
    }

    if(field_name == "id") {
        processed_search_fields.push_back(field_name);
        return Option<bool>(true);
    }

    bool is_wildcard = field_name.find('*') != std::string::npos;
    if (is_wildcard && !handle_wildcard) {
        return Option<bool>(400, "Pattern `" + field_name + "` is not allowed.");
    }

    if (is_wildcard && include_id && field_name.size() < 4 &&
        (field_name == "*" || field_name == "i*" || field_name == "id*")) {
        processed_search_fields.emplace_back("id");
    }

    // If wildcard, remove *
    auto prefix_it = search_schema.equal_prefix_range(field_name.substr(0, field_name.size() - is_wildcard));
    bool field_found = false;

    for(auto kv = prefix_it.first; kv != prefix_it.second; ++kv) {
        bool exact_key_match = (kv.key().size() == field_name.size());
        bool exact_primitive_match = exact_key_match && !kv.value().is_object();
        bool text_embedding = kv.value().type == field_types::FLOAT_ARRAY && kv.value().num_dim > 0;

        if(extract_only_string_fields && !kv.value().is_string() && !text_embedding) {
            if(exact_primitive_match && !is_wildcard) {
                // upstream needs to be returned an error
                return Option<bool>(400, "Field `" + field_name + "` should be a string or a string array.");
            }

            continue;
        }

        if(!exact_key_match && text_embedding) {
            continue;
        }

        if (exact_primitive_match || is_wildcard || text_embedding ||
            // field_name prefix must be followed by a "." to indicate an object search
            (enable_nested_fields && kv.key().size() > field_name.size() && kv.key()[field_name.size()] == '.')) {
            processed_search_fields.push_back(kv.key());
            field_found = true;
        }
    }

    if (is_wildcard && extract_only_string_fields && !field_found) {
        std::string error = "No string or string array field found matching the pattern `" + field_name + "` in the schema.";
        return Option<bool>(404, error);
    } else if (!field_found) {
        std::string error = is_wildcard ? "No field found matching the pattern `" : "Could not find a field named `" +
                                                                                    field_name + "` in the schema.";
        return Option<bool>(404, error);
    }

    return Option<bool>(true);
}

Option<nlohmann::json> Collection::search(std::string raw_query,
                                  const std::vector<std::string>& raw_search_fields,
                                  const std::string & filter_query, const std::vector<std::string>& facet_fields,
                                  const std::vector<sort_by> & sort_fields, const std::vector<uint32_t>& num_typos,
                                  size_t per_page, const size_t page,
                                  token_ordering token_order, const std::vector<bool>& prefixes,
                                  const size_t drop_tokens_threshold,
                                  const spp::sparse_hash_set<std::string> & include_fields,
                                  const spp::sparse_hash_set<std::string> & exclude_fields,
                                  const size_t max_facet_values,
                                  const std::string & simple_facet_query,
                                  const size_t snippet_threshold,
                                  const size_t highlight_affix_num_tokens,
                                  const std::string& highlight_full_fields,
                                  size_t typo_tokens_threshold,
                                  const std::string& pinned_hits_str,
                                  const std::string& hidden_hits_str,
                                  const std::vector<std::string>& raw_group_by_fields,
                                  size_t group_limit,
                                  const std::string& highlight_start_tag,
                                  const std::string& highlight_end_tag,
                                  std::vector<uint32_t> raw_query_by_weights,
                                  size_t limit_hits,
                                  bool prioritize_exact_match,
                                  bool pre_segmented_query,
                                  bool enable_overrides,
                                  const std::string& highlight_fields,
                                  const bool exhaustive_search,
                                  const size_t search_stop_millis,
                                  const size_t min_len_1typo,
                                  const size_t min_len_2typo,
                                  enable_t split_join_tokens,
                                  const size_t max_candidates,
                                  const std::vector<enable_t>& infixes,
                                  const size_t max_extra_prefix,
                                  const size_t max_extra_suffix,
                                  const size_t facet_query_num_typos,
                                  const size_t filter_curated_hits_option,
                                  const bool prioritize_token_position,
                                  const std::string& vector_query_str,
                                  const bool enable_highlight_v1,
                                  const uint64_t search_time_start_us,
                                  const text_match_type_t match_type,
                                  const size_t facet_sample_percent,
                                  const size_t facet_sample_threshold,
                                  const size_t page_offset,
                                  facet_index_type_t facet_index_type,
                                  const size_t remote_embedding_timeout_ms,
                                  const size_t remote_embedding_num_tries,
                                  const std::string& stopwords_set,
                                  const std::vector<std::string>& facet_return_parent,
                                  const std::vector<ref_include_exclude_fields>& ref_include_exclude_fields_vec,
                                  const std::string& drop_tokens_mode,
                                  const bool prioritize_num_matching_fields,
                                  const bool group_missing_values,
                                  const bool conversation,
                                  const std::string& conversation_model_id,
                                  std::string conversation_id,
                                  const std::string& override_tags_str,
                                  const std::string& voice_query) const {
    std::shared_lock lock(mutex);

    // setup thread local vars
    search_stop_us = search_stop_millis * 1000;
    search_begin_us = (search_time_start_us != 0) ? search_time_start_us :
                      std::chrono::duration_cast<std::chrono::microseconds>(
                           std::chrono::system_clock::now().time_since_epoch()).count();
    search_cutoff = false;

    if(raw_query != "*" && raw_search_fields.empty()) {
        return Option<nlohmann::json>(400, "No search fields specified for the query.");
    }

    if(!raw_search_fields.empty() && !raw_query_by_weights.empty() &&
        raw_search_fields.size() != raw_query_by_weights.size()) {
        return Option<nlohmann::json>(400, "Number of weights in `query_by_weights` does not match "
                                           "number of `query_by` fields.");
    }

    if(!raw_group_by_fields.empty() && (group_limit == 0 || group_limit > GROUP_LIMIT_MAX)) {
        return Option<nlohmann::json>(400, "Value of `group_limit` must be between 1 and " +
                                      std::to_string(GROUP_LIMIT_MAX) + ".");
    }

    if(!raw_search_fields.empty() && raw_search_fields.size() != num_typos.size()) {
        if(num_typos.size() != 1) {
            return Option<nlohmann::json>(400, "Number of values in `num_typos` does not match "
                                               "number of `query_by` fields.");
        }
    }

    if(!raw_search_fields.empty() && raw_search_fields.size() != prefixes.size()) {
        if(prefixes.size() != 1) {
            return Option<nlohmann::json>(400, "Number of prefix values in `prefix` does not match "
                                               "number of `query_by` fields.");
        }
    }

    if(!raw_search_fields.empty() && raw_search_fields.size() != infixes.size()) {
        if(infixes.size() != 1) {
            return Option<nlohmann::json>(400, "Number of infix values in `infix` does not match "
                                               "number of `query_by` fields.");
        }
    }

    if(facet_sample_percent > 100) {
        return Option<nlohmann::json>(400, "Value of `facet_sample_percent` must be less than 100.");
    }

    if(raw_group_by_fields.empty()) {
        group_limit = 0;
    }

    vector_query_t vector_query;
    if(!vector_query_str.empty()) {
        bool is_wildcard_query = (raw_query == "*" || raw_query.empty());

        auto parse_vector_op = parse_and_validate_vector_query(vector_query_str, vector_query, is_wildcard_query, remote_embedding_timeout_ms, remote_embedding_num_tries, per_page);

        if(!parse_vector_op.ok()) {
            return Option<nlohmann::json>(parse_vector_op.code(), parse_vector_op.error());
        }
    }

    // validate search fields
    std::vector<search_field_t> processed_search_fields;
    std::vector<uint32_t> query_by_weights;

    size_t num_embed_fields = 0;
    std::string query = raw_query;
    std::string transcribed_query;
    if(!voice_query.empty()) {
        if(!vq_model) {
            return Option<nlohmann::json>(400, "Voice query is not enabled. Please set `voice_query_model` for this collection.");
        }

        auto transcribe_res = vq_model->transcribe(voice_query);
        if(!transcribe_res.ok()) {
            return Option<nlohmann::json>(transcribe_res.code(), transcribe_res.error());
        }
        query = transcribe_res.get();
        transcribed_query = query;
    }

    if(conversation) {
        if(conversation_model_id.empty()) {
            return Option<nlohmann::json>(400, "Conversation is enabled but no conversation model ID is provided.");
        }

        auto conversation_model_op = ConversationModelManager::get_model(conversation_model_id);

        if(!conversation_model_op.ok()) {
            return Option<nlohmann::json>(400, conversation_model_op.error());
        }
    }

    if(!conversation_id.empty()) {
        if(!conversation) {
            return Option<nlohmann::json>(400, "Conversation ID provided but conversation is not enabled for this collection.");
        }

        auto conversation_history_op = ConversationManager::get_instance().get_conversation(conversation_id);
        if(!conversation_history_op.ok()) {
            return Option<nlohmann::json>(400, conversation_history_op.error());
        }

        auto conversation_history = conversation_history_op.get();

        auto truncate_conversation_history = ConversationManager::get_instance().truncate_conversation(conversation_history_op.get()["conversation"]);

        conversation_history["conversation"] = truncate_conversation_history.get();
       

        if(!truncate_conversation_history.ok()) {
            return Option<nlohmann::json>(400, truncate_conversation_history.error());
        }

        auto conversation_model_op = ConversationModelManager::get_model(conversation_model_id);

        auto standalone_question_op = ConversationModel::get_standalone_question(conversation_history, raw_query, conversation_model_op.get());
        if(!standalone_question_op.ok()) {
            return Option<nlohmann::json>(400, standalone_question_op.error());
        }
        query = standalone_question_op.get();
    }

    for(size_t i = 0; i < raw_search_fields.size(); i++) {
        const std::string& field_name = raw_search_fields[i];
        if(field_name == "id") {
            // `id` field needs to be handled separately, we will not handle for now
            std::string error = "Cannot use `id` as a query by field.";
            return Option<nlohmann::json>(400, error);
        } else if (field_name[0] == '$' && field_name.find('(') != std::string::npos &&
                    field_name.find(')') != std::string::npos) {
            return Option<nlohmann::json>(400, "Query by reference is not yet supported.");
        }

        std::vector<std::string> expanded_search_fields;
        auto field_op = extract_field_name(field_name, search_schema, expanded_search_fields, true, enable_nested_fields);
        if(!field_op.ok()) {
            return Option<nlohmann::json>(field_op.code(), field_op.error());
        }

        for(const auto& expanded_search_field: expanded_search_fields) {
            if (search_schema.count(expanded_search_field) == 0) {
                return Option<nlohmann::json>(404, "Could not find `" + expanded_search_field + "` field in the schema.");
            }
            auto search_field = search_schema.at(expanded_search_field);

            if(search_field.num_dim > 0) {
                num_embed_fields++;

                if(num_embed_fields > 1 ||
                    (!vector_query.field_name.empty() && search_field.name != vector_query.field_name)) {
                    std::string error = "Only one embedding field is allowed in the query.";
                    return Option<nlohmann::json>(400, error);
                }

                if(!search_field.index) {
                    std::string error = "Field `" + search_field.name + "` is marked as a non-indexed field in the schema.";
                    return Option<nlohmann::json>(400, error);
                }

                // if(EmbedderManager::model_dir.empty()) {
                //     std::string error = "Text embedding is not enabled. Please set `model-dir` at startup.";
                //     return Option<nlohmann::json>(400, error);
                // }

                if(query == "*") {
                    // ignore embedding field if query is a wildcard
                    continue;
                }

                if(embedding_fields.find(search_field.name) == embedding_fields.end()) {
                    std::string error = "Vector field `" + search_field.name + "` is not an auto-embedding field, do not use `query_by` with it, use `vector_query` instead.";
                    return Option<nlohmann::json>(400, error);
                }

                EmbedderManager& embedder_manager = EmbedderManager::get_instance();
                auto embedder_op = embedder_manager.get_text_embedder(search_field.embed[fields::model_config]);
                if(!embedder_op.ok()) {
                    return Option<nlohmann::json>(400, embedder_op.error());
                }

                auto remote_embedding_timeout_us = remote_embedding_timeout_ms * 1000;
                if((std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count() - search_begin_us) > remote_embedding_timeout_us) {
                    std::string error = "Request timed out.";
                    return Option<nlohmann::json>(500, error);
                }

                auto embedder = embedder_op.get();

                if(embedder->is_remote()) {
                    // return error if prefix search is used with openai embedder
                    if((prefixes.size() == 1 && prefixes[0] == true) || (prefixes.size() > 1 &&  prefixes[i] == true)) {
                        std::string error = "Prefix search is not supported for remote embedders. Please set `prefix=false` as an additional search parameter to disable prefix searching.";
                        return Option<nlohmann::json>(400, error);
                    }

                    if(remote_embedding_num_tries == 0) {
                        std::string error = "`remote_embedding_num_tries` must be greater than 0.";
                        return Option<nlohmann::json>(400, error);
                    }
                }

                std::string embed_query = embedder_manager.get_query_prefix(search_field.embed[fields::model_config]) + query;
                auto embedding_op = embedder->Embed(embed_query, remote_embedding_timeout_ms, remote_embedding_num_tries);
                if(!embedding_op.success) {
                    if(!embedding_op.error["error"].get<std::string>().empty()) {
                        return Option<nlohmann::json>(400, embedding_op.error["error"].get<std::string>());
                    } else {
                        return Option<nlohmann::json>(400, embedding_op.error.dump());
                    }
                }
                std::vector<float> embedding = embedding_op.embedding;
                // params could have been set for an embed field, so we take a backup and restore
                vector_query.values = embedding;
                vector_query.field_name = field_name;
                continue;
            }

            auto query_weight = !raw_query_by_weights.empty() ? raw_query_by_weights[i] : 0;
            auto num_typo = i < num_typos.size() ? num_typos[i] : num_typos[0];
            auto prefix = i < prefixes.size() ? prefixes[i] : prefixes[0];
            auto infix = i < infixes.size() ? infixes[i] : infixes[0];

            processed_search_fields.emplace_back(expanded_search_field, query_weight, num_typo, prefix, infix);
            if(!raw_query_by_weights.empty()) {
                query_by_weights.push_back(query_weight);
            }
        }
    }

    // Set query to * if it is semantic search
    if(!vector_query.field_name.empty() && processed_search_fields.empty()) {
        query = "*";
    }


    if(!vector_query.field_name.empty() && vector_query.values.empty() && num_embed_fields == 0) {
        std::string error = "Vector query could not find any embedded fields.";
        return Option<nlohmann::json>(400, error);
    }

    if(!query_by_weights.empty() && processed_search_fields.size() != query_by_weights.size()) {
        std::string error = "Error, query_by_weights.size != query_by.size.";
        return Option<nlohmann::json>(400, error);
    }

    for(const auto& processed_search_field: processed_search_fields) {
        const auto& field_name = processed_search_field.name;
        field search_field = search_schema.at(field_name);
        if(!search_field.index) {
            std::string error = "Field `" + field_name + "` is marked as a non-indexed field in the schema.";
            return Option<nlohmann::json>(400, error);
        }

        if(search_field.type != field_types::STRING && search_field.type != field_types::STRING_ARRAY) {
            std::string error = "Field `" + field_name + "` should be a string or a string array.";
            return Option<nlohmann::json>(400, error);
        }
    }

    // validate group by fields
    std::vector<std::string> group_by_fields;

    for(const std::string& field_name: raw_group_by_fields) {
        auto field_op = extract_field_name(field_name, search_schema, group_by_fields, false, enable_nested_fields, false);
        if(!field_op.ok()) {
            return Option<nlohmann::json>(404, field_op.error());
        }
    }

    for(const std::string& field_name: group_by_fields) {
        if(field_name == "id") {
            std::string error = "Cannot use `id` as a group by field.";
            return Option<nlohmann::json>(400, error);
        }

        field search_field = search_schema.at(field_name);

        // must be a facet field
        if(!search_field.is_facet()) {
            std::string error = "Group by field `" + field_name + "` should be a facet field.";
            return Option<nlohmann::json>(400, error);
        }
    }

    tsl::htrie_set<char> include_fields_full;
    tsl::htrie_set<char> exclude_fields_full;

    auto include_exclude_op = populate_include_exclude_fields(include_fields, exclude_fields,
                                                              include_fields_full, exclude_fields_full);

    if(!include_exclude_op.ok()) {
        return Option<nlohmann::json>(include_exclude_op.code(), include_exclude_op.error());
    }

    // process weights for search fields
    std::vector<search_field_t> weighted_search_fields;
    process_search_field_weights(processed_search_fields, query_by_weights, weighted_search_fields);

    const std::string doc_id_prefix = std::to_string(collection_id) + "_" + DOC_ID_PREFIX + "_";
    filter_node_t* filter_tree_root = nullptr;
    Option<bool> parse_filter_op = filter::parse_filter_query(filter_query, search_schema,
                                                              store, doc_id_prefix, filter_tree_root);
    std::unique_ptr<filter_node_t> filter_tree_root_guard(filter_tree_root);

    if(!parse_filter_op.ok()) {
        return Option<nlohmann::json>(parse_filter_op.code(), parse_filter_op.error());
    }

    std::vector<facet> facets;
    // validate facet fields
    for(const std::string & facet_field: facet_fields) {
        
        const auto& res = parse_facet(facet_field, facets);
        if(!res.ok()){
            return Option<nlohmann::json>(res.code(), res.error());
        }
    }

    // parse facet query
    facet_query_t facet_query = {"", ""};

    if(!simple_facet_query.empty()) {
        size_t found_colon_index = simple_facet_query.find(':');

        if(found_colon_index == std::string::npos) {
            std::string error = "Facet query must be in the `facet_field: value` format.";
            return Option<nlohmann::json>(400, error);
        }

        if(facet_fields.empty()) {
            std::string error = "The `facet_query` parameter is supplied without a `facet_by` parameter.";
            return Option<nlohmann::json>(400, error);
        }

        std::string&& facet_query_fname = simple_facet_query.substr(0, found_colon_index);
        StringUtils::trim(facet_query_fname);

        std::string&& facet_query_value = simple_facet_query.substr(found_colon_index+1, std::string::npos);
        StringUtils::trim(facet_query_value);

        if(facet_query_value.empty()) {
            // empty facet value, we will treat it as no facet query
            facet_query = {"", ""};
        } else {
            // facet query field must be part of facet fields requested
            facet_query = { StringUtils::trim(facet_query_fname), facet_query_value };
            bool found = false;
            for(const auto& facet : facets) {
                if(facet.field_name == facet_query.field_name) {
                    found=true;
                    break;
                }
            }
            if(!found) {
                std::string error = "Facet query refers to a facet field `" + facet_query.field_name + "` " +
                                    "that is not part of `facet_by` parameter.";
                return Option<nlohmann::json>(400, error);
            }

            if(search_schema.count(facet_query.field_name) == 0 || !search_schema.at(facet_query.field_name).facet) {
                std::string error = "Could not find a facet field named `" + facet_query.field_name + "` in the schema.";
                return Option<nlohmann::json>(404, error);
            }
        }
    }

    if(per_page > PER_PAGE_MAX) {
        std::string message = "Only upto " + std::to_string(PER_PAGE_MAX) + " hits can be fetched per page.";
        return Option<nlohmann::json>(422, message);
    }

    size_t offset = 0;

    if(page == 0 && page_offset != 0) {
        // if only offset is set, use that
        offset = page_offset;
    } else {
        // if both are set or none set, use page value (default is 1)
        size_t actual_page = (page == 0) ? 1 : page;
        offset = (per_page * (actual_page - 1));
    }

    size_t fetch_size = offset + per_page;

    if(fetch_size > limit_hits) {
        std::string message = "Only upto " + std::to_string(limit_hits) + " hits can be fetched. " +
                "Ensure that `page` and `per_page` parameters are within this range.";
        return Option<nlohmann::json>(422, message);
    }

    size_t max_hits = DEFAULT_TOPSTER_SIZE;

    // ensure that `max_hits` never exceeds number of documents in collection
    if(weighted_search_fields.size() <= 1 || query == "*") {
        max_hits = std::min(std::max(fetch_size, max_hits), get_num_documents());
    } else {
        max_hits = std::min(std::max(fetch_size, max_hits), get_num_documents());
    }

    if(token_order == NOT_SET) {
        if(default_sorting_field.empty()) {
            token_order = FREQUENCY;
        } else {
            token_order = MAX_SCORE;
        }
    }

    Option<drop_tokens_param_t> drop_tokens_param_op = parse_drop_tokens_mode(drop_tokens_mode);
    if(!drop_tokens_param_op.ok()) {
        return Option<nlohmann::json>(drop_tokens_param_op.code(), drop_tokens_param_op.error());
    }

    auto drop_tokens_param = drop_tokens_param_op.get();

    std::vector<std::vector<KV*>> raw_result_kvs;
    std::vector<std::vector<KV*>> override_result_kvs;

    size_t total = 0;

    std::vector<uint32_t> excluded_ids;
    std::vector<std::pair<uint32_t, uint32_t>> included_ids; // ID -> position
    std::map<size_t, std::vector<std::string>> pinned_hits;

    Option<bool> pinned_hits_op = parse_pinned_hits(pinned_hits_str, pinned_hits);

    if(!pinned_hits_op.ok()) {
        return Option<nlohmann::json>(400, pinned_hits_op.error());
    }

    std::vector<std::string> hidden_hits;
    StringUtils::split(hidden_hits_str, hidden_hits, ",");

    nlohmann::json override_metadata;
    std::vector<const override_t*> filter_overrides;
    bool filter_curated_hits = false;
    std::string curated_sort_by;
    std::set<std::string> override_tag_set;

    std::vector<std::string> override_tags_vec;
    StringUtils::split(override_tags_str, override_tags_vec, ",");
    for(const auto& tag: override_tags_vec) {
        override_tag_set.insert(tag);
    }

    curate_results(query, filter_query, enable_overrides, pre_segmented_query, override_tag_set,
                   pinned_hits, hidden_hits, included_ids, excluded_ids, filter_overrides, filter_curated_hits,
                   curated_sort_by, override_metadata);

    if(filter_curated_hits_option == 0 || filter_curated_hits_option == 1) {
        // When query param has explicit value set, override level configuration takes lower precedence.
        filter_curated_hits = bool(filter_curated_hits_option);
    }

    /*for(auto& kv: included_ids) {
        LOG(INFO) << "key: " << kv.first;
        for(auto val: kv.second) {
            LOG(INFO) << val;
        }
    }

    LOG(INFO) << "Excludes:";

    for(auto id: excluded_ids) {
        LOG(INFO) << id;
    }

    LOG(INFO) << "included_ids size: " << included_ids.size();
    for(auto& group: included_ids) {
        for(uint32_t& seq_id: group.second) {
            LOG(INFO) << "seq_id: " << seq_id;
        }

        LOG(INFO) << "----";
    }
    */

    // validate sort fields and standardize

    sort_fields_guard_t sort_fields_guard;
    std::vector<sort_by>& sort_fields_std = sort_fields_guard.sort_fields_std;

    bool is_wildcard_query = (query == "*");
    bool is_group_by_query = group_by_fields.size() > 0;
    bool is_vector_query = !vector_query.field_name.empty();

    if(curated_sort_by.empty()) {
        auto sort_validation_op = validate_and_standardize_sort_fields(sort_fields,
                                    sort_fields_std, is_wildcard_query, is_vector_query,
                                    raw_query, is_group_by_query, remote_embedding_timeout_ms, remote_embedding_num_tries);
        if(!sort_validation_op.ok()) {
            return Option<nlohmann::json>(sort_validation_op.code(), sort_validation_op.error());
        }
    } else {
        std::vector<sort_by> curated_sort_fields;
        bool parsed_sort_by = CollectionManager::parse_sort_by_str(curated_sort_by, curated_sort_fields);
        if(!parsed_sort_by) {
            return Option<nlohmann::json>(400, "Parameter `sort_by` is malformed.");
        }

        auto sort_validation_op = validate_and_standardize_sort_fields(curated_sort_fields, 
                                    sort_fields_std, is_wildcard_query, is_vector_query, raw_query, is_group_by_query, remote_embedding_timeout_ms, remote_embedding_num_tries);
        if(!sort_validation_op.ok()) {
            return Option<nlohmann::json>(sort_validation_op.code(), sort_validation_op.error());
        }
    }

    // apply bucketing on text match score
    int match_score_index = -1;
    for(size_t i = 0; i < sort_fields_std.size(); i++) {
        if(sort_fields_std[i].name == sort_field_const::text_match && sort_fields_std[i].text_match_buckets != 0) {
            match_score_index = i;
            break;
        }
    }

    //LOG(INFO) << "Num indices used for querying: " << indices.size();
    std::vector<query_tokens_t> field_query_tokens;
    std::vector<std::string> q_tokens;  // used for auxillary highlighting
    std::vector<std::string> q_include_tokens;

    if(weighted_search_fields.size() == 0) {
        // has to be a wildcard query
        field_query_tokens.emplace_back(query_tokens_t{});
        parse_search_query(query, q_include_tokens,
                           field_query_tokens[0].q_exclude_tokens, field_query_tokens[0].q_phrases, "",
                           false, stopwords_set);

        process_filter_overrides(filter_overrides, q_include_tokens, token_order, filter_tree_root,
                                 included_ids, excluded_ids, override_metadata);

        for(size_t i = 0; i < q_include_tokens.size(); i++) {
            auto& q_include_token = q_include_tokens[i];
            field_query_tokens[0].q_include_tokens.emplace_back(i, q_include_token, (i == q_include_tokens.size() - 1),
                                                                q_include_token.size(), 0);
        }
    } else {
        field_query_tokens.emplace_back(query_tokens_t{});
        const std::string & field_locale = search_schema.at(weighted_search_fields[0].name).locale;
        parse_search_query(query, q_include_tokens,
                           field_query_tokens[0].q_exclude_tokens,
                           field_query_tokens[0].q_phrases,
                           field_locale, pre_segmented_query, stopwords_set);

        // process filter overrides first, before synonyms (order is important)

        // included_ids, excluded_ids
        process_filter_overrides(filter_overrides, q_include_tokens, token_order, filter_tree_root,
                                 included_ids, excluded_ids, override_metadata);

        for(size_t i = 0; i < q_include_tokens.size(); i++) {
            auto& q_include_token = q_include_tokens[i];
            q_tokens.push_back(q_include_token);
            field_query_tokens[0].q_include_tokens.emplace_back(i, q_include_token, (i == q_include_tokens.size() - 1),
                                                                q_include_token.size(), 0);
        }

        for(auto& phrase: field_query_tokens[0].q_phrases) {
            for(auto& token: phrase) {
                q_tokens.push_back(token);
            }
        }

        for(size_t i = 1; i < weighted_search_fields.size(); i++) {
            field_query_tokens.emplace_back(query_tokens_t{});
            field_query_tokens[i] = field_query_tokens[0];
        }
    }

    // search all indices

    size_t index_id = 0;
    search_args* search_params = new search_args(field_query_tokens, weighted_search_fields,
                                                 match_type,
                                                 filter_tree_root, facets, included_ids, excluded_ids,
                                                 sort_fields_std, facet_query, num_typos, max_facet_values, max_hits,
                                                 per_page, offset, token_order, prefixes,
                                                 drop_tokens_threshold, typo_tokens_threshold,
                                                 group_by_fields, group_limit, group_missing_values,
                                                 default_sorting_field,
                                                 prioritize_exact_match, prioritize_token_position,
                                                 prioritize_num_matching_fields,
                                                 exhaustive_search, 4,
                                                 search_stop_millis,
                                                 min_len_1typo, min_len_2typo, max_candidates, infixes,
                                                 max_extra_prefix, max_extra_suffix, facet_query_num_typos,
                                                 filter_curated_hits, split_join_tokens, vector_query,
                                                 facet_sample_percent, facet_sample_threshold, drop_tokens_param);

    std::unique_ptr<search_args> search_params_guard(search_params);

    auto search_op = index->run_search(search_params, this, facet_index_type);

    // filter_tree_root might be updated in Index::static_filter_query_eval.
    filter_tree_root_guard.release();
    filter_tree_root_guard.reset(filter_tree_root);


    if (!search_op.ok()) {
        return Option<nlohmann::json>(search_op.code(), search_op.error());
    }

    // for grouping we have to re-aggregate
    Topster& topster = *search_params->topster;
    Topster& curated_topster = *search_params->curated_topster;

    topster.sort();
    curated_topster.sort();

    populate_result_kvs(&topster, raw_result_kvs, search_params->groups_processed, sort_fields_std);
    populate_result_kvs(&curated_topster, override_result_kvs, search_params->groups_processed, sort_fields_std);

    // for grouping we have to aggregate group set sizes to a count value
    if(group_limit) {
        total = search_params->groups_processed.size() + override_result_kvs.size();
    } else {
        total = search_params->all_result_ids_len;
    }
    

    if(search_cutoff && total == 0) {
        // this can happen if other requests stopped this request from being processed
        // we should return an error so that request can be retried by client
        return Option<nlohmann::json>(408, "Request Timeout");
    }

    if(match_score_index >= 0 && sort_fields_std[match_score_index].text_match_buckets > 0) {
        size_t num_buckets = sort_fields_std[match_score_index].text_match_buckets;
        const size_t max_kvs_bucketed = std::min<size_t>(DEFAULT_TOPSTER_SIZE, raw_result_kvs.size());

        if(max_kvs_bucketed >= num_buckets) {
            spp::sparse_hash_map<uint64_t, int64_t> result_scores;

            // only first `max_kvs_bucketed` elements are bucketed to prevent pagination issues past 250 records
            size_t block_len = (max_kvs_bucketed / num_buckets);
            size_t i = 0;
            while(i < max_kvs_bucketed) {
                int64_t anchor_score = raw_result_kvs[i][0]->scores[raw_result_kvs[i][0]->match_score_index];
                size_t j = 0;
                while(j < block_len && i+j < max_kvs_bucketed) {
                    result_scores[raw_result_kvs[i+j][0]->key] = raw_result_kvs[i+j][0]->scores[raw_result_kvs[i+j][0]->match_score_index];
                    raw_result_kvs[i+j][0]->scores[raw_result_kvs[i+j][0]->match_score_index] = anchor_score;
                    j++;
                }

                i += j;
            }

            // sort again based on bucketed match score
            std::partial_sort(raw_result_kvs.begin(), raw_result_kvs.begin() + max_kvs_bucketed, raw_result_kvs.end(),
                              Topster::is_greater_kv_group);

            // restore original scores
            for(i = 0; i < max_kvs_bucketed; i++) {
                raw_result_kvs[i][0]->scores[raw_result_kvs[i][0]->match_score_index] =
                        result_scores[raw_result_kvs[i][0]->key];
            }
        }
    }

    // Sort based on position in overridden list
    std::sort(
      override_result_kvs.begin(), override_result_kvs.end(),
      [](const std::vector<KV*>& a, std::vector<KV*>& b) -> bool {
          return a[0]->distinct_key < b[0]->distinct_key;
      }
    );

    std::vector<std::vector<KV*>> result_group_kvs;
    size_t override_kv_index = 0;
    size_t raw_results_index = 0;

    // merge raw results and override results
    while(raw_results_index < raw_result_kvs.size()) {
        if(override_kv_index < override_result_kvs.size()) {
            size_t result_position = result_group_kvs.size() + 1;
            uint64_t override_position = override_result_kvs[override_kv_index][0]->distinct_key;
            if(result_position == override_position) {
                override_result_kvs[override_kv_index][0]->match_score_index = CURATED_RECORD_IDENTIFIER;
                result_group_kvs.push_back(override_result_kvs[override_kv_index]);
                override_kv_index++;
                continue;
            }
        }

        result_group_kvs.push_back(raw_result_kvs[raw_results_index]);
        raw_results_index++;
    }

    while(override_kv_index < override_result_kvs.size()) {
        override_result_kvs[override_kv_index][0]->match_score_index = CURATED_RECORD_IDENTIFIER;
        result_group_kvs.push_back({override_result_kvs[override_kv_index]});
        override_kv_index++;
    }

    std::string facet_query_last_token;
    size_t facet_query_num_tokens = 0;       // used to identify drop token scenario

    if(!facet_query.query.empty()) {
        // identify facet hash tokens

        auto fq_field = search_schema.at(facet_query.field_name);
        bool is_cyrillic = Tokenizer::is_cyrillic(fq_field.locale);
        bool normalise = is_cyrillic ? false : true;

        std::vector<std::string> facet_query_tokens;
        Tokenizer(facet_query.query, normalise, !fq_field.is_string(), fq_field.locale,
                  symbols_to_index, token_separators).tokenize(facet_query_tokens);

        facet_query_num_tokens = facet_query_tokens.size();
        facet_query_last_token = facet_query_tokens.empty() ? "" : facet_query_tokens.back();
    }

    const long start_result_index = offset;

    // `end_result_index` could be -1 when max_hits is 0
    const long end_result_index = std::min(fetch_size, std::min(max_hits, result_group_kvs.size())) - 1;

    // handle which fields have to be highlighted

    std::vector<highlight_field_t> highlight_items;
    std::vector<std::string> highlight_field_names;
    StringUtils::split(highlight_fields, highlight_field_names, ",");

    std::vector<std::string> highlight_full_field_names;
    StringUtils::split(highlight_full_fields, highlight_full_field_names, ",");

    if(query != "*") {
        process_highlight_fields(weighted_search_fields, raw_search_fields, include_fields_full, exclude_fields_full,
                                 highlight_field_names, highlight_full_field_names, infixes, q_tokens,
                                 search_params->qtoken_set, highlight_items);
    }

    nlohmann::json result = nlohmann::json::object();
    result["found"] = total;
    if(group_limit != 0) {
        result["found_docs"] = search_params->all_result_ids_len;
    }

    if(exclude_fields.count("out_of") == 0) {
        result["out_of"] = num_documents.load();
    }

    std::string hits_key = group_limit ? "grouped_hits" : "hits";
    result[hits_key] = nlohmann::json::array();

    uint8_t index_symbols[256] = {};
    for(char c: symbols_to_index) {
        index_symbols[uint8_t(c)] = 1;
    }

    nlohmann::json docs_array = nlohmann::json::array();

    // handle analytics query expansion
    std::string first_q = raw_query;
    expand_search_query(raw_query, offset, total, search_params, result_group_kvs, raw_search_fields, first_q);

    // construct results array
    for(long result_kvs_index = start_result_index; result_kvs_index <= end_result_index; result_kvs_index++) {
        const std::vector<KV*> & kv_group = result_group_kvs[result_kvs_index];

        nlohmann::json group_hits;
        if(group_limit) {
            group_hits["hits"] = nlohmann::json::array();
        }

        nlohmann::json& hits_array = group_limit ? group_hits["hits"] : result["hits"];
        nlohmann::json group_key = nlohmann::json::array();

        for(const KV* field_order_kv: kv_group) {
            const std::string& seq_id_key = get_seq_id_key((uint32_t) field_order_kv->key);

            nlohmann::json document;
            const Option<bool> & document_op = get_document_from_store(seq_id_key, document);

            if(!document_op.ok()) {
                LOG(ERROR) << "Document fetch error. " << document_op.error();
                continue;
            }

            nlohmann::json highlight_res = nlohmann::json::object();

            if(!highlight_items.empty()) {
                copy_highlight_doc(highlight_items, enable_nested_fields, document, highlight_res);
                remove_flat_fields(highlight_res);
                remove_reference_helper_fields(highlight_res);
                highlight_res.erase("id");
            }

            nlohmann::json wrapper_doc;

            if(enable_highlight_v1) {
                wrapper_doc["highlights"] = nlohmann::json::array();
            }

            std::vector<highlight_t> highlights;
            StringUtils string_utils;

            tsl::htrie_set<char> hfield_names;
            tsl::htrie_set<char> h_full_field_names;

            for(size_t i = 0; i < highlight_items.size(); i++) {
                auto& highlight_item = highlight_items[i];
                const std::string& field_name = highlight_item.name;
                if(search_schema.count(field_name) == 0) {
                    continue;
                }

                field search_field = search_schema.at(field_name);

                if(query != "*") {
                    highlight_t highlight;
                    highlight.field = search_field.name;

                    bool found_highlight = false;
                    bool found_full_highlight = false;

                    highlight_result(raw_query, search_field, i, highlight_item.qtoken_leaves, field_order_kv,
                                     document, highlight_res,
                                     string_utils, snippet_threshold,
                                     highlight_affix_num_tokens, highlight_item.fully_highlighted, highlight_item.infix,
                                     highlight_start_tag, highlight_end_tag, index_symbols, highlight,
                                     found_highlight, found_full_highlight);
                    if(!highlight.snippets.empty()) {
                        highlights.push_back(highlight);
                    }

                    if(found_highlight) {
                        hfield_names.insert(search_field.name);
                        if(found_full_highlight) {
                            h_full_field_names.insert(search_field.name);
                        }
                    }
                }
            }

            // explicit highlight fields could be parent of searched fields, so we will take a pass at that
            for(auto& hfield_name: highlight_full_field_names) {
                auto it = h_full_field_names.equal_prefix_range(hfield_name);
                if(it.first != it.second) {
                    h_full_field_names.insert(hfield_name);
                }
            }

            if(highlight_field_names.empty()) {
                for(auto& raw_search_field: raw_search_fields) {
                    auto it = hfield_names.equal_prefix_range(raw_search_field);
                    if(it.first != it.second) {
                        hfield_names.insert(raw_search_field);
                    }
                }
            } else {
                for(auto& hfield_name: highlight_field_names) {
                    auto it = hfield_names.equal_prefix_range(hfield_name);
                    if(it.first != it.second) {
                        hfield_names.insert(hfield_name);
                    }
                }
            }

            // remove fields from highlight doc that were not highlighted
            if(!hfield_names.empty()) {
                prune_doc(highlight_res, hfield_names, tsl::htrie_set<char>(), "");
            } else {
                highlight_res.clear();
            }

            if(enable_highlight_v1) {
                std::sort(highlights.begin(), highlights.end());

                for(const auto & highlight: highlights) {
                    auto field_it = search_schema.find(highlight.field);
                    if(field_it == search_schema.end() || field_it->nested) {
                        // nested field highlighting will be available only in the new highlight structure.
                        continue;
                    }

                    nlohmann::json h_json = nlohmann::json::object();
                    h_json["field"] = highlight.field;

                    if(!highlight.indices.empty()) {
                        h_json["matched_tokens"] = highlight.matched_tokens;
                        h_json["indices"] = highlight.indices;
                        h_json["snippets"] = highlight.snippets;
                        if(!highlight.values.empty()) {
                            h_json["values"] = highlight.values;
                        }
                    } else {
                        h_json["matched_tokens"] = highlight.matched_tokens[0];
                        h_json["snippet"] = highlight.snippets[0];
                        if(!highlight.values.empty() && !highlight.values[0].empty()) {
                            h_json["value"] = highlight.values[0];
                        }
                    }

                    wrapper_doc["highlights"].push_back(h_json);
                }
            }

            //wrapper_doc["seq_id"] = (uint32_t) field_order_kv->key;

            if(group_limit && group_key.empty()) {
                for(const auto& field_name: group_by_fields) {
                    if(document.count(field_name) != 0) {
                        group_key.push_back(document[field_name]);
                    }
                }
            }

            remove_flat_fields(document);
            remove_reference_helper_fields(document);
            auto doc_id_op = doc_id_to_seq_id(document["id"].get<std::string>());
            if (!doc_id_op.ok()) {
                return Option<nlohmann::json>(doc_id_op.code(), doc_id_op.error());
            }

            auto prune_op = prune_doc(document,
                                      include_fields_full,
                                      exclude_fields_full,
                                      "",
                                      0,
                                      field_order_kv->reference_filter_results,
                                      const_cast<Collection *>(this), get_seq_id_from_key(seq_id_key),
                                      ref_include_exclude_fields_vec);
            if (!prune_op.ok()) {
                return Option<nlohmann::json>(prune_op.code(), prune_op.error());
            }

            if(conversation) {
                docs_array.push_back(document);
            }

            wrapper_doc["document"] = document;
            wrapper_doc["highlight"] = highlight_res;

            if(field_order_kv->match_score_index == CURATED_RECORD_IDENTIFIER) {
                wrapper_doc["curated"] = true;
            } else if(field_order_kv->match_score_index >= 0) {
                wrapper_doc["text_match"] = field_order_kv->text_match_score;
                wrapper_doc["text_match_info"] = nlohmann::json::object();
                populate_text_match_info(wrapper_doc["text_match_info"],
                                        field_order_kv->text_match_score, match_type);
                if(!vector_query.field_name.empty()) {
                    wrapper_doc["hybrid_search_info"] = nlohmann::json::object();
                    wrapper_doc["hybrid_search_info"]["rank_fusion_score"] = Index::int64_t_to_float(field_order_kv->scores[field_order_kv->match_score_index]);
                }
            }

            nlohmann::json geo_distances;

            for(size_t sort_field_index = 0; sort_field_index < sort_fields_std.size(); sort_field_index++) {
                const auto& sort_field = sort_fields_std[sort_field_index];
                if(sort_field.geopoint != 0) {
                    geo_distances[sort_field.name] = std::abs(field_order_kv->scores[sort_field_index]);
                }
            }

            if(!geo_distances.empty()) {
                wrapper_doc["geo_distance_meters"] = geo_distances;
            }

            if(!vector_query.field_name.empty() && field_order_kv->vector_distance >= 0) {
                wrapper_doc["vector_distance"] = field_order_kv->vector_distance;
            }

            hits_array.push_back(wrapper_doc);
        }

        if(group_limit) {
            group_hits["group_key"] = group_key;

            const auto& itr = search_params->groups_processed.find(kv_group[0]->distinct_key);
            
            if(itr != search_params->groups_processed.end()) {
                group_hits["found"] = itr->second;
            }
            result["grouped_hits"].push_back(group_hits);
        }
    }

    if(conversation) {
        result["conversation"] = nlohmann::json::object();
        result["conversation"]["query"] = raw_query;

        // remove all fields with vector type from docs_array
        for(const auto& field : search_schema) {
            if(field.type == field_types::FLOAT_ARRAY && field.num_dim > 0) {
                for(auto& doc : docs_array) {
                    doc.erase(field.name);
                }
            }
        }

        // remove document with lowest score until total tokens is less than MAX_TOKENS
        while(ConversationManager::get_instance().get_token_count(docs_array) > ConversationManager::get_instance().MAX_TOKENS) {
            try {
                docs_array.erase(docs_array.size() - 1);
            } catch(...) {
                return Option<nlohmann::json>(400, "Failed to remove document from search results.");
            }
        }

        auto conversation_model = ConversationModelManager::get_model(conversation_model_id).get();

        bool has_conversation_history = !conversation_id.empty();
        auto qa_op = ConversationModel::get_answer(docs_array.dump(0), raw_query, conversation_model);
        if(!qa_op.ok()) {
            return Option<nlohmann::json>(qa_op.code(), qa_op.error());
        }
        result["conversation"]["answer"] = qa_op.get();
        result["conversation"]["conversation_id"] = conversation_id;

        auto formatted_question_op = ConversationModel::format_question(raw_query, conversation_model);
        if(!formatted_question_op.ok()) {
            return Option<nlohmann::json>(formatted_question_op.code(), formatted_question_op.error());
        }

        auto formatted_answer_op = ConversationModel::format_answer(qa_op.get(), conversation_model);
        if(!formatted_answer_op.ok()) {
            return Option<nlohmann::json>(formatted_answer_op.code(), formatted_answer_op.error());
        }

        if(has_conversation_history) {
            ConversationManager::get_instance().append_conversation(conversation_id, formatted_question_op.get());
            ConversationManager::get_instance().append_conversation(conversation_id, formatted_answer_op.get());
            auto get_conversation_op = ConversationManager::get_instance().get_conversation(conversation_id);
            if(!get_conversation_op.ok()) {
                return Option<nlohmann::json>(get_conversation_op.code(), get_conversation_op.error());
            }

            auto conversation_history = get_conversation_op.get();
            if(exclude_fields.count("conversation_history") == 0) {
                result["conversation"]["conversation_history"] = conversation_history;
            }
            result["conversation"]["conversation_id"] = conversation_id;
        } else {
            nlohmann::json conversation_history = nlohmann::json::array();
            conversation_history.push_back(formatted_question_op.get());
            conversation_history.push_back(formatted_answer_op.get());

            auto create_conversation_op = ConversationManager::get_instance().create_conversation(conversation_history);
            if(!create_conversation_op.ok()) {
                return Option<nlohmann::json>(create_conversation_op.code(), create_conversation_op.error());
            }

            auto get_conversation_op = ConversationManager::get_instance().get_conversation(create_conversation_op.get());
            if(!get_conversation_op.ok()) {
                return Option<nlohmann::json>(get_conversation_op.code(), get_conversation_op.error());
            }

            if(exclude_fields.count("conversation_history") == 0) {
                result["conversation"]["conversation_history"] = get_conversation_op.get();
            }
            result["conversation"]["conversation_id"] = create_conversation_op.get();
        }
    }

    result["facet_counts"] = nlohmann::json::array();
    
    // populate facets
    for(facet& a_facet: facets) {
        // Don't return zero counts for a wildcard facet.
        if (a_facet.is_wildcard_match &&
                (((a_facet.is_intersected && a_facet.value_result_map.empty())) ||
                (!a_facet.is_intersected && a_facet.result_map.empty()))) {
            continue;
        }

        // check for search cutoff elapse
        if((std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().
            time_since_epoch()).count() - search_begin_us) > search_stop_us) {
            search_cutoff = true;
            break;
        }

        nlohmann::json facet_result = nlohmann::json::object();
        facet_result["field_name"] = a_facet.field_name;
        facet_result["sampled"] = a_facet.sampled;
        facet_result["counts"] = nlohmann::json::array();

        std::vector<facet_value_t> facet_values;
        std::vector<facet_count_t> facet_counts;

        for (const auto & kv : a_facet.result_map) {
            facet_count_t v = kv.second;
            v.fhash = kv.first;
            v.sort_field_val = kv.second.sort_field_val;
            facet_counts.emplace_back(v);
        }

        for (const auto& kv : a_facet.value_result_map) {
            facet_count_t v = kv.second;
            v.fvalue = kv.first;
            v.fhash = StringUtils::hash_wy(kv.first.c_str(), kv.first.size());
            facet_counts.emplace_back(v);
        }
        
        auto max_facets = std::min(max_facet_values, facet_counts.size());
        auto nthElement = max_facets == facet_counts.size() ? max_facets - 1 : max_facets;
        std::nth_element(facet_counts.begin(), facet_counts.begin() + nthElement, facet_counts.end(),
                         Collection::facet_count_compare);

        if(a_facet.is_range_query){
            for(const auto& kv : a_facet.result_map){
                auto facet_range_iter = a_facet.facet_range_map.find(kv.first);
                if(facet_range_iter != a_facet.facet_range_map.end()){
                    auto & facet_count = kv.second;
                    facet_value_t facet_value = {facet_range_iter->second, std::string(), facet_count.count};
                    facet_values.emplace_back(facet_value);
                }
                else{
                    LOG (ERROR) << "range_id not found in result map.";
                }
            }
        } else {
            field the_field;
            if (a_facet.reference_collection_name.empty()) {
                the_field = search_schema.at(a_facet.field_name);
            } else {
                auto& cm = CollectionManager::get_instance();
                auto ref_collection = cm.get_collection(a_facet.reference_collection_name);
                if (ref_collection == nullptr) {
                    continue;
                }

                the_field = ref_collection->get_schema().at(a_facet.field_name);
            }
            bool should_return_parent = std::find(facet_return_parent.begin(), facet_return_parent.end(),
                                                  the_field.name) != facet_return_parent.end();
            bool should_fetch_doc_from_store = ((a_facet.is_intersected && should_return_parent) || !a_facet.is_intersected);

            for(size_t fi = 0; fi < max_facets; fi++) {
                // remap facet value hash with actual string
                auto & facet_count = facet_counts[fi];
                std::string value;
                nlohmann::json document;

                if(should_fetch_doc_from_store) {
                    const std::string &seq_id_key = get_seq_id_key((uint32_t) facet_count.doc_id);
                    const Option<bool> &document_op = get_document_from_store(seq_id_key, document);
                    if (!document_op.ok()) {
                        LOG(ERROR) << "Facet fetch error. " << document_op.error();
                        continue;
                    }
                }

                if(a_facet.is_intersected) {
                    value = facet_count.fvalue;
                    //LOG(INFO) << "used intersection";
                } else {
                    // fetch actual facet value from representative doc id
                    //LOG(INFO) << "used hashes";
                    bool facet_found = facet_value_to_string(a_facet, facet_count, document, value);
                    if(!facet_found) {
                        continue;
                    }
                }

                highlight_t highlight;

                if(!facet_query.query.empty()) {
                    bool use_word_tokenizer = Tokenizer::has_word_tokenizer(the_field.locale);
                    bool normalise = !use_word_tokenizer;

                    std::vector<std::string> fquery_tokens;
                    Tokenizer(facet_query.query, true, false, the_field.locale, symbols_to_index,
                              token_separators).tokenize(fquery_tokens);

                    if(fquery_tokens.empty()) {
                        continue;
                    }

                    std::vector<string>& ftokens = a_facet.is_intersected ? a_facet.fvalue_tokens[facet_count.fvalue] :
                                                   a_facet.hash_tokens[facet_count.fhash];

                    tsl::htrie_map<char, token_leaf> qtoken_leaves;

                    //LOG(INFO) << "working on hash_tokens for hash " << kv.first << " with size " << ftokens.size();
                    for(size_t ti = 0; ti < ftokens.size(); ti++) {
                        if(the_field.is_bool()) {
                            if(ftokens[ti] == "1") {
                                ftokens[ti] = "true";
                            } else {
                                ftokens[ti] = "false";
                            }
                        }

                        Tokenizer(facet_query.query, true, false, the_field.locale, symbols_to_index,
                                  token_separators).tokenize(ftokens[ti]);

                        const std::string& resolved_token = ftokens[ti];
                        size_t root_len = (fquery_tokens.size() == ftokens.size()) ?
                                          fquery_tokens[ti].size() :
                                          resolved_token.size();

                        token_leaf leaf(nullptr, root_len, 0, (ti == ftokens.size()-1));
                        qtoken_leaves.emplace(resolved_token, leaf);
                    }

                    std::vector<std::string> raw_fquery_tokens;
                    Tokenizer(facet_query.query, normalise, false, the_field.locale, symbols_to_index,
                              token_separators).tokenize(raw_fquery_tokens);

                    if(raw_fquery_tokens.empty()) {
                        continue;
                    }

                    size_t prefix_token_num_chars = StringUtils::get_num_chars(raw_fquery_tokens.back());

                    StringUtils string_utils;
                    size_t last_valid_offset = 0;
                    int last_valid_offset_index = -1;
                    match_index_t match_index(Match(), 0, 0);

                    uint8_t index_symbols[256] = {};
                    for(char c: symbols_to_index) {
                        index_symbols[uint8_t(c)] = 1;
                    }

                    handle_highlight_text(value, normalise, the_field, symbols_to_index, token_separators,
                                          highlight, string_utils, use_word_tokenizer,
                                          highlight_affix_num_tokens, qtoken_leaves, last_valid_offset_index,
                                          prefix_token_num_chars, false, snippet_threshold, false, ftokens,
                                          last_valid_offset, highlight_start_tag, highlight_end_tag,
                                          index_symbols, match_index);
                }

                nlohmann::json parent;
                if(the_field.nested && should_return_parent) {
                    parent = get_facet_parent(the_field.name, document);
                }

                const auto& highlighted_text = highlight.snippets.empty() ? value : highlight.snippets[0];
                facet_value_t facet_value = {value, highlighted_text, facet_count.count,
                                             facet_count.sort_field_val, parent};
                facet_values.emplace_back(facet_value);
            }
        }

        if(a_facet.is_sort_by_alpha) {
            bool is_asc = a_facet.sort_order == "asc";
            std::stable_sort(facet_values.begin(), facet_values.end(),
                             [&] (const auto& fv1, const auto& fv2) {
                if(is_asc) {
                    return fv1.value < fv2.value;
                }

                return fv1.value > fv2.value;
            });
        } else if(!a_facet.sort_field.empty()) {
            bool is_asc = a_facet.sort_order == "asc";
            std::stable_sort(facet_values.begin(), facet_values.end(),
                             [&] (const auto& fv1, const auto& fv2) {
                if(is_asc) {
                    return fv1.sort_field_val < fv2.sort_field_val;
                }

                return fv1.sort_field_val > fv2.sort_field_val;
            });
        } else {
            std::stable_sort(facet_values.begin(), facet_values.end(), Collection::facet_count_str_compare);
        }

        for(const auto & facet_count: facet_values) {
            nlohmann::json facet_value_count = nlohmann::json::object();
            const std::string & value = facet_count.value;

            facet_value_count["value"] = value;
            facet_value_count["highlighted"] = facet_count.highlighted;
            facet_value_count["count"] = facet_count.count;

            if(!facet_count.parent.empty()) {
                facet_value_count["parent"] = facet_count.parent;
            }
            facet_result["counts"].push_back(facet_value_count);
        }

        // add facet value stats
        facet_result["stats"] = nlohmann::json::object();
        if(a_facet.stats.fvcount != 0) {
            facet_result["stats"]["min"] = a_facet.stats.fvmin;
            facet_result["stats"]["max"] = a_facet.stats.fvmax;
            facet_result["stats"]["sum"] = a_facet.stats.fvsum;
            facet_result["stats"]["avg"] = (a_facet.stats.fvsum / a_facet.stats.fvcount);
        }

        facet_result["stats"]["total_values"] = facet_values.size();
        result["facet_counts"].push_back(facet_result);
    }

    result["search_cutoff"] = search_cutoff;

    result["request_params"] = nlohmann::json::object();
    result["request_params"]["collection_name"] = name;
    result["request_params"]["per_page"] = per_page;
    if(!raw_query.empty()) {
        result["request_params"]["q"] = raw_query;
        result["request_params"]["first_q"] = first_q;
    }

    if(!voice_query.empty()) {
        result["request_params"]["voice_query"] = nlohmann::json::object();
        result["request_params"]["voice_query"]["transcribed_query"] = transcribed_query;
    }
    if(!override_metadata.empty()) {
        result["metadata"] = override_metadata;
    }

    //long long int timeMillis = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - begin).count();
    //!LOG(INFO) << "Time taken for result calc: " << timeMillis << "us";
    //!store->print_memory_usage();
    return Option<nlohmann::json>(result);
}

void Collection::expand_search_query(const string& raw_query, size_t offset, size_t total, const search_args* search_params,
                                    const std::vector<std::vector<KV*>>& result_group_kvs,
                                    const std::vector<std::string>& raw_search_fields, string& first_q) const {
    if(!Config::get_instance().get_enable_search_analytics()) {
        return ;
    }

    if(offset == 0 && !raw_search_fields.empty() && !search_params->searched_queries.empty() &&
        total != 0 && !result_group_kvs.empty()) {
        // we have to map raw_query (which could contain a prefix) back to expanded version
        auto search_field_it = search_schema.find(raw_search_fields[0]);
        if(search_field_it == search_schema.end() || Tokenizer::has_word_tokenizer(search_field_it->locale)) {
            return ;
        }

        first_q = "";
        auto q_index = result_group_kvs[0][0]->query_index;
        if(q_index >= search_params->searched_queries.size()) {
            return ;
        }

        const auto& qleaves = search_params->searched_queries[q_index];
        Tokenizer tokenizer(raw_query, true, false, search_field_it->locale, symbols_to_index, token_separators);
        std::string raw_token;
        size_t raw_token_index = 0, tok_start = 0, tok_end = 0;

        while(tokenizer.next(raw_token, raw_token_index, tok_start, tok_end)) {
            if(raw_token_index < qleaves.size()) {
                auto leaf = qleaves[raw_token_index];
                std::string tok(reinterpret_cast<char*>(leaf->key), leaf->key_len - 1);
                if(StringUtils::begins_with(tok, raw_token)) {
                    first_q += tok + " ";
                }
            }
        }

        if(!first_q.empty()) {
            first_q.pop_back();
        }
    }
}

void Collection::copy_highlight_doc(std::vector<highlight_field_t>& hightlight_items,
                                    const bool nested_fields_enabled,
                                    const nlohmann::json& src, nlohmann::json& dst) {
    for(const auto& hightlight_item: hightlight_items) {
        if(!nested_fields_enabled && src.count(hightlight_item.name) != 0) {
            dst[hightlight_item.name] = src[hightlight_item.name];
            continue;
        }

        std::string root_field_name;

        for(size_t i = 0; i < hightlight_item.name.size(); i++) {
            if(hightlight_item.name[i] == '.') {
                break;
            }

            root_field_name += hightlight_item.name[i];
        }

        if(dst.count(root_field_name) != 0) {
            // skip if parent "foo" has already has been copied over in e.g. foo.bar, foo.baz
            continue;
        }

        // root field name might not exist if object has primitive field values with "."s in the name
        if(src.count(root_field_name) != 0) {
            // copy whole sub-object
            dst[root_field_name] = src[root_field_name];
        } else if(src.count(hightlight_item.name) != 0) {
            dst[hightlight_item.name] = src[hightlight_item.name];
        }
    }
}

void Collection::process_search_field_weights(const std::vector<search_field_t>& search_fields,
                                              std::vector<uint32_t>& query_by_weights,
                                              std::vector<search_field_t>& weighted_search_fields) const {
    const bool weights_given = !query_by_weights.empty();

    // weights, if given, must be in desc order
    bool weights_in_desc_order = true;
    bool weights_under_max = true;

    for(size_t i = 0; i < search_fields.size(); i++) {
        if(!weights_given) {
            size_t weight = std::max<int>(0, (int(Index::FIELD_MAX_WEIGHT) - i));
            query_by_weights.push_back(weight);
            auto wsearch_field = search_fields[i];
            wsearch_field.weight = weight;
            weighted_search_fields.push_back(wsearch_field);
        } else {
            // check if weights are already sorted
            auto prev_weight = (i == 0) ? query_by_weights[0] : query_by_weights[i-1];
            weights_in_desc_order = weights_in_desc_order && (query_by_weights[i] <= prev_weight);
            weights_under_max = weights_under_max && (query_by_weights[i] <= Index::FIELD_MAX_WEIGHT);
        }
    }

    if(weights_given && (!weights_in_desc_order || !weights_under_max)) {
        // ensure that search fields are sorted on their corresponding weight
        std::vector<std::pair<size_t, size_t>> field_index_and_weights;

        for(size_t i=0; i < search_fields.size(); i++) {
            field_index_and_weights.emplace_back(i, search_fields[i].weight);
        }

        std::sort(field_index_and_weights.begin(), field_index_and_weights.end(), [](const auto& a, const auto& b) {
            return a.second > b.second;
        });

        for(size_t i = 0; i < field_index_and_weights.size(); i++) {
            const auto& index_weight = field_index_and_weights[i];

            // we have to also normalize weights to 0 to Index::FIELD_MAX_WEIGHT range.
            if(i == 0) {
                query_by_weights[i] = Index::FIELD_MAX_WEIGHT;
            } else {
                auto curr_weight = field_index_and_weights[i].second;
                auto prev_weight = field_index_and_weights[i-1].second;

                if(curr_weight == prev_weight) {
                    query_by_weights[i] = query_by_weights[i-1];
                } else {
                    // bound to be lesser than prev_weight since weights have been sorted desc
                    uint32_t bounded_weight = std::max(0, int(query_by_weights[i-1]) - 1);
                    query_by_weights[i] = bounded_weight;
                }
            }

            const auto& search_field = search_fields[index_weight.first];
            const auto weight = query_by_weights[i];
            const size_t orig_index = index_weight.first;
            auto wsearch_field = search_fields[orig_index];
            wsearch_field.weight = weight;
            weighted_search_fields.push_back(wsearch_field);
        }
    }

    if(weighted_search_fields.empty()) {
        for(size_t i=0; i < search_fields.size(); i++) {
            weighted_search_fields.push_back(search_fields[i]);
        }
    }
}

// lsb_offset is zero-based and inclusive
uint64_t Collection::extract_bits(uint64_t value, unsigned lsb_offset, unsigned n) {
    const uint64_t max_n = CHAR_BIT * sizeof(uint64_t);
    if (lsb_offset >= max_n) {
        return 0;
    }
    value >>= lsb_offset;
    if (n >= max_n) {
        return value;
    }
    const uint64_t mask = ((uint64_t(1)) << n) - 1; /* n '1's */
    return value & mask;
}

void Collection::populate_text_match_info(nlohmann::json& info, uint64_t match_score,
                                          const text_match_type_t match_type) const {

    // MAX_SCORE
    // [ sign | tokens_matched | max_field_score | max_field_weight | num_matching_fields ]
    // [   1  |        4       |        48       |       8          |         3           ]  (64 bits)

    // MAX_WEIGHT
    // [ sign | tokens_matched | max_field_weight | max_field_score  | num_matching_fields ]
    // [   1  |        4       |        8         |      48          |         3           ]  (64 bits)

    info["score"] = std::to_string(match_score);

    if(match_type == max_score) {
        info["tokens_matched"] = extract_bits(match_score, 59, 4);
        info["best_field_score"] = std::to_string(extract_bits(match_score, 11, 48));
        info["best_field_weight"] = extract_bits(match_score, 3, 8);
        info["fields_matched"] = extract_bits(match_score, 0, 3);
    } else {
        info["tokens_matched"] = extract_bits(match_score, 59, 4);
        info["best_field_weight"] = extract_bits(match_score, 51, 8);
        info["best_field_score"] = std::to_string(extract_bits(match_score, 3, 48));
        info["fields_matched"] = extract_bits(match_score, 0, 3);
    }
}

void Collection::process_highlight_fields(const std::vector<search_field_t>& search_fields,
                                          const std::vector<std::string>& raw_search_fields,
                                          const tsl::htrie_set<char>& include_fields,
                                          const tsl::htrie_set<char>& exclude_fields,
                                          const std::vector<std::string>& highlight_field_names,
                                          const std::vector<std::string>& highlight_full_field_names,
                                          const std::vector<enable_t>& infixes,
                                          std::vector<std::string>& q_tokens,
                                          const tsl::htrie_map<char, token_leaf>& qtoken_set,
                                          std::vector<highlight_field_t>& highlight_items) const {

    // identify full highlight fields
    spp::sparse_hash_set<std::string> fields_highlighted_fully_set;
    std::vector<std::string> fields_highlighted_fully_expanded;
    for(const std::string& highlight_full_field: highlight_full_field_names) {
        extract_field_name(highlight_full_field, search_schema, fields_highlighted_fully_expanded, true, enable_nested_fields);
    }

    for(std::string & highlight_full_field: fields_highlighted_fully_expanded) {
        fields_highlighted_fully_set.insert(highlight_full_field);
    }

    // identify infix enabled fields
    spp::sparse_hash_set<std::string> fields_infixed_set;

    for(size_t i = 0; i < search_fields.size(); i++) {
        const auto& field_name = search_fields[i].name;

        enable_t field_infix = search_fields[i].infix;
        if(field_infix != off) {
            fields_infixed_set.insert(field_name);
        }
    }

    if(highlight_field_names.empty()) {
        std::vector<std::string> highlight_field_names_expanded;

        for(size_t i = 0; i < raw_search_fields.size(); i++) {
            extract_field_name(raw_search_fields[i], search_schema, highlight_field_names_expanded, false, enable_nested_fields);
        }

        for(size_t i = 0; i < highlight_field_names_expanded.size(); i++) {
            const auto& field_name = highlight_field_names_expanded[i];
            if(exclude_fields.count(field_name) != 0) {
                // should not pick excluded field for highlighting (only for implicit highlighting)
                continue;
            }

            if(!include_fields.empty() && include_fields.count(field_name) == 0) {
                // if include fields have been specified, use that as allow list
                continue;
            }

            bool fully_highlighted = (fields_highlighted_fully_set.count(field_name) != 0);
            bool infixed = (fields_infixed_set.count(field_name) != 0);
            auto schema_it = search_schema.find(field_name);
            bool is_string = (schema_it != search_schema.end()) && schema_it->is_string();

            highlight_items.emplace_back(field_name, fully_highlighted, infixed, is_string);
        }
    } else {
        std::vector<std::string> highlight_field_names_expanded;
        for(size_t i = 0; i < highlight_field_names.size(); i++) {
            extract_field_name(highlight_field_names[i], search_schema, highlight_field_names_expanded, false, enable_nested_fields);
        }

        for(size_t i = 0; i < highlight_field_names_expanded.size(); i++) {
            const auto& highlight_field_name = highlight_field_names_expanded[i];
            auto schema_it = search_schema.find(highlight_field_name);
            if(schema_it == search_schema.end()) {
                // ignore fields not part of schema
                continue;
            }
            bool fully_highlighted = (fields_highlighted_fully_set.count(highlight_field_name) != 0);
            bool infixed = (fields_infixed_set.count(highlight_field_name) != 0);
            bool is_string = schema_it->is_string();
            highlight_items.emplace_back(highlight_field_name, fully_highlighted, infixed, is_string);
        }
    }

    std::string qtoken;
    for(auto it = qtoken_set.begin(); it != qtoken_set.end(); ++it) {
        it.key(qtoken);

        for(auto& highlight_item: highlight_items) {
            if(!highlight_item.is_string) {
                continue;
            }

            const auto& field_name = highlight_item.name;
            art_leaf* leaf = index->get_token_leaf(field_name, (const unsigned char*) qtoken.c_str(), qtoken.size()+1);
            if(leaf) {
                highlight_item.qtoken_leaves.insert(qtoken,
                    token_leaf(leaf, it.value().root_len, it.value().num_typos, it.value().is_prefix)
                );
            }
        }
    }

    // We will also add tokens from the query if they are not already added.
    // This helps handle highlighting of tokens which were dropped from the query to return results.
    for(auto& q_token: q_tokens) {
        if(qtoken_set.find(q_token) == qtoken_set.end()) {
            for(auto& highlight_item: highlight_items) {
                if(!highlight_item.is_string) {
                    continue;
                }
                const auto& field_name = highlight_item.name;
                art_leaf* leaf = index->get_token_leaf(field_name, (const unsigned char*) q_token.c_str(), q_token.size()+1);
                if(leaf) {
                    highlight_item.qtoken_leaves.insert(q_token, token_leaf(leaf, q_token.size(), 0, false));
                }
            }
        }
    }
}

void Collection::process_filter_overrides(std::vector<const override_t*>& filter_overrides,
                                          std::vector<std::string>& q_include_tokens,
                                          token_ordering token_order,
                                          filter_node_t*& filter_tree_root,
                                          std::vector<std::pair<uint32_t, uint32_t>>& included_ids,
                                          std::vector<uint32_t>& excluded_ids,
                                          nlohmann::json& override_metadata) const {

    std::vector<const override_t*> matched_dynamic_overrides;
    index->process_filter_overrides(filter_overrides, q_include_tokens, token_order,
                                    filter_tree_root, matched_dynamic_overrides, override_metadata);

    // we will check the dynamic overrides to see if they also have include/exclude
    std::set<uint32_t> excluded_set;

    for(auto matched_dynamic_override: matched_dynamic_overrides) {
        for(const auto& hit: matched_dynamic_override->drop_hits) {
            Option<uint32_t> seq_id_op = doc_id_to_seq_id(hit.doc_id);
            if(seq_id_op.ok()) {
                excluded_ids.push_back(seq_id_op.get());
                excluded_set.insert(seq_id_op.get());
            }
        }

        for(const auto& hit: matched_dynamic_override->add_hits) {
            Option<uint32_t> seq_id_op = doc_id_to_seq_id(hit.doc_id);
            if(!seq_id_op.ok()) {
                continue;
            }
            uint32_t seq_id = seq_id_op.get();
            bool excluded = (excluded_set.count(seq_id) != 0);
            if(!excluded) {
                included_ids.emplace_back(seq_id, hit.position);
            }
        }
    }
}

void Collection::parse_search_query(const std::string &query, std::vector<std::string>& q_include_tokens,
                                    std::vector<std::vector<std::string>>& q_exclude_tokens,
                                    std::vector<std::vector<std::string>>& q_phrases,
                                    const std::string& locale, const bool already_segmented, const std::string& stopwords_set) const {
    if(query == "*") {
        q_exclude_tokens = {};
        q_include_tokens = {query};
    } else {
        std::vector<std::string> tokens;
        stopword_struct_t stopwordStruct;
        if(!stopwords_set.empty()) {
            const auto &stopword_op = StopwordsManager::get_instance().get_stopword(stopwords_set, stopwordStruct);
            if (!stopword_op.ok()) {
                LOG(ERROR) << stopword_op.error();
                LOG(ERROR) << "Error fetching stopword_list for stopword " << stopwords_set;
            }
        }

        if(already_segmented) {
            StringUtils::split(query, tokens, " ");
        } else {
            std::vector<char> custom_symbols = symbols_to_index;
            custom_symbols.push_back('-');
            custom_symbols.push_back('"');

            Tokenizer(query, true, false, locale, custom_symbols, token_separators).tokenize(tokens);
        }

        for (const auto val: stopwordStruct.stopwords) {
            tokens.erase(std::remove(tokens.begin(), tokens.end(), val), tokens.end());
        }

        bool exclude_operator_prior = false;
        bool phrase_search_op_prior = false;
        std::vector<std::string> phrase;

        auto symbols_to_index_has_minus =
                std::find(symbols_to_index.begin(), symbols_to_index.end(), '-') != symbols_to_index.end();

        for(auto& token: tokens) {
            bool end_of_phrase = false;

            if(token == "-" && !symbols_to_index_has_minus) {
                continue;
            } else if(token[0] == '-' && !symbols_to_index_has_minus) {
                exclude_operator_prior = true;
                token = token.substr(1);
            }

            if(token[0] == '"' && token.size() > 1) {
                phrase_search_op_prior = true;
                token = token.substr(1);
            }

            if(!token.empty() && (token.back() == '"' || (token[0] == '"' && token.size() == 1))) {
                if(phrase_search_op_prior) {
                    // handles single token phrase and a phrase with padded space, like: "some query "
                    end_of_phrase = true;
                    token = token.substr(0, token.size()-1);
                } else if(token[0] == '"' && token.size() == 1) {
                    // handles front padded phrase query, e.g. " some query"
                    phrase_search_op_prior = true;
                }
            }

            // retokenize using collection config (handles hyphens being part of the query)
            std::vector<std::string> sub_tokens;

            if(already_segmented) {
                StringUtils::split(token, sub_tokens, " ");
            } else {
                Tokenizer(token, true, false, locale, symbols_to_index, token_separators).tokenize(sub_tokens);
            }

            for(auto& sub_token: sub_tokens) {
                if(sub_token.size() > 100) {
                    sub_token.erase(100);
                }

                if(exclude_operator_prior) {
                    if(phrase_search_op_prior) {
                        phrase.push_back(sub_token);
                    } else {
                        q_exclude_tokens.push_back({sub_token});
                        exclude_operator_prior = false;
                    }
                } else if(phrase_search_op_prior) {
                    phrase.push_back(sub_token);
                } else {
                    q_include_tokens.push_back(sub_token);
                }
            }

            if(end_of_phrase && phrase_search_op_prior) {
                if(exclude_operator_prior) {
                    q_exclude_tokens.push_back(phrase);
                } else {
                    q_phrases.push_back(phrase);
                }

                phrase_search_op_prior = false;
                exclude_operator_prior = false;
                phrase.clear();
            }
        }

        if(!phrase.empty()) {
            if(exclude_operator_prior) {
                q_exclude_tokens.push_back(phrase);
            } else {
                q_include_tokens.insert(q_include_tokens.end(), phrase.begin(), phrase.end());
            }
        }

        if(q_include_tokens.empty()) {
            if(!stopwords_set.empty()) {
                //this can happen when all tokens in the include are stopwords
                q_include_tokens.emplace_back("##hrhdh##");
            } else {
                // this can happen if the only query token is an exclusion token
                q_include_tokens.emplace_back("*");
            }
        }
    }
}

void Collection::populate_result_kvs(Topster *topster, std::vector<std::vector<KV *>> &result_kvs,
                                const spp::sparse_hash_map<uint64_t, uint32_t>& groups_processed, 
                                const std::vector<sort_by>& sort_by_fields) {
    if(topster->distinct) {
        // we have to pick top-K groups
        Topster gtopster(topster->MAX_SIZE);

        int group_count_index = -1;
        int group_sort_order = 1;

        for(int i = 0; i < sort_by_fields.size(); ++i) {
            if(sort_by_fields[i].name == sort_field_const::group_found) {
                group_count_index = i;
                
                if(sort_by_fields[i].order == sort_field_const::asc) {
                    group_sort_order *= -1;
                } 

                break;
            }
        }

        for(auto& group_topster: topster->group_kv_map) {
            group_topster.second->sort();
            if(group_topster.second->size != 0) {
                KV* kv_head = group_topster.second->getKV(0);
                
                if(group_count_index >= 0) {
                    const auto& itr = groups_processed.find(kv_head->distinct_key);
                    if(itr != groups_processed.end()) {
                        kv_head->scores[group_count_index] = itr->second * group_sort_order;
                    }
                }
                gtopster.add(kv_head);
            }
        }

        gtopster.sort();

        for(size_t i = 0; i < gtopster.size; i++) {
            KV* kv = gtopster.getKV(i);
            const std::vector<KV*> group_kvs(
                topster->group_kv_map[kv->distinct_key]->kvs,
                topster->group_kv_map[kv->distinct_key]->kvs+topster->group_kv_map[kv->distinct_key]->size
            );
            result_kvs.emplace_back(group_kvs);
        }
    } else {
        for(uint32_t t = 0; t < topster->size; t++) {
            KV* kv = topster->getKV(t);
            result_kvs.push_back({kv});
        }
    }
}

Option<bool> Collection::get_filter_ids(const std::string& filter_query, filter_result_t& filter_result) const {
    std::shared_lock lock(mutex);

    const std::string doc_id_prefix = std::to_string(collection_id) + "_" + DOC_ID_PREFIX + "_";
    filter_node_t* filter_tree_root = nullptr;
    Option<bool> filter_op = filter::parse_filter_query(filter_query, search_schema,
                                                        store, doc_id_prefix, filter_tree_root);
    std::unique_ptr<filter_node_t> filter_tree_root_guard(filter_tree_root);

    if(!filter_op.ok()) {
        return filter_op;
    }

    return index->do_filtering_with_lock(filter_tree_root, filter_result, name);
}

Option<bool> Collection::get_related_ids(const std::string& ref_field_name, const uint32_t& seq_id,
                                               std::vector<uint32_t>& result) const {
    return index->get_related_ids_with_lock(name, ref_field_name, seq_id, result);
}

Option<bool> Collection::get_object_array_related_id(const std::string& ref_field_name,
                                                     const uint32_t& seq_id, const uint32_t& object_index,
                                                     uint32_t& result) const {
    return index->get_object_array_related_id(name, ref_field_name, seq_id, object_index, result);
}

Option<bool> Collection::get_reference_filter_ids(const std::string & filter_query,
                                                  filter_result_t& filter_result,
                                                  const std::string& reference_field_name) const {
    std::shared_lock lock(mutex);

    const std::string doc_id_prefix = std::to_string(collection_id) + "_" + DOC_ID_PREFIX + "_";
    filter_node_t* filter_tree_root = nullptr;
    Option<bool> parse_op = filter::parse_filter_query(filter_query, search_schema,
                                                       store, doc_id_prefix, filter_tree_root);
    std::unique_ptr<filter_node_t> filter_tree_root_guard(filter_tree_root);

    if(!parse_op.ok()) {
        return parse_op;
    }

    return index->do_reference_filtering_with_lock(filter_tree_root, filter_result, name, reference_field_name);
}

bool Collection::facet_value_to_string(const facet &a_facet, const facet_count_t &facet_count,
                                       const nlohmann::json &document, std::string &value) const {

    if(document.count(a_facet.field_name) == 0) {
        // check for field exists
        if(search_schema.at(a_facet.field_name).optional) {
            return false;
        }

        LOG(ERROR) << "Could not find field " << a_facet.field_name << " in document during faceting.";
        LOG(ERROR) << "Facet field type: " << search_schema.at(a_facet.field_name).type;
        LOG(ERROR) << "Actual document: " << document;
        return false;
    }

    if(search_schema.at(a_facet.field_name).is_array()) {
        size_t array_sz = document[a_facet.field_name].size();
        if(facet_count.array_pos >= array_sz) {
            LOG(ERROR) << "Facet field array size " << array_sz << " lesser than array pos " <<  facet_count.array_pos
                       << " for facet field " << a_facet.field_name;
            LOG(ERROR) << "Facet field type: " << search_schema.at(a_facet.field_name).type;
            LOG(ERROR) << "Actual document: " << document;
            return false;
        }
    }

    if(search_schema.at(a_facet.field_name).type == field_types::STRING) {
        value =  document[a_facet.field_name];
    } else if(search_schema.at(a_facet.field_name).type == field_types::STRING_ARRAY) {
        value = document[a_facet.field_name][facet_count.array_pos];
    } else if(search_schema.at(a_facet.field_name).type == field_types::INT32) {
        int32_t raw_val = document[a_facet.field_name].get<int32_t>();
        value = std::to_string(raw_val);
    } else if(search_schema.at(a_facet.field_name).type == field_types::INT32_ARRAY) {
        int32_t raw_val = document[a_facet.field_name][facet_count.array_pos].get<int32_t>();
        value = std::to_string(raw_val);
    } else if(search_schema.at(a_facet.field_name).type == field_types::INT64) {
        int64_t raw_val = document[a_facet.field_name].get<int64_t>();
        value = std::to_string(raw_val);
    } else if(search_schema.at(a_facet.field_name).type == field_types::INT64_ARRAY) {
        int64_t raw_val = document[a_facet.field_name][facet_count.array_pos].get<int64_t>();
        value = std::to_string(raw_val);
    } else if(search_schema.at(a_facet.field_name).type == field_types::FLOAT) {
        float raw_val = document[a_facet.field_name].get<float>();
        value = StringUtils::float_to_str(raw_val);
    } else if(search_schema.at(a_facet.field_name).type == field_types::FLOAT_ARRAY) {
        float raw_val = document[a_facet.field_name][facet_count.array_pos].get<float>();
        value = StringUtils::float_to_str(raw_val);
    } else if(search_schema.at(a_facet.field_name).type == field_types::BOOL) {
        value = std::to_string(document[a_facet.field_name].get<bool>());
        value = (value == "1") ? "true" : "false";
    } else if(search_schema.at(a_facet.field_name).type == field_types::BOOL_ARRAY) {
        value = std::to_string(document[a_facet.field_name][facet_count.array_pos].get<bool>());
        value = (value == "1") ? "true" : "false";
    }

    return true;
}

nlohmann::json Collection::get_facet_parent(const std::string& facet_field_name, const nlohmann::json& document) const {
    std::vector<std::string> tokens;
    StringUtils::split(facet_field_name, tokens, ".");
    std::vector<nlohmann::json> level_docs;

    auto doc = document[tokens[0]];
    level_docs.push_back(doc);
    for(auto i = 1; i < tokens.size()-1; ++i) { //just to ignore last token which uis our facet field
        if(doc.contains(tokens[i])) {
            doc = doc[tokens[i]];
            level_docs.push_back(doc);
        } else {
            LOG(ERROR) << tokens[i] << " not found in document";
        }
    }
    bool parent_found = false;
    for(auto i = level_docs.size()-1; i >0; --i) {
        if(level_docs[i].size() > 1) {
            doc = level_docs[i];
            parent_found = true;
            break;
        }
    }

    if(!parent_found) {
        doc = level_docs[0]; //return the top most root
    }
    return doc;
}

bool Collection::is_nested_array(const nlohmann::json& obj, std::vector<std::string> path_parts, size_t part_i) const {
    auto child_it = obj.find(path_parts[part_i]);
    if(child_it == obj.end()) {
        return false;
    }

    if(child_it.value().is_array() && !child_it.value().empty() && child_it.value().at(0).is_object()) {
        return true;
    }

    if(part_i+1 == path_parts.size()) {
        return false;
    }

    return is_nested_array(child_it.value(), path_parts, part_i+1);
}

void Collection::highlight_result(const std::string& raw_query, const field &search_field,
                                  const size_t search_field_index,
                                  const tsl::htrie_map<char, token_leaf>& qtoken_leaves,
                                  const KV* field_order_kv, const nlohmann::json & document,
                                  nlohmann::json& highlight_doc,
                                  StringUtils & string_utils,
                                  const size_t snippet_threshold,
                                  const size_t highlight_affix_num_tokens,
                                  bool highlight_fully,
                                  bool is_infix_search,
                                  const std::string& highlight_start_tag,
                                  const std::string& highlight_end_tag,
                                  const uint8_t* index_symbols,
                                  highlight_t& highlight,
                                  bool& found_highlight,
                                  bool& found_full_highlight) const {

    if(raw_query == "*") {
        return;
    }

    tsl::htrie_set<char> matched_tokens;

    bool use_word_tokenizer = Tokenizer::has_word_tokenizer(search_field.locale);
    bool normalise = !use_word_tokenizer;

    std::vector<std::string> raw_query_tokens;
    Tokenizer(raw_query, normalise, false, search_field.locale, symbols_to_index, token_separators).tokenize(raw_query_tokens);

    if(raw_query_tokens.empty()) {
        return ;
    }

    bool flat_field = highlight_doc.contains(search_field.name);
    std::vector<std::string> path_parts;
    if(enable_nested_fields && !flat_field) {
        StringUtils::split(search_field.name, path_parts, ".");
    } else {
        path_parts = {search_field.name};
    }

    const std::string& last_raw_q_token = raw_query_tokens.back();
    size_t prefix_token_num_chars = StringUtils::get_num_chars(last_raw_q_token);

    std::set<std::string> last_full_q_tokens;
    std::vector<match_index_t> match_indices;

    if(is_infix_search) {
        // could be an optional field
        if(document.contains(search_field.name)) {
            size_t array_len = 1;
            bool field_is_array = document[search_field.name].is_array();
            if(field_is_array) {
                array_len = document[search_field.name].size();
            }

            const std::vector<token_positions_t> empty_offsets;

            for(size_t i = 0; i < array_len; i++) {
                std::string text = field_is_array ? document[search_field.name][i] : document[search_field.name];
                StringUtils::tolowercase(text);
                if(text.size() < 100 && text.find(raw_query_tokens.front()) != std::string::npos) {
                    const Match & this_match = Match(field_order_kv->key, empty_offsets, false, false);
                    uint64_t this_match_score = this_match.get_match_score(0, 1);
                    match_indices.emplace_back(this_match, this_match_score, i);
                }
            }
        }

    } else {

        /*std::string qtok_buff;
        for(auto it = qtoken_leaves.begin(); it != qtoken_leaves.end(); ++it) {
            it.key(qtok_buff);
            LOG(INFO) << "Token: " << qtok_buff << ", root_len: " << it.value().root_len;
        }*/

        if(!qtoken_leaves.empty()) {
            std::vector<void*> posting_lists;
            for(auto token_leaf: qtoken_leaves) {
                posting_lists.push_back(token_leaf.leaf->values);
            }

            std::map<size_t, std::vector<token_positions_t>> array_token_positions;
            posting_t::get_array_token_positions(field_order_kv->key, posting_lists, array_token_positions);

            for(const auto& kv: array_token_positions) {
                const std::vector<token_positions_t>& token_positions = kv.second;
                size_t array_index = kv.first;

                if(token_positions.empty()) {
                    continue;
                }

                const Match & this_match = Match(field_order_kv->key, token_positions, true, true);
                uint64_t this_match_score = this_match.get_match_score(1, token_positions.size());
                match_indices.emplace_back(this_match, this_match_score, array_index);

                /*LOG(INFO) << "doc_id: " << document["id"] << ", search_field: " << search_field.name
                          << ", words_present: " << size_t(this_match.words_present)
                          << ", match_score: " << this_match_score
                          << ", match.distance: " << size_t(this_match.distance);*/
            }
        }
    }

    const size_t max_array_matches = std::min((size_t)MAX_ARRAY_MATCHES, match_indices.size());
    std::partial_sort(match_indices.begin(), match_indices.begin()+max_array_matches, match_indices.end());

    highlight_nested_field(highlight_doc, highlight_doc, path_parts, 0, false, -1,
                           [&](nlohmann::json& h_obj, bool is_arr_obj_ele, int array_i) {
        if(h_obj.is_object()) {
            return ;
        } else if(!h_obj.is_string()) {
            auto val_back = h_obj;
            h_obj = nlohmann::json::object();
            h_obj["snippet"] = to_string(val_back);
            h_obj["matched_tokens"] = nlohmann::json::array();
            if(highlight_fully) {
                h_obj["value"] = val_back;
            }
            return ;
        }

        int matched_index = -1;

        if(!is_arr_obj_ele) {
            // Since we will iterate on both matching and non-matching array elements for highlighting,
            // we need to check if `array_i`exists within match_indices vec.

            for (size_t match_index = 0; match_index < match_indices.size(); match_index++) {
                if (match_indices[match_index].index == array_i) {
                    matched_index = match_index;
                    break;
                }
            }

            if(matched_index == -1) {
                // If an element does not belong to an array of object field and also does not have a matching index
                // we know that there cannot be any matching tokens for highlighting
                std::string text = h_obj.get<std::string>();
                h_obj = nlohmann::json::object();
                h_obj["snippet"] = text;
                h_obj["matched_tokens"] = nlohmann::json::array();
                if(highlight_fully) {
                    h_obj["value"] = text;
                }
                return ;
            }

            std::sort(match_indices[matched_index].match.offsets.begin(),
                      match_indices[matched_index].match.offsets.end());
        } else {
            // array of object element indices will not match indexed offsets, so we will use dummy match
            // the highlighting logic will ignore this and try to do exhaustive highlighting (look at all tokens)
            match_indices.clear();
            match_indices.push_back(match_index_t(Match(), 0, 0));
            matched_index = 0;
        }

        const auto& match_index = match_indices[matched_index];

        size_t last_valid_offset = 0;
        int last_valid_offset_index = -1;

        for(size_t match_offset_i = 0; match_offset_i < match_index.match.offsets.size(); match_offset_i++) {
            const auto& token_offset = match_index.match.offsets[match_offset_i];
            if(token_offset.offset != MAX_DISPLACEMENT) {
                last_valid_offset = token_offset.offset;
                last_valid_offset_index = match_offset_i;
            } else {
                break;
            }
        }

        highlight_t array_highlight = highlight;
        std::string text = h_obj.get<std::string>();
        h_obj = nlohmann::json::object();

        handle_highlight_text(text, normalise, search_field, symbols_to_index,
                              token_separators, array_highlight, string_utils, use_word_tokenizer,
                              highlight_affix_num_tokens,
                              qtoken_leaves, last_valid_offset_index,
                              prefix_token_num_chars,
                              highlight_fully, snippet_threshold, is_infix_search,
                              raw_query_tokens,
                              last_valid_offset, highlight_start_tag, highlight_end_tag,
                              index_symbols, match_index);


        if(array_highlight.snippets.empty() && array_highlight.values.empty()) {
            h_obj["snippet"] = text;
            h_obj["matched_tokens"] = nlohmann::json::array();
        }

        if(!array_highlight.snippets.empty()) {
            found_highlight = found_highlight || true;

            h_obj["snippet"] = array_highlight.snippets[0];
            h_obj["matched_tokens"] = nlohmann::json::array();

            for(auto& token_vec: array_highlight.matched_tokens) {
                for(auto& token: token_vec) {
                    h_obj["matched_tokens"].push_back(token);
                }
            }
        }

        if(!array_highlight.values.empty()) {
            h_obj["value"] = array_highlight.values[0];;
            found_full_highlight = found_full_highlight || true;
        } else if(highlight_fully) {
            h_obj["value"] = text;
        }
    });

    if(!flat_field) {
        return;
    }

    if(!search_field.is_string()) {
        return ;
    }

    if(!is_infix_search && qtoken_leaves.empty()) {
        // none of the tokens from the query were found on this field
        return ;
    }

    if(match_indices.empty()) {
        return ;
    }

    for(size_t array_i = 0; array_i < max_array_matches; array_i++) {
        std::sort(match_indices[array_i].match.offsets.begin(), match_indices[array_i].match.offsets.end());
        const auto& match_index = match_indices[array_i];
        const Match& match = match_index.match;

        size_t last_valid_offset = 0;
        int last_valid_offset_index = -1;

        for(size_t match_offset_index = 0; match_offset_index < match.offsets.size(); match_offset_index++) {
            const auto& token_offset = match.offsets[match_offset_index];
            if(token_offset.offset != MAX_DISPLACEMENT) {
                last_valid_offset = token_offset.offset;
                last_valid_offset_index = match_offset_index;
            } else {
                break;
            }
        }

        if(!document.contains(search_field.name)) {
            // could be an optional field
            continue;
        }

        /*LOG(INFO) << "field: " << document[search_field.name] << ", id: " << field_order_kv->key
                  << ", index: " << match_index.index;*/

        std::string text;

        if(search_field.type == field_types::STRING) {
            text = document[search_field.name];
        } else {
            // since we try to do manual prefix matching on the first array value, we have to check for an empty array
            if(!document[search_field.name].is_array() ||
                match_index.index >= document[search_field.name].size()) {
                continue;
            }

            text = document[search_field.name][match_index.index];
        }

        handle_highlight_text(text, normalise, search_field, symbols_to_index, token_separators,
                              highlight, string_utils, use_word_tokenizer, highlight_affix_num_tokens,
                              qtoken_leaves, last_valid_offset_index, prefix_token_num_chars,
                              highlight_fully, snippet_threshold, is_infix_search, raw_query_tokens,
                              last_valid_offset, highlight_start_tag, highlight_end_tag,
                              index_symbols, match_index);

        if(!highlight.snippets.empty()) {
            found_highlight = found_highlight || true;
            for(auto& token_vec: highlight.matched_tokens) {
                for(auto& token: token_vec) {
                    matched_tokens.insert(token);
                }
            }
        }

        if(!highlight.values.empty()) {
            found_full_highlight = found_full_highlight || true;
        }
    }

    highlight.field = search_field.name;
    highlight.field_index = search_field_index;

    if(!match_indices.empty()) {
        highlight.match_score = match_indices[0].match_score;
    }
}

bool Collection::handle_highlight_text(std::string& text, bool normalise, const field &search_field,
                           const std::vector<char>& symbols_to_index, const std::vector<char>& token_separators,
                           highlight_t& highlight, StringUtils & string_utils, bool use_word_tokenizer,
                           const size_t highlight_affix_num_tokens,
                           const tsl::htrie_map<char, token_leaf>& qtoken_leaves, int last_valid_offset_index,
                           const size_t prefix_token_num_chars, bool highlight_fully, const size_t snippet_threshold,
                           bool is_infix_search, std::vector<std::string>& raw_query_tokens, size_t last_valid_offset,
                           const std::string& highlight_start_tag, const std::string& highlight_end_tag,
                           const uint8_t* index_symbols, const match_index_t& match_index) const {

    const Match& match = match_index.match;

    Tokenizer tokenizer(text, normalise, false, search_field.locale, symbols_to_index, token_separators);

    // word tokenizer is a secondary tokenizer used for specific languages that requires transliteration
    Tokenizer word_tokenizer("", true, false, search_field.locale, symbols_to_index, token_separators);

    if(search_field.locale == "ko") {
        text = string_utils.unicode_nfkd(text);
    }

    // need an ordered map here to ensure that it is ordered by the key (start offset)
    std::map<size_t, size_t> token_offsets;

    int match_offset_index = 0;
    std::string raw_token;
    std::set<std::string> token_hits;  // used to identify repeating tokens
    size_t raw_token_index = 0, tok_start = 0, tok_end = 0;

    // based on `highlight_affix_num_tokens`
    size_t snippet_start_offset = 0, snippet_end_offset = (text.empty() ? 0 : text.size() - 1);

    // window used to locate the starting offset for snippet on the text
    std::list<size_t> snippet_start_window;

    highlight.matched_tokens.emplace_back();
    std::vector<std::string>& matched_tokens = highlight.matched_tokens.back();
    bool found_first_match = false;

    size_t text_len = Tokenizer::is_ascii_char(text[0]) ? text.size() : StringUtils::get_num_chars(text);

    while(tokenizer.next(raw_token, raw_token_index, tok_start, tok_end)) {
        if(use_word_tokenizer) {
            bool found_token = word_tokenizer.tokenize(raw_token);
            if(!found_token) {
                tokenizer.decr_token_counter();
                continue;
            }
        }

        if(!found_first_match) {
            if(snippet_start_window.size() == highlight_affix_num_tokens + 1) {
                snippet_start_window.pop_front();
            }

            snippet_start_window.push_back(tok_start);
        }

        bool token_already_found = (token_hits.find(raw_token) != token_hits.end());
        auto qtoken_it = qtoken_leaves.find(raw_token);

        // ensures that the `snippet_start_offset` is always from a matched token, and not from query suggestion
        bool match_offset_found = (found_first_match && token_already_found) ||
                                  (match_offset_index <= last_valid_offset_index &&
                                   match.offsets[match_offset_index].offset == raw_token_index);

        // Token might not appear in the best matched window, which is limited to a size of 10.
        // If field is marked to be highlighted fully, or field length exceeds snippet_threshold, we will
        // locate all tokens that appear in the query / query candidates
        bool raw_token_found = !match_offset_found && (highlight_fully || text_len < snippet_threshold * 6) &&
                                qtoken_leaves.find(raw_token) != qtoken_leaves.end();

        if (match_offset_found || raw_token_found) {
            if(qtoken_it != qtoken_leaves.end() && qtoken_it.value().is_prefix &&
               qtoken_it.value().root_len < raw_token.size()) {
                // need to ensure that only the prefix portion is highlighted
                // if length diff is within 2, we still might not want to highlight partially in some cases
                // e.g. "samsng" vs "samsung" -> full highlight is preferred, unless it's a full prefix match

                size_t k = tok_start;
                size_t num_letters = 0, prefix_letters = 0, prefix_end = tok_start;

                // group unicode code points and calculate number of actual characters
                while(k <= tok_end) {
                    k++;

                    if(tokenizer.should_skip_char(text[k])) {
                        // used to handle special characters inside a tokenized word, e.g. `foo-bar`
                        continue;
                    }

                    if ((text[k] & 0xC0) == 0x80) k++;
                    if ((text[k] & 0xC0) == 0x80) k++;
                    if ((text[k] & 0xC0) == 0x80) k++;

                    num_letters++;

                    if(num_letters <= prefix_token_num_chars) {
                        prefix_letters++;
                    }

                    if(num_letters == prefix_token_num_chars) {
                        prefix_end = k - 1;
                    }
                }

                size_t char_diff = num_letters - prefix_letters;
                auto new_tok_end = (char_diff <= 2 && qtoken_it.value().num_typos != 0) ? tok_end : prefix_end;
                token_offsets.emplace(tok_start, new_tok_end);
            } else {
                token_offsets.emplace(tok_start, tok_end);
            }

            token_hits.insert(raw_token);

            if(match_offset_found) {
                // to skip over duplicate tokens in the query
                do {
                    match_offset_index++;
                } while(match_offset_index <= last_valid_offset_index &&
                        match.offsets[match_offset_index - 1].offset == match.offsets[match_offset_index].offset);

                if(!found_first_match) {
                    snippet_start_offset = snippet_start_window.front();
                }

                found_first_match = true;
            }
        } else if(is_infix_search && text.size() < 100 &&
                  raw_token.find(raw_query_tokens.front()) != std::string::npos) {
            token_offsets.emplace(tok_start, tok_end);
            token_hits.insert(raw_token);
        }

        if(raw_token_index >= last_valid_offset + highlight_affix_num_tokens) {
            // register end of highlight snippet
            if(snippet_end_offset == text.size() - 1) {
                snippet_end_offset = tok_end;
            }
        }

        // We can break early only if we have:
        // a) run out of matched indices
        // b) token_index exceeds the suffix tokens boundary
        // c) raw_token_index exceeds snippet threshold
        // d) highlight fully is not requested

        if(raw_token_index >= snippet_threshold &&
           match_offset_index > last_valid_offset_index &&
           raw_token_index >= last_valid_offset + highlight_affix_num_tokens &&
           !highlight_fully) {
            break;
        }
    }

    if(token_offsets.empty()) {
        return false;
    }

    if(raw_token_index <= snippet_threshold-1) {
        // fully highlight field whose token size is less than given snippet threshold
        snippet_start_offset = 0;
        snippet_end_offset = text.size() - 1;
    }

    // `token_offsets` has a list of ranges to target for highlighting
    // tokens from query might occur before actual snippet start offset: we skip that
    auto offset_it = token_offsets.begin();
    while(offset_it != token_offsets.end() && offset_it->first < snippet_start_offset) {
        offset_it++;
    }

    std::stringstream highlighted_text;
    highlight_text(highlight_start_tag, highlight_end_tag, text, token_offsets,
                   snippet_end_offset, matched_tokens, offset_it,
                   highlighted_text, index_symbols, snippet_start_offset);

    highlight.snippets.push_back(highlighted_text.str());
    if(search_field.type == field_types::STRING_ARRAY) {
        highlight.indices.push_back(match_index.index);
    }

    if(highlight_fully) {
        std::stringstream value_stream;
        offset_it = token_offsets.begin();
        std::vector<std::string> full_matched_tokens;
        highlight_text(highlight_start_tag, highlight_end_tag, text, token_offsets,
                       text.size()-1, full_matched_tokens, offset_it,
                       value_stream, index_symbols, 0);
        highlight.values.push_back(value_stream.str());
    }

    return true;
}

void Collection::highlight_text(const string& highlight_start_tag, const string& highlight_end_tag,
                                  const string& text,
                                  const std::map<size_t, size_t>& token_offsets,
                                  size_t snippet_end_offset, std::vector<std::string>& matched_tokens,
                                  std::map<size_t, size_t>::iterator& offset_it,
                                  std::stringstream& highlighted_text,
                                  const uint8_t* index_symbols,
                                  size_t snippet_start_offset) {

    while(snippet_start_offset <= snippet_end_offset) {
        if(offset_it != token_offsets.end()) {
            if (snippet_start_offset == offset_it->first) {
                highlighted_text << highlight_start_tag;

                auto end_offset = offset_it->second;

                // if a token ends with one or more puncutation chars, we should not highlight them
                for(int j = end_offset; j >= 0; j--) {
                    if(end_offset >= text.size()) {
                        // this should not happen unless we mess up unicode normalization
                        break;
                    }

                    if(!std::isalnum(text[j]) && Tokenizer::is_ascii_char(text[j]) &&
                        index_symbols[uint8_t(text[j])] != 1) {
                        end_offset--;
                    } else {
                        break;
                    }
                }

                size_t token_len = end_offset - snippet_start_offset + 1;
                const std::string& text_token = text.substr(snippet_start_offset, token_len);
                matched_tokens.push_back(text_token);

                for(size_t j = 0; j < token_len; j++) {
                    if((snippet_start_offset + j) >= text.size()) {
                        LOG(ERROR) << "??? snippet_start_offset: " << snippet_start_offset
                                  << ", offset_it->first: " << offset_it->first
                                  << ", offset_it->second: " << offset_it->second
                                  << ", end_offset: " << end_offset
                                  << ", j: " << j << ", token_len: " << token_len << ", text: " << text;
                        break;
                    }
                    highlighted_text << text[snippet_start_offset + j];
                }

                highlighted_text << highlight_end_tag;
                offset_it++;
                snippet_start_offset += token_len;
                continue;
            }
        }

        highlighted_text << text[snippet_start_offset];
        snippet_start_offset++;
    }
}

Option<nlohmann::json> Collection::get(const std::string & id) const {
    std::string seq_id_str;
    StoreStatus seq_id_status = store->get(get_doc_id_key(id), seq_id_str);

    if(seq_id_status == StoreStatus::NOT_FOUND) {
        return Option<nlohmann::json>(404, "Could not find a document with id: " + id);
    }

    if(seq_id_status == StoreStatus::ERROR) {
        return Option<nlohmann::json>(500, "Error while fetching the document.");
    }

    uint32_t seq_id = (uint32_t) std::stoul(seq_id_str);

    std::string parsed_document;
    StoreStatus doc_status = store->get(get_seq_id_key(seq_id), parsed_document);

    if(doc_status == StoreStatus::NOT_FOUND) {
        LOG(ERROR) << "Sequence ID exists, but document is missing for id: " << id;
        return Option<nlohmann::json>(404, "Could not find a document with id: " + id);
    }

    if(doc_status == StoreStatus::ERROR) {
        return Option<nlohmann::json>(500, "Error while fetching the document.");
    }

    nlohmann::json document;
    try {
        document = nlohmann::json::parse(parsed_document);
    } catch(...) {
        return Option<nlohmann::json>(500, "Error while parsing stored document.");
    }

    return Option<nlohmann::json>(document);
}

void Collection::remove_document(const nlohmann::json & document, const uint32_t seq_id, bool remove_from_store) {
    const std::string& id = document["id"];

    {
        std::unique_lock lock(mutex);

        index->remove(seq_id, document, {}, false);
        num_documents -= 1;
    }

    if(remove_from_store) {
        store->remove(get_doc_id_key(id));
        store->remove(get_seq_id_key(seq_id));
    }

    if (referenced_in.empty()) {
        return;
    }

    CollectionManager& collectionManager = CollectionManager::get_instance();
    // Cascade delete all the references.
    for (const auto &item: referenced_in) {
        auto ref_coll = collectionManager.get_collection(item.first);
        if (ref_coll != nullptr) {
            filter_result_t filter_result;
            ref_coll->get_filter_ids(item.second + ":=" + id, filter_result);
            for (uint32_t i = 0; i < filter_result.count; i++) {
                ref_coll->remove(std::to_string(filter_result.docs[i]));
            }
        }
    }
}

Option<std::string> Collection::remove(const std::string & id, const bool remove_from_store) {
    std::string seq_id_str;
    StoreStatus seq_id_status = store->get(get_doc_id_key(id), seq_id_str);

    if(seq_id_status == StoreStatus::NOT_FOUND) {
        return Option<std::string>(404, "Could not find a document with id: " + id);
    }

    if(seq_id_status == StoreStatus::ERROR) {
        return Option<std::string>(500, "Error while fetching the document.");
    }

    uint32_t seq_id = (uint32_t) std::stoul(seq_id_str);

    nlohmann::json document;
    auto get_doc_op = get_document_from_store(get_seq_id_key(seq_id), document);

    if(!get_doc_op.ok()) {
        if(get_doc_op.code() == 404) {
            LOG(ERROR) << "Sequence ID exists, but document is missing for id: " << id;
            return Option<std::string>(404, "Could not find a document with id: " + id);
        }

        return Option<std::string>(get_doc_op.code(), get_doc_op.error());
    }

    remove_document(document, seq_id, remove_from_store);
    return Option<std::string>(id);
}

Option<bool> Collection::remove_if_found(uint32_t seq_id, const bool remove_from_store) {
    nlohmann::json document;
    auto get_doc_op = get_document_from_store(get_seq_id_key(seq_id), document);

    if(!get_doc_op.ok()) {
        if(get_doc_op.code() == 404) {
            return Option<bool>(false);
        }

        return Option<bool>(500, "Error while fetching the document with seq id: " +
                                 std::to_string(seq_id));
    }

    remove_document(document, seq_id, remove_from_store);
    return Option<bool>(true);
}

Option<uint32_t> Collection::add_override(const override_t & override, bool write_to_store) {
    if(write_to_store) {
        bool inserted = store->insert(Collection::get_override_key(name, override.id), override.to_json().dump());
        if(!inserted) {
            return Option<uint32_t>(500, "Error while storing the override on disk.");
        }
    }

    std::unique_lock lock(mutex);

    if(overrides.count(override.id) != 0 && !overrides[override.id].rule.tags.empty()) {
        // remove existing tags
        for(auto& tag: overrides[override.id].rule.tags) {
            if(override_tags.count(tag) != 0) {
                override_tags[tag].erase(override.id);
            }
        }
    }

    overrides[override.id] = override;
    for(const auto& tag: override.rule.tags) {
        override_tags[tag].insert(override.id);
    }

    return Option<uint32_t>(200);
}

Option<uint32_t> Collection::remove_override(const std::string & id) {
    if(overrides.count(id) != 0) {
        bool removed = store->remove(Collection::get_override_key(name, id));
        if(!removed) {
            return Option<uint32_t>(500, "Error while deleting the override from disk.");
        }

        std::unique_lock lock(mutex);
        for(const auto& tag: overrides[id].rule.tags) {
            if(override_tags.count(tag) != 0) {
                override_tags[tag].erase(id);
            }
        }

        overrides.erase(id);

        return Option<uint32_t>(200);
    }

    return Option<uint32_t>(404, "Could not find that `id`.");
}

uint32_t Collection::get_seq_id_from_key(const std::string & key) {
    // last 4 bytes of the key would be the serialized version of the sequence id
    std::string serialized_seq_id = key.substr(key.length() - 4);
    return StringUtils::deserialize_uint32_t(serialized_seq_id);
}

std::string Collection::get_next_seq_id_key(const std::string & collection_name) {
    return std::string(COLLECTION_NEXT_SEQ_PREFIX) + "_" + collection_name;
}

std::string Collection::get_seq_id_key(uint32_t seq_id) const {
    // We can't simply do std::to_string() because we want to preserve the byte order.
    // & 0xFF masks all but the lowest eight bits.
    const std::string & serialized_id = StringUtils::serialize_uint32_t(seq_id);
    return get_seq_id_collection_prefix() + "_" + serialized_id;
}

std::string Collection::get_doc_id_key(const std::string & doc_id) const {
    return std::to_string(collection_id) + "_" + DOC_ID_PREFIX + "_" + doc_id;
}

std::string Collection::get_name() const {
    std::shared_lock lock(mutex);
    return name;
}

uint64_t Collection::get_created_at() const {
    return created_at.load();
}

size_t Collection::get_num_documents() const {
    return num_documents.load();
}

uint32_t Collection::get_collection_id() const {
    return collection_id.load();
}

Option<uint32_t> Collection::doc_id_to_seq_id_with_lock(const std::string & doc_id) const {
    std::shared_lock lock(mutex);
    return doc_id_to_seq_id(doc_id);
}

Option<uint32_t> Collection::doc_id_to_seq_id(const std::string & doc_id) const {
    std::string seq_id_str;
    StoreStatus status = store->get(get_doc_id_key(doc_id), seq_id_str);
    if(status == StoreStatus::FOUND) {
        uint32_t seq_id = (uint32_t) std::stoi(seq_id_str);
        return Option<uint32_t>(seq_id);
    }

    if(status == StoreStatus::NOT_FOUND) {
        return Option<uint32_t>(404, "Not found.");
    }

    return Option<uint32_t>(500, "Error while fetching doc_id from store.");
}

std::vector<std::string> Collection::get_facet_fields() {
    std::shared_lock lock(mutex);

    std::vector<std::string> facet_fields_copy;
    for(auto it = search_schema.begin(); it != search_schema.end(); ++it) {
        if(it.value().facet) {
            facet_fields_copy.push_back(it.key());
        }
    }

    return facet_fields_copy;
}

std::vector<field> Collection::get_sort_fields() {
    std::shared_lock lock(mutex);

    std::vector<field> sort_fields_copy;
    for(auto it = search_schema.begin(); it != search_schema.end(); ++it) {
        if(it.value().sort) {
            sort_fields_copy.push_back(it.value());
        }
    }

    return sort_fields_copy;
}

std::vector<field> Collection::get_fields() {
    std::shared_lock lock(mutex);
    return fields;
}

bool Collection::contains_field(const std::string &field) {
    std::shared_lock lock(mutex);
    return search_schema.find(field) != search_schema.end();
}

std::unordered_map<std::string, field> Collection::get_dynamic_fields() {
    std::shared_lock lock(mutex);
    return dynamic_fields;
}

tsl::htrie_map<char, field> Collection::get_schema() {
    std::shared_lock lock(mutex);
    return search_schema;
};

tsl::htrie_map<char, field> Collection::get_nested_fields() {
    std::shared_lock lock(mutex);
    return nested_fields;
};

tsl::htrie_map<char, field> Collection::get_embedding_fields() {
    std::shared_lock lock(mutex);
    return embedding_fields;
};

tsl::htrie_set<char> Collection::get_object_reference_helper_fields() const {
    std::shared_lock lock(mutex);
    return object_reference_helper_fields;
}

std::string Collection::get_meta_key(const std::string & collection_name) {
    return std::string(COLLECTION_META_PREFIX) + "_" + collection_name;
}

std::string Collection::get_override_key(const std::string & collection_name, const std::string & override_id) {
    return std::string(COLLECTION_OVERRIDE_PREFIX) + "_" + collection_name + "_" + override_id;
}

std::string Collection::get_seq_id_collection_prefix() const {
    return std::to_string(collection_id) + "_" + std::string(SEQ_ID_PREFIX);
}

std::string Collection::get_default_sorting_field() {
    std::shared_lock lock(mutex);
    return default_sorting_field;
}

Option<bool> Collection::get_document_from_store(const uint32_t& seq_id,
                                                 nlohmann::json& document, bool raw_doc) const {
    return get_document_from_store(get_seq_id_key(seq_id), document, raw_doc);
}

Option<bool> Collection::get_document_from_store(const std::string &seq_id_key,
                                                 nlohmann::json& document, bool raw_doc) const {
    std::string json_doc_str;
    StoreStatus json_doc_status = store->get(seq_id_key, json_doc_str);

    if(json_doc_status != StoreStatus::FOUND) {
        const std::string& seq_id = std::to_string(get_seq_id_from_key(seq_id_key));
        if(json_doc_status == StoreStatus::NOT_FOUND) {
            return Option<bool>(404, "Could not locate the JSON document for sequence ID: " + seq_id);
        }

        return Option<bool>(500, "Error while fetching JSON document for sequence ID: " + seq_id);
    }

    try {
        document = nlohmann::json::parse(json_doc_str);
    } catch(...) {
        return Option<bool>(500, "Error while parsing stored document with sequence ID: " + seq_id_key);
    }

    if(!raw_doc && enable_nested_fields) {
        std::vector<field> flattened_fields;
        field::flatten_doc(document, nested_fields, {}, true, flattened_fields);
    }

    return Option<bool>(true);
}

const Index* Collection::_get_index() const {
    return index;
}

Option<bool> Collection::parse_pinned_hits(const std::string& pinned_hits_str,
                                           std::map<size_t, std::vector<std::string>>& pinned_hits) {
    if(!pinned_hits_str.empty()) {
        std::vector<std::string> pinned_hits_strs;
        StringUtils::split(pinned_hits_str, pinned_hits_strs, ",");

        for(const std::string & pinned_hits_part: pinned_hits_strs) {
            std::vector<std::string> expression_parts;
            int64_t index = pinned_hits_part.size() - 1;
            while(index >= 0 && pinned_hits_part[index] != ':') {
                index--;
            }

            if(index == 0) {
                return Option<bool>(400, "Pinned hits are not in expected format.");
            }

            std::string pinned_id = pinned_hits_part.substr(0, index);
            std::string pinned_pos = pinned_hits_part.substr(index+1);

            if(!StringUtils::is_positive_integer(pinned_pos)) {
                return Option<bool>(400, "Pinned hits are not in expected format.");
            }

            int position = std::stoi(pinned_pos);
            if(position == 0) {
                return Option<bool>(400, "Pinned hits must start from position 1.");
            }

            pinned_hits[position].emplace_back(pinned_id);
        }
    }

    return Option<bool>(true);
}

Option<drop_tokens_param_t> Collection::parse_drop_tokens_mode(const std::string& drop_tokens_mode) {
    drop_tokens_mode_t drop_tokens_mode_val = left_to_right;
    size_t drop_tokens_token_limit = 1000;
    auto drop_tokens_mode_op = magic_enum::enum_cast<drop_tokens_mode_t>(drop_tokens_mode);
    if(drop_tokens_mode_op.has_value()) {
        drop_tokens_mode_val = drop_tokens_mode_op.value();
    } else {
        std::vector<std::string> drop_token_parts;
        StringUtils::split(drop_tokens_mode, drop_token_parts, ":");
        if(drop_token_parts.size() == 2) {
            if(!StringUtils::is_uint32_t(drop_token_parts[1])) {
                return Option<drop_tokens_param_t>(400, "Invalid format for drop tokens mode.");
            }

            drop_tokens_mode_op = magic_enum::enum_cast<drop_tokens_mode_t>(drop_token_parts[0]);
            if(drop_tokens_mode_op.has_value()) {
                drop_tokens_mode_val = drop_tokens_mode_op.value();
            }

            drop_tokens_token_limit = std::stoul(drop_token_parts[1]);

        } else {
            return Option<drop_tokens_param_t>(400, "Invalid format for drop tokens mode.");
        }
    }

    return Option<drop_tokens_param_t>(drop_tokens_param_t(drop_tokens_mode_val, drop_tokens_token_limit));
}

Option<bool> Collection::add_synonym(const nlohmann::json& syn_json, bool write_to_store) {
    std::shared_lock lock(mutex);
    synonym_t synonym;
    Option<bool> syn_op = synonym_t::parse(syn_json, synonym);

    if(!syn_op.ok()) {
        return syn_op;
    }

    return synonym_index->add_synonym(name, synonym, write_to_store);
}

bool Collection::get_synonym(const std::string& id, synonym_t& synonym) {
    std::shared_lock lock(mutex);
    return synonym_index->get_synonym(id, synonym);
}

Option<bool> Collection::remove_synonym(const std::string &id) {
    std::shared_lock lock(mutex);
    return synonym_index->remove_synonym(name, id);
}

void Collection::synonym_reduction(const std::vector<std::string>& tokens,
                                     std::vector<std::vector<std::string>>& results) const {
    std::shared_lock lock(mutex);
    return synonym_index->synonym_reduction(tokens, results);
}

spp::sparse_hash_map<std::string, synonym_t> Collection::get_synonyms() {
    std::shared_lock lock(mutex);
    return synonym_index->get_synonyms();
}

SynonymIndex* Collection::get_synonym_index() {
    return synonym_index;
}

spp::sparse_hash_map<std::string, reference_pair> Collection::get_reference_fields() {
    std::shared_lock lock(mutex);
    return reference_fields;
}

Option<bool> Collection::persist_collection_meta() {
    // first compact nested fields (to keep only parents of expanded children)
    field::compact_nested_fields(nested_fields);

    std::string coll_meta_json;
    StoreStatus status = store->get(Collection::get_meta_key(name), coll_meta_json);

    if(status != StoreStatus::FOUND) {
        return Option<bool>(500, "Could not fetch collection meta from store.");
    }

    nlohmann::json collection_meta;

    try {
        collection_meta = nlohmann::json::parse(coll_meta_json);
    } catch(...) {
        return Option<bool>(500, "Unable to parse collection meta.");
    }

    nlohmann::json fields_json = nlohmann::json::array();
    Option<bool> fields_json_op = field::fields_to_json_fields(fields, default_sorting_field, fields_json);

    if(!fields_json_op.ok()) {
        return Option<bool>(fields_json_op.code(), fields_json_op.error());
    }

    collection_meta[COLLECTION_SEARCH_FIELDS_KEY] = fields_json;
    collection_meta[Collection::COLLECTION_DEFAULT_SORTING_FIELD_KEY] = default_sorting_field;
    collection_meta[Collection::COLLECTION_FALLBACK_FIELD_TYPE] = fallback_field_type;

    bool persisted = store->insert(Collection::get_meta_key(name), collection_meta.dump());
    if(!persisted) {
        return Option<bool>(500, "Could not persist collection meta to store.");
    }

    return Option<bool>(true);
}

Option<bool> Collection::batch_alter_data(const std::vector<field>& alter_fields,
                                          const std::vector<field>& del_fields,
                                          const std::string& this_fallback_field_type) {
    // Update schema with additions (deletions can only be made later)
    std::vector<field> new_fields;
    tsl::htrie_map<char, field> schema_additions;
    std::vector<std::string> nested_field_names;
    bool found_embedding_field = false;

    std::unique_lock ulock(mutex);

    for(auto& f: alter_fields) {
        if(f.name == ".*") {
            fields.push_back(f);
            continue;
        }

        if(f.is_dynamic()) {
            dynamic_fields.emplace(f.name, f);
        } else {
            schema_additions.emplace(f.name, f);
            search_schema.emplace(f.name, f);
            new_fields.push_back(f);
        }

        if(f.nested) {
            nested_fields.emplace(f.name, f);
            nested_field_names.push_back(f.name);
        }

        if(f.embed.count(fields::from) != 0) {
            found_embedding_field = true;
            const auto& text_embedders = EmbedderManager::get_instance()._get_text_embedders();
            const auto& model_name = f.embed[fields::model_config][fields::model_name].get<std::string>();
            if(text_embedders.count(model_name) == 0) {
                size_t dummy_num_dim = 0;
                auto validate_model_res = EmbedderManager::get_instance().validate_and_init_model(f.embed[fields::model_config], dummy_num_dim);
                if(!validate_model_res.ok()) {
                    return Option<bool>(validate_model_res.code(), validate_model_res.error());
                }
            }
            embedding_fields.emplace(f.name, f);
        }

        fields.push_back(f);
    }

    field::compact_nested_fields(nested_fields);

    ulock.unlock();
    std::shared_lock shlock(mutex);

    index->refresh_schemas(new_fields, {});

    // Now, we can index existing data onto the updated schema
    const std::string seq_id_prefix = get_seq_id_collection_prefix();
    std::string upper_bound_key = get_seq_id_collection_prefix() + "`";  // cannot inline this
    rocksdb::Slice upper_bound(upper_bound_key);

    rocksdb::Iterator* iter = store->scan(seq_id_prefix, &upper_bound);
    std::unique_ptr<rocksdb::Iterator> iter_guard(iter);

    size_t num_found_docs = 0;
    std::vector<index_record> iter_batch;
    const size_t index_batch_size = 1000;

    auto begin = std::chrono::high_resolution_clock::now();

    while(iter->Valid() && iter->key().starts_with(seq_id_prefix)) {
        num_found_docs++;
        const uint32_t seq_id = Collection::get_seq_id_from_key(iter->key().ToString());

        nlohmann::json document;

        try {
            document = nlohmann::json::parse(iter->value().ToString());
        } catch(const std::exception& e) {
            return Option<bool>(400, "Bad JSON in document: " + document.dump(-1, ' ', false,
                                                                                nlohmann::detail::error_handler_t::ignore));
        }

        if(enable_nested_fields) {
            std::vector<field> flattened_fields;
            field::flatten_doc(document, nested_fields, {}, true, flattened_fields);
        }

        index_record record(num_found_docs, seq_id, document, index_operation_t::CREATE, DIRTY_VALUES::COERCE_OR_DROP);
        iter_batch.emplace_back(std::move(record));

        // Peek and check for last record right here so that we handle batched indexing correctly
        // Without doing this, the "last batch" would have to be indexed outside the loop.
        iter->Next();
        bool last_record = !(iter->Valid() && iter->key().starts_with(seq_id_prefix));

        if(num_found_docs % index_batch_size == 0 || last_record) {
            // put delete first because a field could be deleted and added in the same change set
            if(!del_fields.empty()) {
                for(auto& rec: iter_batch) {
                    index->remove(seq_id, rec.doc, del_fields, true);
                }
            }

            Index::batch_memory_index(index, iter_batch, default_sorting_field, search_schema, embedding_fields,
                                      fallback_field_type, token_separators, symbols_to_index, true, 200, 60000, 2,
                                      found_embedding_field, true, schema_additions);

            if(found_embedding_field) {
                for(auto& index_record : iter_batch) {
                    if(index_record.indexed.ok()) {
                        remove_flat_fields(index_record.doc);
                        const std::string& serialized_json = index_record.doc.dump(-1, ' ', false, nlohmann::detail::error_handler_t::ignore);
                        bool write_ok = store->insert(get_seq_id_key(index_record.seq_id), serialized_json);

                        if(!write_ok) {
                            LOG(ERROR) << "Inserting doc with new embedding field failed for seq id: " << index_record.seq_id;
                            index_record.index_failure(500, "Could not write to on-disk storage.");
                        } else {
                            index_record.index_success();
                        }
                    }
                }
            }

            iter_batch.clear();
        }

        if(num_found_docs % ((1 << 14)) == 0) {
            // having a cheaper higher layer check to prevent checking clock too often
            auto time_elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::high_resolution_clock::now() - begin).count();

            if(time_elapsed > 30) {
                begin = std::chrono::high_resolution_clock::now();
                LOG(INFO) << "Altered " << num_found_docs << " so far.";
            }
        }
    }

    LOG(INFO) << "Finished altering " << num_found_docs << " document(s).";
    shlock.unlock();
    ulock.lock();

    std::vector<field> garbage_embedding_fields_vec;
    for(auto& del_field: del_fields) {
        search_schema.erase(del_field.name);
        auto new_end = std::remove_if(fields.begin(), fields.end(), [&del_field](const field& f) {
            return f.name == del_field.name;
        });

        fields.erase(new_end, fields.end());

        if(del_field.is_dynamic()) {
            dynamic_fields.erase(del_field.name);
        }

        if(del_field.nested) {
            nested_fields.erase(del_field.name);
        }

        if(del_field.embed.count(fields::from) != 0) {
            remove_embedding_field(del_field.name);
        }

        if(del_field.name == ".*") {
            fallback_field_type = "";
        }

        if(del_field.name == default_sorting_field) {
            default_sorting_field = "";
        }

        process_remove_field_for_embedding_fields(del_field, garbage_embedding_fields_vec);
    }

    ulock.unlock();
    shlock.lock();

    index->refresh_schemas({}, del_fields);
    index->refresh_schemas({}, garbage_embedding_fields_vec);

    auto persist_op = persist_collection_meta();
    if(!persist_op.ok()) {
        return persist_op;
    }

    return Option<bool>(true);
}

Option<bool> Collection::alter(nlohmann::json& alter_payload) {
    std::shared_lock shlock(mutex);

    LOG(INFO) << "Collection " << name << " is being prepared for alter...";

    // Validate that all stored documents are compatible with the proposed schema changes.
    std::vector<field> del_fields;
    std::vector<field> addition_fields;
    std::vector<field> reindex_fields;

    std::string this_fallback_field_type;

    auto validate_op = validate_alter_payload(alter_payload, addition_fields, reindex_fields,
                                              del_fields, this_fallback_field_type);
    if(!validate_op.ok()) {
        LOG(INFO) << "Alter failed validation: " << validate_op.error();
        return validate_op;
    }

    if(!this_fallback_field_type.empty() && !fallback_field_type.empty()) {
        LOG(INFO) << "Alter failed: schema already contains a `.*` field.";
        return Option<bool>(400, "The schema already contains a `.*` field.");
    }

    shlock.unlock();

    if(!this_fallback_field_type.empty() && fallback_field_type.empty()) {
        std::unique_lock ulock(mutex);
        fallback_field_type = this_fallback_field_type;
    }

    LOG(INFO) << "Alter payload validation is successful...";
    if(!reindex_fields.empty()) {
        LOG(INFO) << "Processing field additions and deletions first...";
    }

    auto batch_alter_op = batch_alter_data(addition_fields, del_fields, fallback_field_type);
    if(!batch_alter_op.ok()) {
        LOG(INFO) << "Alter failed during alter data: " << batch_alter_op.error();
        return batch_alter_op;
    }

    if(!reindex_fields.empty()) {
        LOG(INFO) << "Processing field modifications now...";
        batch_alter_op = batch_alter_data(reindex_fields, {}, fallback_field_type);
        if(!batch_alter_op.ok()) {
            LOG(INFO) << "Alter failed during alter data: " << batch_alter_op.error();
            return batch_alter_op;
        }
    }

    // hide credentials in the alter payload return
    for(auto& field_json : alter_payload["fields"]) {
        if(field_json[fields::embed].count(fields::model_config) != 0) {
            hide_credential(field_json[fields::embed][fields::model_config], "api_key");
            hide_credential(field_json[fields::embed][fields::model_config], "access_token");
            hide_credential(field_json[fields::embed][fields::model_config], "refresh_token");
            hide_credential(field_json[fields::embed][fields::model_config], "client_id");
            hide_credential(field_json[fields::embed][fields::model_config], "client_secret");
            hide_credential(field_json[fields::embed][fields::model_config], "project_id");
        }
    }

    return Option<bool>(true);
}

void Collection::remove_flat_fields(nlohmann::json& document) {
    if(document.count(".flat") != 0) {
        for(const auto& flat_key: document[".flat"].get<std::vector<std::string>>()) {
            document.erase(flat_key);
        }
        document.erase(".flat");
    }
}

void Collection::remove_reference_helper_fields(nlohmann::json& document) {
    if(document.count(fields::reference_helper_fields) != 0) {
        for(const auto& key: document[fields::reference_helper_fields].get<std::vector<std::string>>()) {
            document.erase(key);
        }
        document.erase(fields::reference_helper_fields);
    }
}

Option<bool> Collection::prune_ref_doc(nlohmann::json& doc,
                                       const reference_filter_result_t& references,
                                       const tsl::htrie_set<char>& ref_include_fields_full,
                                       const tsl::htrie_set<char>& ref_exclude_fields_full,
                                       const bool& is_reference_array,
                                       const ref_include_exclude_fields& ref_include_exclude) {
    auto const& ref_collection_name = ref_include_exclude.collection_name;
    auto& cm = CollectionManager::get_instance();
    auto ref_collection = cm.get_collection(ref_collection_name);
    if (ref_collection == nullptr) {
        return Option<bool>(400, "Referenced collection `" + ref_collection_name + "` in `include_fields` not found.");
    }

    auto const& alias = ref_include_exclude.alias;
    auto const& strategy = ref_include_exclude.strategy;
    auto error_prefix = "Referenced collection `" + ref_collection_name + "`: ";

    // One-to-one relation.
    if (strategy != ref_include::nest_array && !is_reference_array && references.count == 1) {
        auto ref_doc_seq_id = references.docs[0];

        nlohmann::json ref_doc;
        auto get_doc_op = ref_collection->get_document_from_store(ref_doc_seq_id, ref_doc);
        if (!get_doc_op.ok()) {
            return Option<bool>(get_doc_op.code(), error_prefix + get_doc_op.error());
        }

        remove_flat_fields(ref_doc);
        remove_reference_helper_fields(ref_doc);

        auto prune_op = prune_doc(ref_doc, ref_include_fields_full, ref_exclude_fields_full);
        if (!prune_op.ok()) {
            return Option<bool>(prune_op.code(), error_prefix + prune_op.error());
        }

        auto const key = alias.empty() ? ref_collection_name : alias;
        auto const& nest_ref_doc = (strategy == ref_include::nest);
        if (!ref_doc.empty()) {
            if (nest_ref_doc) {
                doc[key] = ref_doc;
            } else {
                if (!alias.empty()) {
                    auto temp_doc = ref_doc;
                    ref_doc.clear();
                    for (const auto &item: temp_doc.items()) {
                        ref_doc[alias + item.key()] = item.value();
                    }
                }
                doc.update(ref_doc);
            }
        }

        // Include nested join references.
        if (!ref_include_exclude.nested_join_includes.empty() && !references.coll_to_references->empty()) {
            auto nested_include_exclude_op = include_references(nest_ref_doc ? doc[key] : doc, ref_doc_seq_id,
                                                                ref_collection.get(), references.coll_to_references[0],
                                                                ref_include_exclude.nested_join_includes);
            if (!nested_include_exclude_op.ok()) {
                return nested_include_exclude_op;
            }
        }

        return Option<bool>(true);
    }

    // One-to-many relation.
    for (uint32_t i = 0; i < references.count; i++) {
        auto ref_doc_seq_id = references.docs[i];

        nlohmann::json ref_doc;
        auto get_doc_op = ref_collection->get_document_from_store(ref_doc_seq_id, ref_doc);
        if (!get_doc_op.ok()) {
            return Option<bool>(get_doc_op.code(), error_prefix + get_doc_op.error());
        }

        remove_flat_fields(ref_doc);
        remove_reference_helper_fields(ref_doc);

        auto prune_op = prune_doc(ref_doc, ref_include_fields_full, ref_exclude_fields_full);
        if (!prune_op.ok()) {
            return Option<bool>(prune_op.code(), error_prefix + prune_op.error());
        }

        std::string key;
        auto const& nest_ref_doc = (strategy == ref_include::nest || strategy == ref_include::nest_array);
        if (!ref_doc.empty()) {
            if (nest_ref_doc) {
                key = alias.empty() ? ref_collection_name : alias;
                if (doc.contains(key) && !doc[key].is_array()) {
                    return Option<bool>(400, "Could not include the reference document of `" + ref_collection_name +
                                             "` collection. Expected `" + key + "` to be an array. Try " +
                                             (alias.empty() ? "adding an" : "renaming the") + " alias.");
                }

                doc[key] += ref_doc;
            } else {
                for (auto ref_doc_it = ref_doc.begin(); ref_doc_it != ref_doc.end(); ref_doc_it++) {
                    auto const& ref_doc_key = ref_doc_it.key();
                    key = alias + ref_doc_key;
                    if (doc.contains(key) && !doc[key].is_array()) {
                        return Option<bool>(400, "Could not include the value of `" + ref_doc_key +
                                                 "` key of the reference document of `" + ref_collection_name +
                                                 "` collection. Expected `" + key + "` to be an array. Try " +
                                                 (alias.empty() ? "adding an" : "renaming the") + " alias.");
                    }

                    // Add the values of ref_doc as JSON array into doc.
                    doc[key] += ref_doc_it.value();
                }
            }
        }

        // Include nested join references.
        if (!ref_include_exclude.nested_join_includes.empty() &&
                references.coll_to_references != nullptr && !references.coll_to_references->empty()) {
            auto nested_include_exclude_op = include_references(nest_ref_doc ? doc[key].at(i) : doc, ref_doc_seq_id,
                                                                ref_collection.get(), references.coll_to_references[i],
                                                                ref_include_exclude.nested_join_includes);
            if (!nested_include_exclude_op.ok()) {
                return nested_include_exclude_op;
            }
        }
    }

    return Option<bool>(true);
}

Option<bool> Collection::include_references(nlohmann::json& doc, const uint32_t& seq_id, Collection const *const collection,
                                            const std::map<std::string, reference_filter_result_t>& reference_filter_results,
                                            const std::vector<ref_include_exclude_fields>& ref_include_exclude_fields_vec) {
    for (auto const& ref_include_exclude: ref_include_exclude_fields_vec) {
        auto const& ref_collection_name = ref_include_exclude.collection_name;

        auto& cm = CollectionManager::get_instance();
        auto ref_collection = cm.get_collection(ref_collection_name);
        if (ref_collection == nullptr) {
            return Option<bool>(400, "Referenced collection `" + ref_collection_name + "` in `include_fields` not found.");
        }

        auto const joined_on_ref_collection = reference_filter_results.count(ref_collection_name) > 0,
                has_filter_reference = (joined_on_ref_collection &&
                                        reference_filter_results.at(ref_collection_name).count > 0);
        auto doc_has_reference = false, joined_coll_has_reference = false;

        // Reference include_by without join, check if doc itself contains the reference.
        if (!joined_on_ref_collection && collection != nullptr) {
            doc_has_reference = ref_collection->is_referenced_in(collection->name);
        }

        std::string joined_coll_having_reference;
        // Check if the joined collection has a reference.
        if (!joined_on_ref_collection && !doc_has_reference) {
            for (const auto &reference_filter_result: reference_filter_results) {
                joined_coll_has_reference = ref_collection->is_referenced_in(reference_filter_result.first);
                if (joined_coll_has_reference) {
                    joined_coll_having_reference = reference_filter_result.first;
                    break;
                }
            }
        }

        if (!has_filter_reference && !doc_has_reference && !joined_coll_has_reference) {
            continue;
        }

        std::vector<std::string> ref_include_fields_vec, ref_exclude_fields_vec;
        StringUtils::split(ref_include_exclude.include_fields, ref_include_fields_vec, ",");
        StringUtils::split(ref_include_exclude.exclude_fields, ref_exclude_fields_vec, ",");

        spp::sparse_hash_set<std::string> ref_include_fields, ref_exclude_fields;
        ref_include_fields.insert(ref_include_fields_vec.begin(), ref_include_fields_vec.end());
        ref_exclude_fields.insert(ref_exclude_fields_vec.begin(), ref_exclude_fields_vec.end());

        tsl::htrie_set<char> ref_include_fields_full, ref_exclude_fields_full;
        auto include_exclude_op = ref_collection->populate_include_exclude_fields_lk(ref_include_fields,
                                                                                     ref_exclude_fields,
                                                                                     ref_include_fields_full,
                                                                                     ref_exclude_fields_full);
        auto error_prefix = "Referenced collection `" + ref_collection_name + "`: ";
        if (!include_exclude_op.ok()) {
            return Option<bool>(include_exclude_op.code(), error_prefix + include_exclude_op.error());
        }

        Option<bool> prune_doc_op = Option<bool>(true);
        auto const& ref_collection_alias = ref_include_exclude.alias;
        if (has_filter_reference) {
            auto const& ref_filter_result = reference_filter_results.at(ref_collection_name);
            prune_doc_op = prune_ref_doc(doc, ref_filter_result, ref_include_fields_full, ref_exclude_fields_full,
                                         ref_filter_result.is_reference_array_field, ref_include_exclude);
        } else if (doc_has_reference) {
            auto get_reference_field_op = ref_collection->get_referenced_in_field_with_lock(collection->name);
            if (!get_reference_field_op.ok()) {
                continue;
            }
            auto const& field_name = get_reference_field_op.get();
            if (collection->search_schema.count(field_name) == 0) {
                continue;
            }

            if (collection->object_reference_helper_fields.count(field_name) != 0) {
                std::vector<std::string> keys;
                StringUtils::split(field_name, keys, ".");
                if (!doc.contains(keys[0])) {
                    return Option<bool>(400, "Could not find `" + keys[0] +
                                             "` in the document to include the referenced document.");
                }

                if (doc[keys[0]].is_array()) {
                    for (uint32_t i = 0; i < doc[keys[0]].size(); i++) {
                        uint32_t ref_doc_id;
                        auto op = collection->get_object_array_related_id(field_name, seq_id, i, ref_doc_id);
                        if (!op.ok()) {
                            if (op.code() == 404) { // field_name is not indexed.
                                break;
                            } else { // No reference found for this object.
                                continue;
                            }
                        }

                        reference_filter_result_t result(1, new uint32_t[1]{ref_doc_id});
                        prune_doc_op = prune_ref_doc(doc[keys[0]][i], result,
                                                     ref_include_fields_full, ref_exclude_fields_full,
                                                     false, ref_include_exclude);
                        if (!prune_doc_op.ok()) {
                            return prune_doc_op;
                        }
                    }
                } else {
                    std::vector<uint32_t> ids;
                    auto get_references_op = collection->get_related_ids(field_name, seq_id, ids);
                    if (!get_references_op.ok()) {
                        continue;
                    }
                    reference_filter_result_t result(ids.size(), &ids[0]);
                    prune_doc_op = prune_ref_doc(doc[keys[0]], result, ref_include_fields_full, ref_exclude_fields_full,
                                                 collection->search_schema.at(field_name).is_array(), ref_include_exclude);
                    result.docs = nullptr;
                }
            } else {
                std::vector<uint32_t> ids;
                auto get_references_op = collection->get_related_ids(field_name, seq_id, ids);
                if (!get_references_op.ok()) {
                    continue;
                }
                reference_filter_result_t result(ids.size(), &ids[0]);
                prune_doc_op = prune_ref_doc(doc, result, ref_include_fields_full, ref_exclude_fields_full,
                                             collection->search_schema.at(field_name).is_array(), ref_include_exclude);
                result.docs = nullptr;
            }
        } else if (joined_coll_has_reference) {
            auto joined_collection = cm.get_collection(joined_coll_having_reference);
            if (joined_collection == nullptr) {
                continue;
            }

            auto reference_field_name_op = ref_collection->get_referenced_in_field_with_lock(joined_coll_having_reference);
            if (!reference_field_name_op.ok() || joined_collection->get_schema().count(reference_field_name_op.get()) == 0) {
                continue;
            }

            auto const& reference_field_name = reference_field_name_op.get();
            auto const& reference_filter_result = reference_filter_results.at(joined_coll_having_reference);
            auto const& count = reference_filter_result.count;
            std::vector<uint32_t> ids;
            ids.reserve(count);
            for (uint32_t i = 0; i < count; i++) {
                joined_collection->get_related_ids_with_lock(reference_field_name, reference_filter_result.docs[i], ids);
            }
            if (ids.empty()) {
                continue;
            }

            gfx::timsort(ids.begin(), ids.end());
            ids.erase(unique(ids.begin(), ids.end()), ids.end());

            reference_filter_result_t result;
            result.count = ids.size();
            result.docs = &ids[0];
            prune_doc_op = prune_ref_doc(doc, result, ref_include_fields_full, ref_exclude_fields_full,
                                         joined_collection->get_schema().at(reference_field_name).is_array(),
                                         ref_include_exclude);
            result.docs = nullptr;
        }

        if (!prune_doc_op.ok()) {
            return prune_doc_op;
        }
    }

    return Option<bool>(true);
}

Option<bool> Collection::prune_doc(nlohmann::json& doc,
                                   const tsl::htrie_set<char>& include_names,
                                   const tsl::htrie_set<char>& exclude_names,
                                   const std::string& parent_name, size_t depth,
                                   const std::map<std::string, reference_filter_result_t>& reference_filter_results,
                                   Collection const *const collection, const uint32_t& seq_id,
                                   const std::vector<ref_include_exclude_fields>& ref_include_exclude_fields_vec) {
    // doc can only be an object
    auto it = doc.begin();
    while(it != doc.end()) {
        std::string nested_name = parent_name + (parent_name.empty() ? it.key() : "." + it.key());

        //LOG(INFO) << "it.key(): " << it.key() << ", nested_name: " << nested_name;

        // use prefix lookup to prune non-matching sub-trees early
        auto prefix_it = include_names.equal_prefix_range(nested_name);
        if(!include_names.empty() && prefix_it.first == prefix_it.second) {
            // prefix not found in allowed list of highlight field names, so can trim early
            it = doc.erase(it);
            continue ;
        }

        if(exclude_names.count(nested_name) != 0) {
            it = doc.erase(it);
            continue ;
        }

        if(exclude_names.empty() && !include_names.empty() && include_names.count(nested_name) != 0) {
            // without exclusions, we can pick the sub-tree early if parent name is found in include names
            it++;
            continue;
        }

        if(it.value().is_object()) {
            bool is_orig_empty = it.value().empty();
            prune_doc(it.value(), include_names, exclude_names, nested_name, depth+1);
            if(!is_orig_empty && it.value().empty()) {
                it = doc.erase(it);
            } else {
                it++;
            }

            continue;
        }

        else if(it.value().is_array()) {
            bool orig_array_empty = it.value().empty();
            bool primitive_array = true;
            auto arr_it = it.value().begin();
            while(arr_it != it.value().end()) {
                // NOTE: we will not support array of array of nested objects
                primitive_array = primitive_array && !arr_it.value().is_object();
                if(arr_it.value().is_object()) {
                    bool orig_ele_empty = arr_it.value().empty();
                    prune_doc(arr_it.value(), include_names, exclude_names, nested_name, depth+1);
                    // don't remove empty array objects to help frontend
                }

                arr_it++;
            }

            if(!orig_array_empty && it.value().empty()) {
                // only drop field if array became empty because of pruning (and not empty already)
                it = doc.erase(it);
                continue;
            }

            if(!primitive_array) {
                it++;
                continue;
            }
        }

        if(!include_names.empty() && include_names.count(nested_name) == 0) {
            // at this point, name should match fully, otherwise we should erase the value
            it = doc.erase(it);
            continue;
        }

        it++;
    }

    return include_references(doc, seq_id, collection, reference_filter_results, ref_include_exclude_fields_vec);
}

Option<bool> Collection::validate_alter_payload(nlohmann::json& schema_changes,
                                                std::vector<field>& addition_fields,
                                                std::vector<field>& reindex_fields,
                                                std::vector<field>& del_fields,
                                                std::string& fallback_field_type) {
    if(!schema_changes.is_object()) {
        return Option<bool>(400, "Bad JSON.");
    }

    if(schema_changes.size() != 1) {
        return Option<bool>(400, "Only `fields` can be updated at the moment.");
    }

    const std::string err_msg = "The `fields` value should be an array of objects containing "
                                "the field `name` and other properties.";

    if(!schema_changes.contains("fields") || !schema_changes["fields"].is_array() || schema_changes["fields"].empty()) {
        return Option<bool>(400, err_msg);
    }

    // basic validation of fields
    std::vector<field> diff_fields;
    tsl::htrie_map<char, field> updated_search_schema = search_schema;
    tsl::htrie_map<char, field> updated_nested_fields = nested_fields;
    tsl::htrie_map<char, field> updated_embedding_fields = embedding_fields;
    size_t num_auto_detect_fields = 0;

    // since fields can be deleted and added in the same change set,
    // we will first do a pass at basic validations and pick out fields to be deleted
    std::set<std::string> delete_field_names;

    // ensure that drop values are at the top: required for drop+add use case
    std::sort(schema_changes["fields"].begin(), schema_changes["fields"].end(),
              [](nlohmann::json& a, nlohmann::json& b) {
                    return a.contains("drop") > b.contains("drop");
              });

    for(const auto& kv: schema_changes["fields"].items()) {
        if (!kv.value().is_object()) {
            return Option<bool>(400, err_msg);
        }

        if (!kv.value().contains("name")) {
            return Option<bool>(400, err_msg);
        }

        const std::string& field_name = kv.value()["name"].get<std::string>();

        if(field_name == "id") {
            return Option<bool>(400, "Field `" + field_name + "` cannot be altered.");
        }

        if(kv.value().contains("drop")) {
            delete_field_names.insert(field_name);
        }
    }

    std::unordered_map<std::string, field> new_dynamic_fields;
    int json_array_index = -1;

    for(const auto& kv: schema_changes["fields"].items()) {
        json_array_index++;
        const std::string& field_name = kv.value()["name"].get<std::string>();
        const auto& field_it = search_schema.find(field_name);
        auto found_field = (field_it != search_schema.end());

        auto dyn_field_it = dynamic_fields.find(field_name);
        auto found_dyn_field = (dyn_field_it != dynamic_fields.end());

        if(kv.value().contains("drop")) {
            if(!kv.value()["drop"].is_boolean() || !kv.value()["drop"].get<bool>()) {
                return Option<bool>(400, "Field `" + field_name + "` must have a drop value of `true`.");
            }

            if(field_name == ".*") {
                del_fields.emplace_back(".*", field_types::AUTO, false);
                continue;
            }

            if(!found_field && !found_dyn_field) {
                return Option<bool>(400, "Field `" + field_name + "` is not part of collection schema.");
            }

            if(found_field && field_it.value().embed.count(fields::from) != 0) {
                updated_embedding_fields.erase(field_it.key());
            }

            if(found_field) {
                del_fields.push_back(field_it.value());
                updated_search_schema.erase(field_it.key());
                updated_nested_fields.erase(field_it.key());
                
                if(field_it.value().embed.count(fields::from) != 0) {
                    updated_embedding_fields.erase(field_it.key());
                }

                // should also remove children if the field being dropped is an object
                if(field_it.value().nested && enable_nested_fields) {
                    auto prefix_it = search_schema.equal_prefix_range(field_name);
                    for(auto prefix_kv = prefix_it.first; prefix_kv != prefix_it.second; ++prefix_kv) {
                        bool exact_key_match = (prefix_kv.key().size() == field_name.size());
                        if(!exact_key_match) {
                            del_fields.push_back(prefix_kv.value());
                            updated_search_schema.erase(prefix_kv.key());
                            updated_nested_fields.erase(prefix_kv.key());

                            if(prefix_kv.value().embed.count(fields::from) != 0) {
                                updated_embedding_fields.erase(prefix_kv.key());
                            }
                        }
                    }
                }
            }

            // NOTE: fields with type "auto" or "string*" will exist in both `search_schema` and `dynamic_fields`
            if(found_dyn_field) {
                del_fields.push_back(dyn_field_it->second);
                // we will also have to resolve the actual field names which match the dynamic field pattern
                for(auto& a_field: search_schema) {
                    if(std::regex_match(a_field.name, std::regex(dyn_field_it->first))) {
                        del_fields.push_back(a_field);
                        // if schema contains explicit fields that match dynamic field that're going to be removed,
                        // we will have to remove them from the schema so that validation can occur properly
                        updated_search_schema.erase(a_field.name);
                    }
                }
            }
        } else {
            // add or update existing field
            auto is_addition = (!found_field && !found_dyn_field);
            auto is_reindex = (delete_field_names.count(field_name) != 0);

            if(is_addition && is_reindex) {
                return Option<bool>(400, "Field `" + field_name +
                                    "` cannot be added and deleted at the same time.");
            }

            if(is_addition || is_reindex) {
                // must validate fields
                auto parse_op = field::json_field_to_field(enable_nested_fields, kv.value(), diff_fields,
                                                           fallback_field_type, num_auto_detect_fields);
                if (!parse_op.ok()) {
                    return parse_op;
                }

                auto& f = diff_fields.back();

                if(f.is_dynamic()) {
                    new_dynamic_fields[f.name] = f;
                } else {
                    updated_search_schema[f.name] = f;
                }

                if(!f.embed.empty()) {
                    auto validate_res = field::validate_and_init_embed_field(search_schema, schema_changes["fields"][json_array_index], schema_changes["fields"], f);

                    if(!validate_res.ok()) {
                        return validate_res;
                    }
                }

                if(is_reindex) {
                    reindex_fields.push_back(f);
                } else {
                    addition_fields.push_back(f);
                }

                if(f.embed.count(fields::from) != 0) {
                    embedding_fields.emplace(f.name, f);
                }


                if(f.nested && enable_nested_fields) {
                    updated_nested_fields.emplace(f.name, f);

                    // should also add children if the field is an object
                    auto prefix_it = search_schema.equal_prefix_range(field_name);
                    for(auto prefix_kv = prefix_it.first; prefix_kv != prefix_it.second; ++prefix_kv) {
                        bool exact_key_match = (prefix_kv.key().size() == field_name.size());
                        if(!exact_key_match) {
                            updated_search_schema.emplace(prefix_kv.key(), prefix_kv.value());
                            updated_nested_fields.emplace(prefix_kv.key(), prefix_kv.value());

                            if(prefix_kv.value().embed.count(fields::from) != 0) {
                                embedding_fields.emplace(prefix_kv.key(), prefix_kv.value());
                            }

                            if(is_reindex) {
                                reindex_fields.push_back(prefix_kv.value());
                            } else {
                                addition_fields.push_back(prefix_kv.value());
                            }
                        }
                    }
                }



            } else {
                // partial update is not supported for now
                return Option<bool>(400, "Field `" + field_name + "` is already part of the schema: To "
                                         "change this field, drop it first before adding it back to the schema.");
            }
        }
    }

    if(num_auto_detect_fields > 1) {
        return Option<bool>(400, "There can be only one field named `.*`.");
    }

    // data validations: here we ensure that already stored data is compatible with requested schema changes
    const std::string seq_id_prefix = get_seq_id_collection_prefix();
    std::string upper_bound_key = get_seq_id_collection_prefix() + "`";  // cannot inline this
    rocksdb::Slice upper_bound(upper_bound_key);

    rocksdb::Iterator* iter = store->scan(seq_id_prefix, &upper_bound);
    std::unique_ptr<rocksdb::Iterator> iter_guard(iter);

    size_t num_found_docs = 0;
    auto begin = std::chrono::high_resolution_clock::now();

    while(iter->Valid() && iter->key().starts_with(seq_id_prefix)) {
        num_found_docs++;
        const uint32_t seq_id = Collection::get_seq_id_from_key(iter->key().ToString());
        nlohmann::json document;

        try {
            document = nlohmann::json::parse(iter->value().ToString());
        } catch(const std::exception& e) {
            return Option<bool>(400, "Bad JSON in document: " + document.dump(-1, ' ', false,
                                                                                nlohmann::detail::error_handler_t::ignore));
        }

        if(!fallback_field_type.empty() || !new_dynamic_fields.empty() || !updated_nested_fields.empty()) {
            std::vector<field> new_fields;
            Option<bool> new_fields_op = detect_new_fields(document, DIRTY_VALUES::DROP,
                                                           updated_search_schema, new_dynamic_fields,
                                                           updated_nested_fields,
                                                           fallback_field_type, false,
                                                           new_fields,
                                                           enable_nested_fields,
                                                           reference_fields, object_reference_helper_fields);
            if(!new_fields_op.ok()) {
                return new_fields_op;
            }

            for(auto& new_field: new_fields) {
                if(updated_search_schema.find(new_field.name) == updated_search_schema.end()) {
                    if(new_field.nested) {
                        auto del_field_it = std::find_if(del_fields.begin(), del_fields.end(), [&new_field](const field& f) {
                            return f.name == new_field.name;
                        });

                        auto re_field_it = std::find_if(reindex_fields.begin(),
                                                        reindex_fields.end(), [&new_field](const field& f) {
                            return f.name == new_field.name;
                        });

                        if(del_field_it != del_fields.end() && re_field_it == reindex_fields.end()) {
                            // If the discovered field is already being deleted and is not part of reindex fields,
                            // we should ignore. This can happen when we are trying to drop a nested object's child.
                            continue;
                        }
                    }

                    reindex_fields.push_back(new_field);
                    updated_search_schema[new_field.name] = new_field;
                    if(new_field.nested) {
                        updated_nested_fields[new_field.name] = new_field;
                    }
                }
            }
        }

        // validate existing data on disk for compatibility via updated_search_schema
        auto validate_op = validator_t::validate_index_in_memory(document, seq_id, default_sorting_field,
                                                           updated_search_schema,
                                                           updated_embedding_fields,
                                                           index_operation_t::CREATE,
                                                           false,
                                                           fallback_field_type,
                                                           DIRTY_VALUES::COERCE_OR_REJECT);
        if(!validate_op.ok()) {
            std::string err_message = validate_op.error();

            // we've to message the error message to suite the schema alter context
            if(err_message.find("but is not found in the document.") != std::string::npos) {
                // missing field
                err_message.pop_back(); // delete trailing dot
                err_message += "s already present in the collection. If you still want to add this field, "
                               "set it as `optional: true`.";
                return Option<bool>(validate_op.code(), err_message);
            }

            else if(err_message.find("must be") != std::string::npos) {
                // type of an already stored document conflicts with new schema
                std::string type_error = "Schema change is incompatible with the type of documents already stored "
                                         "in this collection.";
                std::vector<std::string> err_parts;
                StringUtils::split(err_message, err_parts, "must be");
                if(err_parts.size() == 2) {
                    err_parts[0][0] = std::tolower(err_parts[0][0]);
                    type_error += " Existing data for " + err_parts[0] + " cannot be coerced into " + err_parts[1];
                }

                return Option<bool>(validate_op.code(), type_error);
            }

            else {
                std::string schema_err = "Schema change is incompatible with the type of documents already stored "
                                         "in this collection. error: " + validate_op.error();
                return Option<bool>(validate_op.code(), schema_err);
            }
        }

        if(num_found_docs % ((1 << 14)) == 0) {
            // having a cheaper higher layer check to prevent checking clock too often
            auto time_elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::high_resolution_clock::now() - begin).count();

            if(time_elapsed > 30) {
                begin = std::chrono::high_resolution_clock::now();
                LOG(INFO) << "Verified " << num_found_docs << " so far.";
            }
        }

        iter->Next();
    }

    return Option<bool>(true);
}

Option<bool> Collection::resolve_field_type(field& new_field,
                                            nlohmann::detail::iter_impl<nlohmann::basic_json<>>& kv,
                                            nlohmann::json& document,
                                            const DIRTY_VALUES& dirty_values,
                                            const bool found_dynamic_field,
                                            const std::string& fallback_field_type,
                                            const bool enable_nested_fields,
                                            std::vector<field>& new_fields) {
    if(!new_field.index) {
        return Option<bool>(true);
    }

    // Type detection scenarios:
    // a) Not a dynamic field + fallback type is explicit: use fallback type
    // b) Dynamic field + type is explicit: use explicit type
    // c) Not a dynamic field + fallback type is auto: detect and assign type
    // d) Dynamic field + type is auto: detect and assign type
    // e) Not a dynamic field + fallback type is string*: map to string/string[]
    // f) Dynamic field + type is string*: map to string/string[]

    const std::string& test_field_type = found_dynamic_field ? new_field.type : fallback_field_type;

    if(test_field_type == field_types::AUTO || field_types::is_string_or_array(test_field_type)) {
        if(kv.key() == ".*") {
            return Option<bool>(true);
        }

        std::string field_type;
        bool parseable = field::get_type(kv.value(), field_type);
        if(!parseable) {

            if(kv.value().is_null() && new_field.optional) {
                // null values are allowed only if field is optional
                kv = document.erase(kv);
                return Option<bool>(false);
            }

            if(kv.value().is_object()) {
                return Option<bool>(true);
            }

            if(kv.value().is_array() && kv.value().empty()) {
                return Option<bool>(true);
            }

            if(dirty_values == DIRTY_VALUES::REJECT || dirty_values == DIRTY_VALUES::COERCE_OR_REJECT) {
                return Option<bool>(400, "Type of field `" + kv.key() + "` is invalid.");
            } else {
                // DROP or COERCE_OR_DROP
                kv = document.erase(kv);
                return Option<bool>(false);
            }
        }

        if(test_field_type == field_types::AUTO) {
            new_field.type = field_type;
            if(new_field.is_object()) {
                new_field.nested = true;
            }
        } else {
            if (kv.value().is_array()) {
                new_field.type = field_types::STRING_ARRAY;
            } else {
                new_field.type = field_types::STRING;
            }
        }
    }

    else {
        new_field.type = test_field_type;
    }

    if (new_field.is_num_sort_field()) {
        // only numerical fields are added to sort index in dynamic type detection
        new_field.sort = true;
    }

    if(enable_nested_fields || !new_field.nested) {
        // only detect nested field if it is enabled explicitly
        new_fields.emplace_back(new_field);
    }

    return Option<bool>(true);
}

Option<bool> Collection::detect_new_fields(nlohmann::json& document,
                                           const DIRTY_VALUES& dirty_values,
                                           const tsl::htrie_map<char, field>& schema,
                                           const std::unordered_map<std::string, field>& dyn_fields,
                                           tsl::htrie_map<char, field>& nested_fields,
                                           const std::string& fallback_field_type,
                                           bool is_update,
                                           std::vector<field>& new_fields,
                                           const bool enable_nested_fields,
                                           const spp::sparse_hash_map<std::string, reference_pair>& reference_fields,
                                           tsl::htrie_set<char>& object_reference_helper_fields) {

    auto kv = document.begin();
    while(kv != document.end()) {
        // we will not index the special "id" key
        if (schema.count(kv.key()) == 0 && kv.key() != "id") {
            const std::string &fname = kv.key();
            field new_field(fname, field_types::STRING, false, true);
            bool found_dynamic_field = false;
            bool skip_field = false;

            // check against dynamic field definitions
            for(auto dyn_field_it = dyn_fields.begin(); dyn_field_it != dyn_fields.end(); dyn_field_it++) {
                auto& dynamic_field = dyn_field_it->second;

                if(std::regex_match (kv.key(), std::regex(dynamic_field.name))) {
                    // to prevent confusion we also disallow dynamic field names that contain ".*"
                    if((kv.key() != ".*" && kv.key().find(".*") != std::string::npos)) {
                        skip_field = true;
                        break;
                    }

                    new_field = dynamic_field;
                    new_field.name = fname;
                    found_dynamic_field = true;

                    if(kv->is_object() && dynamic_field.name.find(".*") == kv.key().size()) {
                        // e.g. { name => price.*, type: float } to match price.USD, price.UK etc.
                        // top-level price field should be treated as type `object` and NOT `float`.
                        new_field.nested = true;
                        new_field.type = field_types::OBJECT;
                        new_field.sort = false;
                    }

                    break;
                }
            }

            if(skip_field) {
                kv++;
                continue;
            }

            if(!found_dynamic_field && fallback_field_type.empty()) {
                // we will not auto detect schema for non-dynamic fields if auto detection is not enabled
                kv++;
                continue;
            }

            auto add_op = resolve_field_type(new_field, kv, document, dirty_values, found_dynamic_field,
                                             fallback_field_type, enable_nested_fields, new_fields);
            if(!add_op.ok()) {
                return add_op;
            }

            bool increment_iter = add_op.get();
            if(!increment_iter) {
                continue;
            }
        }

        kv++;
    }

    if(enable_nested_fields) {
        for(auto& new_field: new_fields) {
            if(new_field.nested) {
                nested_fields.emplace(new_field.name, new_field);
            }
        }

        std::vector<field> flattened_fields;
        auto flatten_op = field::flatten_doc(document, nested_fields, dyn_fields, is_update, flattened_fields);
        if(!flatten_op.ok()) {
            return flatten_op;
        }

        for(const auto& flattened_field: flattened_fields) {
            if(schema.find(flattened_field.name) == schema.end()) {
                new_fields.push_back(flattened_field);
            }
        }
    }

    auto add_reference_helper_fields_op = add_reference_helper_fields(document, schema, reference_fields,
                                                                      object_reference_helper_fields);
    if (!add_reference_helper_fields_op.ok()) {
        return add_reference_helper_fields_op;
    }

    return Option<bool>(true);
}

Index* Collection::init_index() {
    for(const field& field: fields) {
        if(field.is_dynamic()) {
            // regexp fields and fields with auto type are treated as dynamic fields
            dynamic_fields.emplace(field.name, field);
            continue;
        }

        if(field.name == ".*") {
            continue;
        }

        search_schema.emplace(field.name, field);

        if(field.nested) {
            nested_fields.emplace(field.name, field);
        }

        if(field.embed.count(fields::from) != 0) {
            embedding_fields.emplace(field.name, field);
        }

        if(!field.reference.empty()) {
            auto dot_index = field.reference.find('.');
            auto ref_coll_name = field.reference.substr(0, dot_index);
            auto ref_field_name = field.reference.substr(dot_index + 1);

            reference_fields.emplace(field.name, reference_pair(ref_coll_name, ref_field_name));

            auto& collectionManager = CollectionManager::get_instance();
            auto ref_coll = collectionManager.get_collection(ref_coll_name);
            if (ref_coll != nullptr) {
                // Passing reference helper field helps perform operation on doc_id instead of field value.
                ref_coll->add_referenced_in(name, field.name + fields::REFERENCE_HELPER_FIELD_SUFFIX);
            } else {
                // Reference collection has not been created yet.
                collectionManager.add_referenced_in_backlog(ref_coll_name,
                                                            reference_pair{name, field.name + fields::REFERENCE_HELPER_FIELD_SUFFIX});
            }
        }
    }

    field::compact_nested_fields(nested_fields);

    synonym_index = new SynonymIndex(store);

    return new Index(name+std::to_string(0),
                     collection_id,
                     store,
                     synonym_index,
                     CollectionManager::get_instance().get_thread_pool(),
                     search_schema,
                     symbols_to_index, token_separators);
}

DIRTY_VALUES Collection::parse_dirty_values_option(std::string& dirty_values) const {
    std::shared_lock lock(mutex);

    StringUtils::toupper(dirty_values);
    auto dirty_values_op = magic_enum::enum_cast<DIRTY_VALUES>(dirty_values);
    DIRTY_VALUES dirty_values_action;

    if(dirty_values_op.has_value()) {
        dirty_values_action = dirty_values_op.value();
    } else {
        dirty_values_action = (fallback_field_type.empty() && dynamic_fields.empty()) ?
                              DIRTY_VALUES::REJECT : DIRTY_VALUES::COERCE_OR_REJECT;
    }

    return dirty_values_action;
}

std::vector<char> Collection::to_char_array(const std::vector<std::string>& strings) {
    std::vector<char> vec;
    for(const auto& s: strings) {
        if(s.length() == 1) {
            vec.push_back(s[0]);
        }
    }

    return vec;
}

std::vector<char> Collection::get_symbols_to_index() {
    return symbols_to_index;
}

std::vector<char> Collection::get_token_separators() {
    return token_separators;
}

std::string Collection::get_fallback_field_type() {
    return fallback_field_type;
}

bool Collection::get_enable_nested_fields() {
    return enable_nested_fields;
}

Option<bool> Collection::parse_facet_with_lock(const std::string& facet_field, std::vector<facet>& facets) const {
    std::shared_lock lock(mutex);
    return parse_facet(facet_field, facets);
}

Option<bool> Collection::parse_facet(const std::string& facet_field, std::vector<facet>& facets) const {
    const std::regex base_pattern(".+\\(.*\\)");
    const std::regex range_pattern("[[0-9]*[a-z A-Z]+[0-9]*:\\[([+-]?([0-9]*[.])?[0-9]*)\\,\\s*([+-]?([0-9]*[.])?[0-9]*)\\]");
    const std::string _alpha = "_alpha";

    if (facet_field[0] == '$') { // Reference facet_by
        std::string error_message = "Error parsing reference facet: ";
        auto open_paren_pos = facet_field.find('(');
        if (open_paren_pos == std::string::npos) {
            return Option<bool>(400, error_message + "`" + facet_field + "`.");
        }

        std::string ref_collection_name = facet_field.substr(1, open_paren_pos - 1);
        auto &cm = CollectionManager::get_instance();
        auto ref_collection = cm.get_collection(ref_collection_name);
        if (ref_collection == nullptr) {
            return Option<bool>(400, error_message + "Referenced collection `" + ref_collection_name + "` not found.");
        }

        std::string ref_facet_expression = facet_field.substr(open_paren_pos + 1, facet_field.size() - open_paren_pos - 2);
        std::vector<std::string> ref_facet_strings;
        StringUtils::split_facet(ref_facet_expression, ref_facet_strings);

        std::vector<facet> ref_facets;
        for (const auto &ref_facet: ref_facet_strings) {
            auto parse_op = ref_collection->parse_facet_with_lock(ref_facet, ref_facets);
            if (!parse_op.ok()) {
                return Option<bool>(parse_op.code(), error_message + parse_op.error());
            }
        }

        for (auto &ref_facet: ref_facets) {
            ref_facet.reference_collection_name = ref_collection_name;
            facets.emplace_back(std::move(ref_facet));
        }
    }
    else if ((facet_field.find(":") != std::string::npos)
        && (facet_field.find("sort") == std::string::npos)) { //range based facet

       if (!std::regex_match(facet_field, base_pattern)) {
            std::string error = "Facet range value is not valid.";
            return Option<bool>(400, error);
        }

        auto startpos = facet_field.find("(");
        auto field_name = facet_field.substr(0, startpos);

        if(search_schema.count(field_name) == 0) {
            std::string error = "Could not find a facet field named `" + field_name + "` in the schema.";
            return Option<bool>(404, error);
        }

        const field& a_field = search_schema.at(field_name);

        if(!a_field.is_integer() && !a_field.is_float()){
            std::string error = "Range facet is restricted to only integer and float fields.";
            return Option<bool>(400, error);
        }

        facet a_facet(field_name);

        //starting after "(" and excluding ")"
        auto range_string = std::string(facet_field.begin() + startpos + 1, facet_field.end() - 1);

        //split the ranges
        std::vector<std::string> result;
        startpos = 0;
        int index=0;
        int commaFound = 0, rangeFound = 0;
        bool range_open=false;
        while(index < range_string.size()){
            if(range_string[index] == ']'){
                if(range_open == true){
                    std::string range = range_string.substr(startpos, index + 1 - startpos);
                    range=StringUtils::trim(range);
                    result.emplace_back(range);
                    rangeFound++;
                    range_open=false;
                }
                else{
                    result.clear();
                    break;
                }
            }
            else if(range_string[index] == ',' && range_open == false){
                startpos = index+1;
                commaFound++;
            }
            else if(range_string[index] == '['){
                if((commaFound == rangeFound) && range_open==false){
                    range_open=true;
                }
                else{
                    result.clear();
                    break;
                }
            }

            index++;
        }

        if((result.empty()) || (range_open==true)){
            std::string error = "Error splitting the facet range values.";
            return Option<bool>(400, error);
        }

        std::vector<std::tuple<int64_t, int64_t, std::string>> tupVec;

        auto& range_map = a_facet.facet_range_map;
        for(const auto& range : result){
            //validate each range syntax
            if(!std::regex_match(range, range_pattern)){
                std::string error = "Facet range value is not valid.";
                return Option<bool>(400, error);
            }
            auto pos1 = range.find(":");
            std::string range_val = range.substr(0, pos1);

            auto pos2 = range.find(",");
            auto pos3 = range.find("]");

            int64_t lower_range, upper_range;

            if(a_field.is_integer()) {
                auto start = pos1 + 2;
                auto end = pos2 - start;
                auto lower_range_str = range.substr(start, end);
                StringUtils::trim(lower_range_str);
                if(lower_range_str.empty()) {
                    lower_range = INT64_MIN;
                } else {
                    lower_range = std::stoll(lower_range_str);
                }

                start = pos2 + 1;
                end = pos3 - start;
                auto upper_range_str = range.substr(start, end);
                StringUtils::trim(upper_range_str);
                if(upper_range_str.empty()) {
                    upper_range = INT64_MAX;
                } else {
                    upper_range = std::stoll(upper_range_str);
                }
            } else {
                auto start = pos1 + 2;
                auto end = pos2 - start;
                auto lower_range_str = range.substr(start, end);
                StringUtils::trim(lower_range_str);
                if(lower_range_str.empty()) {
                    lower_range = INT64_MIN;
                } else {
                    float val = std::stof(lower_range_str);
                    lower_range = Index::float_to_int64_t(val);
                }

                start = pos2 + 1;
                end = pos3 - start;
                auto upper_range_str = range.substr(start, end);
                StringUtils::trim(upper_range_str);
                if(upper_range_str.empty()) {
                    upper_range = INT64_MAX;
                } else {
                    float val = std::stof(upper_range_str);
                    upper_range = Index::float_to_int64_t(val);
                }
            }

            tupVec.emplace_back(lower_range, upper_range, range_val);
        }

        //sort the range values so that we can check continuity
        sort(tupVec.begin(), tupVec.end());

        for(const auto& tup : tupVec){

            const auto& lower_range = std::get<0>(tup);
            const auto& upper_range = std::get<1>(tup);
            const std::string& range_val = std::get<2>(tup);
            //check if ranges are continous or not
            if((!range_map.empty()) && (range_map.find(lower_range)== range_map.end())){
                std::string error = "Ranges in range facet syntax should be continous.";
                return Option<bool>(400, error);
            }

            range_map[upper_range] =  range_val;
        }

        a_facet.is_range_query = true;

        facets.emplace_back(std::move(a_facet));
    } else if (facet_field.find('*') != std::string::npos) { // Wildcard
       if (facet_field[facet_field.size() - 1] != '*') {
           return Option<bool>(404, "Only prefix matching with a wildcard is allowed.");
       }

        // Trim * from the end.
        auto prefix = facet_field.substr(0, facet_field.size() - 1);
        auto pair = search_schema.equal_prefix_range(prefix);

        if(pair.first == pair.second) {
            // not found
            std::string error = "Could not find a facet field for `" + facet_field + "` in the schema.";
            return Option<bool>(404, error);
        }

        // Collect the fields that match the prefix and are marked as facet.
        for (auto field = pair.first; field != pair.second; field++) {
            if (field->facet) {
                facets.emplace_back(facet(field->name));
                facets.back().is_wildcard_match = true;
            }
        }
    } else {
       // normal facet
       std::string order = "";
       bool sort_alpha = false;
       std::string sort_field = "";
       std::string facet_field_copy = facet_field;
       auto pos = facet_field_copy.find("(");
       if(pos != std::string::npos) {
           facet_field_copy = facet_field_copy.substr(0, pos);
       }

       if (search_schema.count(facet_field_copy) == 0 || !search_schema.at(facet_field_copy).facet) {
           std::string error = "Could not find a facet field named `" + facet_field_copy + "` in the schema.";
           return Option<bool>(404, error);
       }

       if (facet_field.find("sort_by") != std::string::npos) { //sort params are supplied with facet
           std::vector<std::string> tokens;
           StringUtils::split(facet_field, tokens, ":");

           if(tokens.size() != 3) {
               std::string error = "Invalid sort format.";
               return Option<bool>(400, error);
           }

           //remove possible whitespaces
           for(auto i=0; i < 3; ++i) {
               StringUtils::trim(tokens[i]);
           }

           if(tokens[1] == _alpha) {
               const field &a_field = search_schema.at(facet_field_copy);
               if (!a_field.is_string()) {
                   std::string error = "Facet field should be string type to apply alpha sort.";
                   return Option<bool>(400, error);
               }
               sort_alpha = true;
           } else { //sort_field based sort
               sort_field = tokens[1];

               if (search_schema.count(sort_field) == 0 || !search_schema.at(sort_field).facet) {
                   std::string error = "Could not find a facet field named `" + sort_field + "` in the schema.";
                   return Option<bool>(404, error);
               }

               const field &a_field = search_schema.at(sort_field);
               if (a_field.is_string()) {
                   std::string error = "Sort field should be non string type to apply sort.";
                   return Option<bool>(400, error);
               }
           }

           if (tokens[2].find("asc") != std::string::npos) {
               order = "asc";
           } else if (tokens[2].find("desc") != std::string::npos) {
               order = "desc";
           } else {
               std::string error = "Invalid sort param.";
               return Option<bool>(400, error);
           }
       } else if (facet_field != facet_field_copy) {
           std::string error = "Invalid sort format.";
           return Option<bool>(400, error);
       }

       facets.emplace_back(facet(facet_field_copy, {}, false, sort_alpha,
                                 order, sort_field));
   }

    return Option<bool>(true);
}

Option<bool> Collection::compute_facet_infos_with_lock(const std::vector<facet>& facets, facet_query_t& facet_query,
                                                       const size_t facet_query_num_typos,
                                                       const uint32_t* all_result_ids, const size_t& all_result_ids_len,
                                                       const std::vector<std::string>& group_by_fields,
                                                       size_t group_limit, bool is_wildcard_no_filter_query,
                                                       size_t max_candidates,
                                                       std::vector<facet_info_t>& facet_infos, facet_index_type_t facet_index_type) const {
    std::shared_lock lock(mutex);
    return index->compute_facet_infos_with_lock(facets, facet_query, facet_query_num_typos, all_result_ids, all_result_ids_len,
                                                group_by_fields, group_limit, is_wildcard_no_filter_query,
                                                max_candidates, facet_infos, facet_index_type, this);
}

Option<bool> Collection::do_facets_with_lock(std::vector<facet> & facets, facet_query_t & facet_query,
                                             bool estimate_facets, size_t facet_sample_percent,
                                             const std::vector<facet_info_t>& facet_infos,
                                             size_t group_limit, const std::vector<std::string>& group_by_fields,
                                             const bool group_missing_values,
                                             const uint32_t* result_ids, size_t results_size,
                                             int max_facet_count, bool is_wildcard_query,
                                             facet_index_type_t facet_index_type) const {
    std::shared_lock lock(mutex);
    return index->do_facets_with_lock(facets, facet_query, estimate_facets, facet_sample_percent, facet_infos,
                                      group_limit, group_by_fields, group_missing_values, result_ids, results_size,
                                      max_facet_count, is_wildcard_query, facet_index_type, this);
}

Option<bool> Collection::populate_include_exclude_fields(const spp::sparse_hash_set<std::string>& include_fields,
                                                         const spp::sparse_hash_set<std::string>& exclude_fields,
                                                         tsl::htrie_set<char>& include_fields_full,
                                                         tsl::htrie_set<char>& exclude_fields_full) const {

    std::vector<std::string> include_fields_vec;
    std::vector<std::string> exclude_fields_vec;

    for(auto& f_name: include_fields) {
        auto field_op = extract_field_name(f_name, search_schema, include_fields_vec, false, enable_nested_fields, true, true);
        if(!field_op.ok()) {
            if(field_op.code() == 404) {
                // field need not be part of schema to be included (could be a stored value in the doc)
                include_fields_vec.push_back(f_name);
                continue;
            }
            return Option<bool>(field_op.code(), field_op.error());
        }
    }

    for(auto& f_name: exclude_fields) {
        if(f_name == "out_of") {
            // `out_of` is strictly a meta-field, but we handle it since it's useful
            continue;
        }

        if(f_name == "conversation_history") {
            continue;
        }

        auto field_op = extract_field_name(f_name, search_schema, exclude_fields_vec, false, enable_nested_fields, true, true);
        if(!field_op.ok()) {
            if(field_op.code() == 404) {
                // field need not be part of schema to be excluded (could be a stored value in the doc)
                exclude_fields_vec.push_back(f_name);
                continue;
            }
            return Option<bool>(field_op.code(), field_op.error());
        }
    }

    for(auto& f_name: include_fields_vec) {
        include_fields_full.insert(f_name);
    }

    for(auto& f_name: exclude_fields_vec) {
        exclude_fields_full.insert(f_name);
    }

    return Option<bool>(true);
}

Option<bool> Collection::populate_include_exclude_fields_lk(const spp::sparse_hash_set<std::string>& include_fields,
                                                            const spp::sparse_hash_set<std::string>& exclude_fields,
                                                            tsl::htrie_set<char>& include_fields_full,
                                                            tsl::htrie_set<char>& exclude_fields_full) const {
    std::shared_lock lock(mutex);
    return populate_include_exclude_fields(include_fields, exclude_fields, include_fields_full, exclude_fields_full);
}

// Removes the dropped field from embed_from of all embedding fields.
void Collection::process_remove_field_for_embedding_fields(const field& del_field,
                                                           std::vector<field>& garbage_embed_fields) {
    for(auto& field : fields) {
        if(field.embed.count(fields::from) == 0) {
            continue;
        }

        bool found_field = false;
        nlohmann::json& embed_from_names = field.embed[fields::from];
        for(auto it = embed_from_names.begin(); it != embed_from_names.end();) {
            if(it.value() == del_field.name) {
                it = embed_from_names.erase(it);
                found_field = true;
            } else {
                it++;
            }
        }

        if(found_field) {
            // mark this embedding field as "garbage" if it has no more embed_from fields
            if(embed_from_names.empty()) {
                garbage_embed_fields.push_back(field);
            } else {
                // the dropped field was present in `embed_from`, so we have to update the field objects
                field.embed[fields::from] = embed_from_names;
                embedding_fields[field.name].embed[fields::from] = embed_from_names;
            }
        }
    }

    for(auto& garbage_field: garbage_embed_fields) {
        remove_embedding_field(garbage_field.name);
        search_schema.erase(garbage_field.name);
        fields.erase(std::remove_if(fields.begin(), fields.end(), [&garbage_field](const auto &f) {
            return f.name == garbage_field.name;
        }), fields.end());
    }
}

void Collection::hide_credential(nlohmann::json& json, const std::string& credential_name) {
    if(json.count(credential_name) != 0) {
        // hide api key with * except first 5 chars
        std::string credential_name_str = json[credential_name];
        if(credential_name_str.size() > 5) {
            size_t num_chars_to_replace = credential_name_str.size() - 5;
            json[credential_name] = credential_name_str.replace(5, num_chars_to_replace, num_chars_to_replace, '*');
        } else {
            json[credential_name] = "***********";
        }
    }
}

Option<bool> Collection::truncate_after_top_k(const string &field_name, size_t k) {
    std::shared_lock slock(mutex);

    std::vector<uint32_t> seq_ids;
    auto op = index->seq_ids_outside_top_k(field_name, k, seq_ids);

    slock.unlock();

    if(!op.ok()) {
        return op;
    }

    for(auto seq_id: seq_ids) {
        auto remove_op = remove_if_found(seq_id);
        if(!remove_op.ok()) {
            LOG(ERROR) << "Error while truncating top k: " << remove_op.error();
        }
    }

    return Option<bool>(true);
}

void Collection::reference_populate_sort_mapping(int *sort_order, std::vector<size_t> &geopoint_indices,
                                                 std::vector<sort_by> &sort_fields_std,
                                                 std::array<spp::sparse_hash_map<uint32_t, int64_t, Hasher32> *, 3> &field_values)
                                                 const {
    std::shared_lock lock(mutex);
    index->populate_sort_mapping_with_lock(sort_order, geopoint_indices, sort_fields_std, field_values);
}

int64_t Collection::reference_string_sort_score(const string &field_name,  const uint32_t& seq_id) const {
    std::shared_lock lock(mutex);
    return index->reference_string_sort_score(field_name, seq_id);
}

bool Collection::is_referenced_in(const std::string& collection_name) const {
    std::shared_lock lock(mutex);
    return referenced_in.count(collection_name) > 0;
}

void Collection::add_referenced_in(const reference_pair& pair) {
    return add_referenced_in(pair.collection, pair.field);
}

void Collection::add_referenced_ins(const std::set<reference_pair>& pairs) {
    std::shared_lock lock(mutex);
    for (const auto &pair: pairs) {
        referenced_in.emplace(pair.collection, pair.field);
    }
}

void Collection::add_referenced_in(const std::string& collection_name, const std::string& field_name) {
    std::shared_lock lock(mutex);
    referenced_in.emplace(collection_name, field_name);
}

Option<std::string> Collection::get_referenced_in_field_with_lock(const std::string& collection_name) const {
    std::shared_lock lock(mutex);
    return get_referenced_in_field(collection_name);
}

Option<std::string> Collection::get_referenced_in_field(const std::string& collection_name) const {
    if (referenced_in.count(collection_name) == 0) {
        return Option<std::string>(400, "Could not find any field in `" + name + "` referencing the collection `"
                                        + collection_name + "`.");
    }

    return Option<std::string>(referenced_in.at(collection_name));
}

Option<bool> Collection::get_related_ids_with_lock(const std::string& field_name, const uint32_t& seq_id,
                                                   std::vector<uint32_t>& result) const {
    std::shared_lock lock(mutex);
    return index->get_related_ids_with_lock(name, field_name, seq_id, result);
}

Option<uint32_t> Collection::get_sort_index_value_with_lock(const std::string& field_name,
                                                            const uint32_t& seq_id) const {
    std::shared_lock lock(mutex);
    return index->get_sort_index_value_with_lock(name, field_name, seq_id);
}

std::shared_mutex& Collection::get_lifecycle_mutex() {
    return lifecycle_mutex;
}

void Collection::remove_embedding_field(const std::string& field_name) {
    if(embedding_fields.find(field_name) == embedding_fields.end()) {
        return;
    }

    const auto& del_field = embedding_fields[field_name];
    const auto& model_name = del_field.embed[fields::model_config]["model_name"].get<std::string>(); 
    embedding_fields.erase(field_name);
    CollectionManager::get_instance().process_embedding_field_delete(model_name);
}

tsl::htrie_map<char, field> Collection::get_embedding_fields_unsafe() {
    return embedding_fields;
}

void Collection::do_housekeeping() {
    index->repair_hnsw_index();
}

Option<bool> Collection::parse_and_validate_vector_query(const std::string& vector_query_str,
                                                         vector_query_t& vector_query,
                                                         const bool is_wildcard_query,
                                                         const size_t remote_embedding_timeout_ms, 
                                                         const size_t remote_embedding_num_tries,
                                                         size_t& per_page) const {

    auto parse_vector_op = VectorQueryOps::parse_vector_query_str(vector_query_str, vector_query,
                                                                    is_wildcard_query, this, false);
    if(!parse_vector_op.ok()) {
        return Option<bool>(400, parse_vector_op.error());
    }

    auto vector_field_it = search_schema.find(vector_query.field_name);
    if(vector_field_it == search_schema.end() || vector_field_it.value().num_dim == 0) {
        return Option<bool>(400, "Field `" + vector_query.field_name + "` does not have a vector query index.");
    }

    if(!vector_field_it.value().index) {
        return Option<bool>(400, "Field `" + vector_query.field_name + "` is marked as a non-indexed field in the schema.");
    }

    if(!vector_query.queries.empty()) {
        if(embedding_fields.find(vector_query.field_name) == embedding_fields.end()) {
            return Option<bool>(400, "`queries` parameter is only supported for auto-embedding fields.");
        }

        std::vector<std::vector<float>> embeddings;
        for(const auto& q: vector_query.queries) {
            EmbedderManager& embedder_manager = EmbedderManager::get_instance();
            auto embedder_op = embedder_manager.get_text_embedder(vector_field_it.value().embed[fields::model_config]);
            if(!embedder_op.ok()) {
                return Option<bool>(400, embedder_op.error());
            }

            auto remote_embedding_timeout_us = remote_embedding_timeout_ms * 1000;
            if((std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count() - search_begin_us) > remote_embedding_timeout_us) {
                std::string error = "Request timed out.";
                return Option<bool>(500, error);
            }

            auto embedder = embedder_op.get();

            if(embedder->is_remote()) {
                if(remote_embedding_num_tries == 0) {
                    std::string error = "`remote_embedding_num_tries` must be greater than 0.";
                    return Option<bool>(400, error);
                }
            }

            std::string embed_query = embedder_manager.get_query_prefix(vector_field_it.value().embed[fields::model_config]) + q;
            auto embedding_op = embedder->Embed(embed_query, remote_embedding_timeout_ms, remote_embedding_num_tries);

            if(!embedding_op.success) {
                if(!embedding_op.error["error"].get<std::string>().empty()) {
                    return Option<bool>(400, embedding_op.error["error"].get<std::string>());
                } else {
                    return Option<bool>(400, embedding_op.error.dump());
                }
            }

            embeddings.emplace_back(embedding_op.embedding);
        }
        
        if(vector_query.query_weights.empty()) {
            // get average of all embeddings
            std::vector<float> avg_embedding(vector_field_it.value().num_dim, 0);
            for(const auto& embedding: embeddings) {
                for(size_t i = 0; i < embedding.size(); i++) {
                    avg_embedding[i] += embedding[i];
                }
            }
            for(size_t i = 0; i < avg_embedding.size(); i++) {
                avg_embedding[i] /= embeddings.size();
            }

            vector_query.values = avg_embedding;
        } else {
            std::vector<float> embeddings_with_weights(vector_field_it.value().num_dim, 0);
            for(size_t i = 0; i < embeddings.size(); i++) {
                for(size_t j = 0; j < embeddings[i].size(); j++) {
                    embeddings_with_weights[j] += embeddings[i][j] * vector_query.query_weights[i];
                }
            }

            vector_query.values = embeddings_with_weights;
        }
    }

    if(is_wildcard_query) {
        if(vector_query.values.empty() && !vector_query.query_doc_given) {
            // for usability we will treat this as non-vector query
            vector_query.field_name.clear();
            if(vector_query.k != 0) {
                per_page = std::min(per_page, vector_query.k);
            }
        }

        else if(vector_field_it.value().num_dim != vector_query.values.size()) {
            return Option<bool>(400, "Query field `" + vector_query.field_name + "` must have " +
                                                std::to_string(vector_field_it.value().num_dim) + " dimensions.");
        }
    }

    return Option<bool>(true);
}

std::shared_ptr<VQModel> Collection::get_vq_model() {
    return vq_model;
}