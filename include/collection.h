#pragma once

#include <string>
#include <vector>
#include <string>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <shared_mutex>
#include <art.h>
#include <index.h>
#include <number.h>
#include <sparsepp.h>
#include <store.h>
#include <topster.h>
#include <json.hpp>
#include <field.h>
#include <option.h>
#include <tsl/htrie_map.h>
#include "tokenizer.h"
#include "synonym_index.h"
#include "vq_model_manager.h"

struct doc_seq_id_t {
    uint32_t seq_id;
    bool is_new;
};

struct highlight_field_t {
    std::string name;
    bool fully_highlighted;
    bool infix;
    bool is_string;
    tsl::htrie_map<char, token_leaf> qtoken_leaves;

    highlight_field_t(const std::string& name, bool fully_highlighted, bool infix, bool is_string):
            name(name), fully_highlighted(fully_highlighted), infix(infix), is_string(is_string) {

    }
};

struct reference_pair {
    std::string collection;
    std::string field;

    reference_pair(std::string collection, std::string field) : collection(std::move(collection)), field(std::move(field)) {}

    bool operator < (const reference_pair& pair) const {
        return collection < pair.collection;
    }
};

class Collection {
private:

    mutable std::shared_mutex mutex;

    // ensures that a Collection* is not destructed while in use by multiple threads
    mutable std::shared_mutex lifecycle_mutex;

    const uint8_t CURATED_RECORD_IDENTIFIER = 100;

    const size_t DEFAULT_TOPSTER_SIZE = 250;

    struct highlight_t {
        size_t field_index;
        std::string field;
        std::vector<std::string> snippets;
        std::vector<std::string> values;
        std::vector<size_t> indices;
        uint64_t match_score;
        std::vector<std::vector<std::string>> matched_tokens;

        highlight_t(): field_index(0), match_score(0)  {

        }

        bool operator<(const highlight_t& a) const {
            return std::tie(match_score, field_index) > std::tie(a.match_score, field_index);
        }
    };

    struct match_index_t {
        Match match;
        uint64_t match_score = 0;
        size_t index;

        match_index_t(Match match, uint64_t match_score, size_t index): match(match), match_score(match_score),
                                                                        index(index) {

        }

        bool operator<(const match_index_t& a) const {
            if(match_score != a.match_score) {
                return match_score > a.match_score;
            }
            return index < a.index;
        }
    };

    const std::string name;

    const std::atomic<uint32_t> collection_id;

    const std::atomic<uint64_t> created_at;

    std::atomic<size_t> num_documents;

    // Auto incrementing record ID used internally for indexing - not exposed to the client
    std::atomic<uint32_t> next_seq_id;

    Store* store;

    std::vector<field> fields;

    tsl::htrie_map<char, field> search_schema;

    std::map<std::string, override_t> overrides;

    // maps tag name => override_ids
    std::map<std::string, std::set<std::string>> override_tags;

    std::string default_sorting_field;

    const float max_memory_ratio;

    std::string fallback_field_type;

    std::unordered_map<std::string, field> dynamic_fields;

    tsl::htrie_map<char, field> nested_fields;

    tsl::htrie_map<char, field> embedding_fields;

    bool enable_nested_fields;

    std::vector<char> symbols_to_index;

    std::vector<char> token_separators;

    SynonymIndex* synonym_index;

    /// "field name" -> reference_pair(referenced_collection_name, referenced_field_name)
    spp::sparse_hash_map<std::string, reference_pair> reference_fields;

    /// Contains the info where the current collection is referenced.
    /// Useful to perform operations such as cascading delete.
    /// collection_name -> field_name
    spp::sparse_hash_map<std::string, std::string> referenced_in;

    /// Reference helper fields that are part of an object. The reference doc of these fields will be included in the
    /// object rather than in the document.
    tsl::htrie_set<char> object_reference_helper_fields;

    // Keep index as the last field since it is initialized in the constructor via init_index(). Add a new field before it.
    Index* index;

    // methods

    std::string get_doc_id_key(const std::string & doc_id) const;

    std::string get_seq_id_key(uint32_t seq_id) const;

    void highlight_result(const std::string& h_obj,
                          const field &search_field,
                          const size_t search_field_index,
                          const tsl::htrie_map<char, token_leaf>& qtoken_leaves,
                          const KV* field_order_kv, const nlohmann::json &document,
                          nlohmann::json& highlight_doc,
                          StringUtils & string_utils,
                          const size_t snippet_threshold,
                          const size_t highlight_affix_num_tokens,
                          bool highlight_fully,
                          bool is_infix_search,
                          const std::string& highlight_start_tag,
                          const std::string& highlight_end_tag,
                          const uint8_t* index_symbols,
                          highlight_t &highlight,
                          bool& found_highlight,
                          bool& found_full_highlight) const;

    void remove_document(const nlohmann::json & document, const uint32_t seq_id, bool remove_from_store);

    void process_remove_field_for_embedding_fields(const field& del_field, std::vector<field>& garbage_embed_fields);

    bool does_override_match(const override_t& override, std::string& query,
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
                             nlohmann::json& override_metadata) const;

    void curate_results(string& actual_query, const string& filter_query, bool enable_overrides, bool already_segmented,
                        const std::set<std::string>& tags,
                        const std::map<size_t, std::vector<std::string>>& pinned_hits,
                        const std::vector<std::string>& hidden_hits,
                        std::vector<std::pair<uint32_t, uint32_t>>& included_ids,
                        std::vector<uint32_t>& excluded_ids, std::vector<const override_t*>& filter_overrides,
                        bool& filter_curated_hits,
                        std::string& curated_sort_by, nlohmann::json& override_metadata) const;

    static Option<bool> detect_new_fields(nlohmann::json& document,
                                          const DIRTY_VALUES& dirty_values,
                                          const tsl::htrie_map<char, field>& schema,
                                          const std::unordered_map<std::string, field>& dyn_fields,
                                          tsl::htrie_map<char, field>& nested_fields,
                                          const std::string& fallback_field_type,
                                          bool is_update,
                                          std::vector<field>& new_fields,
                                          bool enable_nested_fields,
                                          const spp::sparse_hash_map<std::string, reference_pair>& reference_fields,
                                          tsl::htrie_set<char>& object_reference_helper_fields);

    static bool facet_count_compare(const facet_count_t& a, const facet_count_t& b) {
        return std::tie(a.count, a.fhash) > std::tie(b.count, b.fhash);
    }

    static bool facet_count_str_compare(const facet_value_t& a,
                                        const facet_value_t& b) {
        size_t a_count = a.count;
        size_t b_count = b.count;

        size_t a_value_size = UINT64_MAX - a.value.size();
        size_t b_value_size = UINT64_MAX - b.value.size();

        return std::tie(a_count, a_value_size, a.value) > std::tie(b_count, b_value_size, b.value);
    }

    static Option<bool> parse_pinned_hits(const std::string& pinned_hits_str,
                                   std::map<size_t, std::vector<std::string>>& pinned_hits);

    static Option<drop_tokens_param_t> parse_drop_tokens_mode(const std::string& drop_tokens_mode);

    Index* init_index();

    static std::vector<char> to_char_array(const std::vector<std::string>& strs);

    Option<bool> validate_and_standardize_sort_fields_with_lock(const std::vector<sort_by> & sort_fields,
                                                                std::vector<sort_by>& sort_fields_std,
                                                                bool is_wildcard_query,const bool is_vector_query,
                                                                const std::string& query, bool is_group_by_query = false,
                                                                const size_t remote_embedding_timeout_ms = 30000,
                                                                const size_t remote_embedding_num_tries = 2) const;

    Option<bool> validate_and_standardize_sort_fields(const std::vector<sort_by> & sort_fields,
                                                      std::vector<sort_by>& sort_fields_std,
                                                      const bool is_wildcard_query,
                                                      const bool is_vector_query,
                                                      const std::string& query, bool is_group_by_query = false,
                                                      const size_t remote_embedding_timeout_ms = 30000,
                                                      const size_t remote_embedding_num_tries = 2,
                                                      const bool is_reference_sort = false) const;
    
    Option<bool> persist_collection_meta();

    Option<bool> batch_alter_data(const std::vector<field>& alter_fields,
                                  const std::vector<field>& del_fields,
                                  const std::string& this_fallback_field_type);

    Option<bool> validate_alter_payload(nlohmann::json& schema_changes,
                                        std::vector<field>& addition_fields,
                                        std::vector<field>& reindex_fields,
                                        std::vector<field>& del_fields,
                                        std::string& fallback_field_type);

    void process_filter_overrides(std::vector<const override_t*>& filter_overrides,
                                  std::vector<std::string>& q_include_tokens,
                                  token_ordering token_order,
                                  filter_node_t*& filter_tree_root,
                                  std::vector<std::pair<uint32_t, uint32_t>>& included_ids,
                                  std::vector<uint32_t>& excluded_ids,
                                  nlohmann::json& override_metadata) const;

    void populate_text_match_info(nlohmann::json& info, uint64_t match_score, const text_match_type_t match_type) const;

    bool handle_highlight_text(std::string& text, bool normalise, const field &search_field,
                               const std::vector<char>& symbols_to_index, const std::vector<char>& token_separators,
                               highlight_t& highlight, StringUtils & string_utils, bool use_word_tokenizer,
                               const size_t highlight_affix_num_tokens,
                               const tsl::htrie_map<char, token_leaf>& qtoken_leaves, int last_valid_offset_index,
                               const size_t prefix_token_num_chars, bool highlight_fully, const size_t snippet_threshold,
                               bool is_infix_search, std::vector<std::string>& raw_query_tokens, size_t last_valid_offset,
                               const std::string& highlight_start_tag, const std::string& highlight_end_tag,
                               const uint8_t* index_symbols, const match_index_t& match_index) const;

    static Option<bool> extract_field_name(const std::string& field_name,
                                           const tsl::htrie_map<char, field>& search_schema,
                                           std::vector<std::string>& processed_search_fields,
                                           bool extract_only_string_fields,
                                           bool enable_nested_fields,
                                           const bool handle_wildcard = true,
                                           const bool& include_id = false);

    bool is_nested_array(const nlohmann::json& obj, std::vector<std::string> path_parts, size_t part_i) const;

    template<class T>
    static bool highlight_nested_field(const nlohmann::json& hdoc, nlohmann::json& hobj,
                                       std::vector<std::string>& path_parts, size_t path_index,
                                       bool is_arr_obj_ele, int array_index, T func);

    static Option<bool> resolve_field_type(field& new_field,
                                           nlohmann::detail::iter_impl<nlohmann::basic_json<>>& kv,
                                           nlohmann::json& document,
                                           const DIRTY_VALUES& dirty_values,
                                           const bool found_dynamic_field,
                                           const std::string& fallback_field_type,
                                           bool enable_nested_fields,
                                           std::vector<field>& new_fields);

    static uint64_t extract_bits(uint64_t value, unsigned lsb_offset, unsigned n);

    Option<bool> populate_include_exclude_fields(const spp::sparse_hash_set<std::string>& include_fields,
                                                 const spp::sparse_hash_set<std::string>& exclude_fields,
                                                 tsl::htrie_set<char>& include_fields_full,
                                                 tsl::htrie_set<char>& exclude_fields_full) const;

    Option<std::string> get_referenced_in_field(const std::string& collection_name) const;

    Option<bool> get_related_ids(const std::string& ref_field_name, const uint32_t& seq_id,
                                 std::vector<uint32_t>& result) const;

    Option<bool> get_object_array_related_id(const std::string& ref_field_name,
                                             const uint32_t& seq_id, const uint32_t& object_index,
                                             uint32_t& result) const;

    void remove_embedding_field(const std::string& field_name);

    Option<bool> parse_and_validate_vector_query(const std::string& vector_query_str,
                                                     vector_query_t& vector_query,
                                                     const bool is_wildcard_query,
                                                     const size_t remote_embedding_timeout_ms, 
                                                     const size_t remote_embedding_num_tries,
                                                     size_t& per_page) const;

    std::shared_ptr<VQModel> vq_model = nullptr;

public:

    enum {MAX_ARRAY_MATCHES = 5};

    const size_t PER_PAGE_MAX = 250;

    const size_t GROUP_LIMIT_MAX = 99;

    // Using a $ prefix so that these meta keys stay above record entries in a lexicographically ordered KV store
    static constexpr const char* COLLECTION_META_PREFIX = "$CM";
    static constexpr const char* COLLECTION_NEXT_SEQ_PREFIX = "$CS";
    static constexpr const char* COLLECTION_OVERRIDE_PREFIX = "$CO";
    static constexpr const char* SEQ_ID_PREFIX = "$SI";
    static constexpr const char* DOC_ID_PREFIX = "$DI";

    static constexpr const char* COLLECTION_NAME_KEY = "name";
    static constexpr const char* COLLECTION_ID_KEY = "id";
    static constexpr const char* COLLECTION_SEARCH_FIELDS_KEY = "fields";
    static constexpr const char* COLLECTION_DEFAULT_SORTING_FIELD_KEY = "default_sorting_field";
    static constexpr const char* COLLECTION_CREATED = "created_at";
    static constexpr const char* COLLECTION_NUM_MEMORY_SHARDS = "num_memory_shards";
    static constexpr const char* COLLECTION_FALLBACK_FIELD_TYPE = "fallback_field_type";
    static constexpr const char* COLLECTION_ENABLE_NESTED_FIELDS = "enable_nested_fields";

    static constexpr const char* COLLECTION_SYMBOLS_TO_INDEX = "symbols_to_index";
    static constexpr const char* COLLECTION_SEPARATORS = "token_separators";
    static constexpr const char* COLLECTION_VOICE_QUERY_MODEL = "voice_query_model";

    // methods

    Collection() = delete;

    Collection(const std::string& name, const uint32_t collection_id, const uint64_t created_at,
               const uint32_t next_seq_id, Store *store, const std::vector<field>& fields,
               const std::string& default_sorting_field,
               const float max_memory_ratio, const std::string& fallback_field_type,
               const std::vector<std::string>& symbols_to_index, const std::vector<std::string>& token_separators,
               const bool enable_nested_fields, std::shared_ptr<VQModel> vq_model = nullptr);

    ~Collection();

    static std::string get_next_seq_id_key(const std::string & collection_name);

    static std::string get_meta_key(const std::string & collection_name);

    static std::string get_override_key(const std::string & collection_name, const std::string & override_id);

    std::string get_seq_id_collection_prefix() const;

    std::string get_name() const;

    uint64_t get_created_at() const;

    uint32_t get_collection_id() const;

    uint32_t get_next_seq_id();

    Option<uint32_t> doc_id_to_seq_id_with_lock(const std::string & doc_id) const;

    Option<uint32_t> doc_id_to_seq_id(const std::string & doc_id) const;

    std::vector<std::string> get_facet_fields();

    std::vector<field> get_sort_fields();

    std::vector<field> get_fields();

    bool contains_field(const std::string&);

    std::unordered_map<std::string, field> get_dynamic_fields();

    tsl::htrie_map<char, field> get_schema();

    tsl::htrie_map<char, field> get_nested_fields();

    tsl::htrie_map<char, field> get_embedding_fields();

    tsl::htrie_map<char, field> get_embedding_fields_unsafe();

    tsl::htrie_set<char> get_object_reference_helper_fields() const;

    std::string get_default_sorting_field();

    static Option<bool> add_reference_helper_fields(nlohmann::json& document, const tsl::htrie_map<char, field>& schema,
                                                    const spp::sparse_hash_map<std::string, reference_pair>& reference_fields,
                                                    tsl::htrie_set<char>& object_reference_helper_fields);

    Option<doc_seq_id_t> to_doc(const std::string& json_str, nlohmann::json& document,
                                const index_operation_t& operation,
                                const DIRTY_VALUES dirty_values,
                                const std::string& id="");


    static uint32_t get_seq_id_from_key(const std::string & key);

    Option<bool> get_document_from_store(const std::string & seq_id_key, nlohmann::json & document, bool raw_doc = false) const;

    Option<bool> get_document_from_store(const uint32_t& seq_id, nlohmann::json & document, bool raw_doc = false) const;

    Option<uint32_t> index_in_memory(nlohmann::json & document, uint32_t seq_id,
                                     const index_operation_t op, const DIRTY_VALUES& dirty_values);

    static void remove_flat_fields(nlohmann::json& document);

    static void remove_reference_helper_fields(nlohmann::json& document);

    static Option<bool> prune_ref_doc(nlohmann::json& doc,
                                      const reference_filter_result_t& references,
                                      const tsl::htrie_set<char>& ref_include_fields_full,
                                      const tsl::htrie_set<char>& ref_exclude_fields_full,
                                      const bool& is_reference_array,
                                      const ref_include_exclude_fields& ref_include_exclude);

    static Option<bool> include_references(nlohmann::json& doc, const uint32_t& seq_id, Collection const *const collection,
                                           const std::map<std::string, reference_filter_result_t>& reference_filter_results,
                                           const std::vector<ref_include_exclude_fields>& ref_include_exclude_fields_vec);

    static Option<bool> prune_doc(nlohmann::json& doc, const tsl::htrie_set<char>& include_names,
                                  const tsl::htrie_set<char>& exclude_names, const std::string& parent_name = "",
                                  size_t depth = 0,
                                  const std::map<std::string, reference_filter_result_t>& reference_filter_results = {},
                                  Collection const *const collection = nullptr, const uint32_t& seq_id = 0,
                                  const std::vector<ref_include_exclude_fields>& ref_include_exclude_fields_vec = {});

    const Index* _get_index() const;

    bool facet_value_to_string(const facet &a_facet, const facet_count_t &facet_count, const nlohmann::json &document,
                               std::string &value) const;

    nlohmann::json get_facet_parent(const std::string& facet_field_name, const nlohmann::json& document) const;

    static void populate_result_kvs(Topster *topster, std::vector<std::vector<KV *>> &result_kvs, 
                    const spp::sparse_hash_map<uint64_t, uint32_t>& groups_processed, 
                    const std::vector<sort_by>& sort_by_fields);

    void batch_index(std::vector<index_record>& index_records, std::vector<std::string>& json_out, size_t &num_indexed,
                     const bool& return_doc, const bool& return_id, const size_t remote_embedding_batch_size = 200,
                     const size_t remote_embedding_timeout_ms = 60000, const size_t remote_embedding_num_tries = 2);

    bool is_exceeding_memory_threshold() const;

    void parse_search_query(const std::string &query, std::vector<std::string>& q_include_tokens,
                            std::vector<std::vector<std::string>>& q_exclude_tokens,
                            std::vector<std::vector<std::string>>& q_phrases,
                            const std::string& locale, const bool already_segmented, const std::string& stopword_set="") const;

    // PUBLIC OPERATIONS

    nlohmann::json get_summary_json() const;

    size_t batch_index_in_memory(std::vector<index_record>& index_records, const size_t remote_embedding_batch_size,
                                 const size_t remote_embedding_timeout_ms, const size_t remote_embedding_num_tries, const bool generate_embeddings);

    Option<nlohmann::json> add(const std::string & json_str,
                               const index_operation_t& operation=CREATE, const std::string& id="",
                               const DIRTY_VALUES& dirty_values=DIRTY_VALUES::COERCE_OR_REJECT);

    nlohmann::json add_many(std::vector<std::string>& json_lines, nlohmann::json& document,
                            const index_operation_t& operation=CREATE, const std::string& id="",
                            const DIRTY_VALUES& dirty_values=DIRTY_VALUES::COERCE_OR_REJECT,
                            const bool& return_doc=false, const bool& return_id=false,
                            const size_t remote_embedding_batch_size=200,
                            const size_t remote_embedding_timeout_ms=60000,
                            const size_t remote_embedding_num_tries=2);

    Option<nlohmann::json> update_matching_filter(const std::string& filter_query,
                                                  const std::string & json_str,
                                                  std::string& req_dirty_values,
                                                  const int batch_size = 1000);

    Option<bool> populate_include_exclude_fields_lk(const spp::sparse_hash_set<std::string>& include_fields,
                                                     const spp::sparse_hash_set<std::string>& exclude_fields,
                                                     tsl::htrie_set<char>& include_fields_full,
                                                     tsl::htrie_set<char>& exclude_fields_full) const;

    void do_housekeeping();

    Option<nlohmann::json> search(std::string query, const std::vector<std::string> & search_fields,
                                  const std::string & filter_query, const std::vector<std::string> & facet_fields,
                                  const std::vector<sort_by> & sort_fields, const std::vector<uint32_t>& num_typos,
                                  size_t per_page = 10, size_t page = 1,
                                  token_ordering token_order = FREQUENCY, const std::vector<bool>& prefixes = {true},
                                  size_t drop_tokens_threshold = Index::DROP_TOKENS_THRESHOLD,
                                  const spp::sparse_hash_set<std::string> & include_fields = spp::sparse_hash_set<std::string>(),
                                  const spp::sparse_hash_set<std::string> & exclude_fields = spp::sparse_hash_set<std::string>(),
                                  size_t max_facet_values=10,
                                  const std::string & simple_facet_query = "",
                                  const size_t snippet_threshold = 30,
                                  const size_t highlight_affix_num_tokens = 4,
                                  const std::string & highlight_full_fields = "",
                                  size_t typo_tokens_threshold = Index::TYPO_TOKENS_THRESHOLD,
                                  const std::string& pinned_hits_str="",
                                  const std::string& hidden_hits="",
                                  const std::vector<std::string>& group_by_fields={},
                                  size_t group_limit = 3,
                                  const std::string& highlight_start_tag="<mark>",
                                  const std::string& highlight_end_tag="</mark>",
                                  std::vector<uint32_t> raw_query_by_weights={},
                                  size_t limit_hits=UINT32_MAX,
                                  bool prioritize_exact_match=true,
                                  bool pre_segmented_query=false,
                                  bool enable_overrides=true,
                                  const std::string& highlight_fields="",
                                  const bool exhaustive_search = false,
                                  size_t search_stop_millis = 6000*1000,
                                  size_t min_len_1typo = 4,
                                  size_t min_len_2typo = 7,
                                  enable_t split_join_tokens = fallback,
                                  size_t max_candidates = 4,
                                  const std::vector<enable_t>& infixes = {off},
                                  const size_t max_extra_prefix = INT16_MAX,
                                  const size_t max_extra_suffix = INT16_MAX,
                                  const size_t facet_query_num_typos = 2,
                                  const size_t filter_curated_hits_option = 2,
                                  const bool prioritize_token_position = false,
                                  const std::string& vector_query_str = "",
                                  const bool enable_highlight_v1 = true,
                                  const uint64_t search_time_start_us = 0,
                                  const text_match_type_t match_type = max_score,
                                  const size_t facet_sample_percent = 100,
                                  const size_t facet_sample_threshold = 0,
                                  const size_t page_offset = 0,
                                  facet_index_type_t facet_index_type = HASH,
                                  const size_t remote_embedding_timeout_ms = 30000,
                                  const size_t remote_embedding_num_tries = 2,
                                  const std::string& stopwords_set="",
                                  const std::vector<std::string>& facet_return_parent = {},
                                  const std::vector<ref_include_exclude_fields>& ref_include_exclude_fields_vec = {},
                                  const std::string& drop_tokens_mode = "right_to_left",
                                  const bool prioritize_num_matching_fields = true,
                                  const bool group_missing_values = true,
                                  const bool converstaion = false,
                                  const std::string& conversation_model_id = "",
                                  std::string conversation_id = "",
                                  const std::string& override_tags_str = "",
                                  const std::string& voice_query = "") const;

    Option<bool> get_filter_ids(const std::string & filter_query, filter_result_t& filter_result) const;

    Option<bool> get_reference_filter_ids(const std::string& filter_query,
                                          filter_result_t& filter_result,
                                          const std::string& reference_field_name) const;

    Option<nlohmann::json> get(const std::string & id) const;

    Option<std::string> remove(const std::string & id, bool remove_from_store = true);

    Option<bool> remove_if_found(uint32_t seq_id, bool remove_from_store = true);

    size_t get_num_documents() const;

    DIRTY_VALUES parse_dirty_values_option(std::string& dirty_values) const;

    std::vector<char> get_symbols_to_index();

    std::vector<char> get_token_separators();

    std::string get_fallback_field_type();

    bool get_enable_nested_fields();

    std::shared_ptr<VQModel> get_vq_model();

    Option<bool> parse_facet_with_lock(const std::string& facet_field, std::vector<facet>& facets) const;

    Option<bool> parse_facet(const std::string& facet_field, std::vector<facet>& facets) const;

    Option<bool> compute_facet_infos_with_lock(const std::vector<facet>& facets, facet_query_t& facet_query,
                                               const size_t facet_query_num_typos,
                                               const uint32_t* all_result_ids, const size_t& all_result_ids_len,
                                               const std::vector<std::string>& group_by_fields,
                                               size_t group_limit, bool is_wildcard_no_filter_query,
                                               size_t max_candidates,
                                               std::vector<facet_info_t>& facet_infos, facet_index_type_t facet_index_type) const;

    Option<bool> do_facets_with_lock(std::vector<facet> & facets, facet_query_t & facet_query,
                                     bool estimate_facets, size_t facet_sample_percent,
                                     const std::vector<facet_info_t>& facet_infos,
                                     size_t group_limit, const std::vector<std::string>& group_by_fields,
                                     const bool group_missing_values,
                                     const uint32_t* result_ids, size_t results_size,
                                     int max_facet_count, bool is_wildcard_query,
                                     facet_index_type_t facet_index_type) const;

    // Override operations

    Option<uint32_t> add_override(const override_t & override, bool write_to_store = true);

    Option<uint32_t> remove_override(const std::string & id);

    std::map<std::string, override_t> get_overrides() {
        std::shared_lock lock(mutex);
        return overrides;
    };

    // synonym operations

    spp::sparse_hash_map<std::string, synonym_t> get_synonyms();

    bool get_synonym(const std::string& id, synonym_t& synonym);

    Option<bool> add_synonym(const nlohmann::json& syn_json, bool write_to_store = true);

    Option<bool> remove_synonym(const std::string & id);

    void synonym_reduction(const std::vector<std::string>& tokens,
                           std::vector<std::vector<std::string>>& results) const;

    SynonymIndex* get_synonym_index();

    spp::sparse_hash_map<std::string, reference_pair> get_reference_fields();

    // highlight ops

    static void highlight_text(const string& highlight_start_tag, const string& highlight_end_tag,
                   const string& text, const std::map<size_t, size_t>& token_offsets,
                   size_t snippet_end_offset,
                   std::vector<std::string>& matched_tokens, std::map<size_t, size_t>::iterator& offset_it,
                   std::stringstream& highlighted_text,
                   const uint8_t* index_symbols,
                   size_t snippet_start_offset) ;

    void process_highlight_fields(const std::vector<search_field_t>& search_fields,
                                  const std::vector<std::string>& raw_search_fields,
                                  const tsl::htrie_set<char>& include_fields,
                                  const tsl::htrie_set<char>& exclude_fields,
                                  const std::vector<std::string>& highlight_field_names,
                                  const std::vector<std::string>& highlight_full_field_names,
                                  const std::vector<enable_t>& infixes,
                                  std::vector<std::string>& q_tokens,
                                  const tsl::htrie_map<char, token_leaf>& qtoken_set,
                                  std::vector<highlight_field_t>& highlight_items) const;

    static void copy_highlight_doc(std::vector<highlight_field_t>& hightlight_items,
                                   const bool nested_fields_enabled,
                                   const nlohmann::json& src,
                                   nlohmann::json& dst);

    Option<bool> alter(nlohmann::json& alter_payload);

    void process_search_field_weights(const std::vector<search_field_t>& search_fields,
                                      std::vector<uint32_t>& query_by_weights,
                                      std::vector<search_field_t>& weighted_search_fields) const;

    Option<bool> truncate_after_top_k(const std::string& field_name, size_t k);

    void reference_populate_sort_mapping(int* sort_order, std::vector<size_t>& geopoint_indices,
                                         std::vector<sort_by>& sort_fields_std,
                                         std::array<spp::sparse_hash_map<uint32_t, int64_t, Hasher32>*, 3>& field_values) const;

    int64_t reference_string_sort_score(const std::string& field_name, const uint32_t& seq_id) const;

    bool is_referenced_in(const std::string& collection_name) const;

    void add_referenced_in(const reference_pair& pair);

    void add_referenced_ins(const std::set<reference_pair>& pairs);

    void add_referenced_in(const std::string& collection_name, const std::string& field_name);

    Option<std::string> get_referenced_in_field_with_lock(const std::string& collection_name) const;

    Option<bool> get_related_ids_with_lock(const std::string& field_name, const uint32_t& seq_id,
                                           std::vector<uint32_t>& result) const;

    Option<uint32_t> get_sort_index_value_with_lock(const std::string& field_name, const uint32_t& seq_id) const;

    static void hide_credential(nlohmann::json& json, const std::string& credential_name);

    friend class filter_result_iterator_t;

    std::shared_mutex& get_lifecycle_mutex();

    void expand_search_query(const string& raw_query, size_t offset, size_t total, const search_args* search_params,
                             const std::vector<std::vector<KV*>>& result_group_kvs,
                             const std::vector<std::string>& raw_search_fields, string& first_q) const;
};

template<class T>
bool Collection::highlight_nested_field(const nlohmann::json& hdoc, nlohmann::json& hobj,
                                        std::vector<std::string>& path_parts, size_t path_index,
                                        bool is_arr_obj_ele, int array_index, T func) {
    if(path_index == path_parts.size()) {
        func(hobj, is_arr_obj_ele, array_index);
        return true;
    }

    const std::string& fragment = path_parts[path_index];
    const auto& it = hobj.find(fragment);

    if(it != hobj.end()) {
        if(it.value().is_array()) {
            bool resolved = false;
            for(size_t i = 0; i < it.value().size(); i++) {
                auto& h_ele = it.value().at(i);
                is_arr_obj_ele = is_arr_obj_ele || h_ele.is_object();
                resolved = highlight_nested_field(hdoc, h_ele, path_parts, path_index + 1,
                                                  is_arr_obj_ele, i, func) || resolved;
            }
            return resolved;
        } else {
            return highlight_nested_field(hdoc, it.value(), path_parts, path_index + 1, is_arr_obj_ele, 0, func);
        }
    } {
        return false;
    }
}

