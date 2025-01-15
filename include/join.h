#pragma once

#include "option.h"
#include "json.hpp"
#include "tsl/htrie_map.h"
#include "field.h"
#include "tsl/htrie_set.h"
#include "filter_result_iterator.h"

struct reference_info_t {
    std::string collection;
    std::string field;
    bool is_async;

    std::string referenced_field_name;

    reference_info_t(std::string collection, std::string field, bool is_async, std::string referenced_field_name = "") :
            collection(std::move(collection)), field(std::move(field)), is_async(is_async),
            referenced_field_name(std::move(referenced_field_name)) {}

    bool operator < (const reference_info_t& other) const noexcept {
        if (collection == other.collection) {
            return field < other.field;
        }

        return collection < other.collection;
    }
};

struct ref_include_collection_names_t {
    std::set<std::string> collection_names;
    ref_include_collection_names_t* nested_include = nullptr;

    ~ref_include_collection_names_t() {
        delete nested_include;
    }
};

class Join {
public:

    static Option<bool> add_reference_helper_fields(nlohmann::json& document,
                                                    const tsl::htrie_map<char, field>& schema,
                                                    const spp::sparse_hash_map<std::string, reference_info_t>& reference_fields,
                                                    tsl::htrie_set<char>& object_reference_helper_fields,
                                                    const bool& is_update);

    static Option<bool> prune_ref_doc(nlohmann::json& doc,
                                      const reference_filter_result_t& references,
                                      const tsl::htrie_set<char>& ref_include_fields_full,
                                      const tsl::htrie_set<char>& ref_exclude_fields_full,
                                      const bool& is_reference_array,
                                      const ref_include_exclude_fields& ref_include_exclude);

    static Option<bool> include_references(nlohmann::json& doc, const uint32_t& seq_id, Collection *const collection,
                                           const std::map<std::string, reference_filter_result_t>& reference_filter_results,
                                           const std::vector<ref_include_exclude_fields>& ref_include_exclude_fields_vec,
                                           const nlohmann::json& original_doc);

    static Option<bool> parse_reference_filter(const std::string& filter_query, std::queue<std::string>& tokens, size_t& index,
                                               std::set<std::string>& ref_collection_names);

    static Option<bool> split_reference_include_exclude_fields(const std::string& include_exclude_fields,
                                                               size_t& index, std::string& token);

    static void get_reference_collection_names(const std::string& filter_query, ref_include_collection_names_t*& ref_include);

    // Separate out the reference includes and excludes into `ref_include_exclude_fields_vec`.
    static Option<bool> initialize_ref_include_exclude_fields_vec(const std::string& filter_query,
                                                                  std::vector<std::string>& include_fields_vec,
                                                                  std::vector<std::string>& exclude_fields_vec,
                                                                  std::vector<ref_include_exclude_fields>& ref_include_exclude_fields_vec);

    [[nodiscard]] static bool merge_join_conditions(string& embedded_filter, string& query_filter);

    static Option<bool> single_value_filter_query(nlohmann::json& document, const std::string& field_name,
                                                  const std::string& ref_field_type, std::string& filter_value,
                                                  const bool& is_reference_value = true);
};