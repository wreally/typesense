#pragma once

#include <string>
#include <vector>
#include "option.h"

class Collection;

struct vector_query_t {
    std::string field_name;
    size_t k = 0;
    size_t flat_search_cutoff = 0;
    float distance_threshold = 2.01;
    std::vector<float> values;
    bool is_boolean = false;

    uint32_t seq_id = 0;
    bool query_doc_given = false;
    float alpha = 0.3;

    uint32_t ef = 10;

    std::vector<std::string> queries;
    std::vector<float> query_weights;

    void _reset() {
        // used for testing only
        field_name.clear();
        k = 0;
        distance_threshold = 2.01;
        values.clear();
        seq_id = 0;
        query_doc_given = false;
    }
};

class VectorQueryOps {
public:
    static Option<bool> parse_vector_query_str(const std::string& vector_query_str, vector_query_t& vector_query,
                                               const bool is_wildcard_query,
                                               const Collection* coll,
                                               const bool allow_empty_query);
};