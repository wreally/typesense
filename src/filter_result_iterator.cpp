#include <memory>
#include <queue>
#include <id_list.h>
#include <s2/s2point.h>
#include <s2/s2latlng.h>
#include <s2/s2region_term_indexer.h>
#include <s2/s2cap.h>
#include <s2/s2earth.h>
#include <s2/s2loop.h>
#include <s2/s2builder.h>
#include <timsort.hpp>
#include "filter_result_iterator.h"
#include "index.h"
#include "posting.h"
#include "collection_manager.h"

void copy_references_helper(const std::map<std::string, reference_filter_result_t>* from,
                            std::map<std::string, reference_filter_result_t>*& to, const uint32_t& count) {
    if (from == nullptr) {
        return;
    }

    to = new std::map<std::string, reference_filter_result_t>[count] {};
    for (uint32_t i = 0; i < count; i++) {
        if (from[i].empty()) {
            continue;
        }

        auto& ref = to[i];
        ref.insert(from[i].begin(), from[i].end());
    }
}

void reference_filter_result_t::copy_references(const reference_filter_result_t& from, reference_filter_result_t& to) {
    return copy_references_helper(from.coll_to_references, to.coll_to_references, from.count);
}

void filter_result_t::copy_references(const filter_result_t& from, filter_result_t& to) {
    return copy_references_helper(from.coll_to_references, to.coll_to_references, from.count);
}

void filter_result_t::and_filter_results(const filter_result_t& a, const filter_result_t& b, filter_result_t& result) {
    auto lenA = a.count, lenB = b.count;
    if (lenA == 0 || lenB == 0) {
        return;
    }

    result.docs = new uint32_t[std::min(lenA, lenB)];

    auto A = a.docs, B = b.docs, out = result.docs;
    const uint32_t *endA = A + lenA;
    const uint32_t *endB = B + lenB;

    if (a.coll_to_references != nullptr || b.coll_to_references != nullptr) {
        result.coll_to_references = new std::map<std::string, reference_filter_result_t>[std::min(lenA, lenB)] {};
    }

    while (true) {
        while (*A < *B) {
            SKIP_FIRST_COMPARE:
            if (++A == endA) {
                result.count = out - result.docs;
                return;
            }
        }
        while (*A > *B) {
            if (++B == endB) {
                result.count = out - result.docs;
                return;
            }
        }
        if (*A == *B) {
            *out = *A;

            if (result.coll_to_references != nullptr) {
                // Copy the references of the document from every collection into result.
                auto& ref = result.coll_to_references[out - result.docs];
                if (a.coll_to_references != nullptr) {
                    ref.insert(a.coll_to_references[A - a.docs].begin(), a.coll_to_references[A - a.docs].end());
                }
                if (b.coll_to_references != nullptr) {
                    ref.insert(b.coll_to_references[B - b.docs].begin(), b.coll_to_references[B - b.docs].end());
                }
            }

            out++;

            if (++A == endA || ++B == endB) {
                result.count = out - result.docs;
                return;
            }
        } else {
            goto SKIP_FIRST_COMPARE;
        }
    }
}

void filter_result_t::or_filter_results(const filter_result_t& a, const filter_result_t& b, filter_result_t& result) {
    if (a.count == 0 && b.count == 0) {
        return;
    }

    // If either one of a or b does not have any matches, copy other into result.
    if (a.count == 0) {
        result = b;
        return;
    }
    if (b.count == 0) {
        result = a;
        return;
    }

    size_t indexA = 0, indexB = 0, res_index = 0, lenA = a.count, lenB = b.count;
    result.docs = new uint32_t[lenA + lenB];

    if (a.coll_to_references != nullptr || b.coll_to_references != nullptr) {
        result.coll_to_references = new std::map<std::string, reference_filter_result_t>[lenA + lenB] {};
    }

    while (indexA < lenA && indexB < lenB) {
        if (a.docs[indexA] < b.docs[indexB]) {
            // check for duplicate
            if (res_index == 0 || result.docs[res_index - 1] != a.docs[indexA]) {
                result.docs[res_index] = a.docs[indexA];
                res_index++;
            }

            if (a.coll_to_references != nullptr) {
                // Copy references of the last result document from every collection in a.
                auto &ref = result.coll_to_references[res_index - 1];
                ref.insert(a.coll_to_references[indexA].begin(), a.coll_to_references[indexA].end());
            }

            indexA++;
        } else {
            if (res_index == 0 || result.docs[res_index - 1] != b.docs[indexB]) {
                result.docs[res_index] = b.docs[indexB];
                res_index++;
            }

            if (b.coll_to_references != nullptr) {
                auto &ref = result.coll_to_references[res_index - 1];
                ref.insert(b.coll_to_references[indexB].begin(), b.coll_to_references[indexB].end());
            }

            indexB++;
        }
    }

    while (indexA < lenA) {
        if (res_index == 0 || result.docs[res_index - 1] != a.docs[indexA]) {
            result.docs[res_index] = a.docs[indexA];
            res_index++;
        }

        if (a.coll_to_references != nullptr) {
            auto &ref = result.coll_to_references[res_index - 1];
            ref.insert(a.coll_to_references[indexA].begin(), a.coll_to_references[indexA].end());
        }

        indexA++;
    }

    while (indexB < lenB) {
        if(res_index == 0 || result.docs[res_index - 1] != b.docs[indexB]) {
            result.docs[res_index] = b.docs[indexB];
            res_index++;
        }

        if (b.coll_to_references != nullptr) {
            auto &ref = result.coll_to_references[res_index - 1];
            ref.insert(b.coll_to_references[indexB].begin(), b.coll_to_references[indexB].end());
        }

        indexB++;
    }

    result.count = res_index;

    if (res_index == lenA + lenB) {
        return;
    }

    // shrink fit
    auto out = new uint32_t[res_index];
    memcpy(out, result.docs, res_index * sizeof(uint32_t));
    delete[] result.docs;
    result.docs = out;

    if (result.coll_to_references == nullptr) {
        return;
    }

    auto out_references = new std::map<std::string, reference_filter_result_t>[res_index] {};
    for (uint32_t i = 0; i < res_index; i++) {
        auto& ref = out_references[i];
        ref.insert(result.coll_to_references[i].begin(), result.coll_to_references[i].end());
    }

    delete[] result.coll_to_references;
    result.coll_to_references = out_references;
}

void filter_result_iterator_t::and_filter_iterators() {
    while (left_it->validity && right_it->validity) {
        while (left_it->seq_id < right_it->seq_id) {
            left_it->skip_to(right_it->seq_id);
            if (!left_it->validity) {
                validity = invalid;
                return;
            }
        }

        while (left_it->seq_id > right_it->seq_id) {
            right_it->skip_to(left_it->seq_id);
            if (!right_it->validity) {
                validity = invalid;
                return;
            }
        }

        if (left_it->seq_id == right_it->seq_id) {
            seq_id = left_it->seq_id;
            reference.clear();

            for (const auto& item: left_it->reference) {
                reference[item.first] = item.second;
            }
            for (const auto& item: right_it->reference) {
                reference[item.first] = item.second;
            }

            return;
        }
    }

    validity = invalid;
}

void filter_result_iterator_t::or_filter_iterators() {
    if (left_it->validity && right_it->validity) {
        if (left_it->seq_id < right_it->seq_id) {
            seq_id = left_it->seq_id;
            reference.clear();

            for (const auto& item: left_it->reference) {
                reference[item.first] = item.second;
            }

            return;
        }

        if (left_it->seq_id > right_it->seq_id) {
            seq_id = right_it->seq_id;
            reference.clear();

            for (const auto& item: right_it->reference) {
                reference[item.first] = item.second;
            }

            return;
        }

        seq_id = left_it->seq_id;
        reference.clear();

        for (const auto& item: left_it->reference) {
            reference[item.first] = item.second;
        }
        for (const auto& item: right_it->reference) {
            reference[item.first] = item.second;
        }

        return;
    }

    if (left_it->validity) {
        seq_id = left_it->seq_id;
        reference.clear();

        for (const auto& item: left_it->reference) {
            reference[item.first] = item.second;
        }

        return;
    }

    if (right_it->validity) {
        seq_id = right_it->seq_id;
        reference.clear();

        for (const auto& item: right_it->reference) {
            reference[item.first] = item.second;
        }

        return;
    }

    validity = invalid;
}

void filter_result_iterator_t::advance_string_filter_token_iterators() {
    for (uint32_t i = 0; i < posting_list_iterators.size(); i++) {
        auto& filter_value_tokens = posting_list_iterators[i];

        if (!filter_value_tokens[0].valid() || filter_value_tokens[0].id() != seq_id) {
            continue;
        }

        for (auto& iter: filter_value_tokens) {
            if (iter.valid()) {
                iter.next();
            }
        }
    }
}

void filter_result_iterator_t::get_string_filter_next_match(const bool& field_is_array) {
    // If none of the filter value iterators are valid, mark this node as invalid.
    bool one_is_valid = false;

    // Since we do OR between filter values, the lowest seq_id id from all is selected.
    uint32_t lowest_id = UINT32_MAX;

    if (filter_node->filter_exp.comparators[0] == EQUALS || filter_node->filter_exp.comparators[0] == NOT_EQUALS) {
        bool exact_match_found = false;
        switch (posting_list_iterators.size()) {
            case 1:
                while(true) {
                    // Perform AND between tokens of a filter value.
                    posting_list_t::intersect(posting_list_iterators[0], one_is_valid);

                    if (!one_is_valid) {
                        break;
                    }

                    if (posting_list_t::has_exact_match(posting_list_iterators[0], field_is_array)) {
                        exact_match_found = true;
                        break;
                    } else {
                        // Keep advancing token iterators till exact match is not found.
                        for (auto& iter: posting_list_iterators[0]) {
                            if (!iter.valid()) {
                                break;
                            }

                            iter.next();
                        }
                    }
                }

                if (one_is_valid && exact_match_found) {
                    lowest_id = posting_list_iterators[0][0].id();
                }
            break;

            default :
                for (auto& filter_value_tokens : posting_list_iterators) {
                    bool tokens_iter_is_valid;
                    while(true) {
                        // Perform AND between tokens of a filter value.
                        posting_list_t::intersect(filter_value_tokens, tokens_iter_is_valid);

                        if (!tokens_iter_is_valid) {
                            break;
                        }

                        if (posting_list_t::has_exact_match(filter_value_tokens, field_is_array)) {
                            exact_match_found = true;
                            break;
                        } else {
                            // Keep advancing token iterators till exact match is not found.
                            for (auto &iter: filter_value_tokens) {
                                if (!iter.valid()) {
                                    break;
                                }

                                iter.next();
                            }
                        }
                    }

                    one_is_valid = tokens_iter_is_valid || one_is_valid;

                    if (tokens_iter_is_valid && exact_match_found && filter_value_tokens[0].id() < lowest_id) {
                        lowest_id = filter_value_tokens[0].id();
                    }
                }
        }
    } else {
        switch (posting_list_iterators.size()) {
            case 1:
                // Perform AND between tokens of a filter value.
                posting_list_t::intersect(posting_list_iterators[0], one_is_valid);

                if (one_is_valid) {
                    lowest_id = posting_list_iterators[0][0].id();
                }
            break;

            default:
                for (auto& filter_value_tokens : posting_list_iterators) {
                    // Perform AND between tokens of a filter value.
                    bool tokens_iter_is_valid;
                    posting_list_t::intersect(filter_value_tokens, tokens_iter_is_valid);

                    one_is_valid = tokens_iter_is_valid || one_is_valid;

                    if (tokens_iter_is_valid && filter_value_tokens[0].id() < lowest_id) {
                        lowest_id = filter_value_tokens[0].id();
                    }
                }
        }
    }

    if (one_is_valid) {
        seq_id = lowest_id;
    }

    validity = one_is_valid ? valid : invalid;
}

void filter_result_iterator_t::next() {
    if (validity != valid) {
        return;
    }

    if (timeout_info != nullptr && is_timed_out()) {
        return;
    }

    // No need to traverse iterator tree if there's only one filter or compute_result() has been called.
    if (is_filter_result_initialized) {
        if (++result_index >= filter_result.count) {
            validity = invalid;
            return;
        }

        seq_id = filter_result.docs[result_index];
        reference.clear();
        if (filter_result.coll_to_references != nullptr) {
            auto& ref = filter_result.coll_to_references[result_index];
            reference.insert(ref.begin(), ref.end());
        }

        return;
    }

    if (filter_node->isOperator) {
        // Advance the subtrees and then apply operators to arrive at the next valid doc.
        if (filter_node->filter_operator == AND) {
            left_it->next();
            right_it->next();
            and_filter_iterators();
        } else {
            if (left_it->seq_id == seq_id && right_it->seq_id == seq_id) {
                left_it->next();
                right_it->next();
            } else if (left_it->seq_id == seq_id) {
                left_it->next();
            } else {
                right_it->next();
            }

            or_filter_iterators();
        }

        return;
    }

    const filter a_filter = filter_node->filter_exp;

    if (!index->field_is_indexed(a_filter.field_name)) {
        validity = invalid;
        return;
    }

    field f = index->search_schema.at(a_filter.field_name);

    if (f.is_string()) {
        if (filter_node->filter_exp.apply_not_equals) {
            if (++seq_id < result_index) {
                return;
            }

            uint32_t previous_match;
            do {
                previous_match = seq_id;
                advance_string_filter_token_iterators();
                get_string_filter_next_match(f.is_array());
            } while (validity && previous_match + 1 == seq_id);

            if (!validity) {
                // We've reached the end of the index, no possible matches pending.
                if (previous_match >= index->seq_ids->last_id()) {
                    return;
                }

                validity = valid;
                result_index = index->seq_ids->last_id() + 1;
                seq_id = previous_match + 1;
                return;
            }

            result_index = seq_id;
            seq_id = previous_match + 1;
            return;
        }

        advance_string_filter_token_iterators();
        get_string_filter_next_match(f.is_array());

        return;
    }
}

void numeric_not_equals_filter(num_tree_t* const num_tree,
                               const int64_t value,
                               uint32_t*&& all_ids,
                               uint32_t&& all_ids_length,
                               uint32_t*& result_ids,
                               size_t& result_ids_len) {
    uint32_t* to_exclude_ids = nullptr;
    size_t to_exclude_ids_len = 0;

    num_tree->search(EQUALS, value, &to_exclude_ids, to_exclude_ids_len);

    result_ids_len = ArrayUtils::exclude_scalar(all_ids, all_ids_length, to_exclude_ids, to_exclude_ids_len, &result_ids);

    delete[] all_ids;
    delete[] to_exclude_ids;
}

void apply_not_equals(uint32_t*&& all_ids,
                      uint32_t&& all_ids_length,
                      uint32_t*& result_ids,
                      uint32_t& result_ids_len) {

    uint32_t* to_include_ids = nullptr;
    size_t to_include_ids_len = 0;

    to_include_ids_len = ArrayUtils::exclude_scalar(all_ids, all_ids_length, result_ids,
                                                    result_ids_len, &to_include_ids);

    delete[] all_ids;
    delete[] result_ids;

    result_ids = to_include_ids;
    result_ids_len = to_include_ids_len;
}

void filter_result_iterator_t::get_string_filter_first_match(const bool& field_is_array) {
    get_string_filter_next_match(field_is_array);

    if (filter_node->filter_exp.apply_not_equals) {
        // filter didn't match any id. So by applying not equals, every id in the index is a match.
        if (!validity) {
            validity = valid;
            seq_id = 0;
            result_index = index->seq_ids->last_id() + 1;
            return;
        }

        // [0, seq_id) are a match for not equals.
        if (seq_id > 0) {
            result_index = seq_id;
            seq_id = 0;
            return;
        }

        // Keep ignoring the consecutive matches.
        uint32_t previous_match;
        do {
            previous_match = seq_id;
            advance_string_filter_token_iterators();
            get_string_filter_next_match(field_is_array);
        } while (validity && previous_match + 1 == seq_id);

        if (!validity) {
            // filter matched all the ids in the index. So for not equals, there's no match.
            if (previous_match >= index->seq_ids->last_id()) {
                return;
            }

            validity = valid;
            result_index = index->seq_ids->last_id() + 1;
            seq_id = previous_match + 1;
            return;
        }

        result_index = seq_id;
        seq_id = previous_match + 1;
    }
}

void filter_result_iterator_t::init() {
    if (filter_node == nullptr) {
        return;
    }

    if (filter_node->isOperator) {
        if (filter_node->filter_operator == AND) {
            and_filter_iterators();
            approx_filter_ids_length = std::min(left_it->approx_filter_ids_length, right_it->approx_filter_ids_length);
        } else {
            or_filter_iterators();
            approx_filter_ids_length = std::max(left_it->approx_filter_ids_length, right_it->approx_filter_ids_length);
        }

        // Rearranging the subtree in hope to reduce computation if/when compute_result() is called.
        if (left_it->approx_filter_ids_length > right_it->approx_filter_ids_length) {
            std::swap(left_it, right_it);
        }

        return;
    }

    const filter a_filter = filter_node->filter_exp;

    bool is_referenced_filter = !a_filter.referenced_collection_name.empty();
    if (is_referenced_filter) {
        // Apply filter on referenced collection and get the sequence ids of current collection from the filtered documents.
        auto& cm = CollectionManager::get_instance();
        auto const& ref_collection_name = a_filter.referenced_collection_name;
        auto ref_collection = cm.get_collection(ref_collection_name);
        if (ref_collection == nullptr) {
            status = Option<bool>(400, "Referenced collection `" + ref_collection_name + "` not found.");
            validity = invalid;
            return;
        }

        auto coll = cm.get_collection(collection_name);
        bool is_referenced = coll->referenced_in.count(ref_collection_name) > 0,
                has_reference = ref_collection->is_referenced_in(collection_name);
        if (!is_referenced && !has_reference) {
            status = Option<bool>(400, "Failed to join on `" + ref_collection_name + "`: No reference field found.");
            validity = invalid;
            return;
        }

        if (is_referenced) {
            auto const& field_name = coll->referenced_in.at(ref_collection_name);
            auto reference_filter_op = ref_collection->get_reference_filter_ids(a_filter.field_name,
                                                                                filter_result,
                                                                                field_name);
            if (!reference_filter_op.ok()) {
                status = Option<bool>(400, "Failed to join on `" + a_filter.referenced_collection_name
                                           + "` collection: " + reference_filter_op.error());
                validity = invalid;
                return;
            }
        } else if (has_reference) {
            // Get the doc ids of reference collection matching the filter then apply filter on the current collection's
            // reference helper field.
            filter_result_t result;
            auto reference_filter_op = ref_collection->get_filter_ids(a_filter.field_name, result);
            if (!reference_filter_op.ok()) {
                status = Option<bool>(400, "Failed to join on `" + a_filter.referenced_collection_name
                                           + "` collection: " + reference_filter_op.error());
                validity = invalid;
                return;
            }

            auto get_reference_field_op = ref_collection->get_referenced_in_field_with_lock(collection_name);
            if (!get_reference_field_op.ok()) {
                status = Option<bool>(get_reference_field_op.code(), get_reference_field_op.error());
                validity = invalid;
                return;
            }

            auto const& reference_helper_field_name = get_reference_field_op.get();
            auto op = index->do_filtering_with_reference_ids(reference_helper_field_name, ref_collection_name,
                                                             std::move(result));
            if (!op.ok()) {
                status = Option<bool>(op.code(), op.error());
                validity = invalid;
                return;
            }

            filter_result = op.get();
        }

        if (filter_result.count == 0) {
            validity = invalid;
            return;
        }

        seq_id = filter_result.docs[result_index];
        if (filter_result.coll_to_references != nullptr) {
            auto& ref = filter_result.coll_to_references[result_index];
            reference.insert(ref.begin(), ref.end());
        }

        is_filter_result_initialized = true;
        approx_filter_ids_length = filter_result.count;
        return;
    }

    if (a_filter.field_name == "id") {
        // we handle `ids` separately
        std::vector<uint32_t> result_ids;
        for (const auto& id_str : a_filter.values) {
            result_ids.push_back(std::stoul(id_str));
        }

        std::sort(result_ids.begin(), result_ids.end());

        filter_result.count = result_ids.size();
        filter_result.docs = new uint32_t[result_ids.size()];
        std::copy(result_ids.begin(), result_ids.end(), filter_result.docs);

        if (a_filter.apply_not_equals) {
            apply_not_equals(index->seq_ids->uncompress(), index->seq_ids->num_ids(),
                             filter_result.docs, filter_result.count);
        }

        if (filter_result.count == 0) {
            validity = invalid;
            return;
        }

        seq_id = filter_result.docs[result_index];
        is_filter_result_initialized = true;
        approx_filter_ids_length = filter_result.count;
        return;
    }

    if (!index->field_is_indexed(a_filter.field_name)) {
        status = Option<bool>(400, "Cannot filter on non-indexed field `" + a_filter.field_name + "`.");
        validity = invalid;
        return;
    }

    field f = index->search_schema.at(a_filter.field_name);

    if (f.is_integer()) {
        if (f.range_index) {
            auto const& trie = index->range_index.at(a_filter.field_name);

            for (size_t fi = 0; fi < a_filter.values.size(); fi++) {
                const std::string& filter_value = a_filter.values[fi];
                auto const& value = (int64_t)std::stol(filter_value);

                if (a_filter.comparators[fi] == RANGE_INCLUSIVE && fi+1 < a_filter.values.size()) {
                    const std::string& next_filter_value = a_filter.values[fi + 1];
                    auto const& range_end_value = (int64_t)std::stol(next_filter_value);
                    trie->search_range(value, true, range_end_value, true, filter_result.docs, filter_result.count);
                    fi++;
                } else if (a_filter.comparators[fi] == EQUALS) {
                    trie->search_equal_to(value, filter_result.docs, filter_result.count);
                } else if (a_filter.comparators[fi] == NOT_EQUALS) {
                    uint32_t* to_exclude_ids = nullptr;
                    uint32_t to_exclude_ids_len = 0;
                    trie->search_equal_to(value, to_exclude_ids, to_exclude_ids_len);

                    auto all_ids = index->seq_ids->uncompress();
                    filter_result.count = ArrayUtils::exclude_scalar(all_ids, index->seq_ids->num_ids(),
                                                                     to_exclude_ids, to_exclude_ids_len, &filter_result.docs);

                    delete[] all_ids;
                    delete[] to_exclude_ids;
                } else if (a_filter.comparators[fi] == GREATER_THAN || a_filter.comparators[fi] == GREATER_THAN_EQUALS) {
                    trie->search_greater_than(value, a_filter.comparators[fi] == GREATER_THAN_EQUALS,
                                              filter_result.docs, filter_result.count);
                } else if (a_filter.comparators[fi] == LESS_THAN || a_filter.comparators[fi] == LESS_THAN_EQUALS) {
                    trie->search_less_than(value, a_filter.comparators[fi] == LESS_THAN_EQUALS,
                                           filter_result.docs, filter_result.count);
                }
            }
        } else {
            auto num_tree = index->numerical_index.at(a_filter.field_name);

            for (size_t fi = 0; fi < a_filter.values.size(); fi++) {
                const std::string& filter_value = a_filter.values[fi];
                int64_t value = (int64_t)std::stol(filter_value);

                size_t result_size = filter_result.count;
                if (a_filter.comparators[fi] == RANGE_INCLUSIVE && fi+1 < a_filter.values.size()) {
                    const std::string& next_filter_value = a_filter.values[fi + 1];
                    auto const range_end_value = (int64_t)std::stol(next_filter_value);
                    num_tree->range_inclusive_search(value, range_end_value, &filter_result.docs, result_size);
                    fi++;
                } else if (a_filter.comparators[fi] == NOT_EQUALS) {
                    numeric_not_equals_filter(num_tree, value,
                                              index->seq_ids->uncompress(), index->seq_ids->num_ids(),
                                              filter_result.docs, result_size);
                } else {
                    num_tree->search(a_filter.comparators[fi], value, &filter_result.docs, result_size);
                }

                filter_result.count = result_size;
            }
        }

        if (a_filter.apply_not_equals) {
            apply_not_equals(index->seq_ids->uncompress(), index->seq_ids->num_ids(),
                             filter_result.docs, filter_result.count);
        }

        if (filter_result.count == 0) {
            validity = invalid;
            return;
        }

        seq_id = filter_result.docs[result_index];
        is_filter_result_initialized = true;
        approx_filter_ids_length = filter_result.count;
        return;
    } else if (f.is_float()) {
        if (f.range_index) {
            auto const& trie = index->range_index.at(a_filter.field_name);

            for (size_t fi = 0; fi < a_filter.values.size(); fi++) {
                const std::string& filter_value = a_filter.values[fi];
                float value = (float)std::atof(filter_value.c_str());
                int64_t float_int64 = Index::float_to_int64_t(value);

                if (a_filter.comparators[fi] == RANGE_INCLUSIVE && fi+1 < a_filter.values.size()) {
                    const std::string& next_filter_value = a_filter.values[fi + 1];
                    int64_t range_end_value = Index::float_to_int64_t((float) std::atof(next_filter_value.c_str()));
                    trie->search_range(float_int64, true, range_end_value, true, filter_result.docs, filter_result.count);
                    fi++;
                } else if (a_filter.comparators[fi] == EQUALS) {
                    trie->search_equal_to(float_int64, filter_result.docs, filter_result.count);
                } else if (a_filter.comparators[fi] == NOT_EQUALS) {
                    uint32_t* to_exclude_ids = nullptr;
                    uint32_t to_exclude_ids_len = 0;
                    trie->search_equal_to(float_int64, to_exclude_ids, to_exclude_ids_len);

                    auto all_ids = index->seq_ids->uncompress();
                    filter_result.count = ArrayUtils::exclude_scalar(all_ids, index->seq_ids->num_ids(),
                                                                     to_exclude_ids, to_exclude_ids_len, &filter_result.docs);

                    delete[] all_ids;
                    delete[] to_exclude_ids;
                } else if (a_filter.comparators[fi] == GREATER_THAN || a_filter.comparators[fi] == GREATER_THAN_EQUALS) {
                    trie->search_greater_than(float_int64, a_filter.comparators[fi] == GREATER_THAN_EQUALS,
                                              filter_result.docs, filter_result.count);
                } else if (a_filter.comparators[fi] == LESS_THAN || a_filter.comparators[fi] == LESS_THAN_EQUALS) {
                    trie->search_less_than(float_int64, a_filter.comparators[fi] == LESS_THAN_EQUALS,
                                           filter_result.docs, filter_result.count);
                }
            }
        } else {
            auto num_tree = index->numerical_index.at(a_filter.field_name);

            for (size_t fi = 0; fi < a_filter.values.size(); fi++) {
                const std::string& filter_value = a_filter.values[fi];
                float value = (float)std::atof(filter_value.c_str());
                int64_t float_int64 = Index::float_to_int64_t(value);

                size_t result_size = filter_result.count;
                if (a_filter.comparators[fi] == RANGE_INCLUSIVE && fi+1 < a_filter.values.size()) {
                    const std::string& next_filter_value = a_filter.values[fi+1];
                    int64_t range_end_value = Index::float_to_int64_t((float) std::atof(next_filter_value.c_str()));
                    num_tree->range_inclusive_search(float_int64, range_end_value, &filter_result.docs, result_size);
                    fi++;
                } else if (a_filter.comparators[fi] == NOT_EQUALS) {
                    numeric_not_equals_filter(num_tree, float_int64,
                                              index->seq_ids->uncompress(), index->seq_ids->num_ids(),
                                              filter_result.docs, result_size);
                } else {
                    num_tree->search(a_filter.comparators[fi], float_int64, &filter_result.docs, result_size);
                }

                filter_result.count = result_size;
            }
        }

        if (a_filter.apply_not_equals) {
            apply_not_equals(index->seq_ids->uncompress(), index->seq_ids->num_ids(),
                             filter_result.docs, filter_result.count);
        }

        if (filter_result.count == 0) {
            validity = invalid;
            return;
        }

        seq_id = filter_result.docs[result_index];
        is_filter_result_initialized = true;
        approx_filter_ids_length = filter_result.count;
        return;
    } else if (f.is_bool()) {
        if (f.range_index) {

            auto const& trie = index->range_index.at(a_filter.field_name);

            size_t value_index = 0;
            for (const std::string& filter_value : a_filter.values) {
                int64_t bool_int64 = (filter_value == "1") ? 1 : 0;

                if (a_filter.comparators[value_index] == EQUALS) {
                    trie->search_equal_to(bool_int64, filter_result.docs, filter_result.count);
                } else if (a_filter.comparators[value_index] == NOT_EQUALS) {
                    uint32_t* to_exclude_ids = nullptr;
                    uint32_t to_exclude_ids_len = 0;
                    trie->search_equal_to(bool_int64, to_exclude_ids, to_exclude_ids_len);

                    auto all_ids = index->seq_ids->uncompress();
                    filter_result.count = ArrayUtils::exclude_scalar(all_ids, index->seq_ids->num_ids(),
                                                                     to_exclude_ids, to_exclude_ids_len, &filter_result.docs);

                    delete[] all_ids;
                    delete[] to_exclude_ids;
                }

                value_index++;
            }
        } else {
            auto num_tree = index->numerical_index.at(a_filter.field_name);

            size_t value_index = 0;
            for (const std::string& filter_value : a_filter.values) {
                int64_t bool_int64 = (filter_value == "1") ? 1 : 0;

                size_t result_size = filter_result.count;
                if (a_filter.comparators[value_index] == NOT_EQUALS) {
                    numeric_not_equals_filter(num_tree, bool_int64,
                                              index->seq_ids->uncompress(), index->seq_ids->num_ids(),
                                              filter_result.docs, result_size);
                } else {
                    num_tree->search(a_filter.comparators[value_index], bool_int64, &filter_result.docs, result_size);
                }

                filter_result.count = result_size;
                value_index++;
            }
        }

        if (a_filter.apply_not_equals) {
            apply_not_equals(index->seq_ids->uncompress(), index->seq_ids->num_ids(),
                             filter_result.docs, filter_result.count);
        }

        if (filter_result.count == 0) {
            validity = invalid;
            return;
        }

        seq_id = filter_result.docs[result_index];
        is_filter_result_initialized = true;
        approx_filter_ids_length = filter_result.count;
        return;
    } else if (f.is_geopoint()) {
        for (uint32_t fi = 0; fi < a_filter.values.size(); fi++) {
            const std::string& filter_value = a_filter.values[fi];

            std::vector<uint32_t> geo_result_ids;

            std::vector<std::string> filter_value_parts;
            StringUtils::split(filter_value, filter_value_parts, ",");  // x, y, 2, km (or) list of points

            bool is_polygon = StringUtils::is_float(filter_value_parts.back());
            S2Region* query_region;

            double query_radius_meters;
            if (is_polygon) {
                const int num_verts = int(filter_value_parts.size()) / 2;
                std::vector<S2Point> vertices;
                double sum = 0.0;

                for (size_t point_index = 0; point_index < size_t(num_verts); point_index++) {
                    double lat = std::stod(filter_value_parts[point_index * 2]);
                    double lon = std::stod(filter_value_parts[point_index * 2 + 1]);
                    if (point_index + 1 == size_t(num_verts) &&
                        lat == std::stod(filter_value_parts[0]) &&
                        lon == std::stod(filter_value_parts[1])) {
                        // The last geopoint is same as the first one.
                        break;
                    }

                    S2Point vertex = S2LatLng::FromDegrees(lat, lon).ToPoint();
                    vertices.emplace_back(vertex);
                }

                auto loop = new S2Loop(vertices, S2Debug::DISABLE);
                loop->Normalize();  // if loop is not CCW but CW, change to CCW.

                S2Error error;
                if (loop->FindValidationError(&error)) {
                    delete loop;
                    status = Option<bool>(400, "Polygon" + (a_filter.values.size() > 1 ?
                                                                " at position " + std::to_string(fi + 1) : "")
                                                                + " is invalid: " + error.text());
                    validity = invalid;
                    return;
                } else {
                    query_region = loop;
                }

                query_radius_meters = S2Earth::RadiansToMeters(query_region->GetCapBound().GetRadius().radians());
            } else {
                query_radius_meters = std::stof(filter_value_parts[2]);
                const auto& unit = filter_value_parts[3];

                if (unit == "km") {
                    query_radius_meters *= 1000;
                } else {
                    // assume "mi" (validated upstream)
                    query_radius_meters *= 1609.34;
                }

                S1Angle query_radius_radians = S1Angle::Radians(S2Earth::MetersToRadians(query_radius_meters));
                double query_lat = std::stod(filter_value_parts[0]);
                double query_lng = std::stod(filter_value_parts[1]);
                S2Point center = S2LatLng::FromDegrees(query_lat, query_lng).ToPoint();
                query_region = new S2Cap(center, query_radius_radians);
            }
            std::unique_ptr<S2Region> query_region_guard(query_region);

            S2RegionTermIndexer::Options options;
            options.set_index_contains_points_only(true);
            S2RegionTermIndexer indexer(options);
            auto const& geo_range_index = index->geo_range_index.at(a_filter.field_name);

            std::vector<uint64_t> cell_ids;
            for (const auto& term : indexer.GetQueryTerms(*query_region, "")) {
                auto cell = S2CellId::FromToken(term);
                cell_ids.push_back(cell.id());
            }

            geo_range_index->search_geopoints(cell_ids, geo_result_ids);

            // Skip exact filtering step if query radius is greater than the threshold.
            if (fi < a_filter.params.size() &&
                query_radius_meters > a_filter.params[fi][filter::EXACT_GEO_FILTER_RADIUS_KEY].get<double>()) {
                uint32_t* out = nullptr;
                filter_result.count = ArrayUtils::or_scalar(geo_result_ids.data(), geo_result_ids.size(),
                                                            filter_result.docs, filter_result.count, &out);

                delete[] filter_result.docs;
                filter_result.docs = out;
                continue;
            }

            // `geo_result_ids` will contain all IDs that are within approximately within query radius
            // we still need to do another round of exact filtering on them

            std::vector<uint32_t> exact_geo_result_ids;

            if (f.is_single_geopoint()) {
                auto sort_field_index = index->sort_index.at(f.name);

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
                spp::sparse_hash_map<uint32_t, int64_t*>* geo_field_index = index->geo_array_index.at(f.name);

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
            filter_result.count = ArrayUtils::or_scalar(&exact_geo_result_ids[0], exact_geo_result_ids.size(),
                                                        filter_result.docs, filter_result.count, &out);

            delete[] filter_result.docs;
            filter_result.docs = out;
        }

        if (filter_result.count == 0) {
            validity = invalid;
            return;
        }

        seq_id = filter_result.docs[result_index];
        is_filter_result_initialized = true;
        approx_filter_ids_length = filter_result.count;
        return;
    } else if (f.is_string()) {
        art_tree* t = index->search_index.at(a_filter.field_name);

        for (const std::string& filter_value : a_filter.values) {
            std::vector<void*> raw_posting_lists;

            // there could be multiple tokens in a filter value, which we have to treat as ANDs
            // e.g. country: South Africa
            Tokenizer tokenizer(filter_value, true, false, f.locale, index->symbols_to_index, index->token_separators);

            std::string str_token;
            size_t token_index = 0;
            std::vector<std::string> str_tokens;

            while (tokenizer.next(str_token, token_index)) {
                if (str_token.size() > 100) {
                    str_token.erase(100);
                }
                str_tokens.push_back(str_token);

                art_leaf* leaf = (art_leaf *) art_search(t, (const unsigned char*) str_token.c_str(),
                                                         str_token.length()+1);
                if (leaf == nullptr) {
                    continue;
                }

                approx_filter_ids_length += posting_t::num_ids(leaf->values);
                raw_posting_lists.push_back(leaf->values);
            }

            if (raw_posting_lists.size() != str_tokens.size()) {
                continue;
            }

            std::vector<posting_list_t*> plists;
            posting_t::to_expanded_plists(raw_posting_lists, plists, expanded_plists);

            posting_lists.push_back(plists);
            posting_list_iterators.emplace_back(std::vector<posting_list_t::iterator_t>());
            for (auto const& plist: plists) {
                posting_list_iterators.back().push_back(plist->new_iterator());
            }
        }

        if (a_filter.apply_not_equals && approx_filter_ids_length == 0) {
            approx_filter_ids_length = index->seq_ids->num_ids();
        }

        get_string_filter_first_match(f.is_array());
        return;
    }
}

void filter_result_iterator_t::skip_to(uint32_t id, const bool& override_timeout) {
    if (validity == invalid || (!override_timeout && timeout_info != nullptr && is_timed_out())) {
        return;
    }

    // No need to traverse iterator tree if there's only one filter or compute_result() has been called.
    if (is_filter_result_initialized) {
        ArrayUtils::skip_index_to_id(result_index, filter_result.docs, filter_result.count, id);

        if (result_index >= filter_result.count) {
            validity = invalid;
            return;
        }

        seq_id = filter_result.docs[result_index];
        reference.clear();
        if (filter_result.coll_to_references != nullptr) {
            auto& ref = filter_result.coll_to_references[result_index];
            reference.insert(ref.begin(), ref.end());
        }

        return;
    }

    if (filter_node->isOperator) {
        // Skip the subtrees to id and then apply operators to arrive at the next valid doc.
        left_it->skip_to(id);
        right_it->skip_to(id);

        if (filter_node->filter_operator == AND) {
            and_filter_iterators();
        } else {
            or_filter_iterators();
        }

        return;
    }

    const filter a_filter = filter_node->filter_exp;

    if (!index->field_is_indexed(a_filter.field_name)) {
        validity = invalid;
        return;
    }

    field f = index->search_schema.at(a_filter.field_name);

    if (f.is_string()) {
        if (filter_node->filter_exp.apply_not_equals) {
            if (id < seq_id) {
                return;
            }

            if (id < result_index) {
                seq_id = id;
                return;
            }

            seq_id = result_index;
            uint32_t previous_match;

            // Keep ignoring the found gaps till they cannot contain id.
            do {
                do {
                    previous_match = seq_id;
                    advance_string_filter_token_iterators();
                    get_string_filter_next_match(f.is_array());
                } while (validity && previous_match + 1 == seq_id);
            } while (validity && seq_id <= id);

            if (!validity) {
                // filter matched all the ids in the index. So for not equals, there's no match.
                if (previous_match >= index->seq_ids->last_id()) {
                    return;
                }

                validity = valid;
                seq_id = previous_match + 1;
                result_index = index->seq_ids->last_id() + 1;

                // Skip to id, if possible.
                if (seq_id < id && id < result_index) {
                    seq_id = id;
                }

                return;
            }

            result_index = seq_id;
            seq_id = previous_match + 1;

            if (seq_id < id && id < result_index) {
                seq_id = id;
            }

            return;
        }

        // Skip all the token iterators and find a new match.
        for (auto& filter_value_tokens : posting_list_iterators) {
            for (auto& token: filter_value_tokens) {
                // We perform AND on tokens. Short-circuiting here.
                if (!token.valid()) {
                    break;
                }

                token.skip_to(id);
            }
        }

        get_string_filter_next_match(f.is_array());
        return;
    }
}

int filter_result_iterator_t::is_valid(uint32_t id) {
    if (validity != valid) {
        return -1;
    }

    // No need to traverse iterator tree if there's only one filter or compute_result() has been called.
    if (is_filter_result_initialized) {
        skip_to(id);
        return validity ? (seq_id == id ? 1 : 0) : -1;
    }

    if (timeout_info != nullptr && is_timed_out()) {
        return -1;
    }

    if (filter_node->isOperator) {
        // We only need to consider only valid/invalid state since child nodes can never time out.
        auto left_validity = left_it->is_valid(id), right_validity = right_it->is_valid(id);

        if (filter_node->filter_operator == AND) {
            validity = (left_it->validity == valid && right_it->validity == valid) ? valid : invalid;

            if (left_validity < 1 || right_validity < 1) {
                if (left_validity == -1 || right_validity == -1) {
                    return -1;
                }

                // id did not match the filter but both of the sub-iterators are still valid.
                // Updating seq_id to the next potential match.
                if (left_validity == 0 && right_validity == 0) {
                    seq_id = std::max(left_it->seq_id, right_it->seq_id);
                } else if (left_validity == 0) {
                    seq_id = left_it->seq_id;
                } else {
                    seq_id = right_it->seq_id;
                }

                return 0;
            }

            seq_id = id;
            and_filter_iterators();
            return 1;
        } else {
            validity = (left_it->validity == valid || right_it->validity == valid) ? valid : invalid;

            if (left_validity < 1 && right_validity < 1) {
                if (left_validity == -1 && right_validity == -1) {
                    return -1;
                }

                // id did not match the filter; both of the sub-iterators or one of them might be valid.
                // Updating seq_id to the next match.
                if (left_validity == 0 && right_validity == 0) {
                    seq_id = std::min(left_it->seq_id, right_it->seq_id);
                } else if (left_validity == 0) {
                    seq_id = left_it->seq_id;
                } else {
                    seq_id = right_it->seq_id;
                }

                return 0;
            }

            seq_id = id;
            or_filter_iterators();
            return 1;
        }
    }

    skip_to(id);
    return validity ? (seq_id == id ? 1 : 0) : -1;
}

Option<bool> filter_result_iterator_t::init_status() {
    if (filter_node != nullptr && filter_node->isOperator) {
        auto left_status = left_it->init_status();

        return !left_status.ok() ? left_status : right_it->init_status();
    }

    return status;
}

bool filter_result_iterator_t::contains_atleast_one(const void *obj) {
    if(IS_COMPACT_POSTING(obj)) {
        compact_posting_list_t* list = COMPACT_POSTING_PTR(obj);

        size_t i = 0;
        while(i < list->length && validity == valid) {
            size_t num_existing_offsets = list->id_offsets[i];
            size_t existing_id = list->id_offsets[i + num_existing_offsets + 1];

            if (existing_id == seq_id) {
                return true;
            }

            // advance smallest value
            if (existing_id < seq_id) {
                i += num_existing_offsets + 2;
            } else {
                skip_to(existing_id);
            }
        }
    } else {
        auto list = (posting_list_t*)(obj);
        posting_list_t::iterator_t it = list->new_iterator();

        while(it.valid() && validity == valid) {
            uint32_t id = it.id();

            if(id == seq_id) {
                return true;
            }

            if(id < seq_id) {
                it.skip_to(seq_id);
            } else {
                skip_to(id);
            }
        }
    }

    return false;
}

void filter_result_iterator_t::reset(const bool& override_timeout) {
    if (filter_node == nullptr) {
        return;
    }

    if (!override_timeout && timeout_info != nullptr && is_timed_out()) {
        return;
    }

    // No need to traverse iterator tree if there's only one filter or compute_result() has been called.
    if (is_filter_result_initialized) {
        if (filter_result.count == 0) {
            validity = invalid;
            return;
        }

        result_index = 0;
        seq_id = filter_result.docs[result_index];

        reference.clear();
        if (filter_result.coll_to_references != nullptr) {
            auto& ref = filter_result.coll_to_references[result_index];
            reference.insert(ref.begin(), ref.end());
        }

        validity = valid;
        return;
    }

    if (filter_node->isOperator) {
        // Reset the subtrees then apply operators to arrive at the first valid doc.
        left_it->reset();
        right_it->reset();
        validity = valid;

        if (filter_node->filter_operator == AND) {
            and_filter_iterators();
        } else {
            or_filter_iterators();
        }

        return;
    }

    const filter a_filter = filter_node->filter_exp;

    if (!index->field_is_indexed(a_filter.field_name)) {
        return;
    }

    field f = index->search_schema.at(a_filter.field_name);

    if (f.is_string()) {
        for (uint32_t i = 0; i < posting_lists.size(); i++) {
            auto const& plists = posting_lists[i];

            posting_list_iterators[i].clear();
            for (auto const& plist: plists) {
                posting_list_iterators[i].push_back(plist->new_iterator());
            }
        }

        get_string_filter_first_match(f.is_array());
        return;
    }
}

uint32_t filter_result_iterator_t::to_filter_id_array(uint32_t*& filter_array) {
    if (validity != valid) {
        return 0;
    }

    if (is_filter_result_initialized) {
        filter_array = new uint32_t[filter_result.count];
        std::copy(filter_result.docs, filter_result.docs + filter_result.count, filter_array);
        return filter_result.count;
    }

    std::vector<uint32_t> filter_ids;
    do {
        filter_ids.push_back(seq_id);
        next();
    } while (validity == valid);

    filter_array = new uint32_t[filter_ids.size()];
    std::copy(filter_ids.begin(), filter_ids.end(), filter_array);

    return filter_ids.size();
}

uint32_t filter_result_iterator_t::and_scalar(const uint32_t* A, const uint32_t& lenA, uint32_t*& results) {
    if (validity != valid) {
        return 0;
    }

    if (is_filter_result_initialized) {
        return ArrayUtils::and_scalar(A, lenA, filter_result.docs, filter_result.count, &results);
    }

    std::vector<uint32_t> filter_ids;
    for (uint32_t i = 0; i < lenA; i++) {
        auto result = is_valid(A[i]);

        if (result == -1) {
            break;
        }

        if (result == 1) {
            filter_ids.push_back(A[i]);
        }
    }

    if (filter_ids.empty()) {
        return 0;
    }

    results = new uint32_t[filter_ids.size()];
    std::copy(filter_ids.begin(), filter_ids.end(), results);

    return filter_ids.size();
}

void filter_result_iterator_t::and_scalar(const uint32_t* A, const uint32_t& lenA, filter_result_t& result) {
    if (validity != valid) {
        return;
    }

    if (filter_result.coll_to_references == nullptr) {
        if (is_filter_result_initialized) {
            result.count = ArrayUtils::and_scalar(A, lenA, filter_result.docs, filter_result.count, &result.docs);
            return;
        }

        std::vector<uint32_t> filter_ids;
        for (uint32_t i = 0; i < lenA; i++) {
            auto _result = is_valid(A[i]);

            if (_result == -1) {
                break;
            }

            if (_result == 1) {
                filter_ids.push_back(A[i]);
            }
        }

        if (filter_ids.empty()) {
            return;
        }

        result.count = filter_ids.size();
        result.docs = new uint32_t[filter_ids.size()];
        std::copy(filter_ids.begin(), filter_ids.end(), result.docs);
        return;
    }

    if (!is_filter_result_initialized) {
        compute_result();
    }

    std::vector<uint32_t> match_indexes;
    for (uint32_t i = 0; i < lenA; i++) {
        auto _result = is_valid(A[i]);

        if (_result == -1) {
            break;
        }

        if (_result == 1) {
            match_indexes.push_back(result_index);
        }
    }

    result.count = match_indexes.size();
    result.docs = new uint32_t[match_indexes.size()];
    result.coll_to_references = new std::map<std::string, reference_filter_result_t>[match_indexes.size()] {};

    for (uint32_t i = 0; i < match_indexes.size(); i++) {
        auto const& match_index = match_indexes[i];
        result.docs[i] = filter_result.docs[match_index];

        auto& result_reference = result.coll_to_references[i];
        result_reference.insert(filter_result.coll_to_references[match_index].begin(),
                                 filter_result.coll_to_references[match_index].end());
    }
}

filter_result_iterator_t::filter_result_iterator_t(const std::string& collection_name, const Index *const index,
                                                   const filter_node_t *const filter_node,
                                                   uint64_t search_begin, uint64_t search_stop)  :
        collection_name(collection_name),
        index(index),
        filter_node(filter_node) {
    if (filter_node == nullptr) {
        validity = invalid;
        return;
    }

    // Only initialize timeout_info in the root node. We won't pass search_begin/search_stop parameters to the sub-nodes.
    if (search_stop != UINT64_MAX) {
        timeout_info = std::make_unique<filter_result_iterator_timeout_info>(search_begin, search_stop);
    }

    // Generate the iterator tree and then initialize each node.
    if (filter_node->isOperator) {
        left_it = new filter_result_iterator_t(collection_name, index, filter_node->left);
        right_it = new filter_result_iterator_t(collection_name, index, filter_node->right);
    }

    init();

    if (!validity) {
        this->approx_filter_ids_length = 0;
    }
}

filter_result_iterator_t::~filter_result_iterator_t() {
    // In case the filter was on string field.
    for(auto expanded_plist: expanded_plists) {
        delete expanded_plist;
    }

    if (delete_filter_node) {
        delete filter_node;
    }

    delete left_it;
    delete right_it;
}

filter_result_iterator_t& filter_result_iterator_t::operator=(filter_result_iterator_t&& obj) noexcept {
    if (&obj == this) {
        return *this;
    }

    // In case the filter was on string field.
    for(auto expanded_plist: expanded_plists) {
        delete expanded_plist;
    }

    delete left_it;
    delete right_it;

    collection_name = obj.collection_name;
    index = obj.index;
    filter_node = obj.filter_node;
    left_it = obj.left_it;
    right_it = obj.right_it;

    obj.left_it = nullptr;
    obj.right_it = nullptr;

    result_index = obj.result_index;

    filter_result = std::move(obj.filter_result);

    posting_list_iterators = std::move(obj.posting_list_iterators);
    expanded_plists = std::move(obj.expanded_plists);

    validity = obj.validity;

    seq_id = obj.seq_id;
    reference = std::move(obj.reference);
    status = std::move(obj.status);
    is_filter_result_initialized = obj.is_filter_result_initialized;

    approx_filter_ids_length = obj.approx_filter_ids_length;

    return *this;
}

void filter_result_iterator_t::get_n_ids(const uint32_t& n, filter_result_t*& result, const bool& override_timeout) {
    if (!is_filter_result_initialized) {
        return;
    }

    if (override_timeout) {
        result_index = 0;
    } else if (timeout_info != nullptr) {
        // In Index::search_wildcard number of calls to get_n_ids will be min(number of threads, filter match ids).
        // Therefore, `timeout_info->function_call_counter` won't reach `function_call_modulo` if only incremented on
        // function call.
        if (n > function_call_modulo) {
            timeout_info->function_call_counter = function_call_modulo - 1;
        }
        if (is_timed_out()) {
            return;
        }
    }

    auto result_length = result->count = std::min(n, filter_result.count - result_index);
    result->docs = new uint32_t[result_length];
    if (filter_result.coll_to_references != nullptr) {
        result->coll_to_references = new std::map<std::string, reference_filter_result_t>[result_length] {};
    }

    for (uint32_t i = 0; i < result_length; i++, result_index++) {
        result->docs[i] = filter_result.docs[result_index];

        if (filter_result.coll_to_references == nullptr) {
            continue;
        }

        auto& result_reference = result->coll_to_references[i];
        result_reference = filter_result.coll_to_references[result_index];
    }

    if (!override_timeout) {
        validity = result_index < filter_result.count ? valid : invalid;
    }
}

void filter_result_iterator_t::get_n_ids(const uint32_t& n,
                                         uint32_t& excluded_result_index,
                                         uint32_t const* const excluded_result_ids, const size_t& excluded_result_ids_size,
                                         filter_result_t*& result, const bool& override_timeout) {
    if (excluded_result_ids == nullptr || excluded_result_ids_size == 0 ||
        excluded_result_index >= excluded_result_ids_size) {
        return get_n_ids(n, result, override_timeout);
    }

    // This method is only called in Index::search_wildcard after filter_result_iterator_t::compute_result.
    if (!is_filter_result_initialized) {
        return;
    }

    if (override_timeout) {
        result_index = 0;
    } else if (timeout_info != nullptr) {
        // In Index::search_wildcard number of calls to get_n_ids will be min(number of threads, filter match ids).
        // Therefore, `timeout_info->function_call_counter` won't reach `function_call_modulo` if only incremented on
        // function call.
        if (n > function_call_modulo) {
            timeout_info->function_call_counter = function_call_modulo - 1;
        }
        if (is_timed_out()) {
            return;
        }
    }

    std::vector<uint32_t> match_indexes;
    for (uint32_t count = 0; count < n && result_index < filter_result.count; result_index++) {
        auto id = filter_result.docs[result_index];

        if (!ArrayUtils::skip_index_to_id(excluded_result_index, excluded_result_ids, excluded_result_ids_size, id)) {
            match_indexes.push_back(result_index);
            count++;
        }
    }

    result->count = match_indexes.size();
    result->docs = new uint32_t[match_indexes.size()];
    if (filter_result.coll_to_references != nullptr) {
        result->coll_to_references = new std::map<std::string, reference_filter_result_t>[match_indexes.size()] {};
    }

    for (uint32_t i = 0; i < match_indexes.size(); i++) {
        auto const& match_index = match_indexes[i];
        result->docs[i] = filter_result.docs[match_index];

        if (filter_result.coll_to_references == nullptr) {
            continue;
        }

        auto& result_reference = result->coll_to_references[i];
        result_reference = filter_result.coll_to_references[match_index];
    }

    if (!override_timeout) {
        validity = result_index < filter_result.count ? valid : invalid;
    }
}

filter_result_iterator_t::filter_result_iterator_t(uint32_t approx_filter_ids_length) :
        approx_filter_ids_length(approx_filter_ids_length) {
    filter_node = new filter_node_t(AND, nullptr, nullptr);
    delete_filter_node = true;
}

filter_result_iterator_t::filter_result_iterator_t(uint32_t* ids, const uint32_t& ids_count,
                                                   uint64_t search_begin, uint64_t search_stop) {
    filter_result.count = approx_filter_ids_length = ids_count;
    filter_result.docs = ids;
    validity = ids_count > 0 ? valid : invalid;

    if (validity) {
        seq_id = filter_result.docs[result_index];
        is_filter_result_initialized = true;
        filter_node = new filter_node_t({"dummy", {}, {}});
        delete_filter_node = true;

        if (search_stop != UINT64_MAX) {
            timeout_info = std::make_unique<filter_result_iterator_timeout_info>(search_begin, search_stop);
        }
    }
}

void filter_result_iterator_t::add_phrase_ids(filter_result_iterator_t*& fit,
                                              uint32_t* phrase_result_ids, const uint32_t& phrase_result_count) {
    auto root_iterator = new filter_result_iterator_t(std::min(phrase_result_count, fit->approx_filter_ids_length));
    root_iterator->left_it = new filter_result_iterator_t(phrase_result_ids, phrase_result_count);
    root_iterator->right_it = fit;

    auto& left_it = root_iterator->left_it;
    auto& right_it = root_iterator->right_it;

    while (left_it->validity && right_it->validity && left_it->seq_id != right_it->seq_id) {
        if (left_it->seq_id < right_it->seq_id) {
            left_it->skip_to(right_it->seq_id);
        } else {
            right_it->skip_to(left_it->seq_id);
        }
    }

    root_iterator->timeout_info = std::move(fit->timeout_info);
    root_iterator->validity = (left_it->validity == timed_out || right_it->validity == timed_out) ? timed_out :
                               (left_it->validity == invalid || right_it->validity == invalid) ? invalid : valid;
    root_iterator->seq_id = left_it->seq_id;
    fit = root_iterator;
}

void filter_result_iterator_t::compute_result() {
    if (filter_node == nullptr) {
        validity = invalid;
        is_filter_result_initialized = false;
        LOG(ERROR) << "filter_node is null";
        return;
    } else if (is_filter_result_initialized) { // Filter result is already computed.
        return;
    } else if (timeout_info != nullptr && is_timed_out()) {
        return;
    }

    if (filter_node->isOperator) {
        left_it->compute_result();
        right_it->compute_result();

        if (filter_node->filter_operator == AND) {
            filter_result_t::and_filter_results(left_it->filter_result, right_it->filter_result, filter_result);
        } else {
            filter_result_t::or_filter_results(left_it->filter_result, right_it->filter_result, filter_result);
        }

        // In a complex filter query a sub-expression might not match any document while the full expression does match
        // at least one document. If the full expression doesn't match any document, we return early in the search.
        if (filter_result.count == 0) {
            validity = invalid;
            is_filter_result_initialized = true;
            return;
        }

        result_index = 0;;
        seq_id = filter_result.docs[result_index];
        is_filter_result_initialized = true;
        approx_filter_ids_length = filter_result.count;

        // Deleting subtree since we've already computed the result.
        delete left_it;
        left_it = nullptr;
        delete right_it;
        right_it = nullptr;
        return;
    }

    if (index->search_index.count(filter_node->filter_exp.field_name) == 0) {
        return;
    }

    // Only string field filter needs to be evaluated. For other fields `is_filter_result_initialized` would be true.
    for (uint32_t i = 0; i < posting_lists.size(); i++) {
        // Resetting posting_list_iterators.
        auto const& plists = posting_lists[i];

        posting_list_iterators[i].clear();
        for (auto const& plist: plists) {
            posting_list_iterators[i].push_back(plist->new_iterator());
        }
    }

    auto const& a_filter = filter_node->filter_exp;
    auto const& f = index->search_schema.at(a_filter.field_name);

    uint32_t* or_ids = nullptr;
    size_t or_ids_size = 0;

    // aggregates IDs across array of filter values and reduces excessive ORing
    std::vector<uint32_t> f_id_buff;

    for (uint32_t i = 0; i < posting_lists.size(); i++) {
        auto& p_list = posting_lists[i];
        if (a_filter.comparators[0] == EQUALS || a_filter.comparators[0] == NOT_EQUALS) {
            // needs intersection + exact matching (unlike CONTAINS)
            std::vector<uint32_t> result_id_vec;
            posting_list_t::intersect(p_list, result_id_vec);

            if (result_id_vec.empty()) {
                continue;
            }

            // need to do exact match
            uint32_t* exact_str_ids = new uint32_t[result_id_vec.size()];
            size_t exact_str_ids_size = 0;
            std::unique_ptr<uint32_t[]> exact_str_ids_guard(exact_str_ids);

            posting_list_t::get_exact_matches(posting_list_iterators[i], f.is_array(),
                                              result_id_vec.data(), result_id_vec.size(),
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
            posting_list_t::intersect(p_list, f_id_buff);
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

    filter_result.docs = or_ids;
    filter_result.count = or_ids_size;

    if (a_filter.apply_not_equals) {
        apply_not_equals(index->seq_ids->uncompress(), index->seq_ids->num_ids(), filter_result.docs, filter_result.count);
    }

    if (filter_result.count == 0) {
        validity = invalid;
        return;
    }

    result_index = 0;
    seq_id = filter_result.docs[result_index];
    is_filter_result_initialized = true;
    approx_filter_ids_length = filter_result.count;
}

bool filter_result_iterator_t::is_timed_out() {
    if (validity == timed_out ||
        (++(timeout_info->function_call_counter) % function_call_modulo == 0 && (std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count() - timeout_info->search_begin_us) > timeout_info->search_stop_us)) {
        validity = timed_out;
        return true;
    }
    return false;
}

filter_result_iterator_timeout_info::filter_result_iterator_timeout_info(uint64_t search_begin,
                                                                         uint64_t search_stop) :
                                                                         search_begin_us(search_begin),
                                                                         search_stop_us(search_stop) {}
