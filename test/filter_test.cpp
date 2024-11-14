#include <gtest/gtest.h>
#include <string>
#include <vector>
#include <fstream>
#include <collection_manager.h>
#include <filter.h>
#include <posting.h>
#include <chrono>
#include "collection.h"

class FilterTest : public ::testing::Test {
protected:
    Store *store;
    CollectionManager & collectionManager = CollectionManager::get_instance();
    std::atomic<bool> quit = false;

    std::vector<std::string> query_fields;
    std::vector<sort_by> sort_fields;

    void setupCollection() {
        std::string state_dir_path = "/tmp/typesense_test/collection_join";
        LOG(INFO) << "Truncating and creating: " << state_dir_path;
        system(("rm -rf "+state_dir_path+" && mkdir -p "+state_dir_path).c_str());

        store = new Store(state_dir_path);
        collectionManager.init(store, 1.0, "auth_key", quit);
        collectionManager.load(8, 1000);
    }

    virtual void SetUp() {
        setupCollection();
    }

    virtual void TearDown() {
        collectionManager.dispose();
        delete store;
    }
};

TEST_F(FilterTest, FilterTreeIterator) {
    nlohmann::json schema =
            R"({
                "name": "Collection",
                "fields": [
                    {"name": "name", "type": "string"},
                    {"name": "age", "type": "int32"},
                    {"name": "years", "type": "int32[]"},
                    {"name": "rating", "type": "float"},
                    {"name": "tags", "type": "string[]"}
                ]
            })"_json;

    Collection* coll = collectionManager.create_collection(schema).get();

    std::ifstream infile(std::string(ROOT_DIR)+"test/numeric_array_documents.jsonl");
    std::string json_line;
    while (std::getline(infile, json_line)) {
        auto add_op = coll->add(json_line);
        ASSERT_TRUE(add_op.ok());
    }
    infile.close();

    const std::string doc_id_prefix = std::to_string(coll->get_collection_id()) + "_" + Collection::DOC_ID_PREFIX + "_";
    filter_node_t* filter_tree_root = nullptr;

    auto const enable_lazy_evaluation = true;
    auto iter_null_filter_tree_test = filter_result_iterator_t(coll->get_name(), coll->_get_index(), filter_tree_root,
                                                               enable_lazy_evaluation);

    ASSERT_TRUE(iter_null_filter_tree_test.init_status().ok());
    ASSERT_EQ(filter_result_iterator_t::invalid, iter_null_filter_tree_test.validity);

    Option<bool> filter_op = filter::parse_filter_query("name: foo", coll->get_schema(), store, doc_id_prefix,
                                                        filter_tree_root);
    ASSERT_TRUE(filter_op.ok());

    auto iter_no_match_test = filter_result_iterator_t(coll->get_name(), coll->_get_index(), filter_tree_root,
                                                       enable_lazy_evaluation);

    ASSERT_TRUE(iter_no_match_test.init_status().ok());
    ASSERT_EQ(filter_result_iterator_t::invalid, iter_no_match_test.validity);

    delete filter_tree_root;
    filter_tree_root = nullptr;
    filter_op = filter::parse_filter_query("name: [foo bar, baz]", coll->get_schema(), store, doc_id_prefix,
                                                        filter_tree_root);
    ASSERT_TRUE(filter_op.ok());

    auto iter_no_match_multi_test = filter_result_iterator_t(coll->get_name(), coll->_get_index(), filter_tree_root,
                                                             enable_lazy_evaluation);

    ASSERT_TRUE(iter_no_match_multi_test.init_status().ok());
    ASSERT_EQ(filter_result_iterator_t::invalid, iter_no_match_multi_test.validity);

    delete filter_tree_root;
    filter_tree_root = nullptr;
    filter_op = filter::parse_filter_query("name: Jeremy", coll->get_schema(), store, doc_id_prefix,
                                           filter_tree_root);
    ASSERT_TRUE(filter_op.ok());

    auto iter_contains_test = filter_result_iterator_t(coll->get_name(), coll->_get_index(), filter_tree_root,
                                                       enable_lazy_evaluation);
    ASSERT_TRUE(iter_contains_test.init_status().ok());

    for (uint32_t i = 0; i < 5; i++) {
        ASSERT_EQ(filter_result_iterator_t::valid, iter_contains_test.validity);
        ASSERT_EQ(i, iter_contains_test.seq_id);
        iter_contains_test.next();
    }
    ASSERT_EQ(filter_result_iterator_t::invalid, iter_contains_test.validity);

    delete filter_tree_root;
    filter_tree_root = nullptr;
    filter_op = filter::parse_filter_query("name: [Jeremy, Howard, Richard]", coll->get_schema(), store, doc_id_prefix,
                                           filter_tree_root);
    ASSERT_TRUE(filter_op.ok());

    auto iter_contains_multi_test = filter_result_iterator_t(coll->get_name(), coll->_get_index(), filter_tree_root,
                                                             enable_lazy_evaluation);
    ASSERT_TRUE(iter_contains_multi_test.init_status().ok());

    for (uint32_t i = 0; i < 5; i++) {
        ASSERT_EQ(filter_result_iterator_t::valid, iter_contains_multi_test.validity);
        ASSERT_EQ(i, iter_contains_multi_test.seq_id);
        iter_contains_multi_test.next();
    }
    ASSERT_EQ(filter_result_iterator_t::invalid, iter_contains_multi_test.validity);

    delete filter_tree_root;
    filter_tree_root = nullptr;
    filter_op = filter::parse_filter_query("name:= Jeremy Howard", coll->get_schema(), store, doc_id_prefix,
                                                        filter_tree_root);
    ASSERT_TRUE(filter_op.ok());

    auto iter_exact_match_1_test = filter_result_iterator_t(coll->get_name(), coll->_get_index(), filter_tree_root,
                                                            enable_lazy_evaluation);
    ASSERT_TRUE(iter_exact_match_1_test.init_status().ok());

    for (uint32_t i = 0; i < 5; i++) {
        ASSERT_EQ(filter_result_iterator_t::valid, iter_exact_match_1_test.validity);
        ASSERT_EQ(i, iter_exact_match_1_test.seq_id);
        iter_exact_match_1_test.next();
    }
    ASSERT_EQ(filter_result_iterator_t::invalid, iter_exact_match_1_test.validity);

    delete filter_tree_root;
    filter_tree_root = nullptr;
    filter_op = filter::parse_filter_query("tags:= PLATINUM", coll->get_schema(), store, doc_id_prefix,
                                           filter_tree_root);
    ASSERT_TRUE(filter_op.ok());

    auto iter_exact_match_2_test = filter_result_iterator_t(coll->get_name(), coll->_get_index(), filter_tree_root,
                                                            enable_lazy_evaluation);
    ASSERT_TRUE(iter_exact_match_2_test.init_status().ok());
    ASSERT_EQ(filter_result_iterator_t::invalid, iter_exact_match_2_test.validity);

    delete filter_tree_root;
    filter_tree_root = nullptr;
    filter_op = filter::parse_filter_query("tags:= [gold, silver]", coll->get_schema(), store, doc_id_prefix,
                                           filter_tree_root);
    ASSERT_TRUE(filter_op.ok());

    auto iter_exact_match_multi_test = filter_result_iterator_t(coll->get_name(), coll->_get_index(), filter_tree_root,
                                                                enable_lazy_evaluation);
    ASSERT_TRUE(iter_exact_match_multi_test.init_status().ok());

    std::vector<int> expected = {0, 2, 3, 4};
    for (auto const& i : expected) {
        ASSERT_EQ(filter_result_iterator_t::valid, iter_exact_match_multi_test.validity);
        ASSERT_EQ(i, iter_exact_match_multi_test.seq_id);
        iter_exact_match_multi_test.next();
    }
    ASSERT_EQ(filter_result_iterator_t::invalid, iter_exact_match_multi_test.validity);

    delete filter_tree_root;
    filter_tree_root = nullptr;
    filter_op = filter::parse_filter_query("tags:!= gold", coll->get_schema(), store, doc_id_prefix,
                                           filter_tree_root);
    ASSERT_TRUE(filter_op.ok());

    auto iter_not_equals_test = filter_result_iterator_t(coll->get_name(), coll->_get_index(), filter_tree_root,
                                                         enable_lazy_evaluation);
    ASSERT_TRUE(iter_not_equals_test.init_status().ok());

    expected = {1, 3};
    for (auto const& i : expected) {
        ASSERT_EQ(filter_result_iterator_t::valid, iter_not_equals_test.validity);
        ASSERT_EQ(i, iter_not_equals_test.seq_id);
        iter_not_equals_test.next();
    }

    ASSERT_EQ(filter_result_iterator_t::invalid, iter_not_equals_test.validity);

    delete filter_tree_root;
    filter_tree_root = nullptr;
    filter_op = filter::parse_filter_query("name: James || tags: bronze", coll->get_schema(), store, doc_id_prefix,
                                           filter_tree_root);
    ASSERT_TRUE(filter_op.ok());

    auto doc =
            R"({
                "name": "James Rowdy",
                "age": 36,
                "years": [2005, 2022],
                "rating": 6.03,
                "tags": ["copper"]
            })"_json;
    auto add_op = coll->add(doc.dump());
    ASSERT_TRUE(add_op.ok());

    auto iter_or_test = filter_result_iterator_t(coll->get_name(), coll->_get_index(), filter_tree_root,
                                                 enable_lazy_evaluation);
    ASSERT_TRUE(iter_or_test.init_status().ok());

    expected = {2, 4, 5};
    for (auto const& i : expected) {
        ASSERT_EQ(filter_result_iterator_t::valid, iter_or_test.validity);
        ASSERT_EQ(i, iter_or_test.seq_id);
        iter_or_test.next();
    }

    ASSERT_EQ(filter_result_iterator_t::invalid, iter_or_test.validity);

    delete filter_tree_root;
    filter_tree_root = nullptr;
    filter_op = filter::parse_filter_query("name: James || (tags: gold && tags: silver)", coll->get_schema(), store, doc_id_prefix,
                                           filter_tree_root);
    ASSERT_TRUE(filter_op.ok());

    auto iter_complex_filter_test = filter_result_iterator_t(coll->get_name(), coll->_get_index(), filter_tree_root,
                                                             enable_lazy_evaluation);
    ASSERT_TRUE(iter_complex_filter_test.init_status().ok());

    ASSERT_EQ(filter_result_iterator_t::valid, iter_complex_filter_test.validity);
    ASSERT_EQ(0, iter_complex_filter_test.is_valid(3));
    ASSERT_EQ(4, iter_complex_filter_test.seq_id);

    expected = {4, 5};
    for (auto const& i : expected) {
        ASSERT_EQ(filter_result_iterator_t::valid, iter_complex_filter_test.validity);
        ASSERT_EQ(i, iter_complex_filter_test.seq_id);
        iter_complex_filter_test.next();
    }

    ASSERT_EQ(filter_result_iterator_t::invalid, iter_complex_filter_test.validity);

    delete filter_tree_root;
    filter_tree_root = nullptr;
    filter_op = filter::parse_filter_query("name: James || (tags: gold && tags: [silver, bronze])", coll->get_schema(), store, doc_id_prefix,
                                           filter_tree_root);
    ASSERT_TRUE(filter_op.ok());

    auto iter_validate_ids_test1 = filter_result_iterator_t(coll->get_name(), coll->_get_index(), filter_tree_root,
                                                            enable_lazy_evaluation);
    ASSERT_TRUE(iter_validate_ids_test1.init_status().ok());

    std::vector<int> validate_ids = {0, 1, 2, 3, 4, 5, 6};
    std::vector<int> seq_ids = {0, 2, 2, 4, 4, 5, 5};
    expected = {1, 0, 1, 0, 1, 1, -1};
    for (uint32_t i = 0; i < validate_ids.size(); i++) {
        ASSERT_EQ(expected[i], iter_validate_ids_test1.is_valid(validate_ids[i]));
        ASSERT_EQ(seq_ids[i], iter_validate_ids_test1.seq_id);
    }

    delete filter_tree_root;
    filter_tree_root = nullptr;
    filter_op = filter::parse_filter_query("tags: platinum || name: James", coll->get_schema(), store, doc_id_prefix,
                                           filter_tree_root);
    ASSERT_TRUE(filter_op.ok());

    auto iter_validate_ids_test2 = filter_result_iterator_t(coll->get_name(), coll->_get_index(), filter_tree_root,
                                                            enable_lazy_evaluation);
    ASSERT_TRUE(iter_validate_ids_test2.init_status().ok());

    validate_ids = {0, 1, 2, 3, 4, 5, 6}, seq_ids = {1, 1, 5, 5, 5, 5, 5};
    expected = {0, 1, 0, 0, 0, 1, -1};
    for (uint32_t i = 0; i < validate_ids.size(); i++) {
        ASSERT_EQ(expected[i], iter_validate_ids_test2.is_valid(validate_ids[i]));
        ASSERT_EQ(seq_ids[i], iter_validate_ids_test2.seq_id);
    }

    delete filter_tree_root;
    filter_tree_root = nullptr;
    filter_op = filter::parse_filter_query("tags: gold && rating: < 6", coll->get_schema(), store, doc_id_prefix,
                                           filter_tree_root);
    ASSERT_TRUE(filter_op.ok());

    auto iter_validate_ids_test3 = filter_result_iterator_t(coll->get_name(), coll->_get_index(), filter_tree_root,
                                                            enable_lazy_evaluation);
    ASSERT_TRUE(iter_validate_ids_test3.init_status().ok());
    ASSERT_TRUE(iter_validate_ids_test3._get_is_filter_result_initialized());

    validate_ids = {0, 1, 2, 3, 4, 5, 6}, seq_ids = {0, 4, 4, 4, 4, 4, 4};
    expected = {1, 0, 0, 0, 1, -1, -1};
    for (uint32_t i = 0; i < validate_ids.size(); i++) {
        ASSERT_EQ(expected[i], iter_validate_ids_test3.is_valid(validate_ids[i]));
        ASSERT_EQ(seq_ids[i], iter_validate_ids_test3.seq_id);
    }

    delete filter_tree_root;
    filter_tree_root = nullptr;
    filter_op = filter::parse_filter_query("tags: gold", coll->get_schema(), store, doc_id_prefix,
                                           filter_tree_root);
    ASSERT_TRUE(filter_op.ok());

    auto iter_compact_plist_contains_atleast_one_test1 = filter_result_iterator_t(coll->get_name(), coll->_get_index(),
                                                                                  filter_tree_root, enable_lazy_evaluation);
    ASSERT_TRUE(iter_compact_plist_contains_atleast_one_test1.init_status().ok());

    std::vector<uint32_t> ids = {1, 3, 5};
    std::vector<uint32_t> offset_index = {0, 3, 6};
    std::vector<uint32_t> offsets = {0, 3, 4, 0, 3, 4, 0, 3, 4};

    compact_posting_list_t* c_list1 = compact_posting_list_t::create(3, &ids[0], &offset_index[0], 9, &offsets[0]);
    ASSERT_FALSE(iter_compact_plist_contains_atleast_one_test1.contains_atleast_one(SET_COMPACT_POSTING(c_list1)));
    free(c_list1);

    auto iter_compact_plist_contains_atleast_one_test2 = filter_result_iterator_t(coll->get_name(), coll->_get_index(),
                                                                                  filter_tree_root, enable_lazy_evaluation);
    ASSERT_TRUE(iter_compact_plist_contains_atleast_one_test2.init_status().ok());

    ids = {1, 3, 4};
    offset_index = {0, 3, 6};
    offsets = {0, 3, 4, 0, 3, 4, 0, 3, 4};

    compact_posting_list_t* c_list2 = compact_posting_list_t::create(3, &ids[0], &offset_index[0], 9, &offsets[0]);
    ASSERT_TRUE(iter_compact_plist_contains_atleast_one_test2.contains_atleast_one(SET_COMPACT_POSTING(c_list2)));
    free(c_list2);

    auto iter_plist_contains_atleast_one_test1 = filter_result_iterator_t(coll->get_name(), coll->_get_index(),
                                                                          filter_tree_root, enable_lazy_evaluation);
    ASSERT_TRUE(iter_plist_contains_atleast_one_test1.init_status().ok());

    posting_list_t p_list1(2);
    ids = {1, 3};
    for (const auto &i: ids) {
        p_list1.upsert(i, {1, 2, 3});
    }

    ASSERT_FALSE(iter_plist_contains_atleast_one_test1.contains_atleast_one(&p_list1));

    auto iter_plist_contains_atleast_one_test2 = filter_result_iterator_t(coll->get_name(), coll->_get_index(),
                                                                          filter_tree_root);
    ASSERT_TRUE(iter_plist_contains_atleast_one_test2.init_status().ok());

    posting_list_t p_list2(2);
    ids = {1, 3, 4};
    for (const auto &i: ids) {
        p_list1.upsert(i, {1, 2, 3});
    }

    ASSERT_TRUE(iter_plist_contains_atleast_one_test2.contains_atleast_one(&p_list1));

    delete filter_tree_root;
    filter_tree_root = nullptr;
    filter_op = filter::parse_filter_query("tags:= [gold, silver]", coll->get_schema(), store, doc_id_prefix,
                                           filter_tree_root);
    ASSERT_TRUE(filter_op.ok());

    auto iter_reset_test = filter_result_iterator_t(coll->get_name(), coll->_get_index(), filter_tree_root,
                                                    enable_lazy_evaluation);
    ASSERT_TRUE(iter_reset_test.init_status().ok());

    expected = {0, 2, 3, 4};
    for (auto const& i : expected) {
        ASSERT_EQ(filter_result_iterator_t::valid, iter_reset_test.validity);
        ASSERT_EQ(i, iter_reset_test.seq_id);
        iter_reset_test.next();
    }
    ASSERT_EQ(filter_result_iterator_t::invalid, iter_reset_test.validity);

    iter_reset_test.reset();

    for (auto const& i : expected) {
        ASSERT_EQ(filter_result_iterator_t::valid, iter_reset_test.validity);
        ASSERT_EQ(i, iter_reset_test.seq_id);
        iter_reset_test.next();
    }
    ASSERT_EQ(filter_result_iterator_t::invalid, iter_reset_test.validity);

    auto iter_move_assignment_test = filter_result_iterator_t(coll->get_name(), coll->_get_index(), filter_tree_root,
                                                              enable_lazy_evaluation);

    iter_reset_test.reset();
    iter_move_assignment_test = std::move(iter_reset_test);

    expected = {0, 2, 3, 4};
    for (auto const& i : expected) {
        ASSERT_EQ(filter_result_iterator_t::valid, iter_move_assignment_test.validity);
        ASSERT_EQ(i, iter_move_assignment_test.seq_id);
        iter_move_assignment_test.next();
    }
    ASSERT_EQ(filter_result_iterator_t::invalid, iter_move_assignment_test.validity);

    delete filter_tree_root;
    filter_tree_root = nullptr;
    filter_op = filter::parse_filter_query("tags: gold", coll->get_schema(), store, doc_id_prefix,
                                                filter_tree_root);
    ASSERT_TRUE(filter_op.ok());

    auto iter_to_array_test = filter_result_iterator_t(coll->get_name(), coll->_get_index(), filter_tree_root,
                                                       enable_lazy_evaluation);
    ASSERT_TRUE(iter_to_array_test.init_status().ok());

    uint32_t* filter_ids = nullptr;
    uint32_t filter_ids_length;

    iter_to_array_test.compute_iterators();
    filter_ids_length = iter_to_array_test.to_filter_id_array(filter_ids);
    ASSERT_EQ(3, filter_ids_length);

    expected = {0, 2, 4};
    for (uint32_t i = 0; i < filter_ids_length; i++) {
        ASSERT_EQ(expected[i], filter_ids[i]);
    }

    delete[] filter_ids;

    auto iter_and_scalar_test = filter_result_iterator_t(coll->get_name(), coll->_get_index(), filter_tree_root,
                                                         enable_lazy_evaluation);
    ASSERT_TRUE(iter_and_scalar_test.init_status().ok());

    uint32_t a_ids[6] = {0, 1, 3, 4, 5, 6};
    uint32_t* and_result = nullptr;
    uint32_t and_result_length;
    and_result_length = iter_and_scalar_test.and_scalar(a_ids, 6, and_result);
    ASSERT_EQ(2, and_result_length);

    expected = {0, 4};
    for (uint32_t i = 0; i < and_result_length; i++) {
        ASSERT_EQ(expected[i], and_result[i]);
    }
    ASSERT_EQ(filter_result_iterator_t::invalid, iter_and_scalar_test.validity);

    delete[] and_result;
    delete filter_tree_root;

    doc = R"({
            "name": "James Rowdy",
            "age": 36,
            "years": [2005, 2022],
            "rating": 6.03,
            "tags": ["FINE PLATINUM"]
        })"_json;
    add_op = coll->add(doc.dump());
    ASSERT_TRUE(add_op.ok());

    filter_tree_root = nullptr;
    filter_op = filter::parse_filter_query("tags: bronze", coll->get_schema(), store, doc_id_prefix,
                                           filter_tree_root);

    auto iter_add_phrase_ids_test = new filter_result_iterator_t(coll->get_name(), coll->_get_index(), filter_tree_root,
                                                                 enable_lazy_evaluation);
    std::unique_ptr<filter_result_iterator_t> filter_iter_guard(iter_add_phrase_ids_test);
    ASSERT_TRUE(iter_add_phrase_ids_test->init_status().ok());

    auto phrase_ids = new uint32_t[4];
    for (uint32_t i = 0; i < 4; i++) {
        phrase_ids[i] = i * 2;
    }
    filter_result_iterator_t::add_phrase_ids(iter_add_phrase_ids_test, phrase_ids, 4);
    filter_iter_guard.release();
    filter_iter_guard.reset(iter_add_phrase_ids_test);

    ASSERT_EQ(filter_result_iterator_t::valid, iter_add_phrase_ids_test->validity);
    ASSERT_EQ(2, iter_add_phrase_ids_test->seq_id);
    delete filter_tree_root;

    filter_tree_root = nullptr;
    filter_op = filter::parse_filter_query("tags: [gold]", coll->get_schema(), store, doc_id_prefix,
                                           filter_tree_root);
    ASSERT_TRUE(filter_op.ok());

    auto iter_string_multi_value_test = filter_result_iterator_t(coll->get_name(), coll->_get_index(), filter_tree_root,
                                                                 enable_lazy_evaluation);
    ASSERT_TRUE(iter_string_multi_value_test.init_status().ok());
    ASSERT_FALSE(iter_string_multi_value_test._get_is_filter_result_initialized());

    expected = {0, 2, 4};
    for (auto const& i : expected) {
        ASSERT_EQ(filter_result_iterator_t::valid, iter_string_multi_value_test.validity);
        ASSERT_EQ(i, iter_string_multi_value_test.seq_id);
        iter_string_multi_value_test.next();
    }
    ASSERT_EQ(filter_result_iterator_t::invalid, iter_string_multi_value_test.validity);
    delete filter_tree_root;

    filter_tree_root = nullptr;
    filter_op = filter::parse_filter_query("tags:= bronze", coll->get_schema(), store, doc_id_prefix,
                                           filter_tree_root);
    ASSERT_TRUE(filter_op.ok());

    auto iter_string_equals_test = filter_result_iterator_t(coll->get_name(), coll->_get_index(), filter_tree_root,
                                                            enable_lazy_evaluation);
    ASSERT_TRUE(iter_string_equals_test.init_status().ok());
    ASSERT_TRUE(iter_string_equals_test._get_is_filter_result_initialized());

    expected = {2, 4};
    for (auto const& i : expected) {
        ASSERT_EQ(filter_result_iterator_t::valid, iter_string_equals_test.validity);
        ASSERT_EQ(i, iter_string_equals_test.seq_id);
        iter_string_equals_test.next();
    }
    ASSERT_EQ(filter_result_iterator_t::invalid, iter_string_equals_test.validity);

    delete filter_tree_root;

    filter_tree_root = nullptr;
    filter_op = filter::parse_filter_query("tags: gold", coll->get_schema(), store, doc_id_prefix,
                                           filter_tree_root);
    ASSERT_TRUE(filter_op.ok());

    auto iter_string_equals_test_2 = filter_result_iterator_t(coll->get_name(), coll->_get_index(), filter_tree_root,
                                                              enable_lazy_evaluation);
    ASSERT_TRUE(iter_string_equals_test_2.init_status().ok());
    ASSERT_FALSE(iter_string_equals_test_2._get_is_filter_result_initialized());

    expected = {0, 2, 4};
    for (auto const& i : expected) {
        ASSERT_EQ(filter_result_iterator_t::valid, iter_string_equals_test_2.validity);
        ASSERT_EQ(i, iter_string_equals_test_2.seq_id);
        iter_string_equals_test_2.next();
    }
    ASSERT_EQ(filter_result_iterator_t::invalid, iter_string_equals_test_2.validity);

    delete filter_tree_root;
    filter_tree_root = nullptr;
    filter_op = filter::parse_filter_query("tags: != [gold, silver]", coll->get_schema(), store, doc_id_prefix,
                                           filter_tree_root);
    ASSERT_TRUE(filter_op.ok());
    auto iter_string_not_equals_test_2 = filter_result_iterator_t(coll->get_name(), coll->_get_index(), filter_tree_root,
                                                                  enable_lazy_evaluation);
    ASSERT_TRUE(iter_string_not_equals_test_2.init_status().ok());
    ASSERT_TRUE(iter_string_not_equals_test_2._get_is_filter_result_initialized());

    expected = {1, 5, 6};
    for (auto const& i : expected) {
        ASSERT_EQ(filter_result_iterator_t::valid, iter_string_not_equals_test_2.validity);
        ASSERT_EQ(i, iter_string_not_equals_test_2.seq_id);
        iter_string_not_equals_test_2.next();
    }
    ASSERT_EQ(filter_result_iterator_t::invalid, iter_string_not_equals_test_2.validity);

    coll->remove("0");
    coll->remove("2");

    doc = R"({
            "name": "James Rowdy",
            "age": 16,
            "years": [2022],
            "rating": 2.03,
            "tags": ["FINE PLATINUM"]
        })"_json;
    add_op = coll->add(doc.dump());
    ASSERT_TRUE(add_op.ok());

    delete filter_tree_root;

    Collection *bool_coll;

    std::vector<field> fields = {field("title", field_types::STRING, false),
                                 field("in_stock", field_types::BOOL, false),
                                 field("points", field_types::INT32, false),};

    bool_coll = collectionManager.get_collection("bool_coll").get();
    if(bool_coll == nullptr) {
        bool_coll = collectionManager.create_collection("bool_coll", 1, fields, "points").get();
    }

    for(size_t i=0; i<10; i++) {
        nlohmann::json bool_doc;

        bool_doc["title"] = "title_" + std::to_string(i);
        bool_doc["in_stock"] = (i < 5 || i % 2) ? "true" : "false";
        bool_doc["points"] = i;

        ASSERT_TRUE(bool_coll->add(bool_doc.dump()).ok());
    }

    filter_tree_root = nullptr;
    filter_op = filter::parse_filter_query("in_stock: false", bool_coll->get_schema(), store, doc_id_prefix,
                                           filter_tree_root);
    ASSERT_TRUE(filter_op.ok());

    auto iter_boolean_test = filter_result_iterator_t(bool_coll->get_name(), bool_coll->_get_index(), filter_tree_root,
                                                      enable_lazy_evaluation);
    ASSERT_TRUE(iter_boolean_test.init_status().ok());
    ASSERT_TRUE(iter_boolean_test._get_is_filter_result_initialized());
    ASSERT_EQ(2, iter_boolean_test.approx_filter_ids_length);

    expected = {6, 8};
    for (auto const& i : expected) {
        ASSERT_EQ(filter_result_iterator_t::valid, iter_boolean_test.validity);
        ASSERT_EQ(i, iter_boolean_test.seq_id);
        iter_boolean_test.next();
    }
    ASSERT_EQ(filter_result_iterator_t::invalid, iter_boolean_test.validity);
    delete filter_tree_root;

    filter_tree_root = nullptr;
    filter_op = filter::parse_filter_query("in_stock: true", bool_coll->get_schema(), store, doc_id_prefix,
                                           filter_tree_root);
    ASSERT_TRUE(filter_op.ok());

    auto iter_boolean_test_2 = filter_result_iterator_t(bool_coll->get_name(), bool_coll->_get_index(), filter_tree_root,
                                                        enable_lazy_evaluation);
    ASSERT_TRUE(iter_boolean_test_2.init_status().ok());
    ASSERT_FALSE(iter_boolean_test_2._get_is_filter_result_initialized());
    ASSERT_EQ(8, iter_boolean_test_2.approx_filter_ids_length);

    expected = {0, 1, 2, 3, 4, 5, 7, 9};
    for (auto const& i : expected) {
        ASSERT_EQ(filter_result_iterator_t::valid, iter_boolean_test_2.validity);
        ASSERT_EQ(i, iter_boolean_test_2.seq_id);
        iter_boolean_test_2.next();
    }
    ASSERT_EQ(filter_result_iterator_t::invalid, iter_boolean_test_2.validity);

    iter_boolean_test_2.reset();

    expected = {0, 1, 2, 3, 4, 5, 7, 9};
    for (auto const& i : expected) {
        ASSERT_EQ(filter_result_iterator_t::valid, iter_boolean_test_2.validity);
        ASSERT_EQ(i, iter_boolean_test_2.seq_id);
        iter_boolean_test_2.next();
    }
    ASSERT_EQ(filter_result_iterator_t::invalid, iter_boolean_test_2.validity);

    iter_boolean_test_2.reset();
    ASSERT_EQ(0, iter_boolean_test_2.is_valid(6));
    ASSERT_EQ(filter_result_iterator_t::valid, iter_boolean_test_2.validity);
    ASSERT_EQ(7, iter_boolean_test_2.seq_id);

    ASSERT_EQ(0, iter_boolean_test_2.is_valid(8));
    ASSERT_EQ(filter_result_iterator_t::valid, iter_boolean_test_2.validity);
    ASSERT_EQ(9, iter_boolean_test_2.seq_id);

    ASSERT_EQ(-1, iter_boolean_test_2.is_valid(10));
    ASSERT_EQ(filter_result_iterator_t::invalid, iter_boolean_test_2.validity);
    delete filter_tree_root;

    doc = R"({
            "name": "James rock",
            "age": 20,
            "years": [],
            "rating": 4.51,
            "tags": ["gallium", "Gadolinium"]
        })"_json;
    add_op = coll->add(doc.dump());
    ASSERT_TRUE(add_op.ok());

    search_stop_us = UINT64_MAX; // `Index::fuzzy_search_fields` checks for timeout.
    filter_tree_root = nullptr;
    filter_op = filter::parse_filter_query("tags: g*", coll->get_schema(), store, doc_id_prefix,
                                           filter_tree_root);
    ASSERT_TRUE(filter_op.ok());

    auto iter_string_prefix_value_test = filter_result_iterator_t(coll->get_name(), coll->_get_index(), filter_tree_root,
                                                                  enable_lazy_evaluation);
    ASSERT_TRUE(iter_string_prefix_value_test.init_status().ok());
    ASSERT_FALSE(iter_string_prefix_value_test._get_is_filter_result_initialized());
    ASSERT_EQ(3, iter_string_prefix_value_test.approx_filter_ids_length); // document 0 and 2 have been deleted.

    expected = {4, 8};
    for (auto const& i : expected) {
        ASSERT_EQ(filter_result_iterator_t::valid, iter_string_prefix_value_test.validity);
        ASSERT_EQ(i, iter_string_prefix_value_test.seq_id);
        iter_string_prefix_value_test.next();
    }
    ASSERT_EQ(filter_result_iterator_t::invalid, iter_string_prefix_value_test.validity);

    delete filter_tree_root;
    filter_tree_root = nullptr;
    filter_op = filter::parse_filter_query("tags: != g*", coll->get_schema(), store, doc_id_prefix,
                                           filter_tree_root);
    ASSERT_TRUE(filter_op.ok());

    auto iter_string_prefix_value_test_2 = filter_result_iterator_t(coll->get_name(), coll->_get_index(), filter_tree_root,
                                                                    enable_lazy_evaluation);
    ASSERT_TRUE(iter_string_prefix_value_test_2.init_status().ok());
    ASSERT_FALSE(iter_string_prefix_value_test_2._get_is_filter_result_initialized());
    ASSERT_EQ(4, iter_string_prefix_value_test_2.approx_filter_ids_length); // 7 total docs, 3 approx count for equals.

    validate_ids = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    seq_ids = {1, 2, 3, 4, 5, 6, 7, 8, 9, 9};
    expected = {1, 1, 1, 1, 0, 1, 1, 1, 0, -1};
    std::vector<uint32_t > equals_match_seq_ids = {4, 4, 4, 4, 4, 8, 8, 8, 8, 8};
    std::vector<bool> equals_iterator_valid = {true, true, true, true, true, true, true, true, true, true};
    for (uint32_t i = 0; i < validate_ids.size(); i++) {
        ASSERT_EQ(filter_result_iterator_t::valid, iter_string_prefix_value_test_2.validity);
        ASSERT_EQ(expected[i], iter_string_prefix_value_test_2.is_valid(validate_ids[i]));
        ASSERT_EQ(equals_match_seq_ids[i], iter_string_prefix_value_test_2._get_equals_iterator_id());
        ASSERT_EQ(equals_iterator_valid[i], iter_string_prefix_value_test_2._get_is_equals_iterator_valid());

        if (expected[i] == 1) {
            iter_string_prefix_value_test_2.next();
        }
        ASSERT_EQ(seq_ids[i], iter_string_prefix_value_test_2.seq_id);
    }
    ASSERT_EQ(filter_result_iterator_t::invalid, iter_string_prefix_value_test_2.validity);

    delete filter_tree_root;
}

TEST_F(FilterTest, FilterTreeIteratorTimeout) {
    auto count = 20;
    auto filter_ids = new uint32_t[count];
    for (auto i = 0; i < count; i++) {
        filter_ids[i] = i;
    }
    auto filter_iterator = new filter_result_iterator_t(filter_ids, count, DEFAULT_FILTER_BY_CANDIDATES,
                                                        std::chrono::duration_cast<std::chrono::microseconds>(
                                                        std::chrono::system_clock::now().time_since_epoch()).count(),
                                                        10000000); // Timeout after 10 seconds
    std::unique_ptr<filter_result_iterator_t> filter_iter_guard(filter_iterator);

    ASSERT_EQ(filter_result_iterator_t::valid, filter_iterator->validity);
    std::this_thread::sleep_for(std::chrono::seconds(5));

    for (auto i = 0; i < 20; i++) {
        ASSERT_EQ(filter_result_iterator_t::valid, filter_iterator->validity);
        filter_iterator->next();
    }
    ASSERT_EQ(filter_result_iterator_t::invalid, filter_iterator->validity); // End of iterator reached.

    filter_iterator->reset();
    ASSERT_EQ(filter_result_iterator_t::valid, filter_iterator->validity);
    std::this_thread::sleep_for(std::chrono::seconds(5));

    for (auto i = 0; i < 9; i++) {
        ASSERT_EQ(filter_result_iterator_t::valid, filter_iterator->validity);
        filter_iterator->next();
    }
    ASSERT_EQ(filter_result_iterator_t::timed_out, filter_iterator->validity);

    filter_iterator->reset();
    ASSERT_EQ(filter_result_iterator_t::timed_out, filter_iterator->validity); // Resetting won't help with timeout.

    uint32_t excluded_result_index = 0;
    auto result = new filter_result_t();
    filter_iterator->get_n_ids(count, excluded_result_index, nullptr, 0, result);

    ASSERT_EQ(0, result->count); // Shouldn't return results
    delete result;

    filter_iterator->reset(true);
    result = new filter_result_t();
    filter_iterator->get_n_ids(count, excluded_result_index, nullptr, 0, result, true);

    ASSERT_EQ(count, result->count); // With `override_timeout` true, we should get result.
    delete result;
}

TEST_F(FilterTest, FilterTreeInitialization) {
    nlohmann::json schema =
            R"({
                "name": "Collection",
                "fields": [
                    {"name": "name", "type": "string"},
                    {"name": "age", "type": "int32"},
                    {"name": "years", "type": "int32[]"},
                    {"name": "rating", "type": "float"},
                    {"name": "tags", "type": "string[]"}
                ]
            })"_json;

    Collection* coll = collectionManager.create_collection(schema).get();

    std::ifstream infile(std::string(ROOT_DIR)+"test/numeric_array_documents.jsonl");
    std::string json_line;
    while (std::getline(infile, json_line)) {
        auto add_op = coll->add(json_line);
        ASSERT_TRUE(add_op.ok());
    }
    infile.close();

    const std::string doc_id_prefix = std::to_string(coll->get_collection_id()) + "_" + Collection::DOC_ID_PREFIX + "_";
    filter_node_t* filter_tree_root = nullptr;

    Option<bool> filter_op = filter::parse_filter_query("age: 0 && (rating: >0 && years: 2016)", coll->get_schema(), store, doc_id_prefix,
                                                        filter_tree_root);
    ASSERT_TRUE(filter_op.ok());

    auto const enable_lazy_evaluation = true;
    auto iter_left_subtree_0_matches = filter_result_iterator_t(coll->get_name(), coll->_get_index(), filter_tree_root,
                                                                enable_lazy_evaluation);

    ASSERT_TRUE(iter_left_subtree_0_matches.init_status().ok());
    ASSERT_EQ(filter_result_iterator_t::invalid, iter_left_subtree_0_matches.validity);
    ASSERT_EQ(0, iter_left_subtree_0_matches.approx_filter_ids_length);
    ASSERT_TRUE(iter_left_subtree_0_matches._get_is_filter_result_initialized());
    ASSERT_EQ(nullptr, iter_left_subtree_0_matches._get_left_it());
    ASSERT_EQ(nullptr, iter_left_subtree_0_matches._get_right_it());

    delete filter_tree_root;
    filter_tree_root = nullptr;

    filter_op = filter::parse_filter_query("(rating: >0 && years: 2016) && age: 0", coll->get_schema(), store, doc_id_prefix,
                                                        filter_tree_root);
    ASSERT_TRUE(filter_op.ok());

    auto iter_right_subtree_0_matches = filter_result_iterator_t(coll->get_name(), coll->_get_index(), filter_tree_root,
                                                                 enable_lazy_evaluation);

    ASSERT_TRUE(iter_right_subtree_0_matches.init_status().ok());
    ASSERT_EQ(filter_result_iterator_t::invalid, iter_right_subtree_0_matches.validity);
    ASSERT_EQ(0, iter_right_subtree_0_matches.approx_filter_ids_length);
    ASSERT_TRUE(iter_right_subtree_0_matches._get_is_filter_result_initialized());
    ASSERT_EQ(nullptr, iter_right_subtree_0_matches._get_left_it());
    ASSERT_EQ(nullptr, iter_right_subtree_0_matches._get_right_it());

    delete filter_tree_root;
    filter_tree_root = nullptr;

    filter_op = filter::parse_filter_query("(age: 0 && rating: >0) || (age: 0 && rating: >0)", coll->get_schema(), store, doc_id_prefix,
                                           filter_tree_root);
    ASSERT_TRUE(filter_op.ok());

    auto iter_inner_subtree_0_matches = filter_result_iterator_t(coll->get_name(), coll->_get_index(), filter_tree_root,
                                                                 enable_lazy_evaluation);

    ASSERT_TRUE(iter_inner_subtree_0_matches.init_status().ok());
    ASSERT_EQ(filter_result_iterator_t::invalid, iter_inner_subtree_0_matches.validity);
    ASSERT_EQ(0, iter_inner_subtree_0_matches.approx_filter_ids_length);
    ASSERT_FALSE(iter_inner_subtree_0_matches._get_is_filter_result_initialized());
    ASSERT_NE(nullptr, iter_inner_subtree_0_matches._get_left_it());
    ASSERT_NE(nullptr, iter_inner_subtree_0_matches._get_right_it());

    delete filter_tree_root;
    filter_tree_root = nullptr;
}

TEST_F(FilterTest, NotEqualsStringFilter) {
    nlohmann::json schema =
            R"({
                "name": "Collection",
                "fields": [
                    {"name": "name", "type": "string"},
                    {"name": "tags", "type": "string[]"}
                ]
            })"_json;

    Collection* coll = collectionManager.create_collection(schema).get();

    std::ifstream infile(std::string(ROOT_DIR)+"test/numeric_array_documents.jsonl");
    std::string json_line;
    while (std::getline(infile, json_line)) {
        auto add_op = coll->add(json_line);
        ASSERT_TRUE(add_op.ok());
    }
    infile.close();

    const std::string doc_id_prefix = std::to_string(coll->get_collection_id()) + "_" + Collection::DOC_ID_PREFIX + "_";
    filter_node_t* filter_tree_root = nullptr;

    Option<bool> filter_op = filter::parse_filter_query("tags:!= gold", coll->get_schema(), store, doc_id_prefix,
                                                        filter_tree_root);
    ASSERT_TRUE(filter_op.ok());

    auto const enable_lazy_evaluation = true;
    auto computed_not_equals_test = filter_result_iterator_t(coll->get_name(), coll->_get_index(), filter_tree_root,
                                                             enable_lazy_evaluation);
    ASSERT_TRUE(computed_not_equals_test.init_status().ok());
    ASSERT_TRUE(computed_not_equals_test._get_is_filter_result_initialized());

    std::vector<int> expected = {1, 3};
    for (auto const& i : expected) {
        ASSERT_EQ(filter_result_iterator_t::valid, computed_not_equals_test.validity);
        ASSERT_EQ(i, computed_not_equals_test.seq_id);
        computed_not_equals_test.next();
    }
    ASSERT_EQ(filter_result_iterator_t::invalid, computed_not_equals_test.validity);

    delete filter_tree_root;
    filter_tree_root = nullptr;
    filter_op = filter::parse_filter_query("tags: != fine platinum", coll->get_schema(), store, doc_id_prefix,
                                           filter_tree_root);
    ASSERT_TRUE(filter_op.ok());

    auto iter_string_not_equals_test = filter_result_iterator_t(coll->get_name(), coll->_get_index(), filter_tree_root,
                                                                enable_lazy_evaluation);
    ASSERT_TRUE(iter_string_not_equals_test.init_status().ok());
    ASSERT_FALSE(iter_string_not_equals_test._get_is_filter_result_initialized());

    std::vector<uint32_t> validate_ids = {0, 1, 2, 3, 4, 5};
    std::vector<uint32_t> seq_ids = {1, 2, 3, 4, 5, 5};
    std::vector<uint32_t> equals_match_seq_ids = {1, 1, 1, 1, 1, 1};
    std::vector<bool> equals_iterator_valid = {true, true, false, false, false, false};
    expected = {1, 0, 1, 1, 1, -1};
    for (uint32_t i = 0; i < validate_ids.size(); i++) {
        ASSERT_EQ(filter_result_iterator_t::valid, iter_string_not_equals_test.validity);
        ASSERT_EQ(expected[i], iter_string_not_equals_test.is_valid(validate_ids[i]));
        ASSERT_EQ(equals_match_seq_ids[i], iter_string_not_equals_test._get_equals_iterator_id());
        ASSERT_EQ(equals_iterator_valid[i], iter_string_not_equals_test._get_is_equals_iterator_valid());

        if (expected[i] == 1) {
            iter_string_not_equals_test.next();
        }
        ASSERT_EQ(seq_ids[i], iter_string_not_equals_test.seq_id);
    }
    ASSERT_EQ(filter_result_iterator_t::invalid, iter_string_not_equals_test.validity);

    delete filter_tree_root;
    filter_tree_root = nullptr;
    filter_op = filter::parse_filter_query("tags: != [gold, silver]", coll->get_schema(), store, doc_id_prefix,
                                           filter_tree_root);
    ASSERT_TRUE(filter_op.ok());
    auto iter_string_array_not_equals_test = filter_result_iterator_t(coll->get_name(), coll->_get_index(), filter_tree_root,
                                                                      enable_lazy_evaluation);
    ASSERT_TRUE(iter_string_array_not_equals_test.init_status().ok());
    ASSERT_FALSE(iter_string_array_not_equals_test._get_is_filter_result_initialized());
    ASSERT_EQ(5, iter_string_array_not_equals_test.approx_filter_ids_length);

    validate_ids = {0, 1, 2, 3, 4, 5};
    seq_ids = {1, 2, 3, 4, 5, 5};
    expected = {0, 1, 0, 0, 0, -1};
    for (uint32_t i = 0; i < validate_ids.size(); i++) {
        ASSERT_EQ(filter_result_iterator_t::valid, iter_string_array_not_equals_test.validity);
        ASSERT_EQ(expected[i], iter_string_array_not_equals_test.is_valid(validate_ids[i]));

        if (expected[i] == 1) {
            iter_string_array_not_equals_test.next();
        }
        ASSERT_EQ(seq_ids[i], iter_string_array_not_equals_test.seq_id);
    }
    ASSERT_EQ(filter_result_iterator_t::invalid, iter_string_array_not_equals_test.validity);

    delete filter_tree_root;
    filter_tree_root = nullptr;

    auto docs = {
            R"({
                "name": "James Rowdy",
                "tags": ["copper"]
            })"_json,
            R"({
                "name": "James Rowdy",
                "tags": ["copper"]
            })"_json,
            R"({
                "name": "James Rowdy",
                "tags": ["gold"]
            })"_json
    };

    for (auto const& doc: docs) {
        auto add_op = coll->add(doc.dump());
        ASSERT_TRUE(add_op.ok());
    }

    filter_op = filter::parse_filter_query("tags: != gold", coll->get_schema(), store, doc_id_prefix,
                                           filter_tree_root);
    ASSERT_TRUE(filter_op.ok());

    auto iter_string_not_equals_test_2 = filter_result_iterator_t(coll->get_name(), coll->_get_index(), filter_tree_root,
                                                                  enable_lazy_evaluation);
    ASSERT_TRUE(iter_string_not_equals_test_2.init_status().ok());
    ASSERT_FALSE(iter_string_not_equals_test_2._get_is_filter_result_initialized());

    validate_ids = {1, 2, 3, 4, 5, 6, 7, 8};
    seq_ids = {2, 3, 4, 5, 6, 7, 8, 8};
    expected = {1, 0, 1, 0, 1, 1, 0, -1};
    equals_match_seq_ids = {2, 2, 4, 4, 7, 7, 7, 7};
    equals_iterator_valid = {true, true, true, true, true, true, true, true};
    for (uint32_t i = 0; i < validate_ids.size(); i++) {
        ASSERT_EQ(filter_result_iterator_t::valid, iter_string_not_equals_test_2.validity);
        ASSERT_EQ(expected[i], iter_string_not_equals_test_2.is_valid(validate_ids[i]));
        ASSERT_EQ(equals_match_seq_ids[i], iter_string_not_equals_test_2._get_equals_iterator_id());
        ASSERT_EQ(equals_iterator_valid[i], iter_string_not_equals_test_2._get_is_equals_iterator_valid());

        if (expected[i] == 1) {
            iter_string_not_equals_test_2.next();
        }
        ASSERT_EQ(seq_ids[i], iter_string_not_equals_test_2.seq_id);
    }
    ASSERT_EQ(filter_result_iterator_t::invalid, iter_string_not_equals_test_2.validity);

    iter_string_not_equals_test_2.reset();
    validate_ids = {2, 5, 7, 8};
    seq_ids = {3, 6, 8, 8};
    expected = {0, 1, 0, -1};
    equals_match_seq_ids = {2, 7, 7, 7};
    equals_iterator_valid = {true, true, true, true};
    for (uint32_t i = 0; i < validate_ids.size(); i++) {
        ASSERT_EQ(filter_result_iterator_t::valid, iter_string_not_equals_test_2.validity);
        ASSERT_EQ(expected[i], iter_string_not_equals_test_2.is_valid(validate_ids[i]));
        ASSERT_EQ(equals_match_seq_ids[i], iter_string_not_equals_test_2._get_equals_iterator_id());
        ASSERT_EQ(equals_iterator_valid[i], iter_string_not_equals_test_2._get_is_equals_iterator_valid());

        if (expected[i] == 1) {
            iter_string_not_equals_test_2.next();
        }
        ASSERT_EQ(seq_ids[i], iter_string_not_equals_test_2.seq_id);
    }
    ASSERT_EQ(filter_result_iterator_t::invalid, iter_string_not_equals_test_2.validity);

    delete filter_tree_root;
    filter_tree_root = nullptr;

    filter_op = filter::parse_filter_query("name: James || tags: != bronze", coll->get_schema(), store, doc_id_prefix,
                                           filter_tree_root);
    ASSERT_TRUE(filter_op.ok());

    auto iter_not_equals_or_test = filter_result_iterator_t(coll->get_name(), coll->_get_index(),
                                                            filter_tree_root, enable_lazy_evaluation);
    ASSERT_TRUE(iter_not_equals_or_test.init_status().ok());
    ASSERT_FALSE(iter_not_equals_or_test._get_is_filter_result_initialized());

    validate_ids = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    seq_ids = {1, 2, 3, 4, 5, 6, 7, 8, 8};
    expected = {1, 1, 0, 1, 0, 1, 1, 1, -1};
    for (uint32_t i = 0; i < validate_ids.size(); i++) {
        ASSERT_EQ(filter_result_iterator_t::valid, iter_not_equals_or_test.validity);
        ASSERT_EQ(expected[i], iter_not_equals_or_test.is_valid(validate_ids[i]));

        if (expected[i] == 1) {
            iter_not_equals_or_test.next();
        }
        ASSERT_EQ(seq_ids[i], iter_not_equals_or_test.seq_id);
    }
    ASSERT_EQ(filter_result_iterator_t::invalid, iter_not_equals_or_test.validity);

    delete filter_tree_root;
    filter_tree_root = nullptr;
    filter_op = filter::parse_filter_query("tags: != silver || tags: != gold", coll->get_schema(), store, doc_id_prefix,
                                           filter_tree_root);
    ASSERT_TRUE(filter_op.ok());

    auto iter_not_equals_or_test_2 = filter_result_iterator_t(coll->get_name(), coll->_get_index(),
                                                               filter_tree_root, enable_lazy_evaluation);
    ASSERT_TRUE(iter_not_equals_or_test_2.init_status().ok());

    validate_ids = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    seq_ids = {1, 2, 3, 4, 5, 6, 7, 8, 8};
    expected = {0, 1, 1, 1, 0, 1, 1, 1, -1};
    for (uint32_t i = 0; i < validate_ids.size(); i++) {
        ASSERT_EQ(filter_result_iterator_t::valid, iter_not_equals_or_test_2.validity);
        ASSERT_EQ(expected[i], iter_not_equals_or_test_2.is_valid(validate_ids[i]));

        if (expected[i] == 1) {
            iter_not_equals_or_test_2.next();
        }
        ASSERT_EQ(seq_ids[i], iter_not_equals_or_test_2.seq_id);
    }
    ASSERT_EQ(filter_result_iterator_t::invalid, iter_not_equals_or_test_2.validity);

    delete filter_tree_root;
    filter_tree_root = nullptr;
    filter_op = filter::parse_filter_query("name: James && tags: != gold", coll->get_schema(), store, doc_id_prefix,
                                           filter_tree_root);
    ASSERT_TRUE(filter_op.ok());

    auto iter_not_equals_and_test = filter_result_iterator_t(coll->get_name(), coll->_get_index(),
                                                             filter_tree_root, enable_lazy_evaluation);
    ASSERT_TRUE(iter_not_equals_and_test.init_status().ok());
    ASSERT_TRUE(iter_not_equals_and_test._get_is_filter_result_initialized());

    validate_ids = {4, 5, 6, 7};
    seq_ids = {5, 6, 6, 6};
    expected = {0, 1, 1, -1};

    ASSERT_EQ(filter_result_iterator_t::valid, iter_not_equals_and_test.validity);
    for (uint32_t i = 0; i < validate_ids.size(); i++) {
        ASSERT_EQ(expected[i], iter_not_equals_and_test.is_valid(validate_ids[i]));

        if (expected[i] == 1) {
            iter_not_equals_and_test.next();
        }
        ASSERT_EQ(seq_ids[i], iter_not_equals_and_test.seq_id);
    }
    ASSERT_EQ(filter_result_iterator_t::invalid, iter_not_equals_and_test.validity);

    delete filter_tree_root;
    filter_tree_root = nullptr;
    filter_op = filter::parse_filter_query("tags: != silver && tags: != gold", coll->get_schema(), store, doc_id_prefix,
                                           filter_tree_root);
    ASSERT_TRUE(filter_op.ok());
    ASSERT_TRUE(iter_not_equals_and_test._get_is_filter_result_initialized());

    auto iter_not_equals_and_test_2 = filter_result_iterator_t(coll->get_name(), coll->_get_index(),
                                                                filter_tree_root, enable_lazy_evaluation);
    ASSERT_TRUE(iter_not_equals_and_test_2.init_status().ok());

    validate_ids = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    seq_ids = {1, 5, 5, 5, 5, 6, 6, 6, 6};
    expected = {0, 1, 0, 0, 0, 1, 1, -1, -1};

    ASSERT_EQ(filter_result_iterator_t::valid, iter_not_equals_and_test_2.validity);
    for (uint32_t i = 0; i < validate_ids.size(); i++) {
        ASSERT_EQ(expected[i], iter_not_equals_and_test_2.is_valid(validate_ids[i]));

        if (expected[i] == 1) {
            iter_not_equals_and_test_2.next();
        }
        ASSERT_EQ(seq_ids[i], iter_not_equals_and_test_2.seq_id);
    }
    ASSERT_EQ(filter_result_iterator_t::invalid, iter_not_equals_and_test_2.validity);

    delete filter_tree_root;
}

TEST_F(FilterTest, NumericFilterIterator) {
    nlohmann::json schema =
            R"({
                "name": "Collection",
                "fields": [
                    {"name": "rating", "type": "float"},
                    {"name": "age", "type": "int32"},
                    {"name": "years", "type": "int32[]"},
                    {"name": "timestamps", "type": "int64[]"}
                ]
            })"_json;

    Collection* coll = collectionManager.create_collection(schema).get();

    std::ifstream infile(std::string(ROOT_DIR)+"test/numeric_array_documents.jsonl");
    std::string json_line;
    while (std::getline(infile, json_line)) {
        auto add_op = coll->add(json_line);
        ASSERT_TRUE(add_op.ok());
    }
    infile.close();

    const std::string doc_id_prefix = std::to_string(coll->get_collection_id()) + "_" + Collection::DOC_ID_PREFIX + "_";
    filter_node_t* filter_tree_root = nullptr;

    Option<bool> filter_op = filter::parse_filter_query("age: > 32", coll->get_schema(), store, doc_id_prefix,
                                                        filter_tree_root);
    ASSERT_TRUE(filter_op.ok());

    auto const enable_lazy_evaluation = true;
    auto const disable_lazy_evaluation = false;
    auto computed_greater_than_test = filter_result_iterator_t(coll->get_name(), coll->_get_index(), filter_tree_root,
                                                               enable_lazy_evaluation);
    ASSERT_TRUE(computed_greater_than_test.init_status().ok());
    ASSERT_TRUE(computed_greater_than_test._get_is_filter_result_initialized());

    std::vector<int> expected = {1, 3};
    for (auto const& i : expected) {
        ASSERT_EQ(filter_result_iterator_t::valid, computed_greater_than_test.validity);
        ASSERT_EQ(i, computed_greater_than_test.seq_id);
        computed_greater_than_test.next();
    }
    ASSERT_EQ(filter_result_iterator_t::invalid, computed_greater_than_test.validity);

    delete filter_tree_root;
    filter_tree_root = nullptr;
    filter_op = filter::parse_filter_query("age: >= 32", coll->get_schema(), store, doc_id_prefix,
                                           filter_tree_root);
    ASSERT_TRUE(filter_op.ok());

    auto iter_greater_than_test = filter_result_iterator_t(coll->get_name(), coll->_get_index(), filter_tree_root,
                                                           enable_lazy_evaluation);
    ASSERT_TRUE(iter_greater_than_test.init_status().ok());
    ASSERT_FALSE(iter_greater_than_test._get_is_filter_result_initialized());

    std::vector<uint32_t> validate_ids = {0, 1, 2, 3, 4, 5};
    std::vector<uint32_t> seq_ids = {1, 3, 3, 4, 4, 4};
    expected = {0, 1, 0, 1, 1, -1};
    for (uint32_t i = 0; i < validate_ids.size(); i++) {
        if (i < 5) {
            ASSERT_EQ(filter_result_iterator_t::valid, iter_greater_than_test.validity);
        } else {
            ASSERT_EQ(filter_result_iterator_t::invalid, iter_greater_than_test.validity);
        }
        ASSERT_EQ(expected[i], iter_greater_than_test.is_valid(validate_ids[i]));

        if (expected[i] == 1) {
            iter_greater_than_test.next();
        }
        ASSERT_EQ(seq_ids[i], iter_greater_than_test.seq_id);
    }
    ASSERT_EQ(filter_result_iterator_t::invalid, iter_greater_than_test.validity);

    iter_greater_than_test.reset();
    validate_ids = {0, 1, 3, 5};
    seq_ids = {1, 3, 4, 4};
    expected = {0, 1, 1, -1};
    for (uint32_t i = 0; i < validate_ids.size(); i++) {
        ASSERT_EQ(filter_result_iterator_t::valid, iter_greater_than_test.validity);
        ASSERT_EQ(expected[i], iter_greater_than_test.is_valid(validate_ids[i]));

        if (expected[i] == 1) {
            iter_greater_than_test.next();
        }
        ASSERT_EQ(seq_ids[i], iter_greater_than_test.seq_id);
    }
    ASSERT_EQ(filter_result_iterator_t::invalid, iter_greater_than_test.validity);

    // With enable_lazy_evaluation = false, filter result should be initialized.
    {
        auto iter_greater_than_test_non_lazy = filter_result_iterator_t(coll->get_name(), coll->_get_index(),
                                                                        filter_tree_root, disable_lazy_evaluation);
        ASSERT_TRUE(iter_greater_than_test_non_lazy.init_status().ok());
        ASSERT_TRUE(iter_greater_than_test_non_lazy._get_is_filter_result_initialized());

        validate_ids = {0, 1, 2, 3, 4, 5};
        seq_ids = {1, 3, 3, 4, 4, 4};
        expected = {0, 1, 0, 1, 1, -1};

        ASSERT_EQ(filter_result_iterator_t::valid, iter_greater_than_test_non_lazy.validity);
        for (uint32_t i = 0; i < validate_ids.size(); i++) {
            ASSERT_EQ(expected[i], iter_greater_than_test_non_lazy.is_valid(validate_ids[i]));

            if (expected[i] == 1) {
                iter_greater_than_test_non_lazy.next();
            }
            ASSERT_EQ(seq_ids[i], iter_greater_than_test_non_lazy.seq_id);
        }
        ASSERT_EQ(filter_result_iterator_t::invalid, iter_greater_than_test_non_lazy.validity);
    }

    delete filter_tree_root;
    filter_tree_root = nullptr;
    filter_op = filter::parse_filter_query("age: != 21", coll->get_schema(), store, doc_id_prefix,
                                           filter_tree_root);
    ASSERT_TRUE(filter_op.ok());

    auto iter_not_equals_test = filter_result_iterator_t(coll->get_name(), coll->_get_index(), filter_tree_root,
                                                         enable_lazy_evaluation);
    ASSERT_TRUE(iter_not_equals_test.init_status().ok());
    ASSERT_FALSE(iter_not_equals_test._get_is_filter_result_initialized());

    validate_ids = {0, 1, 2, 3, 4, 5};
    seq_ids = {1, 2, 3, 4, 5, 5};
    expected = {1, 1, 0, 1, 1, -1};

    for (uint32_t i = 0; i < validate_ids.size(); i++) {
        ASSERT_EQ(filter_result_iterator_t::valid, iter_not_equals_test.validity);
        ASSERT_EQ(expected[i], iter_not_equals_test.is_valid(validate_ids[i]));

        if (expected[i] == 1) {
            iter_not_equals_test.next();
        }
        ASSERT_EQ(seq_ids[i], iter_not_equals_test.seq_id);
    }
    ASSERT_EQ(filter_result_iterator_t::invalid, iter_not_equals_test.validity);

    // With enable_lazy_evaluation = false, filter result should be initialized.
    {
        auto iter_not_equals_test_non_lazy = filter_result_iterator_t(coll->get_name(), coll->_get_index(),
                                                                      filter_tree_root, disable_lazy_evaluation);
        ASSERT_TRUE(iter_not_equals_test_non_lazy.init_status().ok());
        ASSERT_TRUE(iter_not_equals_test_non_lazy._get_is_filter_result_initialized());

        validate_ids = {0, 1, 2, 3, 4, 5};
        seq_ids = {1, 3, 3, 4, 4, 4};
        expected = {1, 1, 0, 1, 1, -1};

        ASSERT_EQ(filter_result_iterator_t::valid, iter_not_equals_test_non_lazy.validity);
        for (uint32_t i = 0; i < validate_ids.size(); i++) {
            ASSERT_EQ(expected[i], iter_not_equals_test_non_lazy.is_valid(validate_ids[i]));

            if (expected[i] == 1) {
                iter_not_equals_test_non_lazy.next();
            }
            ASSERT_EQ(seq_ids[i], iter_not_equals_test_non_lazy.seq_id);
        }
        ASSERT_EQ(filter_result_iterator_t::invalid, iter_not_equals_test_non_lazy.validity);
    }

    delete filter_tree_root;
    filter_tree_root = nullptr;
    filter_op = filter::parse_filter_query("age: != [21]", coll->get_schema(), store, doc_id_prefix,
                                           filter_tree_root);
    ASSERT_TRUE(filter_op.ok());

    auto iter_not_equals_test_2 = filter_result_iterator_t(coll->get_name(), coll->_get_index(), filter_tree_root,
                                                           enable_lazy_evaluation);
    ASSERT_TRUE(iter_not_equals_test_2.init_status().ok());
    ASSERT_FALSE(iter_not_equals_test_2._get_is_filter_result_initialized());

    validate_ids = {0, 1, 2, 3, 4, 5};
    seq_ids = {1, 2, 3, 4, 5, 5};
    expected = {1, 1, 0, 1, 1, -1};
    std::vector<bool> equals_iterator_valid = {true, true, true, false, false, false};
    std::vector<uint32_t> equals_match_seq_ids = {2, 2, 2, 2, 2, 2};

    for (uint32_t i = 0; i < validate_ids.size(); i++) {
        ASSERT_EQ(filter_result_iterator_t::valid, iter_not_equals_test_2.validity);
        ASSERT_EQ(expected[i], iter_not_equals_test_2.is_valid(validate_ids[i]));

        ASSERT_EQ(equals_iterator_valid[i], iter_not_equals_test_2._get_is_equals_iterator_valid());
        ASSERT_EQ(equals_match_seq_ids[i], iter_not_equals_test_2._get_equals_iterator_id());

        if (expected[i] == 1) {
            iter_not_equals_test_2.next();
        }
        ASSERT_EQ(seq_ids[i], iter_not_equals_test_2.seq_id);
    }
    ASSERT_EQ(filter_result_iterator_t::invalid, iter_not_equals_test_2.validity);

    // With enable_lazy_evaluation = false, filter result should be initialized.
    {
        auto iter_not_equals_test_2_non_lazy = filter_result_iterator_t(coll->get_name(), coll->_get_index(),
                                                                        filter_tree_root, disable_lazy_evaluation);
        ASSERT_TRUE(iter_not_equals_test_2_non_lazy.init_status().ok());
        ASSERT_TRUE(iter_not_equals_test_2_non_lazy._get_is_filter_result_initialized());

        validate_ids = {0, 1, 2, 3, 4, 5};
        seq_ids = {1, 3, 3, 4, 4, 4};
        expected = {1, 1, 0, 1, 1, -1};

        ASSERT_EQ(filter_result_iterator_t::valid, iter_not_equals_test_2_non_lazy.validity);
        for (uint32_t i = 0; i < validate_ids.size(); i++) {
            ASSERT_EQ(expected[i], iter_not_equals_test_2_non_lazy.is_valid(validate_ids[i]));

            if (expected[i] == 1) {
                iter_not_equals_test_2_non_lazy.next();
            }
            ASSERT_EQ(seq_ids[i], iter_not_equals_test_2_non_lazy.seq_id);
        }
        ASSERT_EQ(filter_result_iterator_t::invalid, iter_not_equals_test_2_non_lazy.validity);
    }

    delete filter_tree_root;
    filter_tree_root = nullptr;
    filter_op = filter::parse_filter_query("age: [<=21, >32]", coll->get_schema(), store, doc_id_prefix,
                                           filter_tree_root);
    ASSERT_TRUE(filter_op.ok());

    auto iter_multivalue_filter = filter_result_iterator_t(coll->get_name(), coll->_get_index(), filter_tree_root,
                                                           enable_lazy_evaluation);
    ASSERT_TRUE(iter_multivalue_filter.init_status().ok());
    ASSERT_FALSE(iter_multivalue_filter._get_is_filter_result_initialized());

    validate_ids = {0, 1, 2, 3, 4, 5};
    seq_ids = {1, 2, 3, 3, 3, 3};
    expected = {0, 1, 1, 1, -1, -1};

    for (uint32_t i = 0; i < validate_ids.size(); i++) {
        if (i < 4) {
            ASSERT_EQ(filter_result_iterator_t::valid, iter_multivalue_filter.validity);
        } else {
            ASSERT_EQ(filter_result_iterator_t::invalid, iter_multivalue_filter.validity);
        }
        ASSERT_EQ(expected[i], iter_multivalue_filter.is_valid(validate_ids[i]));

        if (expected[i] == 1) {
            iter_multivalue_filter.next();
        }
        ASSERT_EQ(seq_ids[i], iter_multivalue_filter.seq_id);
    }
    ASSERT_EQ(filter_result_iterator_t::invalid, iter_multivalue_filter.validity);

    // With enable_lazy_evaluation = false, filter result should be initialized.
    {
        auto iter_multivalue_filter_non_lazy = filter_result_iterator_t(coll->get_name(), coll->_get_index(),
                                                                        filter_tree_root, disable_lazy_evaluation);
        ASSERT_TRUE(iter_multivalue_filter_non_lazy.init_status().ok());
        ASSERT_TRUE(iter_multivalue_filter_non_lazy._get_is_filter_result_initialized());

        validate_ids = {0, 1, 2, 3, 4, 5};
        seq_ids = {1, 2, 3, 3, 3, 3};
        expected = {0, 1, 1, 1, -1, -1};

        ASSERT_EQ(filter_result_iterator_t::valid, iter_multivalue_filter_non_lazy.validity);
        for (uint32_t i = 0; i < validate_ids.size(); i++) {
            ASSERT_EQ(expected[i], iter_multivalue_filter_non_lazy.is_valid(validate_ids[i]));

            if (expected[i] == 1) {
                iter_multivalue_filter_non_lazy.next();
            }
            ASSERT_EQ(seq_ids[i], iter_multivalue_filter_non_lazy.seq_id);
        }
        ASSERT_EQ(filter_result_iterator_t::invalid, iter_multivalue_filter_non_lazy.validity);
    }

    delete filter_tree_root;
    filter_tree_root = nullptr;
    filter_op = filter::parse_filter_query("age: != [<24, >44]", coll->get_schema(), store, doc_id_prefix,
                                           filter_tree_root);
    ASSERT_TRUE(filter_op.ok());

    auto iter_multivalue_filter_2 = filter_result_iterator_t(coll->get_name(), coll->_get_index(), filter_tree_root,
                                                             enable_lazy_evaluation);
    ASSERT_TRUE(iter_multivalue_filter_2.init_status().ok());
    ASSERT_FALSE(iter_multivalue_filter_2._get_is_filter_result_initialized());

    validate_ids = {0, 1, 2, 3, 4, 5};
    seq_ids = {1, 2, 3, 4, 5, 5};
    expected = {1, 1, 0, 0, 1, -1};
    equals_iterator_valid = {true, true, true, true, false, false};
    equals_match_seq_ids = {2, 2, 2, 3, 3, 3};

    for (uint32_t i = 0; i < validate_ids.size(); i++) {
        ASSERT_EQ(filter_result_iterator_t::valid, iter_multivalue_filter_2.validity);
        ASSERT_EQ(expected[i], iter_multivalue_filter_2.is_valid(validate_ids[i]));

        ASSERT_EQ(equals_iterator_valid[i], iter_multivalue_filter_2._get_is_equals_iterator_valid());
        ASSERT_EQ(equals_match_seq_ids[i], iter_multivalue_filter_2._get_equals_iterator_id());

        if (expected[i] == 1) {
            iter_multivalue_filter_2.next();
        }
        ASSERT_EQ(seq_ids[i], iter_multivalue_filter_2.seq_id);
    }
    ASSERT_EQ(filter_result_iterator_t::invalid, iter_multivalue_filter_2.validity);

    // With enable_lazy_evaluation = false, filter result should be initialized.
    {
        auto iter_multivalue_filter_2_non_lazy = filter_result_iterator_t(coll->get_name(), coll->_get_index(),
                                                                          filter_tree_root, disable_lazy_evaluation);
        ASSERT_TRUE(iter_multivalue_filter_2_non_lazy.init_status().ok());
        ASSERT_TRUE(iter_multivalue_filter_2_non_lazy._get_is_filter_result_initialized());

        validate_ids = {0, 1, 2, 3, 4, 5};
        seq_ids = {1, 4, 4, 4, 4, 4};
        expected = {1, 1, 0, 0, 1, -1};

        ASSERT_EQ(filter_result_iterator_t::valid, iter_multivalue_filter_2_non_lazy.validity);
        for (uint32_t i = 0; i < validate_ids.size(); i++) {
            ASSERT_EQ(expected[i], iter_multivalue_filter_2_non_lazy.is_valid(validate_ids[i]));

            if (expected[i] == 1) {
                iter_multivalue_filter_2_non_lazy.next();
            }
            ASSERT_EQ(seq_ids[i], iter_multivalue_filter_2_non_lazy.seq_id);
        }
        ASSERT_EQ(filter_result_iterator_t::invalid, iter_multivalue_filter_2_non_lazy.validity);
    }

    delete filter_tree_root;
    filter_tree_root = nullptr;
    filter_op = filter::parse_filter_query("age: [21..32, >44]", coll->get_schema(), store, doc_id_prefix,
                                           filter_tree_root);
    ASSERT_TRUE(filter_op.ok());

    auto iter_multivalue_filter_3 = filter_result_iterator_t(coll->get_name(), coll->_get_index(), filter_tree_root,
                                                             enable_lazy_evaluation);
    ASSERT_TRUE(iter_multivalue_filter_3.init_status().ok());
    ASSERT_FALSE(iter_multivalue_filter_3._get_is_filter_result_initialized());

    validate_ids = {0, 1, 2, 3, 4, 5};
    seq_ids = {2, 2, 3, 4, 4, 4};
    expected = {1, 0, 1, 1, 1, -1};
    equals_iterator_valid = {true, true, true, true, true, false};
    equals_match_seq_ids = {0, 2, 2, 3, 4, 4};

    for (uint32_t i = 0; i < validate_ids.size(); i++) {
        if (i < 5) {
            ASSERT_EQ(filter_result_iterator_t::valid, iter_multivalue_filter_3.validity);
        } else {
            ASSERT_EQ(filter_result_iterator_t::invalid, iter_multivalue_filter_3.validity);
        }
        ASSERT_EQ(expected[i], iter_multivalue_filter_3.is_valid(validate_ids[i]));

        ASSERT_EQ(equals_iterator_valid[i], iter_multivalue_filter_3._get_is_equals_iterator_valid());
        ASSERT_EQ(equals_match_seq_ids[i], iter_multivalue_filter_3._get_equals_iterator_id());

        if (expected[i] == 1) {
            iter_multivalue_filter_3.next();
        }
        ASSERT_EQ(seq_ids[i], iter_multivalue_filter_3.seq_id);
    }
    ASSERT_EQ(filter_result_iterator_t::invalid, iter_multivalue_filter_3.validity);

    // With enable_lazy_evaluation = false, filter result should be initialized.
    {
        auto iter_multivalue_filter_3_non_lazy = filter_result_iterator_t(coll->get_name(), coll->_get_index(),
                                                                          filter_tree_root, disable_lazy_evaluation);
        ASSERT_TRUE(iter_multivalue_filter_3_non_lazy.init_status().ok());
        ASSERT_TRUE(iter_multivalue_filter_3_non_lazy._get_is_filter_result_initialized());

        validate_ids = {0, 1, 2, 3, 4, 5};
        seq_ids = {2, 2, 3, 4, 4, 4};
        expected = {1, 0, 1, 1, 1, -1};

        ASSERT_EQ(filter_result_iterator_t::valid, iter_multivalue_filter_3_non_lazy.validity);
        for (uint32_t i = 0; i < validate_ids.size(); i++) {
            ASSERT_EQ(expected[i], iter_multivalue_filter_3_non_lazy.is_valid(validate_ids[i]));

            if (expected[i] == 1) {
                iter_multivalue_filter_3_non_lazy.next();
            }
            ASSERT_EQ(seq_ids[i], iter_multivalue_filter_3_non_lazy.seq_id);
        }
        ASSERT_EQ(filter_result_iterator_t::invalid, iter_multivalue_filter_3_non_lazy.validity);
    }

    delete filter_tree_root;
    filter_tree_root = nullptr;
    filter_op = filter::parse_filter_query("rating: <5", coll->get_schema(), store, doc_id_prefix,
                                           filter_tree_root);
    ASSERT_TRUE(filter_op.ok());

    auto computed_greater_than_test_2 = filter_result_iterator_t(coll->get_name(), coll->_get_index(), filter_tree_root,
                                                                 enable_lazy_evaluation);
    ASSERT_TRUE(computed_greater_than_test_2.init_status().ok());
    ASSERT_TRUE(computed_greater_than_test_2._get_is_filter_result_initialized());

    expected = {0, 3};
    for (auto const& i : expected) {
        ASSERT_EQ(filter_result_iterator_t::valid, computed_greater_than_test_2.validity);
        ASSERT_EQ(i, computed_greater_than_test_2.seq_id);
        computed_greater_than_test_2.next();
    }
    ASSERT_EQ(filter_result_iterator_t::invalid, computed_greater_than_test_2.validity);

    delete filter_tree_root;
    filter_tree_root = nullptr;
    filter_op = filter::parse_filter_query("rating: >5", coll->get_schema(), store, doc_id_prefix,
                                           filter_tree_root);
    ASSERT_TRUE(filter_op.ok());

    auto iter_greater_than_test_2 = filter_result_iterator_t(coll->get_name(), coll->_get_index(), filter_tree_root,
                                                             enable_lazy_evaluation);
    ASSERT_TRUE(iter_greater_than_test_2.init_status().ok());
    ASSERT_FALSE(iter_greater_than_test_2._get_is_filter_result_initialized());

    validate_ids = {0, 1, 2, 3, 4, 5};
    seq_ids = {1, 2, 4, 4, 4, 4};
    expected = {0, 1, 1, 0, 1, -1};
    for (uint32_t i = 0; i < validate_ids.size(); i++) {
        if (i < 5) {
            ASSERT_EQ(filter_result_iterator_t::valid, iter_greater_than_test_2.validity);
        } else {
            ASSERT_EQ(filter_result_iterator_t::invalid, iter_greater_than_test_2.validity);
        }
        ASSERT_EQ(expected[i], iter_greater_than_test_2.is_valid(validate_ids[i]));

        if (expected[i] == 1) {
            iter_greater_than_test_2.next();
        }
        ASSERT_EQ(seq_ids[i], iter_greater_than_test_2.seq_id);
    }
    ASSERT_EQ(filter_result_iterator_t::invalid, iter_greater_than_test_2.validity);

    iter_greater_than_test_2.reset();
    validate_ids = {0, 1, 4, 5};
    seq_ids = {1, 2, 4, 4};
    expected = {0, 1, 1, -1};
    for (uint32_t i = 0; i < validate_ids.size(); i++) {
        if (i < 3) {
            ASSERT_EQ(filter_result_iterator_t::valid, iter_greater_than_test_2.validity);
        } else {
            ASSERT_EQ(filter_result_iterator_t::invalid, iter_greater_than_test_2.validity);
        }
        ASSERT_EQ(expected[i], iter_greater_than_test_2.is_valid(validate_ids[i]));

        if (expected[i] == 1) {
            iter_greater_than_test_2.next();
        }
        ASSERT_EQ(seq_ids[i], iter_greater_than_test_2.seq_id);
    }
    ASSERT_EQ(filter_result_iterator_t::invalid, iter_greater_than_test_2.validity);

    delete filter_tree_root;
    filter_tree_root = nullptr;
    filter_op = filter::parse_filter_query("rating: != 7.812", coll->get_schema(), store, doc_id_prefix,
                                           filter_tree_root);
    ASSERT_TRUE(filter_op.ok());

    auto iter_not_equals_test_3 = filter_result_iterator_t(coll->get_name(), coll->_get_index(), filter_tree_root,
                                                           enable_lazy_evaluation);
    ASSERT_TRUE(iter_not_equals_test_3.init_status().ok());
    ASSERT_FALSE(iter_not_equals_test_3._get_is_filter_result_initialized());

    validate_ids = {0, 1, 2, 3, 4, 5};
    seq_ids = {1, 2, 3, 4, 5, 5};
    expected = {1, 1, 0, 1, 1, -1};

    for (uint32_t i = 0; i < validate_ids.size(); i++) {
        ASSERT_EQ(filter_result_iterator_t::valid, iter_not_equals_test_3.validity);
        ASSERT_EQ(expected[i], iter_not_equals_test_3.is_valid(validate_ids[i]));

        if (expected[i] == 1) {
            iter_not_equals_test_3.next();
        }
        ASSERT_EQ(seq_ids[i], iter_not_equals_test_3.seq_id);
    }
    ASSERT_EQ(filter_result_iterator_t::invalid, iter_not_equals_test_3.validity);

    // With enable_lazy_evaluation = false, filter result should be initialized.
    {
        auto iter_not_equals_test_3_non_lazy = filter_result_iterator_t(coll->get_name(), coll->_get_index(),
                                                                        filter_tree_root, disable_lazy_evaluation);
        ASSERT_TRUE(iter_not_equals_test_3_non_lazy.init_status().ok());
        ASSERT_TRUE(iter_not_equals_test_3_non_lazy._get_is_filter_result_initialized());

        validate_ids = {0, 1, 2, 3, 4, 5};
        seq_ids = {1, 3, 3, 4, 4, 4};
        expected = {1, 1, 0, 1, 1, -1};

        ASSERT_EQ(filter_result_iterator_t::valid, iter_not_equals_test_3_non_lazy.validity);
        for (uint32_t i = 0; i < validate_ids.size(); i++) {
            ASSERT_EQ(expected[i], iter_not_equals_test_3_non_lazy.is_valid(validate_ids[i]));

            if (expected[i] == 1) {
                iter_not_equals_test_3_non_lazy.next();
            }
            ASSERT_EQ(seq_ids[i], iter_not_equals_test_3_non_lazy.seq_id);
        }
        ASSERT_EQ(filter_result_iterator_t::invalid, iter_not_equals_test_3_non_lazy.validity);
    }

    delete filter_tree_root;
    filter_tree_root = nullptr;
    filter_op = filter::parse_filter_query("rating: != [7.812]", coll->get_schema(), store, doc_id_prefix,
                                           filter_tree_root);
    ASSERT_TRUE(filter_op.ok());

    auto iter_not_equals_test_4 = filter_result_iterator_t(coll->get_name(), coll->_get_index(), filter_tree_root,
                                                           enable_lazy_evaluation);
    ASSERT_TRUE(iter_not_equals_test_4.init_status().ok());
    ASSERT_FALSE(iter_not_equals_test_4._get_is_filter_result_initialized());

    validate_ids = {0, 1, 2, 3, 4, 5};
    seq_ids = {1, 2, 3, 4, 5, 5};
    expected = {1, 1, 0, 1, 1, -1};
    equals_iterator_valid = {true, true, true, false, false, false};
    equals_match_seq_ids = {2, 2, 2, 2, 2, 2};

    for (uint32_t i = 0; i < validate_ids.size(); i++) {
        ASSERT_EQ(filter_result_iterator_t::valid, iter_not_equals_test_4.validity);
        ASSERT_EQ(expected[i], iter_not_equals_test_4.is_valid(validate_ids[i]));

        ASSERT_EQ(equals_iterator_valid[i], iter_not_equals_test_4._get_is_equals_iterator_valid());
        ASSERT_EQ(equals_match_seq_ids[i], iter_not_equals_test_4._get_equals_iterator_id());

        if (expected[i] == 1) {
            iter_not_equals_test_4.next();
        }
        ASSERT_EQ(seq_ids[i], iter_not_equals_test_4.seq_id);
    }
    ASSERT_EQ(filter_result_iterator_t::invalid, iter_not_equals_test_4.validity);

    // With enable_lazy_evaluation = false, filter result should be initialized.
    {
        auto iter_not_equals_test_4_non_lazy = filter_result_iterator_t(coll->get_name(), coll->_get_index(),
                                                                        filter_tree_root, disable_lazy_evaluation);
        ASSERT_TRUE(iter_not_equals_test_4_non_lazy.init_status().ok());
        ASSERT_TRUE(iter_not_equals_test_4_non_lazy._get_is_filter_result_initialized());

        validate_ids = {0, 1, 2, 3, 4, 5};
        seq_ids = {1, 3, 3, 4, 4, 4};
        expected = {1, 1, 0, 1, 1, -1};

        ASSERT_EQ(filter_result_iterator_t::valid, iter_not_equals_test_4_non_lazy.validity);
        for (uint32_t i = 0; i < validate_ids.size(); i++) {
            ASSERT_EQ(expected[i], iter_not_equals_test_4_non_lazy.is_valid(validate_ids[i]));

            if (expected[i] == 1) {
                iter_not_equals_test_4_non_lazy.next();
            }
            ASSERT_EQ(seq_ids[i], iter_not_equals_test_4_non_lazy.seq_id);
        }
        ASSERT_EQ(filter_result_iterator_t::invalid, iter_not_equals_test_4_non_lazy.validity);
    }

    delete filter_tree_root;
    filter_tree_root = nullptr;
    filter_op = filter::parse_filter_query("rating: [< 1, >6]", coll->get_schema(), store, doc_id_prefix,
                                           filter_tree_root);
    ASSERT_TRUE(filter_op.ok());

    auto iter_multivalue_filter_4 = filter_result_iterator_t(coll->get_name(), coll->_get_index(), filter_tree_root,
                                                             enable_lazy_evaluation);
    ASSERT_TRUE(iter_multivalue_filter_4.init_status().ok());
    ASSERT_FALSE(iter_multivalue_filter_4._get_is_filter_result_initialized());

    validate_ids = {0, 1, 2, 3, 4, 5};
    seq_ids = {1, 2, 3, 3, 3, 3};
    expected = {0, 1, 1, 1, -1, -1};

    for (uint32_t i = 0; i < validate_ids.size(); i++) {
        if (i < 4) {
            ASSERT_EQ(filter_result_iterator_t::valid, iter_multivalue_filter_4.validity);
        } else {
            ASSERT_EQ(filter_result_iterator_t::invalid, iter_multivalue_filter_4.validity);
        }
        ASSERT_EQ(expected[i], iter_multivalue_filter_4.is_valid(validate_ids[i]));

        if (expected[i] == 1) {
            iter_multivalue_filter_4.next();
        }
        ASSERT_EQ(seq_ids[i], iter_multivalue_filter_4.seq_id);
    }
    ASSERT_EQ(filter_result_iterator_t::invalid, iter_multivalue_filter_4.validity);

    // With enable_lazy_evaluation = false, filter result should be initialized.
    {
        auto iter_multivalue_filter_4_non_lazy = filter_result_iterator_t(coll->get_name(), coll->_get_index(),
                                                                          filter_tree_root, disable_lazy_evaluation);
        ASSERT_TRUE(iter_multivalue_filter_4_non_lazy.init_status().ok());
        ASSERT_TRUE(iter_multivalue_filter_4_non_lazy._get_is_filter_result_initialized());

        validate_ids = {0, 1, 2, 3, 4, 5};
        seq_ids = {1, 2, 3, 3, 3, 3};
        expected = {0, 1, 1, 1, -1, -1};

        ASSERT_EQ(filter_result_iterator_t::valid, iter_multivalue_filter_4_non_lazy.validity);
        for (uint32_t i = 0; i < validate_ids.size(); i++) {
            ASSERT_EQ(expected[i], iter_multivalue_filter_4_non_lazy.is_valid(validate_ids[i]));

            if (expected[i] == 1) {
                iter_multivalue_filter_4_non_lazy.next();
            }
            ASSERT_EQ(seq_ids[i], iter_multivalue_filter_4_non_lazy.seq_id);
        }
        ASSERT_EQ(filter_result_iterator_t::invalid, iter_multivalue_filter_4_non_lazy.validity);
    }

    delete filter_tree_root;
    filter_tree_root = nullptr;
    filter_op = filter::parse_filter_query("rating: != [<1, >8]", coll->get_schema(), store, doc_id_prefix,
                                           filter_tree_root);
    ASSERT_TRUE(filter_op.ok());

    auto iter_multivalue_filter_5 = filter_result_iterator_t(coll->get_name(), coll->_get_index(), filter_tree_root,
                                                             enable_lazy_evaluation);
    ASSERT_TRUE(iter_multivalue_filter_5.init_status().ok());
    ASSERT_FALSE(iter_multivalue_filter_5._get_is_filter_result_initialized());

    validate_ids = {0, 1, 2, 3, 4, 5};
    seq_ids = {1, 2, 3, 4, 5, 5};
    expected = {1, 0, 1, 0, 1, -1};
    equals_iterator_valid = {true, true, true, true, false, false};
    equals_match_seq_ids = {1, 1, 3, 3, 3, 3};

    for (uint32_t i = 0; i < validate_ids.size(); i++) {
        ASSERT_EQ(filter_result_iterator_t::valid, iter_multivalue_filter_5.validity);
        ASSERT_EQ(expected[i], iter_multivalue_filter_5.is_valid(validate_ids[i]));

        ASSERT_EQ(equals_iterator_valid[i], iter_multivalue_filter_5._get_is_equals_iterator_valid());
        ASSERT_EQ(equals_match_seq_ids[i], iter_multivalue_filter_5._get_equals_iterator_id());

        if (expected[i] == 1) {
            iter_multivalue_filter_5.next();
        }
        ASSERT_EQ(seq_ids[i], iter_multivalue_filter_5.seq_id);
    }
    ASSERT_EQ(filter_result_iterator_t::invalid, iter_multivalue_filter_5.validity);

    // With enable_lazy_evaluation = false, filter result should be initialized.
    {
        auto iter_multivalue_filter_5_non_lazy = filter_result_iterator_t(coll->get_name(), coll->_get_index(),
                                                                          filter_tree_root, disable_lazy_evaluation);
        ASSERT_TRUE(iter_multivalue_filter_5_non_lazy.init_status().ok());
        ASSERT_TRUE(iter_multivalue_filter_5_non_lazy._get_is_filter_result_initialized());

        validate_ids = {0, 1, 2, 3, 4, 5};
        seq_ids = {2, 2, 4, 4, 4, 4};
        expected = {1, 0, 1, 0, 1, -1};

        ASSERT_EQ(filter_result_iterator_t::valid, iter_multivalue_filter_5_non_lazy.validity);
        for (uint32_t i = 0; i < validate_ids.size(); i++) {
            ASSERT_EQ(expected[i], iter_multivalue_filter_5_non_lazy.is_valid(validate_ids[i]));

            if (expected[i] == 1) {
                iter_multivalue_filter_5_non_lazy.next();
            }
            ASSERT_EQ(seq_ids[i], iter_multivalue_filter_5_non_lazy.seq_id);
        }
        ASSERT_EQ(filter_result_iterator_t::invalid, iter_multivalue_filter_5_non_lazy.validity);
    }

    delete filter_tree_root;
    filter_tree_root = nullptr;
    filter_op = filter::parse_filter_query("rating: [0..6, >8]", coll->get_schema(), store, doc_id_prefix,
                                           filter_tree_root);
    ASSERT_TRUE(filter_op.ok());

    auto iter_multivalue_filter_6 = filter_result_iterator_t(coll->get_name(), coll->_get_index(), filter_tree_root,
                                                             enable_lazy_evaluation);
    ASSERT_TRUE(iter_multivalue_filter_6.init_status().ok());
    ASSERT_FALSE(iter_multivalue_filter_6._get_is_filter_result_initialized());

    validate_ids = {0, 1, 2, 3, 4, 5};
    seq_ids = {1, 3, 3, 4, 4, 4};
    expected = {1, 1, 0, 1, 1, -1};
    equals_iterator_valid = {true, true, true, true, true, false};
    equals_match_seq_ids = {0, 1, 3, 3, 4, 4};

    for (uint32_t i = 0; i < validate_ids.size(); i++) {
        if (i < 5) {
            ASSERT_EQ(filter_result_iterator_t::valid, iter_multivalue_filter_6.validity);
        } else {
            ASSERT_EQ(filter_result_iterator_t::invalid, iter_multivalue_filter_6.validity);
        }
        ASSERT_EQ(expected[i], iter_multivalue_filter_6.is_valid(validate_ids[i]));

        ASSERT_EQ(equals_iterator_valid[i], iter_multivalue_filter_6._get_is_equals_iterator_valid());
        ASSERT_EQ(equals_match_seq_ids[i], iter_multivalue_filter_6._get_equals_iterator_id());

        if (expected[i] == 1) {
            iter_multivalue_filter_6.next();
        }
        ASSERT_EQ(seq_ids[i], iter_multivalue_filter_6.seq_id);
    }
    ASSERT_EQ(filter_result_iterator_t::invalid, iter_multivalue_filter_6.validity);

    // With enable_lazy_evaluation = false, filter result should be initialized.
    {
        auto iter_multivalue_filter_6_non_lazy = filter_result_iterator_t(coll->get_name(), coll->_get_index(),
                                                                          filter_tree_root, disable_lazy_evaluation);
        ASSERT_TRUE(iter_multivalue_filter_6_non_lazy.init_status().ok());
        ASSERT_TRUE(iter_multivalue_filter_6_non_lazy._get_is_filter_result_initialized());

        validate_ids = {0, 1, 2, 3, 4, 5};
        seq_ids = {1, 3, 3, 4, 4, 4};
        expected = {1, 1, 0, 1, 1, -1};

        ASSERT_EQ(filter_result_iterator_t::valid, iter_multivalue_filter_6_non_lazy.validity);
        for (uint32_t i = 0; i < validate_ids.size(); i++) {
            ASSERT_EQ(expected[i], iter_multivalue_filter_6_non_lazy.is_valid(validate_ids[i]));

            if (expected[i] == 1) {
                iter_multivalue_filter_6_non_lazy.next();
            }
            ASSERT_EQ(seq_ids[i], iter_multivalue_filter_6_non_lazy.seq_id);
        }
        ASSERT_EQ(filter_result_iterator_t::invalid, iter_multivalue_filter_6_non_lazy.validity);
    }

    delete filter_tree_root;
}

TEST_F(FilterTest, PrefixStringFilter) {
    auto schema_json =
            R"({
                "name": "Names",
                "fields": [
                    {"name": "name", "type": "string"}
                ]
            })"_json;
    std::vector<nlohmann::json> documents = {
            R"({
                "name": "Steve Jobs"
            })"_json,
            R"({
                "name": "Adam Stator"
            })"_json,
    };

    auto collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_TRUE(collection_create_op.ok());
    Collection* coll = collection_create_op.get();
    for (auto const &json: documents) {
        auto add_op = coll->add(json.dump());
        ASSERT_TRUE(add_op.ok());
    }

    const std::string doc_id_prefix = std::to_string(coll->get_collection_id()) + "_" + Collection::DOC_ID_PREFIX + "_";
    filter_node_t* filter_tree_root = nullptr;

    search_stop_us = UINT64_MAX; // `Index::fuzzy_search_fields` checks for timeout.
    Option<bool> filter_op = filter::parse_filter_query("name:= S*", coll->get_schema(), store, doc_id_prefix,
                                                        filter_tree_root);
    ASSERT_TRUE(filter_op.ok());

    auto const enable_lazy_evaluation = true;
    auto computed_exact_prefix_test = filter_result_iterator_t(coll->get_name(), coll->_get_index(), filter_tree_root,
                                                               enable_lazy_evaluation);
    ASSERT_TRUE(computed_exact_prefix_test.init_status().ok());
    ASSERT_TRUE(computed_exact_prefix_test._get_is_filter_result_initialized());

    std::vector<int> expected = {0};
    for (auto const& i : expected) {
        ASSERT_EQ(filter_result_iterator_t::valid, computed_exact_prefix_test.validity);
        ASSERT_EQ(i, computed_exact_prefix_test.seq_id);
        computed_exact_prefix_test.next();
    }
    ASSERT_EQ(filter_result_iterator_t::invalid, computed_exact_prefix_test.validity);

    delete filter_tree_root;
    filter_tree_root = nullptr;
    filter_op = filter::parse_filter_query("name: S*", coll->get_schema(), store, doc_id_prefix,
                                                        filter_tree_root);
    ASSERT_TRUE(filter_op.ok());

    auto computed_contains_prefix_test = filter_result_iterator_t(coll->get_name(), coll->_get_index(), filter_tree_root,
                                                                  enable_lazy_evaluation);
    ASSERT_TRUE(computed_contains_prefix_test.init_status().ok());
    ASSERT_TRUE(computed_contains_prefix_test._get_is_filter_result_initialized());

    expected = {0, 1};
    for (auto const& i : expected) {
        ASSERT_EQ(filter_result_iterator_t::valid, computed_contains_prefix_test.validity);
        ASSERT_EQ(i, computed_contains_prefix_test.seq_id);
        computed_contains_prefix_test.next();
    }
    ASSERT_EQ(filter_result_iterator_t::invalid, computed_contains_prefix_test.validity);

    delete filter_tree_root;

    documents = {
            R"({
                "name": "Steve Reiley"
            })"_json,
            R"({
                "name": "Storm"
            })"_json,
            R"({
                "name": "Steve Rogers"
            })"_json,
    };

    for (auto const &json: documents) {
        auto add_op = collection_create_op.get()->add(json.dump());
        ASSERT_TRUE(add_op.ok());
    }

    filter_tree_root = nullptr;
    filter_op = filter::parse_filter_query("name:= S*", coll->get_schema(), store, doc_id_prefix,
                                           filter_tree_root);
    ASSERT_TRUE(filter_op.ok());

    auto iter_exact_prefix_test = filter_result_iterator_t(coll->get_name(), coll->_get_index(), filter_tree_root,
                                                           enable_lazy_evaluation);
    ASSERT_TRUE(iter_exact_prefix_test.init_status().ok());
    ASSERT_FALSE(iter_exact_prefix_test._get_is_filter_result_initialized());

    std::vector<uint32_t> validate_ids = {0, 1, 2, 3, 4, 5};
    std::vector<uint32_t> seq_ids = {2, 2, 3, 4, 4, 4};
    std::vector<uint32_t> equals_match_seq_ids = {0, 2, 2, 3, 4, 4};
    std::vector<bool> equals_iterator_valid = {true, true, true, true, true, false};
    expected = {1, 0, 1, 1, 1, -1};
    for (uint32_t i = 0; i < validate_ids.size(); i++) {
        if (i < 5) {
            ASSERT_EQ(filter_result_iterator_t::valid, iter_exact_prefix_test.validity);
        } else {
            ASSERT_EQ(filter_result_iterator_t::invalid, iter_exact_prefix_test.validity);
        }
        ASSERT_EQ(expected[i], iter_exact_prefix_test.is_valid(validate_ids[i]));
        ASSERT_EQ(equals_match_seq_ids[i], iter_exact_prefix_test._get_equals_iterator_id());
        ASSERT_EQ(equals_iterator_valid[i], iter_exact_prefix_test._get_is_equals_iterator_valid());

        if (expected[i] == 1) {
            iter_exact_prefix_test.next();
        }
        ASSERT_EQ(seq_ids[i], iter_exact_prefix_test.seq_id);
    }
    ASSERT_EQ(filter_result_iterator_t::invalid, iter_exact_prefix_test.validity);

    delete filter_tree_root;
    filter_tree_root = nullptr;
    filter_op = filter::parse_filter_query("name: S*", coll->get_schema(), store, doc_id_prefix,
                                           filter_tree_root);
    ASSERT_TRUE(filter_op.ok());

    auto iter_contains_prefix_test = filter_result_iterator_t(coll->get_name(), coll->_get_index(), filter_tree_root,
                                                              enable_lazy_evaluation);
    ASSERT_TRUE(iter_contains_prefix_test.init_status().ok());
    ASSERT_FALSE(iter_contains_prefix_test._get_is_filter_result_initialized());

    validate_ids = {0, 1, 2, 3, 4, 5};
    seq_ids = {1, 2, 3, 4, 4, 4};
    equals_match_seq_ids = {0, 1, 2, 3, 4, 4};
    equals_iterator_valid = {true, true, true, true, true, false};
    expected = {1, 1, 1, 1, 1, -1};
    for (uint32_t i = 0; i < validate_ids.size(); i++) {
        if (i < 5) {
            ASSERT_EQ(filter_result_iterator_t::valid, iter_contains_prefix_test.validity);
        } else {
            ASSERT_EQ(filter_result_iterator_t::invalid, iter_contains_prefix_test.validity);
        }
        ASSERT_EQ(expected[i], iter_contains_prefix_test.is_valid(validate_ids[i]));
        ASSERT_EQ(equals_match_seq_ids[i], iter_contains_prefix_test._get_equals_iterator_id());
        ASSERT_EQ(equals_iterator_valid[i], iter_contains_prefix_test._get_is_equals_iterator_valid());

        if (expected[i] == 1) {
            iter_contains_prefix_test.next();
        }
        ASSERT_EQ(seq_ids[i], iter_contains_prefix_test.seq_id);
    }
    ASSERT_EQ(filter_result_iterator_t::invalid, iter_contains_prefix_test.validity);

    delete filter_tree_root;
    filter_tree_root = nullptr;
    filter_op = filter::parse_filter_query("name:= Steve R*", coll->get_schema(), store, doc_id_prefix,
                                           filter_tree_root);
    ASSERT_TRUE(filter_op.ok());

    auto computed_exact_prefix_test_2 = filter_result_iterator_t(coll->get_name(), coll->_get_index(), filter_tree_root,
                                                                 enable_lazy_evaluation);
    ASSERT_TRUE(computed_exact_prefix_test_2.init_status().ok());
    ASSERT_TRUE(computed_exact_prefix_test_2._get_is_filter_result_initialized());

    expected = {2, 4};
    for (auto const& i : expected) {
        ASSERT_EQ(filter_result_iterator_t::valid, computed_exact_prefix_test_2.validity);
        ASSERT_EQ(i, computed_exact_prefix_test_2.seq_id);
        computed_exact_prefix_test_2.next();
    }
    ASSERT_EQ(filter_result_iterator_t::invalid, computed_exact_prefix_test_2.validity);

    delete filter_tree_root;
    filter_tree_root = nullptr;
    filter_op = filter::parse_filter_query("name: Steve R*", coll->get_schema(), store, doc_id_prefix,
                                           filter_tree_root);
    ASSERT_TRUE(filter_op.ok());

    auto computed_contains_prefix_test_2 = filter_result_iterator_t(coll->get_name(), coll->_get_index(), filter_tree_root,
                                                                    enable_lazy_evaluation);
    ASSERT_TRUE(computed_contains_prefix_test_2.init_status().ok());
    ASSERT_TRUE(computed_contains_prefix_test_2._get_is_filter_result_initialized());

    expected = {2, 4};
    for (auto const& i : expected) {
        ASSERT_EQ(filter_result_iterator_t::valid, computed_contains_prefix_test_2.validity);
        ASSERT_EQ(i, computed_contains_prefix_test_2.seq_id);
        computed_contains_prefix_test_2.next();
    }
    ASSERT_EQ(filter_result_iterator_t::invalid, computed_contains_prefix_test_2.validity);

    delete filter_tree_root;

    documents = {
            R"({
                "name": "Steve Runner foo"
            })"_json,
            R"({
                "name": "foo Steve Runner"
            })"_json,
    };

    for (auto const &json: documents) {
        auto add_op = collection_create_op.get()->add(json.dump());
        ASSERT_TRUE(add_op.ok());
    }

    filter_tree_root = nullptr;
    filter_op = filter::parse_filter_query("name:= Steve R*", coll->get_schema(), store, doc_id_prefix,
                                           filter_tree_root);
    ASSERT_TRUE(filter_op.ok());

    auto iter_exact_prefix_test_2 = filter_result_iterator_t(coll->get_name(), coll->_get_index(), filter_tree_root,
                                                             enable_lazy_evaluation);
    ASSERT_TRUE(iter_exact_prefix_test_2.init_status().ok());
    ASSERT_FALSE(iter_exact_prefix_test_2._get_is_filter_result_initialized());

    validate_ids = {0, 1, 2, 3, 4, 5, 6, 7};
    seq_ids = {2, 2, 4, 4, 5, 5, 5, 5};
    equals_match_seq_ids = {2, 2, 2, 4, 4, 5, 5, 5};
    equals_iterator_valid = {true, true, true, true, true, true, false, false};
    expected = {0, 0, 1, 0, 1, 1, -1, -1};
    for (uint32_t i = 0; i < validate_ids.size(); i++) {
        if (i < 6) {
            ASSERT_EQ(filter_result_iterator_t::valid, iter_exact_prefix_test_2.validity);
        } else {
            ASSERT_EQ(filter_result_iterator_t::invalid, iter_exact_prefix_test_2.validity);
        }
        ASSERT_EQ(expected[i], iter_exact_prefix_test_2.is_valid(validate_ids[i]));
        ASSERT_EQ(equals_match_seq_ids[i], iter_exact_prefix_test_2._get_equals_iterator_id());
        ASSERT_EQ(equals_iterator_valid[i], iter_exact_prefix_test_2._get_is_equals_iterator_valid());

        if (expected[i] == 1) {
            iter_exact_prefix_test_2.next();
        }
        ASSERT_EQ(seq_ids[i], iter_exact_prefix_test_2.seq_id);
    }
    ASSERT_EQ(filter_result_iterator_t::invalid, iter_exact_prefix_test_2.validity);

    delete filter_tree_root;
    filter_tree_root = nullptr;
    filter_op = filter::parse_filter_query("name: Steve R*", coll->get_schema(), store, doc_id_prefix,
                                           filter_tree_root);
    ASSERT_TRUE(filter_op.ok());

    auto iter_contains_prefix_test_2 = filter_result_iterator_t(coll->get_name(), coll->_get_index(), filter_tree_root,
                                                                enable_lazy_evaluation);
    ASSERT_TRUE(iter_contains_prefix_test_2.init_status().ok());
    ASSERT_FALSE(iter_contains_prefix_test_2._get_is_filter_result_initialized());

    validate_ids = {0, 1, 2, 3, 4, 5, 6, 7};
    seq_ids = {2, 2, 4, 4, 5, 6, 6, 6};
    equals_match_seq_ids = {2, 2, 2, 4, 4, 5, 6, 6};
    equals_iterator_valid = {true, true, true, true, true, true, true, false};
    expected = {0, 0, 1, 0, 1, 1, 1, -1};
    for (uint32_t i = 0; i < validate_ids.size(); i++) {
        if (i < 7) {
            ASSERT_EQ(filter_result_iterator_t::valid, iter_contains_prefix_test_2.validity);
        } else {
            ASSERT_EQ(filter_result_iterator_t::invalid, iter_contains_prefix_test_2.validity);
        }
        ASSERT_EQ(expected[i], iter_contains_prefix_test_2.is_valid(validate_ids[i]));
        ASSERT_EQ(equals_match_seq_ids[i], iter_contains_prefix_test_2._get_equals_iterator_id());
        ASSERT_EQ(equals_iterator_valid[i], iter_contains_prefix_test_2._get_is_equals_iterator_valid());

        if (expected[i] == 1) {
            iter_contains_prefix_test_2.next();
        }
        ASSERT_EQ(seq_ids[i], iter_contains_prefix_test_2.seq_id);
    }
    ASSERT_EQ(filter_result_iterator_t::invalid, iter_contains_prefix_test_2.validity);

    delete filter_tree_root;
}
