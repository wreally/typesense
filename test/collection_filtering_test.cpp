#include <gtest/gtest.h>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <collection_manager.h>
#include "collection.h"

class CollectionFilteringTest : public ::testing::Test {
protected:
    std::string state_dir_path = "/tmp/typesense_test/collection_filtering";
    Store *store;
    CollectionManager & collectionManager = CollectionManager::get_instance();
    std::atomic<bool> quit = false;

    std::vector<std::string> query_fields;
    std::vector<sort_by> sort_fields;

    void setupCollection() {
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

TEST_F(CollectionFilteringTest, FilterOnTextFields) {
    Collection *coll_array_fields;

    std::ifstream infile(std::string(ROOT_DIR)+"test/numeric_array_documents.jsonl");
    std::vector<field> fields = {
            field("name", field_types::STRING, false),
            field("age", field_types::INT32, false),
            field("years", field_types::INT32_ARRAY, false),
            field("tags", field_types::STRING_ARRAY, true)
    };

    std::vector<sort_by> sort_fields = { sort_by("age", "DESC") };

    coll_array_fields = collectionManager.get_collection("coll_array_fields").get();
    if(coll_array_fields == nullptr) {
        coll_array_fields = collectionManager.create_collection("coll_array_fields", 4, fields, "age").get();
    }

    std::string json_line;

    while (std::getline(infile, json_line)) {
        coll_array_fields->add(json_line);
    }

    infile.close();

    query_fields = {"name"};
    std::vector<std::string> facets;
    nlohmann::json results = coll_array_fields->search("Jeremy", query_fields, "tags: gold", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(3, results["hits"].size());

    std::vector<std::string> ids = {"4", "0", "2"};

    for(size_t i = 0; i < results["hits"].size(); i++) {
        nlohmann::json result = results["hits"].at(i);
        std::string result_id = result["document"]["id"];
        std::string id = ids.at(i);
        ASSERT_STREQ(id.c_str(), result_id.c_str());
    }

    results = coll_array_fields->search("Jeremy", query_fields, "tags : fine PLATINUM", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(1, results["hits"].size());

    results = coll_array_fields->search("Jeremy", query_fields, "tags : foobarbaz", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(0, results["hits"].size());

    // using just ":", filtering should return documents that contain ALL tokens in the filter expression
    results = coll_array_fields->search("Jeremy", query_fields, "tags : PLATINUM", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(1, results["hits"].size());

    // no documents contain "white"
    results = coll_array_fields->search("Jeremy", query_fields, "tags : WHITE", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(0, results["hits"].size());

    // no documents contain both "white" and "platinum", so
    results = coll_array_fields->search("Jeremy", query_fields, "tags : WHITE PLATINUM", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(0, results["hits"].size());

    // with exact match operator (:=) partial matches are not allowed
    results = coll_array_fields->search("Jeremy", query_fields, "tags:= PLATINUM", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(0, results["hits"].size());

    results = coll_array_fields->search("Jeremy", query_fields, "tags : bronze", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(2, results["hits"].size());

    ids = {"4", "2"};

    for(size_t i = 0; i < results["hits"].size(); i++) {
        nlohmann::json result = results["hits"].at(i);
        std::string result_id = result["document"]["id"];
        std::string id = ids.at(i);
        ASSERT_STREQ(id.c_str(), result_id.c_str());
    }

    // search with a list of tags, also testing extra padding of space
    results = coll_array_fields->search("Jeremy", query_fields, "tags: [bronze,   silver]", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(4, results["hits"].size());

    ids = {"3", "4", "0", "2"};

    for(size_t i = 0; i < results["hits"].size(); i++) {
        nlohmann::json result = results["hits"].at(i);
        std::string result_id = result["document"]["id"];
        std::string id = ids.at(i);
        ASSERT_STREQ(id.c_str(), result_id.c_str());
    }

    // need to be exact matches
    results = coll_array_fields->search("Jeremy", query_fields, "tags: bronze", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(2, results["hits"].size());

    // when comparators are used, they should be ignored
    results = coll_array_fields->search("Jeremy", query_fields, "tags:<bronze", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(2, results["hits"].size());

    results = coll_array_fields->search("Jeremy", query_fields, "tags:<=BRONZE", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(2, results["hits"].size());

    results = coll_array_fields->search("Jeremy", query_fields, "tags:>BRONZE", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(2, results["hits"].size());

    // bad filter value (empty)
    auto res_op = coll_array_fields->search("Jeremy", query_fields, "tags:=", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false});
    ASSERT_FALSE(res_op.ok());
    ASSERT_EQ("Error with filter field `tags`: Filter value cannot be empty.", res_op.error());

    collectionManager.drop_collection("coll_array_fields");

    auto schema_json =
            R"({
                "name": "title",
                "fields": [
                    {"name": "title", "type": "string"},
                    {"name": "titles", "type": "string[]"}
                ]
            })"_json;

    std::vector<nlohmann::json> documents = {
            R"({
                "title": "foo bar baz",
                "titles": []
            })"_json,
            R"({
                "title": "foo bar baz",
                "titles": ["foo bar baz"]
            })"_json,
            R"({
                "title": "foo bar baz",
                "titles": ["bar foo baz", "foo bar baz"]
            })"_json,
            R"({
                "title": "bar foo baz",
                "titles": ["bar foo baz"]
            })"_json,
    };

    auto collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_TRUE(collection_create_op.ok());
    for (auto const &json: documents) {
        auto add_op = collection_create_op.get()->add(json.dump());
        ASSERT_TRUE(add_op.ok());
    }

    std::map<std::string, std::string> req_params = {
            {"collection", "title"},
            {"q", "foo"},
            {"query_by", "title"},
            {"filter_by", "title:= foo bar baz"}
    };
    nlohmann::json embedded_params;
    std::string json_res;
    auto now_ts = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

    auto search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    auto res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(3, res_obj["found"].get<size_t>());
    ASSERT_EQ(3, res_obj["hits"].size());
    ASSERT_EQ("2", res_obj["hits"][0]["document"].at("id"));
    ASSERT_EQ("1", res_obj["hits"][1]["document"].at("id"));
    ASSERT_EQ("0", res_obj["hits"][2]["document"].at("id"));

    req_params = {
            {"collection", "title"},
            {"q", "foo"},
            {"query_by", "titles"},
            {"filter_by", "titles:= foo bar baz"}
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(2, res_obj["found"].get<size_t>());
    ASSERT_EQ(2, res_obj["hits"].size());
    ASSERT_EQ("2", res_obj["hits"][0]["document"].at("id"));
    ASSERT_EQ("1", res_obj["hits"][1]["document"].at("id"));
}

TEST_F(CollectionFilteringTest, FacetFieldStringFiltering) {
    Collection *coll_str;

    std::ifstream infile(std::string(ROOT_DIR)+"test/multi_field_documents.jsonl");
    std::vector<field> fields = {
            field("title", field_types::STRING, false),
            field("starring", field_types::STRING, true),
            field("cast", field_types::STRING_ARRAY, false),
            field("points", field_types::INT32, false)
    };

    std::vector<sort_by> sort_fields = { sort_by("points", "DESC") };

    coll_str = collectionManager.get_collection("coll_str").get();
    if(coll_str == nullptr) {
        coll_str = collectionManager.create_collection("coll_str", 1, fields, "points").get();
    }

    std::string json_line;

    while (std::getline(infile, json_line)) {
        nlohmann::json document = nlohmann::json::parse(json_line);
        coll_str->add(document.dump());
    }

    infile.close();

    query_fields = {"title"};
    std::vector<std::string> facets;

    // exact filter on string field must fail when single token is used
    facets.clear();
    facets.emplace_back("starring");
    auto results = coll_str->search("*", query_fields, "starring:= samuel", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(0, results["hits"].size());
    ASSERT_EQ(0, results["found"].get<size_t>());

    // multiple tokens but with a typo on one of them
    results = coll_str->search("*", query_fields, "starring:= ssamuel l. Jackson", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(0, results["hits"].size());
    ASSERT_EQ(0, results["found"].get<size_t>());

    // same should succeed when verbatim filter is made
    results = coll_str->search("*", query_fields, "starring:= samuel l. Jackson", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(2, results["hits"].size());
    ASSERT_EQ(2, results["found"].get<size_t>());

    // with backticks
    results = coll_str->search("*", query_fields, "starring:= `samuel l. Jackson`", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(2, results["hits"].size());
    ASSERT_EQ(2, results["found"].get<size_t>());

    // contains filter with a single token should work as well
    results = coll_str->search("*", query_fields, "starring: jackson", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(2, results["hits"].size());
    ASSERT_EQ(2, results["found"].get<size_t>());

    results = coll_str->search("*", query_fields, "starring: samuel", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(2, results["hits"].size());
    ASSERT_EQ(2, results["found"].get<size_t>());

    // contains when only 1 token so should not match
    results = coll_str->search("*", query_fields, "starring: samuel johnson", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(0, results["hits"].size());

    collectionManager.drop_collection("coll_str");
}

TEST_F(CollectionFilteringTest, FacetFieldStringArrayFiltering) {
    Collection *coll_array_fields;

    std::ifstream infile(std::string(ROOT_DIR)+"test/numeric_array_documents.jsonl");
    std::vector<field> fields = {field("name", field_types::STRING, false),
                                 field("name_facet", field_types::STRING, true),
                                 field("age", field_types::INT32, true),
                                 field("years", field_types::INT32_ARRAY, true),
                                 field("rating", field_types::FLOAT, true),
                                 field("timestamps", field_types::INT64_ARRAY, true),
                                 field("tags", field_types::STRING_ARRAY, true)};

    std::vector<sort_by> sort_fields = { sort_by("age", "DESC") };

    coll_array_fields = collectionManager.get_collection("coll_array_fields").get();
    if(coll_array_fields == nullptr) {
        coll_array_fields = collectionManager.create_collection("coll_array_fields", 1, fields, "age").get();
    }

    std::string json_line;

    while (std::getline(infile, json_line)) {
        nlohmann::json document = nlohmann::json::parse(json_line);
        document["name_facet"] = document["name"];
        const std::string & patched_json_line = document.dump();
        coll_array_fields->add(patched_json_line);
    }

    infile.close();

    query_fields = {"name"};
    std::vector<std::string> facets = {"tags"};

    // facet with filter on string array field must fail when exact token is used
    facets.clear();
    facets.push_back("tags");
    auto results = coll_array_fields->search("Jeremy", query_fields, "tags:= PLATINUM", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(0, results["hits"].size());
    ASSERT_EQ(0, results["found"].get<size_t>());

    results = coll_array_fields->search("Jeremy", query_fields, "tags:= FINE", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(0, results["hits"].size());

    results = coll_array_fields->search("Jeremy", query_fields, "tags:= FFINE PLATINUM", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(0, results["hits"].size());

    // partial token filter should be made without "=" operator
    results = coll_array_fields->search("Jeremy", query_fields, "tags: PLATINUM", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(1, results["hits"].size());
    ASSERT_EQ(1, results["found"].get<size_t>());

    results = coll_array_fields->search("Jeremy", query_fields, "tags: FINE", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(1, results["hits"].size());
    ASSERT_EQ(1, results["found"].get<size_t>());

    // to make tokens match facet value exactly, use "=" operator
    results = coll_array_fields->search("Jeremy", query_fields, "tags:= FINE PLATINUM", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(1, results["hits"].size());
    ASSERT_EQ(1, results["found"].get<size_t>());

    // allow exact filter on non-faceted field
    results = coll_array_fields->search("Jeremy", query_fields, "name:= Jeremy Howard", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(5, results["hits"].size());
    ASSERT_EQ(5, results["found"].get<size_t>());

    // multi match exact query (OR condition)
    results = coll_array_fields->search("Jeremy", query_fields, "tags:= [Gold, bronze]", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(3, results["hits"].size());
    ASSERT_EQ(3, results["found"].get<size_t>());

    results = coll_array_fields->search("Jeremy", query_fields, "tags:= [Gold, bronze, fine PLATINUM]", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(4, results["hits"].size());
    ASSERT_EQ(4, results["found"].get<size_t>());

    // single array multi match
    results = coll_array_fields->search("Jeremy", query_fields, "tags:= [fine PLATINUM]", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(1, results["hits"].size());
    ASSERT_EQ(1, results["found"].get<size_t>());

    collectionManager.drop_collection("coll_array_fields");
}

TEST_F(CollectionFilteringTest, FilterOnTextFieldWithColon) {
    Collection *coll1;

    std::vector<field> fields = {field("url", field_types::STRING, true),
                                 field("points", field_types::INT32, false)};

    std::vector<sort_by> sort_fields = { sort_by("points", "DESC") };

    coll1 = collectionManager.get_collection("coll1").get();
    if(coll1 == nullptr) {
        coll1 = collectionManager.create_collection("coll1", 4, fields, "points").get();
    }

    nlohmann::json doc1;
    doc1["id"] = "1";
    doc1["url"] = "https://example.com/1";
    doc1["points"] = 1;

    coll1->add(doc1.dump());

    query_fields = {"url"};
    std::vector<std::string> facets;

    auto res = coll1->search("*", query_fields, "url:= https://example.com/1", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(1, res["hits"].size());
    ASSERT_STREQ("1", res["hits"][0]["document"]["id"].get<std::string>().c_str());

    res = coll1->search("*", query_fields, "url: https://example.com/1", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(1, res["hits"].size());
    ASSERT_STREQ("1", res["hits"][0]["document"]["id"].get<std::string>().c_str());

    collectionManager.drop_collection("coll1");
}

TEST_F(CollectionFilteringTest, HandleBadlyFormedFilterQuery) {
    // should not crash when filter query is malformed!
    Collection *coll_array_fields;

    std::ifstream infile(std::string(ROOT_DIR)+"test/numeric_array_documents.jsonl");
    std::vector<field> fields = {field("name", field_types::STRING, false), field("age", field_types::INT32, false),
                                 field("years", field_types::INT32_ARRAY, false),
                                 field("timestamps", field_types::INT64_ARRAY, false),
                                 field("tags", field_types::STRING_ARRAY, true)};

    std::vector<sort_by> sort_fields = { sort_by("age", "DESC") };

    coll_array_fields = collectionManager.get_collection("coll_array_fields").get();
    if(coll_array_fields == nullptr) {
        coll_array_fields = collectionManager.create_collection("coll_array_fields", 4, fields, "age").get();
    }

    std::string json_line;

    while (std::getline(infile, json_line)) {
        coll_array_fields->add(json_line);
    }

    infile.close();

    query_fields = {"name"};
    std::vector<std::string> facets;

    // when filter field does not exist in the schema
    nlohmann::json results = coll_array_fields->search("Jeremy", query_fields, "tagzz: gold", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(0, results["hits"].size());

    // compound filter expression containing an unknown field
    results = coll_array_fields->search("Jeremy", query_fields,
               "(age:>0 ||  timestamps:> 0) || tagzz: gold", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(0, results["hits"].size());

    // unbalanced paranthesis
    results = coll_array_fields->search("Jeremy", query_fields,
                                        "(age:>0 ||  timestamps:> 0) || ", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(0, results["hits"].size());

    // searching using a string for a numeric field
    results = coll_array_fields->search("Jeremy", query_fields, "age: abcdef", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(0, results["hits"].size());

    // searching using a string for a numeric array field
    results = coll_array_fields->search("Jeremy", query_fields, "timestamps: abcdef", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(0, results["hits"].size());

    // malformed k:v syntax
    results = coll_array_fields->search("Jeremy", query_fields, "timestamps abcdef", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(0, results["hits"].size());

    // just spaces - must be treated as empty filter
    results = coll_array_fields->search("Jeremy", query_fields, "  ", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(5, results["hits"].size());

    // wrapping number with quotes
    results = coll_array_fields->search("Jeremy", query_fields, "age: '21'", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(0, results["hits"].size());

    // empty value for a numerical filter field
    auto res_op = coll_array_fields->search("Jeremy", query_fields, "age:", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false});
    ASSERT_FALSE(res_op.ok());
    ASSERT_EQ("Error with filter field `age`: Numerical field has an invalid comparator.", res_op.error());

    // empty value for string filter field
    res_op = coll_array_fields->search("Jeremy", query_fields, "tags:", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false});
    ASSERT_FALSE(res_op.ok());
    ASSERT_EQ("Error with filter field `tags`: Filter value cannot be empty.", res_op.error());

    res_op = coll_array_fields->search("Jeremy", query_fields, "tags:= ", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false});
    ASSERT_FALSE(res_op.ok());
    ASSERT_EQ("Error with filter field `tags`: Filter value cannot be empty.", res_op.error());

    collectionManager.drop_collection("coll_array_fields");
}

TEST_F(CollectionFilteringTest, FilterAndQueryFieldRestrictions) {
    Collection *coll_mul_fields;

    std::ifstream infile(std::string(ROOT_DIR)+"test/multi_field_documents.jsonl");
    std::vector<field> fields = {
        field("title", field_types::STRING, false),
        field("starring", field_types::STRING, false),
        field("cast", field_types::STRING_ARRAY, true),
        field("points", field_types::INT32, false)
    };

    coll_mul_fields = collectionManager.get_collection("coll_mul_fields").get();
    if(coll_mul_fields == nullptr) {
        coll_mul_fields = collectionManager.create_collection("coll_mul_fields", 4, fields, "points").get();
    }

    std::string json_line;

    while (std::getline(infile, json_line)) {
        coll_mul_fields->add(json_line);
    }

    infile.close();

    std::vector<std::string> facets;

    // query shall be allowed on faceted text fields as well
    query_fields = {"cast"};
    Option<nlohmann::json> result_op =
            coll_mul_fields->search("anton", query_fields, "", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false});
    ASSERT_TRUE(result_op.ok());

    nlohmann::json results = result_op.get();
    ASSERT_EQ(1, results["hits"].size());
    std::string solo_id = results["hits"].at(0)["document"]["id"];
    ASSERT_STREQ("14", solo_id.c_str());

    // filtering on string field should be possible
    query_fields = {"title"};
    result_op = coll_mul_fields->search("captain", query_fields, "starring: Samuel L. Jackson", facets, sort_fields, {0}, 10, 1,
                                        FREQUENCY, {false});
    ASSERT_EQ(true, result_op.ok());
    results = result_op.get();
    ASSERT_EQ(1, results["hits"].size());
    solo_id = results["hits"].at(0)["document"]["id"];
    ASSERT_STREQ("6", solo_id.c_str());

    // filtering on facet field should be possible (supports partial word search but without typo tolerance)
    query_fields = {"title"};
    result_op = coll_mul_fields->search("*", query_fields, "cast: chris", facets, sort_fields, {0}, 10, 1,
                                        FREQUENCY, {false});
    ASSERT_EQ(true, result_op.ok());
    results = result_op.get();
    ASSERT_EQ(3, results["hits"].size());

    // bad query string
    result_op = coll_mul_fields->search("captain", query_fields, "BLAH", facets, sort_fields, {0}, 10, 1,
                                        FREQUENCY, {false});
    ASSERT_EQ(false, result_op.ok());
    ASSERT_STREQ("Could not parse the filter query.", result_op.error().c_str());

    // missing field
    result_op = coll_mul_fields->search("captain", query_fields, "age: 100", facets, sort_fields, {0}, 10, 1,
                                        FREQUENCY, {false});
    ASSERT_EQ(false, result_op.ok());
    ASSERT_STREQ("Could not find a filter field named `age` in the schema.", result_op.error().c_str());

    // bad filter value type
    result_op = coll_mul_fields->search("captain", query_fields, "points: \"100\"", facets, sort_fields, {0}, 10, 1,
                                        FREQUENCY, {false});
    ASSERT_EQ(false, result_op.ok());
    ASSERT_STREQ("Error with filter field `points`: Numerical field has an invalid comparator.", result_op.error().c_str());

    result_op = coll_mul_fields->search("captain", query_fields, "points:<= foo", facets, sort_fields, {0}, 10, 1,
                                        FREQUENCY, {false});
    ASSERT_FALSE(result_op.ok());
    ASSERT_EQ("Error with filter field `points`: Not an int32.", result_op.error());

    // bad filter value type - equaling float on an integer field
    result_op = coll_mul_fields->search("captain", query_fields, "points: 100.34", facets, sort_fields, {0}, 10, 1,
                                        FREQUENCY, {false});
    ASSERT_EQ(false, result_op.ok());
    ASSERT_STREQ("Error with filter field `points`: Not an int32.", result_op.error().c_str());

    // bad filter value type - less than float on an integer field
    result_op = coll_mul_fields->search("captain", query_fields, "points: <100.0", facets, sort_fields, {0}, 10, 1,
                                        FREQUENCY, {false});
    ASSERT_EQ(false, result_op.ok());
    ASSERT_STREQ("Error with filter field `points`: Not an int32.", result_op.error().c_str());

    // when an int32 field is queried with a 64-bit number
    result_op = coll_mul_fields->search("captain", query_fields, "points: <2230070399", facets, sort_fields, {0}, 10, 1,
                                        FREQUENCY, {false});
    ASSERT_EQ(false, result_op.ok());
    ASSERT_EQ("Error with filter field `points`: `2230070399` exceeds the range of an int32.", result_op.error());

    result_op = coll_mul_fields->search("captain", query_fields, "points:<= 9223372036854775808", facets, sort_fields, {0}, 10, 1,
                                        FREQUENCY, {false});
    ASSERT_FALSE(result_op.ok());
    ASSERT_EQ("Error with filter field `points`: `9223372036854775808` exceeds the range of an int32.", result_op.error());

    // using a string filter value against an integer field
    result_op = coll_mul_fields->search("captain", query_fields, "points: <sdsdfsdf", facets, sort_fields, {0}, 10, 1,
                                        FREQUENCY, {false});
    ASSERT_EQ(false, result_op.ok());

    // large negative number
    result_op = coll_mul_fields->search("captain", query_fields, "points: >-3230070399", facets, sort_fields, {0}, 10, 1,
                                        FREQUENCY, {false});
    ASSERT_EQ(false, result_op.ok());

    // but should allow small negative number
    result_op = coll_mul_fields->search("captain", query_fields, "points: >-3230", facets, sort_fields, {0}, 10, 1,
                                        FREQUENCY, {false});
    ASSERT_EQ(true, result_op.ok());

    collectionManager.drop_collection("coll_mul_fields");
}

TEST_F(CollectionFilteringTest, FilterOnNumericFields) {
    Collection *coll_array_fields;

    std::ifstream infile(std::string(ROOT_DIR)+"test/numeric_array_documents.jsonl");
    std::vector<field> fields = {
            field("name", field_types::STRING, false),
            field("rating", field_types::FLOAT, false),
            field("age", field_types::INT32, false),
            field("years", field_types::INT32_ARRAY, false),
            field("timestamps", field_types::INT64_ARRAY, false),
            field("tags", field_types::STRING_ARRAY, true)
    };

    std::vector<sort_by> sort_fields = { sort_by("age", "DESC") };

    coll_array_fields = collectionManager.get_collection("coll_array_fields").get();
    if(coll_array_fields == nullptr) {
        // ensure that default_sorting_field is a non-array numerical field
        auto coll_op = collectionManager.create_collection("coll_array_fields", 4, fields, "years");
        ASSERT_EQ(false, coll_op.ok());
        ASSERT_STREQ("Default sorting field `years` is not a sortable type.", coll_op.error().c_str());

        // let's try again properly
        coll_op = collectionManager.create_collection("coll_array_fields", 4, fields, "age");
        coll_array_fields = coll_op.get();
    }

    std::string json_line;

    while (std::getline(infile, json_line)) {
        coll_array_fields->add(json_line);
    }

    infile.close();

    // Plain search with no filters - results should be sorted by rank fields
    query_fields = {"name"};
    std::vector<std::string> facets;
    nlohmann::json results = coll_array_fields->search("Jeremy", query_fields, "", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(5, results["hits"].size());

    std::vector<std::string> ids = {"3", "1", "4", "0", "2"};

    for(size_t i = 0; i < results["hits"].size(); i++) {
        nlohmann::json result = results["hits"].at(i);
        std::string result_id = result["document"]["id"];
        std::string id = ids.at(i);
        ASSERT_STREQ(id.c_str(), result_id.c_str());
    }

    // Searching on an int32 field
    results = coll_array_fields->search("Jeremy", query_fields, "age:>24", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(3, results["hits"].size());

    ids = {"3", "1", "4"};

    for(size_t i = 0; i < results["hits"].size(); i++) {
        nlohmann::json result = results["hits"].at(i);
        std::string result_id = result["document"]["id"];
        std::string id = ids.at(i);
        ASSERT_STREQ(id.c_str(), result_id.c_str());
    }

    results = coll_array_fields->search("Jeremy", query_fields, "age:>=24", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(4, results["hits"].size());

    results = coll_array_fields->search("Jeremy", query_fields, "age:24", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(1, results["hits"].size());

    // alternative `:=` syntax
    results = coll_array_fields->search("Jeremy", query_fields, "age:=24", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(1, results["hits"].size());

    results = coll_array_fields->search("Jeremy", query_fields, "age:= 24", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(1, results["hits"].size());

    // Searching a number against an int32 array field
    results = coll_array_fields->search("Jeremy", query_fields, "years:>2002", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(3, results["hits"].size());

    ids = {"1", "0", "2"};
    for(size_t i = 0; i < results["hits"].size(); i++) {
        nlohmann::json result = results["hits"].at(i);
        std::string result_id = result["document"]["id"];
        std::string id = ids.at(i);
        ASSERT_STREQ(id.c_str(), result_id.c_str());
    }

    results = coll_array_fields->search("Jeremy", query_fields, "years:<1989", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(1, results["hits"].size());

    ids = {"3"};
    for(size_t i = 0; i < results["hits"].size(); i++) {
        nlohmann::json result = results["hits"].at(i);
        std::string result_id = result["document"]["id"];
        std::string id = ids.at(i);
        ASSERT_STREQ(id.c_str(), result_id.c_str());
    }

    // not equals
    results = coll_array_fields->search("Jeremy", query_fields, "age:!= 24", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(4, results["hits"].size());

    ids = {"3", "1", "4", "2"};
    for(size_t i = 0; i < results["hits"].size(); i++) {
        nlohmann::json result = results["hits"].at(i);
        std::string result_id = result["document"]["id"];
        std::string id = ids.at(i);
        ASSERT_STREQ(id.c_str(), result_id.c_str());
    }

    results = coll_array_fields->search("Jeremy", query_fields, "age:!= 0", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(5, results["hits"].size());

    ids = {"3", "1", "4", "0", "2"};
    for(size_t i = 0; i < results["hits"].size(); i++) {
        nlohmann::json result = results["hits"].at(i);
        std::string result_id = result["document"]["id"];
        std::string id = ids.at(i);
        ASSERT_EQ(id, result_id);
    }

    // multiple filters
    results = coll_array_fields->search("Jeremy", query_fields, "years:<2005 && years:>1987", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(1, results["hits"].size());

    ids = {"4"};
    for(size_t i = 0; i < results["hits"].size(); i++) {
        nlohmann::json result = results["hits"].at(i);
        std::string result_id = result["document"]["id"];
        std::string id = ids.at(i);
        ASSERT_STREQ(id.c_str(), result_id.c_str());
    }

    // multiple search values (works like SQL's IN operator) against a single int field
    results = coll_array_fields->search("Jeremy", query_fields, "age:[21, 24, 63]", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(3, results["hits"].size());

    ids = {"3", "0", "2"};
    for(size_t i = 0; i < results["hits"].size(); i++) {
        nlohmann::json result = results["hits"].at(i);
        std::string result_id = result["document"]["id"];
        std::string id = ids.at(i);
        ASSERT_STREQ(id.c_str(), result_id.c_str());
    }

    // alternative `:=` syntax
    results = coll_array_fields->search("Jeremy", query_fields, "age:= [21, 24, 63]", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(3, results["hits"].size());

    // individual comparators can still be applied.
    results = coll_array_fields->search("Jeremy", query_fields, "age: [!=21, >30]", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(4, results["hits"].size());

    ids = {"3", "1", "4", "0"};
    for(size_t i = 0; i < results["hits"].size(); i++) {
        nlohmann::json result = results["hits"].at(i);
        std::string result_id = result["document"]["id"];
        std::string id = ids.at(i);
        ASSERT_EQ(id, result_id);
    }

    // negate multiple search values (works like SQL's NOT IN) against a single int field
    results = coll_array_fields->search("Jeremy", query_fields, "age:!= [21, 24, 63]", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(2, results["hits"].size());

    ids = {"1", "4"};
    for(size_t i = 0; i < results["hits"].size(); i++) {
        nlohmann::json result = results["hits"].at(i);
        std::string result_id = result["document"]["id"];
        std::string id = ids.at(i);
        ASSERT_EQ(id, result_id);
    }

    // individual comparators can still be applied.
    results = coll_array_fields->search("Jeremy", query_fields, "age: != [<30, >60]", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(2, results["hits"].size());

    ids = {"1", "4"};
    for(size_t i = 0; i < results["hits"].size(); i++) {
        nlohmann::json result = results["hits"].at(i);
        std::string result_id = result["document"]["id"];
        std::string id = ids.at(i);
        ASSERT_EQ(id, result_id);
    }

    // multiple search values against an int32 array field - also use extra padding between symbols
    results = coll_array_fields->search("Jeremy", query_fields, "years : [ 2015, 1985 , 1999]", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(4, results["hits"].size());

    ids = {"3", "1", "4", "0"};
    for(size_t i = 0; i < results["hits"].size(); i++) {
        nlohmann::json result = results["hits"].at(i);
        std::string result_id = result["document"]["id"];
        std::string id = ids.at(i);
        ASSERT_STREQ(id.c_str(), result_id.c_str());
    }

    // searching on an int64 array field - also ensure that padded space causes no issues
    results = coll_array_fields->search("Jeremy", query_fields, "timestamps : > 475205222", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(4, results["hits"].size());

    ids = {"1", "4", "0", "2"};

    for(size_t i = 0; i < results["hits"].size(); i++) {
        nlohmann::json result = results["hits"].at(i);
        std::string result_id = result["document"]["id"];
        std::string id = ids.at(i);
        ASSERT_STREQ(id.c_str(), result_id.c_str());
    }

    // range based filter
    results = coll_array_fields->search("Jeremy", query_fields, "age: 21..32", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(3, results["hits"].size());

    ids = {"4", "0", "2"};
    for(size_t i = 0; i < results["hits"].size(); i++) {
        nlohmann::json result = results["hits"].at(i);
        std::string result_id = result["document"]["id"];
        std::string id = ids.at(i);
        ASSERT_STREQ(id.c_str(), result_id.c_str());
    }

    results = coll_array_fields->search("Jeremy", query_fields, "age: 0 .. 100", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(5, results["hits"].size());

    results = coll_array_fields->search("Jeremy", query_fields, "age: [21..24, 40..65]", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(4, results["hits"].size());

    ids = {"3", "1", "0", "2"};
    for(size_t i = 0; i < results["hits"].size(); i++) {
        nlohmann::json result = results["hits"].at(i);
        std::string result_id = result["document"]["id"];
        std::string id = ids.at(i);
        ASSERT_STREQ(id.c_str(), result_id.c_str());
    }

    results = coll_array_fields->search("Jeremy", query_fields, "rating: 7.812 .. 9.999", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(2, results["hits"].size());

    ids = {"1", "2"};
    for(size_t i = 0; i < results["hits"].size(); i++) {
        nlohmann::json result = results["hits"].at(i);
        std::string result_id = result["document"]["id"];
        std::string id = ids.at(i);
        ASSERT_STREQ(id.c_str(), result_id.c_str());
    }

    results = coll_array_fields->search("Jeremy", query_fields, "rating: [7.812 .. 9.999, 1.05 .. 1.09]", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(3, results["hits"].size());

    // when filters don't match any record, no results should be returned
    results = coll_array_fields->search("Jeremy", query_fields, "timestamps:>1591091288061", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(0, results["hits"].size());

    collectionManager.drop_collection("coll_array_fields");
}

TEST_F(CollectionFilteringTest, FilterOnFloatFields) {
    Collection *coll_array_fields;

    std::ifstream infile(std::string(ROOT_DIR)+"test/numeric_array_documents.jsonl");
    std::vector<field> fields = {
            field("name", field_types::STRING, false),
            field("age", field_types::INT32, false),
            field("top_3", field_types::FLOAT_ARRAY, false),
            field("rating", field_types::FLOAT, false)
    };
    std::vector<sort_by> sort_fields_desc = { sort_by("rating", "DESC") };
    std::vector<sort_by> sort_fields_asc = { sort_by("rating", "ASC") };

    coll_array_fields = collectionManager.get_collection("coll_array_fields").get();
    if(coll_array_fields == nullptr) {
        coll_array_fields = collectionManager.create_collection("coll_array_fields", 4, fields, "age").get();
    }

    std::string json_line;

    while (std::getline(infile, json_line)) {
        auto add_op = coll_array_fields->add(json_line);
        ASSERT_TRUE(add_op.ok());
    }

    infile.close();

    // Plain search with no filters - results should be sorted by rating field DESC
    query_fields = {"name"};
    std::vector<std::string> facets;
    nlohmann::json results = coll_array_fields->search("Jeremy", query_fields, "", facets, sort_fields_desc, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(5, results["hits"].size());

    std::vector<std::string> ids = {"1", "2", "4", "0", "3"};

    for(size_t i = 0; i < results["hits"].size(); i++) {
        nlohmann::json result = results["hits"].at(i);
        std::string result_id = result["document"]["id"];
        std::string id = ids.at(i);
        ASSERT_STREQ(id.c_str(), result_id.c_str());
    }

    // Plain search with no filters - results should be sorted by rating field ASC
    results = coll_array_fields->search("Jeremy", query_fields, "", facets, sort_fields_asc, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(5, results["hits"].size());

    ids = {"3", "0", "4", "2", "1"};

    for(size_t i = 0; i < results["hits"].size(); i++) {
        nlohmann::json result = results["hits"].at(i);
        std::string result_id = result["document"]["id"];
        std::string id = ids.at(i);
        ASSERT_STREQ(id.c_str(), result_id.c_str()); //?
    }

    results = coll_array_fields->search("Jeremy", query_fields, "rating:!=0", facets, sort_fields_asc, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(4, results["hits"].size());

    ids = {"0", "4", "2", "1"};
    for(size_t i = 0; i < results["hits"].size(); i++) {
        nlohmann::json result = results["hits"].at(i);
        std::string result_id = result["document"]["id"];
        std::string id = ids.at(i);
        ASSERT_STREQ(id.c_str(), result_id.c_str()); //?
    }

    // Searching on a float field, sorted desc by rating
    results = coll_array_fields->search("Jeremy", query_fields, "rating:>0.0", facets, sort_fields_desc, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(4, results["hits"].size());

    ids = {"1", "2", "4", "0"};

    for(size_t i = 0; i < results["hits"].size(); i++) {
        nlohmann::json result = results["hits"].at(i);
        std::string result_id = result["document"]["id"];
        std::string id = ids.at(i);
        ASSERT_STREQ(id.c_str(), result_id.c_str());
    }

    // Searching a float against an float array field
    results = coll_array_fields->search("Jeremy", query_fields, "top_3:>7.8", facets, sort_fields_desc, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(2, results["hits"].size());

    ids = {"1", "2"};
    for(size_t i = 0; i < results["hits"].size(); i++) {
        nlohmann::json result = results["hits"].at(i);
        std::string result_id = result["document"]["id"];
        std::string id = ids.at(i);
        ASSERT_STREQ(id.c_str(), result_id.c_str());
    }

    // multiple filters
    results = coll_array_fields->search("Jeremy", query_fields, "top_3:>7.8 && rating:>7.9", facets, sort_fields_desc, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(1, results["hits"].size());

    ids = {"1"};
    for(size_t i = 0; i < results["hits"].size(); i++) {
        nlohmann::json result = results["hits"].at(i);
        std::string result_id = result["document"]["id"];
        std::string id = ids.at(i);
        ASSERT_STREQ(id.c_str(), result_id.c_str());
    }

    // multiple search values (works like SQL's IN operator) against a single float field
    results = coll_array_fields->search("Jeremy", query_fields, "rating:[1.09, 7.812]", facets, sort_fields_desc, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(2, results["hits"].size());

    ids = {"2", "0"};
    for(size_t i = 0; i < results["hits"].size(); i++) {
        nlohmann::json result = results["hits"].at(i);
        std::string result_id = result["document"]["id"];
        std::string id = ids.at(i);
        ASSERT_STREQ(id.c_str(), result_id.c_str());
    }

    // negate multiple search values (works like SQL's NOT IN operator) against a single float field
    results = coll_array_fields->search("Jeremy", query_fields, "rating:!= [1.09, 7.812]", facets, sort_fields_desc, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(3, results["hits"].size());

    ids = {"1", "4", "3"};
    for(size_t i = 0; i < results["hits"].size(); i++) {
        nlohmann::json result = results["hits"].at(i);
        std::string result_id = result["document"]["id"];
        std::string id = ids.at(i);
        ASSERT_EQ(id, result_id);
    }

    // individual comparators can still be applied.
    results = coll_array_fields->search("Jeremy", query_fields, "rating: != [<5.4, >9]", facets, sort_fields_desc, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(2, results["hits"].size());

    ids = {"2", "4"};
    for(size_t i = 0; i < results["hits"].size(); i++) {
        nlohmann::json result = results["hits"].at(i);
        std::string result_id = result["document"]["id"];
        std::string id = ids.at(i);
        ASSERT_EQ(id, result_id);
    }

    results = coll_array_fields->search("Jeremy", query_fields, "rating: [!= 1]", facets, sort_fields_desc, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(5, results["hits"].size());

    ids = {"1", "2", "4", "0", "3"};
    for(size_t i = 0; i < results["hits"].size(); i++) {
        nlohmann::json result = results["hits"].at(i);
        std::string result_id = result["document"]["id"];
        std::string id = ids.at(i);
        ASSERT_EQ(id, result_id);
    }

    // multiple search values against a float array field - also use extra padding between symbols
    results = coll_array_fields->search("Jeremy", query_fields, "top_3 : [ 5.431, 0.001 , 7.812, 11.992]", facets, sort_fields_desc, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(3, results["hits"].size());

    ids = {"2", "4", "0"};
    for(size_t i = 0; i < results["hits"].size(); i++) {
        nlohmann::json result = results["hits"].at(i);
        std::string result_id = result["document"]["id"];
        std::string id = ids.at(i);
        ASSERT_STREQ(id.c_str(), result_id.c_str());
    }

    // when filters don't match any record, no results should be returned
    auto results_op = coll_array_fields->search("Jeremy", query_fields, "rating:<-2.78", facets, sort_fields_desc, {0}, 10, 1, FREQUENCY, {false});
    ASSERT_TRUE(results_op.ok());
    results = results_op.get();
    ASSERT_EQ(0, results["hits"].size());

    // rank tokens by default sorting field
    results_op = coll_array_fields->search("j", query_fields, "", facets, sort_fields_desc, {0}, 10, 1, MAX_SCORE, {true});
    ASSERT_TRUE(results_op.ok());
    results = results_op.get();
    ASSERT_EQ(5, results["hits"].size());

    ids = {"1", "2", "4", "0", "3"};

    for(size_t i = 0; i < results["hits"].size(); i++) {
        nlohmann::json result = results["hits"].at(i);
        std::string result_id = result["document"]["id"];
        std::string id = ids.at(i);
        ASSERT_STREQ(id.c_str(), result_id.c_str());
    }

    collectionManager.drop_collection("coll_array_fields");
}

TEST_F(CollectionFilteringTest, FilterOnNegativeNumericalFields) {
    Collection *coll1;

    std::vector<field> fields = {field("title", field_types::STRING, false),
                                 field("int32_field", field_types::INT32, false),
                                 field("int64_field", field_types::INT64, false),
                                 field("float_field", field_types::FLOAT, false)};

    coll1 = collectionManager.get_collection("coll1").get();
    if(coll1 == nullptr) {
        coll1 = collectionManager.create_collection("coll1", 1, fields, "int32_field").get();
    }

    std::vector<std::vector<std::string>> records = {
        {"Title 1", "-100", "5000000", "-10.45124"},
        {"Title 2", "100", "-1000000", "0.45124"},
        {"Title 3", "-200", "3000000", "-0.45124"},
        {"Title 4", "150", "10000000", "1.45124"},
    };

    for(size_t i=0; i<records.size(); i++) {
        nlohmann::json doc;

        doc["id"] = std::to_string(i);
        doc["title"] = records[i][0];
        doc["int32_field"] = std::stoi(records[i][1]);
        doc["int64_field"] = std::stoll(records[i][2]);
        doc["float_field"] = std::stof(records[i][3]);

        ASSERT_TRUE(coll1->add(doc.dump()).ok());
    }

    auto results = coll1->search("*", {}, "int32_field:<0", {}, {}, {0}, 10, 1, FREQUENCY, {true}, 10).get();

    ASSERT_EQ(2, results["found"].get<size_t>());
    ASSERT_EQ(2, results["hits"].size());
    ASSERT_EQ("0", results["hits"][0]["document"]["id"].get<std::string>());
    ASSERT_EQ("2", results["hits"][1]["document"]["id"].get<std::string>());

    results = coll1->search("*", {}, "int64_field:<0", {}, {}, {0}, 10, 1, FREQUENCY, {true}, 10).get();

    ASSERT_EQ(1, results["found"].get<size_t>());
    ASSERT_EQ(1, results["hits"].size());
    ASSERT_EQ("1", results["hits"][0]["document"]["id"].get<std::string>());

    results = coll1->search("*", {}, "float_field:<0", {}, {sort_by("float_field", "desc")}, {0}, 10, 1, FREQUENCY,
                            {true}, 10).get();

    ASSERT_EQ(2, results["found"].get<size_t>());
    ASSERT_EQ(2, results["hits"].size());
    ASSERT_EQ("2", results["hits"][0]["document"]["id"].get<std::string>());
    ASSERT_EQ("0", results["hits"][1]["document"]["id"].get<std::string>());

    collectionManager.drop_collection("coll1");
}

TEST_F(CollectionFilteringTest, ComparatorsOnMultiValuedNumericalField) {
    Collection *coll_array_fields;

    std::ifstream infile(std::string(ROOT_DIR)+"test/numeric_array_documents.jsonl");
    std::vector<field> fields = {
            field("name", field_types::STRING, false),
            field("age", field_types::INT32, false),
            field("top_3", field_types::FLOAT_ARRAY, false),
            field("rating", field_types::FLOAT, false)
    };

    std::vector<sort_by> sort_fields_desc = {sort_by("rating", "DESC")};

    coll_array_fields = collectionManager.get_collection("coll_array_fields").get();
    if (coll_array_fields == nullptr) {
        coll_array_fields = collectionManager.create_collection("coll_array_fields", 4, fields, "age").get();
    }

    std::string json_line;

    while (std::getline(infile, json_line)) {
        auto add_op = coll_array_fields->add(json_line);
        ASSERT_TRUE(add_op.ok());
    }

    infile.close();

    query_fields = {"name"};
    std::vector<std::string> facets;
    nlohmann::json results = coll_array_fields->search("Jeremy", query_fields, "age: [24, >32]",
            facets, sort_fields_desc, {0}, 10, 1,FREQUENCY, {false}).get();

    ASSERT_EQ(3, results["hits"].size());

    std::vector<std::string> ids = {"1", "0", "3"};

    for (size_t i = 0; i < results["hits"].size(); i++) {
        nlohmann::json result = results["hits"].at(i);
        std::string result_id = result["document"]["id"];
        std::string id = ids.at(i);
        ASSERT_STREQ(id.c_str(), result_id.c_str());
    }

    // with <= and >=

    results = coll_array_fields->search("Jeremy", query_fields, "age: [<=24, >=44]",
                                        facets, sort_fields_desc, {0}, 10, 1,FREQUENCY, {false}).get();

    ASSERT_EQ(4, results["hits"].size());

    ids = {"1", "2", "0", "3"};

    for (size_t i = 0; i < results["hits"].size(); i++) {
        nlohmann::json result = results["hits"].at(i);
        std::string result_id = result["document"]["id"];
        std::string id = ids.at(i);
        ASSERT_STREQ(id.c_str(), result_id.c_str());
    }

    collectionManager.drop_collection("coll_array_fields");
}

TEST_F(CollectionFilteringTest, FilteringWithPrefixSearch) {
    Collection *coll1;

    std::vector<field> fields = {field("title", field_types::STRING, false),
                                 field("points", field_types::INT32, false),};

    coll1 = collectionManager.get_collection("coll1").get();
    if(coll1 == nullptr) {
        coll1 = collectionManager.create_collection("coll1", 1, fields, "points").get();
    }

    std::vector<std::vector<std::string>> records = {
            {"elephant"}, {"emerald"}, {"effective"}, {"esther"}, {"eagle"},
            {"empty"}, {"elite"}, {"example"}, {"elated"}, {"end"},
            {"ear"}, {"eager"}, {"earmark"}, {"envelop"}, {"excess"},
            {"ember"}, {"earth"}, {"envoy"}, {"emerge"}, {"emigrant"},
            {"envision"}, {"envy"}, {"envisage"}, {"executive"}, {"end"},
    };

    for(size_t i=0; i<records.size(); i++) {
        nlohmann::json doc;

        doc["id"] = std::to_string(i);
        doc["title"] = records[i][0];
        doc["points"] = i;

        ASSERT_TRUE(coll1->add(doc.dump()).ok());
    }

    // pick a location close to only the Sacre Coeur
    auto res_op = coll1->search("e",
                                {"title"}, "points: 23",
                                {}, {}, {0}, 10, 1, FREQUENCY, {true});

    auto results = res_op.get();

    ASSERT_EQ(1, results["found"].get<size_t>());
    ASSERT_EQ(1, results["hits"].size());

    ASSERT_STREQ("23", results["hits"][0]["document"]["id"].get<std::string>().c_str());

    collectionManager.drop_collection("coll1");
}

TEST_F(CollectionFilteringTest, NumericalFilteringWithAnd) {
    Collection *coll1;

    std::vector<field> fields = {field("company_name", field_types::STRING, false),
                                 field("num_employees", field_types::INT32, false),};

    coll1 = collectionManager.get_collection("coll1").get();
    if(coll1 == nullptr) {
        coll1 = collectionManager.create_collection("coll1", 1, fields, "num_employees").get();
    }

    std::vector<std::vector<std::string>> records = {
            {"123", "Company 1", "50"},
            {"125", "Company 2", "150"},
            {"127", "Company 3", "250"},
            {"129", "Stark Industries 4", "500"},
    };

    for(size_t i=0; i<records.size(); i++) {
        nlohmann::json doc;

        doc["id"] = records[i][0];
        doc["company_name"] = records[i][1];
        doc["num_employees"] = std::stoi(records[i][2]);

        ASSERT_TRUE(coll1->add(doc.dump()).ok());
    }

    std::vector<sort_by> sort_fields = { sort_by("num_employees", "ASC") };

    auto results = coll1->search("*",
                                {}, "num_employees:>=100 && num_employees:<=300",
                                {}, sort_fields, {0}, 10, 1, FREQUENCY, {true}).get();

    ASSERT_EQ(2, results["found"].get<size_t>());
    ASSERT_EQ(2, results["hits"].size());

    ASSERT_STREQ("125", results["hits"][0]["document"]["id"].get<std::string>().c_str());
    ASSERT_STREQ("127", results["hits"][1]["document"]["id"].get<std::string>().c_str());

    // when filter number is well below all values
    results = coll1->search("*",
                                 {}, "num_employees:>=100 && num_employees:<=10",
                                 {}, sort_fields, {0}, 10, 1, FREQUENCY, {true}).get();

    ASSERT_EQ(0, results["found"].get<size_t>());

    // check boundaries
    results = coll1->search("*",
                            {}, "num_employees:>=150 && num_employees:<=250",
                            {}, sort_fields, {0}, 10, 1, FREQUENCY, {true}).get();

    ASSERT_EQ(2, results["found"].get<size_t>());
    ASSERT_STREQ("125", results["hits"][0]["document"]["id"].get<std::string>().c_str());
    ASSERT_STREQ("127", results["hits"][1]["document"]["id"].get<std::string>().c_str());

    results = coll1->search("*",
                            {}, "num_employees:>150 && num_employees:<250",
                            {}, sort_fields, {0}, 10, 1, FREQUENCY, {true}).get();

    ASSERT_EQ(0, results["found"].get<size_t>());


    results = coll1->search("*",
                            {}, "num_employees:>50 && num_employees:<250",
                            {}, sort_fields, {0}, 10, 1, FREQUENCY, {true}).get();

    ASSERT_EQ(1, results["found"].get<size_t>());
    ASSERT_STREQ("125", results["hits"][0]["document"]["id"].get<std::string>().c_str());

    // extreme boundaries

    results = coll1->search("*",
                            {}, "num_employees:>50 && num_employees:<=500",
                            {}, sort_fields, {0}, 10, 1, FREQUENCY, {true}).get();

    ASSERT_EQ(3, results["found"].get<size_t>());
    ASSERT_STREQ("125", results["hits"][0]["document"]["id"].get<std::string>().c_str());
    ASSERT_STREQ("127", results["hits"][1]["document"]["id"].get<std::string>().c_str());
    ASSERT_STREQ("129", results["hits"][2]["document"]["id"].get<std::string>().c_str());

    results = coll1->search("*",
                            {}, "num_employees:>=50 && num_employees:<500",
                            {}, sort_fields, {0}, 10, 1, FREQUENCY, {true}).get();

    ASSERT_EQ(3, results["found"].get<size_t>());
    ASSERT_STREQ("123", results["hits"][0]["document"]["id"].get<std::string>().c_str());
    ASSERT_STREQ("125", results["hits"][1]["document"]["id"].get<std::string>().c_str());
    ASSERT_STREQ("127", results["hits"][2]["document"]["id"].get<std::string>().c_str());

    // no match
    results = coll1->search("*",
                            {}, "num_employees:>3000 && num_employees:<10",
                            {}, sort_fields, {0}, 10, 1, FREQUENCY, {true}).get();

    ASSERT_EQ(0, results["found"].get<size_t>());

    collectionManager.drop_collection("coll1");
}

TEST_F(CollectionFilteringTest, FilteringViaDocumentIds) {
    Collection *coll1;

    std::vector<field> fields = {field("company_name", field_types::STRING, false),
                                 field("num_employees", field_types::INT32, false),};

    coll1 = collectionManager.get_collection("coll1").get();
    if(coll1 == nullptr) {
        coll1 = collectionManager.create_collection("coll1", 1, fields, "num_employees").get();
    }

    std::vector<std::vector<std::string>> records = {
        {"123", "Company 1", "50"},
        {"125", "Company 2", "150"},
        {"127", "Company 3", "250"},
        {"129", "Stark Industries 4", "500"},
    };

    for(size_t i=0; i<records.size(); i++) {
        nlohmann::json doc;

        doc["id"] = records[i][0];
        doc["company_name"] = records[i][1];
        doc["num_employees"] = std::stoi(records[i][2]);

        ASSERT_TRUE(coll1->add(doc.dump()).ok());
    }

    std::vector<sort_by> sort_fields = { sort_by("num_employees", "ASC") };

    auto results = coll1->search("*",
                                 {}, "id: 123",
                                 {}, sort_fields, {0}, 10, 1, FREQUENCY, {true}).get();

    ASSERT_EQ(1, results["found"].get<size_t>());
    ASSERT_EQ(1, results["hits"].size());
    ASSERT_STREQ("123", results["hits"][0]["document"]["id"].get<std::string>().c_str());

    results = coll1->search("*",
                            {}, "id: != 123",
                            {}, sort_fields, {0}, 10, 1, FREQUENCY, {true}).get();

    ASSERT_EQ(3, results["found"].get<size_t>());
    ASSERT_EQ(3, results["hits"].size());
    ASSERT_STREQ("125", results["hits"][0]["document"]["id"].get<std::string>().c_str());
    ASSERT_STREQ("127", results["hits"][1]["document"]["id"].get<std::string>().c_str());
    ASSERT_STREQ("129", results["hits"][2]["document"]["id"].get<std::string>().c_str());

    // single ID with backtick

    results = coll1->search("*",
                            {}, "id: `123`",
                            {}, sort_fields, {0}, 10, 1, FREQUENCY, {true}).get();

    ASSERT_EQ(1, results["found"].get<size_t>());
    ASSERT_EQ(1, results["hits"].size());
    ASSERT_STREQ("123", results["hits"][0]["document"]["id"].get<std::string>().c_str());

    // single ID with condition
    results = coll1->search("*",
                            {}, "id: 125 && num_employees: 150",
                            {}, sort_fields, {0}, 10, 1, FREQUENCY, {true}).get();

    ASSERT_EQ(1, results["found"].get<size_t>());
    ASSERT_EQ(1, results["hits"].size());
    ASSERT_STREQ("125", results["hits"][0]["document"]["id"].get<std::string>().c_str());

    // multiple IDs
    results = coll1->search("*",
                            {}, "id: [123, 125, 127, 129] && num_employees: <300",
                            {}, sort_fields, {0}, 10, 1, FREQUENCY, {true}).get();

    ASSERT_EQ(3, results["found"].get<size_t>());
    ASSERT_EQ(3, results["hits"].size());
    ASSERT_STREQ("123", results["hits"][0]["document"]["id"].get<std::string>().c_str());
    ASSERT_STREQ("125", results["hits"][1]["document"]["id"].get<std::string>().c_str());
    ASSERT_STREQ("127", results["hits"][2]["document"]["id"].get<std::string>().c_str());

    // multiple IDs with exact equals operator with IDs not being ordered
    results = coll1->search("*",
                            {}, "id:= [129, 123, 127, 125] && num_employees: <300",
                            {}, sort_fields, {0}, 10, 1, FREQUENCY, {true}).get();

    ASSERT_EQ(3, results["found"].get<size_t>());
    ASSERT_EQ(3, results["hits"].size());
    ASSERT_STREQ("123", results["hits"][0]["document"]["id"].get<std::string>().c_str());
    ASSERT_STREQ("125", results["hits"][1]["document"]["id"].get<std::string>().c_str());
    ASSERT_STREQ("127", results["hits"][2]["document"]["id"].get<std::string>().c_str());

    // multiple IDs with exact equals operator and backticks
    results = coll1->search("*",
                            {}, "id:= [`123`, `125`, `127`, `129`] && num_employees: <300",
                            {}, sort_fields, {0}, 10, 1, FREQUENCY, {true}).get();

    ASSERT_EQ(3, results["found"].get<size_t>());
    ASSERT_EQ(3, results["hits"].size());
    ASSERT_STREQ("123", results["hits"][0]["document"]["id"].get<std::string>().c_str());
    ASSERT_STREQ("125", results["hits"][1]["document"]["id"].get<std::string>().c_str());
    ASSERT_STREQ("127", results["hits"][2]["document"]["id"].get<std::string>().c_str());

    results = coll1->search("*",
                           {}, "id:!= [123,125] && num_employees: <300",
                           {}, sort_fields, {0}, 10, 1, FREQUENCY, {true}).get();

    ASSERT_EQ(1, results["found"].get<size_t>());
    ASSERT_EQ(1, results["hits"].size());
    ASSERT_STREQ("127", results["hits"][0]["document"]["id"].get<std::string>().c_str());

    // empty id list not allowed
    auto res_op = coll1->search("*", {}, "id:=", {}, sort_fields, {0}, 10, 1, FREQUENCY, {true});
    ASSERT_FALSE(res_op.ok());
    ASSERT_EQ("Error with filter field `id`: Filter value cannot be empty.", res_op.error());

    res_op = coll1->search("*", {}, "id:= ", {}, sort_fields, {0}, 10, 1, FREQUENCY, {true});
    ASSERT_FALSE(res_op.ok());
    ASSERT_EQ("Error with filter field `id`: Filter value cannot be empty.", res_op.error());

    res_op = coll1->search("*", {}, "id: ", {}, sort_fields, {0}, 10, 1, FREQUENCY, {true});
    ASSERT_FALSE(res_op.ok());
    ASSERT_EQ("Error with filter field `id`: Filter value cannot be empty.", res_op.error());

    res_op = coll1->search("*", {}, "id: ``", {}, sort_fields, {0}, 10, 1, FREQUENCY, {true});
    ASSERT_FALSE(res_op.ok());
    ASSERT_EQ("Error with filter field `id`: Filter value cannot be empty.", res_op.error());

    // when no IDs exist
    results = coll1->search("*",
                            {}, "id: [1000] && num_employees: <300",
                            {}, sort_fields, {0}, 10, 1, FREQUENCY, {true}).get();

    ASSERT_EQ(0, results["found"].get<size_t>());

    results = coll1->search("*",
                            {}, "id: 1000",
                            {}, sort_fields, {0}, 10, 1, FREQUENCY, {true}).get();

    ASSERT_EQ(0, results["found"].get<size_t>());

    // match all IDs
    results = coll1->search("*",
                            {}, "id: *",
                            {}, sort_fields, {0}, 10, 1, FREQUENCY, {true}).get();

    ASSERT_EQ(4, results["found"].get<size_t>());

    collectionManager.drop_collection("coll1");
}

TEST_F(CollectionFilteringTest, NumericalFilteringWithArray) {
    Collection *coll1;

    std::vector<field> fields = {field("title", field_types::STRING, false),
                                 field("prices", field_types::INT32_ARRAY, false),};

    coll1 = collectionManager.get_collection("coll1").get();
    if (coll1 == nullptr) {
        coll1 = collectionManager.create_collection("coll1", 1, fields, "").get();
    }

    std::vector<std::vector<std::string>> records = {
        {"1", "T Shirt 1", "1", "2", "3"},
        {"2", "T Shirt 2", "1", "2", "3"},
        {"3", "T Shirt 3", "1", "2", "3"},
        {"4", "T Shirt 4", "1", "1", "1"},
    };

    for (size_t i = 0; i < records.size(); i++) {
        nlohmann::json doc;

        doc["id"] = records[i][0];
        doc["title"] = records[i][1];

        std::vector<int32_t> prices;
        for(size_t j = 2; j <= 4; j++) {
            prices.push_back(std::stoi(records[i][j]));
        }

        doc["prices"] = prices;

        ASSERT_TRUE(coll1->add(doc.dump()).ok());
    }

    // check equals on a repeating price
    auto results = coll1->search("*",
                                 {}, "prices:1",
                                 {}, {}, {0}, 10, 1, FREQUENCY, {true}).get();

    ASSERT_EQ(4, results["found"].get<size_t>());
    ASSERT_EQ(4, results["hits"].size());

    // check ranges

    results = coll1->search("*",
                            {}, "prices:>=1",
                            {}, {}, {0}, 10, 1, FREQUENCY, {true}).get();

    ASSERT_EQ(4, results["found"].get<size_t>());
    ASSERT_EQ(4, results["hits"].size());

    results = coll1->search("*",
                            {}, "prices:>=2",
                            {}, {}, {0}, 10, 1, FREQUENCY, {true}).get();

    ASSERT_EQ(3, results["found"].get<size_t>());
    ASSERT_EQ(3, results["hits"].size());

    results = coll1->search("*",
                            {}, "prices:<4",
                            {}, {}, {0}, 10, 1, FREQUENCY, {true}).get();

    ASSERT_EQ(4, results["found"].get<size_t>());
    ASSERT_EQ(4, results["hits"].size());

    results = coll1->search("*",
                            {}, "prices:<=2",
                            {}, {}, {0}, 10, 1, FREQUENCY, {true}).get();

    ASSERT_EQ(4, results["found"].get<size_t>());
    ASSERT_EQ(4, results["hits"].size());

    collectionManager.drop_collection("coll1");
}

TEST_F(CollectionFilteringTest, NegationOperatorBasics) {
    Collection *coll1;

    std::vector<field> fields = {
            field("title", field_types::STRING, false),
            field("artist", field_types::STRING, false),
            field("points", field_types::INT32, false),};

    coll1 = collectionManager.get_collection("coll1").get();
    if(coll1 == nullptr) {
        coll1 = collectionManager.create_collection("coll1", 2, fields, "points").get();
    }

    std::vector<std::vector<std::string>> records = {
        {"Taylor Swift Karaoke: reputation", "Taylor Swift"},
        {"Beat it", "Michael Jackson"},
        {"Style", "Taylor Swift"},
        {"Thriller", "Michael Joseph Jackson"},
    };

    for(size_t i=0; i<records.size(); i++) {
        nlohmann::json doc;

        doc["id"] = std::to_string(i);
        doc["title"] = records[i][0];
        doc["artist"] = records[i][1];
        doc["points"] = i;

        ASSERT_TRUE(coll1->add(doc.dump()).ok());
    }

    auto results = coll1->search("*", {"artist"}, "artist:!=Michael Jackson", {}, {}, {0}, 10, 1, FREQUENCY, {true}, 10).get();

    ASSERT_EQ(3, results["found"].get<size_t>());

    ASSERT_STREQ("3", results["hits"][0]["document"]["id"].get<std::string>().c_str());
    ASSERT_STREQ("2", results["hits"][1]["document"]["id"].get<std::string>().c_str());
    ASSERT_STREQ("0", results["hits"][2]["document"]["id"].get<std::string>().c_str());

    results = coll1->search("*", {"artist"}, "artist:!= Michael Jackson && points: >0", {}, {}, {0}, 10, 1, FREQUENCY, {true}, 10).get();
    ASSERT_EQ(2, results["found"].get<size_t>());
    ASSERT_STREQ("3", results["hits"][0]["document"]["id"].get<std::string>().c_str());
    ASSERT_STREQ("2", results["hits"][1]["document"]["id"].get<std::string>().c_str());

    // negation operation on multiple values

    results = coll1->search("*", {"artist"}, "artist:!= [Michael Jackson, Taylor Swift]", {}, {}, {0}, 10, 1, FREQUENCY, {true}, 10).get();
    ASSERT_EQ(1, results["found"].get<size_t>());
    ASSERT_STREQ("3", results["hits"][0]["document"]["id"].get<std::string>().c_str());

    // when no such value exists: should return all results
    results = coll1->search("*", {"artist"}, "artist:!=Foobar", {}, {}, {0}, 10, 1, FREQUENCY, {true}, 10).get();
    ASSERT_EQ(4, results["found"].get<size_t>());

    results = coll1->search("*", {"artist"}, "artist:! Jackson", {}, {}, {0}, 10, 1, FREQUENCY, {true}, 10).get();
    ASSERT_EQ(2, results["found"]);
    ASSERT_EQ("2", results["hits"][0]["document"]["id"]);
    ASSERT_EQ("0", results["hits"][1]["document"]["id"]);

    results = coll1->search("*", {"artist"}, "artist:![Swift, Jack]", {}, {}, {0}, 10, 1, FREQUENCY, {true}, 10).get();
    ASSERT_EQ(2, results["found"]);
    ASSERT_EQ("3", results["hits"][0]["document"]["id"]);
    ASSERT_EQ("1", results["hits"][1]["document"]["id"]);

    results = coll1->search("*", {"artist"}, "artist:![Swift, Jackson]", {}, {}, {0}, 10, 1, FREQUENCY, {true}, 10).get();
    ASSERT_EQ(0, results["found"]);

    results = coll1->search("*", {"artist"}, "artist:!=[]", {}, {}, {0}, 10, 1, FREQUENCY, {true}, 10).get();
    ASSERT_EQ(4, results["found"]);

    // empty value (bad filtering)
    auto res_op = coll1->search("*", {"artist"}, "artist:!=", {}, {}, {0}, 10, 1, FREQUENCY, {true}, 10);
    ASSERT_FALSE(res_op.ok());
    ASSERT_EQ("Error with filter field `artist`: Filter value cannot be empty.", res_op.error());

    res_op = coll1->search("*", {"artist"}, "artist:!= ", {}, {}, {0}, 10, 1, FREQUENCY, {true}, 10);
    ASSERT_FALSE(res_op.ok());
    ASSERT_EQ("Error with filter field `artist`: Filter value cannot be empty.", res_op.error());

    res_op = coll1->search("*", {"artist"}, "artist:!=``", {}, {}, {0}, 10, 1, FREQUENCY, {true}, 10);
    ASSERT_FALSE(res_op.ok());
    ASSERT_EQ("Error with filter field `artist`: Filter value cannot be empty.", res_op.error());

    res_op = coll1->search("*", {"artist"}, "artist:!=[`foo`, ``]", {}, {}, {0}, 10, 1, FREQUENCY, {true}, 10);
    ASSERT_FALSE(res_op.ok());
    ASSERT_EQ("Error with filter field `artist`: Filter value cannot be empty.", res_op.error());

    collectionManager.drop_collection("coll1");
}

TEST_F(CollectionFilteringTest, FilterStringsWithComma) {
    Collection *coll1;

    std::vector<field> fields = {field("place", field_types::STRING, true),
                                 field("state", field_types::STRING, false),
                                 field("points", field_types::INT32, false),};

    coll1 = collectionManager.get_collection("coll1").get();
    if(coll1 == nullptr) {
        coll1 = collectionManager.create_collection("coll1", 1, fields, "points").get();
    }

    std::vector<std::vector<std::string>> records = {
        {"St. John's Cathedral, Denver, Colorado", "Colorado"},
        {"Crater Lake National Park, Oregon", "Oregon"},
        {"St. Patrick's Cathedral, Manhattan", "New York"},
    };

    for(size_t i=0; i<records.size(); i++) {
        nlohmann::json doc;

        doc["id"] = std::to_string(i);
        doc["place"] = records[i][0];
        doc["state"] = records[i][1];
        doc["points"] = i;

        ASSERT_TRUE(coll1->add(doc.dump()).ok());
    }

    auto results = coll1->search("*", {"place"}, "place:= St. John's Cathedral, Denver, Colorado", {}, {}, {0}, 10, 1,
                                 FREQUENCY, {true}, 10).get();

    ASSERT_EQ(1, results["found"].get<size_t>());
    ASSERT_STREQ("0", results["hits"][0]["document"]["id"].get<std::string>().c_str());

    results = coll1->search("*", {"place"}, "place:= `St. John's Cathedral, Denver, Colorado`", {}, {}, {0}, 10, 1,
                            FREQUENCY, {true}, 10).get();

    ASSERT_EQ(1, results["found"].get<size_t>());
    ASSERT_STREQ("0", results["hits"][0]["document"]["id"].get<std::string>().c_str());

    results = coll1->search("*", {"place"}, "place:= [`St. John's Cathedral, Denver, Colorado`]", {}, {}, {0}, 10, 1,
                            FREQUENCY, {true}, 10).get();

    ASSERT_EQ(1, results["found"].get<size_t>());
    ASSERT_STREQ("0", results["hits"][0]["document"]["id"].get<std::string>().c_str());

    results = coll1->search("*", {"place"}, "place:= [`St. John's Cathedral, Denver, Colorado`, `St. Patrick's Cathedral, Manhattan`]", {}, {}, {0}, 10, 1,
                            FREQUENCY, {true}, 10).get();

    ASSERT_EQ(2, results["found"].get<size_t>());
    ASSERT_STREQ("2", results["hits"][0]["document"]["id"].get<std::string>().c_str());
    ASSERT_STREQ("0", results["hits"][1]["document"]["id"].get<std::string>().c_str());

    results = coll1->search("*", {"place"}, "place: [`Cathedral, Denver, Colorado`]", {}, {}, {0}, 10, 1,
                            FREQUENCY, {true}, 10).get();

    ASSERT_EQ(1, results["found"].get<size_t>());
    ASSERT_STREQ("0", results["hits"][0]["document"]["id"].get<std::string>().c_str());

    results = coll1->search("*", {"place"}, "place: []", {}, {}, {0}, 10, 1,
                            FREQUENCY, {true}, 10).get();

    ASSERT_EQ(0, results["found"].get<size_t>());

    collectionManager.drop_collection("coll1");
}

TEST_F(CollectionFilteringTest, NumericalRangeFilter) {
    std::vector<field> fields = {field("company", field_types::STRING, true),
                                 field("num_employees", field_types::INT32, false),};

    Collection* coll1 = collectionManager.create_collection("coll1", 1, fields, "num_employees").get();

    std::vector<std::vector<std::string>> records = {
        {"123", "Company 1", "50"},
        {"125", "Company 2", "150"},
        {"127", "Company 3", "250"},
        {"129", "Stark Industries 4", "500"},
    };

    for(size_t i=0; i<records.size(); i++) {
        nlohmann::json doc;

        doc["id"] = records[i][0];
        doc["company"] = records[i][1];
        doc["num_employees"] = std::stoi(records[i][2]);

        ASSERT_TRUE(coll1->add(doc.dump()).ok());
    }

    std::vector<sort_by> sort_fields = { sort_by("num_employees", "ASC") };

    auto results = coll1->search("*", {}, "num_employees:>=100 && num_employees:<=300", {}, sort_fields, {0}, 10, 1,
                                 FREQUENCY, {true}, 10).get();

    ASSERT_EQ(2, results["found"].get<size_t>());
    ASSERT_STREQ("125", results["hits"][0]["document"]["id"].get<std::string>().c_str());
    ASSERT_STREQ("127", results["hits"][1]["document"]["id"].get<std::string>().c_str());

    collectionManager.drop_collection("coll1");
}

TEST_F(CollectionFilteringTest, RangeFilterOnTimestamp) {
    std::vector<field> fields = {field("ts", field_types::INT64, false)};

    Collection* coll1 = collectionManager.create_collection(
            "coll1", 1, fields, "", 0, "", {}, {"."}
    ).get();

    nlohmann::json doc1;
    doc1["id"] = "0";
    doc1["ts"] = 1646092800000;

    nlohmann::json doc2;
    doc2["id"] = "1";
    doc2["ts"] = 1648771199000;

    nlohmann::json doc3;
    doc3["id"] = "2";
    doc3["ts"] = 1647111199000;

    ASSERT_TRUE(coll1->add(doc1.dump()).ok());
    ASSERT_TRUE(coll1->add(doc2.dump()).ok());
    ASSERT_TRUE(coll1->add(doc3.dump()).ok());

    auto results = coll1->search("*", {},"ts:[1646092800000..1648771199000]", {}, {}, {0}, 10,
                                 1, FREQUENCY, {false}).get();

    ASSERT_EQ(3, results["hits"].size());

    collectionManager.drop_collection("coll1");
}

TEST_F(CollectionFilteringTest, QueryBoolFields) {
    Collection *coll_bool;

    std::ifstream infile(std::string(ROOT_DIR)+"test/bool_documents.jsonl");
    std::vector<field> fields = {
            field("popular", field_types::BOOL, false),
            field("title", field_types::STRING, false),
            field("rating", field_types::FLOAT, false),
            field("bool_array", field_types::BOOL_ARRAY, false),
    };

    std::vector<sort_by> sort_fields = { sort_by("popular", "DESC"), sort_by("rating", "DESC") };

    coll_bool = collectionManager.get_collection("coll_bool").get();
    if(coll_bool == nullptr) {
        coll_bool = collectionManager.create_collection("coll_bool", 1, fields, "rating").get();
    }

    std::string json_line;

    while (std::getline(infile, json_line)) {
        coll_bool->add(json_line);
    }

    infile.close();

    // Plain search with no filters - results should be sorted correctly
    query_fields = {"title"};
    std::vector<std::string> facets;
    nlohmann::json results = coll_bool->search("the", query_fields, "", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(5, results["hits"].size());

    std::vector<std::string> ids = {"1", "3", "4", "9", "2"};

    for(size_t i = 0; i < results["hits"].size(); i++) {
        nlohmann::json result = results["hits"].at(i);
        std::string result_id = result["document"]["id"];
        std::string id = ids.at(i);
        ASSERT_STREQ(id.c_str(), result_id.c_str());
    }

    // Searching on a bool field
    results = coll_bool->search("the", query_fields, "popular:true", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(3, results["hits"].size());

    ids = {"1", "3", "4"};

    for(size_t i = 0; i < results["hits"].size(); i++) {
        nlohmann::json result = results["hits"].at(i);
        std::string result_id = result["document"]["id"];
        std::string id = ids.at(i);
        ASSERT_STREQ(id.c_str(), result_id.c_str());
    }

    results = coll_bool->search("*", query_fields, "popular:true", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(7, results["hits"].size());

    ids = {"1", "0", "3", "5", "6", "7", "4"};

    for(size_t i = 0; i < results["hits"].size(); i++) {
        nlohmann::json result = results["hits"].at(i);
        std::string result_id = result["document"]["id"];
        std::string id = ids.at(i);
        ASSERT_EQ(id, result_id);
    }

    // alternative `:=` syntax
    results = coll_bool->search("the", query_fields, "popular:=true", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(3, results["hits"].size());

    results = coll_bool->search("the", query_fields, "popular:false", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(2, results["hits"].size());

    results = coll_bool->search("the", query_fields, "popular:= false", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(2, results["hits"].size());

    ids = {"9", "2"};

    for(size_t i = 0; i < results["hits"].size(); i++) {
        nlohmann::json result = results["hits"].at(i);
        std::string result_id = result["document"]["id"];
        std::string id = ids.at(i);
        ASSERT_STREQ(id.c_str(), result_id.c_str());
    }

    // searching against a bool array field

    // should be able to filter with an array of boolean values
    Option<nlohmann::json> res_op = coll_bool->search("the", query_fields, "bool_array:[true, false]", facets,
                                                      sort_fields, {0}, 10, 1, FREQUENCY, {false});
    ASSERT_TRUE(res_op.ok());
    results = res_op.get();

    ASSERT_EQ(5, results["hits"].size());

    results = coll_bool->search("the", query_fields, "bool_array: true", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(4, results["hits"].size());
    ids = {"1", "4", "9", "2"};

    for(size_t i = 0; i < results["hits"].size(); i++) {
        nlohmann::json result = results["hits"].at(i);
        std::string result_id = result["document"]["id"];
        std::string id = ids.at(i);
        ASSERT_STREQ(id.c_str(), result_id.c_str());
    }

    // should be able to search using array with a single element boolean value

    results = coll_bool->search("the", query_fields, "bool_array:[true]", facets,
                                 sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();

    ASSERT_EQ(4, results["hits"].size());

    for(size_t i = 0; i < results["hits"].size(); i++) {
        nlohmann::json result = results["hits"].at(i);
        std::string result_id = result["document"]["id"];
        std::string id = ids.at(i);
        ASSERT_STREQ(id.c_str(), result_id.c_str());
    }

    // not equals on bool field

    results = coll_bool->search("the", query_fields, "popular:!= true", facets,
                             sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();

    ASSERT_EQ(2, results["hits"].size());
    ASSERT_EQ("9", results["hits"][0]["document"]["id"].get<std::string>());
    ASSERT_EQ("2", results["hits"][1]["document"]["id"].get<std::string>());

    // not equals on bool array field
    results = coll_bool->search("the", query_fields, "bool_array:!= [true]", facets,
                                sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();

    ASSERT_EQ(1, results["hits"].size());
    ASSERT_EQ("3", results["hits"][0]["document"]["id"].get<std::string>());

    // empty filter value
    res_op = coll_bool->search("the", query_fields, "bool_array:=", facets,
                                    sort_fields, {0}, 10, 1, FREQUENCY, {false});

    ASSERT_FALSE(res_op.ok());
    ASSERT_EQ("Error with filter field `bool_array`: Filter value cannot be empty.", res_op.error());

    collectionManager.drop_collection("coll_bool");
}

TEST_F(CollectionFilteringTest, FilteringWithTokenSeparators) {
    std::vector<field> fields = {field("code", field_types::STRING, true)};

    Collection* coll1 = collectionManager.create_collection(
        "coll1", 1, fields, "", 0, "", {}, {"."}
    ).get();

    nlohmann::json doc1;
    doc1["id"] = "0";
    doc1["code"] = "7318.15";

    ASSERT_TRUE(coll1->add(doc1.dump()).ok());

    auto results = coll1->search("*", {},"code:=7318.15", {}, {}, {0}, 10,
                                 1, FREQUENCY, {false}).get();

    ASSERT_EQ(1, results["hits"].size());

    results = coll1->search("*", {},"code:=`7318.15`", {}, {}, {0}, 10,
                            1, FREQUENCY, {false}).get();

    ASSERT_EQ(1, results["hits"].size());

    collectionManager.drop_collection("coll1");

    Collection* coll2 = collectionManager.create_collection(
            "coll2", 1, fields, "", 0, "", {"."}, {}
    ).get();

    doc1["id"] = "0";
    doc1["code"] = "7318.15";

    ASSERT_TRUE(coll2->add(doc1.dump()).ok());

    results = coll2->search("*", {},"code:=7318.15", {}, {}, {0}, 10,
                                 1, FREQUENCY, {false}).get();

    ASSERT_EQ(1, results["hits"].size());

    collectionManager.drop_collection("coll2");
}

TEST_F(CollectionFilteringTest, ExactFilteringSingleQueryTerm) {
    std::vector<field> fields = {field("name", field_types::STRING, false),
                                 field("tags", field_types::STRING_ARRAY, false)};

    Collection* coll1 = collectionManager.create_collection(
        "coll1", 1, fields, "", 0, "", {}, {"."}
    ).get();

    nlohmann::json doc1;
    doc1["id"] = "0";
    doc1["name"] = "AT&T GoPhone";
    doc1["tags"] = {"AT&T GoPhone"};

    nlohmann::json doc2;
    doc2["id"] = "1";
    doc2["name"] = "AT&T";
    doc2["tags"] = {"AT&T"};

    ASSERT_TRUE(coll1->add(doc1.dump()).ok());
    ASSERT_TRUE(coll1->add(doc2.dump()).ok());

    auto results = coll1->search("*", {},"name:=AT&T", {}, {}, {0}, 10,
                                 1, FREQUENCY, {false}).get();
    ASSERT_EQ(1, results["hits"].size());
    ASSERT_EQ("1", results["hits"][0]["document"]["id"].get<std::string>());

    results = coll1->search("*", {},"tags:=AT&T", {}, {}, {0}, 10,
                            1, FREQUENCY, {false}).get();
    ASSERT_EQ(1, results["hits"].size());
    ASSERT_EQ("1", results["hits"][0]["document"]["id"].get<std::string>());

    nlohmann::json doc3;
    doc3["id"] = "2";
    doc3["name"] = "Phone";
    doc3["tags"] = {"Samsung Phone", "Phone"};

    ASSERT_TRUE(coll1->add(doc3.dump()).ok());

    results = coll1->search("*", {},"tags:=Phone", {}, {}, {0}, 10,
                            1, FREQUENCY, {false}).get();
    ASSERT_EQ(1, results["hits"].size());
    ASSERT_EQ("2", results["hits"][0]["document"]["id"].get<std::string>());

    collectionManager.drop_collection("coll1");
}

TEST_F(CollectionFilteringTest, ExactFilteringRepeatingTokensSingularField) {
    std::vector<field> fields = {field("name", field_types::STRING, false)};

    Collection* coll1 = collectionManager.create_collection(
        "coll1", 1, fields, "", 0, "", {}, {"."}
    ).get();

    nlohmann::json doc1;
    doc1["id"] = "0";
    doc1["name"] = "Cardiology - Interventional Cardiology";

    nlohmann::json doc2;
    doc2["id"] = "1";
    doc2["name"] = "Cardiology - Interventional";

    nlohmann::json doc3;
    doc3["id"] = "2";
    doc3["name"] = "Cardiology - Interventional Cardiology Department";

    nlohmann::json doc4;
    doc4["id"] = "3";
    doc4["name"] = "Interventional Cardiology - Interventional Cardiology";

    ASSERT_TRUE(coll1->add(doc1.dump()).ok());
    ASSERT_TRUE(coll1->add(doc2.dump()).ok());
    ASSERT_TRUE(coll1->add(doc3.dump()).ok());
    ASSERT_TRUE(coll1->add(doc4.dump()).ok());

    auto results = coll1->search("*", {},"name:=Cardiology - Interventional Cardiology", {}, {}, {0}, 10,
                                 1, FREQUENCY, {false}).get();
    ASSERT_EQ(1, results["hits"].size());
    ASSERT_EQ("0", results["hits"][0]["document"]["id"].get<std::string>());

    results = coll1->search("*", {},"name:=Cardiology - Interventional", {}, {}, {0}, 10,
                            1, FREQUENCY, {false}).get();
    ASSERT_EQ(1, results["hits"].size());
    ASSERT_EQ("1", results["hits"][0]["document"]["id"].get<std::string>());

    results = coll1->search("*", {},"name:=Interventional Cardiology", {}, {}, {0}, 10,
                            1, FREQUENCY, {false}).get();
    ASSERT_EQ(0, results["hits"].size());

    results = coll1->search("*", {},"name:=Cardiology", {}, {}, {0}, 10,
                            1, FREQUENCY, {false}).get();
    ASSERT_EQ(0, results["hits"].size());

    collectionManager.drop_collection("coll1");
}

TEST_F(CollectionFilteringTest, ExactFilteringRepeatingTokensArrayField) {
    std::vector<field> fields = {field("name", field_types::STRING_ARRAY, false)};

    Collection* coll1 = collectionManager.create_collection(
        "coll1", 1, fields, "", 0, "", {}, {"."}
    ).get();

    nlohmann::json doc1;
    doc1["id"] = "0";
    doc1["name"] = {"Cardiology - Interventional Cardiology"};

    nlohmann::json doc2;
    doc2["id"] = "1";
    doc2["name"] = {"Cardiology - Interventional"};

    nlohmann::json doc3;
    doc3["id"] = "2";
    doc3["name"] = {"Cardiology - Interventional Cardiology Department"};

    nlohmann::json doc4;
    doc4["id"] = "3";
    doc4["name"] = {"Interventional Cardiology - Interventional Cardiology"};

    ASSERT_TRUE(coll1->add(doc1.dump()).ok());
    ASSERT_TRUE(coll1->add(doc2.dump()).ok());
    ASSERT_TRUE(coll1->add(doc3.dump()).ok());
    ASSERT_TRUE(coll1->add(doc4.dump()).ok());

    auto results = coll1->search("*", {},"name:=Cardiology - Interventional Cardiology", {}, {}, {0}, 10,
                                 1, FREQUENCY, {false}).get();
    ASSERT_EQ(1, results["hits"].size());
    ASSERT_EQ("0", results["hits"][0]["document"]["id"].get<std::string>());

    results = coll1->search("*", {},"name:=Cardiology - Interventional", {}, {}, {0}, 10,
                            1, FREQUENCY, {false}).get();
    ASSERT_EQ(1, results["hits"].size());
    ASSERT_EQ("1", results["hits"][0]["document"]["id"].get<std::string>());

    results = coll1->search("*", {},"name:=Interventional Cardiology", {}, {}, {0}, 10,
                            1, FREQUENCY, {false}).get();
    ASSERT_EQ(0, results["hits"].size());

    results = coll1->search("*", {},"name:=Cardiology", {}, {}, {0}, 10,
                            1, FREQUENCY, {false}).get();
    ASSERT_EQ(0, results["hits"].size());

    collectionManager.drop_collection("coll1");
}

TEST_F(CollectionFilteringTest, ExcludeMultipleTokens) {
    Collection *coll1;

    std::vector<field> fields = {field("title", field_types::STRING, false),
                                 field("points", field_types::INT32, false),};

    coll1 = collectionManager.get_collection("coll1").get();
    if(coll1 == nullptr) {
        coll1 = collectionManager.create_collection("coll1", 1, fields, "points").get();
    }

    std::vector<std::vector<std::string>> records = {
        {"alpha"},
        {"TXBT0eiYnFhkJHqz02Wv0PWN5hp1"},
        {"3u7RtEn5S9fcnizoUojWUwW23Yf2"},
        {"HpPALvzDDVc3zMmlAAUySwp8Ir33"},
        {"9oF2qhYI8sdBa2xJSerfmntpvBr2"},
        {"5fAnLlld5obG4vhhNIbIeoHe1uB2"},
        {"4OlIYKbzwIUoAOYy6dfDzCREezg1"},
        {"4JK1BvoqCuTeMwEZorlKj8hnSl02"},
        {"3tQBmRH0AQPEWyoKcDNYJyIxQQe2"},
        {"3Mvl5HZgNwQkHykAqL77oMfo8DW2"},
        {"3Ipnw5JATpYFyCcdUKTBhCicjoH3"},
        {"2rizUF2ntNSUVpaXwPdHmSBB6C63"},
        {"2kMHFOUQhAQK9cQbFNoXGpcAFVD2"},
    };

    for(size_t i=0; i<records.size(); i++) {
        nlohmann::json doc;
        doc["id"] = std::to_string(i);
        doc["title"] = records[i][0];
        doc["points"] = i;

        ASSERT_TRUE(coll1->add(doc.dump()).ok());
    }

    auto results = coll1->search(
            "-TXBT0eiYnFhkJHqz02Wv0PWN5hp1 -3u7RtEn5S9fcnizoUojWUwW23Yf2 -HpPALvzDDVc3zMmlAAUySwp8Ir33 "
            "-9oF2qhYI8sdBa2xJSerfmntpvBr2 -5fAnLlld5obG4vhhNIbIeoHe1uB2 -4OlIYKbzwIUoAOYy6dfDzCREezg1 "
            "-4JK1BvoqCuTeMwEZorlKj8hnSl02 -3tQBmRH0AQPEWyoKcDNYJyIxQQe2 -3Mvl5HZgNwQkHykAqL77oMfo8DW2 "
            "-3Ipnw5JATpYFyCcdUKTBhCicjoH3 -2rizUF2ntNSUVpaXwPdHmSBB6C63 -2kMHFOUQhAQK9cQbFNoXGpcAFVD2",
            {"title"}, "",
            {}, {}, {0}, 10, 1, FREQUENCY).get();

    ASSERT_EQ(1, results["found"].get<size_t>());
    ASSERT_EQ(1, results["hits"].size());

    ASSERT_STREQ("0", results["hits"][0]["document"]["id"].get<std::string>().c_str());

    collectionManager.drop_collection("coll1");
}

TEST_F(CollectionFilteringTest, FilteringAfterUpsertOnArrayWithTokenSeparators) {
    std::vector<field> fields = {field("name", field_types::STRING, false),
                                 field("tags", field_types::STRING_ARRAY, false),
                                 field("tag", field_types::STRING, false)};

    Collection* coll1 = collectionManager.create_collection("coll1", 1, fields, "", 0, "", {}, {"-"}).get();

    nlohmann::json doc1;
    doc1["id"] = "0";
    doc1["name"] = "david";
    doc1["tags"] = {"alpha-beta-gamma", "foo-bar-baz"};
    doc1["tag"] = "foo-bar-baz";

    nlohmann::json doc2;
    doc2["id"] = "1";
    doc2["name"] = "david";
    doc2["tags"] = {"alpha-gamma-beta", "bar-foo-baz"};
    doc2["tag"] = "alpha-beta";

    ASSERT_TRUE(coll1->add(doc1.dump()).ok());
    ASSERT_TRUE(coll1->add(doc2.dump()).ok());

    auto results = coll1->search("david", {"name"},"tags:=[foo-bar-baz]", {}, {}, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(1, results["hits"].size());
    ASSERT_EQ("0", results["hits"][0]["document"]["id"].get<std::string>());

    // upsert with "foo-bar-baz" removed
    doc1["tags"] = {"alpha-beta-gamma"};
    coll1->add(doc1.dump(), UPSERT);

    results = coll1->search("david", {"name"},"tags:=[foo-bar-baz]", {}, {}, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(0, results["hits"].size());

    results = coll1->search("david", {"name"},"tags:=[bar-foo-baz]", {}, {}, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(1, results["hits"].size());
    ASSERT_EQ("1", results["hits"][0]["document"]["id"].get<std::string>());

    // repeat for singular string field: upsert with "foo-bar-baz" removed
    doc1["tag"] = "alpha-beta-gamma";
    coll1->add(doc1.dump(), UPSERT);

    results = coll1->search("david", {"name"},"tag:=[foo-bar-baz]", {}, {}, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(0, results["hits"].size());

    collectionManager.drop_collection("coll1");
}

TEST_F(CollectionFilteringTest, FilteringAfterUpsertOnArrayWithSymbolsToIndex) {
    std::vector<field> fields = {field("name", field_types::STRING, false),
                                 field("tags", field_types::STRING_ARRAY, false),
                                 field("tag", field_types::STRING, false)};

    Collection* coll1 = collectionManager.create_collection("coll1", 1, fields, "", 0, "", {"-"}, {}).get();

    nlohmann::json doc1;
    doc1["id"] = "0";
    doc1["name"] = "david";
    doc1["tags"] = {"alpha-beta-gamma", "foo-bar-baz"};
    doc1["tag"] = "foo-bar-baz";

    nlohmann::json doc2;
    doc2["id"] = "1";
    doc2["name"] = "david";
    doc2["tags"] = {"alpha-gamma-beta", "bar-foo-baz"};
    doc2["tag"] = "alpha-beta";

    ASSERT_TRUE(coll1->add(doc1.dump()).ok());
    ASSERT_TRUE(coll1->add(doc2.dump()).ok());

    auto results = coll1->search("david", {"name"},"tags:=[foo-bar-baz]", {}, {}, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(1, results["hits"].size());
    ASSERT_EQ("0", results["hits"][0]["document"]["id"].get<std::string>());

    // upsert with "foo-bar-baz" removed
    doc1["tags"] = {"alpha-beta-gamma"};
    coll1->add(doc1.dump(), UPSERT);

    results = coll1->search("david", {"name"},"tags:=[foo-bar-baz]", {}, {}, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(0, results["hits"].size());

    results = coll1->search("david", {"name"},"tags:=[bar-foo-baz]", {}, {}, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(1, results["hits"].size());
    ASSERT_EQ("1", results["hits"][0]["document"]["id"].get<std::string>());

    // repeat for singular string field: upsert with "foo-bar-baz" removed
    doc1["tag"] = "alpha-beta-gamma";
    coll1->add(doc1.dump(), UPSERT);

    results = coll1->search("david", {"name"},"tag:=[foo-bar-baz]", {}, {}, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(0, results["hits"].size());

    collectionManager.drop_collection("coll1");
}

TEST_F(CollectionFilteringTest, ComplexFilterQuery) {
    nlohmann::json schema_json =
            R"({
                "name": "ComplexFilterQueryCollection",
                "fields": [
                    {"name": "name", "type": "string"},
                    {"name": "age", "type": "int32"},
                    {"name": "years", "type": "int32[]"},
                    {"name": "rating", "type": "float"},
                    {"name": "tags", "type": "string[]"}
                ]
            })"_json;

    auto op = collectionManager.create_collection(schema_json);
    ASSERT_TRUE(op.ok());
    auto coll = op.get();

    std::ifstream infile(std::string(ROOT_DIR)+"test/numeric_array_documents.jsonl");
    std::string json_line;
    while (std::getline(infile, json_line)) {
        auto add_op = coll->add(json_line);
        ASSERT_TRUE(add_op.ok());
    }
    infile.close();

    std::vector<sort_by> sort_fields_desc = {sort_by("rating", "DESC")};
    nlohmann::json results = coll->search("Jeremy", {"name"}, "(rating:>=0 && years:>2000) && age:>50",
                                          {}, sort_fields_desc, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(0, results["hits"].size());

    results = coll->search("*", {"name"}, "(age:>50 && rating:>5) || years:<2000",
                                          {}, sort_fields_desc, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(2, results["hits"].size());

    results = coll->search("Jeremy", {"name"}, "(age:>50 || rating:>5) && years:<2000",
                                          {}, sort_fields_desc, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(2, results["hits"].size());

    std::vector<std::string> ids = {"4", "3"};
    for (size_t i = 0; i < results["hits"].size(); i++) {
        nlohmann::json result = results["hits"].at(i);
        std::string result_id = result["document"]["id"];
        std::string id = ids.at(i);
        ASSERT_STREQ(id.c_str(), result_id.c_str());
    }

    results = coll->search("Jeremy", {"name"}, "(age:<50 && rating:10) || (years:>2000 && rating:<5)",
                                          {}, sort_fields_desc, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(1, results["hits"].size());

    ids = {"0"};
    for (size_t i = 0; i < results["hits"].size(); i++) {
        nlohmann::json result = results["hits"].at(i);
        std::string result_id = result["document"]["id"];
        std::string id = ids.at(i);
        ASSERT_STREQ(id.c_str(), result_id.c_str());
    }

    results = coll->search("Jeremy", {"name"}, "years:>2000 && ((age:<30 && rating:>5) || (age:>50 && rating:<5))",
                           {}, sort_fields_desc, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(1, results["hits"].size());

    ids = {"2"};
    for (size_t i = 0; i < results["hits"].size(); i++) {
        nlohmann::json result = results["hits"].at(i);
        std::string result_id = result["document"]["id"];
        std::string id = ids.at(i);
        ASSERT_STREQ(id.c_str(), result_id.c_str());
    }

    std::string extreme_filter = "(years:>2000 && ((age:<30 && rating:>5) || (age:>50 && rating:<5))) ||"
                                 "(years:>2000 && ((age:<30 && rating:>5) || (age:>50 && rating:<5))) ||"
                                 "(years:>2000 && ((age:<30 && rating:>5) || (age:>50 && rating:<5))) ||"
                                 "(years:>2000 && ((age:<30 && rating:>5) || (age:>50 && rating:<5))) ||"
                                 "(years:>2000 && ((age:<30 && rating:>5) || (age:>50 && rating:<5))) ||"
                                 "(years:>2000 && ((age:<30 && rating:>5) || (age:>50 && rating:<5))) ||"
                                 "(years:>2000 && ((age:<30 && rating:>5) || (age:>50 && rating:<5))) ||"
                                 "(years:>2000 && ((age:<30 && rating:>5) || (age:>50 && rating:<5))) ||"
                                 "(years:>2000 && ((age:<30 && rating:>5) || (age:>50 && rating:<5))) ||"
                                 "(years:>2000 && ((age:<30 && rating:>5) || (age:>50 && rating:<5)))";

    auto search_op = coll->search("Jeremy", {"name"}, extreme_filter,
                                  {}, sort_fields_desc, {0}, 10, 1, FREQUENCY, {false});
    ASSERT_TRUE(search_op.ok());
    ASSERT_EQ(1, search_op.get()["hits"].size());

    extreme_filter += "|| (years:>2000 && ((age:<30 && rating:>5) || (age:>50 && rating:<5)))";
    search_op = coll->search("Jeremy", {"name"}, extreme_filter,
                             {}, sort_fields_desc, {0}, 10, 1, FREQUENCY, {false});
    ASSERT_FALSE(search_op.ok());
    ASSERT_EQ("`filter_by` has too many operations. Maximum allowed: 100. Use `--filter-by-max-ops` command line "
              "argument to customize this value.", search_op.error());

    collectionManager.dispose();
    delete store;

    store = new Store(state_dir_path);
    collectionManager.init(store, 1.0, "auth_key", quit, 109); // Re-initialize with 109 filter operations allowed.
    auto load_op = collectionManager.load(8, 1000);

    if(!load_op.ok()) {
        LOG(ERROR) << load_op.error();
    }
    ASSERT_TRUE(load_op.ok());

    coll = collectionManager.get_collection_unsafe("ComplexFilterQueryCollection");
    search_op = coll->search("Jeremy", {"name"}, extreme_filter,
                                  {}, sort_fields_desc, {0}, 10, 1, FREQUENCY, {false});
    ASSERT_TRUE(search_op.ok());
    ASSERT_EQ(1, search_op.get()["hits"].size());

    extreme_filter += "|| (years:>2000 && ((age:<30 && rating:>5) || (age:>50 && rating:<5)))";
    search_op = coll->search("Jeremy", {"name"}, extreme_filter,
                             {}, sort_fields_desc, {0}, 10, 1, FREQUENCY, {false});
    ASSERT_FALSE(search_op.ok());
    ASSERT_EQ("`filter_by` has too many operations. Maximum allowed: 109. Use `--filter-by-max-ops` command line "
              "argument to customize this value.", search_op.error());

    collectionManager.drop_collection("ComplexFilterQueryCollection");
}

TEST_F(CollectionFilteringTest, PrefixSearchWithFilter) {
    std::ifstream infile(std::string(ROOT_DIR)+"test/documents.jsonl");
    std::vector<field> search_fields = {
            field("title", field_types::STRING, false),
            field("points", field_types::INT32, false)
    };

    query_fields = {"title"};
    sort_fields = { sort_by(sort_field_const::text_match, "DESC"), sort_by("points", "DESC") };

    auto collection = collectionManager.create_collection("collection", 4, search_fields, "points").get();

    std::string json_line;

    // dummy record for record id 0: to make the test record IDs to match with line numbers
    json_line = "{\"points\":10,\"title\":\"z\"}";
    collection->add(json_line);

    while (std::getline(infile, json_line)) {
        collection->add(json_line);
    }

    infile.close();

    std::vector<std::string> facets;
    auto results = collection->search("what ex", query_fields, "points: >10", facets, sort_fields, {0}, 10, 1, MAX_SCORE, {true}, 10,
                                 spp::sparse_hash_set<std::string>(),
                                 spp::sparse_hash_set<std::string>(), 10, "", 30, 5,
                                 "", 10).get();
    ASSERT_EQ(7, results["hits"].size());
    std::vector<std::string> ids = {"6", "12", "19", "22", "13", "8", "15"};

    for(size_t i = 0; i < results["hits"].size(); i++) {
        nlohmann::json result = results["hits"].at(i);
        std::string result_id = result["document"]["id"];
        std::string id = ids.at(i);
        ASSERT_STREQ(id.c_str(), result_id.c_str());
    }

    collectionManager.drop_collection("collection");
}

TEST_F(CollectionFilteringTest, LargeFilterToken) {
    nlohmann::json json =
            R"({
                "name": "LargeFilterTokenCollection",
                "fields": [
                    {"name": "uri", "type": "string"}
                ],
                "symbols_to_index": [
                    "/",
                    "-"
                ]
            })"_json;

    auto op = collectionManager.create_collection(json);
    ASSERT_TRUE(op.ok());
    auto coll = op.get();

    json.clear();
    std::string token = "rade/aols/insolvenzrecht/persoenliche-risiken-fuer-organe-von-kapitalgesellschaften-gmbh-"
                        "geschaeftsfuehrer-ag-vorstand";
    json["uri"] = token;
    auto add_op = coll->add(json.dump());
    ASSERT_TRUE(add_op.ok());

    auto results = coll->search("*", query_fields, "", {}, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(1, results["hits"].size());

    results = coll->search("*", query_fields, "uri:" + token, {}, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(1, results["hits"].size());

    token.erase(100); // Max token length that's indexed is 100, we'll still get a match.
    results = coll->search("*", query_fields, "uri:" + token, {}, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(1, results["hits"].size());

    token.erase(99);
    results = coll->search("*", query_fields, "uri:" + token, {}, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(0, results["hits"].size());
}

TEST_F(CollectionFilteringTest, NonIndexedFiltering) {
    nlohmann::json json =
            R"({
                "name": "NonIndexedCollection",
                "fields": [
                    {"name": "uri", "type": "string"},
                    {"name": "non_index", "type": "string", "index": false, "optional": true}
                ]
            })"_json;

    auto op = collectionManager.create_collection(json);
    ASSERT_TRUE(op.ok());
    auto coll = op.get();

    json.clear();
    json = R"({
                "uri": "token",
                "non_index": "foo"
            })"_json;
    auto add_op = coll->add(json.dump());
    ASSERT_TRUE(add_op.ok());

    auto search_op = coll->search("*", {}, "", {}, sort_fields, {0}, 10, 1, FREQUENCY, {false});
    ASSERT_EQ(1, search_op.get()["hits"].size());

    search_op = coll->search("*", {}, "non_index:= bar", {}, sort_fields, {0}, 10, 1, FREQUENCY, {false});
    ASSERT_FALSE(search_op.ok());
    ASSERT_EQ("Cannot filter on non-indexed field `non_index`.", search_op.error());
}

TEST_F(CollectionFilteringTest, ComputeFilterResult) {
    Collection *coll1;

    std::vector<field> fields = {field("title", field_types::STRING, false),
                                 field("points", field_types::INT32, false),};

    coll1 = collectionManager.get_collection("coll1").get();
    if(coll1 == nullptr) {
        coll1 = collectionManager.create_collection("coll1", 1, fields, "points").get();
    }

    for(size_t i=0; i<50; i++) {
        nlohmann::json doc;

        doc["title"] = i < 10 ? "foo" : "bar";
        doc["points"] = i;

        ASSERT_TRUE(coll1->add(doc.dump()).ok());
    }

    auto res_op = coll1->search("*", {}, "title: foo",
                                {}, {}, {0}, 10, 1, FREQUENCY, {true});

    auto results = res_op.get();

    ASSERT_EQ(10, results["found"]);

    res_op = coll1->search("*", {}, "title: bar && points:>=10",
                                {}, {}, {0}, 10, 1, FREQUENCY, {true});

    results = res_op.get();

    ASSERT_EQ(40, results["found"]);

    collectionManager.drop_collection("coll1");
}

TEST_F(CollectionFilteringTest, PrefixFilterOnTextFields) {
    Collection *coll_mul_fields;

    std::ifstream infile(std::string(ROOT_DIR)+"test/multi_field_documents.jsonl");
    std::vector<field> fields = {
            field("title", field_types::STRING, false),
            field("starring", field_types::STRING, false),
            field("cast", field_types::STRING_ARRAY, true),
            field("points", field_types::INT32, false)
    };

    coll_mul_fields = collectionManager.get_collection("coll_mul_fields").get();
    if(coll_mul_fields == nullptr) {
        coll_mul_fields = collectionManager.create_collection("coll_mul_fields", 4, fields, "points").get();
    }

    std::string json_line;

    while (std::getline(infile, json_line)) {
        coll_mul_fields->add(json_line);
    }

    infile.close();

    nlohmann::json results = coll_mul_fields->search("*", {}, "cast: Chris", {}, {}, {0},
                                                     10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(3, results["hits"].size());

    std::vector<std::string> ids = {"6", "1", "7"};

    for(size_t i = 0; i < results["hits"].size(); i++) {
        nlohmann::json result = results["hits"].at(i);
        std::string result_id = result["document"]["id"];
        std::string id = ids.at(i);
        ASSERT_EQ(id, result_id);
    }

    results = coll_mul_fields->search("*", {}, "cast: Ch*", {}, {}, {0},
                                                     10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(3, results["hits"].size());

    ids = {"6", "1", "7"};

    for(size_t i = 0; i < results["hits"].size(); i++) {
        nlohmann::json result = results["hits"].at(i);
        std::string result_id = result["document"]["id"];
        std::string id = ids.at(i);
        ASSERT_EQ(id, result_id);
    }

    results = coll_mul_fields->search("*", {}, "cast: M*", {}, {}, {0},
                                                     10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(3, results["hits"].size());

    ids = {"3", "2", "16"};

    for(size_t i = 0; i < results["hits"].size(); i++) {
        nlohmann::json result = results["hits"].at(i);
        std::string result_id = result["document"]["id"];
        std::string id = ids.at(i);
        ASSERT_EQ(id, result_id);
    }

    results = coll_mul_fields->search("*", {}, "cast: Chris P*", {}, {}, {0},
                                                     10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(2, results["hits"].size());

    ids = {"1", "7"};

    for(size_t i = 0; i < results["hits"].size(); i++) {
        nlohmann::json result = results["hits"].at(i);
        std::string result_id = result["document"]["id"];
        std::string id = ids.at(i);
        ASSERT_EQ(id, result_id);
    }

    results = coll_mul_fields->search("*", {}, "cast: [Martin, Chris P*]", {}, {}, {0},
                                                     10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(3, results["hits"].size());

    ids = {"2", "1", "7"};

    for(size_t i = 0; i < results["hits"].size(); i++) {
        nlohmann::json result = results["hits"].at(i);
        std::string result_id = result["document"]["id"];
        std::string id = ids.at(i);
        ASSERT_EQ(id, result_id);
    }

    results = coll_mul_fields->search("*", {}, "cast: [M*, Chris P*]", {}, {}, {0},
                                                     10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(5, results["hits"].size());

    ids = {"3", "2", "16", "1", "7"};

    for(size_t i = 0; i < results["hits"].size(); i++) {
        nlohmann::json result = results["hits"].at(i);
        std::string result_id = result["document"]["id"];
        std::string id = ids.at(i);
        ASSERT_EQ(id, result_id);
    }

    auto schema_json =
            R"({
                "name": "Names",
                "fields": [
                    {"name": "name", "type": "string", "optional": true},
                    {"name": "names", "type": "string[]", "optional": true}
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
    for (auto const &json: documents) {
        auto add_op = collection_create_op.get()->add(json.dump());
        ASSERT_TRUE(add_op.ok());
    }

    std::map<std::string, std::string> req_params = {
            {"collection", "Names"},
            {"q", "*"},
            {"query_by", "name"},
            {"filter_by", "name:= S*"}
    };
    nlohmann::json embedded_params;
    std::string json_res;
    auto now_ts = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

    auto search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    auto res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(1, res_obj["found"].get<size_t>());
    ASSERT_EQ(1, res_obj["hits"].size());
    ASSERT_EQ("Steve Jobs", res_obj["hits"][0]["document"].at("name"));

    req_params = {
            {"collection", "Names"},
            {"q", "*"},
            {"query_by", "name"},
            {"filter_by", "name: S*"}
    };

    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(2, res_obj["found"].get<size_t>());
    ASSERT_EQ(2, res_obj["hits"].size());
    ASSERT_EQ("Adam Stator", res_obj["hits"][0]["document"].at("name"));
    ASSERT_EQ("Steve Jobs", res_obj["hits"][1]["document"].at("name"));

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

    req_params = {
            {"collection", "Names"},
            {"q", "s"},
            {"query_by", "name"},
            {"filter_by", "name:= St*"}
    };

    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(4, res_obj["found"].get<size_t>());
    ASSERT_EQ(4, res_obj["hits"].size());
    ASSERT_EQ("Steve Rogers", res_obj["hits"][0]["document"].at("name"));
    ASSERT_EQ("Storm", res_obj["hits"][1]["document"].at("name"));
    ASSERT_EQ("Steve Reiley", res_obj["hits"][2]["document"].at("name"));
    ASSERT_EQ("Steve Jobs", res_obj["hits"][3]["document"].at("name"));

    req_params = {
            {"collection", "Names"},
            {"q", "s"},
            {"query_by", "name"},
            {"filter_by", "name: St*"}
    };

    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(5, res_obj["found"].get<size_t>());
    ASSERT_EQ(5, res_obj["hits"].size());
    ASSERT_EQ("Steve Rogers", res_obj["hits"][0]["document"].at("name"));
    ASSERT_EQ("Storm", res_obj["hits"][1]["document"].at("name"));
    ASSERT_EQ("Steve Reiley", res_obj["hits"][2]["document"].at("name"));
    ASSERT_EQ("Adam Stator", res_obj["hits"][3]["document"].at("name"));
    ASSERT_EQ("Steve Jobs", res_obj["hits"][4]["document"].at("name"));

    req_params = {
            {"collection", "Names"},
            {"q", "s"},
            {"query_by", "name"},
            {"filter_by", "name:= Steve R*"}
    };

    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(2, res_obj["found"].get<size_t>());
    ASSERT_EQ(2, res_obj["hits"].size());
    ASSERT_EQ("Steve Rogers", res_obj["hits"][0]["document"].at("name"));
    ASSERT_EQ("Steve Reiley", res_obj["hits"][1]["document"].at("name"));

    req_params = {
            {"collection", "Names"},
            {"q", "s"},
            {"query_by", "name"},
            {"filter_by", "name: Steve R*"}
    };

    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(2, res_obj["found"].get<size_t>());
    ASSERT_EQ(2, res_obj["hits"].size());
    ASSERT_EQ("Steve Rogers", res_obj["hits"][0]["document"].at("name"));
    ASSERT_EQ("Steve Reiley", res_obj["hits"][1]["document"].at("name"));

    documents = {
            R"({
                "names": []
            })"_json,
            R"({
                "names": ["Steve Jobs"]
            })"_json,
            R"({
                "names": ["Adam Stator"]
            })"_json
    };

    for (auto const &json: documents) {
        auto add_op = collection_create_op.get()->add(json.dump());
        ASSERT_TRUE(add_op.ok());
    }

    req_params = {
            {"collection", "Names"},
            {"q", "s"},
            {"query_by", "names"},
            {"filter_by", "names:= St*"}
    };

    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(1, res_obj["found"].get<size_t>());
    ASSERT_EQ(1, res_obj["hits"].size());
    ASSERT_EQ("Steve Jobs", res_obj["hits"][0]["document"]["names"][0]);

    req_params = {
            {"collection", "Names"},
            {"q", "s"},
            {"query_by", "names"},
            {"filter_by", "names: St*"}
    };

    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(2, res_obj["found"].get<size_t>());
    ASSERT_EQ(2, res_obj["hits"].size());
    ASSERT_EQ("Adam Stator", res_obj["hits"][0]["document"]["names"][0]);
    ASSERT_EQ("Steve Jobs", res_obj["hits"][1]["document"]["names"][0]);

    documents = {
            R"({
                "names": ["Steve Reiley"]
            })"_json,
            R"({
                "names": ["Storm"]
            })"_json,
            R"({
                "names": ["Adam", "Steve Rogers"]
            })"_json,
    };

    for (auto const &json: documents) {
        auto add_op = collection_create_op.get()->add(json.dump());
        ASSERT_TRUE(add_op.ok());
    }

    req_params = {
            {"collection", "Names"},
            {"q", "s"},
            {"query_by", "names"},
            {"filter_by", "names:= St*"}
    };

    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(4, res_obj["found"].get<size_t>());
    ASSERT_EQ(4, res_obj["hits"].size());
    ASSERT_EQ("Steve Rogers", res_obj["hits"][0]["document"]["names"][1]);
    ASSERT_EQ("Storm", res_obj["hits"][1]["document"]["names"][0]);
    ASSERT_EQ("Steve Reiley", res_obj["hits"][2]["document"]["names"][0]);
    ASSERT_EQ("Steve Jobs", res_obj["hits"][3]["document"]["names"][0]);

    req_params = {
            {"collection", "Names"},
            {"q", "s"},
            {"query_by", "names"},
            {"filter_by", "names: St*"}
    };

    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(5, res_obj["found"].get<size_t>());
    ASSERT_EQ(5, res_obj["hits"].size());
    ASSERT_EQ("Steve Rogers", res_obj["hits"][0]["document"]["names"][1]);
    ASSERT_EQ("Storm", res_obj["hits"][1]["document"]["names"][0]);
    ASSERT_EQ("Steve Reiley", res_obj["hits"][2]["document"]["names"][0]);
    ASSERT_EQ("Adam Stator", res_obj["hits"][3]["document"]["names"][0]);
    ASSERT_EQ("Steve Jobs", res_obj["hits"][4]["document"]["names"][0]);

    req_params = {
            {"collection", "Names"},
            {"q", "s"},
            {"query_by", "names"},
            {"filter_by", "names:= Steve*"}
    };

    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(3, res_obj["found"].get<size_t>());
    ASSERT_EQ(3, res_obj["hits"].size());
    ASSERT_EQ("Steve Rogers", res_obj["hits"][0]["document"]["names"][1]);
    ASSERT_EQ("Steve Reiley", res_obj["hits"][1]["document"]["names"][0]);
    ASSERT_EQ("Steve Jobs", res_obj["hits"][2]["document"]["names"][0]);

    req_params = {
            {"collection", "Names"},
            {"q", "s"},
            {"query_by", "names"},
            {"filter_by", "names: Steve*"}
    };

    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(3, res_obj["found"].get<size_t>());
    ASSERT_EQ(3, res_obj["hits"].size());
    ASSERT_EQ("Steve Rogers", res_obj["hits"][0]["document"]["names"][1]);
    ASSERT_EQ("Steve Reiley", res_obj["hits"][1]["document"]["names"][0]);
    ASSERT_EQ("Steve Jobs", res_obj["hits"][2]["document"]["names"][0]);

    req_params = {
            {"collection", "Names"},
            {"q", "s"},
            {"query_by", "names"},
            {"filter_by", "names:= Steve R*"}
    };

    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(2, res_obj["found"].get<size_t>());
    ASSERT_EQ(2, res_obj["hits"].size());
    ASSERT_EQ("Steve Rogers", res_obj["hits"][0]["document"]["names"][1]);
    ASSERT_EQ("Steve Reiley", res_obj["hits"][1]["document"]["names"][0]);

    req_params = {
            {"collection", "Names"},
            {"q", "s"},
            {"query_by", "names"},
            {"filter_by", "names: Steve R*"}
    };

    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(2, res_obj["found"].get<size_t>());
    ASSERT_EQ(2, res_obj["hits"].size());
    ASSERT_EQ("Steve Rogers", res_obj["hits"][0]["document"]["names"][1]);
    ASSERT_EQ("Steve Reiley", res_obj["hits"][1]["document"]["names"][0]);

    documents = {
            R"({
                "names": ["Steve Runner foo"]
            })"_json,
            R"({
                "names": ["foo Steve Runner"]
            })"_json,
    };

    for (auto const &json: documents) {
        auto add_op = collection_create_op.get()->add(json.dump());
        ASSERT_TRUE(add_op.ok());
    }

    req_params = {
            {"collection", "Names"},
            {"q", "s"},
            {"query_by", "names"},
            {"filter_by", "names:= Steve R*"}
    };

    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(3, res_obj["found"].get<size_t>());
    ASSERT_EQ(3, res_obj["hits"].size());
    ASSERT_EQ("Steve Runner foo", res_obj["hits"][0]["document"]["names"][0]);
    ASSERT_EQ("Steve Rogers", res_obj["hits"][1]["document"]["names"][1]);
    ASSERT_EQ("Steve Reiley", res_obj["hits"][2]["document"]["names"][0]);

    req_params = {
            {"collection", "Names"},
            {"q", "s"},
            {"query_by", "names"},
            {"filter_by", "names: Steve R*"}
    };

    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(4, res_obj["found"].get<size_t>());
    ASSERT_EQ(4, res_obj["hits"].size());
    ASSERT_EQ("foo Steve Runner", res_obj["hits"][0]["document"]["names"][0]);
    ASSERT_EQ("Steve Runner foo", res_obj["hits"][1]["document"]["names"][0]);
    ASSERT_EQ("Steve Rogers", res_obj["hits"][2]["document"]["names"][1]);
    ASSERT_EQ("Steve Reiley", res_obj["hits"][3]["document"]["names"][0]);
}

TEST_F(CollectionFilteringTest, ExactFilterOnLongField) {
    nlohmann::json schema = R"({
         "name": "companies",
         "fields": [
           {"name": "keywords", "type": "string[]"}
         ]
       })"_json;

    auto op = collectionManager.create_collection(schema);
    ASSERT_TRUE(op.ok());

    auto coll = op.get();

    nlohmann::json doc1;
    doc1["id"] = "0";

    std::string arr_value;

    // when value exceeds 128 tokens, we will fail gracefully
    for(size_t i = 0; i < 130; i++) {
        arr_value += "foo" + std::to_string(i) + " ";
    }
    doc1["keywords"] = {arr_value};

    ASSERT_TRUE(coll->add(doc1.dump()).ok());

    auto results = coll->search("*", {}, "keywords:=" + arr_value, {}, {}, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(0, results["hits"].size());
}

TEST_F(CollectionFilteringTest, FilterOnStemmedField) {
    nlohmann::json schema = R"({
         "name": "companies",
         "fields": [
           {"name": "keywords", "type": "string[]", "facet": true, "stem": true }
         ]
       })"_json;

    auto op = collectionManager.create_collection(schema);
    ASSERT_TRUE(op.ok());

    auto coll = op.get();

    nlohmann::json doc1 = {
        {"id", "124"},
        {"keywords", {"Restaurant"}}
    };

    nlohmann::json doc2 = {
        {"id", "125"},
        {"keywords", {"Baking"}}
    };

    ASSERT_TRUE(coll->add(doc1.dump()).ok());
    ASSERT_TRUE(coll->add(doc2.dump()).ok());

    auto results = coll->search("*", {}, "keywords:=Baking", {}, {}, {0}, 10, 1, FREQUENCY, {false}).get();
    ASSERT_EQ(1, results["hits"].size());
    ASSERT_EQ("125", results["hits"][0]["document"]["id"].get<std::string>());

}

TEST_F(CollectionFilteringTest, MaxFilterByCandidates) {
    Collection *coll1;
    std::vector<field> fields = {field("title", field_types::STRING, false),
                                 field("points", field_types::INT32, false)};

    coll1 = collectionManager.get_collection("coll1").get();
    if(coll1 == nullptr) {
        coll1 = collectionManager.create_collection("coll1", 1, fields, "points").get();
    }

    for(size_t i = 0; i < 20; i++) {
        nlohmann::json doc;
        doc["title"] = "Independent" + std::to_string(i);
        doc["points"] = i;
        coll1->add(doc.dump());
    }

    std::map<std::string, std::string> req_params = {
            {"collection", "coll1"},
            {"q", "*"},
            {"filter_by", "title:independent*"},
    };
    nlohmann::json embedded_params;
    std::string json_res;
    auto now_ts = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

    auto search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    auto res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(4, res_obj["found"].get<size_t>());
    ASSERT_EQ(4, res_obj["hits"].size());
    ASSERT_EQ("Independent19", res_obj["hits"][0]["document"]["title"]);
    ASSERT_EQ("Independent18", res_obj["hits"][1]["document"]["title"]);
    ASSERT_EQ("Independent17", res_obj["hits"][2]["document"]["title"]);
    ASSERT_EQ("Independent16", res_obj["hits"][3]["document"]["title"]);

    req_params = {
            {"collection", "coll1"},
            {"q", "*"},
            {"filter_by", "title:independent*"},
            {"max_filter_by_candidates", "0"}
    };

    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(0, res_obj["found"].get<size_t>());
    ASSERT_EQ(0, res_obj["hits"].size());

    req_params = {
            {"collection", "coll1"},
            {"q", "*"},
            {"filter_by", "title:independent*"},
            {"max_filter_by_candidates", "1"}
    };

    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(1, res_obj["found"].get<size_t>());
    ASSERT_EQ(1, res_obj["hits"].size());
    ASSERT_EQ("Independent19", res_obj["hits"][0]["document"]["title"]);
}

TEST_F(CollectionFilteringTest, FilterOnObjectFields) {
    auto schema_json =
            R"({
                "name": "Products",
                "fields": [
                    {"name": "product_id", "type": "string"},
                    {"name": "product_name", "type": "string", "infix": true},
                    {"name": "product_description", "type": "string"},
                    {"name": "embedding", "type":"float[]", "embed":{"from": ["product_description"], "model_config": {"model_name": "ts/e5-small"}}},
                    {"name": "rating", "type": "int32"},
                    {"name": "stocks", "type": "object"},
                    {"name": "stocks.*", "type": "auto", "optional": true}
                ],
                "enable_nested_fields": true
            })"_json;
    std::vector<nlohmann::json> documents = {
            R"({
                "product_id": "product_a",
                "product_name": "shampoo",
                "product_description": "Our new moisturizing shampoo is perfect for those with dry or damaged hair.",
                "rating": "2",
                "stocks": {
                    "26": {
                        "rec": true
                    }
                }
            })"_json,
            R"({
                "product_id": "product_b",
                "product_name": "soap",
                "product_description": "Introducing our all-natural, organic soap bar made with essential oils and botanical ingredients.",
                "rating": "4",
                "stocks": {
                    "26": {
                        "rec": false
                    }
                }
            })"_json,
            R"({
                "product_id": "product_c",
                "product_name": "comb",
                "product_description": "Experience the natural elegance and gentle care of our handcrafted wooden combs – because your hair deserves the best.",
                "rating": "3",
                "stocks": {}
            })"_json
    };

    EmbedderManager::set_model_dir("/tmp/typesense_test/models");

    auto collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_TRUE(collection_create_op.ok());
    for (auto const &json: documents) {
        auto add_op = collection_create_op.get()->add(json.dump());
        if (!add_op.ok()) {
            LOG(INFO) << add_op.error();
        }
        ASSERT_TRUE(add_op.ok());
    }

    std::map<std::string, std::string> req_params = {
            {"collection", "Products"},
            {"q", "*"},
            {"filter_by", "stocks.26.rec:true"},
            {"include_fields", "product_id, product_name, stocks"}
    };
    nlohmann::json embedded_params;
    std::string json_res;
    auto now_ts = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

    auto search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    auto res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(1, res_obj["found"].get<size_t>());
    ASSERT_EQ(1, res_obj["hits"].size());
    ASSERT_EQ("product_a", res_obj["hits"][0]["document"]["product_id"]);
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("stocks"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"]["stocks"].size());
    ASSERT_EQ(1, res_obj["hits"][0]["document"]["stocks"].count("26"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"]["stocks"]["26"].size());
    ASSERT_EQ(1, res_obj["hits"][0]["document"]["stocks"]["26"].count("rec"));
    ASSERT_TRUE(res_obj["hits"][0]["document"]["stocks"]["26"]["rec"]);

    req_params = {
            {"collection", "Products"},
            {"q", "*"},
            {"filter_by", "stocks.26.rec:false"},
            {"include_fields", "product_id, product_name, stocks"}
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(1, res_obj["found"].get<size_t>());
    ASSERT_EQ(1, res_obj["hits"].size());
    ASSERT_EQ("product_b", res_obj["hits"][0]["document"]["product_id"]);
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("stocks"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"]["stocks"].size());
    ASSERT_EQ(1, res_obj["hits"][0]["document"]["stocks"].count("26"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"]["stocks"]["26"].size());
    ASSERT_EQ(1, res_obj["hits"][0]["document"]["stocks"]["26"].count("rec"));
    ASSERT_FALSE(res_obj["hits"][0]["document"]["stocks"]["26"]["rec"]);
}
