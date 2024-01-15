#include <gtest/gtest.h>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <collection_manager.h>
#include "collection.h"

class CollectionFacetingTest : public ::testing::Test {
protected:
    Store *store;
    CollectionManager & collectionManager = CollectionManager::get_instance();
    std::atomic<bool> quit = false;

    std::vector<std::string> query_fields;
    std::vector<sort_by> sort_fields;

    void setupCollection() {
        std::string state_dir_path = "/tmp/typesense_test/collection_faceting";
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

TEST_F(CollectionFacetingTest, FacetCounts) {
    Collection *coll_array_fields;

    std::ifstream infile(std::string(ROOT_DIR)+"test/numeric_array_documents.jsonl");
    std::vector<field> fields = {field("name", field_types::STRING, false),
                                 field("name_facet", field_types::STRING, true),
                                 field("age", field_types::INT32, true),
                                 field("years", field_types::INT32_ARRAY, true),
                                 field("rating", field_types::FLOAT, true),
                                 field("timestamps", field_types::INT64_ARRAY, true),
                                 field("tags", field_types::STRING_ARRAY, true),
                                 field("optional_facet", field_types::INT64_ARRAY, true, true),};

    std::vector<sort_by> sort_fields = { sort_by("age", "DESC") };

    coll_array_fields = collectionManager.get_collection("coll_array_fields").get();
    if(coll_array_fields == nullptr) {
        coll_array_fields = collectionManager.create_collection("coll_array_fields", 4, fields, "age").get();
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

    // single facet with no filters
    nlohmann::json results = coll_array_fields->search("Jeremy", query_fields, "", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();    
    ASSERT_EQ(5, results["hits"].size());

    ASSERT_EQ(1, results["facet_counts"].size());
    ASSERT_EQ(4, results["facet_counts"][0].size());
    ASSERT_EQ("tags", results["facet_counts"][0]["field_name"]);
    ASSERT_EQ(false, results["facet_counts"][0]["sampled"].get<bool>());
    ASSERT_EQ(4, results["facet_counts"][0]["counts"].size());
    ASSERT_EQ(1, results["facet_counts"][0]["stats"].size());
    ASSERT_EQ(4, results["facet_counts"][0]["stats"]["total_values"].get<size_t>());

    ASSERT_STREQ("gold", results["facet_counts"][0]["counts"][0]["value"].get<std::string>().c_str());
    ASSERT_EQ(3, (int) results["facet_counts"][0]["counts"][0]["count"]);

    ASSERT_STREQ("silver", results["facet_counts"][0]["counts"][1]["value"].get<std::string>().c_str());
    ASSERT_EQ(3, (int) results["facet_counts"][0]["counts"][1]["count"]);

    ASSERT_STREQ("bronze", results["facet_counts"][0]["counts"][2]["value"].get<std::string>().c_str());
    ASSERT_EQ(2, (int) results["facet_counts"][0]["counts"][2]["count"]);

    ASSERT_STREQ("FINE PLATINUM", results["facet_counts"][0]["counts"][3]["value"].get<std::string>().c_str());
    ASSERT_EQ(1, (int) results["facet_counts"][0]["counts"][3]["count"]);

    // facet with facet count limit
    results = coll_array_fields->search("Jeremy", query_fields, "", facets, sort_fields, {0}, 10, 1,
                                        FREQUENCY, {false}, 10, spp::sparse_hash_set<std::string>(),
                                        spp::sparse_hash_set<std::string>(), 2).get();
    
    ASSERT_EQ(5, results["hits"].size());

    ASSERT_EQ(1, results["facet_counts"].size());
    ASSERT_STREQ("tags", results["facet_counts"][0]["field_name"].get<std::string>().c_str());
    ASSERT_EQ(2, results["facet_counts"][0]["counts"].size());

    ASSERT_STREQ("gold", results["facet_counts"][0]["counts"][0]["value"].get<std::string>().c_str());
    ASSERT_EQ(3, (int) results["facet_counts"][0]["counts"][0]["count"]);

    ASSERT_STREQ("silver", results["facet_counts"][0]["counts"][1]["value"].get<std::string>().c_str());
    ASSERT_EQ(3, (int) results["facet_counts"][0]["counts"][1]["count"]);

    // 2 facets, 1 text query with no filters
    facets.clear();
    facets.push_back("tags");
    facets.push_back("name_facet");
    results = coll_array_fields->search("Jeremy", query_fields, "", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();

    ASSERT_EQ(5, results["hits"].size());
    ASSERT_EQ(2, results["facet_counts"].size());

    ASSERT_STREQ("tags", results["facet_counts"][0]["field_name"].get<std::string>().c_str());
    ASSERT_STREQ("name_facet", results["facet_counts"][1]["field_name"].get<std::string>().c_str());

    // facet value must one that's stored, not indexed (i.e. no tokenization/standardization)
    ASSERT_STREQ("Jeremy Howard", results["facet_counts"][1]["counts"][0]["value"].get<std::string>().c_str());
    ASSERT_EQ(5, (int) results["facet_counts"][1]["counts"][0]["count"]);

    // facet with filters
    facets.clear();
    facets.push_back("tags");
    results = coll_array_fields->search("Jeremy", query_fields, "age: >24", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();

    ASSERT_EQ(3, results["hits"].size());
    ASSERT_EQ(1, results["facet_counts"].size());

    ASSERT_STREQ("tags", results["facet_counts"][0]["field_name"].get<std::string>().c_str());
    ASSERT_EQ(2, (int) results["facet_counts"][0]["counts"][0]["count"]);
    ASSERT_EQ(1, (int) results["facet_counts"][0]["counts"][1]["count"]);
    ASSERT_EQ(1, (int) results["facet_counts"][0]["counts"][2]["count"]);
    ASSERT_EQ(1, (int) results["facet_counts"][0]["counts"][3]["count"]);

    ASSERT_STREQ("silver", results["facet_counts"][0]["counts"][0]["value"].get<std::string>().c_str());
    ASSERT_STREQ("gold", results["facet_counts"][0]["counts"][1]["value"].get<std::string>().c_str());
    ASSERT_STREQ("bronze", results["facet_counts"][0]["counts"][2]["value"].get<std::string>().c_str());
    ASSERT_STREQ("FINE PLATINUM", results["facet_counts"][0]["counts"][3]["value"].get<std::string>().c_str());

    // facet with wildcard query
    facets.clear();
    facets.push_back("tags");
    results = coll_array_fields->search("*", query_fields, "age: >24", facets, sort_fields, {0}, 10, 1, FREQUENCY, {false}).get();

    ASSERT_EQ(3, results["hits"].size());
    ASSERT_EQ(1, results["facet_counts"].size());

    ASSERT_STREQ("tags", results["facet_counts"][0]["field_name"].get<std::string>().c_str());
    ASSERT_EQ(2, (int) results["facet_counts"][0]["counts"][0]["count"]);
    ASSERT_EQ(1, (int) results["facet_counts"][0]["counts"][1]["count"]);
    ASSERT_EQ(1, (int) results["facet_counts"][0]["counts"][2]["count"]);
    ASSERT_EQ(1, (int) results["facet_counts"][0]["counts"][3]["count"]);

    ASSERT_STREQ("silver", results["facet_counts"][0]["counts"][0]["value"].get<std::string>().c_str());
    ASSERT_STREQ("gold", results["facet_counts"][0]["counts"][1]["value"].get<std::string>().c_str());
    ASSERT_STREQ("bronze", results["facet_counts"][0]["counts"][2]["value"].get<std::string>().c_str());
    ASSERT_STREQ("FINE PLATINUM", results["facet_counts"][0]["counts"][3]["value"].get<std::string>().c_str());

    // facet with facet filter query (allows typo correction!)
    results = coll_array_fields->search("*", query_fields, "", facets, sort_fields, {0}, 10, 1, FREQUENCY,
                                        {false}, Index::DROP_TOKENS_THRESHOLD,
                                        spp::sparse_hash_set<std::string>(),
                                        spp::sparse_hash_set<std::string>(), 10, " tags : sliver").get();

    ASSERT_EQ(5, results["hits"].size());
    ASSERT_EQ(1, results["facet_counts"].size());
    ASSERT_STREQ("tags", results["facet_counts"][0]["field_name"].get<std::string>().c_str());
    ASSERT_EQ(3, (int) results["facet_counts"][0]["counts"][0]["count"]);
    ASSERT_STREQ("silver", results["facet_counts"][0]["counts"][0]["value"].get<std::string>().c_str());

    // facet with facet filter query matching 2 tokens
    results = coll_array_fields->search("*", query_fields, "", facets, sort_fields, {0}, 10, 1, FREQUENCY,
                                        {false}, Index::DROP_TOKENS_THRESHOLD,
                                        spp::sparse_hash_set<std::string>(),
                                        spp::sparse_hash_set<std::string>(), 10, "tags: fxne platim").get();

    ASSERT_EQ(5, results["hits"].size());
    ASSERT_EQ(1, results["facet_counts"].size());
    ASSERT_STREQ("tags", results["facet_counts"][0]["field_name"].get<std::string>().c_str());
    ASSERT_EQ(1, (int) results["facet_counts"][0]["counts"][0]["count"]);
    ASSERT_STREQ("FINE PLATINUM", results["facet_counts"][0]["counts"][0]["value"].get<std::string>().c_str());
    ASSERT_STREQ("<mark>FINE</mark> <mark>PLATIN</mark>UM", results["facet_counts"][0]["counts"][0]["highlighted"].get<std::string>().c_str());

    // facet with facet filter query matching first token of an array
    results = coll_array_fields->search("*", query_fields, "", facets, sort_fields, {0}, 10, 1, FREQUENCY,
                                        {false}, Index::DROP_TOKENS_THRESHOLD,
                                        spp::sparse_hash_set<std::string>(),
                                        spp::sparse_hash_set<std::string>(), 10, "tags: fine").get();

    ASSERT_EQ(5, results["hits"].size());
    ASSERT_EQ(1, results["facet_counts"].size());
    ASSERT_STREQ("tags", results["facet_counts"][0]["field_name"].get<std::string>().c_str());
    ASSERT_EQ(1, (int) results["facet_counts"][0]["counts"][0]["count"]);
    ASSERT_STREQ("FINE PLATINUM", results["facet_counts"][0]["counts"][0]["value"].get<std::string>().c_str());

    // facet with facet filter query matching second token of an array
    results = coll_array_fields->search("*", query_fields, "", facets, sort_fields, {0}, 10, 1, FREQUENCY,
                                        {false}, Index::DROP_TOKENS_THRESHOLD,
                                        spp::sparse_hash_set<std::string>(),
                                        spp::sparse_hash_set<std::string>(), 10, "tags: pltinum").get();

    ASSERT_EQ(5, results["hits"].size());
    ASSERT_EQ(1, results["facet_counts"].size());
    ASSERT_STREQ("tags", results["facet_counts"][0]["field_name"].get<std::string>().c_str());
    ASSERT_EQ(1, (int) results["facet_counts"][0]["counts"][0]["count"]);
    ASSERT_STREQ("FINE PLATINUM", results["facet_counts"][0]["counts"][0]["value"].get<std::string>().c_str());

    // facet with wildcard
    results = coll_array_fields->search("Jeremy", query_fields, "", {"ag*"}, sort_fields, {0}, 10, 1, FREQUENCY,
                                        {false}).get();
    ASSERT_EQ(5, results["hits"].size());
    ASSERT_EQ(1, results["facet_counts"].size());
    ASSERT_STREQ("age", results["facet_counts"][0]["field_name"].get<std::string>().c_str());

    // facet query on an integer field
    results = coll_array_fields->search("*", query_fields, "", {"age"}, sort_fields, {0}, 10, 1, FREQUENCY,
                                        {false}, Index::DROP_TOKENS_THRESHOLD,
                                        spp::sparse_hash_set<std::string>(),
                                        spp::sparse_hash_set<std::string>(), 10, "age: 2").get();

    ASSERT_EQ(5, results["hits"].size());
    ASSERT_EQ(1, results["facet_counts"].size());
    ASSERT_STREQ("age", results["facet_counts"][0]["field_name"].get<std::string>().c_str());

    ASSERT_EQ(1, (int) results["facet_counts"][0]["counts"][0]["count"]);
    ASSERT_STREQ("24", results["facet_counts"][0]["counts"][0]["value"].get<std::string>().c_str());
    ASSERT_STREQ("<mark>2</mark>4", results["facet_counts"][0]["counts"][0]["highlighted"].get<std::string>().c_str());

    ASSERT_EQ(1, (int) results["facet_counts"][0]["counts"][1]["count"]);
    ASSERT_STREQ("21", results["facet_counts"][0]["counts"][1]["value"].get<std::string>().c_str());
    ASSERT_STREQ("<mark>2</mark>1", results["facet_counts"][0]["counts"][1]["highlighted"].get<std::string>().c_str());

    // facet on a float field without query to check on stats
    results = coll_array_fields->search("*", query_fields, "", {"rating"}, sort_fields, {0}, 10, 1, FREQUENCY,
                                        {false}, Index::DROP_TOKENS_THRESHOLD,
                                        spp::sparse_hash_set<std::string>(),
                                        spp::sparse_hash_set<std::string>(), 10, "").get();
    ASSERT_EQ(5, results["facet_counts"][0]["stats"].size());
    ASSERT_FLOAT_EQ(4.880199885368347, results["facet_counts"][0]["stats"]["avg"].get<double>());
    ASSERT_FLOAT_EQ(0.0, results["facet_counts"][0]["stats"]["min"].get<double>());
    ASSERT_FLOAT_EQ(9.99899959564209, results["facet_counts"][0]["stats"]["max"].get<double>());
    ASSERT_FLOAT_EQ(24.400999426841736, results["facet_counts"][0]["stats"]["sum"].get<double>());
    ASSERT_FLOAT_EQ(5, results["facet_counts"][0]["stats"]["total_values"].get<size_t>());

    // check for "0" case
    ASSERT_STREQ("0", results["facet_counts"][0]["counts"][0]["value"].get<std::string>().c_str());
    ASSERT_EQ(1, results["facet_counts"][0]["counts"][0]["count"].get<size_t>());

    // facet query on a float field
    results = coll_array_fields->search("*", query_fields, "", {"rating"}, sort_fields, {0}, 10, 1, FREQUENCY,
                                        {false}, Index::DROP_TOKENS_THRESHOLD,
                                        spp::sparse_hash_set<std::string>(),
                                        spp::sparse_hash_set<std::string>(), 10, "rating: 7").get();
    
    ASSERT_EQ(5, results["hits"].size());
    ASSERT_EQ(1, results["facet_counts"].size());
    ASSERT_STREQ("rating", results["facet_counts"][0]["field_name"].get<std::string>().c_str());
    ASSERT_EQ(1, (int) results["facet_counts"][0]["counts"][0]["count"]);
    ASSERT_STREQ("7.812", results["facet_counts"][0]["counts"][0]["value"].get<std::string>().c_str());
    ASSERT_STREQ("<mark>7</mark>.812", results["facet_counts"][0]["counts"][0]["highlighted"].get<std::string>().c_str());

    ASSERT_EQ(5, results["facet_counts"][0]["stats"].size());
    ASSERT_FLOAT_EQ(4.880199885368347, results["facet_counts"][0]["stats"]["avg"].get<double>());
    ASSERT_FLOAT_EQ(0.0, results["facet_counts"][0]["stats"]["min"].get<double>());
    ASSERT_FLOAT_EQ(9.99899959564209, results["facet_counts"][0]["stats"]["max"].get<double>());
    ASSERT_FLOAT_EQ(24.400999426841736, results["facet_counts"][0]["stats"]["sum"].get<double>());
    ASSERT_FLOAT_EQ(1, results["facet_counts"][0]["stats"]["total_values"].get<size_t>());

    // facet query on an array integer field

    results = coll_array_fields->search("*", query_fields, "", {"timestamps"}, sort_fields, {0}, 10, 1, FREQUENCY,
                                        {false}, Index::DROP_TOKENS_THRESHOLD,
                                        spp::sparse_hash_set<std::string>(),
                                        spp::sparse_hash_set<std::string>(), 10, "timestamps: 142189002").get();    
    ASSERT_EQ(5, results["hits"].size());
    ASSERT_EQ(1, results["facet_counts"].size());
    ASSERT_EQ(1, results["facet_counts"][0]["counts"].size());
    ASSERT_STREQ("timestamps", results["facet_counts"][0]["field_name"].get<std::string>().c_str());
    ASSERT_EQ(2, (int) results["facet_counts"][0]["counts"][0]["count"]);
    ASSERT_STREQ("1421890022", results["facet_counts"][0]["counts"][0]["value"].get<std::string>().c_str());
    ASSERT_STREQ("<mark>142189002</mark>2", results["facet_counts"][0]["counts"][0]["highlighted"].get<std::string>().c_str());

    ASSERT_EQ(5, results["facet_counts"][0]["stats"].size());
    ASSERT_FLOAT_EQ(1106321222, results["facet_counts"][0]["stats"]["avg"].get<double>());
    ASSERT_FLOAT_EQ(348974822, results["facet_counts"][0]["stats"]["min"].get<double>());
    ASSERT_FLOAT_EQ(1453426022, results["facet_counts"][0]["stats"]["max"].get<double>());
    ASSERT_FLOAT_EQ(13275854664, results["facet_counts"][0]["stats"]["sum"].get<double>());
    ASSERT_FLOAT_EQ(1, results["facet_counts"][0]["stats"]["total_values"].get<size_t>());

    // facet query that does not match any indexed value
    results = coll_array_fields->search("*", query_fields, "", {facets}, sort_fields, {0}, 10, 1, FREQUENCY,
                                        {false}, Index::DROP_TOKENS_THRESHOLD,
                                        spp::sparse_hash_set<std::string>(),
                                        spp::sparse_hash_set<std::string>(), 10, " tags : notfound").get();

    ASSERT_EQ(5, results["hits"].size());
    ASSERT_EQ(1, results["facet_counts"].size());
    ASSERT_STREQ("tags", results["facet_counts"][0]["field_name"].get<std::string>().c_str());
    ASSERT_EQ(0, results["facet_counts"][0]["counts"].size());

    // empty facet query value should return all facets without any filtering of facets
    results = coll_array_fields->search("*", query_fields, "", facets, sort_fields, {0}, 10, 1, FREQUENCY,
                                       {false}, Index::DROP_TOKENS_THRESHOLD,
                                       spp::sparse_hash_set<std::string>(),
                                       spp::sparse_hash_set<std::string>(), 10, "tags: ").get();

    ASSERT_EQ(5, results["hits"].size());

    results = coll_array_fields->search("*", query_fields, "", facets, sort_fields, {0}, 10, 1, FREQUENCY,
                                       {false}, Index::DROP_TOKENS_THRESHOLD,
                                       spp::sparse_hash_set<std::string>(),
                                       spp::sparse_hash_set<std::string>(), 10, "tags:").get();

    ASSERT_EQ(5, results["hits"].size());

    // Wildcard facet_by can have partial matches
    results = coll_array_fields->search("*", query_fields, "", {"nam*"}, sort_fields, {0}, 10, 1, FREQUENCY,
                                        {false}, Index::DROP_TOKENS_THRESHOLD,
                                        spp::sparse_hash_set<std::string>(),
                                        spp::sparse_hash_set<std::string>(), 10).get();
    ASSERT_EQ(5, results["hits"].size());
    ASSERT_EQ(1, results["facet_counts"].size());
    ASSERT_EQ("name_facet", results["facet_counts"][0]["field_name"].get<std::string>());

    // Wildcard facet_by having no counts should not be returned
    results = coll_array_fields->search("*", query_fields, "", {"optio*"}, sort_fields, {0}, 10, 1, FREQUENCY,
                                        {false}, Index::DROP_TOKENS_THRESHOLD,
                                        spp::sparse_hash_set<std::string>(),
                                        spp::sparse_hash_set<std::string>(), 10).get();
    ASSERT_EQ(5, results["hits"].size());
    ASSERT_EQ(0, results["facet_counts"].size());

    results = coll_array_fields->search("*", query_fields, "", {"optional_facet"}, sort_fields, {0}, 10, 1, FREQUENCY,
                                        {false}, Index::DROP_TOKENS_THRESHOLD,
                                        spp::sparse_hash_set<std::string>(),
                                        spp::sparse_hash_set<std::string>(), 10).get();
    ASSERT_EQ(5, results["hits"].size());
    ASSERT_EQ(1, results["facet_counts"].size());
    ASSERT_EQ("optional_facet", results["facet_counts"][0]["field_name"].get<std::string>());

    // bad facet query syntax
    auto res_op = coll_array_fields->search("*", query_fields, "", facets, sort_fields, {0}, 10, 1, FREQUENCY,
                                            {false}, Index::DROP_TOKENS_THRESHOLD,
                                            spp::sparse_hash_set<std::string>(),
                                            spp::sparse_hash_set<std::string>(), 10, "foobar");

    ASSERT_FALSE(res_op.ok());
    ASSERT_STREQ("Facet query must be in the `facet_field: value` format.", res_op.error().c_str());

    // unknown facet field
    res_op = coll_array_fields->search("*", query_fields, "", {"foobar"}, sort_fields, {0}, 10, 1, FREQUENCY,
                                       {false}, Index::DROP_TOKENS_THRESHOLD,
                                       spp::sparse_hash_set<std::string>(),
                                       spp::sparse_hash_set<std::string>(), 10, "foobar: baz");

    ASSERT_FALSE(res_op.ok());
    ASSERT_STREQ("Could not find a facet field named `foobar` in the schema.", res_op.error().c_str());

    // only prefix matching is valid
    res_op = coll_array_fields->search("*", query_fields, "", {"*_facet"}, sort_fields, {0}, 10, 1, FREQUENCY,
                                       {false}, Index::DROP_TOKENS_THRESHOLD,
                                       spp::sparse_hash_set<std::string>(),
                                       spp::sparse_hash_set<std::string>(), 10);

    ASSERT_FALSE(res_op.ok());
    ASSERT_STREQ("Only prefix matching with a wildcard is allowed.", res_op.error().c_str());

    // unknown wildcard facet field
    res_op = coll_array_fields->search("*", query_fields, "", {"foo*"}, sort_fields, {0}, 10, 1, FREQUENCY,
                                       {false}, Index::DROP_TOKENS_THRESHOLD,
                                       spp::sparse_hash_set<std::string>(),
                                       spp::sparse_hash_set<std::string>(), 10);

    ASSERT_FALSE(res_op.ok());
    ASSERT_STREQ("Could not find a facet field for `foo*` in the schema.", res_op.error().c_str());

    // when facet query is given but no facet fields are specified, must return an error message
    res_op = coll_array_fields->search("*", query_fields, "", {}, sort_fields, {0}, 10, 1, FREQUENCY,
                                       {false}, Index::DROP_TOKENS_THRESHOLD,
                                       spp::sparse_hash_set<std::string>(),
                                       spp::sparse_hash_set<std::string>(), 10, "tags: foo");

    ASSERT_FALSE(res_op.ok());
    ASSERT_STREQ("The `facet_query` parameter is supplied without a `facet_by` parameter.", res_op.error().c_str());

    res_op = coll_array_fields->search("*", query_fields, "", {""}, sort_fields, {0}, 10, 1, FREQUENCY,
                                       {false}, Index::DROP_TOKENS_THRESHOLD,
                                       spp::sparse_hash_set<std::string>(),
                                       spp::sparse_hash_set<std::string>(), 10, "tags: foo");

    ASSERT_FALSE(res_op.ok());
    ASSERT_STREQ("Could not find a facet field named `` in the schema.", res_op.error().c_str());

    // given facet query field must be part of facet fields requested
    res_op = coll_array_fields->search("*", query_fields, "", facets, sort_fields, {0}, 10, 1, FREQUENCY,
                                       {false}, Index::DROP_TOKENS_THRESHOLD,
                                       spp::sparse_hash_set<std::string>(),
                                       spp::sparse_hash_set<std::string>(), 10, "name_facet: jeremy");

    ASSERT_FALSE(res_op.ok());
    ASSERT_STREQ("Facet query refers to a facet field `name_facet` that is not part of `facet_by` parameter.", res_op.error().c_str());

    // facet query with multiple colons should be fine (only first colon will be treate as separator)
    res_op = coll_array_fields->search("*", query_fields, "", facets, sort_fields, {0}, 10, 1, FREQUENCY,
                                       {false}, Index::DROP_TOKENS_THRESHOLD,
                                       spp::sparse_hash_set<std::string>(),
                                       spp::sparse_hash_set<std::string>(), 10, "tags:foo:bar");

    ASSERT_TRUE(res_op.ok());

    collectionManager.drop_collection("coll_array_fields");
}

TEST_F(CollectionFacetingTest, FacetCountsBool) {
    Collection *coll1;

    std::vector<field> fields = {field("title", field_types::STRING, false),
                                 field("points", field_types::INT32, false),
                                 field("in_stock", field_types::BOOL, true)};

    std::vector<sort_by> sort_fields = {sort_by("points", "DESC")};

    coll1 = collectionManager.get_collection("coll1").get();
    if (coll1 == nullptr) {
        coll1 = collectionManager.create_collection("coll1", 4, fields, "points").get();
    }

    nlohmann::json doc;
    doc["id"] = "100";
    doc["title"] = "Ford Mustang";
    doc["points"] = 25;
    doc["in_stock"] = true;

    coll1->add(doc.dump());

    doc["id"] = "101";
    doc["title"] = "Tesla Model S";
    doc["points"] = 40;
    doc["in_stock"] = false;

    coll1->add(doc.dump());

    doc["id"] = "102";
    doc["title"] = "Chevrolet Beat";
    doc["points"] = 10;
    doc["in_stock"] = true;

    coll1->add(doc.dump());

    std::vector<std::string> facets = {"in_stock"};

    nlohmann::json results = coll1->search("*", {"title"}, "in_stock:true", facets, sort_fields, {0}, 10, 1,
                                           token_ordering::FREQUENCY, {true}, 10, spp::sparse_hash_set<std::string>(),
                                           spp::sparse_hash_set<std::string>(), 10, "in_stock:true").get();

    ASSERT_EQ(1, results["facet_counts"].size());
    ASSERT_EQ(1, results["facet_counts"][0]["counts"].size());
    ASSERT_EQ(1, results["facet_counts"][0]["stats"].size());
    ASSERT_FLOAT_EQ(1, results["facet_counts"][0]["stats"]["total_values"].get<size_t>());

    ASSERT_STREQ("in_stock", results["facet_counts"][0]["field_name"].get<std::string>().c_str());
    ASSERT_EQ(2, (int) results["facet_counts"][0]["counts"][0]["count"]);
    ASSERT_STREQ("true", results["facet_counts"][0]["counts"][0]["value"].get<std::string>().c_str());
    ASSERT_STREQ("<mark>true</mark>",
                 results["facet_counts"][0]["counts"][0]["highlighted"].get<std::string>().c_str());

    collectionManager.drop_collection("coll1");
}

TEST_F(CollectionFacetingTest, FacetCountsFloatPrecision) {
    Collection *coll1;

    std::vector<field> fields = {field("title", field_types::STRING, false),
                                 field("points", field_types::FLOAT, true)};

    std::vector<sort_by> sort_fields = {sort_by("points", "DESC")};

    coll1 = collectionManager.get_collection("coll1").get();
    if (coll1 == nullptr) {
        coll1 = collectionManager.create_collection("coll1", 4, fields, "points").get();
    }

    nlohmann::json doc;
    doc["id"] = "100";
    doc["title"] = "Ford Mustang";
    doc["points"] = 113.4;

    coll1->add(doc.dump());

    std::vector<std::string> facets = {"points"};

    nlohmann::json results = coll1->search("*", {"title"}, "", facets, sort_fields, {0}, 10, 1,
                                           token_ordering::FREQUENCY, {true}).get();

    ASSERT_EQ(1, results["facet_counts"].size());
    ASSERT_EQ(1, results["facet_counts"][0]["counts"].size());

    ASSERT_STREQ("points", results["facet_counts"][0]["field_name"].get<std::string>().c_str());
    ASSERT_EQ(1, (int) results["facet_counts"][0]["counts"][0]["count"]);
    ASSERT_STREQ("113.4", results["facet_counts"][0]["counts"][0]["value"].get<std::string>().c_str());
    ASSERT_STREQ("113.4",results["facet_counts"][0]["counts"][0]["highlighted"].get<std::string>().c_str());

    collectionManager.drop_collection("coll1");
}

TEST_F(CollectionFacetingTest, FacetCountsHighlighting) {
    Collection *coll1;

    std::vector<field> fields = {field("categories", field_types::STRING_ARRAY, true),
                                 field("points", field_types::INT32, false)};

    std::vector<sort_by> sort_fields = { sort_by("points", "DESC") };

    coll1 = collectionManager.get_collection("coll1").get();
    if(coll1 == nullptr) {
        coll1 = collectionManager.create_collection("coll1", 4, fields, "points").get();
    }

    nlohmann::json doc;
    doc["id"] = "100";
    doc["categories"] = {"Cell Phones", "Cell Phone Accessories", "Cell Phone Cases & Clips"};
    doc["points"] = 25;

    coll1->add(doc.dump());

    std::vector<std::string> facets = {"categories"};

    nlohmann::json results = coll1->search("phone", {"categories"}, "", facets, sort_fields, {0}, 10, 1,
                                           token_ordering::FREQUENCY, {true}, 10, spp::sparse_hash_set<std::string>(),
                                           spp::sparse_hash_set<std::string>(), 10, "categories:cell").get();

    ASSERT_EQ(1, results["facet_counts"].size());
    ASSERT_EQ(3, results["facet_counts"][0]["counts"].size());

    ASSERT_STREQ("categories", results["facet_counts"][0]["field_name"].get<std::string>().c_str());

    ASSERT_EQ(1, results["facet_counts"][0]["counts"][0]["count"].get<size_t>());
    ASSERT_STREQ("Cell Phones", results["facet_counts"][0]["counts"][0]["value"].get<std::string>().c_str());
    ASSERT_STREQ("<mark>Cell</mark> Phones", results["facet_counts"][0]["counts"][0]["highlighted"].get<std::string>().c_str());

    ASSERT_EQ(1, results["facet_counts"][0]["counts"][1]["count"].get<size_t>());
    ASSERT_STREQ("Cell Phone Accessories", results["facet_counts"][0]["counts"][1]["value"].get<std::string>().c_str());
    ASSERT_STREQ("<mark>Cell</mark> Phone Accessories", results["facet_counts"][0]["counts"][1]["highlighted"].get<std::string>().c_str());

    ASSERT_EQ(1, results["facet_counts"][0]["counts"][2]["count"].get<size_t>());
    ASSERT_STREQ("Cell Phone Cases & Clips", results["facet_counts"][0]["counts"][2]["value"].get<std::string>().c_str());
    ASSERT_STREQ("<mark>Cell</mark> Phone Cases & Clips", results["facet_counts"][0]["counts"][2]["highlighted"].get<std::string>().c_str());

    coll1->remove("100");

    doc["categories"] = {"Cell Phones", "Unlocked Cell Phones", "All Unlocked Cell Phones" };
    coll1->add(doc.dump());

    results = coll1->search("phone", {"categories"}, "", facets, sort_fields, {0}, 10, 1,
                            token_ordering::FREQUENCY, {true}, 10, spp::sparse_hash_set<std::string>(),
                            spp::sparse_hash_set<std::string>(), 10, "categories:cell").get();

    ASSERT_EQ(1, results["facet_counts"].size());
    ASSERT_STREQ("categories", results["facet_counts"][0]["field_name"].get<std::string>().c_str());

    ASSERT_EQ(1, (int) results["facet_counts"][0]["counts"][0]["count"]);
    ASSERT_STREQ("Cell Phones", results["facet_counts"][0]["counts"][0]["value"].get<std::string>().c_str());
    ASSERT_STREQ("<mark>Cell</mark> Phones", results["facet_counts"][0]["counts"][0]["highlighted"].get<std::string>().c_str());

    ASSERT_EQ(1, (int) results["facet_counts"][0]["counts"][1]["count"]);
    ASSERT_STREQ("Unlocked Cell Phones", results["facet_counts"][0]["counts"][1]["value"].get<std::string>().c_str());
    ASSERT_STREQ("Unlocked <mark>Cell</mark> Phones", results["facet_counts"][0]["counts"][1]["highlighted"].get<std::string>().c_str());

    ASSERT_EQ(1, (int) results["facet_counts"][0]["counts"][2]["count"]);
    ASSERT_STREQ("All Unlocked Cell Phones", results["facet_counts"][0]["counts"][2]["value"].get<std::string>().c_str());
    ASSERT_STREQ("All Unlocked <mark>Cell</mark> Phones", results["facet_counts"][0]["counts"][2]["highlighted"].get<std::string>().c_str());

    coll1->remove("100");
    doc["categories"] = {"Cell Phones", "Cell Phone Accessories", "Cell Phone Cases & Clips"};
    coll1->add(doc.dump());

    results = coll1->search("phone", {"categories"}, "", facets, sort_fields, {0}, 10, 1,
                            token_ordering::FREQUENCY, {true}, 10, spp::sparse_hash_set<std::string>(),
                            spp::sparse_hash_set<std::string>(), 10, "categories:acces").get();


    ASSERT_EQ(1, results["facet_counts"].size());
    ASSERT_EQ(1, results["facet_counts"][0]["counts"].size());
    ASSERT_STREQ("categories", results["facet_counts"][0]["field_name"].get<std::string>().c_str());
    ASSERT_EQ(1, (int) results["facet_counts"][0]["counts"][0]["count"]);
    ASSERT_STREQ("Cell Phone Accessories", results["facet_counts"][0]["counts"][0]["value"].get<std::string>().c_str());
    ASSERT_STREQ("Cell Phone <mark>Acces</mark>sories", results["facet_counts"][0]["counts"][0]["highlighted"].get<std::string>().c_str());

    // ensure that query is NOT case sensitive

    results = coll1->search("phone", {"categories"}, "", facets, sort_fields, {0}, 10, 1,
                            token_ordering::FREQUENCY, {true}, 10, spp::sparse_hash_set<std::string>(),
                            spp::sparse_hash_set<std::string>(), 10, "categories:ACCES").get();

    ASSERT_EQ(1, results["facet_counts"].size());
    ASSERT_EQ(1, results["facet_counts"][0]["counts"].size());
    ASSERT_STREQ("categories", results["facet_counts"][0]["field_name"].get<std::string>().c_str());
    ASSERT_EQ(1, (int) results["facet_counts"][0]["counts"][0]["count"]);
    ASSERT_STREQ("Cell Phone Accessories", results["facet_counts"][0]["counts"][0]["value"].get<std::string>().c_str());
    ASSERT_STREQ("Cell Phone <mark>Acces</mark>sories", results["facet_counts"][0]["counts"][0]["highlighted"].get<std::string>().c_str());

    // ensure that only the last token is treated as prefix search
    coll1->remove("100");
    doc["categories"] = {"Cell Phones", "Cell Phone Accessories", "Cellophanes"};
    coll1->add(doc.dump());

    results = coll1->search("phone", {"categories"}, "", facets, sort_fields, {0}, 10, 1,
                            token_ordering::FREQUENCY, {true}, 10, spp::sparse_hash_set<std::string>(),
                            spp::sparse_hash_set<std::string>(), 10, "categories:cell ph").get();

    ASSERT_EQ(1, results["facet_counts"].size());
    ASSERT_EQ(2, results["facet_counts"][0]["counts"].size());

    ASSERT_STREQ("Cell Phones", results["facet_counts"][0]["counts"][0]["value"].get<std::string>().c_str());
    ASSERT_STREQ("Cell Phone Accessories", results["facet_counts"][0]["counts"][1]["value"].get<std::string>().c_str());

    // facet query longer than a token is correctly matched with typo tolerance
    // also ensure that setting per_page = 0 works fine

    results = coll1->search("phone", {"categories"}, "", facets, sort_fields, {0}, 0, 1,
                            token_ordering::FREQUENCY, {true}, 10, spp::sparse_hash_set<std::string>(),
                            spp::sparse_hash_set<std::string>(), 10, "categories:cellx").get();

    ASSERT_EQ(0, results["hits"].size());
    ASSERT_EQ(1, results["found"].get<uint32_t>());

    ASSERT_EQ(1, results["facet_counts"].size());
    ASSERT_EQ(3, results["facet_counts"][0]["counts"].size());

    ASSERT_STREQ("<mark>Cello</mark>phanes", results["facet_counts"][0]["counts"][0]["highlighted"].get<std::string>().c_str());
    ASSERT_STREQ("<mark>Cell</mark> Phones", results["facet_counts"][0]["counts"][1]["highlighted"].get<std::string>().c_str());
    ASSERT_STREQ("<mark>Cell</mark> Phone Accessories", results["facet_counts"][0]["counts"][2]["highlighted"].get<std::string>().c_str());

    collectionManager.drop_collection("coll1");
}

TEST_F(CollectionFacetingTest, FacetStatOnFloatFields) {
    Collection *coll_float_fields;

    std::ifstream infile(std::string(ROOT_DIR)+"test/float_documents.jsonl");
    std::vector<field> fields = {
            field("title", field_types::STRING, false),
            field("score", field_types::FLOAT, false),
            field("average", field_types::FLOAT, true)
    };

    std::vector<sort_by> sort_fields_desc = { sort_by("average", "DESC") };

    coll_float_fields = collectionManager.get_collection("coll_float_fields").get();
    if(coll_float_fields == nullptr) {
        coll_float_fields = collectionManager.create_collection("coll_float_fields", 4, fields, "average").get();
    }

    std::string json_line;

    while (std::getline(infile, json_line)) {
        coll_float_fields->add(json_line);
    }

    infile.close();

    query_fields = {"title"};
    std::vector<std::string> facets;
    auto res_op = coll_float_fields->search("Jeremy", query_fields, "", {"average"}, sort_fields_desc, {0}, 10,
                                            1, FREQUENCY, {false});

    auto results = res_op.get();

    ASSERT_EQ(7, results["hits"].size());

    ASSERT_EQ(5, results["facet_counts"][0]["stats"].size());
    ASSERT_FLOAT_EQ(-21.3799991607666, results["facet_counts"][0]["stats"]["min"].get<double>());
    ASSERT_FLOAT_EQ(300, results["facet_counts"][0]["stats"]["max"].get<double>());
    ASSERT_FLOAT_EQ(277.8160007725237, results["facet_counts"][0]["stats"]["sum"].get<double>());
    ASSERT_FLOAT_EQ(39.68800011036053, results["facet_counts"][0]["stats"]["avg"].get<double>());
    ASSERT_FLOAT_EQ(7, results["facet_counts"][0]["stats"]["total_values"].get<size_t>());

    results = coll_float_fields->search("*", query_fields, "average:>100", {"average"}, sort_fields_desc, {0}, 10,
                                        1, FREQUENCY, {false}).get();

    ASSERT_EQ(5, results["facet_counts"][0]["stats"].size());
    ASSERT_FLOAT_EQ(300, results["facet_counts"][0]["stats"]["avg"].get<double>());
    ASSERT_FLOAT_EQ(300, results["facet_counts"][0]["stats"]["min"].get<double>());
    ASSERT_FLOAT_EQ(300, results["facet_counts"][0]["stats"]["max"].get<double>());
    ASSERT_FLOAT_EQ(300, results["facet_counts"][0]["stats"]["sum"].get<double>());
    ASSERT_FLOAT_EQ(1, results["facet_counts"][0]["stats"]["total_values"].get<size_t>());

    // facet filter, though should not be considered when calculating facet stats (except total facet values)

    results = coll_float_fields->search("*", query_fields, "", {"average"}, sort_fields_desc, {0}, 10, 1, FREQUENCY,
                                        {false}, Index::DROP_TOKENS_THRESHOLD,
                                        spp::sparse_hash_set<std::string>(),
                                        spp::sparse_hash_set<std::string>(),
                                        10, "average: 11").get();

    ASSERT_EQ(5, results["facet_counts"][0]["stats"].size());
    ASSERT_FLOAT_EQ(39.68800011036053, results["facet_counts"][0]["stats"]["avg"].get<double>());
    ASSERT_FLOAT_EQ(-21.3799991607666, results["facet_counts"][0]["stats"]["min"].get<double>());
    ASSERT_FLOAT_EQ(300, results["facet_counts"][0]["stats"]["max"].get<double>());
    ASSERT_FLOAT_EQ(277.8160007725237, results["facet_counts"][0]["stats"]["sum"].get<double>());
    ASSERT_FLOAT_EQ(1, results["facet_counts"][0]["stats"]["total_values"].get<size_t>());

    collectionManager.drop_collection("coll_float_fields");
}

TEST_F(CollectionFacetingTest, FacetStatsFloatLon) {
    std::vector<field> fields = {
        field("lon", field_types::FLOAT, true),
    };

    Collection* coll1 = collectionManager.create_collection("coll1", 1, fields).get();

    nlohmann::json doc;
    doc["lon"] = -99.184319;
    ASSERT_TRUE(coll1->add(doc.dump()).ok());

    auto results = coll1->search("*", {},
                                 "", {"lon"}, {}, {2}, 10, 1, FREQUENCY, {true}, 1).get();

    ASSERT_EQ(1, results["facet_counts"].size());
    ASSERT_EQ(5, results["facet_counts"][0]["stats"].size());

    ASSERT_FLOAT_EQ(-99.1843, results["facet_counts"][0]["stats"]["avg"].get<double>());
    ASSERT_FLOAT_EQ(-99.1843, results["facet_counts"][0]["stats"]["min"].get<double>());
    ASSERT_FLOAT_EQ(-99.1843, results["facet_counts"][0]["stats"]["max"].get<double>());
    ASSERT_FLOAT_EQ(-99.1843, results["facet_counts"][0]["stats"]["sum"].get<double>());
    ASSERT_FLOAT_EQ(1, results["facet_counts"][0]["stats"]["total_values"].get<size_t>());
}

TEST_F(CollectionFacetingTest, FacetCountOnSimilarStrings) {
    Collection *coll1;

    std::vector<field> fields = {field("categories", field_types::STRING_ARRAY, true),
                                 field("points", field_types::INT32, true)};

    std::vector<sort_by> sort_fields = {sort_by("points", "DESC")};

    coll1 = collectionManager.get_collection("coll1").get();
    if (coll1 == nullptr) {
        coll1 = collectionManager.create_collection("coll1", 4, fields, "points").get();
    }

    nlohmann::json doc;
    doc["id"] = "100";
    doc["categories"] = {"England in India"};
    doc["points"] = 25;

    coll1->add(doc.dump());

    doc["id"] = "101";
    doc["categories"] = {"India in England"};
    doc["points"] = 50;

    coll1->add(doc.dump());

    std::vector<std::string> facets = {"categories"};

    nlohmann::json results = coll1->search("*", {"categories"}, "points:[25, 50]", facets, sort_fields, {0}, 10, 1,
                                           token_ordering::FREQUENCY, {true}, 10, spp::sparse_hash_set<std::string>(),
                                           spp::sparse_hash_set<std::string>(), 10).get();

    ASSERT_EQ(2, results["hits"].size());
    ASSERT_EQ(2, results["facet_counts"][0]["counts"].size());

    ASSERT_STREQ("India in England", results["facet_counts"][0]["counts"][0]["value"].get<std::string>().c_str());
    ASSERT_STREQ("England in India", results["facet_counts"][0]["counts"][1]["value"].get<std::string>().c_str());

    collectionManager.drop_collection("coll1");
}

TEST_F(CollectionFacetingTest, FacetQueryTest) {
    std::vector<field> fields = {
            field("color", field_types::STRING, true),
    };

    Collection* coll1 = collectionManager.create_collection("coll1", 1, fields).get();
    std::vector<std::string> colors = {"apple red", "azure", "amazon green", "apricot orange",
                                       "blue", "barrel blue", "banana yellow", "ball green", "baikal"};

    for(size_t i = 0; i < 100; i++) {
        nlohmann::json doc;
        doc["color"] = colors[i % colors.size()];
        ASSERT_TRUE(coll1->add(doc.dump()).ok());
    }

    // add colors that DON'T start with "b" to push these up the count list
    for(size_t i = 0; i < 4; i++) {
        nlohmann::json doc;
        doc["color"] = colors[i];
        ASSERT_TRUE(coll1->add(doc.dump()).ok());
    }

    auto results = coll1->search("*", {},
                                 "", {"color"}, {}, {2}, 1, 1, FREQUENCY, {true}, 1, spp::sparse_hash_set<std::string>(),
                                 spp::sparse_hash_set<std::string>(), 5, "color:b", 30, 4, "", 20, {}, {}, {}, 0,
                                 "<mark>", "</mark>", {}, 1000, true, false, true, "", false, 6000 * 1000, 4, 7, fallback,
                                 4, {off}, 3, 3, 2, 2, false, "", true, 0, max_score, 100, 0, 4294967295UL, VALUE).get();


    ASSERT_EQ(1, results["facet_counts"].size());
    ASSERT_EQ(4, results["facet_counts"][0]["counts"].size()); // 4 is default candidate size

    // junk string should produce no facets

    results = coll1->search("*", {},
                            "", {"color"}, {}, {2}, 1, 1, FREQUENCY, {true}, 1, spp::sparse_hash_set<std::string>(),
                            spp::sparse_hash_set<std::string>(), 5, "color:xsda", 30, 4, "", 20, {}, {}, {}, 0,
                            "<mark>", "</mark>", {}, 1000, true, false, true, "", false, 6000 * 1000, 4, 7, fallback,
                            4, {off}, 3, 3, 2, 2, false, "", true, 0, max_score, 100, 0, 4294967295UL, VALUE).get();
    ASSERT_EQ(1, results["facet_counts"].size());
    ASSERT_EQ(0, results["facet_counts"][0]["counts"].size());

    results = coll1->search("*", {},
                            "", {"color"}, {}, {2}, 1, 1, FREQUENCY, {true}, 1, spp::sparse_hash_set<std::string>(),
                            spp::sparse_hash_set<std::string>(), 5, "color:green a", 30, 4, "", 20, {}, {}, {}, 0,
                            "<mark>", "</mark>", {}, 1000, true, false, true, "", false, 6000 * 1000, 4, 7, fallback,
                            4, {off}, 3, 3, 2, 2, false, "", true, 0, max_score, 100, 0, 4294967295UL, VALUE).get();

    ASSERT_EQ(1, results["facet_counts"].size());
    ASSERT_EQ(1, results["facet_counts"][0]["counts"].size());
    ASSERT_EQ("amazon green", results["facet_counts"][0]["counts"][0]["value"]);
    ASSERT_EQ("<mark>a</mark>mazon <mark>green</mark>", results["facet_counts"][0]["counts"][0]["highlighted"]);
}

TEST_F(CollectionFacetingTest, FacetQueryOnStringWithColon) {
    std::vector<field> fields = {field("title", field_types::STRING, true),
                                 field("points", field_types::INT32, false)};

    std::vector<sort_by> sort_fields = {sort_by("points", "DESC")};

    Collection* coll1 = collectionManager.create_collection("coll1", 4, fields, "points").get();

    nlohmann::json doc;
    doc["id"] = "100";
    doc["title"] = "foo:bar";
    doc["points"] = 25;

    ASSERT_TRUE(coll1->add(doc.dump()).ok());

    auto res_op = coll1->search("*", {}, "", {"title"}, sort_fields, {0}, 10, 1,
                                           token_ordering::FREQUENCY, {true}, 10, spp::sparse_hash_set<std::string>(),
                                           spp::sparse_hash_set<std::string>(), 10, "title: foo:ba");

    ASSERT_TRUE(res_op.ok());

    auto results = res_op.get();

    ASSERT_STREQ("foo:bar", results["facet_counts"][0]["counts"][0]["value"].get<std::string>().c_str());
    ASSERT_STREQ("<mark>foo:ba</mark>r", results["facet_counts"][0]["counts"][0]["highlighted"].get<std::string>().c_str());

    results = coll1->search("*", {}, "", {"title"}, sort_fields, {0}, 10, 1,
                            token_ordering::FREQUENCY, {true}, 10, spp::sparse_hash_set<std::string>(),
                            spp::sparse_hash_set<std::string>(), 10, "title: ").get();

    ASSERT_STREQ("foo:bar", results["facet_counts"][0]["counts"][0]["value"].get<std::string>().c_str());
    ASSERT_STREQ("foo:bar", results["facet_counts"][0]["counts"][0]["highlighted"].get<std::string>().c_str());

    results = coll1->search("*", {}, "", {"title"}, sort_fields, {0}, 10, 1,
                            token_ordering::FREQUENCY, {true}, 10, spp::sparse_hash_set<std::string>(),
                            spp::sparse_hash_set<std::string>(), 10, "").get();

    ASSERT_STREQ("foo:bar", results["facet_counts"][0]["counts"][0]["value"].get<std::string>().c_str());
    ASSERT_STREQ("foo:bar", results["facet_counts"][0]["counts"][0]["highlighted"].get<std::string>().c_str());

    collectionManager.drop_collection("coll1");
}

TEST_F(CollectionFacetingTest, FacetQueryOnStringArray) {
    Collection* coll1;

    std::vector<field> fields = {field("title", field_types::STRING, false),
                                 field("genres", field_types::STRING_ARRAY, true)};

    coll1 = collectionManager.get_collection("coll1").get();
    if (coll1 == nullptr) {
        coll1 = collectionManager.create_collection("coll1", 2, fields, "").get();
    }

    nlohmann::json doc1;
    doc1["id"] = "0";
    doc1["title"] = "Song 1";
    doc1["genres"] = {"Country Punk Rock", "Country", "Slow"};

    nlohmann::json doc2;
    doc2["id"] = "1";
    doc2["title"] = "Song 2";
    doc2["genres"] = {"Soft Rock", "Rock", "Electronic"};

    nlohmann::json doc3;
    doc3["id"] = "2";
    doc3["title"] = "Song 3";
    doc3["genres"] = {"Rockabilly", "Metal"};

    nlohmann::json doc4;
    doc4["id"] = "3";
    doc4["title"] = "Song 4";
    doc4["genres"] = {"Pop Rock", "Rock", "Fast"};

    nlohmann::json doc5;
    doc5["id"] = "4";
    doc5["title"] = "Song 5";
    doc5["genres"] = {"Pop", "Rockabilly", "Fast"};

    ASSERT_TRUE(coll1->add(doc1.dump()).ok());
    ASSERT_TRUE(coll1->add(doc2.dump()).ok());
    ASSERT_TRUE(coll1->add(doc3.dump()).ok());
    ASSERT_TRUE(coll1->add(doc4.dump()).ok());
    ASSERT_TRUE(coll1->add(doc5.dump()).ok());

    auto results = coll1->search("*", {}, "", {"genres"}, sort_fields, {0}, 0, 1, FREQUENCY,
                                 {false}, Index::DROP_TOKENS_THRESHOLD,
                                 spp::sparse_hash_set<std::string>(),
                                 spp::sparse_hash_set<std::string>(), 10, "genres: roc").get();

    ASSERT_EQ(1, results["facet_counts"].size());
    ASSERT_EQ(5, results["facet_counts"][0]["counts"].size());

    results = coll1->search("*", {}, "", {"genres"}, sort_fields, {0}, 0, 1, FREQUENCY,
                            {false}, Index::DROP_TOKENS_THRESHOLD,
                            spp::sparse_hash_set<std::string>(),
                            spp::sparse_hash_set<std::string>(), 10, "genres: soft roc").get();

    ASSERT_EQ(1, results["facet_counts"].size());
    ASSERT_EQ(1, results["facet_counts"][0]["counts"].size());

    results = coll1->search("*", {}, "", {"genres"}, sort_fields, {0}, 0, 1, FREQUENCY,
                            {false}, Index::DROP_TOKENS_THRESHOLD,
                            spp::sparse_hash_set<std::string>(),
                            spp::sparse_hash_set<std::string>(), 10, "genres: punk roc").get();

    ASSERT_EQ(1, results["facet_counts"].size());
    ASSERT_EQ(1, results["facet_counts"][0]["counts"].size());
    ASSERT_EQ("Country <mark>Punk</mark> <mark>Roc</mark>k", results["facet_counts"][0]["counts"][0]["highlighted"].get<std::string>());

    results = coll1->search("*", {}, "", {"genres"}, sort_fields, {0}, 0, 1, FREQUENCY,
                            {false}, Index::DROP_TOKENS_THRESHOLD,
                            spp::sparse_hash_set<std::string>(),
                            spp::sparse_hash_set<std::string>(), 10, "genres: country roc").get();

    ASSERT_EQ(1, results["facet_counts"].size());
    ASSERT_EQ(1, results["facet_counts"][0]["counts"].size());
    ASSERT_EQ("<mark>Country</mark> Punk <mark>Roc</mark>k", results["facet_counts"][0]["counts"][0]["highlighted"].get<std::string>());

    // with facet query num typo parameter

    results = coll1->search("*", {}, "", {"genres"}, sort_fields, {0}, 0, 1, FREQUENCY,
                            {false}, Index::DROP_TOKENS_THRESHOLD,
                            spp::sparse_hash_set<std::string>(),
                            spp::sparse_hash_set<std::string>(), 10, "genres: eletronic",
                            30, 4, "title", 20, {}, {}, {}, 0,
                            "<mark>", "</mark>", {}, 1000, true, false, true, "", false, 6000 * 1000, 4, 7, fallback,
                            4, {off}, 32767, 32767, 1).get();

    ASSERT_EQ(1, results["facet_counts"].size());
    ASSERT_EQ(1, results["facet_counts"][0]["counts"].size());
    ASSERT_EQ("<mark>Electroni</mark>c", results["facet_counts"][0]["counts"][0]["highlighted"].get<std::string>());

    results = coll1->search("*", {}, "", {"genres"}, sort_fields, {0}, 0, 1, FREQUENCY,
                            {false}, Index::DROP_TOKENS_THRESHOLD,
                            spp::sparse_hash_set<std::string>(),
                            spp::sparse_hash_set<std::string>(), 10, "genres: eletronic",
                            30, 4, "title", 20, {}, {}, {}, 0,
                            "<mark>", "</mark>", {}, 1000, true, false, true, "", false, 6000 * 1000, 4, 7, fallback,
                            4, {off}, 32767, 32767, 0).get();

    ASSERT_EQ(1, results["facet_counts"].size());
    ASSERT_EQ(0, results["facet_counts"][0]["counts"].size());

    collectionManager.drop_collection("coll1");
}

TEST_F(CollectionFacetingTest, FacetQueryReturnAllCandidates) {
    std::vector<field> fields = {field("title", field_types::STRING, true),
                                 field("points", field_types::INT32, false)};

    std::vector<sort_by> sort_fields = {sort_by("points", "DESC")};

    Collection* coll1 = collectionManager.create_collection("coll1", 4, fields, "points").get();

    std::vector<std::string> titles = {
        "everest", "evergreen", "everlast", "estrange", "energy", "extra"
    };

    for(size_t i=0; i < titles.size(); i++) {
        nlohmann::json doc;
        doc["id"] = std::to_string(i);
        doc["points"] = i;
        doc["title"] = titles[i];
        ASSERT_TRUE(coll1->add(doc.dump()).ok());
    }

    auto res_op = coll1->search("*", {}, "", {"title"}, sort_fields, {0}, 10, 1,
                                token_ordering::FREQUENCY, {true}, 10, spp::sparse_hash_set<std::string>(),
                                spp::sparse_hash_set<std::string>(), 10, "title:e", 30, 5,
                                "", 10, {}, {}, {}, 0,
                                "<mark>", "</mark>", {}, 1000, true, false, true, "", false, 6000 * 1000, 4, 7, fallback,
                                10, {off}, 32767, 32767, 2,
                                false, false);

    ASSERT_TRUE(res_op.ok());

    auto results = res_op.get();
    ASSERT_EQ(6, results["facet_counts"][0]["counts"].size());
    collectionManager.drop_collection("coll1");
}

TEST_F(CollectionFacetingTest, FacetValuesShouldBeNormalized) {
    std::vector<field> fields = {field("brand", field_types::STRING, true),};

    Collection* coll1 = collectionManager.create_collection("coll1", 1, fields).get();

    std::vector<std::vector<std::string>> records = {
        {"BUQU"},
        {"Buqu"},
        {"bu-qu"},
    };

    for(size_t i=0; i<records.size(); i++) {
        nlohmann::json doc;
        doc["id"] = std::to_string(i);
        doc["brand"] = records[i][0];
        ASSERT_TRUE(coll1->add(doc.dump()).ok());
    }

    auto results = coll1->search("*", {},
                                 "", {"brand"}, {}, {2}, 10, 1, FREQUENCY, {true}, 1).get();

    ASSERT_EQ(3, results["hits"].size());
    ASSERT_EQ(1, results["facet_counts"].size());
    ASSERT_EQ(3, results["facet_counts"][0]["counts"].size());

    ASSERT_EQ("Buqu", results["facet_counts"][0]["counts"][0]["value"].get<std::string>());
    ASSERT_EQ("BUQU", results["facet_counts"][0]["counts"][1]["value"].get<std::string>());
    ASSERT_EQ("bu-qu", results["facet_counts"][0]["counts"][2]["value"].get<std::string>());

    collectionManager.drop_collection("coll1");
}

TEST_F(CollectionFacetingTest, FacetArrayValuesShouldBeNormalized) {
    std::vector<field> fields = {field("brands", field_types::STRING_ARRAY, true),};

    Collection* coll1 = collectionManager.create_collection("coll1", 1, fields).get();

    std::vector<std::vector<std::string>> records = {
        {"BUQU", "Buqu", "bu-qu"},
    };

    for(size_t i=0; i<records.size(); i++) {
        nlohmann::json doc;
        doc["id"] = std::to_string(i);
        doc["brands"] = nlohmann::json::array();

        for(auto& str: records[i]) {
            doc["brands"].push_back(str);
        }

        ASSERT_TRUE(coll1->add(doc.dump()).ok());
    }

    auto results = coll1->search("*", {},
                                 "", {"brands"}, {}, {2}, 10, 1, FREQUENCY, {true}, 1).get();

    ASSERT_EQ(1, results["hits"].size());
    ASSERT_EQ(1, results["facet_counts"].size());
    ASSERT_EQ(3, results["facet_counts"][0]["counts"].size());

    ASSERT_EQ("Buqu", results["facet_counts"][0]["counts"][0]["value"].get<std::string>());
    ASSERT_EQ("BUQU", results["facet_counts"][0]["counts"][1]["value"].get<std::string>());
    ASSERT_EQ("bu-qu", results["facet_counts"][0]["counts"][2]["value"].get<std::string>());

    collectionManager.drop_collection("coll1");
}

TEST_F(CollectionFacetingTest, FacetByNestedIntField) {
    nlohmann::json schema = R"({
        "name": "coll1",
        "enable_nested_fields": true,
        "fields": [
          {"name": "details", "type": "object", "optional": false },
          {"name": "company.num_employees", "type": "int32", "optional": false, "facet": true },
          {"name": "companyRank", "type": "int32", "optional": false, "facet": true }
        ]
    })"_json;

    auto op = collectionManager.create_collection(schema);
    ASSERT_TRUE(op.ok());
    Collection* coll1 = op.get();

    auto doc1 = R"({
        "details": {"count": 1000},
        "company": {"num_employees": 2000},
        "companyRank": 100
    })"_json;

    auto doc2 = R"({
        "details": {"count": 2000},
        "company": {"num_employees": 2000},
        "companyRank": 101
    })"_json;

    ASSERT_TRUE(coll1->add(doc1.dump(), CREATE).ok());
    ASSERT_TRUE(coll1->add(doc2.dump(), CREATE).ok());

    std::vector<sort_by> sort_fields = { sort_by("details.count", "ASC") };

    auto results = coll1->search("*", {}, "", {"company.num_employees"}, sort_fields, {0}, 10, 1,
                                 token_ordering::FREQUENCY, {true}, 10, spp::sparse_hash_set<std::string>(),
                                 spp::sparse_hash_set<std::string>(), 10, "", 30, 4).get();

    ASSERT_EQ(2, results["found"].get<size_t>());
    ASSERT_EQ(1, results["facet_counts"].size());
    ASSERT_EQ("company.num_employees", results["facet_counts"][0]["field_name"]);
    ASSERT_EQ(1, results["facet_counts"][0]["counts"].size());
    ASSERT_EQ(2, results["facet_counts"][0]["counts"][0]["count"].get<size_t>());
    ASSERT_EQ("2000", results["facet_counts"][0]["counts"][0]["value"].get<std::string>());

    // Nested wildcard faceting
    std::vector<facet> wildcard_facets;
    coll1->parse_facet("company.*", wildcard_facets);

    ASSERT_EQ(1, wildcard_facets.size());
    ASSERT_EQ("company.num_employees", wildcard_facets[0].field_name);

    wildcard_facets.clear();
    coll1->parse_facet("company*", wildcard_facets);

    ASSERT_EQ(2, wildcard_facets.size());
    ASSERT_EQ("company.num_employees", wildcard_facets[0].field_name);
    ASSERT_EQ("companyRank", wildcard_facets[1].field_name);
}

TEST_F(CollectionFacetingTest, FacetByNestedArrayField) {
    nlohmann::json schema = R"({
        "name": "coll1",
        "enable_nested_fields": true,
        "fields": [
          {"name": "data", "type": "object", "optional": false, "facet": true }
        ]
    })"_json;

    auto op = collectionManager.create_collection(schema);
    ASSERT_TRUE(op.ok());
    Collection* coll1 = op.get();

    auto doc1 = R"({
        "data": {"details": [{"name": "Foo"}, {"name": "Foo"}]}
    })"_json;

    auto doc2 = R"({
        "data": {"details": [{"name": "Foo"}, {"name": "Foo"}]}
    })"_json;

    ASSERT_TRUE(coll1->add(doc1.dump(), CREATE).ok());
    ASSERT_TRUE(coll1->add(doc2.dump(), CREATE).ok());

    auto results = coll1->search("*", {}, "", {"data.details.name"}, {}, {0}, 10, 1,
                                 token_ordering::FREQUENCY, {true}, 10, spp::sparse_hash_set<std::string>(),
                                 spp::sparse_hash_set<std::string>(), 10, "", 30, 4).get();

    ASSERT_EQ(2, results["found"].get<size_t>());
    ASSERT_EQ(1, results["facet_counts"].size());
    ASSERT_EQ("data.details.name", results["facet_counts"][0]["field_name"]);
    ASSERT_EQ(1, results["facet_counts"][0]["counts"].size());
    ASSERT_EQ(2, results["facet_counts"][0]["counts"][0]["count"].get<size_t>());
    ASSERT_EQ("Foo", results["facet_counts"][0]["counts"][0]["value"].get<std::string>());
}

TEST_F(CollectionFacetingTest, FacetByArrayField) {
    nlohmann::json schema = R"({
        "name": "coll1",
        "enable_nested_fields": true,
        "fields": [
          {"name": "data", "type": "string[]", "optional": false, "facet": true }
        ]
    })"_json;

    auto op = collectionManager.create_collection(schema);
    ASSERT_TRUE(op.ok());
    Collection* coll1 = op.get();

    auto doc1 = R"({
        "data": ["Foo", "Foo"]
    })"_json;

    auto doc2 = R"({
        "data": ["Foo", "Foo", "Bazinga"]
    })"_json;

    ASSERT_TRUE(coll1->add(doc1.dump(), CREATE).ok());
    ASSERT_TRUE(coll1->add(doc2.dump(), CREATE).ok());

    auto results = coll1->search("*", {}, "", {"data"}, {}, {0}, 10, 1,
                                 token_ordering::FREQUENCY, {true}, 10, spp::sparse_hash_set<std::string>(),
                                 spp::sparse_hash_set<std::string>(), 10, "", 30, 4).get();

    ASSERT_EQ(2, results["found"].get<size_t>());
    ASSERT_EQ(1, results["facet_counts"].size());
    ASSERT_EQ("data", results["facet_counts"][0]["field_name"]);
    ASSERT_EQ(2, results["facet_counts"][0]["counts"].size());
    ASSERT_EQ(2, results["facet_counts"][0]["counts"][0]["count"].get<size_t>());
    ASSERT_EQ("Foo", results["facet_counts"][0]["counts"][0]["value"].get<std::string>());

    ASSERT_EQ(1, results["facet_counts"][0]["counts"][1]["count"].get<size_t>());
    ASSERT_EQ("Bazinga", results["facet_counts"][0]["counts"][1]["value"].get<std::string>());

    results = coll1->search("*", {}, "", {"data"}, {}, {0}, 10, 1,
                            token_ordering::FREQUENCY, {true}, 10, spp::sparse_hash_set<std::string>(),
                            spp::sparse_hash_set<std::string>(), 10, "data:baz", 30, 4).get();

    ASSERT_EQ(2, results["found"].get<size_t>());
    ASSERT_EQ(1, results["facet_counts"].size());
    ASSERT_EQ("data", results["facet_counts"][0]["field_name"]);
    ASSERT_EQ(1, results["facet_counts"][0]["counts"].size());
    ASSERT_EQ(1, results["facet_counts"][0]["counts"][0]["count"].get<size_t>());
    ASSERT_EQ("Bazinga", results["facet_counts"][0]["counts"][0]["value"].get<std::string>());
}

TEST_F(CollectionFacetingTest, FacetParseTest){
    std::vector<field> fields = {
            field("score", field_types::INT32, true),
            field("grade", field_types::INT32, true),
            field("rank", field_types::INT32, true),
            field("range", field_types::INT32, true),
            field("review", field_types::FLOAT, true),
            field("scale", field_types::INT32, false),
    };

    Collection* coll1 = collectionManager.create_collection("coll1", 1, fields).get();

    std::vector<std::string> range_facet_fields {
            "score(fail:[0, 40], pass:[40, 100])",
            "grade(A:[80, 100], B:[60, 80], C:[40, 60])"
    };
    std::vector<facet> range_facets;
    for(const std::string & facet_field: range_facet_fields) {
        coll1->parse_facet(facet_field, range_facets);
    }
    ASSERT_EQ(2, range_facets.size());

    ASSERT_STREQ("score", range_facets[0].field_name.c_str());
    ASSERT_TRUE(range_facets[0].is_range_query);
    ASSERT_GT(range_facets[0].facet_range_map.size(), 0);

    ASSERT_STREQ("grade", range_facets[1].field_name.c_str());
    ASSERT_TRUE(range_facets[1].is_range_query);
    ASSERT_GT(range_facets[1].facet_range_map.size(), 0);

    std::vector<std::string> normal_facet_fields {
            "score",
            "grade"
    };
    std::vector<facet> normal_facets;
    for(const std::string & facet_field: normal_facet_fields) {
        coll1->parse_facet(facet_field, normal_facets);
    }
    ASSERT_EQ(2, normal_facets.size());

    ASSERT_STREQ("score", normal_facets[0].field_name.c_str());
    ASSERT_STREQ("grade", normal_facets[1].field_name.c_str());

    std::vector<std::string> wildcard_facet_fields {
            "ran*",
            "sc*",
    };
    std::vector<facet> wildcard_facets;
    for(const std::string & facet_field: wildcard_facet_fields) {
        coll1->parse_facet(facet_field, wildcard_facets);
    }

    ASSERT_EQ(3, wildcard_facets.size());

    std::set<std::string> expected{"range", "rank", "score"};
    for (size_t i = 0; i < wildcard_facets.size(); i++) {
        ASSERT_TRUE(expected.count(wildcard_facets[i].field_name) == 1);
    }

    wildcard_facets.clear();
    coll1->parse_facet("*", wildcard_facets);

    // Last field is not a facet.
    ASSERT_EQ(fields.size() - 1, wildcard_facets.size());

    expected.clear();
    for (size_t i = 0; i < fields.size() - 1; i++) {
        expected.insert(fields[i].name);
    }

    for (size_t i = 0; i < wildcard_facets.size(); i++) {
        ASSERT_TRUE(expected.count(wildcard_facets[i].field_name) == 1);
    }

    // should also allow zero or more spaces after ","
    std::vector<std::string> mixed_facet_fields {
            "score",
            "grade(A:[80, 100], B:[60,  80], C:[40,60])",
            "ra*",
    };

    std::vector<facet> mixed_facets;
    for(const std::string & facet_field: mixed_facet_fields) {
        coll1->parse_facet(facet_field, mixed_facets);
    }
    ASSERT_EQ(4, mixed_facets.size());

    std::vector<facet*> mixed_facets_ptr;
    for(auto& f: mixed_facets) {
        mixed_facets_ptr.push_back(&f);
    }

    std::sort(mixed_facets_ptr.begin(), mixed_facets_ptr.end(), [](const facet* f1, const facet* f2) {
        return f1->field_name < f2->field_name;
    });

    ASSERT_EQ("score", mixed_facets_ptr[3]->field_name);

    ASSERT_EQ("grade", mixed_facets_ptr[0]->field_name);
    ASSERT_TRUE(mixed_facets_ptr[0]->is_range_query);
    ASSERT_GT(mixed_facets_ptr[0]->facet_range_map.size(), 0);

    ASSERT_EQ("rank", mixed_facets_ptr[2]->field_name);
    ASSERT_EQ("range", mixed_facets_ptr[1]->field_name);

    std::vector<std::string> range_facet_float_fields {
            "review(bad:[0, 2.5], good:[2.5, 5])"
    };

    std::vector<facet> float_facets;
    for(const std::string & facet_field: range_facet_float_fields) {
        auto res = coll1->parse_facet(facet_field, float_facets);

        if(!res.error().empty()) {
            LOG(ERROR) << res.error();
            FAIL();
        }
    }

    std::vector<std::string> range_facet_negative_range {
            "review(bad:[-2.5, 2.5], good:[2.5, 5])"
    };

    std::vector<facet> negative_range;
    for(const std::string & facet_field: range_facet_negative_range) {
        auto res = coll1->parse_facet(facet_field, negative_range);

        if(!res.error().empty()) {
            LOG(ERROR) << res.error();
            FAIL();
        }
    }

    fields = {
            field("ref_score", field_types::INT32, true),
            field("ref_grade", field_types::INT32, true),
            field("ref_rank", field_types::INT32, true),
            field("ref_range", field_types::INT32, true),
            field("ref_scale", field_types::INT32, false),
    };

    collectionManager.create_collection("ref_coll", 1, fields).get();

    range_facet_fields = {
            "$ref_coll(ref_score(fail:[0, 40], pass:[40, 100]), ref_grade(A:[80, 100], B:[60, 80], C:[40, 60]))"
    };
    range_facets.clear();
    for(const std::string & facet_field: range_facet_fields) {
        coll1->parse_facet(facet_field, range_facets);
    }
    ASSERT_EQ(2, range_facets.size());

    ASSERT_EQ("ref_score", range_facets[0].field_name);
    ASSERT_TRUE(range_facets[0].is_range_query);
    ASSERT_EQ(2, range_facets[0].facet_range_map.size());
    ASSERT_EQ("ref_coll", range_facets[0].reference_collection_name);

    ASSERT_EQ("ref_grade", range_facets[1].field_name);
    ASSERT_TRUE(range_facets[1].is_range_query);
    ASSERT_EQ(3, range_facets[1].facet_range_map.size());
    ASSERT_EQ("ref_coll", range_facets[1].reference_collection_name);

    normal_facet_fields = {
            "$ref_coll(ref_score, ref_grade)"
    };
    normal_facets.clear();
    for(const std::string & facet_field: normal_facet_fields) {
        coll1->parse_facet(facet_field, normal_facets);
    }
    ASSERT_EQ(2, normal_facets.size());

    ASSERT_EQ("ref_score", normal_facets[0].field_name);
    ASSERT_EQ("ref_coll", normal_facets[0].reference_collection_name);
    ASSERT_EQ("ref_grade", normal_facets[1].field_name);
    ASSERT_EQ("ref_coll", normal_facets[1].reference_collection_name);

    wildcard_facet_fields = {
            "$ref_coll(ref_ran*, ref_sc*)",
    };
    wildcard_facets.clear();
    for(const std::string & facet_field: wildcard_facet_fields) {
        coll1->parse_facet(facet_field, wildcard_facets);
    }

    ASSERT_EQ(3, wildcard_facets.size());

    expected = {"ref_range", "ref_rank", "ref_score"};
    for (size_t i = 0; i < wildcard_facets.size(); i++) {
        ASSERT_TRUE(expected.count(wildcard_facets[i].field_name) == 1);
        ASSERT_EQ("ref_coll", wildcard_facets[i].reference_collection_name);
    }

    wildcard_facets.clear();
    coll1->parse_facet("$ref_coll(*)", wildcard_facets);

    // Last field is not a facet.
    ASSERT_EQ(fields.size() - 1, wildcard_facets.size());

    expected.clear();
    for (size_t i = 0; i < fields.size() - 1; i++) {
        expected.insert(fields[i].name);
    }

    for (size_t i = 0; i < wildcard_facets.size(); i++) {
        ASSERT_TRUE(expected.count(wildcard_facets[i].field_name) == 1);
        ASSERT_EQ("ref_coll", wildcard_facets[i].reference_collection_name);
    }

    mixed_facet_fields = {
            "$ref_coll(ref_score, ref_grade(A:[80, 100], B:[60,  80], C:[40,60]), ref_ra*)",
    };

    mixed_facets.clear();
    for(const std::string & facet_field: mixed_facet_fields) {
        coll1->parse_facet(facet_field, mixed_facets);
    }
    ASSERT_EQ(4, mixed_facets.size());

    mixed_facets_ptr.clear();
    for(auto& f: mixed_facets) {
        mixed_facets_ptr.push_back(&f);
    }

    std::sort(mixed_facets_ptr.begin(), mixed_facets_ptr.end(), [](const facet* f1, const facet* f2) {
        return f1->field_name < f2->field_name;
    });

    ASSERT_EQ("ref_score", mixed_facets_ptr[3]->field_name);
    ASSERT_EQ("ref_coll", mixed_facets_ptr[3]->reference_collection_name);

    ASSERT_EQ("ref_grade", mixed_facets_ptr[0]->field_name);
    ASSERT_TRUE(mixed_facets_ptr[0]->is_range_query);
    ASSERT_GT(mixed_facets_ptr[0]->facet_range_map.size(), 0);
    ASSERT_EQ("ref_coll", mixed_facets_ptr[0]->reference_collection_name);

    ASSERT_EQ("ref_rank", mixed_facets_ptr[2]->field_name);
    ASSERT_EQ("ref_coll", mixed_facets_ptr[2]->reference_collection_name);
    ASSERT_EQ("ref_range", mixed_facets_ptr[1]->field_name);
    ASSERT_EQ("ref_coll", mixed_facets_ptr[1]->reference_collection_name);
}

TEST_F(CollectionFacetingTest, RangeFacetTest) {
    std::vector<field> fields = {field("place", field_types::STRING, false),
                                 field("state", field_types::STRING, false),
                                 field("visitors", field_types::INT32, true),
                                 field("rating", field_types::FLOAT, true),
                                 field("trackingFrom", field_types::INT32, true),};
    Collection* coll1 = collectionManager.create_collection(
            "coll1", 1, fields, "", 0, "", {}, {}
    ).get();
    nlohmann::json doc1;
    doc1["id"] = "0";
    doc1["place"] = "Mysore Palace";
    doc1["state"] = "Karnataka";
    doc1["visitors"] = 235486;
    doc1["rating"] = 4.7;
    doc1["trackingFrom"] = 1900;

    nlohmann::json doc2;
    doc2["id"] = "1";
    doc2["place"] = "Hampi";
    doc2["state"] = "Karnataka";
    doc2["visitors"] = 187654;
    doc2["rating"] = 2.9;
    doc2["trackingFrom"] = 1900;

    nlohmann::json doc3;
    doc3["id"] = "2";
    doc3["place"] = "Mahabalipuram";
    doc3["state"] = "TamilNadu";
    doc3["visitors"] = 174684;
    doc3["rating"] = 3.8;
    doc3["trackingFrom"] = 1900;

    nlohmann::json doc4;
    doc4["id"] = "3";
    doc4["place"] = "Meenakshi Amman Temple";
    doc4["state"] = "TamilNadu";
    doc4["visitors"] = 246676;
    doc4["rating"] = 4.5;
    doc4["trackingFrom"] = 2000;

    nlohmann::json doc5;
    doc5["id"] = "4";
    doc5["place"] = "Staue of Unity";
    doc5["state"] = "Gujarat";
    doc5["visitors"] = 345878;
    doc5["rating"] = 3.5;
    doc5["trackingFrom"] = 2000;

    ASSERT_TRUE(coll1->add(doc1.dump()).ok());
    ASSERT_TRUE(coll1->add(doc2.dump()).ok());
    ASSERT_TRUE(coll1->add(doc3.dump()).ok());
    ASSERT_TRUE(coll1->add(doc4.dump()).ok());
    ASSERT_TRUE(coll1->add(doc5.dump()).ok());

    auto result = coll1->search("Karnataka", {"state"},
                                 "", {"visitors(Busy:[0, 200000], VeryBusy:[200000, 500000])"},
                                 {}, {2}, 10,
                                 1, FREQUENCY, {true},
                                 10, spp::sparse_hash_set<std::string>(),
                                 spp::sparse_hash_set<std::string>(), 10, "", 30, 4, "", 10, {}, {}, {}, 0,
                                 "<mark>", "</mark>", {}, 1000,
                                 true, false, true, "", true);
    if(!result.ok()) {
        LOG(INFO) << result.error();
    }

    auto results = result.get();

    ASSERT_EQ(2, results["facet_counts"][0]["counts"].size());
    ASSERT_EQ(1, (int) results["facet_counts"][0]["counts"][0]["count"]);
    ASSERT_EQ("Busy", results["facet_counts"][0]["counts"][0]["value"].get<std::string>());
    ASSERT_EQ(1, (int) results["facet_counts"][0]["counts"][1]["count"]);
    ASSERT_EQ("VeryBusy", results["facet_counts"][0]["counts"][1]["value"].get<std::string>());

    auto results2 = coll1->search("Gujarat", {"state"},
                                  "", {"visitors(Busy:[0, 200000], VeryBusy:[200000, 500000])"},
                                  {}, {2}, 10,
                                  1, FREQUENCY, {true},
                                  10, spp::sparse_hash_set<std::string>(),
                                  spp::sparse_hash_set<std::string>(), 10, "", 30, 4, "", 10, {}, {}, {}, 0,
                                  "<mark>", "</mark>", {}, 1000,
                                  true, false, true, "", true).get();
    ASSERT_EQ(1, results2["facet_counts"][0]["counts"].size());
    ASSERT_EQ(1, results2["facet_counts"][0]["counts"][0]["count"].get<std::size_t>());
    ASSERT_STREQ("VeryBusy", results2["facet_counts"][0]["counts"][0]["value"].get<std::string>().c_str());
    ASSERT_TRUE(results2["facet_counts"][0]["counts"][1]["value"] == nullptr);

    // ensure that unknown facet field are handled

    auto results3 = coll1->search("Gujarat", {"state"},
                             "", {"visitorsz(Busy:[0, 200000], VeryBusy:[200000, 500000])"},
                             {}, {2}, 10,
                             1, FREQUENCY, {true},
                             10, spp::sparse_hash_set<std::string>(),
                             spp::sparse_hash_set<std::string>(), 10, "", 30, 4, "", 10, {}, {}, {}, 0,
                             "<mark>", "</mark>", {}, 1000,
                             true, false, true, "", true);
    ASSERT_FALSE(results3.ok());
    ASSERT_EQ("Could not find a facet field named `visitorsz` in the schema.", results3.error());

    auto results4 = coll1->search("*", {"state"},
                                  "", {"trackingFrom(Old:[0, 1910], New:[1910, 2100])"},
                                  {}, {2}, 10,
                                  1, FREQUENCY, {true},
                                  10, spp::sparse_hash_set<std::string>(),
                                  spp::sparse_hash_set<std::string>(), 10, "", 30, 4, "", 10, {}, {}, {}, 0,
                                  "<mark>", "</mark>", {}, 1000,
                                  true, false, true, "", true).get();

    ASSERT_EQ(2, results4["facet_counts"][0]["counts"].size());
    ASSERT_EQ(3, results4["facet_counts"][0]["counts"][0]["count"].get<std::size_t>());
    ASSERT_EQ("Old", results4["facet_counts"][0]["counts"][0]["value"].get<std::string>());

    ASSERT_EQ(2, results4["facet_counts"][0]["counts"][1]["count"].get<std::size_t>());
    ASSERT_EQ("New", results4["facet_counts"][0]["counts"][1]["value"].get<std::string>());

    //range faceting on float fields
    results4 = coll1->search("*", {"state"},
                                  "", {"rating(Average:[0, 3], Good:[3, 4], Best:[4, 5])"},
                                  {}, {2}, 10,
                                  1, FREQUENCY, {true},
                                  10, spp::sparse_hash_set<std::string>(),
                                  spp::sparse_hash_set<std::string>(), 10, "", 30, 4, "", 10, {}, {}, {}, 0,
                                  "<mark>", "</mark>", {}, 1000,
                                  true, false, true, "", true).get();

    ASSERT_EQ(3, results4["facet_counts"][0]["counts"].size());

    ASSERT_EQ(2, results4["facet_counts"][0]["counts"][0]["count"].get<std::size_t>());
    ASSERT_EQ("Good", results4["facet_counts"][0]["counts"][0]["value"].get<std::string>());

    ASSERT_EQ(2, results4["facet_counts"][0]["counts"][1]["count"].get<std::size_t>());
    ASSERT_EQ("Best", results4["facet_counts"][0]["counts"][1]["value"].get<std::string>());

    ASSERT_EQ(1, results4["facet_counts"][0]["counts"][2]["count"].get<std::size_t>());
    ASSERT_EQ("Average", results4["facet_counts"][0]["counts"][2]["value"].get<std::string>());
    
    //stats on float field
    ASSERT_EQ(5, results4["facet_counts"][0]["stats"].size());
    ASSERT_FLOAT_EQ(3.8799999713897706, results4["facet_counts"][0]["stats"]["avg"].get<double>());
    ASSERT_FLOAT_EQ(2.9000000953674316, results4["facet_counts"][0]["stats"]["min"].get<double>());
    ASSERT_FLOAT_EQ(4.699999809265137, results4["facet_counts"][0]["stats"]["max"].get<double>());
    ASSERT_FLOAT_EQ(19.399999856948853, results4["facet_counts"][0]["stats"]["sum"].get<double>());
    ASSERT_FLOAT_EQ(3, results4["facet_counts"][0]["stats"]["total_values"].get<size_t>());

    // ensure that only integer fields are allowed
    auto rop = coll1->search("Karnataka", {"state"},
                                 "", {"state(Busy:[0, 200000], VeryBusy:[200000, 500000])"},
                                 {}, {2}, 10,
                                 1, FREQUENCY, {true},
                                 10, spp::sparse_hash_set<std::string>(),
                                 spp::sparse_hash_set<std::string>(), 10, "", 30, 4, "", 10, {}, {}, {}, 0,
                                 "<mark>", "</mark>", {}, 1000,
                                 true, false, true, "", true);

    ASSERT_FALSE(rop.ok());
    ASSERT_EQ("Range facet is restricted to only integer and float fields.", rop.error());

    // ensure that bad facet range values are handled
    rop = coll1->search("Karnataka", {"state"},
                        "", {"visitors(Busy:[alpha, 200000], VeryBusy:[200000, beta])"},
                        {}, {2}, 10,
                        1, FREQUENCY, {true},
                        10, spp::sparse_hash_set<std::string>(),
                        spp::sparse_hash_set<std::string>(), 10, "", 30, 4, "", 10, {}, {}, {}, 0,
                        "<mark>", "</mark>", {}, 1000,
                        true, false, true, "", true);

    ASSERT_FALSE(rop.ok());
    ASSERT_EQ("Facet range value is not valid.", rop.error());

    collectionManager.drop_collection("coll1");
}

TEST_F(CollectionFacetingTest, RangeFacetContinuity) {
    std::vector<field> fields = {field("place", field_types::STRING, false),
                                 field("state", field_types::STRING, false),
                                 field("visitors", field_types::INT32, true),};
    Collection* coll1 = collectionManager.create_collection(
            "coll1", 1, fields, "", 0, "", {}, {}
    ).get();
    nlohmann::json doc1;
    doc1["id"] = "0";
    doc1["place"] = "Mysore Palace";
    doc1["state"] = "Karnataka";
    doc1["visitors"] = 235486;

    nlohmann::json doc2;
    doc2["id"] = "1";
    doc2["place"] = "Hampi";
    doc2["state"] = "Karnataka";
    doc2["visitors"] = 187654;

    nlohmann::json doc3;
    doc3["id"] = "2";
    doc3["place"] = "Mahabalipuram";
    doc3["state"] = "TamilNadu";
    doc3["visitors"] = 174684;

    nlohmann::json doc4;
    doc4["id"] = "3";
    doc4["place"] = "Meenakshi Amman Temple";
    doc4["state"] = "TamilNadu";
    doc4["visitors"] = 246676;

    nlohmann::json doc5;
    doc5["id"] = "4";
    doc5["place"] = "Staue of Unity";
    doc5["state"] = "Gujarat";
    doc5["visitors"] = 345878;


    ASSERT_TRUE(coll1->add(doc1.dump()).ok());
    ASSERT_TRUE(coll1->add(doc2.dump()).ok());
    ASSERT_TRUE(coll1->add(doc3.dump()).ok());
    ASSERT_TRUE(coll1->add(doc4.dump()).ok());
    ASSERT_TRUE(coll1->add(doc5.dump()).ok());

    auto results = coll1->search("TamilNadu", {"state"},
                                 "", {"visitors(Busy:[0, 200000], VeryBusy:[200001, 500000])"},
                                 {}, {2}, 10,
                                 1, FREQUENCY, {true},
                                 10, spp::sparse_hash_set<std::string>(),
                                 spp::sparse_hash_set<std::string>(), 10, "", 30, 4, "", 10, {}, {}, {}, 0,
                                 "<mark>", "</mark>", {}, 1000,
                                 true, false, true, "", true);
    ASSERT_STREQ("Ranges in range facet syntax should be continous.", results.error().c_str());

    auto results2 = coll1->search("TamilNadu", {"state"},
                                  "", {"visitors(Busy:[0, 200000], VeryBusy:[199999, 500000])"},
                                  {}, {2}, 10,
                                  1, FREQUENCY, {true},
                                  10, spp::sparse_hash_set<std::string>(),
                                  spp::sparse_hash_set<std::string>(), 10, "", 30, 4, "", 10, {}, {}, {}, 0,
                                  "<mark>", "</mark>", {}, 1000,
                                  true, false, true, "", true);
    ASSERT_STREQ("Ranges in range facet syntax should be continous.", results2.error().c_str());

    collectionManager.drop_collection("coll1");
}

TEST_F(CollectionFacetingTest, RangeFacetTypo) {
    std::vector<field> fields = {field("place", field_types::STRING, false),
                                 field("state", field_types::STRING, false),
                                 field("visitors", field_types::INT32, true),};
    Collection* coll1 = collectionManager.create_collection(
            "coll1", 1, fields, "", 0, "", {}, {}
    ).get();
    nlohmann::json doc1;
    doc1["id"] = "0";
    doc1["place"] = "Mysore Palace";
    doc1["state"] = "Karnataka";
    doc1["visitors"] = 235486;

    nlohmann::json doc2;
    doc2["id"] = "1";
    doc2["place"] = "Hampi";
    doc2["state"] = "Karnataka";
    doc2["visitors"] = 187654;

    nlohmann::json doc3;
    doc3["id"] = "2";
    doc3["place"] = "Mahabalipuram";
    doc3["state"] = "TamilNadu";
    doc3["visitors"] = 174684;

    nlohmann::json doc4;
    doc4["id"] = "3";
    doc4["place"] = "Meenakshi Amman Temple";
    doc4["state"] = "TamilNadu";
    doc4["visitors"] = 246676;

    nlohmann::json doc5;
    doc5["id"] = "4";
    doc5["place"] = "Staue of Unity";
    doc5["state"] = "Gujarat";
    doc5["visitors"] = 345878;


    ASSERT_TRUE(coll1->add(doc1.dump()).ok());
    ASSERT_TRUE(coll1->add(doc2.dump()).ok());
    ASSERT_TRUE(coll1->add(doc3.dump()).ok());
    ASSERT_TRUE(coll1->add(doc4.dump()).ok());
    ASSERT_TRUE(coll1->add(doc5.dump()).ok());

    auto results = coll1->search("TamilNadu", {"state"},
                                 "", {"visitors(Busy:[0, 200000], VeryBusy:[200000, 500000)"}, //missing ']' at end
                                 {}, {2}, 10,
                                 1, FREQUENCY, {true},
                                 10, spp::sparse_hash_set<std::string>(),
                                 spp::sparse_hash_set<std::string>(), 10, "", 30, 4, "", 10, {}, {}, {}, 0,
                                 "<mark>", "</mark>", {}, 1000,
                                 true, false, true, "", true);
    ASSERT_STREQ("Error splitting the facet range values.", results.error().c_str());

    auto results2 = coll1->search("TamilNadu", {"state"},
                                  "", {"visitors(Busy:[0, 200000], VeryBusy:200000, 500000])"}, //missing '[' in second range
                                  {}, {2}, 10,
                                  1, FREQUENCY, {true},
                                  10, spp::sparse_hash_set<std::string>(),
                                  spp::sparse_hash_set<std::string>(), 10, "", 30, 4, "", 10, {}, {}, {}, 0,
                                  "<mark>", "</mark>", {}, 1000,
                                  true, false, true, "", true);
    ASSERT_STREQ("Error splitting the facet range values.", results2.error().c_str());

    auto results3 = coll1->search("TamilNadu", {"state"},
                                  "", {"visitors(Busy:[0, 200000] VeryBusy:[200000, 500000])"}, //missing ',' between ranges
                                  {}, {2}, 10,
                                  1, FREQUENCY, {true},
                                  10, spp::sparse_hash_set<std::string>(),
                                  spp::sparse_hash_set<std::string>(), 10, "", 30, 4, "", 10, {}, {}, {}, 0,
                                  "<mark>", "</mark>", {}, 1000,
                                  true, false, true, "", true);
    ASSERT_STREQ("Error splitting the facet range values.", results3.error().c_str());

    auto results4 = coll1->search("TamilNadu", {"state"},
                                  "", {"visitors(Busy:[0 200000], VeryBusy:[200000, 500000])"}, //missing ',' between first ranges values
                                  {}, {2}, 10,
                                  1, FREQUENCY, {true},
                                  10, spp::sparse_hash_set<std::string>(),
                                  spp::sparse_hash_set<std::string>(), 10, "", 30, 4, "", 10, {}, {}, {}, 0,
                                  "<mark>", "</mark>", {}, 1000,
                                  true, false, true, "", true);
    ASSERT_STREQ("Facet range value is not valid.", results4.error().c_str());

    auto results5 = coll1->search("TamilNadu", {"state"},
                                  "", {"visitors(Busy:[0, 200000 VeryBusy:200000, 500000])"}, //missing '],' and '['
                                  {}, {2}, 10,
                                  1, FREQUENCY, {true},
                                  10, spp::sparse_hash_set<std::string>(),
                                  spp::sparse_hash_set<std::string>(), 10, "", 30, 4, "", 10, {}, {}, {}, 0,
                                  "<mark>", "</mark>", {}, 1000,
                                  true, false, true, "", true);
    ASSERT_STREQ("Facet range value is not valid.", results5.error().c_str());

    collectionManager.drop_collection("coll1");
}

TEST_F(CollectionFacetingTest, RangeFacetsFloatRange) {
    std::vector<field> fields = {field("name", field_types::STRING, false),
                                 field("inches", field_types::FLOAT, true),};
    Collection* coll1 = collectionManager.create_collection(
            "coll1", 1, fields, "", 0, "", {}, {}).get();

    nlohmann::json doc;
    doc["id"] = "0";
    doc["name"] = "TV 1";
    doc["inches"] = 32.4;

    ASSERT_TRUE(coll1->add(doc.dump()).ok());

    doc["id"] = "1";
    doc["name"] = "TV 2";
    doc["inches"] = 55;

    ASSERT_TRUE(coll1->add(doc.dump()).ok());

    doc["id"] = "2";
    doc["name"] = "TV 3";
    doc["inches"] = 55.6;

    ASSERT_TRUE(coll1->add(doc.dump()).ok());

    auto results = coll1->search("*", {},
                                 "", {"inches(small:[0, 55.5])"},
                                 {}, {2}, 10,
                                 1, FREQUENCY, {true},
                                 10, spp::sparse_hash_set<std::string>(),
                                 spp::sparse_hash_set<std::string>(), 10, "", 30, 4, "", 10, {}, {}, {}, 0,
                                 "<mark>", "</mark>", {}, 1000,
                                 true, false, true, "", true).get();

    ASSERT_EQ(1, results["facet_counts"][0]["counts"].size());
    ASSERT_EQ(2, (int) results["facet_counts"][0]["counts"][0]["count"]);
    ASSERT_EQ("small", results["facet_counts"][0]["counts"][0]["value"]);
}

TEST_F(CollectionFacetingTest, RangeFacetsMinMaxRange) {
    std::vector<field> fields = {field("name", field_types::STRING, false),
                                 field("inches", field_types::FLOAT, true),};
    Collection* coll1 = collectionManager.create_collection(
            "coll1", 1, fields, "", 0, "", {}, {}).get();

    nlohmann::json doc;
    doc["id"] = "0";
    doc["name"] = "TV 1";
    doc["inches"] = 32.4;

    ASSERT_TRUE(coll1->add(doc.dump()).ok());

    doc["id"] = "1";
    doc["name"] = "TV 2";
    doc["inches"] = 55;

    ASSERT_TRUE(coll1->add(doc.dump()).ok());

    doc["id"] = "2";
    doc["name"] = "TV 3";
    doc["inches"] = 55.6;

    ASSERT_TRUE(coll1->add(doc.dump()).ok());

    auto results = coll1->search("*", {},
                                 "", {"inches(small:[0, 55], large:[55, ])"},
                                 {}, {2}, 10,
                                 1, FREQUENCY, {true},
                                 10, spp::sparse_hash_set<std::string>(),
                                 spp::sparse_hash_set<std::string>(), 10, "", 30, 4, "", 10, {}, {}, {}, 0,
                                 "<mark>", "</mark>", {}, 1000,
                                 true, false, true, "", true).get();

    ASSERT_EQ(2, results["facet_counts"][0]["counts"].size());
    ASSERT_EQ(2, (int) results["facet_counts"][0]["counts"][0]["count"]);
    ASSERT_EQ("small", results["facet_counts"][0]["counts"][0]["value"]);
    ASSERT_EQ(1, (int) results["facet_counts"][0]["counts"][1]["count"]);
    ASSERT_EQ("large", results["facet_counts"][0]["counts"][1]["value"]);

    results = coll1->search("*", {},
                                 "", {"inches(small:[,55])"},
                                 {}, {2}, 10,
                                 1, FREQUENCY, {true},
                                 10, spp::sparse_hash_set<std::string>(),
                                 spp::sparse_hash_set<std::string>(), 10, "", 30, 4, "", 10, {}, {}, {}, 0,
                                 "<mark>", "</mark>", {}, 1000,
                                 true, false, true, "", true).get();

    ASSERT_EQ(1, results["facet_counts"][0]["counts"].size());
    ASSERT_EQ(2, (int) results["facet_counts"][0]["counts"][0]["count"]);
    ASSERT_EQ("small", results["facet_counts"][0]["counts"][0]["value"]);
}

TEST_F(CollectionFacetingTest, RangeFacetRangeLabelWithSpace) {
    std::vector<field> fields = {field("name", field_types::STRING, false),
                                 field("inches", field_types::FLOAT, true),};
    Collection* coll1 = collectionManager.create_collection(
            "coll1", 1, fields, "", 0, "", {}, {}).get();

    nlohmann::json doc;
    doc["id"] = "0";
    doc["name"] = "TV 1";
    doc["inches"] = 32.4;

    ASSERT_TRUE(coll1->add(doc.dump()).ok());

    doc["id"] = "1";
    doc["name"] = "TV 2";
    doc["inches"] = 55;

    ASSERT_TRUE(coll1->add(doc.dump()).ok());

    doc["id"] = "2";
    doc["name"] = "TV 3";
    doc["inches"] = 55.6;

    ASSERT_TRUE(coll1->add(doc.dump()).ok());

    auto results = coll1->search("*", {},
                            "", {"inches(small tvs with display size:[0,55])"},
                            {}, {2}, 10,
                            1, FREQUENCY, {true},
                            10, spp::sparse_hash_set<std::string>(),
                            spp::sparse_hash_set<std::string>(), 10, "", 30, 4, "", 10, {}, {}, {}, 0,
                            "<mark>", "</mark>", {}, 1000,
                            true, false, true, "", true).get();

    ASSERT_EQ(1, results["facet_counts"][0]["counts"].size());
    ASSERT_EQ(2, (int) results["facet_counts"][0]["counts"][0]["count"]);
    ASSERT_EQ("small tvs with display size", results["facet_counts"][0]["counts"][0]["value"]);
}

TEST_F(CollectionFacetingTest, RangeFacetRangeNegativeRanges) {
    std::vector<field> fields = {field("team", field_types::STRING, false),
                                 field("nrr", field_types::FLOAT, true),};
    Collection* coll1 = collectionManager.create_collection(
            "coll1", 1, fields, "", 0, "",
            {},{}).get();

    nlohmann::json doc;
    doc["id"] = "0";
    doc["team"] = "india";
    doc["nrr"] = 1.353;
    ASSERT_TRUE(coll1->add(doc.dump()).ok());

    doc["id"] = "1";
    doc["team"] = "australia";
    doc["nrr"] = -0.193;
    ASSERT_TRUE(coll1->add(doc.dump()).ok());

    doc["id"] = "2";
    doc["team"] = "pakistan";
    doc["nrr"] = -0.400;
    ASSERT_TRUE(coll1->add(doc.dump()).ok());

    doc["id"] = "3";
    doc["team"] = "afghanistan";
    doc["nrr"] = -0.969;
    ASSERT_TRUE(coll1->add(doc.dump()).ok());

    doc["id"] = "4";
    doc["team"] = "srilanka";
    doc["nrr"] = -1.048;
    ASSERT_TRUE(coll1->add(doc.dump()).ok());

    doc["id"] = "5";
    doc["team"] = "england";
    doc["nrr"] = -1.248;
    ASSERT_TRUE(coll1->add(doc.dump()).ok());

    doc["id"] = "6";
    doc["team"] = "bangladesh";
    doc["nrr"] = -1.253;
    ASSERT_TRUE(coll1->add(doc.dump()).ok());

    doc["id"] = "7";
    doc["team"] = "new zealand";
    doc["nrr"] = 1.481;
    ASSERT_TRUE(coll1->add(doc.dump()).ok());

    auto results = coll1->search("*", {},
                                 "", {"nrr(poor:[-1.5,-1], decent:[-1,0], good:[0,2])"},
                                 {}, {2}, 10,
                                 1, FREQUENCY, {true},
                                 10, spp::sparse_hash_set<std::string>(),
                                 spp::sparse_hash_set<std::string>(), 10, "", 30, 4, "", 10, {}, {}, {}, 0,
                                 "<mark>", "</mark>", {}, 1000,
                                 true, false, true, "", true).get();

    ASSERT_EQ(3, results["facet_counts"][0]["counts"].size());
    ASSERT_EQ(3, (int) results["facet_counts"][0]["counts"][0]["count"]);
    ASSERT_EQ("poor", results["facet_counts"][0]["counts"][0]["value"]);
    ASSERT_EQ(3, (int) results["facet_counts"][0]["counts"][1]["count"]);
    ASSERT_EQ("decent", results["facet_counts"][0]["counts"][1]["value"]);
    ASSERT_EQ(2, (int) results["facet_counts"][0]["counts"][2]["count"]);
    ASSERT_EQ("good", results["facet_counts"][0]["counts"][2]["value"]);
}

TEST_F(CollectionFacetingTest, FacetWithPhraseSearch) {
    Collection *coll1;

    std::vector<field> fields = {field("title", field_types::STRING, true),
                                 field("rating", field_types::FLOAT, false)};

    std::vector<sort_by> sort_fields = {sort_by("rating", "DESC")};

    coll1 = collectionManager.get_collection("coll1").get();
    if (coll1 == nullptr) {
        coll1 = collectionManager.create_collection("coll1", 4, fields, "rating").get();
    }

    nlohmann::json doc;
    doc["id"] = "0";
    doc["title"] = "The Shawshank Redemption";
    doc["rating"] = 9.3;

    coll1->add(doc.dump());

    doc["id"] = "1";
    doc["title"] = "The Godfather";
    doc["rating"] = 9.2;

    coll1->add(doc.dump());

    std::vector<std::string> facets = {"title"};

    nlohmann::json results = coll1->search(R"("shawshank")", {"title"}, "", facets, sort_fields, {0}, 10, 1,
                                           token_ordering::FREQUENCY, {true}, 10, spp::sparse_hash_set<std::string>(),
                                           spp::sparse_hash_set<std::string>(), 2,"",  30UL, 4UL,
                                           "", 1UL, "", "", {}, 3UL, "<mark>", "</mark>", {},
                                           4294967295UL, true, false, true, "", false, 6000000UL, 4UL,
                                           7UL, fallback, 4UL, {off}, 32767UL, 32767UL, 2UL, 2UL, false,
                                           "", true, 0UL, max_score, 100UL, 0UL, 4294967295UL, HASH).get();

    ASSERT_EQ(1, results["facet_counts"].size());
    ASSERT_EQ(1, results["facet_counts"][0]["counts"].size());
    ASSERT_EQ("The Shawshank Redemption", results["facet_counts"][0]["counts"][0]["value"]);
}

TEST_F(CollectionFacetingTest, SampleFacetCounts) {
    nlohmann::json schema = R"({
            "name": "coll1",
            "fields": [
                {"name": "color", "type": "string", "facet": true}
            ]
        })"_json;

    Collection* coll1 = collectionManager.create_collection(schema).get();

    std::mt19937 gen(137723); // use constant seed to make sure that counts don't jump around
    std::uniform_int_distribution<> distr(1, 100); // 1 to 100 inclusive

    size_t count_blue = 0, count_red = 0;

    for(size_t i = 0; i < 1000; i++) {
        nlohmann::json doc;
        if(distr(gen) % 2 == 0) {
            doc["color"] = "blue";
            count_blue++;
        } else {
            doc["color"] = "red";
            count_red++;
        }

        ASSERT_TRUE(coll1->add(doc.dump()).ok());
    }

    auto res = coll1->search("*", {}, "", {"color"}, {}, {0}, 3, 1, FREQUENCY, {true}, 5,
                             spp::sparse_hash_set<std::string>(),
                             spp::sparse_hash_set<std::string>(), 10, "", 30, 4, "", 20, {}, {}, {}, 0,
                             "<mark>", "</mark>", {}, 1000, true, false, true, "", false, 6000 * 1000, 4, 7, fallback,
                             4, {off}, 3, 3, 2, 2, false, "", true, 0, max_score, 10, 0).get();

    ASSERT_EQ(1000, res["found"].get<size_t>());
    ASSERT_EQ(1, res["facet_counts"].size());
    ASSERT_EQ(2, res["facet_counts"][0]["counts"].size());

    // verify approximate counts
    ASSERT_GE(res["facet_counts"][0]["counts"][0]["count"].get<size_t>(), 250);
    ASSERT_GE(res["facet_counts"][0]["counts"][1]["count"].get<size_t>(), 250);
    ASSERT_TRUE(res["facet_counts"][0]["sampled"].get<bool>());

    // when sample threshold is high, don't estimate
    res = coll1->search("*", {}, "", {"color"}, {}, {0}, 3, 1, FREQUENCY, {true}, 5,
                        spp::sparse_hash_set<std::string>(),
                        spp::sparse_hash_set<std::string>(), 10, "", 30, 4, "", 20, {}, {}, {}, 0,
                        "<mark>", "</mark>", {}, 1000, true, false, true, "", false, 6000 * 1000, 4, 7, fallback,
                        4, {off}, 3, 3, 2, 2, false, "", true, 0, max_score, 10, 10000).get();

    ASSERT_EQ(1000, res["found"].get<size_t>());
    ASSERT_EQ(1, res["facet_counts"].size());
    ASSERT_EQ(2, res["facet_counts"][0]["counts"].size());

    for(size_t i = 0; i < res["facet_counts"][0]["counts"].size(); i++) {
        if(res["facet_counts"][0]["counts"][i]["value"].get<std::string>() == "red") {
            ASSERT_EQ(count_red, res["facet_counts"][0]["counts"][i]["count"].get<size_t>());
        } else {
            ASSERT_EQ(count_blue, res["facet_counts"][0]["counts"][i]["count"].get<size_t>());
        }
    }

    ASSERT_FALSE(res["facet_counts"][0]["sampled"].get<bool>());

    // test for sample percent > 100

    auto res_op = coll1->search("*", {}, "", {"color"}, {}, {0}, 3, 1, FREQUENCY, {true}, 5,
                                spp::sparse_hash_set<std::string>(),
                                spp::sparse_hash_set<std::string>(), 10, "", 30, 4, "", 20, {}, {}, {}, 0,
                                "<mark>", "</mark>", {}, 1000, true, false, true, "", false, 6000 * 1000, 4, 7, fallback,
                                4, {off}, 3, 3, 2, 2, false, "", true, 0, max_score, 200, 0);

    ASSERT_FALSE(res_op.ok());
    ASSERT_EQ("Value of `facet_sample_percent` must be less than 100.", res_op.error());
}

TEST_F(CollectionFacetingTest, FacetOnArrayFieldWithSpecialChars) {
    std::vector<field> fields = {
            field("tags", field_types::STRING_ARRAY, true),
            field("points", field_types::INT32, true),
    };

    Collection* coll1 = collectionManager.create_collection("coll1", 1, fields).get();

    nlohmann::json doc;
    doc["tags"] = {"gamma"};
    doc["points"] = 10;
    ASSERT_TRUE(coll1->add(doc.dump()).ok());

    doc["tags"] = {"alpha", "| . |", "beta", "gamma"};
    doc["points"] = 10;
    ASSERT_TRUE(coll1->add(doc.dump()).ok());

    auto results = coll1->search("*", {},
                                 "", {"tags"}, {}, {2}, 10, 1, FREQUENCY, {true}, 1).get();

    ASSERT_EQ(1, results["facet_counts"].size());
    ASSERT_EQ(4, results["facet_counts"][0]["counts"].size());

    for(size_t i = 0; i < results["facet_counts"][0]["counts"].size(); i++) {
        auto fvalue = results["facet_counts"][0]["counts"][i]["value"].get<std::string>();
        if(fvalue == "gamma") {
            ASSERT_EQ(2, results["facet_counts"][0]["counts"][i]["count"].get<size_t>());
        } else {
            ASSERT_EQ(1, results["facet_counts"][0]["counts"][i]["count"].get<size_t>());
        }
    }
}

TEST_F(CollectionFacetingTest, FloatFieldValueTruncation) {
    std::vector<field> fields = {
            field("tags", field_types::STRING_ARRAY, true),
            field("points", field_types::FLOAT, true),
    };

    Collection* coll1 = collectionManager.create_collection("coll1", 1, fields).get();

    nlohmann::json doc;
    doc["tags"] = {"gamma"};
    doc["points"] = 300;
    ASSERT_TRUE(coll1->add(doc.dump()).ok());

    auto results = coll1->search("*", {},
                                 "", {"points"}, {}, {2}, 10, 1, FREQUENCY, {true}, 1).get();

    ASSERT_EQ(1, results["facet_counts"].size());
    ASSERT_EQ(1, results["facet_counts"][0]["counts"].size());

    ASSERT_EQ("300", results["facet_counts"][0]["counts"][0]["value"].get<std::string>());
}

TEST_F(CollectionFacetingTest, FacetingReturnParent) {
    nlohmann::json schema = R"({
        "name": "coll1",
        "enable_nested_fields": true,
        "fields": [
          {"name": "value.color", "type": "string", "optional": false, "facet": true },
          {"name": "value.r", "type": "int32", "optional": false, "facet": true },
          {"name": "value.g", "type": "int32", "optional": false, "facet": true },
          {"name": "value.b", "type": "int32", "optional": false, "facet": true }
        ]
    })"_json;

    auto op = collectionManager.create_collection(schema);
    ASSERT_TRUE(op.ok());
    Collection* coll1 = op.get();

    nlohmann::json doc1 = R"({
        "value": {
            "color": "red",
            "r": 255,
            "g": 0,
            "b": 0
        }
    })"_json;

    nlohmann::json doc2 = R"({
        "value": {
            "color": "blue",
            "r": 0,
            "g": 0,
            "b": 255
        }
    })"_json;

    auto add_op = coll1->add(doc1.dump(), CREATE);
    ASSERT_TRUE(add_op.ok());
    add_op = coll1->add(doc2.dump(), CREATE);
    ASSERT_TRUE(add_op.ok());

    auto search_op = coll1->search("*", {},"", {"value.color"},
                                   {}, {2}, 10, 1,FREQUENCY, {true},
                                   1, spp::sparse_hash_set<std::string>(),
                                   spp::sparse_hash_set<std::string>(),10, "",
                                   30, 4, "",
                                   Index::TYPO_TOKENS_THRESHOLD, "", "",{},
                                   3, "<mark>", "</mark>", {},
                                   UINT32_MAX, true, false, true,
                                 "", false, 6000*1000, 4, 7,
                                 fallback, 4, {off}, INT16_MAX, INT16_MAX,
                                 2, 2, false, "",
                                 true, 0, max_score, 100,
                                 0, 0, HASH, 30000,
                                 2, "", {"value.color"});

    if(!search_op.ok()) {
        LOG(ERROR) << search_op.error();
        FAIL();
    }
    auto results = search_op.get();
    ASSERT_EQ(1, results["facet_counts"].size());
    ASSERT_EQ(2, results["facet_counts"][0]["counts"].size());
    ASSERT_EQ("{\"b\":0,\"color\":\"red\",\"g\":0,\"r\":255}", results["facet_counts"][0]["counts"][0]["parent"].dump());
    ASSERT_EQ("red", results["facet_counts"][0]["counts"][0]["value"]);
    ASSERT_EQ("{\"b\":255,\"color\":\"blue\",\"g\":0,\"r\":0}", results["facet_counts"][0]["counts"][1]["parent"].dump());
    ASSERT_EQ("blue", results["facet_counts"][0]["counts"][1]["value"]);

    //not passing facet_fields in facet_return_parent list will only return facet value, not immediate parent for those field
    search_op = coll1->search("*", {},"", {"value.color"},
                                   {}, {2}, 10, 1,FREQUENCY, {true},
                                   1, spp::sparse_hash_set<std::string>(),
                                   spp::sparse_hash_set<std::string>(),10, "",
                                   30, 4, "",
                                   Index::TYPO_TOKENS_THRESHOLD, "", "",{},
                                   3, "<mark>", "</mark>", {},
                                   UINT32_MAX, true, false, true,
                                   "", false, 6000*1000, 4, 7,
                                   fallback, 4, {off}, INT16_MAX, INT16_MAX,
                                   2, 2, false, "",
                                   true, 0, max_score, 100,
                                   0, 0, HASH, 30000,
                                   2, "", {});

    if(!search_op.ok()) {
        LOG(ERROR) << search_op.error();
        FAIL();
    }
    results = search_op.get();
    ASSERT_EQ(1, results["facet_counts"].size());
    ASSERT_EQ(2, results["facet_counts"][0]["counts"].size());
    ASSERT_EQ("red", results["facet_counts"][0]["counts"][0]["value"]);
    ASSERT_EQ("blue", results["facet_counts"][0]["counts"][1]["value"]);

    search_op = coll1->search("*", {},"", {"value.color", "value.r"},
                              {}, {2}, 10, 1,FREQUENCY, {true},
                              1, spp::sparse_hash_set<std::string>(),
                              spp::sparse_hash_set<std::string>(),10, "",
                              30, 4, "",
                              Index::TYPO_TOKENS_THRESHOLD, "", "",{},
                              3, "<mark>", "</mark>", {},
                              UINT32_MAX, true, false, true,
                              "", false, 6000*1000, 4, 7,
                              fallback, 4, {off}, INT16_MAX, INT16_MAX,
                              2, 2, false, "",
                              true, 0, max_score, 100,
                              0, 0, HASH, 30000,
                              2, "", {"value.r"});

    if(!search_op.ok()) {
        LOG(ERROR) << search_op.error();
        FAIL();
    }
    results = search_op.get();
    ASSERT_EQ(2, results["facet_counts"].size());

    ASSERT_EQ(2, results["facet_counts"][0]["counts"].size());
    ASSERT_EQ("red", results["facet_counts"][0]["counts"][0]["value"]);
    ASSERT_EQ("blue", results["facet_counts"][0]["counts"][1]["value"]);

    ASSERT_EQ(2, results["facet_counts"][1]["counts"].size());
    ASSERT_EQ("{\"b\":255,\"color\":\"blue\",\"g\":0,\"r\":0}", results["facet_counts"][1]["counts"][0]["parent"].dump());
    ASSERT_EQ("0", results["facet_counts"][1]["counts"][0]["value"]);
    ASSERT_EQ("{\"b\":0,\"color\":\"red\",\"g\":0,\"r\":255}", results["facet_counts"][1]["counts"][1]["parent"].dump());
    ASSERT_EQ("255", results["facet_counts"][1]["counts"][1]["value"]);
}

TEST_F(CollectionFacetingTest, FacetingReturnParentDeepNested) {
    nlohmann::json schema = R"({
        "name": "coll1",
        "enable_nested_fields": true,
        "fields": [
          {"name": "product.specification.detail.width", "type": "int32", "optional": false, "facet": true }
        ]
    })"_json;

    auto op = collectionManager.create_collection(schema);
    ASSERT_TRUE(op.ok());
    Collection* coll1 = op.get();

    nlohmann::json doc1 = R"({
       "product" : {
            "specification": {
                "detail" : {
                    "width": 25
                }
            }
        }
    })"_json;

    nlohmann::json doc2 = R"({
        "product" : {
            "specification": {
                "detail" : {
                    "width": 30
                }
            }
        }
    })"_json;

    auto add_op = coll1->add(doc1.dump(), CREATE);
    ASSERT_TRUE(add_op.ok());
    add_op = coll1->add(doc2.dump(), CREATE);
    ASSERT_TRUE(add_op.ok());

    auto search_op = coll1->search("*", {},"", {"product.specification.detail.width"},
                                   {}, {2}, 10, 1,FREQUENCY, {true},
                                   1, spp::sparse_hash_set<std::string>(),
                                   spp::sparse_hash_set<std::string>(),10, "",
                                   30, 4, "",
                                   Index::TYPO_TOKENS_THRESHOLD, "", "",{},
                                   3, "<mark>", "</mark>", {},
                                   UINT32_MAX, true, false, true,
                                   "", false, 6000*1000, 4, 7,
                                   fallback, 4, {off}, INT16_MAX, INT16_MAX,
                                   2, 2, false, "",
                                   true, 0, max_score, 100,
                                   0, 0, HASH, 30000,
                                   2, "", {"product.specification.detail.width"});

    if(!search_op.ok()) {
        LOG(ERROR) << search_op.error();
        FAIL();
    }
    auto results = search_op.get();
    ASSERT_EQ(1, results["facet_counts"].size());
    ASSERT_EQ(2, results["facet_counts"][0]["counts"].size());
    ASSERT_EQ("{\"specification\":{\"detail\":{\"width\":30}}}", results["facet_counts"][0]["counts"][0]["parent"].dump());
    ASSERT_EQ("30", results["facet_counts"][0]["counts"][0]["value"]);
    ASSERT_EQ("{\"specification\":{\"detail\":{\"width\":25}}}", results["facet_counts"][0]["counts"][1]["parent"].dump());
    ASSERT_EQ("25", results["facet_counts"][0]["counts"][1]["value"]);
}

TEST_F(CollectionFacetingTest, FacetingReturnParentObject) {
    nlohmann::json schema = R"({
        "name": "coll1",
        "enable_nested_fields": true,
        "fields": [
          {"name": "value", "type": "object", "optional": false, "facet": true }
        ]
    })"_json;

    auto op = collectionManager.create_collection(schema);
    ASSERT_TRUE(op.ok());
    Collection* coll1 = op.get();

    nlohmann::json doc1 = R"({
        "value": {
            "color": "red",
            "r": 255,
            "g": 0,
            "b": 0
        }
    })"_json;

    nlohmann::json doc2 = R"({
        "value": {
            "color": "blue",
            "r": 0,
            "g": 0,
            "b": 255
        }
    })"_json;

    auto add_op = coll1->add(doc1.dump(), CREATE);
    ASSERT_TRUE(add_op.ok());
    add_op = coll1->add(doc2.dump(), CREATE);
    ASSERT_TRUE(add_op.ok());

    auto search_op = coll1->search("*", {},"", {"value.color"},
                                   {}, {2}, 10, 1,FREQUENCY, {true},
                                   1, spp::sparse_hash_set<std::string>(),
                                   spp::sparse_hash_set<std::string>(),10, "",
                                   30, 4, "",
                                   Index::TYPO_TOKENS_THRESHOLD, "", "",{},
                                   3, "<mark>", "</mark>", {},
                                   UINT32_MAX, true, false, true,
                                   "", false, 6000*1000, 4, 7,
                                   fallback, 4, {off}, INT16_MAX, INT16_MAX,
                                   2, 2, false, "",
                                   true, 0, max_score, 100,
                                   0, 0, HASH, 30000,
                                   2, "", {"value.color"});

    if(!search_op.ok()) {
        LOG(ERROR) << search_op.error();
        FAIL();
    }
    auto results = search_op.get();
    ASSERT_EQ(1, results["facet_counts"].size());
    ASSERT_EQ(2, results["facet_counts"][0]["counts"].size());
    ASSERT_EQ("{\"b\":0,\"color\":\"red\",\"g\":0,\"r\":255}", results["facet_counts"][0]["counts"][0]["parent"].dump());
    ASSERT_EQ("red", results["facet_counts"][0]["counts"][0]["value"]);
    ASSERT_EQ("{\"b\":255,\"color\":\"blue\",\"g\":0,\"r\":0}", results["facet_counts"][0]["counts"][1]["parent"].dump());
    ASSERT_EQ("blue", results["facet_counts"][0]["counts"][1]["value"]);
}


TEST_F(CollectionFacetingTest, FacetSortByAlpha) {
    nlohmann::json schema = R"({
        "name": "coll1",
        "fields": [
          {"name": "phone", "type": "string", "optional": false, "facet": true },
          {"name": "brand", "type": "string", "optional": false, "facet": true },
          {"name": "rating", "type": "float", "optional": false, "facet": true }
        ]
    })"_json;

    auto op = collectionManager.create_collection(schema);
    ASSERT_TRUE(op.ok());
    Collection *coll1 = op.get();

    nlohmann::json doc;

    doc["phone"] = "Oneplus 11R";
    doc["brand"] = "Oneplus";
    doc["rating"] = 4.6;
    auto add_op = coll1->add(doc.dump(), CREATE);
    ASSERT_TRUE(add_op.ok());

    doc["phone"] = "Fusion Plus";
    doc["brand"] = "Moto";
    doc["rating"] = 4.2;
    add_op = coll1->add(doc.dump(), CREATE);
    ASSERT_TRUE(add_op.ok());

    doc["phone"] = "S22 Ultra";
    doc["brand"] = "Samsung";
    doc["rating"] = 4.1;
    add_op = coll1->add(doc.dump(), CREATE);
    ASSERT_TRUE(add_op.ok());

    doc["phone"] = "GT Master";
    doc["brand"] = "Realme";
    doc["rating"] = 4.4;
    add_op = coll1->add(doc.dump(), CREATE);
    ASSERT_TRUE(add_op.ok());

    doc["phone"] = "T2";
    doc["brand"] = "Vivo";
    doc["rating"] = 4.0;
    add_op = coll1->add(doc.dump(), CREATE);
    ASSERT_TRUE(add_op.ok());

    doc["phone"] = "Mi 6";
    doc["brand"] = "Xiaomi";
    doc["rating"] = 3.9;
    add_op = coll1->add(doc.dump(), CREATE);
    ASSERT_TRUE(add_op.ok());

    doc["phone"] = "Z6 Lite";
    doc["brand"] = "Iqoo";
    doc["rating"] = 4.3;
    add_op = coll1->add(doc.dump(), CREATE);
    ASSERT_TRUE(add_op.ok());

    //sort facets by phone in asc order
    auto search_op = coll1->search("*", {}, "", {"phone(sort_by:_alpha:asc)"},
                                   {}, {2});

    if (!search_op.ok()) {
        LOG(ERROR) << search_op.error();
        FAIL();
    }

    auto results = search_op.get();
    ASSERT_EQ(1, results["facet_counts"].size());
    ASSERT_EQ(7, results["facet_counts"][0]["counts"].size());
    ASSERT_EQ("Fusion Plus", results["facet_counts"][0]["counts"][0]["value"]);
    ASSERT_EQ("GT Master", results["facet_counts"][0]["counts"][1]["value"]);
    ASSERT_EQ("Mi 6", results["facet_counts"][0]["counts"][2]["value"]);
    ASSERT_EQ("Oneplus 11R", results["facet_counts"][0]["counts"][3]["value"]);
    ASSERT_EQ("S22 Ultra", results["facet_counts"][0]["counts"][4]["value"]);
    ASSERT_EQ("T2", results["facet_counts"][0]["counts"][5]["value"]);
    ASSERT_EQ("Z6 Lite", results["facet_counts"][0]["counts"][6]["value"]);

    //sort facets by brand in desc order
    search_op = coll1->search("*", {}, "", {"brand(sort_by:_alpha:desc)"},
                              {}, {2});

    if (!search_op.ok()) {
        LOG(ERROR) << search_op.error();
        FAIL();
    }

    results = search_op.get();
    ASSERT_EQ(1, results["facet_counts"].size());
    ASSERT_EQ(7, results["facet_counts"][0]["counts"].size());
    ASSERT_EQ("Xiaomi", results["facet_counts"][0]["counts"][0]["value"]);
    ASSERT_EQ("Vivo", results["facet_counts"][0]["counts"][1]["value"]);
    ASSERT_EQ("Samsung", results["facet_counts"][0]["counts"][2]["value"]);
    ASSERT_EQ("Realme", results["facet_counts"][0]["counts"][3]["value"]);
    ASSERT_EQ("Oneplus", results["facet_counts"][0]["counts"][4]["value"]);
    ASSERT_EQ("Moto", results["facet_counts"][0]["counts"][5]["value"]);
    ASSERT_EQ("Iqoo", results["facet_counts"][0]["counts"][6]["value"]);

    //sort facets by brand in desc order and phone by asc order
    search_op = coll1->search("*", {}, "", {"brand(sort_by:_alpha:desc)",
                                            "phone(sort_by:_alpha:asc)"},
                              {}, {2});

    if (!search_op.ok()) {
        LOG(ERROR) << search_op.error();
        FAIL();
    }

    results = search_op.get();
    ASSERT_EQ(2, results["facet_counts"].size());

    ASSERT_EQ(7, results["facet_counts"][0]["counts"].size());
    ASSERT_EQ("Xiaomi", results["facet_counts"][0]["counts"][0]["value"]);
    ASSERT_EQ("Vivo", results["facet_counts"][0]["counts"][1]["value"]);
    ASSERT_EQ("Samsung", results["facet_counts"][0]["counts"][2]["value"]);
    ASSERT_EQ("Realme", results["facet_counts"][0]["counts"][3]["value"]);
    ASSERT_EQ("Oneplus", results["facet_counts"][0]["counts"][4]["value"]);
    ASSERT_EQ("Moto", results["facet_counts"][0]["counts"][5]["value"]);
    ASSERT_EQ("Iqoo", results["facet_counts"][0]["counts"][6]["value"]);

    ASSERT_EQ(7, results["facet_counts"][1]["counts"].size());
    ASSERT_EQ("Fusion Plus", results["facet_counts"][1]["counts"][0]["value"]);
    ASSERT_EQ("GT Master", results["facet_counts"][1]["counts"][1]["value"]);
    ASSERT_EQ("Mi 6", results["facet_counts"][1]["counts"][2]["value"]);
    ASSERT_EQ("Oneplus 11R", results["facet_counts"][1]["counts"][3]["value"]);
    ASSERT_EQ("S22 Ultra", results["facet_counts"][1]["counts"][4]["value"]);
    ASSERT_EQ("T2", results["facet_counts"][1]["counts"][5]["value"]);
    ASSERT_EQ("Z6 Lite", results["facet_counts"][1]["counts"][6]["value"]);
}

TEST_F(CollectionFacetingTest, FacetSortByOtherField) {
    nlohmann::json schema = R"({
        "name": "coll1",
        "enable_nested_fields": true,
        "fields": [
          {"name": "receipe", "type": "object", "optional": false, "facet": true }
        ]
    })"_json;

    auto op = collectionManager.create_collection(schema);
    ASSERT_TRUE(op.ok());
    Collection* coll1 = op.get();

    nlohmann::json doc1 = R"({
        "receipe": {
            "name": "cheese pizza",
            "calories": 300,
            "origin": "america"
        }
    })"_json;

    auto add_op = coll1->add(doc1.dump(), CREATE);
    ASSERT_TRUE(add_op.ok());

    nlohmann::json doc2 = R"({
          "receipe": {
            "name": "noodles",
            "calories": 250,
            "origin": "china"
        }
    })"_json;

    add_op = coll1->add(doc2.dump(), CREATE);
    ASSERT_TRUE(add_op.ok());

    nlohmann::json doc3 = R"({
          "receipe": {
            "name": "hamburger",
            "calories": 350,
            "origin": "america"
        }
    })"_json;

    add_op = coll1->add(doc3.dump(), CREATE);
    ASSERT_TRUE(add_op.ok());

    nlohmann::json doc4 = R"({
          "receipe": {
            "name": "schezwan rice",
            "calories": 150,
            "origin": "china"
        }
    })"_json;

    add_op = coll1->add(doc4.dump(), CREATE);
    ASSERT_TRUE(add_op.ok());

    nlohmann::json doc5 = R"({
          "receipe": {
            "name": "butter chicken",
            "calories": 270,
            "origin": "india"
        }
    })"_json;

    add_op = coll1->add(doc5.dump(), CREATE);
    ASSERT_TRUE(add_op.ok());

    //search by calories in asc order
    auto search_op = coll1->search("*", {},"",
                                   {"receipe.name(sort_by:receipe.calories:asc)"},
                                   {}, {2});

    if(!search_op.ok()) {
        LOG(ERROR) << search_op.error();
        FAIL();
    }
    auto results = search_op.get();

    ASSERT_EQ(1, results["facet_counts"].size());
    ASSERT_EQ(5, results["facet_counts"][0]["counts"].size());
    ASSERT_EQ("schezwan rice", results["facet_counts"][0]["counts"][0]["value"]);
    ASSERT_EQ("noodles", results["facet_counts"][0]["counts"][1]["value"]);
    ASSERT_EQ("butter chicken", results["facet_counts"][0]["counts"][2]["value"]);
    ASSERT_EQ("cheese pizza", results["facet_counts"][0]["counts"][3]["value"]);
    ASSERT_EQ("hamburger", results["facet_counts"][0]["counts"][4]["value"]);

    //search by calories in desc order
    search_op = coll1->search("*", {},"",
                                   {"receipe.name(sort_by:receipe.calories:desc)"},
                                   {}, {2});

    if(!search_op.ok()) {
        LOG(ERROR) << search_op.error();
        FAIL();
    }
    results = search_op.get();

    ASSERT_EQ(1, results["facet_counts"].size());
    ASSERT_EQ(5, results["facet_counts"][0]["counts"].size());
    ASSERT_EQ("hamburger", results["facet_counts"][0]["counts"][0]["value"]);
    ASSERT_EQ("cheese pizza", results["facet_counts"][0]["counts"][1]["value"]);
    ASSERT_EQ("butter chicken", results["facet_counts"][0]["counts"][2]["value"]);
    ASSERT_EQ("noodles", results["facet_counts"][0]["counts"][3]["value"]);
    ASSERT_EQ("schezwan rice", results["facet_counts"][0]["counts"][4]["value"]);
}

TEST_F(CollectionFacetingTest, FacetSortByOtherFloatField) {
    nlohmann::json schema = R"({
        "name": "coll1",
        "enable_nested_fields": true,
        "fields": [
          {"name": "investment", "type": "object", "optional": false, "facet": true }
        ]
    })"_json;

    auto op = collectionManager.create_collection(schema);
    ASSERT_TRUE(op.ok());
    Collection* coll1 = op.get();

    nlohmann::json doc1 = R"({
        "investment": {
            "name": "Term Deposits",
            "interest_rate": 7.1,
            "class": "fixed"
        }
    })"_json;

    auto add_op = coll1->add(doc1.dump(), CREATE);
    ASSERT_TRUE(add_op.ok());

    nlohmann::json doc2 = R"({
         "investment": {
            "name": "Gold",
            "interest_rate": 5.4,
            "class": "fixed"
        }
    })"_json;

    add_op = coll1->add(doc2.dump(), CREATE);
    ASSERT_TRUE(add_op.ok());

    nlohmann::json doc3 = R"({
          "investment": {
            "name": "Mutual Funds",
            "interest_rate": 12,
            "class": "Equity"
        }
    })"_json;

    add_op = coll1->add(doc3.dump(), CREATE);
    ASSERT_TRUE(add_op.ok());

    nlohmann::json doc4 = R"({
          "investment": {
            "name": "Land",
            "interest_rate": 9.1,
            "class": "real estate"
        }
    })"_json;

    add_op = coll1->add(doc4.dump(), CREATE);
    ASSERT_TRUE(add_op.ok());

    nlohmann::json doc5 = R"({
          "investment": {
            "name": "Bonds",
            "interest_rate": 7.24,
            "class": "g-sec"
        }
    })"_json;

    add_op = coll1->add(doc5.dump(), CREATE);
    ASSERT_TRUE(add_op.ok());

    //search by calories in asc order
    auto search_op = coll1->search("*", {},"",
                                   {"investment.name(sort_by:investment.interest_rate:asc)"},
                                   {}, {2});

    if(!search_op.ok()) {
        LOG(ERROR) << search_op.error();
        FAIL();
    }
    auto results = search_op.get();

    ASSERT_EQ(1, results["facet_counts"].size());
    ASSERT_EQ(5, results["facet_counts"][0]["counts"].size());
    ASSERT_EQ("Gold", results["facet_counts"][0]["counts"][0]["value"]);
    ASSERT_EQ("Term Deposits", results["facet_counts"][0]["counts"][1]["value"]);
    ASSERT_EQ("Bonds", results["facet_counts"][0]["counts"][2]["value"]);
    ASSERT_EQ("Land", results["facet_counts"][0]["counts"][3]["value"]);
    ASSERT_EQ("Mutual Funds", results["facet_counts"][0]["counts"][4]["value"]);

    //search by calories in desc order
    search_op = coll1->search("*", {},"",
                              {"investment.name(sort_by:investment.interest_rate:desc)"},
                              {}, {2});

    if(!search_op.ok()) {
        LOG(ERROR) << search_op.error();
        FAIL();
    }
    results = search_op.get();

    ASSERT_EQ(1, results["facet_counts"].size());
    ASSERT_EQ(5, results["facet_counts"][0]["counts"].size());
    ASSERT_EQ("Mutual Funds", results["facet_counts"][0]["counts"][0]["value"]);
    ASSERT_EQ("Land", results["facet_counts"][0]["counts"][1]["value"]);
    ASSERT_EQ("Bonds", results["facet_counts"][0]["counts"][2]["value"]);
    ASSERT_EQ("Term Deposits", results["facet_counts"][0]["counts"][3]["value"]);
    ASSERT_EQ("Gold", results["facet_counts"][0]["counts"][4]["value"]);
}

TEST_F(CollectionFacetingTest, FacetSortValidation) {
    nlohmann::json schema = R"({
        "name": "coll1",
        "fields": [
          {"name": "phone", "type": "string", "optional": false, "facet": true },
          {"name": "brand", "type": "string", "optional": false, "facet": true },
          {"name": "rating", "type": "float", "optional": false, "facet": true }
        ]
    })"_json;

    auto op = collectionManager.create_collection(schema);
    ASSERT_TRUE(op.ok());
    Collection *coll1 = op.get();

    nlohmann::json doc;

    doc["phone"] = "Oneplus 11R";
    doc["brand"] = "Oneplus";
    doc["rating"] = 4.6;
    auto add_op = coll1->add(doc.dump(), CREATE);
    ASSERT_TRUE(add_op.ok());

    doc["phone"] = "Fusion Plus";
    doc["brand"] = "Moto";
    doc["rating"] = 4.2;
    add_op = coll1->add(doc.dump(), CREATE);
    ASSERT_TRUE(add_op.ok());

    doc["phone"] = "S22 Ultra";
    doc["brand"] = "Samsung";
    doc["rating"] = 4.1;
    add_op = coll1->add(doc.dump(), CREATE);
    ASSERT_TRUE(add_op.ok());

    //try sort on non string field
    auto search_op = coll1->search("*", {}, "", {"rating(sort_by:_alpha:desc)"},
                              {}, {2});

    ASSERT_EQ(400, search_op.code());
    ASSERT_EQ("Facet field should be string type to apply alpha sort.", search_op.error());

    //try sort by string field
    search_op = coll1->search("*", {}, "", {"phone(sort_by:brand:desc)"},
                              {}, {2});

    ASSERT_EQ(400, search_op.code());
    ASSERT_EQ("Sort field should be non string type to apply sort.", search_op.error());

    //incorrect syntax
    search_op = coll1->search("*", {}, "", {"phone(sort_by:desc)"},
                              {}, {2});

    ASSERT_EQ(400, search_op.code());
    ASSERT_EQ("Invalid sort format.", search_op.error());

    search_op = coll1->search("*", {}, "", {"phone(sort:_alpha:desc)"},
                              {}, {2});

    ASSERT_EQ(400, search_op.code());
    ASSERT_EQ("Invalid sort format.", search_op.error());

    //invalid param
    search_op = coll1->search("*", {}, "", {"phone(sort_by:_alpha:foo)"},
                              {}, {2});

    ASSERT_EQ(400, search_op.code());
    ASSERT_EQ("Invalid sort param.", search_op.error());

    //whitespace is allowed
    search_op = coll1->search("*", {}, "", {"phone(  sort_by: _alpha : asc)"},
                              {}, {2});

    if (!search_op.ok()) {
        LOG(ERROR) << search_op.error();
        FAIL();
    }

    auto results = search_op.get();
    ASSERT_EQ(1, results["facet_counts"].size());
    ASSERT_EQ(3, results["facet_counts"][0]["counts"].size());
    ASSERT_EQ("Fusion Plus", results["facet_counts"][0]["counts"][0]["value"]);
    ASSERT_EQ("Oneplus 11R", results["facet_counts"][0]["counts"][1]["value"]);
    ASSERT_EQ("S22 Ultra", results["facet_counts"][0]["counts"][2]["value"]);

    //facet sort with facet query should work
    search_op = coll1->search("*", query_fields, "", {"phone(sort_by:_alpha:desc)"},
                              sort_fields, {0}, 10, 1, FREQUENCY,{false},
                              Index::DROP_TOKENS_THRESHOLD,spp::sparse_hash_set<std::string>(),
                              spp::sparse_hash_set<std::string>(),10, "phone: plus",
                              30UL, 4UL,"", 1UL,
                              "", "", {}, 3UL, "<mark>",
                              "</mark>", {},4294967295UL, true,
                              false, true, "", false, 6000000UL,
                              4UL,7UL, fallback, 4UL, {off}, 32767UL,
                              32767UL, 2UL, 2UL, false,
                              "", true, 0UL, max_score, 100UL,
                              0UL, 4294967295UL, HASH);

    results = search_op.get();
    ASSERT_EQ(1, results["facet_counts"].size());
    ASSERT_EQ(1, results["facet_counts"][0]["counts"].size());
    ASSERT_EQ("Fusion Plus", results["facet_counts"][0]["counts"][0]["value"]);
}

TEST_F(CollectionFacetingTest, FacetQueryWithDifferentLocale) {
    nlohmann::json schema = R"({
        "name": "coll1",
        "fields": [
          {"name": "phone", "type": "string", "optional": false, "facet": true },
          {"name": "brand", "type": "string", "optional": false, "facet": true },
          {"name": "rating", "type": "float", "optional": false, "facet": true }
        ]
    })"_json;

    auto op = collectionManager.create_collection(schema);
    ASSERT_TRUE(op.ok());
    Collection *coll1 = op.get();

    nlohmann::json doc;
    doc["phone"] = "çapeta";
    doc["brand"] = "Samsung";
    doc["rating"] = 4.1;
    auto add_op = coll1->add(doc.dump(), CREATE);
    ASSERT_TRUE(add_op.ok());

    doc["phone"] = "teléfono justo";
    doc["brand"] = "Oneplus";
    doc["rating"] = 4.6;
    add_op = coll1->add(doc.dump(), CREATE);
    ASSERT_TRUE(add_op.ok());

    auto search_op = coll1->search("*", query_fields, "", {"phone(sort_by:_alpha:desc)"},
                              sort_fields, {0}, 10, 1, FREQUENCY,{false},
                              Index::DROP_TOKENS_THRESHOLD,spp::sparse_hash_set<std::string>(),
                              spp::sparse_hash_set<std::string>(),10, "phone: ç",
                              30UL, 4UL,"", 1UL,
                              "", "", {}, 3UL, "<mark>",
                              "</mark>", {},4294967295UL, true,
                              false, true, "", false, 6000000UL,
                              4UL,7UL, fallback, 4UL, {off}, 32767UL,
                              32767UL, 2UL, 2UL, false,
                              "", true, 0UL, max_score, 100UL,
                              0UL, 4294967295UL, HASH);

    auto results = search_op.get();
    ASSERT_EQ(1, results["facet_counts"].size());
    ASSERT_EQ(1, results["facet_counts"][0]["counts"].size());
    ASSERT_EQ("çapeta", results["facet_counts"][0]["counts"][0]["value"]);

    search_op = coll1->search("*", query_fields, "", {"phone(sort_by:_alpha:desc)"},
                                   sort_fields, {0}, 10, 1, FREQUENCY,{false},
                                   Index::DROP_TOKENS_THRESHOLD,spp::sparse_hash_set<std::string>(),
                                   spp::sparse_hash_set<std::string>(),10, "phone: telé",
                                   30UL, 4UL,"", 1UL,
                                   "", "", {}, 3UL, "<mark>",
                                   "</mark>", {},4294967295UL, true,
                                   false, true, "", false, 6000000UL,
                                   4UL,7UL, fallback, 4UL, {off}, 32767UL,
                                   32767UL, 2UL, 2UL, false,
                                   "", true, 0UL, max_score, 100UL,
                                   0UL, 4294967295UL, HASH);

    results = search_op.get();
    ASSERT_EQ(1, results["facet_counts"].size());
    ASSERT_EQ(1, results["facet_counts"][0]["counts"].size());
    ASSERT_EQ("teléfono justo", results["facet_counts"][0]["counts"][0]["value"]);
}

TEST_F(CollectionFacetingTest, FhashInt64MapTest) {
    std::vector<int64_t> visitors = {227489798, 124098972, 180247624};
    facet_index_t facet_index_v4;
    std::unordered_map<facet_value_id_t, std::vector<uint32_t>, facet_value_id_t::Hash> fvalue_to_seq_ids;
    std::unordered_map<uint32_t, std::vector<facet_value_id_t>> seq_id_to_fvalues;

    facet_index_v4.initialize("visitors");

    //insert timestamps
    int seq_id = 0;
    for (auto it = visitors.begin(); it != visitors.end(); ++it) {
        auto val = std::to_string(*it);
        facet_value_id_t facet_value_id(val);
        fvalue_to_seq_ids[facet_value_id].push_back(seq_id);
        seq_id_to_fvalues[seq_id].push_back(facet_value_id);
        ++seq_id;
    }

    facet_index_v4.insert("visitors", fvalue_to_seq_ids, seq_id_to_fvalues);
    ASSERT_EQ(3, facet_index_v4.get_fhash_int64_map("visitors").size());

    field visitorsf("visitors", field_types::INT64, true);
    nlohmann::json doc;
    doc["visitors"] = 227489798;

    facet_index_v4.remove(doc, visitorsf, 0);
    ASSERT_EQ(2, facet_index_v4.get_fhash_int64_map("visitors").size());

    fvalue_to_seq_ids.clear();
    seq_id_to_fvalues.clear();

    facet_value_id_t facet_value_id("124798721");
    fvalue_to_seq_ids[facet_value_id].push_back(seq_id);
    seq_id_to_fvalues[seq_id].push_back(facet_value_id);

    facet_index_v4.insert("visitors", fvalue_to_seq_ids, seq_id_to_fvalues);
    ASSERT_EQ(3, facet_index_v4.get_fhash_int64_map("visitors").size());
}

TEST_F(CollectionFacetingTest, RangeFacetTestWithGroupBy) {
    std::vector<field> fields = {field("place", field_types::STRING, false),
                                 field("state", field_types::STRING, true),
                                 field("visitors", field_types::INT32, true),
                                 field("rating", field_types::FLOAT, true),
                                 field("trackingFrom", field_types::INT32, true),};
    Collection* coll1 = collectionManager.create_collection(
            "coll1", 1, fields, "", 0, "", {}, {}
    ).get();
    nlohmann::json doc1;
    doc1["id"] = "0";
    doc1["place"] = "Mysore Palace";
    doc1["state"] = "Karnataka";
    doc1["visitors"] = 235486;
    doc1["rating"] = 4.5;
    doc1["trackingFrom"] = 1900;

    nlohmann::json doc2;
    doc2["id"] = "1";
    doc2["place"] = "Hampi";
    doc2["state"] = "Karnataka";
    doc2["visitors"] = 201022;
    doc2["rating"] = 4.5;
    doc2["trackingFrom"] = 1900;

    nlohmann::json doc3;
    doc3["id"] = "2";
    doc3["place"] = "Mahabalipuram";
    doc3["state"] = "TamilNadu";
    doc3["visitors"] = 174684;
    doc3["rating"] = 3.8;
    doc3["trackingFrom"] = 1900;

    nlohmann::json doc4;
    doc4["id"] = "3";
    doc4["place"] = "Meenakshi Amman Temple";
    doc4["state"] = "TamilNadu";
    doc4["visitors"] = 246676;
    doc4["rating"] = 4.5;
    doc4["trackingFrom"] = 2000;

    nlohmann::json doc5;
    doc5["id"] = "4";
    doc5["place"] = "Staue of Unity";
    doc5["state"] = "Gujarat";
    doc5["visitors"] = 345878;
    doc5["rating"] = 3.8;
    doc5["trackingFrom"] = 2000;

    ASSERT_TRUE(coll1->add(doc1.dump()).ok());
    ASSERT_TRUE(coll1->add(doc2.dump()).ok());
    ASSERT_TRUE(coll1->add(doc3.dump()).ok());
    ASSERT_TRUE(coll1->add(doc4.dump()).ok());
    ASSERT_TRUE(coll1->add(doc5.dump()).ok());

    auto result = coll1->search("Karnataka", {"state"},
                                "", {"visitors(Busy:[0, 200000], VeryBusy:[200000, 500000])"},
                                {}, {2}, 10,
                                1, FREQUENCY, {true},
                                10, spp::sparse_hash_set<std::string>(),
                                spp::sparse_hash_set<std::string>(), 10, "", 30, 4, "", 10, {}, {}, {}, 0,
                                "<mark>", "</mark>", {}, 1000,
                                true, false, true, "", true);
    if(!result.ok()) {
        LOG(INFO) << result.error();
    }

    auto results = result.get();

    ASSERT_EQ(1, results["facet_counts"][0]["counts"].size());
    ASSERT_EQ(2, (int) results["facet_counts"][0]["counts"][0]["count"]);
    ASSERT_EQ("VeryBusy", results["facet_counts"][0]["counts"][0]["value"].get<std::string>());

    //apply group_by
    result = coll1->search("*", {"state"},
                           "", {"visitors(Busy:[0, 200000], VeryBusy:[200000, 500000])"},
                           {}, {2}, 10,
                           1, FREQUENCY, {true},
                           10, spp::sparse_hash_set<std::string>(),
                           spp::sparse_hash_set<std::string>(), 10, "",
                           30, 4, "", 10,
                           {}, {}, {"rating"}, 10,"<mark>",
                           "</mark>", {}, 1000,true,
                           false, true, "", true);

    if(!result.ok()) {
        LOG(INFO) << result.error();
    }

    results = result.get();

    ASSERT_EQ(2, results["facet_counts"][0]["counts"].size());
    ASSERT_EQ(2, (int) results["facet_counts"][0]["counts"][0]["count"]);
    ASSERT_EQ("VeryBusy", results["facet_counts"][0]["counts"][0]["value"].get<std::string>());
    ASSERT_EQ(1, (int) results["facet_counts"][0]["counts"][1]["count"]);
    ASSERT_EQ("Busy", results["facet_counts"][0]["counts"][1]["value"].get<std::string>());

    ASSERT_EQ(2, results["grouped_hits"].size());

    ASSERT_EQ(2, results["grouped_hits"][0]["hits"].size());
    ASSERT_EQ(3.8, results["grouped_hits"][0]["group_key"][0]);

    ASSERT_EQ(3, results["grouped_hits"][1]["hits"].size());
    ASSERT_EQ(4.5, results["grouped_hits"][1]["group_key"][0]);

    collectionManager.drop_collection("coll1");
}

TEST_F(CollectionFacetingTest, RangeFacetAlphanumericLabels) {
    std::vector<field> fields = {field("monuments", field_types::STRING, false),
                                 field("year", field_types::INT32, true),};
    Collection* coll1 = collectionManager.create_collection(
            "coll1", 1, fields, "", 0, "",
            {},{}).get();

    nlohmann::json doc;
    doc["id"] = "0";
    doc["monuments"] = "Statue Of Unity";
    doc["year"] = 2018;
    ASSERT_TRUE(coll1->add(doc.dump()).ok());

    doc["id"] = "1";
    doc["monuments"] = "Taj Mahal";
    doc["year"] = 1653;
    ASSERT_TRUE(coll1->add(doc.dump()).ok());

    doc["id"] = "2";
    doc["monuments"] = "Mysore Palace";
    doc["year"] = 1897;
    ASSERT_TRUE(coll1->add(doc.dump()).ok());

    doc["id"] = "3";
    doc["monuments"] = "Chennakesava Temple";
    doc["year"] = 1117;
    ASSERT_TRUE(coll1->add(doc.dump()).ok());

    auto results = coll1->search("*", {},
                                 "", {"year(10thAD:[1000,1500], 15thAD:[1500,2000], 20thAD:[2000, ])"},
                                 {}, {2}, 10,
                                 1, FREQUENCY, {true},
                                 10, spp::sparse_hash_set<std::string>(),
                                 spp::sparse_hash_set<std::string>(), 10, "", 30, 4, "", 10, {}, {}, {}, 0,
                                 "<mark>", "</mark>", {}, 1000,
                                 true, false, true, "", true).get();

    ASSERT_EQ(3, results["facet_counts"][0]["counts"].size());
    ASSERT_EQ(2, results["facet_counts"][0]["counts"][0]["count"]);
    ASSERT_EQ("15thAD", results["facet_counts"][0]["counts"][0]["value"]);
    ASSERT_EQ(1, results["facet_counts"][0]["counts"][1]["count"]);
    ASSERT_EQ("20thAD", results["facet_counts"][0]["counts"][1]["value"]);
    ASSERT_EQ(1, results["facet_counts"][0]["counts"][2]["count"]);
    ASSERT_EQ("10thAD", results["facet_counts"][0]["counts"][2]["value"]);
}