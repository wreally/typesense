#include <gtest/gtest.h>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <collection_manager.h>
#include "collection.h"

class CollectionJoinTest : public ::testing::Test {
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

TEST_F(CollectionJoinTest, SchemaReferenceField) {
    nlohmann::json schema_json =
            R"({
                "name": "Customers",
                "fields": [
                    {"name": "product_.*", "type": "string", "reference": "Products.product_id"}
                ]
            })"_json;

    auto collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_FALSE(collection_create_op.ok());
    ASSERT_EQ("Wildcard field cannot have a reference.", collection_create_op.error());

    schema_json =
            R"({
                "name": "Customers",
                "fields": [
                    {"name": ".*", "type": "auto", "reference": "Products.product_id"}
                ]
            })"_json;

    collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_FALSE(collection_create_op.ok());
    ASSERT_EQ("Field `.*` cannot be a reference field.", collection_create_op.error());

    schema_json =
            R"({
                "name": "Customers",
                "fields": [
                    {"name": "product_id", "type": "string", "reference": 123},
                    {"name": "customer_name", "type": "string"},
                    {"name": "product_price", "type": "float"}
                ]
            })"_json;

    collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_FALSE(collection_create_op.ok());
    ASSERT_EQ("Reference should be a string.", collection_create_op.error());

    schema_json =
            R"({
                "name": "Customers",
                "fields": [
                    {"name": "product_id", "type": "string", "reference": "foo"},
                    {"name": "customer_name", "type": "string"},
                    {"name": "product_price", "type": "float"}
                ]
            })"_json;

    collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_FALSE(collection_create_op.ok());
    ASSERT_EQ("Invalid reference `foo`.", collection_create_op.error());

    schema_json =
            R"({
                "name": "Customers",
                "fields": [
                    {"name": "Object.object.field", "type": "string", "reference": "Products.product_id"},
                    {"name": "customer_name", "type": "string"},
                    {"name": "product_price", "type": "float"}
                ]
            })"_json;

    collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_FALSE(collection_create_op.ok());
    ASSERT_EQ("`Object.object.field` field cannot have a reference. Only the top-level field of an object is allowed.",
              collection_create_op.error());

    schema_json =
            R"({
                "name": "Customers",
                "fields": [
                    {"name": "product_id", "type": "string", "reference": "Products.product_id"},
                    {"name": "customer_name", "type": "string"},
                    {"name": "product_price", "type": "float"}
                ]
            })"_json;
    collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_TRUE(collection_create_op.ok());
    auto collection = collection_create_op.get();
    auto schema = collection->get_schema();

    ASSERT_EQ(schema.count("customer_name"), 1);
    ASSERT_TRUE(schema.at("customer_name").reference.empty());
    ASSERT_EQ(schema.count("product_id"), 1);
    ASSERT_FALSE(schema.at("product_id").reference.empty());

    auto reference_fields = collection->get_reference_fields();
    ASSERT_EQ(reference_fields.count("product_id"), 1);
    ASSERT_EQ(reference_fields.at("product_id").collection, "Products");
    ASSERT_EQ(reference_fields.at("product_id").field, "product_id");

    // Add a `foo_sequence_id` field in the schema for `foo` reference field.
    ASSERT_EQ(schema.count("product_id_sequence_id"), 1);
    ASSERT_TRUE(schema.at("product_id_sequence_id").index);

    collectionManager.drop_collection("Customers");
}

TEST_F(CollectionJoinTest, IndexDocumentHavingReferenceField) {
    auto customers_schema_json =
            R"({
                "name": "Customers",
                "fields": [
                    {"name": "customer_id", "type": "string"},
                    {"name": "customer_name", "type": "string"},
                    {"name": "product_price", "type": "float"},
                    {"name": "reference_id", "type": "string", "reference": "products.product_id"}
                ]
            })"_json;
    auto collection_create_op = collectionManager.create_collection(customers_schema_json);
    ASSERT_TRUE(collection_create_op.ok());
    auto customer_collection = collection_create_op.get();

    nlohmann::json customer_json = R"({
                                        "customer_id": "customer_a",
                                        "customer_name": "Joe",
                                        "product_price": 143
                                    })"_json;
    auto add_doc_op = customer_collection->add(customer_json.dump());

    ASSERT_FALSE(add_doc_op.ok());
    ASSERT_EQ("Missing the required reference field `reference_id` in the document.", add_doc_op.error());

    customer_json = R"({
                        "customer_id": "customer_a",
                        "customer_name": "Joe",
                        "product_price": 143,
                        "reference_id": "a"
                    })"_json;
    add_doc_op = customer_collection->add(customer_json.dump());
    ASSERT_FALSE(add_doc_op.ok());
    ASSERT_EQ("Referenced collection `products` not found.", add_doc_op.error());
    collectionManager.drop_collection("Customers");

    customers_schema_json =
            R"({
                "name": "Customers",
                "fields": [
                    {"name": "customer_id", "type": "string"},
                    {"name": "customer_name", "type": "string"},
                    {"name": "product_price", "type": "float"},
                    {"name": "reference_id", "type": "string", "reference": "Products.foo"}
                ]
            })"_json;
    collection_create_op = collectionManager.create_collection(customers_schema_json);
    ASSERT_TRUE(collection_create_op.ok());
    customer_collection = collection_create_op.get();

    auto products_schema_json =
            R"({
                "name": "Products",
                "fields": [
                    {"name": "product_id", "type": "string", "index": false, "optional": true},
                    {"name": "product_name", "type": "string"},
                    {"name": "product_description", "type": "string"}
                ]
            })"_json;
    collection_create_op = collectionManager.create_collection(products_schema_json);
    ASSERT_TRUE(collection_create_op.ok());

    add_doc_op = customer_collection->add(customer_json.dump());
    ASSERT_FALSE(add_doc_op.ok());
    ASSERT_EQ("Referenced field `foo` not found in the collection `Products`.", add_doc_op.error());
    collectionManager.drop_collection("Customers");

    customers_schema_json =
            R"({
                "name": "Customers",
                "fields": [
                    {"name": "customer_id", "type": "string"},
                    {"name": "customer_name", "type": "string"},
                    {"name": "product_price", "type": "float"},
                    {"name": "reference_id", "type": "string", "reference": "Products.product_id"}
                ]
            })"_json;
    collection_create_op = collectionManager.create_collection(customers_schema_json);
    ASSERT_TRUE(collection_create_op.ok());

    customer_collection = collection_create_op.get();
    add_doc_op = customer_collection->add(customer_json.dump());
    ASSERT_FALSE(add_doc_op.ok());
    ASSERT_EQ("Referenced field `product_id` in the collection `Products` must be indexed.", add_doc_op.error());

    collectionManager.drop_collection("Products");
    products_schema_json =
            R"({
                "name": "Products",
                "fields": [
                    {"name": "product_id", "type": "string"},
                    {"name": "product_name", "type": "string"},
                    {"name": "product_description", "type": "string"}
                ]
            })"_json;
    collection_create_op = collectionManager.create_collection(products_schema_json);
    ASSERT_TRUE(collection_create_op.ok());

    add_doc_op = customer_collection->add(customer_json.dump());
    ASSERT_EQ("Reference document having `product_id:= a` not found in the collection `Products`.", add_doc_op.error());

    std::vector<nlohmann::json> products = {
            R"({
                "product_id": "product_a",
                "product_name": "shampoo",
                "product_description": "Our new moisturizing shampoo is perfect for those with dry or damaged hair."
            })"_json,
            R"({
                "product_id": "product_a",
                "product_name": "soap",
                "product_description": "Introducing our all-natural, organic soap bar made with essential oils and botanical ingredients."
            })"_json
    };
    for (auto const &json: products){
        auto add_op = collection_create_op.get()->add(json.dump());
        if (!add_op.ok()) {
            LOG(INFO) << add_op.error();
        }
        ASSERT_TRUE(add_op.ok());
    }

    customer_json["reference_id"] = "product_a";
    add_doc_op = customer_collection->add(customer_json.dump());
    ASSERT_EQ("Multiple documents having `product_id:= product_a` found in the collection `Products`.", add_doc_op.error());

    collectionManager.drop_collection("Products");
    products[1]["product_id"] = "product_b";
    products_schema_json =
            R"({
                "name": "Products",
                "fields": [
                    {"name": "product_id", "type": "string"},
                    {"name": "product_name", "type": "string"},
                    {"name": "product_description", "type": "string"}
                ]
            })"_json;
    collection_create_op = collectionManager.create_collection(products_schema_json);
    ASSERT_TRUE(collection_create_op.ok());
    for (auto const &json: products){
        auto add_op = collection_create_op.get()->add(json.dump());
        if (!add_op.ok()) {
            LOG(INFO) << add_op.error();
        }
        ASSERT_TRUE(add_op.ok());
    }

    collectionManager.drop_collection("Customers");
    customers_schema_json =
            R"({
                "name": "Customers",
                "fields": [
                    {"name": "customer_id", "type": "string"},
                    {"name": "customer_name", "type": "string"},
                    {"name": "product_price", "type": "float"},
                    {"name": "reference_id", "type": "string", "reference": "Products.product_id"}
                ]
            })"_json;
    collection_create_op = collectionManager.create_collection(customers_schema_json);
    ASSERT_TRUE(collection_create_op.ok());

    customer_collection = collection_create_op.get();
    add_doc_op = customer_collection->add(customer_json.dump());
    ASSERT_TRUE(add_doc_op.ok());

    auto customer_doc = customer_collection->get("0").get();
    ASSERT_EQ(0, customer_doc.at("reference_id_sequence_id"));
    ASSERT_EQ(1, customer_doc.count(".ref"));
    ASSERT_EQ(1, customer_doc[".ref"].size());
    ASSERT_EQ("reference_id_sequence_id", customer_doc[".ref"].at(0));

    nlohmann::json product_doc;
    // Referenced document's sequence_id must be valid.
    auto get_op = collectionManager.get_collection("Products")->get_document_from_store(
                                                                customer_doc["reference_id_sequence_id"].get<uint32_t>(),
                                                                product_doc);
    ASSERT_TRUE(get_op.ok());
    ASSERT_EQ(product_doc.count("product_id"), 1);
    ASSERT_EQ(product_doc["product_id"], "product_a");
    ASSERT_EQ(product_doc["product_name"], "shampoo");

    auto id_ref_schema_json =
            R"({
                "name": "id_ref",
                "fields": [
                    {"name": "id_reference", "type": "string", "reference": "Products.id", "optional": true},
                    {"name": "multi_id_reference", "type": "string[]", "reference": "Products.id", "optional": true}
                ]
            })"_json;
    collection_create_op = collectionManager.create_collection(id_ref_schema_json);
    ASSERT_TRUE(collection_create_op.ok());

    auto id_ref_collection = collection_create_op.get();
    auto id_ref_json = R"({
                            "id_reference": 123
                        })"_json;
    add_doc_op = id_ref_collection->add(id_ref_json.dump());
    ASSERT_FALSE(add_doc_op.ok());
    ASSERT_EQ("Field `id_reference` must have string value.", add_doc_op.error());

    id_ref_json = R"({
                        "id_reference": "foo"
                    })"_json;
    add_doc_op = id_ref_collection->add(id_ref_json.dump());
    ASSERT_FALSE(add_doc_op.ok());
    ASSERT_EQ("Referenced document having `id: foo` not found in the collection `Products`.", add_doc_op.error());

    id_ref_json = R"({
                        "multi_id_reference": ["0", 1]
                    })"_json;
    add_doc_op = id_ref_collection->add(id_ref_json.dump());
    ASSERT_FALSE(add_doc_op.ok());
    ASSERT_EQ("Field `multi_id_reference` must have string value.", add_doc_op.error());

    id_ref_json = R"({
                        "multi_id_reference": ["0", "foo"]
                    })"_json;
    add_doc_op = id_ref_collection->add(id_ref_json.dump());
    ASSERT_FALSE(add_doc_op.ok());
    ASSERT_EQ("Referenced document having `id: foo` not found in the collection `Products`.", add_doc_op.error());

    collectionManager.drop_collection("id_ref");
    id_ref_schema_json =
            R"({
                "name": "id_ref",
                "fields": [
                    {"name": "id_reference", "type": "string", "reference": "Products.id", "optional": true},
                    {"name": "multi_id_reference", "type": "string[]", "reference": "Products.id", "optional": true}
                ]
            })"_json;
    collection_create_op = collectionManager.create_collection(id_ref_schema_json);
    ASSERT_TRUE(collection_create_op.ok());
    id_ref_collection = collection_create_op.get();

    id_ref_json = R"({
                        "id_reference": "0"
                    })"_json;
    add_doc_op = id_ref_collection->add(id_ref_json.dump());
    ASSERT_TRUE(add_doc_op.ok());

    auto doc = id_ref_collection->get("0").get();
    ASSERT_EQ(0, doc["id_reference_sequence_id"]);
    ASSERT_EQ(1, doc.count(".ref"));
    ASSERT_EQ(1, doc[".ref"].size());
    ASSERT_EQ("id_reference_sequence_id", doc[".ref"].at(0));

    id_ref_json = R"({
                        "multi_id_reference": ["1"]
                    })"_json;
    add_doc_op = id_ref_collection->add(id_ref_json.dump());
    ASSERT_TRUE(add_doc_op.ok());

    doc = id_ref_collection->get("1").get();
    ASSERT_EQ(1, doc["multi_id_reference_sequence_id"].size());
    ASSERT_EQ(1, doc["multi_id_reference_sequence_id"][0]);
    ASSERT_EQ(1, doc.count(".ref"));
    ASSERT_EQ(1, doc[".ref"].size());
    ASSERT_EQ("multi_id_reference_sequence_id", doc[".ref"][0]);

    id_ref_json = R"({
                        "multi_id_reference": ["0", "1"]
                    })"_json;
    add_doc_op = id_ref_collection->add(id_ref_json.dump());
    ASSERT_TRUE(add_doc_op.ok());

    doc = id_ref_collection->get("2").get();
    ASSERT_EQ(2, doc["multi_id_reference_sequence_id"].size());
    ASSERT_EQ(0, doc["multi_id_reference_sequence_id"][0]);
    ASSERT_EQ(1, doc["multi_id_reference_sequence_id"][1]);

    auto result = id_ref_collection->search("*", {}, "", {}, {}, {0}).get();
    ASSERT_EQ(3, result["found"].get<size_t>());
    ASSERT_EQ(3, result["hits"].size());
    ASSERT_EQ(0, result["hits"][0]["document"].count("multi_id_reference_sequence_id"));
    ASSERT_EQ(0, result["hits"][1]["document"].count("multi_id_reference_sequence_id"));
    ASSERT_EQ(0, result["hits"][2]["document"].count("id_reference_sequence_id"));

    collectionManager.drop_collection("Customers");
    collectionManager.drop_collection("Products");
    collectionManager.drop_collection("id_ref");

    auto schema_json =
            R"({
                "name": "coll1",
                "enable_nested_fields": true,
                "fields": [
                    {"name": "string_field", "type": "string", "optional": true},
                    {"name": "string_array_field", "type": "string[]", "optional": true},
                    {"name": "int32_field", "type": "int32", "optional": true},
                    {"name": "int32_array_field", "type": "int32[]", "optional": true},
                    {"name": "int64_field", "type": "int64", "optional": true},
                    {"name": "int64_array_field", "type": "int64[]", "optional": true},
                    {"name": "float_field", "type": "float", "optional": true},
                    {"name": "float_array_field", "type": "float[]", "optional": true},
                    {"name": "bool_field", "type": "bool", "optional": true},
                    {"name": "bool_array_field", "type": "bool[]", "optional": true},
                    {"name": "geopoint_field", "type": "geopoint", "optional": true},
                    {"name": "geopoint_array_field", "type": "geopoint[]", "optional": true},
                    {"name": "object_field", "type": "object", "optional": true},
                    {"name": "object_array_field", "type": "object[]", "optional": true}
                ]
            })"_json;
    collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_TRUE(collection_create_op.ok());
    auto coll1 = collection_create_op.get();

    schema_json =
            R"({
                "name": "coll2",
                "enable_nested_fields": true,
                "fields": [
                    {"name": "ref_string_field", "type": "string", "optional": true, "reference": "coll1.string_field"},
                    {"name": "ref_string_array_field", "type": "string[]", "optional": true, "reference": "coll1.string_array_field"},
                    {"name": "ref_int32_field", "type": "int32", "optional": true, "reference": "coll1.int32_field"},
                    {"name": "ref_int32_array_field", "type": "int32[]", "optional": true, "reference": "coll1.int32_array_field"},
                    {"name": "ref_int64_field", "type": "int64", "optional": true, "reference": "coll1.int64_field"},
                    {"name": "ref_int64_array_field", "type": "int64[]", "optional": true, "reference": "coll1.int64_array_field"},
                    {"name": "ref_float_field", "type": "float", "optional": true, "reference": "coll1.float_field"},
                    {"name": "ref_float_array_field", "type": "float[]", "optional": true, "reference": "coll1.float_array_field"},
                    {"name": "ref_bool_field", "type": "bool", "optional": true, "reference": "coll1.bool_field"},
                    {"name": "ref_bool_array_field", "type": "bool[]", "optional": true, "reference": "coll1.bool_array_field"},
                    {"name": "ref_geopoint_field", "type": "geopoint", "optional": true, "reference": "coll1.geopoint_field"},
                    {"name": "ref_geopoint_array_field", "type": "geopoint[]", "optional": true, "reference": "coll1.geopoint_array_field"},
                    {"name": "ref_object_field", "type": "object", "optional": true, "reference": "coll1.object_field"},
                    {"name": "ref_object_array_field", "type": "object[]", "optional": true, "reference": "coll1.object_array_field"},
                    {"name": "non_indexed_object.ref_field", "type": "string", "optional": true, "reference": "coll1.string_field"},
                    {"name": "object.ref_field", "type": "string", "optional": true, "reference": "coll1.string_field"},
                    {"name": "object.ref_array_field", "type": "string[]", "optional": true, "reference": "coll1.string_array_field"},
                    {"name": "object", "type": "object", "optional": true},
                    {"name": "object_array.ref_field", "type": "string", "optional": true, "reference": "coll1.string_field"},
                    {"name": "object_array.ref_array_field", "type": "string[]", "optional": true, "reference": "coll1.string_array_field"},
                    {"name": "object_array", "type": "object[]", "optional": true}
                ]
            })"_json;
    auto temp_json = schema_json;
    collection_create_op = collectionManager.create_collection(temp_json);
    ASSERT_TRUE(collection_create_op.ok());
    auto coll2 = collection_create_op.get();

    // string/string[] reference fields
    auto doc_json = R"({
                        "string_field": "a",
                        "string_array_field": ["b", "c"]
                    })"_json;
    add_doc_op = coll1->add(doc_json.dump());
    ASSERT_TRUE(add_doc_op.ok());
    doc_json = R"({
                    "string_field": "d",
                    "string_array_field": ["e", "f"]
                })"_json;
    add_doc_op = coll1->add(doc_json.dump());
    ASSERT_TRUE(add_doc_op.ok());

    doc_json = R"({
                    "ref_string_field": 1
                })"_json;
    add_doc_op = coll2->add(doc_json.dump());
    ASSERT_FALSE(add_doc_op.ok());
    ASSERT_EQ("Field `ref_string_field` must have `string` value.", add_doc_op.error());

    doc_json = R"({
                    "ref_string_array_field": ["a", 1]
                })"_json;
    add_doc_op = coll2->add(doc_json.dump());
    ASSERT_FALSE(add_doc_op.ok());
    ASSERT_EQ("Field `ref_string_array_field` must only have `string` values.", add_doc_op.error());

    collectionManager.drop_collection("coll2");
    temp_json = schema_json;
    collection_create_op = collectionManager.create_collection(temp_json);
    ASSERT_TRUE(collection_create_op.ok());
    coll2 = collection_create_op.get();

    doc_json = R"({
                    "ref_string_field": "d"
                })"_json;
    add_doc_op = coll2->add(doc_json.dump());
    ASSERT_TRUE(add_doc_op.ok());

    doc = coll2->get("0").get();
    ASSERT_EQ(1, doc.count("ref_string_field_sequence_id"));
    ASSERT_EQ(1, doc["ref_string_field_sequence_id"]);
    ASSERT_EQ(1, doc.count(".ref"));
    ASSERT_EQ(1, doc[".ref"].size());
    ASSERT_EQ("ref_string_field_sequence_id", doc[".ref"][0]);

    doc_json = R"({
                    "ref_string_array_field": ["foo"]
                })"_json;
    add_doc_op = coll2->add(doc_json.dump());
    ASSERT_TRUE(add_doc_op.ok());

    result = coll2->search("*", {}, "", {}, {}, {0}).get();
    ASSERT_EQ(0, result["hits"][0]["document"]["ref_string_array_field_sequence_id"].size());

    doc = coll2->get("1").get();
    ASSERT_EQ(1, doc.count("ref_string_array_field_sequence_id"));
    ASSERT_EQ(0, doc["ref_string_array_field_sequence_id"].size());
    ASSERT_EQ(1, doc.count(".ref"));
    ASSERT_EQ(1, doc[".ref"].size());
    ASSERT_EQ("ref_string_array_field_sequence_id", doc[".ref"][0]);

    doc_json = R"({
                    "ref_string_array_field": ["b", "foo"]
                })"_json;
    add_doc_op = coll2->add(doc_json.dump());
    ASSERT_TRUE(add_doc_op.ok());

    doc = coll2->get("2").get();
    ASSERT_EQ(1, doc.count("ref_string_array_field_sequence_id"));
    ASSERT_EQ(1, doc["ref_string_array_field_sequence_id"].size());
    ASSERT_EQ(0, doc["ref_string_array_field_sequence_id"][0]);

    doc_json = R"({
                    "ref_string_array_field": ["c", "e"]
                })"_json;
    add_doc_op = coll2->add(doc_json.dump());
    ASSERT_TRUE(add_doc_op.ok());

    doc = coll2->get("3").get();
    ASSERT_EQ(1, doc.count("ref_string_array_field_sequence_id"));
    ASSERT_EQ(2, doc["ref_string_array_field_sequence_id"].size());
    ASSERT_EQ(0, doc["ref_string_array_field_sequence_id"][0]);
    ASSERT_EQ(1, doc["ref_string_array_field_sequence_id"][1]);

    // int32/int32[] reference fields
    doc_json = R"({
                    "int32_field": 1
                })"_json;
    add_doc_op = coll1->add(doc_json.dump());
    ASSERT_TRUE(add_doc_op.ok());

    doc_json = R"({
                    "int32_field": 1,
                    "int32_array_field": [2, -2147483648]
                })"_json;
    add_doc_op = coll1->add(doc_json.dump());
    ASSERT_TRUE(add_doc_op.ok());

    doc_json = R"({
                    "int32_field": 4,
                    "int32_array_field": [5, 2147483647]
                })"_json;
    add_doc_op = coll1->add(doc_json.dump());
    ASSERT_TRUE(add_doc_op.ok());

    doc_json = R"({
                    "ref_int32_field": "1"
                })"_json;
    add_doc_op = coll2->add(doc_json.dump());
    ASSERT_FALSE(add_doc_op.ok());
    ASSERT_EQ("Field `ref_int32_field` must have `int32` value.", add_doc_op.error());

    doc_json = R"({
                    "ref_int32_field": 2147483648
                })"_json;
    add_doc_op = coll2->add(doc_json.dump());
    ASSERT_FALSE(add_doc_op.ok());
    ASSERT_EQ("Field `ref_int32_field` must have `int32` value.", add_doc_op.error());

    doc_json = R"({
                    "ref_int32_field": 0
                })"_json;
    add_doc_op = coll2->add(doc_json.dump());
    ASSERT_FALSE(add_doc_op.ok());
    ASSERT_EQ("Reference document having `int32_field: 0` not found in the collection `coll1`.", add_doc_op.error());

    doc_json = R"({
                    "ref_int32_field": 1
                })"_json;
    add_doc_op = coll2->add(doc_json.dump());
    ASSERT_FALSE(add_doc_op.ok());
    ASSERT_EQ("Multiple documents having `int32_field: 1` found in the collection `coll1`.", add_doc_op.error());

    doc_json = R"({
                    "ref_int32_array_field": [1, "2"]
                })"_json;
    add_doc_op = coll2->add(doc_json.dump());
    ASSERT_FALSE(add_doc_op.ok());
    ASSERT_EQ("Field `ref_int32_array_field` must only have `int32` values.", add_doc_op.error());

    doc_json = R"({
                    "ref_int32_array_field": [1, -2147483649]
                })"_json;
    add_doc_op = coll2->add(doc_json.dump());
    ASSERT_FALSE(add_doc_op.ok());
    ASSERT_EQ("Field `ref_int32_array_field` must only have `int32` values.", add_doc_op.error());

    doc_json = R"({
                    "ref_int32_array_field": [1, 2147483648]
                })"_json;
    add_doc_op = coll2->add(doc_json.dump());
    ASSERT_FALSE(add_doc_op.ok());
    ASSERT_EQ("Field `ref_int32_array_field` must only have `int32` values.", add_doc_op.error());

    collectionManager.drop_collection("coll2");
    temp_json = schema_json;
    collection_create_op = collectionManager.create_collection(temp_json);
    ASSERT_TRUE(collection_create_op.ok());
    coll2 = collection_create_op.get();

    doc_json = R"({
                    "ref_int32_field": 4
                })"_json;
    add_doc_op = coll2->add(doc_json.dump());
    ASSERT_TRUE(add_doc_op.ok());

    doc = coll2->get("0").get();
    ASSERT_EQ(1, doc.count("ref_int32_field_sequence_id"));
    ASSERT_EQ(4, doc["ref_int32_field_sequence_id"]);
    ASSERT_EQ(1, doc.count(".ref"));
    ASSERT_EQ(1, doc[".ref"].size());
    ASSERT_EQ("ref_int32_field_sequence_id", doc[".ref"][0]);

    doc_json = R"({
                    "ref_int32_array_field": [1]
                })"_json;
    add_doc_op = coll2->add(doc_json.dump());
    ASSERT_TRUE(add_doc_op.ok());

    doc = coll2->get("1").get();
    ASSERT_EQ(1, doc.count("ref_int32_array_field_sequence_id"));
    ASSERT_EQ(0, doc["ref_int32_array_field_sequence_id"].size());
    ASSERT_EQ(1, doc.count(".ref"));
    ASSERT_EQ(1, doc[".ref"].size());
    ASSERT_EQ("ref_int32_array_field_sequence_id", doc[".ref"][0]);

    doc_json = R"({
                    "ref_int32_array_field": [1, 2]
                })"_json;
    add_doc_op = coll2->add(doc_json.dump());
    ASSERT_TRUE(add_doc_op.ok());

    doc = coll2->get("2").get();
    ASSERT_EQ(1, doc.count("ref_int32_array_field_sequence_id"));
    ASSERT_EQ(1, doc["ref_int32_array_field_sequence_id"].size());
    ASSERT_EQ(3, doc["ref_int32_array_field_sequence_id"][0]);

    doc_json = R"({
                    "ref_int32_array_field": [2, 5]
                })"_json;
    add_doc_op = coll2->add(doc_json.dump());
    ASSERT_TRUE(add_doc_op.ok());

    doc = coll2->get("3").get();
    ASSERT_EQ(1, doc.count("ref_int32_array_field_sequence_id"));
    ASSERT_EQ(2, doc["ref_int32_array_field_sequence_id"].size());
    ASSERT_EQ(3, doc["ref_int32_array_field_sequence_id"][0]);
    ASSERT_EQ(4, doc["ref_int32_array_field_sequence_id"][1]);

    doc_json = R"({
                    "ref_int32_array_field": [-2147483648]
                })"_json;
    add_doc_op = coll2->add(doc_json.dump());
    ASSERT_TRUE(add_doc_op.ok());

    doc = coll2->get("4").get();
    ASSERT_EQ(1, doc.count("ref_int32_array_field_sequence_id"));
    ASSERT_EQ(1, doc["ref_int32_array_field_sequence_id"].size());
    ASSERT_EQ(3, doc["ref_int32_array_field_sequence_id"][0]);

    // int64/int64[] reference fields
    doc_json = R"({
                    "int64_field": 1
                })"_json;
    add_doc_op = coll1->add(doc_json.dump());
    ASSERT_TRUE(add_doc_op.ok());

    doc_json = R"({
                    "int64_field": 1,
                    "int64_array_field": [2, -9223372036854775808]
                })"_json;
    add_doc_op = coll1->add(doc_json.dump());
    ASSERT_TRUE(add_doc_op.ok());

    doc_json = R"({
                    "int64_field": 4,
                    "int64_array_field": [5,  9223372036854775807]
                })"_json;
    add_doc_op = coll1->add(doc_json.dump());
    ASSERT_TRUE(add_doc_op.ok());

    doc_json = R"({
                    "ref_int64_field": "1"
                })"_json;
    add_doc_op = coll2->add(doc_json.dump());
    ASSERT_FALSE(add_doc_op.ok());
    ASSERT_EQ("Field `ref_int64_field` must have `int64` value.", add_doc_op.error());

    doc_json = R"({
                    "ref_int64_field": 0
                })"_json;
    add_doc_op = coll2->add(doc_json.dump());
    ASSERT_FALSE(add_doc_op.ok());
    ASSERT_EQ("Reference document having `int64_field: 0` not found in the collection `coll1`.", add_doc_op.error());

    doc_json = R"({
                    "ref_int64_field": 1
                })"_json;
    add_doc_op = coll2->add(doc_json.dump());
    ASSERT_FALSE(add_doc_op.ok());
    ASSERT_EQ("Multiple documents having `int64_field: 1` found in the collection `coll1`.", add_doc_op.error());

    doc_json = R"({
                    "ref_int64_array_field": [1, "2"]
                })"_json;
    add_doc_op = coll2->add(doc_json.dump());
    ASSERT_FALSE(add_doc_op.ok());
    ASSERT_EQ("Field `ref_int64_array_field` must only have `int64` values.", add_doc_op.error());

    doc_json = R"({
                    "ref_int64_array_field": [1, -9223372036854775809]
                })"_json;
    add_doc_op = coll2->add(doc_json.dump());
    ASSERT_FALSE(add_doc_op.ok());
    ASSERT_EQ("Field `ref_int64_array_field` must only have `int64` values.", add_doc_op.error());

    doc_json = R"({
                    "ref_int64_array_field": [1, 1.5]
                })"_json;
    add_doc_op = coll2->add(doc_json.dump());
    ASSERT_FALSE(add_doc_op.ok());
    ASSERT_EQ("Field `ref_int64_array_field` must only have `int64` values.", add_doc_op.error());

    collectionManager.drop_collection("coll2");
    temp_json = schema_json;
    collection_create_op = collectionManager.create_collection(temp_json);
    ASSERT_TRUE(collection_create_op.ok());
    coll2 = collection_create_op.get();

    doc_json = R"({
                    "ref_int64_field": 4
                })"_json;
    add_doc_op = coll2->add(doc_json.dump());
    ASSERT_TRUE(add_doc_op.ok());

    doc = coll2->get("0").get();
    ASSERT_EQ(1, doc.count("ref_int64_field_sequence_id"));
    ASSERT_EQ(7, doc["ref_int64_field_sequence_id"]);
    ASSERT_EQ(1, doc.count(".ref"));
    ASSERT_EQ(1, doc[".ref"].size());
    ASSERT_EQ("ref_int64_field_sequence_id", doc[".ref"][0]);

    doc_json = R"({
                    "ref_int64_array_field": [1]
                })"_json;
    add_doc_op = coll2->add(doc_json.dump());
    ASSERT_TRUE(add_doc_op.ok());

    doc = coll2->get("1").get();
    ASSERT_EQ(1, doc.count("ref_int64_array_field_sequence_id"));
    ASSERT_EQ(0, doc["ref_int64_array_field_sequence_id"].size());
    ASSERT_EQ(1, doc.count(".ref"));
    ASSERT_EQ(1, doc[".ref"].size());
    ASSERT_EQ("ref_int64_array_field_sequence_id", doc[".ref"][0]);

    doc_json = R"({
                    "ref_int64_array_field": [1, 2]
                })"_json;
    add_doc_op = coll2->add(doc_json.dump());
    ASSERT_TRUE(add_doc_op.ok());

    doc = coll2->get("2").get();
    ASSERT_EQ(1, doc.count("ref_int64_array_field_sequence_id"));
    ASSERT_EQ(1, doc["ref_int64_array_field_sequence_id"].size());
    ASSERT_EQ(6, doc["ref_int64_array_field_sequence_id"][0]);

    doc_json = R"({
                    "ref_int64_array_field": [2, 5]
                })"_json;
    add_doc_op = coll2->add(doc_json.dump());
    ASSERT_TRUE(add_doc_op.ok());

    doc = coll2->get("3").get();
    ASSERT_EQ(1, doc.count("ref_int64_array_field_sequence_id"));
    ASSERT_EQ(2, doc["ref_int64_array_field_sequence_id"].size());
    ASSERT_EQ(6, doc["ref_int64_array_field_sequence_id"][0]);
    ASSERT_EQ(7, doc["ref_int64_array_field_sequence_id"][1]);

    doc_json = R"({
                    "ref_int64_array_field": [-9223372036854775808]
                })"_json;
    add_doc_op = coll2->add(doc_json.dump());
    ASSERT_TRUE(add_doc_op.ok());

    doc = coll2->get("4").get();
    ASSERT_EQ(1, doc.count("ref_int64_array_field_sequence_id"));
    ASSERT_EQ(1, doc["ref_int64_array_field_sequence_id"].size());
    ASSERT_EQ(6, doc["ref_int64_array_field_sequence_id"][0]);

    // reference field inside object/object[]
    doc_json = R"({
                    "non_indexed_object": {
                        "ref_field": "foo"
                    }
                })"_json;
    add_doc_op = coll2->add(doc_json.dump());
    ASSERT_FALSE(add_doc_op.ok());
    ASSERT_EQ("Could not find `non_indexed_object` object/object[] field in the schema.", add_doc_op.error());

    doc_json = R"({
                    "object": {
                        "ref_field": 1
                    }
                })"_json;
    add_doc_op = coll2->add(doc_json.dump());
    ASSERT_FALSE(add_doc_op.ok());
    ASSERT_EQ("Field `object.ref_field` must have `string` value.", add_doc_op.error());

    doc_json = R"({
                    "object": {
                        "ref_array_field": [1]
                    }
                })"_json;
    add_doc_op = coll2->add(doc_json.dump());
    ASSERT_FALSE(add_doc_op.ok());
    ASSERT_EQ("Field `object.ref_array_field` must only have `string` values.", add_doc_op.error());

    doc_json = R"({
                    "object_array": [
                        {
                            "ref_field": 1
                        }
                    ]
                })"_json;
    add_doc_op = coll2->add(doc_json.dump());
    ASSERT_FALSE(add_doc_op.ok());
    ASSERT_EQ("Field `object_array.ref_field` must have `string` value.", add_doc_op.error());

    doc_json = R"({
                    "object_array": [
                        {
                            "ref_field": "foo"
                        }
                    ]
                })"_json;
    add_doc_op = coll2->add(doc_json.dump());
    ASSERT_FALSE(add_doc_op.ok());
    ASSERT_EQ("Reference document having `string_field:= foo` not found in the collection `coll1`.", add_doc_op.error());

    doc_json = R"({
                    "object_array": [
                        {
                            "ref_field": "a"
                        }
                    ]
                })"_json;
    add_doc_op = coll2->add(doc_json.dump());
    ASSERT_FALSE(add_doc_op.ok());
    ASSERT_EQ("Field `object_array.ref_field` has an incorrect type."
              " Hint: field inside an array of objects must be an array type as well.", add_doc_op.error());

    doc_json = R"({
                    "object_array": [
                        {
                            "ref_array_field": "foo"
                        }
                    ]
                })"_json;
    add_doc_op = coll2->add(doc_json.dump());
    ASSERT_FALSE(add_doc_op.ok());
    ASSERT_EQ("Reference document having `string_array_field:= foo` not found in the collection `coll1`.", add_doc_op.error());

    collectionManager.drop_collection("coll2");
    temp_json = schema_json;
    collection_create_op = collectionManager.create_collection(temp_json);
    ASSERT_TRUE(collection_create_op.ok());
    coll2 = collection_create_op.get();

    doc_json = R"({
                    "object": {
                        "ref_field": "d"
                    }
                })"_json;
    add_doc_op = coll2->add(doc_json.dump());
    ASSERT_TRUE(add_doc_op.ok());

    doc = coll2->get("0").get();
    ASSERT_EQ(1, doc.count("object.ref_field_sequence_id"));
    ASSERT_EQ(1, doc["object.ref_field_sequence_id"]);
    ASSERT_EQ(1, doc.count(".ref"));
    ASSERT_EQ(1, doc[".ref"].size());
    ASSERT_EQ("object.ref_field_sequence_id", doc[".ref"][0]);
    ASSERT_EQ(1, coll2->get_object_reference_helper_fields().count("object.ref_field_sequence_id"));

    doc_json = R"({
                    "object": {
                        "ref_array_field": ["foo"]
                    }
                })"_json;
    add_doc_op = coll2->add(doc_json.dump());
    ASSERT_TRUE(add_doc_op.ok());

    doc = coll2->get("1").get();
    ASSERT_EQ(1, doc.count("object.ref_array_field_sequence_id"));
    ASSERT_EQ(0, doc["object.ref_array_field_sequence_id"].size());
    ASSERT_EQ(1, doc.count(".ref"));
    ASSERT_EQ(1, doc[".ref"].size());
    ASSERT_EQ("object.ref_array_field_sequence_id", doc[".ref"][0]);

    doc_json = R"({
                    "object": {
                        "ref_array_field": ["b", "foo"]
                    }
                })"_json;
    add_doc_op = coll2->add(doc_json.dump());
    ASSERT_TRUE(add_doc_op.ok());

    doc = coll2->get("2").get();
    ASSERT_EQ(1, doc.count("object.ref_array_field_sequence_id"));
    ASSERT_EQ(1, doc["object.ref_array_field_sequence_id"].size());
    ASSERT_EQ(0, doc["object.ref_array_field_sequence_id"][0]);

    doc_json = R"({
                    "object_array": [
                        {
                            "ref_array_field": "c"
                        },
                        {
                            "ref_array_field": "e"
                        }
                    ]
                })"_json;
    add_doc_op = coll2->add(doc_json.dump());
    ASSERT_TRUE(add_doc_op.ok());

    doc = coll2->get("3").get();
    ASSERT_EQ(1, doc.count("object_array.ref_array_field_sequence_id"));
    ASSERT_EQ(2, doc["object_array.ref_array_field_sequence_id"].size());
    ASSERT_EQ(2, doc["object_array.ref_array_field_sequence_id"][0].size());
    ASSERT_EQ(0, doc["object_array.ref_array_field_sequence_id"][0][0]);
    ASSERT_EQ(0, doc["object_array.ref_array_field_sequence_id"][0][1]);
    ASSERT_EQ(2, doc["object_array.ref_array_field_sequence_id"][1].size());
    ASSERT_EQ(1, doc["object_array.ref_array_field_sequence_id"][1][0]);
    ASSERT_EQ(1, doc["object_array.ref_array_field_sequence_id"][1][1]);
    ASSERT_EQ(1, doc.count(".ref"));
    ASSERT_EQ(1, doc[".ref"].size());
    ASSERT_EQ("object_array.ref_array_field_sequence_id", doc[".ref"][0]);

    // float/float[] reference fields
    doc_json = R"({
                    "ref_float_field": 1.5
                })"_json;
    add_doc_op = coll2->add(doc_json.dump());
    ASSERT_FALSE(add_doc_op.ok());
    ASSERT_EQ("Cannot add a reference to `coll1.float_field` of type `float`.", add_doc_op.error());

    doc_json = R"({
                    "ref_float_array_field": [1.5]
                })"_json;
    add_doc_op = coll2->add(doc_json.dump());
    ASSERT_FALSE(add_doc_op.ok());
    ASSERT_EQ("Cannot add a reference to `coll1.float_array_field` of type `float[]`.", add_doc_op.error());

    // bool/bool[] reference fields
    doc_json = R"({
                    "ref_bool_field": "true"
                })"_json;
    add_doc_op = coll2->add(doc_json.dump());
    ASSERT_FALSE(add_doc_op.ok());
    ASSERT_EQ("Cannot add a reference to `coll1.bool_field` of type `bool`.", add_doc_op.error());

    doc_json = R"({
                    "ref_bool_array_field": ["true"]
                })"_json;
    add_doc_op = coll2->add(doc_json.dump());
    ASSERT_FALSE(add_doc_op.ok());
    ASSERT_EQ("Cannot add a reference to `coll1.bool_array_field` of type `bool[]`.", add_doc_op.error());

    // geopoint/geopoint[] reference fields
    doc_json = R"({
                    "ref_geopoint_field": [13.12631, 80.20252]
                })"_json;
    add_doc_op = coll2->add(doc_json.dump());
    ASSERT_FALSE(add_doc_op.ok());
    ASSERT_EQ("Cannot add a reference to `coll1.geopoint_field` of type `geopoint`.", add_doc_op.error());

    doc_json = R"({
                    "ref_geopoint_array_field": [[13.12631, 80.20252]]
                })"_json;
    add_doc_op = coll2->add(doc_json.dump());
    ASSERT_FALSE(add_doc_op.ok());
    ASSERT_EQ("Cannot add a reference to `coll1.geopoint_array_field` of type `geopoint[]`.", add_doc_op.error());

    // object/object[] reference fields
    doc_json = R"({
                    "ref_object_field": {
                        "foo": "bar"
                    }
                })"_json;
    add_doc_op = coll2->add(doc_json.dump());
    ASSERT_FALSE(add_doc_op.ok());
    ASSERT_EQ("Cannot add a reference to `coll1.object_field` of type `object`.", add_doc_op.error());

    doc_json = R"({
                    "ref_object_array_field": [
                        {
                            "foo": "bar"
                        }
                    ]
                })"_json;
    add_doc_op = coll2->add(doc_json.dump());
    ASSERT_FALSE(add_doc_op.ok());
    ASSERT_EQ("Cannot add a reference to `coll1.object_array_field` of type `object[]`.", add_doc_op.error());
}

TEST_F(CollectionJoinTest, FilterByReference_SingleMatch) {
    auto schema_json =
            R"({
                "name": "Products",
                "fields": [
                    {"name": "product_id", "type": "string"},
                    {"name": "product_name", "type": "string"},
                    {"name": "product_description", "type": "string"},
                    {"name": "rating", "type": "int32"}
                ]
            })"_json;
    std::vector<nlohmann::json> documents = {
            R"({
                "product_id": "product_a",
                "product_name": "shampoo",
                "product_description": "Our new moisturizing shampoo is perfect for those with dry or damaged hair.",
                "rating": "2"
            })"_json,
            R"({
                "product_id": "product_b",
                "product_name": "soap",
                "product_description": "Introducing our all-natural, organic soap bar made with essential oils and botanical ingredients.",
                "rating": "4"
            })"_json
    };
    auto collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_TRUE(collection_create_op.ok());
    for (auto const &json: documents) {
        auto add_op = collection_create_op.get()->add(json.dump());
        if (!add_op.ok()) {
            LOG(INFO) << add_op.error();
        }
        ASSERT_TRUE(add_op.ok());
    }

    schema_json =
            R"({
                "name": "Customers",
                "fields": [
                    {"name": "customer_id", "type": "string"},
                    {"name": "customer_name", "type": "string"},
                    {"name": "product_price", "type": "float"},
                    {"name": "product_id", "type": "string", "reference": "Products.product_id"}
                ]
            })"_json;
    documents = {
            R"({
                "customer_id": "customer_a",
                "customer_name": "Joe",
                "product_price": 143,
                "product_id": "product_a"
            })"_json,
            R"({
                "customer_id": "customer_a",
                "customer_name": "Joe",
                "product_price": 73.5,
                "product_id": "product_b"
            })"_json,
            R"({
                "customer_id": "customer_b",
                "customer_name": "Dan",
                "product_price": 75,
                "product_id": "product_a"
            })"_json,
            R"({
                "customer_id": "customer_b",
                "customer_name": "Dan",
                "product_price": 140,
                "product_id": "product_b"
            })"_json
    };
    collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_TRUE(collection_create_op.ok());
    for (auto const &json: documents) {
        auto add_op = collection_create_op.get()->add(json.dump());
        if (!add_op.ok()) {
            LOG(INFO) << add_op.error();
        }
        ASSERT_TRUE(add_op.ok());
    }

    schema_json =
            R"({
                "name": "Dummy",
                "fields": [
                    {"name": "dummy_id", "type": "string"}
                ]
            })"_json;
    collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_TRUE(collection_create_op.ok());

    auto coll = collectionManager.get_collection_unsafe("Products");
    auto search_op = coll->search("s", {"product_name"}, "$foo:=customer_a", {}, {}, {0},
                               10, 1, FREQUENCY, {true}, Index::DROP_TOKENS_THRESHOLD);
    ASSERT_FALSE(search_op.ok());
    ASSERT_EQ(search_op.error(), "Could not parse the reference filter.");

    search_op = coll->search("s", {"product_name"}, "$foo(:=customer_a", {}, {}, {0},
                                  10, 1, FREQUENCY, {true}, Index::DROP_TOKENS_THRESHOLD);
    ASSERT_FALSE(search_op.ok());
    ASSERT_EQ(search_op.error(), "Could not parse the reference filter.");

    search_op = coll->search("s", {"product_name"}, "$foo(:=customer_a)", {}, {}, {0},
                                  10, 1, FREQUENCY, {true}, Index::DROP_TOKENS_THRESHOLD);
    ASSERT_FALSE(search_op.ok());
    ASSERT_EQ(search_op.error(), "Referenced collection `foo` not found.");

    search_op = coll->search("s", {"product_name"}, "$Dummy(dummy_id:=customer_a)", {}, {}, {0},
                             10, 1, FREQUENCY, {true}, Index::DROP_TOKENS_THRESHOLD);
    ASSERT_FALSE(search_op.ok());
    ASSERT_EQ(search_op.error(), "Failed to join on `Dummy`: No reference field found.");

    search_op = coll->search("s", {"product_name"}, "$Customers(foo:=customer_a)", {}, {}, {0},
                             10, 1, FREQUENCY, {true}, Index::DROP_TOKENS_THRESHOLD);
    ASSERT_FALSE(search_op.ok());
    ASSERT_EQ(search_op.error(), "Failed to join on `Customers` collection: Could not find a filter "
                                 "field named `foo` in the schema.");

    auto result = coll->search("s", {"product_name"}, "$Customers(customer_id:=customer_a && product_price:<100)", {},
                               {}, {0}, 10, 1, FREQUENCY, {true}, Index::DROP_TOKENS_THRESHOLD).get();

    ASSERT_EQ(1, result["found"].get<size_t>());
    ASSERT_EQ(1, result["hits"].size());
    ASSERT_EQ("soap", result["hits"][0]["document"]["product_name"].get<std::string>());

    std::map<std::string, std::string> req_params = {
            {"collection", "Customers"},
            {"q", "Dan"},
            {"query_by", "customer_name"},
            {"filter_by", "$Products(foo:>3)"},
    };
    nlohmann::json embedded_params;
    std::string json_res;
    auto now_ts = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

    auto search_op_bool = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_FALSE(search_op_bool.ok());
    ASSERT_EQ(search_op_bool.error(), "Failed to join on `Products` collection: Could not find a filter "
                                        "field named `foo` in the schema.");

    req_params = {
            {"collection", "Customers"},
            {"q", "Dan"},
            {"query_by", "customer_name"},
            {"filter_by", "$Products(rating:>3)"},
            {"include_fields", "$Products(*:merge)"},
    };

    search_op_bool = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op_bool.ok());

    auto res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(1, res_obj["found"].get<size_t>());
    ASSERT_EQ(1, res_obj["hits"].size());
    ASSERT_EQ("soap", res_obj["hits"][0]["document"]["product_name"].get<std::string>());

    req_params = {
            {"collection", "Customers"},
            {"q", "Dan"},
            {"query_by", "customer_name"},
            {"filter_by", "$Products(id:*) && product_price:>100"},
            {"include_fields", "$Products(*:merge)"},
    };

    search_op_bool = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op_bool.ok());

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(1, res_obj["found"].get<size_t>());
    ASSERT_EQ(1, res_obj["hits"].size());
    ASSERT_EQ("soap", res_obj["hits"][0]["document"]["product_name"].get<std::string>());

    collectionManager.drop_collection("Customers");
    collectionManager.drop_collection("Products");
}

TEST_F(CollectionJoinTest, FilterByReference_MultipleMatch) {
    auto schema_json =
            R"({
                "name": "Users",
                "fields": [
                    {"name": "user_id", "type": "string"},
                    {"name": "user_name", "type": "string"}
                ]
            })"_json;
    std::vector<nlohmann::json> documents = {
            R"({
                "user_id": "user_a",
                "user_name": "Roshan"
            })"_json,
            R"({
                "user_id": "user_b",
                "user_name": "Ruby"
            })"_json,
            R"({
                "user_id": "user_c",
                "user_name": "Joe"
            })"_json,
            R"({
                "user_id": "user_d",
                "user_name": "Aby"
            })"_json
    };
    auto collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_TRUE(collection_create_op.ok());
    for (auto const &json: documents) {
        auto add_op = collection_create_op.get()->add(json.dump());
        if (!add_op.ok()) {
            LOG(INFO) << add_op.error();
        }
        ASSERT_TRUE(add_op.ok());
    }

    schema_json =
            R"({
                "name": "Repos",
                "fields": [
                    {"name": "repo_id", "type": "string"},
                    {"name": "repo_content", "type": "string"}
                ]
            })"_json;
    documents = {
            R"({
                "repo_id": "repo_a",
                "repo_content": "body1"
            })"_json,
            R"({
                "repo_id": "repo_b",
                "repo_content": "body2"
            })"_json,
            R"({
                "repo_id": "repo_c",
                "repo_content": "body3"
            })"_json
    };
    collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_TRUE(collection_create_op.ok());
    for (auto const &json: documents) {
        auto add_op = collection_create_op.get()->add(json.dump());
        if (!add_op.ok()) {
            LOG(INFO) << add_op.error();
        }
        ASSERT_TRUE(add_op.ok());
    }

    schema_json =
            R"({
                "name": "Links",
                "fields": [
                    {"name": "repo_id", "type": "string", "reference": "Repos.repo_id"},
                    {"name": "user_id", "type": "string", "reference": "Users.user_id"}
                ]
            })"_json;
    documents = {
            R"({
                "repo_id": "repo_a",
                "user_id": "user_b"
            })"_json,
            R"({
                "repo_id": "repo_a",
                "user_id": "user_c"
            })"_json,
            R"({
                "repo_id": "repo_b",
                "user_id": "user_a"
            })"_json,
            R"({
                "repo_id": "repo_b",
                "user_id": "user_b"
            })"_json,
            R"({
                "repo_id": "repo_b",
                "user_id": "user_d"
            })"_json,
            R"({
                "repo_id": "repo_c",
                "user_id": "user_a"
            })"_json,
            R"({
                "repo_id": "repo_c",
                "user_id": "user_b"
            })"_json,
            R"({
                "repo_id": "repo_c",
                "user_id": "user_c"
            })"_json,
            R"({
                "repo_id": "repo_c",
                "user_id": "user_d"
            })"_json
    };
    collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_TRUE(collection_create_op.ok());

    for (auto const &json: documents) {
        auto add_op = collection_create_op.get()->add(json.dump());
        if (!add_op.ok()) {
            LOG(INFO) << add_op.error();
        }
        ASSERT_TRUE(add_op.ok());
    }

    auto coll = collectionManager.get_collection_unsafe("Users");

    // Search for users linked to repo_b
    auto result = coll->search("R", {"user_name"}, "$Links(repo_id:=repo_b)", {}, {}, {0},
                               10, 1, FREQUENCY, {true}, Index::DROP_TOKENS_THRESHOLD).get();

    ASSERT_EQ(2, result["found"].get<size_t>());
    ASSERT_EQ(2, result["hits"].size());
    ASSERT_EQ("user_b", result["hits"][0]["document"]["user_id"].get<std::string>());
    ASSERT_EQ("user_a", result["hits"][1]["document"]["user_id"].get<std::string>());

    collectionManager.drop_collection("Users");
    collectionManager.drop_collection("Repos");
    collectionManager.drop_collection("Links");
}

TEST_F(CollectionJoinTest, AndFilterResults_NoReference) {
    filter_result_t a;
    a.count = 9;
    a.docs = new uint32_t[a.count];
    for (size_t i = 0; i < a.count; i++) {
        a.docs[i] = i;
    }

    filter_result_t b;
    b.count = 0;
    uint32_t limit = 10;
    b.docs = new uint32_t[limit];
    for (size_t i = 2; i < limit; i++) {
        if (i % 3 == 0) {
            b.docs[b.count++] = i;
        }
    }

    // a.docs: [0..8] , b.docs: [3, 6, 9]
    filter_result_t result;
    filter_result_t::and_filter_results(a, b, result);

    ASSERT_EQ(2, result.count);
    ASSERT_EQ(nullptr, result.coll_to_references);

    std::vector<uint32_t> docs = {3, 6};

    for(size_t i = 0; i < result.count; i++) {
        ASSERT_EQ(docs[i], result.docs[i]);
    }
}

TEST_F(CollectionJoinTest, AndFilterResults_WithReferences) {
    filter_result_t a;
    a.count = 9;
    a.docs = new uint32_t[a.count];
    a.coll_to_references = new std::map<std::string, reference_filter_result_t>[a.count] {};

    for (size_t i = 0; i < a.count; i++) {
        a.docs[i] = i;

        auto& reference = a.coll_to_references[i];
        // Having only one reference of each document for brevity.
        auto reference_docs = new uint32_t[1];
        reference_docs[0] = 10 - i;
        reference["foo"] = reference_filter_result_t(1, reference_docs);
    }

    filter_result_t b;
    b.count = 0;
    uint32_t limit = 10;
    b.docs = new uint32_t[limit];
    b.coll_to_references = new std::map<std::string, reference_filter_result_t>[limit] {};
    for (size_t i = 2; i < limit; i++) {
        if (i % 3 == 0) {
            b.docs[b.count] = i;

            auto& reference = b.coll_to_references[b.count++];
            auto reference_docs = new uint32_t[1];
            reference_docs[0] = 2 * i;
            reference["bar"] = reference_filter_result_t(1, reference_docs);
        }
    }

    // a.docs: [0..8] , b.docs: [3, 6, 9]
    filter_result_t result;
    filter_result_t::and_filter_results(a, b, result);

    ASSERT_EQ(2, result.count);
    ASSERT_EQ(2, result.coll_to_references[0].size());
    ASSERT_EQ(1, result.coll_to_references[0].count("foo"));
    ASSERT_EQ(1, result.coll_to_references[0].count("bar"));

    std::vector<uint32_t> docs = {3, 6}, foo_reference = {7, 4}, bar_reference = {6, 12};

    for(size_t i = 0; i < result.count; i++) {
        ASSERT_EQ(docs[i], result.docs[i]);

        // result should contain correct references to the foo and bar collection.
        ASSERT_EQ(1, result.coll_to_references[i].at("foo").count);
        ASSERT_EQ(foo_reference[i], result.coll_to_references[i].at("foo").docs[0]);
        ASSERT_EQ(1, result.coll_to_references[i].at("bar").count);
        ASSERT_EQ(bar_reference[i], result.coll_to_references[i].at("bar").docs[0]);
    }
}

TEST_F(CollectionJoinTest, OrFilterResults_NoReference) {
    filter_result_t a, b;
    a.count = 0;
    uint32_t limit = 10;
    a.docs = new uint32_t[limit];
    for (size_t i = 2; i < limit; i++) {
        if (i % 3 == 0) {
            a.docs[a.count++] = i;
        }
    }

    // a.docs: [3, 6, 9], b.docs: []
    filter_result_t result1;
    filter_result_t::or_filter_results(a, b, result1);
    ASSERT_EQ(3, result1.count);
    ASSERT_EQ(nullptr, result1.coll_to_references);

    std::vector<uint32_t> expected = {3, 6, 9};
    for (size_t i = 0; i < result1.count; i++) {
        ASSERT_EQ(expected[i], result1.docs[i]);
    }

    b.count = 9;
    b.docs = new uint32_t[b.count];
    for (size_t i = 0; i < b.count; i++) {
        b.docs[i] = i;
    }

    // a.docs: [3, 6, 9], b.docs: [0..8]
    filter_result_t result2;
    filter_result_t::or_filter_results(a, b, result2);
    ASSERT_EQ(10, result2.count);
    ASSERT_EQ(nullptr, result2.coll_to_references);

    expected = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    for (size_t i = 0; i < result2.count; i++) {
        ASSERT_EQ(expected[i], result2.docs[i]);
    }


    filter_result_t c, result3;

    std::vector<uint32_t> vec = {0, 4, 5};
    c.docs = new uint32_t[vec.size()];
    auto j = 0;
    for(auto i: vec) {
        a.docs[j++] = i;
    }

    // b.docs: [0..8], c.docs: [0, 4, 5]
    filter_result_t::or_filter_results(b, c, result3);
    ASSERT_EQ(9, result3.count);
    ASSERT_EQ(nullptr, result3.coll_to_references);

    expected = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    for(size_t i = 0; i < result3.count; i++) {
        ASSERT_EQ(expected[i], result3.docs[i]);
    }
}

TEST_F(CollectionJoinTest, OrFilterResults_WithReferences) {
    filter_result_t a, b;
    uint32_t limit = 10;

    a.count = 0;
    a.docs = new uint32_t[limit];
    a.coll_to_references = new std::map<std::string, reference_filter_result_t>[limit] {};
    for (size_t i = 2; i < limit; i++) {
        if (i % 3 == 0) {
            a.docs[a.count] = i;

            auto& reference = a.coll_to_references[a.count++];
            auto reference_docs = new uint32_t[1];
            reference_docs[0] = 2 * i;
            reference["foo"] = reference_filter_result_t(1, reference_docs);
        }
    }

    // a.docs: [3, 6, 9], b.docs: []
    filter_result_t result1;
    filter_result_t::or_filter_results(a, b, result1);

    ASSERT_EQ(3, result1.count);
    ASSERT_EQ(1, result1.coll_to_references[0].size());
    ASSERT_EQ(1, result1.coll_to_references[0].count("foo"));

    std::vector<uint32_t> expected = {3, 6, 9}, foo_reference = {6, 12, 18};
    for (size_t i = 0; i < result1.count; i++) {
        ASSERT_EQ(expected[i], result1.docs[i]);

        ASSERT_EQ(1, result1.coll_to_references[i].at("foo").count);
        ASSERT_EQ(foo_reference[i], result1.coll_to_references[i].at("foo").docs[0]);
    }

    b.count = 9;
    b.docs = new uint32_t[b.count];
    b.coll_to_references = new std::map<std::string, reference_filter_result_t>[b.count] {};
    for (size_t i = 0; i < b.count; i++) {
        b.docs[i] = i;

        auto& reference = b.coll_to_references[i];
        auto reference_docs = new uint32_t[1];
        reference_docs[0] = 10 - i;
        reference["bar"] = reference_filter_result_t(1, reference_docs);
    }

    // a.docs: [3, 6, 9], b.docs: [0..8]
    filter_result_t result2;
    filter_result_t::or_filter_results(a, b, result2);
    ASSERT_EQ(10, result2.count);

    expected = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    // doc_id -> reference_id
    std::map<uint32_t, uint32_t> foo_map = {{3, 6}, {6, 12}, {9, 18}}, bar_map = {{0, 10}, {1, 9}, {2, 8}, {3, 7},
                                                                                  {4, 6}, {5, 5}, {6, 4}, {7, 3}, {8, 2}};
    for (size_t i = 0; i < result2.count; i++) {
        ASSERT_EQ(expected[i], result2.docs[i]);

        if (foo_map.count(i) != 0) {
            ASSERT_EQ(1, result2.coll_to_references[i].at("foo").count);
            ASSERT_EQ(foo_map[i], result2.coll_to_references[i].at("foo").docs[0]);
        } else {
            // foo didn't have any reference to current doc.
            ASSERT_EQ(0, result2.coll_to_references[i].count("foo"));
        }

        if (bar_map.count(i) != 0) {
            ASSERT_EQ(1, result2.coll_to_references[i].at("bar").count);
            ASSERT_EQ(bar_map[i], result2.coll_to_references[i].at("bar").docs[0]);
        } else {
            ASSERT_EQ(0, result2.coll_to_references[i].count("bar"));
        }
    }

    filter_result_t c, result3;

    std::map<uint32_t, uint32_t> baz_map = {{0, 2}, {4, 0}, {5, 8}};
    c.count = baz_map.size();
    c.docs = new uint32_t[baz_map.size()];
    c.coll_to_references = new std::map<std::string, reference_filter_result_t>[baz_map.size()] {};
    auto j = 0;
    for(auto i: baz_map) {
        c.docs[j] = i.first;

        auto& reference = c.coll_to_references[j++];
        auto reference_docs = new uint32_t[1];
        reference_docs[0] = i.second;
        reference["baz"] = reference_filter_result_t(1, reference_docs);
    }

    // b.docs: [0..8], c.docs: [0, 4, 5]
    filter_result_t::or_filter_results(b, c, result3);
    ASSERT_EQ(9, result3.count);

    expected = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    for (size_t i = 0; i < result3.count; i++) {
        ASSERT_EQ(expected[i], result3.docs[i]);

        if (bar_map.count(i) != 0) {
            ASSERT_EQ(1, result3.coll_to_references[i].at("bar").count);
            ASSERT_EQ(bar_map[i], result3.coll_to_references[i].at("bar").docs[0]);
        } else {
            ASSERT_EQ(0, result3.coll_to_references[i].count("bar"));
        }

        if (baz_map.count(i) != 0) {
            ASSERT_EQ(1, result3.coll_to_references[i].at("baz").count);
            ASSERT_EQ(baz_map[i], result3.coll_to_references[i].at("baz").docs[0]);
        } else {
            ASSERT_EQ(0, result3.coll_to_references[i].count("baz"));
        }
    }
}

TEST_F(CollectionJoinTest, FilterByNReferences) {
    auto schema_json =
            R"({
                "name": "Users",
                "fields": [
                    {"name": "user_id", "type": "string"},
                    {"name": "user_name", "type": "string"}
                ]
            })"_json;
    std::vector<nlohmann::json> documents = {
            R"({
                "user_id": "user_a",
                "user_name": "Roshan"
            })"_json,
            R"({
                "user_id": "user_b",
                "user_name": "Ruby"
            })"_json,
            R"({
                "user_id": "user_c",
                "user_name": "Joe"
            })"_json,
            R"({
                "user_id": "user_d",
                "user_name": "Aby"
            })"_json
    };
    auto collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_TRUE(collection_create_op.ok());
    for (auto const &json: documents) {
        auto add_op = collection_create_op.get()->add(json.dump());
        if (!add_op.ok()) {
            LOG(INFO) << add_op.error();
        }
        ASSERT_TRUE(add_op.ok());
    }

    schema_json =
            R"({
                "name": "Repos",
                "fields": [
                    {"name": "repo_id", "type": "string"},
                    {"name": "repo_content", "type": "string"},
                    {"name": "repo_stars", "type": "int32"},
                    {"name": "repo_is_private", "type": "bool"}
                ]
            })"_json;
    documents = {
            R"({
                "repo_id": "repo_a",
                "repo_content": "body1",
                "repo_stars": 431,
                "repo_is_private": true
            })"_json,
            R"({
                "repo_id": "repo_b",
                "repo_content": "body2",
                "repo_stars": 4562,
                "repo_is_private": false
            })"_json,
            R"({
                "repo_id": "repo_c",
                "repo_content": "body3",
                "repo_stars": 945,
                "repo_is_private": false
            })"_json
    };
    collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_TRUE(collection_create_op.ok());
    for (auto const &json: documents) {
        auto add_op = collection_create_op.get()->add(json.dump());
        if (!add_op.ok()) {
            LOG(INFO) << add_op.error();
        }
        ASSERT_TRUE(add_op.ok());
    }

    schema_json =
            R"({
                "name": "Links",
                "fields": [
                    {"name": "repo_id", "type": "string", "reference": "Repos.repo_id"},
                    {"name": "user_id", "type": "string", "reference": "Users.user_id"}
                ]
            })"_json;
    documents = {
            R"({
                "repo_id": "repo_a",
                "user_id": "user_b"
            })"_json,
            R"({
                "repo_id": "repo_a",
                "user_id": "user_c"
            })"_json,
            R"({
                "repo_id": "repo_b",
                "user_id": "user_a"
            })"_json,
            R"({
                "repo_id": "repo_b",
                "user_id": "user_b"
            })"_json,
            R"({
                "repo_id": "repo_b",
                "user_id": "user_d"
            })"_json,
            R"({
                "repo_id": "repo_c",
                "user_id": "user_a"
            })"_json,
            R"({
                "repo_id": "repo_c",
                "user_id": "user_b"
            })"_json,
            R"({
                "repo_id": "repo_c",
                "user_id": "user_c"
            })"_json,
            R"({
                "repo_id": "repo_c",
                "user_id": "user_d"
            })"_json
    };
    collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_TRUE(collection_create_op.ok());

    for (auto const &json: documents) {
        auto add_op = collection_create_op.get()->add(json.dump());
        if (!add_op.ok()) {
            LOG(INFO) << add_op.error();
        }
        ASSERT_TRUE(add_op.ok());
    }

    schema_json =
            R"({
                "name": "Organizations",
                "fields": [
                    {"name": "org_id", "type": "string"},
                    {"name": "org_name", "type": "string"}
                ]
            })"_json;
    documents = {
            R"({
                "org_id": "org_a",
                "org_name": "Typesense"
            })"_json
    };
    collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_TRUE(collection_create_op.ok());

    for (auto const &json: documents) {
        auto add_op = collection_create_op.get()->add(json.dump());
        if (!add_op.ok()) {
            LOG(INFO) << add_op.error();
        }
        ASSERT_TRUE(add_op.ok());
    }

    schema_json =
            R"({
                "name": "Participants",
                "fields": [
                    {"name": "user_id", "type": "string", "reference": "Users.user_id"},
                    {"name": "org_id", "type": "string", "reference": "Organizations.org_id"}
                ]
            })"_json;
    documents = {
            R"({
                "user_id": "user_a",
                "org_id": "org_a"
            })"_json,
            R"({
                "user_id": "user_b",
                "org_id": "org_a"
            })"_json,
            R"({
                "user_id": "user_d",
                "org_id": "org_a"
            })"_json,
    };
    collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_TRUE(collection_create_op.ok());

    for (auto const &json: documents) {
        auto add_op = collection_create_op.get()->add(json.dump());
        if (!add_op.ok()) {
            LOG(INFO) << add_op.error();
        }
        ASSERT_TRUE(add_op.ok());
    }

    auto coll = collectionManager.get_collection_unsafe("Users");

    // Search for users within an organization with access to a particular repo.
    auto result = coll->search("R", {"user_name"}, "$Participants(org_id:=org_a) && $Links(repo_id:=repo_b)", {}, {}, {0},
                               10, 1, FREQUENCY, {true}, Index::DROP_TOKENS_THRESHOLD).get();

    ASSERT_EQ(2, result["found"].get<size_t>());
    ASSERT_EQ(2, result["hits"].size());
    ASSERT_EQ("user_b", result["hits"][0]["document"]["user_id"].get<std::string>());
    ASSERT_EQ("user_a", result["hits"][1]["document"]["user_id"].get<std::string>());

    collectionManager.drop_collection("Users");
    collectionManager.drop_collection("Repos");
    collectionManager.drop_collection("Links");
}

TEST_F(CollectionJoinTest, FilterByNestedReferences) {
    auto schema_json =
            R"({
                "name": "Coll_A",
                "fields": [
                    {"name": "title", "type": "string"}
                ]
            })"_json;
    std::vector<nlohmann::json> documents = {
            R"({
                "title": "coll_a_0"
            })"_json,
            R"({
                "title": "coll_a_1"
            })"_json
    };
    auto collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_TRUE(collection_create_op.ok());
    for (auto const &json: documents) {
        auto add_op = collection_create_op.get()->add(json.dump());
        if (!add_op.ok()) {
            LOG(INFO) << add_op.error();
        }
        ASSERT_TRUE(add_op.ok());
    }
    schema_json =
            R"({
                "name": "Coll_B",
                "fields": [
                    {"name": "title", "type": "string"},
                    {"name": "ref_coll_a", "type": "string", "reference": "Coll_A.id"}
                ]
            })"_json;
    documents = {
            R"({
                "title": "coll_b_0",
                "ref_coll_a": "1"
            })"_json,
            R"({
                "title": "coll_b_1",
                "ref_coll_a": "0"
            })"_json,
            R"({
                "title": "coll_b_2",
                "ref_coll_a": "0"
            })"_json
    };
    collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_TRUE(collection_create_op.ok());
    for (auto const &json: documents) {
        auto add_op = collection_create_op.get()->add(json.dump());
        if (!add_op.ok()) {
            LOG(INFO) << add_op.error();
        }
        ASSERT_TRUE(add_op.ok());
    }
    schema_json =
            R"({
                "name": "Coll_C",
                "fields": [
                    {"name": "title", "type": "string"},
                    {"name": "ref_coll_b", "type": "string[]", "reference": "Coll_B.id"}
                ]
            })"_json;
    documents = {
            R"({
                "title": "coll_c_0",
                "ref_coll_b": ["0"]
            })"_json,
            R"({
                "title": "coll_c_1",
                "ref_coll_b": ["1"]
            })"_json,
            R"({
                "title": "coll_c_2",
                "ref_coll_b": ["0", "1"]
            })"_json,
            R"({
                "title": "coll_c_3",
                "ref_coll_b": ["2"]
            })"_json
    };
    collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_TRUE(collection_create_op.ok());
    for (auto const &json: documents) {
        auto add_op = collection_create_op.get()->add(json.dump());
        if (!add_op.ok()) {
            LOG(INFO) << add_op.error();
        }
        ASSERT_TRUE(add_op.ok());
    }

    std::map<std::string, std::string> req_params = {
            {"collection", "Coll_A"},
            {"q", "*"},
            {"filter_by", "$Coll_B($Coll_C(id: [1, 3]))"},
            {"include_fields", "title, $Coll_B(title, $Coll_C(title))"}
    };
    nlohmann::json embedded_params;
    std::string json_res;
    auto now_ts = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

    auto search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    nlohmann::json res_obj = nlohmann::json::parse(json_res);
    //              coll_b_1 <- coll_c_1
    // coll_a_0  <
    //             coll_b_2 <- coll_c_3
    ASSERT_EQ(1, res_obj["found"].get<size_t>());
    ASSERT_EQ(1, res_obj["hits"].size());
    ASSERT_EQ(2, res_obj["hits"][0]["document"].size());
    ASSERT_EQ("coll_a_0", res_obj["hits"][0]["document"]["title"]);
    ASSERT_EQ(2, res_obj["hits"][0]["document"]["Coll_B"].size());
    ASSERT_EQ("coll_b_1", res_obj["hits"][0]["document"]["Coll_B"][0]["title"]);
    ASSERT_EQ(1, res_obj["hits"][0]["document"]["Coll_B"][0]["Coll_C"].size());
    ASSERT_EQ("coll_c_1", res_obj["hits"][0]["document"]["Coll_B"][0]["Coll_C"][0]["title"]);
    ASSERT_EQ("coll_b_2", res_obj["hits"][0]["document"]["Coll_B"][1]["title"]);
    ASSERT_EQ(1, res_obj["hits"][0]["document"]["Coll_B"][1]["Coll_C"].size());
    ASSERT_EQ("coll_c_3", res_obj["hits"][0]["document"]["Coll_B"][1]["Coll_C"][0]["title"]);

    req_params = {
            {"collection", "Coll_A"},
            {"q", "*"},
            {"filter_by", "$Coll_B($Coll_C(id: != 0))"},
            {"include_fields", "title, $Coll_B(title, $Coll_C(title):nest_array)"}
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    res_obj = nlohmann::json::parse(json_res);
    // coll_a_1 <- coll_b_0 <- coll_c_2
    //
    //             coll_b_1 <- coll_c_1, coll_c_2
    // coll_a_0  <
    //             coll_b_2 <- coll_c_3
    ASSERT_EQ(2, res_obj["found"].get<size_t>());
    ASSERT_EQ(2, res_obj["hits"].size());
    ASSERT_EQ(2, res_obj["hits"][0]["document"].size());
    ASSERT_EQ("coll_a_1", res_obj["hits"][0]["document"]["title"]);
    ASSERT_EQ(1, res_obj["hits"][0]["document"]["Coll_B"].size());
    ASSERT_EQ("coll_b_0", res_obj["hits"][0]["document"]["Coll_B"][0]["title"]);
    ASSERT_EQ(1, res_obj["hits"][0]["document"]["Coll_B"][0]["Coll_C"].size());
    ASSERT_EQ("coll_c_2", res_obj["hits"][0]["document"]["Coll_B"][0]["Coll_C"][0]["title"]);

    ASSERT_EQ("coll_a_0", res_obj["hits"][1]["document"]["title"]);
    ASSERT_EQ(2, res_obj["hits"][1]["document"]["Coll_B"].size());
    ASSERT_EQ("coll_b_1", res_obj["hits"][1]["document"]["Coll_B"][0]["title"]);
    ASSERT_EQ(2, res_obj["hits"][1]["document"]["Coll_B"][0]["Coll_C"].size());
    ASSERT_EQ("coll_c_1", res_obj["hits"][1]["document"]["Coll_B"][0]["Coll_C"][0]["title"]);
    ASSERT_EQ("coll_c_2", res_obj["hits"][1]["document"]["Coll_B"][0]["Coll_C"][1]["title"]);
    ASSERT_EQ("coll_b_2", res_obj["hits"][1]["document"]["Coll_B"][1]["title"]);
    ASSERT_EQ(1, res_obj["hits"][1]["document"]["Coll_B"][1]["Coll_C"].size());
    ASSERT_EQ("coll_c_3", res_obj["hits"][1]["document"]["Coll_B"][1]["Coll_C"][0]["title"]);

    req_params = {
            {"collection", "Coll_C"},
            {"q", "*"},
            {"filter_by", "$Coll_B($Coll_A(id: 0))"},
            {"include_fields", "title, $Coll_B(title, $Coll_A(title))"}
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    res_obj = nlohmann::json::parse(json_res);
    // coll_c_3 -> coll_b_2 -> coll_a_0
    //
    // coll_c_2 -> coll_b_1 -> coll_a_0
    //
    // coll_c_1 -> coll_b_1 -> coll_a_0
    ASSERT_EQ(3, res_obj["found"].get<size_t>());
    ASSERT_EQ(3, res_obj["hits"].size());
    ASSERT_EQ(2, res_obj["hits"][0]["document"].size());
    ASSERT_EQ("coll_c_3", res_obj["hits"][0]["document"]["title"]);
    ASSERT_EQ(2, res_obj["hits"][0]["document"]["Coll_B"].size());
    ASSERT_EQ("coll_b_2", res_obj["hits"][0]["document"]["Coll_B"]["title"]);
    ASSERT_EQ(1, res_obj["hits"][0]["document"]["Coll_B"]["Coll_A"].size());
    ASSERT_EQ("coll_a_0", res_obj["hits"][0]["document"]["Coll_B"]["Coll_A"]["title"]);

    ASSERT_EQ(2, res_obj["hits"][1]["document"].size());
    ASSERT_EQ("coll_c_2", res_obj["hits"][1]["document"]["title"]);
    ASSERT_EQ(2, res_obj["hits"][1]["document"]["Coll_B"].size());
    ASSERT_EQ("coll_b_1", res_obj["hits"][1]["document"]["Coll_B"]["title"]);
    ASSERT_EQ(1, res_obj["hits"][1]["document"]["Coll_B"]["Coll_A"].size());
    ASSERT_EQ("coll_a_0", res_obj["hits"][1]["document"]["Coll_B"]["Coll_A"]["title"]);

    ASSERT_EQ(2, res_obj["hits"][2]["document"].size());
    ASSERT_EQ("coll_c_1", res_obj["hits"][2]["document"]["title"]);
    ASSERT_EQ(2, res_obj["hits"][2]["document"]["Coll_B"].size());
    ASSERT_EQ("coll_b_1", res_obj["hits"][2]["document"]["Coll_B"]["title"]);
    ASSERT_EQ(1, res_obj["hits"][2]["document"]["Coll_B"]["Coll_A"].size());
    ASSERT_EQ("coll_a_0", res_obj["hits"][2]["document"]["Coll_B"]["Coll_A"]["title"]);

    schema_json =
            R"({
                "name": "Coll_D",
                "fields": [
                    {"name": "title", "type": "string"},
                    {"name": "ref_coll_c", "type": "string[]", "reference": "Coll_C.id"}
                ]
            })"_json;
    documents = {
            R"({
                "title": "coll_d_0",
                "ref_coll_c": []
            })"_json,
            R"({
                "title": "coll_d_1",
                "ref_coll_c": ["1", "3"]
            })"_json,
            R"({
                "title": "coll_d_2",
                "ref_coll_c": ["2", "3"]
            })"_json,
    };
    collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_TRUE(collection_create_op.ok());
    for (auto const &json: documents) {
        auto add_op = collection_create_op.get()->add(json.dump());
        if (!add_op.ok()) {
            LOG(INFO) << add_op.error();
        }
        ASSERT_TRUE(add_op.ok());
    }

    req_params = {
            {"collection", "Coll_B"},
            {"q", "*"},
            {"filter_by", "$Coll_C($Coll_D(id: *))"},
            {"include_fields", "title, $Coll_C(title, $Coll_D(title:nest_array):nest_array)"}
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    res_obj = nlohmann::json::parse(json_res);
    // coll_b_2 <- coll_c_3 <- coll_d_1, coll_d_2
    //
    //             coll_c_1 <- coll_d_1
    // coll_b_1  <
    //             coll_c_2 <- coll_d_2
    //
    // coll_b_0 <- coll_c_2 <- coll_d_2
    ASSERT_EQ(3, res_obj["found"].get<size_t>());
    ASSERT_EQ(3, res_obj["hits"].size());
    ASSERT_EQ(2, res_obj["hits"][0]["document"].size());
    ASSERT_EQ("coll_b_2", res_obj["hits"][0]["document"]["title"]);
    ASSERT_EQ(1, res_obj["hits"][0]["document"]["Coll_C"].size());
    ASSERT_EQ("coll_c_3", res_obj["hits"][0]["document"]["Coll_C"][0]["title"]);
    ASSERT_EQ(2, res_obj["hits"][0]["document"]["Coll_C"][0]["Coll_D"].size());
    ASSERT_EQ("coll_d_1", res_obj["hits"][0]["document"]["Coll_C"][0]["Coll_D"][0]["title"]);
    ASSERT_EQ("coll_d_2", res_obj["hits"][0]["document"]["Coll_C"][0]["Coll_D"][1]["title"]);

    ASSERT_EQ(2, res_obj["hits"][1]["document"].size());
    ASSERT_EQ("coll_b_1", res_obj["hits"][1]["document"]["title"]);
    ASSERT_EQ(2, res_obj["hits"][1]["document"]["Coll_C"].size());
    ASSERT_EQ("coll_c_1", res_obj["hits"][1]["document"]["Coll_C"][0]["title"]);
    ASSERT_EQ(1, res_obj["hits"][1]["document"]["Coll_C"][0]["Coll_D"].size());
    ASSERT_EQ("coll_d_1", res_obj["hits"][1]["document"]["Coll_C"][0]["Coll_D"][0]["title"]);
    ASSERT_EQ("coll_c_2", res_obj["hits"][1]["document"]["Coll_C"][1]["title"]);
    ASSERT_EQ(1, res_obj["hits"][1]["document"]["Coll_C"][1]["Coll_D"].size());
    ASSERT_EQ("coll_d_2", res_obj["hits"][1]["document"]["Coll_C"][1]["Coll_D"][0]["title"]);

    ASSERT_EQ(2, res_obj["hits"][2]["document"].size());
    ASSERT_EQ("coll_b_0", res_obj["hits"][2]["document"]["title"]);
    ASSERT_EQ(1, res_obj["hits"][2]["document"]["Coll_C"].size());
    ASSERT_EQ("coll_c_2", res_obj["hits"][2]["document"]["Coll_C"][0]["title"]);
    ASSERT_EQ(1, res_obj["hits"][2]["document"]["Coll_C"][0]["Coll_D"].size());
    ASSERT_EQ("coll_d_2", res_obj["hits"][2]["document"]["Coll_C"][0]["Coll_D"][0]["title"]);

    req_params = {
            {"collection", "Coll_D"},
            {"q", "*"},
            {"filter_by", "$Coll_C($Coll_B(id: [0, 1]))"},
            {"include_fields", "title, $Coll_C(title, $Coll_B(title:nest_array):nest_array)"}
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    res_obj = nlohmann::json::parse(json_res);
    LOG(INFO) << res_obj.dump();
    // coll_d_2 -> coll_c_2 -> coll_b_0, coll_b_1
    //
    // coll_d_1 -> coll_c_1 -> coll_b_1
    ASSERT_EQ(2, res_obj["found"].get<size_t>());
    ASSERT_EQ(2, res_obj["hits"].size());
    ASSERT_EQ(2, res_obj["hits"][0]["document"].size());
    ASSERT_EQ("coll_d_2", res_obj["hits"][0]["document"]["title"]);
    ASSERT_EQ(1, res_obj["hits"][0]["document"]["Coll_C"].size());
    ASSERT_EQ("coll_c_2", res_obj["hits"][0]["document"]["Coll_C"][0]["title"]);
    ASSERT_EQ(2, res_obj["hits"][0]["document"]["Coll_C"][0]["Coll_B"].size());
    ASSERT_EQ("coll_b_0", res_obj["hits"][0]["document"]["Coll_C"][0]["Coll_B"][0]["title"]);
    ASSERT_EQ("coll_b_1", res_obj["hits"][0]["document"]["Coll_C"][0]["Coll_B"][1]["title"]);

    ASSERT_EQ(2, res_obj["hits"][1]["document"].size());
    ASSERT_EQ("coll_d_1", res_obj["hits"][1]["document"]["title"]);
    ASSERT_EQ(1, res_obj["hits"][1]["document"]["Coll_C"].size());
    ASSERT_EQ("coll_c_1", res_obj["hits"][1]["document"]["Coll_C"][0]["title"]);
    ASSERT_EQ(1, res_obj["hits"][1]["document"]["Coll_C"][0]["Coll_B"].size());
    ASSERT_EQ("coll_b_1", res_obj["hits"][1]["document"]["Coll_C"][0]["Coll_B"][0]["title"]);

    schema_json =
            R"({
                "name": "products",
                "fields": [
                    {"name": "title", "type": "string"}
                ]
            })"_json;
    documents = {
            R"({
                "title": "shampoo"
            })"_json,
            R"({
                "title": "soap"
            })"_json
    };
    collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_TRUE(collection_create_op.ok());
    for (auto const &json: documents) {
        auto add_op = collection_create_op.get()->add(json.dump());
        if (!add_op.ok()) {
            LOG(INFO) << add_op.error();
        }
        ASSERT_TRUE(add_op.ok());
    }

    schema_json =
            R"({
                "name": "product_variants",
                "fields": [
                    {"name": "title", "type": "string"},
                    {"name": "product_id", "type": "string", "reference": "products.id"}
                ]
            })"_json;
    documents = {
            R"({
                "title": "panteen",
                "product_id": "0"
            })"_json,
            R"({
                "title": "loreal",
                "product_id": "0"
            })"_json,
            R"({
                "title": "pears",
                "product_id": "1"
            })"_json,
            R"({
                "title": "lifebuoy",
                "product_id": "1"
            })"_json
    };
    collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_TRUE(collection_create_op.ok());
    for (auto const &json: documents) {
        auto add_op = collection_create_op.get()->add(json.dump());
        if (!add_op.ok()) {
            LOG(INFO) << add_op.error();
        }
        ASSERT_TRUE(add_op.ok());
    }

    schema_json =
            R"({
                "name": "retailers",
                "fields": [
                    {"name": "title", "type": "string"},
                    {"name": "location", "type": "geopoint"}
                ]
            })"_json;
    documents = {
            R"({
                "title": "retailer 1",
                "location": [48.872576479306765, 2.332291112241466]
            })"_json,
            R"({
                "title": "retailer 2",
                "location": [48.888286721920934, 2.342340862419206]
            })"_json,
            R"({
                "title": "retailer 3",
                "location": [48.87538726829884, 2.296113163780903]
            })"_json
    };
    collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_TRUE(collection_create_op.ok());
    for (auto const &json: documents) {
        auto add_op = collection_create_op.get()->add(json.dump());
        if (!add_op.ok()) {
            LOG(INFO) << add_op.error();
        }
        ASSERT_TRUE(add_op.ok());
    }

    schema_json =
            R"({
                "name": "inventory",
                "fields": [
                    {"name": "qty", "type": "int32"},
                    {"name": "retailer_id", "type": "string", "reference": "retailers.id"},
                    {"name": "product_variant_id", "type": "string", "reference": "product_variants.id"}
                ]
            })"_json;
    documents = {
            R"({
                "qty": "1",
                "retailer_id": "0",
                "product_variant_id": "0"
            })"_json,
            R"({
                "qty": "2",
                "retailer_id": "0",
                "product_variant_id": "1"
            })"_json,
            R"({
                "qty": "3",
                "retailer_id": "0",
                "product_variant_id": "2"
            })"_json,
            R"({
                "qty": "4",
                "retailer_id": "0",
                "product_variant_id": "3"
            })"_json,
            R"({
                "qty": "5",
                "retailer_id": "1",
                "product_variant_id": "0"
            })"_json,
            R"({
                "qty": "6",
                "retailer_id": "1",
                "product_variant_id": "1"
            })"_json,
            R"({
                "qty": "7",
                "retailer_id": "1",
                "product_variant_id": "2"
            })"_json,
            R"({
                "qty": "8",
                "retailer_id": "1",
                "product_variant_id": "3"
            })"_json,
            R"({
                "qty": "9",
                "retailer_id": "2",
                "product_variant_id": "0"
            })"_json,
            R"({
                "qty": "10",
                "retailer_id": "2",
                "product_variant_id": "1"
            })"_json,
            R"({
                "qty": "11",
                "retailer_id": "2",
                "product_variant_id": "2"
            })"_json,
            R"({
                "qty": "12",
                "retailer_id": "2",
                "product_variant_id": "3"
            })"_json,
    };
    collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_TRUE(collection_create_op.ok());
    for (auto const &json: documents) {
        auto add_op = collection_create_op.get()->add(json.dump());
        if (!add_op.ok()) {
            LOG(INFO) << add_op.error();
        }
        ASSERT_TRUE(add_op.ok());
    }

    req_params = {
            {"collection", "products"},
            {"q", "*"},
            {"filter_by", "$product_variants($inventory($retailers(location:(48.87538726829884, 2.296113163780903,1 km))))"},
            {"include_fields", "$product_variants(id,$inventory(qty,sku,$retailers(id,title)))"}
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(2, res_obj["found"].get<size_t>());
    ASSERT_EQ(2, res_obj["hits"].size());
    ASSERT_EQ("1", res_obj["hits"][0]["document"]["id"]);
    ASSERT_EQ("soap", res_obj["hits"][0]["document"]["title"]);
    ASSERT_EQ(2, res_obj["hits"][0]["document"]["product_variants"].size());

    ASSERT_EQ("2", res_obj["hits"][0]["document"]["product_variants"][0]["id"]);
    ASSERT_EQ(2, res_obj["hits"][0]["document"]["product_variants"][0]["inventory"].size());
    ASSERT_EQ(11, res_obj["hits"][0]["document"]["product_variants"][0]["inventory"]["qty"]);
    ASSERT_EQ(2, res_obj["hits"][0]["document"]["product_variants"][0]["inventory"]["retailers"].size());
    ASSERT_EQ("2", res_obj["hits"][0]["document"]["product_variants"][0]["inventory"]["retailers"]["id"]);
    ASSERT_EQ("retailer 3", res_obj["hits"][0]["document"]["product_variants"][0]["inventory"]["retailers"]["title"]);

    ASSERT_EQ("3", res_obj["hits"][0]["document"]["product_variants"][1]["id"]);
    ASSERT_EQ(2, res_obj["hits"][0]["document"]["product_variants"][1]["inventory"].size());
    ASSERT_EQ(12, res_obj["hits"][0]["document"]["product_variants"][1]["inventory"]["qty"]);
    ASSERT_EQ(2, res_obj["hits"][0]["document"]["product_variants"][1]["inventory"]["retailers"].size());
    ASSERT_EQ("2", res_obj["hits"][0]["document"]["product_variants"][1]["inventory"]["retailers"]["id"]);
    ASSERT_EQ("retailer 3", res_obj["hits"][0]["document"]["product_variants"][1]["inventory"]["retailers"]["title"]);

    ASSERT_EQ("0", res_obj["hits"][1]["document"]["id"]);
    ASSERT_EQ("shampoo", res_obj["hits"][1]["document"]["title"]);
    ASSERT_EQ(2, res_obj["hits"][1]["document"]["product_variants"].size());

    ASSERT_EQ("0", res_obj["hits"][1]["document"]["product_variants"][0]["id"]);
    ASSERT_EQ(2, res_obj["hits"][1]["document"]["product_variants"][0]["inventory"].size());
    ASSERT_EQ(9, res_obj["hits"][1]["document"]["product_variants"][0]["inventory"]["qty"]);
    ASSERT_EQ(2, res_obj["hits"][1]["document"]["product_variants"][0]["inventory"]["retailers"].size());
    ASSERT_EQ("2", res_obj["hits"][1]["document"]["product_variants"][0]["inventory"]["retailers"]["id"]);
    ASSERT_EQ("retailer 3", res_obj["hits"][1]["document"]["product_variants"][0]["inventory"]["retailers"]["title"]);

    ASSERT_EQ("1", res_obj["hits"][1]["document"]["product_variants"][1]["id"]);
    ASSERT_EQ(2, res_obj["hits"][1]["document"]["product_variants"][1]["inventory"].size());
    ASSERT_EQ(10, res_obj["hits"][1]["document"]["product_variants"][1]["inventory"]["qty"]);
    ASSERT_EQ(2, res_obj["hits"][1]["document"]["product_variants"][1]["inventory"]["retailers"].size());
    ASSERT_EQ("2", res_obj["hits"][1]["document"]["product_variants"][1]["inventory"]["retailers"]["id"]);
    ASSERT_EQ("retailer 3", res_obj["hits"][1]["document"]["product_variants"][1]["inventory"]["retailers"]["title"]);

    req_params = {
            {"collection", "products"},
            {"q", "*"},
            {"filter_by", "$product_variants($inventory($retailers(id: [0, 1]) && qty: [4..5]))"},
            {"include_fields", "$product_variants(id,$inventory(qty,sku,$retailers(id,title)))"},
            {"exclude_fields", "$product_variants($inventory($retailers(id)))"}
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(2, res_obj["found"].get<size_t>());
    ASSERT_EQ(2, res_obj["hits"].size());
    ASSERT_EQ("1", res_obj["hits"][0]["document"]["id"]);
    ASSERT_EQ("soap", res_obj["hits"][0]["document"]["title"]);
    ASSERT_EQ("3", res_obj["hits"][0]["document"]["product_variants"]["id"]);
    ASSERT_EQ(2, res_obj["hits"][0]["document"]["product_variants"]["inventory"].size());
    ASSERT_EQ(4, res_obj["hits"][0]["document"]["product_variants"]["inventory"]["qty"]);
    ASSERT_EQ(1, res_obj["hits"][0]["document"]["product_variants"]["inventory"]["retailers"].size());
    ASSERT_EQ("retailer 1", res_obj["hits"][0]["document"]["product_variants"]["inventory"]["retailers"]["title"]);

    ASSERT_EQ("0", res_obj["hits"][1]["document"]["id"]);
    ASSERT_EQ("shampoo", res_obj["hits"][1]["document"]["title"]);
    ASSERT_EQ("0", res_obj["hits"][1]["document"]["product_variants"]["id"]);
    ASSERT_EQ(2, res_obj["hits"][1]["document"]["product_variants"]["inventory"].size());
    ASSERT_EQ(5, res_obj["hits"][1]["document"]["product_variants"]["inventory"]["qty"]);
    ASSERT_EQ(1, res_obj["hits"][1]["document"]["product_variants"]["inventory"]["retailers"].size());
    ASSERT_EQ("retailer 2", res_obj["hits"][1]["document"]["product_variants"]["inventory"]["retailers"]["title"]);
}

TEST_F(CollectionJoinTest, IncludeExcludeFieldsByReference) {
    auto schema_json =
            R"({
                "name": "Products",
                "fields": [
                    {"name": "product_id", "type": "string"},
                    {"name": "product_name", "type": "string", "infix": true},
                    {"name": "product_description", "type": "string"},
                    {"name": "embedding", "type":"float[]", "embed":{"from": ["product_description"], "model_config": {"model_name": "ts/e5-small"}}},
                    {"name": "rating", "type": "int32"}
                ]
            })"_json;
    std::vector<nlohmann::json> documents = {
            R"({
                "product_id": "product_a",
                "product_name": "shampoo",
                "product_description": "Our new moisturizing shampoo is perfect for those with dry or damaged hair.",
                "rating": "2"
            })"_json,
            R"({
                "product_id": "product_b",
                "product_name": "soap",
                "product_description": "Introducing our all-natural, organic soap bar made with essential oils and botanical ingredients.",
                "rating": "4"
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

    schema_json =
            R"({
                "name": "Customers",
                "fields": [
                    {"name": "customer_id", "type": "string"},
                    {"name": "customer_name", "type": "string"},
                    {"name": "product_price", "type": "float"},
                    {"name": "product_id", "type": "string", "reference": "Products.product_id"}
                ]
            })"_json;
    documents = {
            R"({
                "customer_id": "customer_a",
                "customer_name": "Joe",
                "product_price": 143,
                "product_id": "product_a"
            })"_json,
            R"({
                "customer_id": "customer_a",
                "customer_name": "Joe",
                "product_price": 73.5,
                "product_id": "product_b"
            })"_json,
            R"({
                "customer_id": "customer_b",
                "customer_name": "Dan",
                "product_price": 75,
                "product_id": "product_a"
            })"_json,
            R"({
                "customer_id": "customer_b",
                "customer_name": "Dan",
                "product_price": 140,
                "product_id": "product_b"
            })"_json
    };
    collection_create_op = collectionManager.create_collection(schema_json);
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
            {"query_by", "product_name"},
            {"filter_by", "$Customers(customer_id:=customer_a && product_price:<100)"},
            {"include_fields", "$foo.bar"}
    };
    nlohmann::json embedded_params;
    std::string json_res;
    auto now_ts = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

    auto search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_FALSE(search_op.ok());
    ASSERT_EQ("Invalid reference `$foo.bar` in include_fields/exclude_fields, expected `$CollectionName(fieldA, ...)`.",
              search_op.error());

    req_params["include_fields"] = "$foo(bar";
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_FALSE(search_op.ok());
    ASSERT_EQ("Invalid reference `$foo(bar` in include_fields/exclude_fields, expected `$CollectionName(fieldA, ...)`.",
              search_op.error());

    req_params["include_fields"] = "$foo(bar)";
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_FALSE(search_op.ok());
    ASSERT_EQ("Referenced collection `foo` in `include_fields` not found.", search_op.error());

    req_params = {
            {"collection", "Products"},
            {"q", "*"},
            {"query_by", "product_name"},
            {"filter_by", "$Customers(customer_id:=customer_a && product_price:<100)"},
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);

    nlohmann::json res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(1, res_obj["found"].get<size_t>());
    ASSERT_EQ(1, res_obj["hits"].size());
    // No fields are mentioned in `include_fields`, should include all fields of Products and Customers by default.
    ASSERT_EQ(7, res_obj["hits"][0]["document"].size());
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("id"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("product_id"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("product_name"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("product_description"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("embedding"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("rating"));
    // Default strategy of reference includes is nest. No alias was provided, collection name becomes the field name.
    ASSERT_EQ(5, res_obj["hits"][0]["document"]["Customers"].size());
    ASSERT_EQ(1, res_obj["hits"][0]["document"]["Customers"].count("customer_id"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"]["Customers"].count("customer_name"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"]["Customers"].count("id"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"]["Customers"].count("product_id"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"]["Customers"].count("product_price"));

    req_params = {
            {"collection", "Products"},
            {"q", "*"},
            {"query_by", "product_name"},
            {"filter_by", "$Customers(customer_id:=customer_a && product_price:<100)"},
            {"include_fields", "*, $Customers(*:nest_array) as Customers"}
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(1, res_obj["found"].get<size_t>());
    ASSERT_EQ(1, res_obj["hits"].size());
    // No fields are mentioned in `include_fields`, should include all fields of Products and Customers by default.
    ASSERT_EQ(7, res_obj["hits"][0]["document"].size());
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("id"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("product_id"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("product_name"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("product_description"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("embedding"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("rating"));
    // In nest_array strategy we return the referenced docs in an array.
    ASSERT_EQ(1, res_obj["hits"][0]["document"]["Customers"].size());
    ASSERT_EQ(1, res_obj["hits"][0]["document"]["Customers"][0].count("customer_id"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"]["Customers"][0].count("customer_name"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"]["Customers"][0].count("id"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"]["Customers"][0].count("product_id"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"]["Customers"][0].count("product_price"));

    req_params = {
            {"collection", "Products"},
            {"q", "*"},
            {"query_by", "product_name"},
            {"filter_by", "$Customers(customer_id:=customer_a && product_price:<100)"},
            {"include_fields", "*, $Customers(*:merge) as Customers"}
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(1, res_obj["found"].get<size_t>());
    ASSERT_EQ(1, res_obj["hits"].size());
    ASSERT_EQ(11, res_obj["hits"][0]["document"].size());
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("id"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("product_id"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("product_name"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("product_description"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("embedding"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("rating"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("Customers.customer_id"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("Customers.customer_name"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("Customers.id"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("Customers.product_price"));

    req_params = {
        {"collection", "Products"},
        {"q", "*"},
        {"query_by", "product_name"},
        {"filter_by", "$Customers(customer_id:=customer_a && product_price:<100)"},
        {"include_fields", "$Customers(bar:merge)"}
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(1, res_obj["found"].get<size_t>());
    ASSERT_EQ(1, res_obj["hits"].size());
    // No fields of Products collection are mentioned in `include_fields`, should include all of its fields by default.
    ASSERT_EQ(6, res_obj["hits"][0]["document"].size());
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("id"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("product_id"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("product_name"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("product_description"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("embedding"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("rating"));

    req_params = {
            {"collection", "Products"},
            {"q", "*"},
            {"query_by", "product_name"},
            {"filter_by", "$Customers(customer_id:=customer_a && product_price:<100)"},
            {"include_fields", "$Customers(product_price:merge)"}
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(1, res_obj["found"].get<size_t>());
    ASSERT_EQ(1, res_obj["hits"].size());
    ASSERT_EQ(7, res_obj["hits"][0]["document"].size());
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("product_price"));
    ASSERT_EQ(73.5, res_obj["hits"][0]["document"].at("product_price"));

    req_params = {
            {"collection", "Products"},
            {"q", "*"},
            {"query_by", "product_name"},
            {"filter_by", "$Customers(customer_id:=customer_a && product_price:<100)"},
            {"include_fields", "$Customers(product_price, customer_id:merge)"}
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(1, res_obj["found"].get<size_t>());
    ASSERT_EQ(1, res_obj["hits"].size());
    ASSERT_EQ(8, res_obj["hits"][0]["document"].size());
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("product_price"));
    ASSERT_EQ(73.5, res_obj["hits"][0]["document"].at("product_price"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("customer_id"));
    ASSERT_EQ("customer_a", res_obj["hits"][0]["document"].at("customer_id"));

    req_params = {
            {"collection", "Products"},
            {"q", "*"},
            {"query_by", "product_name"},
            {"filter_by", "$Customers(customer_id:=customer_a && product_price:<100)"},
            {"include_fields", "*, $Customers(product_price, customer_id:merge)"}
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(1, res_obj["found"].get<size_t>());
    ASSERT_EQ(1, res_obj["hits"].size());
    // 6 fields in Products document and 2 fields from Customers document
    ASSERT_EQ(8, res_obj["hits"][0]["document"].size());

    req_params = {
            {"collection", "Products"},
            {"q", "*"},
            {"query_by", "product_name"},
            {"filter_by", "$Customers(customer_id:=customer_a && product_price:<100)"},
            {"include_fields", "$Customers(product*:merge)"}
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(1, res_obj["found"].get<size_t>());
    ASSERT_EQ(1, res_obj["hits"].size());
    // 6 fields in Products document and 1 field from Customers document
    ASSERT_EQ(7, res_obj["hits"][0]["document"].size());
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("product_price"));

    req_params = {
            {"collection", "Products"},
            {"q", "s"},
            {"query_by", "product_name"},
            {"filter_by", "$Customers(customer_id:=customer_a && product_price:<100)"},
            {"include_fields", "$Customers(product*:merge)"},
            {"exclude_fields", "$Customers(product_id_sequence_id)"}
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(1, res_obj["found"].get<size_t>());
    ASSERT_EQ(1, res_obj["hits"].size());
    // 6 fields in Products document and 1 fields from Customers document
    ASSERT_EQ(7, res_obj["hits"][0]["document"].size());
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("product_id"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("product_name"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("product_description"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("product_price"));
    ASSERT_EQ(73.5, res_obj["hits"][0]["document"].at("product_price"));

    // Exclude token search
    req_params = {
            {"collection", "Products"},
            {"q", "-shampoo"},
            {"query_by", "product_name"},
            {"filter_by", "$Customers(product_price:<100)"}, // This filter will match both shampoo and soap.
            {"include_fields", "product_name"},
            {"exclude_fields", "$Customers(*)"}
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(1, res_obj["found"].get<size_t>());
    ASSERT_EQ(1, res_obj["hits"].size());
    ASSERT_EQ(1, res_obj["hits"][0]["document"].size());
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("product_name"));
    ASSERT_EQ("soap", res_obj["hits"][0]["document"].at("product_name"));

    // Phrase search
    req_params = {
            {"collection", "Products"},
            {"q", R"("soap")"},
            {"query_by", "product_name"},
            {"filter_by", "$Customers(product_price:<100)"}, // This filter will match both shampoo and soap.
            {"include_fields", "product_name"},
            {"exclude_fields", "$Customers(*)"}
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(1, res_obj["found"].get<size_t>());
    ASSERT_EQ(1, res_obj["hits"].size());
    ASSERT_EQ(1, res_obj["hits"][0]["document"].size());
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("product_name"));
    ASSERT_EQ("soap", res_obj["hits"][0]["document"].at("product_name"));

    // Combining normal and reference filter
    req_params = {
            {"collection", "Products"},
            {"q", "*"},
            {"query_by", "product_name"},
            {"filter_by", "product_name:soap && $Customers(product_price:>100)"},
            {"include_fields", "product_name, $Customers(product_price:merge)"},
            {"exclude_fields", ""}
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(1, res_obj["found"].get<size_t>());
    ASSERT_EQ(1, res_obj["hits"].size());
    ASSERT_EQ(2, res_obj["hits"][0]["document"].size());
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("product_name"));
    ASSERT_EQ("soap", res_obj["hits"][0]["document"].at("product_name"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("product_price"));
    ASSERT_EQ(140, res_obj["hits"][0]["document"].at("product_price"));

    // Multiple references
    req_params = {
            {"collection", "Products"},
            {"q", "soap"},
            {"query_by", "product_name"},
            {"filter_by", "$Customers(product_price: >0)"},
            {"include_fields", "product_name, $Customers(customer_name, product_price:merge)"},
            {"exclude_fields", ""}
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(1, res_obj["found"].get<size_t>());
    ASSERT_EQ(1, res_obj["hits"].size());
    ASSERT_EQ(3, res_obj["hits"][0]["document"].size());
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("product_name"));
    ASSERT_EQ("soap", res_obj["hits"][0]["document"].at("product_name"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("customer_name"));
    ASSERT_EQ("Joe", res_obj["hits"][0]["document"].at("customer_name").at(0));
    ASSERT_EQ("Dan", res_obj["hits"][0]["document"].at("customer_name").at(1));
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("product_price"));
    ASSERT_EQ(73.5, res_obj["hits"][0]["document"].at("product_price").at(0));
    ASSERT_EQ(140, res_obj["hits"][0]["document"].at("product_price").at(1));

    // Vector search
    req_params = {
            {"collection", "Products"},
            {"q", "natural products"},
            {"query_by", "embedding"},
            {"filter_by", "$Customers(customer_id:=customer_a && product_price:<100)"},
            {"include_fields", "product_name, $Customers(product_price:merge)"},
            {"exclude_fields", ""}
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(1, res_obj["found"].get<size_t>());
    ASSERT_EQ(1, res_obj["hits"].size());
    ASSERT_EQ(2, res_obj["hits"][0]["document"].size());
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("product_name"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("product_price"));
    ASSERT_EQ(73.5, res_obj["hits"][0]["document"].at("product_price"));

    nlohmann::json model_config = R"({
        "model_name": "ts/e5-small"
    })"_json;
    auto query_embedding = EmbedderManager::get_instance().get_text_embedder(model_config).get()->Embed("natural products");
    std::string vec_string = "[";
    for (auto const& i : query_embedding.embedding) {
        vec_string += std::to_string(i);
        vec_string += ",";
    }
    vec_string[vec_string.size() - 1] = ']';

    req_params = {
            {"collection", "Products"},
            {"q", "*"},
            {"vector_query", "embedding:(" + vec_string + ", flat_search_cutoff: 0)"},
            {"filter_by", "$Customers(customer_id:=customer_a && product_price:<100)"},
            {"include_fields", "product_name, $Customers(product_price : merge)"},
            {"exclude_fields", ""}
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(1, res_obj["found"].get<size_t>());
    ASSERT_EQ(1, res_obj["hits"].size());
    ASSERT_EQ(2, res_obj["hits"][0]["document"].size());
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("product_name"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("product_price"));
    ASSERT_EQ(73.5, res_obj["hits"][0]["document"].at("product_price"));

    // Hybrid search - Both text match and vector match
    req_params = {
            {"collection", "Products"},
            {"q", "soap"},
            {"query_by", "product_name, embedding"},
            {"filter_by", "$Customers(customer_id:=customer_a && product_price:<100)"},
            {"include_fields", "product_name, $Customers(product_price: merge)"},
            {"exclude_fields", ""}
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(1, res_obj["found"].get<size_t>());
    ASSERT_EQ(1, res_obj["hits"].size());
    ASSERT_EQ(2, res_obj["hits"][0]["document"].size());
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("product_name"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("product_price"));
    ASSERT_EQ(73.5, res_obj["hits"][0]["document"].at("product_price"));
    ASSERT_NE(0, res_obj["hits"][0].at("text_match"));
    ASSERT_NE(0, res_obj["hits"][0].at("vector_distance"));

    // Hybrid search - Only vector match
    req_params = {
            {"collection", "Products"},
            {"q", "natural products"},
            {"query_by", "product_name, embedding"},
            {"filter_by", "$Customers(customer_id:=customer_a && product_price:<100)"},
            {"include_fields", "product_name, $Customers(product_price :merge)"},
            {"exclude_fields", ""}
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(1, res_obj["found"].get<size_t>());
    ASSERT_EQ(1, res_obj["hits"].size());
    ASSERT_EQ(2, res_obj["hits"][0]["document"].size());
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("product_name"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("product_price"));
    ASSERT_EQ(73.5, res_obj["hits"][0]["document"].at("product_price"));
    ASSERT_EQ(0, res_obj["hits"][0].at("text_match"));
    ASSERT_NE(0, res_obj["hits"][0].at("vector_distance"));

    // Infix search
    req_params = {
            {"collection", "Products"},
            {"q", "ap"},
            {"query_by", "product_name"},
            {"infix", "always"},
            {"filter_by", "$Customers(customer_id:=customer_a && product_price:<100)"},
            {"include_fields", "product_name, $Customers(product_price:merge)"},
            {"exclude_fields", ""}
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(1, res_obj["found"].get<size_t>());
    ASSERT_EQ(1, res_obj["hits"].size());
    ASSERT_EQ(2, res_obj["hits"][0]["document"].size());
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("product_name"));
    ASSERT_EQ("soap", res_obj["hits"][0]["document"].at("product_name"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("product_price"));
    ASSERT_EQ(73.5, res_obj["hits"][0]["document"].at("product_price"));

    req_params = {
            {"collection", "Customers"},
            {"q", "Dan"},
            {"query_by", "customer_name"},
            {"filter_by", "$Products(rating:>3)"},
            {"include_fields", "$Products(product_name:merge), product_price"}
    };

    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(1, res_obj["found"].get<size_t>());
    ASSERT_EQ(1, res_obj["hits"].size());
    ASSERT_EQ(2, res_obj["hits"][0]["document"].size());
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("product_name"));
    ASSERT_EQ("soap", res_obj["hits"][0]["document"].at("product_name"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("product_price"));
    ASSERT_EQ(140, res_obj["hits"][0]["document"].at("product_price"));

    // Reference include_by without join
    req_params = {
            {"collection", "Customers"},
            {"q", "Joe"},
            {"query_by", "customer_name"},
            {"filter_by", "product_price:<100"},
            {"include_fields", "$Products(product_name: merge), product_price"}
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(1, res_obj["found"].get<size_t>());
    ASSERT_EQ(1, res_obj["hits"].size());
    ASSERT_EQ(2, res_obj["hits"][0]["document"].size());
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("product_name"));
    ASSERT_EQ("soap", res_obj["hits"][0]["document"].at("product_name"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("product_price"));
    ASSERT_EQ(73.5, res_obj["hits"][0]["document"].at("product_price"));

    // Add alias using `as`
    req_params = {
            {"collection", "Products"},
            {"q", "soap"},
            {"query_by", "product_name"},
            {"filter_by", "$Customers(id:*)"},
            {"include_fields", "id, $Customers(id :merge)"}
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_FALSE(search_op.ok());
    ASSERT_EQ("Could not include the value of `id` key of the reference document of `Customers` collection."
              " Expected `id` to be an array. Try adding an alias.", search_op.error());

    req_params = {
            {"collection", "Products"},
            {"q", "soap"},
            {"query_by", "product_name"},
            {"filter_by", "$Customers(id:*)"},
            {"include_fields", "id, $Customers(id :nest) as id"}
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_FALSE(search_op.ok());
    ASSERT_EQ("Could not include the reference document of `Customers` collection."
              " Expected `id` to be an array. Try renaming the alias.", search_op.error());

    req_params = {
            {"collection", "Customers"},
            {"q", "Joe"},
            {"query_by", "customer_name"},
            {"filter_by", "product_price:<100"},
            // With merge, alias is prepended
            {"include_fields", "$Products(product_name:merge) as prod, product_price"}
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(1, res_obj["found"].get<size_t>());
    ASSERT_EQ(1, res_obj["hits"].size());
    ASSERT_EQ(2, res_obj["hits"][0]["document"].size());
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("prod.product_name"));
    ASSERT_EQ("soap", res_obj["hits"][0]["document"].at("prod.product_name"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("product_price"));
    ASSERT_EQ(73.5, res_obj["hits"][0]["document"].at("product_price"));

    req_params = {
            {"collection", "Customers"},
            {"q", "Joe"},
            {"query_by", "customer_name"},
            {"filter_by", "product_price:<100"},
            // With nest, alias becomes the key
            {"include_fields", "$Products(product_name:nest) as prod, product_price"}
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(1, res_obj["found"].get<size_t>());
    ASSERT_EQ(1, res_obj["hits"].size());
    ASSERT_EQ(2, res_obj["hits"][0]["document"].size());
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("prod"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"]["prod"].count("product_name"));
    ASSERT_EQ("soap", res_obj["hits"][0]["document"]["prod"].at("product_name"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("product_price"));
    ASSERT_EQ(73.5, res_obj["hits"][0]["document"].at("product_price"));

    req_params = {
            {"collection", "Products"},
            {"q", "soap"},
            {"query_by", "product_name"},
            {"filter_by", "$Customers(id:*)"},
            // With nest, alias becomes the key
            {"include_fields", "$Customers(customer_name, product_price :nest) as CustomerPrices, product_name"}
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(1, res_obj["found"].get<size_t>());
    ASSERT_EQ(1, res_obj["hits"].size());
    ASSERT_EQ(2, res_obj["hits"][0]["document"].size());
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("product_name"));
    ASSERT_EQ("soap", res_obj["hits"][0]["document"]["product_name"]);
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("CustomerPrices"));
    ASSERT_EQ(2, res_obj["hits"][0]["document"]["CustomerPrices"].size());

    ASSERT_EQ("Joe", res_obj["hits"][0]["document"]["CustomerPrices"].at(0)["customer_name"]);
    ASSERT_EQ(73.5, res_obj["hits"][0]["document"]["CustomerPrices"].at(0)["product_price"]);

    ASSERT_EQ("Dan", res_obj["hits"][0]["document"]["CustomerPrices"].at(1)["customer_name"]);
    ASSERT_EQ(140, res_obj["hits"][0]["document"]["CustomerPrices"].at(1)["product_price"]);

    schema_json =
            R"({
                "name": "Users",
                "fields": [
                    {"name": "user_id", "type": "string"},
                    {"name": "user_name", "type": "string"}
                ]
            })"_json;
    documents = {
            R"({
                "user_id": "user_a",
                "user_name": "Roshan"
            })"_json,
            R"({
                "user_id": "user_b",
                "user_name": "Ruby"
            })"_json,
            R"({
                "user_id": "user_c",
                "user_name": "Joe"
            })"_json,
            R"({
                "user_id": "user_d",
                "user_name": "Aby"
            })"_json
    };
    collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_TRUE(collection_create_op.ok());
    for (auto const &json: documents) {
        auto add_op = collection_create_op.get()->add(json.dump());
        if (!add_op.ok()) {
            LOG(INFO) << add_op.error();
        }
        ASSERT_TRUE(add_op.ok());
    }

    schema_json =
            R"({
                "name": "Repos",
                "fields": [
                    {"name": "repo_id", "type": "string"},
                    {"name": "repo_content", "type": "string"},
                    {"name": "repo_stars", "type": "int32"},
                    {"name": "repo_is_private", "type": "bool"}
                ]
            })"_json;
    documents = {
            R"({
                "repo_id": "repo_a",
                "repo_content": "body1",
                "repo_stars": 431,
                "repo_is_private": true
            })"_json,
            R"({
                "repo_id": "repo_b",
                "repo_content": "body2",
                "repo_stars": 4562,
                "repo_is_private": false
            })"_json,
            R"({
                "repo_id": "repo_c",
                "repo_content": "body3",
                "repo_stars": 945,
                "repo_is_private": false
            })"_json
    };
    collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_TRUE(collection_create_op.ok());
    for (auto const &json: documents) {
        auto add_op = collection_create_op.get()->add(json.dump());
        if (!add_op.ok()) {
            LOG(INFO) << add_op.error();
        }
        ASSERT_TRUE(add_op.ok());
    }

    schema_json =
            R"({
                "name": "Links",
                "fields": [
                    {"name": "repo_id", "type": "string", "reference": "Repos.repo_id"},
                    {"name": "user_id", "type": "string", "reference": "Users.user_id"}
                ]
            })"_json;
    documents = {
            R"({
                "repo_id": "repo_a",
                "user_id": "user_b"
            })"_json,
            R"({
                "repo_id": "repo_a",
                "user_id": "user_c"
            })"_json,
            R"({
                "repo_id": "repo_b",
                "user_id": "user_a"
            })"_json,
            R"({
                "repo_id": "repo_b",
                "user_id": "user_b"
            })"_json,
            R"({
                "repo_id": "repo_b",
                "user_id": "user_d"
            })"_json,
            R"({
                "repo_id": "repo_c",
                "user_id": "user_a"
            })"_json,
            R"({
                "repo_id": "repo_c",
                "user_id": "user_b"
            })"_json,
            R"({
                "repo_id": "repo_c",
                "user_id": "user_c"
            })"_json,
            R"({
                "repo_id": "repo_c",
                "user_id": "user_d"
            })"_json
    };
    collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_TRUE(collection_create_op.ok());

    for (auto const &json: documents) {
        auto add_op = collection_create_op.get()->add(json.dump());
        if (!add_op.ok()) {
            LOG(INFO) << add_op.error();
        }
        ASSERT_TRUE(add_op.ok());
    }

    schema_json =
            R"({
                "name": "Organizations",
                "fields": [
                    {"name": "org_id", "type": "string"},
                    {"name": "name", "type": "object"},
                    {"name": "name.first", "type": "string"},
                    {"name": "name.last", "type": "string"}
                ],
                "enable_nested_fields": true
            })"_json;
    documents = {
            R"({
                "org_id": "org_a",
                "name": {
                    "first": "type",
                    "last": "sense"
                }
            })"_json
    };
    collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_TRUE(collection_create_op.ok());

    for (auto const &json: documents) {
        auto add_op = collection_create_op.get()->add(json.dump());
        if (!add_op.ok()) {
            LOG(INFO) << add_op.error();
        }
        ASSERT_TRUE(add_op.ok());
    }

    schema_json =
            R"({
                "name": "Participants",
                "fields": [
                    {"name": "user_id", "type": "string", "reference": "Users.user_id"},
                    {"name": "org_id", "type": "string", "reference": "Organizations.org_id"}
                ]
            })"_json;
    documents = {
            R"({
                "user_id": "user_a",
                "org_id": "org_a"
            })"_json,
            R"({
                "user_id": "user_b",
                "org_id": "org_a"
            })"_json,
            R"({
                "user_id": "user_d",
                "org_id": "org_a"
            })"_json,
    };
    collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_TRUE(collection_create_op.ok());

    for (auto const &json: documents) {
        auto add_op = collection_create_op.get()->add(json.dump());
        if (!add_op.ok()) {
            LOG(INFO) << add_op.error();
        }
        ASSERT_TRUE(add_op.ok());
    }

    // Search for users within an organization with access to a particular repo.
    req_params = {
            {"collection", "Users"},
            {"q", "R"},
            {"query_by", "user_name"},
            {"filter_by", "$Participants(org_id:=org_a) && $Links(repo_id:=repo_b)"},
            {"include_fields", "user_id, user_name, $Repos(repo_content:merge), $Organizations(name:merge) as org"},
            {"exclude_fields", "$Participants(*), $Links(*), "}
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(2, res_obj["found"].get<size_t>());
    ASSERT_EQ(2, res_obj["hits"].size());
    ASSERT_EQ(4, res_obj["hits"][0]["document"].size());

    ASSERT_EQ("user_b", res_obj["hits"][0]["document"].at("user_id"));
    ASSERT_EQ("Ruby", res_obj["hits"][0]["document"].at("user_name"));
    ASSERT_EQ("body2", res_obj["hits"][0]["document"].at("repo_content"));
    ASSERT_EQ("type", res_obj["hits"][0]["document"]["org.name"].at("first"));
    ASSERT_EQ("sense", res_obj["hits"][0]["document"]["org.name"].at("last"));

    ASSERT_EQ("user_a", res_obj["hits"][1]["document"].at("user_id"));
    ASSERT_EQ("Roshan", res_obj["hits"][1]["document"].at("user_name"));
    ASSERT_EQ("body2", res_obj["hits"][1]["document"].at("repo_content"));
    ASSERT_EQ("type", res_obj["hits"][0]["document"]["org.name"].at("first"));
    ASSERT_EQ("sense", res_obj["hits"][0]["document"]["org.name"].at("last"));
}

TEST_F(CollectionJoinTest, FilterByReferenceArrayField) {
    auto schema_json =
            R"({
                "name": "genres",
                "fields": [
                    { "name": "id", "type": "string" },
                    { "name": "name", "type": "string" }
                ]
            })"_json;
    std::vector<nlohmann::json> documents = {
            R"({"id":"0","name":"Grunge"})"_json,
            R"({"id":"1","name":"Arena rock"})"_json,
            R"({"id":"2","name":"Blues"})"_json
    };
    auto collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_TRUE(collection_create_op.ok());
    for (auto const &json: documents) {
        auto add_op = collection_create_op.get()->add(json.dump());
        if (!add_op.ok()) {
            LOG(INFO) << add_op.error();
        }
        ASSERT_TRUE(add_op.ok());
    }

    schema_json =
            R"({
                "name": "songs",
                "fields": [
                    { "name": "title", "type": "string" },
                    { "name": "genres", "type": "string[]", "reference": "genres.id"}
                ]
           })"_json;
    documents = {
            R"({"title":"Dil De Rani", "genres":[]})"_json,
            R"({"title":"Corduroy", "genres":["0"]})"_json,
            R"({"title":"Achilles Last Stand", "genres":["1","2"]})"_json
    };
    collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_TRUE(collection_create_op.ok());
    for (auto const &json: documents) {
        auto add_op = collection_create_op.get()->add(json.dump());
        if (!add_op.ok()) {
            LOG(INFO) << add_op.error();
        }
        ASSERT_TRUE(add_op.ok());
    }

    std::map<std::string, std::string> req_params = {
            {"collection", "songs"},
            {"q", "*"},
            {"include_fields", "$genres(name:merge) as genre"},
            {"exclude_fields", "genres_sequence_id"},
    };
    nlohmann::json embedded_params;
    std::string json_res;
    auto now_ts = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

    auto search_op_bool = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op_bool.ok());

    auto res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(3, res_obj["found"].get<size_t>());
    ASSERT_EQ(3, res_obj["hits"].size());

    ASSERT_EQ("Achilles Last Stand", res_obj["hits"][0]["document"]["title"].get<std::string>());
    ASSERT_EQ(2, res_obj["hits"][0]["document"]["genre.name"].size());
    ASSERT_EQ("Arena rock", res_obj["hits"][0]["document"]["genre.name"][0]);
    ASSERT_EQ("Blues", res_obj["hits"][0]["document"]["genre.name"][1]);

    ASSERT_EQ("Corduroy", res_obj["hits"][1]["document"]["title"].get<std::string>());
    ASSERT_EQ(1, res_obj["hits"][1]["document"]["genre.name"].size());
    ASSERT_EQ("Grunge", res_obj["hits"][1]["document"]["genre.name"][0]);

    ASSERT_EQ("Dil De Rani", res_obj["hits"][2]["document"]["title"].get<std::string>());
    ASSERT_EQ(0, res_obj["hits"][2]["document"]["genre.name"].size());

    req_params = {
            {"collection", "genres"},
            {"q", "*"},
            {"filter_by", "$songs(id: *)"},
            {"include_fields", "$songs(title:merge) as song"},
    };
    search_op_bool = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op_bool.ok());

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(3, res_obj["found"].get<size_t>());
    ASSERT_EQ(3, res_obj["hits"].size());

    ASSERT_EQ("Blues", res_obj["hits"][0]["document"]["name"].get<std::string>());
    ASSERT_EQ(1, res_obj["hits"][0]["document"]["song.title"].size());
    ASSERT_EQ("Achilles Last Stand", res_obj["hits"][0]["document"]["song.title"][0]);

    ASSERT_EQ("Arena rock", res_obj["hits"][1]["document"]["name"].get<std::string>());
    ASSERT_EQ(1, res_obj["hits"][1]["document"]["song.title"].size());
    ASSERT_EQ("Achilles Last Stand", res_obj["hits"][1]["document"]["song.title"][0]);

    ASSERT_EQ("Grunge", res_obj["hits"][2]["document"]["name"].get<std::string>());
    ASSERT_EQ(1, res_obj["hits"][2]["document"]["song.title"].size());
    ASSERT_EQ("Corduroy", res_obj["hits"][2]["document"]["song.title"][0]);
}

TEST_F(CollectionJoinTest, FilterByObjectReferenceField) {
    auto schema_json =
            R"({
                "name": "Products",
                "fields": [
                    {"name": "product_id", "type": "string"},
                    {"name": "price", "type": "int32"},
                    {"name": "name", "type": "string"}
                ]
            })"_json;
    std::vector<nlohmann::json> documents = {
            R"({
                "product_id": "product_a",
                "price": 50,
                "name": "soap"
            })"_json,
            R"({
                "product_id": "product_b",
                "price": 10,
                "name": "shampoo"
            })"_json,
            R"({
                "product_id": "product_c",
                "price": 120,
                "name": "milk"
            })"_json
    };

    auto collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_TRUE(collection_create_op.ok());
    for (auto const &json: documents) {
        auto add_op = collection_create_op.get()->add(json.dump());
        if (!add_op.ok()) {
            LOG(INFO) << add_op.error();
        }
        ASSERT_TRUE(add_op.ok());
    }

    schema_json =
            R"({
                "name": "coll1",
                "fields": [
                    {"name": "coll_id", "type": "string"},
                    {"name": "object.reference", "type": "string", "reference": "Products.product_id", "optional": true},
                    {"name": "object", "type": "object"}
                ],
                "enable_nested_fields": true
            })"_json;
    documents = {
            R"({
                "coll_id": "a",
                "object": {}
            })"_json,
            R"({
                "coll_id": "b",
                "object": {
                    "reference": "product_c"
                }
            })"_json
    };

    collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_TRUE(collection_create_op.ok());
    for (auto const &json: documents) {
        auto add_op = collection_create_op.get()->add(json.dump());
        if (!add_op.ok()) {
            LOG(INFO) << add_op.error();
        }
        ASSERT_TRUE(add_op.ok());
    }

    std::map<std::string, std::string> req_params = {
            {"collection", "coll1"},
            {"q", "*"},
            {"include_fields", "$Products(product_id)"}
    };
    nlohmann::json embedded_params;
    std::string json_res;
    auto now_ts = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

    auto search_op_bool = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op_bool.ok());

    auto res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(2, res_obj["found"].get<size_t>());
    ASSERT_EQ(2, res_obj["hits"].size());
    ASSERT_EQ(3, res_obj["hits"][0]["document"].size());
    ASSERT_EQ("b", res_obj["hits"][0]["document"]["coll_id"]);
    ASSERT_EQ(2, res_obj["hits"][0]["document"]["object"].size());
    ASSERT_EQ("product_c", res_obj["hits"][0]["document"]["object"]["reference"]);
    ASSERT_EQ(1, res_obj["hits"][0]["document"]["object"].count("Products"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"]["object"]["Products"].size());
    ASSERT_EQ(1, res_obj["hits"][0]["document"]["object"]["Products"].count("product_id"));
    ASSERT_EQ("product_c", res_obj["hits"][0]["document"]["object"]["Products"]["product_id"]);
    ASSERT_EQ(3, res_obj["hits"][1]["document"].size());
    ASSERT_EQ("a", res_obj["hits"][1]["document"]["coll_id"]);
    ASSERT_EQ(0, res_obj["hits"][1]["document"]["object"].size());

    req_params = {
            {"collection", "Products"},
            {"q", "*"},
            {"filter_by", "$coll1(id: *)"},
            {"include_fields", "$coll1(coll_id)"}
    };
    search_op_bool = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op_bool.ok());

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(1, res_obj["found"].get<size_t>());
    ASSERT_EQ(1, res_obj["hits"].size());
    ASSERT_EQ(5, res_obj["hits"][0]["document"].size());
    ASSERT_EQ("product_c", res_obj["hits"][0]["document"]["product_id"]);
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("coll1"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"]["coll1"].size());
    ASSERT_EQ(1, res_obj["hits"][0]["document"]["coll1"].count("coll_id"));
    ASSERT_EQ("b", res_obj["hits"][0]["document"]["coll1"]["coll_id"]);

    schema_json =
            R"({
                "name": "coll2",
                "fields": [
                    {"name": "coll_id", "type": "string"},
                    {"name": "object.reference_array", "type": "string[]", "reference": "Products.product_id", "optional": true},
                    {"name": "object", "type": "object"}
                ],
                "enable_nested_fields": true
            })"_json;
    documents = {
            R"({
                "coll_id": "a",
                "object": {}
            })"_json,
            R"({
                "coll_id": "b",
                "object": {
                    "reference_array": ["product_a", "product_b"]
                }
            })"_json
    };

    collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_TRUE(collection_create_op.ok());
    for (auto const &json: documents) {
        auto add_op = collection_create_op.get()->add(json.dump());
        if (!add_op.ok()) {
            LOG(INFO) << add_op.error();
        }
        ASSERT_TRUE(add_op.ok());
    }

    req_params = {
            {"collection", "coll2"},
            {"q", "*"},
            {"include_fields", "$Products(product_id)"}
    };
    search_op_bool = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op_bool.ok());

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(2, res_obj["found"].get<size_t>());
    ASSERT_EQ(2, res_obj["hits"].size());
    ASSERT_EQ(3, res_obj["hits"][0]["document"].size());
    ASSERT_EQ("b", res_obj["hits"][0]["document"]["coll_id"]);
    ASSERT_EQ(2, res_obj["hits"][0]["document"]["object"].size());
    ASSERT_EQ("product_a", res_obj["hits"][0]["document"]["object"]["reference_array"][0]);
    ASSERT_EQ("product_b", res_obj["hits"][0]["document"]["object"]["reference_array"][1]);
    ASSERT_EQ(1, res_obj["hits"][0]["document"]["object"].count("Products"));
    ASSERT_EQ(2, res_obj["hits"][0]["document"]["object"]["Products"].size());
    ASSERT_EQ(1, res_obj["hits"][0]["document"]["object"]["Products"][0].count("product_id"));
    ASSERT_EQ("product_a", res_obj["hits"][0]["document"]["object"]["Products"][0]["product_id"]);
    ASSERT_EQ(1, res_obj["hits"][0]["document"]["object"]["Products"][1].count("product_id"));
    ASSERT_EQ("product_b", res_obj["hits"][0]["document"]["object"]["Products"][1]["product_id"]);
    ASSERT_EQ(3, res_obj["hits"][1]["document"].size());
    ASSERT_EQ("a", res_obj["hits"][1]["document"]["coll_id"]);
    ASSERT_EQ(0, res_obj["hits"][1]["document"]["object"].size());

    req_params = {
            {"collection", "Products"},
            {"q", "*"},
            {"filter_by", "$coll2(id: *)"},
            {"include_fields", "$coll2(coll_id)"}
    };
    search_op_bool = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op_bool.ok());

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(2, res_obj["found"].get<size_t>());
    ASSERT_EQ(2, res_obj["hits"].size());
    ASSERT_EQ(5, res_obj["hits"][0]["document"].size());
    ASSERT_EQ("product_b", res_obj["hits"][0]["document"]["product_id"]);
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("coll2"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"]["coll2"].size());
    ASSERT_EQ(1, res_obj["hits"][0]["document"]["coll2"][0].count("coll_id"));
    ASSERT_EQ("b", res_obj["hits"][0]["document"]["coll2"][0]["coll_id"]);
    ASSERT_EQ("product_a", res_obj["hits"][1]["document"]["product_id"]);
    ASSERT_EQ(1, res_obj["hits"][1]["document"].count("coll2"));
    ASSERT_EQ(1, res_obj["hits"][1]["document"]["coll2"].size());
    ASSERT_EQ(1, res_obj["hits"][1]["document"]["coll2"][0].count("coll_id"));
    ASSERT_EQ("b", res_obj["hits"][1]["document"]["coll2"][0]["coll_id"]);

    schema_json =
            R"({
                "name": "Portions",
                "fields": [
                    {"name": "portion_id", "type": "string"},
                    {"name": "quantity", "type": "int32"},
                    {"name": "unit", "type": "string"}
                ]
            })"_json;
    documents = {
            R"({
                "portion_id": "portion_a",
                "quantity": 500,
                "unit": "g"
            })"_json,
            R"({
                "portion_id": "portion_b",
                "quantity": 1,
                "unit": "lt"
            })"_json,
            R"({
                "portion_id": "portion_c",
                "quantity": 500,
                "unit": "ml"
            })"_json
    };

    collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_TRUE(collection_create_op.ok());
    for (auto const &json: documents) {
        auto add_op = collection_create_op.get()->add(json.dump());
        if (!add_op.ok()) {
            LOG(INFO) << add_op.error();
        }
        ASSERT_TRUE(add_op.ok());
    }

    schema_json =
            R"({
                "name": "Foods",
                "fields": [
                    {"name": "name", "type": "string"},
                    {"name": "portions", "type": "object[]"},
                    {"name": "portions.portion_id", "type": "string[]", "reference": "Portions.portion_id", "optional": true}
                ],
                "enable_nested_fields": true
            })"_json;
    documents = {
            R"({
                "name": "Bread",
                "portions": [
                    {
                        "portion_id": "portion_a",
                        "count": 10
                    }
                ]
            })"_json,
            R"({
                "name": "Milk",
                "portions": [
                    {
                        "portion_id": "portion_b",
                        "count": 3
                    },
                    {
                        "count": 3
                    },
                    {
                        "portion_id": "portion_c",
                        "count": 1
                    }
                ]
            })"_json
    };

    collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_TRUE(collection_create_op.ok());
    for (auto const &json: documents) {
        auto add_op = collection_create_op.get()->add(json.dump(), CREATE, "", DIRTY_VALUES::REJECT);
        if (!add_op.ok()) {
            LOG(INFO) << add_op.error();
        }
        ASSERT_TRUE(add_op.ok());
    }

    req_params = {
            {"collection", "Foods"},
            {"q", "*"},
            {"include_fields", "$Portions(*:merge)"}
    };
    search_op_bool = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op_bool.ok());

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(2, res_obj["found"].get<size_t>());
    ASSERT_EQ(2, res_obj["hits"].size());
    ASSERT_EQ(3, res_obj["hits"][0]["document"].size());
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("name"));

    ASSERT_EQ("Milk", res_obj["hits"][0]["document"]["name"]);
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("portions"));
    ASSERT_EQ(3, res_obj["hits"][0]["document"]["portions"].size());

    ASSERT_EQ(5, res_obj["hits"][0]["document"]["portions"][0].size());
    ASSERT_EQ("portion_b", res_obj["hits"][0]["document"]["portions"][0].at("portion_id"));
    ASSERT_EQ(1 , res_obj["hits"][0]["document"]["portions"][0].at("quantity"));
    ASSERT_EQ("lt", res_obj["hits"][0]["document"]["portions"][0].at("unit"));
    ASSERT_EQ(3 , res_obj["hits"][0]["document"]["portions"][0].at("count"));

    ASSERT_EQ(1, res_obj["hits"][0]["document"]["portions"][1].size());
    ASSERT_EQ(3 , res_obj["hits"][0]["document"]["portions"][1].at("count"));

    ASSERT_EQ(5, res_obj["hits"][0]["document"]["portions"][2].size());
    ASSERT_EQ("portion_c", res_obj["hits"][0]["document"]["portions"][2].at("portion_id"));
    ASSERT_EQ(500 , res_obj["hits"][0]["document"]["portions"][2].at("quantity"));
    ASSERT_EQ("ml", res_obj["hits"][0]["document"]["portions"][2].at("unit"));
    ASSERT_EQ(1 , res_obj["hits"][0]["document"]["portions"][2].at("count"));


    ASSERT_EQ("Bread", res_obj["hits"][1]["document"]["name"]);
    ASSERT_EQ(1, res_obj["hits"][1]["document"].count("portions"));
    ASSERT_EQ(1, res_obj["hits"][1]["document"]["portions"].size());

    ASSERT_EQ(5, res_obj["hits"][1]["document"]["portions"][0].size());
    ASSERT_EQ("portion_a", res_obj["hits"][1]["document"]["portions"][0].at("portion_id"));
    ASSERT_EQ(500 , res_obj["hits"][1]["document"]["portions"][0].at("quantity"));
    ASSERT_EQ("g", res_obj["hits"][1]["document"]["portions"][0].at("unit"));
    ASSERT_EQ(10 , res_obj["hits"][1]["document"]["portions"][0].at("count"));
}

TEST_F(CollectionJoinTest, CascadeDeletion) {
    auto schema_json =
            R"({
                "name": "Products",
                "fields": [
                    {"name": "product_idx", "type": "string"},
                    {"name": "product_name", "type": "string", "infix": true},
                    {"name": "product_description", "type": "string"}
                ]
            })"_json;
    std::vector<nlohmann::json> documents = {
            R"({
                "product_idx": "product_a",
                "product_name": "shampoo",
                "product_description": "Our new moisturizing shampoo is perfect for those with dry or damaged hair."
            })"_json,
            R"({
                "product_idx": "product_b",
                "product_name": "soap",
                "product_description": "Introducing our all-natural, organic soap bar made with essential oils and botanical ingredients."
            })"_json
    };

    auto collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_TRUE(collection_create_op.ok());
    for (auto const &json: documents) {
        auto add_op = collection_create_op.get()->add(json.dump());
        ASSERT_TRUE(add_op.ok());
    }

    schema_json =
            R"({
                "name": "Users",
                "fields": [
                    {"name": "user_id", "type": "string"},
                    {"name": "user_name", "type": "string"}
                ]
            })"_json;
    documents = {
            R"({
                "user_id": "user_a",
                "user_name": "Joe"
            })"_json,
            R"({
                "user_id": "user_b",
                "user_name": "Dan"
            })"_json,
    };
    collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_TRUE(collection_create_op.ok());
    for (auto const &json: documents) {
        auto add_op = collection_create_op.get()->add(json.dump());
        ASSERT_TRUE(add_op.ok());
    }

    schema_json =
            R"({
                "name": "CustomerProductPrices",
                "fields": [
                    {"name": "product_price", "type": "float"},
                    {"name": "user_id", "type": "string", "reference": "Users.user_id"},
                    {"name": "product_id", "type": "string", "reference": "Products.product_idx"}
                ]
            })"_json;
    documents = {
            R"({
                "user_id": "user_a",
                "product_price": 143,
                "product_id": "product_a"
            })"_json,
            R"({
                "user_id": "user_a",
                "product_price": 73.5,
                "product_id": "product_b"
            })"_json,
            R"({
                "user_id": "user_b",
                "product_price": 75,
                "product_id": "product_a"
            })"_json,
            R"({
                "user_id": "user_b",
                "product_price": 140,
                "product_id": "product_b"
            })"_json
    };
    collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_TRUE(collection_create_op.ok());
    for (auto const &json: documents) {
        auto add_op = collection_create_op.get()->add(json.dump());
        ASSERT_TRUE(add_op.ok());
    }

    std::map<std::string, std::string> req_params = {
            {"collection", "Products"},
            {"q", "*"},
            {"filter_by", "$CustomerProductPrices(user_id:= user_a)"},
            {"include_fields", "$CustomerProductPrices(product_price)"}
    };
    nlohmann::json embedded_params;
    std::string json_res;
    auto now_ts = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

    auto search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    nlohmann::json res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(2, res_obj["found"].get<size_t>());
    ASSERT_EQ("product_b", res_obj["hits"][0]["document"].at("product_idx"));
    ASSERT_EQ("product_a", res_obj["hits"][1]["document"].at("product_idx"));

    req_params = {
            {"collection", "CustomerProductPrices"},
            {"q", "*"},
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(4, res_obj["found"].get<size_t>());
    ASSERT_EQ("product_b", res_obj["hits"][0]["document"].at("product_id"));
    ASSERT_EQ("product_a", res_obj["hits"][1]["document"].at("product_id"));
    ASSERT_EQ("product_b", res_obj["hits"][0]["document"].at("product_id"));
    ASSERT_EQ("product_a", res_obj["hits"][1]["document"].at("product_id"));

    req_params = {
            {"collection", "Products"},
            {"q", "*"},
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(2, res_obj["found"].get<size_t>());
    ASSERT_EQ("product_b", res_obj["hits"][0]["document"].at("product_idx"));
    ASSERT_EQ("product_a", res_obj["hits"][1]["document"].at("product_idx"));

    collectionManager.get_collection_unsafe("Products")->remove("0");

    req_params = {
            {"collection", "Products"},
            {"q", "*"},
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(1, res_obj["found"].get<size_t>());
    ASSERT_EQ("product_b", res_obj["hits"][0]["document"].at("product_idx"));

    req_params = {
            {"collection", "CustomerProductPrices"},
            {"q", "*"},
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(2, res_obj["found"].get<size_t>());
    ASSERT_EQ("product_b", res_obj["hits"][0]["document"].at("product_id"));
    ASSERT_EQ("product_b", res_obj["hits"][1]["document"].at("product_id"));

    collectionManager.get_collection_unsafe("Users")->remove("1");

    req_params = {
            {"collection", "Users"},
            {"q", "*"},
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(1, res_obj["found"].get<size_t>());
    ASSERT_EQ("user_a", res_obj["hits"][0]["document"].at("user_id"));

    req_params = {
            {"collection", "CustomerProductPrices"},
            {"q", "*"},
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(1, res_obj["found"].get<size_t>());
    ASSERT_EQ("product_b", res_obj["hits"][0]["document"].at("product_id"));
    ASSERT_EQ("user_a", res_obj["hits"][0]["document"].at("user_id"));
}

TEST_F(CollectionJoinTest, SortByReference) {
    auto schema_json =
            R"({
                "name": "Products",
                "fields": [
                    {"name": "product_id", "type": "string"},
                    {"name": "product_name", "type": "string", "sort": true, "infix": true},
                    {"name": "product_description", "type": "string"},
                    {"name": "embedding", "type":"float[]", "embed":{"from": ["product_description"], "model_config": {"model_name": "ts/e5-small"}}}
                ]
            })"_json;
    std::vector<nlohmann::json> documents = {
            R"({
                "product_id": "product_a",
                "product_name": "shampoo",
                "product_description": "Our new moisturizing shampoo is perfect for those with dry or damaged hair."
            })"_json,
            R"({
                "product_id": "product_b",
                "product_name": "soap",
                "product_description": "Introducing our all-natural, organic soap bar made with essential oils and botanical ingredients."
            })"_json
    };

    EmbedderManager::set_model_dir("/tmp/typesense_test/models");

    auto collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_TRUE(collection_create_op.ok());
    for (auto const &json: documents) {
        auto add_op = collection_create_op.get()->add(json.dump());
        ASSERT_TRUE(add_op.ok());
    }

    schema_json =
            R"({
                "name": "Customers",
                "fields": [
                    {"name": "customer_id", "type": "string"},
                    {"name": "customer_name", "type": "string", "sort": true},
                    {"name": "product_price", "type": "float"},
                    {"name": "product_available", "type": "bool"},
                    {"name": "product_location", "type": "geopoint"},
                    {"name": "product_id", "type": "string", "reference": "Products.product_id", "sort": true}
                ]
            })"_json;
    documents = {
            R"({
                "customer_id": "customer_a",
                "customer_name": "Joe",
                "product_price": 143,
                "product_available": true,
                "product_location": [48.872576479306765, 2.332291112241466],
                "product_id": "product_a"
            })"_json,
            R"({
                "customer_id": "customer_a",
                "customer_name": "Joe",
                "product_price": 73.5,
                "product_available": false,
                "product_location": [48.888286721920934, 2.342340862419206],
                "product_id": "product_b"
            })"_json,
            R"({
                "customer_id": "customer_b",
                "customer_name": "Dan",
                "product_price": 75,
                "product_available": true,
                "product_location": [48.872576479306765, 2.332291112241466],
                "product_id": "product_a"
            })"_json,
            R"({
                "customer_id": "customer_b",
                "customer_name": "Dan",
                "product_price": 140,
                "product_available": false,
                "product_location": [48.888286721920934, 2.342340862419206],
                "product_id": "product_b"
            })"_json
    };
    collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_TRUE(collection_create_op.ok());
    for (auto const &json: documents) {
        auto add_op = collection_create_op.get()->add(json.dump());
        ASSERT_TRUE(add_op.ok());
    }

    std::map<std::string, std::string> req_params = {
            {"collection", "Products"},
            {"q", "*"},
            {"query_by", "product_name"},
            {"filter_by", "$Customers(customer_id:=customer_a)"},
            {"sort_by", "$foo(product_price:asc"}
    };
    nlohmann::json embedded_params;
    std::string json_res;
    auto now_ts = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

    auto search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_FALSE(search_op.ok());
    ASSERT_EQ("Parameter `sort_by` is malformed.", search_op.error());

    req_params = {
            {"collection", "Products"},
            {"q", "*"},
            {"query_by", "product_name"},
            {"filter_by", "$Customers(customer_id:=customer_a)"},
            {"sort_by", "$Customers(product_price)"}
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_FALSE(search_op.ok());
    ASSERT_EQ("Reference `sort_by` is malformed.", search_op.error());

    req_params = {
            {"collection", "Products"},
            {"q", "*"},
            {"query_by", "product_name"},
            {"filter_by", "$Customers(customer_id:=customer_a)"},
            {"sort_by", "$foo(product_price:asc)"}
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_FALSE(search_op.ok());
    ASSERT_EQ("Referenced collection `foo` in `sort_by` not found.", search_op.error());

    req_params = {
            {"collection", "Products"},
            {"q", "*"},
            {"query_by", "product_name"},
            {"filter_by", "$Customers(customer_id:=customer_a)"},
            {"sort_by", "$Customers(foo:asc)"}
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_FALSE(search_op.ok());
    ASSERT_EQ("Referenced collection `Customers`: Could not find a field named `foo` in the schema for sorting.",
              search_op.error());

    // Sort by reference numeric field
    req_params = {
            {"collection", "Products"},
            {"q", "*"},
            {"query_by", "product_name"},
            {"filter_by", "$Customers(customer_id:=customer_a)"},
            {"sort_by", "$Customers(product_price:asc)"},
            {"include_fields", "product_id, $Customers(product_price:merge)"},
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    auto res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(2, res_obj["found"].get<size_t>());
    ASSERT_EQ(2, res_obj["hits"].size());
    ASSERT_EQ("product_b", res_obj["hits"][0]["document"].at("product_id"));
    ASSERT_EQ(73.5, res_obj["hits"][0]["document"].at("product_price"));
    ASSERT_EQ("product_a", res_obj["hits"][1]["document"].at("product_id"));
    ASSERT_EQ(143, res_obj["hits"][1]["document"].at("product_price"));

    req_params = {
            {"collection", "Products"},
            {"q", "*"},
            {"query_by", "product_name"},
            {"filter_by", "$Customers(customer_id:=customer_a)"},
            {"sort_by", "$Customers(product_price:desc)"},
            {"include_fields", "product_id, $Customers(product_price:merge)"},
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(2, res_obj["found"].get<size_t>());
    ASSERT_EQ(2, res_obj["hits"].size());
    ASSERT_EQ("product_a", res_obj["hits"][0]["document"].at("product_id"));
    ASSERT_EQ(143, res_obj["hits"][0]["document"].at("product_price"));
    ASSERT_EQ("product_b", res_obj["hits"][1]["document"].at("product_id"));
    ASSERT_EQ(73.5, res_obj["hits"][1]["document"].at("product_price"));

    // Sort by reference string field
    req_params = {
            {"collection", "Products"},
            {"q", "*"},
            {"query_by", "product_name"},
            {"filter_by", "$Customers(customer_id:=customer_a)"},
            {"sort_by", "$Customers(product_id:asc)"},
            {"include_fields", "product_id, $Customers(product_price:merge)"},
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(2, res_obj["found"].get<size_t>());
    ASSERT_EQ(2, res_obj["hits"].size());
    ASSERT_EQ("product_a", res_obj["hits"][0]["document"].at("product_id"));
    ASSERT_EQ(143, res_obj["hits"][0]["document"].at("product_price"));
    ASSERT_EQ("product_b", res_obj["hits"][1]["document"].at("product_id"));
    ASSERT_EQ(73.5, res_obj["hits"][1]["document"].at("product_price"));

    // Sort by reference optional filtering.
    req_params = {
            {"collection", "Products"},
            {"q", "*"},
            {"query_by", "product_name"},
            {"filter_by", "$Customers(customer_id:=customer_a)"},
            {"sort_by", "$Customers(_eval(product_available:true):asc)"},
            {"include_fields", "product_id, $Customers(product_price:merge)"},
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(2, res_obj["found"].get<size_t>());
    ASSERT_EQ(2, res_obj["hits"].size());
    ASSERT_EQ("product_b", res_obj["hits"][0]["document"].at("product_id"));
    ASSERT_EQ(73.5, res_obj["hits"][0]["document"].at("product_price"));
    ASSERT_EQ("product_a", res_obj["hits"][1]["document"].at("product_id"));
    ASSERT_EQ(143, res_obj["hits"][1]["document"].at("product_price"));

    req_params = {
            {"collection", "Products"},
            {"q", "*"},
            {"query_by", "product_name"},
            {"filter_by", "$Customers(customer_id:=customer_a)"},
            {"sort_by", "$Customers(_eval(product_available:true):desc)"},
            {"include_fields", "product_id, $Customers(product_price:merge)"},
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(2, res_obj["found"].get<size_t>());
    ASSERT_EQ(2, res_obj["hits"].size());
    ASSERT_EQ("product_a", res_obj["hits"][0]["document"].at("product_id"));
    ASSERT_EQ(143, res_obj["hits"][0]["document"].at("product_price"));
    ASSERT_EQ("product_b", res_obj["hits"][1]["document"].at("product_id"));
    ASSERT_EQ(73.5, res_obj["hits"][1]["document"].at("product_price"));

    // Text search
    req_params = {
            {"collection", "Products"},
            {"q", "s"},
            {"query_by", "product_name"},
            {"filter_by", "$Customers(customer_id:=customer_a)"},
            {"sort_by", "$Customers(product_price:desc)"},
            {"include_fields", "product_id, $Customers(product_price:merge)"},
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(2, res_obj["found"].get<size_t>());
    ASSERT_EQ(2, res_obj["hits"].size());
    ASSERT_EQ("product_a", res_obj["hits"][0]["document"].at("product_id"));
    ASSERT_EQ(143, res_obj["hits"][0]["document"].at("product_price"));
    ASSERT_EQ("product_b", res_obj["hits"][1]["document"].at("product_id"));
    ASSERT_EQ(73.5, res_obj["hits"][1]["document"].at("product_price"));

    // Phrase search
    req_params = {
            {"collection", "Products"},
            {"q", R"("our")"},
            {"query_by", "product_description"},
            {"filter_by", "$Customers(customer_id:=customer_a)"},
            {"include_fields", "product_id, $Customers(product_price:merge)"},
            {"sort_by", "$Customers(product_price:desc)"},
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(2, res_obj["found"].get<size_t>());
    ASSERT_EQ(2, res_obj["hits"].size());
    ASSERT_EQ("product_a", res_obj["hits"][0]["document"].at("product_id"));
    ASSERT_EQ(143, res_obj["hits"][0]["document"].at("product_price"));
    ASSERT_EQ("product_b", res_obj["hits"][1]["document"].at("product_id"));
    ASSERT_EQ(73.5, res_obj["hits"][1]["document"].at("product_price"));

    // Vector search
    req_params = {
            {"collection", "Products"},
            {"q", "natural products"},
            {"query_by", "embedding"},
            {"filter_by", "$Customers(customer_id:=customer_a)"},
            {"include_fields", "product_id, $Customers(product_price:merge)"},
            {"sort_by", "$Customers(product_price:desc)"},
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(2, res_obj["found"].get<size_t>());
    ASSERT_EQ(2, res_obj["hits"].size());
    ASSERT_EQ("product_a", res_obj["hits"][0]["document"].at("product_id"));
    ASSERT_EQ(143, res_obj["hits"][0]["document"].at("product_price"));
    ASSERT_EQ("product_b", res_obj["hits"][1]["document"].at("product_id"));
    ASSERT_EQ(73.5, res_obj["hits"][1]["document"].at("product_price"));
    auto product_a_score = res_obj["hits"][0].at("vector_distance");
    auto product_b_score = res_obj["hits"][1].at("vector_distance");
    // product_b is a better match for the vector query but sort_by overrides the order.
    ASSERT_TRUE(product_b_score < product_a_score);

    nlohmann::json model_config = R"({
        "model_name": "ts/e5-small"
    })"_json;
    auto query_embedding = EmbedderManager::get_instance().get_text_embedder(model_config).get()->Embed("natural products");
    std::string vec_string = "[";
    for (auto const& i : query_embedding.embedding) {
        vec_string += std::to_string(i);
        vec_string += ",";
    }
    vec_string[vec_string.size() - 1] = ']';

    req_params = {
            {"collection", "Products"},
            {"q", "*"},
            {"vector_query", "embedding:(" + vec_string + ", flat_search_cutoff: 0)"},
            {"filter_by", "$Customers(customer_id:=customer_a)"},
            {"include_fields", "product_id, $Customers(product_price:merge)"},
            {"sort_by", "$Customers(product_price:desc)"},
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(2, res_obj["found"].get<size_t>());
    ASSERT_EQ(2, res_obj["hits"].size());
    ASSERT_EQ("product_a", res_obj["hits"][0]["document"].at("product_id"));
    ASSERT_EQ(143, res_obj["hits"][0]["document"].at("product_price"));
    ASSERT_EQ("product_b", res_obj["hits"][1]["document"].at("product_id"));
    ASSERT_EQ(73.5, res_obj["hits"][1]["document"].at("product_price"));
    product_a_score = res_obj["hits"][0].at("vector_distance");
    product_b_score = res_obj["hits"][1].at("vector_distance");
    // product_b is a better match for the vector query but sort_by overrides the order.
    ASSERT_TRUE(product_b_score < product_a_score);

    // Hybrid search - Both text match and vector match
    req_params = {
            {"collection", "Products"},
            {"q", "soap"},
            {"query_by", "product_name, embedding"},
            {"filter_by", "$Customers(customer_id:=customer_a)"},
            {"include_fields", "product_id, $Customers(product_price:merge)"},
            {"sort_by", "$Customers(product_price:desc)"},
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(2, res_obj["found"].get<size_t>());
    ASSERT_EQ(2, res_obj["hits"].size());
    ASSERT_EQ("product_a", res_obj["hits"][0]["document"].at("product_id"));
    ASSERT_EQ(143, res_obj["hits"][0]["document"].at("product_price"));
    ASSERT_EQ("product_b", res_obj["hits"][1]["document"].at("product_id"));
    ASSERT_EQ(73.5, res_obj["hits"][1]["document"].at("product_price"));
    product_a_score = res_obj["hits"][0].at("text_match");
    product_b_score = res_obj["hits"][1].at("text_match");
    ASSERT_TRUE(product_b_score > product_a_score);
    product_a_score = res_obj["hits"][0].at("vector_distance");
    product_b_score = res_obj["hits"][1].at("vector_distance");
    ASSERT_TRUE(product_b_score < product_a_score);

    // Hybrid search - Only vector match
    req_params = {
            {"collection", "Products"},
            {"q", "natural products"},
            {"query_by", "product_name, embedding"},
            {"filter_by", "$Customers(customer_id:=customer_a)"},
            {"include_fields", "product_id, $Customers(product_price:merge)"},
            {"sort_by", "$Customers(product_price:desc)"},
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(2, res_obj["found"].get<size_t>());
    ASSERT_EQ(2, res_obj["hits"].size());
    ASSERT_EQ("product_a", res_obj["hits"][0]["document"].at("product_id"));
    ASSERT_EQ(143, res_obj["hits"][0]["document"].at("product_price"));
    ASSERT_EQ("product_b", res_obj["hits"][1]["document"].at("product_id"));
    ASSERT_EQ(73.5, res_obj["hits"][1]["document"].at("product_price"));
    product_a_score = res_obj["hits"][0].at("vector_distance");
    product_b_score = res_obj["hits"][1].at("vector_distance");
    // product_b is a better match for the vector query but sort_by overrides the order.
    ASSERT_TRUE(product_b_score < product_a_score);

    // Infix search
    req_params = {
            {"collection", "Products"},
            {"q", "p"},
            {"query_by", "product_name"},
            {"infix", "always"},
            {"filter_by", "$Customers(customer_id:=customer_a)"},
            {"include_fields", "product_id, $Customers(product_price:merge)"},
            {"sort_by", "$Customers(product_price:desc)"},
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(2, res_obj["found"].get<size_t>());
    ASSERT_EQ(2, res_obj["hits"].size());
    ASSERT_EQ("product_a", res_obj["hits"][0]["document"].at("product_id"));
    ASSERT_EQ(143, res_obj["hits"][0]["document"].at("product_price"));
    ASSERT_EQ("product_b", res_obj["hits"][1]["document"].at("product_id"));
    ASSERT_EQ(73.5, res_obj["hits"][1]["document"].at("product_price"));

    // Reference sort_by without join
    req_params = {
            {"collection", "Customers"},
            {"q", "*"},
            {"filter_by", "customer_name:= [Joe, Dan] && product_price:<100"},
            {"include_fields", "$Products(product_name:merge), product_price"},
            {"sort_by", "$Products(product_name:desc)"},
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(2, res_obj["found"].get<size_t>());
    ASSERT_EQ(2, res_obj["hits"].size());
    ASSERT_EQ(2, res_obj["hits"][0]["document"].size());
    ASSERT_EQ("soap", res_obj["hits"][0]["document"].at("product_name"));
    ASSERT_EQ(73.5, res_obj["hits"][0]["document"].at("product_price"));
    ASSERT_EQ("shampoo", res_obj["hits"][1]["document"].at("product_name"));
    ASSERT_EQ(75, res_obj["hits"][1]["document"].at("product_price"));

    req_params = {
            {"collection", "Customers"},
            {"q", "*"},
            {"filter_by", "customer_name:= [Joe, Dan] && product_price:<100"},
            {"include_fields", "$Products(product_name:merge), product_price"},
            {"sort_by", "$Products(product_name:asc)"},
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(2, res_obj["found"].get<size_t>());
    ASSERT_EQ(2, res_obj["hits"].size());
    ASSERT_EQ(2, res_obj["hits"][0]["document"].size());
    ASSERT_EQ("shampoo", res_obj["hits"][0]["document"].at("product_name"));
    ASSERT_EQ(75, res_obj["hits"][0]["document"].at("product_price"));
    ASSERT_EQ("soap", res_obj["hits"][1]["document"].at("product_name"));
    ASSERT_EQ(73.5, res_obj["hits"][1]["document"].at("product_price"));

    req_params = {
            {"collection", "Customers"},
            {"q", "*"},
            {"include_fields", "$Products(product_name:merge), customer_name, id"},
            {"sort_by", "$Products(product_name:asc), customer_name:desc"},
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(4, res_obj["found"].get<size_t>());
    ASSERT_EQ(4, res_obj["hits"].size());
    ASSERT_EQ(3, res_obj["hits"][0]["document"].size());
    ASSERT_EQ("0", res_obj["hits"][0]["document"].at("id"));
    ASSERT_EQ("Joe", res_obj["hits"][0]["document"].at("customer_name"));
    ASSERT_EQ("shampoo", res_obj["hits"][0]["document"].at("product_name"));
    ASSERT_EQ(3, res_obj["hits"][1]["document"].size());
    ASSERT_EQ("2", res_obj["hits"][1]["document"].at("id"));
    ASSERT_EQ("Dan", res_obj["hits"][1]["document"].at("customer_name"));
    ASSERT_EQ("shampoo", res_obj["hits"][1]["document"].at("product_name"));
    ASSERT_EQ(3, res_obj["hits"][2]["document"].size());
    ASSERT_EQ("1", res_obj["hits"][2]["document"].at("id"));
    ASSERT_EQ("Joe", res_obj["hits"][2]["document"].at("customer_name"));
    ASSERT_EQ("soap", res_obj["hits"][2]["document"].at("product_name"));
    ASSERT_EQ(3, res_obj["hits"][3]["document"].size());
    ASSERT_EQ("3", res_obj["hits"][3]["document"].at("id"));
    ASSERT_EQ("Dan", res_obj["hits"][3]["document"].at("customer_name"));
    ASSERT_EQ("soap", res_obj["hits"][3]["document"].at("product_name"));

    schema_json =
            R"({
                "name": "Users",
                "fields": [
                    {"name": "user_id", "type": "string"},
                    {"name": "user_name", "type": "string", "sort": true}
                ]
            })"_json;
    documents = {
            R"({
                "user_id": "user_a",
                "user_name": "Roshan"
            })"_json,
            R"({
                "user_id": "user_b",
                "user_name": "Ruby"
            })"_json,
            R"({
                "user_id": "user_c",
                "user_name": "Joe"
            })"_json,
            R"({
                "user_id": "user_d",
                "user_name": "Aby"
            })"_json
    };
    collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_TRUE(collection_create_op.ok());
    for (auto const &json: documents) {
        auto add_op = collection_create_op.get()->add(json.dump());
        if (!add_op.ok()) {
            LOG(INFO) << add_op.error();
        }
        ASSERT_TRUE(add_op.ok());
    }

    schema_json =
            R"({
                "name": "Repos",
                "fields": [
                    {"name": "repo_id", "type": "string"},
                    {"name": "repo_content", "type": "string"},
                    {"name": "repo_stars", "type": "int32"},
                    {"name": "repo_is_private", "type": "bool"}
                ]
            })"_json;
    documents = {
            R"({
                "repo_id": "repo_a",
                "repo_content": "body1",
                "repo_stars": 431,
                "repo_is_private": true
            })"_json,
            R"({
                "repo_id": "repo_b",
                "repo_content": "body2",
                "repo_stars": 4562,
                "repo_is_private": false
            })"_json,
            R"({
                "repo_id": "repo_c",
                "repo_content": "body3",
                "repo_stars": 945,
                "repo_is_private": false
            })"_json,
            R"({
                "repo_id": "repo_d",
                "repo_content": "body4",
                "repo_stars": 95,
                "repo_is_private": true
            })"_json
    };
    collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_TRUE(collection_create_op.ok());
    for (auto const &json: documents) {
        auto add_op = collection_create_op.get()->add(json.dump());
        if (!add_op.ok()) {
            LOG(INFO) << add_op.error();
        }
        ASSERT_TRUE(add_op.ok());
    }

    schema_json =
            R"({
                "name": "Links",
                "fields": [
                    {"name": "repo_id", "type": "string", "reference": "Repos.repo_id"},
                    {"name": "user_id", "type": "string", "reference": "Users.user_id"}
                ]
            })"_json;
    documents = {
            R"({
                "repo_id": "repo_a",
                "user_id": "user_b"
            })"_json,
            R"({
                "repo_id": "repo_a",
                "user_id": "user_c"
            })"_json,
            R"({
                "repo_id": "repo_b",
                "user_id": "user_a"
            })"_json,
            R"({
                "repo_id": "repo_b",
                "user_id": "user_b"
            })"_json,
            R"({
                "repo_id": "repo_b",
                "user_id": "user_d"
            })"_json,
            R"({
                "repo_id": "repo_c",
                "user_id": "user_a"
            })"_json,
            R"({
                "repo_id": "repo_c",
                "user_id": "user_b"
            })"_json,
            R"({
                "repo_id": "repo_c",
                "user_id": "user_c"
            })"_json,
            R"({
                "repo_id": "repo_c",
                "user_id": "user_d"
            })"_json,
            R"({
                "repo_id": "repo_d",
                "user_id": "user_d"
            })"_json
    };
    collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_TRUE(collection_create_op.ok());

    for (auto const &json: documents) {
        auto add_op = collection_create_op.get()->add(json.dump());
        if (!add_op.ok()) {
            LOG(INFO) << add_op.error();
        }
        ASSERT_TRUE(add_op.ok());
    }

    req_params = {
            {"collection", "Users"},
            {"q", "*"},
            {"filter_by", "$Links(repo_id:=[repo_a, repo_d])"},
            {"include_fields", "user_id, user_name, $Repos(repo_content, repo_stars:merge), "},
            {"exclude_fields", "$Links(*), "},
            {"sort_by", "$Repos(repo_stars: asc)"}
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(3, res_obj["found"].get<size_t>());
    ASSERT_EQ(3, res_obj["hits"].size());
    ASSERT_EQ(4, res_obj["hits"][0]["document"].size());
    ASSERT_EQ("user_d", res_obj["hits"][0]["document"].at("user_id"));
    ASSERT_EQ("Aby", res_obj["hits"][0]["document"].at("user_name"));
    ASSERT_EQ("body4", res_obj["hits"][0]["document"].at("repo_content"));
    ASSERT_EQ(95, res_obj["hits"][0]["document"].at("repo_stars"));

    ASSERT_EQ("user_c", res_obj["hits"][1]["document"].at("user_id"));
    ASSERT_EQ("Joe", res_obj["hits"][1]["document"].at("user_name"));
    ASSERT_EQ("body1", res_obj["hits"][1]["document"].at("repo_content"));
    ASSERT_EQ(431, res_obj["hits"][1]["document"].at("repo_stars"));

    ASSERT_EQ("user_b", res_obj["hits"][2]["document"].at("user_id"));
    ASSERT_EQ("Ruby", res_obj["hits"][2]["document"].at("user_name"));
    ASSERT_EQ("body1", res_obj["hits"][2]["document"].at("repo_content"));
    ASSERT_EQ(431, res_obj["hits"][2]["document"].at("repo_stars"));

    req_params = {
            {"collection", "Users"},
            {"q", "*"},
            {"filter_by", "$Links(repo_id:=[repo_a, repo_d])"},
            {"include_fields", "user_id, user_name, $Repos(repo_content, repo_stars:merge), "},
            {"exclude_fields", "$Links(*), "},
            {"sort_by", "$Repos(repo_stars: desc), user_name:desc"}
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_TRUE(search_op.ok());

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(3, res_obj["found"].get<size_t>());
    ASSERT_EQ(3, res_obj["hits"].size());
    ASSERT_EQ(4, res_obj["hits"][0]["document"].size());
    ASSERT_EQ("user_b", res_obj["hits"][0]["document"].at("user_id"));
    ASSERT_EQ("Ruby", res_obj["hits"][0]["document"].at("user_name"));
    ASSERT_EQ("body1", res_obj["hits"][0]["document"].at("repo_content"));
    ASSERT_EQ(431, res_obj["hits"][0]["document"].at("repo_stars"));

    ASSERT_EQ("user_c", res_obj["hits"][1]["document"].at("user_id"));
    ASSERT_EQ("Joe", res_obj["hits"][1]["document"].at("user_name"));
    ASSERT_EQ("body1", res_obj["hits"][1]["document"].at("repo_content"));
    ASSERT_EQ(431, res_obj["hits"][1]["document"].at("repo_stars"));

    ASSERT_EQ("user_d", res_obj["hits"][2]["document"].at("user_id"));
    ASSERT_EQ("Aby", res_obj["hits"][2]["document"].at("user_name"));
    ASSERT_EQ("body4", res_obj["hits"][2]["document"].at("repo_content"));
    ASSERT_EQ(95, res_obj["hits"][2]["document"].at("repo_stars"));

    // Multiple references - Wildcard search
    req_params = {
            {"collection", "Products"},
            {"q", "*"},
            {"query_by", "product_name"},
            {"filter_by", "$Customers(product_price: >0)"},
            {"sort_by", "$Customers(product_price:desc)"},
            {"include_fields", "product_id, $Customers(product_price)"},
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_FALSE(search_op.ok());
    ASSERT_EQ("Multiple references found to sort by on `Customers.product_price`.", search_op.error());

    // Multiple references - Text search
    req_params = {
            {"collection", "Products"},
            {"q", "s"},
            {"query_by", "product_name"},
            {"filter_by", "$Customers(product_price: >0)"},
            {"sort_by", "$Customers(product_price:desc)"},
            {"include_fields", "product_id, $Customers(product_price)"}
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_FALSE(search_op.ok());
    ASSERT_EQ("Multiple references found to sort by on `Customers.product_price`.", search_op.error());

    // Multiple references - Phrase search
    req_params = {
            {"collection", "Products"},
            {"q", R"("our")"},
            {"query_by", "product_description"},
            {"filter_by", "$Customers(product_price: >0)"},
            {"include_fields", "product_id, $Customers(product_price)"},
            {"sort_by", "$Customers(product_price:desc)"},
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_FALSE(search_op.ok());
    ASSERT_EQ("Multiple references found to sort by on `Customers.product_price`.", search_op.error());

    // Multiple references - Vector search
    req_params = {
            {"collection", "Products"},
            {"q", "natural products"},
            {"query_by", "embedding"},
            {"filter_by", "$Customers(product_price:>0)"},
            {"include_fields", "product_name, $Customers(product_price)"},
            {"sort_by", "$Customers(product_price:desc)"},
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_FALSE(search_op.ok());
    ASSERT_EQ("Multiple references found to sort by on `Customers.product_price`.", search_op.error());

    req_params = {
            {"collection", "Products"},
            {"q", "*"},
            {"vector_query", "embedding:(" + vec_string + ", flat_search_cutoff: 0)"},
            {"filter_by", "$Customers(product_price: >0)"},
            {"include_fields", "product_id, $Customers(product_price)"},
            {"sort_by", "$Customers(product_price:desc)"},
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_FALSE(search_op.ok());
    ASSERT_EQ("Multiple references found to sort by on `Customers.product_price`.", search_op.error());

    // Multiple references - Hybrid search
    req_params = {
            {"collection", "Products"},
            {"q", "soap"},
            {"query_by", "product_name, embedding"},
            {"filter_by", "$Customers(product_price: >0)"},
            {"include_fields", "product_id, $Customers(product_price)"},
            {"sort_by", "$Customers(product_price:desc)"},
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_FALSE(search_op.ok());
    ASSERT_EQ("Multiple references found to sort by on `Customers.product_price`.", search_op.error());

    req_params = {
            {"collection", "Products"},
            {"q", "natural products"},
            {"query_by", "product_name, embedding"},
            {"filter_by", "$Customers(product_price: >0)"},
            {"include_fields", "product_id, $Customers(product_price)"},
            {"sort_by", "$Customers(product_price:desc)"},
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_FALSE(search_op.ok());
    ASSERT_EQ("Multiple references found to sort by on `Customers.product_price`.", search_op.error());

    // Multiple references - Infix search
    req_params = {
            {"collection", "Products"},
            {"q", "p"},
            {"query_by", "product_name"},
            {"infix", "always"},
            {"filter_by", "$Customers(product_price: >0)"},
            {"include_fields", "product_id, $Customers(product_price)"},
            {"sort_by", "$Customers(product_price:desc)"},
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_FALSE(search_op.ok());
    ASSERT_EQ("Multiple references found to sort by on `Customers.product_price`.", search_op.error());
}

TEST_F(CollectionJoinTest, FilterByReferenceAlias) {
    auto schema_json =
            R"({
                "name": "Products",
                "fields": [
                    {"name": "product_id", "type": "string"},
                    {"name": "product_name", "type": "string", "infix": true},
                    {"name": "product_description", "type": "string"},
                    {"name": "embedding", "type":"float[]", "embed":{"from": ["product_description"], "model_config": {"model_name": "ts/e5-small"}}},
                    {"name": "rating", "type": "int32"}
                ]
            })"_json;
    std::vector<nlohmann::json> documents = {
            R"({
                "product_id": "product_a",
                "product_name": "shampoo",
                "product_description": "Our new moisturizing shampoo is perfect for those with dry or damaged hair.",
                "rating": "2"
            })"_json,
            R"({
                "product_id": "product_b",
                "product_name": "soap",
                "product_description": "Introducing our all-natural, organic soap bar made with essential oils and botanical ingredients.",
                "rating": "4"
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

    schema_json =
            R"({
                "name": "Customers",
                "fields": [
                    {"name": "customer_id", "type": "string"},
                    {"name": "customer_name", "type": "string"},
                    {"name": "product_price", "type": "float"},
                    {"name": "product_id", "type": "string", "reference": "Products.product_id"}
                ]
            })"_json;
    documents = {
            R"({
                "customer_id": "customer_a",
                "customer_name": "Joe",
                "product_price": 143,
                "product_id": "product_a"
            })"_json,
            R"({
                "customer_id": "customer_a",
                "customer_name": "Joe",
                "product_price": 73.5,
                "product_id": "product_b"
            })"_json,
            R"({
                "customer_id": "customer_b",
                "customer_name": "Dan",
                "product_price": 75,
                "product_id": "product_a"
            })"_json,
            R"({
                "customer_id": "customer_b",
                "customer_name": "Dan",
                "product_price": 140,
                "product_id": "product_b"
            })"_json
    };
    collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_TRUE(collection_create_op.ok());
    for (auto const &json: documents) {
        auto add_op = collection_create_op.get()->add(json.dump());
        if (!add_op.ok()) {
            LOG(INFO) << add_op.error();
        }
        ASSERT_TRUE(add_op.ok());
    }

    auto symlink_op = collectionManager.upsert_symlink("Products_alias", "Products");
    ASSERT_TRUE(symlink_op.ok());

    symlink_op = collectionManager.upsert_symlink("$Customers_alias", "$Customers");
    ASSERT_TRUE(symlink_op.ok());

    std::map<std::string, std::string> req_params = {
            {"collection", "Products_alias"},
            {"q", "*"},
            {"query_by", "product_name"},
            {"filter_by", "$Customers(customer_id:=customer_a && product_price:<100)"},
    };
    nlohmann::json embedded_params;
    std::string json_res;
    auto now_ts = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

    auto search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);

    nlohmann::json res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(1, res_obj["found"].get<size_t>());
    ASSERT_EQ(1, res_obj["hits"].size());
    // No fields are mentioned in `include_fields`, should include all fields of Products and Customers by default.
    ASSERT_EQ(7, res_obj["hits"][0]["document"].size());
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("id"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("product_id"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("product_name"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("product_description"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("embedding"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("rating"));
    // Default strategy of reference includes is nest. No alias was provided, collection name becomes the field name.
    ASSERT_EQ(5, res_obj["hits"][0]["document"]["Customers"].size());
    ASSERT_EQ(1, res_obj["hits"][0]["document"]["Customers"].count("customer_id"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"]["Customers"].count("customer_name"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"]["Customers"].count("id"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"]["Customers"].count("product_id"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"]["Customers"].count("product_price"));

    req_params = {
            {"collection", "Products"},
            {"q", "*"},
            {"query_by", "product_name"},
            {"filter_by", "$Customers_alias(customer_id:=customer_a && product_price:<100)"},
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(1, res_obj["found"].get<size_t>());
    ASSERT_EQ(1, res_obj["hits"].size());
    // No fields are mentioned in `include_fields`, should include all fields of Products and Customers by default.
    ASSERT_EQ(7, res_obj["hits"][0]["document"].size());
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("id"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("product_id"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("product_name"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("product_description"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("embedding"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("rating"));
    // Default strategy of reference includes is nest. No alias was provided, collection name becomes the field name.
    ASSERT_EQ(5, res_obj["hits"][0]["document"]["Customers"].size());
    ASSERT_EQ(1, res_obj["hits"][0]["document"]["Customers"].count("customer_id"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"]["Customers"].count("customer_name"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"]["Customers"].count("id"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"]["Customers"].count("product_id"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"]["Customers"].count("product_price"));

    req_params = {
            {"collection", "Products_alias"},
            {"q", "*"},
            {"query_by", "product_name"},
            {"filter_by", "$Customers_alias(customer_id:=customer_a && product_price:<100)"},
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(1, res_obj["found"].get<size_t>());
    ASSERT_EQ(1, res_obj["hits"].size());
    // No fields are mentioned in `include_fields`, should include all fields of Products and Customers by default.
    ASSERT_EQ(7, res_obj["hits"][0]["document"].size());
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("id"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("product_id"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("product_name"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("product_description"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("embedding"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("rating"));
    // Default strategy of reference includes is nest. No alias was provided, collection name becomes the field name.
    ASSERT_EQ(5, res_obj["hits"][0]["document"]["Customers"].size());
    ASSERT_EQ(1, res_obj["hits"][0]["document"]["Customers"].count("customer_id"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"]["Customers"].count("customer_name"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"]["Customers"].count("id"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"]["Customers"].count("product_id"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"]["Customers"].count("product_price"));

    collectionManager.drop_collection("Customers");

    // Alias in reference.
    schema_json =
            R"({
                "name": "Customers",
                "fields": [
                    {"name": "customer_id", "type": "string"},
                    {"name": "customer_name", "type": "string"},
                    {"name": "product_price", "type": "float"},
                    {"name": "product_id", "type": "string", "reference": "Products_alias.product_id"}
                ]
            })"_json;
    documents = {
            R"({
                "customer_id": "customer_a",
                "customer_name": "Joe",
                "product_price": 143,
                "product_id": "product_a"
            })"_json,
            R"({
                "customer_id": "customer_a",
                "customer_name": "Joe",
                "product_price": 73.5,
                "product_id": "product_b"
            })"_json,
            R"({
                "customer_id": "customer_b",
                "customer_name": "Dan",
                "product_price": 75,
                "product_id": "product_a"
            })"_json,
            R"({
                "customer_id": "customer_b",
                "customer_name": "Dan",
                "product_price": 140,
                "product_id": "product_b"
            })"_json
    };
    collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_TRUE(collection_create_op.ok());
    for (auto const &json: documents) {
        auto add_op = collection_create_op.get()->add(json.dump());
        if (!add_op.ok()) {
            LOG(INFO) << add_op.error();
        }
        ASSERT_TRUE(add_op.ok());
    }

    req_params = {
            {"collection", "Products"},
            {"q", "*"},
            {"query_by", "product_name"},
            {"filter_by", "$Customers(customer_id:=customer_a && product_price:<100)"},
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);

    res_obj = nlohmann::json::parse(json_res);
    ASSERT_EQ(1, res_obj["found"].get<size_t>());
    ASSERT_EQ(1, res_obj["hits"].size());
    // No fields are mentioned in `include_fields`, should include all fields of Products and Customers by default.
    ASSERT_EQ(7, res_obj["hits"][0]["document"].size());
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("id"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("product_id"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("product_name"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("product_description"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("embedding"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("rating"));
    // Default strategy of reference includes is nest. No alias was provided, collection name becomes the field name.
    ASSERT_EQ(5, res_obj["hits"][0]["document"]["Customers"].size());
    ASSERT_EQ(1, res_obj["hits"][0]["document"]["Customers"].count("customer_id"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"]["Customers"].count("customer_name"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"]["Customers"].count("id"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"]["Customers"].count("product_id"));
    ASSERT_EQ(1, res_obj["hits"][0]["document"]["Customers"].count("product_price"));
}

TEST_F(CollectionJoinTest, QueryByReference) {
    auto schema_json =
            R"({
                "name": "Products",
                "fields": [
                    {"name": "product_id", "type": "string"},
                    {"name": "product_name", "type": "string"},
                    {"name": "product_description", "type": "string"}
                ]
            })"_json;
    std::vector<nlohmann::json> documents = {
            R"({
                "product_id": "product_a",
                "product_name": "shampoo",
                "product_description": "Our new moisturizing shampoo is perfect for those with dry or damaged hair."
            })"_json,
            R"({
                "product_id": "product_b",
                "product_name": "soap",
                "product_description": "Introducing our all-natural, organic soap bar made with essential oils and botanical ingredients."
            })"_json
    };

    auto collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_TRUE(collection_create_op.ok());
    for (auto const &json: documents) {
        auto add_op = collection_create_op.get()->add(json.dump());
        ASSERT_TRUE(add_op.ok());
    }

    schema_json =
            R"({
                "name": "Customers",
                "fields": [
                    {"name": "customer_id", "type": "string"},
                    {"name": "customer_name", "type": "string", "sort": true},
                    {"name": "product_price", "type": "float"},
                    {"name": "product_id", "type": "string", "reference": "Products.product_id", "sort": true}
                ]
            })"_json;
    documents = {
            R"({
                "customer_id": "customer_a",
                "customer_name": "Joe",
                "product_price": 143,
                "product_id": "product_a"
            })"_json,
            R"({
                "customer_id": "customer_a",
                "customer_name": "Joe",
                "product_price": 73.5,
                "product_id": "product_b"
            })"_json,
            R"({
                "customer_id": "customer_b",
                "customer_name": "Dan",
                "product_price": 75,
                "product_id": "product_a"
            })"_json,
            R"({
                "customer_id": "customer_b",
                "customer_name": "Dan",
                "product_price": 140,
                "product_id": "product_b"
            })"_json
    };
    collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_TRUE(collection_create_op.ok());
    for (auto const &json: documents) {
        auto add_op = collection_create_op.get()->add(json.dump());
        ASSERT_TRUE(add_op.ok());
    }

    std::map<std::string, std::string> req_params = {
            {"collection", "Products"},
            {"q", "*"},
            {"query_by", "$Customers(customer_name)"}
    };
    nlohmann::json embedded_params;
    std::string json_res;
    auto now_ts = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

    auto search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_FALSE(search_op.ok());
    ASSERT_EQ("Query by reference is not yet supported.", search_op.error());

    req_params = {
            {"collection", "Products"},
            {"q", "*"},
            {"query_by", "$Customers(customer_name"}
    };
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_FALSE(search_op.ok());
    ASSERT_EQ("Could not find `$Customers(customer_name` field in the schema.", search_op.error());
}

//TEST_F(CollectionJoinTest, FacetByReference) {
//    auto schema_json =
//        R"({
//            "name": "Products",
//            "fields": [
//                {"name": "product_id", "type": "string"},
//                {"name": "product_name", "type": "string"},
//                {"name": "product_description", "type": "string"},
//                {"name": "rating", "type": "int32", "facet": true}
//            ]
//        })"_json;
//    std::vector<nlohmann::json> documents = {
//            R"({
//            "product_id":  "product_a",
//            "product_name": "shampoo",
//            "product_description": "Our new moisturizing shampoo is perfect for those with dry or damaged hair.",
//            "rating": "2"
//        })"_json,
//            R"({
//            "product_id": "product_b",
//            "product_name": "soap",
//            "product_description": "Introducing our all-natural, organic soap bar made with essential oils and botanical ingredients.",
//            "rating": "4"
//        })"_json
//    };
//    auto collection_create_op = collectionManager.create_collection(schema_json);
//    ASSERT_TRUE(collection_create_op.ok());
//    for (auto const &json: documents) {
//        auto add_op = collection_create_op.get()->add(json.dump());
//        if (!add_op.ok()) {
//            LOG(INFO) << add_op.error();
//        }
//        ASSERT_TRUE(add_op.ok());
//    }
//
//    schema_json =
//            R"({
//            "name": "Customers",
//            "fields": [
//                {"name": "customer_id", "type": "string"},
//                {"name": "customer_name", "type": "string"},
//                {"name": "product_price", "type": "float", "facet": true},
//                {"name": "product_id", "type": "string", "reference": "Products.product_id"}
//            ]
//        })"_json;
//    documents = {
//            R"({
//            "customer_id": "customer_a",
//            "customer_name": "Joe",
//            "product_price": 143,
//            "product_id": "product_a"
//        })"_json,
//            R"({
//            "customer_id": "customer_a",
//            "customer_name": "Joe",
//            "product_price": 73.5,
//            "product_id": "product_b"
//        })"_json,
//            R"({
//            "customer_id": "customer_b",
//            "customer_name": "Dan",
//            "product_price": 75,
//            "product_id": "product_a"
//        })"_json,
//            R"({
//            "customer_id": "customer_b",
//            "customer_name": "Dan",
//            "product_price": 140,
//            "product_id": "product_b"
//        })"_json
//    };
//    collection_create_op = collectionManager.create_collection(schema_json);
//    ASSERT_TRUE(collection_create_op.ok());
//    for (auto const &json: documents) {
//        auto add_op = collection_create_op.get()->add(json.dump());
//        if (!add_op.ok()) {
//            LOG(INFO) << add_op.error();
//        }
//        ASSERT_TRUE(add_op.ok());
//    }
//
//    std::map<std::string, std::string> req_params = {
//            {"collection", "Products"},
//            {"q", "*"},
//            {"filter_by", "$Customers(customer_id: customer_a)"},
//            {"facet_by", "$Customers(product_price)"}
//    };
//    nlohmann::json embedded_params;
//    std::string json_res;
//    auto now_ts = std::chrono::duration_cast<std::chrono::microseconds>(
//            std::chrono::system_clock::now().time_since_epoch()).count();
//
//    auto search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
//    ASSERT_TRUE(search_op.ok());
//
//    nlohmann::json res_obj = nlohmann::json::parse(json_res);
//    LOG(INFO) << res_obj.dump();
//}
