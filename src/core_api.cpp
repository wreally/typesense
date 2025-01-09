#include <chrono>
#include <thread>
#include <app_metrics.h>
#include <regex>
#include <analytics_manager.h>
#include <housekeeper.h>
#include "typesense_server_utils.h"
#include "core_api.h"
#include "string_utils.h"
#include "collection.h"
#include "collection_manager.h"
#include "system_metrics.h"
#include "logger.h"
#include "core_api_utils.h"
#include "lru/lru.hpp"
#include "ratelimit_manager.h"
#include "event_manager.h"
#include "http_proxy.h"
#include "include/stopwords_manager.h"
#include "conversation_manager.h"
#include "conversation_model_manager.h"
#include "conversation_model.h"
#include "personalization_model_manager.h"

using namespace std::chrono_literals;

std::shared_mutex mutex;
LRU::Cache<uint64_t, cached_res_t> res_cache;

std::shared_mutex alter_mutex;
std::set<std::string> alters_in_progress;

class alter_guard_t {
    std::string collection_name;
public:
    alter_guard_t(const std::string& collection) {
        std::unique_lock ulock(alter_mutex);
        collection_name = collection;
        alters_in_progress.insert(collection_name);
    }

    ~alter_guard_t() {
        std::unique_lock ulock(alter_mutex);
        alters_in_progress.erase(collection_name);
    }
};


class in_flight_req_guard_t {
    uint64_t req_id;
public:
    in_flight_req_guard_t(const std::shared_ptr<http_req>& req) {
        req_id = req->start_ts;
        HouseKeeper::get_instance().add_req(req);
    }

    ~in_flight_req_guard_t() {
        HouseKeeper::get_instance().remove_req(req_id);
    }
};

void init_api(uint32_t cache_num_entries) {
    std::unique_lock lock(mutex);
    res_cache.capacity(cache_num_entries);
}

bool get_alter_in_progress(const std::string& collection) {
    std::shared_lock lock(alter_mutex);
    return alters_in_progress.count(collection) != 0;
}

bool handle_authentication(std::map<std::string, std::string>& req_params,
                           std::vector<nlohmann::json>& embedded_params_vec,
                           const std::string& body,
                           const route_path& rpath,
                           const std::string& req_auth_key) {

    if(rpath.handler == get_health) {
        // health endpoint requires no authentication
        return true;
    }

    if(rpath.handler == get_health_with_resource_usage) {
        // health_rusage end-point will be authenticated via pre-determined keys
        return !req_auth_key.empty() && (
                req_auth_key == Config::get_instance().get_api_key() ||
                req_auth_key == Config::get_instance().get_health_rusage_api_key()
                );
    }

    CollectionManager & collectionManager = CollectionManager::get_instance();
    std::vector<collection_key_t> collections;
    get_collections_for_auth(req_params, body, rpath, req_auth_key, collections, embedded_params_vec);

    if(collections.size() != embedded_params_vec.size()) {
        LOG(ERROR) << "Impossible error: size of collections and embedded_params_vec don't match, "
                   << "collections.size: " << collections.size()
                   << ", embedded_params_vec.size: " << embedded_params_vec.size();
        return false;
    }

    return collectionManager.auth_key_matches(req_auth_key, rpath.action, collections, req_params, embedded_params_vec);
}

void stream_response(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    if(!res->is_alive) {
        // underlying request is dead or this is a raft log playback
        return ;
    }

    // wait for previous chunk to finish (if any)
    res->wait();

    auto req_res = new async_req_res_t(req, res, true);
    server->get_message_dispatcher()->send_message(HttpServer::STREAM_RESPONSE_MESSAGE, req_res);
}

void defer_processing(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res, size_t timeout_ms) {
    defer_processing_t* defer = new defer_processing_t(req, res, timeout_ms, server);
    //LOG(INFO) << "core_api req " << req.get() << ", use count: " << req.use_count();
    server->get_message_dispatcher()->send_message(HttpServer::DEFER_PROCESSING_MESSAGE, defer);
}

// we cannot return errors here because that will end up as auth failure and won't convey
// bad schema errors
void get_collections_for_auth(std::map<std::string, std::string>& req_params,
                                      const string& body,
                                      const route_path& rpath, const string& req_auth_key,
                                      std::vector<collection_key_t>& collections,
                                      std::vector<nlohmann::json>& embedded_params_vec) {

    if(rpath.handler == post_multi_search) {
        nlohmann::json req_obj;

        // If a `preset` parameter is present, we've to only load a pre-existing search configuration
        // and ignore the actual request body.
        auto preset_it = req_params.find("preset");
        if(preset_it != req_params.end()) {
            CollectionManager::get_instance().get_preset(preset_it->second, req_obj);
        } else {
            req_obj = nlohmann::json::parse(body, nullptr, false);
        }

        if(!req_obj.is_discarded() && req_obj.count("searches") != 0  && req_obj["searches"].is_array()) {
            for(auto& el : req_obj["searches"]) {
                if(el.is_object()) {
                    std::string coll_name;
                    if(el.count("collection") != 0 && el["collection"].is_string()) {
                        coll_name = el["collection"].get<std::string>();
                    } else if(req_params.count("collection") != 0) {
                        coll_name = req_params["collection"];
                    } else {
                        // if preset exists, that should be the lowest priority
                        if(el.count("preset") != 0) {
                            nlohmann::json preset_obj;
                            auto preset_op = CollectionManager::get_instance().
                                    get_preset(el["preset"].get<std::string>(), preset_obj);
                            if(preset_op.ok() && preset_obj.count("collection") != 0  &&
                                preset_obj["collection"].is_string()) {
                                coll_name = preset_obj["collection"].get<std::string>();
                            }
                        }
                    }

                    const std::string& access_key = (el.count("x-typesense-api-key") != 0 &&
                                                     el["x-typesense-api-key"].is_string()) ?
                                                    el["x-typesense-api-key"].get<std::string>() :
                                                    req_auth_key;

                    collections.emplace_back(coll_name, access_key);
                    embedded_params_vec.emplace_back(nlohmann::json::object());
                } else {
                    collections.emplace_back("", req_auth_key);
                    embedded_params_vec.emplace_back(nlohmann::json::object());
                }
            }
        } else {
            //LOG(ERROR) << "Multi search request body is malformed, body: " << body;
        }
    } else {
        if(rpath.handler == post_create_collection) {
            nlohmann::json obj = nlohmann::json::parse(body, nullptr, false);

            if(obj.is_discarded()) {
                LOG(ERROR) << "Create collection request body is malformed.";
            }

            else if(obj.count("name") != 0 && obj["name"].is_string()) {
                collections.emplace_back(obj["name"].get<std::string>(), req_auth_key);
                embedded_params_vec.emplace_back(nlohmann::json::object());
            }

        } else if(req_params.count("collection") != 0) {
            collections.emplace_back(req_params.at("collection"), req_auth_key);
            embedded_params_vec.emplace_back(nlohmann::json::object());
        }
    }

    if(collections.empty()) {
        collections.emplace_back("", req_auth_key);
        embedded_params_vec.emplace_back(nlohmann::json::object());
    }
}

index_operation_t get_index_operation(const std::string& action) {
    if(action == "create") {
        return CREATE;
    } else if(action == "update") {
        return UPDATE;
    } else if(action == "upsert") {
        return UPSERT;
    } else if(action == "emplace") {
        return EMPLACE;
    }

    return CREATE;
}

bool get_collections(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    CollectionManager & collectionManager = CollectionManager::get_instance();

    uint32_t offset = 0, limit = 0;
    std::vector<std::string> exclude_fields;

    if(req->params.count("offset") != 0) {
        const auto &offset_str = req->params["offset"];
        if(!StringUtils::is_uint32_t(offset_str)) {
            res->set(400, "Offset param should be unsigned integer.");
            return false;
        }
        offset = std::stoi(offset_str);
    }

    if(req->params.count("limit") != 0) {
        const auto &limit_str = req->params["limit"];
        if(!StringUtils::is_uint32_t(limit_str)) {
            res->set(400, "Limit param should be unsigned integer.");
            return false;
        }
        limit = std::stoi(limit_str);
    }

    if(req->params.count("exclude_fields") != 0) {
        const auto& exclude_fields_str = req->params["exclude_fields"];
        StringUtils::split(exclude_fields_str, exclude_fields, ",");
    }

    AuthManager &auth_manager = collectionManager.getAuthManager();
    auto api_key_collections = auth_manager.get_api_key_collections(req->api_auth_key);

    auto collections_summaries_op = collectionManager.get_collection_summaries(limit, offset, exclude_fields,
                                                                               api_key_collections);
    if(!collections_summaries_op.ok()) {
        res->set(collections_summaries_op.code(), collections_summaries_op.error());
        return false;
    }

    nlohmann::json json_response = collections_summaries_op.get();
    res->set_200(json_response.dump());
    return true;
}

bool post_create_collection(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    nlohmann::json req_json;

    try {
        req_json = nlohmann::json::parse(req->body);
    } catch(const std::exception& e) {
        //LOG(ERROR) << "JSON error: " << e.what();
        res->set_400("Bad JSON.");
        return false;
    }

    const std::string SRC_COLL_NAME = "src_name";

    /*if(res->is_alive && req_json.is_object() && req_json.count("enable_nested_fields") == 0) {
        // This patch ensures that nested fields are only enabled for collections created on Typesense versions
        // which support nested fields. This ensures that ".*" schema does not end up duplicating fields on
        // manually flattened collection schemas that also contain nested versions for convenience.
        // TO BE ENABLED WHEN READY!
        // req_json["enable_nested_fields"] = true;
    }*/

    CollectionManager& collectionManager = CollectionManager::get_instance();
    const Option<Collection*> &collection_op = req->params.count(SRC_COLL_NAME) != 0 ?
               collectionManager.clone_collection(req->params[SRC_COLL_NAME], req_json) :
               CollectionManager::create_collection(req_json);

    if(collection_op.ok()) {
        nlohmann::json json_response = collection_op.get()->get_summary_json();
        res->set_201(json_response.dump());
        return true;
    }

    res->set(collection_op.code(), collection_op.error());
    return false;
}

bool patch_update_collection(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    nlohmann::json req_json;
    std::set<std::string> allowed_keys = {"metadata", "fields"};

    // Ensures that only one alter can run per collection.
    // The actual check for this, happens in `ReplicationState::write` which is called only during live writes.
    alter_guard_t alter_guard(req->params["collection"]);

    try {
        req_json = nlohmann::json::parse(req->body);
    } catch(const std::exception& e) {
        //LOG(ERROR) << "JSON error: " << e.what();
        res->set_400("Bad JSON.");
        return false;
    }

    if(req_json.empty()) {
        res->set_400("Alter payload is empty.");
        return false;
    }

    for(auto it : req_json.items()) {
        if(allowed_keys.count(it.key()) == 0) {
            res->set_400("Only `fields` and `metadata` can be updated at the moment.");
            return false;
        }
    }

    CollectionManager & collectionManager = CollectionManager::get_instance();
    auto collection = collectionManager.get_collection(req->params["collection"]);

    if(collection == nullptr) {
        res->set_404();
        return false;
    }

    if(req_json.contains("metadata")) {
        if(!req_json["metadata"].is_object()) {
            res->set_400("The `metadata` value should be an object.");
            return false;
        }

        //update in collection metadata and store in db
        auto op = collectionManager.update_collection_metadata(req->params["collection"], req_json["metadata"]);
        if(!op.ok()) {
            res->set(op.code(), op.error());
            return false;
        }
    }

    if(req_json.contains("fields")) {
        nlohmann::json alter_payload;
        alter_payload["fields"] = req_json["fields"];
        auto alter_op = collection->alter(alter_payload);
        if(!alter_op.ok()) {
            res->set(alter_op.code(), alter_op.error());
            return false;
        }
        // without this line, response will return full api key without being masked
        req_json["fields"] = alter_payload["fields"];
    }

    res->set_200(req_json.dump());
    return true;
}

bool del_drop_collection(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    bool compact_store = false;

    if(req->params.count("compact_store") != 0) {
        compact_store = (req->params["compact_store"] == "true");
    }

    CollectionManager & collectionManager = CollectionManager::get_instance();
    Option<nlohmann::json> drop_op = collectionManager.drop_collection(req->params["collection"], true, compact_store);

    if(!drop_op.ok()) {
        res->set(drop_op.code(), drop_op.error());
        return false;
    }

    res->set_200(drop_op.get().dump());
    return true;
}

bool get_debug(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    bool log_inflight_queries = false;

    if(req->params.count("log_inflight_queries") != 0) {
        log_inflight_queries = (req->params["log_inflight_queries"] == "true");
    }

    if(log_inflight_queries) {
        HouseKeeper::get_instance().log_running_queries();
    }

    nlohmann::json result;
    result["version"] = server->get_version();

    uint64_t state = server->node_state();
    result["state"] = state;

    res->set_200(result.dump());
    return true;
}

bool get_health_with_resource_usage(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    nlohmann::json result;
    bool alive = server->is_alive();

    auto resource_check = cached_resource_stat_t::get_instance().has_enough_resources(
        Config::get_instance().get_data_dir(),
        Config::get_instance().get_disk_used_max_percentage(),
        Config::get_instance().get_memory_used_max_percentage()
    );

    if (resource_check != cached_resource_stat_t::resource_check_t::OK) {
        result["resource_error"] = std::string(magic_enum::enum_name(resource_check));
    }

    if(req->params.count("cpu_threshold") != 0 && StringUtils::is_float(req->params["cpu_threshold"])) {
        float cpu_threshold = std::stof(req->params["cpu_threshold"]);
        std::vector<cpu_stat_t> cpu_stats = SystemMetrics::get_instance().get_cpu_stats();
        if(!cpu_stats.empty() && StringUtils::is_float(cpu_stats[0].active)) {
            alive = alive && (std::stof(cpu_stats[0].active) < cpu_threshold);
        }
    }

    result["ok"] = alive;

    if(alive) {
        res->set_body(200, result.dump());
    } else {
        res->set_body(503, result.dump());
    }

    return alive;
}

bool get_health(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    nlohmann::json result;
    bool alive = server->is_alive();
    result["ok"] = alive;

    auto resource_check = cached_resource_stat_t::get_instance().has_enough_resources(
        Config::get_instance().get_data_dir(),
        Config::get_instance().get_disk_used_max_percentage(),
        Config::get_instance().get_memory_used_max_percentage()
    );

    if (resource_check != cached_resource_stat_t::resource_check_t::OK) {
        result["resource_error"] = std::string(magic_enum::enum_name(resource_check));
    }

    if(alive) {
        res->set_body(200, result.dump());
    } else {
        res->set_body(503, result.dump());
    }

    return alive;
}

bool post_health(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    nlohmann::json result;
    bool alive = server->is_alive();
    result["ok"] = alive;

    if(alive) {
        res->set_body(200, result.dump());
    } else {
        res->set_body(503, result.dump());
    }

    return alive;
}

bool get_metrics_json(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    nlohmann::json result;

    CollectionManager & collectionManager = CollectionManager::get_instance();
    const std::string & data_dir_path = collectionManager.get_store()->get_state_dir_path();

    SystemMetrics::get_instance().get(data_dir_path, result);

    res->set_body(200, result.dump(2));
    return true;
}

bool get_stats_json(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    nlohmann::json result;
    AppMetrics::get_instance().get("requests_per_second", "latency_ms", result);
    result["pending_write_batches"] = server->get_num_queued_writes();

    res->set_body(200, result.dump(2));
    return true;
}

bool get_status(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    nlohmann::json status = server->node_status();
    res->set_body(200, status.dump());
    return true;
}

uint64_t hash_request(const std::shared_ptr<http_req>& req) {
    std::stringstream ss;
    ss << req->route_hash << req->body;

    for(auto& kv: req->params) {
        if(kv.first != "use_cache") {
            ss << kv.second;
        }
    }

    const std::string& req_str = ss.str();
    return StringUtils::hash_wy(req_str.c_str(), req_str.size());
}

bool get_search(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    const auto use_cache_it = req->params.find("use_cache");
    bool use_cache = (use_cache_it != req->params.end()) && (use_cache_it->second == "1" || use_cache_it->second == "true");
    uint64_t req_hash = 0;

    in_flight_req_guard_t in_flight_req_guard(req);

    if(use_cache) {
        // cache enabled, let's check if request is already in the cache
        req_hash = hash_request(req);

        //LOG(INFO) << "req_hash = " << req_hash;

        std::unique_lock lock(mutex);
        auto hit_it = res_cache.find(req_hash);
        if(hit_it != res_cache.end()) {
            //LOG(INFO) << "Result found in cache.";
            const auto& cached_value = hit_it.value();

            // we still need to check that TTL has not expired
            uint32_t ttl = cached_value.ttl;
            uint64_t seconds_elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::high_resolution_clock::now() - cached_value.created_at).count();

            if(seconds_elapsed < cached_value.ttl) {
                res->set_content(cached_value.status_code, cached_value.content_type_header, cached_value.body, true);
                return true;
            }

            // Result found in cache but ttl has lapsed.
            res_cache.erase(req_hash);
        }
    }

    if(req->embedded_params_vec.empty()) {
        res->set_500("Embedded params is empty.");
        return false;
    }

    std::string results_json_str;
    Option<bool> search_op = CollectionManager::do_search(req->params, req->embedded_params_vec[0],
                                                          results_json_str, req->conn_ts);

    if(!search_op.ok()) {
        res->set(search_op.code(), search_op.error());
        if(search_op.code() == 408) {
            req->overloaded = true;
        }
        return false;
    }

    res->set_200(results_json_str);

    // we will cache only successful requests
    if(use_cache) {
        //LOG(INFO) << "Adding to cache, key = " << req_hash;
        auto now = std::chrono::high_resolution_clock::now();
        const auto cache_ttl_it = req->params.find("cache_ttl");
        uint32_t cache_ttl = 60;
        if(cache_ttl_it != req->params.end() && StringUtils::is_int32_t(cache_ttl_it->second)) {
            cache_ttl = std::stoul(cache_ttl_it->second);
        }

        cached_res_t cached_res;
        cached_res.load(res->status_code, res->content_type_header, res->body, now, cache_ttl, req_hash);

        std::unique_lock lock(mutex);
        res_cache.insert(req_hash, cached_res);
    }

    return true;
}

bool post_multi_search(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    const auto use_cache_it = req->params.find("use_cache");
    bool use_cache = (use_cache_it != req->params.end()) && (use_cache_it->second == "1" || use_cache_it->second == "true");
    uint64_t req_hash = 0;

    in_flight_req_guard_t in_flight_req_guard(req);

    if(use_cache) {
        // cache enabled, let's check if request is already in the cache
        req_hash = hash_request(req);

        //LOG(INFO) << "req_hash = " << req_hash;

        std::unique_lock lock(mutex);
        auto hit_it = res_cache.find(req_hash);
        if(hit_it != res_cache.end()) {
            //LOG(INFO) << "Result found in cache.";
            const auto& cached_value = hit_it.value();

            // we still need to check that TTL has not expired
            uint32_t ttl = cached_value.ttl;
            uint64_t seconds_elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::high_resolution_clock::now() - cached_value.created_at).count();

            if(seconds_elapsed < cached_value.ttl) {
                res->set_content(cached_value.status_code, cached_value.content_type_header, cached_value.body, true);
                return true;
            }

            // Result found in cache but ttl has lapsed.
            res_cache.erase(req_hash);
        }
    }

    nlohmann::json req_json;

    const auto preset_it = req->params.find("preset");
    if(preset_it != req->params.end()) {
        CollectionManager::get_instance().get_preset(preset_it->second, req_json);
    } else {
        try {
            req_json = nlohmann::json::parse(req->body);
        } catch(const std::exception& e) {
            LOG(ERROR) << "JSON error: " << e.what();
            res->set_400("Bad JSON.");
            return false;
        }
    }

    if(req_json.count("searches") == 0) {
        res->set_400("Missing `searches` array.");
        return false;
    }

    if(!req_json["searches"].is_array()) {
        res->set_400("Missing `searches` array.");
        return false;
    }

    if(req->embedded_params_vec.empty()) {
        res->set_400("Missing embedded params array.");
        return false;
    }

    auto orig_req_params = req->params;
    const char* LIMIT_MULTI_SEARCHES = "limit_multi_searches";
    size_t limit_multi_searches = 50;

    if(orig_req_params.count(LIMIT_MULTI_SEARCHES) != 0 &&
        StringUtils::is_uint32_t(orig_req_params[LIMIT_MULTI_SEARCHES])) {
        limit_multi_searches = std::stoi(orig_req_params[LIMIT_MULTI_SEARCHES]);
    }

    const auto& first_embedded_param = req->embedded_params_vec[0];
    if(first_embedded_param.count(LIMIT_MULTI_SEARCHES) != 0 && first_embedded_param[LIMIT_MULTI_SEARCHES].is_number_integer()) {
        limit_multi_searches = first_embedded_param[LIMIT_MULTI_SEARCHES].get<size_t>();
    }

    if(req_json["searches"].size() > limit_multi_searches) {
        res->set_400(std::string("Number of multi searches exceeds `") + LIMIT_MULTI_SEARCHES + "` parameter.");
        return false;
    }

    nlohmann::json response;

    nlohmann::json& searches = req_json["searches"];

    if(searches.size() != req->embedded_params_vec.size()) {
        LOG(ERROR) << "Embedded params parsing error: length does not match multi search array, searches.size(): "
                   << searches.size() << ", embedded_params_vec.size: " << req->embedded_params_vec.size()
                   << ", req_body: " << req->body;
        res->set_500("Embedded params parsing error.");
        return false;
    }

    // Get API key and IP
    if(!req->metadata.empty()) {
        auto api_key_ip_op = get_api_key_and_ip(req->metadata);
        if(!api_key_ip_op.ok()) {
            res->set(api_key_ip_op.code(), api_key_ip_op.error());
            return false;
        }
        const auto& api_key_ip = api_key_ip_op.get();
        auto rate_limit_manager = RateLimitManager::getInstance();

        // Check rate limiting first before doing any search, don't want to waste time if we're rate limited
        for(size_t i = 0; i < searches.size(); i++) {
            if(RateLimitManager::getInstance()->is_rate_limited({RateLimitedEntityType::api_key, api_key_ip.first}, {RateLimitedEntityType::ip, api_key_ip.second})) {
                res->set(429, "Rate limit exceeded or blocked");
                return false;
            }
        }
    }

    const char* UNION_RESULT = "union";
    auto is_union = false;
    auto it = req_json.find(UNION_RESULT);
    if (it != req_json.end() && it.value().is_boolean()) {
        is_union = it.value();
    }

    bool conversation = orig_req_params["conversation"] == "true" && !is_union;
    bool conversation_history = orig_req_params.find("conversation_id") != orig_req_params.end();
    std::string common_query;

    if(!conversation && conversation_history) {
        res->set_400("`conversation_id` can only be used if `conversation` is enabled.");
        return false;
    }

    if(conversation) {
        if(orig_req_params.find("q") == orig_req_params.end()) {
            res->set_400("`q` parameter has to be common for all searches if conversation is enabled. Please set `q` as a query parameter in the request, instead of inside the POST body");
            return false;
        }

        if(orig_req_params.find("conversation_model_id") == orig_req_params.end()) {
            res->set_400("`conversation_model_id` is needed if conversation is enabled.");
            return false;
        }

        const std::string& conversation_model_id = orig_req_params["conversation_model_id"];
        auto conversation_model_op = ConversationModelManager::get_model(conversation_model_id);

        if(!conversation_model_op.ok()) {
            res->set_400("`conversation_model_id` is invalid.");
            return false;
        }
        auto conversation_model = conversation_model_op.get();
        if(conversation_history) {
            std::string conversation_id = orig_req_params["conversation_id"];

            auto conversation_history = ConversationManager::get_instance().get_conversation(conversation_id, conversation_model);

            if(!conversation_history.ok()) {
                res->set_400("`conversation_id` is invalid.");
                return false;
            }
        }

        common_query = orig_req_params["q"];

        if(conversation_history) {
            const std::string& conversation_model_id = orig_req_params["conversation_model_id"];
            auto conversation_id = orig_req_params["conversation_id"];
            auto conversation_history = ConversationManager::get_instance().get_conversation(conversation_id, conversation_model).get();
            auto generate_standalone_q = ConversationModel::get_standalone_question(conversation_history, common_query, conversation_model);

            if(!generate_standalone_q.ok()) {
                res->set_400(generate_standalone_q.error());
                return false;
            }

            orig_req_params["q"] = generate_standalone_q.get();
        }
    }

    if (searches.size() > 1 && is_union) {
        Option<bool> union_op = CollectionManager::do_union(req->params, req->embedded_params_vec, searches,
                                                            response, req->conn_ts);
        if(!union_op.ok() && union_op.code() == 408) {
            res->set(union_op.code(), union_op.error());
            req->overloaded = true;
            return false;
        }
    } else {
        response["results"] = nlohmann::json::array();

        for(size_t i = 0; i < searches.size(); i++) {
            auto& search_params = searches[i];
            req->params = orig_req_params;

            auto validate_op = multi_search_validate_and_add_params(req->params, search_params, conversation);
            if (!validate_op.ok()) {
                res->set_400(validate_op.error());
                return false;
            }

            std::string results_json_str;
            Option<bool> search_op = CollectionManager::do_search(req->params, req->embedded_params_vec[i],
                                                                  results_json_str, req->conn_ts);

            if(search_op.ok()) {
                auto results_json = nlohmann::json::parse(results_json_str);
                if(conversation) {
                    results_json["request_params"]["q"] = common_query;
                }
                response["results"].push_back(results_json);
            } else {
                if(search_op.code() == 408) {
                    res->set(search_op.code(), search_op.error());
                    req->overloaded = true;
                    return false;
                }
                nlohmann::json err_res;
                err_res["error"] = search_op.error();
                err_res["code"] = search_op.code();
                response["results"].push_back(err_res);
            }
        }
    }

    if(conversation) {
        nlohmann::json result_docs_arr = nlohmann::json::array();
        int res_index = 0;
        for(const auto& result : response["results"]) {
            if(result.count("code") != 0) {
                continue;
            }

            nlohmann::json result_docs = nlohmann::json::array();

            std::vector<std::string> vector_fields;

            auto collection = CollectionManager::get_instance().get_collection(searches[res_index]["collection"].get<std::string>());
            auto search_schema = collection->get_schema();

            for(const auto& field : search_schema) {
                if(field.type == field_types::FLOAT_ARRAY) {
                    vector_fields.push_back(field.name);
                }
            }

            if(result.contains("grouped_hits")) {
                for(const auto& grouped_hit : result["grouped_hits"]) {
                    for(const auto& hit : grouped_hit["hits"]) {
                        auto doc = hit["document"];
                        for(const auto& vector_field : vector_fields) {
                            if(doc.contains(vector_field)) {
                                doc.erase(vector_field);
                            }
                        }
                        result_docs.push_back(doc);
                    }
                }
            }
            else {
                for(const auto& hit : result["hits"]) {
                    auto doc = hit["document"];
                    for(const auto& vector_field : vector_fields) {
                        if(doc.contains(vector_field)) {
                            doc.erase(vector_field);
                        }
                    }
                    result_docs.push_back(doc);
                }
            }

            result_docs_arr.push_back(result_docs);
        }

        const std::string& conversation_model_id = orig_req_params["conversation_model_id"];
        auto conversation_model = ConversationModelManager::get_model(conversation_model_id).get();
        auto min_required_bytes_op = ConversationModel::get_minimum_required_bytes(conversation_model);
        if(!min_required_bytes_op.ok()) {
            res->set_400(min_required_bytes_op.error());
            return false;
        }
        auto min_required_bytes = min_required_bytes_op.get();
        auto prompt = req->params["q"];
        if(conversation_model["max_bytes"].get<size_t>() < min_required_bytes + prompt.size()) {
            res->set_400("`max_bytes` of the conversation model is less than the minimum required bytes(" + std::to_string(min_required_bytes) + ").");
            return false;
        }

        // remove document with lowest score until total tokens is less than MAX_TOKENS
        while(result_docs_arr.dump(0).size() > conversation_model["max_bytes"].get<size_t>() - min_required_bytes - prompt.size()) {
            // sort the result_docs_arr by size descending
            std::sort(result_docs_arr.begin(), result_docs_arr.end(), [](const auto& a, const auto& b) {
                return a.size() > b.size();
            });

            // pop the last element from first array
            if(result_docs_arr.size() > 0 && result_docs_arr[0].size() > 0) {
                result_docs_arr[0].erase(result_docs_arr[0].size() - 1);
            }
        }

        // Make result_docs_arr 1D
        nlohmann::json result_docs = nlohmann::json::array();
        for(const auto& result_doc : result_docs_arr) {
            for(const auto& doc : result_doc) {
                result_docs.push_back(doc);
            }
        }

        auto answer_op = ConversationModel::get_answer(result_docs.dump(0), prompt, conversation_model);

        if(!answer_op.ok()) {
            res->set_400(answer_op.error());
            return false;
        }

        response["conversation"] = nlohmann::json::object();
        response["conversation"]["query"] = common_query;
        response["conversation"]["answer"] = answer_op.get();
        std::string conversation_id = conversation_history ? orig_req_params["conversation_id"] : "";
        auto conversation_history_op = ConversationManager::get_instance().get_full_conversation(common_query, answer_op.get(), conversation_model, conversation_id);
        if(!conversation_history_op.ok()) {
            res->set_400(conversation_history_op.error());
            return false;
        }

        auto conversation_history = conversation_history_op.get();

        std::vector<std::string> exclude_fields;
        StringUtils::split(req->params["exclude_fields"], exclude_fields, ",");
        bool exclude_conversation_history = std::find(exclude_fields.begin(), exclude_fields.end(), "conversation_history") != exclude_fields.end();
        
        auto new_conversation_op = ConversationManager::get_last_n_messages(conversation_history["conversation"], 2);
        if(!new_conversation_op.ok()) {
            res->set_400(new_conversation_op.error());
            return false;
        }
        auto new_conversation = new_conversation_op.get();

        auto add_conversation_op = ConversationManager::get_instance().add_conversation(new_conversation, conversation_model, conversation_id);
        if(!add_conversation_op.ok()) {
            res->set_400(add_conversation_op.error());
            return false;
        }

        if(!exclude_conversation_history) {
            response["conversation"]["conversation_history"] = conversation_history;
        }
        response["conversation"]["conversation_id"] = add_conversation_op.get();

    }

    res->set_200(response.dump());

    // we will cache only successful requests
    if(use_cache) {
        //LOG(INFO) << "Adding to cache, key = " << req_hash;
        auto now = std::chrono::high_resolution_clock::now();
        const auto cache_ttl_it = req->params.find("cache_ttl");
        uint32_t cache_ttl = 60;
        if(cache_ttl_it != req->params.end() && StringUtils::is_int32_t(cache_ttl_it->second)) {
            cache_ttl = std::stoul(cache_ttl_it->second);
        }

        cached_res_t cached_res;
        cached_res.load(res->status_code, res->content_type_header, res->body, now, cache_ttl, req_hash);

        std::unique_lock lock(mutex);
        res_cache.insert(req_hash, cached_res);
    }

    return true;
}

bool get_collection_summary(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    CollectionManager& collectionManager = CollectionManager::get_instance();
    auto collection = collectionManager.get_collection(req->params["collection"]);

    if(collection == nullptr) {
        res->set_404();
        return false;
    }

    nlohmann::json json_response = collection->get_summary_json();
    res->set_200(json_response.dump(-1, ' ', false, nlohmann::detail::error_handler_t::ignore));

    return true;
}

bool get_export_documents(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    // NOTE: this is a streaming response end-point so this handler will be called multiple times
    CollectionManager & collectionManager = CollectionManager::get_instance();
    auto collection = collectionManager.get_collection(req->params["collection"]);

    if(collection == nullptr) {
        req->last_chunk_aggregate = true;
        res->final = true;
        res->set_404();
        stream_response(req, res);
        return false;
    }

    const char* FILTER_BY = "filter_by";
    const char* INCLUDE_FIELDS = "include_fields";
    const char* EXCLUDE_FIELDS = "exclude_fields";
    const char* BATCH_SIZE = "batch_size";
    const char* VALIDATE_FIELD_NAMES = "validate_field_names";

    export_state_t* export_state = nullptr;

    const std::string seq_id_prefix = collection->get_seq_id_collection_prefix();

    if(req->data == nullptr) {
        export_state = new export_state_t();
        export_state->collection = collection.get();

        // destruction of data is managed by req destructor
        req->data = export_state;

        std::string filter_query;
        std::vector<std::string> include_fields_vec;
        std::vector<std::string> exclude_fields_vec;
        spp::sparse_hash_set<std::string> exclude_fields;
        spp::sparse_hash_set<std::string> include_fields;

        if(req->params.count(FILTER_BY) != 0) {
            filter_query = req->params[FILTER_BY];
        }

        if(req->params.count(INCLUDE_FIELDS) != 0) {
            auto op = StringUtils::split_include_exclude_fields(req->params[INCLUDE_FIELDS], include_fields_vec);
            if (!op.ok()) {
                res->set(op.code(), op.error());
                req->last_chunk_aggregate = true;
                res->final = true;
                stream_response(req, res);
                return false;
            }
        }

        if(req->params.count(EXCLUDE_FIELDS) != 0) {
            auto op = StringUtils::split_include_exclude_fields(req->params[EXCLUDE_FIELDS], exclude_fields_vec);
            if (!op.ok()) {
                res->set(op.code(), op.error());
                req->last_chunk_aggregate = true;
                res->final = true;
                stream_response(req, res);
                return false;
            }
        }

        auto initialize_op = Join::initialize_ref_include_exclude_fields_vec(filter_query, include_fields_vec, exclude_fields_vec,
                                                                             export_state->ref_include_exclude_fields_vec);
        if (!initialize_op.ok()) {
            res->set(initialize_op.code(), initialize_op.error());
            req->last_chunk_aggregate = true;
            res->final = true;
            stream_response(req, res);
            return false;
        }

        include_fields.insert(include_fields_vec.begin(), include_fields_vec.end());
        exclude_fields.insert(exclude_fields_vec.begin(), exclude_fields_vec.end());

        collection->populate_include_exclude_fields_lk(include_fields, exclude_fields,
                                                      export_state->include_fields, export_state->exclude_fields);

        if(req->params.count(BATCH_SIZE) != 0 && StringUtils::is_uint32_t(req->params[BATCH_SIZE])) {
            export_state->export_batch_size = std::stoul(req->params[BATCH_SIZE]);
        }

        if(filter_query.empty()) {
            export_state->iter_upper_bound_key = collection->get_seq_id_collection_prefix() + "`";  // cannot inline this
            export_state->iter_upper_bound = new rocksdb::Slice(export_state->iter_upper_bound_key);
            export_state->it = collectionManager.get_store()->scan(seq_id_prefix, export_state->iter_upper_bound);
        } else {
            bool validate_field_names = true;
            if (req->params.count(VALIDATE_FIELD_NAMES) != 0 && req->params[VALIDATE_FIELD_NAMES] == "false") {
                validate_field_names = false;
            }

            auto filter_ids_op = collection->get_filter_ids(filter_query, export_state->filter_result, false,
                                                            validate_field_names);

            if(!filter_ids_op.ok()) {
                res->set(filter_ids_op.code(), filter_ids_op.error());
                req->last_chunk_aggregate = true;
                res->final = true;
                stream_response(req, res);
                return false;
            }

            export_state->res_body = &res->body;
        }
    } else {
        export_state = dynamic_cast<export_state_t*>(req->data);
    }

    if(export_state->it != nullptr) {
        rocksdb::Iterator* it = export_state->it;
        size_t batch_counter = 0;
        std::string().swap(res->body);

        while(it->Valid() && it->key().ToString().compare(0, seq_id_prefix.size(), seq_id_prefix) == 0) {
            nlohmann::json doc = nlohmann::json::parse(it->value().ToString());
            Collection::remove_flat_fields(doc);
            Collection::remove_reference_helper_fields(doc);

            if(export_state->include_fields.empty() && export_state->exclude_fields.empty()) {
                res->body += doc.dump();
            } else {
                if (doc.count("id") == 0 || !doc.at("id").is_string()) {
                    res->set(500, "Could not find `id` field in the document: " + doc.dump());
                    req->last_chunk_aggregate = true;
                    res->final = true;
                    stream_response(req, res);
                    return false;
                }

                auto const& coll = export_state->collection;
                auto const seq_id_op = coll->doc_id_to_seq_id_with_lock(doc.at("id"));
                if (!seq_id_op.ok()) {
                    res->set(seq_id_op.code(), seq_id_op.error());
                    req->last_chunk_aggregate = true;
                    res->final = true;
                    stream_response(req, res);
                    return false;
                }

                std::map<std::string, reference_filter_result_t> references = {};
                coll->prune_doc_with_lock(doc, export_state->include_fields, export_state->exclude_fields,
                                          references, seq_id_op.get(), export_state->ref_include_exclude_fields_vec);
                res->body += doc.dump();
            }

            it->Next();

            // append a new line character if there is going to be one more record to send
            if(it->Valid() && it->key().ToString().compare(0, seq_id_prefix.size(), seq_id_prefix) == 0) {
                res->body += "\n";
                req->last_chunk_aggregate = false;
                res->final = false;
            } else {
                req->last_chunk_aggregate = true;
                res->final = true;
            }

            batch_counter++;
            if(batch_counter == export_state->export_batch_size) {
                break;
            }
        }
    } else {
        bool done;
        stateful_export_docs(export_state, export_state->export_batch_size, done);

        if(!done) {
            req->last_chunk_aggregate = false;
            res->final = false;
        } else {
            req->last_chunk_aggregate = true;
            res->final = true;
        }
    }

    res->content_type_header = "text/plain; charset=utf-8";
    res->status_code = 200;

    stream_response(req, res);
    return true;
}

bool post_import_documents(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    //LOG(INFO) << "Import, req->body_index=" << req->body_index << ", body size: " << req->body.size();
    //LOG(INFO) << "req->first_chunk=" << req->first_chunk_aggregate << ", last_chunk=" << req->last_chunk_aggregate;

    const char *BATCH_SIZE = "batch_size";
    const char *ACTION = "action";
    const char *DIRTY_VALUES = "dirty_values";
    const char *RETURN_DOC = "return_doc";
    const char *RETURN_ID = "return_id";
    const char *REMOTE_EMBEDDING_BATCH_SIZE = "remote_embedding_batch_size";
    const char *REMOTE_EMBEDDING_TIMEOUT_MS = "remote_embedding_timeout_ms";
    const char *REMOTE_EMBEDDING_NUM_TRIES = "remote_embedding_num_tries";

    if(req->params.count(BATCH_SIZE) == 0) {
        req->params[BATCH_SIZE] = "40";
    }

    if(req->params.count(REMOTE_EMBEDDING_BATCH_SIZE) == 0) {
        req->params[REMOTE_EMBEDDING_BATCH_SIZE] = "200";
    }

    if(req->params.count(ACTION) == 0) {
        req->params[ACTION] = "create";
    }

    if(req->params.count(DIRTY_VALUES) == 0) {
        req->params[DIRTY_VALUES] = "";  // set it empty as default will depend on `index_all_fields`
    }

    if(req->params.count(RETURN_DOC) == 0) {
        req->params[RETURN_DOC] = "false";
    }

    if(req->params.count(RETURN_ID) == 0) {
        req->params[RETURN_ID] = "false";
    }

    if(!StringUtils::is_uint32_t(req->params[BATCH_SIZE])) {
        res->final = true;
        res->set_400("Parameter `" + std::string(BATCH_SIZE) + "` must be a positive integer.");
        stream_response(req, res);
        return false;
    }

    if(req->params[ACTION] != "create" && req->params[ACTION] != "update" && req->params[ACTION] != "upsert" &&
       req->params[ACTION] != "emplace") {
        res->final = true;
        res->set_400("Parameter `" + std::string(ACTION) + "` must be a create|update|upsert.");
        stream_response(req, res);
        return false;
    }

    if(req->params[RETURN_DOC] != "true" && req->params[RETURN_DOC] != "false") {
        res->final = true;
        res->set_400("Parameter `" + std::string(RETURN_DOC) + "` must be a true|false.");
        stream_response(req, res);
        return false;
    }

    if(req->params[RETURN_ID] != "true" && req->params[RETURN_ID] != "false") {
        res->final = true;
        res->set_400("Parameter `" + std::string(RETURN_ID) + "` must be a true|false.");
        stream_response(req, res);
        return false;
    }

    if(req->params.count(REMOTE_EMBEDDING_TIMEOUT_MS) == 0) {
        req->params[REMOTE_EMBEDDING_TIMEOUT_MS] = "60000";
    }

    if(req->params.count(REMOTE_EMBEDDING_NUM_TRIES) == 0) {
        req->params[REMOTE_EMBEDDING_NUM_TRIES] = "2";
    }

    const size_t IMPORT_BATCH_SIZE = std::stoi(req->params[BATCH_SIZE]);
    const size_t REMOTE_EMBEDDING_BATCH_SIZE_VAL = std::stoi(req->params[REMOTE_EMBEDDING_BATCH_SIZE]);
    const size_t REMOTE_EMBEDDING_TIMEOUT_MS_VAL = std::stoi(req->params[REMOTE_EMBEDDING_TIMEOUT_MS]);
    const size_t REMOTE_EMBEDDING_NUM_TRIES_VAL = std::stoi(req->params[REMOTE_EMBEDDING_NUM_TRIES]);

    if(IMPORT_BATCH_SIZE == 0) {
        res->final = true;
        res->set_400("Parameter `" + std::string(BATCH_SIZE) + "` must be a positive integer.");
        stream_response(req, res);
        return false;
    }

    if(REMOTE_EMBEDDING_BATCH_SIZE_VAL == 0) {
        res->final = true;
        res->set_400("Parameter `" + std::string(REMOTE_EMBEDDING_BATCH_SIZE) + "` must be a positive integer.");
        stream_response(req, res);
        return false;
    }

    if(REMOTE_EMBEDDING_TIMEOUT_MS_VAL == 0) {
        res->final = true;
        res->set_400("Parameter `" + std::string(REMOTE_EMBEDDING_TIMEOUT_MS) + "` must be a positive integer.");
        stream_response(req, res);
        return false;
    }

    if(REMOTE_EMBEDDING_NUM_TRIES_VAL == 0) {
        res->final = true;
        res->set_400("Parameter `" + std::string(REMOTE_EMBEDDING_NUM_TRIES) + "` must be a positive integer.");
        stream_response(req, res);
        return false;
    }

    if(req->body_index == 0) {
        // will log for every major chunk of request body
        //LOG(INFO) << "Import, req->body.size=" << req->body.size() << ", batch_size=" << IMPORT_BATCH_SIZE;
        //int nminusten_pos = std::max(0, int(req->body.size())-10);
        //LOG(INFO) << "Last 10 chars: " << req->body.substr(nminusten_pos);
    }

    CollectionManager & collectionManager = CollectionManager::get_instance();
    auto collection = collectionManager.get_collection(req->params["collection"]);

    if(collection == nullptr) {
        //LOG(INFO) << "collection == nullptr, for collection: " << req->params["collection"];
        res->final = true;
        res->set_404();
        stream_response(req, res);
        return false;
    }

    //LOG(INFO) << "Import, " << "req->body_index=" << req->body_index << ", req->body.size: " << req->body.size();
    //LOG(INFO) << "req body %: " << (float(req->body_index)/req->body.size())*100;

    std::vector<std::string> json_lines;
    StringUtils::split(req->body, json_lines, "\n", false, false);

    //LOG(INFO) << "json_lines.size before: " << json_lines.size() << ", req->body_index: " << req->body_index;

    if(req->last_chunk_aggregate) {
        //LOG(INFO) << "req->last_chunk_aggregate is true";
        req->body = "";
    } else {
        if(!json_lines.empty()) {
            // check if req->body had complete last record
            bool complete_document;

            try {
                nlohmann::json document = nlohmann::json::parse(json_lines.back());
                complete_document = document.is_object();
            } catch(const std::exception& e) {
                complete_document = false;
            }

            if(!complete_document) {
                // eject partial record
                req->body = json_lines.back();
                json_lines.pop_back();
            } else {
                req->body = "";
            }
        }
    }

    //LOG(INFO) << "json_lines.size after: " << json_lines.size() << ", stream_proceed: " << stream_proceed;
    //LOG(INFO) << "json_lines.size: " << json_lines.size() << ", req->res_state: " << req->res_state;

    // When only one partial record arrives as a chunk, an empty body is pushed to response stream
    bool single_partial_record_body = (json_lines.empty() && !req->body.empty());
    std::stringstream response_stream;

    //LOG(INFO) << "single_partial_record_body: " << single_partial_record_body;

    const index_operation_t operation = get_index_operation(req->params[ACTION]);

    if(!single_partial_record_body) {
        nlohmann::json document;

        const auto& dirty_values = collection->parse_dirty_values_option(req->params[DIRTY_VALUES]);
        const bool& return_doc = req->params[RETURN_DOC] == "true";
        const bool& return_id = req->params[RETURN_ID] == "true";
        nlohmann::json json_res = collection->add_many(json_lines, document, operation, "",
                                                       dirty_values, return_doc, return_id, REMOTE_EMBEDDING_BATCH_SIZE_VAL, REMOTE_EMBEDDING_TIMEOUT_MS_VAL, REMOTE_EMBEDDING_NUM_TRIES_VAL);
        //const std::string& import_summary_json = json_res->dump();
        //response_stream << import_summary_json << "\n";

        for (size_t i = 0; i < json_lines.size(); i++) {
            bool res_start = (res->status_code == 0) && (i == 0);

            if(res_start) {
                // indicates first import result to be streamed
                response_stream << json_lines[i];
            } else {
                response_stream << "\n" << json_lines[i];
            }
        }

        // Since we use `res->status_code == 0` for flagging `res_start`, we will only set this
        // when we have accumulated enough response data to stream.
        // Otherwise, we will send an empty line as first response.
        res->status_code = 200;
    }

    res->content_type_header = "text/plain; charset=utf-8";
    res->body = response_stream.str();

    res->final.store(req->last_chunk_aggregate);
    stream_response(req, res);

    return true;
}

bool post_add_document(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    const char *ACTION = "action";
    const char *DIRTY_VALUES_PARAM = "dirty_values";

    if(req->params.count(ACTION) == 0) {
        req->params[ACTION] = "create";
    }

    if(req->params[ACTION] != "create" && req->params[ACTION] != "update" && req->params[ACTION] != "upsert" &&
       req->params[ACTION] != "emplace") {
        res->set_400("Parameter `" + std::string(ACTION) + "` must be a create|update|upsert.");
        return false;
    }

    if(req->params.count(DIRTY_VALUES_PARAM) == 0) {
        req->params[DIRTY_VALUES_PARAM] = "";  // set it empty as default will depend on whether schema is enabled
    }


    CollectionManager & collectionManager = CollectionManager::get_instance();
    auto collection = collectionManager.get_collection(req->params["collection"]);

    if(collection == nullptr) {
        res->set_404();
        return false;
    }

    const index_operation_t operation = get_index_operation(req->params[ACTION]);
    const auto& dirty_values = collection->parse_dirty_values_option(req->params[DIRTY_VALUES_PARAM]);

    size_t remote_embedding_timeout_ms = 60000;
    size_t remote_embedding_num_tries = 2;

    if(req->params.count("remote_embedding_timeout_ms") != 0) {
        remote_embedding_timeout_ms = std::stoul(req->params["remote_embedding_timeout_ms"]);
    }

    if(req->params.count("remote_embedding_num_tries") != 0) {
        remote_embedding_num_tries = std::stoul(req->params["remote_embedding_num_tries"]);
    }

    nlohmann::json document;
    std::vector<std::string> json_lines = {req->body};
    const nlohmann::json& inserted_doc_op = collection->add_many(json_lines, document, operation, "", dirty_values,
                                                                 false, false, 200, remote_embedding_timeout_ms,
                                                                 remote_embedding_num_tries);

    if(!inserted_doc_op["success"].get<bool>()) {
        nlohmann::json res_doc;

        try {
            res_doc = nlohmann::json::parse(json_lines[0]);
        } catch(const std::exception& e) {
            LOG(ERROR) << "JSON error: " << e.what();
            res->set_400("Bad JSON.");
            return false;
        }

        res->status_code = res_doc["code"].get<size_t>();
        // erase keys from res_doc except error and embedding_error
        for(auto it = res_doc.begin(); it != res_doc.end(); ) {
            if(it.key() != "error" && it.key() != "embedding_error") {
                it = res_doc.erase(it);
            } else {
                ++it;
            }
        }

        // rename error to message if not empty and exists
        if(res_doc.count("error") != 0 && !res_doc["error"].get<std::string>().empty()) {
            res_doc["message"] = res_doc["error"];
            res_doc.erase("error");
        }

        res->body = res_doc.dump();
        return false;
    }

    res->set_201(document.dump(-1, ' ', false, nlohmann::detail::error_handler_t::ignore));
    return true;
}

bool patch_update_document(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    std::string doc_id = req->params["id"];

    CollectionManager & collectionManager = CollectionManager::get_instance();
    auto collection = collectionManager.get_collection(req->params["collection"]);

    if(collection == nullptr) {
        res->set_404();
        return false;
    }

    const char* DIRTY_VALUES_PARAM = "dirty_values";

    if(req->params.count(DIRTY_VALUES_PARAM) == 0) {
        req->params[DIRTY_VALUES_PARAM] = "";  // set it empty as default will depend on whether schema is enabled
    }

    const auto& dirty_values = collection->parse_dirty_values_option(req->params[DIRTY_VALUES_PARAM]);
    Option<nlohmann::json> upserted_doc_op = collection->add(req->body, index_operation_t::UPDATE, doc_id, dirty_values);

    if(!upserted_doc_op.ok()) {
        res->set(upserted_doc_op.code(), upserted_doc_op.error());
        return false;
    }

    res->set_200(upserted_doc_op.get().dump(-1, ' ', false, nlohmann::detail::error_handler_t::ignore));
    return true;
}

bool patch_update_documents(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    const char *FILTER_BY = "filter_by";
    std::string filter_query;
    if(req->params.count(FILTER_BY) == 0) {
        res->set_400("Parameter `" + std::string(FILTER_BY) + "` must be provided.");
        return false;
    } else {
        filter_query = req->params[FILTER_BY];
    }

    CollectionManager & collectionManager = CollectionManager::get_instance();
    auto collection = collectionManager.get_collection(req->params["collection"]);
    if(collection == nullptr) {
        res->set_404();
        return false;
    }

    const char* DIRTY_VALUES_PARAM = "dirty_values";
    if(req->params.count(DIRTY_VALUES_PARAM) == 0) {
        req->params[DIRTY_VALUES_PARAM] = "";  // set it empty as default will depend on whether schema is enabled
    }

    const char* VALIDATE_FIELD_NAMES = "validate_field_names";
    bool validate_field_names = true;
    if (req->params.count(VALIDATE_FIELD_NAMES) != 0 && req->params[VALIDATE_FIELD_NAMES] == "false") {
        validate_field_names = false;
    }

    search_stop_us = UINT64_MAX; // Filtering shouldn't timeout during update operation.
    auto update_op = collection->update_matching_filter(filter_query, req->body, req->params[DIRTY_VALUES_PARAM],
                                                        validate_field_names);
    if(update_op.ok()) {
        res->set_200(update_op.get().dump());
    } else {
        res->set(update_op.code(), update_op.error());
    }

    return update_op.ok();
}

bool get_fetch_document(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    std::string doc_id = req->params["id"];

    const char* INCLUDE_FIELDS = "include_fields";
    const char* EXCLUDE_FIELDS = "exclude_fields";

    spp::sparse_hash_set<std::string> exclude_fields;
    spp::sparse_hash_set<std::string> include_fields;

    CollectionManager & collectionManager = CollectionManager::get_instance();
    auto collection = collectionManager.get_collection(req->params["collection"]);
    if(collection == nullptr) {
        res->set_404();
        return false;
    }

    Option<nlohmann::json> doc_option = collection->get(doc_id);

    if(!doc_option.ok()) {
        res->set(doc_option.code(), doc_option.error());
        return false;
    }

    if(req->params.count(INCLUDE_FIELDS) != 0) {
        std::vector<std::string> include_fields_vec;
        StringUtils::split(req->params[INCLUDE_FIELDS], include_fields_vec, ",");
        include_fields = spp::sparse_hash_set<std::string>(include_fields_vec.begin(), include_fields_vec.end());
    }

    if(req->params.count(EXCLUDE_FIELDS) != 0) {
        std::vector<std::string> exclude_fields_vec;
        StringUtils::split(req->params[EXCLUDE_FIELDS], exclude_fields_vec, ",");
        exclude_fields = spp::sparse_hash_set<std::string>(exclude_fields_vec.begin(), exclude_fields_vec.end());
    }

    nlohmann::json doc = doc_option.get();

    for(auto it = doc.begin(); it != doc.end(); ++it) {
        if(!include_fields.empty() && include_fields.count(it.key()) == 0) {
            doc.erase(it.key());
            continue;
        }

        if(!exclude_fields.empty() && exclude_fields.count(it.key()) != 0) {
            doc.erase(it.key());
            continue;
        }
    }


    res->set_200(doc.dump(-1, ' ', false, nlohmann::detail::error_handler_t::ignore));
    return true;
}

bool del_remove_document(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    std::string doc_id = req->params["id"];

    bool ignore_not_found = false;
    if((req->params.count("ignore_not_found") != 0) && (req->params["ignore_not_found"] == "true")) {
        ignore_not_found = true;
    }

    CollectionManager & collectionManager = CollectionManager::get_instance();
    auto collection = collectionManager.get_collection(req->params["collection"]);
    if(collection == nullptr) {
        res->set_404();
        return false;
    }

    Option<nlohmann::json> doc_option = collection->get(doc_id);

    if (!doc_option.ok()) {
        if (ignore_not_found && doc_option.code() == 404) {
            nlohmann::json resp;
            resp["id"] = doc_id;
            res->set_200(resp.dump());
            return true;
        }

        res->set(doc_option.code(), doc_option.error());
        return false;
    }

    Option<std::string> deleted_id_op = collection->remove(doc_id);

    if (!deleted_id_op.ok()) {
        if (ignore_not_found && deleted_id_op.code() == 404) {
            nlohmann::json resp;
            resp["id"] = doc_id;
            res->set_200(resp.dump());
            return true;
        }

        res->set(deleted_id_op.code(), deleted_id_op.error());
        return false;
    }

    nlohmann::json doc = doc_option.get();
    res->set_200(doc.dump(-1, ' ', false, nlohmann::detail::error_handler_t::ignore));

    return true;
}

bool del_remove_documents(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    // defaults: will get overridden later if needed
    res->content_type_header = "application/json";
    res->status_code = 200;

    // NOTE: this is a streaming response end-point so this handler will be called multiple times
    CollectionManager & collectionManager = CollectionManager::get_instance();
    auto collection = collectionManager.get_collection(req->params["collection"]);

    if(collection == nullptr) {
        req->last_chunk_aggregate = true;
        res->final = true;
        res->set_404();
        stream_response(req, res);
        return false;
    }

    const char *BATCH_SIZE = "batch_size";
    const char *FILTER_BY = "filter_by";
    const char *TOP_K_BY = "top_k_by";
    const char* VALIDATE_FIELD_NAMES = "validate_field_names";
    const char* TRUNCATE = "truncate";

    if(req->params.count(TOP_K_BY) != 0) {
        std::vector<std::string> parts;
        StringUtils::split(req->params[TOP_K_BY], parts, ":");

        if(parts.size() != 2 || !StringUtils::is_uint32_t(parts[1])) {
            req->last_chunk_aggregate = true;
            res->final = true;
            res->set_400("The `top_k_by` parameter is not valid.");
            stream_response(req, res);
            return false;
        }

        const std::string& field_name = parts[0];
        const size_t k = std::stoull(parts[1]);
        auto op = collection->truncate_after_top_k(field_name, k);

        req->last_chunk_aggregate = true;
        res->final = true;

        if(!op.ok()) {
            res->set_500(op.error());
            stream_response(req, res);
            return false;
        }

        res->set_200(R"({"ok": true})");
        stream_response(req, res);
        return true;
    }

    if(req->params.count(BATCH_SIZE) == 0) {
        req->params[BATCH_SIZE] = "1000000000"; // 1 Billion
    }

    if(req->params.count(FILTER_BY) == 0) {
        req->last_chunk_aggregate = true;
        res->final = true;
        res->set_400("Parameter `" + std::string(FILTER_BY) + "` must be provided.");
        stream_response(req, res);
        return false;
    }

    if(!StringUtils::is_uint32_t(req->params[BATCH_SIZE])) {
        req->last_chunk_aggregate = true;
        res->final = true;
        res->set_400("Parameter `" + std::string(BATCH_SIZE) + "` must be a positive integer.");
        stream_response(req, res);
        return false;
    }

    const size_t DELETE_BATCH_SIZE = std::stoi(req->params[BATCH_SIZE]);

    if(DELETE_BATCH_SIZE == 0) {
        req->last_chunk_aggregate = true;
        res->final = true;
        res->set_400("Parameter `" + std::string(BATCH_SIZE) + "` must be a positive integer.");
        stream_response(req, res);
        return false;
    }

    std::string simple_filter_query;

    if(req->params.count(FILTER_BY) != 0) {
        simple_filter_query = req->params[FILTER_BY];
    }

    deletion_state_t* deletion_state = nullptr;
    nlohmann::json response;


    if(req->params.count(TRUNCATE) != 0 && req->params[TRUNCATE] == "true") {
        auto op = collection->remove_all_docs();

        if (!op.ok()) {
            res->set(op.code(), op.error());
        } else {
            response["num_deleted"] = op.get();
            res->body = response.dump();
        }

        req->last_chunk_aggregate = true;
        res->final = true;
    } else {
        if (req->data == nullptr) {
            deletion_state = new deletion_state_t{};
            // destruction of data is managed by req destructor
            req->data = deletion_state;

            bool validate_field_names = true;
            if (req->params.count(VALIDATE_FIELD_NAMES) != 0 && req->params[VALIDATE_FIELD_NAMES] == "false") {
                validate_field_names = false;
            }

            filter_result_t filter_result;
            auto filter_ids_op = collection->get_filter_ids(simple_filter_query, filter_result, false,
                                                            validate_field_names);

            if (!filter_ids_op.ok()) {
                res->set(filter_ids_op.code(), filter_ids_op.error());
                req->last_chunk_aggregate = true;
                res->final = true;
                stream_response(req, res);
                return false;
            }

            deletion_state->index_ids.emplace_back(filter_result.count, filter_result.docs);
            filter_result.docs = nullptr;

            for (size_t i = 0; i < deletion_state->index_ids.size(); i++) {
                deletion_state->offsets.push_back(0);
            }
            deletion_state->collection = collection.get();
            deletion_state->num_removed = 0;
        } else {
            deletion_state = dynamic_cast<deletion_state_t *>(req->data);
        }

        bool done = true;
        Option<bool> remove_op = stateful_remove_docs(deletion_state, DELETE_BATCH_SIZE, done);

        //LOG(INFO) << "Deletion batch size: " << DELETE_BATCH_SIZE << ", done: " << done;

        if (!remove_op.ok()) {
            res->set(remove_op.code(), remove_op.error());
            req->last_chunk_aggregate = true;
            res->final = true;
        } else {
            if (!done) {
                req->last_chunk_aggregate = false;
                res->final = false;
            } else {
                response["num_deleted"] = deletion_state->num_removed;

                req->last_chunk_aggregate = true;
                res->body = response.dump();
                res->final = true;
            }
        }
    }

    if(res->final) {
        stream_response(req, res);
    } else {
        defer_processing(req, res, 0);
    }

    return true;
}

bool get_aliases(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    CollectionManager & collectionManager = CollectionManager::get_instance();
    const spp::sparse_hash_map<std::string, std::string> & symlinks = collectionManager.get_symlinks();
    nlohmann::json res_json = nlohmann::json::object();
    res_json["aliases"] = nlohmann::json::array();

    for(const auto & symlink_collection: symlinks) {
        nlohmann::json symlink;
        symlink["name"] = symlink_collection.first;
        symlink["collection_name"] = symlink_collection.second;
        res_json["aliases"].push_back(symlink);
    }

    res->set_200(res_json.dump());
    return true;
}

bool get_alias(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    const std::string & alias = req->params["alias"];
    CollectionManager & collectionManager = CollectionManager::get_instance();
    Option<std::string> collection_name_op = collectionManager.resolve_symlink(alias);

    if(!collection_name_op.ok()) {
        res->set_404();
        return false;
    }

    nlohmann::json res_json;
    res_json["name"] = alias;
    res_json["collection_name"] = collection_name_op.get();

    res->set_200(res_json.dump());
    return true;
}

bool put_upsert_alias(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    nlohmann::json req_json;

    try {
        req_json = nlohmann::json::parse(req->body);
    } catch(const std::exception& e) {
        LOG(ERROR) << "JSON error: " << e.what();
        res->set_400("Bad JSON.");
        return false;
    }

    CollectionManager & collectionManager = CollectionManager::get_instance();
    const std::string & alias = req->params["alias"];

    const char* COLLECTION_NAME = "collection_name";

    if(req_json.count(COLLECTION_NAME) == 0) {
        res->set_400(std::string("Parameter `") + COLLECTION_NAME + "` is required.");
        return false;
    }

    Option<bool> success_op = collectionManager.upsert_symlink(alias, req_json[COLLECTION_NAME]);
    if(!success_op.ok()) {
        res->set_500(success_op.error());
        return false;
    }

    req_json["name"] = alias;
    res->set_200(req_json.dump());
    return true;
}

bool del_alias(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    const std::string & alias = req->params["alias"];
    CollectionManager & collectionManager = CollectionManager::get_instance();

    Option<std::string> collection_name_op = collectionManager.resolve_symlink(alias);
    if(!collection_name_op.ok()) {
        res->set_404();
        return false;
    }

    Option<bool> delete_op = collectionManager.delete_symlink(alias);

    if(!delete_op.ok()) {
        res->set_500(delete_op.error());
        return false;
    }

    nlohmann::json res_json;
    res_json["name"] = alias;
    res_json["collection_name"] = collection_name_op.get();
    res->set_200(res_json.dump());
    return true;
}

bool get_overrides(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    CollectionManager & collectionManager = CollectionManager::get_instance();
    auto collection = collectionManager.get_collection(req->params["collection"]);

    if(collection == nullptr) {
        res->set_404();
        return false;
    }

    uint32_t offset = 0, limit = 0;
    if(req->params.count("offset") != 0) {
        const auto &offset_str = req->params["offset"];
        if(!StringUtils::is_uint32_t(offset_str)) {
            res->set(400, "Offset param should be unsigned integer.");
            return false;
        }
        offset = std::stoi(offset_str);
    }

    if(req->params.count("limit") != 0) {
        const auto &limit_str = req->params["limit"];
        if(!StringUtils::is_uint32_t(limit_str)) {
            res->set(400, "Limit param should be unsigned integer.");
            return false;
        }
        limit = std::stoi(limit_str);
    }

    nlohmann::json res_json;
    res_json["overrides"] = nlohmann::json::array();

    auto overrides_op = collection->get_overrides(limit, offset);
    if(!overrides_op.ok()) {
        res->set(overrides_op.code(), overrides_op.error());
        return false;
    }

    const auto overrides = overrides_op.get();

    for(const auto &kv: overrides) {
        res_json["overrides"].push_back(kv.second->to_json());
    }

    res->set_200(res_json.dump());
    return true;
}

bool get_override(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    CollectionManager & collectionManager = CollectionManager::get_instance();
    auto collection = collectionManager.get_collection(req->params["collection"]);

    if(collection == nullptr) {
        res->set_404();
        return false;
    }

    std::string override_id = req->params["id"];

    auto overrides_op = collection->get_override(override_id);

    if(!overrides_op.ok()) {
        res->set(overrides_op.code(), overrides_op.error());
        return false;
    }

    res->set_200(overrides_op.get().to_json().dump());
    return true;
}

bool put_override(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    CollectionManager & collectionManager = CollectionManager::get_instance();
    auto collection = collectionManager.get_collection(req->params["collection"]);

    std::string override_id = req->params["id"];

    if(collection == nullptr) {
        res->set_404();
        return false;
    }

    nlohmann::json req_json;

    try {
        req_json = nlohmann::json::parse(req->body);
    } catch(const std::exception& e) {
        LOG(ERROR) << "JSON error: " << e.what();
        res->set_400("Bad JSON.");
        return false;
    }
    
    override_t override;
    Option<bool> parse_op = override_t::parse(req_json, override_id, override, "",
                                              collection->get_symbols_to_index(),
                                              collection->get_token_separators());
    if(!parse_op.ok()) {
        res->set(parse_op.code(), parse_op.error());
        return false;
    }
    
    Option<uint32_t> add_op = collection->add_override(override);

    if(!add_op.ok()) {
        res->set(add_op.code(), add_op.error());
        return false;
    }

    req_json["id"] = override.id;

    res->set_200(req_json.dump());
    return true;
}

bool del_override(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    CollectionManager & collectionManager = CollectionManager::get_instance();
    auto collection = collectionManager.get_collection(req->params["collection"]);

    if(collection == nullptr) {
        res->set_404();
        return false;
    }

    Option<uint32_t> rem_op = collection->remove_override(req->params["id"]);
    if(!rem_op.ok()) {
        res->set(rem_op.code(), rem_op.error());
        return false;
    }

    nlohmann::json res_json;
    res_json["id"] = req->params["id"];

    res->set_200(res_json.dump());
    return true;
}

bool get_keys(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    CollectionManager & collectionManager = CollectionManager::get_instance();
    AuthManager &auth_manager = collectionManager.getAuthManager();

    const Option<std::vector<api_key_t>>& keys_op = auth_manager.list_keys();
    if(!keys_op.ok()) {
        res->set(keys_op.code(), keys_op.error());
        return false;
    }

    nlohmann::json res_json;
    res_json["keys"] = nlohmann::json::array();

    const std::vector<api_key_t>& keys = keys_op.get();
    for(const auto & key: keys) {
        nlohmann::json key_obj = key.to_json();
        key_obj["value_prefix"] = key_obj["value"];
        key_obj.erase("value");
        res_json["keys"].push_back(key_obj);
    }

    res->set_200(res_json.dump());
    return true;
}

bool post_create_key(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    //LOG(INFO) << "post_create_key";

    CollectionManager & collectionManager = CollectionManager::get_instance();
    AuthManager &auth_manager = collectionManager.getAuthManager();

    nlohmann::json req_json;

    try {
        req_json = nlohmann::json::parse(req->body);
    } catch(const std::exception& e) {
        LOG(ERROR) << "JSON error: " << e.what();
        res->set_400("Bad JSON.");
        return false;
    }


    const Option<uint32_t>& validate_op = api_key_t::validate(req_json);
    if(!validate_op.ok()) {
        res->set(validate_op.code(), validate_op.error());
        return false;
    }

    if(req_json.count("expires_at") == 0) {
        req_json["expires_at"] = api_key_t::FAR_FUTURE_TIMESTAMP;
    }

    if(req_json.count("autodelete") == 0) {
        req_json["autodelete"] = false;
    }

    const std::string &rand_key = (req_json.count("value") != 0) ?
            req_json["value"].get<std::string>() : req->metadata;

    api_key_t api_key(
        rand_key,
        req_json["description"].get<std::string>(),
        req_json["actions"].get<std::vector<std::string>>(),
        req_json["collections"].get<std::vector<std::string>>(),
        req_json["expires_at"].get<uint64_t>(),
        req_json["autodelete"].get<bool>()
    );

    const Option<api_key_t>& api_key_op = auth_manager.create_key(api_key);
    if(!api_key_op.ok()) {
        res->set(api_key_op.code(), api_key_op.error());
        return false;
    }

    res->set_201(api_key_op.get().to_json().dump());
    return true;
}

bool get_key(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    CollectionManager & collectionManager = CollectionManager::get_instance();
    AuthManager &auth_manager = collectionManager.getAuthManager();

    const std::string& key_id_str = req->params["id"];
    uint32_t key_id = (uint32_t) std::stoul(key_id_str);

    const Option<api_key_t>& key_op = auth_manager.get_key(key_id);

    if(!key_op.ok()) {
        res->set(key_op.code(), key_op.error());
        return false;
    }

    nlohmann::json key_obj = key_op.get().to_json();
    key_obj["value_prefix"] = key_obj["value"];
    key_obj.erase("value");

    res->set_200(key_obj.dump());
    return true;
}

bool del_key(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    CollectionManager & collectionManager = CollectionManager::get_instance();
    AuthManager &auth_manager = collectionManager.getAuthManager();

    const std::string& key_id_str = req->params["id"];
    uint32_t key_id = (uint32_t) std::stoul(key_id_str);

    const Option<api_key_t> &del_op = auth_manager.remove_key(key_id);

    if(!del_op.ok()) {
        res->set(del_op.code(), del_op.error());
        return false;
    }

    nlohmann::json res_json;
    res_json["id"] = del_op.get().id;

    res->set_200(res_json.dump(-1, ' ', false, nlohmann::detail::error_handler_t::ignore));
    return true;
}

bool post_snapshot(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    const std::string SNAPSHOT_PATH = "snapshot_path";

    res->status_code = 201;
    res->content_type_header = "application/json";

    if(req->params.count(SNAPSHOT_PATH) == 0) {
        req->params[SNAPSHOT_PATH] = "";
    }

    server->do_snapshot(req->params[SNAPSHOT_PATH], req, res);

    return true;
}

bool post_vote(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    res->status_code = 200;
    res->content_type_header = "application/json";

    nlohmann::json response;
    response["success"] = server->trigger_vote();
    res->body = response.dump();

    return true;
}

bool post_config(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    nlohmann::json req_json;

    try {
        req_json = nlohmann::json::parse(req->body);
    } catch(const std::exception& e) {
        LOG(ERROR) << "JSON error: " << e.what();
        res->set_400("Bad JSON.");
        return false;
    }

    auto config_update_op = Config::get_instance().update_config(req_json);

    if(!config_update_op.ok()) {
        res->set(config_update_op.code(), config_update_op.error());
    } else {
        // for cache config, we have to resize the cache
        if(req_json.count("cache-num-entries") != 0) {
            std::unique_lock lock(mutex);
            res_cache.capacity(Config::get_instance().get_cache_num_entries());
        }
        nlohmann::json response;
        response["success"] = true;
        res->set_201(response.dump());
    }

    return true;
}

bool post_clear_cache(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    {
        std::unique_lock lock(mutex);
        res_cache.clear();
    }

    nlohmann::json response;
    response["success"] = true;
    res->set_200(response.dump());

    return true;
}

bool post_compact_db(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    CollectionManager& collectionManager = CollectionManager::get_instance();
    rocksdb::Status status = collectionManager.get_store()->compact_all();

    nlohmann::json response;
    response["success"] = status.ok();

    if(!status.ok()) {
        response["error"] = "Error code: " + std::to_string(status.code());
        res->set_500(response.dump());
    } else {
        res->set_200(response.dump());
    }

    return true;
}

bool post_reset_peers(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    res->status_code = 200;
    res->content_type_header = "application/json";

    nlohmann::json response;
    response["success"] = server->reset_peers();
    res->body = response.dump();

    return true;
}

bool get_schema_changes(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    CollectionManager & collectionManager = CollectionManager::get_instance();
    auto op = collectionManager.get_collection_alter_status();

    if(!op.ok()) {
        res->set(op.code(), op.error());
    }

    res->set_200(op.get().dump());

    return true;
}

bool get_synonyms(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    CollectionManager & collectionManager = CollectionManager::get_instance();
    auto collection = collectionManager.get_collection(req->params["collection"]);

    if(collection == nullptr) {
        res->set_404();
        return false;
    }

    uint32_t offset = 0, limit = 0;
    if(req->params.count("offset") != 0) {
        const auto &offset_str = req->params["offset"];
        if(!StringUtils::is_uint32_t(offset_str)) {
            res->set(400, "Offset param should be unsigned integer.");
            return false;
        }
        offset = std::stoi(offset_str);
    }

    if(req->params.count("limit") != 0) {
        const auto &limit_str = req->params["limit"];
        if(!StringUtils::is_uint32_t(limit_str)) {
            res->set(400, "Limit param should be unsigned integer.");
            return false;
        }
        limit = std::stoi(limit_str);
    }

    nlohmann::json res_json;
    res_json["synonyms"] = nlohmann::json::array();

    auto synonyms_op = collection->get_synonyms(limit, offset);
    if(!synonyms_op.ok()) {
        res->set(synonyms_op.code(), synonyms_op.error());
        return false;
    }

    const auto synonyms = synonyms_op.get();
    for(const auto & kv: synonyms) {
        res_json["synonyms"].push_back(kv.second->to_view_json());
    }

    res->set_200(res_json.dump());
    return true;
}

bool get_synonym(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    CollectionManager & collectionManager = CollectionManager::get_instance();
    auto collection = collectionManager.get_collection(req->params["collection"]);

    if(collection == nullptr) {
        res->set_404();
        return false;
    }

    std::string synonym_id = req->params["id"];

    synonym_t synonym;
    bool found = collection->get_synonym(synonym_id, synonym);

    if(found) {
        nlohmann::json synonym_json = synonym.to_view_json();
        res->set_200(synonym_json.dump());
        return true;
    }

    res->set_404();
    return false;
}

bool put_synonym(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    CollectionManager & collectionManager = CollectionManager::get_instance();
    auto collection = collectionManager.get_collection(req->params["collection"]);

    std::string synonym_id = req->params["id"];

    if(collection == nullptr) {
        res->set_404();
        return false;
    }

    nlohmann::json syn_json;

    try {
        syn_json = nlohmann::json::parse(req->body);
    } catch(const std::exception& e) {
        LOG(ERROR) << "JSON error: " << e.what();
        res->set_400("Bad JSON.");
        return false;
    }

    if(!syn_json.is_object()) {
        res->set_400("Bad JSON.");
        return false;
    }

    // These checks should be inside `add_synonym` but older versions of Typesense wrongly persisted
    // `root` as an array, so we have to do it here so that on-disk synonyms are loaded properly
    if(syn_json.count("root") != 0 && !syn_json["root"].is_string()) {
        res->set_400("Key `root` should be a string.");
        return false;
    }

    if(syn_json.count("synonyms") && syn_json["synonyms"].is_array()) {
        if(syn_json["synonyms"].empty()) {
            res->set_400("Could not find a valid string array of `synonyms`");
            return false;
        }

        for(const auto& synonym: syn_json["synonyms"]) {
            if (!synonym.is_string() || synonym.empty()) {
                res->set_400("Could not find a valid string array of `synonyms`");
                return false;
            }
        }
    }

    syn_json["id"] = synonym_id;
    Option<bool> upsert_op = collection->add_synonym(syn_json);

    if(!upsert_op.ok()) {
        res->set(upsert_op.code(), upsert_op.error());
        return false;
    }

    res->set_200(syn_json.dump());
    return true;
}

bool del_synonym(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    CollectionManager & collectionManager = CollectionManager::get_instance();
    auto collection = collectionManager.get_collection(req->params["collection"]);

    if(collection == nullptr) {
        res->set_404();
        return false;
    }

    Option<bool> rem_op = collection->remove_synonym(req->params["id"]);
    if(!rem_op.ok()) {
        res->set(rem_op.code(), rem_op.error());
        return false;
    }

    nlohmann::json res_json;
    res_json["id"] = req->params["id"];

    res->set_200(res_json.dump());
    return true;
}

bool is_doc_import_route(uint64_t route_hash) {
    route_path* rpath;
    bool found = server->get_route(route_hash, &rpath);
    return found && (rpath->handler == post_import_documents);
}

bool is_coll_create_route(uint64_t route_hash) {
    route_path* rpath;
    bool found = server->get_route(route_hash, &rpath);
    return found && (rpath->handler == post_create_collection);
}

bool is_drop_collection_route(uint64_t route_hash) {
    route_path* rpath;
    bool found = server->get_route(route_hash, &rpath);
    return found && (rpath->handler == del_drop_collection);
}

bool is_doc_write_route(uint64_t route_hash) {
    route_path* rpath;
    bool found = server->get_route(route_hash, &rpath);
    return found && (rpath->handler == post_add_document || rpath->handler == patch_update_document);
}

bool is_doc_del_route(uint64_t route_hash) {
    route_path* rpath;
    bool found = server->get_route(route_hash, &rpath);
    return found && (rpath->handler == del_remove_document || rpath->handler == del_remove_documents);
}

bool get_presets(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    CollectionManager & collectionManager = CollectionManager::get_instance();
    const spp::sparse_hash_map<std::string, nlohmann::json> & presets = collectionManager.get_presets();
    nlohmann::json res_json = nlohmann::json::object();
    res_json["presets"] = nlohmann::json::array();

    for(const auto& preset_kv: presets) {
        nlohmann::json preset;
        preset["name"] = preset_kv.first;
        preset["value"] = preset_kv.second;
        res_json["presets"].push_back(preset);
    }

    res->set_200(res_json.dump());
    return true;
}

bool get_preset(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    const std::string & preset_name = req->params["name"];
    CollectionManager & collectionManager = CollectionManager::get_instance();

    nlohmann::json preset;
    Option<bool> preset_op = collectionManager.get_preset(preset_name, preset);

    if(!preset_op.ok()) {
        res->set(preset_op.code(), preset_op.error());
        return false;
    }

    nlohmann::json res_json;
    res_json["name"] = preset_name;
    res_json["value"] = preset;

    res->set_200(res_json.dump());
    return true;
}

bool put_upsert_preset(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    nlohmann::json req_json;

    try {
        req_json = nlohmann::json::parse(req->body);
    } catch(const std::exception& e) {
        LOG(ERROR) << "JSON error: " << e.what();
        res->set_400("Bad JSON.");
        return false;
    }

    CollectionManager & collectionManager = CollectionManager::get_instance();
    const std::string & preset_name = req->params["name"];

    const char* PRESET_VALUE = "value";

    if(req_json.count(PRESET_VALUE) == 0) {
        res->set_400(std::string("Parameter `") + PRESET_VALUE + "` is required.");
        return false;
    }

    Option<bool> success_op = collectionManager.upsert_preset(preset_name, req_json[PRESET_VALUE]);
    if(!success_op.ok()) {
        res->set_500(success_op.error());
        return false;
    }

    req_json["name"] = preset_name;

    res->set_200(req_json.dump());
    return true;
}

bool del_preset(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    const std::string & preset_name = req->params["name"];
    CollectionManager & collectionManager = CollectionManager::get_instance();

    nlohmann::json preset;
    Option<bool> preset_op = collectionManager.get_preset(preset_name, preset);
    if(!preset_op.ok()) {
        res->set(preset_op.code(), preset_op.error());
        return false;
    }

    Option<bool> delete_op = collectionManager.delete_preset(preset_name);

    if(!delete_op.ok()) {
        res->set_500(delete_op.error());
        return false;
    }

    nlohmann::json res_json;
    res_json["name"] = preset_name;
    res_json["value"] = preset;
    res->set_200(res_json.dump());
    return true;
}

bool get_stopwords(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    StopwordsManager& stopwordManager = StopwordsManager::get_instance();
    const spp::sparse_hash_map<std::string, stopword_struct_t>& stopwords = stopwordManager.get_stopwords();
    nlohmann::json res_json = nlohmann::json::object();
    res_json["stopwords"] = nlohmann::json::array();

    for(const auto& stopwords_kv: stopwords) {
        auto stopword = stopwords_kv.second.to_json();
        res_json["stopwords"].push_back(stopword);
    }

    res->set_200(res_json.dump());
    return true;
}

bool get_stopword(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    const std::string & stopword_name = req->params["name"];
    StopwordsManager& stopwordManager = StopwordsManager::get_instance();

    const Option<stopword_struct_t>& stopword_op = stopwordManager.get_stopword(stopword_name);

    if(!stopword_op.ok()) {
        res->set(stopword_op.code(), stopword_op.error());
        return false;
    }

    nlohmann::json res_json;

    res_json["stopwords"] = stopword_op.get().to_json();

    res->set_200(res_json.dump());
    return true;
}

bool put_upsert_stopword(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    nlohmann::json req_json;

    try {
        req_json = nlohmann::json::parse(req->body);
    } catch(const std::exception& e) {
        LOG(ERROR) << "JSON error: " << e.what();
        res->set_400("Bad JSON.");
        return false;
    }

    StopwordsManager& stopwordManager = StopwordsManager::get_instance();
    const std::string & stopword_name = req->params["name"];

    Option<bool> success_op = stopwordManager.upsert_stopword(stopword_name, req_json, true);
    if(!success_op.ok()) {
        res->set(success_op.code(), success_op.error());
        return false;
    }

    req_json["id"] = stopword_name;

    res->set_200(req_json.dump());
    return true;
}

bool del_stopword(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    const std::string & stopword_name = req->params["name"];
    StopwordsManager& stopwordManager = StopwordsManager::get_instance();

    Option<bool> delete_op = stopwordManager.delete_stopword(stopword_name);

    if(!delete_op.ok()) {
        res->set(delete_op.code(), delete_op.error());
        return false;
    }

    nlohmann::json res_json;
    res_json["id"] = stopword_name;

    res->set_200(res_json.dump());
    return true;
}

bool get_rate_limits(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    RateLimitManager* rateLimitManager = RateLimitManager::getInstance();

    res->set_200(rateLimitManager->get_all_rules_json().dump());
    return true;
}

bool get_rate_limit(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    RateLimitManager* rateLimitManager = RateLimitManager::getInstance();
    // Convert param id to uint64_t
    if(!StringUtils::is_uint32_t(req->params["id"])) {
        res->set_400("{\"message\": \"Invalid ID\"}");
        return false;
    }
    uint64_t id = std::stoull(req->params["id"]);
    const auto& rule_option = rateLimitManager->find_rule_by_id(id);

    if(!rule_option.ok()) {
        res->set(rule_option.code(), rule_option.error());
        return false;
    }

    res->set_200(rule_option.get().dump());
    return true;
}

bool put_rate_limit(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    RateLimitManager* rateLimitManager = RateLimitManager::getInstance();
    nlohmann::json req_json;
    if(!StringUtils::is_uint32_t(req->params["id"])) {
        res->set_400("{\"message\": \"Invalid ID\"}");
        return false;
    }
    uint64_t id = std::stoull(req->params["id"]);

    try {
        req_json = nlohmann::json::parse(req->body);
    } catch(const nlohmann::json::parse_error& e) {
        res->set_400("Invalid JSON");
        return false;
    }

    const auto& edit_rule_result = rateLimitManager->edit_rule(id, req_json);

    if(!edit_rule_result.ok()) {
        res->set(edit_rule_result.code(), edit_rule_result.error());
        return false;
    }

    res->set_200(edit_rule_result.get().dump());
    return true;

}

bool del_rate_limit(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    RateLimitManager* rateLimitManager = RateLimitManager::getInstance();
    if(!StringUtils::is_uint32_t(req->params["id"])) {
        res->set_400("{\"message\": \"Invalid ID\"}");
        return false;
    }
    uint64_t id = std::stoull(req->params["id"]);
    const auto& rule_option = rateLimitManager->find_rule_by_id(id);

    if(!rule_option.ok()) {
        res->set(rule_option.code(), rule_option.error());
        return false;
    }

    rateLimitManager->delete_rule_by_id(id);
    nlohmann::json res_json;
    res_json["id"] = id;
    res->set_200(res_json.dump());
    return true;
}

bool post_rate_limit(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    RateLimitManager* rateLimitManager = RateLimitManager::getInstance();
    nlohmann::json req_json;

    try {
        req_json = nlohmann::json::parse(req->body);
    } catch (const std::exception & e) {
        res->set_400("Bad JSON.");
        return false;
    }

    auto add_rule_result = rateLimitManager->add_rule(req_json);

    if(!add_rule_result.ok()) {
        res->set(add_rule_result.code(), add_rule_result.error());
        return false;
    }

    res->set_200(add_rule_result.get().dump());
    return true;
}

bool get_active_throttles(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    RateLimitManager* rateLimitManager = RateLimitManager::getInstance();
    res->set_200(rateLimitManager->get_throttled_entities_json().dump());
    return true;
}

bool del_throttle(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    RateLimitManager* rateLimitManager = RateLimitManager::getInstance();
    if(!StringUtils::is_uint32_t(req->params["id"])) {
        res->set_400("{\"message\": \"Invalid ID\"}");
        return false;
    }
    uint64_t id = std::stoull(req->params["id"]);
    bool res_ = rateLimitManager->delete_ban_by_id(id);
    if(!res_) {
        res->set_400("{\"message\": \"Invalid ID\"}");
        return false;
    }
    nlohmann::json res_json;
    res_json["id"] = id;
    res->set_200(res_json.dump());
    return true;
}

bool del_exceed(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    RateLimitManager* rateLimitManager = RateLimitManager::getInstance();
    if(!StringUtils::is_uint32_t(req->params["id"])) {
        res->set_400("{\"message\": \"Invalid ID\"}");
        return false;
    }
    uint64_t id = std::stoull(req->params["id"]);
    bool res_ = rateLimitManager->delete_throttle_by_id(id);
    if(!res_) {
        res->set_400("{\"message\": \"Invalid ID\"}");
        return false;
    }
    nlohmann::json res_json;
    res_json["id"] = id;
    res->set_200(res_json.dump());
    return true;
}

bool get_limit_exceed_counts(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    RateLimitManager* rateLimitManager = RateLimitManager::getInstance();
    res->set_200(rateLimitManager->get_exceeded_entities_json().dump());
    return true;
}

Option<std::pair<std::string,std::string>> get_api_key_and_ip(const std::string& metadata) {
    // format <length of api_key>:<api_key><ip>
    // length of api_key is a uint32_t
    if(metadata.size() < 10) {
        if(metadata.size() >= 2 && metadata[0] == '0' && metadata[1] == ':') {
            // e.g. "0:0.0.0.0" (when api key is not provided at all)
            std::string ip = metadata.substr(metadata.find(":") + 1);
            return Option<std::pair<std::string,std::string>>(std::make_pair("", ip));
        }

        return Option<std::pair<std::string,std::string>>(400, "Invalid metadata");
    }

    if(metadata.find(":") == std::string::npos) {
        return Option<std::pair<std::string,std::string>>(400, "Invalid metadata");
    }

    std::string key_len_str = metadata.substr(0, metadata.find(":"));

    if(!StringUtils::is_uint32_t(key_len_str)) {
        return Option<std::pair<std::string,std::string>>(400, "Invalid metadata");
    }

    uint32_t api_key_length = static_cast<uint32_t>(std::stoul(key_len_str));

    if(metadata.size() < api_key_length + metadata.find(":") + 7) {
        return Option<std::pair<std::string,std::string>>(400, "Invalid metadata");
    }

    std::string api_key = metadata.substr(metadata.find(":") + 1, api_key_length);
    std::string ip = metadata.substr(metadata.find(":") + 1 + api_key_length);

    // validate IP address
    std::regex ip_pattern("\\b(([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\\.){3}([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\\b");
    if(!std::regex_match(ip, ip_pattern)) {
        return Option<std::pair<std::string,std::string>>(400, "Invalid metadata");
    }

    return Option<std::pair<std::string,std::string>>(std::make_pair(api_key, ip));
}

bool post_create_event(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    nlohmann::json req_json;

    try {
        req_json = nlohmann::json::parse(req->body);
    } catch(const std::exception& e) {
        LOG(ERROR) << "JSON error: " << e.what();
        res->set_400("Bad JSON.");
        return false;
    }

    auto add_event_op = EventManager::get_instance().add_event(req_json, req->client_ip);
    if(add_event_op.ok()) {
        res->set_201(R"({"ok": true})");
        return true;
    }

    res->set_400(add_event_op.error());
    return false;
}

bool get_analytics_rules(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    auto rules_op = AnalyticsManager::get_instance().list_rules();

    if(!rules_op.ok()) {
        res->set(rules_op.code(), rules_op.error());
        return false;
    }

    res->set_200(rules_op.get().dump());
    return true;
}

bool get_analytics_rule(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    auto rules_op = AnalyticsManager::get_instance().get_rule(req->params["name"]);

    if(!rules_op.ok()) {
        res->set(rules_op.code(), rules_op.error());
        return false;
    }

    res->set_200(rules_op.get().dump());
    return true;
}

bool post_create_analytics_rules(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    nlohmann::json req_json;

    try {
        req_json = nlohmann::json::parse(req->body);
    } catch(const std::exception& e) {
        LOG(ERROR) << "JSON error: " << e.what();
        res->set_400("Bad JSON.");
        return false;
    }

    auto op = AnalyticsManager::get_instance().create_rule(req_json, false, true);

    if(!op.ok()) {
        res->set(op.code(), op.error());
        return false;
    }

    res->set_201(req_json.dump());
    return true;
}

bool put_upsert_analytics_rules(const std::shared_ptr<http_req> &req, const std::shared_ptr<http_res> &res) {
    nlohmann::json req_json;

    try {
        req_json = nlohmann::json::parse(req->body);
    } catch(const std::exception& e) {
        LOG(ERROR) << "JSON error: " << e.what();
        res->set_400("Bad JSON.");
        return false;
    }

    req_json["name"] = req->params["name"];
    auto op = AnalyticsManager::get_instance().create_rule(req_json, true, true);

    if(!op.ok()) {
        res->set(op.code(), op.error());
        return false;
    }

    res->set_200(req_json.dump());
    return true;
}

bool del_analytics_rules(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    auto op = AnalyticsManager::get_instance().remove_rule(req->params["name"]);
    if(!op.ok()) {
        res->set(op.code(), op.error());
        return false;
    }

    nlohmann::json res_json;
    res_json["name"] = req->params["name"];

    res->set_200(res_json.dump());
    return true;
}

bool post_write_analytics_to_db(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    nlohmann::json req_json;

    try {
        req_json = nlohmann::json::parse(req->body);
    } catch(const std::exception& e) {
        LOG(ERROR) << "JSON error: " << e.what();
        res->set_400("Bad JSON.");
        return false;
    }

    if(!AnalyticsManager::get_instance().write_to_db(req_json)) {
        res->set_500(R"({"ok": false})");
        return false;
    }

    res->set_200(R"({"ok": true})");
    return true;
}

bool post_import_stemming_dictionary(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    const char *BATCH_SIZE = "batch_size";
    const char *ID = "id";

    if(req->params.count(BATCH_SIZE) == 0) {
        req->params[BATCH_SIZE] = "40";
    }

    if(req->params.count(ID) == 0) {
        res->final = true;
        res->set_400("Parameter `" + std::string(ID) + "` must be provided while importing dictionary words.");
        stream_response(req, res);
        return false;
    }

    if(!StringUtils::is_uint32_t(req->params[BATCH_SIZE])) {
        res->final = true;
        res->set_400("Parameter `" + std::string(BATCH_SIZE) + "` must be a positive integer.");
        stream_response(req, res);
        return false;
    }

    std::vector<std::string> json_lines;
    StringUtils::split(req->body, json_lines, "\n", false, false);

    if(req->last_chunk_aggregate) {
        //LOG(INFO) << "req->last_chunk_aggregate is true";
        req->body = "";
    }

    // When only one partial record arrives as a chunk, an empty body is pushed to response stream
    bool single_partial_record_body = (json_lines.empty() && !req->body.empty());
    std::stringstream response_stream;

    if(!single_partial_record_body) {
        auto op = StemmerManager::get_instance().upsert_stemming_dictionary(req->params.at(ID), json_lines);
        if(!op.ok()) {
            res->set(op.code(), op.error());
            stream_response(req, res);
        }

        for (size_t i = 0; i < json_lines.size(); i++) {
            bool res_start = (res->status_code == 0) && (i == 0);

            if(res_start) {
                // indicates first import result to be streamed
                response_stream << json_lines[i];
            } else {
                response_stream << "\n" << json_lines[i];
            }
        }

        // Since we use `res->status_code == 0` for flagging `res_start`, we will only set this
        // when we have accumulated enough response data to stream.
        // Otherwise, we will send an empty line as first response.
        res->status_code = 200;
    }

    res->content_type_header = "text/plain; charset=utf-8";
    res->body = response_stream.str();

    res->final.store(req->last_chunk_aggregate);
    stream_response(req, res);

    return true;
}

bool get_stemming_dictionaries(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    nlohmann::json dictionaries;
    StemmerManager::get_instance().get_stemming_dictionaries(dictionaries);

    res->set_200(dictionaries.dump());
    return true;
}

bool get_stemming_dictionary(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    const std::string& id = req->params["id"];
    nlohmann::json dictionary;

    if(!StemmerManager::get_instance().get_stemming_dictionary(id, dictionary)) {
        res->set_404();
        return false;
    }

    res->set_200(dictionary.dump());
    return true;
}

bool del_stemming_dictionary(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    const std::string& id = req->params["id"];
    nlohmann::json dictionary;

    auto delete_op = StemmerManager::get_instance().del_stemming_dictionary(id);

    if(!delete_op.ok()) {
        res->set(delete_op.code(), delete_op.error());
    }

    nlohmann::json res_json;
    res_json["id"] = id;
    res->set_200(res_json.dump());

    return true;
}

bool get_analytics_events(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    const char* N = "n";

    uint32_t n = 10;
    if(req->params.count(N) != 0 && !StringUtils::is_uint32_t(req->params[N])) {
        res->set_400("Parameter `n` must be a positive integer.");
        return false;
    }

    if (req->params.count(N)) {
        n = std::stoi(req->params[N]);
    }

    auto get_events_op = AnalyticsManager::get_instance().get_events(n);

    if(!get_events_op.ok()) {
        res->set(get_events_op.code(), get_events_op.error());
        return false;
    }
    nlohmann::json response = get_events_op.get();
    res->set_200(response.dump());
    return true;
}

bool post_proxy(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    HttpProxy& proxy = HttpProxy::get_instance();

    nlohmann::json req_json;

    try {
        req_json = nlohmann::json::parse(req->body);
    } catch(const nlohmann::json::parse_error& e) {
        LOG(ERROR) << "JSON error: " << e.what();
        res->set_400("Bad JSON.");
        return false;
    }

    std::string body, url, method;
    std::unordered_map<std::string, std::string> headers;

    if(req_json.count("url") == 0 || req_json.count("method") == 0) {
        res->set_400("Missing required fields.");
        return false;
    }

    if(!req_json["url"].is_string() || !req_json["method"].is_string() || req_json["url"].get<std::string>().empty() || req_json["method"].get<std::string>().empty()) {
        res->set_400("URL and method must be non-empty strings.");
        return false;
    }

    try {        
        if(req_json.count("body") != 0 && !req_json["body"].is_string()) {
            res->set_400("Body must be a string.");
            return false;
        }
        if(req_json.count("headers") != 0 && !req_json["headers"].is_object()) {
            res->set_400("Headers must be a JSON object.");
            return false;
        }
        if(req_json.count("body")) {
            body = req_json["body"].get<std::string>();
        }
        url = req_json["url"].get<std::string>();
        method = req_json["method"].get<std::string>();
        if(req_json.count("headers")) {
            headers = req_json["headers"].get<std::unordered_map<std::string, std::string>>();
        }
    } catch(const std::exception& e) {
        LOG(ERROR) << "JSON error: " << e.what();
        res->set_400("Bad JSON.");
        return false;
    }

    auto response = proxy.send(url, method, body, headers);
    
    if(response.status_code != 200) {
        int code = response.status_code;
        res->set_body(code, response.body);
        return false;
    }

    res->set_200(response.body);
    return true;
}


bool post_conversation_model(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    nlohmann::json model_json;

    try {
        model_json = nlohmann::json::parse(req->body);
    } catch(const nlohmann::json::parse_error& e) {
        LOG(ERROR) << "JSON error: " << e.what();
        res->set_400("Bad JSON.");
        return false;
    }

    if(!model_json.is_object()) {
        res->set_400("Bad JSON.");
        return false;
    }

    std::string model_id = req->metadata;

    auto add_model_op = ConversationModelManager::add_model(model_json, model_id, true);

    if(!add_model_op.ok()) {
        res->set(add_model_op.code(), add_model_op.error());
        return false;
    }

    Collection::hide_credential(model_json, "api_key");

    res->set_200(model_json.dump());
    return true;
}

bool get_conversation_model(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    const std::string& model_id = req->params["id"];

    auto model_op = ConversationModelManager::get_model(model_id);

    if(!model_op.ok()) {
        res->set(model_op.code(), model_op.error());
        return false;
    }

    auto model = model_op.get();
    Collection::hide_credential(model, "api_key");

    res->set_200(model.dump());
    return true;
}

bool get_conversation_models(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    auto models_op = ConversationModelManager::get_all_models();

    if(!models_op.ok()) {
        res->set(models_op.code(), models_op.error());
        return false;
    }

    auto models = models_op.get();

    for(auto& model: models) {
        Collection::hide_credential(model, "api_key");
    }

    res->set_200(models.dump());
    return true;
}

bool del_conversation_model(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    const std::string& model_id = req->params["id"];

    auto model_op = ConversationModelManager::delete_model(model_id);

    if(!model_op.ok()) {
        res->set(model_op.code(), model_op.error());
        return false;
    }

    auto model = model_op.get();

    Collection::hide_credential(model, "api_key");

    res->set_200(model.dump());
    return true;
}

bool put_conversation_model(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    const std::string& model_id = req->params["id"];

    nlohmann::json req_json;

    try {
        req_json = nlohmann::json::parse(req->body);
    } catch(const nlohmann::json::parse_error& e) {
        LOG(ERROR) << "JSON error: " << e.what();
        res->set_400("Bad JSON.");
        return false;
    }

    if(!req_json.is_object()) {
        res->set_400("Bad JSON.");
        return false;
    }

    auto model_op = ConversationModelManager::update_model(model_id, req_json);

    if(!model_op.ok()) {
        res->set(model_op.code(), model_op.error());
        return false;
    }

    auto model = model_op.get();

    Collection::hide_credential(model, "api_key");

    res->set_200(model.dump());
    return true;
}

bool post_personalization_model(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    nlohmann::json req_json;
    
    if (!req->params.count("name") || !req->params.count("collection") || !req->params.count("type")) {
        res->set_400("Missing required parameters 'name', 'collection' and 'type'.");
        return false;
    }

    req_json = {
        {"name", req->params["name"]},
        {"collection", req->params["collection"]},
        {"type", req->params["type"]}
    };

    std::string model_id;
    if (req->params.count("id")) {
        req_json["id"] = req->params["id"];
        model_id = req->params["id"];
    }

    const std::string model_data = req->body;
    auto create_op = PersonalizationModelManager::add_model(req_json, model_id, true, model_data);
    if(!create_op.ok()) {
        res->set(create_op.code(), create_op.error());
        return false;
    }
    
    auto model = create_op.get();
    res->set_200(nlohmann::json{{"ok", true}, {"model_id", model}}.dump());
    
    return true;
}

bool get_personalization_model(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    const std::string& model_id = req->params["id"];
    
    auto model_op = PersonalizationModelManager::get_model(model_id);
    if (!model_op.ok()) {
        res->set(model_op.code(), model_op.error());
        return false;
    }

    auto model = model_op.get();

    if (model.contains("model_path")) {
        model.erase("model_path");
    }
    res->set_200(model.dump());
    return true;
}

bool get_personalization_models(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    auto models_op = PersonalizationModelManager::get_all_models();
    if (!models_op.ok()) {
        res->set(models_op.code(), models_op.error());
        return false;
    }

    auto models = models_op.get();
    for (auto& model : models) {
        if (model.contains("model_path")) {
            model.erase("model_path");
        }
    }

    res->set_200(models.dump());
    return true;
}

bool del_personalization_model(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    const std::string& model_id = req->params["id"];
    
    auto delete_op = PersonalizationModelManager::delete_model(model_id);
    if (!delete_op.ok()) {
        res->set(delete_op.code(), delete_op.error());
        return false;
    }

    auto deleted_model = delete_op.get();
    if (deleted_model.contains("model_path")) {
        deleted_model.erase("model_path");
    }
    res->set_200(deleted_model.dump());
    return true;
}

bool put_personalization_model(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res) {
    nlohmann::json req_json;
    
    if (req->params.count("name") && !req->params["name"].empty()) {
        req_json["name"] = req->params["name"];
    }
    if (req->params.count("collection") && !req->params["collection"].empty()) {
        req_json["collection"] = req->params["collection"];
    }
    if (req->params.count("type") && !req->params["type"].empty()) {
        req_json["type"] = req->params["type"];
    }

    if (!req->params.count("id")) {
        res->set_400("Missing required parameter 'id'.");
        return false;
    }
    std::string model_id = req->params["id"];

    const std::string model_data = req->body;
    auto update_op = PersonalizationModelManager::update_model(model_id, req_json, model_data);
    if(!update_op.ok()) {
        res->set(update_op.code(), update_op.error());
        return false;
    }

    nlohmann::json response = update_op.get();
    if (response.contains("model_path")) {
        response.erase("model_path");
    }
    res->set_200(response.dump());
    return true;
}