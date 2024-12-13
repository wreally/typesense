#include <mutex>
#include <thread>
#include "analytics_manager.h"
#include "tokenizer.h"
#include "http_client.h"
#include "collection_manager.h"
#include "string_utils.h"

#define EVENTS_RATE_LIMIT_SEC 60

Option<bool> AnalyticsManager::create_rule(nlohmann::json& payload, bool upsert, bool write_to_disk) {

    if(!payload.contains("type") || !payload["type"].is_string()) {
        return Option<bool>(400, "Request payload contains invalid type.");
    }

    if(!payload.contains("name") || !payload["name"].is_string()) {
        return Option<bool>(400, "Bad or missing name.");
    }

    if(!payload.contains("params") || !payload["params"].is_object()) {
        return Option<bool>(400, "Bad or missing params.");
    }

    if(payload["type"] == POPULAR_QUERIES_TYPE || payload["type"] == NOHITS_QUERIES_TYPE
        || payload["type"] == COUNTER_TYPE || payload["type"] == LOG_TYPE) {
        return create_index(payload, upsert, write_to_disk);
    }

    return Option<bool>(400, "Invalid type.");
}

Option<bool> AnalyticsManager::create_index(nlohmann::json &payload, bool upsert, bool write_to_disk) {
    // params and name are validated upstream
    const std::string& suggestion_config_name = payload["name"].get<std::string>();
    bool already_exists = suggestion_configs.find(suggestion_config_name) != suggestion_configs.end();

    if(!upsert && already_exists) {
        return Option<bool>(400, "There's already another configuration with the name `" +
                                 suggestion_config_name + "`.");
    }

    const auto& params = payload["params"];

    if(!params.contains("source") || !params["source"].is_object()) {
        return Option<bool>(400, "Bad or missing source.");
    }

    size_t limit = 1000;
    bool expand_query = false;
    bool enable_auto_aggregation = true;

    if(params.contains("limit") && params["limit"].is_number_integer()) {
        limit = params["limit"].get<size_t>();
    }

    if(params.contains("expand_query") && params["expand_query"].is_boolean()) {
        expand_query = params["expand_query"].get<bool>();
    }

    std::string counter_field;
    std::string destination_collection;
    std::vector<std::string> src_collections;

    suggestion_config_t suggestion_config;
    suggestion_config.name = suggestion_config_name;
    suggestion_config.limit = limit;
    suggestion_config.expand_query = expand_query;
    suggestion_config.rule_type = payload["type"];

    //for all types source collection is needed.
    if(!params["source"].contains("collections") || !params["source"]["collections"].is_array()) {
        return Option<bool>(400, "Must contain a valid list of source collections.");
    } else {
        for(const auto& coll: params["source"]["collections"]) {
            if (!coll.is_string()) {
                return Option<bool>(400, "Source collections value should be a string.");
            }
            auto collection = CollectionManager::get_instance().get_collection(coll.get<std::string>());
            if (collection == nullptr) {
                LOG(WARNING) << "Collection `" + coll.get<std::string>() + "` is not found for rule `" + suggestion_config_name + "`";
            }

            const std::string &src_collection = coll.get<std::string>();
            src_collections.push_back(src_collection);
            destination_collection = src_collection;
        }
    }

    if((payload["type"] == POPULAR_QUERIES_TYPE || payload["type"] == NOHITS_QUERIES_TYPE)
        && (params.contains("enable_auto_aggregation"))) {

        if(!params["enable_auto_aggregation"].is_boolean()) {
            return Option<bool>(400, "enable_auto_aggregation should be boolean.");
        }

        enable_auto_aggregation = params["enable_auto_aggregation"];
    }

    bool valid_events_found = params["source"].contains("events") && !params["source"]["events"].empty()
                                                     && params["source"]["events"].is_array()
                                                     && params["source"]["events"][0].is_object();

    if(valid_events_found) {
        suggestion_config.events = params["source"]["events"];
    } else if(payload["type"] == LOG_TYPE || payload["type"] == COUNTER_TYPE) {
        //events array is mandatory for LOG and COUNTER EVENTS
        return Option<bool>(400, "Bad or missing events.");
    }

    if(payload["type"] != LOG_TYPE) {
        if(!params.contains("destination") || !params["destination"].is_object()) {
            return Option<bool>(400, "Bad or missing destination.");
        }

        if(!params["destination"].contains("collection") || !params["destination"]["collection"].is_string()) {
            return Option<bool>(400, "Must contain a valid destination collection.");
        }

        if(params["destination"].contains("counter_field")) {
            if(!params["destination"]["counter_field"].is_string()) {
                return Option<bool>(400, "Must contain a valid counter_field.");
            }
            counter_field = params["destination"]["counter_field"].get<std::string>();
            suggestion_config.counter_field = counter_field;
        }

        destination_collection = params["destination"]["collection"].get<std::string>();
    }

    auto coll = CollectionManager::get_instance().get_collection(destination_collection);
    if (coll == nullptr) {
        LOG(WARNING) << "Collection `" + destination_collection + "` is not found for rule `" + suggestion_config_name + "`";
    }

    if(payload["type"] == POPULAR_QUERIES_TYPE) {
        if(!upsert && popular_queries.count(destination_collection) != 0) {
            return Option<bool>(400, "There's already another configuration for this destination collection.");
        }
    } else if(payload["type"] == NOHITS_QUERIES_TYPE) {
        if(!upsert && nohits_queries.count(destination_collection) != 0) {
            return Option<bool>(400, "There's already another configuration for this destination collection.");
        }
    } else if(payload["type"] == COUNTER_TYPE) {
        if(!upsert && counter_events.count(destination_collection) != 0) {
            return Option<bool>(400, "There's already another configuration for this destination collection.");
        }

        auto coll = CollectionManager::get_instance().get_collection(destination_collection).get();
        if(coll != nullptr) {
            if(!coll->contains_field(counter_field)) {
                return Option<bool>(404,
                                    "counter_field `" + counter_field + "` not found in destination collection.");
            }
        } else {
            LOG(WARNING) << "Collection `" + destination_collection + "` is not found for rule `" + suggestion_config_name + "`";
        }
    }

    std::unique_lock lock(mutex);

    if(already_exists) {
        // remove the previous configuration with same name (upsert)
        Option<bool> remove_op = remove_index(suggestion_config_name);
        if(!remove_op.ok()) {
            return Option<bool>(500, "Error erasing the existing configuration.");;
        }
    }

    if(query_collection_events.count(destination_collection) == 0) {
        std::vector<event_t> vec;
        query_collection_events.emplace(destination_collection, vec);
    }

    std::map<std::string, uint16_t> event_weight_map;
    bool log_to_store = payload["type"] == LOG_TYPE;

    for (const std::string coll: src_collections) {
        if(query_collection_events.count(coll) == 0) {
            std::vector<event_t> vec;
            query_collection_events.emplace(coll, vec);
        }
    }

    if(payload["type"] == POPULAR_QUERIES_TYPE) {
        QueryAnalytics* popularQueries = new QueryAnalytics(limit, enable_auto_aggregation);
        popularQueries->set_expand_query(suggestion_config.expand_query);
        popular_queries.emplace(destination_collection, popularQueries);
    } else if(payload["type"] == NOHITS_QUERIES_TYPE) {
        QueryAnalytics* noresultsQueries = new QueryAnalytics(limit, enable_auto_aggregation);
        nohits_queries.emplace(destination_collection, noresultsQueries);
    }

    if(valid_events_found) {
        for(const auto& event: params["source"]["events"]) {
            if(!event.contains("name") || event_collection_map.count(event["name"]) != 0) {
                return Option<bool>(400, "Events must contain a unique name.");
            }

            bool event_log_to_store = false;
            if(payload["type"] == COUNTER_TYPE) {
                if(!event.contains("weight") || !event["weight"].is_number()) {
                    return Option<bool>(400, "Counter events must contain a weight value.");
                }
                event_weight_map[event["name"]] = event["weight"];
            }

            if(event.contains("log_to_store")) {
                event_log_to_store = event["log_to_store"].get<bool>();
                if(event_log_to_store && !analytics_store) {
                    return Option<bool>(400, "Event can't be logged when analytics-db is not defined.");
                }
            }

            event_type_collection ec{event["type"], destination_collection, src_collections, event_log_to_store || log_to_store, suggestion_config_name};

            //keep pointer for /events API
            if(payload["type"] == POPULAR_QUERIES_TYPE) {
                ec.queries_ptr = popular_queries.at(destination_collection);
            } else if(payload["type"] == NOHITS_QUERIES_TYPE) {
                ec.queries_ptr = nohits_queries.at(destination_collection);
            }

            event_collection_map.emplace(event["name"], ec);
        }

        //store counter events data
        if(payload["type"] == COUNTER_TYPE) {
            counter_events.emplace(destination_collection, counter_event_t{counter_field, {}, event_weight_map});
        }
    }

    suggestion_config.destination_collection = destination_collection;
    suggestion_config.src_collections = src_collections;

    suggestion_configs.emplace(suggestion_config_name, suggestion_config);

    for(const auto& query_coll: suggestion_config.src_collections) {
        query_collection_mapping[query_coll].push_back(destination_collection);
    }

    if(write_to_disk) {
        auto suggestion_key = std::string(ANALYTICS_RULE_PREFIX) + "_" + suggestion_config_name;
        bool inserted = store->insert(suggestion_key, payload.dump());
        if(!inserted) {
            return Option<bool>(500, "Error while storing the config to disk.");
        }
    }

    return Option<bool>(true);
}

AnalyticsManager::~AnalyticsManager() {
    std::unique_lock lock(mutex);

    for(auto& kv: popular_queries) {
        delete kv.second;
    }

    for(auto& kv: nohits_queries) {
        delete kv.second;
    }
}

Option<nlohmann::json> AnalyticsManager::list_rules() {
    std::unique_lock lock(mutex);

    nlohmann::json rules = nlohmann::json::object();
    rules["rules"]= nlohmann::json::array();

    for(const auto& suggestion_config: suggestion_configs) {
        nlohmann::json rule;
        suggestion_config.second.to_json(rule);
        rules["rules"].push_back(rule);
    }

    return Option<nlohmann::json>(rules);
}

Option<nlohmann::json> AnalyticsManager::get_rule(const std::string& name) {
    nlohmann::json rule;
    std::unique_lock lock(mutex);

    auto suggestion_config_it = suggestion_configs.find(name);
    if(suggestion_config_it == suggestion_configs.end()) {
        return Option<nlohmann::json>(404, "Rule not found.");
    }

    suggestion_config_it->second.to_json(rule);
    return Option<nlohmann::json>(rule);
}

Option<bool> AnalyticsManager::remove_rule(const std::string &name) {
    std::unique_lock lock(mutex);

    auto suggestion_configs_it = suggestion_configs.find(name);
    if(suggestion_configs_it != suggestion_configs.end()) {
        return remove_index(name);
    }

    return Option<bool>(404, "Rule not found.");
}

Option<bool> AnalyticsManager::remove_all_rules() {
    std::unique_lock lock(mutex);

    std::vector<std::string> rules_list;
    //populate rules to delete later
    for(const auto& suggestion_config_it : suggestion_configs) {
        rules_list.emplace_back(suggestion_config_it.first);
    }

    for(const auto& rule : rules_list) {
        remove_index(rule);
    }

    return Option<bool>(true);
}

Option<bool> AnalyticsManager::remove_index(const std::string &name) {
    // lock is held by caller
    auto suggestion_configs_it = suggestion_configs.find(name);

    if(suggestion_configs_it == suggestion_configs.end()) {
        return Option<bool>(404, "Rule not found.");
    }

    const auto& suggestion_collection = suggestion_configs_it->second.destination_collection;

    for(const auto& query_collection: suggestion_configs_it->second.src_collections) {
        query_collection_mapping.erase(query_collection);
    }

    if(popular_queries.count(suggestion_collection) != 0) {
        delete popular_queries[suggestion_collection];
        popular_queries.erase(suggestion_collection);
    }

    if(nohits_queries.count(suggestion_collection) != 0) {
        delete nohits_queries[suggestion_collection];
        nohits_queries.erase(suggestion_collection);
    }

    if(counter_events.count(suggestion_collection) != 0) {
        counter_events.erase(suggestion_collection);
    }

    if(query_collection_events.count(suggestion_collection) != 0) {
        query_collection_events.erase(suggestion_collection);
    }

    suggestion_configs.erase(name);

    //remove corresponding events with rule
    for(auto it = event_collection_map.begin(); it != event_collection_map.end();) {
        if(it->second.analytic_rule == name) {
            event_collection_map.erase(it++);
        } else {
            ++it;
        }
    }

    auto suggestion_key = std::string(ANALYTICS_RULE_PREFIX) + "_" + name;
    bool erased = store->remove(suggestion_key);
    if(!erased) {
        return Option<bool>(500, "Error while deleting from disk.");
    }

    return Option<bool>(true);
}

void AnalyticsManager::add_suggestion(const std::string &query_collection,
                                      const std::string& query, const std::string& expanded_query,
                                      const bool live_query, const std::string& user_id) {
    // look up suggestion collections for the query collection
    std::unique_lock lock(mutex);
    const auto& suggestion_collections_it = query_collection_mapping.find(query_collection);
    if(suggestion_collections_it != query_collection_mapping.end()) {
        for(const auto& suggestion_collection: suggestion_collections_it->second) {
            const auto& popular_queries_it = popular_queries.find(suggestion_collection);
            if(popular_queries_it != popular_queries.end() && popular_queries_it->second->is_auto_aggregation_enabled()) {
                popular_queries_it->second->add(query, expanded_query, live_query, user_id);
            }
        }
    }
}

Option<bool> AnalyticsManager::add_event(const std::string& client_ip, const std::string& event_type,
                                         const std::string& event_name, const nlohmann::json& event_json) {
    std::unique_lock lock(mutex);

    const auto event_collection_map_it = event_collection_map.find(event_name);
    if(event_collection_map_it == event_collection_map.end()) {
        return Option<bool>(404, "No analytics rule defined for event name " + event_name);
    }

    if(event_collection_map_it->second.event_type != event_type) {
        return Option<bool>(400, "event_type mismatch in analytic rules.");
    }

    std::string destination_collection = event_collection_map_it->second.destination_collection;
    std::vector<std::string> src_collections = event_collection_map_it->second.src_collections;

    std::string src_collection;
    if (!event_json.contains("collection") && src_collections.size() == 1) {
        src_collection = src_collections[0];
    } else if(!event_json.contains("collection") && src_collections.size() > 1) {
        return Option<bool>(400, "Multiple source collections. 'collection' should be specified");
    } else if (event_json.contains("collection")) {
        if(std::find(src_collections.begin(), src_collections.end(), event_json["collection"]) == src_collections.end()) {
            return Option<bool>(400, event_json["collection"].get<std::string>() + " not found in the rule " + event_name);
        }
        src_collection = event_json["collection"];
    }

    const auto& query_collection_events_it = query_collection_events.find(src_collection);
    if(query_collection_events_it != query_collection_events.end()) {
        auto &events_vec = query_collection_events_it->second;
#ifdef TEST_BUILD
        if (isRateLimitEnabled) {
#endif
        auto now_ts_seconds = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
        auto events_cache_it = events_cache.find(client_ip);

        if (events_cache_it != events_cache.end()) {
            // event found in events cache
            if ((now_ts_seconds - events_cache_it->second.last_update_time) < EVENTS_RATE_LIMIT_SEC) {
                if (events_cache_it->second.count >= analytics_minute_rate_limit) {
                    return Option<bool>(500, "event rate limit reached.");
                } else {
                    events_cache_it->second.count++;
                }
            } else {
                events_cache_it->second.last_update_time = now_ts_seconds;
                events_cache_it->second.count = 1;
            }
        } else {
            event_cache_t eventCache{(uint64_t) now_ts_seconds, 1};
            events_cache.insert(client_ip, eventCache);
        }
#ifdef TEST_BUILD
        }
#endif
        auto now_ts_useconds = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();

        std::string query;
        std::string user_id;
        std::string doc_id;
        std::vector<std::pair<std::string, std::string>> custom_data;

        if(event_type == SEARCH_EVENT) {
            query = event_json["q"].get<std::string>();
            user_id = event_json["user_id"].get<std::string>();

            //add to respective popular queries/nohits queries
            if(event_collection_map_it->second.queries_ptr) {
                event_collection_map_it->second.queries_ptr->add(query, query, false, user_id);
            } else {
                return Option<bool>(500, "Error in /events endpoint for event " + event_name);
            }
        } else if (event_type == CUSTOM_EVENT) {
            for(auto itr = event_json.begin(); itr != event_json.end(); ++itr) {
                if (itr.key() == "query") {
                    query = itr.value().get<std::string>();
                } else if (itr.key()== "user_id") {
                    user_id = itr.value().get<std::string>();
                } else if (itr.key() == "doc_id") {
                    doc_id = itr.value().get<std::string>();
                } else {
                    auto kv = std::make_pair(itr.key(), itr.value().get<std::string>());
                    custom_data.push_back(kv);
                }
            }
        } else {
            query = event_json.contains("q") ? event_json["q"].get<std::string>() : "";
            user_id = event_json.contains("user_id") ? event_json["user_id"].get<std::string>() : "";
            doc_id = event_json["doc_id"].get<std::string>();
        }

        if(event_collection_map_it->second.log_to_store) {
            //only store events if log_to_store is specified in rule
            //remove any '%' found in userid
            user_id.erase(std::remove(user_id.begin(), user_id.end(), '%'), user_id.end());

            event_t event(query, event_type, now_ts_useconds, user_id, doc_id,
                          event_name, event_collection_map[event_name].log_to_store, custom_data);
            events_vec.emplace_back(event);
        }

        if (!counter_events.empty()) {
            auto counter_events_it = counter_events.find(destination_collection);
            if (counter_events_it != counter_events.end()) {
                auto event_weight_map_it = counter_events_it->second.event_weight_map.find(event_name);
                if (event_weight_map_it != counter_events_it->second.event_weight_map.end()) {
                    auto inc_val = event_weight_map_it->second;
                    counter_events_it->second.docid_counts[doc_id] += inc_val;
                } else {
                    LOG(ERROR) << "event_name " << event_name
                               << " not defined in analytic rule for counter events.";
                }
            } else {
                LOG(ERROR) << "collection " << destination_collection << " not found in analytics rule.";
            }
        }
    } else {
        return Option<bool>(500, "Failure in adding an event.");
    }
    return Option<bool>(true);
}

void AnalyticsManager::add_nohits_query(const std::string &query_collection, const std::string &query,
                                        bool live_query, const std::string &user_id) {
    // look up suggestion collections for the query collection
    std::unique_lock lock(mutex);
    const auto& suggestion_collections_it = query_collection_mapping.find(query_collection);
    if(suggestion_collections_it != query_collection_mapping.end()) {
        for(const auto& suggestion_collection: suggestion_collections_it->second) {
            const auto& noresults_queries_it = nohits_queries.find(suggestion_collection);
            if(noresults_queries_it != nohits_queries.end() && noresults_queries_it->second->is_auto_aggregation_enabled()) {
                noresults_queries_it->second->add(query, query, live_query, user_id);
            }
        }
    }
}

void AnalyticsManager::run(ReplicationState* raft_server) {
    uint64_t prev_persistence_s = std::chrono::duration_cast<std::chrono::seconds>(
                                    std::chrono::system_clock::now().time_since_epoch()).count();

    while(!quit) {
        std::unique_lock lk(mutex);
        cv.wait_for(lk, std::chrono::seconds(QUERY_COMPACTION_INTERVAL_S), [&] { return quit.load(); });

        //LOG(INFO) << "AnalyticsManager::run";

        if(quit) {
            lk.unlock();
            break;
        }

        auto now_ts_seconds = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();

        if(now_ts_seconds - prev_persistence_s < Config::get_instance().get_analytics_flush_interval()) {
            // we will persist aggregation every hour
            // LOG(INFO) << "QuerySuggestions::run interval is less, continuing";
            continue;
        }

        persist_query_events(raft_server, prev_persistence_s);
        persist_events(raft_server, prev_persistence_s);
        persist_popular_events(raft_server, prev_persistence_s);

        prev_persistence_s = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();

        lk.unlock();
    }

    dispose();
}

void AnalyticsManager::persist_query_events(ReplicationState *raft_server, uint64_t prev_persistence_s) {
    // lock is held by caller

    auto send_http_response = [&](QueryAnalytics* queryAnalyticsPtr,
            const std::string& import_payload, const std::string& suggestion_coll, const std::string& query_type) {
        // send http request
        std::string leader_url = raft_server->get_leader_url();
        if(!leader_url.empty()) {
            const std::string& base_url = leader_url + "collections/" + suggestion_coll;
            std::string res;

            const std::string& update_url = base_url + "/documents/import?action=emplace";
            std::map<std::string, std::string> res_headers;
            long status_code = HttpClient::post_response(update_url, import_payload,
                                                         res, res_headers, {}, 10*1000, true);

            if(status_code != 200) {
                LOG(ERROR) << "Error while sending "<< query_type <<" events to leader. "
                           << "Status code: " << status_code << ", response: " << res;
            } else {
                LOG(INFO) << "Query aggregation for collection: " + suggestion_coll;
                queryAnalyticsPtr->reset_local_counts();

                if(raft_server->is_leader()) {
                    // try to run top-K compaction of suggestion collection
                    const std::string top_k_param = "count:" + std::to_string(queryAnalyticsPtr->get_k());
                    const std::string& truncate_topk_url = base_url + "/documents?top_k_by=" + top_k_param;
                    res.clear();
                    res_headers.clear();
                    status_code = HttpClient::delete_response(truncate_topk_url, res, res_headers, 10*1000, true);
                    if(status_code != 200) {
                        LOG(ERROR) << "Error while running top K for " << query_type <<" suggestions collection. "
                                   << "Status code: " << status_code << ", response: " << res;
                    } else {
                        LOG(INFO) << "Top K aggregation for collection: " + suggestion_coll;
                    }
                }
            }
        }
    };


    for(const auto& suggestion_config: suggestion_configs) {
        const std::string& sink_name = suggestion_config.first;
        const std::string& suggestion_coll = suggestion_config.second.destination_collection;

        auto popular_queries_it = popular_queries.find(suggestion_coll);
        auto nohits_queries_it = nohits_queries.find(suggestion_coll);

        // need to prepare the counts as JSON docs for import into the suggestion collection
        // {"id": "432432", "q": "foo", "$operations": {"increment": {"count": 100}}}
        std::string import_payload;

        if(popular_queries_it != popular_queries.end()) {
            import_payload.clear();
            QueryAnalytics *popularQueries = popular_queries_it->second;

            // aggregate prefix queries to their final form
            auto now = std::chrono::system_clock::now().time_since_epoch();
            auto now_ts_us = std::chrono::duration_cast<std::chrono::microseconds>(now).count();
            popularQueries->compact_user_queries(now_ts_us);

            popularQueries->serialize_as_docs(import_payload);
            send_http_response(popularQueries, import_payload, suggestion_coll, "popular queries");
        }

        if(nohits_queries_it != nohits_queries.end()) {
            import_payload.clear();
            QueryAnalytics *nohitsQueries = nohits_queries_it->second;
            // aggregate prefix queries to their final form
            auto now = std::chrono::system_clock::now().time_since_epoch();
            auto now_ts_us = std::chrono::duration_cast<std::chrono::microseconds>(now).count();
            nohitsQueries->compact_user_queries(now_ts_us);

            nohitsQueries->serialize_as_docs(import_payload);
            send_http_response(nohitsQueries, import_payload, suggestion_coll, "nohits queries");
        }

        if(import_payload.empty()) {
            continue;
        }
    }
}

void AnalyticsManager::persist_events(ReplicationState *raft_server, uint64_t prev_persistence_s) {
    // lock is held by caller

    auto send_http_response = [&](const std::string& import_payload) {
        // send http request
        if(raft_server == nullptr) {
            return;
        }

        std::string leader_url = raft_server->get_leader_url();
        if(!leader_url.empty()) {
            const std::string& base_url = leader_url + "analytics/";
            std::string res;

            const std::string& update_url = base_url + "aggregate_events";
            std::map<std::string, std::string> res_headers;
            long status_code = HttpClient::post_response(update_url, import_payload,
                                                         res, res_headers, {}, 10*1000, true);

            if(status_code != 200) {
                LOG(ERROR) << "Error while sending "<<" log events to leader. "
                           << "Status code: " << status_code << ", response: " << res;
            }
        }
    };

    nlohmann::json payload = nlohmann::json::array();
    for (auto &events_collection_it: query_collection_events) {
        const auto& collection = events_collection_it.first;
        for (const auto &event: events_collection_it.second) {
            if (event.log_to_store) {
                nlohmann::json event_data;
                event.to_json(event_data, collection);
                payload.push_back(event_data);
            }
        }
        if(!payload.empty()) {
            send_http_response(payload.dump());
            events_collection_it.second.clear();
        }
    }
}

void AnalyticsManager::persist_popular_events(ReplicationState *raft_server, uint64_t prev_persistence_s) {
    auto send_http_response = [&](const std::string& import_payload, const std::string& collection) {
        if (raft_server == nullptr) {
            return;
        }

        std::string leader_url = raft_server->get_leader_url();
        if (!leader_url.empty()) {
            const std::string &base_url = leader_url + "collections/" + collection;
            std::string res;

            const std::string &update_url = base_url + "/documents/import?action=update";
            std::map<std::string, std::string> res_headers;
            long status_code = HttpClient::post_response(update_url, import_payload,
                                                         res, res_headers, {}, 10 * 1000, true);

            if (status_code != 200) {
                LOG(ERROR) << "Error while sending popular_clicks events to leader. "
                           << "Status code: " << status_code << ", response: " << res;
            }
        }
    };

    for(auto& counter_event_it : counter_events) {
        auto coll = counter_event_it.first;
        std::string docs;
        counter_event_it.second.serialize_as_docs(docs);
        send_http_response(docs, coll);
        counter_event_it.second.docid_counts.clear();
    }
}

void AnalyticsManager::stop() {
    quit = true;
    dispose();
    cv.notify_all();
}

void AnalyticsManager::dispose() {
    std::unique_lock lk(mutex);

    for(auto& kv: popular_queries) {
        delete kv.second;
    }

    popular_queries.clear();

    for(auto& kv: nohits_queries) {
        delete kv.second;
    }

    nohits_queries.clear();

    suggestion_configs.clear();

    query_collection_mapping.clear();

    counter_events.clear();

    query_collection_events.clear();

    event_collection_map.clear();

    events_cache.clear();
}

void AnalyticsManager::init(Store* store, Store* analytics_store, uint32_t analytics_minute_rate_limit) {
    this->store = store;
    this->analytics_store = analytics_store;
    this->analytics_minute_rate_limit = analytics_minute_rate_limit;

    if(analytics_store) {
        events_cache.capacity(1024);
    }
}

std::unordered_map<std::string, QueryAnalytics*> AnalyticsManager::get_popular_queries() {
    std::unique_lock lk(mutex);
    return popular_queries;
}

std::unordered_map<std::string, QueryAnalytics*> AnalyticsManager::get_nohits_queries() {
    std::unique_lock lk(mutex);
    return nohits_queries;
}

std::unordered_map<std::string, counter_event_t> AnalyticsManager::get_popular_clicks() {
    std::unique_lock lk(mutex);
    return counter_events;
}

void AnalyticsManager::resetToggleRateLimit(bool toggle) {
    std::unique_lock lk(mutex);
    events_cache.clear();
    isRateLimitEnabled = toggle;
}

void counter_event_t::serialize_as_docs(std::string &docs) {
    for (auto kv: docid_counts) {
        nlohmann::json doc;
        doc["id"] = kv.first;
        doc["$operations"]["increment"][counter_field] = kv.second;
        docs += doc.dump(-1, ' ', false, nlohmann::detail::error_handler_t::ignore) + "\n";
    }

    if (!docs.empty()) {
        docs.pop_back();
    }
}

bool AnalyticsManager::write_to_db(const nlohmann::json& payload) {
    if(analytics_store) {
        for(const auto& event: payload) {
            std::string userid = event["user_id"].get<std::string>();
            std::string event_name = event["name"].get<std::string>();
            std::string ts = StringUtils::serialize_uint64_t(event["timestamp"].get<uint64_t>());

            std::string key =  userid + "%" + event_name + "%" + ts;

            bool inserted = analytics_store->insert(key, event.dump());
            if(!inserted) {
                LOG(ERROR) << "Error while dumping events to analytics db.";
                return false;
            }
        }
    } else {
        LOG(ERROR) << "Analytics DB not initialized!!";
        return false;
    }

    return true;
}

void AnalyticsManager::get_last_N_events(const std::string& userid, const std::string& event_name, uint32_t N,
                                            std::vector<std::string>& values) {
    std::string user_id = userid;

    //erase any '%' in userid
    user_id.erase(std::remove(user_id.begin(), user_id.end(), '%'), user_id.end());

    auto userid_prefix = user_id + "%";
    if(event_name != "*") {
        userid_prefix += event_name;
    }

    analytics_store->get_last_N_values(userid_prefix, N, values);
}

Option<nlohmann::json> AnalyticsManager::get_events(uint32_t N) {
    std::vector<std::string> values;
    if (N > 1000) {
        return Option<nlohmann::json>(400, "N cannot be greater than 1000");
    }

    std::string userid_prefix;
    analytics_store->get_last_N_values(userid_prefix, N, values);
    nlohmann::json response;
    response["events"] = nlohmann::json::array();
    for(const auto& event: values) {
        response["events"].push_back(nlohmann::json::parse(event));
    }
    return Option<nlohmann::json>(response);
}

void event_t::to_json(nlohmann::json& obj, const std::string& coll) const {
    obj["query"] = query;
    obj["type"] = event_type;
    obj["timestamp"] = timestamp;
    obj["user_id"] = user_id;
    obj["doc_id"] = doc_id;
    obj["name"] = name;
    obj["collection"] = coll;

    if(event_type == "custom") {
        for(const auto& kv : data) {
            obj[kv.first] = kv.second;
        }
    }
}

Option<event_type_collection> AnalyticsManager::get_event_details(const std::string& event_name) {
    const auto event_it = event_collection_map.find(event_name);
    if(event_it == event_collection_map.end()) {
        return Option<event_type_collection>(404, "Event name not found");
    }

    return Option<event_type_collection>(event_it->second);
}
