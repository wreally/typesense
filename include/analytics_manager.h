#pragma once
#include "query_analytics.h"
#include "option.h"
#include "raft_server.h"
#include <vector>
#include <string>
#include <unordered_map>
#include <shared_mutex>
#include "lru/lru.hpp"

struct event_type_collection {
    std::string event_type;
    std::string destination_collection;
    std::vector<std::string> src_collections;
    bool log_to_store = false;
    std::string analytic_rule;
    QueryAnalytics* queries_ptr = nullptr;
};

struct event_t {
    std::string query;
    std::string event_type;
    uint64_t timestamp;
    std::string user_id;
    std::string doc_id;
    std::string name;
    std::vector<std::pair<std::string, std::string>> data;
    bool log_to_store;

    event_t() = delete;

    ~event_t() = default;

    event_t(const std::string& q, const std::string& type, uint64_t ts, const std::string& uid, const std::string& id,
            const std::string& event_name, bool should_log_to_store, const std::vector<std::pair<std::string, std::string>> datavec) {
        query = q;
        event_type = type;
        timestamp = ts;
        user_id = uid;
        doc_id = id;
        name = event_name;
        log_to_store = should_log_to_store;
        data = datavec;
    }

    event_t& operator=(event_t& other) {
        if (this != &other) {
            query = other.query;
            event_type = other.event_type;
            timestamp = other.timestamp;
            user_id = other.user_id;
            doc_id = other.doc_id;
            name = other.name;
            data = other.data;
            return *this;
        }
    }

    void to_json(nlohmann::json& obj, const std::string& coll) const;
};

struct counter_event_t {
    std::string counter_field;
    std::map<std::string, uint64_t> docid_counts;
    std::map<std::string, uint16_t> event_weight_map;

    void serialize_as_docs(std::string& docs);
};

struct event_cache_t {
    uint64_t last_update_time;
    uint64_t count;

    bool operator == (const event_cache_t& res) const {
        return last_update_time == res.last_update_time;
    }

    bool operator != (const event_cache_t& res) const {
        return last_update_time != res.last_update_time;
    }
};

class AnalyticsManager {
private:
    mutable std::mutex mutex;
    std::condition_variable cv;
    const size_t QUERY_COMPACTION_INTERVAL_S = 30;

    std::atomic<bool> quit = false;
    struct suggestion_config_t {
        std::string name;
        std::string destination_collection;
        std::vector<std::string> src_collections;
        size_t limit;
        std::string rule_type;
        bool expand_query = false;
        nlohmann::json events;
        std::string counter_field;

        void to_json(nlohmann::json& obj) const {
            obj["name"] = name;
            obj["type"] = rule_type;
            obj["params"] = nlohmann::json::object();
            obj["params"]["limit"] = limit;
            obj["params"]["source"]["collections"] = src_collections;
            obj["params"]["destination"]["collection"] = destination_collection;

            if(rule_type == POPULAR_QUERIES_TYPE) {
                obj["params"]["expand_query"] = expand_query;
            }

            if(!events.empty()) {
                obj["params"]["source"]["events"] = events;
                obj["params"]["destination"]["counter_field"] = counter_field;
            }
        }
    };

    // config name => config
    std::unordered_map<std::string, suggestion_config_t> suggestion_configs;

    // query collection => suggestion collections
    std::unordered_map<std::string, std::vector<std::string>> query_collection_mapping;

    // suggestion collection => popular queries
    std::unordered_map<std::string, QueryAnalytics*> popular_queries;

    // suggestion collection => nohits queries
    std::unordered_map<std::string, QueryAnalytics*> nohits_queries;

    // collection => popular clicks
    std::unordered_map<std::string, counter_event_t> counter_events;

    //query collection => events
    std::unordered_map<std::string, std::vector<event_t>> query_collection_events;

    //event_name  => collection
    std::unordered_map<std::string, event_type_collection> event_collection_map;

    // per_ip cache for rate limiting
    LRU::Cache<std::string, event_cache_t> events_cache;

    Store* store = nullptr;
    Store* analytics_store = nullptr;

    bool isRateLimitEnabled = true;

    uint32_t analytics_minute_rate_limit;

    AnalyticsManager() {}

    ~AnalyticsManager();

    Option<bool> remove_index(const std::string& name);

    Option<bool> create_index(nlohmann::json &payload,
                              bool upsert,
                              bool write_to_disk);

public:

    static constexpr const char* ANALYTICS_RULE_PREFIX = "$AR";
    static constexpr const char* POPULAR_QUERIES_TYPE = "popular_queries";
    static constexpr const char* NOHITS_QUERIES_TYPE = "nohits_queries";
    static constexpr const char* COUNTER_TYPE = "counter";
    static constexpr const char* LOG_TYPE = "log";
    static constexpr const char* CLICK_EVENT = "click";
    static constexpr const char* CONVERSION_EVENT = "conversion";
    static constexpr const char* VISIT_EVENT = "visit";
    static constexpr const char* CUSTOM_EVENT = "custom";
    static constexpr const char* SEARCH_EVENT = "search";

    static AnalyticsManager& get_instance() {
        static AnalyticsManager instance;
        return instance;
    }

    AnalyticsManager(AnalyticsManager const&) = delete;
    void operator=(AnalyticsManager const&) = delete;

    void init(Store* store, Store* analytics_store, uint32_t analytics_minute_rate_limit);

    void run(ReplicationState* raft_server);

    Option<nlohmann::json> list_rules();

    Option<nlohmann::json> get_rule(const std::string& name);

    Option<bool> create_rule(nlohmann::json& payload, bool upsert, bool write_to_disk);

    Option<bool> remove_rule(const std::string& name);

    Option<bool> remove_all_rules();

    void add_suggestion(const std::string& query_collection,
                        const std::string& query, const std::string& expanded_query,
                        bool live_query, const std::string& user_id);

    void stop();

    void dispose();

    void persist_query_events(ReplicationState *raft_server, uint64_t prev_persistence_s);

    std::unordered_map<std::string, QueryAnalytics*> get_popular_queries();

    Option<bool> add_event(const std::string& client_ip, const std::string& event_type,
                           const std::string& event_name, const nlohmann::json& event_data);

    void persist_events(ReplicationState *raft_server, uint64_t prev_persistence_s);

    void persist_popular_events(ReplicationState *raft_server, uint64_t prev_persistence_s);

    std::unordered_map<std::string, counter_event_t> get_popular_clicks();

    void add_nohits_query(const std::string& query_collection,
                          const std::string& query, bool live_query, const std::string& user_id);

    std::unordered_map<std::string, QueryAnalytics*> get_nohits_queries();

    void resetToggleRateLimit(bool toggle);

    bool write_to_db(const nlohmann::json& payload);

    void get_last_N_events(const std::string& userid, const std::string& event_name, uint32_t N, std::vector<std::string>& values);

    Option<nlohmann::json> get_events(uint32_t N);

    Option<event_type_collection> get_event_details(const std::string& event_name);

#ifdef TEST_BUILD
    std::unordered_map<std::string, std::vector<event_t>> get_log_events() {
        return query_collection_events;
    }
#endif
};
