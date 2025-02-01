#pragma once

#include "sparsepp.h"
#include "json.hpp"
#include "logger.h"
#include "tsconfig.h"
#include <mutex>
#include <string>
#include <shared_mutex>
#include <mutex>
#include <fstream>
#include "tdigest.h"

class AppMetrics {
private:
    mutable std::shared_mutex mutex;

    // stores last complete window
    spp::sparse_hash_map<std::string, uint64_t>* counts;
    spp::sparse_hash_map<std::string, TDigest>* durations;

    // stores the current window
    spp::sparse_hash_map<std::string, uint64_t>* current_counts;
    spp::sparse_hash_map<std::string, TDigest>* current_durations;

    std::string access_log_path;
    std::ofstream access_log;

    AppMetrics() {
        current_counts = new spp::sparse_hash_map<std::string, uint64_t>();
        counts = new spp::sparse_hash_map<std::string, uint64_t>();

        current_durations = new spp::sparse_hash_map<std::string, TDigest>();
        durations = new spp::sparse_hash_map<std::string, TDigest>();

        access_log_path = Config::get_instance().get_access_log_path();
        if(Config::get_instance().get_enable_access_logging() && !access_log_path.empty()) {
            access_log.open(access_log_path, std::ofstream::out | std::ofstream::app);
        }
    }

    ~AppMetrics() {
        delete current_counts;
        delete counts;

        delete current_durations;
        delete durations;
    }

public:
    static inline const std::string SEARCH_LABEL = "search";
    static inline const std::string DOC_WRITE_LABEL = "write";
    static inline const std::string IMPORT_LABEL = "import";
    static inline const std::string DOC_DELETE_LABEL = "delete";
    static inline const std::string OVERLOADED_LABEL = "overloaded";

    static const uint64_t METRICS_REFRESH_INTERVAL_MS = 10 * 1000;

    static AppMetrics & get_instance() {
        static AppMetrics instance;
        return instance;
    }

    AppMetrics(AppMetrics const&) = delete;
    void operator=(AppMetrics const&) = delete;

    void increment_count(const std::string& identifier, uint64_t count) {
        std::unique_lock lock(mutex);
        (*current_counts)[identifier] += count;
    }

    void increment_duration(const std::string& identifier, uint64_t duration) {
        std::unique_lock lock(mutex);
        (*current_durations)[identifier].add(duration);
    }

    void increment_write_metrics(uint64_t route_hash, uint64_t duration);

    void write_access_log(const uint64_t epoch_millis, const char* remote_ip, const std::string& path);

    void flush_access_log();

    void window_reset();

    void get(const std::string& rps_key, const std::string& latency_key, nlohmann::json &result) const;
};
