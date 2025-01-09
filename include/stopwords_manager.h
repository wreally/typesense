#pragma once

#include "sparsepp.h"
#include "option.h"
#include "json.hpp"
#include "shared_mutex"
#include "mutex"
#include "store.h"

struct stopword_struct_t {
    std::string id;
    spp::sparse_hash_set<std::string> stopwords;
    spp::sparse_hash_set<std::string> stopwords_phrases;
    std::string locale;

    nlohmann::json to_json() const {
        nlohmann::json doc;

        doc["id"] = id;
        if(!locale.empty()) {
            doc["locale"] = locale;
        }

        for(const auto& stopword : stopwords) {
            doc["stopwords"].push_back(stopword);
        }

        for(const auto& stopword : stopwords_phrases) {
            doc["stopwords"].push_back(stopword);
        }

        return doc;
    }
};

class StopwordsManager{
private:
    spp::sparse_hash_map<std::string, stopword_struct_t> stopword_configs;

    static std::string get_stopword_key(const std::string & stopword_name);

    mutable std::shared_mutex mutex;

    Store* store = nullptr;
public:
    StopwordsManager() = default;

    ~StopwordsManager() {
        stopword_configs.clear();
    }

    static constexpr const char* STOPWORD_PREFIX = "$SW";

    static StopwordsManager& get_instance() {
        static StopwordsManager instance;
        return instance;
    }

    void init(Store* store);

    const spp::sparse_hash_map<std::string, stopword_struct_t>& get_stopwords() const;

    Option<stopword_struct_t> get_stopword(const std::string&) const;

    Option<bool> upsert_stopword(const std::string&, const nlohmann::json&, bool write_to_store=false);

    Option<bool> delete_stopword(const std::string&);

    void dispose();

    bool stopword_set_exists(const std::string &stopword);

    bool is_stopword(const std::string& stopword_set, const std::string& token);

    void process_stopwords(const std::string& stopword_set, std::vector<std::string>& tokens);
};