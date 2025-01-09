#include "include/stopwords_manager.h"
#include "include/tokenizer.h"

void StopwordsManager::init(Store* _store) {
    store = _store;
}

const spp::sparse_hash_map<std::string, stopword_struct_t>& StopwordsManager::get_stopwords() const {
    std::shared_lock lock(mutex);
    return stopword_configs;
}

Option<stopword_struct_t> StopwordsManager::get_stopword(const std::string& stopword_name) const {
    std::shared_lock lock(mutex);

    const auto& it = stopword_configs.find(stopword_name);
    if(it != stopword_configs.end()) {
        return Option<stopword_struct_t>(it->second);
    }

    return Option<stopword_struct_t>(404, "Stopword `" + stopword_name +"` not found.");
}

Option<bool> StopwordsManager::upsert_stopword(const std::string& stopword_name, const nlohmann::json& stopwords_json,
                                               bool write_to_store) {
    std::unique_lock lock(mutex);

    const char* STOPWORD_VALUES = "stopwords";
    const char* STOPWORD_LOCALE = "locale";
    std::string locale  = "";

    if(stopwords_json.count(STOPWORD_VALUES) == 0){
        return Option<bool>(400, (std::string("Parameter `") + STOPWORD_VALUES + "` is required"));
    }

    if(stopwords_json[STOPWORD_VALUES].empty()) {
        return Option<bool>(400, (std::string("Parameter `") + STOPWORD_VALUES + "` is empty"));
    }

    if((!stopwords_json[STOPWORD_VALUES].is_array()) || (!stopwords_json[STOPWORD_VALUES][0].is_string())) {
        return Option<bool>(400, (std::string("Parameter `") + STOPWORD_VALUES + "` is required as string array value"));
    }

    if(stopwords_json.count(STOPWORD_LOCALE) != 0) {
        if (!stopwords_json[STOPWORD_LOCALE].is_string()) {
            return Option<bool>(400, (std::string("Parameter `") + STOPWORD_LOCALE + "` is required as string value"));
        }
        locale = stopwords_json[STOPWORD_LOCALE];
    }

    if(write_to_store) {
        bool inserted = store->insert(get_stopword_key(stopword_name), stopwords_json.dump());
        if (!inserted) {
            return Option<bool>(500, "Unable to insert into store.");
        }
    }

    std::vector<std::string> tokens;
    spp::sparse_hash_set<std::string> stopwords_set;
    spp::sparse_hash_set<std::string> stopwords_phrases;
    const auto& stopwords = stopwords_json[STOPWORD_VALUES];
    std::vector<char> custom_symbols;

    for (const auto &stopword: stopwords.items()) {
        const auto& val = stopword.value().get<std::string>();

        //if first and last char is double quote, should add ' ' as custom symbol to tokenize whole string
        if (val[0] == '\"' && val[val.size() - 1] == '\"') {
            custom_symbols.push_back(' ');
        }

        Tokenizer(val, true, false, locale, custom_symbols, {}).tokenize(tokens);

        if(custom_symbols.empty()) { //store indvidual stopwords and stopword phrases separately
            for (const auto &tok: tokens) {
                stopwords_set.emplace(tok);
            }
        } else {
            for (const auto &tok: tokens) {
                stopwords_phrases.emplace(tok);
            }
        }
        tokens.clear();
        custom_symbols.clear();
    }
    stopword_configs[stopword_name] = stopword_struct_t{stopword_name, stopwords_set, stopwords_phrases, locale};
    return Option<bool>(true);
}

std::string StopwordsManager::get_stopword_key(const std::string& stopword_name) {
    return std::string(STOPWORD_PREFIX) + "_" + stopword_name;
}

Option<bool> StopwordsManager::delete_stopword(const std::string& stopword_name) {
    std::unique_lock lock(mutex);

    if(stopword_configs.find(stopword_name) == stopword_configs.end()) {
        return Option<bool>(404, "Stopword `" + stopword_name + "` not found.");
    }

    stopword_configs.erase(stopword_name);

    bool removed = store->remove(get_stopword_key(stopword_name));
    if(!removed) {
        return Option<bool>(500, "Unable to delete from store.");
    }

    return Option<bool>(true);
}

void StopwordsManager::dispose() {
    std::unique_lock lock(mutex);
    stopword_configs.clear();
}

bool StopwordsManager::stopword_set_exists(const std::string &stopword) {
    std::shared_lock lock(mutex);

    return stopword_configs.find(stopword) != stopword_configs.end();
}

bool StopwordsManager::is_stopword(const std::string &stopword_set, const std::string &token) {
    if(stopword_set_exists(stopword_set)) {
        const auto& stopwords = stopword_configs.at(stopword_set).stopwords;
        const auto& stopword_phrases = stopword_configs.at(stopword_set).stopwords_phrases;

        if(stopwords.find(token) != stopwords.end() || stopword_phrases.find(token) != stopword_phrases.end()) {
            return true;
        }
    }

    return false;
}

void StopwordsManager::process_stopwords(const std::string& stopword_set, std::vector<std::string>& tokens) {
    std::string merged_str;
    std::vector<int> stopword_indices;

    for (int i = tokens.size(); i >= 1; --i) {
        // Loop through all subsequences of length i
        for (int j = 0; j <= tokens.size() - i; ++j) {
            merged_str.clear();
            // Create a string from the subsequence of length i starting at index j
            for (int k = j; k < j + i; ++k) {
                if (k != j) {
                    merged_str += " ";  // Add space between words
                }
                merged_str += tokens[k];
            }

            if(is_stopword(stopword_set, merged_str)) {
                // store the indices of tokens which are stopword or stopword phrases
                for (int k = j; k < j + i; ++k) {
                    stopword_indices.push_back(k);
                }
            }
        }
    }

    // remove stopword/stopword phrases
    for(const auto& ind : stopword_indices) {
        tokens[ind].clear();
    }

    tokens.erase(std::remove_if(tokens.begin(), tokens.end(), [&](const auto& str) {
        return str.empty();
    }));
}
