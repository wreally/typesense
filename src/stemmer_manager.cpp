#include "stemmer_manager.h"


Stemmer::Stemmer(const char * language, const std::string& dictionary_name) {
    if(dictionary_name.empty()) {
        this->stemmer = sb_stemmer_new(language, nullptr);
    } else {
        this->dictionary_name = dictionary_name;
    }

    this->cache = LRU::Cache<std::string, std::string>(20);
}

Stemmer::~Stemmer() {
    if(stemmer) {
        sb_stemmer_delete(stemmer);
    }
}

std::string Stemmer::stem(const std::string & word) {
    std::unique_lock<std::mutex> lock(mutex);
    std::string stemmed_word;
    if (cache.contains(word)) {
        return cache.lookup(word);
    }

    if(dictionary_name.empty()) {
        auto stemmed = sb_stemmer_stem(stemmer, reinterpret_cast<const sb_symbol *>(word.c_str()), word.length());
        stemmed_word = std::string(reinterpret_cast<const char *>(stemmed));
    } else {
        const auto& stem_dictionary = StemmerManager::get_instance().get_dictionary(dictionary_name);
        auto it = stem_dictionary.find(word);
        if(it == stem_dictionary.end()) {
            stemmed_word = word;
        } else {
            stemmed_word = it->second;
        }
    }

    cache.insert(word, stemmed_word);
    return stemmed_word;
}

StemmerManager::~StemmerManager() {
    delete_all_stemmers();
    stem_dictionary.clear();
}

std::shared_ptr<Stemmer> StemmerManager::get_stemmer(const std::string& language, const std::string& dictionary_name) {
    std::unique_lock<std::mutex> lock(mutex);
    // use english as default language
    const std::string language_ = language.empty() ? "english" : language;
    if (stemmers.find(language_) == stemmers.end()) {
        stemmers[language] = std::make_shared<Stemmer>(language_.c_str(), dictionary_name);
    }
    return stemmers[language];
}

void StemmerManager::delete_stemmer(const std::string& language) {
    std::unique_lock<std::mutex> lock(mutex);
    if (stemmers.find(language) != stemmers.end()) {
        stemmers.erase(language);
    }
}

void StemmerManager::delete_all_stemmers() {
    std::unique_lock<std::mutex> lock(mutex);
    stemmers.clear();
}

const bool StemmerManager::validate_language(const std::string& language) {
    const std::string language_ = language.empty() ? "english" : language;
    auto stemmer = sb_stemmer_new(language_.c_str(), nullptr);
    if (stemmer == nullptr) {
        return false;
    }
    sb_stemmer_delete(stemmer);
    return true;
}

bool StemmerManager::save_words(const std::string& dictionary_name, const std::vector<std::string> &json_lines) {
    if(json_lines.empty()) {
        return false;
    }

    nlohmann::json json_line;
    for(const auto& line_str : json_lines) {
        try {
            json_line = nlohmann::json::parse(line_str);
        } catch(...) {
            return false;
        }

        if(!json_line.contains("word") || !json_line.contains("root")) {
            return false;
        }
        stem_dictionary[dictionary_name].emplace(json_line["word"], json_line["root"]);
    }
    return true;
}

spp::sparse_hash_map<std::string, std::string> StemmerManager::get_dictionary(const std::string& dictionary_name) {
    if(stem_dictionary.count(dictionary_name) != 0) {
        return stem_dictionary.at(dictionary_name);
    }

    return spp::sparse_hash_map<std::string, std::string>();
}