#pragma once
#include <string>
#include <vector>
#include <map>
#include "embedder_manager.h"
#include <json.hpp>
#include <memory>
#include <mutex>
#include <core/session/onnxruntime_cxx_api.h>
#include "text_embedder_remote.h" // Added for embedding_res_t

class PersonalizationModel {
public:
    static inline const std::map<std::string, std::vector<std::string>> valid_model_names = {
        {"recommendation", {"tyrec-1"}},
        {"search", {"tyrec-2"}}
    };
    
    PersonalizationModel(const std::string& model_id);
    ~PersonalizationModel();

    static std::string get_model_subdir(const std::string& model_id);
    static Option<bool> validate_model(const nlohmann::json& model_json);
    static Option<nlohmann::json> create_model(const std::string& model_id, const nlohmann::json& model_json, const std::string model_data);
    static Option<nlohmann::json> update_model(const std::string& model_id, const nlohmann::json& model_json, const std::string model_data);
    static Option<bool> delete_model(const std::string& model_id);

    size_t get_input_dims() const { return input_dims_; }
    size_t get_output_dims() const { return output_dims_; }
    embedding_res_t embed_vector(const std::vector<float>& input_vector);
    std::vector<embedding_res_t> batch_embed_vectors(const std::vector<std::vector<float>>& input_vectors);
    Option<bool> validate_model_io();

private:
    std::string model_path_;
    std::string model_id_;
    size_t input_dims_;
    size_t output_dims_;
    std::shared_ptr<Ort::Env> env_;
    std::shared_ptr<Ort::Session> session_;
    std::mutex mutex_;
    void initialize_session();
};