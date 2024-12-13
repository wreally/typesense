#pragma once

#include <sentencepiece_processor.h>
#include <core/session/onnxruntime_cxx_api.h>
#include <tokenizer/bert_tokenizer.hpp>
#include <vector>
#include "option.h"
#include "text_embedder_tokenizer.h"
#include "text_embedder_remote.h"


class TextEmbedder {
    public:
        // Constructor for local or public models
        TextEmbedder(const std::string& model_path, const bool is_public_model);
        // Constructor for remote models
        TextEmbedder(const nlohmann::json& model_config, size_t num_dims, const bool has_custom_dims = false);
        ~TextEmbedder();
        embedding_res_t Embed(const std::string& text, const size_t remote_embedder_timeout_ms = 30000, const size_t remote_embedding_num_tries = 2);
        std::vector<embedding_res_t> batch_embed(const std::vector<std::string>& inputs, const size_t remote_embedding_batch_size = 200,
                                                 const size_t remote_embedding_timeout_ms = 60000, const size_t remote_embedding_num_tries = 2);
        const std::string& get_vocab_file_name() const;
        const size_t get_num_dim() const;
        bool is_remote() {
            return remote_embedder_ != nullptr;
        }
        Option<bool> validate();

        std::shared_ptr<Ort::Session> get_session() {
            return session_;
        }

        std::shared_ptr<Ort::Env> get_env() {
            return env_;
        }

        const TokenizerType get_tokenizer_type() {
            return tokenizer_->get_tokenizer_type();
        }

        bool update_remote_embedder_apikey(const std::string& api_key) {
            return remote_embedder_->update_api_key(api_key);
        }
        static std::vector<float> mean_pooling(const std::vector<std::vector<float>>& input, const std::vector<int64_t>& attention_mask);


    private:
        std::shared_ptr<Ort::Session> session_;
        std::shared_ptr<Ort::Env> env_;
        encoded_input_t Encode(const std::string& text);
        batch_encoded_input_t batch_encode(const std::vector<std::string>& inputs);
        std::unique_ptr<TextEmbeddingTokenizer> tokenizer_;
        std::unique_ptr<RemoteEmbedder> remote_embedder_;
        std::string vocab_file_name;
        std::string output_tensor_name;
        size_t num_dim;
        std::mutex mutex_;
};
