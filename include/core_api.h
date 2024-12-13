#pragma once

#include "http_server.h"
#include "auth_manager.h"
#include "ratelimit_manager.h"

bool handle_authentication(std::map<std::string, std::string>& req_params,
                           std::vector<nlohmann::json>& embedded_params_vec,
                           const std::string& body, const route_path& rpath,
                           const std::string& req_auth_key);

bool get_alter_in_progress(const std::string& collection);

// Collections

bool get_collections(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool post_create_collection(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool patch_update_collection(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool del_drop_collection(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool get_collection_summary(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

// Documents

bool get_search(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool post_multi_search(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool get_export_documents(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool post_add_document(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool patch_update_document(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool patch_update_documents(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool post_import_documents(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool get_fetch_document(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool del_remove_document(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool del_remove_documents(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

// Alias

bool get_alias(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool get_aliases(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool put_upsert_alias(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool del_alias(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

// Presets

bool get_presets(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool get_preset(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool put_upsert_preset(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool del_preset(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

//stopwords

bool get_stopwords(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool get_stopword(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool put_upsert_stopword(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool del_stopword(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

// Overrides

bool get_overrides(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool get_override(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool put_override(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool del_override(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

// Synonyms

bool get_synonyms(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool get_synonym(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool put_synonym(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool del_synonym(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

// Keys

bool get_keys(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool post_create_key(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool get_key(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool del_key(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

// Health + Metrics

bool get_debug(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool get_health(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool get_health_with_resource_usage(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool post_health(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool get_metrics_json(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool get_stats_json(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool get_status(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

// operations

bool post_snapshot(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool post_vote(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool post_config(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool post_clear_cache(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool post_compact_db(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool post_reset_peers(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

// Rate Limiting

bool get_rate_limits(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool get_rate_limit(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool put_rate_limit(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool del_rate_limit(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool post_rate_limit(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool get_active_throttles(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool get_limit_exceed_counts(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool del_throttle(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool del_exceed(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

// Analytics

bool post_create_event(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool get_analytics_rules(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool get_analytics_rule(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool post_create_analytics_rules(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool put_upsert_analytics_rules(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool del_analytics_rules(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool post_write_analytics_to_db(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool get_analytics_events(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

//plurals, nouns
bool post_import_stemming_dictionary(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool get_stemming_dictionaries(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool get_stemming_dictionary(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool del_stemming_dictionary(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

// Misc helpers

void get_collections_for_auth(std::map<std::string, std::string>& req_params, const std::string& body,
                              const route_path& rpath, const std::string& req_auth_key,
                              std::vector<collection_key_t>& collections,
                              std::vector<nlohmann::json>& embedded_params_vec);

bool is_doc_import_route(uint64_t route_hash);

bool is_coll_create_route(uint64_t route_hash);

bool is_drop_collection_route(uint64_t route_hash);

bool is_doc_write_route(uint64_t route_hash);

bool is_doc_del_route(uint64_t route_hash);

Option<std::pair<std::string,std::string>> get_api_key_and_ip(const std::string& metadata);

void init_api(uint32_t cache_num_entries);


bool post_proxy(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);




bool post_conversation_model(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool get_conversation_model(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool del_conversation_model(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool get_conversation_models(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool put_conversation_model(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool post_personalization_model(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool get_personalization_model(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool del_personalization_model(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool get_personalization_models(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool put_personalization_model(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);

bool test_personalization_model(const std::shared_ptr<http_req>& req, const std::shared_ptr<http_res>& res);
