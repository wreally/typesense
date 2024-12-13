#include <housekeeper.h>
#include "typesense_server_utils.h"
#include "core_api.h"
#include "tsconfig.h"
#include "stackprinter.h"
#include "backward.hpp"
#include "butil/at_exit.h"

#ifndef ASAN_BUILD
extern "C" {
#include "jemalloc.h"
}

#ifdef __APPLE__
extern "C" {
    extern void je_zone_register();
}
#endif
#endif

void master_server_routes() {
    // collection operations
    // NOTE: placing this first to score an immediate hit on O(N) route search
    server->get("/collections/:collection/documents/search", get_search);
    server->post("/multi_search", post_multi_search);

    // document management
    // NOTE:`/documents/:id` end-points must be placed last in the list
    server->post("/collections/:collection/documents", post_add_document);
    server->del("/collections/:collection/documents", del_remove_documents, false, true);

    server->post("/collections/:collection/documents/import", post_import_documents, true, true);
    server->get("/collections/:collection/documents/export", get_export_documents, false, true);

    server->get("/collections/:collection/documents/:id", get_fetch_document);
    server->patch("/collections/:collection/documents/:id", patch_update_document);
    server->patch("/collections/:collection/documents", patch_update_documents);
    server->del("/collections/:collection/documents/:id", del_remove_document);

    server->get("/collections/:collection/overrides", get_overrides);
    server->get("/collections/:collection/overrides/:id", get_override);
    server->put("/collections/:collection/overrides/:id", put_override);
    server->del("/collections/:collection/overrides/:id", del_override);

    server->get("/collections/:collection/synonyms", get_synonyms);
    server->get("/collections/:collection/synonyms/:id", get_synonym);
    server->put("/collections/:collection/synonyms/:id", put_synonym);
    server->del("/collections/:collection/synonyms/:id", del_synonym);

    // collection management
    server->post("/collections", post_create_collection);
    server->patch("/collections/:collection", patch_update_collection);
    server->get("/collections", get_collections);
    server->del("/collections/:collection", del_drop_collection);
    server->get("/collections/:collection", get_collection_summary);

    server->get("/aliases", get_aliases);
    server->get("/aliases/:alias", get_alias);
    server->put("/aliases/:alias", put_upsert_alias);
    server->del("/aliases/:alias", del_alias);

    server->get("/keys", get_keys);
    server->get("/keys/:id", get_key);
    server->post("/keys", post_create_key);
    server->del("/keys/:id", del_key);

    server->get("/presets", get_presets);
    server->get("/presets/:name", get_preset);
    server->put("/presets/:name", put_upsert_preset);
    server->del("/presets/:name", del_preset);

    server->get("/stopwords", get_stopwords);
    server->get("/stopwords/:name", get_stopword);
    server->put("/stopwords/:name", put_upsert_stopword);
    server->del("/stopwords/:name", del_stopword);

    // analytics
    server->get("/analytics/rules", get_analytics_rules);
    server->get("/analytics/rules/:name", get_analytics_rule);
    server->post("/analytics/rules", post_create_analytics_rules);
    server->put("/analytics/rules/:name", put_upsert_analytics_rules);
    server->del("/analytics/rules/:name", del_analytics_rules);
    server->post("/analytics/events", post_create_event);
    server->post("/analytics/aggregate_events", post_write_analytics_to_db);
    server->get("/analytics/events", get_analytics_events);

    // for plurals, nouns
    server->post("/stemming/dictionaries/import", post_import_stemming_dictionary, true, true);
    server->get("/stemming/dictionaries", get_stemming_dictionaries);
    server->get("/stemming/dictionaries/:id", get_stemming_dictionary);

    // meta
    server->get("/metrics.json", get_metrics_json);
    server->get("/stats.json", get_stats_json);
    server->get("/debug", get_debug);
    server->get("/health", get_health);
    server->get("/health_with_rusage", get_health_with_resource_usage);
    server->post("/health", post_health);
    server->get("/status", get_status);

    server->post("/operations/snapshot", post_snapshot, false, true);
    server->post("/operations/vote", post_vote, false, false);
    server->post("/operations/cache/clear", post_clear_cache, false, false);
    server->post("/operations/db/compact", post_compact_db, false, false);
    server->post("/operations/reset_peers", post_reset_peers, false, false);

    server->post("/conversations/models", post_conversation_model);
    server->get("/conversations/models", get_conversation_models);
    server->get("/conversations/models/:id", get_conversation_model);
    server->put("/conversations/models/:id", put_conversation_model);
    server->del("/conversations/models/:id", del_conversation_model);
    
    server->post("/personalization/models", post_personalization_model);
    server->get("/personalization/models", get_personalization_models);
    server->get("/personalization/models/:id", get_personalization_model);
    server->del("/personalization/models/:id", del_personalization_model);
    server->put("/personalization/models/:id", put_personalization_model);
    server->get("/personalization/models/:id/test", test_personalization_model);
    

    server->get("/limits", get_rate_limits);
    server->get("/limits/active", get_active_throttles);
    server->get("/limits/exceeds", get_limit_exceed_counts);
    server->get("/limits/:id", get_rate_limit);
    server->post("/limits", post_rate_limit);
    server->put("/limits/:id", put_rate_limit);
    server->del("/limits/:id", del_rate_limit);
    server->del("/limits/active/:id", del_throttle);
    server->del("/limits/exceeds/:id", del_exceed);
    server->post("/config", post_config, false, false);

    // for proxying remote embedders
    server->post("/proxy", post_proxy);
}

void (*backward::SignalHandling::_callback)(int sig, backward::StackTrace&) = nullptr;

void crash_callback(int sig, backward::StackTrace& st) {
    backward::TraceResolver tr; tr.load_stacktrace(st);
    for (size_t i = 0; i < st.size(); ++i) {
        backward::ResolvedTrace trace = tr.resolve(st[i]);
        if(trace.object_function.find("BatchedIndexer") != std::string::npos ||
           trace.object_function.find("batch_memory_index") != std::string::npos) {
            server->persist_applying_index();
            break;
        }
    }

    HouseKeeper::get_instance().log_running_queries();
    LOG(ERROR) << "Typesense " << TYPESENSE_VERSION << " is terminating abruptly.";
}

int main(int argc, char **argv) {
#ifndef ASAN_BUILD
    #ifdef __APPLE__
    // On OS X, je_zone_register registers jemalloc with the system allocator.
    // We have to force the presence of these symbols on macOS by explicitly calling this method.
    // See these issues:
    // - https://github.com/jemalloc/jemalloc/issues/708
    // - https://github.com/ClickHouse/ClickHouse/pull/11897
    je_zone_register();
    #endif
#endif

    butil::AtExitManager exit_manager;

    Config& config = Config::get_instance();

    cmdline::parser options;
    init_cmdline_options(options, argc, argv);
    options.parse(argc, argv);

    // Command line args override env vars
    config.load_config_env();
    config.load_config_file(options);
    config.load_config_cmd_args(options);

    Option<bool> config_validitation = config.is_valid();

    if(!config_validitation.ok()) {
        std::cerr << "Typesense " << TYPESENSE_VERSION << std::endl;
        std::cerr << "Invalid configuration: " << config_validitation.error() << std::endl;
        std::cerr << "Command line " << options.usage() << std::endl;
        std::cerr << "You can also pass these arguments as environment variables such as "
                  << "TYPESENSE_DATA_DIR, TYPESENSE_API_KEY, etc." << std::endl;
        exit(1);
    }

    int ret_code = init_root_logger(config, TYPESENSE_VERSION);
    if(ret_code != 0) {
        return ret_code;
    }

#ifdef __APPLE__
    #ifdef USE_BACKWARD
        backward::SignalHandling sh;
        sh._callback = crash_callback;
    #else
        signal(SIGABRT, StackPrinter::bt_sighandler);
        signal(SIGFPE, StackPrinter::bt_sighandler);
        signal(SIGILL, StackPrinter::bt_sighandler);
        signal(SIGSEGV, StackPrinter::bt_sighandler);
    #endif
#elif __linux__
    backward::SignalHandling sh;
    sh._callback = crash_callback;
#endif

    // we can install new signal handlers only after overriding above
    signal(SIGINT, catch_interrupt);
    signal(SIGTERM, catch_interrupt);

    init_api(config.get_cache_num_entries());

    return run_server(config, TYPESENSE_VERSION, &master_server_routes);
}