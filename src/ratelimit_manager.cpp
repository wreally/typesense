#include "ratelimit_manager.h"
#include "string_utils.h"
#include "logger.h"
#include <iterator>

RateLimitManager * RateLimitManager::getInstance() {
    if(!instance) {
        instance = new RateLimitManager();
    }

    return instance;
}

void RateLimitManager::temp_ban_entity(const rate_limit_entity_t& entity, const uint64_t number_of_hours) {
    // lock mutex
    std::unique_lock<std::shared_mutex>lock(rate_limit_mutex);
    temp_ban_entity_wrapped(entity, number_of_hours);
}

bool RateLimitManager::is_rate_limited(const std::vector<rate_limit_entity_t> &entities) {
    // lock mutex
    std::unique_lock<std::shared_mutex>lock(rate_limit_mutex);
    rate_limit_entity_t ip_entity, api_key_entity;
    std::vector<rate_limit_rule_t*> rules_bucket;

    for(const auto &entity : entities) {
        if(entity.entity_type == RateLimitedEntityType::ip) {
            ip_entity = entity;
        } else if(entity.entity_type == RateLimitedEntityType::api_key) {
            api_key_entity = entity;
        }
    }
    // get wildcard rules
    if(rate_limit_entities.count(rate_limit_entity_t{RateLimitedEntityType::ip, ".*"}) > 0) {
        std::copy_if(rate_limit_entities.at(rate_limit_entity_t{RateLimitedEntityType::ip, ".*"}).begin(), rate_limit_entities.at(rate_limit_entity_t{RateLimitedEntityType::ip, ".*"}).end(), std::back_inserter(rules_bucket), [&](rate_limit_rule_t* rule) {
            return std::find_if(rule->entities.begin(), rule->entities.end(), [](const rate_limit_entity_t& entity) {
                return entity.entity_type == RateLimitedEntityType::api_key;
            }) == rule->entities.end() || std::find_if(rule->entities.begin(), rule->entities.end(), [&](const rate_limit_entity_t& entity) {
                return entity.entity_type == RateLimitedEntityType::api_key && (entity.entity_id == ".*" || entity.entity_id == api_key_entity.entity_id);
        }) != rule->entities.end();
        });
    }

    if(rate_limit_entities.count(rate_limit_entity_t{RateLimitedEntityType::api_key, ".*"}) > 0) {
        std::copy_if(rate_limit_entities.at(rate_limit_entity_t{RateLimitedEntityType::api_key, ".*"}).begin(), rate_limit_entities.at(rate_limit_entity_t{RateLimitedEntityType::api_key, ".*"}).end(), std::back_inserter(rules_bucket), [&](rate_limit_rule_t* rule) {
            return std::find_if(rule->entities.begin(), rule->entities.end(), [](const rate_limit_entity_t& entity) {
                return entity.entity_type == RateLimitedEntityType::ip;
            }) == rule->entities.end() || std::find_if(rule->entities.begin(), rule->entities.end(), [&](const rate_limit_entity_t& entity) {
                return entity.entity_type == RateLimitedEntityType::ip && (entity.entity_id == ".*" || entity.entity_id == ip_entity.entity_id);
        }) != rule->entities.end();
        });
    }

    // get rules for the IP entity
    if(rate_limit_entities.count(ip_entity) > 0) {
        std::copy_if(rate_limit_entities.at(ip_entity).begin(), rate_limit_entities.at(ip_entity).end(), std::back_inserter(rules_bucket), [&](rate_limit_rule_t* rule) {
            return std::find_if(rule->entities.begin(), rule->entities.end(), [](const rate_limit_entity_t& entity) {
                return entity.entity_type == RateLimitedEntityType::api_key;
            }) == rule->entities.end() || std::find_if(rule->entities.begin(), rule->entities.end(), [&](const rate_limit_entity_t& entity) {
                return entity.entity_type == RateLimitedEntityType::api_key && (entity.entity_id == ".*" || entity.entity_id == api_key_entity.entity_id);
        }) != rule->entities.end();
        });
    }

    // get rules for the API key entity
    if(rate_limit_entities.count(api_key_entity) > 0) {
        std::copy_if(rate_limit_entities.at(api_key_entity).begin(), rate_limit_entities.at(api_key_entity).end(), std::back_inserter(rules_bucket), [&](rate_limit_rule_t* rule) {
            return std::find_if(rule->entities.begin(), rule->entities.end(), [](const rate_limit_entity_t& entity) {
                return entity.entity_type == RateLimitedEntityType::ip;
            }) == rule->entities.end() || std::find_if(rule->entities.begin(), rule->entities.end(), [&](const rate_limit_entity_t& entity) {
                return entity.entity_type == RateLimitedEntityType::ip && (entity.entity_id == ".*" || entity.entity_id == ip_entity.entity_id);
        }) != rule->entities.end();
        });
    }

    if(rules_bucket.empty()) {
        return false;
    }

    // sort rules_bucket by priority in descending order
    std::sort(rules_bucket.begin(), rules_bucket.end(), [](rate_limit_rule_t* rule1, rate_limit_rule_t* rule2) {
        return rule1->priority > rule2->priority;
    });

    // get the rule with the highest priority
    auto& rule = *rules_bucket.front();
    // get key for throttling if exists
    auto throttle_key = get_throttle_key(ip_entity, api_key_entity);

    if(rule.action == RateLimitAction::block) {
        return true;
    }
    else if(rule.action == RateLimitAction::allow) {
        return false;
    }

    // check if any throttle exists and still valid
    while(throttle_key.ok()) {
        auto key = throttle_key.get();
        // Check ifban duration is not over
        if(throttled_entities.at(key).throttling_to > get_current_time()) {
            return true;
        }
        // Remove ban from DB store
        std::string ban_key = std::string(BANS_PREFIX) + "_" + std::to_string(throttled_entities.at(key).status_id);
        store->remove(ban_key);
        // Remove ban
        throttled_entities.erase(key);
        rate_limit_exceeds.erase(key);
        // Reset request counts
        auto& request_counts = rate_limit_request_counts.lookup(key);
        request_counts.reset();
        // Get next throttle key if exists
        throttle_key = get_throttle_key(ip_entity, api_key_entity);
    }

    // get request counter key according to rule type
    auto request_counter_key = get_request_counter_key(rule, ip_entity, api_key_entity);
    
    if(!rate_limit_request_counts.contains(request_counter_key)){
        rate_limit_request_counts.insert(request_counter_key, request_counter_t{});
    }
    auto& request_counts = rate_limit_request_counts.lookup(request_counter_key);
    // Check iflast reset time was more than 1 minute ago
    if(request_counts.last_reset_time_minute <= get_current_time() - 60) {
        request_counts.previous_requests_count_minute = request_counts.current_requests_count_minute;
        request_counts.current_requests_count_minute = 0;
        if(request_counts.last_reset_time_minute <= get_current_time() - 120) {
            request_counts.previous_requests_count_minute = 0;
        }
        request_counts.last_reset_time_minute = get_current_time();
    }
    // Check iflast reset time was more than 1 hour ago
    if(request_counts.last_reset_time_hour <= get_current_time() - 3600) {
        request_counts.previous_requests_count_hour = request_counts.current_requests_count_hour;
        request_counts.current_requests_count_hour = 0;
        if(request_counts.last_reset_time_hour <= get_current_time() - 7200) {
            request_counts.previous_requests_count_hour = 0;
        }
        request_counts.last_reset_time_hour = get_current_time();
    }
    // Check if request count is over the limit
    auto current_rate_for_minute = (60 - (get_current_time() - request_counts.last_reset_time_minute)) / 60  * request_counts.previous_requests_count_minute;
    current_rate_for_minute += request_counts.current_requests_count_minute;
    if(rule.max_requests.minute_threshold >= 0 && current_rate_for_minute >= rule.max_requests.minute_threshold) {
        bool auto_ban_is_enabled = (rule.auto_ban_threshold_num > 0 && rule.auto_ban_num_hours > 0);
        // If key is not in exceed map that means, it is a new exceed, not a continued exceed
        if(rate_limit_exceeds.count(request_counter_key) == 0) {
            rate_limit_exceeds.insert({request_counter_key, rate_limit_exceed_t{request_counter_key, 1}});
            request_counts.threshold_exceed_count_minute++;
        } else {
            // else it is a continued exceed, so just increment the request count
            rate_limit_exceeds[request_counter_key].request_count++;
        }
        // If auto ban is enabled, check if threshold is exceeded
        if(auto_ban_is_enabled) {
            if(request_counts.threshold_exceed_count_minute > rule.auto_ban_threshold_num) {
                temp_ban_entity_wrapped(request_counter_key.substr(0, request_counter_key.find("_")) == ".*" ? rate_limit_entity_t{RateLimitedEntityType::api_key, ".*"} : api_key_entity, rule.auto_ban_num_hours, (request_counter_key.substr((request_counter_key.find("_") + 1)) == ".*" && !rule.apply_limit_per_entity) ? nullptr : &ip_entity);
            }
        } 
        return true;
    }
    auto current_rate_for_hour = (3600 - (get_current_time() - request_counts.last_reset_time_hour)) / 3600  * request_counts.previous_requests_count_hour;
    current_rate_for_hour += request_counts.current_requests_count_hour;
    if(rule.max_requests.hour_threshold >= 0 && current_rate_for_hour >= rule.max_requests.hour_threshold) {
        if(rate_limit_exceeds.count(request_counter_key) == 0) {
            rate_limit_exceeds.insert({request_counter_key, rate_limit_exceed_t{request_counter_key, 1}});
        } else {
            rate_limit_exceeds[request_counter_key].request_count++;
        }
        return true;
    }
    // Increment request counts
    request_counts.current_requests_count_minute++;
    request_counts.current_requests_count_hour++;
    // If key is in exceed map that means, it is no longer exceed, so remove it from the map
    if(rate_limit_exceeds.count(request_counter_key) > 0) {
        rate_limit_exceeds.erase(request_counter_key);
    }
    return false;
}

Option<nlohmann::json> RateLimitManager::find_rule_by_id(const uint64_t id) {
    std::shared_lock<std::shared_mutex> lock(rate_limit_mutex);
    if(rule_store.count(id) > 0) {
        return Option<nlohmann::json>(rule_store.at(id).to_json());
    }
    return Option<nlohmann::json>(404, "Not Found");
}

bool RateLimitManager::delete_rule_by_id(const uint64_t id) {
    std::unique_lock<std::shared_mutex> lock(rate_limit_mutex);
    const std::string rule_store_key = get_rule_key(id);
    bool deleted = store->remove(rule_store_key);
    if(!deleted) {
        return false;
    }
    // Check ifa rule exists for the given ID
    if(rule_store.count(id) > 0) {
        auto rule = rule_store.at(id);
        // Remove rule from rule store
        rule_store.erase(id);
        // Remove rule from rate limit rule pointer
        for(auto &entity : rate_limit_entities) {
            std::remove_if(entity.second.begin(), entity.second.end(), [&rule](rate_limit_rule_t *rule_ptr) {
                return rule_ptr->id == rule.id;
            });
        }
        return true;
    }
    return false;
}


const std::vector<rate_limit_rule_t> RateLimitManager::get_all_rules() {
    std::shared_lock<std::shared_mutex> lock(rate_limit_mutex);
    // Get all rules in a vector
    std::vector<rate_limit_rule_t> rules;
    for(const auto &rule : rule_store) {
        rules.push_back(rule.second);
    }
    return rules;
}

const std::vector<rate_limit_status_t> RateLimitManager::get_banned_entities(const RateLimitedEntityType entity_type) {
    std::shared_lock<std::shared_mutex> lock(rate_limit_mutex);
    std::vector <rate_limit_status_t> banned_entities;
    for (auto & element: throttled_entities) {
        if(element.second.entity.entity_type == entity_type) {
            banned_entities.push_back(element.second);
        }

        if(element.second.and_entity.ok()) {
            if(element.second.and_entity.get().entity_type == entity_type) {
                banned_entities.push_back(element.second);
            }
        }
    }
    // Get permanent bans
    for (auto & element: rule_store) {
        if(element.second.action == RateLimitAction::block) {
            for(const auto& entity: element.second.entities) {
                if(entity.entity_type == entity_type) {
                    banned_entities.push_back(rate_limit_status_t(0, 0, 0, entity));
                }
            }
        }
    }
    return banned_entities;
}

void RateLimitManager::clear_all() {
    std::unique_lock<std::shared_mutex> lock(rate_limit_mutex);
    rate_limit_request_counts.clear();
    rate_limit_entities.clear();
    throttled_entities.clear();
    rate_limit_exceeds.clear();
    rule_store.clear();
    last_rule_id = 0;
    last_ban_id = 0;
    base_timestamp = 0;
}

void RateLimitManager::temp_ban_entity_wrapped(const rate_limit_entity_t& entity, const uint64_t number_of_hours, const rate_limit_entity_t* and_entity) {
    std::string key = entity.entity_id + "_" + (and_entity != nullptr ? and_entity->entity_id : ".*");
    // Check ifentity is already banned
    if(throttled_entities.count(key) > 0) {
        return;
    }
    auto now = get_current_time();
    // Add entity to throttled_entities for the given number of days
    rate_limit_status_t status(last_ban_id, now, now + (number_of_hours * 60 * 60), entity, and_entity);
    std::string ban_key = get_ban_key(last_ban_id);
    store->insert(ban_key, status.to_json().dump());
    throttled_entities.insert({key, status});
    last_ban_id++;
    if(rate_limit_request_counts.contains(key)){
        // Reset counters for the given entity
        rate_limit_request_counts.lookup(key).current_requests_count_minute = 0;
        rate_limit_request_counts.lookup(key).current_requests_count_hour = 0;
    }
}

const nlohmann::json RateLimitManager::get_all_rules_json() {
    std::shared_lock<std::shared_mutex> lock(rate_limit_mutex);
    nlohmann::json rules_json = nlohmann::json::array();
    for(const auto &rule : rule_store) {
        rules_json.push_back(rule.second.to_json());
    }
    return rules_json;
}

const nlohmann::json rate_limit_rule_t::to_json() const {
    nlohmann::json rule;
    nlohmann::json api_keys_json = nlohmann::json::array();
    nlohmann::json ip_addresses_json = nlohmann::json::array();
    rule["id"] = id;
    rule["action"] = magic_enum::enum_name(action);
    rule["priority"] = priority;
    for(const auto &entity : entities) {
        if(entity.entity_type == RateLimitedEntityType::api_key) {
            api_keys_json.push_back(entity.entity_id);
        } else if(entity.entity_type == RateLimitedEntityType::ip) {
            ip_addresses_json.push_back(entity.entity_id);
        }
    }
    if(max_requests.minute_threshold >= 0) {
        rule["max_requests"]["minute_threshold"] = max_requests.minute_threshold;
    }
    if(max_requests.hour_threshold >= 0) {
        rule["max_requests"]["hour_threshold"] = max_requests.hour_threshold;
    }
    if(auto_ban_threshold_num >= 0) {
        rule["auto_ban_threshold_num"] = auto_ban_threshold_num;
    }
    if(auto_ban_num_hours >= 0) {
        rule["auto_ban_num_hours"] = auto_ban_num_hours;
    }
    if(!api_keys_json.empty()) {
        rule["api_keys"] = api_keys_json;
    }
    if(!ip_addresses_json.empty()) {
        rule["ip_addresses"] = ip_addresses_json;
    }
    return rule;
}

const nlohmann::json rate_limit_status_t::to_json() const {
    nlohmann::json status;
    status["id"] = status_id;
    status["throttling_from"] = throttling_from;
    status["throttling_to"] = throttling_to;
    status["value"] = entity.entity_id;
    status["entity_type"] = magic_enum::enum_name(entity.entity_type);
    if(and_entity.ok()) {
        auto and_entity_value = and_entity.get();
        status["and_entity"] = nlohmann::json::object();
        status["and_entity"]["value"] = and_entity_value.entity_id;
        status["and_entity"]["entity_type"] = magic_enum::enum_name(and_entity_value.entity_type);
    }
    return status;
}

void rate_limit_status_t::parse_json(const nlohmann::json &json) {
    status_id = json["id"];
    throttling_from = json["throttling_from"];
    throttling_to = json["throttling_to"];
    entity.entity_id = json["value"];
    entity.entity_type = magic_enum::enum_cast<RateLimitedEntityType>(json["entity_type"].get<std::string>()).value();
    if(json.contains("and_entity")) {
        and_entity = Option<rate_limit_entity_t>(rate_limit_entity_t{magic_enum::enum_cast<RateLimitedEntityType>(json["and_entity"]["entity_type"].get<std::string>()).value(), json["and_entity"]["value"]});
    } else {
        and_entity = Option<rate_limit_entity_t>(404, "No and_entity found");
    }
}


Option<nlohmann::json> RateLimitManager::add_rule(const nlohmann::json &rule_json) {
    std::unique_lock<std::shared_mutex> lock(rate_limit_mutex);
    auto rule_validation_result = is_valid_rule(rule_json);
    if(!rule_validation_result.ok()) {
        return Option<nlohmann::json>(rule_validation_result.code(), rule_validation_result.error());
    }
    auto parsed_rule_option = parse_rule(rule_json);
    if(!parsed_rule_option.ok()) {
        return Option<nlohmann::json>(parsed_rule_option.code(), parsed_rule_option.error());
    }
    rate_limit_rule_t parsed_rule = parsed_rule_option.get();
    parsed_rule.id = last_rule_id++;
    const std::string rule_store_key = get_rule_key(parsed_rule.id);
    bool inserted = store->insert(rule_store_key, parsed_rule.to_json().dump());
    if(!inserted) {
        return Option<nlohmann::json>(500, "Failed to insert rule into the DB store");
    }
    store->increment(std::string(RULES_NEXT_ID), 1);
    // Insert rule to rule store
    lock.unlock();
    insert_rule(parsed_rule);
    lock.lock();
    nlohmann::json response;
    response["message"] = "Rule added successfully.";
    response["rule"] = parsed_rule.to_json();
    return Option<nlohmann::json>(response);
}

Option<nlohmann::json> RateLimitManager::edit_rule(const uint64_t id, const nlohmann::json &rule_json) {
    std::unique_lock<std::shared_mutex> lock(rate_limit_mutex);
    const auto& rule_option = find_rule_by_id(id);
    if(!rule_option.ok()) {
        return Option<nlohmann::json>(rule_option.code(), rule_option.error());
    }
    auto rule_validation_result = is_valid_rule(rule_json);
    if(!rule_validation_result.ok()) {
        return Option<nlohmann::json>(rule_validation_result.code(), rule_validation_result.error());
    }
    auto parsed_rule_option = parse_rule(rule_json);
    if(!parsed_rule_option.ok()) {
        return Option<nlohmann::json>(parsed_rule_option.code(), parsed_rule_option.error());
    }
    rate_limit_rule_t parsed_rule = parsed_rule_option.get();
    parsed_rule.id = id;
    const std::string rule_store_key = get_rule_key(parsed_rule.id);
    bool inserted = store->insert(rule_store_key, parsed_rule.to_json().dump());
    if(!inserted) {
        return Option<nlohmann::json>(500, "Failed to update rule in the DB store");
    }
    auto old_rule = rule_store.at(id);
    // Remove rule from rate limit rule pointer
    for(const auto &entity : old_rule.entities) {
        auto& vec = rate_limit_entities.at(entity);
        std::remove_if(vec.begin(), vec.end(), [&](const auto &rule) {
            return rule->id == id;
        });
    }
    // Insert new rule to rule store
    lock.unlock();
    insert_rule(parsed_rule);
    lock.lock();
    nlohmann::json response;
    response["message"] = "Rule updated successfully.";
    response["rule"] = parsed_rule.to_json();
    return Option<nlohmann::json>(response);
}

Option<bool> RateLimitManager::is_valid_rule(const nlohmann::json &rule_json) {
    if(rule_json.count("action") == 0) {
        return Option<bool>(400, "Parameter `action` is required.");
    }
    if(rule_json.count("apply_limit_per_entity") > 0 && rule_json["apply_limit_per_entity"].is_boolean() == false) {
        return Option<bool>(400, "Parameter `apply_limit_per_entity` must be a boolean.");
    }
    if((rule_json.count("ip_addresses") == 0 && rule_json.count("api_keys") == 0) || (rule_json.count("ip_addresses") > 1 && rule_json.count("api_keys") > 1)) {
            return Option<bool>(400, "Invalid combination of `ip_addresses` and `api_keys`.");
    }
    if(rule_json.count("ip_addresses") > 0 && !rule_json["ip_addresses"].is_array() && !rule_json["ip_addresses"][0].is_string()) {
        return Option<bool>(400, "Parameter `ip_addresses` must be an array of strings.");
    }
    if(rule_json.count("api_keys") > 0 && !rule_json["api_keys"].is_array() && !rule_json["api_keys"][0].is_string()) {
        return Option<bool>(400, "Parameter `api_keys` must be an array of strings.");
    }
    if(rule_json["action"].is_string() == false) {
        return Option<bool>(400, "Parameter `action` must be a string.");
    }
    if(rule_json["action"] == "allow") {
        return Option<bool>(true);
    } else if(rule_json["action"] == "block") {
        return Option<bool>(true);
    } else if(rule_json["action"] == "throttle") {
        if(rule_json.count("max_requests_1m") == 0 && rule_json.count("max_requests_1h") == 0) {
            return Option<bool>(400, "At least  one of `max_requests_1m` or `max_requests_1h` is required.");
        }
        if(rule_json.count("max_requests_1m") > 0 && rule_json["max_requests_1m"].is_number_integer() == false) {
            return Option<bool>(400, "Parameter `max_requests_1m` must be an integer.");
        }
        if(rule_json.count("max_requests_1h") > 0 && rule_json["max_requests_1h"].is_number_integer() == false) {
            return Option<bool>(400, "Parameter `max_requests_1h` must be an integer.");
        }
        if((rule_json.count("auto_ban_threshold_num") > 0 && rule_json.count("auto_ban_num_hours") == 0) || (rule_json.count("auto_ban_threshold_num") == 0 && rule_json.count("auto_ban_num_hours") > 0)) {
            return Option<bool>(400, "Both `auto_ban_threshold_num` and `auto_ban_num_hours` are required ifeither is specified.");

        }
        if(rule_json.count("auto_ban_threshold_num") > 0 && rule_json.count("auto_ban_num_hours") > 0) {
            if(!rule_json["auto_ban_threshold_num"].is_number_integer() || !rule_json["auto_ban_num_hours"].is_number_integer()) {
                return Option<bool>(400, "Parameters `auto_ban_threshold_num` and `auto_ban_num_hours` must be integers.");
            }
            if(rule_json["auto_ban_threshold_num"].get<int>() < 0 || rule_json["auto_ban_num_hours"].get<int>() < 0) {
                return Option<bool>(400, "Both `auto_ban_threshold_num` and `auto_ban_num_hours` must be greater than 0.");
            }
        }
    } else {
        return Option<bool>(400, "Invalid action.");
    }
    return Option<bool>(true);
}

Option<rate_limit_rule_t> RateLimitManager::parse_rule(const nlohmann::json &rule_json)
{
    rate_limit_rule_t new_rule;
    new_rule.action = magic_enum::enum_cast<RateLimitAction>(rule_json["action"].get<std::string>()).value();
    if(rule_json.count("ip_addresses") > 0) {
        for(const auto& ip: rule_json["ip_addresses"]) {
            new_rule.entities.push_back(rate_limit_entity_t{RateLimitedEntityType::ip, ip});
        }
    }
    if(rule_json.count("api_keys") > 0) {
        for(const auto& api_key: rule_json["api_keys"]) {
            new_rule.entities.push_back(rate_limit_entity_t{RateLimitedEntityType::api_key, api_key});
        }
    }
    if(rule_json.count("max_requests_1m") > 0) {
        new_rule.max_requests.minute_threshold = rule_json["max_requests_1m"];
    }
    if(rule_json.count("max_requests_1h") > 0) {
        new_rule.max_requests.hour_threshold = rule_json["max_requests_1h"];
    }
    if(rule_json.count("auto_ban_threshold_num") > 0 && rule_json.count("auto_ban_num_hours") > 0) {
        new_rule.auto_ban_threshold_num = rule_json["auto_ban_threshold_num"];
        new_rule.auto_ban_num_hours = rule_json["auto_ban_num_hours"];
    }
    if(rule_json.count("apply_limit_per_entity") > 0) {
        new_rule.apply_limit_per_entity = rule_json["apply_limit_per_entity"].get<bool>();
    }
    if(rule_json.count("priority") > 0) {
        new_rule.priority = rule_json["priority"];
    }
    return Option<rate_limit_rule_t>(new_rule);
}

void RateLimitManager::insert_rule(const rate_limit_rule_t &rule) {
    std::unique_lock<std::shared_mutex> lock(rate_limit_mutex);
    rule_store[rule.id] = rule;
    for(const auto &entity : rule.entities) {
        rate_limit_entities[entity].push_back(&rule_store[rule.id]);
    }
}


Option<bool> RateLimitManager::init(Store *store) {
    std::unique_lock<std::shared_mutex> lock(rate_limit_mutex);
    this->store = store;
    // Load rules from database
    std::string last_rule_id_str;
    StoreStatus last_rule_id_status = store->get(std::string(RULES_NEXT_ID), last_rule_id_str);
    if(last_rule_id_status == StoreStatus::ERROR) {
        return Option<bool>(500, "Error while fetching rule next id from database.");
    }
    else if(last_rule_id_status == StoreStatus::FOUND) {
        last_rule_id = StringUtils::deserialize_uint32_t(last_rule_id_str);
    }
    else {
        last_rule_id = 0;
    }

    std::vector<std::string> rule_json_strs;
    store->scan_fill(std::string(RULES_PREFIX) + "_", std::string(RULES_PREFIX) + "`", rule_json_strs);

    for(const auto& rule_json_str: rule_json_strs) {
        nlohmann::json rule_json = nlohmann::json::parse(rule_json_str);
        Option<rate_limit_rule_t> rule_option = parse_rule(rule_json);
        if(!rule_option.ok()) {
            return Option<bool>(rule_option.code(), rule_option.error());
        }
        auto rule = rule_option.get();
        rule.id = rule_json["id"];
        lock.unlock();
        insert_rule(rule);
        lock.lock();
    }
    // Load bans from database
    std::string last_ban_id_str;
    StoreStatus last_ban_id_status = store->get(BANS_NEXT_ID, last_ban_id_str);
    if(last_ban_id_status == StoreStatus::ERROR) {
        return Option<bool>(500, "Error while fetching ban next id from database.");
    }
    else if(last_ban_id_status == StoreStatus::FOUND) {
        last_ban_id = StringUtils::deserialize_uint32_t(last_ban_id_str);
    }
    else {
        last_ban_id = 0;
    }
    std::vector<std::string> ban_json_strs;
    store->scan_fill(std::string(BANS_PREFIX) + "_", std::string(BANS_PREFIX) + "`", ban_json_strs);

    for(const auto& ban_json_str: ban_json_strs) {
        nlohmann::json ban_json = nlohmann::json::parse(ban_json_str);
        rate_limit_status_t ban_status;
        ban_status.parse_json(ban_json);
        std::string key = ban_status.entity.entity_id + "_" + (ban_status.and_entity.ok() ? ban_status.and_entity.get().entity_id : ".*");
        throttled_entities.insert({key, ban_status});
    }
    LOG(INFO) << "Loaded " << rule_store.size() << " rate limit rules.";
    LOG(INFO) << "Loaded " << throttled_entities.size() << " rate limit bans.";
    return Option<bool>(true);
}

std::string RateLimitManager::get_rule_key(const uint32_t id) {
    return std::string(RULES_PREFIX) + "_" + std::to_string(id);
}

std::string RateLimitManager::get_ban_key(const uint32_t id) {
    return std::string(BANS_PREFIX) + "_" + std::to_string(id);
}

time_t RateLimitManager::get_current_time() {
    return  base_timestamp + std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());;
}

void RateLimitManager::_set_base_timestamp(const time_t& timestamp) {
    base_timestamp = timestamp;
}

Option<std::string> RateLimitManager::get_throttle_key(const rate_limit_entity_t& ip_entity, const rate_limit_entity_t& api_key_entity) {
    if(throttled_entities.count(api_key_entity.entity_id + "_" + ip_entity.entity_id) > 0) {
        return Option<std::string>(api_key_entity.entity_id + "_" + ip_entity.entity_id);
    }
    else if(throttled_entities.count(api_key_entity.entity_id + "_.*") > 0) {
        return Option<std::string>(api_key_entity.entity_id + "_.*");
    }
    else if(throttled_entities.count(".*_" + ip_entity.entity_id) > 0) {
        return Option<std::string>(".*_" + ip_entity.entity_id);
    }
    else if(throttled_entities.count(".*_.*") > 0) {
        return Option<std::string>(".*_.*");
    }
    return Option<std::string>(404, "No throttle found.");
}

const std::string RateLimitManager::get_request_counter_key(const rate_limit_rule_t& rule, const rate_limit_entity_t& ip_entity, const rate_limit_entity_t& api_key_entity) {
    bool has_api_key = false, has_ip = false, has_wildcard_ip = false, has_wildcard_api_key = false;
    for(const auto& entity: rule.entities) {
        if(entity.entity_type == RateLimitedEntityType::ip) {
            has_ip = true;
            if(entity.entity_id == ".*") {
                has_wildcard_ip = true;
            }
        }
        else if(entity.entity_type == RateLimitedEntityType::api_key) {
            has_api_key = true;
            if(entity.entity_id == ".*") {
                has_wildcard_api_key = true;
            }
        }
    }
    std::string key;
    if(!has_api_key || has_wildcard_api_key) {
        key += ".*";
    } else {
        key += api_key_entity.entity_id;
    }
    key += "_";
    if((!has_ip || has_wildcard_ip) && !rule.apply_limit_per_entity) {
        key += ".*";
    } else {
        key += ip_entity.entity_id;
    }
    return key;
}

const nlohmann::json RateLimitManager::get_exceeded_entities_json() {
    std::shared_lock<std::shared_mutex> lock(rate_limit_mutex);
    nlohmann::json exceeded_entities_json = nlohmann::json::array();
    for(const auto& entity: rate_limit_exceeds) {
        exceeded_entities_json.push_back(entity.second.to_json());
    }
    return exceeded_entities_json;
}

const nlohmann::json RateLimitManager::get_throttled_entities_json() {
    std::shared_lock<std::shared_mutex> lock(rate_limit_mutex);
    nlohmann::json throttled_entities_json = nlohmann::json::array();
    for(const auto& entity: throttled_entities) {
        auto json = entity.second.to_json();
        json[json["entity_type"].get<std::string>() == "ip" ? "ip_address" : "api_key"] = json["value"];
        json.erase("entity_type");
        json.erase("value");
        if(json["and_entity"].is_object()) {
            json[json["and_entity"]["entity_type"].get<std::string>() == "ip" ? "ip_address" : "api_key"] = json["and_entity"]["value"];
            json.erase("and_entity");
        }
        if(json["api_key"] == ".*") {
            json.erase("api_key");
        }
        throttled_entities_json.push_back(json);
    }
    return throttled_entities_json;
}

bool RateLimitManager::delete_throttle_by_id(const uint64_t id) {
    std::unique_lock<std::shared_mutex> lock(rate_limit_mutex);
    std::string ban_key = get_ban_key(id);
    bool deleted = store->remove(ban_key);
    if(!deleted) {
        return false;
    }
    auto ban = std::find_if(throttled_entities.begin(), throttled_entities.end(), [id](const auto& ban) {
        return ban.second.status_id == id;
    });
    if(ban != throttled_entities.end()) {
        throttled_entities.erase(ban);
    } else {
        return false;
    }
    return true;
}