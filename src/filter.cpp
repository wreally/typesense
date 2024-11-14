#include <collection_manager.h>
#include <posting.h>
#include <timsort.hpp>
#include <stack>
#include "filter.h"

Option<bool> filter::validate_numerical_filter_value(field _field, const string &raw_value) {
    if(_field.is_int32()) {
        if (!StringUtils::is_integer(raw_value)) {
            return Option<bool>(400, "Error with filter field `" + _field.name + "`: Not an int32.");
        } else if (!StringUtils::is_int32_t(raw_value)) {
            return Option<bool>(400, "Error with filter field `" + _field.name +
                                                        "`: `" + raw_value + "` exceeds the range of an int32.");
        }
        return Option<bool>(true);
    }

    else if(_field.is_int64() && !StringUtils::is_int64_t(raw_value)) {
        return Option<bool>(400, "Error with filter field `" + _field.name + "`: Not an int64.");
    }

    else if(_field.is_float() && !StringUtils::is_float(raw_value)) {
        return Option<bool>(400, "Error with filter field `" + _field.name + "`: Not a float.");
    }

    return Option<bool>(true);
}

Option<NUM_COMPARATOR> filter::extract_num_comparator(string &comp_and_value) {
    auto num_comparator = EQUALS;

    if(StringUtils::is_integer(comp_and_value) || StringUtils::is_float(comp_and_value)) {
        num_comparator = EQUALS;
    }

        // the ordering is important - we have to compare 2-letter operators first
    else if(comp_and_value.compare(0, 2, "<=") == 0) {
        num_comparator = LESS_THAN_EQUALS;
    }

    else if(comp_and_value.compare(0, 2, ">=") == 0) {
        num_comparator = GREATER_THAN_EQUALS;
    }

    else if(comp_and_value.compare(0, 2, "!=") == 0) {
        num_comparator = NOT_EQUALS;
    }

    else if(comp_and_value.compare(0, 1, "<") == 0) {
        num_comparator = LESS_THAN;
    }

    else if(comp_and_value.compare(0, 1, ">") == 0) {
        num_comparator = GREATER_THAN;
    }

    // "=" case is handled upstream.

    else if(comp_and_value.find("..") != std::string::npos) {
        num_comparator = RANGE_INCLUSIVE;
    }

    else {
        return Option<NUM_COMPARATOR>(400, "Numerical field has an invalid comparator.");
    }

    if(num_comparator == LESS_THAN || num_comparator == GREATER_THAN) {
        comp_and_value = comp_and_value.substr(1);
    } else if(num_comparator == LESS_THAN_EQUALS || num_comparator == GREATER_THAN_EQUALS || num_comparator == NOT_EQUALS) {
        comp_and_value = comp_and_value.substr(2);
    }

    comp_and_value = StringUtils::trim(comp_and_value);

    return Option<NUM_COMPARATOR>(num_comparator);
}

Option<bool> filter::parse_geopoint_filter_value(std::string& raw_value,
                                                 const std::string& format_err_msg,
                                                 std::string& processed_filter_val,
                                                 NUM_COMPARATOR& num_comparator) {

    num_comparator = LESS_THAN_EQUALS;

    if(!(raw_value[0] == '(' && raw_value[raw_value.size() - 1] == ')')) {
        return Option<bool>(400, format_err_msg);
    }

    std::vector<std::string> filter_values;
    auto raw_val_without_paran = raw_value.substr(1, raw_value.size() - 2);
    StringUtils::split(raw_val_without_paran, filter_values, ",");

    // we will end up with: "10.45 34.56 2 km" or "10.45 34.56 2mi" or a geo polygon

    if(filter_values.size() < 3) {
        return Option<bool>(400, format_err_msg);
    }

    // do validation: format should match either a point + radius or polygon

    size_t num_floats = 0;
    for(const auto& fvalue: filter_values) {
        if(StringUtils::is_float(fvalue)) {
            num_floats++;
        }
    }

    bool is_polygon = (num_floats == filter_values.size());
    if(!is_polygon) {
        // we have to ensure that this is a point + radius match
        if(!StringUtils::is_float(filter_values[0]) || !StringUtils::is_float(filter_values[1])) {
            return Option<bool>(400, format_err_msg);
        }

        if(filter_values[0] == "nan" || filter_values[0] == "NaN" ||
           filter_values[1] == "nan" || filter_values[1] == "NaN") {
            return Option<bool>(400, format_err_msg);
        }
    }

    if(is_polygon) {
        processed_filter_val = raw_val_without_paran;
    } else {
        // point + radius
        // filter_values[2] is distance, get the unit, validate it and split on that
        if(filter_values[2].size() < 2) {
            return Option<bool>(400, "Unit must be either `km` or `mi`.");
        }

        std::string unit = filter_values[2].substr(filter_values[2].size()-2, 2);

        if(unit != "km" && unit != "mi") {
            return Option<bool>(400, "Unit must be either `km` or `mi`.");
        }

        std::vector<std::string> dist_values;
        StringUtils::split(filter_values[2], dist_values, unit);

        if(dist_values.size() != 1) {
            return Option<bool>(400, format_err_msg);
        }

        if(!StringUtils::is_float(dist_values[0])) {
            return Option<bool>(400, format_err_msg);
        }

        processed_filter_val = filter_values[0] + ", " + filter_values[1] + ", " + // co-ords
                               dist_values[0] + ", " +  unit;           // X km
    }

    return Option<bool>(true);
}

Option<bool> validate_geofilter_distance(std::string& raw_value, const string& format_err_msg,
                                         std::string& distance, std::string& unit) {
    if (raw_value.size() < 2) {
        return Option<bool>(400, "Unit must be either `km` or `mi`.");
    }

    unit = raw_value.substr(raw_value.size() - 2, 2);

    if (unit != "km" && unit != "mi") {
        return Option<bool>(400, "Unit must be either `km` or `mi`.");
    }

    std::vector<std::string> dist_values;
    StringUtils::split(raw_value, dist_values, unit);

    if (dist_values.size() != 1) {
        return Option<bool>(400, format_err_msg);
    }

    if (!StringUtils::is_float(dist_values[0])) {
        return Option<bool>(400, format_err_msg);
    }

    distance = std::string(dist_values[0]);
    return Option<bool>(true);
}

Option<bool> filter::parse_geopoint_filter_value(string& raw_value, const string& format_err_msg, filter& filter_exp) {
    // FORMAT:
    // [ ([48.853, 2.344], radius: 1km, exact_filter_radius: 100km),
    //   ([48.8662, 2.3255, 48.8581, 2.3209, 48.8561, 2.3448, 48.8641, 2.3469], exact_filter_radius: 100km) ]

    // Every open parenthesis represent a geo filter value.
    auto open_parenthesis_count = std::count(raw_value.begin(), raw_value.end(), '(');
    if (open_parenthesis_count < 1) {
        return Option<bool>(400, format_err_msg);
    }

    filter_exp.comparators.push_back(LESS_THAN_EQUALS);
    bool is_multivalued = raw_value[0] == '[';
    size_t i = is_multivalued;

    for (auto j = 0; j < open_parenthesis_count; j++) {
        if (is_multivalued) {
            auto pos = raw_value.find('(', i);
            if (pos == std::string::npos) {
                return Option<bool>(400, format_err_msg);
            }
            i = pos;
        }

        i++;
        if (i >= raw_value.size()) {
            return Option<bool>(400, format_err_msg);
        }

        auto value_end_index = raw_value.find(')', i);
        if (value_end_index == std::string::npos) {
            return Option<bool>(400, format_err_msg);
        }

        // [48.853, 2.344], radius: 1km, exact_filter_radius: 100km
        // [48.8662, 2.3255, 48.8581, 2.3209, 48.8561, 2.3448, 48.8641, 2.3469], exact_filter_radius: 100km
        std::string value_str = raw_value.substr(i, value_end_index - i);
        StringUtils::trim(value_str);

        if (value_str.empty() || value_str[0] != '[' || value_str.find(']', 1) == std::string::npos) {
            return Option<bool>(400, format_err_msg);
        }

        auto points_str = value_str.substr(1, value_str.find(']', 1) - 1);
        std::vector<std::string> geo_points;
        StringUtils::split(points_str, geo_points, ",");

        if (geo_points.size() < 2 || geo_points.size() % 2) {
            return Option<bool>(400, format_err_msg);
        }

        bool is_polygon = geo_points.size() > 2;
        for (const auto& geo_point: geo_points) {
            if (geo_point == "nan" || geo_point == "NaN" || !StringUtils::is_float(geo_point)) {
                return Option<bool>(400, format_err_msg);
            }
        }

        if (is_polygon) {
            filter_exp.values.push_back(points_str);
        }

        // Handle options.
        // , radius: 1km, exact_filter_radius: 100km
        i = raw_value.find(']', i) + 1;

        std::vector<std::string> options;
        StringUtils::split(raw_value.substr(i, value_end_index - i), options, ",");

        if (options.empty()) {
            if (!is_polygon) {
                // Missing radius option
                return Option<bool>(400, format_err_msg);
            }

            nlohmann::json param;
            param[EXACT_GEO_FILTER_RADIUS_KEY] = DEFAULT_EXACT_GEO_FILTER_RADIUS_VALUE;
            filter_exp.params.push_back(param);

            continue;
        }

        bool is_radius_present = false;
        for (auto const& option: options) {
            std::vector<std::string> key_value;
            StringUtils::split(option, key_value, ":");

            if (key_value.size() < 2) {
                continue;
            }

            if (key_value[0] == GEO_FILTER_RADIUS_KEY && !is_polygon) {
                is_radius_present = true;

                std::string distance, unit;
                auto validate_op = validate_geofilter_distance(key_value[1], format_err_msg, distance, unit);
                if (!validate_op.ok()) {
                    return validate_op;
                }

                filter_exp.values.push_back(points_str + ", " + distance + ", " + unit);
            } else if (key_value[0] == EXACT_GEO_FILTER_RADIUS_KEY) {
                std::string distance, unit;
                auto validate_op = validate_geofilter_distance(key_value[1], format_err_msg, distance, unit);
                if (!validate_op.ok()) {
                    return validate_op;
                }

                double exact_under_radius = std::stof(distance);

                if (unit == "km") {
                    exact_under_radius *= 1000;
                } else {
                    // assume "mi" (validated upstream)
                    exact_under_radius *= 1609.34;
                }

                nlohmann::json param;
                param[EXACT_GEO_FILTER_RADIUS_KEY] = exact_under_radius;
                filter_exp.params.push_back(param);

                // Only EXACT_GEO_FILTER_RADIUS_KEY option would be present for a polygon. We can also stop if we've
                // parsed the radius in case of a single geopoint since there are only two options.
                if (is_polygon || is_radius_present) {
                    break;
                }
            }
        }

        if (!is_radius_present && !is_polygon) {
            return Option<bool>(400, format_err_msg);
        }

        // EXACT_GEO_FILTER_RADIUS_KEY was not present.
        if (filter_exp.params.size() < filter_exp.values.size()) {
            nlohmann::json param;
            param[EXACT_GEO_FILTER_RADIUS_KEY] = DEFAULT_EXACT_GEO_FILTER_RADIUS_VALUE;
            filter_exp.params.push_back(param);
        }
    }

    return Option<bool>(true);
}

bool isOperator(const std::string& expression) {
    return expression == "&&" || expression == "||";
}

// https://en.wikipedia.org/wiki/Shunting_yard_algorithm
Option<bool> toPostfix(std::queue<std::string>& tokens, std::queue<std::string>& postfix) {
    std::stack<std::string> operatorStack;

    while (!tokens.empty()) {
        auto expression = tokens.front();
        tokens.pop();

        if (isOperator(expression)) {
            // We only have two operators &&, || having the same precedence and both being left associative.
            while (!operatorStack.empty() && operatorStack.top() != "(") {
                postfix.push(operatorStack.top());
                operatorStack.pop();
            }

            operatorStack.push(expression);
        } else if (expression == "(") {
            operatorStack.push(expression);
        } else if (expression == ")") {
            while (!operatorStack.empty() && operatorStack.top() != "(") {
                postfix.push(operatorStack.top());
                operatorStack.pop();
            }

            if (operatorStack.empty() || operatorStack.top() != "(") {
                return Option<bool>(400, "Could not parse the filter query: unbalanced parentheses.");
            }
            operatorStack.pop();
        } else {
            postfix.push(expression);
        }
    }

    while (!operatorStack.empty()) {
        if (operatorStack.top() == "(") {
            return Option<bool>(400, "Could not parse the filter query: unbalanced parentheses.");
        }
        postfix.push(operatorStack.top());
        operatorStack.pop();
    }

    return Option<bool>(true);
}

Option<bool> toMultiValueNumericFilter(std::string& raw_value, filter& filter_exp, const field& _field) {
    std::vector<std::string> filter_values;
    StringUtils::split(raw_value.substr(1, raw_value.size() - 2), filter_values, ",");
    filter_exp = {_field.name, {}, {}};
    for (std::string& filter_value: filter_values) {
        Option<NUM_COMPARATOR> op_comparator = filter::extract_num_comparator(filter_value);
        if (!op_comparator.ok()) {
            return Option<bool>(400, "Error with filter field `" + _field.name + "`: " + op_comparator.error());
        }
        if (op_comparator.get() == RANGE_INCLUSIVE) {
            // split the value around range operator to extract bounds
            std::vector<std::string> range_values;
            StringUtils::split(filter_value, range_values, filter::RANGE_OPERATOR());
            for (const std::string& range_value: range_values) {
                auto validate_op = filter::validate_numerical_filter_value(_field, range_value);
                if (!validate_op.ok()) {
                    return validate_op;
                }
                filter_exp.values.push_back(range_value);
                filter_exp.comparators.push_back(op_comparator.get());
            }
        } else {
            auto validate_op = filter::validate_numerical_filter_value(_field, filter_value);
            if (!validate_op.ok()) {
                return validate_op;
            }
            filter_exp.values.push_back(filter_value);
            filter_exp.comparators.push_back(op_comparator.get());
        }
    }

    return Option<bool>(true);
}

Option<bool> toFilter(const std::string expression,
                      filter& filter_exp,
                      const tsl::htrie_map<char, field>& search_schema,
                      const Store* store,
                      const std::string& doc_id_prefix) {
    // split into [field_name, value]
    size_t found_index = expression.find(':');
    if (found_index == std::string::npos) {
        return Option<bool>(400, "Could not parse the filter query.");
    }
    std::string&& field_name = expression.substr(0, found_index);
    StringUtils::trim(field_name);
    if (field_name == "id") {
        std::string&& raw_value = expression.substr(found_index + 1, std::string::npos);
        StringUtils::trim(raw_value);
        std::string empty_filter_err = "Error with filter field `id`: Filter value cannot be empty.";
        if (raw_value.empty()) {
            return Option<bool>(400, empty_filter_err);
        }
        filter_exp = {field_name, {}, {}};
        NUM_COMPARATOR id_comparator = EQUALS;
        size_t filter_value_index = 0;
        if (raw_value == "*") { // Match all.
            // NOT_EQUALS comparator with no id match will get all the ids.
            id_comparator = NOT_EQUALS;
            filter_exp.apply_not_equals = true;
        } else if (raw_value[0] == '=') {
            id_comparator = EQUALS;
            while (++filter_value_index < raw_value.size() && raw_value[filter_value_index] == ' ');
        } else if (raw_value.size() >= 2 && raw_value[0] == '!' && raw_value[1] == '=') {
            id_comparator = NOT_EQUALS;
            filter_exp.apply_not_equals = true;
            filter_value_index++;
            while (++filter_value_index < raw_value.size() && raw_value[filter_value_index] == ' ');
        }
        if (filter_value_index != 0) {
            raw_value = raw_value.substr(filter_value_index);
        }
        if (raw_value.empty()) {
            return Option<bool>(400, empty_filter_err);
        }
        if (raw_value[0] == '[' && raw_value[raw_value.size() - 1] == ']') {
            std::vector<std::string> doc_ids;
            StringUtils::split_to_values(raw_value.substr(1, raw_value.size() - 2), doc_ids);
            for (std::string& doc_id: doc_ids) {
                // we have to convert the doc_id to seq id
                std::string seq_id_str;
                StoreStatus seq_id_status = store->get(doc_id_prefix + doc_id, seq_id_str);
                if (seq_id_status != StoreStatus::FOUND) {
                    continue;
                }
                filter_exp.values.push_back(seq_id_str);
                filter_exp.comparators.push_back(id_comparator);
            }
        } else {
            std::vector<std::string> doc_ids;
            StringUtils::split_to_values(raw_value, doc_ids); // to handle backticks
            if (doc_ids.empty()) {
                return Option<bool>(400, empty_filter_err);
            }

            std::string seq_id_str;
            StoreStatus seq_id_status = store->get(doc_id_prefix + doc_ids[0], seq_id_str);
            if (seq_id_status == StoreStatus::FOUND) {
                filter_exp.values.push_back(seq_id_str);
                filter_exp.comparators.push_back(id_comparator);
            }
        }
        return Option<bool>(true);
    }

    auto field_it = search_schema.find(field_name);

    if (field_it == search_schema.end()) {
        return Option<bool>(404, "Could not find a filter field named `" + field_name + "` in the schema.");
    }

    if (field_it->num_dim > 0) {
        return Option<bool>(404, "Cannot filter on vector field `" + field_name + "`.");
    }

    const field& _field = field_it.value();
    std::string&& raw_value = expression.substr(found_index + 1, std::string::npos);
    StringUtils::trim(raw_value);
    // skip past optional `:=` operator, which has no meaning for non-string fields
    if (!_field.is_string() && raw_value[0] == '=') {
        size_t filter_value_index = 0;
        while (raw_value[++filter_value_index] == ' ');
        raw_value = raw_value.substr(filter_value_index);
    }
    if (_field.is_integer() || _field.is_float()) {
        // could be a single value or a list
        if (raw_value[0] == '[' && raw_value[raw_value.size() - 1] == ']') {
            Option<bool> op = toMultiValueNumericFilter(raw_value, filter_exp, _field);
            if (!op.ok()) {
                return op;
            }
        } else {
            Option<NUM_COMPARATOR> op_comparator = filter::extract_num_comparator(raw_value);
            if (!op_comparator.ok()) {
                return Option<bool>(400, "Error with filter field `" + _field.name + "`: " + op_comparator.error());
            }
            if (op_comparator.get() == RANGE_INCLUSIVE) {
                // split the value around range operator to extract bounds
                std::vector<std::string> range_values;
                StringUtils::split(raw_value, range_values, filter::RANGE_OPERATOR());
                filter_exp.field_name = field_name;
                for (const std::string& range_value: range_values) {
                    auto validate_op = filter::validate_numerical_filter_value(_field, range_value);
                    if (!validate_op.ok()) {
                        return validate_op;
                    }
                    filter_exp.values.push_back(range_value);
                    filter_exp.comparators.push_back(op_comparator.get());
                }
            } else if (op_comparator.get() == NOT_EQUALS && raw_value[0] == '[' && raw_value[raw_value.size() - 1] == ']') {
                Option<bool> op = toMultiValueNumericFilter(raw_value, filter_exp, _field);
                if (!op.ok()) {
                    return op;
                }
                filter_exp.apply_not_equals = true;
            } else {
                auto validate_op = filter::validate_numerical_filter_value(_field, raw_value);
                if (!validate_op.ok()) {
                    return validate_op;
                }
                filter_exp = {field_name, {raw_value}, {op_comparator.get()}};
            }
        }
    } else if (_field.is_bool()) {
        NUM_COMPARATOR bool_comparator = EQUALS;
        size_t filter_value_index = 0;
        if (raw_value[0] == '=') {
            bool_comparator = EQUALS;
            while (++filter_value_index < raw_value.size() && raw_value[filter_value_index] == ' ');
        } else if (raw_value.size() >= 2 && raw_value[0] == '!' && raw_value[1] == '=') {
            bool_comparator = NOT_EQUALS;
            filter_value_index++;
            while (++filter_value_index < raw_value.size() && raw_value[filter_value_index] == ' ');
        }
        if (filter_value_index != 0) {
            raw_value = raw_value.substr(filter_value_index);
        }
        if (filter_value_index == raw_value.size()) {
            return Option<bool>(400, "Error with filter field `" + _field.name +
                                     "`: Filter value cannot be empty.");
        }
        if (raw_value[0] == '[' && raw_value[raw_value.size() - 1] == ']') {
            std::vector<std::string> filter_values;
            StringUtils::split(raw_value.substr(1, raw_value.size() - 2), filter_values, ",");
            filter_exp = {field_name, {}, {}};
            for (std::string& filter_value: filter_values) {
                if (filter_value != "true" && filter_value != "false") {
                    return Option<bool>(400, "Values of filter field `" + _field.name +
                                             "`: must be `true` or `false`.");
                }
                filter_value = (filter_value == "true") ? "1" : "0";
                filter_exp.values.push_back(filter_value);
                filter_exp.comparators.push_back(bool_comparator);
            }
        } else {
            if (raw_value != "true" && raw_value != "false") {
                return Option<bool>(400, "Value of filter field `" + _field.name + "` must be `true` or `false`.");
            }
            std::string bool_value = (raw_value == "true") ? "1" : "0";
            filter_exp = {field_name, {bool_value}, {bool_comparator}};
        }
    } else if (_field.is_geopoint()) {
        filter_exp = {field_name, {}, {}};
        NUM_COMPARATOR num_comparator;

        if ((raw_value[0] == '(' && std::count(raw_value.begin(), raw_value.end(), '[') > 0) ||
            std::count(raw_value.begin(), raw_value.end(), '[') > 1 ||
            std::count(raw_value.begin(), raw_value.end(), ':') > 0) {

            const std::string& format_err_msg = "Value of filter field `" + _field.name + "`: must be in the "
                                                "`([-44.50, 170.29], radius: 0.75 km, exact_filter_radius: 5 km)` or "
                                                "([56.33, -65.97, 23.82, -127.82], exact_filter_radius: 7 km) format.";

            auto parse_op = filter::parse_geopoint_filter_value(raw_value, format_err_msg, filter_exp);
            return parse_op;
        }

        const std::string& format_err_msg = "Value of filter field `" + _field.name +
                                            "`: must be in the `(-44.50, 170.29, 0.75 km)` or "
                                            "(56.33, -65.97, 23.82, -127.82) format.";
        // could be a single value or a list
        if (raw_value[0] == '[' && raw_value[raw_value.size() - 1] == ']') {
            std::vector<std::string> filter_values;
            StringUtils::split(raw_value.substr(1, raw_value.size() - 2), filter_values, "),");
            for (std::string& filter_value: filter_values) {
                filter_value += ")";
                std::string processed_filter_val;
                auto parse_op = filter::parse_geopoint_filter_value(filter_value, format_err_msg, processed_filter_val,
                                                                    num_comparator);
                if (!parse_op.ok()) {
                    return parse_op;
                }
                filter_exp.values.push_back(processed_filter_val);
                filter_exp.comparators.push_back(num_comparator);
            }
        } else {
            // single value, e.g. (10.45, 34.56, 2 km)
            std::string processed_filter_val;
            auto parse_op = filter::parse_geopoint_filter_value(raw_value, format_err_msg, processed_filter_val,
                                                                num_comparator);
            if (!parse_op.ok()) {
                return parse_op;
            }
            filter_exp.values.push_back(processed_filter_val);
            filter_exp.comparators.push_back(num_comparator);
        }
    } else if (_field.is_string()) {
        size_t filter_value_index = 0;
        NUM_COMPARATOR str_comparator = CONTAINS;
        auto apply_not_equals = false;
        if (raw_value[0] == '=') {
            // string filter should be evaluated in strict "equals" mode
            str_comparator = EQUALS;
            while (++filter_value_index < raw_value.size() && raw_value[filter_value_index] == ' ');
        } else if (raw_value.size() >= 2 && raw_value[0] == '!') {
            if (raw_value[1] == '=') {
                str_comparator = NOT_EQUALS;
                filter_value_index++;
            }

            apply_not_equals = true;
            while (++filter_value_index < raw_value.size() && raw_value[filter_value_index] == ' ');
        }
        if (filter_value_index == raw_value.size()) {
            return Option<bool>(400, "Error with filter field `" + _field.name +
                                     "`: Filter value cannot be empty.");
        }
        if (raw_value[filter_value_index] == '[' && raw_value[raw_value.size() - 1] == ']') {
            std::vector<std::string> filter_values;
            StringUtils::split_to_values(
                    raw_value.substr(filter_value_index + 1, raw_value.size() - filter_value_index - 2), filter_values);
            if(_field.stem) {
                auto stemmer = _field.get_stemmer();
                for (std::string& filter_value: filter_values) {
                    filter_value = stemmer->stem(filter_value);
                }
            }
            filter_exp = {field_name, filter_values, {str_comparator}};
        } else {
            std::string filter_value = raw_value.substr(filter_value_index);
            if(_field.stem) {
                auto stemmer = _field.get_stemmer();
                filter_value = stemmer->stem(filter_value);
            }
            filter_exp = {field_name, {filter_value}, {str_comparator}};
        }

        filter_exp.apply_not_equals = apply_not_equals;
    } else {
        return Option<bool>(400, "Error with filter field `" + _field.name +
                                 "`: Unidentified field data type, see docs for supported data types.");
    }

    return Option<bool>(true);
}

// https://stackoverflow.com/a/423914/11218270
Option<bool> toParseTree(std::queue<std::string>& postfix, filter_node_t*& root,
                         const tsl::htrie_map<char, field>& search_schema,
                         const Store* store,
                         const std::string& doc_id_prefix) {
    std::stack<filter_node_t*> nodeStack;
    bool is_successful = true;
    std::string error_message;

    filter_node_t *filter_node = nullptr;

    while (!postfix.empty()) {
        const std::string expression = postfix.front();
        postfix.pop();

        if (isOperator(expression)) {
            if (nodeStack.empty()) {
                is_successful = false;
                error_message = "Could not parse the filter query: unbalanced `" + expression + "` operands.";
                break;
            }
            auto operandB = nodeStack.top();
            nodeStack.pop();

            if (nodeStack.empty()) {
                delete operandB;
                is_successful = false;
                error_message = "Could not parse the filter query: unbalanced `" + expression + "` operands.";
                break;
            }
            auto operandA = nodeStack.top();
            nodeStack.pop();

            filter_node = new filter_node_t(expression == "&&" ? AND : OR, operandA, operandB);
            filter_node->filter_query = operandA->filter_query + " " + expression + " " + operandB->filter_query;
        } else {
            filter filter_exp;

            // Expected value: $Collection(...)
            bool is_referenced_filter = (expression[0] == '$' && expression[expression.size() - 1] == ')');
            if (is_referenced_filter) {
                size_t parenthesis_index = expression.find('(');

                std::string collection_name = expression.substr(1, parenthesis_index - 1);
                auto &cm = CollectionManager::get_instance();
                auto collection = cm.get_collection(collection_name);
                if (collection == nullptr) {
                    is_successful = false;
                    error_message = "Referenced collection `" + collection_name + "` not found.";
                    break;
                }

                filter_exp = {expression.substr(parenthesis_index + 1, expression.size() - parenthesis_index - 2)};
                filter_exp.referenced_collection_name = collection_name;
            } else {
                Option<bool> toFilter_op = toFilter(expression, filter_exp, search_schema, store, doc_id_prefix);
                if (!toFilter_op.ok()) {
                    is_successful = false;
                    error_message = toFilter_op.error();
                    break;
                }
            }

            filter_node = new filter_node_t(filter_exp);
            filter_node->filter_query = expression;
        }

        nodeStack.push(filter_node);
    }

    if (!is_successful) {
        while (!nodeStack.empty()) {
            auto filterNode = nodeStack.top();
            delete filterNode;
            nodeStack.pop();
        }

        return Option<bool>(400, error_message);
    }

    if (nodeStack.empty()) {
        return Option<bool>(400, "Filter query cannot be empty.");
    }
    root = nodeStack.top();

    return Option<bool>(true);
}

Option<bool> filter::parse_filter_query(const std::string& filter_query,
                                        const tsl::htrie_map<char, field>& search_schema,
                                        const Store* store,
                                        const std::string& doc_id_prefix,
                                        filter_node_t*& root) {
    auto _filter_query = filter_query;
    StringUtils::trim(_filter_query);
    if (_filter_query.empty()) {
        return Option<bool>(true);
    }

    std::queue<std::string> tokens;
    Option<bool> tokenize_op = StringUtils::tokenize_filter_query(filter_query, tokens);
    if (!tokenize_op.ok()) {
        return tokenize_op;
    }

    std::queue<std::string> postfix;
    Option<bool> toPostfix_op = toPostfix(tokens, postfix);
    if (!toPostfix_op.ok()) {
        return toPostfix_op;
    }

    auto const& max_ops = CollectionManager::get_instance().filter_by_max_ops;
    if (postfix.size() > max_ops) {
        return Option<bool>(400, "`filter_by` has too many operations. Maximum allowed: " + std::to_string(max_ops) +
                                 ". Use `--filter-by-max-ops` command line argument to customize this value.");
    }

    Option<bool> toParseTree_op = toParseTree(postfix,
                                              root,
                                              search_schema,
                                              store,
                                              doc_id_prefix);
    if (!toParseTree_op.ok()) {
        return toParseTree_op;
    }

    root->filter_query = filter_query;
    return Option<bool>(true);
}
