// https://arxiv.org/pdf/1902.04023
// refer: https://medium.com/@mani./t-digest-an-interesting-datastructure-to-estimate-quantiles-accurately-b99a50eaf4f7

#pragma once

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cassert>

class TDigest {
public:
    TDigest(double compression = 100.0) : compression(compression) {}

    void add(double value) {
        // Add a new value to the t-digest
        if (data.empty() || value < data.front().mean) {
            data.insert(data.begin(), {value, 1});
        } else if (value > data.back().mean) {
            data.push_back({value, 1});
        } else {
            auto it = std::lower_bound(data.begin(), data.end(), value, [](const Node& node, double val) {
                return node.mean < val;
            });
            if (it != data.end() && it->mean == value) {
                it->count++;
            } else {
                data.insert(it, {value, 1});
            }
        }

        if(value < min_val) {
            min_val = value;
        }

        if(value > max_val) {
            max_val = value;
        }

        total_size++;
        total_value += value;
        compress();
    }

    double percentile(double p) {
        // Calculate the approximate percentile
        if (data.empty()) return 0.0;

        double total_count = 0;
        for (const auto& node : data) {
            total_count += node.count;
        }

        double target = (p * total_count)/100.f;
        double cumulative_count = 0;

        for (const auto& node : data) {
            cumulative_count += node.count;
            if (cumulative_count >= target) {
                return node.mean;
            }
        }
        return data.back().mean; // Fallback
    }

    uint64_t size() const {
        return total_size;
    }

    uint64_t min() const {
        return min_val;
    }

    uint64_t max() const {
        return max_val;
    }

    uint64_t sum() const {
        return total_value;
    }

private:
    struct Node {
        double mean;
        int count;

        bool operator<(const Node& other) const {
            return mean < other.mean;
        }
    };

    std::vector<Node> data;
    double compression;
    uint64_t min_val = UINT64_MAX;
    uint64_t max_val = 0;
    uint64_t total_size = 0;
    uint64_t total_value = 0;

    void compress() {
        // Compress the data if necessary
        if (data.size() > compression) {
            std::vector<Node> new_data;
            for (const auto& node : data) {
                if (new_data.empty() || new_data.back().mean != node.mean) {
                    new_data.push_back(node);
                } else {
                    new_data.back().count += node.count;
                }
            }
            data = std::move(new_data);
        }
    }
};


