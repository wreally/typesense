#include <gtest/gtest.h>
#include "app_metrics.h"

class AppMetricsTest : public ::testing::Test {
protected:
    AppMetrics& metrics = AppMetrics::get_instance();

    virtual void SetUp() {

    }

    virtual void TearDown() {

    }
};


TEST_F(AppMetricsTest, StatefulRemoveDocs) {
    metrics.increment_count("GET /collections", 1);
    metrics.increment_count("GET /collections", 1);
    metrics.increment_count("GET /operations/vote", 1);

    metrics.increment_duration("GET /collections", 2);
    metrics.increment_duration("GET /collections", 4);
    metrics.increment_duration("GET /operations/vote", 5);

    metrics.increment_count(AppMetrics::SEARCH_LABEL, 1);
    metrics.increment_count(AppMetrics::SEARCH_LABEL, 1);
    metrics.increment_duration(AppMetrics::SEARCH_LABEL, 16);
    metrics.increment_duration(AppMetrics::SEARCH_LABEL, 12);

    metrics.window_reset();

    nlohmann::json result;
    metrics.get("rps", "latency", result);

    ASSERT_EQ(result["search_latency"].get<double>(), 14.0);
    ASSERT_EQ(result["search_rps"].get<double>(), 0.2);

    ASSERT_EQ(result["latency"]["GET /collections"].get<double>(), 3.0);
    ASSERT_EQ(result["latency"]["GET /operations/vote"].get<double>(), 5.0);

    ASSERT_EQ(result["rps"]["GET /collections"].get<double>(), 0.2);
    ASSERT_EQ(result["rps"]["GET /operations/vote"].get<double>(), 0.1);
}

TEST_F(AppMetricsTest, EstimateQuantileDuration) {
    //add 100 random durations
    std::mt19937 rng;
    std::uniform_int_distribution<uint32_t> distrib(0, 1000);
    rng.seed(1);

    std::vector<int> durations;
    for(auto i = 0; i < 10000; ++i) {
        durations.push_back(distrib(rng));
    }

    std::sort(durations.begin(), durations.end());

    // add to appmetrics to get approximate percentile
    for(auto i = 0; i < 10000; ++i) {
        metrics.increment_count(AppMetrics::SEARCH_LABEL, 1);
        metrics.increment_duration(AppMetrics::SEARCH_LABEL, durations[i]);
    }

    metrics.window_reset();

    nlohmann::json result;
    metrics.get("rps", "latency", result);
    ASSERT_EQ(result["search_70Percentile_latency"], 701.0);
    ASSERT_EQ(result["search_95Percentile_latency"], 950.0);
    ASSERT_EQ(result["search_99Percentile_latency"], 990.0);

    // compute accurate percentile
    auto computeNthPercentile = [&](int percentile) -> int {
        auto total_elems = durations.size();
        auto index = (percentile * total_elems)/100.f;
        index = lround(index) - 1; //array bounds
        return durations[index];
    };

    ASSERT_EQ(computeNthPercentile(70), 701);
    ASSERT_EQ(computeNthPercentile(95), 950);
    ASSERT_EQ(computeNthPercentile(99), 990);
}