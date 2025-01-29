#include <gtest/gtest.h>
#include <index.h>
#include "topster.h"
#include "match_score.h"
#include <fstream>

TEST(TopsterTest, MaxIntValues) {
    Topster<KV> topster(5);

    struct {
        uint16_t query_index;
        uint64_t key;
        uint64_t match_score;
        int64_t primary_attr;
        int64_t secondary_attr;
    } data[14] = {
        {0, 1, 11, 20, 30},
        {0, 1, 12, 20, 32},
        {0, 2, 4, 20, 30},
        {2, 3, 7, 20, 30},
        {0, 4, 14, 20, 30},
        {1, 5, 9, 20, 30},
        {1, 5, 10, 20, 32},
        {1, 5, 9, 20, 30},
        {0, 6, 6, 20, 30},
        {2, 7, 6, 22, 30},
        {2, 7, 6, 22, 30},
        {1, 8, 9, 20, 30},
        {0, 9, 8, 20, 30},
        {3, 10, 5, 20, 30},
    };

    for(int i = 0; i < 14; i++) {
        int64_t scores[3];
        scores[0] = int64_t(data[i].match_score);
        scores[1] = data[i].primary_attr;
        scores[2] = data[i].secondary_attr;

        KV kv(data[i].query_index, data[i].key, data[i].key, 0, scores);
        topster.add(&kv);
    }

    topster.sort();

    std::vector<uint64_t> ids = {4, 1, 5, 8, 9};

    for(uint32_t i = 0; i < topster.size; i++) {
        EXPECT_EQ(ids[i], topster.getKeyAt(i));

        if(ids[i] == 1) {
            EXPECT_EQ(12, (int) topster.getKV(i)->scores[topster.getKV(i)->match_score_index]);
        }

        if(ids[i] == 5) {
            EXPECT_EQ(10, (int) topster.getKV(i)->scores[topster.getKV(i)->match_score_index]);
        }
    }
}

TEST(TopsterTest, StableSorting) {
    // evaluate if the positions of the documents in Topster<1000> is the same in Topster 250, 500 and 750

    std::ifstream infile("test/resources/record_values.txt");
    std::string line;
    std::vector<std::pair<uint64_t, int64_t>> records;

    while (std::getline(infile, line)) {
        std::vector<std::string> parts;
        StringUtils::split(line, parts, ",");
        uint64_t key = std::stoll(parts[0]);
        records.emplace_back(key, std::stoi(parts[1]));
    }

    infile.close();

    Topster<KV> topster1K(1000);

    for(auto id_score: records) {
        int64_t scores[3] = {id_score.second, 0, 0};
        KV kv(0, id_score.first, id_score.first, 0, scores);
        topster1K.add(&kv);
    }

    topster1K.sort();

    std::vector<uint64_t> record_ids;

    for(uint32_t i = 0; i < topster1K.size; i++) {
        record_ids.push_back(topster1K.getKeyAt(i));
    }

    // check on Topster<250>
    Topster<KV> topster250(250);

    for(auto id_score: records) {
        int64_t scores[3] = {id_score.second, 0, 0};
        KV kv(0, id_score.first, id_score.first, 0, scores);
        topster250.add(&kv);
    }

    topster250.sort();

    for(uint32_t i = 0; i < topster250.size; i++) {
        ASSERT_EQ(record_ids[i], topster250.getKeyAt(i));
    }

    // check on Topster<500>
    Topster<KV> topster500(500);

    for(auto id_score: records) {
        int64_t scores[3] = {id_score.second, 0, 0};
        KV kv(0, id_score.first, id_score.first, 0, scores);
        topster500.add(&kv);
    }

    topster500.sort();

    for(uint32_t i = 0; i < topster500.size; i++) {
        ASSERT_EQ(record_ids[i], topster500.getKeyAt(i));
    }

    // check on Topster<750>
    Topster<KV> topster750(750);

    for(auto id_score: records) {
        int64_t scores[3] = {id_score.second, 0, 0};
        KV kv(0, id_score.first, id_score.first, 0, scores);
        topster750.add(&kv);
    }

    topster750.sort();

    for(uint32_t i = 0; i < topster750.size; i++) {
        ASSERT_EQ(record_ids[i], topster750.getKeyAt(i));
    }
}

TEST(TopsterTest, MaxFloatValues) {
    Topster<KV> topster(5);

    struct {
        uint16_t query_index;
        uint64_t key;
        uint64_t match_score;
        float primary_attr;
        int64_t secondary_attr;
    } data[12] = {
        {0, 1, 11, 1.09, 30},
        {0, 2, 11, -20, 30},
        {2, 3, 11, -20, 30},
        {0, 4, 11, 7.812, 30},
        {0, 4, 11, 7.912, 30},
        {1, 5, 11, 0.0, 34},
        {0, 6, 11, -22, 30},
        {2, 7, 11, -22, 30},
        {1, 8, 11, -9.998, 30},
        {1, 8, 11, -9.998, 30},
        {0, 9, 11, -9.999, 30},
        {3, 10, 11, -20, 30},
    };

    for(int i = 0; i < 12; i++) {
        int64_t scores[3];
        scores[0] = int64_t(data[i].match_score);
        scores[1] = Index::float_to_int64_t(data[i].primary_attr);
        scores[2] = data[i].secondary_attr;

        KV kv(data[i].query_index, data[i].key, data[i].key, 0, scores);
        topster.add(&kv);
    }

    topster.sort();

    std::vector<uint64_t> ids = {4, 1, 5, 8, 9};

    for(uint32_t i = 0; i < topster.size; i++) {
        EXPECT_EQ(ids[i], topster.getKeyAt(i));
    }
}

TEST(TopsterTest, DistinctIntValues) {
    Topster<KV> dist_topster(5, 2, false, false);

    struct {
        uint16_t query_index;
        uint64_t distinct_key;
        uint64_t match_score;
        int64_t primary_attr;
        int64_t secondary_attr;
    } data[14] = {
        {0, 1, 11, 20, 30},
        {0, 1, 12, 20, 32},
        {0, 2, 4, 20, 30},
        {2, 3, 7, 20, 30},
        {0, 4, 14, 20, 30},
        {1, 5, 9, 20, 30},
        {1, 5, 10, 20, 32},
        {1, 5, 9, 20, 30},
        {0, 6, 6, 20, 30},
        {2, 7, 6, 22, 30},
        {2, 7, 6, 22, 30},
        {1, 8, 9, 20, 30},
        {0, 9, 8, 20, 30},
        {3, 10,  5, 20, 30},
    };

    for(int i = 0; i < 14; i++) {
        int64_t scores[3];
        scores[0] = int64_t(data[i].match_score);
        scores[1] = data[i].primary_attr;
        scores[2] = data[i].secondary_attr;

        KV kv(data[i].query_index, i+100, data[i].distinct_key, 0, scores);
        dist_topster.add(&kv);
    }

    dist_topster.sort();

    std::vector<uint64_t> distinct_ids = {4, 1, 8, 5, 9};

    for(uint32_t i = 0; i < dist_topster.size; i++) {
        EXPECT_EQ(distinct_ids[i], dist_topster.getDistinctKeyAt(i));

        if(distinct_ids[i] == 1) {
            EXPECT_EQ(12, (int) dist_topster.getKV(i)->scores[dist_topster.getKV(i)->match_score_index]);
            EXPECT_EQ(2, dist_topster.group_kv_map[dist_topster.getDistinctKeyAt(i)]->size);
            EXPECT_EQ(12, dist_topster.group_kv_map[dist_topster.getDistinctKeyAt(i)]->getKV(0)->scores[0]);
            EXPECT_EQ(11, dist_topster.group_kv_map[dist_topster.getDistinctKeyAt(i)]->getKV(1)->scores[0]);
        }

        if(distinct_ids[i] == 5) {
            EXPECT_EQ(9, (int) dist_topster.getKV(i)->scores[dist_topster.getKV(i)->match_score_index]);
            EXPECT_EQ(2, dist_topster.group_kv_map[dist_topster.getDistinctKeyAt(i)]->size);
            EXPECT_EQ(10, dist_topster.group_kv_map[dist_topster.getDistinctKeyAt(i)]->getKV(0)->scores[0]);
            EXPECT_EQ(9, dist_topster.group_kv_map[dist_topster.getDistinctKeyAt(i)]->getKV(1)->scores[0]);
        }
    }

    Topster<KV> dist_topster_first_pass(7, 2, true, true);

    for(int i = 0; i < 14; i++) {
        int64_t scores[3];
        scores[0] = int64_t(data[i].match_score);
        scores[1] = data[i].primary_attr;
        scores[2] = data[i].secondary_attr;

        KV kv(data[i].query_index, i+100, data[i].distinct_key, 0, scores);
        dist_topster_first_pass.add(&kv);
    }

    dist_topster_first_pass.sort();

    distinct_ids = {4, 1, 5, 8, 9, 3, 7};
    std::vector<uint64_t> ids = {104, 101, 106, 111, 112, 103, 110};

    for(uint32_t i = 0; i < dist_topster_first_pass.size; i++) {
        ASSERT_EQ(distinct_ids[i], dist_topster_first_pass.getDistinctKeyAt(i));
        ASSERT_EQ(ids[i], dist_topster_first_pass.getKeyAt(i));
    }

    ASSERT_TRUE(dist_topster_first_pass.group_kv_map.empty());
    ASSERT_EQ(10, dist_topster_first_pass.hyperloglog_counter.count());
}