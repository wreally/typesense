#pragma once

#include <cstdint>
#include <climits>
#include <cstdio>
#include <algorithm>
#include <unordered_map>
#include <field.h>
#include "filter_result_iterator.h"
#include "hll/distinct_counter.h"

struct KV {
    int8_t match_score_index{};
    uint16_t query_index{};
    uint16_t array_index{};
    uint64_t key{};
    uint64_t distinct_key{};
    int64_t scores[3]{};  // match score + 2 custom attributes

    // only to be used in hybrid search
    float vector_distance = -1.0f;
    int64_t text_match_score = 0;

    // to be used only in final aggregation
    uint64_t* query_indices = nullptr;

    std::map<std::string, reference_filter_result_t> reference_filter_results;

    KV(uint16_t queryIndex, uint64_t key, uint64_t distinct_key, int8_t match_score_index, const int64_t *scores,
       std::map<std::string, reference_filter_result_t>  reference_filter_results = {}):
            match_score_index(match_score_index), query_index(queryIndex), array_index(0), key(key),
            distinct_key(distinct_key), reference_filter_results(std::move(reference_filter_results)) {
        this->scores[0] = scores[0];
        this->scores[1] = scores[1];
        this->scores[2] = scores[2];

        if(match_score_index >= 0){
            this->text_match_score = scores[match_score_index];
        }
    }

    KV() = default;

    KV(KV& kv) = default;

    KV(KV&& kv) noexcept : match_score_index(kv.match_score_index),
                 query_index(kv.query_index), array_index(kv.array_index),
                 key(kv.key), distinct_key(kv.distinct_key) {
                    
        scores[0] = kv.scores[0];
        scores[1] = kv.scores[1];
        scores[2] = kv.scores[2];

        query_indices = kv.query_indices;
        kv.query_indices = nullptr;

        vector_distance = kv.vector_distance;
        text_match_score = kv.text_match_score;

        reference_filter_results = std::move(kv.reference_filter_results);
    }

    KV& operator=(KV&& kv) noexcept  {
        if (this != &kv) {
            match_score_index = kv.match_score_index;
            query_index = kv.query_index;
            array_index = kv.array_index;
            key = kv.key;
            distinct_key = kv.distinct_key;

            scores[0] = kv.scores[0];
            scores[1] = kv.scores[1];
            scores[2] = kv.scores[2];

            delete[] query_indices;
            query_indices = kv.query_indices;
            kv.query_indices = nullptr;

            vector_distance = kv.vector_distance;
            text_match_score = kv.text_match_score;

            reference_filter_results = std::move(kv.reference_filter_results);
        }

        return *this;
    }

    KV& operator=(KV& kv) noexcept  {
        if (this != &kv) {
            match_score_index = kv.match_score_index;
            query_index = kv.query_index;
            array_index = kv.array_index;
            key = kv.key;
            distinct_key = kv.distinct_key;

            scores[0] = kv.scores[0];
            scores[1] = kv.scores[1];
            scores[2] = kv.scores[2];

            delete[] query_indices;
            query_indices = kv.query_indices;
            kv.query_indices = nullptr;

            vector_distance = kv.vector_distance;
            text_match_score = kv.text_match_score;

            reference_filter_results.clear();

            for (const auto& item: kv.reference_filter_results) {
                reference_filter_results[item.first] = item.second;
            }
        }

        return *this;
    }

    ~KV() {
        delete [] query_indices;
        query_indices = nullptr;
    }

    static bool is_greater(const KV* i, const KV* j) {
        return std::tie(i->scores[0], i->scores[1], i->scores[2], i->key) >
                    std::tie(j->scores[0], j->scores[1], j->scores[2], j->key);
    }

    static bool is_smaller(const struct KV* i, const struct KV* j) {
        return std::tie(i->scores[0], i->scores[1], i->scores[2], i->key) <
                    std::tie(j->scores[0], j->scores[1], j->scores[2], j->key);
    }

    static bool is_greater_kv_group(const std::vector<KV*>& i, const std::vector<KV*>& j) {
        return std::tie(i[0]->scores[0], i[0]->scores[1], i[0]->scores[2], i[0]->key) >
                    std::tie(j[0]->scores[0], j[0]->scores[1], j[0]->scores[2], j[0]->key);
    }

    static constexpr uint64_t get_key(const KV* kv) {
        return kv->key;
    }

    static constexpr uint64_t get_distinct_key(const KV* kv) {
        return kv->distinct_key;
    }
};

struct Union_KV : public KV {
    uint32_t search_index{};

    Union_KV(KV& kv, uint32_t search_index) : KV(kv.query_index, kv.key, kv.distinct_key, kv.match_score_index, kv.scores),
                                               search_index(search_index) {
        reference_filter_results = std::move(kv.reference_filter_results);
    }

    Union_KV() = default;

    Union_KV& operator=(Union_KV&& kv) noexcept  {
        if (this != &kv) {
            search_index = kv.search_index;
            KV::operator=(std::move(kv));
        }

        return *this;
    }

    Union_KV& operator=(Union_KV& kv) noexcept {
        if (this != &kv) {
            search_index = kv.search_index;
            KV::operator=(kv);
        }

        return *this;
    }

    static bool is_greater(const Union_KV* i, const Union_KV* j) {
        return std::tie(i->scores[0], i->scores[1], i->scores[2], i->search_index, i->key) >
                    std::tie(j->scores[0], j->scores[1], j->scores[2], j->search_index, j->key);
    }

    static bool is_smaller(const Union_KV* i, const Union_KV* j) {
        return std::tie(i->scores[0], i->scores[1], i->scores[2], i->search_index, i->key) <
                    std::tie(j->scores[0], j->scores[1], j->scores[2], j->search_index, j->key);
    }

    static constexpr uint64_t get_key(const Union_KV* union_kv) {
        return StringUtils::hash_combine(union_kv->search_index, union_kv->key);
    }

    static constexpr uint64_t get_distinct_key(const Union_KV* union_kv) {
        return StringUtils::hash_combine(union_kv->search_index, union_kv->distinct_key);
    }
};

struct pair_hash {
    template <class T1, class T2>
    std::size_t operator() (const std::pair<T1, T2> &pair) const {
        return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
    }
};

/*
* Remembers the max-K elements seen so far using a min-heap
*/
template <typename T, const auto& get_key = KV::get_key, const auto& get_distinct_key = KV::get_distinct_key,
            const auto& is_greater = KV::is_greater, const auto& is_smaller = KV::is_smaller>
struct Topster {
    const uint32_t MAX_SIZE;
    uint32_t size;

    T *data;
    T** kvs;

    std::unordered_map<uint64_t, T*> map;

    bool is_group_by_first_pass;
    size_t distinct;
    spp::sparse_hash_set<uint64_t> group_doc_seq_ids;
    spp::sparse_hash_map<uint64_t, Topster<T, get_key, get_distinct_key, is_greater, is_smaller>*> group_kv_map;

    bool should_count_distinct;
    hyperloglog_hip::distinct_counter<5> hyperloglog_counter = hyperloglog_hip::distinct_counter<5>(12);

    explicit Topster(size_t capacity): Topster(capacity, 0, false, false) {
    }

    explicit Topster(size_t capacity, size_t distinct, bool is_group_by_first_pass, bool should_count_distinct) :
                                                                MAX_SIZE(capacity), size(0), distinct(distinct),
                                                                is_group_by_first_pass(is_group_by_first_pass),
                                                                should_count_distinct(should_count_distinct) {
        // we allocate data first to get a memory block whose indices are then assigned to `kvs`
        // we use separate **kvs for easier pointer swaps
        data = new T[capacity];
        kvs = new T*[capacity];

        for(size_t i=0; i<capacity; i++) {
            data[i].match_score_index = 0;
            data[i].query_index = 0;
            data[i].array_index = i;
            data[i].key = 0;
            data[i].distinct_key = 0;
            kvs[i] = &data[i];
        }
    }

    ~Topster() {
        delete[] data;
        delete[] kvs;
        for(auto& kv: group_kv_map) {
            delete kv.second;
        }

        data = nullptr;
        kvs = nullptr;

        group_kv_map.clear();
    }

    static inline void swapMe(T** a, T** b) {
        T *temp = *a;
        *a = *b;
        *b = temp;

        uint16_t a_index = (*a)->array_index;
        (*a)->array_index = (*b)->array_index;
        (*b)->array_index = a_index;
    }

    int add(T* kv) {
        /*LOG(INFO) << "kv_map size: " << kv_map.size() << " -- kvs[0]: " << kvs[0]->scores[kvs[0]->match_score_index];
        for(auto& mkv: kv_map) {
            LOG(INFO) << "kv key: " << mkv.first << " => " << mkv.second->scores[mkv.second->match_score_index];
        }*/

        int ret = 1;
       
        bool less_than_min_heap = (size >= MAX_SIZE) && is_smaller(kv, kvs[0]);
        size_t heap_op_index = 0;
        const bool& is_group_by_second_pass = distinct && !is_group_by_first_pass;

        if(!is_group_by_second_pass && less_than_min_heap) {
            if (is_group_by_first_pass && should_count_distinct) {
                hyperloglog_counter.insert(get_distinct_key(kv));
            }
            // for non-distinct or first group_by pass, if incoming value is smaller than min-heap ignore
            return 0;
        }

        bool SIFT_DOWN = true;

        if(is_group_by_second_pass) {
            const auto& doc_seq_id_exists = 
                (group_doc_seq_ids.find(kv->key) != group_doc_seq_ids.end());
        
            if(doc_seq_id_exists) {
                ret = 2;
            }
            group_doc_seq_ids.emplace(kv->key);
            
            // Grouping cannot be a streaming operation, so aggregate the KVs associated with every group.
            auto kvs_it = group_kv_map.find(kv->distinct_key);
            if(kvs_it != group_kv_map.end()) {
                kvs_it->second->add(kv);
            } else {
                auto g_topster = new Topster<T, get_key, get_distinct_key, is_greater, is_smaller>(distinct, 0, false,
                                                                                                   false);
                g_topster->add(kv);
                group_kv_map.insert({kv->distinct_key, g_topster});
            }
            
            return ret;

        } else { // not distinct or first group_by pass
            //LOG(INFO) << "Searching for key: " << kv->key;

            const auto& key = is_group_by_first_pass ? get_distinct_key(kv) : get_key(kv);
            const auto& found_it = map.find(key);
            bool is_duplicate_key = (found_it != map.end());

            /*
               is_duplicate_key: SIFT_DOWN regardless of `size`.
               Else:
                   Do SIFT_UP if size < max_size
                   Else SIFT_DOWN
            */

            if(is_duplicate_key) {
                // Need to check if kv is greater than existing duplicate kv.
                auto existing_kv = found_it->second;
                //LOG(INFO) << "existing_kv: " << existing_kv->key << " -> " << existing_kv->match_score;

                bool smaller_than_existing = is_smaller(kv, existing_kv);
                if(smaller_than_existing) {
                    return 0;
                }

                SIFT_DOWN = true;

                // replace existing kv and sift down
                heap_op_index = existing_kv->array_index;
                map.erase(is_group_by_first_pass ? get_distinct_key(kvs[heap_op_index]) : get_key(kvs[heap_op_index]));
            } else {  // not duplicate
                if (is_group_by_first_pass && should_count_distinct) {
                    hyperloglog_counter.insert(key);
                }

                if(size < MAX_SIZE) {
                    // we just copy to end of array
                    SIFT_DOWN = false;
                    heap_op_index = size;
                    size++;
                } else {
                    // kv is guaranteed to be > min heap.
                    // we have to replace min heap element since array is full
                    SIFT_DOWN = true;
                    heap_op_index = 0;
                    map.erase(is_group_by_first_pass ? get_distinct_key(kvs[heap_op_index]) : get_key(kvs[heap_op_index]));
                }
            }

            // kv will be copied into the pointer at heap_op_index
            map.emplace(key, kvs[heap_op_index]);
        }

        // we have to replace the existing element in the heap and sift down
        kv->array_index = heap_op_index;
        *kvs[heap_op_index] = *kv;

        // sift up/down to maintain heap property

        if(SIFT_DOWN) {
            while ((2 * heap_op_index + 1) < size) {
                uint32_t next = (2 * heap_op_index + 1);  // left child
                if (next+1 < size && is_greater(kvs[next], kvs[next + 1])) {
                    // for min heap we compare with the minimum of children
                    next++;  // right child (2n + 2)
                }

                if (is_greater(kvs[heap_op_index], kvs[next])) {
                    swapMe(&kvs[heap_op_index], &kvs[next]);
                } else {
                    break;
                }

                heap_op_index = next;
            }
        } else {
            // SIFT UP
            while(heap_op_index > 0) {
                uint32_t parent = (heap_op_index - 1) / 2;
                if (is_greater(kvs[parent], kvs[heap_op_index])) {
                    swapMe(&kvs[heap_op_index], &kvs[parent]);
                    heap_op_index = parent;
                } else {
                    break;
                }
            }
        }

        return ret;
    }

    // topster must be sorted before iterated upon to remove dead array entries
    void sort() {
        if(!distinct || is_group_by_first_pass) {
            std::stable_sort(kvs, kvs + size, is_greater);
        }
    }

    void clear(){
        size = 0;
    }

    uint64_t getKeyAt(uint32_t index) {
        return kvs[index]->key;
    }

    uint64_t getDistinctKeyAt(uint32_t index) {
        return get_distinct_key(kvs[index]);
    }

    T* getKV(uint32_t index) {
        return kvs[index];
    }
};

