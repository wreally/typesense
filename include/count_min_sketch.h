/**
    Daniel Alabi
    Count-Min Sketch Implementation based on paper by
    Muthukrishnan and Cormode, 2004
    Source: https://github.com/alabid/countminsketch
**/

#include <cmath>
#include <ctime>
#include <cstdint>

class CountMinSketch {
private:
    uint32_t w, d;
    float eps;
    float gamma;
    uint64_t **C;
    int **hashes;

    void genajbj(int* hash, int i);

public:

    CountMinSketch(float ep, float gamm) {
        eps = (0.009 <= ep && ep < 1) ? ep : 0.01;
        gamma = (0 < gamm && gamm < 1) ? gamm : 0.1;

        w = ceil(exp(1) / eps);
        d = ceil(log(1 / gamma));

        C = new uint64_t *[d];
        for (size_t i = 0; i < d; i++) {
            C[i] = new uint64_t[w]{0};
        }

        srand(time(NULL));

        hashes = new int* [d];
        for (size_t i = 0; i < d; i++) {
            hashes[i] = new int[2];
            genajbj(hashes[i], i);
        }
    }

    ~CountMinSketch() {
        for (size_t i = 0; i < d; i++) {
            delete[] C[i];
        }
        delete[] C;

        for (size_t i = 0; i < d; i++) {
            delete[] hashes[i];
        }
        delete[] hashes;
    }

    void update(uint64_t item, uint32_t count) {
        count = std::max((uint32_t) 1, count);

        for (size_t j = 0; j < d; j++) {
            size_t index = (hashes[j][0] * item + hashes[j][1]) % w;
            C[j][index] += count;
        }
    }

    uint32_t estimate(uint64_t item) {
        uint64_t min_val = UINT64_MAX;

        for (size_t j = 0; j < d; j++) {
            size_t index = (hashes[j][0] * item + hashes[j][1]) % w;
            min_val = std::min(min_val, C[j][index]);
        }

        return min_val;
    }
};

inline void CountMinSketch::genajbj(int* hash, int i) {
    constexpr auto LONG_PRIME = 32993;
    hash[0] = int(float(rand())*float(LONG_PRIME)/float(RAND_MAX) + 1);
    hash[1] = int(float(rand())*float(LONG_PRIME)/float(RAND_MAX) + 1);
}
