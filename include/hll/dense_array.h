// Source : https://github.com/iwiwi/hyperloglog-hip
#ifndef HYPERLOGLOG_HIP_DENSE_ARRAY_H_
#define HYPERLOGLOG_HIP_DENSE_ARRAY_H_

#include <algorithm>
#include <climits>
#include <cstdint>
#include <memory>
#include <type_traits>

namespace hyperloglog_hip {
    template<size_t NumRegisterBits, typename Value = uint8_t>
    class dense_array {
    public:
        typedef Value value_type;

        dense_array(size_t num_registers)
                : data_(new value_type[data_length(num_registers)]()) {}

        value_type get(size_t pos) {
            const size_t b = pos * num_register_bits();
            const size_t i1 = b / num_value_bits();
            const size_t o1 = b - i1 * num_value_bits();
            const size_t n1 = num_value_bits() - o1;
            value_type v = data_[i1] >> o1;

            if (n1 > num_register_bits()) {
                v &= (value_type(1) << num_register_bits()) - 1;
            }
            else if (n1 < num_register_bits()) {
                const size_t i2 = i1 + 1;
                const size_t n2 = num_register_bits() - n1;
                v |= (data_[i2] & ((value_type(1) << n2) - 1)) << n1;
            }
            return v;
        }

        void set(size_t pos, value_type val) {
            const size_t b = pos * num_register_bits();

            const size_t i1 = b / num_value_bits();
            const size_t o1 = b - i1 * num_value_bits();
            const size_t n1 = std::min(num_value_bits() - o1, num_register_bits());
            data_[i1] &= value_type(-1) ^ (((value_type(1) << n1) - 1) << o1);
            data_[i1] |= val << o1;

            if (n1 < num_register_bits()) {
                const size_t i2 = i1 + 1;
                const size_t n2 = num_register_bits() - n1;
                data_[i2] &= value_type(-1) ^ ((value_type(1) << n2) - 1);
                data_[i2] |= val >> n1;
            }
        }

    private:
        std::unique_ptr<value_type[]> data_;

        static constexpr size_t num_register_bits() {
            return NumRegisterBits;
        }

        static constexpr size_t num_value_bits() {
            return sizeof(Value) * CHAR_BIT;
        }

        static constexpr size_t data_length(size_t num_registers) {
            return (num_registers * num_register_bits() + num_value_bits() - 1) / num_value_bits();
        }

        static_assert(std::is_unsigned<value_type>::value,
                      "Value should be an unsigned integral type.");

        static_assert(sizeof(value_type) * CHAR_BIT >= NumRegisterBits,
                      "Value should have at least NumRegisterBits bits.");
    };

    template<typename Value>
    class dense_array_primitive {
    public:
        typedef Value value_type;

        dense_array_primitive(size_t size) : data_(new value_type[size]()) {}
        virtual ~dense_array_primitive() {}

        value_type get(size_t pos) const {
            return data_[pos];
        }

        void set(size_t pos, value_type val) {
            data_[pos] = val;
        }

    private:
        std::unique_ptr<value_type[]> data_;
    };

    template<>
    class dense_array<8, uint8_t> : public dense_array_primitive<uint8_t> {
    public:
        dense_array(size_t size) : dense_array_primitive(size) {}
    };

    template<>
    class dense_array<16, uint16_t> : public dense_array_primitive<uint16_t> {
    public:
        dense_array(size_t size) : dense_array_primitive(size) {}
    };

    template<>
    class dense_array<32, uint32_t> : public dense_array_primitive<uint32_t> {
    public:
        dense_array(size_t size) : dense_array_primitive(size) {}
    };

    template<>
    class dense_array<64, uint64_t> : public dense_array_primitive<uint64_t> {
    public:
        dense_array(size_t size) : dense_array_primitive(size) {}
    };
}  // namespace hyperloglog_hip

#endif  // HYEPRLOGLOG_HIP_DENSE_ARRAY_H_
