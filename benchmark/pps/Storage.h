// benchmark/pps/Storage.h
#pragma once

#include "benchmark/pps/Schema.h"

namespace star {
namespace pps {

static constexpr std::size_t PPS_MAX_PARTS_PER_TXN = 100;

struct Storage {
  Storage() {
    memset(parts_keys,      0, sizeof(parts_keys));
    memset(parts_values,    0, sizeof(parts_values));
    memset(&product_key,    0, sizeof(product_key));
    memset(&product_value,  0, sizeof(product_value));
    memset(&supplier_key,   0, sizeof(supplier_key));
    memset(&supplier_value, 0, sizeof(supplier_value));
  }

  // Parts: up to PPS_MAX_PARTS_PER_TXN accessed per transaction
  parts::key   parts_keys[PPS_MAX_PARTS_PER_TXN];
  parts::value parts_values[PPS_MAX_PARTS_PER_TXN];

  // Product: at most 1 per transaction
  products::key   product_key;
  products::value product_value;

  // Supplier: at most 1 per transaction
  suppliers::key   supplier_key;
  suppliers::value supplier_value;
};

} // namespace pps
} // namespace star
