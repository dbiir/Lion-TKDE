// benchmark/pps/Query.h
#pragma once

#include "benchmark/pps/Context.h"
#include "benchmark/pps/Random.h"
#include "benchmark/pps/Schema.h"
#include "common/Zipf.h"
#include <vector>

namespace star {
namespace pps {

// Forward declare Database so makePPSQuery can take a Database& without circular include
class Database;

enum PPSTxnType {
  GET_PART = 0,           // read-only, single-partition: read PARTS[part_key]
  GET_PRODUCT,            // read-only, single-partition: read PRODUCTS[product_key]
  GET_SUPPLIER,           // read-only, single-partition: read SUPPLIERS[supplier_key]
  GET_PART_BY_PRODUCT,    // read-only, multi-partition: read PRODUCTS + PARTS[p1..pN]
  GET_PART_BY_SUPPLIER,   // read-only, multi-partition: read SUPPLIERS + PARTS[p1..pN]
  ORDER_PRODUCT,          // write, multi-partition: read PRODUCTS + update PARTS[p1..pN].amount--
  UPDATE_PRODUCT_PART,    // write, single-partition: update PRODUCTS[product_key].PR_F01
  UPDATE_PART,            // write, single-partition: update PARTS[part_key].P_AMOUNT++
};

struct PPSQuery {
  PPSTxnType txn_type = GET_PART;

  int32_t part_key     = 0;
  int32_t product_key  = 0;
  int32_t supplier_key = 0;

  // Pre-resolved part keys for multi-part transactions
  std::vector<int32_t> part_keys;

  std::vector<uint64_t> record_keys; // for migration/replay
};

class makePPSQuery {
public:
  // Select keys using Zipf or uniform distribution
  static int32_t selectPartKey(const Context &context, Random &random) {
    std::size_t total = context.getTotalKeys(0);
    if (context.isUniform) {
      return static_cast<int32_t>(random.uniform_dist(0, total - 1));
    } else {
      return static_cast<int32_t>(Zipf::globalZipf().value(random));
    }
  }

  static int32_t selectProductKey(const Context &context, Random &random) {
    std::size_t total = context.getTotalKeys(1);
    return static_cast<int32_t>(random.uniform_dist(0, total - 1));
  }

  static int32_t selectSupplierKey(const Context &context, Random &random) {
    std::size_t total = context.getTotalKeys(2);
    return static_cast<int32_t>(random.uniform_dist(0, total - 1));
  }

  // Local part key: biased toward home partition
  static int32_t selectLocalPartKey(const Context &context, Random &random,
                                     std::size_t partitionID) {
    std::size_t kpp = context.getKeysPerPartition(0);
    std::size_t lo  = partitionID * kpp;
    std::size_t hi  = lo + kpp - 1;
    if (context.isUniform) {
      return static_cast<int32_t>(random.uniform_dist(lo, hi));
    } else {
      int32_t k = static_cast<int32_t>(Zipf::globalZipf().value(random));
      return static_cast<int32_t>(lo + static_cast<std::size_t>(k) % kpp);
    }
  }

  static int32_t selectLocalProductKey(const Context &context, Random &random,
                                        std::size_t partitionID) {
    std::size_t kpp = context.getKeysPerPartition(1);
    std::size_t lo  = partitionID * kpp;
    std::size_t hi  = lo + kpp - 1;
    return static_cast<int32_t>(random.uniform_dist(lo, hi));
  }

  static int32_t selectLocalSupplierKey(const Context &context, Random &random,
                                         std::size_t partitionID) {
    std::size_t kpp = context.getKeysPerPartition(2);
    std::size_t lo  = partitionID * kpp;
    std::size_t hi  = lo + kpp - 1;
    return static_cast<int32_t>(random.uniform_dist(lo, hi));
  }

  // Primary query generation: called at transaction creation time
  // Database is forward-declared; full definition available via Database.h at call site
  PPSQuery operator()(const Context &context, std::size_t partitionID,
                      Random &random, Database &db) const;

  // Reconstruct from flat key vector (for unpack_transaction / migration replay)
  // Encoding: keys[0] = (txn_type << 48) | (table_id << 32) | raw_key
  //           keys[i>0] = (table_id << 32) | raw_key
  PPSQuery operator()(const std::vector<size_t> &keys,
                      const std::vector<bool>   &updates) const {
    PPSQuery query;
    if (keys.empty()) return query;

    // Extract txn_type from high 16 bits of first key
    query.txn_type = static_cast<PPSTxnType>(keys[0] >> 48);

    for (std::size_t i = 0; i < keys.size(); i++) {
      uint64_t encoded = keys[i];
      if (i == 0) encoded = encoded & 0x0000FFFFFFFFFFFFull; // strip txn_type
      size_t   table_id = (encoded >> 32) & 0xFFFF;
      int32_t  raw_key  = static_cast<int32_t>(encoded & 0xFFFFFFFF);

      if (table_id == pps::parts::tableID) {
        if (i == 0) query.part_key = raw_key;
        else        query.part_keys.push_back(raw_key);
      } else if (table_id == pps::products::tableID) {
        query.product_key = raw_key;
      } else if (table_id == pps::suppliers::tableID) {
        query.supplier_key = raw_key;
      }
    }
    return query;
  }
};

} // namespace pps
} // namespace star
