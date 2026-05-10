// benchmark/pps/Query.cpp
#include "benchmark/pps/Query.h"
#include "benchmark/pps/Database.h"

namespace star {
namespace pps {

// Out-of-class definitions required for ODR-use of static constexpr members (C++14)
constexpr std::size_t parts::tableID;
constexpr std::size_t products::tableID;
constexpr std::size_t suppliers::tableID;

PPSQuery makePPSQuery::operator()(const Context &context, std::size_t partitionID,
                                   Random &random, Database &db) const {
  PPSQuery query;

  int crossRoll = random.uniform_dist(1, 100);
  bool isCross  = (crossRoll <= context.crossPartitionProbability &&
                   context.partition_num > 1);

  int rwRoll   = random.uniform_dist(1, 100);
  bool isWrite = (rwRoll > context.readWriteRatio);

  // Override partition selection with Zipf when skewed workload is enabled.
  // This makes some partitions "hotter" than others, replacing the uniform
  // random partition_id chosen by the upstream generator.
  if (!context.isUniform) {
    partitionID = static_cast<std::size_t>(
        Zipf::partitionZipf().value(random.next_double()));
  }

  if (isCross) {
    if (isWrite) {
      query.txn_type    = ORDER_PRODUCT;
      query.product_key = selectProductKey(context, random);
      if (context.isUniform) {
        query.part_keys = db.getProductParts(query.product_key);
      } else {
        // Use the same Zipf offset as local transactions so cross-partition writes
        // conflict with local reads/writes on the same hot keys.
        std::size_t kpp = context.getKeysPerPartition(0);
        std::size_t offset = static_cast<std::size_t>(
            Zipf::offsetZipf().value(random.next_double())) % kpp;
        for (std::size_t p = 0; p < context.partition_num; p++) {
          query.part_keys.push_back(static_cast<int32_t>(p * kpp + offset));
        }
      }
    } else {
      int which = random.uniform_dist(0, 1);
      if (which == 0) {
        query.txn_type    = GET_PART_BY_PRODUCT;
        query.product_key = selectProductKey(context, random);
        query.part_keys   = db.getProductParts(query.product_key);
      } else {
        query.txn_type     = GET_PART_BY_SUPPLIER;
        query.supplier_key = selectSupplierKey(context, random);
        query.part_keys    = db.getSupplierParts(query.supplier_key);
      }
    }
  } else {
    if (isWrite) {
      int which = random.uniform_dist(0, 1);
      if (which == 0) {
        query.txn_type    = UPDATE_PRODUCT_PART;
        query.product_key = selectLocalProductKey(context, random, partitionID);
      } else {
        query.txn_type = UPDATE_PART;
        if (context.isUniform) {
          query.part_key = selectLocalPartKey(context, random, partitionID);
        } else {
          // Zipf mode: update partsPerProduct keys from the local partition,
          // all drawn from offsetZipf. High theta concentrates writes on the
          // same hot offsets across threads -> 2PL write-write conflicts ->
          // throughput decreases monotonically with theta even at cross_ratio=0.
          std::size_t kpp = context.getKeysPerPartition(0);
          std::size_t lo  = partitionID * kpp;
          for (std::size_t j = 0; j < context.partsPerProduct; j++) {
            std::size_t offset = static_cast<std::size_t>(
                Zipf::offsetZipf().value(random.next_double())) % kpp;
            query.part_keys.push_back(static_cast<int32_t>(lo + offset));
          }
        }
      }
    } else {
      int which = random.uniform_dist(0, 2);
      if (which == 0) {
        query.txn_type = GET_PART;
        query.part_key = selectLocalPartKey(context, random, partitionID);
      } else if (which == 1) {
        query.txn_type    = GET_PRODUCT;
        query.product_key = selectLocalProductKey(context, random, partitionID);
      } else {
        query.txn_type     = GET_SUPPLIER;
        query.supplier_key = selectLocalSupplierKey(context, random, partitionID);
      }
    }
  }

  return query;
}

} // namespace pps
} // namespace star
