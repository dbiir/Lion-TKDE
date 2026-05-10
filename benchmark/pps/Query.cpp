// benchmark/pps/Query.cpp
#include "benchmark/pps/Query.h"
#include "benchmark/pps/Database.h"
DECLARE_double(zipf);

namespace star {
namespace pps {

// Out-of-class definitions required for ODR-use of static constexpr members (C++14)
constexpr std::size_t parts::tableID;
constexpr std::size_t products::tableID;
constexpr std::size_t suppliers::tableID;

static const char* txnTypeName(PPSTxnType t) {
  switch (t) {
    case GET_PART:            return "GET_PART";
    case GET_PRODUCT:         return "GET_PRODUCT";
    case GET_SUPPLIER:        return "GET_SUPPLIER";
    case GET_PART_BY_PRODUCT: return "GET_PART_BY_PRODUCT";
    case GET_PART_BY_SUPPLIER:return "GET_PART_BY_SUPPLIER";
    case ORDER_PRODUCT:       return "ORDER_PRODUCT";
    case UPDATE_PRODUCT_PART: return "UPDATE_PRODUCT_PART";
    case UPDATE_PART:         return "UPDATE_PART";
    default:                  return "UNKNOWN";
  }
}

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
          std::sort(query.part_keys.begin(), query.part_keys.end());
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

  // 1% sampling log
  if (random.uniform_dist(1, 100) == 1) {
    std::string msg = "zipf=" + std::to_string(context.isUniform ? 0.0 : FLAGS_zipf)
                    + " txn=" + txnTypeName(query.txn_type);
    switch (query.txn_type) {
      case GET_PART:
      case UPDATE_PART:
        msg += " parts:key=" + std::to_string(query.part_key)
             + ",pid=" + std::to_string(context.getPartitionID(0, query.part_key));
        break;
      case GET_PRODUCT:
      case UPDATE_PRODUCT_PART:
        msg += " products:key=" + std::to_string(query.product_key)
             + ",pid=" + std::to_string(context.getPartitionID(1, query.product_key));
        break;
      case GET_SUPPLIER:
        msg += " suppliers:key=" + std::to_string(query.supplier_key)
             + ",pid=" + std::to_string(context.getPartitionID(2, query.supplier_key));
        break;
      case GET_PART_BY_PRODUCT:
      case ORDER_PRODUCT:
        msg += " products:key=" + std::to_string(query.product_key)
             + ",pid=" + std::to_string(context.getPartitionID(1, query.product_key));
        break;
      case GET_PART_BY_SUPPLIER:
        msg += " suppliers:key=" + std::to_string(query.supplier_key)
             + ",pid=" + std::to_string(context.getPartitionID(2, query.supplier_key));
        break;
      default:
        msg += " unknown_txn_type";
        break;
    }
    if (!query.part_keys.empty()) {
      msg += " part_keys=[";
      for (std::size_t i = 0; i < query.part_keys.size(); i++) {
        if (i > 0) msg += ",";
        msg += std::to_string(query.part_keys[i])
             + "(pid=" + std::to_string(context.getPartitionID(0, query.part_keys[i])) + ")";
      }
      msg += "]";
    }
    // LOG(INFO) << msg;
  }

  return query;
}

} // namespace pps
} // namespace star
