// benchmark/pps/Query.cpp
#include "benchmark/pps/Query.h"
#include "benchmark/pps/Database.h"

namespace star {
namespace pps {

PPSQuery makePPSQuery::operator()(const Context &context, std::size_t partitionID,
                                   Random &random, Database &db) const {
  PPSQuery query;

  int crossRoll = random.uniform_dist(1, 100);
  bool isCross  = (crossRoll <= context.crossPartitionProbability &&
                   context.partition_num > 1);

  int rwRoll   = random.uniform_dist(1, 100);
  bool isWrite = (rwRoll > context.readWriteRatio);

  if (isCross) {
    if (isWrite) {
      query.txn_type    = ORDER_PRODUCT;
      query.product_key = selectProductKey(context, random);
      query.part_keys   = db.getProductParts(query.product_key);
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
        query.part_key = selectLocalPartKey(context, random, partitionID);
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
