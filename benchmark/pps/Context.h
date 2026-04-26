// benchmark/pps/Context.h
#pragma once

#include "core/Context.h"
#include <glog/logging.h>

namespace star {
namespace pps {

class Context : public star::Context {
public:
  // Returns partition ID for a key in a given table (range partitioning)
  std::size_t getPartitionID(std::size_t table_id, std::size_t key) const {
    return key / getKeysPerPartition(table_id);
  }

  // Parts: keysPerPartition; Products/Suppliers: keysPerPartition/10
  std::size_t getKeysPerPartition(std::size_t table_id) const {
    if (table_id == 0) return keysPerPartition;
    return keysPerPartition / 10;
  }

  std::size_t getTotalKeys(std::size_t table_id) const {
    return getKeysPerPartition(table_id) * partition_num;
  }

  Context get_single_partition_context() const {
    Context c = *this;
    c.crossPartitionProbability = 0;
    c.operation_replication = this->operation_replication;
    c.star_sync_in_single_master_phase = false;
    return c;
  }

  Context get_cross_partition_context() const {
    Context c = *this;
    c.crossPartitionProbability = 100;
    c.operation_replication = false;
    c.star_sync_in_single_master_phase = this->star_sync_in_single_master_phase;
    return c;
  }

public:
  int readWriteRatio = 80;            // out of 100: probability a txn is read-only
  int crossPartitionProbability = 0;  // out of 100: probability a txn is multi-partition

  std::size_t keysPerPartition = 200000;  // parts per partition; products/suppliers = /10

  std::size_t partsPerProduct  = 10;  // parts each product uses (USES relation)
  std::size_t partsPerSupplier = 10;  // parts each supplier supplies (SUPPLIES relation)

  bool isUniform = true;  // false => use Zipf for part key selection
};

} // namespace pps
} // namespace star
