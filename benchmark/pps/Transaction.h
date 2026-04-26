// benchmark/pps/Transaction.h
#pragma once

#include "benchmark/pps/Database.h"
#include "benchmark/pps/Query.h"
#include "benchmark/pps/Schema.h"
#include "benchmark/pps/Storage.h"
#include "common/Operation.h"
#include "core/Defs.h"
#include "core/Partitioner.h"
#include "core/Table.h"
#include <glog/logging.h>
#include <set>
#include <unordered_map>

namespace star {
namespace pps {

template <class Transaction>
class PPSTransaction : public Transaction {
public:
  using DatabaseType = Database;
  using ContextType  = typename DatabaseType::ContextType;
  using RandomType   = typename DatabaseType::RandomType;
  using StorageType  = Storage;

  // Constructor 1: fresh transaction from context
  PPSTransaction(std::size_t coordinator_id, std::size_t partition_id,
                 std::atomic<uint32_t> &worker_status,
                 DatabaseType &db, const ContextType &context,
                 RandomType &random, Partitioner &partitioner,
                 Storage &storage, double /*cur_timestamp*/)
      : Transaction(coordinator_id, partition_id, partitioner),
        worker_status_(worker_status), db(db),
        context(context), random(random), storage(storage),
        partition_id(partition_id),
        query(makePPSQuery()(context, partition_id, random, db)) {}

  // Constructor 2: unpack from simpleTransaction
  PPSTransaction(std::size_t coordinator_id, std::size_t partition_id,
                 std::atomic<uint32_t> &worker_status,
                 DatabaseType &db, const ContextType &context,
                 RandomType &random, Partitioner &partitioner,
                 Storage &storage, simpleTransaction &simple_txn)
      : Transaction(coordinator_id, partition_id, partitioner),
        worker_status_(worker_status), db(db),
        context(context), random(random), storage(storage),
        partition_id(partition_id),
        query(makePPSQuery()(simple_txn.keys, simple_txn.update)) {
    is_transmit_request = simple_txn.is_transmit_request;
  }

  // Constructor 3: reset/clone from existing Transaction
  PPSTransaction(std::size_t coordinator_id, std::size_t partition_id,
                 std::atomic<uint32_t> &worker_status,
                 DatabaseType &db, const ContextType &context,
                 RandomType &random, Partitioner &partitioner,
                 Storage &storage, Transaction &txn)
      : Transaction(coordinator_id, partition_id, partitioner),
        worker_status_(worker_status), db(db),
        context(context), random(random), storage(storage),
        partition_id(partition_id),
        query(makePPSQuery()(txn.get_query(), txn.get_query_update())) {
    this->on_replica_id = txn.on_replica_id;
  }

  virtual ~PPSTransaction() override = default;

  // ---- Public overrides ----

  bool is_transmit_requests() override {
    return is_transmit_request;
  }

  ExecutorStatus get_worker_status() override {
    return static_cast<ExecutorStatus>(worker_status_.load());
  }

  std::size_t get_partition_id() override {
    return partition_id;
  }

  TransactionResult execute(std::size_t worker_id) override {
    issue_reads_updates();

    if (this->process_requests(worker_id)) {
      return TransactionResult::ABORT;
    }
    if (is_transmit_request) {
      return TransactionResult::TRANSMIT_REQUEST;
    }
    return applyWrites();
  }

  TransactionResult transmit_execute(std::size_t worker_id) override {
    issue_reads_updates();

    if (this->process_migrate_requests(worker_id)) {
      return TransactionResult::ABORT;
    }
    return TransactionResult::TRANSMIT_REQUEST;
  }

  TransactionResult prepare_read_execute(std::size_t worker_id) override {
    issue_reads_updates();
    return TransactionResult::READY_TO_COMMIT;
  }

  TransactionResult read_execute(std::size_t worker_id,
                                 ReadMethods local_read_only) override {
    TransactionResult ret = TransactionResult::READY_TO_COMMIT;
    switch (local_read_only) {
    case ReadMethods::REMOTE_READ_ONLY:
      if (this->process_read_only_requests(worker_id)) {
        ret = TransactionResult::ABORT;
      }
      break;
    case ReadMethods::LOCAL_READ:
      if (this->process_migrate_requests(worker_id)) {
        ret = TransactionResult::NOT_LOCAL_NORETRY;
      }
      break;
    case ReadMethods::REMOTE_READ_WITH_TRANSFER:
      if (this->process_requests(worker_id)) {
        ret = TransactionResult::ABORT;
      }
      break;
    case ReadMethods::REMASTER_ONLY:
      if (this->process_remaster_requests(worker_id)) {
        ret = TransactionResult::ABORT;
      }
      break;
    default:
      DCHECK(false);
      break;
    }
    return ret;
  }

  TransactionResult prepare_update_execute(std::size_t worker_id) override {
    return applyWrites();
  }

  void reset_query() override {
    query = makePPSQuery()(context, partition_id, random, db);
  }

  std::string print_raw_query_str() override {
    DCHECK(false);
    return "";
  }

  const std::vector<uint64_t> get_query() override {
    using T = uint64_t;
    std::vector<T> keys;
    uint64_t tt64 = static_cast<uint64_t>(query.txn_type);

    switch (query.txn_type) {
    case GET_PART:
      keys.push_back((tt64 << 48) |
                     (static_cast<uint64_t>(parts::tableID) << 32) |
                     static_cast<uint32_t>(query.part_key));
      break;
    case GET_PRODUCT:
      keys.push_back((tt64 << 48) |
                     (static_cast<uint64_t>(products::tableID) << 32) |
                     static_cast<uint32_t>(query.product_key));
      break;
    case GET_SUPPLIER:
      keys.push_back((tt64 << 48) |
                     (static_cast<uint64_t>(suppliers::tableID) << 32) |
                     static_cast<uint32_t>(query.supplier_key));
      break;
    case GET_PART_BY_PRODUCT:
      keys.push_back((tt64 << 48) |
                     (static_cast<uint64_t>(products::tableID) << 32) |
                     static_cast<uint32_t>(query.product_key));
      for (auto pk : query.part_keys) {
        keys.push_back((static_cast<uint64_t>(parts::tableID) << 32) |
                       static_cast<uint32_t>(pk));
      }
      break;
    case GET_PART_BY_SUPPLIER:
      keys.push_back((tt64 << 48) |
                     (static_cast<uint64_t>(suppliers::tableID) << 32) |
                     static_cast<uint32_t>(query.supplier_key));
      for (auto pk : query.part_keys) {
        keys.push_back((static_cast<uint64_t>(parts::tableID) << 32) |
                       static_cast<uint32_t>(pk));
      }
      break;
    case ORDER_PRODUCT:
      keys.push_back((tt64 << 48) |
                     (static_cast<uint64_t>(products::tableID) << 32) |
                     static_cast<uint32_t>(query.product_key));
      for (auto pk : query.part_keys) {
        keys.push_back((static_cast<uint64_t>(parts::tableID) << 32) |
                       static_cast<uint32_t>(pk));
      }
      break;
    case UPDATE_PRODUCT_PART:
      keys.push_back((tt64 << 48) |
                     (static_cast<uint64_t>(products::tableID) << 32) |
                     static_cast<uint32_t>(query.product_key));
      break;
    case UPDATE_PART:
      keys.push_back((tt64 << 48) |
                     (static_cast<uint64_t>(parts::tableID) << 32) |
                     static_cast<uint32_t>(query.part_key));
      break;
    default:
      DCHECK(false);
      break;
    }
    return keys;
  }

  const std::vector<bool> get_query_update() override {
    std::vector<bool> upd;
    switch (query.txn_type) {
    case ORDER_PRODUCT:
      // product is read, each part is a write
      upd.push_back(false);
      for (std::size_t i = 0; i < query.part_keys.size(); i++) {
        upd.push_back(true);
      }
      break;
    case UPDATE_PRODUCT_PART:
      upd.push_back(true);
      break;
    case UPDATE_PART:
      upd.push_back(true);
      break;
    default:
      // read-only: all false
      {
        auto keys = get_query();
        for (std::size_t i = 0; i < keys.size(); i++) {
          upd.push_back(false);
        }
      }
      break;
    }
    return upd;
  }

  const std::vector<uint64_t> get_query_master() override {
    using T = uint64_t;
    std::vector<T> masters;
    auto keys = get_query();
    for (auto encoded : keys) {
      uint64_t stripped = encoded & 0x0000FFFFFFFFFFFFull;
      std::size_t table_id = (stripped >> 32) & 0xFFFF;
      int32_t raw_key = static_cast<int32_t>(stripped & 0xFFFFFFFF);
      std::size_t c_id = db.get_dynamic_coordinator_id(
          context.coordinator_num, table_id, (void *)&raw_key);
      masters.push_back(static_cast<T>(c_id));
    }
    return masters;
  }

  const std::string get_query_printed() override {
    std::string s;
    for (auto v : get_query()) {
      s += " " + std::to_string(v);
    }
    return s;
  }

  std::vector<size_t> debug_record_keys() override {
    std::vector<size_t> ret;
    for (auto v : get_query()) {
      ret.push_back(static_cast<size_t>(v));
    }
    return ret;
  }

  std::vector<size_t> debug_record_keys_master() override {
    std::vector<size_t> ret;
    for (auto v : get_query_master()) {
      ret.push_back(static_cast<size_t>(v));
    }
    return ret;
  }

  std::set<int> txn_nodes_involved(bool is_dynamic) override {
    std::set<int> nodes;
    auto keys = get_query();
    for (auto encoded : keys) {
      uint64_t stripped = encoded & 0x0000FFFFFFFFFFFFull;
      std::size_t table_id = (stripped >> 32) & 0xFFFF;
      int32_t raw_key = static_cast<int32_t>(stripped & 0xFFFFFFFF);
      std::size_t c_id;
      if (is_dynamic) {
        c_id = db.get_dynamic_coordinator_id(context.coordinator_num,
                                             table_id, (void *)&raw_key);
      } else {
        std::size_t pid =
            static_cast<std::size_t>(raw_key) /
            context.getKeysPerPartition(table_id);
        c_id = (pid + 1) % context.coordinator_num;
      }
      nodes.insert(static_cast<int>(c_id));
    }
    return nodes;
  }

  std::unordered_map<int, int> txn_nodes_involved(int &max_node,
                                                   bool is_dynamic) override {
    std::unordered_map<int, int> counts;
    auto keys = get_query();
    int max_cnt = 0;
    for (auto encoded : keys) {
      uint64_t stripped = encoded & 0x0000FFFFFFFFFFFFull;
      std::size_t table_id = (stripped >> 32) & 0xFFFF;
      int32_t raw_key = static_cast<int32_t>(stripped & 0xFFFFFFFF);
      std::size_t c_id;
      if (is_dynamic) {
        c_id = db.get_dynamic_coordinator_id(context.coordinator_num,
                                             table_id, (void *)&raw_key);
      } else {
        std::size_t pid =
            static_cast<std::size_t>(raw_key) /
            context.getKeysPerPartition(table_id);
        c_id = (pid + 1) % context.coordinator_num;
      }
      int node = static_cast<int>(c_id);
      if (!counts.count(node)) {
        counts[node] = 1;
      } else {
        counts[node] += 1;
      }
      if (counts[node] > max_cnt) {
        max_cnt = counts[node];
        max_node = node;
      }
    }
    return counts;
  }

  bool check_cross_node_txn(bool is_dynamic) override {
    auto keys = get_query();
    int nodes[20] = {0};
    for (auto encoded : keys) {
      uint64_t stripped = encoded & 0x0000FFFFFFFFFFFFull;
      std::size_t table_id = (stripped >> 32) & 0xFFFF;
      int32_t raw_key = static_cast<int32_t>(stripped & 0xFFFFFFFF);
      std::size_t c_id;
      if (is_dynamic) {
        c_id = db.get_dynamic_coordinator_id(context.coordinator_num,
                                             table_id, (void *)&raw_key);
      } else {
        std::size_t pid =
            static_cast<std::size_t>(raw_key) /
            context.getKeysPerPartition(table_id);
        c_id = (pid + 1) % context.coordinator_num;
      }
      if (c_id < 20) nodes[c_id]++;
    }
    int cnt = 0;
    for (int i = 0; i < 20; i++) {
      if (nodes[i] > 0) cnt++;
    }
    return cnt > 1;
  }

private:
  // ---- Issue helpers ----

  void issuePartRead(std::size_t slot, int32_t part_key) {
    DCHECK(slot < PPS_MAX_PARTS_PER_TXN);
    std::size_t pid = static_cast<std::size_t>(part_key) /
                      context.getKeysPerPartition(parts::tableID);
    storage.parts_keys[slot].P_KEY = part_key;
    this->search_for_read(parts::tableID, pid,
                          storage.parts_keys[slot],
                          storage.parts_values[slot]);
  }

  void issuePartUpdate(std::size_t slot, int32_t part_key) {
    DCHECK(slot < PPS_MAX_PARTS_PER_TXN);
    std::size_t pid = static_cast<std::size_t>(part_key) /
                      context.getKeysPerPartition(parts::tableID);
    storage.parts_keys[slot].P_KEY = part_key;
    this->search_for_update(parts::tableID, pid,
                            storage.parts_keys[slot],
                            storage.parts_values[slot]);
  }

  void issueProductRead(int32_t product_key) {
    std::size_t pid = static_cast<std::size_t>(product_key) /
                      context.getKeysPerPartition(products::tableID);
    storage.product_key.PR_KEY = product_key;
    this->search_for_read(products::tableID, pid,
                          storage.product_key,
                          storage.product_value);
  }

  void issueProductUpdate(int32_t product_key) {
    std::size_t pid = static_cast<std::size_t>(product_key) /
                      context.getKeysPerPartition(products::tableID);
    storage.product_key.PR_KEY = product_key;
    this->search_for_update(products::tableID, pid,
                            storage.product_key,
                            storage.product_value);
  }

  void issueSupplierRead(int32_t supplier_key) {
    std::size_t pid = static_cast<std::size_t>(supplier_key) /
                      context.getKeysPerPartition(suppliers::tableID);
    storage.supplier_key.S_KEY = supplier_key;
    this->search_for_read(suppliers::tableID, pid,
                          storage.supplier_key,
                          storage.supplier_value);
  }

  // Issues all reads/updates for the current txn_type without processing
  void issue_reads_updates() {
    switch (query.txn_type) {
    case GET_PART:
      issuePartRead(0, query.part_key);
      break;
    case GET_PRODUCT:
      issueProductRead(query.product_key);
      break;
    case GET_SUPPLIER:
      issueSupplierRead(query.supplier_key);
      break;
    case GET_PART_BY_PRODUCT:
      issueProductRead(query.product_key);
      for (std::size_t i = 0; i < query.part_keys.size(); i++) {
        issuePartRead(i, query.part_keys[i]);
      }
      break;
    case GET_PART_BY_SUPPLIER:
      issueSupplierRead(query.supplier_key);
      for (std::size_t i = 0; i < query.part_keys.size(); i++) {
        issuePartRead(i, query.part_keys[i]);
      }
      break;
    case ORDER_PRODUCT:
      issueProductRead(query.product_key);
      for (std::size_t i = 0; i < query.part_keys.size(); i++) {
        issuePartUpdate(i, query.part_keys[i]);
      }
      break;
    case UPDATE_PRODUCT_PART:
      issueProductUpdate(query.product_key);
      break;
    case UPDATE_PART:
      issuePartUpdate(0, query.part_key);
      break;
    default:
      DCHECK(false);
      break;
    }
  }

  TransactionResult applyWrites() {
    if (!this->execution_phase) {
      return TransactionResult::READY_TO_COMMIT;
    }

    RandomType local_random;

    switch (query.txn_type) {
    case ORDER_PRODUCT:
      for (std::size_t i = 0; i < query.part_keys.size(); i++) {
        storage.parts_values[i].P_AMOUNT--;
        std::size_t pid = static_cast<std::size_t>(query.part_keys[i]) /
                          context.getKeysPerPartition(parts::tableID);
        this->update(parts::tableID, pid,
                     storage.parts_keys[i],
                     storage.parts_values[i]);
      }
      break;
    case UPDATE_PRODUCT_PART:
      storage.product_value.PR_F01.assign(
          local_random.a_string(PPS_FIELD_SIZE, PPS_FIELD_SIZE));
      {
        std::size_t pid = static_cast<std::size_t>(query.product_key) /
                          context.getKeysPerPartition(products::tableID);
        this->update(products::tableID, pid,
                     storage.product_key,
                     storage.product_value);
      }
      break;
    case UPDATE_PART:
      storage.parts_values[0].P_AMOUNT++;
      {
        std::size_t pid = static_cast<std::size_t>(query.part_key) /
                          context.getKeysPerPartition(parts::tableID);
        this->update(parts::tableID, pid,
                     storage.parts_keys[0],
                     storage.parts_values[0]);
      }
      break;
    default:
      // read-only: nothing to write
      break;
    }

    return TransactionResult::READY_TO_COMMIT;
  }

  // ---- Member variables ----
  std::atomic<uint32_t> &worker_status_;
  DatabaseType          &db;
  const ContextType     &context;
  RandomType            &random;
  Storage               &storage;
  std::size_t            partition_id;
  PPSQuery               query;
  bool                   is_transmit_request = false;
};

} // namespace pps
} // namespace star
