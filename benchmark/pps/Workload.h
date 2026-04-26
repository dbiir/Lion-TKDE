// benchmark/pps/Workload.h
#pragma once

#include "benchmark/pps/Context.h"
#include "benchmark/pps/Database.h"
#include "benchmark/pps/Random.h"
#include "benchmark/pps/Storage.h"
#include "benchmark/pps/Transaction.h"
#include "core/Partitioner.h"

namespace star {
namespace pps {

template <class Transaction>
class Workload {
public:
  using TransactionType = Transaction;
  using DatabaseType    = Database;
  using ContextType     = Context;
  using RandomType      = Random;
  using StorageType     = Storage;

  static myTestSet which_workload;

  Workload(std::size_t coordinator_id,
           std::atomic<uint32_t> &worker_status,
           DatabaseType &db, RandomType &random,
           Partitioner &partitioner,
           std::chrono::steady_clock::time_point start_time)
      : coordinator_id(coordinator_id),
        worker_status(worker_status), db(db), random(random),
        partitioner(partitioner), start_time(start_time) {}

  std::unique_ptr<TransactionType> next_transaction(const ContextType &context,
                                                     std::size_t &partition_id,
                                                     StorageType &storage) {
    double cur_timestamp =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - start_time)
            .count() * 1.0 / 1000 / 1000;

    return std::make_unique<PPSTransaction<Transaction>>(
        coordinator_id, partition_id, worker_status,
        db, context, random, partitioner, storage, cur_timestamp);
  }

  std::unique_ptr<TransactionType> unpack_transaction(const ContextType &context,
                                                       std::size_t partition_id,
                                                       StorageType &storage,
                                                       simpleTransaction &simple_txn) {
    return std::make_unique<PPSTransaction<Transaction>>(
        coordinator_id, partition_id, worker_status,
        db, context, random, partitioner, storage, simple_txn);
  }

  std::unique_ptr<TransactionType> unpack_transaction(const ContextType &context,
                                                       std::size_t partition_id,
                                                       StorageType &storage,
                                                       simpleTransaction &simple_txn,
                                                       bool /*is_transmit*/) {
    return std::make_unique<PPSTransaction<Transaction>>(
        coordinator_id, partition_id, worker_status,
        db, context, random, partitioner, storage, simple_txn);
  }

  std::unique_ptr<TransactionType> reset_transaction(const ContextType &context,
                                                      std::size_t partition_id,
                                                      StorageType &storage,
                                                      TransactionType &txn) {
    auto p = std::make_unique<PPSTransaction<Transaction>>(
        coordinator_id, partition_id, worker_status,
        db, context, random, partitioner, storage, txn);
    p->startTime = txn.startTime;
    return p;
  }

private:
  std::size_t coordinator_id;
  std::atomic<uint32_t> &worker_status;
  DatabaseType &db;
  RandomType &random;
  Partitioner &partitioner;

public:
  std::chrono::steady_clock::time_point start_time;
};

template <class Transaction>
myTestSet Workload<Transaction>::which_workload = myTestSet::PPS;

} // namespace pps
} // namespace star
