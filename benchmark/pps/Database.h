// benchmark/pps/Database.h
#pragma once

#include "benchmark/pps/Context.h"
#include "benchmark/pps/Random.h"
#include "benchmark/pps/Schema.h"
#include "common/Operation.h"
#include "core/Partitioner.h"
#include "core/RouterValue.h"
#include "core/Table.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <glog/logging.h>
#include <thread>
#include <vector>

namespace star {
namespace pps {

class Database {
public:
  using MetaDataType = std::atomic<uint64_t>;
  using ContextType  = Context;
  using RandomType   = Random;
  static myTestSet which_workload() { return myTestSet::PPS; }

  // ---- Table access ----
  ITable *find_table(std::size_t table_id, std::size_t partition_id) {
    DCHECK(table_id < tbl_vecs.size());
    DCHECK(partition_id < tbl_vecs[table_id].size());
    DCHECK(!isolation_replica);
    return tbl_vecs[table_id][partition_id];
  }

  ITable *find_router_lock_table(std::size_t table_id, std::size_t partition_id) {
    DCHECK(table_id < tbl_vecs_router_lock.size());
    DCHECK(partition_id < tbl_vecs_router_lock[table_id].size());
    DCHECK(!isolation_replica);
    return tbl_vecs_router_lock[table_id][partition_id];
  }

  ImyRouterTable *find_router_table(std::size_t table_id) {
    DCHECK(!isolation_replica);
    return tbl_vecs_router[table_id];
  }

  std::size_t get_dynamic_coordinator_id(std::size_t coordinator_num,
                                          std::size_t table_id,
                                          const void *key) {
    DCHECK(!isolation_replica);
    ImyRouterTable *tab = find_router_table(table_id);
    return ((RouterValue *)(tab->search_value(key)))->get_dynamic_coordinator_id();
  }

  std::size_t get_secondary_coordinator_id(std::size_t coordinator_num,
                                             std::size_t table_id,
                                             const void *key) {
    DCHECK(!isolation_replica);
    ImyRouterTable *tab = find_router_table(table_id);
    return ((RouterValue *)(tab->search_value(key)))->get_secondary_coordinator_id();
  }

  // Hermes replica variants — stubbed (isolation_replica not yet supported)
  ITable *find_table(std::size_t table_id, std::size_t partition_id, int replica_id) {
    DCHECK(isolation_replica && replica_id != -1);
    return tbl_vecs_[replica_id][table_id][partition_id];
  }
  ImyRouterTable *find_router_table(std::size_t table_id, int replica_id) {
    DCHECK(isolation_replica && replica_id != -1);
    return tbl_vecs_router_[replica_id][table_id];
  }
  std::size_t get_dynamic_coordinator_id(std::size_t coordinator_num,
                                          std::size_t table_id,
                                          const void *key, int replica_id) {
    DCHECK(isolation_replica && replica_id != -1);
    ImyRouterTable *tab = find_router_table(table_id, replica_id);
    return ((RouterValue *)(tab->search_value(key)))->get_dynamic_coordinator_id();
  }
  std::size_t get_secondary_coordinator_id(std::size_t coordinator_num,
                                             std::size_t table_id,
                                             const void *key, int replica_id) {
    DCHECK(isolation_replica && replica_id != -1);
    ImyRouterTable *tab = find_router_table(table_id, replica_id);
    return ((RouterValue *)(tab->search_value(key)))->get_secondary_coordinator_id();
  }

  // ---- In-memory relationship maps ----
  const std::vector<int32_t> &getProductParts(int32_t product_key) const {
    DCHECK(product_key >= 0 &&
           static_cast<std::size_t>(product_key) < product_parts_.size());
    return product_parts_[product_key];
  }
  const std::vector<int32_t> &getSupplierParts(int32_t supplier_key) const {
    DCHECK(supplier_key >= 0 &&
           static_cast<std::size_t>(supplier_key) < supplier_parts_.size());
    return supplier_parts_[supplier_key];
  }

  // ---- Initialization ----
  void initialize(const Context &context) {
    coordinator_id = context.coordinator_id;
    partitionNum   = context.partition_num;
    threadsNum     = std::max(context.worker_num, std::size_t(8));

    if (context.protocol == "Hermes") isolation_replica = true;

    auto partitioner = PartitionerFactory::create_partitioner(
        context.partitioner, coordinator_id, context.coordinator_num);

    // Create table partitions
    for (auto pid = 0u; pid < partitionNum; pid++) {
      tbl_parts_vec.push_back(
          std::make_unique<Table<100860, parts::key, parts::value>>(
              parts::tableID, pid));
      tbl_parts_vec_router_lock.push_back(
          std::make_unique<Table<100860, parts::key, parts::value>>(
              parts::tableID, pid));
      tbl_products_vec.push_back(
          std::make_unique<Table<100860, products::key, products::value>>(
              products::tableID, pid));
      tbl_products_vec_router_lock.push_back(
          std::make_unique<Table<100860, products::key, products::value>>(
              products::tableID, pid));
      tbl_suppliers_vec.push_back(
          std::make_unique<Table<100860, suppliers::key, suppliers::value>>(
              suppliers::tableID, pid));
      tbl_suppliers_vec_router_lock.push_back(
          std::make_unique<Table<100860, suppliers::key, suppliers::value>>(
              suppliers::tableID, pid));
    }

    // Build tbl_vecs[table_id][partition_id]
    tbl_vecs.resize(3);
    tbl_vecs_router_lock.resize(3);
    auto tFunc = [](auto &t) { return t.get(); };
    std::transform(tbl_parts_vec.begin(), tbl_parts_vec.end(),
                   std::back_inserter(tbl_vecs[0]), tFunc);
    std::transform(tbl_products_vec.begin(), tbl_products_vec.end(),
                   std::back_inserter(tbl_vecs[1]), tFunc);
    std::transform(tbl_suppliers_vec.begin(), tbl_suppliers_vec.end(),
                   std::back_inserter(tbl_vecs[2]), tFunc);
    std::transform(tbl_parts_vec_router_lock.begin(), tbl_parts_vec_router_lock.end(),
                   std::back_inserter(tbl_vecs_router_lock[0]), tFunc);
    std::transform(tbl_products_vec_router_lock.begin(), tbl_products_vec_router_lock.end(),
                   std::back_inserter(tbl_vecs_router_lock[1]), tFunc);
    std::transform(tbl_suppliers_vec_router_lock.begin(), tbl_suppliers_vec_router_lock.end(),
                   std::back_inserter(tbl_vecs_router_lock[2]), tFunc);

    // Initialize table data in parallel
    initTables("parts", [&context, this](std::size_t pid) {
      partsInit(context, pid, tbl_parts_vec[pid].get());
    }, partitionNum, threadsNum, partitioner.get());
    initTables("products", [&context, this](std::size_t pid) {
      productsInit(context, pid, tbl_products_vec[pid].get());
    }, partitionNum, threadsNum, partitioner.get());
    initTables("suppliers", [&context, this](std::size_t pid) {
      suppliersInit(context, pid, tbl_suppliers_vec[pid].get());
    }, partitionNum, threadsNum, partitioner.get());

    // Initialize lock tables (same data)
    initTables("parts_lock", [&context, this](std::size_t pid) {
      partsInit(context, pid, tbl_parts_vec_router_lock[pid].get());
    }, partitionNum, threadsNum, nullptr);
    initTables("products_lock", [&context, this](std::size_t pid) {
      productsInit(context, pid, tbl_products_vec_router_lock[pid].get());
    }, partitionNum, threadsNum, nullptr);
    initTables("suppliers_lock", [&context, this](std::size_t pid) {
      suppliersInit(context, pid, tbl_suppliers_vec_router_lock[pid].get());
    }, partitionNum, threadsNum, nullptr);

    // Create router tables (one per table)
    std::size_t partsTotal     = context.getTotalKeys(0);
    std::size_t productsTotal  = context.getTotalKeys(1);
    std::size_t suppliersTotal = context.getTotalKeys(2);

    tbl_parts_router     = std::make_unique<myRouterTable<parts::key,     RouterValue>>(
                               partsTotal,     parts::tableID,     0);
    tbl_products_router  = std::make_unique<myRouterTable<products::key,  RouterValue>>(
                               productsTotal,  products::tableID,  0);
    tbl_suppliers_router = std::make_unique<myRouterTable<suppliers::key, RouterValue>>(
                               suppliersTotal, suppliers::tableID, 0);

    tbl_vecs_router.resize(3);
    tbl_vecs_router[0] = tbl_parts_router.get();
    tbl_vecs_router[1] = tbl_products_router.get();
    tbl_vecs_router[2] = tbl_suppliers_router.get();

    // Build in-memory relationship maps
    initRelationshipMaps(context);

    // Fill router tables
    if (context.protocol == "Star") {
      init_star_router_table(context);
    } else {
      init_router_table(context, partitioner);
    }
  }

  void apply_operation(const Operation &operation) { CHECK(false); }

private:
  void partsInit(const Context &context, std::size_t partitionID, ITable *table) {
    Random random;
    std::size_t kpp = context.getKeysPerPartition(0);
    for (auto i = partitionID * kpp; i < (partitionID + 1) * kpp; i++) {
      parts::key   k(static_cast<int32_t>(i));
      parts::value v;
      v.P_AMOUNT = 1000;
      v.P_F01.assign(random.a_string(PPS_FIELD_SIZE, PPS_FIELD_SIZE));
      v.P_F02.assign(random.a_string(PPS_FIELD_SIZE, PPS_FIELD_SIZE));
      v.P_F03.assign(random.a_string(PPS_FIELD_SIZE, PPS_FIELD_SIZE));
      v.P_F04.assign(random.a_string(PPS_FIELD_SIZE, PPS_FIELD_SIZE));
      v.P_F05.assign(random.a_string(PPS_FIELD_SIZE, PPS_FIELD_SIZE));
      v.P_F06.assign(random.a_string(PPS_FIELD_SIZE, PPS_FIELD_SIZE));
      v.P_F07.assign(random.a_string(PPS_FIELD_SIZE, PPS_FIELD_SIZE));
      v.P_F08.assign(random.a_string(PPS_FIELD_SIZE, PPS_FIELD_SIZE));
      v.P_F09.assign(random.a_string(PPS_FIELD_SIZE, PPS_FIELD_SIZE));
      v.P_F10.assign(random.a_string(PPS_FIELD_SIZE, PPS_FIELD_SIZE));
      table->insert(&k, &v);
    }
  }

  void productsInit(const Context &context, std::size_t partitionID, ITable *table) {
    Random random;
    std::size_t kpp = context.getKeysPerPartition(1);
    for (auto i = partitionID * kpp; i < (partitionID + 1) * kpp; i++) {
      products::key   k(static_cast<int32_t>(i));
      products::value v;
      v.PR_F01.assign(random.a_string(PPS_FIELD_SIZE, PPS_FIELD_SIZE));
      v.PR_F02.assign(random.a_string(PPS_FIELD_SIZE, PPS_FIELD_SIZE));
      v.PR_F03.assign(random.a_string(PPS_FIELD_SIZE, PPS_FIELD_SIZE));
      v.PR_F04.assign(random.a_string(PPS_FIELD_SIZE, PPS_FIELD_SIZE));
      v.PR_F05.assign(random.a_string(PPS_FIELD_SIZE, PPS_FIELD_SIZE));
      v.PR_F06.assign(random.a_string(PPS_FIELD_SIZE, PPS_FIELD_SIZE));
      v.PR_F07.assign(random.a_string(PPS_FIELD_SIZE, PPS_FIELD_SIZE));
      v.PR_F08.assign(random.a_string(PPS_FIELD_SIZE, PPS_FIELD_SIZE));
      v.PR_F09.assign(random.a_string(PPS_FIELD_SIZE, PPS_FIELD_SIZE));
      v.PR_F10.assign(random.a_string(PPS_FIELD_SIZE, PPS_FIELD_SIZE));
      table->insert(&k, &v);
    }
  }

  void suppliersInit(const Context &context, std::size_t partitionID, ITable *table) {
    Random random;
    std::size_t kpp = context.getKeysPerPartition(2);
    for (auto i = partitionID * kpp; i < (partitionID + 1) * kpp; i++) {
      suppliers::key   k(static_cast<int32_t>(i));
      suppliers::value v;
      v.S_F01.assign(random.a_string(PPS_FIELD_SIZE, PPS_FIELD_SIZE));
      v.S_F02.assign(random.a_string(PPS_FIELD_SIZE, PPS_FIELD_SIZE));
      v.S_F03.assign(random.a_string(PPS_FIELD_SIZE, PPS_FIELD_SIZE));
      v.S_F04.assign(random.a_string(PPS_FIELD_SIZE, PPS_FIELD_SIZE));
      v.S_F05.assign(random.a_string(PPS_FIELD_SIZE, PPS_FIELD_SIZE));
      v.S_F06.assign(random.a_string(PPS_FIELD_SIZE, PPS_FIELD_SIZE));
      v.S_F07.assign(random.a_string(PPS_FIELD_SIZE, PPS_FIELD_SIZE));
      v.S_F08.assign(random.a_string(PPS_FIELD_SIZE, PPS_FIELD_SIZE));
      v.S_F09.assign(random.a_string(PPS_FIELD_SIZE, PPS_FIELD_SIZE));
      v.S_F10.assign(random.a_string(PPS_FIELD_SIZE, PPS_FIELD_SIZE));
      table->insert(&k, &v);
    }
  }

  void initRelationshipMaps(const Context &context) {
    std::size_t productsTotal  = context.getTotalKeys(1);
    std::size_t suppliersTotal = context.getTotalKeys(2);
    std::size_t partsTotal     = context.getTotalKeys(0);
    Random random;

    product_parts_.resize(productsTotal);
    for (std::size_t pid = 0; pid < productsTotal; pid++) {
      product_parts_[pid].resize(context.partsPerProduct);
      for (std::size_t j = 0; j < context.partsPerProduct; j++) {
        product_parts_[pid][j] =
            static_cast<int32_t>(random.uniform_dist(0, partsTotal - 1));
      }
    }

    supplier_parts_.resize(suppliersTotal);
    for (std::size_t sid = 0; sid < suppliersTotal; sid++) {
      supplier_parts_[sid].resize(context.partsPerSupplier);
      for (std::size_t j = 0; j < context.partsPerSupplier; j++) {
        supplier_parts_[sid][j] =
            static_cast<int32_t>(random.uniform_dist(0, partsTotal - 1));
      }
    }
  }

  void init_router_table(const Context &context,
                          std::unique_ptr<Partitioner> &partitioner) {
    auto fill = [&](std::size_t table_id, std::size_t kpp,
                    ImyRouterTable *router) {
      std::size_t replica_num = partitioner->replica_num();
      for (auto pid = 0u; pid < context.partition_num; pid++) {
        for (auto i = pid * kpp; i < (pid + 1) * kpp; i++) {
          int32_t raw_key = static_cast<int32_t>(i);
          size_t last_replica  = (pid + 1) % context.coordinator_num;
          size_t first_replica = (last_replica - replica_num + 1 +
                                  context.coordinator_num) %
                                 context.coordinator_num;
          RouterValue rv;
          rv.set_dynamic_coordinator_id(last_replica);
          if (first_replica <= last_replica) {
            for (size_t k = first_replica; k <= last_replica; k++)
              rv.set_secondary_coordinator_id(k);
          } else {
            for (size_t k = 0; k <= last_replica; k++)
              rv.set_secondary_coordinator_id(k);
            for (size_t k = first_replica; k < context.coordinator_num; k++)
              rv.set_secondary_coordinator_id(k);
          }
          router->insert(&raw_key, &rv);
        }
      }
    };
    fill(0, context.getKeysPerPartition(0), tbl_parts_router.get());
    fill(1, context.getKeysPerPartition(1), tbl_products_router.get());
    fill(2, context.getKeysPerPartition(2), tbl_suppliers_router.get());
  }

  void init_star_router_table(const Context &context) {
    auto fill = [&](std::size_t kpp, ImyRouterTable *router) {
      for (auto pid = 0u; pid < context.partition_num; pid++) {
        for (auto i = pid * kpp; i < (pid + 1) * kpp; i++) {
          int32_t raw_key = static_cast<int32_t>(i);
          RouterValue rv;
          rv.set_dynamic_coordinator_id(0);
          rv.set_secondary_coordinator_id(0);
          rv.set_secondary_coordinator_id(
              static_cast<int>(pid % context.coordinator_num));
          router->insert(&raw_key, &rv);
        }
      }
    };
    fill(context.getKeysPerPartition(0), tbl_parts_router.get());
    fill(context.getKeysPerPartition(1), tbl_products_router.get());
    fill(context.getKeysPerPartition(2), tbl_suppliers_router.get());
  }

  template <class InitFunc>
  void initTables(const std::string &name, InitFunc initFunc,
                  std::size_t partitionNum, std::size_t threadsNum,
                  Partitioner *partitioner) {
    std::vector<int> all_parts;
    for (auto i = 0u; i < partitionNum; i++) {
      if (!partitioner || partitioner->is_partition_replicated_on_me(i))
        all_parts.push_back(static_cast<int>(i));
    }
    auto now = std::chrono::steady_clock::now();
    std::vector<std::thread> v;
    for (auto tid = 0u; tid < threadsNum; tid++) {
      v.emplace_back([=]() {
        for (auto i = tid; i < all_parts.size(); i += threadsNum)
          initFunc(all_parts[i]);
      });
    }
    for (auto &t : v) t.join();
    LOG(INFO) << name << " init done in "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::steady_clock::now() - now)
                     .count()
              << " ms";
  }

private:
  std::vector<std::vector<ITable *>>  tbl_vecs;             // [table_id][partition_id]
  std::vector<ImyRouterTable *>       tbl_vecs_router;      // [table_id]
  std::vector<std::vector<ITable *>>  tbl_vecs_router_lock; // [table_id][partition_id]

  std::vector<std::unique_ptr<ITable>> tbl_parts_vec;
  std::vector<std::unique_ptr<ITable>> tbl_products_vec;
  std::vector<std::unique_ptr<ITable>> tbl_suppliers_vec;

  std::vector<std::unique_ptr<ITable>> tbl_parts_vec_router_lock;
  std::vector<std::unique_ptr<ITable>> tbl_products_vec_router_lock;
  std::vector<std::unique_ptr<ITable>> tbl_suppliers_vec_router_lock;

  std::unique_ptr<ImyRouterTable> tbl_parts_router;
  std::unique_ptr<ImyRouterTable> tbl_products_router;
  std::unique_ptr<ImyRouterTable> tbl_suppliers_router;

  // Hermes replica support (stubbed, isolation_replica guards usage)
  bool isolation_replica = false;
  std::vector<std::vector<ITable *>>  tbl_vecs_[2];
  std::vector<ImyRouterTable *>       tbl_vecs_router_[2];

  // In-memory relationship maps
  std::vector<std::vector<int32_t>> product_parts_;   // [product_key] -> [part_key, ...]
  std::vector<std::vector<int32_t>> supplier_parts_;  // [supplier_key] -> [part_key, ...]

  std::size_t coordinator_id = 0;
  std::size_t partitionNum   = 0;
  std::size_t threadsNum     = 0;
};

} // namespace pps
} // namespace star
