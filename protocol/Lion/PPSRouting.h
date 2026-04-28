#pragma once

#include "core/Defs.h"

#include <climits>
#include <cstdint>
#include <vector>

namespace star {
namespace lion {

static constexpr std::size_t PPS_ROUTING_MAX_COORDINATOR_NUM = 80;

struct PPSDecodedKey {
  std::size_t table_id;
  int32_t raw_key;
};

inline PPSDecodedKey decode_pps_key(uint64_t encoded) {
  uint64_t stripped = encoded & 0x0000FFFFFFFFFFFFull;
  return {
      static_cast<std::size_t>((stripped >> 32) & 0xFFFF),
      static_cast<int32_t>(stripped & 0xFFFFFFFF),
  };
}

inline std::size_t pps_keys_per_partition(std::size_t table_id,
                                          std::size_t keys_per_partition) {
  return table_id == 0 ? keys_per_partition : keys_per_partition / 10;
}

template <class LookupFn>
inline void route_pps_transaction(
    simpleTransaction* t,
    std::vector<std::vector<int>>& txns_coord_cost_,
    std::vector<int>& busy_local,
    std::vector<int>& replicate_busy_local,
    std::size_t coordinator_num,
    bool migration_only,
    std::size_t keys_per_partition,
    bool random_router,
    LookupFn&& lookup) {
  int from_nodes_id[PPS_ROUTING_MAX_COORDINATOR_NUM] = {0};
  int from_nodes_id_secondary[PPS_ROUTING_MAX_COORDINATOR_NUM] = {0};

  for (std::size_t j = 0; j < t->keys.size(); j++) {
    auto decoded = decode_pps_key(t->keys[j]);
    auto routing = lookup(decoded.table_id, decoded.raw_key);
    std::size_t cur_c_id = routing.first;
    std::size_t secondary_c_ids = routing.second;

    from_nodes_id[cur_c_id] += 1;

    for (std::size_t i = 0; i <= coordinator_num; i++) {
      if ((secondary_c_ids & 1) && i != cur_c_id) {
        from_nodes_id_secondary[i] += 1;
      }
      secondary_c_ids = secondary_c_ids >> 1;
    }
  }

  int max_cnt = INT_MIN;
  int max_node = -1;
  int replica_most_cnt = INT_MIN;
  int replica_max_node = -1;

  for (std::size_t cur_c_id = 0; cur_c_id < coordinator_num; cur_c_id++) {
    int cur_score = 0;
    std::size_t cnt_master = from_nodes_id[cur_c_id];
    std::size_t cnt_secondary = from_nodes_id_secondary[cur_c_id];

    if (migration_only) {
      cur_score += 100 * cnt_master;
    } else if (cnt_master == t->keys.size()) {
      cur_score += 100 * static_cast<int>(t->keys.size());
    } else if (cnt_secondary + cnt_master == t->keys.size()) {
      cur_score += 50 * cnt_master + 25 * cnt_secondary;
    } else {
      cur_score += 25 * cnt_master + 15 * cnt_secondary;
    }

    if (cur_score > max_cnt) {
      max_node = static_cast<int>(cur_c_id);
      max_cnt = cur_score;
    }

    if (static_cast<int>(cnt_secondary) > replica_most_cnt) {
      replica_most_cnt = static_cast<int>(cnt_secondary);
      replica_max_node = static_cast<int>(cur_c_id);
    }

    txns_coord_cost_[t->idx_][cur_c_id] =
        10 * static_cast<int>(t->keys.size()) - cur_score;
    replicate_busy_local[cur_c_id] += cnt_secondary;
  }

  if (random_router && !t->keys.empty()) {
    auto decoded = decode_pps_key(t->keys[0]);
    std::size_t kpp = pps_keys_per_partition(decoded.table_id, keys_per_partition);
    max_node = static_cast<int>((decoded.raw_key / kpp + 1) % coordinator_num);
  }

  t->destination_coordinator = max_node;
  t->execution_cost = 10 * static_cast<int>(t->keys.size()) - max_cnt;
  t->is_real_distributed =
      (max_cnt == 100 * static_cast<int>(t->keys.size())) ? false : true;
  t->replica_heavy_node = replica_max_node;
}

}  // namespace lion
}  // namespace star
