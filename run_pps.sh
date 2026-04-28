#!/usr/bin/env bash
# Run PPS workload with the Lion protocol on a single machine.
#
# Node layout (3 entries in --servers → coordinator_num = peers.size()-1 = 2):
#   id=0  127.0.0.1:10010  data coordinator 0  → create_workers()
#   id=1  127.0.0.1:10011  data coordinator 1  → create_workers()
#   id=2  127.0.0.1:10012  generator           → create_generator()  (id == coordinator_num)
#
# Fix 1 (generator never started): all three processes are launched here.
#         In pps_0.log the generator node (id=coordinator_num=2) was missing,
#         leaving executors idle with zero commits.
#
# Fix 2 (missing Metis partition file): --lion_with_metis_init=0
#         The LionMetisGenerator skips the read loop when this flag is 0,
#         avoiding the FAILED TO DO read_file_from_mmap error on
#         resultss_partition_30_60.xls.  To enable Metis-guided migration,
#         generate the partition files offline with the metis_test binary first,
#         set --data_src_path_dir to that directory, and change this flag to 1.
#
# Fix 3 (n_distributed = 0%): coordinator_num=2 + cross_ratio=50
#         With two data coordinators the router table distributes partitions
#         across both nodes via (pid+1) % coordinator_num.  cross_ratio=50
#         makes half of all transactions touch data on both coordinators.
#         partition_num=12 per coordinator → total=24 partitions;
#         worker_num * coordinator_num = 12*2 = 24, so 24 % 24 == 0 ✓.

set -euo pipefail

BINARY=/root/Lion-TKDE/build/bench_pps
SERVERS="127.0.0.1:10010;127.0.0.1:10011;127.0.0.1:10012"

COMMON_FLAGS=(
  --logtostderr=1
  --servers="${SERVERS}"
  --protocol=Lion
  --partitioner=Lion
  --threads=12
  --partition_num=12
  --batch_size=10000
  --batch_flush=500
  --cross_ratio=50
  --read_write_ratio=80
  --lion_with_metis_init=0
  --migration_only=0
  --time_to_run=60
  --workload_time=60
  --sample_time_interval=3
  --random_router=0
)

# Kill any stale processes from a previous run.
pkill -f "bench_pps" 2>/dev/null || true
sleep 1

# Start all three nodes in parallel, each with its own log file.
"${BINARY}" "${COMMON_FLAGS[@]}" --id=0 > pps_0.log 2>&1 &
"${BINARY}" "${COMMON_FLAGS[@]}" --id=1 > pps_1.log 2>&1 &
"${BINARY}" "${COMMON_FLAGS[@]}" --id=2 > pps_2.log 2>&1 &  # generator

wait
echo "All nodes finished. Logs: pps_0.log  pps_1.log  pps_2.log"
