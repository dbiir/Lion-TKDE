#include "benchmark/pps/Database.h"
#include "core/Coordinator.h"
#include "core/Macros.h"

DEFINE_int32(read_write_ratio, 80, "read-only transaction probability (out of 100)");
DEFINE_int32(cross_ratio,      30, "cross-partition transaction probability (out of 100)");
DEFINE_int32(keys,         200000, "parts per partition (products/suppliers = keys/10)");
DEFINE_double(zipf,           0.0, "Zipf skew factor for part key selection (0 = uniform)");

// ./bench_pps --logtostderr=1 --id=1 --servers="127.0.0.1:10010;127.0.0.1:10011"

template <class Context> class InferType {};

template <> class InferType<star::pps::Context> {
public:
  template <class Transaction>
  using WorkloadType = star::pps::Workload<Transaction>;
};

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  google::ParseCommandLineFlags(&argc, &argv, true);

  star::pps::Context context;
  SETUP_CONTEXT(context);

  context.readWriteRatio            = FLAGS_read_write_ratio;
  context.crossPartitionProbability = FLAGS_cross_ratio;
  context.keysPerPartition          = static_cast<std::size_t>(FLAGS_keys);

  if (FLAGS_zipf > 0) {
    context.isUniform = false;
    star::Zipf::globalZipf().init(context.keysPerPartition, FLAGS_zipf);
  }
  DCHECK(context.peers.size() >= 2) << " The size of ip peers must gt 2.(At least one generator, one worker)";

  star::pps::Database db;
  db.initialize(context);

  star::Coordinator c(FLAGS_id, db, context);
  c.connectToPeers();
  c.start();

  google::ShutdownGoogleLogging();

  return 0;
}
