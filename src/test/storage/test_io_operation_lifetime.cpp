// Real O_DIRECT operations with observable kernel submissions. Exceptions are
// injected while OTHER requests are still outstanding, not just at entry.
#include "index/index_disk.hpp"
#include "shuffle/layout.hpp"
#include "support/partition_io_fixture.hpp"
#include "disk_build/cluster_wow/passes.hpp"
#include "disk_build/cluster_wow/scratch.hpp"
#include "disk_build/partition/membership.hpp"
#include "support/disk_build_test_peer.hpp"
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <new>
#include <numeric>
#include <omp.h>
#include <unistd.h>

using namespace alliance;
using Index = AllianceIndexDisk<int, float>;
namespace fs = std::filesystem;
namespace probe {
struct Pending {
  io_context_t ctx{};
  void *request{};
};
static std::array<Pending, 4096> requests{};
static bool tracking = false, fail_new = false, fail_distance = false, fail_poll = false;
static bool fail_write_poll = false, destroy_with_pending = false;
static int fail_read_result = 0; // 1: short completion, 2: negative completion.
static uint64_t failed_read_offset = UINT64_MAX;
static size_t completions = 0, injections = 0, peak = 0, failure_cases = 0;
static int fail_align_after = -1;
static size_t fail_aligned_bytes = 0;
static int fail_aligned_after = -1;
static size_t Count() {
  size_t n = 0;
  for (const auto &p : requests)
    n += p.request != nullptr;
  return n;
}
static void Add(io_context_t ctx, void *request) {
  for (auto &p : requests)
    if (!p.request) {
      p = {ctx, request};
      peak = std::max(peak, Count());
      return;
    }
  std::abort();
}
static void Remove(io_context_t ctx, void *request) {
  for (auto &p : requests)
    if (p.ctx == ctx && p.request == request) {
      p = {};
      return;
    }
}
static bool Contains(io_context_t ctx, void *request) {
  for (const auto &p : requests)
    if (p.ctx == ctx && p.request == request)
      return true;
  return false;
}
} // namespace probe
void *operator new(size_t bytes) {
  if (probe::fail_new && probe::completions && probe::Count()) {
    probe::fail_new = false;
    ++probe::injections;
    throw std::bad_alloc();
  }
  if (void *p = std::malloc(bytes ? bytes : 1))
    return p;
  throw std::bad_alloc();
}
void *operator new[](size_t bytes) {
  return ::operator new(bytes);
}
// Keep the test allocator consistent when libstdc++ uses nothrow new internally.
void *operator new(size_t bytes, const std::nothrow_t &) noexcept {
  try {
    return ::operator new(bytes);
  } catch (...) {
    return nullptr;
  }
}
void *operator new[](size_t bytes, const std::nothrow_t &) noexcept {
  try {
    return ::operator new[](bytes);
  } catch (...) {
    return nullptr;
  }
}
void operator delete(void *p) noexcept {
  std::free(p);
}
void operator delete[](void *p) noexcept {
  std::free(p);
}
void operator delete(void *p, size_t) noexcept {
  std::free(p);
}
void operator delete[](void *p, size_t) noexcept {
  std::free(p);
}
void operator delete(void *p, const std::nothrow_t &) noexcept {
  std::free(p);
}
void operator delete[](void *p, const std::nothrow_t &) noexcept {
  std::free(p);
}
extern "C" int __real_io_submit(io_context_t, long, iocb **);
extern "C" int __real_io_getevents(io_context_t, long, long, io_event *, timespec *);
extern "C" int __real_io_destroy(io_context_t);
extern "C" int __real_posix_memalign(void **, size_t, size_t);
extern "C" void *__real_aligned_alloc(size_t, size_t);
extern "C" void *__wrap_aligned_alloc(size_t align, size_t bytes) {
  if ((!probe::fail_aligned_bytes || bytes == probe::fail_aligned_bytes) &&
      probe::fail_aligned_after >= 0 && probe::fail_aligned_after-- == 0) {
    ++probe::injections;
    return nullptr;
  }
  return __real_aligned_alloc(align, bytes);
}
extern "C" int __wrap_io_submit(io_context_t ctx, long n, iocb **cbs) {
  const int ret = __real_io_submit(ctx, n, cbs);
  if (probe::tracking)
    for (int i = 0; i < ret; ++i) {
      // IOHandler publishes request addresses. AioBatch/stream use small slot
      // numbers and have separate ownership protocols; exclude those contexts.
      if (reinterpret_cast<uintptr_t>(cbs[i]->data) > 4096) {
        probe::Add(ctx, cbs[i]->data);
        if (probe::fail_write_poll && cbs[i]->aio_lio_opcode == IO_CMD_PWRITE) {
          probe::fail_write_poll = false;
          probe::fail_poll = true;
        }
      }
    }
  return ret;
}
extern "C" int __wrap_io_getevents(io_context_t ctx, long min, long max, io_event *events,
                                   timespec *timeout) {
  if (probe::tracking && probe::fail_poll && probe::Count()) {
    probe::fail_poll = false;
    ++probe::injections;
    return -EIO;
  }
  // Leave multiple completions outstanding while application processing runs.
  const int ret =
      __real_io_getevents(ctx, min, probe::tracking ? std::min(max, 1L) : max, events, timeout);
  if (probe::tracking)
    for (int i = 0; i < ret; ++i) {
      // Alter an actual completed request, with other kernel requests still
      // outstanding. Successful io_getevents does not imply this read succeeded.
      if (probe::fail_read_result && probe::Count() > 1 && probe::Contains(ctx, events[i].data)) {
        probe::failed_read_offset = static_cast<const IORequest *>(events[i].data)->offset;
        events[i].res = probe::fail_read_result == 1 ? events[i].res - 1
                                                     : static_cast<decltype(events[i].res)>(-EIO);
        probe::fail_read_result = 0;
        ++probe::injections;
      }
      probe::Remove(ctx, events[i].data);
      ++probe::completions;
    }
  return ret;
}
extern "C" int __wrap_io_destroy(io_context_t ctx) {
  if (probe::tracking)
    for (const auto &p : probe::requests)
      if (p.ctx == ctx && p.request)
        probe::destroy_with_pending = true;
  const int ret = __real_io_destroy(ctx);
  if (!ret)
    for (auto &p : probe::requests)
      if (p.ctx == ctx)
        p = {};
  return ret;
}
extern "C" int __wrap_posix_memalign(void **p, size_t align, size_t bytes) {
  if (probe::fail_align_after >= 0 && probe::fail_align_after-- == 0) {
    ++probe::injections;
    return ENOMEM;
  }
  return __real_posix_memalign(p, align, bytes);
}
static DISTFUNC<float> original_distance;
static float ThrowingDistance(const void *a, const void *b, const void *param) {
  if (probe::fail_distance && probe::Count()) {
    probe::fail_distance = false;
    ++probe::injections;
    throw std::bad_alloc();
  }
  return original_distance(a, b, param);
}
static void Check(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}
struct Scratch {
  fs::path path;
  Scratch() {
    const char *base = std::getenv("TEST_TMPDIR");
    std::string pattern =
        ((base ? fs::path(base) : fs::current_path()) / "io-lifetime-XXXXXX").string();
    std::vector<char> b(pattern.begin(), pattern.end());
    b.push_back(0);
    Check(mkdtemp(b.data()), "mkdtemp");
    path = b.data();
  }
  ~Scratch() {
    std::error_code ec;
    fs::remove_all(path, ec);
  }
};
template <class T> static void Rows(const fs::path &p, const std::vector<T> &v, int d) {
  std::ofstream f(p, std::ios::binary);
  for (size_t i = 0; i < v.size(); i += d) {
    f.write(reinterpret_cast<const char *>(&d), 4);
    f.write(reinterpret_cast<const char *>(v.data() + i), d * sizeof(T));
  }
  Check(bool(f), "write rows");
}
template <class F>
static void Failure(const char *name, F action, int mode, const char *expected_message = nullptr) {
  Check(probe::Count() == 0, "test began with outstanding IO");
  std::cout << "IO_UNWIND_BEGIN " << name << std::endl;
  probe::completions = probe::injections = probe::peak = 0;
  probe::destroy_with_pending = false;
  probe::failed_read_offset = UINT64_MAX;
  probe::tracking = true;
  probe::fail_new = mode == 0;
  probe::fail_distance = mode == 1;
  probe::fail_poll = mode == 2;
  probe::fail_write_poll = mode == 6;
  probe::fail_read_result = mode == 7 ? 1 : (mode == 8 ? 2 : 0);
  probe::fail_align_after = mode == 3 ? 2 : -1;
  probe::fail_aligned_bytes = mode == 5 ? 64 * 4096 : 0;
  probe::fail_aligned_after = (mode == 4 || mode == 5) ? 0 : -1;
  bool caught = false;
  std::string exception_message;
  try {
    action();
  } catch (const std::exception &e) {
    exception_message = e.what();
    caught = !expected_message || exception_message == expected_message;
  }
  probe::fail_new = probe::fail_distance = probe::fail_poll = false;
  probe::fail_align_after = -1;
  probe::fail_write_poll = false;
  probe::fail_read_result = 0;
  probe::fail_aligned_after = -1;
  // Check BEFORE destroying the context; its destructor is too late for local requests.
  const auto check_case = [&](bool ok, const char *message) {
    if (!ok) {
      std::cerr << "IO_UNWIND_FAIL " << name << " injections=" << probe::injections
                << " pending=" << probe::Count() << " actual_exception="
                << (exception_message.empty() ? "<none>" : exception_message) << '\n';
      throw std::runtime_error(std::string(name) + ": " + message);
    }
  };
  check_case(caught && probe::injections == 1, "failure injection did not reach the operation");
  check_case(probe::Count() == 0, "operation released requests with outstanding IO");
  check_case(!probe::destroy_with_pending,
             "operation relied on context destruction to release requests");
  probe::tracking = false;
  ++probe::failure_cases;
  std::cout << "IO_UNWIND_PASS " << name << " completed=" << probe::completions
            << " peak=" << probe::peak << '\n';
}
int main(int argc, char **argv) {
  try {
    omp_set_num_threads(1);
    Scratch scratch;
    const bool baseline = argc == 2 && std::string(argv[1]) == "--regression";
    Check(argc == 1 || baseline, "invalid arguments");
    constexpr size_t n = 512, d = 128;
    std::vector<float> base(n * d), centers(4 * d), query(d);
    std::vector<int> attrs(n), ids(n);
    for (size_t i = 0; i < n; ++i) {
      attrs[i] = int(i);
      ids[i] = i % 4;
      for (size_t j = 0; j < d; ++j)
        base[i * d + j] = std::sin(float(i * 19 + j * 31) * .023f);
    }
    std::copy_n(base.data(), 4 * d, centers.data());
    std::copy_n(base.data() + 7 * d, d, query.data());
    Rows(scratch.path / "centers", centers, int(d));
    Rows(scratch.path / "ids", ids, 1);
    const std::string graph = (scratch.path / "graph").string(),
                      ml = (scratch.path / "ml").string();
    auto owner = std::make_unique<Index>(n, d, 16, 64, "l2", graph, 2, 2, false, 32);
    auto &idx = *owner;
    idx.build_config_.cluster_K_ = 4;
    idx.build_config_.cluster_wow_efc_ = 64;
    idx.build_config_.wow_threshold_ = 64;
    idx.InitRaBitQ((scratch.path / "centers").string(), (scratch.path / "ids").string(), "l2", 3);
    idx.Build(base.data(), attrs.data(), n, 1);
    LayoutExportConfig cfg;
    cfg.block_size = 4096;
    cfg.level_config = MultiLevelConfig{1, 1, 2, {}};
    cfg.method = ShuffleMethod::Identity;
    cfg.out_graph_path = ml;
    cfg.out_meta_path = ml + ".meta";
    ExportMultiLevelLayout(&idx, cfg);
    idx.LoadMultiLevelLayout(ml, ml + ".meta", 1024 * 1024, DynamicBlockCache::EvictPolicy::FIFO);
    idx.per_query_cache_ = true;
    DiskSearchContext ctx(32, graph, ml);
    ctx.filter_mode = DiskSearchContext::FilterMode::Graph;
    ctx.edge_beam_width = 8;
    ctx.vec_beam_width = 8;
    ctx.rerank_beam_width = 8;
    Index::RaBitQEstimator estimator(query.data(), &idx);
    std::vector<dist_id_pair> candidates;
    for (size_t i = 0; i < 32; ++i) {
      const tableint id = (i * 13) % n;
      candidates.emplace_back(estimator.dist_estimate(idx.cids_[id], id), id);
    }
    std::vector<std::pair<dist_t, label_t>> result;
    original_distance = idx.fstdistfunc_;
    idx.fstdistfunc_ = &ThrowingDistance;
    auto rerank = [&] { idx.RerankMultiLevel(query.data(), candidates, &ctx, result); };
    Failure("ml_rerank", rerank, 1);
    if (baseline)
      return 0;
    rerank();
    Check(result.size() == candidates.size(), "rerank retry");
    auto bounded = [&] {
      idx.RerankMultiLevelBounded(query.data(), candidates, candidates.size(), estimator, &ctx,
                                  result);
    };
    Failure("bounded_rerank", bounded, 1);
    bounded();
    Check(result.size() == candidates.size(), "bounded retry");
    std::vector<float> cache;
    std::unordered_map<tableint, size_t> cache_idx;
    auto legacy_rr = [&] {
      idx.RerankPipeline(query.data(), candidates, cache, cache_idx, &ctx, result);
    };
    Failure("record_rerank", legacy_rr, 1);
    legacy_rr();
    Failure("partial_slot_allocation", legacy_rr, 3);
    legacy_rr();

    // Seed distinct neighbor blocks, guaranteeing several live requests before
    // the first completion. Candidate APIs isolate traversal from final rerank.
    std::vector<tableint> route_entries;
    const attr_range<int> full_range{0, int(n - 1)};
    const auto layers = idx.DecideLayerRange(full_range, route_entries);
    const size_t level = idx.ml_layer_to_level_[layers.u_];
    std::vector<dist_id_pair> read_seeds;
    std::vector<int32_t> seed_blocks;
    for (tableint id = 0; id < n && read_seeds.size() < 4; ++id) {
      const auto block = idx.ml_disk_blocks_per_level_[level][id];
      if (std::find(seed_blocks.begin(), seed_blocks.end(), block) != seed_blocks.end())
        continue;
      seed_blocks.push_back(block);
      read_seeds.emplace_back(estimator.dist_estimate(idx.SearchCID(id), id), id);
    }
    Check(read_seeds.size() == 4, "short-read fixture needs four distinct graph blocks");
    const auto entry_samples = ctx.entry_samples;
    ctx.entry_samples = 0;
    auto raw_candidates = [&] {
      cache.clear();
      cache_idx.clear();
      return idx.SearchCandidatesQueryPipeline(read_seeds, query.data(), full_range, layers, 80, 10,
                                               &ctx, cache, cache_idx);
    };
    auto ml_candidates = [&] {
      return idx.SearchCandidatesQueryMultiLevelPipeline(read_seeds, query.data(), full_range,
                                                         layers, 80, 10, &ctx, &estimator);
    };
    for (int mode : {7, 8}) {
      Failure(mode == 7 ? "raw_traversal_short_read" : "raw_traversal_failed_read", raw_candidates,
              mode, "short or failed read in raw graph traversal");
      Check(probe::peak > 1, "raw traversal did not retain outstanding requests");
      Check(!raw_candidates().empty(), "raw traversal retry after failed completion");
      Failure(mode == 7 ? "ml_traversal_short_read" : "ml_traversal_failed_read", ml_candidates,
              mode, "short or failed read in multi-level traversal");
      Check(probe::peak > 1, "ML traversal did not retain outstanding requests");
      // Event order and slot scan order differ: another successful block may
      // legitimately be decoded first. Identify the injected block itself.
      const auto failed_seed =
          std::find_if(read_seeds.begin(), read_seeds.end(), [&](const auto &seed) {
            return probe::failed_read_offset ==
                   idx.ml_header_bytes_ + uint64_t(idx.ml_disk_blocks_per_level_[level][seed.id_]) *
                                              idx.ml_block_size_;
          });
      Check(failed_seed != read_seeds.end(), "failed ML request was not an isolated seed block");
      auto &worker = ctx.Workspace(&idx, idx.ml_cache_config_.generation);
      Check(!worker.Cache().Lookup(failed_seed->id_).found,
            "failed ML block was decoded into the dynamic cache");
      Check(!ml_candidates().empty(), "ML traversal retry after failed completion");
      // Force every candidate through raw-vector IO, then retry on the same context.
      cache.clear();
      cache_idx.clear();
      Failure(mode == 7 ? "raw_rerank_short_read" : "raw_rerank_failed_read", legacy_rr, mode,
              "short or failed read in raw vector rerank");
      legacy_rr();
      Check(result.size() == candidates.size(), "raw rerank retry after failed completion");
    }
    ctx.entry_samples = entry_samples;
    auto legacy = [&] {
      result = idx.searchKNNPipeline(query.data(), 80, 10, attr_range<int>{0, n - 1}, &ctx);
    };
    Failure("record_traversal_allocation", legacy, 0);
    legacy();
    Check(result.size() == 10, "legacy retry");
    auto multi = [&] {
      result =
          idx.searchKNNMultiLevelPipeline(query.data(), 80, 10, attr_range<int>{0, n - 1}, &ctx);
    };
    Failure("ml_traversal_allocation", multi, 0);
    multi();
    Check(result.size() == 10, "ML retry");
    Failure("final_rerank_distance", multi, 1);
    multi();
    Check(result.size() == 10, "final rerank retry");
    Failure("poll_failure", rerank, 2);
    rerank();
    // Serial build IO failure must unwind the raw stage buffer as well as requests.
    for (int mode : {2, 6}) {
      // Fresh graph per fault: never retry a partly written resident phase.
      Index resident(n, d, 16, 64, "l2", (scratch.path / ("left_" + std::to_string(mode))).string(),
                     2, 2, false, 32);
      resident.build_config_.left_stream_batch_ = 300;
      resident.build_config_.merge_build_threads_ = 1;
      resident.PopulateElements(base.data(), attrs.data(), n);
      resident.graph_file_->Sync();
      const auto old_id_only = resident.id_only_start_layer_;
      Failure(
          mode == 2 ? "left_stream_read_poll" : "left_stream_write_poll",
          [&] { resident.BuildLeftStreamLayers(1, n); }, mode);
      Check(probe::peak > 1, "LeftStream fixture did not submit multiple extents");
      Check(!probe::destroy_with_pending,
            "LeftStream relied on context destruction to release requests");
      Check(!resident.left_stream_complete_ && resident.last_completed_layer_ == 0 &&
                resident.id_only_start_layer_ == old_id_only,
            "failed resident phase marked graph complete");
    }
    {
      IOHandler io(64);
      io.Open(graph);
      bool buffer_released = false;
      size_t pending_at_release = 1;
      auto partition_read = [&] {
        test_support::ReadRecordBatch(io, idx.graph_file_->node_offset(0), idx.node_disk_bytes_, 64,
                                      [&]() noexcept {
                                        pending_at_release = probe::Count();
                                        buffer_released = true;
                                      });
      };
      Failure("partition_poll", partition_read, 2);
      Check(probe::peak > 1, "partition fixture did not submit a batch");
      Check(buffer_released && pending_at_release == 0,
            "partition buffer released with outstanding IO");
    }
    {
      IOHandler io(64);
      io.Open(idx.build_vec_file_path_);
      std::vector<uint32_t> ranks(n);
      std::iota(ranks.begin(), ranks.end(), 0);
      std::vector<float> gathered(n * d);
      auto gather = [&] { idx.GatherVecsByRank(io, ranks.data(), nullptr, n, gathered.data()); };
      Failure("gather_poll", gather, 2);
      gather();
      Check(gathered == base, "gather retry");
    }
    idx.ClusterPrepDisk(nullptr, nullptr, n, 4);
    const std::vector<size_t> half_window{idx.window_size_[2] / 2};
    const size_t head = 2 * sizeof(uint32_t);
    auto make_scratch = [&] {
      return std::make_unique<disk_build::ClusterScratchDirectory>(
          graph, idx.ClusterScratch()->cluster_rank_offsets_.data(), idx.ClusterScratch()->K);
    };
    auto make_plan = [&](disk_build::ClusterScratchDirectory &owner) {
      return disk_build::ClusterSpillPlan{
          2,    n, 1, head, head + idx.M_ * sizeof(tableint), half_window, graph, idx.PlanPhase2(n),
          owner};
    };
    // Every fault owns a fresh generation. Scratch destruction must occur after
    // pending requests unwind, including faults inside input distribution.
    for (int mode : {4, 6}) {
      std::string generation;
      auto pass_a = [&] {
        auto stage_owner = make_scratch();
        generation = stage_owner->Path();
        const auto plan = make_plan(*stage_owner);
        DiskBuildTestPeer::BuildClusterPass(idx, plan);
      };
      Failure(mode == 4 ? "cluster_input_buffer_allocation" : "cluster_input_write_poll", pass_a,
              mode);
      Check(!generation.empty() && !fs::exists(generation), "failed pass A retained owned scratch");
    }
    // Pass B consumes and deletes exhausted streams. Rebuild its disposable
    // inputs outside each injection window; never retry partly consumed files.
    for (int mode : {5, 6}) {
      auto prepared = make_scratch();
      const std::string generation = prepared->Path();
      {
        const auto plan = make_plan(*prepared);
        DiskBuildTestPeer::BuildClusterPass(idx, plan);
      }
      auto pass_b = [&] {
        auto stage_owner = std::move(prepared);
        const auto plan = make_plan(*stage_owner);
        DiskBuildTestPeer::CommitClusterPass(idx, plan);
        stage_owner->Complete();
      };
      Failure(mode == 5 ? "pass_b_staging_allocation" : "pass_b_write_poll", pass_b, mode);
      Check(!prepared && !fs::exists(generation), "failed pass B retained owned scratch");
    }
    {
      auto stage_owner = make_scratch();
      const std::string generation = stage_owner->Path();
      const auto plan = make_plan(*stage_owner);
      DiskBuildTestPeer::BuildClusterPass(idx, plan);
      DiskBuildTestPeer::CommitClusterPass(idx, plan);
      stage_owner->Complete();
      Check(!fs::exists(generation), "successful cluster passes retained owned scratch");
    }
    idx.fstdistfunc_ = original_distance;
    Check(probe::Count() == 0, "pending IO at exit");
    std::cout << "IO_OPERATION_LIFETIME_PASS failures=" << probe::failure_cases << "\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
}
