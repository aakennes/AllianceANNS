// Ownership contracts on normal exits, constructor failures and build retries.
#include "index/index_disk.hpp"
#include "disk_build/partition/membership.hpp"
#include <dirent.h>
#include <cstdlib>
#include <new>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <omp.h>
#include <unistd.h>

using Index = alliance::AllianceIndexDisk<int, float>;
namespace fs = std::filesystem;
// Inject only serial CSR setup allocations; production interfaces need no test hook.
static int fail_new_after = -1;
void *operator new(size_t bytes) {
  if (fail_new_after >= 0 && fail_new_after-- == 0)
    throw std::bad_alloc();
  if (void *p = std::malloc(bytes ? bytes : 1))
    return p;
  throw std::bad_alloc();
}
void *operator new[](size_t bytes) {
  return ::operator new(bytes);
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
static void Check(bool pass, const char *message) {
  if (!pass)
    throw std::runtime_error(message);
}
static size_t Fds() {
  DIR *dir = opendir("/proc/self/fd");
  Check(dir, "opendir");
  size_t count = 0;
  while (auto *e = readdir(dir))
    if (e->d_name[0] != '.')
      ++count;
  closedir(dir);
  return count;
}
template <class F> static void Reject(F action, const char *text) {
  try {
    action();
  } catch (const std::exception &e) {
    Check(std::string(e.what()).find(text) != std::string::npos, "wrong failure");
    return;
  }
  throw std::runtime_error("expected runtime failure");
}
template <class F> static void NoMemory(int after, F action) {
  fail_new_after = after;
  bool caught = false;
  try {
    action();
  } catch (const std::bad_alloc &) {
    caught = true;
  }
  fail_new_after = -1;
  Check(caught, "allocation failure not propagated");
}
struct Scratch {
  fs::path path;
  Scratch() {
    const char *base = std::getenv("TEST_TMPDIR");
    std::string name = ((base ? fs::path(base) : fs::current_path()) / "lifetime-XXXXXX").string();
    std::vector<char> b(name.begin(), name.end());
    b.push_back(0);
    Check(mkdtemp(b.data()), "mkdtemp");
    path = b.data();
  }
  ~Scratch() {
    std::error_code ec;
    fs::remove_all(path, ec);
  }
};
static std::vector<char> Read(const fs::path &p) {
  std::ifstream f(p, std::ios::binary);
  Check(bool(f), "open fixture");
  return {std::istreambuf_iterator<char>(f), {}};
}
template <class T> static void Rows(const fs::path &p, const std::vector<T> &v, int d) {
  std::ofstream f(p, std::ios::binary);
  for (size_t i = 0; i < v.size(); i += d) {
    f.write(reinterpret_cast<const char *>(&d), sizeof(d));
    f.write(reinterpret_cast<const char *>(v.data() + i), d * sizeof(T));
  }
  Check(bool(f), "write rows");
}
struct Fixture {
  static constexpr size_t n = 32, d = 128, K = 2;
  fs::path dir;
  std::vector<float> base, centers;
  std::vector<int> attrs;
  std::vector<uint32_t> ids;
  std::unique_ptr<Index> index;
  explicit Fixture(const fs::path &path, bool quant = false)
      : dir(path), base(n * d), centers(K * d), attrs(n), ids(2 * n) {
    fs::create_directories(dir);
    for (size_t i = 0; i < n; ++i) {
      attrs[i] = int(i);
      ids[2 * i] = i % K;
      ids[2 * i + 1] = (i + 1) % K;
      for (size_t j = 0; j < d; ++j)
        base[i * d + j] = float((i * 17 + j * 11) % 71) * .01f;
    }
    for (size_t c = 0; c < K; ++c)
      std::copy_n(base.data() + c * d, d, centers.data() + c * d);
    index = std::make_unique<Index>(n, d, 8, 24, "l2", (dir / "graph").string(), 2, 0, false, 128);
    if (quant) {
      Rows(dir / "centroids", centers, int(d));
      std::vector<int> search_ids(n);
      for (size_t i = 0; i < n; ++i)
        search_ids[i] = int(i % K);
      Rows(dir / "ids", search_ids, 1);
      index->InitRaBitQ((dir / "centroids").string(), (dir / "ids").string(), "l2", 3);
    }
    index->PopulateElements(base.data(), attrs.data(), n);
    index->build_config_.merge_build_threads_ = 1;
    index->build_config_.cluster_K_ = K;
    index->cluster_K_file_ = K;
    index->cluster_centroids_ext_ = centers.data();
    index->cluster_ids_ext_ = ids.data();
  }
  void Setup() {
    index->SetupClusterBuildDisk(ids.data(), K, n);
  }
  void Prep() {
    index->ClusterPrepDisk(centers.data(), ids.data(), n, K);
  }
};
static void Constructors(const fs::path &dir) {
  const size_t fds = Fds();
  for (int repeat = 0; repeat < 3; ++repeat) {
    Fixture f(dir, true);
    auto &idx = *f.index;
    idx.Save((dir / "meta").string());
    {
      auto loaded =
          std::make_unique<Index>((dir / "graph").string(), (dir / "meta").string(), "l2");
      Check(loaded->quant_vecs_ == idx.quant_vecs_, "loaded codes changed");
    }
    // A controlled inconsistent padded dimension fails after table/rotator allocation.
    const size_t dim = idx.padded_dim_;
    // Patch only the dimension field in a valid saved metadata file.
    auto bytes = Read(dir / "meta");
    const size_t offset =
        sizeof(idx.max_elements_) + sizeof(idx.vec_d_) + sizeof(idx.wp_) + sizeof(idx.o_) +
        sizeof(idx.M_) + sizeof(idx.efc_) + sizeof(idx.curvec_num_) + sizeof(idx.cur_max_layer_) +
        sizeof(idx.node_record_bytes_) + sizeof(idx.node_disk_bytes_) + sizeof(size_t) +
        idx.window_size_.size() * sizeof(size_t) +
        Fixture::n * (sizeof(alliance::label_t) + sizeof(int) + 2 * sizeof(alliance::tableint)) +
        sizeof(bool);
    const size_t wrong_dim = dim + 64;
    std::memcpy(bytes.data() + offset, &wrong_dim, sizeof(wrong_dim));
    {
      std::ofstream out(dir / "bad-meta", std::ios::binary);
      out.write(bytes.data(), bytes.size());
    }
    const size_t active = Fds();
    Reject(
        [&] {
          auto p =
              std::make_unique<Index>((dir / "graph").string(), (dir / "bad-meta").string(), "l2");
        },
        "padded_dim mismatch");
    Reject(
        [&] {
          auto p =
              std::make_unique<Index>(32, 128, 8, 24, "l2", (dir / "missing" / "graph").string());
        },
        "open");
    Check(Fds() == active, "constructor failure leaked fd");
  }
  Check(Fds() == fds, "index destruction leaked fd");
}
static void BuildFailure(const fs::path &dir, bool only_fd) {
  Fixture f(dir, true);
  auto &idx = *f.index;
  idx.Save((dir / "checkpoint").string());
  const auto graph = Read(dir / "graph"), vecs = Read(idx.build_vec_file_path_),
             ckpt = Read(dir / "checkpoint");
  const size_t fds = Fds();
  idx.build_config_.build_mem_gb_ = 1e-9;
  Reject([&] { idx.BuildPostPopulate(1); }, "--build_mem_gb");
  Check(Fds() == fds, "failed build leaked descriptors");
  Check(!idx.ClusterScratch(), "failed build kept cluster state");
  Check(Read(dir / "graph") == graph && Read(idx.build_vec_file_path_) == vecs &&
            Read(dir / "checkpoint") == ckpt,
        "failure modified recovery files");
  if (only_fd)
    return;
  idx.build_config_.build_mem_gb_ = 0;
  idx.BuildPostPopulate(1);
  Check(!idx.ClusterScratch() && Fds() == fds, "successful retry leaked resources");
  auto result = idx.searchKNN(f.base.data(), 32, 5, alliance::attr_range<int>{0, 31});
  Check(result.size() == 5 && result.front().second == 0, "retry graph is not searchable");
}
static void ClusterFailures(const fs::path &dir) {
  const size_t initial = Fds();
  {
    Fixture f(dir);
    auto &idx = *f.index;
    const auto centers = f.centers;
    const auto ids = f.ids;
    for (int i = 0; i < 4; ++i) {
      f.Setup();
      NoMemory(i, [&] { f.Setup(); });
      Check(!idx.ClusterScratch(), "failed CSR setup kept partial state");
      f.Prep();
      Check(idx.ClusterScratch() && idx.ClusterScratch()->cluster_ranks_.size() == 2 * Fixture::n,
            "nearest-two CSR retry did not finish");
      idx.TeardownClusterBuildDisk();
      Check(!idx.ClusterScratch(), "explicit cluster teardown retained state");
    }
    const size_t fds = Fds();
    f.ids[1] = f.ids[0];
    Reject([&] { f.Prep(); }, "nearest-two membership");
    Check(!idx.ClusterScratch() && Fds() == fds, "invalid partition retained resources");
    f.ids = ids;
    f.Prep();
    Check(f.centers == centers && f.ids == ids, "borrowed input arrays changed");
    // Intentionally omit teardown: index destruction owns the surviving state.
  }
  Check(Fds() == initial, "cluster destruction leaked fd");
}
int main(int argc, char **argv) {
  try {
    omp_set_num_threads(1);
    Scratch scratch;
    const std::string mode = argc == 2 ? argv[1] : "";
    Check(argc == 1 || mode == "--fd-regression" || mode == "--destruction-only",
          "invalid argument");
    if (mode != "--fd-regression")
      Constructors(scratch.path / "constructors");
    if (mode != "--destruction-only") {
      BuildFailure(scratch.path / "build", mode == "--fd-regression");
      if (mode.empty())
        ClusterFailures(scratch.path / "cluster");
    }
    std::cout << "INDEX_LIFETIME_PASS mode=" << mode << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
}
