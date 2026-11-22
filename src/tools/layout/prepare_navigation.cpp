// Preparation-only tool. Builds a small RAM navigation sidecar, never the SSD graph.
#include "index/index_disk.hpp"
#include "search/memory_navigation.hpp"
#include "disk_build/cluster_wow/graph.hpp"
#include <cerrno>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <iostream>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>

namespace {
using Index = alliance::AllianceIndexDisk<int, float>;
using Navigation = alliance::MemoryNavigationGraph;

struct Options {
  std::string graph, meta, ml_graph, ml_meta, output, space = "ip";
  std::string counts = "16384,32768,65536";
  size_t degree = 16, efc = 256;
  int threads = 16;
  bool validate_only = false;
};

size_t Positive(const std::string &text) {
  if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
    throw std::invalid_argument("expected a positive integer: " + text);
  size_t consumed = 0;
  auto value = std::stoull(text, &consumed);
  if (!value || consumed != text.size() || value > SIZE_MAX)
    throw std::invalid_argument("integer out of range: " + text);
  return static_cast<size_t>(value);
}

Options Parse(int argc, char **argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (key == "--validate_only") {
      options.validate_only = true;
      continue;
    }
    if (++i == argc)
      throw std::invalid_argument("missing value for " + key);
    const std::string value = argv[i];
    if (key == "--graph_file") options.graph = value;
    else if (key == "--meta_file") options.meta = value;
    else if (key == "--ml_graph_file") options.ml_graph = value;
    else if (key == "--ml_meta_file") options.ml_meta = value;
    else if (key == "--output") options.output = value;
    else if (key == "--space") options.space = value;
    else if (key == "--nodes") options.counts = value;
    else if (key == "--degree") options.degree = Positive(value);
    else if (key == "--efc") options.efc = Positive(value);
    else if (key == "--threads") {
      const auto threads = Positive(value);
      if (threads > 1024) throw std::invalid_argument("too many threads");
      options.threads = static_cast<int>(threads);
    } else throw std::invalid_argument("unknown option: " + key);
  }
  if (options.graph.empty() || options.meta.empty() || options.ml_graph.empty() ||
      options.ml_meta.empty() || options.output.empty() ||
      (options.space != "ip" && options.space != "l2") ||
      options.degree > 1024 || options.efc < options.degree || options.efc > 65536)
    throw std::invalid_argument("invalid navigation preparation arguments");
  if (!options.validate_only && std::filesystem::exists(options.output))
    throw std::invalid_argument("refusing to replace an existing navigation sidecar");
  return options;
}

void WriteAll(int fd, const void *data, size_t bytes) {
  const auto *next = static_cast<const char *>(data);
  while (bytes) {
    const ssize_t written = ::write(fd, next, bytes);
    if (written < 0 && errno == EINTR) continue;
    if (written <= 0) throw std::runtime_error("navigation write failed");
    next += written;
    bytes -= static_cast<size_t>(written);
  }
}

// Preserve NVGRAPH1 exactly. The current production reader independently checks
// the checksum, every edge/rank, nesting, geometry and new-index identity.
void Publish(const std::string &output, const Navigation &navigation, const Index &index) {
  const std::string temporary = output + ".partial";
  int fd = ::open(temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
  if (fd < 0) throw std::runtime_error("cannot create navigation staging file: " + temporary);
  try {
    uint64_t hash = 14695981039346656037ULL;
    auto bytes = [&](const void *data, size_t count) {
      WriteAll(fd, data, count);
      hash = Navigation::Hash(data, count, hash);
    };
    auto word = [&](uint64_t value) { bytes(&value, sizeof(value)); };
    word(0x314850415247564EULL); word(1);
    word(navigation.population); word(navigation.dimension);
    word(navigation.order_hash); word(navigation.metric);
    word(navigation.degree); word(navigation.windows.size()); word(navigation.groups.size());
    bytes(navigation.windows.data(), navigation.windows.size() * sizeof(uint64_t));
    for (const auto &group : navigation.groups) {
      word(group.first); word(group.last); word(group.ids.size());
      for (const auto *values : {&group.ranks, &group.ids, &group.edges})
        bytes(values->data(), values->size() * sizeof(uint32_t));
    }
    WriteAll(fd, &hash, sizeof(hash));
    if (::fsync(fd)) throw std::runtime_error("navigation fsync failed");
    if (::close(fd)) { fd = -1; throw std::runtime_error("navigation close failed"); }
    fd = -1;
    auto checked = Navigation::Load(temporary, index.curvec_num_, index.vec_d_);
    checked.Match(index, navigation.metric);
    if (checked.degree != navigation.degree || checked.groups.size() != navigation.groups.size())
      throw std::runtime_error("navigation readback header mismatch");
    for (size_t i = 0; i < checked.groups.size(); ++i) {
      const auto &a = checked.groups[i], &b = navigation.groups[i];
      if (a.first != b.first || a.last != b.last || a.ranks != b.ranks || a.ids != b.ids || a.edges != b.edges)
        throw std::runtime_error("navigation readback payload mismatch");
    }
    // link() is atomic and rejects an existing final name; no overwrite race.
    if (::link(temporary.c_str(), output.c_str()))
      throw std::runtime_error("navigation commit failed: " + std::string(std::strerror(errno)));
    if (::unlink(temporary.c_str())) throw std::runtime_error("navigation staging unlink failed");
    auto parent = std::filesystem::path(output).parent_path();
    if (parent.empty()) parent = ".";
    const int directory = ::open(parent.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (directory < 0) throw std::runtime_error("cannot open navigation output directory");
    const int result = ::fsync(directory);
    ::close(directory);
    if (result) throw std::runtime_error("navigation directory fsync failed");
  } catch (...) {
    if (fd >= 0) ::close(fd);
    // Retain any partial file as failure evidence; never silently retry.
    throw;
  }
}

void Describe(const Navigation &navigation) {
  std::cout << "# nav_verified: nodes=" << navigation.population << " dimensions=" << navigation.dimension
            << " degree=" << navigation.degree << " order_hash=" << navigation.order_hash
            << " bytes=" << navigation.Bytes() << " groups=" << navigation.groups.size() << '\n';
  for (size_t i = 0; i < navigation.groups.size(); ++i) {
    const auto &g = navigation.groups[i];
    std::cout << "# nav_group: group=" << i << " first=" << g.first << " last=" << g.last
              << " nodes=" << g.ids.size() << '\n';
  }
}
} // namespace

int main(int argc, char **argv) {
  try {
    const Options options = Parse(argc, argv);
    const auto started = std::chrono::steady_clock::now();
    Index index(options.graph, options.meta, options.space);
    index.LoadMultiLevelLayout(options.ml_graph, options.ml_meta, 0);
    if (index.ml_layer_stride_ != 1)
      throw std::runtime_error("navigation preparation requires all basic layers");
    const uint64_t metric = options.space == "ip" ? 1 : 2;
    if (options.validate_only) {
      auto navigation = Navigation::Load(options.output, index.curvec_num_, index.vec_d_);
      navigation.Match(index, metric);
      Describe(navigation);
      return 0;
    }

    std::vector<size_t> counts;
    std::stringstream parts(options.counts);
    std::string part;
    while (std::getline(parts, part, ',')) counts.push_back(Positive(part));
    if (counts.size() != index.ml_levels_.size())
      throw std::invalid_argument("--nodes needs one count for each SSD owner group");
    for (size_t i = 0; i < counts.size(); ++i)
      if (counts[i] > index.curvec_num_ ||
          (i && (counts[i] < counts[i - 1] || counts[i] % counts[i - 1])))
        throw std::invalid_argument("representative counts must be nested integer multiples <= N");

    Navigation navigation;
    navigation.population = index.curvec_num_;
    navigation.dimension = index.vec_d_;
    navigation.degree = options.degree;
    navigation.metric = metric;
    navigation.order_hash = Navigation::OrderHash(index.sorted_ids_, index.atts_);
    navigation.windows.assign(index.window_size_.begin(), index.window_size_.begin() + index.cur_max_layer_ + 1);
    // Representatives use global attribute ranks. If even the closest pair is
    // farther apart than the half-window, layer zero cannot contain an edge.
    const bool skip_bottom = navigation.windows.size() > 1 &&
        navigation.population / counts.back() > navigation.windows.front() / 2;
    if (skip_bottom && index.ml_levels_.back().owner_end == 0) {
      const size_t released = counts.back();
      counts.pop_back();
      // Preserve original group weights; largest remainders distribute the
      // rounding residue without changing the total representative budget.
      uint64_t weight = 0;
      for (size_t count : counts) weight += count;
      std::vector<std::pair<uint64_t, size_t>> remainders;
      size_t assigned = 0;
      for (size_t g = 0; g < counts.size(); ++g) {
        const uint64_t numerator = uint64_t(released) * counts[g];
        const size_t extra = numerator / weight;
        remainders.emplace_back(numerator % weight, g);
        counts[g] += extra;
        assigned += extra;
      }
      std::sort(remainders.rbegin(), remainders.rend());
      for (size_t i = 0; i < released - assigned; ++i)
        ++counts[remainders[i].second];
      if (counts.back() > navigation.population)
        throw std::invalid_argument("redistributed group exceeds population");
    }
    // Select each coarser set from its immediate successor. This preserves
    // nesting when redistribution produces counts without integer ratios.
    std::vector<std::vector<size_t>> slots(counts.size());
    for (size_t g = counts.size(); g-- > 0;) {
      for (size_t i = 0; i < counts[g]; ++i)
        slots[g].push_back(g + 1 == counts.size() ? i : slots[g + 1][i * counts[g + 1] / counts[g]]);
    }
    const size_t finest = counts.back();
    std::vector<uint32_t> ranks(finest);
    for (size_t i = 0; i < finest; ++i)
      ranks[i] = ((2 * uint64_t(i) + 1) * navigation.population) / (2 * uint64_t(finest));

    for (size_t gi = 0; gi < counts.size(); ++gi) {
      const auto &level = index.ml_levels_[gi];
      Navigation::Group group;
      group.first = skip_bottom ? std::max<size_t>(1, level.owner_begin) : level.owner_begin;
      group.last = level.owner_end;
      const size_t count = counts[gi];
      if (group.first > group.last || group.last >= navigation.windows.size())
        throw std::runtime_error("invalid SSD owner range");
      group.ranks.reserve(count);
      group.ids.reserve(count);
      std::vector<float> vectors(count * navigation.dimension);
      // Same nested rank representatives as the historical builder. Only these
      // vectors are read; the former full rank-ordered .rawvecs copy is unnecessary.
      for (size_t i = 0; i < count; ++i) {
        const uint32_t rank = ranks[slots[gi][i]], id = index.sorted_ids_[rank];
        group.ranks.push_back(rank);
        group.ids.push_back(id);
        index.ReadVec(id, vectors.data() + i * navigation.dimension);
      }
      if (std::any_of(vectors.begin(), vectors.end(), [](float v) { return !std::isfinite(v); }))
        throw std::runtime_error("non-finite representative vector");
      std::vector<size_t> half{navigation.windows[group.first] / 2};
      for (size_t layer = group.first; layer <= group.last; ++layer)
        half.push_back(navigation.windows[layer] / 2);
      auto distance = [&](uint32_t a, uint32_t b) {
        return index.fstdistfunc_(vectors.data() + size_t(a) * navigation.dimension,
                                  vectors.data() + size_t(b) * navigation.dimension, index.dist_func_param_);
      };
      alliance::disk_build::ClusterWowGraph<decltype(distance)> wow(
          group.ranks.data(), count, options.degree, half, distance);
      for (size_t i = 0; i < count; ++i) wow.Seed(i, {});
      wow.Build(options.efc, 1, options.threads);
      group.edges.assign((group.last - group.first + 1) * count * options.degree, Navigation::invalid);
      for (size_t layer = group.first; layer <= group.last; ++layer) {
        size_t total = 0, isolated = 0;
        for (size_t i = 0; i < count; ++i) {
          const auto edges = wow.Edges(i, layer - group.first + 1);
          total += edges.size();
          isolated += edges.empty();
          for (size_t k = 0; k < edges.size(); ++k)
            group.edges[((layer - group.first) * count + i) * options.degree + k] = edges[k].id_;
        }
        std::cout << "# nav_layer: group=" << gi << " layer=" << layer << " window=" << navigation.windows[layer]
                  << " nodes=" << count << " mean_degree=" << double(total) / count
                  << " isolated=" << isolated << std::endl;
      }
      navigation.groups.push_back(std::move(group));
    }
    navigation.Validate();
    navigation.Match(index, metric);
    Publish(options.output, navigation, index);
    auto loaded = Navigation::Load(options.output, index.curvec_num_, index.vec_d_);
    loaded.Match(index, metric);
    Describe(loaded);
    std::cout << "# nav_build: status=passed efc=" << options.efc << " passes=1 threads=" << options.threads
              << " seconds=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()
              << " formal_queries=0\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "navigation preparation failed: " << error.what() << '\n';
    return 1;
  }
}
