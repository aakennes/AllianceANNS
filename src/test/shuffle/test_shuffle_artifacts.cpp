#include "shuffle/stages/artifacts.hpp"
#include "index/index_disk.hpp"
#include "shuffle/layout.hpp"
#include <filesystem>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <sys/file.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <unistd.h>

namespace fs = std::filesystem;
using alliance::shuffle_detail::ShuffleArtifacts;
namespace {
void Check(bool valid, const char *message) {
  if (!valid)
    throw std::runtime_error(message);
}
void Put(const fs::path &path, const std::string &text) {
  std::ofstream(path) << text;
}
std::string Get(const fs::path &path) {
  std::ifstream stream(path);
  return std::string(std::istreambuf_iterator<char>(stream), {});
}
template <class Function> void Reject(Function fn, const char *message) {
  bool rejected = false;
  try {
    fn();
  } catch (const std::exception &) {
    rejected = true;
  }
  Check(rejected, message);
}
} // namespace

int main() {
  const char *configured_tmp = std::getenv("TEST_TMPDIR");
  const fs::path temporary_root =
      configured_tmp ? fs::path(configured_tmp) : fs::temp_directory_path();
  std::string pattern = (temporary_root / "alliance-shuffle-artifact-test-XXXXXX").string();
  char *name = ::mkdtemp(pattern.data());
  if (!name)
    return 1;
  const fs::path root(name), input = root / "input", meta = root / "input.meta";
  try {
    Put(input, "source");
    Put(meta, "metadata");
    const auto graph = root / "graph", output_meta = root / "meta";
    {
      ShuffleArtifacts artifacts(input, meta, graph, output_meta);
      Put(artifacts.Graph(), "graph bytes");
      Put(artifacts.Layout(), "mapping bytes");
      Put(artifacts.Metadata(), "metadata");
      artifacts.Commit(7, 4, 2, 3, 4096, 8192);
    }
    const auto receipt = Get(graph.string() + ".completion.json");
    Check(receipt.find("\"status\":\"complete\"") != std::string::npos &&
              receipt.find("\"sha256\":") != std::string::npos && fs::file_size(graph) == 11,
          "durable publication missing receipt or data");
    Reject([&] { ShuffleArtifacts again(input, meta, graph, output_meta); },
           "existing output overwritten");
    Check(Get(graph) == "graph bytes", "existing output changed");
    Reject([&] { ShuffleArtifacts bad(input, meta, input, root / "x"); },
           "input output alias accepted");
    fs::create_symlink(input, root / "symlink");
    Reject([&] { ShuffleArtifacts bad(root / "symlink", meta, root / "a", root / "b"); },
           "symlink input accepted");
    fs::create_directories(root / "real" / "child");
    Put(root / "real" / "input", "different source");
    fs::create_directory_symlink(root / "real" / "child", root / "parent-link");
    Reject(
        [&] {
          ShuffleArtifacts bad(root / "parent-link" / ".." / "input", meta, root / "a", root / "b");
        },
        "symlink hidden by parent traversal accepted");
    Reject(
        [&] { ShuffleArtifacts bad(input, meta, root / "parent-link" / ".." / "a", root / "b"); },
        "output symlink hidden by parent traversal accepted");
    {
      ShuffleArtifacts canonical(root / "real" / ".." / "input", meta, root / "canon",
                                 root / "canon.meta");
      Check(canonical.InputGraph() == input.string(), "lock/read path normalization differs");
    }

    // Force a publish collision after the private stage is created. Only this
    // session's already-linked outputs may roll back; the foreign file survives.
    std::string staging;
    {
      ShuffleArtifacts artifacts(input, meta, root / "race.graph", root / "race.meta");
      staging = fs::path(artifacts.Graph()).parent_path();
      Put(artifacts.Graph(), "staged graph");
      Put(artifacts.Layout(), "layout");
      Put(artifacts.Metadata(), "metadata");
      Put(root / "race.graph.layout", "foreign");
      Reject([&] { artifacts.Commit(7, 4, 2, 2, 4096, 4096); },
             "publication replaced racing output");
      artifacts.Fail("expected collision");
    }
    Check(!fs::exists(root / "race.graph") && !fs::exists(root / "race.graph.completion.json") &&
              Get(root / "race.graph.layout") == "foreign",
          "rollback removed foreign output or retained own output");
    Check(!fs::exists(fs::path(staging) / "graph") && fs::exists(fs::path(staging) / "failure.txt"),
          "failed session must remove data and retain diagnostics");

    // The directory lock serializes even distinct output prefixes.
    {
      ShuffleArtifacts first(input, meta, root / "lock.graph", root / "lock.meta");
      Reject(
          [&] { ShuffleArtifacts second(input, meta, root / "lock2.graph", root / "lock2.meta"); },
          "concurrent output session accepted");
    }
    const int source_lock = ::open((input.string() + ".build.lock").c_str(), O_RDWR);
    Check(source_lock >= 0 && ::flock(source_lock, LOCK_EX | LOCK_NB) == 0, "source test lock");
    Reject([&] { ShuffleArtifacts blocked(input, meta, root / "blocked", root / "blocked.meta"); },
           "shuffle ran against an active builder");
    ::close(source_lock);
    {
      // A corrupt/unreadable source graph must not be touched when the requested
      // vector page is already known to be impossible from metadata geometry.
      const auto source = root / "geometry-source";
      auto index = std::make_unique<alliance::AllianceIndexDisk<int, float>>(
          4, 256, 4, 8, "l2", source.string(), 2, 2, false);
      std::vector<float> vectors(4 * 256, 0.0f);
      const int attributes[] = {0, 1, 2, 3};
      index->PopulateElements(vectors.data(), attributes, 4);
      fs::resize_file(source, 0);
      alliance::LayoutExportConfig config;
      config.vector_block_size = 512;
      config.out_graph_path = (root / "bad-geometry").string();
      config.out_meta_path = (root / "bad-geometry.layout").string();
      std::string failure;
      try {
        alliance::ExportMultiLevelLayout(index.get(), config);
      } catch (const std::exception &error) {
        failure = error.what();
      }
      Check(failure.find("vector does not fit") != std::string::npos,
            "invalid vector geometry reached the graph ordering pass");
      Check(!fs::exists(config.out_graph_path), "invalid geometry created data output");
    }
    fs::remove_all(root);
    std::cout << "shuffle artifact lifecycle passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << "; retained fixture: " << root << '\n';
    return 1;
  }
}
