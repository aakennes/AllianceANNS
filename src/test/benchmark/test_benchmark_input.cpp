#include "benchmark/strict_input.hpp"
#include <cstdlib>
#include <iostream>

namespace fs = std::filesystem;
namespace input = benchmark::formal;
void Check(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
template<class Action> void Reject(Action action) {
  bool rejected = false;
  try { action(); } catch (const std::exception&) { rejected = true; }
  Check(rejected, "malformed input accepted");
}
template<class T> void Write(const fs::path& path, const std::vector<T>& values) {
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  file.write(reinterpret_cast<const char*>(values.data()), values.size()*sizeof(T));
  Check(bool(file), "fixture write failed");
}
struct Scratch {
  fs::path path;
  Scratch() {
    const char* root = std::getenv("TEST_TMPDIR");
    auto pattern = ((root ? fs::path(root) : fs::current_path())/"benchmark-input-XXXXXX").string();
    std::vector<char> name(pattern.begin(), pattern.end()); name.push_back(0);
    Check(mkdtemp(name.data()) != nullptr, "mkdtemp failed"); path = name.data();
  }
  ~Scratch() { std::error_code error; fs::remove_all(path, error); }
};
int main() {
  try {
    Scratch scratch;
    const auto query=scratch.path/"query.fvecs", ranges=scratch.path/"ranges.bin", gt=scratch.path/"gt.ivecs";
    // Floating-point zero payloads; explicit row headers are the only nonzero words.
    std::vector<uint32_t> queries{3,0,0,0,3,0,0,0};
    Write(query,queries);
    auto loaded=input::ReadQueries(query.string(),3,2);
    Check(loaded[0]==0 && loaded[5]==0,"valid query read changed data");
    queries[4]=1000000; Write(query,queries);
    Reject([&]{input::ReadQueries(query.string(),3,2);});
    queries[4]=3; queries.push_back(0); Write(query,queries);
    Reject([&]{input::ReadQueries(query.string(),3,2);});
    queries.resize(7); Write(query,queries);
    Reject([&]{input::ReadQueries(query.string(),3,2);});
    Reject([&]{input::ReadQueries(query.string(),0,2);});
    Write<int32_t>(ranges,{-10,20,0,30});
    auto bounds=input::ReadRanges(ranges.string(),2);
    Check(bounds[0].l_==-10 && bounds[1].u_==30,"valid ranges changed");
    Write<int32_t>(ranges,{20,-10,0,30});
    Reject([&]{input::ReadRanges(ranges.string(),2);});
    Write<uint32_t>(gt,{2,3,7,2,1,5});
    auto truth=input::ReadGroundTruth(gt.string(),2,2);
    Check(truth[0][1]==7 && truth[1][0]==1,"valid GT changed");
    Write<uint32_t>(gt,{2,3,7,1,1,5});
    Reject([&]{input::ReadGroundTruth(gt.string(),2,2);});
    Write<uint32_t>(gt,{2,3,7,2,1});
    Reject([&]{input::ReadGroundTruth(gt.string(),2,2);});
    Reject([&]{input::InputBytes(std::numeric_limits<uint64_t>::max(),2);});
    std::cout<<"BENCHMARK_INPUT_PASS exact rows, dimension headers, short/trailing data, ranges and GT\n";
    return 0;
  } catch(const std::exception& error) { std::cerr<<error.what()<<'\n';return 1; }
}
