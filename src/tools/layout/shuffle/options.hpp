#pragma once
#include "shuffle/config.hpp"
#include <string>

namespace benchmark::shuffle {
struct Options {
  std::string graph, meta, output, output_meta, space{"l2"};
  alliance::ShuffleMethod method{alliance::ShuffleMethod::BlockBNF};
  alliance::BNFParams ordering;
  alliance::MultiLevelConfig levels;
  size_t vector_block_size{0};
  double memory_observation_gib{0};
  bool help{false};
  static Options Parse(int argc, char **argv);
};
int Run(const Options &options);
} // namespace benchmark::shuffle
