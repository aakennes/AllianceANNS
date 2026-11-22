#pragma once
#include "search/search_config.hpp"

namespace benchmark {
struct SearchOptionState;
// Compatibility facade retained for CLI users; runtime state is pure values.
struct DiskSearchOptions : DiskSearchConfig {
  bool ParseAndValidate(int argc, char **argv);

private:
  bool NormalizeAndValidate(const SearchOptionState &state);
};
} // namespace benchmark
