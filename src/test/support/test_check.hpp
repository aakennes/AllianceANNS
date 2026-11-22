#pragma once

#include <iostream>
#include <stdexcept>
#include <string>

namespace alliance::test {
// Test checks must remain active when production code is compiled with NDEBUG.
// A failed check is distinct from expected runtime_error failures in IO tests.
class Failure : public std::logic_error {
 public:
  using std::logic_error::logic_error;
};

inline void Require(bool passed, const char *expression, const char *file, int line) {
  if (!passed)
    throw Failure(std::string(file) + ":" + std::to_string(line) +
                  ": check failed: " + expression);
}

template <class Tests>
int RunSuite(const char *name, Tests tests) {
  try {
    tests();
    std::cout << name << " passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << name << ": " << error.what() << '\n';
    return 1;
  }
}
}  // namespace alliance::test

#define ALLIANCE_TEST_CHECK(...) \
  ::alliance::test::Require(bool((__VA_ARGS__)), #__VA_ARGS__, __FILE__, __LINE__)
