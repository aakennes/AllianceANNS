#pragma once
#include <stdexcept>
#include <string>
namespace benchmark::formal {
inline void Require(bool ok, const char *why) {
  if (!ok)
    throw std::runtime_error(why);
}
} // namespace benchmark::formal
