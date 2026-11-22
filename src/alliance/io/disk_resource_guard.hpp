#pragma once
// Internal scope ownership for index construction and build-stage resources.
// Callbacks must be non-throwing and must not outlive the resources they borrow.
#include <cstdio>
#include <memory>
#include <type_traits>
#include <utility>

namespace alliance::detail {
template <typename Cleanup>
class ScopeExit {
public:
  explicit ScopeExit(Cleanup cleanup) noexcept : cleanup_(std::move(cleanup)) {
    static_assert(std::is_nothrow_invocable_v<Cleanup>);
    static_assert(std::is_nothrow_move_constructible_v<Cleanup>);
  }
  ~ScopeExit() noexcept { if (active_) cleanup_(); }
  ScopeExit(const ScopeExit &) = delete;
  ScopeExit &operator=(const ScopeExit &) = delete;
  void Release() noexcept { active_ = false; }
private:
  Cleanup cleanup_;
  bool active_{true};
};

struct CloseFile {
  void operator()(FILE *file) const noexcept { if (file) std::fclose(file); }
};
using OwnedFile = std::unique_ptr<FILE, CloseFile>;
}  // namespace alliance::detail
