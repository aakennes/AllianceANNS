#pragma once
// alliance/index/window_ladder.hpp -- the rank-window ladder shared by
// AllianceIndex (in-memory) and AllianceIndexDisk.
//
// A layer l is defined by one number: window_size[l], the rank span a node may
// draw neighbours from at that layer. Layer 0 is the enumeration layer and the
// top layer wp is the one a query with no attribute filter lands on. Everything
// downstream -- Phase 1's halo, Phase 1.5's activation points, Phase 2's
// half_window, DecideLayerRange's filter-cardinality-to-layer mapping -- reads
// window_size and nothing else, so this file is the single place the ladder's
// shape is decided.
//
// Two anchors:
//
//   kBottom (historical, still the default)
//     window_size[0] = enum_threshold, window_size[l] = o * window_size[l-1].
//     The ladder is pinned at the bottom and the top falls wherever o^wp lands.
//     It usually does not land on n: at arxiv-2M (n=2138591, o=4, wp=5) the top
//     window is 2097152 < n, so the top layer never covers the full set. The
//     WoW activation test (window_size[l]-1 < batch_end) can then never fire
//     for it and the build has to patch the layer in afterwards ("Global-layer
//     fallback"), while a query filtering on ~n candidates gets routed to a
//     layer whose window is narrower than its own filter.
//
//   kTop
//     window_size[wp] = n exactly, and the ladder is derived downward from
//     there: base = ceil(n / o^wp), window_size[l] = base * o^l for l < wp.
//     Same o-fold ratio between adjacent layers, but the *top* is pinned and
//     the enumeration window becomes a derived quantity rather than the input.
//     At arxiv-2M: 2089 / 8356 / 33424 / 133696 / 534784 / 2138591.
//
// enum_threshold keeps its second job under either anchor -- it is also the WoW
// batch size (PlanPhase15's batch_size) -- but under kTop it stops *setting*
// window_size[0]. It becomes a target for it instead:
//
//   * with auto_raise_wp (i.e. no explicit --wp), kTop reverse-solves wp as the
//     largest one whose derived base ceil(n/o^wp) is still >= enum_threshold.
//     Taking the largest such wp -- rather than the smallest wp whose base drops
//     below it -- is what keeps base >= enum_threshold, so the derived ladder is
//     never taller than the bottom-anchored one for the same inputs. At
//     arxiv-2M (n=2138591, o=4, enum_threshold=2048) this lands on wp=5, the
//     same ladder an explicit --wp 5 produces. Without this, wp=0 would be taken
//     literally and the whole graph would collapse into one window of n.
//   * with an explicit wp, a derived base below max(M+1, 64) is clamped up and
//     the caller warned -- a node cannot pick M neighbours out of fewer than
//     M+1 candidates.
//
// The anchor is NOT a search-time knob: the ladder it produces is persisted
// (AllianceIndexDisk::Save writes window_size verbatim) and determines layer
// routing at query time, so a graph built one way must be searched the same
// way. The disk meta records the anchor alongside the ladder purely so a
// resumed build can refuse to switch mid-graph.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace alliance {

enum class WindowAnchor : uint8_t { kBottom = 0, kTop = 1 };

inline auto WindowAnchorName(WindowAnchor a) -> const char * {
  return a == WindowAnchor::kTop ? "top" : "bottom";
}

inline auto ParseWindowAnchor(const std::string &s) -> WindowAnchor {
  if (s == "top")
    return WindowAnchor::kTop;
  if (s == "bottom")
    return WindowAnchor::kBottom;
  throw std::runtime_error("unknown window anchor '" + s + "' (expected top or bottom)");
}

struct WindowLadder {
  std::vector<size_t> window_size;
  size_t wp{0};             // possibly raised (kBottom + auto_raise_wp only)
  size_t base{0};           // window_size[0] as decided by the anchor
  bool wp_raised{false};    // auto_raise_wp fired
  bool wp_generous{false};  // kBottom: the ladder covered n in fewer than wp+1 layers
  bool base_clamped{false}; // kTop: derived base was below the M-neighbour floor
};

/// Builds the ladder over `n` points, `out.wp`+1 layers tall. `M` is only
/// consulted for kTop's base floor. With `auto_raise_wp` the incoming `wp` is
/// merely a starting point -- both anchors derive their own -- so callers must
/// adopt `out.wp`, which sizes their per-node record layout.
inline auto BuildWindowLadder(size_t n, size_t o, size_t wp, size_t enum_threshold, size_t M,
                              WindowAnchor anchor, bool auto_raise_wp) -> WindowLadder {
  WindowLadder out;
  out.wp = wp;

  if (anchor == WindowAnchor::kBottom) {
    // Verbatim reproduction of the growth loop both constructors used to carry
    // inline, including the padding branch's habit of emitting wp+2 entries
    // when the geometric ladder already covers n within wp+1. That trailing
    // entry is always > n and therefore unreachable by DecideLayerRange's
    // lower_bound, but it is in every meta file ever written, so it stays.
    out.window_size.emplace_back(enum_threshold);
    while (out.window_size.back() < n) {
      out.window_size.emplace_back(o * out.window_size.back());
    }
    if (wp + 1 < out.window_size.size()) {
      if (auto_raise_wp) {
        out.wp = out.window_size.size() - 1;
        out.wp_raised = true;
      } else {
        out.window_size.resize(wp + 1);
      }
    } else {
      out.wp_generous = (wp + 1 != out.window_size.size());
      for (size_t i = out.window_size.size(); i <= wp + 1; ++i) {
        out.window_size.emplace_back(o * out.window_size.back());
      }
    }
    out.base = out.window_size.front();
    return out;
  }

  // ===== kTop =====
  if (o < 2) {
    throw std::runtime_error("BuildWindowLadder: top anchor needs o >= 2 (got " +
                             std::to_string(o) + ")");
  }
  if (n == 0) {
    throw std::runtime_error("BuildWindowLadder: top anchor needs n > 0");
  }
  // A layer-0 window below M+1 cannot even fill one edge list, and below 64 the
  // enumeration batch stops amortizing anything.
  const size_t base_floor = std::max(M + 1, (size_t)64);

  size_t span; // o^out.wp
  if (auto_raise_wp) {
    // Reverse-solve wp from the target bottom window. Climbing while the *next*
    // layer's base still clears the target means we stop at the largest wp with
    // base >= target, so the enumeration window never ends up narrower than the
    // caller asked for and base_clamped cannot fire. Terminates in <= log_o(n)
    // steps: span grows o-fold each turn and target >= 64 > 1.
    const size_t target = std::max(enum_threshold, base_floor);
    out.wp = 0;
    span = 1;
    while (span <= SIZE_MAX / o) {
      const size_t next_span = span * o;
      const size_t next_base = (next_span >= n) ? 1 : (n + next_span - 1) / next_span;
      if (next_base < target)
        break;
      span = next_span;
      ++out.wp;
    }
    out.wp_raised = (out.wp != wp);
  } else {
    // o^wp, saturating: once it reaches n the derived base is 1 regardless, so
    // there is no reason to keep multiplying into an overflow.
    span = 1;
    for (size_t i = 0; i < wp && span < n; ++i) {
      if (span > SIZE_MAX / o) {
        span = SIZE_MAX;
        break;
      }
      span *= o;
    }
  }
  size_t base = (span >= n) ? 1 : (n + span - 1) / span;

  // Clamping keeps an explicitly-over-tall ladder usable at the cost of the top
  // few layers colliding at n, which is harmless (monotone non-decreasing is all
  // lower_bound and the activation test need) but worth a warning from the
  // caller. On the auto_raise_wp path it can only fire for n < max(M+1, 64),
  // where wp lands on 0 and base on n -- a dataset too small to have layers.
  if (base < base_floor) {
    base = std::min(base_floor, n);
    out.base_clamped = true;
  }
  out.base = base;

  out.window_size.resize(out.wp + 1);
  size_t w = base;
  for (size_t l = 0; l < out.wp; ++l) {
    out.window_size[l] = std::min(w, n);
    w = (w > SIZE_MAX / o) ? SIZE_MAX : w * o;
  }
  out.window_size[out.wp] = n; // the anchor
  return out;
}

} // namespace alliance
