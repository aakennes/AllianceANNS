#pragma once
#include <cstddef>
#include <cstdint>
#include "quantization/rabitq_code_layout.hpp"
#include "quantization/external_rabitq.hpp"

namespace alliance::detail {
// Actual external RaBitQ runtime dispatch; full unsigned codes, no query SQ.
// Legacy canonical multibit data is retained as an explicit compatibility path.
inline double FloatQueryCodeDot(const uint8_t *code, const float *query, size_t dim, unsigned bits,
                                RaBitQCodeLayout layout = RaBitQCodeLayout::BitPlanes) {
  if (layout != RaBitQCodeLayout::BitPlanes || bits == 1)
    return alliance_rabitq_dot(query, code, dim, bits);
  double dot = 0;
  for (unsigned p = 0; p < bits; ++p)
    dot += static_cast<double>(alliance_rabitq_dot(query, code + p * (dim / 8), dim, 1)) *
           static_cast<double>(1u << p);
  return dot;
}
} // namespace alliance::detail
