#include "quantization/external_rabitq.hpp"
#include <array>
#include <stdexcept>
#include <rabitqlib/utils/space.hpp>
#include <rabitqlib/core/pack_excode.hpp>

extern "C" float alliance_rabitq_dot(const float *query, const uint8_t *code, size_t dim,
                                     unsigned bits) {
  static const auto functions = [] {
    std::array<rabitqlib::ex_ipfunc, 9> result{};
    for (unsigned b = 1; b <= 8; ++b)
      result[b] = rabitqlib::select_excode_ipfunc(b);
    return result;
  }();
  if (bits < 1 || bits > 8)
    throw std::invalid_argument("RaBitQ width must be 1..8");
  return functions[bits](query, code, dim);
}
extern "C" void alliance_rabitq_pack(const uint8_t *raw, uint8_t *packed, size_t dim,
                                     unsigned bits) {
  rabitqlib::quant::rabitq_impl::ex_bits::packing_rabitqplus_code(raw, packed, dim, bits);
}
extern "C" void alliance_rabitq_unpack(const uint8_t *packed, uint8_t *raw, size_t dim,
                                       unsigned bits) {
  rabitqlib::quant::rabitq_impl::ex_bits::unpacking_rabitqplus_code(packed, raw, dim, bits);
}
extern "C" const char *alliance_rabitq_version() {
  return "VectorDB-NTU/RaBitQ-Library v0.7.1 e8fdff963509d527cbe870bb683776b6c03601fd";
}
