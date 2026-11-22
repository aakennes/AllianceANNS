#include "quantization/packed_rabitq_storage.hpp"
#include <iostream>
#include <sstream>
void Check(bool x) {
  if (!x)
    throw std::runtime_error("packed storage contract");
}
int main() {
  for (size_t dim : {64, 128, 192, 384, 768})
    for (size_t bits : {1, 2, 3, 4, 8})
      for (bool native : {false, true}) {
        if (native && bits == 1)
          continue;
        const size_t n = 8193, bytes = dim * bits / 8;
        std::vector<uint8_t> codes(n * bytes);
        std::vector<float> factors(n * 4);
        for (size_t i = 0; i < codes.size(); ++i)
          codes[i] = (i * 71 + 13) & 255;
        for (size_t i = 0; i < factors.size(); ++i)
          factors[i] = float(int(i % 997) - 498) * .03125f;
        alliance::PackedRaBitQStorage p;
        p.Assign(codes, factors, n, bytes,
                 native ? alliance::detail::NativeCodeLayout(bits)
                        : alliance::detail::RaBitQCodeLayout::BitPlanes);
        Check(reinterpret_cast<uintptr_t>(p.Code(0)) % 64 == 0);
        Check(p.Bytes() == p.Stride() * n);
        std::vector<uint8_t> canonical(bytes), raw(dim), expected(bytes);
        for (size_t i = 0; i < n; ++i) {
          p.CopyCanonicalCode(i, canonical.data());
          Check(!std::memcmp(canonical.data(), codes.data() + i * bytes, bytes));
          Check(!std::memcmp(p.Factors(i), factors.data() + i * 4, 16));
          if (native) {
            for (size_t d = 0; d < dim; ++d) {
              raw[d] = 0;
              for (size_t b = 0; b < bits; ++b)
                raw[d] |= ((codes[i * bytes + b * dim / 8 + d / 8] >> (d % 8)) & 1) << b;
            }
            alliance_rabitq_pack(raw.data(), expected.data(), dim, bits);
            Check(!std::memcmp(p.Code(i), expected.data(), bytes));
          }
        }
        std::ostringstream out(std::ios::binary);
        p.WriteCanonical(out);
        auto s = out.str();
        Check(s.size() == n * (bytes + 16));
        Check(!std::memcmp(s.data(), codes.data(), codes.size()));
        Check(!std::memcmp(s.data() + codes.size(), factors.data(), factors.size() * 4));
        bool threw = false;
        try {
          p.Assign(codes, factors, n + 1, bytes);
        } catch (const std::invalid_argument &) {
          threw = true;
        }
        Check(threw);
        Check(p.Bytes() > 0);
        p.Clear();
        Check(p.Empty() && p.Bytes() == 0);
      }
  std::cout
      << "packed storage: dimensions64/128/192/384/768, bits1/2/3/4/8, compact2/3/4/8,8193points, canonical bytes and batch boundaries passed\n";
}
