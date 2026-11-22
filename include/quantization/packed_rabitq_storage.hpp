#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <vector>
#include "quantization/rabitq_code_layout.hpp"
namespace alliance {
// RAM layout only. Canonical metadata remains all codes followed by all factors.
class PackedRaBitQStorage {
  struct Free {
    void operator()(uint8_t *p) const noexcept {
      std::free(p);
    }
  };
  std::unique_ptr<uint8_t, Free> data_;
  size_t count_{}, code_bytes_{}, stride_{};
  detail::RaBitQCodeLayout layout_{detail::RaBitQCodeLayout::BitPlanes};

public:
  detail::RaBitQCodeLayout Layout() const noexcept {
    return layout_;
  }
  bool Empty() const noexcept {
    return !data_;
  }
  size_t Bytes() const noexcept {
    return count_ * stride_;
  }
  size_t Stride() const noexcept {
    return stride_;
  }
  const uint8_t *Code(size_t id) const noexcept {
    return data_.get() + id * stride_;
  }
  const float *Factors(size_t id) const noexcept {
    return reinterpret_cast<const float *>(Code(id) + code_bytes_);
  }
  void Clear() noexcept {
    data_.reset();
    count_ = code_bytes_ = stride_ = 0;
    layout_ = detail::RaBitQCodeLayout::BitPlanes;
  }
  void Assign(const std::vector<uint8_t> &codes, const std::vector<float> &factors, size_t n,
              size_t code_bytes,
              detail::RaBitQCodeLayout layout = detail::RaBitQCodeLayout::BitPlanes) {
    if (!n || !code_bytes || code_bytes % alignof(float) ||
        code_bytes > std::numeric_limits<size_t>::max() - 31)
      throw std::invalid_argument("invalid packed RaBitQ shape");
    if (layout != detail::RaBitQCodeLayout::BitPlanes &&
        (detail::NativeCodeLayout(static_cast<unsigned>(layout)) != layout ||
         code_bytes % (8 * static_cast<unsigned>(layout))))
      throw std::invalid_argument(
          "native compact codes require supported bits and 64-coordinate padding");
    const size_t stride = (code_bytes + 16 + 15) / 16 * 16;
    if (n > std::numeric_limits<size_t>::max() / stride ||
        n > std::numeric_limits<size_t>::max() / 4 || codes.size() != n * code_bytes ||
        factors.size() != n * 4)
      throw std::invalid_argument("packed RaBitQ size mismatch");
    void *p = nullptr;
    if (posix_memalign(&p, 64, n * stride))
      throw std::bad_alloc();
    std::unique_ptr<uint8_t, Free> next(static_cast<uint8_t *>(p));
    for (size_t i = 0; i < n; ++i) {
      if (layout != detail::RaBitQCodeLayout::BitPlanes)
        detail::PackCompact(codes.data() + i * code_bytes, next.get() + i * stride, code_bytes,
                            layout);
      else
        std::memcpy(next.get() + i * stride, codes.data() + i * code_bytes, code_bytes);
      std::memcpy(next.get() + i * stride + code_bytes, factors.data() + i * 4, 16);
    }
    data_ = std::move(next);
    count_ = n;
    code_bytes_ = code_bytes;
    stride_ = stride;
    layout_ = layout;
  }
  void CopyCanonicalCode(size_t id, uint8_t *out) const {
    if (layout_ != detail::RaBitQCodeLayout::BitPlanes)
      detail::UnpackCompact(Code(id), out, code_bytes_, layout_);
    else
      std::memcpy(out, Code(id), code_bytes_);
  }
  void WriteCanonical(std::ostream &out) const {
    const size_t batch = 4096;
    std::vector<uint8_t> buf(batch * std::max(code_bytes_, size_t(16)));
    for (size_t first = 0; first < count_; first += batch) {
      size_t n = std::min(batch, count_ - first);
      for (size_t i = 0; i < n; ++i)
        CopyCanonicalCode(first + i, buf.data() + i * code_bytes_);
      out.write(reinterpret_cast<const char *>(buf.data()), n * code_bytes_);
    }
    for (size_t first = 0; first < count_; first += batch) {
      size_t n = std::min(batch, count_ - first);
      for (size_t i = 0; i < n; ++i)
        std::memcpy(buf.data() + i * 16, Factors(first + i), 16);
      out.write(reinterpret_cast<const char *>(buf.data()), n * 16);
    }
    if (!out)
      throw std::runtime_error("packed RaBitQ serialization failed");
  }
};
} // namespace alliance
