#pragma once
#include <cstddef>
#include <cstdint>
// Narrow ABI: no Eigen, rotator or quantizer object crosses library versions.
#if defined(_WIN32)
#define ALLIANCE_RABITQ_EXPORT __declspec(dllexport)
#else
#define ALLIANCE_RABITQ_EXPORT __attribute__((visibility("default")))
#endif
extern "C" {
ALLIANCE_RABITQ_EXPORT float alliance_rabitq_dot(const float *query, const uint8_t *code,
                                                 size_t dim, unsigned bits);
ALLIANCE_RABITQ_EXPORT void alliance_rabitq_pack(const uint8_t *raw, uint8_t *packed, size_t dim,
                                                 unsigned bits);
ALLIANCE_RABITQ_EXPORT void alliance_rabitq_unpack(const uint8_t *packed, uint8_t *raw, size_t dim,
                                                   unsigned bits);
ALLIANCE_RABITQ_EXPORT const char *alliance_rabitq_version();
}
