#pragma once
#include "shuffle/shuffle.hpp"
namespace alliance::shuffle_detail {
template <typename att_t, typename vec_t>
auto ShuffleBlockBNF(AllianceIndexDisk<att_t, vec_t> *idx, const BNFParams &params, layer_t first,
                     layer_t last, size_t record_layers, size_t stride, bool collect_diagnostics)
    -> std::vector<tableint>;
} // namespace alliance::shuffle_detail
