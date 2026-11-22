#pragma once
#include "shuffle/adjacency/edge_codec.hpp"
#include "shuffle/adjacency/neighbor_filter.hpp"
#include "shuffle/adjacency/scratch_io.hpp"
#include "shuffle/memory/external_budget.hpp"

namespace alliance::shuffle_detail {
// A group-scoped adjacency backend. Construction writes CSR once; Prepare()
// streams outgoing rows and transposed incoming rows in fixed processing order.
// Round() then reads both sequentially, with no random reciprocal-row lookup.
class ExternalGraph {
public:
  ExternalGraph(const std::filesystem::path &parent, size_t nodes, uint8_t maximum, size_t edges,
                ExternalBudget budget);
  ~ExternalGraph();
  const EdgeCodec &Codec() const {
    return codec_;
  }
  void Write(size_t edge_offset, const void *data, size_t bytes);
  void Prepare(const std::vector<size_t> &order, const std::vector<size_t> &offsets);
  size_t LoadBatch(size_t rank, size_t requested);
  EncodedRow Row(size_t rank) const;
  bool ContainsIncoming(size_t rank, tableint source) const;
  bool MayContain(tableint row, const NeighborMembershipFilter::Mask &mask) const;
  size_t MaxDegree() const {
    return max_degree_;
  }
  size_t MaxScore() const {
    return max_score_;
  }
  const ScratchIo &Io() const {
    return io_;
  }

private:
  ScratchIo io_;
  ScratchDirectory directory_;
  EdgeCodec codec_;
  ExternalBudget budget_;
  size_t nodes_, max_degree_{}, max_score_{};
  std::unique_ptr<ScratchFile> raw_, outgoing_, incoming_;
  std::vector<size_t> outgoing_offsets_, incoming_offsets_;
  std::vector<NeighborMembershipFilter> filters_;
  std::vector<uint8_t> outgoing_batch_;
  std::vector<tableint> incoming_batch_;
  size_t batch_rank_{}, batch_count_{};
  bool large_incoming_row_{false};
};
} // namespace alliance::shuffle_detail
