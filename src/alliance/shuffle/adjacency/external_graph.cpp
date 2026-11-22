#include "shuffle/adjacency/external_graph.hpp"
#include "shuffle/adjacency/external_sort.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>

namespace alliance::shuffle_detail {
ExternalGraph::ExternalGraph(const std::filesystem::path &parent, size_t nodes, uint8_t maximum,
                             size_t edges, ExternalBudget budget)
    : directory_(parent), codec_(nodes, maximum), budget_(budget), nodes_(nodes) {
  raw_ = std::make_unique<ScratchFile>(directory_.Path() / "raw-csr", io_);
  raw_->Resize(uint64_t(edges) * codec_.Bytes());
}
ExternalGraph::~ExternalGraph() {
  std::fprintf(
      stderr,
      "  [shuffle-external-io] read_calls=%llu read_bytes=%llu write_calls=%llu write_bytes=%llu\n",
      (unsigned long long)io_.read_calls.load(), (unsigned long long)io_.read_bytes.load(),
      (unsigned long long)io_.write_calls.load(), (unsigned long long)io_.write_bytes.load());
  std::fprintf(
      stderr,
      "  [shuffle-external-aio] backend=libaio direct=1 reads=%llu writes=%llu read_bytes=%llu write_bytes=%llu completed=%llu completed_bytes=%llu submits=%llu waits=%llu max_inflight=%llu errors=%llu\n",
      (unsigned long long)io_.aio_reads.load(), (unsigned long long)io_.aio_writes.load(),
      (unsigned long long)io_.aio_read_bytes.load(), (unsigned long long)io_.aio_write_bytes.load(),
      (unsigned long long)io_.aio_completed.load(),
      (unsigned long long)io_.aio_completed_bytes.load(),
      (unsigned long long)io_.submit_calls.load(), (unsigned long long)io_.wait_calls.load(),
      (unsigned long long)io_.max_inflight.load(), (unsigned long long)io_.errors.load());
}
void ExternalGraph::Write(size_t offset, const void *data, size_t bytes) {
  raw_->Write(uint64_t(offset) * codec_.Bytes(), data, bytes);
}
void ExternalGraph::Prepare(const std::vector<size_t> &order, const std::vector<size_t> &offsets) {
  const auto start = std::chrono::steady_clock::now();
  if (order.size() != nodes_ || offsets.size() != nodes_ + 1 || nodes_ > UINT32_MAX)
    throw std::invalid_argument("external adjacency order geometry");
  raw_->Seal();
  std::vector<tableint> rank(nodes_, UINT32_MAX);
  for (size_t r = 0; r < nodes_; ++r) {
    if (order[r] >= nodes_ || rank[order[r]] != UINT32_MAX)
      throw std::invalid_argument("external adjacency requires a permutation");
    rank[order[r]] = static_cast<tableint>(r);
    max_degree_ = std::max(max_degree_, offsets[order[r] + 1] - offsets[order[r]]);
  }
  if (max_degree_ > budget_.batch_bytes / codec_.Bytes())
    throw std::invalid_argument("outgoing row exceeds external batch budget");
  outgoing_ = std::make_unique<ScratchFile>(directory_.Path() / "outgoing", io_);
  incoming_ = std::make_unique<ScratchFile>(directory_.Path() / "incoming", io_);
  outgoing_offsets_.assign(nodes_ + 1, 0);
  incoming_offsets_.assign(nodes_ + 1, 0);
  filters_.resize(nodes_);
  IncomingSorter sorter(directory_.Path(), io_, budget_.sort_bytes, budget_.fan_in);
  ScratchWriter writer(*outgoing_);
  // Batch independent CSR row gathers into real O_DIRECT/libaio requests.
  // Processing and key emission still follow the original fixed node order.
  const size_t row_stride = std::max<size_t>(1, max_degree_) * codec_.Bytes();
  const size_t gather_nodes = std::min<size_t>(64, std::max<size_t>(1, (1 << 20) / row_stride));
  std::vector<uint8_t> rows(gather_nodes * row_stride);
  std::vector<ScratchRead> reads;
  for (size_t first = 0; first < nodes_; first += gather_nodes) {
    const size_t count = std::min(gather_nodes, nodes_ - first);
    reads.clear();
    for (size_t i = 0; i < count; ++i) {
      const size_t u = order[first + i];
      reads.push_back({offsets[u] * codec_.Bytes(), rows.data() + i * row_stride,
                       (offsets[u + 1] - offsets[u]) * codec_.Bytes()});
    }
    raw_->ReadMany(reads);
    for (size_t i = 0; i < count; ++i) {
      const size_t r = first + i;
      const auto *row = rows.data() + i * row_stride;
      const auto u = static_cast<tableint>(order[r]);
      const size_t degree = offsets[u + 1] - offsets[u];
      writer.Append(row, degree * codec_.Bytes());
      outgoing_offsets_[r + 1] = outgoing_offsets_[r] + degree;
      size_t score = 0;
      for (size_t j = 0; j < degree; ++j) {
        const auto *edge = row + j * codec_.Bytes();
        const tableint v = codec_.Id(edge);
        const size_t weight = codec_.Weight(edge);
        filters_[u].Add(v);
        score += weight * weight;
        // v in incoming(u) is exactly the old u in outgoing(v) tie-break check.
        sorter.Add((uint64_t(rank[v]) << 32) | u);
      }
      max_score_ = std::max(max_score_, score);
    }
  }
  writer.Finish();
  std::vector<tableint>().swap(rank);
  std::vector<uint8_t>().swap(rows);
  ScratchWriter incoming_writer(*incoming_);
  size_t next_rank = 0, count = 0;
  sorter.Finish([&](uint64_t key) {
    const size_t r = key >> 32;
    while (next_rank <= r)
      incoming_offsets_[next_rank++] = count;
    const tableint id = static_cast<tableint>(key);
    incoming_writer.Append(&id, sizeof(id));
    ++count;
  });
  while (next_rank <= nodes_)
    incoming_offsets_[next_rank++] = count;
  incoming_writer.Finish();
  if (count != offsets.back() || outgoing_offsets_.back() != count)
    throw std::runtime_error("external adjacency transpose edge count mismatch");
  raw_.reset(); // Both streams are complete before the original scratch CSR is removed.
  std::fprintf(
      stderr,
      "  [shuffle-external-prepare] seconds=%.6f edges=%zu edge_bytes=%zu max_degree=%zu max_score=%zu stream_bytes=%llu\n",
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(), count,
      codec_.Bytes(), max_degree_, max_score_,
      (unsigned long long)(outgoing_->Size() + incoming_->Size()));
}
size_t ExternalGraph::LoadBatch(size_t rank, size_t requested) {
  if (rank >= nodes_ || !requested)
    throw std::invalid_argument("external batch range");
  size_t count = std::min(requested, nodes_ - rank);
  auto fitting = [&](const std::vector<size_t> &offsets, size_t bytes) {
    const size_t limit = offsets[rank] + budget_.batch_bytes / bytes;
    const auto end =
        std::upper_bound(offsets.begin() + rank + 1, offsets.begin() + rank + count + 1, limit);
    return size_t(end - offsets.begin()) - rank - 1;
  };
  count = std::min(count, fitting(outgoing_offsets_, codec_.Bytes()));
  if (!count)
    throw std::runtime_error("outgoing row exceeds external batch budget");
  const size_t incoming_fit = fitting(incoming_offsets_, sizeof(tableint));
  large_incoming_row_ = incoming_fit == 0;
  count = large_incoming_row_ ? 1 : std::min(count, incoming_fit);
  batch_rank_ = rank;
  batch_count_ = count;
  const size_t begin = outgoing_offsets_[rank], end = outgoing_offsets_[rank + count];
  outgoing_batch_.resize((end - begin) * codec_.Bytes());
  outgoing_->Read(begin * codec_.Bytes(), outgoing_batch_.data(), outgoing_batch_.size());
  if (large_incoming_row_) {
    // A hub can have unbounded incoming degree. Keep memory bounded and use
    // exact file binary search for this single row rather than allocating it.
    incoming_batch_.clear();
  } else {
    const size_t first = incoming_offsets_[rank], last = incoming_offsets_[rank + count];
    incoming_batch_.resize(last - first);
    incoming_->Read(first * sizeof(tableint), incoming_batch_.data(),
                    incoming_batch_.size() * sizeof(tableint));
  }
  return count;
}
EncodedRow ExternalGraph::Row(size_t rank) const {
  if (rank < batch_rank_ || rank >= batch_rank_ + batch_count_)
    throw std::logic_error("external row outside active batch");
  const size_t size = outgoing_offsets_[rank + 1] - outgoing_offsets_[rank];
  const size_t begin = (outgoing_offsets_[rank] - outgoing_offsets_[batch_rank_]) * codec_.Bytes();
  return {size ? outgoing_batch_.data() + begin : nullptr, size, &codec_};
}
bool ExternalGraph::ContainsIncoming(size_t rank, tableint source) const {
  if (rank < batch_rank_ || rank >= batch_rank_ + batch_count_)
    throw std::logic_error("external incoming row outside active batch");
  size_t begin = incoming_offsets_[rank], end = incoming_offsets_[rank + 1];
  if (!large_incoming_row_) {
    const size_t offset = incoming_offsets_[batch_rank_];
    return std::binary_search(incoming_batch_.begin() + begin - offset,
                              incoming_batch_.begin() + end - offset, source);
  }
  const size_t limit = end;
  while (begin < end) {
    const size_t mid = begin + (end - begin) / 2;
    tableint id;
    incoming_->Read(mid * sizeof(id), &id, sizeof(id));
    if (id < source)
      begin = mid + 1;
    else
      end = mid;
  }
  if (begin == limit)
    return false;
  tableint id;
  incoming_->Read(begin * sizeof(id), &id, sizeof(id));
  return id == source;
}
bool ExternalGraph::MayContain(tableint row, const NeighborMembershipFilter::Mask &mask) const {
  return filters_[row].MayContain(mask);
}
} // namespace alliance::shuffle_detail
