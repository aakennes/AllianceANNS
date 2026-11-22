// Round-trip unit test for PagedFile + AioSeqStreamReader disk IO primitives.
// Registered as a CTest regression; also runnable as ./bin/test_paged_file.

#include <cstdio>
#include <cstring>
#include <vector>
#include <random>
#include <cassert>
#include "storage/paged_file.hpp"
#include "io/aio_stream.hpp"

using namespace alliance;

int main() {
  const std::string path = "/tmp/alliance_test_paged_file.bin";
  const size_t header_bytes = kDiskPageSize;
  const size_t node_record_bytes = 4524; // e.g. wp=10,M=32 edges (1452B) + d=768 vector (3072B)
  const size_t node_disk_bytes = kDiskPageSize; // one page = one node
  const size_t n = 500;

  printf("node_record_bytes=%zu node_disk_bytes=%zu (kDiskPageSize=%zu)\n", node_record_bytes,
         node_disk_bytes, kDiskPageSize);
  assert(node_record_bytes <= node_disk_bytes);
  assert(node_disk_bytes == 8192);

  // ---- Test 1: PagedFile round trip ----
  {
    PagedFile file(path, /*create=*/true, header_bytes, node_disk_bytes);
    file.Preallocate(n);

    std::mt19937 rng(42);
    std::vector<std::vector<char>> expected(n, std::vector<char>(node_record_bytes));
    for (size_t i = 0; i < n; ++i) {
      for (auto &b : expected[i])
        b = static_cast<char>(rng());
    }

    // Write full node records (padded with zeros beyond node_record_bytes; only
    // the first node_record_bytes bytes are meaningful, so pad a full buffer).
    std::vector<char> padded(node_disk_bytes, 0);
    for (tableint i = 0; i < n; ++i) {
      std::memcpy(padded.data(), expected[i].data(), node_record_bytes);
      file.WriteNodeSlice(i, 0, node_disk_bytes, padded.data());
    }

    // Read back full records, verify byte-exact match on the meaningful prefix.
    std::vector<char> readbuf(node_disk_bytes);
    for (tableint i = 0; i < n; ++i) {
      file.ReadNodeSlice(i, 0, node_disk_bytes, readbuf.data());
      if (std::memcmp(readbuf.data(), expected[i].data(), node_record_bytes) != 0) {
        fprintf(stderr, "FAIL: node %u full-record mismatch\n", i);
        return 1;
      }
    }
    printf("PASS: full node record round-trip (%zu nodes)\n", n);

    // Partial slice read/write: overwrite just one "layer" slice within a node
    // and verify only that slice changed.
    size_t slice_off = 132; // e.g. one layer's edge list offset
    size_t slice_len = 132; // (M_+1)*4 for M=32
    std::vector<char> new_slice(slice_len);
    for (auto &b : new_slice)
      b = static_cast<char>(0x7A);

    tableint target = 17;
    file.WriteNodeSlice(target, slice_off, slice_len, new_slice.data());
    std::vector<char> read_slice(slice_len);
    file.ReadNodeSlice(target, slice_off, slice_len, read_slice.data());
    if (std::memcmp(read_slice.data(), new_slice.data(), slice_len) != 0) {
      fprintf(stderr, "FAIL: node %u slice mismatch after partial write\n", target);
      return 1;
    }
    // Verify bytes outside the slice are untouched.
    file.ReadNodeSlice(target, 0, node_disk_bytes, readbuf.data());
    if (std::memcmp(readbuf.data(), expected[target].data(), slice_off) != 0 ||
        std::memcmp(readbuf.data() + slice_off + slice_len,
                    expected[target].data() + slice_off + slice_len,
                    node_record_bytes - slice_off - slice_len) != 0) {
      fprintf(stderr, "FAIL: node %u bytes outside slice were corrupted\n", target);
      return 1;
    }
    printf("PASS: partial slice read/write leaves surrounding bytes untouched\n");

    // Header round trip.
    struct Header {
      size_t a, b, c;
    };
    Header h{111, 222, 333};
    file.WriteHeader(&h, sizeof(h));
    Header h2{};
    const AioSlice header_slice{&h2, sizeof(h2), 0};
    file.ReadSlices(&header_slice, 1);
    assert(h2.a == 111 && h2.b == 222 && h2.c == 333);
    printf("PASS: header round-trip\n");

    // ---- Test 3: AioSeqStreamReader Skip/Read slice extraction ----
    // This is the shape the RaBitQ passes use to pull one slice out of every
    // record: Skip the bytes before it, Read it, Skip the rest. A Skip that
    // advanced the stream by the wrong amount would shift every later record,
    // so comparing all n slices catches an off-by-anything.
    {
      file.Sync();
      const size_t slice_at = 1024;
      const size_t slice_bytes = 256;
      AioSeqStreamReader reader(file.path(), file.node_offset(0),
                                file.node_offset(static_cast<tableint>(n)));
      std::vector<char> got(slice_bytes);
      for (size_t i = 0; i < n; ++i) {
        reader.Skip(slice_at);
        reader.Read(got.data(), slice_bytes);
        reader.Skip(node_disk_bytes - slice_at - slice_bytes);
        // Node 17 had a slice overwritten above, and its record is otherwise
        // `expected`; slice_at is past that write, so expected still holds.
        if (std::memcmp(got.data(), expected[i].data() + slice_at, slice_bytes) != 0) {
          fprintf(stderr, "FAIL: streamed slice of node %zu mismatch\n", i);
          return 1;
        }
      }
      printf("PASS: AioSeqStreamReader Skip/Read slice sweep over %zu records\n", n);
    }
  }

  std::remove(path.c_str());
  printf("ALL TESTS PASSED\n");
  return 0;
}
