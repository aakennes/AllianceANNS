// bench_iops.cpp — Standalone IOPS benchmark using libaio + O_DIRECT.
// Tests random read performance at various block sizes, thread counts, and IO depths.
// Usage: ./bench_iops [--file /path/to/testfile] [--size-gb 4] [--runtime 10]

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <libaio.h>
#include <sys/stat.h>
#include <unistd.h>

static constexpr size_t kPageSize = 4096;

struct BenchConfig {
  std::string file_path = "/tmp/bench_iops_test";
  size_t file_size_gb = 4;
  int runtime_sec = 10;
};

struct BenchResult {
  double iops;
  double bw_mbs;
  double lat_avg_us;
};

static void CreateTestFile(const std::string &path, size_t size_bytes) {
  int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT, 0644);
  if (fd < 0) {
    fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
      fprintf(stderr, "Failed to create test file: %s\n", path.c_str());
      exit(1);
    }
  }

  constexpr size_t kChunkSize = 1 << 20; // 1MB
  char *buf = static_cast<char *>(aligned_alloc(kPageSize, kChunkSize));
  memset(buf, 0xAB, kChunkSize);

  size_t written = 0;
  while (written < size_bytes) {
    size_t to_write = std::min(kChunkSize, size_bytes - written);
    ssize_t ret = write(fd, buf, to_write);
    if (ret <= 0) break;
    written += ret;
  }
  fsync(fd);
  close(fd);
  free(buf);
  printf("  Created test file: %s (%.1f GB)\n", path.c_str(), (double)written / (1 << 30));
}

static BenchResult RunBench(const std::string &path, size_t file_size,
                            size_t block_size, int num_threads, int iodepth,
                            int runtime_sec) {
  std::atomic<uint64_t> total_ios{0};
  std::atomic<uint64_t> total_lat_ns{0};
  std::atomic<bool> running{true};

  size_t max_offset = (file_size / block_size) * block_size;
  if (max_offset < block_size) max_offset = block_size;

  auto worker = [&](int tid) {
    int fd = open(path.c_str(), O_RDONLY | O_DIRECT);
    if (fd < 0) {
      fd = open(path.c_str(), O_RDONLY);
      if (fd < 0) return;
    }

    io_context_t ctx{};
    if (io_setup(iodepth, &ctx) < 0) {
      close(fd);
      return;
    }

    std::vector<char *> bufs(iodepth);
    for (int i = 0; i < iodepth; ++i) {
      bufs[i] = static_cast<char *>(aligned_alloc(kPageSize, block_size));
    }

    std::mt19937_64 rng(tid * 12345 + 67890);
    size_t num_blocks = max_offset / block_size;

    // Submit initial batch
    std::vector<struct iocb> cbs(iodepth);
    std::vector<struct iocb *> cb_ptrs(iodepth);
    std::vector<uint64_t> submit_ts(iodepth);

    for (int i = 0; i < iodepth; ++i) {
      uint64_t off = (rng() % num_blocks) * block_size;
      io_prep_pread(&cbs[i], fd, bufs[i], block_size, off);
      cbs[i].data = reinterpret_cast<void *>((intptr_t)i);
      cb_ptrs[i] = &cbs[i];
      submit_ts[i] = std::chrono::steady_clock::now().time_since_epoch().count();
    }
    io_submit(ctx, iodepth, cb_ptrs.data());

    uint64_t local_ios = 0;
    uint64_t local_lat_ns = 0;
    struct io_event events[64];

    int in_flight = iodepth;

    while (running.load(std::memory_order_relaxed)) {
      int ret = io_getevents(ctx, 1, std::min(iodepth, 64), events, nullptr);
      if (ret <= 0) continue;

      auto now = std::chrono::steady_clock::now().time_since_epoch().count();

      for (int i = 0; i < ret; ++i) {
        int slot = (int)(intptr_t)events[i].data;
        local_lat_ns += (now - submit_ts[slot]);
        local_ios++;

        // Resubmit only if still running
        if (running.load(std::memory_order_relaxed)) {
          uint64_t off = (rng() % num_blocks) * block_size;
          io_prep_pread(&cbs[slot], fd, bufs[slot], block_size, off);
          cbs[slot].data = reinterpret_cast<void *>((intptr_t)slot);
          struct iocb *p = &cbs[slot];
          submit_ts[slot] = now;
          io_submit(ctx, 1, &p);
        } else {
          in_flight--;
        }
      }
    }

    // Drain all remaining in-flight IOs (blocking)
    while (in_flight > 0) {
      int ret = io_getevents(ctx, 1, std::min(in_flight, 64), events, nullptr);
      if (ret <= 0) break;
      in_flight -= ret;
      local_ios += ret;
    }

    total_ios.fetch_add(local_ios, std::memory_order_relaxed);
    total_lat_ns.fetch_add(local_lat_ns, std::memory_order_relaxed);

    io_destroy(ctx);
    for (int i = 0; i < iodepth; ++i) free(bufs[i]);
    close(fd);
  };

  std::vector<std::thread> threads;
  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back(worker, t);
  }

  std::this_thread::sleep_for(std::chrono::seconds(runtime_sec));
  running.store(false, std::memory_order_relaxed);

  for (auto &t : threads) t.join();

  uint64_t ios = total_ios.load();
  uint64_t lat = total_lat_ns.load();

  BenchResult r;
  r.iops = (double)ios / runtime_sec;
  r.bw_mbs = r.iops * block_size / (1024.0 * 1024.0);
  r.lat_avg_us = ios > 0 ? (double)lat / ios / 1000.0 : 0;
  return r;
}

int main(int argc, char **argv) {
  BenchConfig cfg;

  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--file") == 0 && i + 1 < argc) {
      cfg.file_path = argv[++i];
    } else if (strcmp(argv[i], "--size-gb") == 0 && i + 1 < argc) {
      cfg.file_size_gb = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--runtime") == 0 && i + 1 < argc) {
      cfg.runtime_sec = atoi(argv[++i]);
    } else {
      fprintf(stderr, "Usage: %s [--file path] [--size-gb N] [--runtime sec]\n", argv[0]);
      return 1;
    }
  }

  size_t file_size = cfg.file_size_gb * (size_t)(1 << 30);

  printf("==========================================\n");
  printf(" IOPS Benchmark (libaio + O_DIRECT)\n");
  printf("==========================================\n");
  printf(" file=%s size=%zuGB runtime=%ds\n\n", cfg.file_path.c_str(), cfg.file_size_gb, cfg.runtime_sec);

  // Check if file exists with sufficient size
  struct stat st;
  if (stat(cfg.file_path.c_str(), &st) != 0 || (size_t)st.st_size < file_size) {
    printf("  Preparing test file...\n");
    CreateTestFile(cfg.file_path, file_size);
  } else {
    printf("  Using existing file: %s (%.1f GB)\n", cfg.file_path.c_str(), (double)st.st_size / (1 << 30));
  }

  // Drop page cache
  int drop_fd = open(cfg.file_path.c_str(), O_RDONLY);
  if (drop_fd >= 0) {
    posix_fadvise(drop_fd, 0, 0, POSIX_FADV_DONTNEED);
    close(drop_fd);
  }

  int block_sizes[] = {4096, 8192, 16384};
  int thread_counts[] = {1, 2, 4, 8, 16};
  int iodepths[] = {1, 8, 32};

  printf("%-6s %-8s %-8s | %-12s %-12s %-12s\n",
         "bs", "threads", "iodepth", "IOPS", "BW(MB/s)", "lat_avg(us)");
  printf("---------------------------------------------------------------------\n");

  for (int bs : block_sizes) {
    for (int t : thread_counts) {
      for (int d : iodepths) {
        // Drop cache before each run
        int dfd = open(cfg.file_path.c_str(), O_RDONLY);
        if (dfd >= 0) {
          posix_fadvise(dfd, 0, 0, POSIX_FADV_DONTNEED);
          close(dfd);
        }

        auto r = RunBench(cfg.file_path, file_size, bs, t, d, cfg.runtime_sec);
        printf("%-6d %-8d %-8d | %-12.0f %-12.1f %-12.1f\n",
               bs, t, d, r.iops, r.bw_mbs, r.lat_avg_us);
      }
    }
  }

  printf("\n==========================================\n");
  printf(" Key configurations for AllianceANNS disk build:\n");
  printf("   - SearchCandidatesBuild: 8KB random read, libaio, iodepth=8/thread\n");
  printf("   - ReadVec (sync): 8KB random pread, iodepth=1\n");
  printf("==========================================\n");

  // Cleanup
  unlink(cfg.file_path.c_str());
  printf("Done. Test file removed.\n");
  return 0;
}
