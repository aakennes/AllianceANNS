#include "search/refinement_score_cache.hpp"
#pragma once
// alliance/index/index_disk.hpp -- SSD-resident phased build (AllianceIndexDisk)
//
// Disk-backed counterpart of AllianceIndex's Build. Graph edges and
// original vectors live on SSD via PagedFile (one node = one or more
// exclusive 4096B pages); only small O(N) bookkeeping (sorted order, label,
// attribute, RaBitQ quantization codes) stays resident in RAM.

#include <memory>
#include "build/disk_build_state.hpp"
#include "build/disk_build_config.hpp"
#include "build/disk_build_plan.hpp"
#include "build/disk_build_geometry.hpp"
#include "quantization/packed_rabitq_storage.hpp"
#include "search/guided_entry_directory.hpp"
#include "search/memory_navigation.hpp"
#include <mutex>
#include <atomic>
#include <array>
#include <type_traits>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include "cache/dynamic_block_cache.hpp"
#include "cache/static_layer_cache.hpp"
#include "cache/static_node_lookup.hpp"
#include "io/io.hpp"
#include "metrics/search_metrics.hpp"
#include "search/search_memory.hpp"
#include "search/disk_search_context.hpp"
#include "storage/storage_open_mode.hpp"
#include "io/io_handler.hpp"
#include "storage/paged_file.hpp"
#include "core/types.hpp"
#include "core/visit_list.hpp"
#include "storage/order_table.hpp"
#include "storage/window_ladder.hpp"
#include "space/space.hpp"
#include "rabitq/quantization/rabitq.hpp"
#include "rabitq/utils/rotator.hpp"

namespace alliance {

namespace disk_build {
struct ClusterSpillPlan;
}

template <typename att_t = int, typename vec_t = float> class AllianceIndexDisk {
public:
  static constexpr size_t kEdgeLockShards = 1 << 16;

  /// Build constructor: creates a fresh on-disk graph file.
  /// `node_page_bytes` overrides the per-node page size; 0 (the default) picks the
  /// smallest multiple of kMinNodePageBytes that holds one node record.
  /// `win_anchor` picks which end of the rank-window ladder is pinned -- see
  /// alliance/storage/window_ladder.hpp.
  AllianceIndexDisk(size_t max_elements, size_t vec_d, size_t M, size_t efc, std::string space_name,
                    std::string graph_file_path, size_t o = 4, size_t wp = 10,
                    bool auto_raise_wp = true, size_t enum_threshold = 2048,
                    size_t node_page_bytes = 0, WindowAnchor win_anchor = WindowAnchor::kBottom);

  /// Default load requires raw graph storage. Explicit ML-only load defers
  /// search readiness until a complete shuffled neighbor/vector layout loads.
  AllianceIndexDisk(const std::string &graph_file_path, const std::string &meta_file_path,
                    std::string space_name);
  AllianceIndexDisk(const std::string &graph_file_path, const std::string &meta_file_path,
                    std::string space_name, StorageOpenMode mode);

  bool IsMultiLevelOnly() const noexcept {
    return storage_open_mode_ == StorageOpenMode::MultiLevelOnly;
  }
  void RequireRawGraph(const char *operation) const {
    if (IsMultiLevelOnly() || !graph_file_)
      throw std::runtime_error(std::string(operation) +
                               " requires raw graph storage; unavailable in ML-only mode");
  }

  AllianceIndexDisk(const AllianceIndexDisk &) = delete;
  AllianceIndexDisk &operator=(const AllianceIndexDisk &) = delete;
  AllianceIndexDisk(AllianceIndexDisk &&) = delete;
  AllianceIndexDisk &operator=(AllianceIndexDisk &&) = delete;
  AllianceIndexDisk() = delete;

  ~AllianceIndexDisk();

private:
  StorageOpenMode storage_open_mode_{StorageOpenMode::RequireRawGraph};

public:
  // ===== Disk-backed per-node record layout =====
  // Per-layer: [(M+1) tableints: IDs + count] [(M) dist_ts: distances]

  // Appended metadata marks a contiguous upper suffix with ID-only edges.
  // Raw record offsets/pages are unchanged; obsolete distance slots are zero.
  size_t id_only_start_layer_{std::numeric_limits<size_t>::max()};
  bool HasStoredEdgeDistances(layer_t layer) const {
    return static_cast<size_t>(layer) < id_only_start_layer_;
  }

  inline auto EdgeIdBytes() const -> size_t {
    return (M_ + 1) * sizeof(tableint);
  }
  inline auto EdgeDistBytes() const -> size_t {
    return M_ * sizeof(dist_t);
  }
  inline auto EdgeBytes() const -> size_t {
    return EdgeIdBytes() + EdgeDistBytes();
  }

  inline auto EdgeOffsetInRecord(layer_t layer) const -> size_t {
    return static_cast<size_t>(wp_ - layer) * EdgeBytes();
  }

  inline auto VecOffsetInRecord() const -> size_t {
    return (wp_ + 1) * EdgeBytes();
  }

  inline auto VecBytes() const -> size_t {
    return vec_d_ * sizeof(vec_t);
  }

  /// Translates internal node ID to its physical disk slot (= rank in attribute-sorted order).
  inline auto DiskSlot(tableint id) const -> tableint {
    return static_cast<tableint>(position_by_id_[id]);
  }

  // ===== Disk-backed accessors (go through PagedFile, no RAM copy of the
  //       full graph/vectors) =====

  /// Reads node `id`'s edge list for `layer` into `out_edges` (M_+1 tableints:
  /// M_ neighbor ids followed by the count at out_edges[M_]).
  void ReadEdges(tableint id, layer_t layer, tableint *out_edges) const;

  /// Reads edge IDs and exact distances; ID-only upper layers recompute on demand.
  void ReadEdgesWithDist(tableint id, layer_t layer, tableint *out_ids, dist_t *out_dists) const;

  /// Overwrites node `id`'s edge list for `layer` on disk (in place).
  void WriteEdges(tableint id, layer_t layer, const tableint *neighbor_ids, tableint count) const;

  /// Writes IDs; stores distances only for layers that retain them on SSD.
  void WriteEdgesWithDist(tableint id, layer_t layer, const tableint *ids, const dist_t *dists,
                          tableint count) const;

  /// Reads node `id`'s original fp32 vector into `out_vec` (vec_d_ elements).
  void ReadVec(tableint id, vec_t *out_vec) const;

  // ===== Build state (mirrors AllianceIndex) =====

  /// Writes every node's zeroed placeholder record (all layers empty) + its
  /// original vector to disk, builds order_table_/sorted_ids_/position_by_id_.
  void PopulateElements(const vec_t *base_vecs, const att_t *att_vec, size_t n);

  /// Streaming variant: reads vectors from fvecs file without loading all into RAM.
  void PopulateElements(const std::string &fvecs_path, const att_t *att_vec, size_t n,
                        const std::string &format = "fvecs");

  /// Builds layers 0..wow_max_layer in attribute order with bounded left
  /// history, configured batches and parallel workers. Completion is recorded
  /// only after all graph records have been written.
  void BuildLeftStreamLayers(layer_t wow_max_layer, size_t n);

  /// Full SSD-resident build: PopulateElements + the phased layer build.
  void Build(const vec_t *base_vecs, const att_t *att_vec, size_t n, int threads = 1);

  /// Phased build after PopulateElements has already been called. Allows the
  /// caller to free basevecs before the build runs (vectors accessed from disk).
  ///
  ///   Resident: LeftStream builds layers 0..wow_max.
  ///   Upper:    cluster build for layers wow_max+1..wp_.
  ///
  /// `start_layer`: first upper-layer to process (1..wp_). Layers below this
  /// are assumed already complete on disk (from a prior run) and are skipped,
  /// provided the LeftStream resident phase is complete. Legacy resident
  /// checkpoints remain load/search-only.
  ///
  /// Completion progress is updated after each successful stage, even when
  /// checkpoint persistence is disabled. A non-empty `checkpoint_meta_path`
  /// additionally saves that state for resume through the load constructor.
  void BuildPostPopulate(int threads = 1, size_t start_layer = 1,
                         const std::string &checkpoint_meta_path = "");

  // ===== RaBitQ quantization =====

  /// Initialize RaBitQ: loads centroids & cluster IDs, computes rotated
  /// centroids. Must be called before PopulateElements (which computes
  /// per-point quantization codes) or before searchKNN.
  void InitRaBitQ(const std::string &centroids_path, const std::string &cids_path,
                  const std::string &space_name, size_t total_bits = 1);

  /// Replace the loaded RaBitQ state with a different centroid set and
  /// re-quantize every vector (read back from the graph file) against it.
  /// The graph build only has one --centroids input, which historically fed
  /// both the search-side quantization (baked into meta) and the Phase 2
  /// cluster build -- so a graph built with build-side clustering (e.g. nc16)
  /// carries coarse search-side codes. This hook lets search swap in a finer
  /// set (nc64/128) without rebuilding the graph.
  void RequantizeRaBitQ(const std::string &centroids_path, const std::string &cids_path,
                        const std::string &space_name, size_t total_bits = 1);

  /// Query-time beam search using RaBitQ distance estimation for candidate
  /// expansion (no raw vector reads during traversal), reading only edge
  /// lists from disk. Returns approximate top-ef candidates.
  template <typename filter_t>
  auto SearchCandidatesQuery(std::vector<dist_id_pair> &eps, const vec_t *query_vec,
                             const filter_t &filter, const attr_range<layer_t> &layer_rng,
                             size_t ef, size_t k) -> std::vector<dist_id_pair>;

  /// DiskANN-style query: RaBitQ-driven graph traversal + fp32 rerank from disk.
  template <typename filter_t = attr_range<att_t>>
  auto searchKNN(vec_t *query_vec, size_t efs, size_t k, const filter_t &filter)
      -> std::vector<std::pair<dist_t, label_t>>;

  /// Pipeline variant of searchKNN using libaio for async IO.
  template <typename filter_t = attr_range<att_t>>
  auto searchKNNPipeline(vec_t *query_vec, size_t efs, size_t k, const filter_t &filter,
                         DiskSearchContext *ctx) -> std::vector<std::pair<dist_t, label_t>>;

  /// Pipelined graph traversal: reads full node pages asynchronously,
  /// extracts edges for RaBitQ expansion, caches raw vectors for rerank.
  template <typename filter_t>
  auto SearchCandidatesQueryPipeline(std::vector<dist_id_pair> &eps, const vec_t *query_vec,
                                     const filter_t &filter, const attr_range<layer_t> &layer_rng,
                                     size_t ef, size_t k, DiskSearchContext *ctx,
                                     std::vector<vec_t> &vec_cache_buf,
                                     std::unordered_map<tableint, size_t> &vec_cache_idx)
      -> std::vector<dist_id_pair>;

  /// Pipelined rerank: uses cached vectors from traversal; issues async IO
  /// only for uncached nodes (rare).
  void RerankPipeline(const vec_t *query_vec, const std::vector<dist_id_pair> &candidates,
                      const std::vector<vec_t> &vec_cache_buf,
                      const std::unordered_map<tableint, size_t> &vec_cache_idx,
                      DiskSearchContext *ctx, std::vector<std::pair<dist_t, label_t>> &out_results);

  /// Brute-force search for small filter ranges: sequentially reads raw vectors
  /// from the rank-sorted vec_file and computes exact fp32 distances.
  template <typename filter_t = attr_range<att_t>>
  auto BruteForceSearch(vec_t *query_vec, size_t k, const filter_t &filter)
      -> std::vector<std::pair<dist_t, label_t>>;

  /// Writes raw vectors in rank order (sorted_ids_ order) to vec_file_path_.
  void WriteVecFile(const vec_t *base_vecs);

  /// Writes raw vectors from graph_file_ (disk) in rank order to vec_file_path_.
  void WriteVecFileFromDisk();

  /// Probes vec_file and marks it usable (sets vec_file_available_).
  void OpenVecFile();

  auto DecideLayerRange(const attr_range<att_t> &filter_range, std::vector<tableint> &OUT_eps)
      -> attr_range<layer_t>;

  /// RaBitQ distance estimator: precomputes query-side quantities once per
  /// query, then provides O(1) distance estimates per candidate.
  class RaBitQEstimator {
  public:
    RaBitQEstimator(const vec_t *query, AllianceIndexDisk *idx);
    ~RaBitQEstimator();

    auto dist_estimate_precise(uint8_t cid, tableint internal_id) -> double;
    auto dist_estimate(uint8_t cid, tableint internal_id) -> float;
    detail::RefinementScoreCache refinement_cache_;
    size_t refinement_reused_{0}, refinement_recomputed_{0};
    auto dist_estimate_with_errorbound(uint8_t cid, tableint internal_id)
        -> std::tuple<float, float>;
    // Full L2 lower bound: query norm, library model error and numerical guard.
    // Returns -infinity when a usable bound cannot be computed.
    auto l2_lower_bound(uint8_t cid, tableint internal_id) -> float;
    auto l2_lower_bound(uint8_t cid, tableint internal_id, float estimate) -> float;

    // Distance interval in the metric's native units (squared L2 or 1-IP).
    // Invalid input gives [-inf,+inf], forcing exact IO.
    auto distance_bounds(uint8_t cid, tableint internal_id, float estimate,
                         bool recompute_float = false) -> std::pair<float, float>;
    double float_query_sum_{0};

    AllianceIndexDisk *index_{nullptr};
    float *rotated_query_{nullptr};
    std::vector<float> dist_to_centroids_;
    // Lazily computed only for centroids used by LB rerank. No cost for LB-off.
    std::vector<double> norm_to_centroids_;
    float max_abs_query_{0};

    uint32_t bits = 1; // = index_->rabitq_bits_ (data-side total bits), set in ctor
  };

  inline auto GetLabelByID(tableint id) const -> label_t {
    return labels_[id];
  }
  inline auto GetAttByID(tableint id) const -> att_t {
    return atts_[id];
  }

  /// Save RAM-resident metadata (params + sorted_ids + labels + atts + RaBitQ
  /// codes/factors/cids + rotator + centroids) to a binary file.
  void Save(const std::string &meta_file_path) const;

public:
  // ===== Index parameters =====
  size_t max_elements_{0};
  size_t vec_d_{0};
  size_t wp_{0};
  size_t o_{4};
  size_t M_{32};
  size_t efc_{256};
  float alpha_{1.2f};
  size_t curvec_num_{0};
  size_t cur_max_layer_{0};

  // ===== Build checkpoint/resume state =====
  // Highest durable layer committed by BuildPostPopulate. The separate
  // left_stream_complete_ flag distinguishes an unfinished resident build
  // from a completed layer-0-only graph.
  size_t last_completed_layer_{0};

  // ===== Disk layout constants =====
  size_t node_record_bytes_{0};
  size_t node_disk_bytes_{kDiskPageSize};
  size_t header_bytes_{kDiskPageSize};

  std::unique_ptr<PagedFile> graph_file_;

  disk_build::BuildConfig build_config_;
  size_t enum_threshold_{2048};
  WindowAnchor win_anchor_{WindowAnchor::kBottom};
  // Checkpoint state is distinct from the policy for the next build session.
  bool left_stream_complete_{false};
  size_t left_stream_max_{0};
  // Borrowed only while the build driver owns its cluster inputs.
  const float *cluster_centroids_ext_{nullptr};
  const uint32_t *cluster_ids_ext_{nullptr};
  size_t cluster_K_file_{0};

  /// RAM the build holds regardless of which layer is running: the O(n)
  /// bookkeeping arrays and the order table. Per-phase working sets are added
  /// on top of this; Phase 2 uses the measured process floor when available.
  auto EstimateBaseBuildBytes(size_t n) const -> size_t;

  /// Phase 1.5 budget derivation. Reverse-solves the largest node-window that
  /// fits AVAIL = budget - BASE and the wow_max it yields. See §四 of the plan.
  using Phase15Plan = disk_build::Phase15Plan;
  auto PlanPhase15(size_t n) const -> Phase15Plan;

  /// Phase 2 (cluster) budget derivation. Three sub-phases with materially
  /// different footprints, all modelled separately because the budget must clear
  /// the worst of them and the per-phase RSS lines are only useful if each is
  /// compared against its own number:
  ///
  ///  - prep:   Vamana partition workspace, rank-major memberships/CSR and
  ///            bounded sequential distribution of vectors and seed edges.
  ///  - pass A: the largest cluster's exact vectors and WoW adjacency, plus
  ///            per-thread graph-search scratch. Memberships are nearest-two;
  ///            the measured maximum cluster size controls capacity.
  ///  - pass B: the state plus K double-buffered merge streams and the kBlk
  ///            node-record staging block. Pass A's per-cluster buffers are
  ///            scoped to pass A, so they are NOT charged here.
  ///
  /// residency_bytes is the max of the three; the reverse-solve finds the
  /// smallest K whose residency fits AVAIL.
  using Phase2Plan = disk_build::Phase2Plan;
  auto PlanPhase2(size_t n) const -> Phase2Plan;

  // Phase 2 state is defined in disk_build/partition/membership.hpp.
  // Setup/Teardown manage the stage lifetime; index destruction is the fallback.
  using DiskClusterBuildState = disk_build::ClusterBuildState;
  auto BuildWorkspace() -> disk_build::BuildState<att_t, vec_t> &;
  auto ClusterScratch() noexcept -> DiskClusterBuildState * {
    return build_state_ ? build_state_->cluster.get() : nullptr;
  }
  auto ClusterScratch() const noexcept -> const DiskClusterBuildState * {
    return build_state_ ? build_state_->cluster.get() : nullptr;
  }

  /// Prepare nearest-two Vamana memberships and rank-major CSR. External
  /// partitions keep their supplied K; otherwise sampling and budget checks
  /// choose K. Search-side quantization is independent of this workspace.
  void ClusterPrepDisk(const float *centroids, const uint32_t *cluster_ids, size_t n,
                       size_t K_file);
  void AssignNearestTwoClustersDisk(const float *centroids, size_t K, size_t n,
                                    std::vector<uint32_t> &OUT_cluster_ids);
  void SynthesizeVamanaClustersDisk(size_t &K, size_t n, std::vector<float> &centroids,
                                    std::vector<uint32_t> &ids);
  void SetupClusterBuildDisk(const uint32_t *cluster_ids, size_t K, size_t n);
  void TeardownClusterBuildDisk();

  /// Pass A builds each Vamana cluster with exact-distance WoW and spills IDs.
  /// Pass B merges the two memberships in rank order, deduplicates, applies
  /// the fixed per-node shuffle and truncates to M before writing graph pages.
  void ClusterBuildUpperLayersDisk(layer_t start_layer, layer_t end_layer, size_t n);

private:
  void BuildClusterWoWEdgesDisk(const disk_build::ClusterSpillPlan &plan);
  void CommitClusterEdgesDisk(const disk_build::ClusterSpillPlan &plan);

public:
  /// Prints the EstimateBaseBuildBytes breakdown and, when build_mem_gb_ > 0,
  /// the Phase 1.5 reverse-solve (AVAIL, derived wow_threshold, wow_max).
  void ReportBuildMemoryBudget(size_t n) const;

  /// Current and peak RSS in bytes from /proc/self/status, 0 0 where /proc is
  /// not mounted. (Not index IO -- procfs has no O_DIRECT and these are two
  /// ~1KB reads per phase, so they stay on stdio.)
  static void ProcRss(size_t *cur_bytes, size_t *peak_bytes);

  /// Resets VmHWM to the current VmRSS via /proc/self/clear_refs, so the next
  /// phase's peak is attributable to that phase instead of being whatever the
  /// largest earlier phase happened to reach. Needs Linux >= 4.0
  /// (CLEAR_REFS_MM_HIWATER_RSS); a silent no-op elsewhere, which just leaves
  /// the peaks cumulative.
  static void ResetProcRssPeak();

  /// One model-vs-measured line at a phase boundary. `model_bytes` is what the
  /// planner predicted this phase would hold resident; the whole point is that
  /// --build_mem_gb reverse-solves thresholds from that number, so a model off
  /// by 2x is more dangerous than no budget at all. Prints the ratio and, under
  /// a budget, whether the measured peak actually stayed inside it. Resets the
  /// peak on the way out so each phase reports its own.
  void ReportPhaseRss(const char *stage, size_t model_bytes) const;

  /// Measured VmRSS at the Phase 1.5 -> Phase 2 boundary, after the codes and
  /// LeftStream temporary buffers are gone and malloc_trim has run: the real floor Phase 2's
  /// residency sits on top of. 0 until that boundary is crossed (dry-run
  /// reports, --resume before Phase 1.5), in which case PlanPhase2 falls back to
  /// EstimateBaseBuildBytes.
  ///
  /// Why measure rather than estimate: on arxiv-2M the estimate is 89.7 MB and
  /// the measurement 97.9 MB, and prep then leaves ~70 MB of arena that no
  /// malloc_trim reaches. Solving derived_K against the estimate spends that
  /// ~78 MB twice -- once as unmodelled floor, once as cluster residency -- and
  /// at --build_mem_gb 1.0 that is most of why K=10's honest 1.13 GB was
  /// reported as fitting in 1.00 GB.
  size_t measured_base_bytes_{0};

  /// RSS retained above base + persistent memberships after ClusterPrep trim.
  /// Subsequent partition estimates charge this residue to distribution and both
  /// graph passes. It is zero before preparation; it is not a separate hard limit.
  size_t measured_prep_residue_bytes_{0};

  // ===== RAM-resident O(N) bookkeeping (small: label + att only) =====
  std::vector<label_t> labels_;
  std::vector<att_t> atts_;
  std::vector<tableint> sorted_ids_;
  std::vector<tableint> position_by_id_; // n < 2^32, so 4 B/pt suffices
  std::vector<size_t> window_size_;

  // Owned by this index. Public raw views are retained for compatibility;
  // callers borrow them and must not delete or replace owned resources.
  SortedVectorOrderTable<att_t> *order_table_{nullptr};

  // ===== Distance =====
  SpaceInterface<vec_t> *space_{nullptr};
  DISTFUNC<vec_t> fstdistfunc_{nullptr};
  void *dist_func_param_{nullptr};

  // ===== Concurrency: sharded locks for resident graph edge updates =====
  std::array<std::mutex, kEdgeLockShards> edge_locks_;
  inline auto EdgeLockFor(tableint id) -> std::mutex & {
    return edge_locks_[id & (kEdgeLockShards - 1)];
  }

  // ===== RaBitQ quantization state (RAM-resident, small per point) =====
  rabitqlib::Rotator<float> *rotator_{nullptr};
  rabitqlib::quant::RabitqConfig rabitq_config_;
  size_t padded_dim_{0};
  size_t rabitq_bits_{1};     // total quantization bits per dim (1/2/3/4/8)
  size_t quant_vec_bytes_{0}; // padded_dim_ * rabitq_bits_ / 8 (bit-plane packed)
  size_t maxC_{0};
  float *centroids_{nullptr};       // maxC_ * padded_dim_ (rotated centroids)
  std::vector<uint8_t> cids_;       // N cluster IDs
  std::vector<uint8_t> quant_vecs_; // N * quant_vec_bytes_ (bit-plane packed)
  std::vector<float> factors_;      // N * 4 (f_add, f_rescale, f_error, popcount)
  PackedRaBitQStorage packed_rabitq_;
  std::vector<uint8_t> search_cids_;

private:
  // Immutable after loading; load/clear requires exclusive index access.
  std::unique_ptr<MemoryNavigationGraph> navigation_;

public:
  bool HasNavigation() const noexcept {
    return bool(navigation_);
  }
  size_t NavigationBytes() const noexcept {
    return navigation_ ? navigation_->Bytes() : 0;
  }
  const MemoryNavigationGraph &Navigation() const {
    if (!navigation_)
      throw std::logic_error("RAM navigation is not loaded");
    return *navigation_;
  }
  void ClearNavigation() noexcept {
    navigation_.reset();
  }
  void LoadNavigation(const std::string &path) {
    if (!ml_layout_loaded_ || !rabitq_initialized_ || !order_table_ ||
        sorted_ids_.size() != curvec_num_)
      throw std::logic_error(
          "navigation requires a loaded layout, quantization and full attribute order");
    if (rabitq_space_name_ != "ip" && rabitq_space_name_ != "l2")
      throw std::invalid_argument("navigation requires L2 or IP quantization");
    auto next = std::make_unique<MemoryNavigationGraph>(
        MemoryNavigationGraph::Load(path, curvec_num_, vec_d_));
    next->Match(*this, rabitq_space_name_ == "ip" ? 1 : 2);
    // A loaded navigation graph owns entry selection; do not pay for both directories.
    guided_entries_.Clear();
    navigation_ = std::move(next);
  }
  GuidedEntryDirectory guided_entries_;
  void PrepareGuidedEntries() {
    if (HasNavigation())
      throw std::logic_error("guided entries cannot be combined with RAM navigation");
    if (!rabitq_initialized_ || sorted_ids_.size() != curvec_num_)
      throw std::logic_error("guided entries require quantization and full attribute order");
    guided_entries_.Prepare(sorted_ids_, maxC_, [&](tableint id) { return SearchCID(id); });
  }
  const uint8_t *RaBitQCode(tableint id) const noexcept {
    return packed_rabitq_.Empty() ? quant_vecs_.data() + size_t(id) * quant_vec_bytes_
                                  : packed_rabitq_.Code(id);
  }
  detail::RaBitQCodeLayout RaBitQLayout() const noexcept {
    return packed_rabitq_.Layout();
  }
  void CopyCanonicalRaBitQCode(tableint id, uint8_t *out) const {
    if (packed_rabitq_.Empty())
      std::memcpy(out, RaBitQCode(id), quant_vec_bytes_);
    else
      packed_rabitq_.CopyCanonicalCode(id, out);
  }
  const float *RaBitQFactors(tableint id) const noexcept {
    return packed_rabitq_.Empty() ? factors_.data() + size_t(id) * 4 : packed_rabitq_.Factors(id);
  }
  uint8_t SearchCID(tableint id) const noexcept {
    return search_cids_.empty() ? cids_[labels_[id]] : search_cids_[id];
  }
  void PrepareSearchCIDs() {
    ClearNavigation();
    guided_entries_.Clear();
    if (!rabitq_initialized_)
      throw std::runtime_error("RaBitQ not initialized");
    search_cids_.resize(curvec_num_);
    for (size_t i = 0; i < curvec_num_; ++i)
      search_cids_[i] = cids_.at(labels_[i]);
  }
  void PackRaBitQForSearch() {
    if (!rabitq_initialized_)
      throw std::runtime_error("RaBitQ not initialized");
    if (!packed_rabitq_.Empty())
      return;
    packed_rabitq_.Assign(
        quant_vecs_, factors_, curvec_num_, quant_vec_bytes_,
        // Use the original library compact kernels for all supported multibit
        // widths. The one-bit canonical layout already matches its library kernel.
        detail::NativeCodeLayout(rabitq_bits_));
    decltype(quant_vecs_){}.swap(quant_vecs_);
    decltype(factors_){}.swap(factors_);
  }
  bool rabitq_initialized_{false};
  std::string rabitq_space_name_;
  SpaceInterface<vec_t> *rabitq_space_{nullptr};
  DISTFUNC<vec_t> rabitq_distfunc_{nullptr};
  void *rabitq_dist_func_param_{nullptr};

  // ===== Dense build vector file (4KB-page-aligned, for fast build-time access) =====
  std::string build_vec_file_path_;
  // Derived from dimension; existing <=1024D buildvecs remain byte-compatible.
  // Storage pages are powers of two. Buffer alignment is independent of length.
  static auto BuildVectorPageBytes(size_t vector_bytes) -> size_t {
    return disk_build::BuildVectorPageBytes(vector_bytes);
  }
  auto VecPageBytes() const -> size_t {
    return BuildVectorPageBytes(VecBytes());
  }
  size_t vecs_per_page_{0}; // floor(VecPageBytes() / VecBytes())

  void WriteBuildVecFile(const vec_t *base_vecs, size_t n);

  /// Pack a per-dim RaBitQ code (`code[d]` in 0..2^rabitq_bits_-1, one byte per
  /// padded dim) into `rabitq_bits_` bit-planes at `packed` (quant_vec_bytes_
  /// total). Plane p occupies [p*padded_dim_/8, (p+1)*padded_dim_/8); bit d of
  /// plane p is (code[d]>>p)&1. Returns pop_count = sum_d code[d] (the dequant
  /// popcount term; equals set-bit count when rabitq_bits_==1).
  void PackMultiBitCode(const uint8_t *code, uint8_t *packed, float &pop_count_out) const;

  /// Same output as WriteBuildVecFile, but sources the vectors from the graph
  /// file, which already holds them in rank order once PopulateElements' record
  /// write pass is done. Costs O(chunk) RAM instead of an O(n * vec_d) copy of
  /// the whole base set, and needs no extra pass over the input file.
  void WriteBuildVecFileFromGraph(size_t n);

  /// The only read path into the build vec file. It replaced a per-vector
  /// synchronous `pread` helper, which was both non-async and page-cache-
  /// mediated: here pages that several requested vectors share are read once,
  /// and consecutive pages are merged into a single multi-page read, so a
  /// rank-ascending sweep moves 1MB per iocb instead of 4KB.
  ///
  /// `ranks` is indexed by slot. `perm`, when non-null, maps rank-ascending
  /// position -> slot, so the reads walk ascending pages (which is what makes
  /// both the dedup and the extent merging pay off). Pass `perm == nullptr`
  /// when `ranks` is already ascending.
  ///
  /// `out_by_slot` picks the destination layout: true writes each vector at
  /// `out + slot * vec_d_`, so `out` must span every slot `perm` can name --
  /// use it when several threads fill disjoint slots of one shared buffer.
  /// False writes at `out + k * vec_d_` in call order, so `out` only needs
  /// `count` vectors -- use it when a thread consumes its own chunk. The two
  /// coincide when `perm == nullptr`.
  void GatherVecsByRank(IOHandler &io, const uint32_t *ranks, const uint32_t *perm, size_t count,
                        vec_t *out, bool out_by_slot = true) const;

  // ===== Raw vector file for brute-force search =====
  std::string vec_file_path_; // path to rank-sorted raw vectors file
  bool vec_file_available_{false};

  // ===== Multi-level disk layout (optional; written by ExportMultiLevelLayout,
  //       loaded via LoadMultiLevelLayout). When ml_layout_loaded_ is false
  //       (the default), all edge reads use the legacy single-record ReadEdges
  //       path above -- this is purely additive, opt-in behavior. =====
  bool ml_layout_loaded_{false};
  std::vector<LevelRange> ml_levels_;
  std::vector<size_t> ml_record_size_per_level_;
  std::vector<std::vector<int32_t>> ml_disk_blocks_per_level_;
  // O(1) layer-to-level lookup (replaces linear scan in FindLevelForLayer).
  // Indexed by layer; value = level index within ml_levels_.
  std::vector<size_t> ml_layer_to_level_;
  int ml_fd_{-1};
  size_t ml_header_bytes_{0};
  size_t ml_block_size_{0}; // neighbor block bytes
  size_t ml_vec_block_size_{0};
  // v2 block IDs use the graph header as their base; v3 IDs use the vector
  // region base. Resolve that distinction on load, before any query IO.
  uint64_t ml_vec_base_bytes_{0};
  uint64_t MultiLevelVectorOffset(size_t block) const {
    return ml_vec_base_bytes_ + static_cast<uint64_t>(block) * ml_vec_block_size_;
  }
  size_t ml_layer_stride_{1};     // from layout header; 1 = all layers
  layer_t ml_layer_grid_base_{0}; // lowest selected layer = cur_max_layer % stride

  // ===== Colocated search-vector region of the multi-level layout (written
  //       by ExportMultiLevelLayout when enable_search_vectors is on and no
  //       separate out_vec_path was used; mappings come from the sidecar).
  //       When loaded, searchKNNMultiLevelPipeline reranks via
  //       RerankMultiLevel instead of falling back to per-candidate node
  //       record reads from the baseline graph file. =====
  bool ml_vec_region_loaded_{false};
  std::vector<size_t> ml_vec_block_ids_; // v2: header-relative; v3: vector-region-relative
  std::vector<size_t> ml_vec_slots_;     // node_id -> slot within that block
  struct MlCacheConfig {
    // One unified thread-local cache spanning all levels (WoW-style): the
    // full budget is NOT split per level; pieces are sized to the largest
    // level record so any level's record fits any slot.
    size_t capacity_bytes{0};
    size_t piece_bytes{0};
    size_t num_nodes{0};
    DynamicBlockCache::EvictPolicy policy{DynamicBlockCache::EvictPolicy::FIFO};
    uint64_t generation{0};
  };
  MlCacheConfig ml_cache_config_;

  void InvalidateMultiLevelCache() {
    ++ml_cache_config_.generation;
  }

  bool per_query_cache_{false};
  // Size the dynamic cache to one query's working set instead of the full
  // --ml_cache_bytes budget. Only takes effect together with per_query_cache_,
  // which is what makes the budget unusable in the first place: the cache is
  // wiped between queries, so slots beyond the per-query working set can never
  // be hit. Hit rate is unchanged once the size converges; only memory drops.
  bool ml_cache_auto_size_{true};
  // Per-thread maxima of the thread-local dynamic caches, published at the
  // per-query Clear(). Reported by the search driver for memory accounting.
  mutable std::atomic<size_t> metric_ml_cache_resident_bytes_{0};
  mutable std::atomic<size_t> metric_ml_cache_peak_slots_{0};
  mutable std::atomic<size_t> metric_ml_cache_peak_wanted_{0};
  // Max-update the published footprint from the calling thread's cache. Called
  // once per query right after Clear(), so the cost has to stay at a few
  // relaxed loads: the compare-exchange only runs when a value actually grew,
  // which after auto-sizing converges is never.
  void PublishMultiLevelCacheFootprint(const DynamicBlockCache &cache) const {
    auto grow = [](std::atomic<size_t> &slot, size_t value) {
      size_t cur = slot.load(std::memory_order_relaxed);
      while (value > cur && !slot.compare_exchange_weak(cur, value, std::memory_order_relaxed)) {
      }
    };
    grow(metric_ml_cache_resident_bytes_, cache.resident_bytes());
    grow(metric_ml_cache_peak_slots_, cache.peak_slots());
    grow(metric_ml_cache_peak_wanted_, cache.peak_wanted_slots());
  }
  void ClearThreadLocalMultiLevelCache() const;

  // ===== Shared read-only static node cache =====
  // Checked before the dynamic cache. Budgets are allocated per level using
  // the landing-layer distribution, or equally when no training weight exists.
  // Node selection uses in-edge weights over each level's owned layers.
  // Each selected node stores a merged range covering its selected levels;
  // overlapping layers are stored once. Prepared at export or sidecar-build
  // time, persisted, then loaded read-only for queries.
  std::vector<StaticLayerGroup> static_groups_;      // per-level budget (build-time only)
  std::vector<StaticNodeMeta64> static_node_meta64_; // populated only after 32-bit overflow
  std::vector<StaticNodeMeta> static_node_meta_;     // build-time only; v2 persistence
  StaticNodeLookup static_node_lookup_;              // compact shared read-only query lookup
  size_t StaticMetadataBytes() const {
    return static_node_meta_.capacity() * sizeof(StaticNodeMeta) +
           static_node_meta64_.capacity() * sizeof(StaticNodeMeta64) + static_node_lookup_.Bytes();
  }
  size_t StaticResidentBytes() const {
    return (static_cache_bytes_ ? ((static_cache_bytes_ + 4095) & ~size_t(4095)) : 0) +
           StaticMetadataBytes();
  }
  char *static_cache_buf_{nullptr};     // contiguous buffer of per-node records
  size_t static_cache_bytes_{0};        // size of static_cache_buf_
  size_t static_cache_budget_bytes_{0}; // configured budget (bytes) across all levels
  bool static_cache_loaded_{false};     // static cache usable at query time
  // Optional offline landing-layer histogram for static-cache preparation.
  // Loaded by the sidecar preparer; search never collects or mutates it.
  std::vector<uint64_t> landing_layer_cnt_;

  // ---- Static cache build (stage 2: allocation + node selection) ----
  // Split `budget_bytes` across levels by the landing-layer distribution.
  void AllocateStaticGroups(const std::vector<LevelRange> &levels,
                            const std::vector<size_t> &record_sizes, size_t level_k, size_t level_m,
                            size_t stride, size_t budget_bytes, size_t num_nodes);
  // Compute full-level in-edge weights for [layer_begin, layer_end] at the
  // given stride. Returns in_weights[node_id] = number of edges pointing to
  // that node across all selected layers in the range.
  auto ComputeLevelInWeights(layer_t layer_begin, layer_t layer_end, size_t stride) const
      -> std::vector<size_t>;
  // Select nodes for all levels using full-level in-weights, merge cross-level
  // overlaps, redistribute saved memory. Populates static_node_meta_.
  void SelectAndMergeStaticNodes(const std::vector<LevelRange> &levels,
                                 const std::vector<size_t> &record_sizes, size_t stride,
                                 const std::vector<std::vector<size_t>> *weights = nullptr);
  // Build new v2 sidecars using existing graph blocks; graph is never rewritten.
  // Each output must be new. Budget covers query-resident payload and metadata.
  void PrepareStaticCacheSidecars(const std::string &graph, const std::string &sidecar,
                                  const std::vector<std::pair<std::string, size_t>> &outputs);
  // ---- Static cache stage 3: buffer fill + persistence ----
  // Allocate static_cache_buf_ (page-aligned) based on static_node_meta_ layout.
  void AllocateStaticCacheBuffer();
  // Copy assembled layers into a cached node's record. Called from
  // ExportMultiLevelLayout's record assembly loop -- copies only the layers
  // within [level_lb, level_le] that fall inside the node's cached range.
  void FillStaticCacheRecord(size_t li, tableint node_id, const char *record, size_t bytes);
  // ---- Static cache stage 4: query-time lookup ----
  // O(1) per-node lookup. Returns covering Entry when the node's cached range
  // includes `layer`; otherwise {found=false}.
  auto StaticCacheLookup(tableint id, layer_t layer) const -> DynamicBlockCache::Entry;
  // Public sum counters share their registry with worker-local values, reset
  // and publication. Names, mutable atomic types and C-array interfaces stay
  // compatible; cache-footprint maxima keep their separate lifecycle above.
  // Block reads count decode/insert events; submits count physical requests
  // after coalescing. Inflight_sum / loop_iters is sampled queue occupancy.
#define ALLIANCE_SEARCH_METRIC(name) mutable std::atomic<size_t> name{0};
#define ALLIANCE_SEARCH_METRIC_ARRAY(name, count) mutable std::atomic<size_t> name[count]{};
#include "metrics/search_metric_fields.inc"
#undef ALLIANCE_SEARCH_METRIC_ARRAY
#undef ALLIANCE_SEARCH_METRIC
  static constexpr size_t kMaxMetricLevels =
      std::extent_v<decltype(metric_ml_block_reads_per_level_)>;
  static constexpr size_t kMaxMetricLayers =
      std::extent_v<decltype(metric_ml_route_hits_per_layer_)>;

  /// Loads a multi-level layout previously written by ExportMultiLevelLayout:
  /// opens a second (read-only) fd for `graph_path`, parses its header + level
  /// entries, and loads the sidecar (`meta_path`) node_id -> block_id mappings.
  /// After this call, SearchCandidatesQuery reads edges via ReadEdgesMultiLevel
  /// instead of the legacy single-record ReadEdges.
  void LoadMultiLevelLayout(
      const std::string &graph_path, const std::string &meta_path,
      size_t cache_capacity_bytes = 32 * 1024 * 1024,
      DynamicBlockCache::EvictPolicy policy = DynamicBlockCache::EvictPolicy::FIFO);

  /// Reads node `id`'s edge list for `layer` via the multi-level layout: finds
  /// the owning level (FindLevelForLayer), checks ml_cache_, and on a miss
  /// reads the whole physical block, decoding + caching every record found in
  /// it (not just the requested node) before returning the requested slice.
  /// Requires LoadMultiLevelLayout to have been called first.
  void ReadEdgesMultiLevel(tableint id, layer_t layer, tableint *out_edges) const;

  /// Pipeline variant of the multi-level search: same RaBitQ-driven beam
  /// search as searchKNNPipeline, but edge reads go through the multi-level
  /// cache (per SearchCandidatesQueryMultiLevelPipeline) instead of one full
  /// node-page read per hop. Requires LoadMultiLevelLayout to have been
  /// called first, and `ctx->ml_io_handler` to be open (construct
  /// DiskSearchContext with the ml_graph_path argument).
  template <typename filter_t = attr_range<att_t>>
  auto searchKNNMultiLevelPipeline(vec_t *query_vec, size_t efs, size_t k, const filter_t &filter,
                                   DiskSearchContext *ctx)
      -> std::vector<std::pair<dist_t, label_t>>;

  /// Async multi-level graph traversal. A cache hit enters ready_queue without
  /// occupying an IO slot. Expansion visits only layers covered by that cached
  /// record and finishes at its lower boundary; it does not pause and resume
  /// across levels. Concurrent candidates whose miss maps to the same physical
  /// block share one IO (see in_flight_block_slot in the .cpp).
  /// Returns up to GraphRerankBudget(k) unique candidates, including reserve
  /// entries when that budget exceeds ef. Reserve never expands graph edges.
  template <typename filter_t, bool post_filter = false>
  auto SearchCandidatesQueryMultiLevelPipeline(
      std::vector<dist_id_pair> &eps, const vec_t *query_vec, const filter_t &filter,
      const attr_range<layer_t> &layer_rng, size_t ef, size_t k, DiskSearchContext *ctx,
      // Optional borrowed estimator, prepared for this exact query and index.
      // Lifetime is owned by the caller; omitted for standalone candidate use.
      RaBitQEstimator *query_estimator = nullptr) -> std::vector<dist_id_pair>;

  /// Rerank candidates by reading raw fp32 vectors from the multi-level
  /// layout's colocated vector region: candidates are grouped by physical
  /// vector block and deduped, so each distinct block costs exactly one
  /// (batched, async) IO. Requires ml_vec_region_loaded_ and an open
  /// ctx->ml_io_handler. Counts IOs in metric_ml_rerank_io_submits_.
  void RerankMultiLevel(const vec_t *query_vec, const std::vector<dist_id_pair> &candidates,
                        DiskSearchContext *ctx,
                        std::vector<std::pair<dist_t, label_t>> &out_results);
  void RerankMultiLevelBounded(const vec_t *query_vec, const std::vector<dist_id_pair> &candidates,
                               size_t k, RaBitQEstimator &estimator, DiskSearchContext *ctx,
                               std::vector<std::pair<dist_t, label_t>> &out_results);

  // ===== Default graph rerank budget =====
  // Default graph rerank multiplier when ctx->rerank_num is zero. An explicit
  // per-query budget overrides it; reserve candidates can extend beyond ef.
  // The default 8 is ported from StorageAirRFANNS's rerankt_times_upper_bound_.
  size_t rerank_times_upper_bound_{8};

  // ===== Legacy/build counters outside the ML sum registry =====
  std::atomic<size_t> metric_disk_reads_{0};

private:
  friend struct DiskBuildTestPeer;
  // Idempotent cleanup for the existing owned raw fields. No persistent
  // files are removed, and borrowed external centroid/CID arrays are untouched.
  std::unique_ptr<disk_build::BuildState<att_t, vec_t>> build_state_;
  void ReleaseBuildResources() noexcept;
  void ReleaseOwnedResources() noexcept;
};

} // namespace alliance
