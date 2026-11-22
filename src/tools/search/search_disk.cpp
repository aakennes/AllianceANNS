#include "index/index_disk.hpp"
#include "common/bench_utils.hpp"
#include "search/search_disk_options.hpp"
#include "search/search_disk_checkpoint.hpp"
#include "search/search_preparation.hpp"
#include <iomanip>
#include "search/search_disk_runner.hpp"
#include "search/search_disk_report.hpp"
#include "search/search_io_budget.hpp"
#include <algorithm>
#include <chrono>
#include <fstream>

int main(int argc, char **argv) {
  std::cout << std::setprecision(17);
  benchmark::DiskSearchOptions options;
  try {
    if (!options.ParseAndValidate(argc, argv))
      return 1;
  } catch (const std::exception &error) {
    std::cerr << "ERROR: " << error.what() << std::endl;
    return 1;
  }

  const auto io_plans =
      alliance::PlanSearchIOBudget(options.search_threads, options.beam_width,
                                   options.rerank_beam_width, options.io_budget_total);
  size_t planned_peak = 0;
  for (const auto &p : io_plans)
    planned_peak += std::max(p.search, p.rerank);
  std::cout << "# io_budget: total=" << options.io_budget_total
            << " workers=" << options.search_threads << " search_min=" << io_plans.back().search
            << " search_max=" << io_plans.front().search << " rerank_min=" << io_plans.back().rerank
            << " rerank_max=" << io_plans.front().rerank << " outstanding_bound=" << planned_peak
            << " policy=fixed_worker_shares" << std::endl;

  std::cout << "=== AllianceIndexDisk Search ===\n"
            << "  graph_file: " << options.graph_file << "\n"
            << "  meta_file: " << options.meta_file << "\n"
            << "  query_vec: " << options.query_vec_path << "\n"
            << "  query_rng: " << options.query_rng_path << "\n"
            << "  gt_file: " << options.gt_file << "\n"
            << "  k=" << options.k << " space=" << options.space
            << " pipeline=" << options.use_pipeline << " beam_width=" << options.beam_width
            << " rerank_beam_width=" << options.rerank_beam_width
            << " threads=" << options.search_threads << "\n";
  if (options.ml_pipeline_mode) {
    std::cout << "  ml_pipeline_mode: ON (ml_graph_file=" << options.ml_graph_file
              << " ml_meta_file=" << options.ml_meta_file
              << " ml_cache_per_thread_bytes=" << options.ml_cache_bytes
              << " ml_cache_total_payload_bytes=" << options.ml_cache_bytes * options.search_threads
              << " ml_cache_policy=" << options.ml_cache_policy << ")\n";
  } else if (options.compare_mode) {
    std::cout << "  compare_mode: ON (ml_graph_file=" << options.ml_graph_file
              << " ml_meta_file=" << options.ml_meta_file
              << " ml_cache_bytes=" << options.ml_cache_bytes
              << " ml_cache_policy=" << options.ml_cache_policy << ")\n";
  }

  benchmark::PrintMemoryPhase("before_index", options.process_memory_budget_bytes);
  const auto storage_mode = options.ml_only ? alliance::StorageOpenMode::MultiLevelOnly
                                            : alliance::StorageOpenMode::RequireRawGraph;
  alliance::AllianceIndexDisk<int, float> index(options.graph_file, options.meta_file,
                                                options.space, storage_mode);
  benchmark::PrintMemoryPhase("base_index_loaded", options.process_memory_budget_bytes);

  // Optional search-side re-encoding; build clustering has separate inputs.
  // Frozen checkpoints retain their original shape/fingerprint checks below.
  if (options.rabitq_bits != 1 && options.rabitq_bits != 2 && options.rabitq_bits != 3 &&
      options.rabitq_bits != 4 && options.rabitq_bits != 8) {
    std::cerr << "--rabitq_bits must be 1, 2, 3, 4, or 8 (got " << options.rabitq_bits << ")\n";
    return 1;
  }
  // Changing the code bit width means re-encoding every vector, which
  // RequantizeRaBitQ does (reading raw vectors back from the graph). It needs a
  // centroid set: reuse the graph's own via --rabitq_centroids/--rabitq_cids.
  if (options.rabitq_bits != index.rabitq_bits_ &&
      (options.rabitq_centroids.empty() || options.rabitq_cids.empty())) {
    std::cerr << "ERROR: --rabitq_bits " << options.rabitq_bits
              << " differs from the graph's baked " << index.rabitq_bits_
              << "-bit codes; pass --rabitq_centroids/--rabitq_cids to re-encode\n";
    return 1;
  }
  uint64_t checkpoint_hash = 0;
  if (!options.quant_checkpoint.empty()) {
    checkpoint_hash = benchmark::LoadSearchCheckpoint(index, options);
    std::cout << "# frozen_quantization: " << checkpoint_hash << std::endl;
  } else if (!options.rabitq_centroids.empty() || !options.rabitq_cids.empty()) {
    if (options.rabitq_centroids.empty() || options.rabitq_cids.empty()) {
      std::cerr << "ERROR: --rabitq_centroids and --rabitq_cids must be given together\n";
      return 1;
    }
    index.RequantizeRaBitQ(options.rabitq_centroids, options.rabitq_cids, options.space,
                           options.rabitq_bits);
  }

  std::vector<benchmark::CheckpointProbe> checkpoint_probe;
  if (!options.quant_checkpoint.empty())
    checkpoint_probe = benchmark::CheckpointScoreProbe(index, false);
  if (options.ml_pipeline_mode) {
    benchmark::PrepareSearchCodes(index, options);
    benchmark::PrepareGuidedEntriesForSearch(index, options);
  }
  std::cout << "# rabitq_code_layout: " << alliance::detail::CodeLayoutName(index.RaBitQLayout())
            << std::endl;
  std::vector<int> checkpoint_attributes;
  if (!options.quant_checkpoint.empty()) {
    if (benchmark::QuantizationFingerprint(index) != checkpoint_hash)
      throw std::runtime_error("packed quantization changed canonical code/factor fingerprint");
    benchmark::ValidateCheckpointProbes(checkpoint_probe,
                                        benchmark::CheckpointScoreProbe(index, true));
    checkpoint_attributes = benchmark::DenseCheckpointAttributes(index);
  }
  if (options.has_ml) {
    benchmark::LoadSearchLayout(index, options);
    benchmark::LoadNavigationForSearch(index, options);
    benchmark::ConfigureSearchCache(index, options, benchmark::CachePreparation::Ordinary);
  }

  benchmark::PrintMemoryPhase("layout_loaded", options.process_memory_budget_bytes);
  benchmark::PrintSharedMemory(index);

  benchmark::SearchRunProfile profile; // outlives the contexts below
  benchmark::SearchContexts search_contexts;
  if (options.use_pipeline) {
    const size_t context_count = options.ml_pipeline_mode ? options.search_threads : 1;
    search_contexts.reserve(context_count);
    for (size_t tid = 0; tid < context_count; ++tid) {
      const auto &io_plan = io_plans[tid];
      if (options.ml_pipeline_mode) {
        search_contexts.emplace_back(std::make_unique<alliance::DiskSearchContext>(
            io_plan.queue_depth, options.graph_file, options.ml_graph_file, storage_mode));
      } else {
        search_contexts.emplace_back(std::make_unique<alliance::DiskSearchContext>(
            options.beam_width + options.rerank_beam_width, options.graph_file));
      }
      auto *search_ctx = search_contexts.back().get();
      if (options.ml_pipeline_mode)
        search_ctx->UseOwnedWorkspace();
      benchmark::ConfigureSearchPolicy(*search_ctx, options, io_plan);
    }
    auto *search_ctx = search_contexts.front().get();
    benchmark::PrintSearchEntryConfiguration(index, options);
    std::cout << "# filter_config: mode=" << options.filter_mode
              << " threshold=" << options.prefilter_threshold
              << " rerank=" << options.prefilter_rerank << " (0=max(k,efs))" << std::endl;
    std::cout << "# rerank_config: error_bound=" << options.rerank_error_bound
              << " query_estimator=float_rabitq_native float_refine=" << options.rerank_float_refine
              << " candidate_factor=" << options.rerank_candidate_factor
              << " prescreen=" << options.rerank_bound_prescreen
              << " bound_rank_factor=" << options.rerank_bound_rank_factor
              << " warm_blocks=" << options.rerank_warm_blocks << " model_error=library_unscaled"
              << " batch_blocks=" << options.rerank_batch_blocks
              << " beam_width=" << options.beam_width
              << " rerank_beam_width=" << search_ctx->RerankWidth()
              << " queue_depth=" << search_ctx->queue_depth << " rerank_num=" << options.rerank_num
              << " rerank_factor=" << options.rerank_factor << std::endl;
  }

  size_t d, nq;
  float *query_vecs = benchmark::fvecs_read(options.query_vec_path, d, nq);
  std::cout << "Loaded query vectors: nq=" << nq << " d=" << d << "\n";

  auto query_filters = benchmark::LoadRange(options.query_rng_path);
  nq = std::min(nq, query_filters.size());

  std::vector<std::vector<alliance::label_t>> gt;
  if (!options.gt_file.empty()) {
    gt = benchmark::LoadGroundTruth(options.gt_file);
    nq = std::min(nq, gt.size());
  }
  if (options.query_limit > 0)
    nq = std::min(nq, options.query_limit);
  if (nq == 0)
    throw std::runtime_error("no queries to run");
  if (d != index.vec_d_)
    throw std::runtime_error("query vector dimension mismatch");

  std::vector<size_t> efs_list = {800, 600, 400, 300, 250, 200, 180, 160, 140, 120, 100, 90, 80,
                                  70,  60,  55,  50,  45,  40,  35,  30,  25,  20,  15,  10};
  if (options.single_efs > 0)
    efs_list = {options.single_efs};
  if (!options.efs_list.empty())
    efs_list = options.efs_list;

  const benchmark::SearchQueries queries{query_vecs, d, nq, query_filters};
  if (options.compare_mode) {
    std::cout << "\nefs,recall_orig,qps_orig,disk_reads_orig,recall_ml,qps_ml,disk_reads_ml,"
              << "ml_block_reads,ml_cache_hit_rate\n";
    for (auto efs : efs_list) {
      index.InvalidateMultiLevelCache();
      auto orig = benchmark::RunSearchPass(index, /*use_ml=*/false, queries, efs, options.k, gt);
      auto ml = benchmark::RunSearchPass(index, /*use_ml=*/true, queries, efs, options.k, gt);
      std::cout << efs << "," << orig.recall << "," << orig.qps << "," << orig.disk_reads_avg << ","
                << ml.recall << "," << ml.qps << "," << ml.disk_reads_avg << ","
                << ml.ml_block_reads_avg << "," << ml.ml_cache_hit_rate << std::endl;
    }
  } else if (options.ml_pipeline_mode) {
    std::cout
        << "\nefs,recall,qps,dist_comps,hops,layers_per_hop,io_submits,io_dedup_saved,"
        << "ml_block_reads,ml_dynamic_hit_rate,ml_static_hit_rate,ml_overall_hit_rate,"
        << "rerank_io_submits,ml_static_hits,unique_recall,post_rerank_io,"
        << "bound_pruned,rerank_duplicate_ids,output_duplicate_ids,rerank_candidates,rerank_reserve_candidates\n";
    for (auto efs : efs_list) {
      std::cout << "# final_case: ef=" << efs << std::endl;
      for (auto &ctx : search_contexts) {
        ctx->rerank_float_refined = 0;
        ctx->rerank_float_reused = 0;
        ctx->rerank_float_recomputed = 0;
      }
      const auto *search_ctx = search_contexts.front().get();
      const size_t graph_budget =
          search_ctx->GraphRerankBudgetForSearch(options.k, efs, index.rerank_times_upper_bound_);
      std::cout << "# graph_rerank_policy: efs=" << efs << " budget=" << graph_budget
                << " main_capacity=" << std::min(efs, graph_budget)
                << " reserve_capacity=" << graph_budget - std::min(efs, graph_budget)
                << " candidate_limit=" << graph_budget << " policy=" << options.rerank_policy
                << " (graph route only; unique candidates may be fewer)" << std::endl;
      index.InvalidateMultiLevelCache();
      auto batch =
          benchmark::RunPipelineBatch(index, queries, efs, options, search_contexts, profile);
      std::cout
          << "efs,recall,qps,dist_comps,hops,layers_per_hop,io_submits,io_dedup_saved,"
             "ml_block_reads,ml_dynamic_hit_rate,ml_static_hit_rate,ml_overall_hit_rate,"
             "rerank_io_submits,ml_static_hits,unique_recall,post_rerank_io,"
             "bound_pruned,rerank_duplicate_ids,output_duplicate_ids,rerank_candidates,rerank_reserve_candidates\n";
      benchmark::PrintPipelineReport(index, options, efs, gt, search_contexts, profile, batch);
      size_t refined = 0;
      for (const auto &ctx : search_contexts)
        refined += ctx->rerank_float_refined;
      std::cout << "# float_refine: refined_per_query=" << double(refined) / nq << std::endl;
      size_t reused = 0, recomputed = 0;
      for (const auto &ctx : search_contexts) {
        reused += ctx->rerank_float_reused;
        recomputed += ctx->rerank_float_recomputed;
      }
      std::cout << "# refinement_reuse: enabled=" << options.rerank_reuse_estimates
                << " reused_per_query=" << double(reused) / nq
                << " recomputed_per_query=" << double(recomputed) / nq << std::endl;
      size_t hits = 0, invalid = 0, duplicates = 0, predicate = 0, shorts = 0;
      for (size_t qi = 0; qi < nq; ++qi) {
        std::unordered_set<alliance::label_t> seen;
        shorts += batch.results[qi].size() != options.k;
        std::unordered_set<alliance::label_t> truth;
        if (!gt.empty())
          truth.insert(gt[qi].begin(), gt[qi].begin() + std::min(options.k, gt[qi].size()));
        for (auto id : batch.results[qi]) {
          const bool unique = seen.insert(id).second;
          duplicates += !unique;
          hits += unique && truth.count(id);
          if (!checkpoint_attributes.empty()) {
            if (id >= checkpoint_attributes.size()) {
              ++invalid;
              continue;
            }
            predicate += !query_filters[qi].Test(checkpoint_attributes[id]);
          }
        }
      }
      if (!options.quant_checkpoint.empty())
        std::cout << "# final_validation: {\"invalid\":" << invalid
                  << ",\"duplicates\":" << duplicates << ",\"predicate\":" << predicate
                  << ",\"short\":" << shorts << "}" << std::endl;
      if (invalid || duplicates || predicate)
        throw std::runtime_error("search output validation failed");
      const double measured_recall = gt.empty() ? 0.0 : double(hits) / (nq * options.k);
      std::cout << "# measured_recall: " << measured_recall << std::endl;
      if (options.recall_stop && measured_recall >= options.recall_stop)
        break;
      std::cout << "# entry_stats: samples=" << index.metric_ml_entry_samples_.load()
                << " estimates=" << index.metric_ml_entry_estimates_.load()
                << " admitted=" << index.metric_ml_entry_admitted_.load() << std::endl;
    }
  } else {
    std::cout << "\nefs,recall,qps,dist_comps,hops,disk_reads\n";
    for (auto efs : efs_list) {
      const auto stats =
          benchmark::RunLegacyPass(index, queries, efs, options, search_contexts, gt);
      std::cout << efs << "," << stats.recall << "," << stats.qps << "," << stats.dist_comps_avg
                << "," << stats.hops_avg << "," << stats.disk_reads_avg << std::endl;
    }
  }

  if (!options.quant_checkpoint.empty()) {
    const auto hash = benchmark::QuantizationFingerprint(index);
    if (hash != checkpoint_hash)
      throw std::runtime_error("search modified frozen quantization");
    std::cout << "# frozen_quantization_end: " << hash << std::endl;
  }
  std::cout << "search done" << std::endl;
  delete[] query_vecs;
  return 0;
}
