#pragma once
#include "alliance/index/index_disk.hpp"
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>

namespace benchmark {

struct DiskSearchOptions {
  std::string graph_file, meta_file, query_vec_path, query_rng_path, gt_file, space = "l2";
  std::string ml_graph_file, ml_meta_file, ml_cache_policy = "fifo";
  size_t ml_cache_bytes = 32 * 1024 * 1024;
  size_t ml_cache_total_bytes = 0;
  bool has_ml_cache_total_bytes = false;
  size_t k = 10;
  int use_pipeline = 0;
  int use_memory = 0;
  size_t beam_width = 8;
  size_t enum_thresh = 2048;
  int export_threads = 1;
  size_t rerank_interval_max = 8;
  std::string collect_landing_path;   // if set: train landing histogram, dump, exit
  size_t train_efs = 200;             // single-efs pass for landing collection
  std::string rabitq_centroids, rabitq_cids;  // override search-side RaBitQ centroids
  size_t rabitq_bits = 1;   // re-encode bit width when requantizing (1/2/3/4/8)
  bool per_query_cache = false;
  bool packed_rabitq = true;
  bool top_layer_postfilter = true;
  size_t entry_samples = 256, entry_count = 4;
  std::string entry_sampling = "guided";
  std::string navigation_file;
  size_t navigation_ef=64, navigation_budget=256, navigation_count=16;
  int ml_cache_auto = 1;    // size the dynamic cache to one query's working set
  std::string inline_rerank_mode = "off";
  size_t inline_rerank_efs_factor = 2;
  size_t rerank_num = 0;
  size_t rerank_factor = 0;
  std::string rerank_policy = "error_bound";
  bool has_rerank_policy = false, no_rerank_error_bound = false;
  bool has_candidate_factor = false, has_prescreen = false;
  bool rerank_float_refine = true;
  bool rerank_reuse_estimates = false;
  using GraphRerankPolicy = alliance::DiskSearchContext::GraphRerankPolicy;
  GraphRerankPolicy selected_rerank_policy = GraphRerankPolicy::Fixed;
  bool has_rerank_factor = false;
  bool graph_rerank_control_set = false;
  bool rerank_error_bound = false;
  size_t rerank_batch_blocks = 32;
  size_t rerank_candidate_factor = 0;
  bool rerank_bound_prescreen = false;
  size_t rerank_bound_rank_factor = 1;
  size_t rerank_warm_blocks = 0;
  size_t inline_rerank_beam_width = 0;
  bool inline_rerank_idle_only = false;
  size_t single_efs = 0;
  size_t query_limit = 0;
  std::string quant_checkpoint;
  std::vector<size_t> efs_list;
  double recall_stop = 0;
  std::string filter_mode = "legacy";
  size_t prefilter_threshold = 8192;
  size_t prefilter_rerank = 0;
  size_t search_threads = 1;
  size_t io_budget_total = 0;
  size_t rerank_beam_width = 8;

  size_t process_memory_budget_bytes = 0;
  bool has_ml = false, ml_pipeline_mode = false, compare_mode = false;
  using FilterMode = alliance::DiskSearchContext::FilterMode;
  FilterMode selected_filter_mode = FilterMode::Legacy;

  bool ParseAndValidate(int argc, char **argv) {
    for (int i = 1; i < argc; ++i) {
      const auto next_value = [&]() -> const char * {
        if (i + 1 >= argc) throw std::invalid_argument("missing value for " + std::string(argv[i]));
        return argv[++i];
      };
      if (strcmp(argv[i], "--memory_budget_bytes") == 0) {
        process_memory_budget_bytes=ParseNonnegativeSize(next_value());
      } else if (strcmp(argv[i], "--packed_rabitq") == 0) {
        const auto v=ParseNonnegativeSize(next_value());if(v>1)throw std::invalid_argument("packed_rabitq must be0/1");packed_rabitq=v;
      } else if (strcmp(argv[i], "--navigation_file") == 0) {
        navigation_file=next_value();
      } else if (strcmp(argv[i], "--navigation_ef") == 0) {
        navigation_ef=ParseNonnegativeSize(next_value());
      } else if (strcmp(argv[i], "--navigation_budget") == 0) {
        navigation_budget=ParseNonnegativeSize(next_value());
      } else if (strcmp(argv[i], "--navigation_count") == 0) {
        navigation_count=ParseNonnegativeSize(next_value());
      } else if (strcmp(argv[i], "--entry_samples") == 0) {
        entry_samples=ParseNonnegativeSize(next_value());
      } else if (strcmp(argv[i], "--entry_count") == 0) {
        entry_count=ParseNonnegativeSize(next_value());
      } else if (strcmp(argv[i], "--entry_sampling") == 0) {
        entry_sampling=next_value();
        if(entry_sampling!="global" && entry_sampling!="range" && entry_sampling!="guided")
          throw std::invalid_argument("entry_sampling must be global, range or guided");
      } else if (strcmp(argv[i], "--top_layer_postfilter") == 0) {
        const auto v=ParseNonnegativeSize(next_value());if(v>1)throw std::invalid_argument("top_layer_postfilter must be0/1");top_layer_postfilter=v;
      } else if (strcmp(argv[i], "--graph_file") == 0) {
        graph_file = argv[++i];
      } else if (strcmp(argv[i], "--meta_file") == 0) {
        meta_file = argv[++i];
      } else if (strcmp(argv[i], "--query_vec") == 0) {
        query_vec_path = argv[++i];
      } else if (strcmp(argv[i], "--query_rng") == 0) {
        query_rng_path = argv[++i];
      } else if (strcmp(argv[i], "--gt_file") == 0) {
        gt_file = argv[++i];
      } else if (strcmp(argv[i], "--k") == 0) {
        k = std::stoul(argv[++i]);
      } else if (strcmp(argv[i], "--space") == 0) {
        space = argv[++i];
      } else if (strcmp(argv[i], "--pipeline") == 0) {
        use_pipeline = std::stoi(argv[++i]);
      } else if (strcmp(argv[i], "--beam_width") == 0) {
        beam_width = std::stoul(argv[++i]);
      } else if (strcmp(argv[i], "--exact") == 0 || strcmp(argv[i], "--basevec") == 0) {
        std::cerr << "ERROR: --exact/--basevec were retired; use --memory for the original "
                     "FP32 memory search on the exported disk graph, or alliance_search for a memory index.\n";
        return false;
      } else if (strcmp(argv[i], "--memory") == 0) {
        use_memory = 1;
      } else if (strcmp(argv[i], "--enum_thresh") == 0) {
        enum_thresh = std::stoul(argv[++i]);
      } else if (strcmp(argv[i], "--export_threads") == 0) {
        export_threads = std::stoi(argv[++i]);
      } else if (strcmp(argv[i], "--ml_graph_file") == 0) {
        ml_graph_file = argv[++i];
      } else if (strcmp(argv[i], "--ml_meta_file") == 0) {
        ml_meta_file = argv[++i];
      } else if (strcmp(argv[i], "--ml_cache_bytes") == 0) {
        ml_cache_bytes = std::stoull(argv[++i]);
      } else if (strcmp(argv[i], "--ml_cache_total_bytes") == 0) {
        ml_cache_total_bytes = ParseNonnegativeSize(next_value());
        has_ml_cache_total_bytes = true;
      } else if (strcmp(argv[i], "--ml_cache_policy") == 0) {
        ml_cache_policy = argv[++i];
      } else if (strcmp(argv[i], "--ml_cache_auto") == 0) {
        ml_cache_auto = std::stoi(next_value());
      } else if (strcmp(argv[i], "--skip_extra_level_io") == 0) {
        // Historical option: current traversal has no alternate level-miss
        // policy. Consume it for frozen command compatibility, without state.
        (void)std::stoi(next_value());
      } else if (strcmp(argv[i], "--rerank_interval_max") == 0) {
        rerank_interval_max = std::stoull(argv[++i]);
      } else if (strcmp(argv[i], "--inline_rerank") == 0) {
        inline_rerank_mode = next_value();
        graph_rerank_control_set = true;
      } else if (strcmp(argv[i], "--inline_rerank_efs_factor") == 0) {
        inline_rerank_efs_factor = ParseNonnegativeSize(next_value());
        graph_rerank_control_set = true;
      } else if (strcmp(argv[i], "--rerank_num") == 0) {
        rerank_num = ParseNonnegativeSize(next_value());
        graph_rerank_control_set = true;
      } else if (strcmp(argv[i], "--rerank_policy") == 0) {
        rerank_policy = next_value();
        has_rerank_policy = true;
        graph_rerank_control_set = true;
      } else if (strcmp(argv[i], "--rerank_factor") == 0) {
        rerank_factor = ParseNonnegativeSize(next_value());
        has_rerank_factor = true;
        graph_rerank_control_set = true;
      } else if (strcmp(argv[i], "--no_inline_rerank") == 0) {
        inline_rerank_mode = "off";
        graph_rerank_control_set = true;
      } else if (strcmp(argv[i], "--rerank_candidate_factor") == 0) {
        rerank_candidate_factor = ParseNonnegativeSize(next_value());
        has_candidate_factor = true;
        graph_rerank_control_set = true;
      } else if (strcmp(argv[i], "--rerank_bound_prescreen") == 0) {
        const auto v = ParseNonnegativeSize(next_value());
        if (v > 1) throw std::invalid_argument("rerank_bound_prescreen must be 0/1");
        rerank_bound_prescreen = v;
        has_prescreen = true;
      } else if (strcmp(argv[i], "--rerank_bound_rank_factor") == 0) {
        rerank_bound_rank_factor = ParseNonnegativeSize(next_value());
      } else if (strcmp(argv[i], "--rerank_warm_blocks") == 0) {
        rerank_warm_blocks = ParseNonnegativeSize(next_value());
      } else if (strcmp(argv[i], "--inline_rerank_beam_width") == 0) {
        inline_rerank_beam_width = ParseNonnegativeSize(next_value());
        graph_rerank_control_set = true;
      } else if (strcmp(argv[i], "--inline_rerank_idle_only") == 0) {
        const auto v = ParseNonnegativeSize(next_value());
        if (v > 1) throw std::invalid_argument("inline_rerank_idle_only must be 0/1");
        inline_rerank_idle_only = v;
        graph_rerank_control_set = true;
      } else if (strcmp(argv[i], "--rerank_error_bound") == 0) {
        rerank_error_bound = true;
      } else if (strcmp(argv[i], "--no_rerank_error_bound") == 0) {
        no_rerank_error_bound = true;
      } else if (strcmp(argv[i], "--rerank_float_refine") == 0) {
        const auto v = ParseNonnegativeSize(next_value());
        if (v > 1) throw std::invalid_argument("rerank_float_refine must be 0/1");
        rerank_float_refine = v;
      } else if (strcmp(argv[i], "--rerank_reuse_estimates") == 0) {
        const auto v=ParseNonnegativeSize(next_value());
        if(v>1)throw std::invalid_argument("rerank_reuse_estimates must be 0/1");
        rerank_reuse_estimates=v;
      } else if (strcmp(argv[i], "--quant_checkpoint") == 0) {
        quant_checkpoint = next_value();
      } else if (strcmp(argv[i], "--efs_list") == 0) {
        const std::string list = next_value();
        if (list.empty() || list.back() == ',') throw std::invalid_argument("empty ef list item");
        std::istringstream in(list); std::string item;
        while (std::getline(in, item, ',')) {
          const size_t ef = ParseNonnegativeSize(item.c_str());
          if (!ef || (!efs_list.empty() && ef <= efs_list.back())) throw std::invalid_argument("ef list must increase");
          efs_list.push_back(ef);
        }
      } else if (strcmp(argv[i], "--recall_stop") == 0) {
        recall_stop = std::stod(next_value());
      } else if (strcmp(argv[i], "--rerank_bound_alpha") == 0) {
        std::cerr << "ERROR: --rerank_bound_alpha was removed; the library model error is used without extra scaling. Remove this option.\n";
        return false;
      } else if (strcmp(argv[i], "--rerank_batch_blocks") == 0) {
        rerank_batch_blocks = ParseNonnegativeSize(next_value());
      } else if (strcmp(argv[i], "--efs") == 0) {
        single_efs = std::stoull(argv[++i]);
      } else if (strcmp(argv[i], "--query_limit") == 0) {
        query_limit = std::stoull(argv[++i]);
      } else if (strcmp(argv[i], "--filter_mode") == 0) {
        filter_mode = next_value();
      } else if (strcmp(argv[i], "--prefilter_threshold") == 0) {
        prefilter_threshold = ParseNonnegativeSize(next_value());
      } else if (strcmp(argv[i], "--prefilter_rerank") == 0) {
        prefilter_rerank = ParseNonnegativeSize(next_value());
      } else if (strcmp(argv[i], "--io_budget_total") == 0) {
        io_budget_total = ParseNonnegativeSize(next_value());
      } else if (strcmp(argv[i], "--threads") == 0) {
        search_threads = ParseNonnegativeSize(next_value());
      } else if (strcmp(argv[i], "--rerank_beam_width") == 0) {
        rerank_beam_width = ParseNonnegativeSize(next_value());
      } else if (strcmp(argv[i], "--collect_landing") == 0) {
        collect_landing_path = argv[++i];
      } else if (strcmp(argv[i], "--train_efs") == 0) {
        train_efs = std::stoul(argv[++i]);
      } else if (strcmp(argv[i], "--rabitq_centroids") == 0) {
        rabitq_centroids = argv[++i];
      } else if (strcmp(argv[i], "--rabitq_cids") == 0) {
        rabitq_cids = argv[++i];
      } else if (strcmp(argv[i], "--rabitq_bits") == 0) {
        rabitq_bits = std::stoul(argv[++i]);
      } else if (strcmp(argv[i], "--per_query_cache") == 0) {
        per_query_cache = true;
      } else {
        throw std::runtime_error("unknown argument: " + std::string(argv[i]));
      }
    }

    if (graph_file.empty() || meta_file.empty() || query_vec_path.empty() || query_rng_path.empty()) {
      std::cerr << "Usage: alliance_search_disk --graph_file <path> --meta_file <path>\n"
                << "  --query_vec <path> --query_rng <path>\n"
                << "  [--gt_file <path>] [--k 10] [--space l2]\n"
                << "  [--pipeline 0] [--beam_width 8]\n"
                << "  [--entry_sampling guided|global|range] [--entry_samples 256] [--entry_count 4]\n"
                << "    ML graph pipeline only; samples=0 disables supplemental rank entries.\n"
                << "  [--threads 1] [--rerank_beam_width 8] [--io_budget_total 0]\n"
                << "    0=independent beams; otherwise split total outstanding requests across workers.\n"
                << "  [--efs <single ef>] [--query_limit 0]  (0=all queries)\n"
                << "  [--filter_mode legacy|graph|prefilter|auto]  (default legacy; ML pipeline only)\n"
                << "  [--prefilter_threshold 8192]  (auto: inclusive range cardinality)\n"
                << "  [--prefilter_rerank 0]  (0=max(k,efs); otherwise >=k; independent of graph budget)\n"
                << "  [--inline_rerank auto|on|off] [--inline_rerank_efs_factor 2]\n"
                << "    Off by default; auto: inline only when efs<=factor*k; factor is a positive integer.\n"
                << "    --no_inline_rerank is an alias for --inline_rerank off.\n"
                << "  [--rerank_policy error_bound|fixed|step|linear] (default error_bound for ML graph route)\n"
                << "    step: 2*k at ef<=2*k, 3*k at ef<=8*k, else 4*k.\n"
                << "    linear: min(4*k, 2*k+ceil(max(0,ef-2*k)/7)).\n"
                << "    Dynamic policies cannot combine with factor or nonzero rerank_num.\n"
                << "  [--rerank_factor C] (positive integer: graph budget C*k, independent of efs)\n"
                << "  [--rerank_num 0] (graph: 0=automatic; explicit >=k; cannot combine with factor)\n"
                << "    Explicit fixed policy without overrides: 8*k. Default error_bound retains max(k,2*ef).\n"
                << "  [--rerank_candidate_factor 2] (nonzero: retain max(k,factor*ef); excludes fixed/step/linear overrides)\n"
                << "  [--rerank_error_bound] [--no_rerank_error_bound] [--rerank_float_refine 0|1] (default refine=1)\n"
                << "  [--rerank_reuse_estimates 0|1] (default 0: opt-in retained full-precision score reuse)\n"
                << "  [--rerank_bound_prescreen 0|1] (default 1 with error_bound policy) [--rerank_bound_rank_factor 1] [--rerank_warm_blocks 0]\n"
                << "    Scheme B: upper-bound rank factor*k prescreen, then exact top-k lower-bound checks.\n"
                << "  [--inline_rerank_beam_width 0] [--inline_rerank_idle_only 0|1]\n"
                << "    Inline width is independent of search/final rerank; 0 inherits final width.\n"
                << "    idle_only=0 (default): independent beams; 1: historical shared-beam ablation.\n"
                << "  [--rerank_batch_blocks 32] (max requests/submit; rerank beam caps in-flight IO)\n"
                << "  [--quant_checkpoint <path>] [--efs_list 40,80,160] [--recall_stop 0]\n"
                << "  [--memory --enum_thresh <n> --export_threads 1]\n"
                << "  [--ml_graph_file <path> --ml_meta_file <path>]\n"
                << "    Without --pipeline: compare mode (orig vs ml sync search, side by side).\n"
                << "    With --pipeline 1: multi-level pipeline mode -- uses\n"
                << "      searchKNNMultiLevelPipeline (async IO, block dedup, cross-level\n"
                << "      pause/resume). Outputs: recall,qps,dist_comps,hops,layers_per_hop,\n"
                << "      io_submits,io_dedup_saved,ml_block_reads,ml_dynamic_hit_rate,\n"
                << "      ml_static_hit_rate,ml_overall_hit_rate,rerank_io_submits,ml_static_hits.\n"
                << "  [--ml_cache_bytes 33554432] (per thread)\n"
                << "  [--ml_cache_total_bytes N] (split across threads) [--ml_cache_policy fifo|lru]\n"
                << "  [--ml_cache_auto 1]        with --per_query_cache, treat --ml_cache_bytes as an\n"
                << "                             upper bound and grow to the per-query working set\n"
                << "  [--skip_extra_level_io 0]  (deprecated; accepted for compatibility, no effect)\n"
                << "  [--rabitq_centroids <path> --rabitq_cids <path>]  re-quantize against a finer centroid set\n"
                << "  [--rabitq_bits 1]          re-encode bit width (1/2/3/4/8); needs --rabitq_centroids/--rabitq_cids\n";
      return false;
    }

    has_ml = !ml_graph_file.empty() || !ml_meta_file.empty();
    if (has_ml) {
      if (ml_graph_file.empty() || ml_meta_file.empty()) {
        std::cerr << "ERROR: --ml_graph_file and --ml_meta_file must be given together\n";
        return false;
      }
      if (ml_cache_policy != "fifo" && ml_cache_policy != "lru") {
        std::cerr << "ERROR: --ml_cache_policy must be fifo or lru\n";
        return false;
      }
    }
    ml_pipeline_mode = has_ml && use_pipeline;
    if(!navigation_file.empty()) {
      if(!ml_pipeline_mode || !navigation_ef || !navigation_count ||
         navigation_count>navigation_ef || navigation_budget<navigation_count ||
         navigation_ef>65536 || navigation_budget>65536) {
        std::cerr<<"ERROR: navigation requires ML pipeline, positive count<=ef and count<=budget (<=65536)\n";
        return false;
      }
      entry_samples=0;entry_count=0;entry_sampling="range";
    }
    compare_mode = has_ml && !use_pipeline && !use_memory;
    if (search_threads == 0 || search_threads > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        rerank_beam_width == 0 || rerank_beam_width > 256) {
      std::cerr << "ERROR: threads must be positive and rerank_beam_width must be in [1,256]\n";
      return false;
    }
    if (search_threads > 1 && !ml_pipeline_mode) {
      std::cerr << "ERROR: --threads > 1 requires multi-level pipeline mode\n";
      return false;
    }
    if (search_threads > 1 && !collect_landing_path.empty()) {
      std::cerr << "ERROR: --collect_landing requires --threads 1\n";
      return false;
    }
    if (has_ml_cache_total_bytes) ml_cache_bytes = ml_cache_total_bytes / search_threads;
    if (filter_mode == "legacy") selected_filter_mode = FilterMode::Legacy;
    else if (filter_mode == "graph") selected_filter_mode = FilterMode::Graph;
    else if (filter_mode == "prefilter") selected_filter_mode = FilterMode::Prefilter;
    else if (filter_mode == "auto") selected_filter_mode = FilterMode::Auto;
    else { std::cerr << "ERROR: --filter_mode must be legacy, graph, prefilter, or auto\n"; return false; }
    if (filter_mode != "legacy" && !ml_pipeline_mode) {
      std::cerr << "ERROR: --filter_mode requires multi-level pipeline mode\n";
      return false;
    }
    if (filter_mode == "prefilter" || filter_mode == "auto") {
      if ((prefilter_rerank != 0 && prefilter_rerank < k) ||
          (filter_mode == "auto" && prefilter_threshold == 0)) {
        std::cerr << "ERROR: prefilter requires R=0 or R>=k and a positive auto threshold\n";
        return false;
      }
    }
    if (k == 0 || beam_width == 0 || beam_width > 64 || rerank_interval_max == 0 ||
        rerank_batch_blocks == 0) {
      std::cerr << "ERROR: positive k/widths/interval required; beam <= 64\n";
      return false;
    }
    if (inline_rerank_mode != "auto" && inline_rerank_mode != "on" && inline_rerank_mode != "off") {
      std::cerr << "ERROR: --inline_rerank must be auto, on, or off\n";
      return false;
    }
    if (inline_rerank_efs_factor == 0 || (rerank_num != 0 && rerank_num < k)) {
      std::cerr << "ERROR: inline_rerank_efs_factor must be positive; rerank_num must be 0 or >=k\n";
      return false;
    }
    if (!std::isfinite(recall_stop) || recall_stop < 0 || recall_stop > 1 ||
        (recall_stop && gt_file.empty()) || (single_efs && !efs_list.empty()) ||
        ((!efs_list.empty() || recall_stop || !quant_checkpoint.empty()) && !ml_pipeline_mode)) {
      std::cerr << "ERROR: sweep/checkpoint requires ML pipeline, exclusive ef controls, and recall stop in [0,1] with GT\n";
      return false;
    }
    if (!has_rerank_policy && (!ml_pipeline_mode || rerank_num || has_rerank_factor || no_rerank_error_bound))
      rerank_policy = "fixed";
    if (no_rerank_error_bound && (rerank_error_bound || rerank_policy == "error_bound")) {
      std::cerr << "ERROR: conflicting error-bound controls\n"; return false;
    }
    if (rerank_policy == "error_bound") {
      if (rerank_num || has_rerank_factor) {
        std::cerr << "ERROR: error_bound policy cannot combine with fixed rerank budgets\n"; return false;
      }
      rerank_error_bound = true;
      if (!has_candidate_factor) rerank_candidate_factor = 2;
      if (!rerank_candidate_factor) {
        std::cerr << "ERROR: error_bound policy requires a positive candidate factor\n"; return false;
      }
      if (!has_prescreen) rerank_bound_prescreen = true;
      selected_rerank_policy = GraphRerankPolicy::Fixed;
    } else if (rerank_policy == "fixed") selected_rerank_policy = GraphRerankPolicy::Fixed;
    else if (rerank_policy == "step") selected_rerank_policy = GraphRerankPolicy::Step;
    else if (rerank_policy == "linear") selected_rerank_policy = GraphRerankPolicy::Linear;
    else { std::cerr << "ERROR: rerank_policy must be error_bound, fixed, step or linear\n"; return false; }
    if (selected_rerank_policy != GraphRerankPolicy::Fixed && (has_rerank_factor || rerank_num != 0)) {
      std::cerr << "ERROR: dynamic rerank cannot combine with factor or nonzero rerank_num\n";
      return false;
    }
    if (has_rerank_factor && (rerank_factor == 0 || rerank_num != 0)) {
      std::cerr << "ERROR: rerank_factor must be positive and cannot combine with nonzero rerank_num\n";
      return false;
    }
    if ((graph_rerank_control_set || rerank_error_bound) && !ml_pipeline_mode) {
      std::cerr << "ERROR: rerank controls require multi-level pipeline mode\n";
      return false;
    }
    if (rerank_error_bound && space != "l2" && space != "ip") {
      std::cerr << "ERROR: --rerank_error_bound requires --space l2 or ip\n";
      return false;
    }
    if ((rerank_candidate_factor && (rerank_num || has_rerank_factor || (rerank_policy != "fixed" && rerank_policy != "error_bound"))) ||
        (rerank_bound_prescreen && !rerank_error_bound) || !rerank_bound_rank_factor ||
        rerank_warm_blocks > 256 || inline_rerank_beam_width > 256) {
      std::cerr << "ERROR: conflicting candidate budget, prescreen without bounds, or invalid bound/inline width\n";
      return false;
    }
    if (has_ml && !ml_pipeline_mode && !compare_mode) {
      std::cerr << "ERROR: --ml_graph_file can combine with --pipeline (ml pipeline mode) "
                << "or standalone (compare mode), not with --memory\n";
      return false;
    }

    if (io_budget_total && (!ml_pipeline_mode || filter_mode == "legacy")) {
      std::cerr << "ERROR: --io_budget_total requires ML pipeline and filter_mode graph/prefilter/auto; legacy raw-vector fallback is not bounded\n";
      return false;
    }
    if (io_budget_total && io_budget_total / search_threads < (inline_rerank_mode == "off" ? 1u : 2u)) {
      std::cerr << "ERROR: io_budget_total needs at least one request per worker (two with inline on/auto)\n";
      return false;
    }
    return true;
  }

 private:
  static size_t ParseNonnegativeSize(const char *value) {
    const std::string text(value);
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
      throw std::invalid_argument("size arguments must be nonnegative integers");
    const auto parsed = std::stoull(text);
    if (parsed > std::numeric_limits<size_t>::max()) throw std::out_of_range("size argument");
    return static_cast<size_t>(parsed);
  }
};

}  // namespace benchmark
