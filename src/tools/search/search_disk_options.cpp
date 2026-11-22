#include "search/search_disk_options.hpp"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace benchmark {
struct SearchOptionState {
  bool has_ml_cache_total_bytes = false;
  bool navigation_control_set = false;
  bool navigation_file_set = false, entry_sampling_set = false;
  bool has_rerank_policy = false, no_rerank_error_bound = false;
  bool has_candidate_factor = false, has_prescreen = false;
  bool has_rerank_factor = false, graph_rerank_control_set = false;
};
namespace {
size_t ParseNonnegativeSize(const char *value) {
  const std::string text(value);
  if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
    throw std::invalid_argument("size arguments must be nonnegative integers");
  const auto parsed = std::stoull(text);
  if (parsed > std::numeric_limits<size_t>::max())
    throw std::out_of_range("size argument");
  return static_cast<size_t>(parsed);
}
bool ParseCompatibilityOption(int argc, char **argv, int &i) {
  const std::string key = argv[i];
  const auto next = [&]() -> const char * {
    if (i + 1 >= argc)
      throw std::invalid_argument("missing value for " + key);
    return argv[++i];
  };
  if (key == "--skip_extra_level_io") {
    (void)std::stoi(next());
    return true;
  }
  if (key == "--no_inline_rerank")
    return true;
  if (key == "--inline_rerank") {
    if (std::string(next()) != "off")
      throw std::invalid_argument(
          "inline rerank was retired; only legacy --inline_rerank off is accepted");
    return true;
  }
  if (key == "--collect_landing" || key == "--train_efs")
    throw std::invalid_argument(
        "query-based landing collection was retired; use alliance_collect_landing: " + key);
  for (const char *retired :
       {"--memory", "--enum_thresh", "--export_threads", "--exact", "--basevec",
        "--rerank_interval_max", "--inline_rerank_efs_factor", "--inline_rerank_beam_width",
        "--inline_rerank_idle_only"})
    if (key == retired)
      throw std::invalid_argument("search option was retired: " + key);
  return false;
}
} // namespace

bool DiskSearchOptions::ParseAndValidate(int argc, char **argv) {
  SearchOptionState state;
  for (int i = 1; i < argc; ++i) {
    if (ParseCompatibilityOption(argc, argv, i))
      continue;
    const auto next_value = [&]() -> const char * {
      if (i + 1 >= argc)
        throw std::invalid_argument("missing value for " + std::string(argv[i]));
      return argv[++i];
    };
    if (strcmp(argv[i], "--memory_budget_bytes") == 0) {
      process_memory_budget_bytes = ParseNonnegativeSize(next_value());
    } else if (strcmp(argv[i], "--packed_rabitq") == 0) {
      const auto v = ParseNonnegativeSize(next_value());
      if (v > 1)
        throw std::invalid_argument("packed_rabitq must be0/1");
      packed_rabitq = v;
    } else if (strcmp(argv[i], "--navigation_file") == 0) {
      navigation_file = next_value();
      state.navigation_control_set = true;
      state.navigation_file_set = true;
    } else if (strcmp(argv[i], "--navigation_ef") == 0) {
      navigation_ef = ParseNonnegativeSize(next_value());
      state.navigation_control_set = true;
    } else if (strcmp(argv[i], "--navigation_budget") == 0) {
      navigation_budget = ParseNonnegativeSize(next_value());
      state.navigation_control_set = true;
    } else if (strcmp(argv[i], "--navigation_count") == 0) {
      navigation_count = ParseNonnegativeSize(next_value());
      state.navigation_control_set = true;
    } else if (strcmp(argv[i], "--entry_samples") == 0) {
      entry_samples = ParseNonnegativeSize(next_value());
    } else if (strcmp(argv[i], "--entry_count") == 0) {
      entry_count = ParseNonnegativeSize(next_value());
    } else if (strcmp(argv[i], "--entry_sampling") == 0) {
      entry_sampling = next_value();
      state.entry_sampling_set = true;
      if (entry_sampling != "navigation" && entry_sampling != "global" &&
          entry_sampling != "range" && entry_sampling != "guided")
        throw std::invalid_argument("entry_sampling must be navigation, global, range or guided");
    } else if (strcmp(argv[i], "--top_layer_postfilter") == 0) {
      const auto v = ParseNonnegativeSize(next_value());
      if (v > 1)
        throw std::invalid_argument("top_layer_postfilter must be0/1");
      top_layer_postfilter = v;
    } else if (strcmp(argv[i], "--graph_file") == 0) {
      graph_file = next_value();
    } else if (strcmp(argv[i], "--meta_file") == 0) {
      meta_file = next_value();
    } else if (strcmp(argv[i], "--query_vec") == 0) {
      query_vec_path = next_value();
    } else if (strcmp(argv[i], "--query_rng") == 0) {
      query_rng_path = next_value();
    } else if (strcmp(argv[i], "--gt_file") == 0) {
      gt_file = next_value();
    } else if (strcmp(argv[i], "--k") == 0) {
      k = std::stoul(next_value());
    } else if (strcmp(argv[i], "--space") == 0) {
      space = next_value();
    } else if (strcmp(argv[i], "--pipeline") == 0) {
      use_pipeline = std::stoi(next_value());
    } else if (strcmp(argv[i], "--beam_width") == 0) {
      beam_width = std::stoul(next_value());
    } else if (strcmp(argv[i], "--ml_only") == 0) {
      ml_only = true;
    } else if (strcmp(argv[i], "--ml_graph_file") == 0) {
      ml_graph_file = next_value();
    } else if (strcmp(argv[i], "--ml_meta_file") == 0) {
      ml_meta_file = next_value();
    } else if (strcmp(argv[i], "--ml_cache_bytes") == 0) {
      ml_cache_bytes = std::stoull(next_value());
    } else if (strcmp(argv[i], "--ml_cache_total_bytes") == 0) {
      ml_cache_total_bytes = ParseNonnegativeSize(next_value());
      state.has_ml_cache_total_bytes = true;
    } else if (strcmp(argv[i], "--ml_cache_policy") == 0) {
      ml_cache_policy = next_value();
    } else if (strcmp(argv[i], "--ml_cache_auto") == 0) {
      ml_cache_auto = std::stoi(next_value());
    } else if (strcmp(argv[i], "--rerank_num") == 0) {
      rerank_num = ParseNonnegativeSize(next_value());
      state.graph_rerank_control_set = true;
    } else if (strcmp(argv[i], "--rerank_policy") == 0) {
      rerank_policy = next_value();
      state.has_rerank_policy = true;
      state.graph_rerank_control_set = true;
    } else if (strcmp(argv[i], "--rerank_factor") == 0) {
      rerank_factor = ParseNonnegativeSize(next_value());
      state.has_rerank_factor = true;
      state.graph_rerank_control_set = true;
    } else if (strcmp(argv[i], "--rerank_candidate_factor") == 0) {
      rerank_candidate_factor = ParseNonnegativeSize(next_value());
      state.has_candidate_factor = true;
      state.graph_rerank_control_set = true;
    } else if (strcmp(argv[i], "--rerank_bound_prescreen") == 0) {
      const auto v = ParseNonnegativeSize(next_value());
      if (v > 1)
        throw std::invalid_argument("rerank_bound_prescreen must be 0/1");
      rerank_bound_prescreen = v;
      state.has_prescreen = true;
    } else if (strcmp(argv[i], "--rerank_bound_rank_factor") == 0) {
      rerank_bound_rank_factor = ParseNonnegativeSize(next_value());
    } else if (strcmp(argv[i], "--rerank_warm_blocks") == 0) {
      rerank_warm_blocks = ParseNonnegativeSize(next_value());
    } else if (strcmp(argv[i], "--rerank_error_bound") == 0) {
      rerank_error_bound = true;
    } else if (strcmp(argv[i], "--no_rerank_error_bound") == 0) {
      state.no_rerank_error_bound = true;
    } else if (strcmp(argv[i], "--rerank_float_refine") == 0) {
      const auto v = ParseNonnegativeSize(next_value());
      if (v > 1)
        throw std::invalid_argument("rerank_float_refine must be 0/1");
      rerank_float_refine = v;
    } else if (strcmp(argv[i], "--rerank_reuse_estimates") == 0) {
      const auto v = ParseNonnegativeSize(next_value());
      if (v > 1)
        throw std::invalid_argument("rerank_reuse_estimates must be 0/1");
      rerank_reuse_estimates = v;
    } else if (strcmp(argv[i], "--quant_checkpoint") == 0) {
      quant_checkpoint = next_value();
    } else if (strcmp(argv[i], "--efs_list") == 0) {
      const std::string list = next_value();
      if (list.empty() || list.back() == ',')
        throw std::invalid_argument("empty ef list item");
      std::istringstream in(list);
      std::string item;
      while (std::getline(in, item, ',')) {
        const size_t ef = ParseNonnegativeSize(item.c_str());
        if (!ef || (!efs_list.empty() && ef <= efs_list.back()))
          throw std::invalid_argument("ef list must increase");
        efs_list.push_back(ef);
      }
    } else if (strcmp(argv[i], "--recall_stop") == 0) {
      recall_stop = std::stod(next_value());
    } else if (strcmp(argv[i], "--rerank_bound_alpha") == 0) {
      std::cerr
          << "ERROR: --rerank_bound_alpha was removed; the library model error is used without extra scaling. Remove this option.\n";
      return false;
    } else if (strcmp(argv[i], "--rerank_batch_blocks") == 0) {
      rerank_batch_blocks = ParseNonnegativeSize(next_value());
    } else if (strcmp(argv[i], "--efs") == 0) {
      single_efs = std::stoull(next_value());
    } else if (strcmp(argv[i], "--query_limit") == 0) {
      query_limit = std::stoull(next_value());
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
    } else if (strcmp(argv[i], "--rabitq_centroids") == 0) {
      rabitq_centroids = next_value();
    } else if (strcmp(argv[i], "--rabitq_cids") == 0) {
      rabitq_cids = next_value();
    } else if (strcmp(argv[i], "--rabitq_bits") == 0) {
      rabitq_bits = std::stoul(next_value());
    } else if (strcmp(argv[i], "--per_query_cache") == 0) {
      per_query_cache = true;
    } else {
      throw std::runtime_error("unknown argument: " + std::string(argv[i]));
    }
  }

  return NormalizeAndValidate(state);
}

bool DiskSearchOptions::NormalizeAndValidate(const SearchOptionState &state) {
  if ((!ml_only && graph_file.empty()) || meta_file.empty() || query_vec_path.empty() ||
      query_rng_path.empty()) {
    std::cerr
        << "Usage: alliance_search_disk --graph_file <path> --meta_file <path>\n"
        << "  --query_vec <path> --query_rng <path>\n"
        << "  [--gt_file <path>] [--k 10] [--space l2]\n"
        << "  [--pipeline 0] [--beam_width 8]\n"
        << "  [--entry_sampling navigation|guided|global|range] [--entry_samples 256] [--entry_count 4]\n"
        << "    Default navigation for ML graph/auto; other routes retain guided.\n"
        << "    Explicit guided/global/range selects rank entries; samples=0 disables supplements.\n"
        << "  [--navigation_file <path>] [--navigation_ef 64] [--navigation_budget 256] [--navigation_count 16]\n"
        << "    RAM graph defaults to navigation.bin beside graph_file (else ml_graph_file).\n"
        << "    Requires ML graph/auto; missing navigation is an error, without fallback.\n"
        << "  [--threads 1] [--rerank_beam_width 8] [--io_budget_total 0]\n"
        << "    0=independent beams; otherwise split total outstanding requests across workers.\n"
        << "  [--efs <single ef>] [--query_limit 0]  (0=all queries)\n"
        << "  [--filter_mode legacy|graph|prefilter|auto]  (default legacy; ML pipeline only)\n"
        << "  [--prefilter_threshold 8192]  (auto: inclusive range cardinality)\n"
        << "  [--prefilter_rerank 0]  (0=max(k,efs); otherwise >=k; independent of graph budget)\n"
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
        << "  [--rerank_batch_blocks 32] (max requests/submit; rerank beam caps in-flight IO)\n"
        << "  [--quant_checkpoint <path>] [--efs_list 40,80,160] [--recall_stop 0]\n"
        << "  [--ml_graph_file <path> --ml_meta_file <path>]\n"
        << "  [--ml_only] complete ML neighbors/vectors without opening raw graph/rawvec files;\n"
        << "    requires --pipeline 1 and explicit graph/auto/prefilter, forbids compare/requantization.\n"
        << "    Without --pipeline: compare mode (orig vs ml sync search, side by side).\n"
        << "    With --pipeline 1: multi-level pipeline mode -- uses\n"
        << "      searchKNNMultiLevelPipeline (async IO, block dedup, cached-level\n"
        << "      expansion). Outputs: recall,qps,dist_comps,hops,layers_per_hop,\n"
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
  compare_mode = has_ml && !use_pipeline;
  const bool navigation_route =
      ml_pipeline_mode && (filter_mode == "graph" || filter_mode == "auto");
  // Keep non-graph routes compatible unless navigation was explicitly requested.
  if (!state.entry_sampling_set && !state.navigation_control_set && !navigation_route)
    entry_sampling = "guided";
  if (state.navigation_file_set && navigation_file.empty())
    throw std::invalid_argument("--navigation_file must be nonempty");
  if (entry_sampling == "navigation") {
    if (!navigation_route)
      throw std::invalid_argument("RAM navigation requires ML pipeline and graph/auto routing");
    if (navigation_file.empty()) {
      const std::filesystem::path source = graph_file.empty() ? ml_graph_file : graph_file;
      navigation_file = (source.parent_path() / "navigation.bin").string();
    }
  }
  if (state.navigation_control_set && navigation_file.empty())
    throw std::invalid_argument("navigation controls require a nonempty --navigation_file");
  if (!navigation_file.empty()) {
    if (!navigation_route)
      throw std::invalid_argument("RAM navigation requires ML pipeline and graph/auto routing");
    if (!navigation_count || navigation_count > navigation_ef || navigation_count > navigation_budget ||
        navigation_ef > 65536 || navigation_budget > 65536)
      throw std::invalid_argument("navigation requires 0 < count <= min(ef, budget), ef/budget <= 65536");
    // Loading a navigation graph selects an exclusive entry strategy. Disabling
    // supplemental entries here also makes the effective policy explicit in logs.
    entry_samples = entry_count = 0;
    entry_sampling = "range";
  }
  if (ml_only &&
      (!ml_pipeline_mode || compare_mode || filter_mode == "legacy" ||
       (quant_checkpoint.empty() && (!rabitq_centroids.empty() || !rabitq_cids.empty())))) {
    std::cerr
        << "ERROR: --ml_only requires complete ML pipeline, explicit graph/auto/prefilter routing, "
           "and metadata or checkpoint quantization; raw/compare/requantization unavailable\n";
    return false;
  }
  if (search_threads == 0 ||
      search_threads > static_cast<size_t>(std::numeric_limits<int>::max()) ||
      rerank_beam_width == 0 || rerank_beam_width > 256) {
    std::cerr << "ERROR: threads must be positive and rerank_beam_width must be in [1,256]\n";
    return false;
  }
  if (search_threads > 1 && !ml_pipeline_mode) {
    std::cerr << "ERROR: --threads > 1 requires multi-level pipeline mode\n";
    return false;
  }
  if (state.has_ml_cache_total_bytes)
    ml_cache_bytes = ml_cache_total_bytes / search_threads;
  if (filter_mode == "legacy")
    selected_filter_mode = FilterMode::Legacy;
  else if (filter_mode == "graph")
    selected_filter_mode = FilterMode::Graph;
  else if (filter_mode == "prefilter")
    selected_filter_mode = FilterMode::Prefilter;
  else if (filter_mode == "auto")
    selected_filter_mode = FilterMode::Auto;
  else {
    std::cerr << "ERROR: --filter_mode must be legacy, graph, prefilter, or auto\n";
    return false;
  }
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
  if (k == 0 || beam_width == 0 || beam_width > 64 || rerank_batch_blocks == 0) {
    std::cerr << "ERROR: positive k/widths required; beam <= 64\n";
    return false;
  }
  if (rerank_num != 0 && rerank_num < k) {
    std::cerr << "ERROR: rerank_num must be 0 or >=k\n";
    return false;
  }
  if (!std::isfinite(recall_stop) || recall_stop < 0 || recall_stop > 1 ||
      (recall_stop && gt_file.empty()) || (single_efs && !efs_list.empty()) ||
      ((!efs_list.empty() || recall_stop || !quant_checkpoint.empty()) && !ml_pipeline_mode)) {
    std::cerr
        << "ERROR: sweep/checkpoint requires ML pipeline, exclusive ef controls, and recall stop in [0,1] with GT\n";
    return false;
  }
  if (!state.has_rerank_policy &&
      (!ml_pipeline_mode || rerank_num || state.has_rerank_factor || state.no_rerank_error_bound))
    rerank_policy = "fixed";
  if (state.no_rerank_error_bound && (rerank_error_bound || rerank_policy == "error_bound")) {
    std::cerr << "ERROR: conflicting error-bound controls\n";
    return false;
  }
  if (rerank_policy == "error_bound") {
    if (rerank_num || state.has_rerank_factor) {
      std::cerr << "ERROR: error_bound policy cannot combine with fixed rerank budgets\n";
      return false;
    }
    rerank_error_bound = true;
    if (!state.has_candidate_factor)
      rerank_candidate_factor = 2;
    if (!rerank_candidate_factor) {
      std::cerr << "ERROR: error_bound policy requires a positive candidate factor\n";
      return false;
    }
    if (!state.has_prescreen)
      rerank_bound_prescreen = true;
    selected_rerank_policy = GraphRerankPolicy::Fixed;
  } else if (rerank_policy == "fixed")
    selected_rerank_policy = GraphRerankPolicy::Fixed;
  else if (rerank_policy == "step")
    selected_rerank_policy = GraphRerankPolicy::Step;
  else if (rerank_policy == "linear")
    selected_rerank_policy = GraphRerankPolicy::Linear;
  else {
    std::cerr << "ERROR: rerank_policy must be error_bound, fixed, step or linear\n";
    return false;
  }
  if (selected_rerank_policy != GraphRerankPolicy::Fixed &&
      (state.has_rerank_factor || rerank_num != 0)) {
    std::cerr << "ERROR: dynamic rerank cannot combine with factor or nonzero rerank_num\n";
    return false;
  }
  if (state.has_rerank_factor && (rerank_factor == 0 || rerank_num != 0)) {
    std::cerr
        << "ERROR: rerank_factor must be positive and cannot combine with nonzero rerank_num\n";
    return false;
  }
  if ((state.graph_rerank_control_set || rerank_error_bound) && !ml_pipeline_mode) {
    std::cerr << "ERROR: rerank controls require multi-level pipeline mode\n";
    return false;
  }
  if (rerank_error_bound && space != "l2" && space != "ip") {
    std::cerr << "ERROR: --rerank_error_bound requires --space l2 or ip\n";
    return false;
  }
  if ((rerank_candidate_factor && (rerank_num || state.has_rerank_factor ||
                                   (rerank_policy != "fixed" && rerank_policy != "error_bound"))) ||
      (rerank_bound_prescreen && !rerank_error_bound) || !rerank_bound_rank_factor ||
      rerank_warm_blocks > 256) {
    std::cerr
        << "ERROR: conflicting candidate budget, prescreen without bounds, or invalid bound budget\n";
    return false;
  }
  if (io_budget_total && (!ml_pipeline_mode || filter_mode == "legacy")) {
    std::cerr
        << "ERROR: --io_budget_total requires ML pipeline and filter_mode graph/prefilter/auto; legacy raw-vector fallback is not bounded\n";
    return false;
  }
  if (io_budget_total && io_budget_total / search_threads < 1u) {
    std::cerr << "ERROR: io_budget_total needs at least one request per worker\n";
    return false;
  }
  return true;
}
} // namespace benchmark
