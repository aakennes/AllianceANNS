#include "search/search_disk_options.hpp"
#include "search/search_preparation.hpp"
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>

bool Parse(std::vector<std::string> tail, benchmark::DiskSearchOptions &options,
           std::string *error = nullptr) {
  std::vector<std::string> args = {"test",   "--graph_file",    "unused", "--meta_file",
                                   "unused", "--query_vec",     "unused", "--query_rng",
                                   "unused", "--ml_graph_file", "unused", "--ml_meta_file",
                                   "unused", "--pipeline",      "1"};
  args.insert(args.end(), tail.begin(), tail.end());
  std::vector<char *> argv;
  for (auto &arg : args)
    argv.push_back(arg.data());
  argv.push_back(nullptr);
  try {
    return options.ParseAndValidate(args.size(), argv.data());
  } catch (const std::exception &e) {
    if (error)
      *error = e.what();
    return false;
  }
}
void Check(bool ok) {
  if (!ok)
    throw std::runtime_error("search options CLI contract failed");
}
int main() {
  // The value default selects RAM navigation; parsing keeps legacy/non-graph
  // routes compatible and resolves graph/auto paths without a file-existence
  // fallback. Loading the resolved path is responsible for reporting failures.
  Check(benchmark::DiskSearchConfig{}.entry_sampling == "navigation");
  {
    benchmark::DiskSearchOptions defaults;
    Check(Parse({}, defaults));
    Check(defaults.rerank_policy == "error_bound" && defaults.rerank_error_bound &&
          defaults.rerank_float_refine && defaults.rerank_candidate_factor == 2 &&
          defaults.rerank_bound_prescreen && defaults.rerank_beam_width == 8);

    Check(defaults.entry_samples == 256 && defaults.entry_count == 4 &&
          defaults.entry_sampling == "guided" && defaults.navigation_file.empty());

    benchmark::DiskSearchOptions explicit_guided;
    Check(Parse({"--entry_sampling", "guided"}, explicit_guided));
    Check(explicit_guided.entry_samples == 256);
    benchmark::DiskSearchOptions local;
    Check(Parse({"--entry_sampling", "range", "--entry_samples", "256", "--entry_count", "4"},
                local));
    Check(local.entry_sampling == "range" && local.entry_samples == 256);
    for (auto args : std::vector<std::vector<std::string>>{{"--entry_samples", "-1"},
                                                           {"--entry_count", "1.5"},
                                                           {"--entry_sampling", "id"},
                                                           {"--entry_samples"},
                                                           {"--entry_sampling"}}) {
      benchmark::DiskSearchOptions bad;
      Check(!Parse(args, bad));
    }
    benchmark::DiskSearchOptions off;
    Check(Parse({"--entry_samples", "0"}, off));
  }
  for (const auto *route : {"graph", "auto"}) {
    benchmark::DiskSearchOptions defaults;
    Check(Parse({"--filter_mode", route, "--graph_file", "indexes/raw/graph.bin",
                 "--ml_graph_file", "indexes/ml/graph.ml.bin"}, defaults));
    Check(std::filesystem::path(defaults.navigation_file) ==
          std::filesystem::path("indexes/raw/navigation.bin"));
    Check(defaults.entry_sampling == "range" && defaults.entry_samples == 0 &&
          defaults.entry_count == 0);

    benchmark::DiskSearchOptions ml_only;
    Check(Parse({"--filter_mode", route, "--ml_only", "--graph_file", "",
                 "--ml_graph_file", "indexes/ml/graph.ml.bin"}, ml_only));
    Check(std::filesystem::path(ml_only.navigation_file) ==
          std::filesystem::path("indexes/ml/navigation.bin"));

    benchmark::DiskSearchOptions explicit_navigation;
    Check(Parse({"--filter_mode", route, "--entry_sampling", "navigation"}, explicit_navigation));
    Check(explicit_navigation.navigation_file == "navigation.bin" &&
          explicit_navigation.entry_samples == 0 && explicit_navigation.entry_count == 0);

    for (const auto *mode : {"guided", "global", "range"}) {
      benchmark::DiskSearchOptions explicit_rank;
      Check(Parse({"--filter_mode", route, "--entry_sampling", mode}, explicit_rank));
      Check(explicit_rank.entry_sampling == mode && explicit_rank.navigation_file.empty() &&
            explicit_rank.entry_samples == 256 && explicit_rank.entry_count == 4);
    }
    benchmark::DiskSearchOptions default_controls;
    Check(Parse({"--filter_mode", route, "--navigation_ef", "32", "--navigation_budget", "80",
                 "--navigation_count", "8"}, default_controls));
    Check(default_controls.navigation_file == "navigation.bin" && default_controls.navigation_ef == 32 &&
          default_controls.navigation_budget == 80 && default_controls.navigation_count == 8);
  }
  for (const auto &arguments : std::vector<std::vector<std::string>>{
           {"--filter_mode", "prefilter"}, {"--filter_mode", "legacy"},
           {"--pipeline", "0"},
           {"--ml_graph_file", "", "--ml_meta_file", "", "--pipeline", "0"}}) {
    benchmark::DiskSearchOptions compatible;
    Check(Parse(arguments, compatible));
    Check(compatible.entry_sampling == "guided" && compatible.navigation_file.empty() &&
          compatible.entry_samples == 256 && compatible.entry_count == 4);
    auto explicit_navigation = arguments;
    explicit_navigation.insert(explicit_navigation.end(), {"--entry_sampling", "navigation"});
    benchmark::DiskSearchOptions invalid;
    Check(!Parse(explicit_navigation, invalid));
  }
  // The ordinary and formal executable call this same preparation helper.
  // Non-guided modes must not allocate the full N-entry directory, even when
  // the default sample/count options remain nonzero.
  struct Entries {
    size_t preparations = 0;
    void PrepareGuidedEntries() {
      ++preparations;
    }
  };
  for (const auto &mode : {"global", "range", "guided"}) {
    benchmark::DiskSearchOptions options;
    Check(Parse({"--entry_sampling", mode}, options));
    Entries index;
    benchmark::PrepareGuidedEntriesForSearch(index, options);
    Check(index.preparations == (std::string(mode) == "guided" ? 1u : 0u));
    options.entry_count = 0;
    benchmark::PrepareGuidedEntriesForSearch(index, options);
    Check(index.preparations == (std::string(mode) == "guided" ? 1u : 0u));
  }
  {
    benchmark::DiskSearchOptions navigation;
    Check(Parse({"--filter_mode", "graph", "--navigation_file", "fixture.nav"}, navigation));
    Check(navigation.navigation_ef == 64 && navigation.navigation_budget == 256 &&
          navigation.navigation_count == 16 && navigation.entry_samples == 0 &&
          navigation.entry_count == 0);
    Entries index;
    benchmark::PrepareGuidedEntriesForSearch(index, navigation);
    Check(index.preparations == 0);
    alliance::DiskSearchPolicy policy;
    benchmark::ConfigureSearchPolicy(policy, navigation, {0, 8, 8, 8});
    Check(policy.navigation_ef == 64 && policy.navigation_budget == 256 &&
          policy.navigation_count == 16 && policy.entry_samples == 0 && policy.entry_count == 0);
    benchmark::DiskSearchOptions explicit_navigation;
    Check(Parse({"--filter_mode", "graph", "--entry_sampling", "range",
                 "--navigation_file", "fixture.nav", "--navigation_ef", "32",
                 "--navigation_budget", "80", "--navigation_count", "8"}, explicit_navigation));
    Check(explicit_navigation.navigation_ef == 32 && explicit_navigation.navigation_budget == 80 &&
          explicit_navigation.navigation_count == 8 && explicit_navigation.navigation_file == "fixture.nav");
    benchmark::DiskSearchOptions independent_budget;
    Check(Parse({"--filter_mode", "graph", "--navigation_file", "fixture.nav", "--navigation_ef", "128",
                 "--navigation_budget", "64", "--navigation_count", "16"}, independent_budget));
    Check(independent_budget.navigation_ef == 128 && independent_budget.navigation_budget == 64);
    benchmark::DiskSearchOptions maximum_budget;
    Check(Parse({"--filter_mode", "graph", "--navigation_file", "fixture.nav", "--navigation_ef", "65536",
                 "--navigation_budget", "65536", "--navigation_count", "16"}, maximum_budget));
    for (auto arguments : std::vector<std::vector<std::string>>{
             {"--entry_sampling", "guided", "--navigation_ef", "32"},
             {"--entry_sampling", "range", "--navigation_count", "4"},
             {"--entry_sampling", "global", "--navigation_budget", "256"},
             {"--navigation_file", ""},
             {"--navigation_file", "fixture.nav", "--navigation_ef", "-1"},
             {"--navigation_file", "fixture.nav", "--navigation_ef", "1.5"},
             {"--navigation_file", "fixture.nav", "--navigation_count", "0"},
             {"--navigation_file", "fixture.nav", "--navigation_count", "65"},
             {"--navigation_file", "fixture.nav", "--navigation_budget", "15"},
             {"--navigation_file", "fixture.nav", "--navigation_ef", "65537"},
             {"--navigation_file", "fixture.nav", "--navigation_budget", "65537"},
             {"--navigation_file", "fixture.nav", "--pipeline", "0"},
             {"--navigation_file", "fixture.nav", "--filter_mode", "prefilter"},
             {"--navigation_file", "fixture.nav", "--filter_mode", "legacy"}}) {
      benchmark::DiskSearchOptions invalid;
      arguments.insert(arguments.begin(), {"--filter_mode", "graph"});
      Check(!Parse(arguments, invalid));
    }
  }
  // Both drivers map normalized values identically without touching the I/O
  // resource's constructed queue depth. Distinct widths exercise budget clamps.
  {
    benchmark::DiskSearchConfig config;
    config.top_layer_postfilter = false;
    config.entry_samples = 17;
    config.entry_count = 3;
    config.entry_sampling = "range";
    config.rerank_num = 41;
    config.rerank_factor = 7;
    config.selected_rerank_policy = benchmark::DiskSearchConfig::GraphRerankPolicy::Linear;
    config.rerank_error_bound = true;
    config.rerank_float_refine = false;
    config.rerank_reuse_estimates = true;
    config.rerank_candidate_factor = 9;
    config.rerank_bound_prescreen = true;
    config.rerank_bound_rank_factor = 5;
    config.rerank_warm_blocks = 13;
    config.rerank_batch_blocks = 31;
    config.selected_filter_mode = benchmark::DiskSearchConfig::FilterMode::Auto;
    config.prefilter_threshold = 73;
    config.prefilter_rerank = 29;
    alliance::DiskSearchPolicy policy;
    policy.queue_depth = 97;
    benchmark::ConfigureSearchPolicy(policy, config, {0, 4, 11, 15});
    Check(!policy.top_layer_postfilter && policy.entry_samples == 17 && policy.entry_count == 3 &&
          policy.entry_sampling == alliance::DiskSearchPolicy::EntrySampling::Range);
    Check(policy.edge_beam_width == 4 && policy.vec_beam_width == 4 &&
          policy.rerank_beam_width == 11 && policy.rerank_batch_blocks == 11 &&
          policy.queue_depth == 97);
    Check(policy.rerank_num == 41 && policy.rerank_factor == 7 &&
          policy.rerank_policy == alliance::DiskSearchPolicy::GraphRerankPolicy::Linear &&
          policy.rerank_error_bound && !policy.rerank_float_refine &&
          policy.rerank_reuse_estimates);
    Check(policy.rerank_candidate_factor == 9 && policy.rerank_bound_prescreen &&
          policy.rerank_bound_rank_factor == 5 && policy.rerank_warm_blocks == 13 &&
          policy.filter_mode == alliance::DiskSearchPolicy::FilterMode::Auto &&
          policy.prefilter_threshold == 73 && policy.prefilter_rerank == 29);
    config.rerank_batch_blocks = 2;
    benchmark::ConfigureSearchPolicy(policy, config, {0, 4, 11, 15});
    Check(policy.rerank_batch_blocks == 2);
  }
  for (const auto *flag : {"--memory_budget_bytes",
                           "--packed_rabitq",
                           "--entry_samples",
                           "--entry_count",
                           "--entry_sampling",
                           "--navigation_file",
                           "--navigation_ef",
                           "--navigation_budget",
                           "--navigation_count",
                           "--top_layer_postfilter",
                           "--graph_file",
                           "--meta_file",
                           "--query_vec",
                           "--query_rng",
                           "--gt_file",
                           "--k",
                           "--space",
                           "--pipeline",
                           "--beam_width",
                           "--ml_graph_file",
                           "--ml_meta_file",
                           "--ml_cache_bytes",
                           "--ml_cache_total_bytes",
                           "--ml_cache_policy",
                           "--ml_cache_auto",
                           "--skip_extra_level_io",
                           "--inline_rerank",
                           "--rerank_num",
                           "--rerank_policy",
                           "--rerank_factor",
                           "--rerank_candidate_factor",
                           "--rerank_bound_prescreen",
                           "--rerank_bound_rank_factor",
                           "--rerank_warm_blocks",
                           "--rerank_float_refine",
                           "--rerank_reuse_estimates",
                           "--quant_checkpoint",
                           "--efs_list",
                           "--recall_stop",
                           "--rerank_batch_blocks",
                           "--efs",
                           "--query_limit",
                           "--filter_mode",
                           "--prefilter_threshold",
                           "--prefilter_rerank",
                           "--io_budget_total",
                           "--threads",
                           "--rerank_beam_width",
                           "--rabitq_centroids",
                           "--rabitq_cids",
                           "--rabitq_bits"}) {
    benchmark::DiskSearchOptions options;
    std::string error;
    Check(!Parse({flag}, options, &error));
    Check(error == "missing value for " + std::string(flag));
  }
  // Retired memory requests must fail during parsing, before any
  // graph or query path is opened. Rejection also applies without a value.
  for (const char *flag : {"--memory", "--enum_thresh", "--export_threads", "--collect_landing",
                           "--train_efs", "--exact", "--basevec"}) {
    for (const auto &args : std::vector<std::vector<std::string>>{{flag}, {flag, "1"}}) {
      benchmark::DiskSearchOptions retired;
      std::string error;
      Check(!Parse(args, retired, &error));
      Check(error.find("retired") != std::string::npos);
    }
  }
  for (auto args :
       std::vector<std::vector<std::string>>{{"--rerank_factor", "1"},
                                             {"--rerank_factor", "3", "--k", "25"},
                                             {"--k", "25", "--rerank_factor", "3"},
                                             {"--rerank_num", "10"},
                                             {"--rerank_num", "0", "--rerank_factor", "3"}}) {
    benchmark::DiskSearchOptions options;
    Check(Parse(args, options));
  }
  for (auto args :
       std::vector<std::vector<std::string>>{{"--rerank_factor", "0"},
                                             {"--rerank_factor", "-1"},
                                             {"--rerank_factor", "1.5"},
                                             {"--rerank_factor", "nan"},
                                             {"--rerank_factor", "3", "--rerank_num", "30"},
                                             {"--rerank_num", "30", "--rerank_factor", "3"},
                                             {"--rerank_num", "9"},
                                             {"--k", "25", "--rerank_num", "24"},
                                             {"--rerank_factor"}}) {
    benchmark::DiskSearchOptions options;
    Check(!Parse(args, options));
  }
  for (const auto &policy : {"step", "linear"}) {
    benchmark::DiskSearchOptions dynamic;
    Check(Parse({"--rerank_policy", policy, "--rerank_num", "0"}, dynamic));
    Check(dynamic.selected_rerank_policy != benchmark::DiskSearchOptions::GraphRerankPolicy::Fixed);
    Check(!dynamic.rerank_error_bound && dynamic.rerank_candidate_factor == 0);
    for (auto args :
         std::vector<std::vector<std::string>>{{"--rerank_policy", policy, "--rerank_factor", "2"},
                                               {"--rerank_factor", "2", "--rerank_policy", policy},
                                               {"--rerank_policy", policy, "--rerank_num", "20"},
                                               {"--rerank_num", "20", "--rerank_policy", policy}}) {
      benchmark::DiskSearchOptions conflict;
      Check(!Parse(args, conflict));
    }
  }
  benchmark::DiskSearchOptions invalid;
  Check(!Parse({"--rerank_policy", "other"}, invalid));
  benchmark::DiskSearchOptions fixed;
  Check(Parse({"--rerank_policy", "fixed", "--rerank_factor", "2"}, fixed));
  benchmark::DiskSearchOptions options;
  Check(Parse({"--rerank_factor", "3", "--k", "25"}, options));
  Check(options.rerank_factor == 3 && options.k == 25 && options.rerank_num == 0);
  for (auto args : std::vector<std::vector<std::string>>{{"--rerank_candidate_factor", "2",
                                                          "--rerank_error_bound",
                                                          "--rerank_bound_prescreen", "1"},
                                                         {"--rerank_error_bound", "--space", "ip"},
                                                         {"--rerank_candidate_factor", "1"}}) {
    benchmark::DiskSearchOptions b;
    Check(Parse(args, b));
  }
  benchmark::DiskSearchOptions independent;
  Check(Parse({"--beam_width", "8", "--rerank_beam_width", "16"}, independent));
  Check(independent.beam_width == 8 && independent.rerank_beam_width == 16);
  for (auto args : std::vector<std::vector<std::string>>{
           {"--inline_rerank", "off"}, {"--no_inline_rerank"}, {"--skip_extra_level_io", "1"}}) {
    benchmark::DiskSearchOptions compatible;
    Check(Parse(args, compatible));
    Check(compatible.rerank_error_bound && compatible.rerank_candidate_factor == 2);
  }
  for (auto args :
       std::vector<std::vector<std::string>>{{"--inline_rerank", "on"},
                                             {"--inline_rerank", "auto"},
                                             {"--inline_rerank", "on", "--inline_rerank", "off"},
                                             {"--inline_rerank_beam_width", "0"},
                                             {"--inline_rerank_idle_only", "0"},
                                             {"--inline_rerank_efs_factor", "2"},
                                             {"--rerank_interval_max", "8"}}) {
    benchmark::DiskSearchOptions retired;
    std::string error;
    Check(!Parse(args, retired, &error));
    Check(error.find("retired") != std::string::npos);
  }
  for (auto args : std::vector<std::vector<std::string>>{
           {"--rerank_candidate_factor", "1", "--rerank_factor", "2"},
           {"--rerank_candidate_factor", "1", "--rerank_policy", "step"},
           {"--rerank_policy", "step", "--rerank_bound_prescreen", "1"},
           {"--rerank_bound_rank_factor", "0"},
           {"--inline_rerank_idle_only", "2"},
           {"--inline_rerank_beam_width", "257"},
           {"--rerank_warm_blocks", "257"},
           {"--rerank_bound_alpha"}}) {
    benchmark::DiskSearchOptions b;
    Check(!Parse(args, b));
  }
  for (auto args : std::vector<std::vector<std::string>>{
           {"--rerank_policy", "error_bound", "--rerank_num", "20"},
           {"--rerank_num", "20", "--rerank_policy", "error_bound"},
           {"--rerank_sq_refine", "0"},
           {"--rerank_sq_refine", "1"},
           {"--rerank_candidate_factor", "0"},
           {"--rerank_float_refine", "2"},
           {"--rerank_bound_alpha", "1"},
           {"--rerank_bound_alpha", "2"},
           {"--rerank_bound_alpha", "0.9"},
           {"--rerank_bound_alpha", "nan"},
           {"--rerank_policy", "error_bound", "--no_rerank_error_bound"},
           {"--efs_list", "40,80,40"},
           {"--efs_list", "40,"},
           {"--efs_list", "40,80", "--efs", "40"}}) {
    benchmark::DiskSearchOptions bad;
    Check(!Parse(args, bad));
  }
  benchmark::DiskSearchOptions coarse;
  Check(Parse({"--rerank_float_refine", "0"}, coarse));
  Check(coarse.rerank_error_bound && !coarse.rerank_float_refine);
  benchmark::DiskSearchOptions disabled;
  Check(Parse({"--no_rerank_error_bound"}, disabled));
  Check(!disabled.rerank_error_bound && disabled.rerank_candidate_factor == 0);
  std::cout << "search options CLI contracts PASSED\n";
}
