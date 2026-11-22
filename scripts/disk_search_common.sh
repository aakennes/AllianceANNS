#!/usr/bin/env bash
# Compatibility helper for Arxiv/SIFT presets. Arguments are kept as arrays.
set -euo pipefail
if (($# == 0)) || [[ $1 != arxiv && $1 != sift1m ]]; then
  echo 'Usage: bash scripts/disk_search_common.sh {arxiv|sift1m} [options]' >&2
  exit 2
fi
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DIR=${ALLIANCE_BIN_DIR:-$ROOT_DIR/build/bin}
DATASET=$1; shift
if [[ $DATASET == arxiv ]]; then
  DATA_ROOT=${ALLIANCE_DATA_ROOT:-$HOME/data}/Arxiv
  INDEX_NAME=wp1; SPACE=ip; SEARCH_NC=256
  QUERY_NAME=test.fvecs; CENTROID_NAME=base.2M; SELECTIVITIES=mixed
else
  DATA_ROOT=${ALLIANCE_DATA_ROOT:-$HOME/data}/Sift1m
  INDEX_NAME=sift_wp1; SPACE=l2; SEARCH_NC=128
  QUERY_NAME=sift_query.fvecs; CENTROID_NAME=base.1M; SELECTIVITIES=1,4,7,10
fi
MODEL_DIR=$ROOT_DIR/build/_p3; OUT_DIR=$ROOT_DIR/build/_p3
GRAPH_FILE=; META_FILE=; QUERY_VEC=; QUERY_RNG=; GT_FILE=; TAG=
ML_GRAPH_FILE=; ML_META_FILE=; SEARCH_CENTROIDS=; SEARCH_CIDS=
K=25; ML_LEVELS=2,2; ML_CACHE_BYTES=47185920; ML_CACHE_POLICY=fifo
BEAM_WIDTH=8; SHUFFLE_METHOD=block_bnf; METHODS=; SHUFFLE_O=2
BLOCK_SIZE=4096; MAX_ITERATIONS=32
STATIC_CACHE_BUDGET=0; SHUFFLE_MEM_GB=0
STATIC_LANDING_FILE=
STATIC_CACHE_USE_LANDING=false; STATIC_MODE_EXPLICIT=false
STATIC_LANDING_QUERY_RNG=; STATIC_CACHE_DIR=; INPUTS_DIR=
RABITQ_BITS=1; FILTER_MODE=auto; PREFILTER_THRESHOLD=8192; PREFILTER_RERANK=0
ENTRY_SAMPLING=; NAVIGATION_FILE=; NAVIGATION_EF=64; NAVIGATION_BUDGET=256; NAVIGATION_COUNT=16
EFS=30,50,100,200,400; QUERY_LIMIT=0
RERANK_NUM=heuristic
RERANK_FACTOR=; RERANK_POLICY=error_bound; RERANK_POLICY_EXPLICIT=false; RERANK_FLOAT_REFINE=1
RERANK_BATCH_BLOCKS=32; CPU=auto; RERANK_BEAM_WIDTH=8
THREADS=1; IO_BUDGET_TOTAL=0; ML_CACHE_TOTAL_BYTES=; MEMORY_BUDGET_BYTES=0
PER_QUERY_CACHE=true; SKIP_SHUFFLE=false; SKIP_SEARCH=false; DRY_RUN=false
RERANK_ERROR_BOUND=false; SUMMARY=true; ML_CACHE_AUTO=1
die() { echo "ERROR: $*" >&2; exit 2; }
help() {
  cat <<'EOF'
Usage: bash scripts/disk_search_common.sh {arxiv|sift1m} [options]
Compatibility helper; the JSON workflow uses shuffle.sh/search.sh DATASET.
Defaults: k25, beam8, cache45MiB FIFO cleared per query, bits1, inline OFF,
  auto prefilter for cardinality <=8192; graph otherwise; Float-query error-bound with unscaled library model error.
  NC: Arxiv256 / SIFT128. efs: 30,50,100,200,400 (one process per point).
  graph default: candidates=max(k,2*ef); bound pruning chooses exact reads.
  ML graph/auto uses RAM navigation at <graph directory>/navigation.bin (64/256/16).
  Use --entry_sampling guided for explicit guided entries; prefilter/legacy need no navigation.
  Explicit fixed policy retains the historical bits-based heuristic.
  prefilter R=0 means max(k,efs), independent of graph R.
Paths: --data_root DIR --model_dir DIR --graph_file FILE --meta_file FILE
  --query_vec FILE --query_rng FILE --gt_file FILE --out_dir DIR --tag NAME
  --inputs_dir DIR  Canonical <dataset>_<origin|mixed|pN>.ranges.ibin and
    <dataset>_<label>.k<K>.gt.ivecs; selectivities accepts origin,mixed,p1,p4,p7,p10.
  --ml_graph_file FILE --ml_meta_file SIDECAR (reuse a supplied ML layout)
  --search_centroids FILE --search_cids FILE --search_nc N
Search: --efs N or N,N,... --selectivities 1,4,7,10 (SIFT default: all four)
  --query_limit N --rabitq_bits 1|2|3|4|8 --k N --space ip|l2 --cpu N
  --filter_mode auto|graph|prefilter|legacy --prefilter_threshold N
  --entry_sampling navigation|guided|global|range
  --navigation_file FILE --navigation_ef N --navigation_budget N --navigation_count N
    Navigation must match the selected graph/layout. New builds cannot reuse an old graph's file.
  --prefilter_rerank N --rerank_num heuristic|N --rerank_factor C
    Factor is a positive integer: R=C*k in the search process, independent of ef.
    It overrides the bits heuristic; cannot combine with nonzero rerank_num.
  --rerank_policy error_bound|fixed|step|linear  Dynamic policies use current k and ef.
    They override the bits heuristic; cannot combine with factor or nonzero num.
  --rerank_efs_factor N  Deprecated compatibility option; ignored (R is bits-based).
  --rerank_error_bound --rerank_batch_blocks N
    Bounds support the ML graph route with L2/IP; --rerank_float_refine 0|1 (default 1). Batch caps requests/submit,
    NOT queue depth: up to rerank_beam_width reads stay in flight, refilled.
  --rerank_beam_width N (default 8) --equal_io_bw
    Requests per worker, not achieved bandwidth. --equal_io_bw sets rerank
    width equal to search width; this does not guarantee equal IO latency.
  --threads N --io_budget_total N (default 0 = independent per-worker beams)
    Split a total outstanding-request budget across workers (at least 1 each).
  --ml_cache_total_bytes N  Split a total dynamic cache payload across workers.
    Without this option --ml_cache_bytes remains a PER-WORKER limit.
  --memory_budget_bytes N  Validate total process peak RSS; 0 disables.
    Includes initialization peaks; this is not an allocator hard limit.
  --cpu auto|LIST  auto pins a single worker to CPU 1; multiple workers inherit
    process affinity. An explicit LIST is passed to taskset.
  --per_query_cache (default) --cross_query_cache --no_summary
  --fixed_ml_cache  Allocate the whole --ml_cache_bytes budget. By default a
    per-query cache is grown to the working set instead (same hit rate, far
    less memory), because nothing survives the between-query Clear() anyway.
Layout: --ml_levels 2,2 --shuffle_o 2 --shuffle_method block_bnf
  --methods block_bnf,identity --beam_width N --ml_cache_bytes N
  --ml_cache_policy fifo|lru --static_cache_budget N --shuffle_mem_gb N
  --static_cache_use_landing true|false (default false)
    true: validate/reuse or generate a histogram for each query range BEFORE
    loading search; false: equal level allocation, ignoring any landing file.
  --static_cache_dir DIR  Default <out_dir>/static_cache. Immutable, mode-separated
    sidecars are reused across EF values and threads with the same budget.
  --static_landing_query_rng FILE  Optional independent training ranges.
  --static_landing_file FILE  Optional existing/combined training histogram.
    For backward compatibility this implies true unless the boolean is explicit.
  --block_size N --max_iterations N (default 32; BlockBNF uses fixed margin 1)
  --skip_shuffle --skip_search --dry_run --help
Automatic reuse requires a matching complete GRAPH.completion.json receipt.
Use --skip_shuffle for an explicitly supplied legacy layout without a receipt.
--skip_search still prepares requested static caches, without executing queries.
Use a new --tag for different layout parameters. Logs get a unique run directory.
EOF
}
while (($#)); do
  case $1 in
    --help|-h) help; exit 0 ;;
    --dry_run) DRY_RUN=true; shift ;;
    --skip_shuffle) SKIP_SHUFFLE=true; shift ;;
    --skip_search) SKIP_SEARCH=true; shift ;;
    --per_query_cache) PER_QUERY_CACHE=true; shift ;;
    --cross_query_cache) PER_QUERY_CACHE=false; shift ;;
    --rerank_error_bound) RERANK_ERROR_BOUND=true; shift ;;
    --rerank_policy) (($# >= 2)) || die 'missing rerank_policy'; RERANK_POLICY=$2; RERANK_POLICY_EXPLICIT=true; shift 2 ;;
    --no_inline_rerank) shift ;; # frozen disabled-only compatibility
    --inline_rerank)
      (($# >= 2)) && [[ $2 == off ]] || die 'inline rerank was retired; only legacy off is accepted'
      shift 2 ;;
    --inline_rerank_efs_factor|--inline_rerank_beam_width|--inline_rerank_idle_only|--rerank_interval_max)
      die "inline rerank tuning was retired: $1" ;;
    --equal_io_bw) RERANK_BEAM_WIDTH=EQUAL; shift ;;
    --no_summary) SUMMARY=false; shift ;;
    --fixed_ml_cache) ML_CACHE_AUTO=0; shift ;;
    --min_colocatable_for_bnf|--move_margin)
      die "retired shuffle tuning: $1; BlockBNF uses fixed margin 1" ;;
    --static_cache_use_landing)
      (($# >= 2)) || die "missing value for $1"
      STATIC_CACHE_USE_LANDING=$2; STATIC_MODE_EXPLICIT=true; shift 2 ;;
    --rerank_efs_factor)
      # Kept for old experiment commands; no state or range constraint remains.
      (($# >= 2)) && [[ $2 != --* && -n $2 ]] || die "missing value for $1"
      shift 2 ;;
    --data_root|--model_dir|--graph_file|--meta_file|--query_vec|--query_rng|--gt_file|--out_dir|--tag|--ml_graph_file|--ml_meta_file|--search_centroids|--search_cids|--search_nc|--efs|--selectivities|--query_limit|--rabitq_bits|--k|--space|--cpu|--threads|--io_budget_total|--ml_cache_total_bytes|--memory_budget_bytes|--filter_mode|--prefilter_threshold|--prefilter_rerank|--rerank_num|--rerank_factor|--rerank_float_refine|--rerank_batch_blocks|--ml_levels|--shuffle_o|--shuffle_method|--methods|--beam_width|--ml_cache_bytes|--ml_cache_policy|--static_cache_budget|--shuffle_mem_gb|--block_size|--max_iterations|--rerank_beam_width|--static_landing_file|--static_landing_query_rng|--static_cache_dir|--inputs_dir|--entry_sampling|--navigation_file|--navigation_ef|--navigation_budget|--navigation_count)
      (($# >= 2)) && [[ $2 != --* && -n $2 ]] || die "missing value for $1"
      key=${1#--}; printf -v "${key^^}" '%s' "$2"; shift 2 ;;
    *) die "unknown option: $1" ;;
  esac
done
if ! $STATIC_MODE_EXPLICIT && [[ -n $STATIC_LANDING_FILE ]]; then STATIC_CACHE_USE_LANDING=true; fi
case $STATIC_CACHE_USE_LANDING in true|false) ;; *) die 'static_cache_use_landing must be true or false';; esac
STATIC_CACHE_DIR=${STATIC_CACHE_DIR:-$OUT_DIR/static_cache}
GRAPH_FILE=${GRAPH_FILE:-$MODEL_DIR/${INDEX_NAME}_graph.bin}
META_FILE=${META_FILE:-$MODEL_DIR/${INDEX_NAME}_meta.bin}
QUERY_VEC=${QUERY_VEC:-$DATA_ROOT/$QUERY_NAME}
SEARCH_CENTROIDS=${SEARCH_CENTROIDS:-$DATA_ROOT/Centroids/$CENTROID_NAME.centroids${SEARCH_NC}.fvecs}
SEARCH_CIDS=${SEARCH_CIDS:-$DATA_ROOT/Centroids/$CENTROID_NAME.cids${SEARCH_NC}.ivecs}
METHODS=${METHODS:-$SHUFFLE_METHOD}
IFS=, read -ra METHOD_LIST <<< "$METHODS"
for method in "${METHOD_LIST[@]}"; do
  case $method in block_bnf|identity) ;; *) die "retired or unknown shuffle method: $method";; esac
done
case $RABITQ_BITS in 1|2|3|4|8) ;; *) die 'bits must be 1,2,3,4,8';; esac
case $FILTER_MODE in auto|graph|prefilter|legacy) ;; *) die 'invalid filter_mode';; esac
if [[ -z $ENTRY_SAMPLING ]]; then
  case $FILTER_MODE in auto|graph) ENTRY_SAMPLING=navigation;; *) ENTRY_SAMPLING=guided;; esac
fi
case $ENTRY_SAMPLING in navigation|guided|global|range) ;; *) die 'invalid entry_sampling';; esac
if [[ $ENTRY_SAMPLING == navigation || -n $NAVIGATION_FILE ]]; then
  [[ $FILTER_MODE == graph || $FILTER_MODE == auto ]] || die 'navigation requires graph/auto routing'
  NAVIGATION_FILE=${NAVIGATION_FILE:-$(dirname "$GRAPH_FILE")/navigation.bin}
fi
if ! $RERANK_POLICY_EXPLICIT && [[ -n $RERANK_FACTOR || ( $RERANK_NUM != heuristic && $RERANK_NUM != 0 ) ]]; then RERANK_POLICY=fixed; fi
case $RERANK_POLICY in error_bound|fixed|step|linear) ;; *) die 'invalid rerank_policy';; esac
case $SPACE in l2|ip) ;; *) die 'space must be l2 or ip';; esac
[[ $RERANK_BEAM_WIDTH != EQUAL ]] || RERANK_BEAM_WIDTH=$BEAM_WIDTH
for key in THREADS IO_BUDGET_TOTAL MEMORY_BUDGET_BYTES K SEARCH_NC BEAM_WIDTH RERANK_BEAM_WIDTH ML_CACHE_BYTES STATIC_CACHE_BUDGET PREFILTER_THRESHOLD PREFILTER_RERANK QUERY_LIMIT RERANK_BATCH_BLOCKS NAVIGATION_EF NAVIGATION_BUDGET NAVIGATION_COUNT; do
  [[ ${!key} =~ ^[0-9]+$ ]] || die "$key must be a nonnegative integer"
done
((NAVIGATION_COUNT>0 && NAVIGATION_COUNT<=NAVIGATION_EF && NAVIGATION_COUNT<=NAVIGATION_BUDGET && NAVIGATION_EF<=65536 && NAVIGATION_BUDGET<=65536)) || die 'navigation requires 0 < count <= min(ef, budget), ef/budget <=65536'
((THREADS>0 && K>0 && BEAM_WIDTH>0 && BEAM_WIDTH<=64 && RERANK_BATCH_BLOCKS>0)) || die 'invalid k/beam/batch/factors (beam <=64)'
((RERANK_BEAM_WIDTH>0 && RERANK_BEAM_WIDTH<=256)) || die 'rerank_beam_width must be in [1,256]'
[[ -z $ML_CACHE_TOTAL_BYTES || $ML_CACHE_TOTAL_BYTES =~ ^[0-9]+$ ]] || die 'invalid ml_cache_total_bytes'
PIN=()
if [[ $CPU == auto ]]; then
  if ((THREADS==1)); then PIN=(taskset -c 1); fi
else
  PIN=(taskset -c "$CPU")
fi
((PREFILTER_RERANK==0 || PREFILTER_RERANK>=K)) || die 'prefilter_rerank must be 0 or >=k'
[[ $FILTER_MODE != auto ]] || ((PREFILTER_THRESHOLD>0)) || die 'auto threshold must be positive'
[[ $RERANK_FLOAT_REFINE == 0 || $RERANK_FLOAT_REFINE == 1 ]] || die 'rerank_float_refine must be 0/1'
[[ $RERANK_NUM == heuristic || $RERANK_NUM =~ ^[0-9]+$ ]] || die 'invalid rerank_num'
if [[ $RERANK_NUM != heuristic ]]; then
  ((RERANK_NUM==0 || RERANK_NUM>=K)) || die 'rerank_num must be 0 or >=k'
fi
if [[ -n $RERANK_FACTOR ]]; then
  [[ $RERANK_FACTOR =~ ^[1-9][0-9]*$ ]] || die 'rerank_factor must be a positive integer'
  [[ $RERANK_NUM == heuristic || $RERANK_NUM == 0 ]] || die 'choose rerank_factor or nonzero rerank_num'
fi
if [[ $RERANK_POLICY != fixed ]]; then
  [[ -z $RERANK_FACTOR && ( $RERANK_NUM == heuristic || $RERANK_NUM == 0 ) ]] || die 'dynamic rerank conflicts with fixed override'
fi
[[ $ML_LEVELS =~ ^[1-9][0-9]*,[1-9][0-9]*$ ]] || die 'ml_levels must be k,m'
IFS=, read -r LEVEL_K LEVEL_M <<< "$ML_LEVELS"
IFS=, read -ra EF_LIST <<< "$EFS"
for ef in "${EF_LIST[@]}"; do [[ $ef =~ ^[1-9][0-9]*$ ]] || die 'efs must contain positive integers'; done
[[ $STATIC_CACHE_USE_LANDING == false ]] || ((STATIC_CACHE_BUDGET>0)) ||
  die 'static_landing_file needs a non-zero --static_cache_budget (the binary would ignore it)'
if ! $SKIP_SEARCH; then
  [[ -z $QUERY_RNG && -z $GT_FILE || -n $QUERY_RNG && -n $GT_FILE ]] || die 'supply both query_rng and gt_file'
fi
[[ -z $ML_META_FILE || -n $ML_GRAPH_FILE ]] || die 'ml_meta_file requires ml_graph_file'
if [[ -n $ML_GRAPH_FILE ]]; then
  [[ $METHODS != *,* ]] || die 'supplied ML layout requires a single method'
  ML_META_FILE=${ML_META_FILE:-$ML_GRAPH_FILE.layout}; SKIP_SHUFFLE=true
fi
if [[ -n $QUERY_RNG ]]; then SELECTIVITIES=custom; fi
IFS=, read -ra SELECTIVITY_LIST <<< "$SELECTIVITIES"
for sel in "${SELECTIVITY_LIST[@]}"; do
  if [[ -n $INPUTS_DIR ]]; then
    [[ $sel == custom || $sel =~ ^(origin|mixed|p?1|p?4|p?7|p?10)$ ]] || die "invalid selectivity: $sel"
  else
    [[ $sel == custom || $DATASET == arxiv && $sel == mixed || $DATASET == sift1m && $sel =~ ^(1|4|7|10)$ ]] || die "invalid selectivity: $sel (use inputs_dir for canonical ranges)"
  fi
done
require_file() { $DRY_RUN || [[ -f $1 ]] || die "file not found: $1"; }
for f in "$GRAPH_FILE" "$META_FILE"; do require_file "$f"; done
if ! $SKIP_SEARCH; then
  [[ -z $NAVIGATION_FILE ]] || require_file "$NAVIGATION_FILE"
  for f in "$BIN_DIR/alliance_search_disk" "$QUERY_VEC"; do require_file "$f"; done
  if ((SEARCH_NC>0)); then require_file "$SEARCH_CENTROIDS"; require_file "$SEARCH_CIDS"; fi
fi
if $DRY_RUN; then RUN_DIR=$OUT_DIR/dry_run; else mkdir -p "$OUT_DIR"; RUN_DIR=$(mktemp -d "$OUT_DIR/search_${DATASET}_$(date +%Y%m%d_%H%M%S)_XXXXXX"); fi
run() {
  local logfile=$1; shift
  printf '%q ' "$@"; printf '\n'
  if ! $DRY_RUN; then
    printf '%q ' "$@" > "$logfile.command"; printf '\n' >> "$logfile.command"
    OMP_NUM_THREADS="$THREADS" OMP_DYNAMIC=FALSE OPENBLAS_NUM_THREADS=1 "$@" 2>&1 | tee "$logfile"
  fi
}
GRAPH_BASE=$(basename "$GRAPH_FILE" .bin); GRAPH_BASE=${GRAPH_BASE%_graph}
for method in "${METHOD_LIST[@]}"; do
  local_tag=${TAG:-o${SHUFFLE_O}_k${LEVEL_K}m${LEVEL_M}_${method}}
  if [[ -n $TAG && $METHODS == *,* ]]; then local_tag=${TAG}_${method}; fi
  [[ $local_tag != */* && $method != */* ]] || die 'tag/method must not contain slashes'
  ml_graph=${ML_GRAPH_FILE:-$OUT_DIR/${GRAPH_BASE}_ml_${local_tag}_graph.bin}
  sidecar=${ML_META_FILE:-$ml_graph.layout}
  ml_meta=$OUT_DIR/${GRAPH_BASE}_ml_${local_tag}_meta.bin
  if $SKIP_SHUFFLE; then
    require_file "$ml_graph"; require_file "$sidecar"
  elif [[ -f $ml_graph && -f $sidecar && -f $ml_meta ]]; then
    python3 "$ROOT_DIR/scripts/validate_shuffle_completion.py" "$ml_graph" "$sidecar" "$ml_meta" ||
      die "layout is not committed or has changed: $ml_graph; legacy reuse requires explicit --skip_shuffle"
    echo "Reuse layout: $ml_graph (use a new --tag when changing layout parameters)"
  else
    [[ ! -e $ml_graph && ! -e $sidecar && ! -e $ml_meta && ! -e $ml_graph.completion.json ]] ||
      die "partial layout exists: $ml_graph; use a new --tag"
    require_file "$BIN_DIR/alliance_shuffle_disk"
    # Search-time caches are prepared per selectivity, separately from layout.
    shuffle_args=(--graph_file "$GRAPH_FILE" --meta_file "$META_FILE" --out_graph_file "$ml_graph" --out_meta_file "$ml_meta"
      --method "$method" --multi_level --shuffle_o "$SHUFFLE_O" --level_k "$LEVEL_K" --level_m "$LEVEL_M"
      --block_size "$BLOCK_SIZE" --max_iterations "$MAX_ITERATIONS"
      --shuffle_mem_gb "$SHUFFLE_MEM_GB" --space "$SPACE")
    run "$RUN_DIR/${method}_shuffle.log" ${PIN[@]+"${PIN[@]}"} "$BIN_DIR/alliance_shuffle_disk" "${shuffle_args[@]}"
    if ! $DRY_RUN; then
      python3 "$ROOT_DIR/scripts/validate_shuffle_completion.py" "$ml_graph" "$sidecar" "$ml_meta" ||
        die "shuffle did not publish a valid completion receipt: $ml_graph"
    fi
  fi
  if $SKIP_SEARCH && ((STATIC_CACHE_BUDGET==0)); then continue; fi
  for sel in "${SELECTIVITY_LIST[@]}"; do
    if [[ $sel == custom ]]; then qr=$QUERY_RNG; gt=$GT_FILE
    elif [[ -n $INPUTS_DIR ]]; then
      ds=$DATASET; [[ $ds != arxiv ]] || ds=arxiv2m
      label=$sel; [[ ! $label =~ ^[0-9]+$ ]] || label=p$label
      qr=$INPUTS_DIR/${ds}_${label}.ranges.ibin; gt=$INPUTS_DIR/${ds}_${label}.k${K}.gt.ivecs
    elif [[ $DATASET == arxiv ]]; then qr=$DATA_ROOT/test.ranges.ibin; gt=$DATA_ROOT/test.gt.ivecs
    else qr=$DATA_ROOT/sift_query_range_2-${sel}.ibin; gt=$DATA_ROOT/sift_query_2-${sel}.gt.ivecs; fi
    if ! $SKIP_SEARCH; then require_file "$qr"; require_file "$gt"; fi
    if ((STATIC_CACHE_BUDGET>0)) && $STATIC_CACHE_USE_LANDING && [[ -z $STATIC_LANDING_FILE ]]; then
      require_file "${STATIC_LANDING_QUERY_RNG:-$qr}"
    fi
    search_sidecar=$sidecar
    if ((STATIC_CACHE_BUDGET>0)); then
      preparer=$ROOT_DIR/scripts/prepare_selectivity_static_cache.py
      require_file "$preparer"
      prep=(python3 "$preparer" --graph_file "$GRAPH_FILE" --meta_file "$META_FILE"
        --ml_graph_file "$ml_graph" --ml_meta_file "$sidecar" --space "$SPACE" --bin_dir "$BIN_DIR"
        --query_rng "$qr" --filter_mode "$FILTER_MODE" --prefilter_threshold "$PREFILTER_THRESHOLD"
        --query_limit "$QUERY_LIMIT" --budgets "$STATIC_CACHE_BUDGET" --cache_dir "$STATIC_CACHE_DIR"
        --static_cache_use_landing "$STATIC_CACHE_USE_LANDING" --print_sidecar)
      if $STATIC_CACHE_USE_LANDING; then
        [[ -z $STATIC_LANDING_FILE ]] || prep+=(--static_landing_file "$STATIC_LANDING_FILE")
        [[ -z $STATIC_LANDING_QUERY_RNG ]] || prep+=(--static_landing_query_rng "$STATIC_LANDING_QUERY_RNG")
      fi
      printf '%q ' "${prep[@]}"; printf '\n'
      if $DRY_RUN; then
        search_sidecar=$STATIC_CACHE_DIR/DRY_RUN_${STATIC_CACHE_USE_LANDING}_${sel}_${STATIC_CACHE_BUDGET}.layout
      else
        printf '%q ' "${prep[@]}" > "$RUN_DIR/${method}_${sel}_static.command"; printf '\n' >> "$RUN_DIR/${method}_${sel}_static.command"
        search_sidecar=$("${prep[@]}")
        require_file "$search_sidecar"
        printf '%s\n' "$search_sidecar" > "$RUN_DIR/${method}_${sel}_static.path"
      fi
    fi
    if $SKIP_SEARCH; then continue; fi
    for ef in "${EF_LIST[@]}"; do
      r=$RERANK_NUM
      if [[ $RERANK_POLICY != fixed || -n $RERANK_FACTOR ]]; then
        r=0  # Keep the factor symbolic; the query API applies it to k.
      elif [[ $r == heuristic ]]; then
        # Historical Arxiv heuristic, independent of ef. Saturation is workload
        # dependent: Deep10M can need a larger factor even with 3-bit codes.
        # An explicit --rerank_factor selects a calibrated workload policy.
        case $RABITQ_BITS in
          1) c=16 ;;
          2) c=4 ;;
          *) c=2 ;;   # bits>=3
        esac
        r=$((c*K))
      fi
      args=(--graph_file "$GRAPH_FILE" --meta_file "$META_FILE" --query_vec "$QUERY_VEC" --query_rng "$qr" --gt_file "$gt"
        --k "$K" --space "$SPACE" --pipeline 1 --beam_width "$BEAM_WIDTH" --ml_graph_file "$ml_graph" --ml_meta_file "$search_sidecar"
        --ml_cache_bytes "$ML_CACHE_BYTES" --ml_cache_policy "$ML_CACHE_POLICY"
        --rerank_num "$r" --filter_mode "$FILTER_MODE" --prefilter_threshold "$PREFILTER_THRESHOLD" --prefilter_rerank "$PREFILTER_RERANK"
        --efs "$ef" --query_limit "$QUERY_LIMIT" --rabitq_bits "$RABITQ_BITS"
        --rerank_float_refine "$RERANK_FLOAT_REFINE" --rerank_batch_blocks "$RERANK_BATCH_BLOCKS"
        --rerank_beam_width "$RERANK_BEAM_WIDTH" --ml_cache_auto "$ML_CACHE_AUTO"
        --threads "$THREADS" --io_budget_total "$IO_BUDGET_TOTAL" --memory_budget_bytes "$MEMORY_BUDGET_BYTES")
      if [[ -n $ML_CACHE_TOTAL_BYTES ]]; then args+=(--ml_cache_total_bytes "$ML_CACHE_TOTAL_BYTES"); fi
      args+=(--rerank_policy "$RERANK_POLICY")
      args+=(--entry_sampling "$ENTRY_SAMPLING")
      if [[ -n $NAVIGATION_FILE ]]; then
        args+=(--navigation_file "$NAVIGATION_FILE" --navigation_ef "$NAVIGATION_EF"
          --navigation_budget "$NAVIGATION_BUDGET" --navigation_count "$NAVIGATION_COUNT")
      fi
      budget_label=$r
      if [[ $RERANK_POLICY != fixed ]]; then budget_label=$RERANK_POLICY; fi
      if [[ -n $RERANK_FACTOR ]]; then
        args+=(--rerank_factor "$RERANK_FACTOR")
        budget_label="${RERANK_FACTOR}xk"
      fi
      if ((SEARCH_NC>0)); then args+=(--rabitq_centroids "$SEARCH_CENTROIDS" --rabitq_cids "$SEARCH_CIDS"); fi
      if $PER_QUERY_CACHE; then args+=(--per_query_cache); fi
      if $RERANK_ERROR_BOUND; then args+=(--rerank_error_bound); fi
      run "$RUN_DIR/${method}_p${sel}_b${RABITQ_BITS}_ef${ef}_R${budget_label}.log" ${PIN[@]+"${PIN[@]}"} "$BIN_DIR/alliance_search_disk" "${args[@]}"
    done
  done
done
SUMMARIZER=$ROOT_DIR/scripts/summarize_search_logs.py
if $SUMMARY && ! $DRY_RUN && ! $SKIP_SEARCH && [[ -f $SUMMARIZER ]]; then
  # Never fatal: the experiment already ran, a summary failure must not mask it.
  python3 "$SUMMARIZER" "$RUN_DIR" -o "$RUN_DIR/summary.csv" || echo "WARN: summary failed" >&2
fi
echo "Logs: $RUN_DIR"
