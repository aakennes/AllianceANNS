#!/usr/bin/env bash
set -euo pipefail
if (($# == 0)) || [[ $1 != arxiv && $1 != sift1m ]]; then
  echo 'Usage: bash scripts/disk_build_common.sh {arxiv|sift1m} [options]' >&2
  exit 2
fi
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DIR=${ALLIANCE_BIN_DIR:-$ROOT_DIR/build/bin}
DATASET=$1; shift
if [[ $DATASET == arxiv ]]; then
  DATA_ROOT=${ALLIANCE_DATA_ROOT:-$HOME/data}/Arxiv; NAME=wp1; SPACE=ip
  BASE_NAME=base.2M.fvecs; ATTR_NAME=base.attribute.ibin
else
  DATA_ROOT=${ALLIANCE_DATA_ROOT:-$HOME/data}/Sift1m; NAME=sift_wp1; SPACE=l2
  BASE_NAME=sift_base.fvecs; ATTR_NAME=sift_base.attribute.ibin
fi
OUT_DIR=$ROOT_DIR/build/_p3; GRAPH_FILE=; META_FILE=; CHECKPOINT_META=
BASEVEC=; BASEATT=; M=32; EFC=256; O=4; WP=5; THREADS=32
WOW_EFC=256; RESIDENT_BUILDER=left_stream; RESIDENT_BATCH=256; BUILD_MEM_GB=1.0; WIN_ANCHOR=top
DRY_RUN=false; RESUME=false; REPORT_ONLY=false; EXTRA=()
die() { echo "ERROR: $*" >&2; exit 2; }
while (($#)); do
  case $1 in
    --help|-h)
      cat <<'EOF'
Usage: bash scripts/disk_build_common.sh {arxiv|sift1m} [options]
Compatibility helper; the JSON workflow uses build.sh DATASET.
Defaults: M32 efc256 o4 wp5 threads32 wow_efc256 resident_builder left_stream resident_batch256 build_mem_gb1.0 win_anchor top.
Arxiv outputs build/_p3/wp1_{graph,meta,ckpt}.bin; SIFT outputs sift_wp1_*.
disk_search_common.sh uses the same default graph/meta paths. Existing outputs are never overwritten.
Options: --data_root DIR --out_dir DIR --basevec FILE --baseatt FILE
  --graph_file FILE --meta_file FILE --checkpoint_meta FILE --space l2|ip
  --m N --efc N --o N --wp N --threads N --wow_efc N --resident_builder left_stream --resident_batch N
  --build_mem_gb GB --win_anchor top|bottom --resume --dry_run --help
  --n_limit N --centroids FILE --cids FILE --cluster_centroids FILE --cluster_cids FILE
  --rabitq_bits N --node_page_bytes N --enum_thresh N --wow_thresh N --build_mem_report
--build_mem_report reads inputs and prints the plan without creating outputs or logs;
it may inspect existing graph/meta paths. --dry_run only prints the command.
For a new build after an earlier run, choose a new --out_dir (and use it as search --model_dir).
EOF
      exit 0 ;;
    --dry_run) DRY_RUN=true; shift ;;
    --resume) RESUME=true; shift ;;
    --copy_top|--wow_online_compat) die "$1 is retired; only LeftStream is supported" ;;
    --wow_passes)
      (($#>=2)) && [[ $2 != --* && -n $2 ]] || die "missing value for $1"
      [[ $2 == 1 ]] || die "resident refinement passes are retired; --wow_passes accepts only 1"
      shift 2 ;;
    --build_mem_report) REPORT_ONLY=true; EXTRA+=("$1"); shift ;;
    --n_limit|--centroids|--cids|--cluster_centroids|--cluster_cids|--rabitq_bits|--node_page_bytes|--enum_thresh|--wow_thresh)
      (($#>=2)) && [[ $2 != --* && -n $2 ]] || die "missing value for $1"
      EXTRA+=("$1" "$2"); shift 2 ;;
    --data_root|--out_dir|--basevec|--baseatt|--graph_file|--meta_file|--checkpoint_meta|--space|--m|--efc|--o|--wp|--threads|--wow_efc|--resident_builder|--resident_batch|--build_mem_gb|--win_anchor)
      (($#>=2)) && [[ $2 != --* && -n $2 ]] || die "missing value for $1"
      key=${1#--}; printf -v "${key^^}" '%s' "$2"; shift 2 ;;
    *) die "unknown option: $1" ;;
  esac
done
BASEVEC=${BASEVEC:-$DATA_ROOT/$BASE_NAME}; BASEATT=${BASEATT:-$DATA_ROOT/$ATTR_NAME}
GRAPH_FILE=${GRAPH_FILE:-$OUT_DIR/${NAME}_graph.bin}; META_FILE=${META_FILE:-$OUT_DIR/${NAME}_meta.bin}
CHECKPOINT_META=${CHECKPOINT_META:-$OUT_DIR/${NAME}_ckpt.bin}
for key in M EFC O WP THREADS WOW_EFC RESIDENT_BATCH; do
  [[ ${!key} =~ ^[0-9]+$ ]] || die "$key must be a nonnegative integer"
done
[[ $RESIDENT_BUILDER == left_stream ]] || die 'resident builder is retired or unknown; only left_stream is supported'
((RESIDENT_BATCH>0)) || die 'resident_batch must be positive'
((M>0 && EFC>0 && O>1 && THREADS>0)) || die 'm/efc/threads must be positive, o>1'
[[ $BUILD_MEM_GB =~ ^[0-9]+([.][0-9]+)?$ ]] || die 'invalid build_mem_gb'
[[ $SPACE == l2 || $SPACE == ip ]] || die 'space must be l2 or ip'
[[ $WIN_ANCHOR == top || $WIN_ANCHOR == bottom ]] || die 'invalid win_anchor'
args=(--basevec "$BASEVEC" --baseatt "$BASEATT" --graph_file "$GRAPH_FILE" --meta_file "$META_FILE"
  --checkpoint_meta "$CHECKPOINT_META" --m "$M" --efc "$EFC" --o "$O" --wp "$WP" --threads "$THREADS" --space "$SPACE"
  --wow_efc "$WOW_EFC" --resident_builder "$RESIDENT_BUILDER" --resident_batch "$RESIDENT_BATCH" --build_mem_gb "$BUILD_MEM_GB" --win_anchor "$WIN_ANCHOR")
if ((${#EXTRA[@]}>0)); then args+=("${EXTRA[@]}"); fi
if $RESUME; then args+=(--resume); fi
printf '%q ' "$BIN_DIR/alliance_build_disk" "${args[@]}"; printf '\n'
$DRY_RUN && exit 0
[[ -x $BIN_DIR/alliance_build_disk ]] || die 'build binary missing; compile alliance_build_disk first'
if $REPORT_ONLY; then exec "$BIN_DIR/alliance_build_disk" "${args[@]}"; fi
if $RESUME; then
  [[ -f $GRAPH_FILE && -f $CHECKPOINT_META && ! -e $META_FILE ]] || die 'resume needs graph/checkpoint and no completed meta file'
else
  [[ -f $BASEVEC && -f $BASEATT ]] || die 'base vectors/attributes missing'
  for f in "$GRAPH_FILE" "$META_FILE" "$CHECKPOINT_META"; do [[ ! -e $f ]] || die "output already exists: $f"; done
fi
mkdir -p "$OUT_DIR" "$(dirname "$GRAPH_FILE")" "$(dirname "$META_FILE")" "$(dirname "$CHECKPOINT_META")"
LOG=$(mktemp "$OUT_DIR/build_${DATASET}_$(date +%Y%m%d_%H%M%S)_XXXXXX.log")
printf '%q ' "$BIN_DIR/alliance_build_disk" "${args[@]}" > "$LOG.command"; printf '\n' >> "$LOG.command"
"$BIN_DIR/alliance_build_disk" "${args[@]}" 2>&1 | tee "$LOG"
echo "Build log: $LOG"
