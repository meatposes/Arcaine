#!/usr/bin/env bash
# A/B the existing ARCAINE_QWEN35_* codepath flags on one checkpoint.
#
# Every flag here already gates an alternate implementation of the same math, so
# this measures which of them is actually the faster choice for a given
# checkpoint rather than for the one they were tuned on. No code changes.
#
# Usage: scripts/qwen35_flag_sweep.sh <model_dir> [reps]
set -uo pipefail

MODEL="${1:?usage: $0 <model_dir> [reps]}"
REPS="${2:-3}"
BENCH="${BENCH:-./build/arcaine_mbench}"
PP="${PP:-512,2048}"
TG="${TG:-32}"
MAX_SEQ="${MAX_SEQ:-4096}"

run() {
    local label="$1"; shift
    echo "=============================================================="
    echo "== $label"
    echo "==   env: $*"
    echo "=============================================================="
    # Drop the per-layer load chatter and the table chrome, but never filter on
    # a whitelist: a configuration that throws must show its error rather than
    # silently producing an empty block.
    env "$@" "$BENCH" "$MODEL" -p "$PP" -n "$TG" -r "$REPS" -w 1 \
        --max-seq "$MAX_SEQ" 2>&1 |
        grep -Ev '^\[qwen35-load\]|^ (test|─)|^$|^loading model|^backend |^model  '
    echo
}

run "baseline (shipped defaults)"
run "nvfp4 dpas MLP on"        ARCAINE_QWEN35_NVFP4_DPAS=1
run "esimd deltanet off"       ARCAINE_QWEN35_ESIMD_DELTA=0
run "projection fusion off"    ARCAINE_QWEN35_FUSED_PROJECTIONS=0
run "xmx attention off"        ARCAINE_QWEN35_XMX_ATTENTION=0
