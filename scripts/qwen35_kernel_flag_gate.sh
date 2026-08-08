#!/usr/bin/env bash
# Per-flag numerical A/B for the Qwen3.5 fast kernels.
#
# Captures a golden logit trajectory with every fast path disabled, then
# re-runs with one enabled at a time, so a divergence is attributed to a
# specific kernel instead of the whole set. Timing benchmarks cannot tell a
# kernel that is fast from one that is fast and wrong; this can.
#
# Run the control first (baseline compared against its own golden). On a
# deterministic engine it reports max |dlogit| = 0, which is what makes any
# nonzero result from the sweep meaningful.
#
# Usage:
#   scripts/qwen35_kernel_flag_gate.sh <model_dir> [extra golden args]
#
# Example, real text on the scratch GPU:
#   ZE_AFFINITY_MASK=2 scripts/qwen35_kernel_flag_gate.sh /models/Qwen3.6-27B-NVFP4 \
#       --steps 24 --prefill 24 --prompt "$(cat sample.txt)"
#
# FUSED_ESIMD_DELTA_DECODE requires ESIMD_DELTA: the fused decode path and the
# scalar baseline use different recurrent-state layouts, and mixing them
# produces garbage rather than an error, so the sweep enables the two together.
set -u

MODEL=${1:?usage: qwen35_kernel_flag_gate.sh <model_dir> [args...]}
shift
BENCH=${ARCAINE_MBENCH:-./build/arcaine_mbench}
GOLDEN=${GOLDEN:-golden_baseline.bin}

FLAGS=(ARCAINE_QWEN35_XMX_ATTENTION
       ARCAINE_QWEN35_ESIMD_DELTA
       ARCAINE_QWEN35_FUSED_ESIMD_DELTA_DECODE
       ARCAINE_QWEN35_FUSED_BA_PROJECTION)

all_off() { for f in "${FLAGS[@]}"; do export "$f"=0; done; }
report() { grep -E "max \|dlogit\||mean \|dlogit\||top-1 agreement|KL max|perplexity|PASS|FAIL:"; }

echo "=== capture: all fast paths off ==="
all_off
"$BENCH" --model "$MODEL" --golden capture --out "$GOLDEN" "$@" || exit 1

echo
echo "=== control: baseline vs its own golden (expect max |dlogit| = 0) ==="
all_off
"$BENCH" --model "$MODEL" --golden compare --golden-file "$GOLDEN" | report

for on in "${FLAGS[@]}"; do
    all_off
    export "$on"=1
    label="$on"
    if [ "$on" = ARCAINE_QWEN35_FUSED_ESIMD_DELTA_DECODE ]; then
        export ARCAINE_QWEN35_ESIMD_DELTA=1
        label="$on (+ESIMD_DELTA)"
    fi
    echo
    echo "=== $label ==="
    "$BENCH" --model "$MODEL" --golden compare --golden-file "$GOLDEN" | report
done

echo
echo "=== all fast paths on (production default) ==="
for f in "${FLAGS[@]}"; do unset "$f"; done
"$BENCH" --model "$MODEL" --golden compare --golden-file "$GOLDEN" | report
