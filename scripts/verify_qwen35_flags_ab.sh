#!/usr/bin/env bash
# Per-flag numerical A/B for the Qwen3.5 fast kernels.
#
# Captures a golden logit trajectory with every fast path disabled, then
# re-runs with one enabled at a time, so a divergence is attributed to a
# specific kernel instead of the whole set. Timing benchmarks cannot tell a
# kernel that is fast from one that is fast and wrong; this can.
#
# Run the control first (compare the baseline against its own golden). On a
# deterministic engine it reports max |Δlogit| = 0, which is what makes any
# nonzero result from the sweep meaningful.
#
# Usage:
#   scripts/verify_qwen35_flags_ab.sh <model_dir> [extra arcaine_verify args]
#
# Example, on the scratch GPU with a real-text prompt:
#   ZE_AFFINITY_MASK=2 scripts/verify_qwen35_flags_ab.sh /models/Qwen3.6-27B-NVFP4 \
#       --steps 64 --prefill 32 --prompt "$(cat sample.txt)"
#
# Note that FUSED_ESIMD_DELTA_DECODE requires ESIMD_DELTA: the fused decode
# path and the scalar baseline use different recurrent-state layouts, and
# mixing them produces garbage rather than an error. The sweep therefore
# enables the two together.
set -u

MODEL=${1:?usage: verify_qwen35_flags_ab.sh <model_dir> [args...]}
shift
BIN=${ARCAINE_VERIFY:-./build/arcaine_verify}
GOLDEN=${GOLDEN:-golden_baseline.bin}

FLAGS=(ARCAINE_QWEN35_XMX_ATTENTION
       ARCAINE_QWEN35_ESIMD_DELTA
       ARCAINE_QWEN35_FUSED_ESIMD_DELTA_DECODE
       ARCAINE_QWEN35_FUSED_BA_PROJECTION)

all_off() { for f in "${FLAGS[@]}"; do export "$f"=0; done; }

echo "=== capture: all fast paths off ==="
all_off
"$BIN" capture --model "$MODEL" --out "$GOLDEN" "$@" || exit 1

echo
echo "=== control: baseline vs its own golden (expect max |Δlogit| = 0) ==="
all_off
"$BIN" compare --model "$MODEL" --golden "$GOLDEN" \
    | grep -E "max \||mean \||top-1|KL max|perplexity|PASS|FAIL:"

for on in "${FLAGS[@]}"; do
    all_off
    export "$on"=1
    label="$on"
    # The fused decode path is only valid on top of the ESIMD layout.
    if [ "$on" = ARCAINE_QWEN35_FUSED_ESIMD_DELTA_DECODE ]; then
        export ARCAINE_QWEN35_ESIMD_DELTA=1
        label="$on (+ESIMD_DELTA)"
    fi
    echo
    echo "=== $label ==="
    "$BIN" compare --model "$MODEL" --golden "$GOLDEN" \
        | grep -E "max \||mean \||top-1|KL max|perplexity|PASS|FAIL:"
done

echo
echo "=== all fast paths on (production default) ==="
for f in "${FLAGS[@]}"; do unset "$f"; done
"$BIN" compare --model "$MODEL" --golden "$GOLDEN" \
    | grep -E "max \||mean \||top-1|KL max|perplexity|PASS|FAIL:"
