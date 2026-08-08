#pragma once

#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "cache.hpp"
#include "config.hpp"
#include "kernels.hpp"
#include "weights.hpp"
#include "workspace.hpp"
#include "../../runtime/quantization/fp8.hpp"
#include "../../runtime/quantization/nvfp4.hpp"
#include "../../runtime/gpu/ops.hpp"
#include "runtime/kernels/elementwise.hpp"
#include "runtime/kernels/rms_norm.hpp"

using namespace qwen35_kernels;

inline bool qwen35_nvfp4_dpas_enabled() {
    static bool enabled = [] {
        const char* value = std::getenv("ARCAINE_QWEN35_NVFP4_DPAS");
        // The current dense Xe2 kernel remains available for explicit A/B
        // measurements, but oneDNN's BMG f4 implementation is materially
        // faster for both M=1 decode and large-M prefill on this checkpoint.
        if (!value) return false;
        return std::strcmp(value, "0") != 0 && std::strcmp(value, "off") != 0 &&
               std::strcmp(value, "false") != 0 && std::strcmp(value, "no") != 0;
    }();
    return enabled;
}

inline bool qwen35_subgroup_attention_enabled() {
    static bool enabled = [] {
        const char* value = std::getenv("ARCAINE_QWEN35_SUBGROUP_ATTENTION");
        // Experimental scalar/SIMD baseline. The production optimization is
        // the XMX/DPAS tiled attention path, not this reduction-only variant.
        if (!value) return false;
        return std::strcmp(value, "0") != 0 && std::strcmp(value, "off") != 0 &&
               std::strcmp(value, "false") != 0 && std::strcmp(value, "no") != 0;
    }();
    return enabled;
}

inline bool qwen35_xmx_attention_enabled() {
    static bool enabled = [] {
        const char* value = std::getenv("ARCAINE_QWEN35_XMX_ATTENTION");
        if (!value) return true;
        return std::strcmp(value, "0") != 0 && std::strcmp(value, "off") != 0 &&
               std::strcmp(value, "false") != 0 && std::strcmp(value, "no") != 0;
    }();
    return enabled;
}

inline bool qwen35_esimd_delta_enabled() {
    static bool enabled = [] {
        const char* value = std::getenv("ARCAINE_QWEN35_ESIMD_DELTA");
        // The scalar/SIMT implementation remains available as the A/B
        // baseline. The ESIMD path is exact at BF16 output precision and keeps
        // the 128x128 recurrent state in registers across the sequence.
        if (!value) return true;
        return std::strcmp(value, "0") != 0 && std::strcmp(value, "off") != 0 &&
               std::strcmp(value, "false") != 0 && std::strcmp(value, "no") != 0;
    }();
    return enabled;
}

inline bool qwen35_fused_esimd_delta_decode_enabled() {
    static bool enabled = [] {
        const char* value =
            std::getenv("ARCAINE_QWEN35_FUSED_ESIMD_DELTA_DECODE");
        if (!value) return true;
        return std::strcmp(value, "0") != 0 && std::strcmp(value, "off") != 0 &&
               std::strcmp(value, "false") != 0 && std::strcmp(value, "no") != 0;
    }();
    return enabled;
}

// M=1 ESIMD GEMV for the NVFP4 MLP. Off by default: 249.7 ms/token against 79.0
// for the general f4 path on Qwen3.6-27B.
//
// Kept wired, with a tunable work-group, because it is the obvious-looking fix
// for the decode bandwidth gap and the measurements say it is not one:
//
//   work-group size is not the problem. 1/8/16/32/64 rows per group all land
//   between 249 and 265 ms. Under-occupancy was the plausible explanation and
//   it is wrong.
//
//   the strided weight-scale load is most of the cost. weight_scale is
//   [K/16, N], so a row-per-work-item GEMV reads it with stride N — 320
//   scattered single-byte loads per output row against 2560 bytes of actual
//   weight. Replacing that with a contiguous load (numerically wrong, measured
//   as a diagnostic) drops the kernel to 76.6 ms, a 3.26x speedup.
//
//   but that only ties oneDNN. 76.6 ms against 79.0, both near 250 GB/s and
//   ~43% of roofline. Two independent implementations converging there is the
//   useful result: at M=1 the ceiling belongs to the W4A4 format, not to the
//   kernel, so an n-major scale copy (~0.94 GB) would buy about 3%.
//
// See notes/qwen3_5_27b/decode_bandwidth_gap.md.
inline bool qwen35_nvfp4_decode_gemv_enabled() {
    static bool enabled = [] {
        const char* value = std::getenv("ARCAINE_QWEN35_NVFP4_DECODE_GEMV");
        if (!value) return false;
        return std::strcmp(value, "0") != 0 && std::strcmp(value, "off") != 0 &&
               std::strcmp(value, "false") != 0 && std::strcmp(value, "no") != 0;
    }();
    return enabled;
}

// Rows per work-group for the M=1 NVFP4 GEMV. The kernels default to 1, which
// is one work-item per work-group; this model's N is large enough that the
// launch geometry, not the arithmetic, decides whether they are usable.
inline int qwen35_nvfp4_decode_gemv_wg() {
    static int wg = [] {
        const char* value = std::getenv("ARCAINE_QWEN35_NVFP4_DECODE_GEMV_WG");
        if (!value) return 32;
        int parsed = std::atoi(value);
        return parsed > 0 ? parsed : 32;
    }();
    return wg;
}

// How many NVFP4 MLP layers to requantize to FP8 at load, from layer 0 up.
// Off by default: it roughly doubles the MLP's residency, which is a choice
// about the deployment and not something to impose. See
// notes/qwen3_5_27b/decode_bandwidth_gap.md for the measured trade.
inline int qwen35_mlp_fp8_layers() {
    static int layers = [] {
        const char* value = std::getenv("ARCAINE_QWEN35_MLP_FP8_LAYERS");
        if (!value) return 0;
        if (std::strcmp(value, "all") == 0) return INT_MAX;
        int parsed = std::atoi(value);
        return parsed > 0 ? parsed : 0;
    }();
    return layers;
}


// Clip factor for the FP8 row scale. See requantize_nvfp4_to_fp8.
//
// Re-measured on a deterministic engine over 1000 records of prose, all 56
// layers converted, against the NVFP4 weights they replace: top-1 agreement is
// 0.945 at clips 0.90 and 0.95 and 0.949 at 1.00, so the conversion changes
// about 5.5% of predicted tokens whatever the clip. Perplexity is not
// monotonic in clip at any sample size tried and is dominated by a handful of
// high-loss steps, so **clip is not tunable with the current instrument** and
// 0.9 is a default rather than an optimum. An earlier comment here claimed a
// flat optimum between 0.8 and 0.9 with a cliff below; that was measured on a
// nondeterministic engine and does not hold.
//
// See notes/qwen3_5_27b/decode_bandwidth_gap.md.
inline float qwen35_mlp_fp8_clip() {
    static float clip = [] {
        const char* value = std::getenv("ARCAINE_QWEN35_MLP_FP8_CLIP");
        if (!value) return 0.9f;
        float parsed = (float)std::atof(value);
        return parsed > 0.0f ? parsed : 1.0f;
    }();
    return clip;
}

// Largest batch the per-token fused decode core is used for. Above this the
// chunked path wins, and prefill is far above it. Speculative verify windows
// are a handful of tokens, so the default covers them.
inline int qwen35_fused_decode_max_seq() {
    static int limit = [] {
        const char* value = std::getenv("ARCAINE_QWEN35_FUSED_DECODE_MAX_SEQ");
        if (!value) return 8;
        int parsed = std::atoi(value);
        return parsed > 0 ? parsed : 8;
    }();
    return limit;
}

inline void matmul_proj(const bf16* A, int M, int K, const Qwen35Proj& W,
                        bf16* C, GpuEngine& context) {
    if (const auto* w = std::get_if<Fp8Linear>(&W))
        matmul_fp8(A, M, K, *w, C, context);
    else
        matmul_int4(A, M, K, std::get<Int4Linear>(W), C, context);
}

// The MTP head is on by default when the checkpoint carries one. Off keeps the
// weights resident but idle, so speculative and plain decoding can be compared
// without reloading the model.
inline bool qwen35_mtp_enabled() {
    static bool enabled = [] {
        const char* value = std::getenv("ARCAINE_QWEN35_MTP");
        if (!value) return true;
        return std::strcmp(value, "0") != 0 && std::strcmp(value, "off") != 0 &&
               std::strcmp(value, "false") != 0 && std::strcmp(value, "no") != 0;
    }();
    return enabled;
}

// Gates the fused [b|a] projection on the M=1 decode path only. The
// multi-token prefill path cannot use it - the fused output interleaves b and
// a per token, which is not the layout its consumers read - and no longer
// consults this flag.
inline bool qwen35_fused_ba_projection_enabled() {
    static bool enabled = [] {
        const char* value =
            std::getenv("ARCAINE_QWEN35_FUSED_BA_PROJECTION");
        if (!value) return true;
        return std::strcmp(value, "0") != 0 && std::strcmp(value, "off") != 0 &&
               std::strcmp(value, "false") != 0 && std::strcmp(value, "no") != 0;
    }();
    return enabled;
}

inline void qwen35_full_attention_forward(
    GpuEngine& context,
    const Qwen35FullAttentionWeights& weights,
    Qwen35KvLayerCache& cache,
    Qwen35Workspace& workspace,
    const bf16* hidden,
    const int32_t* positions,
    bf16* output,
    int seq,
    int past,
    const Qwen35Config& config) {
    const auto& c = config.text;
    auto& queue = context.queue;
    int query_dim = c.num_attention_heads * c.head_dim;
    int key_value_dim = c.num_key_value_heads * c.head_dim;
    if (cache.filled != past)
        throw std::runtime_error("Qwen3.5 KV cache position mismatch");
    if (past + seq > cache.capacity)
        throw std::runtime_error("Qwen3.5 KV cache overflow");

    if (weights.fused_projections) {
        matmul_proj(hidden, seq, c.hidden_size, weights.qkv_proj,
                    workspace.tmp0.data(), context);
        qwen35_split_q_gate_kv(
            queue, workspace.tmp0.data(), workspace.tmp2.data(),
            workspace.tmp3.data(), workspace.tmp1.data(), workspace.tmp4.data(),
            seq, c.num_attention_heads, c.num_key_value_heads, c.head_dim);
    } else {
        matmul_proj(hidden, seq, c.hidden_size, weights.q_proj,
                    workspace.tmp0.data(), context);
        qwen35_split_q_gate(queue, workspace.tmp0.data(), workspace.tmp2.data(),
                            workspace.tmp3.data(), seq, c.num_attention_heads,
                            c.head_dim);
        matmul_proj(hidden, seq, c.hidden_size, weights.k_proj,
                    workspace.tmp1.data(), context);
        matmul_proj(hidden, seq, c.hidden_size, weights.v_proj,
                    workspace.tmp4.data(), context);
    }
    rms_norm(queue, workspace.tmp2.data(), weights.q_norm.data(), workspace.tmp2.data(),
             seq * c.num_attention_heads, c.head_dim, c.rms_norm_eps);
    rms_norm(queue, workspace.tmp1.data(), weights.k_norm.data(), workspace.tmp1.data(),
             seq * c.num_key_value_heads, c.head_dim, c.rms_norm_eps);
    qwen35_apply_mrope(queue, workspace.tmp2.data(), workspace.tmp1.data(), positions,
                       seq, c.num_attention_heads, c.num_key_value_heads,
                       c.head_dim, c.rotary_dim(), c.rope.theta, c.rope.mrope_section);

    size_t offset = (size_t)past * key_value_dim;
    size_t count = (size_t)seq * key_value_dim;
    queue.memcpy(cache.key.data() + offset, workspace.tmp1.data(), count * sizeof(bf16));
    queue.memcpy(cache.value.data() + offset, workspace.tmp4.data(), count * sizeof(bf16));
    cache.filled = past + seq;

    if (qwen35_xmx_attention_enabled()) {
        qwen35_xmx_attention(
            queue, workspace.tmp2.data(), cache.key.data(), cache.value.data(),
            workspace.tmp2.data(), seq, past, c.num_attention_heads,
            c.num_key_value_heads, c.head_dim,
            1.0f / std::sqrt((float)c.head_dim));
    } else if (qwen35_subgroup_attention_enabled()) {
        qwen35_online_attention_subgroup(
            queue, workspace.tmp2.data(), cache.key.data(), cache.value.data(),
            workspace.tmp2.data(), seq, past, c.num_attention_heads,
            c.num_key_value_heads, c.head_dim,
            1.0f / std::sqrt((float)c.head_dim));
    } else {
        qwen35_online_attention(queue, workspace.tmp2.data(), cache.key.data(),
                                cache.value.data(), workspace.tmp2.data(), seq, past,
                                c.num_attention_heads, c.num_key_value_heads,
                                c.head_dim, 1.0f / std::sqrt((float)c.head_dim));
    }
    mul_sigmoid_inplace(queue, workspace.tmp2.data(), workspace.tmp3.data(),
                        (size_t)seq * query_dim);
    matmul_proj(workspace.tmp2.data(), seq, query_dim, weights.o_proj, output, context);
}

inline void qwen35_linear_attention_forward(
    GpuEngine& context,
    const Qwen35LinearAttentionWeights& weights,
    Qwen35DeltaLayerCache& cache,
    Qwen35Workspace& workspace,
    const bf16* hidden,
    bf16* output,
    int seq,
    const Qwen35Config& config) {
    const auto& c = config.text;
    auto& queue = context.queue;
    int key_dim = c.linear_num_key_heads * c.linear_key_head_dim;
    int value_dim = c.linear_num_value_heads * c.linear_value_head_dim;
    int conv_dim = 2 * key_dim + value_dim;
    int heads = c.linear_num_value_heads;
    size_t head_values = (size_t)seq * heads;

    int projected_stride = conv_dim;
    if (weights.fused_projections) {
        projected_stride = conv_dim + value_dim;
        matmul_proj(hidden, seq, c.hidden_size, weights.in_proj_qkvz,
                    workspace.tmp0.data(), context);
    } else {
        matmul_proj(hidden, seq, c.hidden_size, weights.in_proj_qkv,
                    workspace.tmp0.data(), context);
    }
    // The fused decode core handles one token, so a short batch runs it once
    // per token rather than falling through to the chunked path below. Batch
    // size then stops selecting between two implementations that do not agree
    // numerically: before this, a two-token forward and two one-token forwards
    // over the same tokens produced different logits, which is visible as soon
    // as anything verifies a batch against sequential decoding.
    //
    // Only the recurrent core loops. The projection above is already batched
    // over the whole window, so this re-reads the recurrent state and the small
    // conv/gate tensors, not the weights.
    // qwen35_esimd_delta_enabled() belongs in this condition even though this
    // arm never calls qwen35_recurrent_delta_esimd.
    //
    // Both arms advance the same cache.recurrent_state buffer, and the two
    // recurrent implementations do not agree on its layout: the scalar kernel
    // stores [head][key][value], while the ESIMD kernel and the fused decode
    // core below both store [head][value][key]. K and V are equal at 128 here,
    // so a mismatch is exactly size-compatible - nothing faults, the state is
    // silently transposed between the prefill that wrote it and the decode
    // that reads it.
    //
    // Without this term, ARCAINE_QWEN35_ESIMD_DELTA=0 selects the scalar
    // kernel for prefill while leaving the fused ESIMD core on for decode, and
    // the two corrupt each other. Measured over 400 records that arm agreed
    // with the default on 33.75% of tokens, against 92% or better for every
    // other flag - and the fp64 oracle says both kernels are individually
    // correct, so the fault was never in either kernel.
    //
    // The flag now switches the whole DeltaNet path coherently, which is what
    // a baseline A/B switch should do. The deeper fix is to give the scalar
    // kernel the same [head][value][key] layout so no combination can mix
    // them; this makes the unsafe combination unreachable in the meantime.
    if (seq >= 1 && seq <= qwen35_fused_decode_max_seq() &&
        weights.fused_projections && qwen35_fused_esimd_delta_decode_enabled() &&
        qwen35_esimd_delta_enabled()) {
        for (int token = 0; token < seq; ++token) {
            const bf16* token_hidden = hidden + (size_t)token * c.hidden_size;
            const bf16* projected =
                workspace.tmp0.data() + (size_t)token * projected_stride;
            bf16* ba = workspace.tmp1.data() + (size_t)token * 2 * heads;
            bf16* core = workspace.tmp4.data() + (size_t)token * value_dim;
            bf16* gate = workspace.tmp2.data() + (size_t)token * value_dim;

            // Kept at M=1 so a token sees the same projection arithmetic it
            // would have seen decoding alone. These weights are a few hundred
            // KB against the tens of GB a decode step already moves.
            if (qwen35_fused_ba_projection_enabled())
                matmul_bf16(token_hidden, 1, c.hidden_size,
                            weights.in_proj_ba.data(), 2 * heads, ba, context);
            else {
                matmul_bf16(token_hidden, 1, c.hidden_size,
                            weights.in_proj_b.data(), heads, ba, context);
                matmul_bf16(token_hidden, 1, c.hidden_size,
                            weights.in_proj_a.data(), heads, ba + heads, context);
            }
            qwen35_delta_decode_fused_esimd(
                queue, projected, projected_stride,
                weights.conv1d_time_major.data(), cache.conv_state.data(),
                weights.A_log.data(), weights.dt_bias.data(), ba,
                cache.recurrent_state.data(), core, gate,
                c.linear_num_key_heads, heads, c.linear_key_head_dim,
                c.linear_value_head_dim, conv_dim, c.linear_conv_kernel_dim,
                c.rms_norm_eps);
            // Shifts this token into the history before the next reads it. The
            // in-order queue keeps the core and the update interleaved.
            qwen35_update_conv_state_time_major(queue, projected,
                                                projected_stride,
                                                cache.conv_state.data(), conv_dim);
        }
        cache.has_state = true;
        gated_rmsnorm(
            queue, workspace.tmp4.data(), workspace.tmp2.data(),
            weights.norm.data(), workspace.tmp4.data(), seq * heads,
            c.linear_value_head_dim, c.rms_norm_eps);
        matmul_proj(workspace.tmp4.data(), seq, value_dim, weights.out_proj,
                    output, context);
        return;
    }
    qwen35_conv_causal(queue, workspace.tmp0.data(), weights.conv1d.data(),
                       cache.conv_state.data(), workspace.tmp1.data(), seq,
                       conv_dim, c.linear_conv_kernel_dim, cache.has_state,
                       projected_stride);
    qwen35_update_conv_state(queue, workspace.tmp0.data(), cache.conv_state.data(),
                             seq, conv_dim, c.linear_conv_kernel_dim, cache.has_state,
                             projected_stride);
    qwen35_extract_qkv(queue, workspace.tmp1.data(), workspace.tmp2.data(),
                       workspace.tmp3.data(), workspace.tmp4.data(), seq,
                       c.linear_num_key_heads, heads, c.linear_key_head_dim,
                       c.linear_value_head_dim);
    l2norm(queue, workspace.tmp2.data(), workspace.tmp2.data(), seq * heads,
           c.linear_key_head_dim, c.rms_norm_eps);
    l2norm(queue, workspace.tmp3.data(), workspace.tmp3.data(), seq * heads,
           c.linear_key_head_dim, c.rms_norm_eps);
    scale_inplace(queue, workspace.tmp2.data(),
                  (size_t)seq * heads * c.linear_key_head_dim,
                  1.0f / std::sqrt((float)c.linear_key_head_dim));

    // The unfused path reuses tmp0 for z. The fused path keeps z at the tail
    // of each projected row until the recurrent core has consumed q/k/v.
    if (!weights.fused_projections)
        matmul_proj(hidden, seq, c.hidden_size, weights.in_proj_z,
                    workspace.tmp0.data(), context);
    bf16* beta = workspace.tmp1.data();
    bf16* g = workspace.tmp1.data() + head_values;
    // Two matmuls, deliberately, even though a fused [b|a] weight exists.
    //
    // The fused weight stacks b's rows then a's, so C(seq, 2*heads) comes back
    // row-major with each token's b and a adjacent: token t occupies
    // [t*2*heads, (t+1)*2*heads). The consumers below index beta and g as
    // [token * heads + head] over two separate blocks, which describes the
    // same bytes only when seq == 1. Using the fused matmul here therefore fed
    // the recurrence a's values as beta for every token after the first.
    //
    // It survived because the decode path above is per-token M=1, where the
    // two layouts coincide, and because the damage is a plausible-looking
    // perturbation rather than an obvious break. It also made the engine
    // nondeterministic: see notes/qwen3_5_27b/nondeterminism.md.
    //
    // These are hidden_size x heads GEMMs, launch-bound at any sequence
    // length, so splitting them costs nothing worth measuring.
    matmul_bf16(hidden, seq, c.hidden_size, weights.in_proj_b.data(), heads,
                beta, context);
    matmul_bf16(hidden, seq, c.hidden_size, weights.in_proj_a.data(), heads,
                g, context);
    sigmoid_inplace(queue, beta, head_values);
    qwen35_compute_g(queue, g, weights.A_log.data(), weights.dt_bias.data(),
                     g, seq, heads);
    if (qwen35_esimd_delta_enabled()) {
        qwen35_recurrent_delta_esimd(
            queue, workspace.tmp2.data(), workspace.tmp3.data(),
            workspace.tmp4.data(), beta, g, cache.recurrent_state.data(),
            workspace.tmp4.data(), seq, heads, c.linear_key_head_dim,
            c.linear_value_head_dim);
    } else {
        qwen35_recurrent_delta(queue, workspace.tmp2.data(), workspace.tmp3.data(),
                               workspace.tmp4.data(), beta, g,
                               cache.recurrent_state.data(), workspace.tmp4.data(),
                               seq, heads, c.linear_key_head_dim,
                               c.linear_value_head_dim);
    }
    cache.has_state = true;
    bf16* z = workspace.tmp0.data();
    if (weights.fused_projections) {
        qwen35_copy_strided(queue, workspace.tmp0.data(), projected_stride,
                            conv_dim, workspace.tmp1.data(), seq, value_dim);
        z = workspace.tmp1.data();
    }
    gated_rmsnorm(queue, workspace.tmp4.data(), z,
                  weights.norm.data(), workspace.tmp4.data(), seq * heads,
                  c.linear_value_head_dim, c.rms_norm_eps);
    matmul_proj(workspace.tmp4.data(), seq, value_dim, weights.out_proj,
                output, context);
}

inline void qwen35_mlp_forward(
    GpuEngine& context,
    const Qwen35MlpWeights& weights,
    Qwen35Workspace& workspace,
    const bf16* hidden,
    bf16* output,
    int seq,
    const Qwen35Config& config) {
    int H = config.text.hidden_size;
    int I = config.text.intermediate_size;
    auto& queue = context.queue;
    if (std::holds_alternative<Int4Linear>(weights.gate_up)) {
        const auto& gate_up = std::get<Int4Linear>(weights.gate_up);
        const auto& down = std::get<Int4Linear>(weights.down);
        matmul_int4(hidden, seq, H, gate_up, workspace.tmp0.data(), context);
        swiglu_strided(queue, workspace.tmp0.data(), workspace.tmp1.data(), seq, I);
        matmul_int4(workspace.tmp1.data(), seq, I, down, output, context);
    } else if (std::holds_alternative<Nvfp4Linear>(weights.gate_up)) {
        const auto& gate_up = std::get<Nvfp4Linear>(weights.gate_up);
        const auto& down = std::get<Nvfp4Linear>(weights.down);
        if (seq == 1 && qwen35_nvfp4_decode_gemv_enabled()) {
            // M=1 GEMV specialization. The general f4 path is the whole decode
            // bandwidth gap: measured per layer group, the NVFP4 layers run at
            // ~36% of achievable bandwidth against ~88% for the checkpoint's
            // FP8 layers, which move nearly half again as many bytes and are
            // still 1.69x faster per layer. Neither the Xe2 pack kernel nor
            // oneDNN's weight-layout reorder moves that number, because both
            // are shaped for large M.
            //
            // These ESIMD kernels stream the packed weights one output row per
            // work-item, which is what a decode GEMV wants. Same weights, same
            // scales, same activation quantization as the general path, so this
            // is a scheduling change and not a numerical one.
            pack_bf16_to_nvfp4(queue, hidden, workspace.input_packed.data(),
                               workspace.input_scale.data(), 1, H,
                               gate_up.input_global_scale);
            int wg = qwen35_nvfp4_decode_gemv_wg();
            matmul_nvfp4_decode_swiglu_esimd(
                workspace.input_packed.data(), workspace.input_scale.data(), H,
                gate_up, workspace.tmp1.data(), I, context, wg);
            pack_bf16_to_nvfp4(queue, workspace.tmp1.data(),
                               workspace.activation_packed.data(),
                               workspace.activation_scale.data(), 1, I,
                               down.input_global_scale);
            matmul_nvfp4_decode_gemv_esimd(
                workspace.activation_packed.data(),
                workspace.activation_scale.data(), I, down, output, context, wg);
        } else if (qwen35_nvfp4_dpas_enabled()) {
            pack_bf16_to_nvfp4(queue, hidden, workspace.input_packed.data(),
                               workspace.input_scale.data(), seq, H,
                               gate_up.input_global_scale);
            matmul_nvfp4_swiglu_pack_xe2(
                context, workspace.input_packed.data(), workspace.input_scale.data(),
                seq, H, gate_up, down, workspace.activation_packed.data(),
                workspace.activation_scale.data());
            matmul_nvfp4_packed_xe2(
                context, workspace.activation_packed.data(),
                workspace.activation_scale.data(), seq, I, down, output);
        } else {
            matmul_nvfp4(hidden, seq, H, gate_up, workspace.tmp0.data(), context,
                         workspace.input_packed.data(), workspace.input_scale.data());
            swiglu_strided(queue, workspace.tmp0.data(), workspace.tmp1.data(), seq, I);
            matmul_nvfp4(workspace.tmp1.data(), seq, I, down, output, context,
                         workspace.activation_packed.data(),
                         workspace.activation_scale.data());
        }
    } else {
        const auto& gate_up = std::get<Fp8Linear>(weights.gate_up);
        const auto& down = std::get<Fp8Linear>(weights.down);
        matmul_fp8(hidden, seq, H, gate_up, workspace.tmp0.data(), context);
        swiglu_strided(queue, workspace.tmp0.data(), workspace.tmp1.data(), seq, I);
        matmul_fp8(workspace.tmp1.data(), seq, I, down, output, context);
    }
}
