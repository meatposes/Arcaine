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
#include "../../common/gpu/fp8.hpp"
#include "../../common/gpu/nvfp4.hpp"
#include "../../common/gpu/ops.hpp"
#include "../../common/kernels/elementwise.hpp"
#include "../../common/kernels/rms_norm.hpp"

// Above this M, expanding an NVFP4 weight to BF16 and running the library BF16
// GEMM beats oneDNN's f4 matmul, which sustains only ~31-33 TFLOP/s against
// ~129-150 for bf16 on Xe2. Measured crossover on Arc Pro B60 sits between M=64
// (f4 wins 2.7x) and M=512 (dequant wins 1.7x); 2048 gives 3.0x and 5888 gives
// 3.7x. Default 256 sits inside the unmeasured gap, deliberately conservative:
// being late to switch costs a little prefill, being early costs decode.
//
// On by default, on end-to-end evidence rather than the kernel bench: 3 reps of
// arcaine_mbench give pp512 515 -> 781, pp2048 491 -> 983, pp4096 440 -> 826,
// pp8192 341 -> 553 tok/s, with decode identical to three digits at every depth
// because M=1 stays below the threshold. That distinction matters here --
// selecting the attention kernel by phase looked like a 20% win in isolation and
// measured 2% slower in the model, so the isolated bench is not sufficient.
//
// Greedy output is unchanged on a prompt that crosses the threshold, and the
// bench's --check puts weight rounding at rel_rms <= 2.5e-05.
inline bool qwen35_prefill_dequant_bf16_enabled() {
    static bool enabled = [] {
        const char* value = std::getenv("ARCAINE_QWEN35_PREFILL_DEQUANT_BF16");
        if (!value) return true;
        return std::strcmp(value, "0") != 0 && std::strcmp(value, "off") != 0 &&
               std::strcmp(value, "false") != 0 && std::strcmp(value, "no") != 0;
    }();
    return enabled;
}

// Cap on the expansion of a single weight. The scratch holds one projection at
// a time, but "one projection" is checkpoint-dependent: gate_up expands to
// 357 MB here, while an NVFP4 LM head on this vocabulary would need 2.5 GB.
// Weights above the cap keep the f4 path rather than silently claiming VRAM
// that the KV cache needs.
inline size_t qwen35_dequant_bf16_max_bytes() {
    static size_t bytes = [] {
        const char* value = std::getenv("ARCAINE_QWEN35_DEQUANT_BF16_MAX_MB");
        long parsed = value ? std::atol(value) : 0;
        return (size_t)(parsed > 0 ? parsed : 512) * 1024u * 1024u;
    }();
    return bytes;
}

inline int qwen35_dequant_bf16_min_m() {
    static int threshold = [] {
        const char* value = std::getenv("ARCAINE_QWEN35_DEQUANT_BF16_MIN_M");
        int parsed = value ? std::atoi(value) : 0;
        return parsed > 0 ? parsed : 256;
    }();
    return threshold;
}

// Single entry point for every projection in the model. The kind was decided by
// probing the checkpoint at load time, so the forward path never has to know
// which recipe produced the weights it was given.
inline void qwen35_matmul(
    const bf16* A,
    int M,
    int K,
    const Qwen35Linear& weights,
    bf16* C,
    GpuEngine& context,
    uint8_t* packed_scratch = nullptr,
    uint8_t* scale_scratch = nullptr,
    bf16* dequant_scratch = nullptr,
    size_t dequant_capacity = 0) {
    switch (weights.kind) {
        case Qwen35Linear::Kind::Nvfp4:
            if (dequant_scratch && M >= qwen35_dequant_bf16_min_m() &&
                (size_t)weights.in_features * (size_t)weights.out_features <=
                    dequant_capacity &&
                qwen35_prefill_dequant_bf16_enabled()) {
                dequantize_nvfp4_to_bf16(context.queue, weights.nvfp4,
                                         dequant_scratch);
                matmul_bf16(A, M, K, dequant_scratch, weights.out_features, C,
                            context);
                return;
            }
            matmul_nvfp4(A, M, K, weights.nvfp4, C, context, packed_scratch,
                         scale_scratch);
            return;
        case Qwen35Linear::Kind::Fp8:
            matmul_fp8(A, M, K, weights.fp8, C, context);
            return;
        case Qwen35Linear::Kind::Bf16:
            matmul_bf16(A, M, K, weights.dense.data(), weights.out_features, C,
                        context);
            return;
        case Qwen35Linear::Kind::Missing:
            break;
    }
    throw std::runtime_error("Qwen3.5 projection was never loaded");
}

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

    uint8_t* packed = workspace.input_packed.data();
    uint8_t* packed_scale = workspace.input_scale.data();
    if (weights.fused_projections) {
        qwen35_matmul(hidden, seq, c.hidden_size, weights.qkv_proj,
                      workspace.tmp0.data(), context, packed, packed_scale,
                      workspace.dequant_weight.data(),
                      workspace.dequant_weight.count());
        qwen35_split_q_gate_kv(
            queue, workspace.tmp0.data(), workspace.tmp2.data(),
            workspace.tmp3.data(), workspace.tmp1.data(), workspace.tmp4.data(),
            seq, c.num_attention_heads, c.num_key_value_heads, c.head_dim);
    } else {
        qwen35_matmul(hidden, seq, c.hidden_size, weights.q_proj,
                      workspace.tmp0.data(), context, packed, packed_scale,
                      workspace.dequant_weight.data(),
                      workspace.dequant_weight.count());
        qwen35_split_q_gate(queue, workspace.tmp0.data(), workspace.tmp2.data(),
                            workspace.tmp3.data(), seq, c.num_attention_heads,
                            c.head_dim);
        qwen35_matmul(hidden, seq, c.hidden_size, weights.k_proj,
                      workspace.tmp1.data(), context, packed, packed_scale,
                      workspace.dequant_weight.data(),
                      workspace.dequant_weight.count());
        qwen35_matmul(hidden, seq, c.hidden_size, weights.v_proj,
                      workspace.tmp4.data(), context, packed, packed_scale,
                      workspace.dequant_weight.data(),
                      workspace.dequant_weight.count());
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
    qwen35_matmul(workspace.tmp2.data(), seq, query_dim, weights.o_proj, output,
                  context, packed, packed_scale,
                      workspace.dequant_weight.data(),
                      workspace.dequant_weight.count());
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

    uint8_t* packed = workspace.input_packed.data();
    uint8_t* packed_scale = workspace.input_scale.data();
    int projected_stride = conv_dim;
    if (weights.fused_projections) {
        projected_stride = conv_dim + value_dim;
        qwen35_matmul(hidden, seq, c.hidden_size, weights.in_proj_qkvz,
                      workspace.tmp0.data(), context, packed, packed_scale,
                      workspace.dequant_weight.data(),
                      workspace.dequant_weight.count());
    } else {
        qwen35_matmul(hidden, seq, c.hidden_size, weights.in_proj_qkv,
                      workspace.tmp0.data(), context, packed, packed_scale,
                      workspace.dequant_weight.data(),
                      workspace.dequant_weight.count());
    }
    if (seq == 1 && weights.fused_projections &&
        qwen35_fused_esimd_delta_decode_enabled()) {
        if (qwen35_fused_ba_projection_enabled())
            matmul_bf16(hidden, 1, c.hidden_size, weights.in_proj_ba.data(),
                        2 * heads, workspace.tmp1.data(), context);
        else {
            matmul_bf16(hidden, 1, c.hidden_size, weights.in_proj_b.data(), heads,
                        workspace.tmp1.data(), context);
            matmul_bf16(hidden, 1, c.hidden_size, weights.in_proj_a.data(), heads,
                        workspace.tmp1.data() + heads, context);
        }
        qwen35_delta_decode_fused_esimd(
            queue, workspace.tmp0.data(), projected_stride,
            weights.conv1d_time_major.data(), cache.conv_state.data(),
            weights.A_log.data(), weights.dt_bias.data(), workspace.tmp1.data(),
            cache.recurrent_state.data(), workspace.tmp4.data(),
            workspace.tmp2.data(), c.linear_num_key_heads, heads,
            c.linear_key_head_dim, c.linear_value_head_dim, conv_dim,
            c.linear_conv_kernel_dim, c.rms_norm_eps);
        qwen35_update_conv_state_time_major(
            queue, workspace.tmp0.data(), projected_stride,
            cache.conv_state.data(), conv_dim);
        cache.has_state = true;
        gated_rmsnorm(
            queue, workspace.tmp4.data(), workspace.tmp2.data(),
            weights.norm.data(), workspace.tmp4.data(), heads,
            c.linear_value_head_dim, c.rms_norm_eps);
        qwen35_matmul(workspace.tmp4.data(), 1, value_dim, weights.out_proj,
                      output, context, packed, packed_scale,
                      workspace.dequant_weight.data(),
                      workspace.dequant_weight.count());
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
        qwen35_matmul(hidden, seq, c.hidden_size, weights.in_proj_z,
                      workspace.tmp0.data(), context, packed, packed_scale,
                      workspace.dequant_weight.data(),
                      workspace.dequant_weight.count());
    bf16* beta = workspace.tmp1.data();
    bf16* g = workspace.tmp1.data() + head_values;
    if (qwen35_fused_ba_projection_enabled())
        matmul_bf16(hidden, seq, c.hidden_size, weights.in_proj_ba.data(),
                    2 * heads, beta, context);
    else {
        matmul_bf16(hidden, seq, c.hidden_size, weights.in_proj_b.data(), heads,
                    beta, context);
        matmul_bf16(hidden, seq, c.hidden_size, weights.in_proj_a.data(), heads,
                    g, context);
    }
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
    qwen35_matmul(workspace.tmp4.data(), seq, value_dim, weights.out_proj,
                  output, context, packed, packed_scale,
                      workspace.dequant_weight.data(),
                      workspace.dequant_weight.count());
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
    // The Xe2 pack-fused variant consumes the NVFP4 weights directly and only
    // exists for an all-NVFP4 MLP; everything else goes through the shared
    // projection dispatch.
    if (weights.gate_up.kind == Qwen35Linear::Kind::Nvfp4 &&
        weights.down.kind == Qwen35Linear::Kind::Nvfp4 &&
        qwen35_nvfp4_dpas_enabled()) {
        const Nvfp4Linear& gate_up = weights.gate_up.nvfp4;
        const Nvfp4Linear& down = weights.down.nvfp4;
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
        return;
    }
    qwen35_matmul(hidden, seq, H, weights.gate_up, workspace.tmp0.data(), context,
                  workspace.input_packed.data(), workspace.input_scale.data(),
                  workspace.dequant_weight.data(),
                      workspace.dequant_weight.count());
    swiglu_strided(queue, workspace.tmp0.data(), workspace.tmp1.data(), seq, I);
    qwen35_matmul(workspace.tmp1.data(), seq, I, weights.down, output, context,
                  workspace.activation_packed.data(),
                  workspace.activation_scale.data(),
                  workspace.dequant_weight.data(),
                      workspace.dequant_weight.count());
}
