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
#include "../../utils/profile.hpp"

// Prefill is bound by compute and decode by weight bandwidth, so a breakdown
// that merges them hides the shape of both. Every scope below is tagged with
// the phase it ran in; pick the label from the sequence length.
inline const char* qwen35_phase(int seq, const char* prefill, const char* decode) {
    return seq > 1 ? prefill : decode;
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

// One selector for the three full-attention kernels, replacing two overlapping
// booleans that could express states nobody wanted (both set, neither set).
//
// Isolated, `arcaine_kbench qwen35-attention` on one Arc Pro B60 (q_heads=24,
// kv_heads=4, head_dim=256) ranks them:
//
//   phase             xmx      subgroup   baseline
//   prefill q=512    2.77 ms   12.05 ms   13.68 ms
//   decode  kv=513   0.31 ms    0.25 ms    0.47 ms
//   decode  kv=2049  1.19 ms    0.97 ms    1.87 ms
//
// which says to run subgroup at decode and XMX at prefill. End to end it does
// not hold. Selecting subgroup for seq==1 on this checkpoint measures *slower*
// at every KV depth -- 9.06 / 8.48 / 7.50 tok/s against XMX's 9.14 / 8.62 /
// 7.72 at depths 512 / 1024 / 2048 -- consistently, in the direction opposite
// to the kernel benchmark. In the real forward the kernel is one of several
// enqueued back to back rather than run alone behind a queue wait, and the
// isolated ranking does not survive that.
//
// So Auto stays on XMX for both phases: the measurement that counts is the one
// on the whole model. Worth revisiting if decode attention is rewritten, since
// both kernels are far off memory bandwidth for what they read, and subgroup is
// exact at q=1 where XMX carries ~5% relative error from its bf16 accumulation.
enum class Qwen35AttentionKernel { Auto, Xmx, Subgroup, Baseline, ByPhase };

inline Qwen35AttentionKernel qwen35_attention_kernel() {
    static Qwen35AttentionKernel kernel = [] {
        const char* value = std::getenv("ARCAINE_QWEN35_ATTENTION_KERNEL");
        if (!value) return Qwen35AttentionKernel::Auto;
        if (std::strcmp(value, "xmx") == 0) return Qwen35AttentionKernel::Xmx;
        if (std::strcmp(value, "subgroup") == 0) return Qwen35AttentionKernel::Subgroup;
        if (std::strcmp(value, "baseline") == 0) return Qwen35AttentionKernel::Baseline;
        if (std::strcmp(value, "by-phase") == 0) return Qwen35AttentionKernel::ByPhase;
        return Qwen35AttentionKernel::Auto;
    }();
    return kernel;
}

inline Qwen35AttentionKernel qwen35_attention_kernel_for(int seq) {
    Qwen35AttentionKernel kernel = qwen35_attention_kernel();
    if (kernel == Qwen35AttentionKernel::Auto) return Qwen35AttentionKernel::Xmx;
    if (kernel == Qwen35AttentionKernel::ByPhase)
        return seq > 1 ? Qwen35AttentionKernel::Xmx : Qwen35AttentionKernel::Subgroup;
    return kernel;
}

inline const char* qwen35_attention_kernel_name() {
    switch (qwen35_attention_kernel()) {
        case Qwen35AttentionKernel::Xmx:      return "xmx";
        case Qwen35AttentionKernel::Subgroup: return "subgroup";
        case Qwen35AttentionKernel::Baseline: return "baseline";
        case Qwen35AttentionKernel::ByPhase:  return "by-phase (xmx prefill, subgroup decode)";
        case Qwen35AttentionKernel::Auto:     break;
    }
    return "auto (xmx)";
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

    DIFF_PROF(queue, qwen35_phase(seq, "pp.attn", "tg.attn"));
    {
    DIFF_PROF(queue, qwen35_phase(seq, "pp.attn.qkv_proj", "tg.attn.qkv_proj"));
    if (weights.fused_projections) {
        matmul_fp8(hidden, seq, c.hidden_size, weights.qkv_proj,
                   workspace.tmp0.data(), context);
        qwen35_split_q_gate_kv(
            queue, workspace.tmp0.data(), workspace.tmp2.data(),
            workspace.tmp3.data(), workspace.tmp1.data(), workspace.tmp4.data(),
            seq, c.num_attention_heads, c.num_key_value_heads, c.head_dim);
    } else {
        matmul_fp8(hidden, seq, c.hidden_size, weights.q_proj,
                   workspace.tmp0.data(), context);
        qwen35_split_q_gate(queue, workspace.tmp0.data(), workspace.tmp2.data(),
                            workspace.tmp3.data(), seq, c.num_attention_heads,
                            c.head_dim);
        matmul_fp8(hidden, seq, c.hidden_size, weights.k_proj,
                   workspace.tmp1.data(), context);
        matmul_fp8(hidden, seq, c.hidden_size, weights.v_proj,
                   workspace.tmp4.data(), context);
    }
    }
    {
    DIFF_PROF(queue, qwen35_phase(seq, "pp.attn.norm_rope", "tg.attn.norm_rope"));
    rms_norm(queue, workspace.tmp2.data(), weights.q_norm.data(), workspace.tmp2.data(),
             seq * c.num_attention_heads, c.head_dim, c.rms_norm_eps);
    rms_norm(queue, workspace.tmp1.data(), weights.k_norm.data(), workspace.tmp1.data(),
             seq * c.num_key_value_heads, c.head_dim, c.rms_norm_eps);
    qwen35_apply_mrope(queue, workspace.tmp2.data(), workspace.tmp1.data(), positions,
                       seq, c.num_attention_heads, c.num_key_value_heads,
                       c.head_dim, c.rotary_dim(), c.rope.theta, c.rope.mrope_section);
    }

    {
    DIFF_PROF(queue, qwen35_phase(seq, "pp.attn.kv_write", "tg.attn.kv_write"));
    size_t offset = (size_t)past * key_value_dim;
    size_t count = (size_t)seq * key_value_dim;
    queue.memcpy(cache.key.data() + offset, workspace.tmp1.data(), count * sizeof(bf16));
    queue.memcpy(cache.value.data() + offset, workspace.tmp4.data(), count * sizeof(bf16));
    cache.filled = past + seq;
    }

    {
    DIFF_PROF(queue, qwen35_phase(seq, "pp.attn.core", "tg.attn.core"));
    Qwen35AttentionKernel kernel = qwen35_attention_kernel_for(seq);
    if (kernel == Qwen35AttentionKernel::Xmx) {
        qwen35_xmx_attention(
            queue, workspace.tmp2.data(), cache.key.data(), cache.value.data(),
            workspace.tmp2.data(), seq, past, c.num_attention_heads,
            c.num_key_value_heads, c.head_dim,
            1.0f / std::sqrt((float)c.head_dim));
    } else if (kernel == Qwen35AttentionKernel::Subgroup) {
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
    }
    DIFF_PROF(queue, qwen35_phase(seq, "pp.attn.o_proj", "tg.attn.o_proj"));
    matmul_fp8(workspace.tmp2.data(), seq, query_dim, weights.o_proj, output, context);
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

    DIFF_PROF(queue, qwen35_phase(seq, "pp.linear_attn", "tg.linear_attn"));
    int projected_stride = conv_dim;
    {
    DIFF_PROF(queue, qwen35_phase(seq, "pp.linear_attn.in_proj", "tg.linear_attn.in_proj"));
    if (weights.fused_projections) {
        projected_stride = conv_dim + value_dim;
        matmul_fp8(hidden, seq, c.hidden_size, weights.in_proj_qkvz,
                   workspace.tmp0.data(), context);
    } else {
        matmul_fp8(hidden, seq, c.hidden_size, weights.in_proj_qkv,
                   workspace.tmp0.data(), context);
    }
    }
    if (seq == 1 && weights.fused_projections &&
        qwen35_fused_esimd_delta_decode_enabled()) {
        DIFF_PROF(queue, "tg.linear_attn.fused_decode");
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
        matmul_fp8(workspace.tmp4.data(), 1, value_dim, weights.out_proj,
                   output, context);
        return;
    }
    {
    DIFF_PROF(queue, qwen35_phase(seq, "pp.linear_attn.conv", "tg.linear_attn.conv"));
    qwen35_conv_causal(queue, workspace.tmp0.data(), weights.conv1d.data(),
                       cache.conv_state.data(), workspace.tmp1.data(), seq,
                       conv_dim, c.linear_conv_kernel_dim, cache.has_state,
                       projected_stride);
    qwen35_update_conv_state(queue, workspace.tmp0.data(), cache.conv_state.data(),
                             seq, conv_dim, c.linear_conv_kernel_dim, cache.has_state,
                             projected_stride);
    }
    {
    DIFF_PROF(queue, qwen35_phase(seq, "pp.linear_attn.qkv_norm", "tg.linear_attn.qkv_norm"));
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
    }

    {
    DIFF_PROF(queue, qwen35_phase(seq, "pp.linear_attn.z_ba_proj", "tg.linear_attn.z_ba_proj"));
    // The unfused path reuses tmp0 for z. The fused path keeps z at the tail
    // of each projected row until the recurrent core has consumed q/k/v.
    if (!weights.fused_projections)
        matmul_fp8(hidden, seq, c.hidden_size, weights.in_proj_z,
                   workspace.tmp0.data(), context);
    bf16* beta_local = workspace.tmp1.data();
    bf16* g_local = workspace.tmp1.data() + head_values;
    if (qwen35_fused_ba_projection_enabled())
        matmul_bf16(hidden, seq, c.hidden_size, weights.in_proj_ba.data(),
                    2 * heads, beta_local, context);
    else {
        matmul_bf16(hidden, seq, c.hidden_size, weights.in_proj_b.data(), heads,
                    beta_local, context);
        matmul_bf16(hidden, seq, c.hidden_size, weights.in_proj_a.data(), heads,
                    g_local, context);
    }
    sigmoid_inplace(queue, beta_local, head_values);
    qwen35_compute_g(queue, g_local, weights.A_log.data(), weights.dt_bias.data(),
                     g_local, seq, heads);
    }
    bf16* beta = workspace.tmp1.data();
    bf16* g = workspace.tmp1.data() + head_values;
    {
    DIFF_PROF(queue, qwen35_phase(seq, "pp.linear_attn.delta", "tg.linear_attn.delta"));
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
    }
    cache.has_state = true;
    {
    DIFF_PROF(queue, qwen35_phase(seq, "pp.linear_attn.gated_norm", "tg.linear_attn.gated_norm"));
    bf16* z = workspace.tmp0.data();
    if (weights.fused_projections) {
        qwen35_copy_strided(queue, workspace.tmp0.data(), projected_stride,
                            conv_dim, workspace.tmp1.data(), seq, value_dim);
        z = workspace.tmp1.data();
    }
    gated_rmsnorm(queue, workspace.tmp4.data(), z,
                  weights.norm.data(), workspace.tmp4.data(), seq * heads,
                  c.linear_value_head_dim, c.rms_norm_eps);
    }
    DIFF_PROF(queue, qwen35_phase(seq, "pp.linear_attn.out_proj", "tg.linear_attn.out_proj"));
    matmul_fp8(workspace.tmp4.data(), seq, value_dim, weights.out_proj,
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
    DIFF_PROF(queue, qwen35_phase(seq, "pp.mlp", "tg.mlp"));
    if (weights.nvfp4) {
        const auto& gate_up = std::get<Nvfp4Linear>(weights.gate_up);
        const auto& down = std::get<Nvfp4Linear>(weights.down);
        if (qwen35_nvfp4_dpas_enabled()) {
            {
            DIFF_PROF(queue, qwen35_phase(seq, "pp.mlp.pack", "tg.mlp.pack"));
            pack_bf16_to_nvfp4(queue, hidden, workspace.input_packed.data(),
                               workspace.input_scale.data(), seq, H,
                               gate_up.input_global_scale);
            }
            {
            DIFF_PROF(queue, qwen35_phase(seq, "pp.mlp.gate_up", "tg.mlp.gate_up"));
            matmul_nvfp4_swiglu_pack_xe2(
                context, workspace.input_packed.data(), workspace.input_scale.data(),
                seq, H, gate_up, down, workspace.activation_packed.data(),
                workspace.activation_scale.data());
            }
            DIFF_PROF(queue, qwen35_phase(seq, "pp.mlp.down", "tg.mlp.down"));
            matmul_nvfp4_packed_xe2(
                context, workspace.activation_packed.data(),
                workspace.activation_scale.data(), seq, I, down, output);
        } else {
            // Spelled out as pack + packed-matmul rather than calling
            // matmul_nvfp4, which does exactly these two steps internally. The
            // split is the point: W4A4 re-quantizes the activations for every
            // projection, and that cost has to be visible separately from the
            // GEMM it feeds before it is worth trying to remove.
            {
            DIFF_PROF(queue, qwen35_phase(seq, "pp.mlp.pack", "tg.mlp.pack"));
            pack_bf16_to_nvfp4(queue, hidden, workspace.input_packed.data(),
                               workspace.input_scale.data(), seq, H,
                               gate_up.input_global_scale);
            }
            {
            DIFF_PROF(queue, qwen35_phase(seq, "pp.mlp.gate_up", "tg.mlp.gate_up"));
            matmul_nvfp4_packed(workspace.input_packed.data(),
                                workspace.input_scale.data(), seq, H, gate_up,
                                workspace.tmp0.data(), context);
            }
            {
            DIFF_PROF(queue, qwen35_phase(seq, "pp.mlp.swiglu", "tg.mlp.swiglu"));
            swiglu_strided(queue, workspace.tmp0.data(), workspace.tmp1.data(), seq, I);
            }
            {
            DIFF_PROF(queue, qwen35_phase(seq, "pp.mlp.pack", "tg.mlp.pack"));
            pack_bf16_to_nvfp4(queue, workspace.tmp1.data(),
                               workspace.activation_packed.data(),
                               workspace.activation_scale.data(), seq, I,
                               down.input_global_scale);
            }
            DIFF_PROF(queue, qwen35_phase(seq, "pp.mlp.down", "tg.mlp.down"));
            matmul_nvfp4_packed(workspace.activation_packed.data(),
                                workspace.activation_scale.data(), seq, I, down,
                                output, context);
        }
    } else {
        const auto& gate_up = std::get<Fp8Linear>(weights.gate_up);
        const auto& down = std::get<Fp8Linear>(weights.down);
        {
        DIFF_PROF(queue, qwen35_phase(seq, "pp.mlp.gate_up", "tg.mlp.gate_up"));
        matmul_fp8(hidden, seq, H, gate_up, workspace.tmp0.data(), context);
        }
        {
        DIFF_PROF(queue, qwen35_phase(seq, "pp.mlp.swiglu", "tg.mlp.swiglu"));
        swiglu_strided(queue, workspace.tmp0.data(), workspace.tmp1.data(), seq, I);
        }
        DIFF_PROF(queue, qwen35_phase(seq, "pp.mlp.down", "tg.mlp.down"));
        matmul_fp8(workspace.tmp1.data(), seq, I, down, output, context);
    }
}
