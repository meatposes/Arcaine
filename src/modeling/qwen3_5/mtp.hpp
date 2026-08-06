#pragma once

// Qwen3.5 multi-token-prediction head.
//
// The checkpoint ships 15 `mtp.*` tensors that the loader has always read into
// VRAM and the generation loop has never run. The head is one full-attention
// decoder layer that predicts the token two positions ahead: given the
// backbone's hidden state at position t and the token that occupies t+1, it
// produces the distribution over t+2. That is a draft model with no extra
// weights to download and no second checkpoint to keep in sync.
//
// The contract matches the reference implementation exactly, and two details
// of it are easy to get backwards:
//
//   embeds = pre_fc_norm_embedding(embed(x_{t+1}))
//   hidden = pre_fc_norm_hidden(h_t)
//   x      = fc(concat([embeds, hidden]))     <- embedding first, not hidden
//   x      = decoder_layer(x)
//   out    = mtp_norm(x)                      <- caller applies the shared lm_head
//
// `h_t` is the backbone hidden state *before* the model's final norm. Feeding
// the post-norm state normalizes twice and silently degrades the head rather
// than failing, so the caller must pass the pre-norm tensor.

#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#include "cache.hpp"
#include "config.hpp"
#include "kernels.hpp"
#include "operators.hpp"
#include "weights.hpp"
#include "workspace.hpp"
#include "../../runtime/gpu/buffer.hpp"
#include "../../runtime/gpu/engine.hpp"
#include "../../runtime/gpu/ops.hpp"
#include "runtime/kernels/elementwise.hpp"
#include "runtime/kernels/embedding.hpp"
#include "runtime/kernels/rms_norm.hpp"

using namespace qwen35_kernels;

// KV cache and scratch for the MTP layer. Kept separate from the backbone's
// caches: the head runs at different positions than the model it drafts for,
// and mixing the two would corrupt both.
struct Qwen35MtpState {
    GpuBuffer<bf16> key, value;
    int filled = 0;
    int capacity = 0;

    // Scratch. Sized by the draft window rather than the full context, because
    // the head only ever runs over a handful of positions at a time.
    int max_seq = 0;
    GpuBuffer<bf16> embeds;     // [seq, H]
    GpuBuffer<bf16> hidden;     // [seq, H]
    GpuBuffer<bf16> merged;     // [seq, 2H]
    GpuBuffer<bf16> residual;   // [seq, H]
    GpuBuffer<bf16> normed;     // [seq, H]
    GpuBuffer<bf16> sublayer;   // [seq, H]
    GpuBuffer<bf16> gate;       // [seq, I]
    GpuBuffer<bf16> up;         // [seq, I]

    void init(const Qwen35Config& config, int context_len, int draft_window,
              sycl::queue& queue) {
        const auto& c = config.text;
        size_t kv = (size_t)context_len * c.num_key_value_heads * c.head_dim;
        key = GpuBuffer<bf16>(kv, queue);
        value = GpuBuffer<bf16>(kv, queue);
        capacity = context_len;
        filled = 0;

        max_seq = draft_window;
        size_t s = (size_t)draft_window;
        embeds = GpuBuffer<bf16>(s * c.hidden_size, queue);
        hidden = GpuBuffer<bf16>(s * c.hidden_size, queue);
        merged = GpuBuffer<bf16>(s * 2 * c.hidden_size, queue);
        residual = GpuBuffer<bf16>(s * c.hidden_size, queue);
        normed = GpuBuffer<bf16>(s * c.hidden_size, queue);
        sublayer = GpuBuffer<bf16>(s * c.hidden_size, queue);
        gate = GpuBuffer<bf16>(s * c.intermediate_size, queue);
        up = GpuBuffer<bf16>(s * c.intermediate_size, queue);
    }

    void reset() { filled = 0; }
    bool ready() const { return !key.empty(); }
};

// Per-stage trace. Each step waits on the queue and prints, so a device hang
// is attributed to a specific kernel instead of to the enclosing call.
inline bool qwen35_mtp_trace_enabled() {
    static bool enabled = std::getenv("ARCAINE_QWEN35_MTP_TRACE") != nullptr;
    return enabled;
}

inline void qwen35_mtp_trace(sycl::queue& queue, const char* stage, int seq,
                             int past) {
    if (!qwen35_mtp_trace_enabled()) return;
    queue.wait();
    std::fprintf(stderr, "[mtp] seq=%d past=%d %s\n", seq, past, stage);
    std::fflush(stderr);
}

// [seq, H] + [seq, H] -> [seq, 2H], first source in the low half of each row.
inline void qwen35_mtp_concat(sycl::queue& queue, const bf16* first,
                              const bf16* second, bf16* out, int seq, int hidden) {
    size_t total = (size_t)seq * hidden;
    queue.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(total), [=](sycl::id<1> id) {
            size_t i = id[0];
            int token = (int)(i / hidden);
            int lane = (int)(i % hidden);
            size_t row = (size_t)token * 2 * hidden;
            out[row + lane] = first[i];
            out[row + hidden + lane] = second[i];
        });
    });
}

// out = silu(gate) * up, with gate and up held in separate buffers. The shared
// swiglu_strided expects one fused [seq, 2I] tensor; the MTP checkpoint keeps
// gate_proj and up_proj apart, so it gets its own kernel rather than a repack.
inline void qwen35_mtp_swiglu(sycl::queue& queue, const bf16* gate,
                              const bf16* up, bf16* out, size_t n) {
    queue.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
            float g = bf16_to_float(gate[id[0]]);
            float u = bf16_to_float(up[id[0]]);
            out[id[0]] = float_to_bf16((g / (1.0f + sycl::exp(-g))) * u);
        });
    });
}

// The MTP layer's attention. Same shape and same math as a backbone
// full-attention layer, including the sigmoid-gated query, but the weights are
// BF16 and the projections are never fused, so it cannot reuse
// qwen35_full_attention_forward directly.
inline void qwen35_mtp_attention(
    GpuEngine& context, const Qwen35MtpWeights& weights, Qwen35MtpState& state,
    Qwen35Workspace& workspace, const bf16* hidden, const int32_t* positions,
    bf16* output, int seq, int past, const Qwen35Config& config) {
    const auto& c = config.text;
    auto& queue = context.queue;
    int query_dim = c.num_attention_heads * c.head_dim;
    int key_value_dim = c.num_key_value_heads * c.head_dim;
    if (past + seq > state.capacity)
        throw std::runtime_error("Qwen3.5 MTP KV cache overflow");

    qwen35_mtp_trace(queue, "attn:enter", seq, past);
    matmul_bf16(hidden, seq, c.hidden_size, weights.q_proj.data(),
                2 * query_dim, workspace.tmp0.data(), context);
    qwen35_split_q_gate(queue, workspace.tmp0.data(), workspace.tmp2.data(),
                        workspace.tmp3.data(), seq, c.num_attention_heads,
                        c.head_dim);
    matmul_bf16(hidden, seq, c.hidden_size, weights.k_proj.data(),
                key_value_dim, workspace.tmp1.data(), context);
    matmul_bf16(hidden, seq, c.hidden_size, weights.v_proj.data(),
                key_value_dim, workspace.tmp4.data(), context);

    rms_norm(queue, workspace.tmp2.data(), weights.q_norm.data(),
             workspace.tmp2.data(), seq * c.num_attention_heads, c.head_dim,
             c.rms_norm_eps);
    rms_norm(queue, workspace.tmp1.data(), weights.k_norm.data(),
             workspace.tmp1.data(), seq * c.num_key_value_heads, c.head_dim,
             c.rms_norm_eps);
    qwen35_mtp_trace(queue, "attn:qkv+norm", seq, past);
    qwen35_apply_mrope(queue, workspace.tmp2.data(), workspace.tmp1.data(),
                       positions, seq, c.num_attention_heads,
                       c.num_key_value_heads, c.head_dim, c.rotary_dim(),
                       c.rope.theta, c.rope.mrope_section);

    size_t offset = (size_t)past * key_value_dim;
    size_t count = (size_t)seq * key_value_dim;
    queue.memcpy(state.key.data() + offset, workspace.tmp1.data(),
                 count * sizeof(bf16));
    queue.memcpy(state.value.data() + offset, workspace.tmp4.data(),
                 count * sizeof(bf16));
    state.filled = past + seq;

    qwen35_mtp_trace(queue, "attn:rope+cache", seq, past);
    float scale = 1.0f / std::sqrt((float)c.head_dim);
    if (qwen35_xmx_attention_enabled()) {
        qwen35_xmx_attention(queue, workspace.tmp2.data(), state.key.data(),
                             state.value.data(), workspace.tmp2.data(), seq, past,
                             c.num_attention_heads, c.num_key_value_heads,
                             c.head_dim, scale);
    } else {
        qwen35_online_attention(queue, workspace.tmp2.data(), state.key.data(),
                                state.value.data(), workspace.tmp2.data(), seq,
                                past, c.num_attention_heads,
                                c.num_key_value_heads, c.head_dim, scale);
    }
    qwen35_mtp_trace(queue, "attn:core", seq, past);
    mul_sigmoid_inplace(queue, workspace.tmp2.data(), workspace.tmp3.data(),
                        (size_t)seq * query_dim);
    matmul_bf16(workspace.tmp2.data(), seq, query_dim, weights.o_proj.data(),
                c.hidden_size, output, context);
}

// Runs the head over `seq` consecutive positions. `hidden_pre_norm` is the
// backbone state at those positions before the model's final norm, and
// `next_token_ids` are the tokens one position later. Writes the post-mtp-norm
// hidden state to `out`, which the caller feeds to the shared lm_head.
inline void qwen35_mtp_forward(
    GpuEngine& context, const Qwen35MtpWeights& weights,
    const GpuBuffer<bf16>& embed_tokens, Qwen35MtpState& state,
    Qwen35Workspace& workspace, const bf16* hidden_pre_norm,
    const int32_t* next_token_ids, const int32_t* positions, bf16* out,
    int seq, int past, const Qwen35Config& config) {
    const auto& c = config.text;
    auto& queue = context.queue;
    if (seq <= 0 || seq > state.max_seq)
        throw std::runtime_error("Qwen3.5 MTP draft window out of range");

    qwen35_mtp_trace(queue, "fwd:enter", seq, past);
    embedding_lookup(queue, embed_tokens.data(), next_token_ids,
                     state.embeds.data(), seq, c.hidden_size, 1.0f);
    rms_norm(queue, state.embeds.data(), weights.pre_fc_norm_embedding.data(),
             state.embeds.data(), seq, c.hidden_size, c.rms_norm_eps);
    rms_norm(queue, hidden_pre_norm, weights.pre_fc_norm_hidden.data(),
             state.hidden.data(), seq, c.hidden_size, c.rms_norm_eps);
    qwen35_mtp_concat(queue, state.embeds.data(), state.hidden.data(),
                      state.merged.data(), seq, c.hidden_size);
    matmul_bf16(state.merged.data(), seq, 2 * c.hidden_size, weights.fc.data(),
                c.hidden_size, state.residual.data(), context);

    qwen35_mtp_trace(queue, "fwd:fc", seq, past);
    rms_norm(queue, state.residual.data(), weights.input_layernorm.data(),
             state.normed.data(), seq, c.hidden_size, c.rms_norm_eps);
    qwen35_mtp_attention(context, weights, state, workspace,
                         state.normed.data(), positions, state.sublayer.data(),
                         seq, past, config);
    add_inplace(queue, state.residual.data(), state.sublayer.data(),
                (size_t)seq * c.hidden_size);

    qwen35_mtp_trace(queue, "fwd:attn-done", seq, past);
    rms_norm(queue, state.residual.data(), weights.post_attention_layernorm.data(),
             state.normed.data(), seq, c.hidden_size, c.rms_norm_eps);
    matmul_bf16(state.normed.data(), seq, c.hidden_size, weights.gate_proj.data(),
                c.intermediate_size, state.gate.data(), context);
    matmul_bf16(state.normed.data(), seq, c.hidden_size, weights.up_proj.data(),
                c.intermediate_size, state.up.data(), context);
    qwen35_mtp_swiglu(queue, state.gate.data(), state.up.data(),
                      state.gate.data(), (size_t)seq * c.intermediate_size);
    matmul_bf16(state.gate.data(), seq, c.intermediate_size,
                weights.down_proj.data(), c.hidden_size, state.sublayer.data(),
                context);
    add_inplace(queue, state.residual.data(), state.sublayer.data(),
                (size_t)seq * c.hidden_size);

    qwen35_mtp_trace(queue, "fwd:mlp", seq, past);
    rms_norm(queue, state.residual.data(), weights.norm.data(), out, seq,
             c.hidden_size, c.rms_norm_eps);
}
