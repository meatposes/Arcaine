#pragma once

// Multi-token-prediction head, run purely to measure how often its draft would
// have been accepted. No speculative decoding: nothing here changes the tokens
// the model emits, and the main model's caches are untouched.
//
// It exists because the decision to build speculative decoding on this
// checkpoint rests on one unknown -- the acceptance rate -- and the published
// prior art is negative. A llama.cpp port of MTP for this same model family
// measured 47.5% acceptance and a net 26% slowdown, blaming checkpoint/restore
// of the recurrent state. That reason does not survive contact with this
// engine: the snapshot measures 0.213 ms here against a 102 ms decode step.
// With decode this slow the fixed draft cost is proportionally small, and the
// arithmetic puts break-even near 5% rather than their 70%. Whether that holds
// depends entirely on the acceptance rate, so measure it before building
// anything that depends on it.
//
// Semantics are corroborated from the vLLM and mlx-lm implementations and from
// what this loader already encodes; there is no upstream reference, because
// transformers discards these weights outright
// (_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]). The order is:
//
//   h      = backbone hidden at position t, BEFORE the final norm
//   e      = embed(token t+1)
//   x      = fc @ concat[ norm_e(e), norm_h(h) ]        -- embed first
//   x      = one full-attention decoder layer (its own KV)
//   logits = lm_head( mtp_norm(x) )                     -- predicts token t+2
//
// Both pre-FC norms carry the +1 offset, already baked in by the loader's
// upload_plus_one, which is what corroborates the convention.

#include <cmath>

#include "config.hpp"
#include "kernels.hpp"
#include "operators.hpp"
#include "weights.hpp"
#include "../qwen3_5_moe/kernels.hpp"  // swiglu_strided
#include "../../common/gpu/ops.hpp"
#include "../../common/kernels/elementwise.hpp"
#include "../../common/kernels/embedding.hpp"
#include "../../common/kernels/rms_norm.hpp"

inline bool qwen35_mtp_acceptance_enabled() {
    static bool enabled = [] {
        const char* value = std::getenv("ARCAINE_QWEN35_MTP_ACCEPTANCE");
        if (!value) return false;
        return std::strcmp(value, "0") != 0 && std::strcmp(value, "off") != 0 &&
               std::strcmp(value, "false") != 0 && std::strcmp(value, "no") != 0;
    }();
    return enabled;
}

// Scratch and KV for the MTP layer. Separate from the main model's caches --
// the head must not perturb them, or the measurement would change the tokens it
// is trying to measure.
struct Qwen35MtpState {
    GpuBuffer<bf16> key;
    GpuBuffer<bf16> value;
    int filled = 0;
    int capacity = 0;

    GpuBuffer<int32_t> token;
    GpuBuffer<int32_t> positions;
    GpuBuffer<bf16> embedding;
    GpuBuffer<bf16> fused;      // concat[norm_e, norm_h], 2H
    GpuBuffer<bf16> hidden;     // H
    GpuBuffer<bf16> normalized; // H
    GpuBuffer<bf16> attn0;      // q_proj output, gated
    GpuBuffer<bf16> attn1;      // K
    GpuBuffer<bf16> attn2;      // Q
    GpuBuffer<bf16> attn3;      // gate
    GpuBuffer<bf16> attn4;      // V
    GpuBuffer<bf16> mlp0;       // 2I
    GpuBuffer<bf16> mlp1;       // I
    GpuBuffer<bf16> logits_bf16;
    GpuBuffer<float> logits_f32;

    // Draft produced last step for the token after next, and the position it
    // was drafted for. -1 when there is nothing outstanding.
    int pending_draft = -1;
    int pending_for_position = -1;
    long long drafted = 0;
    long long accepted = 0;

    void init(const Qwen35Config& config, int max_seq, sycl::queue& queue) {
        const auto& c = config.text;
        int q_out = c.num_attention_heads * c.head_dim * 2;
        int kv_out = c.num_key_value_heads * c.head_dim;
        int attn_out = c.num_attention_heads * c.head_dim;
        capacity = max_seq;
        key = GpuBuffer<bf16>((size_t)max_seq * kv_out, queue);
        value = GpuBuffer<bf16>((size_t)max_seq * kv_out, queue);
        token = GpuBuffer<int32_t>(1, queue);
        positions = GpuBuffer<int32_t>(3, queue);
        embedding = GpuBuffer<bf16>(c.hidden_size, queue);
        fused = GpuBuffer<bf16>((size_t)2 * c.hidden_size, queue);
        hidden = GpuBuffer<bf16>(c.hidden_size, queue);
        normalized = GpuBuffer<bf16>(c.hidden_size, queue);
        attn0 = GpuBuffer<bf16>(q_out, queue);
        attn1 = GpuBuffer<bf16>(kv_out, queue);
        attn2 = GpuBuffer<bf16>(attn_out, queue);
        attn3 = GpuBuffer<bf16>(attn_out, queue);
        attn4 = GpuBuffer<bf16>(kv_out, queue);
        mlp0 = GpuBuffer<bf16>((size_t)2 * c.intermediate_size, queue);
        mlp1 = GpuBuffer<bf16>(c.intermediate_size, queue);
        logits_bf16 = GpuBuffer<bf16>(c.vocab_size, queue);
        logits_f32 = GpuBuffer<float>(c.vocab_size, queue);
    }

    void reset() {
        filled = 0;
        pending_draft = -1;
        pending_for_position = -1;
    }
};

// Draft the token at `position + 1`, from the backbone hidden at `position - 1`
// and the embedding of the token at `position`. Returns the argmax token id.
inline int qwen35_mtp_draft(
    GpuEngine& context,
    const Qwen35Weights& weights,
    const Qwen35Config& config,
    Qwen35MtpState& state,
    const bf16* backbone_hidden,  // pre-final-norm hidden, H
    int next_token,
    int position) {
    const auto& c = config.text;
    auto& queue = context.queue;
    const int H = c.hidden_size;
    const int I = c.intermediate_size;
    const int q_out = c.num_attention_heads * c.head_dim * 2;
    const int kv_out = c.num_key_value_heads * c.head_dim;
    const int attn_out = c.num_attention_heads * c.head_dim;
    const Qwen35MtpWeights& w = weights.mtp;

    if (position >= state.capacity) return -1;

    // e = embed(token t+1)
    queue.memcpy(state.token.data(), &next_token, sizeof(int32_t)).wait();
    embedding_lookup(queue, weights.embed_tokens.data(), state.token.data(),
                     state.embedding.data(), 1, H, 1.0f);

    // concat[ norm_e(e), norm_h(h) ] -- embed first, per the reference order.
    rms_norm(queue, state.embedding.data(), w.pre_fc_norm_embedding.data(),
             state.fused.data(), 1, H, c.rms_norm_eps);
    rms_norm(queue, backbone_hidden, w.pre_fc_norm_hidden.data(),
             state.fused.data() + H, 1, H, c.rms_norm_eps);
    matmul_bf16(state.fused.data(), 1, 2 * H, w.fc.data(), H,
                state.hidden.data(), context);

    // --- one full-attention decoder layer ---
    rms_norm(queue, state.hidden.data(), w.input_layernorm.data(),
             state.normalized.data(), 1, H, c.rms_norm_eps);

    matmul_bf16(state.normalized.data(), 1, H, w.q_proj.data(), q_out,
                state.attn0.data(), context);
    qwen35_split_q_gate(queue, state.attn0.data(), state.attn2.data(),
                        state.attn3.data(), 1, c.num_attention_heads, c.head_dim);
    matmul_bf16(state.normalized.data(), 1, H, w.k_proj.data(), kv_out,
                state.attn1.data(), context);
    matmul_bf16(state.normalized.data(), 1, H, w.v_proj.data(), kv_out,
                state.attn4.data(), context);

    rms_norm(queue, state.attn2.data(), w.q_norm.data(), state.attn2.data(),
             c.num_attention_heads, c.head_dim, c.rms_norm_eps);
    rms_norm(queue, state.attn1.data(), w.k_norm.data(), state.attn1.data(),
             c.num_key_value_heads, c.head_dim, c.rms_norm_eps);

    std::vector<int32_t> host_positions(3, position);
    queue.memcpy(state.positions.data(), host_positions.data(),
                 host_positions.size() * sizeof(int32_t)).wait();
    qwen35_apply_mrope(queue, state.attn2.data(), state.attn1.data(),
                       state.positions.data(), 1, c.num_attention_heads,
                       c.num_key_value_heads, c.head_dim, c.rotary_dim(),
                       c.rope.theta, c.rope.mrope_section);

    // The head keeps its own KV across steps, as the real thing would; drafting
    // against a single-token context would understate acceptance.
    if (state.filled > position) state.filled = position;
    size_t offset = (size_t)state.filled * kv_out;
    queue.memcpy(state.key.data() + offset, state.attn1.data(),
                 (size_t)kv_out * sizeof(bf16));
    queue.memcpy(state.value.data() + offset, state.attn4.data(),
                 (size_t)kv_out * sizeof(bf16));
    int past = state.filled;
    state.filled += 1;

    qwen35_xmx_attention(queue, state.attn2.data(), state.key.data(),
                         state.value.data(), state.attn2.data(), 1, past,
                         c.num_attention_heads, c.num_key_value_heads,
                         c.head_dim, 1.0f / std::sqrt((float)c.head_dim));
    mul_sigmoid_inplace(queue, state.attn2.data(), state.attn3.data(),
                        (size_t)attn_out);
    matmul_bf16(state.attn2.data(), 1, attn_out, w.o_proj.data(), H,
                state.normalized.data(), context);
    add_inplace(queue, state.hidden.data(), state.normalized.data(), (size_t)H);

    rms_norm(queue, state.hidden.data(), w.post_attention_layernorm.data(),
             state.normalized.data(), 1, H, c.rms_norm_eps);
    matmul_bf16(state.normalized.data(), 1, H, w.gate_proj.data(), I,
                state.mlp0.data(), context);
    matmul_bf16(state.normalized.data(), 1, H, w.up_proj.data(), I,
                state.mlp0.data() + I, context);
    swiglu_strided(queue, state.mlp0.data(), state.mlp1.data(), 1, I);
    matmul_bf16(state.mlp1.data(), 1, I, w.down_proj.data(), H,
                state.normalized.data(), context);
    add_inplace(queue, state.hidden.data(), state.normalized.data(), (size_t)H);

    // Shared head: MTP's own final norm, then the model's LM head.
    rms_norm(queue, state.hidden.data(), w.norm.data(), state.normalized.data(),
             1, H, c.rms_norm_eps);
    qwen35_matmul(state.normalized.data(), 1, H, weights.lm_head,
                  state.logits_bf16.data(), context);
    bf16_to_f32(queue, state.logits_bf16.data(), state.logits_f32.data(),
                c.vocab_size);

    std::vector<float> logits(c.vocab_size);
    queue.memcpy(logits.data(), state.logits_f32.data(),
                 logits.size() * sizeof(float)).wait();
    int best = 0;
    float best_value = logits[0];
    for (int i = 1; i < c.vocab_size; ++i)
        if (logits[i] > best_value) { best_value = logits[i]; best = i; }
    return best;
}
