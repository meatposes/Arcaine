#pragma once

#include <string>
#include <vector>
#include <algorithm>
#include <cstdint>

#include "cache.hpp"
#include "config.hpp"
#include "mtp.hpp"
#include "weights.hpp"
#include "workspace.hpp"
#include "../../preprocessing/image_processor.hpp"
#include "../../preprocessing/audio_processor.hpp"

// ---------------------------------------------------------------------------
// Model-local interface types (relocated per the model-isolation rule).
// Each autoregressive model owns its copy; the model's session / request_mapper
// / benchmark drive the concrete model through these. DiffusionGemma does not
// use them (its generate() takes a direct prompt + step count).
// ---------------------------------------------------------------------------
struct ForwardInput {
    const std::vector<int>&        token_ids;
    int                            past_len = 0;
    const std::vector<ImageInput>* images            = nullptr;
    const std::vector<AudioInput>* audio             = nullptr;
    const std::vector<int32_t>*    mm_token_type_ids = nullptr;
};

struct PreparedInput {
    std::vector<int>        tokens;
    std::vector<ImageInput> images;
    std::vector<AudioInput> audio;
    std::vector<int32_t>    mm_token_type_ids;  // 0=text, 1=image, 2=video, 3=audio
};

// Analytic memory traffic for one batch-1 decode step. Decode is
// bandwidth-bound, so ms/token only means something next to the bytes that had
// to move to produce the token: this is the denominator that turns a
// throughput number into a fraction of the device's measured bandwidth. It is
// computed from the resident tensors, never measured, so a gap against the
// achieved figure is a finding rather than noise.
struct DecodeTrafficClass {
    std::string name;    // "lm_head", "mlp", "attn_proj", ...
    size_t      bytes = 0;
};

struct DecodeTraffic {
    // Read once per decode step regardless of cache depth: weights, plus any
    // persistent recurrent state re-read every token.
    std::vector<DecodeTrafficClass> fixed;
    // Re-read for every cached position, so KV traffic is this times depth.
    size_t bytes_per_kv_position = 0;

    bool empty() const { return fixed.empty(); }
    size_t fixed_bytes() const {
        size_t total = 0;
        for (const auto& c : fixed) total += c.bytes;
        return total;
    }
    size_t bytes_at_depth(size_t kv_depth) const {
        return fixed_bytes() + bytes_per_kv_position * kv_depth;
    }
};

struct ModelInfo {
    int   vocab_size   = 0;
    int   max_seq_len  = 0;
    int   bos_token_id = -1;
    std::vector<int> eos_token_ids;
    std::vector<int> suppress_tokens;
    float temperature  = 1.0f;
    int   top_k        = 64;
    float top_p        = 0.95f;
    std::string model_dir;
    std::string description;
    // Empty when the architecture has not filled it in; the roofline report is
    // skipped rather than guessed at.
    DecodeTraffic decode_traffic;

    bool is_eos(int id) const {
        return std::find(eos_token_ids.begin(), eos_token_ids.end(), id)
               != eos_token_ids.end();
    }
};

class Qwen35Model final {
public:
    explicit Qwen35Model(const std::string& model_dir, int max_seq_len = 2048);

    PreparedInput prepare_input(const std::string& prompt,
                                const std::vector<std::string>& image_paths,
                                const std::vector<std::string>& audio_paths,
                                const std::string& vad_model);
    std::vector<float> forward(const ForwardInput& input);
    void reset_cache();
    const ModelInfo& info() const { return info_; }

    // ---- multi-token prediction -----------------------------------------
    //
    // The checkpoint ships an MTP head: one full-attention decoder layer that
    // predicts the token two positions ahead from the backbone hidden state
    // and the embedding of the token after it. A draft model with no extra
    // weights to download and no second checkpoint to keep in sync.
    bool has_mtp() const { return mtp_state_.ready(); }

    // Drafts the token after `next_token`, which occupies `position`, from the
    // hidden state the last forward already produced. Empty without a head.
    std::vector<float> mtp_draft(int next_token, int position);

    // Logits for every position of a batch, laid out [seq, vocab]. The
    // speculative loop needs the distribution at the drafted position as well
    // as the one after it.
    std::vector<float> forward_verify(const std::vector<int>& tokens, int past);

    // Greedy generation driven by the head. Each round drafts one token,
    // verifies it alongside the pending token in a single backbone pass, and
    // rolls the caches back when the draft misses.
    struct SpecStats {
        int rounds = 0, forwards = 0, drafts = 0, accepts = 0;
        double draft_ms = 0.0, verify_ms = 0.0, rollback_ms = 0.0;
    };
    std::vector<int> generate_speculative(const std::vector<int>& prompt,
                                          int max_tokens, SpecStats& stats);

private:
    // Runs the head over consecutive positions to keep its KV in step with the
    // backbone. `next_tokens[i]` is the token at `first_embedded_position + i`,
    // paired with the backbone hidden state one position earlier.
    void advance_mtp(const std::vector<int>& next_tokens,
                     const std::vector<int32_t>& positions,
                     int first_embedded_position);
    std::vector<float> mtp_logits_from(const bf16* mtp_hidden);

    std::vector<int32_t> build_positions(const std::vector<int>& tokens,
                                         const std::vector<int32_t>* token_types,
                                         const std::vector<ImageInput>* images,
                                         int past);
    void run_layer(GpuEngine& context, Qwen35LayerWeights& layer,
                   Qwen35Workspace& workspace, Qwen35KvLayerCache& kv,
                   Qwen35DeltaLayerCache& delta, bf16* hidden,
                   bf16* normalized, bf16* sublayer, const int32_t* positions,
                   int seq, int past);

    Qwen35Config config_;
    Qwen35Weights weights_;
    Qwen35Caches caches_;
    int split_layer_ = 0;
    int max_seq_len_ = 0;
    int rope_delta_ = 0;
    Qwen35Workspace workspace0_;
    Qwen35Workspace workspace1_;
    GpuBuffer<bf16> hidden0_;
    GpuBuffer<bf16> normalized0_;
    GpuBuffer<bf16> sublayer0_;
    GpuBuffer<bf16> hidden1_;
    GpuBuffer<bf16> normalized1_;
    GpuBuffer<bf16> sublayer1_;
    GpuBuffer<int32_t> token_ids0_;
    GpuBuffer<int32_t> positions0_;
    GpuBuffer<int32_t> positions1_;
    GpuBuffer<bf16> logits_bf16_;
    GpuBuffer<float> logits_f32_;
    std::vector<bf16> transfer_host_;
    ModelInfo info_;

    // MTP. `backbone_hidden_` holds the pre-final-norm state of every position
    // the last forward covered: that is the head's contract, and the final norm
    // that follows can run in place over the same buffer, so it is copied out
    // before it can be overwritten.
    Qwen35MtpState mtp_state_;
    GpuBuffer<bf16> backbone_hidden_;
    GpuBuffer<bf16> mtp_out_;
    GpuBuffer<int32_t> mtp_tokens_;
    GpuBuffer<int32_t> mtp_positions_;
    int mtp_window_ = 0;
    int backbone_hidden_len_ = 0;   // positions valid in backbone_hidden_
    int backbone_hidden_base_ = 0;  // index of its first position

    // Verify scratch. Depth-1 speculation needs two positions of logits; the
    // cap keeps a 248k-wide vocabulary from turning this into hundreds of MB.
    static constexpr int kMaxVerify = 4;
    GpuBuffer<bf16> verify_normed_;
    GpuBuffer<bf16> verify_logits_bf16_;
    GpuBuffer<float> verify_logits_f32_;
};
