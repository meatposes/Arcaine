#pragma once

#include <string>
#include <vector>

#include "cache.hpp"
#include "config.hpp"
#include "mtp.hpp"
#include "weights.hpp"
#include "workspace.hpp"
#include "../../common/model_interface.hpp"

class Qwen35Model final : public Model {
public:
    explicit Qwen35Model(const std::string& model_dir, int max_seq_len = 2048);

    PreparedInput prepare_input(const std::string& prompt,
                                const std::vector<std::string>& image_paths,
                                const std::vector<std::string>& audio_paths,
                                const std::string& vad_model) override;
    std::vector<float> forward(const ForwardInput& input) override;
    void reset_cache() override;
    const ModelInfo& info() const override { return info_; }

    bool has_mtp() const override { return mtp_state_.ready(); }
    std::vector<float> mtp_draft(int next_token, int position) override;

    // Logits for every position of a batch, laid out [seq, vocab]. Used by the
    // speculative loop, which needs the distribution at the drafted position
    // as well as the one after it.
    std::vector<float> forward_verify(const std::vector<int>& tokens, int past);

    // Greedy generation driven by the MTP head. Each round drafts one token,
    // verifies it alongside the pending token in a single backbone pass, and
    // rolls the caches back when the draft misses.
    struct SpecStats {
        int rounds = 0, forwards = 0, drafts = 0, accepts = 0;
        double draft_ms = 0.0, verify_ms = 0.0, rollback_ms = 0.0;
    };
    std::vector<int> generate_speculative(const std::vector<int>& prompt,
                                          int max_tokens, SpecStats& stats);

private:
    // Runs the MTP head over consecutive positions to keep its KV cache in step
    // with the backbone. `next_tokens[i]` is the token at
    // `start_position + i + 1`, the one whose embedding the head consumes, and
    // it is paired with the backbone hidden state at `start_position + i`.
    // Split into chunks of the draft window so scratch stays bounded.
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
    // the last forward covered: the head's contract is that input, and the
    // final norm that follows is applied in place, so it has to be copied out
    // before it is overwritten.
    Qwen35MtpState mtp_state_;
    GpuBuffer<bf16> backbone_hidden_;
    GpuBuffer<bf16> mtp_out_;
    GpuBuffer<int32_t> mtp_tokens_;
    GpuBuffer<int32_t> mtp_positions_;
    int mtp_window_ = 0;
    int backbone_hidden_len_ = 0;   // positions valid in backbone_hidden_
    int backbone_hidden_base_ = 0;  // index of its first position
    // Absolute position where the head's KV starts. A token embedded at
    // position p occupies slot p - 1 - mtp_base_. Non-zero when the prompt was
    // longer than the head's window and only its tail was covered.
    int mtp_base_ = 0;

    // Verify scratch. Depth-1 speculation needs two positions of logits; the
    // cap keeps a 248k-wide vocabulary from turning this into hundreds of MB.
    static constexpr int kMaxVerify = 4;
    GpuBuffer<bf16> verify_normed_;
    GpuBuffer<bf16> verify_logits_bf16_;
    GpuBuffer<float> verify_logits_f32_;
};
