#include "model.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <utility>

#include "loader.hpp"
#include "operators.hpp"
#include "vision.hpp"
#include "../../runtime/gpu/engine.hpp"
#include "../../runtime/quantization/fp8.hpp"
#include "runtime/kernels/elementwise.hpp"
#include "runtime/kernels/embedding.hpp"
#include "runtime/kernels/rms_norm.hpp"
#include "runtime/kernels/scatter.hpp"
#include "../../preprocessing/chat_template.hpp"

namespace {

bool qwen35_persistent_io_enabled() {
    static bool enabled = [] {
        const char* value = std::getenv("ARCAINE_QWEN35_PERSISTENT_IO");
        if (!value) return true;
        return std::strcmp(value, "0") != 0 && std::strcmp(value, "off") != 0 &&
               std::strcmp(value, "false") != 0 && std::strcmp(value, "no") != 0;
    }();
    return enabled;
}

// ---------------------------------------------------------------------------
// Per-decode-step traffic accounting.
//
// Walks the tensors that were actually loaded rather than re-deriving sizes
// from the config, so ARCAINE_QWEN35_MAX_LAYERS truncation and any fused or
// absent projection are reflected without a second source of truth.
//
// Only bytes a single decode step really moves are counted. The embedding
// table is the case that matters: 2.5 GiB resident, but decode gathers one row
// of it, so counting the table would inflate the denominator ~10x and make
// every efficiency number meaningless.

size_t buffer_bytes(const GpuBuffer<bf16>& b) { return b.count() * sizeof(bf16); }
size_t buffer_bytes(const GpuBuffer<float>& b) { return b.count() * sizeof(float); }
size_t buffer_bytes(const GpuBuffer<uint8_t>& b) { return b.count(); }

size_t linear_bytes(const Fp8Linear& w) {
    return buffer_bytes(w.weight) + buffer_bytes(w.weight_scale);
}

// zp_offset is empty for a symmetric checkpoint, so it contributes nothing
// there and its real cost is counted when it is present.
size_t linear_bytes(const Int4Linear& w) {
    return buffer_bytes(w.weight_packed) + buffer_bytes(w.weight_scale) +
           buffer_bytes(w.zp_offset);
}

// weight_any / weight_coal are alternate layouts materialized lazily on first
// use; at construction only weight_packed exists, and a populated alternate
// replaces rather than adds to it, so the packed size is the traffic either way.
size_t linear_bytes(const Nvfp4Linear& w) {
    return buffer_bytes(w.weight_packed) + buffer_bytes(w.weight_scale) +
           buffer_bytes(w.dst_scale);
}

size_t proj_bytes(const Qwen35Proj& w) {
    return std::visit([](const auto& v) { return linear_bytes(v); }, w);
}

size_t mlp_bytes(const Qwen35MlpWeights& mlp) {
    auto visit = [](const auto& v) { return linear_bytes(v); };
    return std::visit(visit, mlp.gate_up) + std::visit(visit, mlp.down);
}

DecodeTraffic qwen35_decode_traffic(const Qwen35Weights& w,
                                    const Qwen35Caches& caches,
                                    const Qwen35TextConfig& c,
                                    int hidden_size) {
    size_t embed_row = 0, head = 0, attn = 0, delta = 0, mlp = 0, norms = 0;
    size_t state = 0;

    // One gathered row of [vocab, hidden], not the table.
    embed_row = w.embed_tokens.empty() ? 0 : (size_t)hidden_size * sizeof(bf16);
    if (const auto* fp8 = std::get_if<Fp8Linear>(&w.lm_head))
        head = linear_bytes(*fp8);
    else
        head = buffer_bytes(std::get<GpuBuffer<bf16>>(w.lm_head));
    norms += buffer_bytes(w.final_norm);

    for (const auto& layer : w.layers) {
        norms += buffer_bytes(layer.input_layernorm) +
                 buffer_bytes(layer.post_attention_layernorm);
        mlp += mlp_bytes(layer.mlp);
        if (const auto* full = std::get_if<Qwen35FullAttentionWeights>(&layer.mixer)) {
            attn += proj_bytes(full->qkv_proj) + proj_bytes(full->q_proj) +
                    proj_bytes(full->k_proj) + proj_bytes(full->v_proj) +
                    proj_bytes(full->o_proj) +
                    buffer_bytes(full->q_norm) + buffer_bytes(full->k_norm);
        } else if (const auto* lin =
                       std::get_if<Qwen35LinearAttentionWeights>(&layer.mixer)) {
            delta += proj_bytes(lin->in_proj_qkvz) + proj_bytes(lin->in_proj_qkv) +
                     proj_bytes(lin->in_proj_z) + proj_bytes(lin->out_proj) +
                     buffer_bytes(lin->in_proj_a) + buffer_bytes(lin->in_proj_b) +
                     buffer_bytes(lin->in_proj_ba) + buffer_bytes(lin->conv1d) +
                     buffer_bytes(lin->conv1d_time_major) + buffer_bytes(lin->A_log) +
                     buffer_bytes(lin->dt_bias) + buffer_bytes(lin->norm);
        }
    }

    // DeltaNet state is read and written in place every token, so it costs
    // twice its size and does not grow with sequence length the way KV does.
    for (const auto& d : caches.delta) {
        if (d.recurrent_state.empty()) continue;
        state += 2 * (buffer_bytes(d.recurrent_state) + buffer_bytes(d.conv_state));
    }

    // Every cached position of every full-attention layer is re-read per step.
    size_t per_position = 0;
    for (const auto& kv : caches.kv) {
        if (kv.key.empty()) continue;
        per_position += 2 * (size_t)c.num_key_value_heads * c.head_dim * sizeof(bf16);
    }

    DecodeTraffic t;
    auto push = [&](const char* name, size_t bytes) {
        if (bytes) t.fixed.push_back({name, bytes});
    };
    push("lm_head", head);
    push("mlp", mlp);
    push("delta_proj", delta);
    push("attn_proj", attn);
    push("delta_state_rw", state);
    push("norms", norms);
    push("embed_row", embed_row);
    t.bytes_per_kv_position = per_position;
    return t;
}

}  // namespace

Qwen35Model::Qwen35Model(const std::string& model_dir, int max_seq_len)
    : config_(Qwen35Config::from_dir(model_dir)), max_seq_len_(max_seq_len) {
    if (max_seq_len <= 0 || max_seq_len > config_.text.max_position_embeddings)
        throw std::runtime_error("Invalid Qwen3.5 maximum sequence length");
    split_layer_ = GpuEngine::count() >= 2 ? config_.text.num_hidden_layers / 2
                                           : config_.text.num_hidden_layers;
    int max_layers = -1;
    if (const char* value = std::getenv("ARCAINE_QWEN35_MAX_LAYERS")) {
        int parsed = std::atoi(value);
        if (parsed > 0) max_layers = parsed;
    }
    std::printf("[qwen35] %d GPU(s), layer split=%d, NVFP4 DPAS=%s, "
                "XMX attention=%s, ESIMD DeltaNet=%s, "
                "fused Delta decode=%s, fused BA=%s, "
                "persistent IO=%s\n",
                GpuEngine::count(), split_layer_,
                qwen35_nvfp4_dpas_enabled() ? "on" : "off",
                qwen35_xmx_attention_enabled() ? "on" : "off",
                qwen35_esimd_delta_enabled() ? "on" : "off",
                qwen35_fused_esimd_delta_decode_enabled() ? "on" : "off",
                qwen35_fused_ba_projection_enabled() ? "on" : "off",
                qwen35_persistent_io_enabled() ? "on" : "off");
    ShardedSafetensors checkpoint(model_dir);
    weights_ = load_qwen35_weights(checkpoint, config_, split_layer_, max_layers);
    caches_.init(config_, max_seq_len, split_layer_);

    size_t activations = (size_t)max_seq_len * config_.text.hidden_size;
    auto& queue0 = GpuEngine::get(0).queue;
    workspace0_.init(config_, max_seq_len, queue0);
    hidden0_ = GpuBuffer<bf16>(activations, queue0);
    normalized0_ = GpuBuffer<bf16>(activations, queue0);
    sublayer0_ = GpuBuffer<bf16>(activations, queue0);
    if (qwen35_persistent_io_enabled()) {
        token_ids0_ = GpuBuffer<int32_t>(max_seq_len, queue0);
        positions0_ = GpuBuffer<int32_t>((size_t)3 * max_seq_len, queue0);
        logits_bf16_ = GpuBuffer<bf16>(config_.text.vocab_size, queue0);
        logits_f32_ = GpuBuffer<float>(config_.text.vocab_size, queue0);
        transfer_host_.resize(activations);
    }
    if (GpuEngine::count() >= 2) {
        auto& queue1 = GpuEngine::get(1).queue;
        workspace1_.init(config_, max_seq_len, queue1);
        hidden1_ = GpuBuffer<bf16>(activations, queue1);
        normalized1_ = GpuBuffer<bf16>(activations, queue1);
        sublayer1_ = GpuBuffer<bf16>(activations, queue1);
        if (qwen35_persistent_io_enabled())
            positions1_ = GpuBuffer<int32_t>((size_t)3 * max_seq_len, queue1);
    }

    info_.vocab_size = config_.text.vocab_size;
    info_.max_seq_len = max_seq_len;
    info_.bos_token_id = config_.bos_token_id;
    info_.eos_token_ids = config_.eos_token_ids;
    info_.temperature = config_.temperature;
    info_.top_k = config_.top_k;
    info_.top_p = config_.top_p;
    info_.model_dir = model_dir;
    char description[256];
    std::snprintf(description, sizeof(description),
        "qwen3_5 conditional: %d layers, H=%d, %d full/%d linear, vision=%d blocks",
        config_.text.num_hidden_layers, config_.text.hidden_size,
        config_.text.num_hidden_layers / config_.text.full_attention_interval,
        config_.text.num_hidden_layers -
            config_.text.num_hidden_layers / config_.text.full_attention_interval,
        config_.vision.depth);
    info_.description = description;
    info_.decode_traffic = qwen35_decode_traffic(
        weights_, caches_, config_.text, config_.text.hidden_size);

    // The head is optional: a checkpoint without one still loads, and
    // ARCAINE_QWEN35_MTP=0 keeps the weights resident but idle so speculative
    // and plain decoding can be A/B'd without reloading.
    if (!weights_.mtp.fc.empty() && qwen35_mtp_enabled()) {
        mtp_window_ = 64;
        if (const char* value = std::getenv("ARCAINE_QWEN35_MTP_WINDOW")) {
            int parsed = std::atoi(value);
            if (parsed > 0) mtp_window_ = parsed;
        }
        mtp_window_ = std::min(mtp_window_, max_seq_len);
        mtp_state_.init(config_, max_seq_len, mtp_window_, queue0);
        backbone_hidden_ = GpuBuffer<bf16>(activations, queue0);
        mtp_out_ = GpuBuffer<bf16>((size_t)mtp_window_ * config_.text.hidden_size,
                                   queue0);
        mtp_tokens_ = GpuBuffer<int32_t>(mtp_window_, queue0);
        mtp_positions_ = GpuBuffer<int32_t>((size_t)3 * mtp_window_, queue0);
        verify_normed_ = GpuBuffer<bf16>(
            (size_t)kMaxVerify * config_.text.hidden_size, queue0);
        verify_logits_bf16_ = GpuBuffer<bf16>(
            (size_t)kMaxVerify * config_.text.vocab_size, queue0);
        verify_logits_f32_ = GpuBuffer<float>(
            (size_t)kMaxVerify * config_.text.vocab_size, queue0);
        caches_.init_snapshot(config_, split_layer_);
        std::printf("[qwen35] MTP head active, draft window %d\n", mtp_window_);
    }
}

PreparedInput Qwen35Model::prepare_input(
    const std::string& prompt,
    const std::vector<std::string>& image_paths,
    const std::vector<std::string>& audio_paths,
    const std::string&) {
    if (!audio_paths.empty())
        throw std::runtime_error("Qwen3.5 supports image/video, not audio inputs");
    PreparedInput output;
    std::vector<int> image_token_counts;
    for (const auto& path : image_paths) {
        ImageInput image = preprocess_qwen_image(
            path, config_.vision.patch_size, config_.vision.temporal_patch_size,
            config_.vision.spatial_merge_size, config_.vision.min_pixels,
            config_.vision.max_pixels, config_.vision.rescale_factor,
            config_.vision.do_rescale, config_.vision.do_normalize,
            config_.vision.image_mean, config_.vision.image_std);
        image_token_counts.push_back(image.num_valid_patches);
        output.images.push_back(std::move(image));
    }
    auto built = build_chat_prompt(config_.model_dir, prompt, image_token_counts, {},
                                   true, false);
    output.tokens = std::move(built.tokens);
    output.mm_token_type_ids = std::move(built.mm_token_type_ids);
    return output;
}

std::vector<int32_t> Qwen35Model::build_positions(
    const std::vector<int>& tokens,
    const std::vector<int32_t>* token_types,
    const std::vector<ImageInput>* images,
    int past) {
    int seq = static_cast<int>(tokens.size());
    std::vector<int32_t> positions((size_t)3 * seq);
    if (past > 0) {
        for (int axis = 0; axis < 3; ++axis)
            for (int token = 0; token < seq; ++token)
                positions[(size_t)axis * seq + token] = past + rope_delta_ + token;
        return positions;
    }
    if (!token_types || token_types->empty() || !images || images->empty()) {
        for (int axis = 0; axis < 3; ++axis)
            for (int token = 0; token < seq; ++token)
                positions[(size_t)axis * seq + token] = token;
        rope_delta_ = 0;
        return positions;
    }
    if (static_cast<int>(token_types->size()) != seq)
        throw std::runtime_error("Qwen3.5 multimodal token type length mismatch");

    int current_position = 0;
    size_t image_index = 0;
    int token = 0;
    int maximum = 0;
    while (token < seq) {
        int type = token_types->at(token);
        int end = token + 1;
        while (end < seq && token_types->at(end) == type) ++end;
        if (type == 0) {
            for (int at = token; at < end; ++at) {
                int position = current_position + at - token;
                for (int axis = 0; axis < 3; ++axis)
                    positions[(size_t)axis * seq + at] = position;
                maximum = std::max(maximum, position);
            }
            current_position += end - token;
        } else if (type == 1 || type == 2) {
            if (image_index >= images->size())
                throw std::runtime_error("Qwen3.5 vision group has no matching grid");
            const ImageInput& image = images->at(image_index++);
            int merge = config_.vision.spatial_merge_size;
            int grid_t = image.grid_thw[0];
            int grid_h = image.grid_thw[1] / merge;
            int grid_w = image.grid_thw[2] / merge;
            if (end - token != grid_t * grid_h * grid_w)
                throw std::runtime_error("Qwen3.5 placeholder count does not match vision grid");
            int at = token;
            for (int t = 0; t < grid_t; ++t)
                for (int y = 0; y < grid_h; ++y)
                    for (int x = 0; x < grid_w; ++x, ++at) {
                        positions[at] = current_position + t;
                        positions[(size_t)seq + at] = current_position + y;
                        positions[(size_t)2 * seq + at] = current_position + x;
                        maximum = std::max({maximum, current_position + t,
                                            current_position + y, current_position + x});
                    }
            current_position += std::max(image.grid_thw[1], image.grid_thw[2]) / merge;
        } else {
            throw std::runtime_error("Unsupported Qwen3.5 multimodal token type");
        }
        token = end;
    }
    if (image_index != images->size())
        throw std::runtime_error("Unused Qwen3.5 image grids in prompt");
    rope_delta_ = maximum + 1 - seq;
    return positions;
}

void Qwen35Model::run_layer(
    GpuEngine& context, Qwen35LayerWeights& layer, Qwen35Workspace& workspace,
    Qwen35KvLayerCache& kv, Qwen35DeltaLayerCache& delta, bf16* hidden,
    bf16* normalized, bf16* sublayer, const int32_t* positions,
    int seq, int past) {
    const auto& c = config_.text;
    rms_norm(context.queue, hidden, layer.input_layernorm.data(), normalized,
             seq, c.hidden_size, c.rms_norm_eps);
    if (layer.full_attention) {
        qwen35_full_attention_forward(
            context, std::get<Qwen35FullAttentionWeights>(layer.mixer), kv,
            workspace, normalized, positions, sublayer, seq, past, config_);
    } else {
        qwen35_linear_attention_forward(
            context, std::get<Qwen35LinearAttentionWeights>(layer.mixer), delta,
            workspace, normalized, sublayer, seq, config_);
    }
    add_inplace(context.queue, hidden, sublayer, (size_t)seq * c.hidden_size);
    rms_norm(context.queue, hidden, layer.post_attention_layernorm.data(), normalized,
             seq, c.hidden_size, c.rms_norm_eps);
    qwen35_mlp_forward(context, layer.mlp, workspace, normalized, sublayer, seq, config_);
    add_inplace(context.queue, hidden, sublayer, (size_t)seq * c.hidden_size);
}

std::vector<float> Qwen35Model::forward(const ForwardInput& input) {
    int seq = static_cast<int>(input.token_ids.size());
    if (seq <= 0 || seq > max_seq_len_)
        throw std::runtime_error("Qwen3.5 forward sequence length out of range");
    auto& context0 = GpuEngine::get(0);
    auto& queue0 = context0.queue;
    std::vector<int32_t> host_ids(input.token_ids.begin(), input.token_ids.end());
    GpuBuffer<int32_t> token_ids_local;
    int32_t* token_ids = nullptr;
    if (qwen35_persistent_io_enabled()) {
        token_ids = token_ids0_.data();
        queue0.memcpy(token_ids, host_ids.data(), host_ids.size() * sizeof(int32_t));
    } else {
        token_ids_local = GpuBuffer<int32_t>(seq, queue0);
        token_ids_local.upload(host_ids.data(), host_ids.size());
        token_ids = token_ids_local.data();
    }
    embedding_lookup(queue0, weights_.embed_tokens.data(), token_ids,
                     hidden0_.data(), seq, config_.text.hidden_size, 1.0f);

    if (input.past_len == 0 && input.images && !input.images->empty()) {
        int search_from = 0;
        for (const ImageInput& image : *input.images) {
            GpuBuffer<bf16> features = qwen35_vision_forward(weights_.vision, config_.vision, image);
            std::vector<uint8_t> mask(seq, 0);
            int selected = 0;
            for (int i = search_from; i < seq && selected < image.num_valid_patches; ++i) {
                if (input.token_ids[i] == config_.image_token_id) {
                    mask[i] = 1;
                    ++selected;
                    search_from = i + 1;
                }
            }
            if (selected != image.num_valid_patches)
                throw std::runtime_error("Qwen3.5 image feature/token count mismatch");
            GpuBuffer<uint8_t> mask_device(seq, queue0);
            mask_device.upload(mask.data(), mask.size());
            auto offsets = compute_scatter_offsets(queue0, mask.data(), seq);
            masked_scatter_bf16(queue0, hidden0_.data(), features.data(), offsets.data(),
                                mask_device.data(), seq, config_.text.hidden_size);
        }
    }

    std::vector<int32_t> host_positions = build_positions(
        input.token_ids, input.mm_token_type_ids, input.images, input.past_len);
    GpuBuffer<int32_t> positions0_local;
    int32_t* positions0 = nullptr;
    if (qwen35_persistent_io_enabled()) {
        positions0 = positions0_.data();
        queue0.memcpy(positions0, host_positions.data(),
                      host_positions.size() * sizeof(int32_t));
    } else {
        positions0_local = GpuBuffer<int32_t>(host_positions.size(), queue0);
        positions0_local.upload(host_positions.data(), host_positions.size());
        positions0 = positions0_local.data();
    }

    int loaded_layers = static_cast<int>(weights_.layers.size());
    int first_stage_end = std::min(split_layer_, loaded_layers);
    for (int layer = 0; layer < first_stage_end; ++layer)
        run_layer(context0, weights_.layers[layer], workspace0_, caches_.kv[layer],
                  caches_.delta[layer], hidden0_.data(), normalized0_.data(),
                  sublayer0_.data(), positions0, seq, input.past_len);

    bf16* last_device = hidden0_.data() + (size_t)(seq - 1) * config_.text.hidden_size;
    if (loaded_layers > split_layer_) {
        auto& context1 = GpuEngine::get(1);
        size_t hidden_count = (size_t)seq * config_.text.hidden_size;
        std::vector<bf16> transfer_local;
        bf16* transfer = nullptr;
        if (qwen35_persistent_io_enabled()) {
            transfer = transfer_host_.data();
        } else {
            transfer_local.resize(hidden_count);
            transfer = transfer_local.data();
        }
        queue0.memcpy(transfer, hidden0_.data(), hidden_count * sizeof(bf16)).wait();
        context1.queue.memcpy(hidden1_.data(), transfer,
                              hidden_count * sizeof(bf16)).wait();
        GpuBuffer<int32_t> positions1_local;
        int32_t* positions1 = nullptr;
        if (qwen35_persistent_io_enabled()) {
            positions1 = positions1_.data();
            context1.queue.memcpy(positions1, host_positions.data(),
                                  host_positions.size() * sizeof(int32_t));
        } else {
            positions1_local = GpuBuffer<int32_t>(host_positions.size(), context1.queue);
            positions1_local.upload(host_positions.data(), host_positions.size());
            positions1 = positions1_local.data();
        }
        for (int layer = split_layer_; layer < loaded_layers; ++layer)
            run_layer(context1, weights_.layers[layer], workspace1_, caches_.kv[layer],
                      caches_.delta[layer], hidden1_.data(), normalized1_.data(),
                      sublayer1_.data(), positions1, seq, input.past_len);
        if (mtp_state_.ready()) {
            // The head runs on GPU 0 over every position, so the split model
            // has to bring the whole stage-2 output back rather than just the
            // final row. One seq x H transfer per forward, amortized over the
            // prefill and negligible at seq == 1.
            context1.queue.memcpy(transfer, hidden1_.data(),
                                  hidden_count * sizeof(bf16)).wait();
            queue0.memcpy(backbone_hidden_.data(), transfer,
                          hidden_count * sizeof(bf16)).wait();
        }
        std::vector<bf16> last_local;
        bf16* last = transfer;
        if (!qwen35_persistent_io_enabled()) {
            last_local.resize(config_.text.hidden_size);
            last = last_local.data();
        }
        context1.queue.memcpy(last,
            hidden1_.data() + (size_t)(seq - 1) * config_.text.hidden_size,
            (size_t)config_.text.hidden_size * sizeof(bf16)).wait();
        queue0.memcpy(normalized0_.data(), last,
                      (size_t)config_.text.hidden_size * sizeof(bf16)).wait();
        last_device = normalized0_.data();
    } else if (mtp_state_.ready()) {
        queue0.memcpy(backbone_hidden_.data(), hidden0_.data(),
                      (size_t)seq * config_.text.hidden_size * sizeof(bf16)).wait();
    }

    // Captured before the final norm below, which can run in place over the
    // same buffer. The head's checkpoint expects the pre-norm state; handing it
    // the normalized one degrades the head silently.
    if (mtp_state_.ready()) {
        backbone_hidden_base_ = input.past_len;
        backbone_hidden_len_ = seq;
        // Positions 1..seq-1 pair a known hidden state with a known following
        // token, so the head can advance now. The last position has no
        // successor yet and waits for mtp_draft. A long prompt is covered a
        // window at a time, keeping the head's scratch O(window) while its
        // cache still spans the whole context.
        if (seq > 1) {
            int pairs = seq - 1;
            std::vector<int> following(input.token_ids.begin() + 1,
                                       input.token_ids.end());
            std::vector<int32_t> following_positions((size_t)3 * pairs);
            for (int axis = 0; axis < 3; ++axis)
                for (int token = 0; token < pairs; ++token)
                    following_positions[(size_t)axis * pairs + token] =
                        host_positions[(size_t)axis * seq + token + 1];
            advance_mtp(following, following_positions, input.past_len + 1);
        }
    }

    rms_norm(queue0, last_device, weights_.final_norm.data(), normalized0_.data(),
             1, config_.text.hidden_size, config_.text.rms_norm_eps);
    GpuBuffer<bf16> logits_bf16_local;
    bf16* logits_bf16 = nullptr;
    if (qwen35_persistent_io_enabled())
        logits_bf16 = logits_bf16_.data();
    else {
        logits_bf16_local = GpuBuffer<bf16>(config_.text.vocab_size, queue0);
        logits_bf16 = logits_bf16_local.data();
    }
    if (const auto* fp8 = std::get_if<Fp8Linear>(&weights_.lm_head)) {
        matmul_fp8(normalized0_.data(), 1, config_.text.hidden_size,
                   *fp8, logits_bf16, context0);
    } else {
        matmul_bf16(normalized0_.data(), 1, config_.text.hidden_size,
                    std::get<GpuBuffer<bf16>>(weights_.lm_head).data(),
                    config_.text.vocab_size, logits_bf16, context0);
    }
    GpuBuffer<float> logits_f32_local;
    float* logits_f32 = nullptr;
    if (qwen35_persistent_io_enabled())
        logits_f32 = logits_f32_.data();
    else {
        logits_f32_local = GpuBuffer<float>(config_.text.vocab_size, queue0);
        logits_f32 = logits_f32_local.data();
    }
    bf16_to_f32(queue0, logits_bf16, logits_f32, config_.text.vocab_size);
    std::vector<float> logits(config_.text.vocab_size);
    queue0.memcpy(logits.data(), logits_f32,
                  logits.size() * sizeof(float)).wait();
    return logits;
}

// `next_tokens[i]` is the token at `first_embedded_position + i`, paired with
// the backbone hidden state one position earlier. Its KV slot is position - 1,
// so the slots the prefill fills stay contiguous with the slot each later
// draft appends. Long prompts are covered a window at a time.
//
// A batch is issued in one call only when the head's cache is empty. Otherwise
// it goes one token at a time: the attention kernel handles a large batch at
// past == 0 and a single token at any past, but a small batch against a large
// past drives the device into UR_RESULT_ERROR_DEVICE_LOST, and nothing before
// the head ever produced that shape. The per-token path is what decoding uses
// anyway, so the cost is only the extra launches.
void Qwen35Model::advance_mtp(const std::vector<int>& next_tokens,
                              const std::vector<int32_t>& positions,
                              int first_embedded_position) {
    auto& context0 = GpuEngine::get(0);
    auto& queue0 = context0.queue;
    int total = static_cast<int>(next_tokens.size());
    if (total <= 0) return;
    int hidden_size = config_.text.hidden_size;
    int first_slot = first_embedded_position - 1;

    for (int done = 0; done < total;) {
        int slot = first_slot + done;
        int count = std::min(mtp_window_, total - done);
        std::vector<int32_t> tokens(next_tokens.begin() + done,
                                    next_tokens.begin() + done + count);
        std::vector<int32_t> slice((size_t)3 * count);
        for (int axis = 0; axis < 3; ++axis)
            for (int i = 0; i < count; ++i)
                slice[(size_t)axis * count + i] =
                    positions[(size_t)axis * total + done + i];

        // Blocking. sycl::queue::memcpy is asynchronous and these sources are
        // loop-local, so letting them fall out of scope hands the copy freed
        // host memory. The token ids it then lands in the device buffer are
        // arbitrary, embedding_lookup indexes the table out of bounds with
        // them, and the failure surfaces as UR_RESULT_ERROR_DEVICE_LOST far
        // from here. A few hundred bytes once per window costs nothing.
        queue0.memcpy(mtp_tokens_.data(), tokens.data(),
                      tokens.size() * sizeof(int32_t)).wait();
        queue0.memcpy(mtp_positions_.data(), slice.data(),
                      slice.size() * sizeof(int32_t)).wait();

        // Hidden state one position before each embedded token.
        int hidden_index = first_embedded_position - 1 + done;
        const bf16* hidden =
            backbone_hidden_.data() +
            (size_t)(hidden_index - backbone_hidden_base_) * hidden_size;
        qwen35_mtp_forward(context0, weights_.mtp, weights_.embed_tokens,
                           mtp_state_, workspace0_, hidden, mtp_tokens_.data(),
                           mtp_positions_.data(), mtp_out_.data(), count,
                           slot, config_);
        done += count;
    }
    queue0.wait();
}

std::vector<float> Qwen35Model::mtp_logits_from(const bf16* mtp_hidden) {
    auto& context0 = GpuEngine::get(0);
    auto& queue0 = context0.queue;
    // The head shares the backbone's output projection; the checkpoint sets
    // mtp_use_dedicated_embeddings=false.
    GpuBuffer<bf16> logits_bf16_local;
    bf16* logits_bf16 = nullptr;
    if (qwen35_persistent_io_enabled())
        logits_bf16 = logits_bf16_.data();
    else {
        logits_bf16_local = GpuBuffer<bf16>(config_.text.vocab_size, queue0);
        logits_bf16 = logits_bf16_local.data();
    }
    if (const auto* fp8 = std::get_if<Fp8Linear>(&weights_.lm_head)) {
        matmul_fp8(mtp_hidden, 1, config_.text.hidden_size, *fp8, logits_bf16,
                   context0);
    } else {
        matmul_bf16(mtp_hidden, 1, config_.text.hidden_size,
                    std::get<GpuBuffer<bf16>>(weights_.lm_head).data(),
                    config_.text.vocab_size, logits_bf16, context0);
    }
    GpuBuffer<float> logits_f32_local;
    float* logits_f32 = nullptr;
    if (qwen35_persistent_io_enabled())
        logits_f32 = logits_f32_.data();
    else {
        logits_f32_local = GpuBuffer<float>(config_.text.vocab_size, queue0);
        logits_f32 = logits_f32_local.data();
    }
    bf16_to_f32(queue0, logits_bf16, logits_f32, config_.text.vocab_size);
    std::vector<float> logits(config_.text.vocab_size);
    queue0.memcpy(logits.data(), logits_f32, logits.size() * sizeof(float)).wait();
    return logits;
}

std::vector<float> Qwen35Model::mtp_draft(int next_token, int position) {
    if (!mtp_state_.ready()) return {};
    if (backbone_hidden_len_ <= 0)
        throw std::runtime_error("Qwen3.5 MTP draft before any forward");
    // The draft pairs the hidden state of the position before `next_token`
    // with that token's embedding, so the caller's sampled token must sit
    // immediately after the range the last forward covered.
    int hidden_index = position - 1;
    if (hidden_index < backbone_hidden_base_ ||
        hidden_index >= backbone_hidden_base_ + backbone_hidden_len_)
        throw std::runtime_error("Qwen3.5 MTP draft position outside the last forward");

    auto& context0 = GpuEngine::get(0);
    auto& queue0 = context0.queue;
    // Blocking for the same reason as advance_mtp: both sources are locals.
    int32_t token = next_token;
    queue0.memcpy(mtp_tokens_.data(), &token, sizeof(int32_t)).wait();
    std::vector<int32_t> positions(3, position + rope_delta_);
    queue0.memcpy(mtp_positions_.data(), positions.data(),
                  positions.size() * sizeof(int32_t)).wait();

    const bf16* hidden =
        backbone_hidden_.data() +
        (size_t)(hidden_index - backbone_hidden_base_) * config_.text.hidden_size;
    qwen35_mtp_forward(context0, weights_.mtp, weights_.embed_tokens, mtp_state_,
                       workspace0_, hidden, mtp_tokens_.data(),
                       mtp_positions_.data(), mtp_out_.data(), 1,
                       position - 1, config_);
    return mtp_logits_from(mtp_out_.data());
}

std::vector<float> Qwen35Model::forward_verify(const std::vector<int>& tokens,
                                               int past) {
    int seq = static_cast<int>(tokens.size());
    if (seq < 1 || seq > kMaxVerify)
        throw std::runtime_error("Qwen3.5 verify batch out of range");
    if (!mtp_state_.ready())
        throw std::runtime_error("Qwen3.5 verify requires the MTP path");

    // The ordinary forward already runs the stack and leaves every position's
    // pre-final-norm hidden state in backbone_hidden_; only the projection to
    // logits has to be repeated for the positions it discarded.
    forward(ForwardInput{tokens, past});

    auto& context0 = GpuEngine::get(0);
    auto& queue0 = context0.queue;
    const auto& c = config_.text;
    rms_norm(queue0, backbone_hidden_.data(), weights_.final_norm.data(),
             verify_normed_.data(), seq, c.hidden_size, c.rms_norm_eps);
    if (const auto* fp8 = std::get_if<Fp8Linear>(&weights_.lm_head)) {
        matmul_fp8(verify_normed_.data(), seq, c.hidden_size, *fp8,
                   verify_logits_bf16_.data(), context0);
    } else {
        matmul_bf16(verify_normed_.data(), seq, c.hidden_size,
                    std::get<GpuBuffer<bf16>>(weights_.lm_head).data(),
                    c.vocab_size, verify_logits_bf16_.data(), context0);
    }
    bf16_to_f32(queue0, verify_logits_bf16_.data(), verify_logits_f32_.data(),
                seq * c.vocab_size);
    std::vector<float> logits((size_t)seq * c.vocab_size);
    queue0.memcpy(logits.data(), verify_logits_f32_.data(),
                  logits.size() * sizeof(float)).wait();
    return logits;
}

std::vector<int> Qwen35Model::generate_speculative(
    const std::vector<int>& prompt, int max_tokens, SpecStats& stats) {
    if (!mtp_state_.ready())
        throw std::runtime_error("Qwen3.5 speculative decode requires the MTP head");
    const int vocab = config_.text.vocab_size;
    auto pick = [&](const std::vector<float>& logits, int position) {
        const float* row = logits.data() + (size_t)position * vocab;
        int best = 0;
        for (int i = 1; i < vocab; ++i)
            if (row[i] > row[best]) best = i;
        return best;
    };
    auto now = [] { return std::chrono::steady_clock::now(); };
    auto ms = [](auto start) {
        return std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - start).count();
    };

    std::vector<int> output;
    std::vector<float> logits = forward(ForwardInput{prompt, 0});
    ++stats.forwards;
    int past = static_cast<int>(prompt.size());
    int pending = pick(logits, 0);

    while ((int)output.size() < max_tokens) {
        if (info_.is_eos(pending)) break;
        output.push_back(pending);
        if ((int)output.size() >= max_tokens) break;
        if (past + 2 > max_seq_len_) break;

        auto t_draft = now();
        std::vector<float> draft_logits = mtp_draft(pending, past);
        stats.draft_ms += ms(t_draft);
        ++stats.drafts;
        int draft = pick(draft_logits, 0);

        // Snapshot before the drafted token touches any in-place state.
        auto t_save = now();
        caches_.save();
        int mtp_filled = mtp_state_.filled;
        stats.rollback_ms += ms(t_save);

        auto t_verify = now();
        std::vector<float> verified = forward_verify({pending, draft}, past);
        stats.verify_ms += ms(t_verify);
        ++stats.forwards;
        ++stats.rounds;

        int truth = pick(verified, 0);   // the real token at past + 1
        if (truth == draft) {
            ++stats.accepts;
            output.push_back(draft);
            pending = pick(verified, 1);
            past += 2;
            continue;
        }

        // Miss. The caches now carry the rejected token, so roll them back and
        // replay the same two positions with the token the backbone actually
        // produced. That costs a second pass but still yields two tokens, so a
        // miss degrades to ordinary decoding rather than below it.
        auto t_restore = now();
        caches_.restore();
        mtp_state_.filled = mtp_filled;
        stats.rollback_ms += ms(t_restore);

        auto t_replay = now();
        std::vector<float> replayed = forward_verify({pending, truth}, past);
        stats.verify_ms += ms(t_replay);
        ++stats.forwards;
        output.push_back(truth);
        pending = pick(replayed, 1);
        past += 2;
    }
    return output;
}

void Qwen35Model::reset_cache() {
    caches_.reset();
    mtp_state_.reset();
    backbone_hidden_len_ = 0;
    backbone_hidden_base_ = 0;
    rope_delta_ = 0;
}
