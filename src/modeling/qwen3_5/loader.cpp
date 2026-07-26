#include "loader.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

#include "../../common/gpu/engine.hpp"

namespace {

bool fused_projections_enabled() {
    const char* value = std::getenv("ARCAINE_QWEN35_FUSED_PROJECTIONS");
    if (!value) return true;
    return std::strcmp(value, "0") != 0 && std::strcmp(value, "off") != 0 &&
           std::strcmp(value, "false") != 0 && std::strcmp(value, "no") != 0;
}

class TrackingTensorSource final : public TensorSource {
public:
    explicit TrackingTensorSource(const ShardedSafetensors& source) : source_(source) {}

    const TensorView& get(const std::string& name) const override {
        const TensorView& view = source_.get(name);
        consumed_.insert(name);
        return view;
    }

    bool has(const std::string& name) const override { return source_.has(name); }
    bool consumed(const std::string& name) const { return consumed_.count(name) != 0; }

private:
    const ShardedSafetensors& source_;
    mutable std::unordered_set<std::string> consumed_;
};

void expect_tensor(const TensorSource& source, const std::string& name,
                   const char* dtype, std::vector<int64_t> shape) {
    const TensorView& view = source.get(name);
    if (view.dtype != dtype || view.shape != shape) {
        std::ostringstream message;
        message << "Unexpected tensor metadata for " << name << ": dtype="
                << view.dtype << " shape=(";
        for (size_t i = 0; i < view.shape.size(); ++i) {
            if (i) message << ',';
            message << view.shape[i];
        }
        message << ')';
        throw std::runtime_error(message.str());
    }
}

GpuBuffer<bf16> load_bf16(const TensorSource& source, const std::string& name,
                          std::vector<int64_t> shape, sycl::queue& queue,
                          bool add_one = false) {
    expect_tensor(source, name, "BF16", std::move(shape));
    return add_one ? upload_plus_one(source.get(name), queue, name.c_str())
                   : upload(source.get(name), queue, name.c_str());
}

// Which compressed form a projection is stored in is decided by the checkpoint,
// not by the architecture, so probe the tensors rather than assuming a layout.
// `prefixes` holds one entry for a plain projection or several to fuse into a
// single concatenated weight.
Qwen35Linear load_linear(const TensorSource& source,
                         const std::vector<std::string>& prefixes,
                         sycl::queue& queue) {
    if (prefixes.empty()) throw std::runtime_error("load_linear needs a prefix");
    const std::string& first = prefixes.front();

    bool all_nvfp4 = true;
    bool all_fp8 = true;
    for (const std::string& prefix : prefixes) {
        if (!source.has(prefix + ".weight_packed")) all_nvfp4 = false;
        if (!source.has(prefix + ".weight_scale") || !source.has(prefix + ".weight"))
            all_fp8 = false;
    }

    Qwen35Linear linear;
    if (all_nvfp4) {
        linear.kind = Qwen35Linear::Kind::Nvfp4;
        linear.nvfp4 = prefixes.size() == 1
            ? upload_nvfp4_linear(source, first, queue)
            : upload_nvfp4_linear_concat(source, prefixes, queue);
        linear.in_features = linear.nvfp4.in_features;
        linear.out_features = linear.nvfp4.out_features;
        return linear;
    }
    if (all_fp8) {
        linear.kind = Qwen35Linear::Kind::Fp8;
        linear.fp8 = prefixes.size() == 1
            ? upload_fp8_linear(source, first, queue)
            : upload_fp8_linear_concat(source, prefixes, queue);
        linear.in_features = linear.fp8.in_features;
        linear.out_features = linear.fp8.out_features;
        return linear;
    }
    if (prefixes.size() == 1 && source.has(first + ".weight")) {
        const TensorView& weight = source.get(first + ".weight");
        if (weight.shape.size() != 2)
            throw std::runtime_error("Expected 2D dense weight: " + first);
        linear.kind = Qwen35Linear::Kind::Bf16;
        linear.out_features = static_cast<int>(weight.shape[0]);
        linear.in_features = static_cast<int>(weight.shape[1]);
        linear.dense = upload(weight, queue, (first + ".weight").c_str());
        return linear;
    }
    throw std::runtime_error(
        "No usable weights for projection " + first +
        " (expected NVFP4 .weight_packed, FP8 .weight + .weight_scale, or a dense .weight)");
}

// A fused weight carries one destination scale, which NVFP4 only permits when
// every input shares both global scales. Quantizers calibrate per module, so
// whether a fusion is legal is a property of the checkpoint.
bool can_fuse(const TensorSource& source, const std::vector<std::string>& prefixes) {
    if (!fused_projections_enabled()) return false;
    bool any_nvfp4 = false;
    for (const std::string& prefix : prefixes)
        if (source.has(prefix + ".weight_packed")) any_nvfp4 = true;
    if (!any_nvfp4) return true;
    return nvfp4_globals_match(source, prefixes);
}

void expect_linear(const Qwen35Linear& linear, int in, int out, const std::string& name) {
    if (linear.in_features != in || linear.out_features != out)
        throw std::runtime_error(
            "Unexpected linear shape for " + name + ": got (" +
            std::to_string(linear.in_features) + "," + std::to_string(linear.out_features) +
            ") expected (" + std::to_string(in) + "," + std::to_string(out) + ")");
}

// Some recipes emit tensors this engine has no use for: static KV-cache scales
// (the KV cache is BF16 here) and the activation scale of a projection that is
// decompressed to dense BF16 at load. Consume them so the completeness check
// below still reports genuinely unread tensors.
void ignore_if_present(const TensorSource& source, const std::string& name) {
    if (source.has(name)) (void)source.get(name);
}

// Load a small projection as dense BF16 whichever form it is stored in. The
// linear-attention beta/gate projections are [num_value_heads, hidden] -- far
// too narrow for a decompression GEMM to pay for itself, and the DeltaNet path
// wants them as plain BF16 operands.
GpuBuffer<bf16> load_dense_projection(const TensorSource& source, const std::string& prefix,
                                      std::vector<int64_t> shape, sycl::queue& queue) {
    if (!source.has(prefix + ".weight_packed"))
        return load_bf16(source, prefix + ".weight", std::move(shape), queue);
    int out_features = 0;
    int in_features = 0;
    GpuBuffer<bf16> weight =
        dequantize_nvfp4_to_bf16(source, prefix, queue, &out_features, &in_features);
    // Decompressing the weight leaves the activation scale unused: this
    // projection runs as a plain BF16 matmul, so its inputs are never packed.
    ignore_if_present(source, prefix + ".input_global_scale");
    if (shape.size() != 2 || out_features != shape[0] || in_features != shape[1])
        throw std::runtime_error("Unexpected NVFP4 dense projection shape: " + prefix);
    return weight;
}

Qwen35VisionWeights load_vision(const TensorSource& source,
                                const Qwen35Config& config,
                                sycl::queue& queue) {
    const auto& v = config.vision;
    const std::string prefix = "model.visual.";
    Qwen35VisionWeights weights;
    weights.patch_weight = load_bf16(source, prefix + "patch_embed.proj.weight",
        {v.hidden_size, v.in_channels, v.temporal_patch_size, v.patch_size, v.patch_size}, queue);
    weights.patch_bias = load_bf16(source, prefix + "patch_embed.proj.bias",
        {v.hidden_size}, queue);
    weights.position_embedding = load_bf16(source, prefix + "pos_embed.weight",
        {v.num_position_embeddings, v.hidden_size}, queue);
    weights.blocks.reserve(v.depth);
    for (int i = 0; i < v.depth; ++i) {
        std::string block = prefix + "blocks." + std::to_string(i) + ".";
        Qwen35VisionBlockWeights w;
        w.norm1_weight = load_bf16(source, block + "norm1.weight", {v.hidden_size}, queue);
        w.norm1_bias = load_bf16(source, block + "norm1.bias", {v.hidden_size}, queue);
        w.norm2_weight = load_bf16(source, block + "norm2.weight", {v.hidden_size}, queue);
        w.norm2_bias = load_bf16(source, block + "norm2.bias", {v.hidden_size}, queue);
        w.qkv_weight = load_bf16(source, block + "attn.qkv.weight",
                                 {3 * v.hidden_size, v.hidden_size}, queue);
        w.qkv_bias = load_bf16(source, block + "attn.qkv.bias", {3 * v.hidden_size}, queue);
        w.proj_weight = load_bf16(source, block + "attn.proj.weight",
                                  {v.hidden_size, v.hidden_size}, queue);
        w.proj_bias = load_bf16(source, block + "attn.proj.bias", {v.hidden_size}, queue);
        w.fc1_weight = load_bf16(source, block + "mlp.linear_fc1.weight",
                                 {v.intermediate_size, v.hidden_size}, queue);
        w.fc1_bias = load_bf16(source, block + "mlp.linear_fc1.bias",
                               {v.intermediate_size}, queue);
        w.fc2_weight = load_bf16(source, block + "mlp.linear_fc2.weight",
                                 {v.hidden_size, v.intermediate_size}, queue);
        w.fc2_bias = load_bf16(source, block + "mlp.linear_fc2.bias",
                               {v.hidden_size}, queue);
        weights.blocks.push_back(std::move(w));
    }
    int merged = v.hidden_size * v.spatial_merge_size * v.spatial_merge_size;
    weights.merger_norm_weight = load_bf16(source, prefix + "merger.norm.weight",
                                            {v.hidden_size}, queue);
    weights.merger_norm_bias = load_bf16(source, prefix + "merger.norm.bias",
                                          {v.hidden_size}, queue);
    weights.merger_fc1_weight = load_bf16(source, prefix + "merger.linear_fc1.weight",
                                           {merged, merged}, queue);
    weights.merger_fc1_bias = load_bf16(source, prefix + "merger.linear_fc1.bias",
                                         {merged}, queue);
    weights.merger_fc2_weight = load_bf16(source, prefix + "merger.linear_fc2.weight",
                                           {v.out_hidden_size, merged}, queue);
    weights.merger_fc2_bias = load_bf16(source, prefix + "merger.linear_fc2.bias",
                                         {v.out_hidden_size}, queue);
    return weights;
}

Qwen35MtpWeights load_mtp(const TensorSource& source,
                          const Qwen35Config& config,
                          sycl::queue& queue) {
    if (config.mtp_num_hidden_layers != 1)
        throw std::runtime_error("Only the checkpoint's one-layer MTP head is supported");
    int H = config.text.hidden_size;
    int I = config.text.intermediate_size;
    int q_out = config.text.num_attention_heads * config.text.head_dim * 2;
    int kv_out = config.text.num_key_value_heads * config.text.head_dim;
    int attn_out = config.text.num_attention_heads * config.text.head_dim;
    Qwen35MtpWeights w;
    w.fc = load_bf16(source, "mtp.fc.weight", {H, 2 * H}, queue);
    w.pre_fc_norm_embedding = load_bf16(source, "mtp.pre_fc_norm_embedding.weight", {H}, queue, true);
    w.pre_fc_norm_hidden = load_bf16(source, "mtp.pre_fc_norm_hidden.weight", {H}, queue, true);
    w.input_layernorm = load_bf16(source, "mtp.layers.0.input_layernorm.weight", {H}, queue, true);
    w.post_attention_layernorm = load_bf16(
        source, "mtp.layers.0.post_attention_layernorm.weight", {H}, queue, true);
    w.q_proj = load_bf16(source, "mtp.layers.0.self_attn.q_proj.weight", {q_out, H}, queue);
    w.k_proj = load_bf16(source, "mtp.layers.0.self_attn.k_proj.weight", {kv_out, H}, queue);
    w.v_proj = load_bf16(source, "mtp.layers.0.self_attn.v_proj.weight", {kv_out, H}, queue);
    w.o_proj = load_bf16(source, "mtp.layers.0.self_attn.o_proj.weight", {H, attn_out}, queue);
    w.q_norm = load_bf16(source, "mtp.layers.0.self_attn.q_norm.weight",
                          {config.text.head_dim}, queue, true);
    w.k_norm = load_bf16(source, "mtp.layers.0.self_attn.k_norm.weight",
                          {config.text.head_dim}, queue, true);
    w.gate_proj = load_bf16(source, "mtp.layers.0.mlp.gate_proj.weight", {I, H}, queue);
    w.up_proj = load_bf16(source, "mtp.layers.0.mlp.up_proj.weight", {I, H}, queue);
    w.down_proj = load_bf16(source, "mtp.layers.0.mlp.down_proj.weight", {H, I}, queue);
    w.norm = load_bf16(source, "mtp.norm.weight", {H}, queue, true);
    return w;
}

}  // namespace

Qwen35Weights load_qwen35_weights(
    const ShardedSafetensors& checkpoint,
    const Qwen35Config& config,
    int split_layer,
    int max_layers) {
    TrackingTensorSource source(checkpoint);
    const auto& c = config.text;
    int total_layers = c.num_hidden_layers;
    if (max_layers <= 0 || max_layers > total_layers) max_layers = total_layers;
    if (split_layer < 0 || split_layer > total_layers) split_layer = total_layers;

    auto& queue0 = GpuEngine::get(0).queue;
    Qwen35Weights weights;
    weights.embed_tokens = load_bf16(
        source, "model.language_model.embed_tokens.weight",
        {c.vocab_size, c.hidden_size}, queue0);
    weights.final_norm = load_bf16(
        source, "model.language_model.norm.weight", {c.hidden_size}, queue0, true);
    weights.lm_head = load_linear(source, {"lm_head"}, queue0);
    expect_linear(weights.lm_head, c.hidden_size, c.vocab_size, "lm_head");

    weights.layers.reserve(max_layers);
    for (int i = 0; i < max_layers; ++i) {
        int gpu = (i < split_layer || GpuEngine::count() < 2) ? 0 : 1;
        auto& queue = GpuEngine::get(gpu).queue;
        std::string layer_prefix = "model.language_model.layers." + std::to_string(i) + ".";
        Qwen35LayerWeights layer;
        layer.index = i;
        layer.gpu = gpu;
        layer.full_attention = c.is_full_attn(i);
        layer.input_layernorm = load_bf16(
            source, layer_prefix + "input_layernorm.weight", {c.hidden_size}, queue, true);
        layer.post_attention_layernorm = load_bf16(
            source, layer_prefix + "post_attention_layernorm.weight", {c.hidden_size}, queue, true);

        if (layer.full_attention) {
            std::string prefix = layer_prefix + "self_attn.";
            Qwen35FullAttentionWeights attention;
            std::vector<std::string> qkv = {prefix + "q_proj", prefix + "k_proj",
                                            prefix + "v_proj"};
            attention.fused_projections = can_fuse(source, qkv);
            if (attention.fused_projections)
                attention.qkv_proj = load_linear(source, qkv, queue);
            else {
                attention.q_proj = load_linear(source, {qkv[0]}, queue);
                attention.k_proj = load_linear(source, {qkv[1]}, queue);
                attention.v_proj = load_linear(source, {qkv[2]}, queue);
            }
            attention.o_proj = load_linear(source, {prefix + "o_proj"}, queue);
            int q_out = c.num_attention_heads * c.head_dim * 2;
            int kv_out = c.num_key_value_heads * c.head_dim;
            int attn_out = c.num_attention_heads * c.head_dim;
            if (attention.fused_projections)
                expect_linear(attention.qkv_proj, c.hidden_size, q_out + 2 * kv_out,
                              prefix + "qkv_proj");
            else {
                expect_linear(attention.q_proj, c.hidden_size, q_out, prefix + "q_proj");
                expect_linear(attention.k_proj, c.hidden_size, kv_out, prefix + "k_proj");
                expect_linear(attention.v_proj, c.hidden_size, kv_out, prefix + "v_proj");
            }
            expect_linear(attention.o_proj, attn_out, c.hidden_size, prefix + "o_proj");
            attention.q_norm = load_bf16(source, prefix + "q_norm.weight", {c.head_dim}, queue, true);
            attention.k_norm = load_bf16(source, prefix + "k_norm.weight", {c.head_dim}, queue, true);
            ignore_if_present(source, prefix + "k_scale");
            ignore_if_present(source, prefix + "v_scale");
            layer.mixer = std::move(attention);
        } else {
            std::string prefix = layer_prefix + "linear_attn.";
            Qwen35LinearAttentionWeights attention;
            int key_dim = c.linear_num_key_heads * c.linear_key_head_dim;
            int value_dim = c.linear_num_value_heads * c.linear_value_head_dim;
            int conv_dim = 2 * key_dim + value_dim;
            std::vector<std::string> qkvz = {prefix + "in_proj_qkv", prefix + "in_proj_z"};
            attention.fused_projections = can_fuse(source, qkvz);
            if (attention.fused_projections)
                attention.in_proj_qkvz = load_linear(source, qkvz, queue);
            else {
                attention.in_proj_qkv = load_linear(source, {qkvz[0]}, queue);
                attention.in_proj_z = load_linear(source, {qkvz[1]}, queue);
            }
            attention.out_proj = load_linear(source, {prefix + "out_proj"}, queue);
            if (attention.fused_projections)
                expect_linear(attention.in_proj_qkvz, c.hidden_size,
                              conv_dim + value_dim, prefix + "in_proj_qkvz");
            else {
                expect_linear(attention.in_proj_qkv, c.hidden_size, conv_dim,
                              prefix + "in_proj_qkv");
                expect_linear(attention.in_proj_z, c.hidden_size, value_dim,
                              prefix + "in_proj_z");
            }
            expect_linear(attention.out_proj, value_dim, c.hidden_size, prefix + "out_proj");
            attention.in_proj_a = load_dense_projection(
                source, prefix + "in_proj_a",
                {c.linear_num_value_heads, c.hidden_size}, queue);
            attention.in_proj_b = load_dense_projection(
                source, prefix + "in_proj_b",
                {c.linear_num_value_heads, c.hidden_size}, queue);
            attention.in_proj_ba = GpuBuffer<bf16>(
                (size_t)2 * c.linear_num_value_heads * c.hidden_size, queue);
            queue.memcpy(
                attention.in_proj_ba.data(), attention.in_proj_b.data(),
                (size_t)c.linear_num_value_heads * c.hidden_size * sizeof(bf16));
            queue.memcpy(
                attention.in_proj_ba.data() +
                    (size_t)c.linear_num_value_heads * c.hidden_size,
                attention.in_proj_a.data(),
                (size_t)c.linear_num_value_heads * c.hidden_size * sizeof(bf16));
            attention.conv1d = load_bf16(source, prefix + "conv1d.weight",
                                          {conv_dim, 1, c.linear_conv_kernel_dim}, queue);
            attention.conv1d_time_major = GpuBuffer<bf16>(
                (size_t)conv_dim * c.linear_conv_kernel_dim, queue);
            {
                const bf16* source_weight = attention.conv1d.data();
                bf16* destination_weight = attention.conv1d_time_major.data();
                int kernel = c.linear_conv_kernel_dim;
                queue.submit([&](sycl::handler& handler) {
                    handler.parallel_for(
                        sycl::range<2>((size_t)kernel, (size_t)conv_dim),
                        [=](sycl::id<2> id) {
                            int tap = static_cast<int>(id[0]);
                            int channel = static_cast<int>(id[1]);
                            destination_weight[(size_t)tap * conv_dim + channel] =
                                source_weight[(size_t)channel * kernel + tap];
                        });
                });
            }
            attention.A_log = load_bf16(source, prefix + "A_log",
                                         {c.linear_num_value_heads}, queue);
            attention.dt_bias = load_bf16(source, prefix + "dt_bias",
                                           {c.linear_num_value_heads}, queue);
            attention.norm = load_bf16(source, prefix + "norm.weight",
                                        {c.linear_value_head_dim}, queue);
            layer.mixer = std::move(attention);
        }

        std::string mlp = layer_prefix + "mlp.";
        layer.mlp.gate_up = load_linear(
            source, {mlp + "gate_proj", mlp + "up_proj"}, queue);
        layer.mlp.down = load_linear(source, {mlp + "down_proj"}, queue);
        expect_linear(layer.mlp.gate_up, c.hidden_size, 2 * c.intermediate_size,
                      mlp + "gate_up");
        expect_linear(layer.mlp.down, c.intermediate_size, c.hidden_size,
                      mlp + "down_proj");
        weights.layers.push_back(std::move(layer));
        std::printf("[qwen35-load] layer %d/%d on GPU %d\n", i + 1, max_layers, gpu);
    }

    weights.vision = load_vision(source, config, queue0);
    weights.mtp = load_mtp(source, config, queue0);

    if (max_layers == total_layers) {
        std::vector<std::string> missing;
        for (const std::string& name : checkpoint.names())
            if (!source.consumed(name)) missing.push_back(name);
        if (!missing.empty()) {
            std::sort(missing.begin(), missing.end());
            std::ostringstream message;
            message << "Qwen3.5 checkpoint has " << missing.size()
                    << " unconsumed tensors; first entries:";
            for (size_t i = 0; i < std::min<size_t>(missing.size(), 16); ++i)
                message << "\n  " << missing[i];
            throw std::runtime_error(message.str());
        }
    }
    return weights;
}
