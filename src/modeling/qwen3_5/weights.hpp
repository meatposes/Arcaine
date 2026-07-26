#pragma once

#include <variant>
#include <vector>

#include "../../common/gpu/buffer.hpp"
#include "../../common/gpu/fp8.hpp"
#include "../../common/gpu/nvfp4.hpp"

// Which modules a checkpoint quantizes, and with what, is a property of the
// quantization recipe rather than of the architecture: published Qwen3.5 builds
// disagree about whether attention, the linear-attention projections and the
// LM head are NVFP4, FP8 or left in BF16. Every projection a recipe might skip
// or quantize differently is probed at load time and carried in this form, so
// one code path serves all of them.
struct Qwen35Linear {
    enum class Kind { Missing, Nvfp4, Fp8, Bf16 };
    Kind kind = Kind::Missing;
    int in_features = 0;
    int out_features = 0;
    Nvfp4Linear nvfp4;
    Fp8Linear fp8;
    GpuBuffer<bf16> dense;  // [out_features, in_features], row-major
    bool empty() const { return kind == Kind::Missing; }
};

struct Qwen35FullAttentionWeights {
    bool fused_projections = false;
    Qwen35Linear qkv_proj;
    Qwen35Linear q_proj;
    Qwen35Linear k_proj;
    Qwen35Linear v_proj;
    Qwen35Linear o_proj;
    GpuBuffer<bf16> q_norm;
    GpuBuffer<bf16> k_norm;
};

struct Qwen35LinearAttentionWeights {
    bool fused_projections = false;
    Qwen35Linear in_proj_qkvz;
    Qwen35Linear in_proj_qkv;
    Qwen35Linear in_proj_z;
    Qwen35Linear out_proj;
    GpuBuffer<bf16> in_proj_a;
    GpuBuffer<bf16> in_proj_b;
    GpuBuffer<bf16> in_proj_ba;
    GpuBuffer<bf16> conv1d;
    // [kernel, channel], used by the fused ESIMD M=1 DeltaNet path.
    GpuBuffer<bf16> conv1d_time_major;
    GpuBuffer<bf16> A_log;
    GpuBuffer<bf16> dt_bias;
    GpuBuffer<bf16> norm;
};

struct Qwen35MlpWeights {
    Qwen35Linear gate_up;
    Qwen35Linear down;
};

struct Qwen35LayerWeights {
    int index = 0;
    int gpu = 0;
    bool full_attention = false;
    std::variant<Qwen35FullAttentionWeights, Qwen35LinearAttentionWeights> mixer;
    Qwen35MlpWeights mlp;
    GpuBuffer<bf16> input_layernorm;
    GpuBuffer<bf16> post_attention_layernorm;
};

struct Qwen35VisionBlockWeights {
    GpuBuffer<bf16> norm1_weight;
    GpuBuffer<bf16> norm1_bias;
    GpuBuffer<bf16> norm2_weight;
    GpuBuffer<bf16> norm2_bias;
    GpuBuffer<bf16> qkv_weight;
    GpuBuffer<bf16> qkv_bias;
    GpuBuffer<bf16> proj_weight;
    GpuBuffer<bf16> proj_bias;
    GpuBuffer<bf16> fc1_weight;
    GpuBuffer<bf16> fc1_bias;
    GpuBuffer<bf16> fc2_weight;
    GpuBuffer<bf16> fc2_bias;
};

struct Qwen35VisionWeights {
    GpuBuffer<bf16> patch_weight;
    GpuBuffer<bf16> patch_bias;
    GpuBuffer<bf16> position_embedding;
    std::vector<Qwen35VisionBlockWeights> blocks;
    GpuBuffer<bf16> merger_norm_weight;
    GpuBuffer<bf16> merger_norm_bias;
    GpuBuffer<bf16> merger_fc1_weight;
    GpuBuffer<bf16> merger_fc1_bias;
    GpuBuffer<bf16> merger_fc2_weight;
    GpuBuffer<bf16> merger_fc2_bias;
};

struct Qwen35MtpWeights {
    GpuBuffer<bf16> fc;
    GpuBuffer<bf16> pre_fc_norm_embedding;
    GpuBuffer<bf16> pre_fc_norm_hidden;
    GpuBuffer<bf16> input_layernorm;
    GpuBuffer<bf16> post_attention_layernorm;
    GpuBuffer<bf16> q_proj;
    GpuBuffer<bf16> k_proj;
    GpuBuffer<bf16> v_proj;
    GpuBuffer<bf16> o_proj;
    GpuBuffer<bf16> q_norm;
    GpuBuffer<bf16> k_norm;
    GpuBuffer<bf16> gate_proj;
    GpuBuffer<bf16> up_proj;
    GpuBuffer<bf16> down_proj;
    GpuBuffer<bf16> norm;
};

struct Qwen35Weights {
    GpuBuffer<bf16> embed_tokens;
    GpuBuffer<bf16> final_norm;
    Qwen35Linear lm_head;
    std::vector<Qwen35LayerWeights> layers;
    Qwen35VisionWeights vision;
    Qwen35MtpWeights mtp;
};
