#include "quant_loader.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <vector>
#include <nlohmann/json.hpp>

// ---------------------------------------------------------------------------
// ShardedSafetensors
// ---------------------------------------------------------------------------
ShardedSafetensors::ShardedSafetensors(const std::string& model_dir) {
    std::ifstream f(model_dir + "/model.safetensors.index.json");
    if (!f) {
        // Single-file checkpoint: no shard index, just model.safetensors.
        // Map every tensor name to that one shard.
        std::string path = model_dir + "/model.safetensors";
        int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0)
            throw std::runtime_error(
                "Cannot open model.safetensors.index.json or model.safetensors in " + model_dir);
        posix_fadvise(fd, 0, 0, POSIX_FADV_WILLNEED);
        close(fd);
        shards_.push_back(std::make_unique<SafetensorsFile>(path));
        for (auto& [name, tv] : shards_[0]->all()) name_to_shard_[name] = 0;
        std::printf("[load] %zu tensors in single safetensors file\n",
                    name_to_shard_.size());
        return;
    }
    auto idx = nlohmann::json::parse(f);
    auto& wm = idx.at("weight_map");

    std::unordered_map<std::string, int> shard_id;
    for (auto& [name, shard] : wm.items()) {
        std::string s = shard.get<std::string>();
        auto it = shard_id.find(s);
        if (it == shard_id.end()) {
            shard_id[s] = (int)shards_.size();
            std::string path = model_dir + "/" + s;
            // Kick off async readahead of the whole shard so the staging
            // memcpys hit warm page cache instead of faulting from disk.
            int fd = open(path.c_str(), O_RDONLY);
            if (fd >= 0) { posix_fadvise(fd, 0, 0, POSIX_FADV_WILLNEED); close(fd); }
            shards_.push_back(std::make_unique<SafetensorsFile>(path));
            it = shard_id.find(s);
        }
        name_to_shard_[name] = it->second;
    }
    std::printf("[load] %zu tensors across %zu shards\n",
                name_to_shard_.size(), shards_.size());
}

const TensorView& ShardedSafetensors::get(const std::string& name) const {
    auto it = name_to_shard_.find(name);
    if (it == name_to_shard_.end())
        throw std::runtime_error("tensor not found: " + name);
    return shards_[it->second]->get(name);
}

bool ShardedSafetensors::has(const std::string& name) const {
    return name_to_shard_.count(name) > 0;
}

std::vector<std::string> ShardedSafetensors::names() const {
    std::vector<std::string> out;
    out.reserve(name_to_shard_.size());
    for (const auto& entry : name_to_shard_) out.push_back(entry.first);
    return out;
}

// Set DIFF_LOAD_TRACE=1 (legacy) or QUANT_LOAD_TRACE=1 for per-tensor
// stall debugging.
static bool g_trace = std::getenv("DIFF_LOAD_TRACE") != nullptr ||
                      std::getenv("QUANT_LOAD_TRACE") != nullptr;

// ---------------------------------------------------------------------------
// SYCL H2D memcpy directly from cold file-backed mmap pages degrades to one
// synchronous 4 KB fault at a time (~10 MB/s).  Stage through a reusable host
// buffer: the CPU memcpy fault path gets kernel readahead (GB/s) and the
// device copy from malloc'd memory runs at full PCIe speed.
// ---------------------------------------------------------------------------
GpuBuffer<bf16> upload(const TensorView& tv, sycl::queue& q, const char* name) {
    size_t n = tv.numel();
    if (g_trace) std::fprintf(stderr, "[trace] alloc+upload %s (%.1f MB)\n",
                              name, n * 2.0 / 1e6);
    static std::vector<bf16> staging;
    if (staging.size() < n) staging.resize(n);
    if (tv.dtype == "BF16") {
        std::memcpy(staging.data(), tv.data, n * sizeof(bf16));
    } else if (tv.dtype == "F32") {
        const float* src = static_cast<const float*>(tv.data);
        for (size_t i = 0; i < n; ++i) staging[i] = float_to_bf16(src[i]);
    } else if (tv.dtype == "F16") {
        const uint16_t* src = static_cast<const uint16_t*>(tv.data);
        for (size_t i = 0; i < n; ++i) {
            uint16_t h = src[i];
            uint32_t sign = (h & 0x8000u) << 16;
            uint32_t exp = (h >> 10) & 0x1fu;
            uint32_t mant = h & 0x03ffu;
            uint32_t out;
            if (exp == 0) {
                if (mant == 0) out = sign;
                else {
                    exp = 1;
                    while ((mant & 0x0400u) == 0) { mant <<= 1; --exp; }
                    mant &= 0x03ffu;
                    out = sign | ((exp + 112u) << 23) | (mant << 13);
                }
            } else if (exp == 31) {
                out = sign | 0x7f800000u | (mant << 13);
            } else {
                out = sign | ((exp + 112u) << 23) | (mant << 13);
            }
            float f;
            std::memcpy(&f, &out, sizeof(f));
            staging[i] = float_to_bf16(f);
        }
    } else {
        throw std::runtime_error(std::string("Expected BF16/F16/F32 for ") + name + ", got " + tv.dtype);
    }
    GpuBuffer<bf16> buf(n, q);
    buf.upload(staging.data(), n);
    if (g_trace) std::fprintf(stderr, "[trace]   done %s\n", name);
    return buf;
}

// (1+w) RMSNorm weight upload: convert to BF16 staging then add 1.0 per
// element before H2D copy. Norm weights are small ([2048] or [256]) so a
// local staging vector is fine (no need for the shared upload() buffer).
GpuBuffer<bf16> upload_plus_one(const TensorView& tv, sycl::queue& q, const char* name) {
    size_t n = tv.numel();
    std::vector<bf16> staging(n);
    if (tv.dtype == "BF16") {
        const bf16* src = static_cast<const bf16*>(tv.data);
        for (size_t i = 0; i < n; ++i)
            staging[i] = float_to_bf16(bf16_to_float(src[i]) + 1.0f);
    } else if (tv.dtype == "F32") {
        const float* src = static_cast<const float*>(tv.data);
        for (size_t i = 0; i < n; ++i)
            staging[i] = float_to_bf16(src[i] + 1.0f);
    } else {
        throw std::runtime_error(std::string("Expected BF16/F32 for +1 norm ") + name + ", got " + tv.dtype);
    }
    GpuBuffer<bf16> buf(n, q);
    buf.upload(staging.data(), n);
    return buf;
}

GpuBuffer<uint8_t> upload_u8(const TensorView& tv, sycl::queue& q, const char* name) {
    if (tv.dtype != "U8" && tv.dtype != "F8_E4M3")
        throw std::runtime_error(std::string("Expected U8/F8_E4M3 for ") + name + ", got " + tv.dtype);
    GpuBuffer<uint8_t> buf(tv.nbytes, q);
    buf.upload(static_cast<const uint8_t*>(tv.data), tv.nbytes);
    return buf;
}

Fp8Linear upload_fp8_linear(const TensorSource& sf, const std::string& prefix,
                            sycl::queue& q) {
    const TensorView& weight = sf.get(prefix + ".weight");
    const TensorView& scale = sf.get(prefix + ".weight_scale");
    if (weight.dtype != "F8_E4M3" || weight.shape.size() != 2)
        throw std::runtime_error("Expected F8_E4M3 [N,K] weight: " + prefix);
    if (scale.dtype != "BF16" || scale.shape.size() != 2 ||
        scale.shape[0] != weight.shape[0] || scale.shape[1] != 1)
        throw std::runtime_error("Expected BF16 [N,1] weight scale: " + prefix);

    Fp8Linear lin;
    lin.out_features = static_cast<int>(weight.shape[0]);
    lin.in_features = static_cast<int>(weight.shape[1]);
    lin.weight = upload_u8(weight, q, (prefix + ".weight").c_str());
    lin.weight_scale = upload(scale, q, (prefix + ".weight_scale").c_str());
    return lin;
}

Fp8Linear upload_fp8_linear_pair(const TensorSource& sf,
                                 const std::string& first_prefix,
                                 const std::string& second_prefix,
                                 sycl::queue& q) {
    return upload_fp8_linear_concat(sf, {first_prefix, second_prefix}, q);
}

Fp8Linear upload_fp8_linear_concat(
    const TensorSource& sf, const std::vector<std::string>& prefixes,
    sycl::queue& q) {
    if (prefixes.empty())
        throw std::runtime_error("FP8 concat requires at least one projection");
    int in = -1;
    int total_out = 0;
    size_t total_weight_bytes = 0;
    std::vector<const TensorView*> weights;
    std::vector<const TensorView*> scales;
    for (const std::string& prefix : prefixes) {
        const TensorView& weight = sf.get(prefix + ".weight");
        const TensorView& scale = sf.get(prefix + ".weight_scale");
        if (weight.dtype != "F8_E4M3" || weight.shape.size() != 2)
            throw std::runtime_error("Expected FP8 [N,K] weight: " + prefix);
        if (scale.dtype != "BF16" || scale.shape.size() != 2 ||
            scale.shape[0] != weight.shape[0] || scale.shape[1] != 1)
            throw std::runtime_error("Expected BF16 [N,1] FP8 scale: " + prefix);
        int current_in = static_cast<int>(weight.shape[1]);
        if (in < 0) in = current_in;
        if (current_in != in)
            throw std::runtime_error("FP8 concat K mismatch: " + prefix);
        total_out += static_cast<int>(weight.shape[0]);
        total_weight_bytes += weight.nbytes;
        weights.push_back(&weight);
        scales.push_back(&scale);
    }
    std::vector<uint8_t> host_weight(total_weight_bytes);
    std::vector<bf16> host_scale(total_out);
    size_t weight_offset = 0;
    size_t scale_offset = 0;
    for (size_t i = 0; i < weights.size(); ++i) {
        std::memcpy(host_weight.data() + weight_offset, weights[i]->data,
                    weights[i]->nbytes);
        size_t outputs = static_cast<size_t>(weights[i]->shape[0]);
        std::memcpy(host_scale.data() + scale_offset, scales[i]->data,
                    outputs * sizeof(bf16));
        weight_offset += weights[i]->nbytes;
        scale_offset += outputs;
    }
    Fp8Linear linear;
    linear.in_features = in;
    linear.out_features = total_out;
    linear.weight = GpuBuffer<uint8_t>(host_weight.size(), q);
    linear.weight.upload(host_weight.data(), host_weight.size());
    linear.weight_scale = GpuBuffer<bf16>(host_scale.size(), q);
    linear.weight_scale.upload(host_scale.data(), host_scale.size());
    return linear;
}

float scalar_f32(const TensorView& tv, const char* name) {
    if (tv.dtype != "F32") throw std::runtime_error(std::string("Expected F32 for ") + name + ", got " + tv.dtype);
    if (tv.nbytes != sizeof(float)) throw std::runtime_error(std::string("Expected scalar F32 for ") + name);
    float out;
    std::memcpy(&out, tv.data, sizeof(float));
    return out;
}

GpuBuffer<uint8_t> upload_nvfp4_scales_transposed(
    const TensorView& tv, int out_features, int groups, sycl::queue& q,
    const char* name) {
    if (tv.dtype != "F8_E4M3")
        throw std::runtime_error(std::string("Expected F8_E4M3 for ") + name + ", got " + tv.dtype);
    if (tv.shape.size() != 2 || tv.shape[0] != out_features || tv.shape[1] != groups)
        throw std::runtime_error(std::string("Unexpected NVFP4 scale shape for ") + name);

    const uint8_t* src = static_cast<const uint8_t*>(tv.data); // model layout: (N, K/16)
    std::vector<uint8_t> transposed((size_t)groups * out_features);
    for (int n = 0; n < out_features; ++n)
        for (int g = 0; g < groups; ++g)
            transposed[(size_t)g * out_features + n] = src[(size_t)n * groups + g];

    GpuBuffer<uint8_t> buf(transposed.size(), q);
    buf.upload(transposed.data(), transposed.size());
    return buf;
}

Nvfp4Linear upload_nvfp4_linear(const TensorSource& sf, const std::string& prefix,
                                 sycl::queue& q) {
    const TensorView& packed = sf.get(prefix + ".weight_packed");
    if (packed.dtype != "U8") throw std::runtime_error("Expected U8 packed weight: " + prefix);
    if (packed.shape.size() != 2) throw std::runtime_error("Expected 2D packed weight: " + prefix);

    Nvfp4Linear lin;
    lin.out_features = (int)packed.shape[0];
    lin.in_features = (int)packed.shape[1] * 2;
    if (lin.in_features % 16 != 0) throw std::runtime_error("NVFP4 K not divisible by 16: " + prefix);
    int groups = lin.in_features / 16;

    lin.weight_packed = upload_u8(packed, q, (prefix + ".weight_packed").c_str());
    lin.weight_scale = upload_nvfp4_scales_transposed(
        sf.get(prefix + ".weight_scale"), lin.out_features, groups, q,
        (prefix + ".weight_scale").c_str());
    lin.input_global_scale = scalar_f32(sf.get(prefix + ".input_global_scale"),
                                        (prefix + ".input_global_scale").c_str());
    lin.weight_global_scale = scalar_f32(sf.get(prefix + ".weight_global_scale"),
                                          (prefix + ".weight_global_scale").c_str());
    float dst_scale = lin.input_global_scale * lin.weight_global_scale;
    lin.dst_scale = GpuBuffer<float>(1, q);
    lin.dst_scale.upload(&dst_scale, 1);
    return lin;
}

Nvfp4Linear upload_nvfp4_linear_pair(const TensorSource& sf,
                                     const std::string& gate_prefix,
                                     const std::string& up_prefix,
                                     sycl::queue& q) {
    return upload_nvfp4_linear_concat(sf, {gate_prefix, up_prefix}, q);
}

Nvfp4Linear upload_nvfp4_linear_concat(
    const TensorSource& sf, const std::vector<std::string>& prefixes,
    sycl::queue& q) {
    if (prefixes.empty())
        throw std::runtime_error("NVFP4 concat requires at least one projection");

    int packed_cols = -1;
    int total_out = 0;
    std::vector<const TensorView*> packed_views;
    std::vector<const TensorView*> scale_views;
    for (const std::string& prefix : prefixes) {
        const TensorView& packed = sf.get(prefix + ".weight_packed");
        if (packed.dtype != "U8" || packed.shape.size() != 2)
            throw std::runtime_error("Expected U8 [N,K/2] packed weight: " + prefix);
        int current_cols = (int)packed.shape[1];
        if (packed_cols < 0) packed_cols = current_cols;
        if (current_cols != packed_cols)
            throw std::runtime_error("NVFP4 concat K mismatch: " + prefix);
        total_out += (int)packed.shape[0];
        packed_views.push_back(&packed);
        scale_views.push_back(&sf.get(prefix + ".weight_scale"));
    }

    Nvfp4Linear lin;
    lin.out_features = total_out;
    lin.in_features = packed_cols * 2;
    if (lin.in_features % 16 != 0)
        throw std::runtime_error("NVFP4 K not divisible by 16: " + prefixes.front());
    int groups = lin.in_features / 16;

    std::vector<uint8_t> packed((size_t)total_out * packed_cols);
    std::vector<uint8_t> transposed((size_t)groups * total_out);
    size_t packed_offset = 0;
    int out_offset = 0;
    for (size_t i = 0; i < prefixes.size(); ++i) {
        int out = (int)packed_views[i]->shape[0];
        std::memcpy(packed.data() + packed_offset, packed_views[i]->data,
                    (size_t)out * packed_cols);
        packed_offset += (size_t)out * packed_cols;

        const TensorView& scale = *scale_views[i];
        if (scale.dtype != "F8_E4M3" || scale.shape.size() != 2 ||
            scale.shape[0] != out || scale.shape[1] != groups)
            throw std::runtime_error("Unexpected NVFP4 scale shape: " + prefixes[i]);
        const uint8_t* src = static_cast<const uint8_t*>(scale.data);
        for (int g = 0; g < groups; ++g)
            for (int n = 0; n < out; ++n)
                transposed[(size_t)g * total_out + out_offset + n] =
                    src[(size_t)n * groups + g];
        out_offset += out;
    }
    lin.weight_packed = GpuBuffer<uint8_t>(packed.size(), q);
    lin.weight_packed.upload(packed.data(), packed.size());
    lin.weight_scale = GpuBuffer<uint8_t>(transposed.size(), q);
    lin.weight_scale.upload(transposed.data(), transposed.size());

    // A fused linear folds one dst_scale for the whole [sum(N_i), K] weight, so
    // the inputs must agree on both globals. Callers that cannot guarantee that
    // should probe with nvfp4_globals_match() and stay unfused.
    lin.input_global_scale = scalar_f32(sf.get(prefixes.front() + ".input_global_scale"),
                                        (prefixes.front() + ".input_global_scale").c_str());
    lin.weight_global_scale = scalar_f32(sf.get(prefixes.front() + ".weight_global_scale"),
                                         (prefixes.front() + ".weight_global_scale").c_str());
    for (size_t i = 1; i < prefixes.size(); ++i) {
        if (scalar_f32(sf.get(prefixes[i] + ".input_global_scale"), "input_global_scale") !=
                lin.input_global_scale ||
            scalar_f32(sf.get(prefixes[i] + ".weight_global_scale"), "weight_global_scale") !=
                lin.weight_global_scale)
            throw std::runtime_error("NVFP4 fused global scales differ: " + prefixes[i]);
    }

    float dst_scale = lin.input_global_scale * lin.weight_global_scale;
    lin.dst_scale = GpuBuffer<float>(1, q);
    lin.dst_scale.upload(&dst_scale, 1);
    return lin;
}

bool nvfp4_globals_match(const TensorSource& sf,
                         const std::vector<std::string>& prefixes) {
    if (prefixes.empty()) return false;
    float input_global = 0.0f;
    float weight_global = 0.0f;
    for (size_t i = 0; i < prefixes.size(); ++i) {
        const std::string& prefix = prefixes[i];
        if (!sf.has(prefix + ".weight_packed") ||
            !sf.has(prefix + ".input_global_scale") ||
            !sf.has(prefix + ".weight_global_scale"))
            return false;
        float current_input = scalar_f32(sf.get(prefix + ".input_global_scale"),
                                         "input_global_scale");
        float current_weight = scalar_f32(sf.get(prefix + ".weight_global_scale"),
                                          "weight_global_scale");
        if (i == 0) {
            input_global = current_input;
            weight_global = current_weight;
        } else if (current_input != input_global || current_weight != weight_global) {
            return false;
        }
    }
    return true;
}

GpuBuffer<bf16> dequantize_nvfp4_to_bf16(const TensorSource& sf,
                                         const std::string& prefix,
                                         sycl::queue& q,
                                         int* out_features_out,
                                         int* in_features_out) {
    const TensorView& packed = sf.get(prefix + ".weight_packed");
    const TensorView& scale = sf.get(prefix + ".weight_scale");
    if (packed.dtype != "U8" || packed.shape.size() != 2)
        throw std::runtime_error("Expected U8 [N,K/2] packed weight: " + prefix);
    int out_features = (int)packed.shape[0];
    int in_features = (int)packed.shape[1] * 2;
    if (in_features % 16 != 0)
        throw std::runtime_error("NVFP4 K not divisible by 16: " + prefix);
    int groups = in_features / 16;
    if (scale.dtype != "F8_E4M3" || scale.shape.size() != 2 ||
        scale.shape[0] != out_features || scale.shape[1] != groups)
        throw std::runtime_error("Unexpected NVFP4 scale shape: " + prefix);

    // The matmul path divides its result by weight_global_scale (oneDNN's DST
    // scale divides), so the reconstructed weight carries that division too.
    float weight_global = scalar_f32(sf.get(prefix + ".weight_global_scale"),
                                     (prefix + ".weight_global_scale").c_str());
    if (weight_global == 0.0f)
        throw std::runtime_error("NVFP4 weight_global_scale is zero: " + prefix);
    float inv_global = 1.0f / weight_global;

    const uint8_t* packed_data = static_cast<const uint8_t*>(packed.data);
    const uint8_t* scale_data = static_cast<const uint8_t*>(scale.data);
    std::vector<bf16> host((size_t)out_features * in_features);
    for (int n = 0; n < out_features; ++n) {
        for (int g = 0; g < groups; ++g) {
            float group_scale =
                nvfp4_e4m3_fast(scale_data[(size_t)n * groups + g]) * inv_global;
            for (int e = 0; e < 16; e += 2) {
                int k = g * 16 + e;
                uint8_t byte = packed_data[((size_t)n * in_features + k) / 2];
                host[(size_t)n * in_features + k] =
                    float_to_bf16(nvfp4_e2m1_fast(byte & 0x0f) * group_scale);
                host[(size_t)n * in_features + k + 1] =
                    float_to_bf16(nvfp4_e2m1_fast(byte >> 4) * group_scale);
            }
        }
    }
    GpuBuffer<bf16> buffer(host.size(), q);
    buffer.upload(host.data(), host.size());
    if (out_features_out) *out_features_out = out_features;
    if (in_features_out) *in_features_out = in_features;
    return buffer;
}
