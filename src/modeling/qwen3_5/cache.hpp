#pragma once

#include <stdexcept>
#include <vector>

#include "config.hpp"
#include "../../common/gpu/buffer.hpp"
#include "../../common/gpu/engine.hpp"

struct Qwen35KvLayerCache {
    GpuBuffer<bf16> key;
    GpuBuffer<bf16> value;
    int filled = 0;
    int capacity = 0;
};

struct Qwen35DeltaLayerCache {
    GpuBuffer<bf16> conv_state;
    GpuBuffer<float> recurrent_state;
    bool has_state = false;
};

struct Qwen35Caches {
    std::vector<Qwen35KvLayerCache> kv;
    std::vector<Qwen35DeltaLayerCache> delta;

    void init(const Qwen35Config& config, int max_seq_len, int split_layer) {
        const auto& c = config.text;
        kv.resize(c.num_hidden_layers);
        delta.resize(c.num_hidden_layers);
        int key_dim = c.linear_num_key_heads * c.linear_key_head_dim;
        int value_dim = c.linear_num_value_heads * c.linear_value_head_dim;
        int conv_dim = 2 * key_dim + value_dim;
        for (int layer = 0; layer < c.num_hidden_layers; ++layer) {
            int gpu = (layer < split_layer || GpuEngine::count() < 2) ? 0 : 1;
            auto& queue = GpuEngine::get(gpu).queue;
            if (c.is_full_attn(layer)) {
                size_t count = (size_t)max_seq_len * c.num_key_value_heads * c.head_dim;
                kv[layer].key = GpuBuffer<bf16>(count, queue);
                kv[layer].value = GpuBuffer<bf16>(count, queue);
                kv[layer].capacity = max_seq_len;
            } else {
                delta[layer].conv_state = GpuBuffer<bf16>(
                    (size_t)conv_dim * (c.linear_conv_kernel_dim - 1), queue);
                delta[layer].recurrent_state = GpuBuffer<float>(
                    (size_t)c.linear_num_value_heads * c.linear_key_head_dim *
                    c.linear_value_head_dim, queue);
                delta[layer].conv_state.zero();
                delta[layer].recurrent_state.zero();
            }
        }
    }

    void reset() {
        for (auto& layer : kv) layer.filled = 0;
        for (auto& layer : delta) {
            if (!layer.conv_state.empty()) {
                layer.conv_state.zero();
                layer.recurrent_state.zero();
            }
            layer.has_state = false;
        }
        snapshot_valid = false;
    }

    // ---- speculative rollback -------------------------------------------
    //
    // Rejecting a drafted token means undoing its effect on the caches. For
    // attention that is just a counter: the KV rows past `filled` are
    // overwritten before they are read again. The DeltaNet layers are the hard
    // case, because the recurrent and convolution states are updated in place
    // and carry no position, so a rejected token cannot be subtracted back out
    // and the previous values have to have been kept.
    //
    // The shadow buffers are allocated once. A save is ~151 MB of
    // device-to-device copy for a 64-layer 27B, about 0.26 ms against a 79 ms
    // decode step.
    std::vector<GpuBuffer<bf16>> shadow_conv;
    std::vector<GpuBuffer<float>> shadow_recurrent;
    std::vector<sycl::queue*> shadow_queue;
    std::vector<int> shadow_kv_filled;
    std::vector<char> shadow_has_state;
    bool snapshot_valid = false;

    void init_snapshot(const Qwen35Config& config, int split_layer) {
        shadow_conv.resize(delta.size());
        shadow_recurrent.resize(delta.size());
        shadow_queue.assign(delta.size(), nullptr);
        shadow_kv_filled.assign(kv.size(), 0);
        shadow_has_state.assign(delta.size(), 0);
        (void)config;
        for (size_t layer = 0; layer < delta.size(); ++layer) {
            if (delta[layer].conv_state.empty()) continue;
            // Each shadow lives on the same device as the state it mirrors, so
            // the copy stays device-local rather than crossing the split. The
            // placement must match Qwen35Caches::init exactly.
            int gpu = ((int)layer < split_layer || GpuEngine::count() < 2) ? 0 : 1;
            sycl::queue& queue = GpuEngine::get(gpu).queue;
            shadow_queue[layer] = &queue;
            shadow_conv[layer] =
                GpuBuffer<bf16>(delta[layer].conv_state.count(), queue);
            shadow_recurrent[layer] =
                GpuBuffer<float>(delta[layer].recurrent_state.count(), queue);
        }
    }

    bool snapshot_ready() const { return !shadow_conv.empty(); }

    void save() {
        for (size_t layer = 0; layer < kv.size(); ++layer)
            shadow_kv_filled[layer] = kv[layer].filled;
        for (size_t layer = 0; layer < delta.size(); ++layer) {
            if (delta[layer].conv_state.empty()) continue;
            shadow_has_state[layer] = delta[layer].has_state ? 1 : 0;
            sycl::queue& queue = *shadow_queue[layer];
            queue.memcpy(shadow_conv[layer].data(), delta[layer].conv_state.data(),
                         delta[layer].conv_state.count() * sizeof(bf16));
            queue.memcpy(shadow_recurrent[layer].data(),
                         delta[layer].recurrent_state.data(),
                         delta[layer].recurrent_state.count() * sizeof(float));
        }
        for (int gpu = 0; gpu < GpuEngine::count(); ++gpu)
            GpuEngine::get(gpu).queue.wait();
        snapshot_valid = true;
    }

    void restore() {
        if (!snapshot_valid)
            throw std::runtime_error("Qwen3.5 cache restore without a snapshot");
        for (size_t layer = 0; layer < kv.size(); ++layer)
            kv[layer].filled = shadow_kv_filled[layer];
        for (size_t layer = 0; layer < delta.size(); ++layer) {
            if (delta[layer].conv_state.empty()) continue;
            delta[layer].has_state = shadow_has_state[layer] != 0;
            sycl::queue& queue = *shadow_queue[layer];
            queue.memcpy(delta[layer].conv_state.data(), shadow_conv[layer].data(),
                         shadow_conv[layer].count() * sizeof(bf16));
            queue.memcpy(delta[layer].recurrent_state.data(),
                         shadow_recurrent[layer].data(),
                         shadow_recurrent[layer].count() * sizeof(float));
        }
        for (int gpu = 0; gpu < GpuEngine::count(); ++gpu)
            GpuEngine::get(gpu).queue.wait();
    }
};
