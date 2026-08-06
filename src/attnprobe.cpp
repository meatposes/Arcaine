// Standalone driver for qwen35_xmx_attention.
//
// The MTP head made the engine issue attention with a small batch against a
// large history for the first time, and that shape takes the device down with
// UR_RESULT_ERROR_DEVICE_LOST. This calls the kernel directly with random data
// so the shape can be swept in seconds instead of behind a 22 GB model load,
// and so a hang kills one small process rather than a benchmark run.
//
// Each shape is run in its own invocation by the caller; the process prints
// the shape before launching so a hang is attributable from the log alone.
//
//   attnprobe <seq> <past> [query_heads] [key_heads]

#include <cstdio>
#include <cstdlib>
#include <vector>

#include <sycl/sycl.hpp>

#include "common/gpu/buffer.hpp"
#include "common/gpu/engine.hpp"
#include "modeling/qwen3_5/kernels.hpp"

int main(int argc, char** argv) {
    int seq = argc > 1 ? std::atoi(argv[1]) : 64;
    int past = argc > 2 ? std::atoi(argv[2]) : 128;
    int query_heads = argc > 3 ? std::atoi(argv[3]) : 24;
    int key_heads = argc > 4 ? std::atoi(argv[4]) : 4;
    const int head_dim = 256;

    auto& engine = GpuEngine::get(0);
    auto& queue = engine.queue;
    std::printf("seq=%d past=%d qh=%d kh=%d ... ", seq, past, query_heads,
                key_heads);
    std::fflush(stdout);

    int capacity = past + seq + 16;
    GpuBuffer<bf16> query((size_t)seq * query_heads * head_dim, queue);
    GpuBuffer<bf16> key((size_t)capacity * key_heads * head_dim, queue);
    GpuBuffer<bf16> value((size_t)capacity * key_heads * head_dim, queue);
    GpuBuffer<bf16> output((size_t)seq * query_heads * head_dim, queue);

    auto fill = [&](GpuBuffer<bf16>& buffer, uint32_t salt) {
        size_t n = buffer.count();
        bf16* data = buffer.data();
        queue.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
                uint32_t x = (uint32_t)id[0] * 2654435761u + salt;
                x ^= x >> 15;
                data[id[0]] = float_to_bf16(((float)(x & 0xFFFF) / 32768.0f) - 1.0f);
            });
        }).wait();
    };
    fill(query, 1);
    fill(key, 2);
    fill(value, 3);
    output.zero();

    qwen35_xmx_attention(queue, query.data(), key.data(), value.data(),
                         output.data(), seq, past, query_heads, key_heads,
                         head_dim, 1.0f / 16.0f);
    queue.wait();

    // Read one row back so a silently-skipped launch is visible as zeros.
    std::vector<bf16> host(head_dim);
    queue.memcpy(host.data(), output.data(), head_dim * sizeof(bf16)).wait();
    double sum = 0.0;
    for (bf16 v : host) sum += bf16_to_float(v);
    std::printf("ok (row0 sum %.4f)\n", sum);
    return 0;
}
