// qwen3_5 — XMX attention across the (seq, past) shape space, no model load.
//
// Exists because a device hang was once blamed on this kernel. Driving it
// directly with random data showed every accused shape passing, which moved the
// search to the caller and found the real cause: a host buffer freed while an
// async memcpy over it was still pending. A kernel is easier to exonerate than
// to convict, and doing it behind a 22 GB model load is too slow to iterate on.
//
// Each shape prints before it launches, so a hang is attributable from the log
// alone, and `--sweep` covers the shapes the engine actually produces: decode,
// prefill, chunked prefill, and a small batch against a large history — the
// last of which nothing generated until the MTP head began advancing its own
// cache a window at a time.
//
//   ./build/arcaine_kbench qwen35-attention-shape [--seq N] [--past N] [--sweep]

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>

#include "benchmarks/registry.hpp"
#include "runtime/gpu/buffer.hpp"
#include "runtime/gpu/device_select.hpp"
#include "runtime/gpu/engine.hpp"
#include "modeling/qwen3_5/kernels.hpp"

using namespace qwen35_kernels;

namespace {

int probe(int seq, int past, int query_heads, int key_heads) {
    const int head_dim = 256;
    auto& engine = GpuEngine::get(0);
    auto& queue = engine.queue;
    std::printf("seq=%-5d past=%-5d qh=%d kh=%d ... ", seq, past, query_heads,
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

    // Read one row back so a silently-skipped launch shows up as zeros rather
    // than as a pass.
    std::vector<bf16> host(head_dim);
    queue.memcpy(host.data(), output.data(), head_dim * sizeof(bf16)).wait();
    double sum = 0.0;
    for (bf16 v : host) sum += bf16_to_float(v);
    std::printf("ok (row0 sum %.4f)\n", sum);
    return 0;
}

int run(int argc, char** argv) {
    int seq = 64, past = 128, qh = 24, kh = 4;
    bool sweep = false;
    std::string device;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
            return argv[++i];
        };
        if      (a == "--seq")    seq  = std::stoi(next());
        else if (a == "--past")   past = std::stoi(next());
        else if (a == "--qh")     qh   = std::stoi(next());
        else if (a == "--kh")     kh   = std::stoi(next());
        else if (a == "--device") device = next();
        else if (a == "--sweep")  sweep = true;
        else if (a == "-h" || a == "--help") {
            std::fputs("Usage: arcaine_kbench qwen35-attention-shape "
                       "[--seq N] [--past N] [--qh N] [--kh N] [--sweep] "
                       "[--device N]\n", stderr);
            return 0;
        } else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 1; }
    }
    if (!device.empty()) gpu_device_control::apply_device_index(device);
    if (!sweep) return probe(seq, past, qh, kh);

    const int shapes[][2] = {{1, 0}, {1, 128}, {1, 2048}, {8, 0}, {64, 0},
                             {512, 0}, {512, 512}, {2, 128}, {64, 64},
                             {64, 128}, {63, 448}, {64, 1984}};
    int failures = 0;
    for (const auto& s : shapes)
        if (probe(s[0], s[1], qh, kh) != 0) ++failures;
    return failures ? 1 : 0;
}

}  // namespace

REGISTER_BENCH("qwen35-attention-shape",
    "Qwen3.5 XMX attention across (seq, past) shapes; no model load",
    run)
