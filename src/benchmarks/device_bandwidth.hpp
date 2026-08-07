#pragma once

// Measured device memory bandwidth, for use as a roofline denominator.
//
// Two details decide whether the number means anything, and both were wrong in
// the first version of this probe:
//
//   The buffer must hold incompressible data. This GPU compresses memory
//   losslessly, so a memset or uniform buffer reports ~2400 GB/s where the same
//   probe over random data reports ~590 GB/s on the same card. The first figure
//   measures the compressor. Model weights are incompressible, so noise is the
//   fill that matches the workload being measured.
//
//   Every lane must store its own partial sum. A guard of the form
//   `if ((id & mask) == 0 && sum == magic)` short-circuits on the id term, which
//   makes the accumulation dead code for every lane that fails the mask and
//   lets the compiler delete the loads the probe exists to time.
//
// Buffers below ~1 GiB also understate: 64 MiB reads about 20% low against the
// figure 1-4 GiB converge on.
//
// A datasheet constant is a poor substitute. The default that used to sit in
// nvfp4_roofline_bench understated this device by 29%, which turns a kernel at
// 62% of peak into one that reports 80% and looks finished.

#include <chrono>
#include <cstddef>
#include <cstdint>

#include <sycl/sycl.hpp>

namespace arcaine::bench {

struct DeviceBandwidth {
    double read_gbs = 0.0;   // streaming read, the decode-relevant figure
    double copy_gbs = 0.0;   // read + write
    bool   valid = false;
};

// `bytes` is the probe buffer size; two of them are allocated. Returns
// valid=false when the device cannot spare the allocation, so a caller can fall
// back rather than report a zero.
inline DeviceBandwidth measure_device_bandwidth(sycl::queue& queue,
                                                size_t bytes = (size_t)1 << 30,
                                                int reps = 20);


namespace {
using Clk = std::chrono::high_resolution_clock;
inline double now_ms() {
    return std::chrono::duration<double, std::milli>(
               Clk::now().time_since_epoch()).count();
}
}  // namespace

inline DeviceBandwidth measure_device_bandwidth(sycl::queue& queue,
                                               size_t bytes, int reps) {
    using v4 = sycl::vec<float, 4>;
    const size_t n = bytes / sizeof(v4);
    const size_t threads = 1u << 20;
    if (n == 0 || reps <= 0) return {};

    v4* src = sycl::malloc_device<v4>(n, queue);
    v4* dst = sycl::malloc_device<v4>(n, queue);
    float* partial = sycl::malloc_device<float>(threads, queue);
    if (!src || !dst || !partial) {
        sycl::free(src, queue);
        sycl::free(dst, queue);
        sycl::free(partial, queue);
        return {};
    }

    // Incompressible fill. See the header for why this is not a memset.
    queue.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
            uint64_t x = (uint64_t)id[0] * 6364136223846793005ull +
                         1442695040888963407ull;
            x ^= x >> 33; x *= 0xff51afd7ed558ccdull; x ^= x >> 33;
            v4 v;
            for (int k = 0; k < 4; ++k)
                v[k] = (float)(int32_t)(uint32_t)(x >> (k * 8));
            src[id[0]] = v;
        });
    }).wait();

    auto read_once = [&] {
        queue.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(threads), [=](sycl::id<1> id) {
                size_t i = id[0];
                v4 acc(0.f);
                for (size_t j = i; j < n; j += threads) acc += src[j];
                partial[i] = acc[0] + acc[1] + acc[2] + acc[3];
            });
        });
    };
    auto copy_once = [&] { queue.memcpy(dst, src, n * sizeof(v4)); };

    auto time = [&](auto&& fn, double moved) {
        fn();
        queue.wait();                       // warm
        double t0 = now_ms();
        for (int r = 0; r < reps; ++r) fn();
        queue.wait();
        double ms = now_ms() - t0;
        return ms > 0.0 ? moved * reps / (ms * 1e6) : 0.0;   // GB/s
    };

    DeviceBandwidth bw;
    bw.read_gbs = time(read_once, (double)n * sizeof(v4));
    bw.copy_gbs = time(copy_once, (double)n * sizeof(v4) * 2.0);
    bw.valid = bw.read_gbs > 0.0;

    sycl::free(src, queue);
    sycl::free(dst, queue);
    sycl::free(partial, queue);
    return bw;
}


}  // namespace arcaine::bench
