// Device memory bandwidth, and the validation for the roofline denominator
// every other bench divides by.
//
// Reports the compressible and incompressible fills side by side because the
// gap between them is the whole point: this GPU compresses memory losslessly,
// so a uniform buffer measures the compressor and not the bus. Weights are
// incompressible, so the random figure is the one a roofline should use.
//
// Also sweeps buffer size (small buffers understate) and cross-checks the host
// clock against SYCL event profiling, so a disagreement between the two shows
// up here rather than as a quietly wrong percentage somewhere else.
//
//   ./build/arcaine_kbench device-bandwidth [--reps N] [--device N]

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>

#include "benchmarks/registry.hpp"
#include "runtime/gpu/device_select.hpp"
#include "runtime/gpu/engine.hpp"

namespace {

using Clk = std::chrono::high_resolution_clock;

int run(int argc, char** argv) {
    int reps = 20;
    std::string device;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
            return argv[++i];
        };
        if      (a == "--reps")   reps = std::stoi(next());
        else if (a == "--device") device = next();
        else if (a == "-h" || a == "--help") {
            std::fputs("Usage: arcaine_kbench device-bandwidth "
                       "[--reps N] [--device N]\n", stderr);
            return 0;
        } else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 1; }
    }
    if (!device.empty()) gpu_device_control::apply_device_index(device);

    sycl::queue queue{GpuEngine::get(0).queue.get_context(),
                      GpuEngine::get(0).queue.get_device(),
                      sycl::property_list{sycl::property::queue::in_order{},
                                          sycl::property::queue::enable_profiling{}}};
    sycl::device dev = queue.get_device();
    std::printf("device: %s, %zu GiB\n",
                dev.get_info<sycl::info::device::name>().c_str(),
                dev.get_info<sycl::info::device::global_mem_size>() >> 30);

    using v4 = sycl::vec<float, 4>;
    for (size_t mib : {64u, 256u, 1024u, 4096u}) {
        const size_t bytes = mib << 20;
        const size_t n = bytes / sizeof(v4);
        const size_t threads = 1u << 20;
        v4* src = sycl::malloc_device<v4>(n, queue);
        v4* dst = sycl::malloc_device<v4>(n, queue);
        float* partial = sycl::malloc_device<float>(threads, queue);
        if (!src || !dst || !partial) {
            std::printf("%5zu MiB: allocation failed\n", mib);
            sycl::free(src, queue); sycl::free(dst, queue); sycl::free(partial, queue);
            continue;
        }

        auto fill_uniform = [&] { queue.fill(src, v4(1.0f), n).wait(); };
        auto fill_random = [&] {
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
        };

        // Every lane stores its own partial, so no branch can make the loop
        // dead and the host can verify the whole buffer was read.
        auto read_once = [&] {
            return queue.submit([&](sycl::handler& h) {
                h.parallel_for(sycl::range<1>(threads), [=](sycl::id<1> id) {
                    size_t i = id[0];
                    v4 acc(0.f);
                    for (size_t j = i; j < n; j += threads) acc += src[j];
                    partial[i] = acc[0] + acc[1] + acc[2] + acc[3];
                });
            });
        };

        const double gb = (double)bytes * reps / 1e9;
        auto timed_read = [&] {
            auto t0 = Clk::now();
            uint64_t ns = 0;
            for (int r = 0; r < reps; ++r) {
                sycl::event e = read_once();
                e.wait();
                ns += e.get_profiling_info<sycl::info::event_profiling::command_end>() -
                      e.get_profiling_info<sycl::info::event_profiling::command_start>();
            }
            queue.wait();
            double wall_ms =
                std::chrono::duration<double, std::milli>(Clk::now() - t0).count();
            return std::pair<double, double>{gb / (wall_ms * 1e-3), gb / (ns * 1e-9)};
        };
        auto timed_copy = [&] {
            auto t0 = Clk::now();
            for (int r = 0; r < reps; ++r) queue.memcpy(dst, src, bytes);
            queue.wait();
            double ms = std::chrono::duration<double, std::milli>(Clk::now() - t0).count();
            return 2.0 * gb / (ms * 1e-3);
        };

        fill_uniform();
        read_once().wait();
        std::vector<float> host(threads);
        queue.memcpy(host.data(), partial, threads * sizeof(float)).wait();
        double sum = 0.0;
        for (float v : host) sum += (double)v;
        double expect = 4.0 * (double)n;
        auto [u_wall, u_event] = timed_read();
        double u_copy = timed_copy();
        std::printf("%5zu MiB  uniform: read %7.1f GB/s (event %7.1f) | copy %7.1f"
                    " | checksum %s\n",
                    mib, u_wall, u_event, u_copy, sum == expect ? "ok" : "MISMATCH");

        fill_random();
        read_once().wait();
        auto [r_wall, r_event] = timed_read();
        double r_copy = timed_copy();
        std::printf("%5zu MiB  random : read %7.1f GB/s (event %7.1f) | copy %7.1f\n",
                    mib, r_wall, r_event, r_copy);

        sycl::free(src, queue); sycl::free(dst, queue); sycl::free(partial, queue);
    }
    return 0;
}

}  // namespace

REGISTER_BENCH("device-bandwidth",
    "Measured memory bandwidth; contrasts compressible and incompressible fills",
    run)
