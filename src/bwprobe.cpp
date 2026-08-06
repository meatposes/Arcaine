// Standalone bandwidth probe. Exists to validate the one inside arcaine_mbench
// against something that cannot silently skip the work: the reduction result
// is copied back and checked, and every timing is taken twice, once from the
// host clock and once from SYCL event profiling.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <sycl/sycl.hpp>

using Clk = std::chrono::high_resolution_clock;

int main(int argc, char** argv) {
    int reps = 20;
    if (argc > 1) reps = std::atoi(argv[1]);

    sycl::queue q{sycl::gpu_selector_v,
                  sycl::property_list{sycl::property::queue::in_order{},
                                      sycl::property::queue::enable_profiling{}}};
    sycl::device dev = q.get_device();
    std::printf("device: %s, %zu GiB\n",
                dev.get_info<sycl::info::device::name>().c_str(),
                dev.get_info<sycl::info::device::global_mem_size>() >> 30);

    using v4 = sycl::vec<float, 4>;
    for (size_t mib : {64u, 256u, 1024u, 4096u}) {
        const size_t bytes = mib << 20;
        const size_t n = bytes / sizeof(v4);
        v4* src = sycl::malloc_device<v4>(n, q);
        v4* dst = sycl::malloc_device<v4>(n, q);
        float* partial = nullptr;
        const size_t threads = 1u << 20;
        partial = sycl::malloc_device<float>(threads, q);
        if (!src || !dst || !partial) {
            std::printf("%5zu MiB: allocation failed\n", mib);
            sycl::free(src, q); sycl::free(dst, q); sycl::free(partial, q);
            continue;
        }

        // Two fills. A uniform buffer is perfectly compressible, and this GPU
        // compresses memory losslessly, so timing it measures the compressor
        // rather than the bus. Model weights are incompressible, so the random
        // fill is the one whose number belongs in a roofline.
        auto fill_uniform = [&] { q.fill(src, v4(1.0f), n).wait(); };
        auto fill_random = [&] {
            q.submit([&](sycl::handler& h) {
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
        fill_uniform();

        // Every lane stores its own partial, so no branch can make the loop
        // dead and the host can verify the whole buffer was actually read.
        auto read_once = [&] {
            return q.submit([&](sycl::handler& h) {
                h.parallel_for(sycl::range<1>(threads), [=](sycl::id<1> id) {
                    size_t i = id[0];
                    v4 acc(0.f);
                    for (size_t j = i; j < n; j += threads) acc += src[j];
                    partial[i] = acc[0] + acc[1] + acc[2] + acc[3];
                });
            });
        };

        read_once().wait();
        std::vector<float> host(threads);
        q.memcpy(host.data(), partial, threads * sizeof(float)).wait();
        double sum = 0.0;
        for (float v : host) sum += (double)v;
        double expect = 4.0 * (double)n;

        const double gb = (double)bytes * reps / 1e9;

        auto timed_read = [&] {
            auto t0 = Clk::now();
            uint64_t ns_total = 0;
            for (int r = 0; r < reps; ++r) {
                sycl::event e = read_once();
                e.wait();
                ns_total +=
                    e.get_profiling_info<sycl::info::event_profiling::command_end>() -
                    e.get_profiling_info<sycl::info::event_profiling::command_start>();
            }
            q.wait();
            double wall_ms =
                std::chrono::duration<double, std::milli>(Clk::now() - t0).count();
            return std::pair<double, double>{gb / (wall_ms * 1e-3),
                                             gb / (ns_total * 1e-9)};
        };
        auto timed_copy = [&] {
            auto t1 = Clk::now();
            for (int r = 0; r < reps; ++r) q.memcpy(dst, src, bytes);
            q.wait();
            double ms =
                std::chrono::duration<double, std::milli>(Clk::now() - t1).count();
            return 2.0 * gb / (ms * 1e-3);
        };

        auto [u_wall, u_event] = timed_read();
        double u_copy = timed_copy();
        std::printf("%5zu MiB  uniform: read %7.1f GB/s (event %7.1f) | "
                    "copy %7.1f | checksum %.0f/%.0f %s\n",
                    mib, u_wall, u_event, u_copy, sum, expect,
                    sum == expect ? "ok" : "MISMATCH");

        fill_random();
        read_once().wait();
        auto [r_wall, r_event] = timed_read();
        double r_copy = timed_copy();
        std::printf("%5zu MiB  random : read %7.1f GB/s (event %7.1f) | "
                    "copy %7.1f\n",
                    mib, r_wall, r_event, r_copy);

        sycl::free(src, q); sycl::free(dst, q); sycl::free(partial, q);
    }
    return 0;
}
