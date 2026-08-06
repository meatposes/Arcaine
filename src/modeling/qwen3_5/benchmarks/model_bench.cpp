// qwen3_5 — model-owned inference benchmark (llama-bench style).
// Drives the module's concrete Qwen35Model engine directly (forward/reset_cache)
// to measure real GPU prefill (PP) and decode (TG) throughput at various KV-cache
// depths. Registered with the central arcaine_mbench dispatcher as "qwen3_5"
// (matches config.json::model_type).
//
//   ./build/arcaine_mbench --model <dir> [options]
//     -p, --p P,...   prefill prompt sizes (default: 512,1024,2048,4096)
//     -n, --n N,...   new-token counts      (default: 128)
//     -d D,...        starting KV depths     (default: 0,512,1024,2048)
//     -r, --r R       timed repetitions      (default: 3)
//     -w, --w W       warmup runs            (default: 1)
//     --max-seq N     KV cache capacity      (default: auto)
//     --device N      restrict to one Level Zero GPU

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <memory>

#include "modeling/qwen3_5/model.hpp"
#include "runtime/gpu/device_select.hpp"
#include "runtime/gpu/engine.hpp"
#include "benchmarks/model_bench_registry.hpp"
#include "benchmarks/model_bench_util.hpp"

using Clk = std::chrono::high_resolution_clock;
using Ms  = std::chrono::duration<double, std::milli>;
static double now_ms() { return Ms(Clk::now().time_since_epoch()).count(); }

static const char* USAGE =
    "Usage: arcaine_mbench --model <dir> [options]   (qwen3_5)\n"
    "  -h, --help    show this help text\n"
    "  -p, --p P,... prefill sizes          (default: 512,1024,2048,4096)\n"
    "  -n, --n N,... new-token counts       (default: 128)\n"
    "  -d D,...      KV-cache depths        (default: 0,512,1024,2048)\n"
    "  -r, --r R     timed repetitions      (default: 3)\n"
    "  -w, --w W     warmup runs            (default: 1)\n"
    "  --max-seq N   KvCache capacity       (default: auto)\n"
    "  --device N    run with one visible Level Zero GPU\n"
    "  --spec        measure the MTP head instead: acceptance, draft cost, and\n"
    "                a greedy baseline-vs-speculative A/B\n"
    "  --spec-tokens N  tokens to generate in --spec mode (default: 128)\n"
    "  --spec-prompt T  prompt for --spec mode\n"
    "  --golden SUB  numerical gate (capture|compare); see golden_bench.cpp.\n"
    "                Remaining flags are forwarded to it.\n";

// Numerical gate, implemented in golden_bench.cpp so this file stays a
// throughput benchmark.
namespace qwen35_golden { int run_golden(int argc, char** argv); }

// ---------------------------------------------------------------------------
// MTP measurement.
//
// Whether the head is worth anything comes down to two numbers. Acceptance is
// how often its draft matches what the backbone actually produces, and it
// doubles as the correctness check: the head is wired through a concatenation
// whose order, and whose choice of pre- or post-final-norm hidden state, cannot
// be inferred from the tensor shapes, and getting either wrong leaves the model
// running with acceptance near zero. Draft cost is one decoder layer against
// sixty four, and it is exactly the break-even acceptance rate.
//
// The end-to-end arm runs the same prompt greedily twice. Both arms must emit
// the identical sequence, because speculative decoding verifies every token it
// emits; anything else means state is lost on rollback or a batched forward
// disagrees with a sequential one. A control replays the baseline's own tokens
// two at a time with no drafting and no rollback at all, which tells those two
// causes apart.
static int argmax_of(const std::vector<float>& x, int row, int vocab) {
    const float* r = x.data() + (size_t)row * vocab;
    int best = 0;
    for (int i = 1; i < vocab; ++i) if (r[i] > r[best]) best = i;
    return best;
}

static double median_of(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t mid = v.size() / 2;
    return v.size() % 2 ? v[mid] : 0.5 * (v[mid - 1] + v[mid]);
}

static int run_spec(Qwen35Model& model, const std::string& prompt, int tokens) {
    const ModelInfo& info = model.info();
    const int vocab = info.vocab_size;
    if (!model.has_mtp()) {
        std::fputs("[spec] this checkpoint reports no MTP head\n", stderr);
        return 1;
    }
    PreparedInput prepared = model.prepare_input(prompt, {}, {}, "");
    const std::vector<int>& prompt_tokens = prepared.tokens;
    std::printf("[spec] prompt %d tokens, generating %d\n",
                (int)prompt_tokens.size(), tokens);

    // Acceptance and cost, scored against a plain greedy decode.
    model.reset_cache();
    std::vector<float> logits = model.forward(ForwardInput{
        prompt_tokens, 0, nullptr, nullptr, &prepared.mm_token_type_ids});
    int past = (int)prompt_tokens.size();
    int accepted = 0, scored = 0, pending_draft = -1;
    std::vector<double> backbone_ms, draft_ms;
    std::vector<int> baseline;

    for (int step = 0; step < tokens; ++step) {
        int next = argmax_of(logits, 0, vocab);
        if (info.is_eos(next)) break;
        if (pending_draft >= 0) { ++scored; if (pending_draft == next) ++accepted; }
        baseline.push_back(next);

        double t0 = now_ms();
        std::vector<float> draft = model.mtp_draft(next, past);
        draft_ms.push_back(now_ms() - t0);
        pending_draft = draft.empty() ? -1 : argmax_of(draft, 0, vocab);

        std::vector<int> one{next};
        double t1 = now_ms();
        logits = model.forward(ForwardInput{one, past});
        backbone_ms.push_back(now_ms() - t1);
        ++past;
    }

    double bb = median_of(backbone_ms), df = median_of(draft_ms);
    double rate = scored ? (double)accepted / scored : 0.0;
    double cost = bb > 0.0 ? df / bb : 0.0;
    std::printf("\n  backbone step        %.2f ms (median of %zu)\n",
                bb, backbone_ms.size());
    std::printf("  mtp draft            %.2f ms  (%.3f of a step)\n", df, cost);
    std::printf("  acceptance           %.4f  (%d/%d)\n", rate, accepted, scored);
    std::printf("  break-even           %.4f\n", cost);
    if (rate < 0.05)
        std::printf("  acceptance this low means the head is mis-wired, not weak\n");

    // Control: two-token forwards over the baseline's own tokens, no drafting
    // and no rollback.
    model.reset_cache();
    model.forward(ForwardInput{prompt_tokens, 0, nullptr, nullptr,
                               &prepared.mm_token_type_ids});
    int control_past = (int)prompt_tokens.size();
    size_t control_ok = 0;
    for (size_t i = 0; i + 1 < baseline.size(); i += 2) {
        std::vector<float> pair =
            model.forward_verify({baseline[i], baseline[i + 1]}, control_past);
        if (argmax_of(pair, 0, vocab) != baseline[i + 1]) break;
        ++control_ok;
        if (i + 2 < baseline.size() && argmax_of(pair, 1, vocab) != baseline[i + 2])
            break;
        ++control_ok;
        control_past += 2;
    }

    // End to end.
    model.reset_cache();
    double t_base = now_ms();
    std::vector<float> bl = model.forward(ForwardInput{
        prompt_tokens, 0, nullptr, nullptr, &prepared.mm_token_type_ids});
    std::vector<int> plain;
    int plain_past = (int)prompt_tokens.size();
    while ((int)plain.size() < tokens) {
        int next = argmax_of(bl, 0, vocab);
        if (info.is_eos(next)) break;
        plain.push_back(next);
        std::vector<int> one{next};
        bl = model.forward(ForwardInput{one, plain_past});
        ++plain_past;
    }
    double base_ms = now_ms() - t_base;

    model.reset_cache();
    Qwen35Model::SpecStats stats;
    double t_spec = now_ms();
    std::vector<int> spec = model.generate_speculative(prompt_tokens, tokens, stats);
    double spec_ms = now_ms() - t_spec;

    size_t common = 0;
    while (common < plain.size() && common < spec.size() &&
           plain[common] == spec[common]) ++common;

    std::printf("\n  --- end to end, greedy, %d tokens ---\n", tokens);
    std::printf("  baseline             %.1f ms   %.2f tok/s\n",
                base_ms, plain.size() / (base_ms * 1e-3));
    std::printf("  speculative          %.1f ms   %.2f tok/s\n",
                spec_ms, spec.size() / (spec_ms * 1e-3));
    std::printf("  speedup              %.3fx\n",
                (spec.size() / (spec_ms * 1e-3)) / (plain.size() / (base_ms * 1e-3)));
    std::printf("  rounds               %d  (%d accepted)\n",
                stats.rounds, stats.accepts);
    std::printf("  backbone passes      %d for %zu tokens  (%.3f tok/pass)\n",
                stats.forwards, spec.size(),
                stats.forwards ? (double)spec.size() / stats.forwards : 0.0);
    std::printf("  draft/verify/rollback  %.1f / %.1f / %.1f ms\n",
                stats.draft_ms, stats.verify_ms, stats.rollback_ms);
    std::printf("  batched control      %zu/%zu tokens reproduced\n",
                control_ok, plain.size());
    if (common == plain.size() && common == spec.size()) {
        std::printf("  sequences            identical (%zu tokens)\n", common);
        return 0;
    }
    if (control_ok <= common)
        std::printf("  diverges at %zu; the control diverges at %zu, so this is "
                    "M=1 vs M=2 kernel numerics, not rollback\n", common, control_ok);
    else
        std::printf("  MISMATCH at %zu while the control reproduced %zu — "
                    "rollback is losing state\n", common, control_ok);
    return 1;
}

static int run(int argc, char* argv[]) {
    // --golden owns its own flag set, so it is split off before this file's
    // parser runs. argv is rebuilt as {"golden", <sub>, ...rest}, the shape
    // run_golden expects.
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--golden") != 0) continue;
        if (i + 1 >= argc) {
            std::fputs("--golden needs a subcommand (capture|compare)\n", stderr);
            return 1;
        }
        std::vector<char*> forwarded{argv[0], argv[i + 1]};
        for (int j = 1; j < argc; ++j)
            if (j != i && j != i + 1) forwarded.push_back(argv[j]);
        return qwen35_golden::run_golden((int)forwarded.size(), forwarded.data());
    }

    std::string model_dir;
    std::vector<int> pp_list = {512, 1024, 2048, 4096};
    std::vector<int> tg_list = {128};
    std::vector<int> depths  = {0, 512, 1024, 2048};
    std::string device_index;
    int reps = 3, warmup = 1, max_seq = -1;
    bool device_index_set = false;
    bool spec = false;
    int spec_tokens = 128;
    std::string spec_prompt =
        "Write a short technical explanation of why autoregressive decoding in "
        "a large language model is limited by memory bandwidth.";

    for (int i = 1; i < argc; ++i) {
        if      (!strcmp(argv[i], "--model") && i+1<argc) model_dir = argv[++i];
        else if ((!strcmp(argv[i], "-p") || !strcmp(argv[i], "--p")) && i+1<argc)
            pp_list = arcaine::bench::parse_int_csv(argv[++i]);
        else if ((!strcmp(argv[i], "-n") || !strcmp(argv[i], "--n")) && i+1<argc)
            tg_list = arcaine::bench::parse_int_csv(argv[++i]);
        else if (!strcmp(argv[i], "-d") && i+1<argc) depths = arcaine::bench::parse_int_csv(argv[++i]);
        else if ((!strcmp(argv[i], "-r") || !strcmp(argv[i], "--r") || !strcmp(argv[i], "--reps")) && i+1<argc) reps = atoi(argv[++i]);
        else if ((!strcmp(argv[i], "-w") || !strcmp(argv[i], "--w") || !strcmp(argv[i], "--warmup")) && i+1<argc) warmup = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--max-seq") && i+1<argc) max_seq = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--device") && i+1<argc) { device_index = argv[++i]; device_index_set = true; }
        else if (!strcmp(argv[i], "--spec")) spec = true;
        else if (!strcmp(argv[i], "--spec-tokens") && i+1<argc) spec_tokens = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--spec-prompt") && i+1<argc) spec_prompt = argv[++i];
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { std::fputs(USAGE, stderr); return 0; }
        else if (argv[i][0] != '-' && model_dir.empty()) model_dir = argv[i];
        else { std::fprintf(stderr, "Unknown argument: %s\n", argv[i]); std::fputs(USAGE, stderr); return 1; }
    }

    if (model_dir.empty() || pp_list.empty() || tg_list.empty() || depths.empty() || reps <= 0 || warmup < 0) {
        std::fputs(USAGE, stderr); return 1;
    }
    for (int v : pp_list) if (v <= 0) { std::fputs("--p values must be positive\n", stderr); return 1; }
    for (int v : tg_list) if (v <= 0) { std::fputs("--n values must be positive\n", stderr); return 1; }

    try { if (device_index_set) gpu_device_control::apply_device_index(device_index); }
    catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }

    if (max_seq < 0) {
        if (spec) {
            max_seq = std::max(2048, spec_tokens + 512);
        } else {
            int max_d = *std::max_element(depths.begin(), depths.end());
            int max_p = *std::max_element(pp_list.begin(), pp_list.end());
            int max_n = *std::max_element(tg_list.begin(), tg_list.end());
            max_seq = std::max(max_d + max_n, max_p);
        }
    }

    std::printf("loading model from %s ...\n", model_dir.c_str());
    double t0 = now_ms();
    Qwen35Model model(model_dir, max_seq);
    double load_s = (now_ms() - t0) * 0.001;
    const ModelInfo& info = model.info();
    std::printf("model   : %s\n", info.description.c_str());
    std::printf("backend : SYCL + oneDNN | GPUs: %d | max_seq: %d",
                GpuEngine::count(), max_seq);
    if (const char* active_gpus = gpu_device_control::active_gpus_spec())
        std::printf(" | ZE_AFFINITY_MASK=%s", active_gpus);
    std::printf("\nload    : %.1f s\n", load_s);

    if (spec) return run_spec(model, spec_prompt, spec_tokens);

    const int bos_id = info.bos_token_id;
    const std::vector<int> single(1, bos_id);
    arcaine::bench::print_pp_tg_header();

    for (int pp : pp_list) {
        if (pp > max_seq) {
            char name[32]; std::snprintf(name, sizeof name, "pp %d", pp);
            std::printf(" %-18s %9s   [skip: pp=%d > max_seq=%d]\n", name, "—", pp, max_seq);
            continue;
        }
        const std::vector<int> prompt(pp, bos_id);
        std::vector<double> times;
        for (int r = 0; r < warmup + reps; ++r) {
            model.reset_cache();
            double t = now_ms();
            model.forward(ForwardInput{prompt, 0});
            double dt = now_ms() - t;
            if (r >= warmup) times.push_back(dt);
        }
        char name[32]; std::snprintf(name, sizeof name, "pp %d", pp);
        arcaine::bench::print_pp_tg_row(name, "—", arcaine::bench::compute_pp_tg_stats(times, pp));
    }

    for (int tg : tg_list) {
        for (int depth : depths) {
            char name[32]; std::snprintf(name, sizeof name, "tg %d", tg);
            char dstr[32]; std::snprintf(dstr, sizeof dstr, "%d", depth);
            if (depth + tg > max_seq) {
                arcaine::bench::print_pp_tg_row(name, dstr, {}, /*skipped=*/true);
                continue;
            }
            std::vector<double> times;
            for (int r = 0; r < warmup + reps; ++r) {
                model.reset_cache();
                if (depth > 0) {
                    constexpr int CHUNK = 512;
                    for (int pos = 0; pos < depth; pos += CHUNK) {
                        int sz = std::min(CHUNK, depth - pos);
                        std::vector<int> chunk(sz, bos_id);
                        model.forward(ForwardInput{chunk, pos});
                    }
                }
                double t = now_ms();
                for (int step = 0; step < tg; ++step)
                    model.forward(ForwardInput{single, depth + step});
                double dt = now_ms() - t;
                if (r >= warmup) times.push_back(dt);
            }
            arcaine::bench::print_pp_tg_row(name, dstr, arcaine::bench::compute_pp_tg_stats(times, tg));
        }
    }
    std::printf("\n");
    return 0;
}

REGISTER_MODEL_BENCH("qwen3_5", "Qwen3.5 AR DeltaNet (PP/TG KV-depth throughput)", run)
