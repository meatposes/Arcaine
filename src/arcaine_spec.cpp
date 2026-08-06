// arcaine_spec — measures what the MTP head is worth before anything is built
// on top of it.
//
// The head drafts the token two positions ahead. Whether that is useful comes
// down to two numbers, and both are measured here against the same greedy
// decode:
//
//   acceptance  how often the draft matches what the backbone actually
//               produces. This doubles as the correctness check: the head is
//               wired through a concatenation whose order and whose choice of
//               pre- or post-final-norm hidden state cannot be inferred from
//               the tensor shapes, and getting either wrong leaves the model
//               running and the acceptance rate near zero.
//
//   cost        the head is one decoder layer against the backbone's sixty
//               four, so a draft should be cheap. The break-even acceptance
//               rate is just that cost ratio, and it is reported alongside so
//               the margin is explicit rather than assumed.
//
// The backbone is never skipped here: every token is produced normally and the
// draft is scored against it. That keeps this a measurement of the head rather
// than of a speculative loop, and it is the number that says whether the loop
// is worth writing.
//
// Usage:
//   arcaine_spec --model DIR [--prompt TEXT] [--tokens N] [--device N]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "common/model_interface.hpp"
#include "common/registry.hpp"
#include "common/gpu/device_select.hpp"
#include "modeling/qwen3_5/model.hpp"

namespace {

using Clk = std::chrono::high_resolution_clock;
double ms_since(Clk::time_point t) {
    return std::chrono::duration<double, std::milli>(Clk::now() - t).count();
}

int argmax(const std::vector<float>& x) {
    return (int)(std::max_element(x.begin(), x.end()) - x.begin());
}

double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t mid = v.size() / 2;
    return v.size() % 2 ? v[mid] : 0.5 * (v[mid - 1] + v[mid]);
}

void usage() {
    std::fputs(
        "Usage: arcaine_spec --model DIR [options]\n"
        "  --prompt TEXT   prompt to decode from\n"
        "  --tokens N      tokens to generate    (default: 128)\n"
        "  --max-seq N     KV capacity           (default: auto)\n"
        "  --device N      restrict to one Level Zero GPU\n",
        stderr);
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_dir, device_index;
    std::string prompt =
        "Write a short technical explanation of why autoregressive decoding in "
        "a large language model is limited by memory bandwidth.";
    int tokens = 128, max_seq = -1;
    bool device_index_set = false;

    for (int i = 1; i < argc; ++i) {
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", argv[i]); std::exit(1); }
            return argv[++i];
        };
        std::string a = argv[i];
        if      (a == "--model")   model_dir = next();
        else if (a == "--prompt")  prompt    = next();
        else if (a == "--tokens")  tokens    = std::stoi(next());
        else if (a == "--max-seq") max_seq   = std::stoi(next());
        else if (a == "--device") { device_index = next(); device_index_set = true; }
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else { std::fprintf(stderr, "unknown argument: %s\n", a.c_str()); usage(); return 1; }
    }
    if (model_dir.empty() || tokens < 1) { usage(); return 1; }

    try {
        if (device_index_set) gpu_device_control::apply_device_index(device_index);
        register_builtin_architectures();

        if (max_seq < 0) max_seq = std::max(2048, tokens + 512);
        std::printf("[spec] loading %s\n", model_dir.c_str());
        std::unique_ptr<Model> model =
            ModelRegistry::instance().create(model_dir, max_seq);
        const ModelInfo& info = model->info();
        std::printf("[spec] %s\n", info.description.c_str());
        if (!model->has_mtp()) {
            std::fputs("[spec] this model reports no MTP head; nothing to measure\n",
                       stderr);
            return 1;
        }

        PreparedInput prepared = model->prepare_input(prompt, {}, {}, "");
        const std::vector<int>& prompt_tokens = prepared.tokens;
        std::printf("[spec] prompt %d tokens, generating %d\n",
                    (int)prompt_tokens.size(), tokens);

        auto t_prefill = Clk::now();
        std::vector<float> logits = model->forward(ForwardInput{
            prompt_tokens, 0, nullptr, nullptr, &prepared.mm_token_type_ids});
        double prefill_ms = ms_since(t_prefill);

        int past = (int)prompt_tokens.size();
        int accepted = 0, scored = 0;
        std::vector<double> backbone_ms, draft_ms;
        // The draft produced on the previous step, waiting to be scored
        // against the token the backbone actually emits next.
        int pending_draft = -1;

        for (int step = 0; step < tokens; ++step) {
            for (int id : info.suppress_tokens)
                if (id >= 0 && id < (int)logits.size())
                    logits[id] = -std::numeric_limits<float>::infinity();
            int next = argmax(logits);
            if (info.is_eos(next)) {
                std::printf("[spec] EOS after %d tokens\n", step);
                break;
            }

            if (pending_draft >= 0) {
                ++scored;
                if (pending_draft == next) ++accepted;
            }

            // Draft the token after `next`, from the hidden state the last
            // forward already produced. Position of `next` is `past`.
            auto t_draft = Clk::now();
            std::vector<float> draft = model->mtp_draft(next, past);
            draft_ms.push_back(ms_since(t_draft));
            pending_draft = draft.empty() ? -1 : argmax(draft);

            std::vector<int> step_tokens{next};
            auto t_step = Clk::now();
            logits = model->forward(ForwardInput{step_tokens, past});
            backbone_ms.push_back(ms_since(t_step));
            ++past;
        }

        double bb = median(backbone_ms);
        double df = median(draft_ms);
        double rate = scored ? (double)accepted / scored : 0.0;
        double cost = bb > 0.0 ? df / bb : 0.0;

        std::printf("\n  prefill              %.1f ms (%d tokens)\n",
                    prefill_ms, (int)prompt_tokens.size());
        std::printf("  backbone step        %.2f ms (median of %zu)\n",
                    bb, backbone_ms.size());
        std::printf("  mtp draft            %.2f ms (median of %zu)\n",
                    df, draft_ms.size());
        std::printf("  draft cost           %.3f of a backbone step\n", cost);
        std::printf("  acceptance           %.4f  (%d/%d)\n", rate, accepted, scored);
        std::printf("  break-even           %.4f\n", cost);

        // With one draft token per step, a speculative loop runs the backbone
        // once per accepted pair instead of twice, so the ceiling is
        // (1 + rate) tokens per backbone step against the draft's overhead.
        double speedup = (1.0 + rate) / (1.0 + cost);
        std::printf("  projected speedup    %.3fx  (depth-1, greedy)\n", speedup);
        if (rate < 0.05)
            std::printf("\n  Acceptance this low means the head is mis-wired, not weak.\n");

        // ---- end-to-end A/B --------------------------------------------
        // The projection above is arithmetic. This is the wall clock, run over
        // the same prompt with the same greedy rule, so the two sequences must
        // come out identical; a mismatch means the rollback is losing state.
        auto* qwen = dynamic_cast<Qwen35Model*>(model.get());
        if (!qwen) return 0;

        model->reset_cache();
        auto t_base = Clk::now();
        std::vector<float> base_logits = model->forward(ForwardInput{
            prompt_tokens, 0, nullptr, nullptr, &prepared.mm_token_type_ids});
        std::vector<int> baseline;
        int base_past = (int)prompt_tokens.size();
        while ((int)baseline.size() < tokens) {
            int next = argmax(base_logits);
            if (info.is_eos(next)) break;
            baseline.push_back(next);
            std::vector<int> step{next};
            base_logits = model->forward(ForwardInput{step, base_past});
            ++base_past;
        }
        double base_ms = ms_since(t_base);

        // Control. Replays the baseline's own tokens two at a time through the
        // same verify pass the speculative loop uses, with no drafting and no
        // rollback. If this diverges, the cause is that a two-token forward
        // does not reproduce a one-token forward, and no amount of correct
        // state restoration would help.
        model->reset_cache();
        qwen->forward(ForwardInput{prompt_tokens, 0, nullptr, nullptr,
                                   &prepared.mm_token_type_ids});
        int batched_past = (int)prompt_tokens.size();
        size_t batched_match = 0;
        for (size_t i = 0; i + 1 < baseline.size(); i += 2) {
            std::vector<float> pair = qwen->forward_verify(
                {baseline[i], baseline[i + 1]}, batched_past);
            int predicted_first = argmax(std::vector<float>(
                pair.begin(), pair.begin() + info.vocab_size));
            if (i + 1 < baseline.size() && predicted_first != baseline[i + 1]) break;
            ++batched_match;
            int predicted_second = argmax(std::vector<float>(
                pair.begin() + info.vocab_size, pair.begin() + 2 * info.vocab_size));
            if (i + 2 < baseline.size() && predicted_second != baseline[i + 2]) break;
            ++batched_match;
            batched_past += 2;
        }

        model->reset_cache();
        Qwen35Model::SpecStats stats;
        auto t_spec = Clk::now();
        std::vector<int> speculative =
            qwen->generate_speculative(prompt_tokens, tokens, stats);
        double spec_ms = ms_since(t_spec);

        size_t common = 0;
        while (common < baseline.size() && common < speculative.size() &&
               baseline[common] == speculative[common])
            ++common;

        std::printf("\n  --- end to end, greedy, %d tokens ---\n", tokens);
        std::printf("  baseline             %.1f ms   %.2f tok/s   (%zu tokens)\n",
                    base_ms, baseline.size() / (base_ms * 1e-3), baseline.size());
        std::printf("  speculative          %.1f ms   %.2f tok/s   (%zu tokens)\n",
                    spec_ms, speculative.size() / (spec_ms * 1e-3),
                    speculative.size());
        std::printf("  speedup              %.3fx\n",
                    (speculative.size() / (spec_ms * 1e-3)) /
                        (baseline.size() / (base_ms * 1e-3)));
        std::printf("  rounds               %d  (%d accepted, %.4f)\n",
                    stats.rounds, stats.accepts,
                    stats.rounds ? (double)stats.accepts / stats.rounds : 0.0);
        std::printf("  backbone passes      %d for %zu tokens  (%.3f tok/pass)\n",
                    stats.forwards, speculative.size(),
                    stats.forwards ? (double)speculative.size() / stats.forwards : 0.0);
        std::printf("  draft / verify / rollback   %.1f / %.1f / %.1f ms\n",
                    stats.draft_ms, stats.verify_ms, stats.rollback_ms);
        std::printf("  batched control      %zu/%zu tokens reproduced by "
                    "2-token forwards, no rollback\n",
                    batched_match, baseline.size());
        if (common == baseline.size() && common == speculative.size()) {
            std::printf("  sequences            identical (%zu tokens)\n", common);
        } else if (batched_match <= common) {
            // The control diverged no later than the speculative run, so a
            // two-token forward already fails to reproduce a one-token forward
            // and the drafting is not what moved the sequence. Batch size
            // selects different kernels in this engine — see the seq == 1
            // branch in qwen35_linear_attention_forward — so this is a
            // property of the backbone, not of speculation.
            std::printf("  diverges at %zu of %zu; the batched control diverges "
                        "at %zu, so this is M=1 vs M=2 kernel numerics, not "
                        "rollback\n",
                        common, baseline.size(), batched_match);
        } else {
            std::printf("  MISMATCH             diverge at %zu of %zu/%zu while "
                        "the batched control reproduced %zu — rollback is "
                        "losing state\n",
                        common, baseline.size(), speculative.size(), batched_match);
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
