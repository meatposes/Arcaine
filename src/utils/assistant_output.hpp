#pragma once

#include <string>
#include <vector>

// Parsing of raw assistant text into the OpenAI response shape.
//
// Tool-call and reasoning syntax is a property of the checkpoint's chat
// template, not of the API. Gemma4 emits `<|tool_call>call: name {...}<tool_call|>`
// with JSON arguments and `<|channel>thought` blocks; Qwen3.5 emits
// `<tool_call><function=name><parameter=p>value</parameter></function></tool_call>`
// with `<think>` blocks. Applying one architecture's grammar to another does not
// fail loudly -- it silently returns the model's tool call as prose and leaves
// tool_calls empty -- so the format is selected per model rather than assumed.
struct ParsedToolCall {
    std::string id;
    std::string name;
    std::string arguments;  // JSON object, serialized
};

struct ParsedAssistantOutput {
    std::string content;
    // Chain-of-thought, separated from content when the template emits it in a
    // dedicated block. Surfaced as `reasoning_content` on the response message.
    std::string reasoning;
    std::vector<ParsedToolCall> tool_calls;
};

enum class AssistantOutputFormat {
    // `<|tool_call>` / `<|channel>` markers. Also the fallback for any
    // architecture with no known grammar: it strips Gemma's markers, which
    // other templates do not produce, so it degrades to a passthrough.
    Gemma4,
    // `<tool_call><function=...>` XML and `<think>` blocks. Qwen3.5 / Qwen3.6.
    Qwen35,
};

// Select by config.json::model_type.
AssistantOutputFormat assistant_output_format_for(const std::string& model_type);

ParsedAssistantOutput parse_assistant_output(const std::string& raw_text,
                                             AssistantOutputFormat format);

// Architecture-specific entry points, exposed for testing.
ParsedAssistantOutput parse_gemma4_assistant_output(const std::string& raw_text);
ParsedAssistantOutput parse_qwen35_assistant_output(const std::string& raw_text);
