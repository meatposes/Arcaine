#include "assistant_output.hpp"

#include <nlohmann/json.hpp>

namespace {

using json = nlohmann::ordered_json;

std::string trim_copy(const std::string& text) {
    size_t begin = text.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) return {};
    size_t end = text.find_last_not_of(" \t\r\n");
    return text.substr(begin, end - begin + 1);
}

void erase_all(std::string& text, const std::string& needle) {
    size_t pos = 0;
    while ((pos = text.find(needle, pos)) != std::string::npos)
        text.erase(pos, needle.size());
}

// A parameter body is raw text, and the template carries no type information,
// so a value that happens to be valid JSON is taken at its parsed type and
// everything else stays a string. This is what makes `limit: 5` an integer and
// `query: Paris` a string without consulting the tool schema.
json parameter_value(const std::string& raw) {
    std::string text = trim_copy(raw);
    if (text.empty()) return json("");
    try {
        json parsed = json::parse(text);
        if (parsed.is_number() || parsed.is_boolean() || parsed.is_null() ||
            parsed.is_array() || parsed.is_object())
            return parsed;
    } catch (const json::exception&) {
        // Not JSON: keep the literal text.
    }
    return json(text);
}

// One `<function=NAME> <parameter=P>V</parameter> ... </function>` block.
bool parse_qwen35_function(const std::string& body, size_t index,
                           ParsedToolCall& out) {
    const std::string function_open = "<function=";
    size_t start = body.find(function_open);
    if (start == std::string::npos) return false;
    size_t name_end = body.find('>', start + function_open.size());
    if (name_end == std::string::npos) return false;
    std::string name =
        trim_copy(body.substr(start + function_open.size(),
                              name_end - start - function_open.size()));
    if (name.empty()) return false;

    json arguments = json::object();
    const std::string parameter_open = "<parameter=";
    const std::string parameter_close = "</parameter>";
    size_t cursor = name_end + 1;
    while (true) {
        size_t p = body.find(parameter_open, cursor);
        if (p == std::string::npos) break;
        size_t key_end = body.find('>', p + parameter_open.size());
        if (key_end == std::string::npos) break;
        std::string key =
            trim_copy(body.substr(p + parameter_open.size(),
                                  key_end - p - parameter_open.size()));
        size_t value_end = body.find(parameter_close, key_end + 1);
        if (value_end == std::string::npos) break;
        std::string value = body.substr(key_end + 1, value_end - key_end - 1);
        if (!key.empty()) arguments[key] = parameter_value(value);
        cursor = value_end + parameter_close.size();
    }

    out.id = "call_" + std::to_string(index);
    out.name = std::move(name);
    out.arguments = arguments.dump();
    return true;
}

// Mirrors the split the checkpoint's own chat template performs when it replays
// a prior assistant turn:
//
//   {%- if '</think>' in content %}
//     reasoning = content.split('</think>')[0] ... split('<think>')[-1]
//     content   = content.split('</think>')[-1]
//
// The template ends the prompt with a bare `<think>\n`, so generation begins
// *inside* the block and the opening tag never appears in the output. Keying on
// the closing tag rather than a matched pair is what makes that work; keying on
// a pair would classify every reasoning turn as content.
void split_qwen35_reasoning(std::string& content, std::string& reasoning) {
    const std::string think_close = "</think>";
    size_t close = content.find(think_close);
    if (close == std::string::npos) return;

    std::string head = content.substr(0, close);
    const std::string think_open = "<think>";
    size_t open = head.rfind(think_open);
    if (open != std::string::npos) head = head.substr(open + think_open.size());

    reasoning = trim_copy(head);
    content = content.substr(close + think_close.size());
}

}  // namespace

ParsedAssistantOutput parse_qwen35_assistant_output(const std::string& raw_text) {
    ParsedAssistantOutput out;
    std::string content = raw_text;

    split_qwen35_reasoning(content, out.reasoning);

    const std::string call_open = "<tool_call>";
    const std::string call_close = "</tool_call>";
    size_t search = 0;
    size_t index = 0;
    while (true) {
        size_t s = content.find(call_open, search);
        if (s == std::string::npos) break;
        size_t body_start = s + call_open.size();
        size_t e = content.find(call_close, body_start);
        if (e == std::string::npos) break;

        ParsedToolCall call;
        if (parse_qwen35_function(content.substr(body_start, e - body_start),
                                  index, call)) {
            out.tool_calls.push_back(std::move(call));
            ++index;
        }
        // A block that does not parse is dropped rather than left in the text:
        // the template forbids prose after a call, so a malformed block is a
        // failed call, not an answer.
        content.erase(s, e + call_close.size() - s);
        search = s;
    }

    erase_all(content, "<|im_end|>");
    erase_all(content, "<|endoftext|>");
    out.content = trim_copy(content);
    return out;
}

AssistantOutputFormat assistant_output_format_for(const std::string& model_type) {
    if (model_type == "qwen3_5" || model_type == "qwen3_5_moe_text")
        return AssistantOutputFormat::Qwen35;
    return AssistantOutputFormat::Gemma4;
}

ParsedAssistantOutput parse_assistant_output(const std::string& raw_text,
                                             AssistantOutputFormat format) {
    switch (format) {
        case AssistantOutputFormat::Qwen35:
            return parse_qwen35_assistant_output(raw_text);
        case AssistantOutputFormat::Gemma4:
            break;
    }
    return parse_gemma4_assistant_output(raw_text);
}
