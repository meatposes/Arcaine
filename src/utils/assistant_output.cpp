#include "assistant_output.hpp"

#include <cstring>

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
//
// Booleans and null need the extra pass: the model writes them Python-style
// (`True` / `False` / `None`), which json::parse rejects, so without this a
// parameter the schema declares boolean reaches the client as the string
// "True". Only an exact match on the whole value is rewritten -- a `query` of
// "None" stays text.
json parameter_value(const std::string& raw) {
    std::string text = trim_copy(raw);
    if (text.empty()) return json("");
    if (text == "True" || text == "true") return json(true);
    if (text == "False" || text == "false") return json(false);
    if (text == "None" || text == "null") return json(nullptr);
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
//
// Deliberately lenient about the opening `<`. This checkpoint drops it and emits
//
//   <tool_call>
//   function=get_weather>
//   <parameter=city>
//   ...
//
// verbatim, confirmed from the raw decode -- the tokenizer round-trips
// `<function=abc>` exactly, so the model is omitting the bracket, not the
// decoder losing it. Abliterated merges adhere to their own template loosely.
// A parser that insists on the documented spelling turns a recoverable
// formatting slip into a dropped response.
bool parse_qwen35_function(const std::string& body, size_t index,
                           ParsedToolCall& out) {
    // Accept "function=" with or without the leading '<'. "</function>" cannot
    // match: it spells "function>", not "function=".
    size_t start = body.find("function=");
    if (start == std::string::npos) return false;
    size_t name_start = start + strlen("function=");
    size_t name_end = body.find('>', name_start);
    if (name_end == std::string::npos) return false;
    std::string name = trim_copy(body.substr(name_start, name_end - name_start));
    if (name.empty()) return false;

    json arguments = json::object();
    size_t cursor = name_end + 1;
    while (true) {
        size_t p = body.find("parameter=", cursor);
        if (p == std::string::npos) break;
        size_t key_start = p + strlen("parameter=");
        size_t key_end = body.find('>', key_start);
        if (key_end == std::string::npos) break;
        std::string key = trim_copy(body.substr(key_start, key_end - key_start));

        // Closing tag, tolerating the same dropped bracket.
        size_t value_end = body.find("</parameter>", key_end + 1);
        size_t close_len = strlen("</parameter>");
        if (value_end == std::string::npos) {
            value_end = body.find("/parameter>", key_end + 1);
            close_len = strlen("/parameter>");
        }
        if (value_end == std::string::npos) break;
        std::string value = body.substr(key_end + 1, value_end - key_end - 1);
        if (!key.empty()) arguments[key] = parameter_value(value);
        cursor = value_end + close_len;
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
            content.erase(s, e + call_close.size() - s);
            search = s;
            continue;
        }
        // Never erase output that failed to parse. An earlier version dropped
        // the block on the theory that a malformed call is a failed call rather
        // than an answer, which turned every unrecognized spelling into an
        // empty response: content empty, tool_calls empty, nothing emitted at
        // all, and the generated tokens discarded with no trace outside the
        // server log. Leaving the text in place degrades to visible-but-wrong,
        // which a caller can see and report.
        search = e + call_close.size();
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
