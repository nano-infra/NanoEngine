/// Qwen XML tool call parser.
///
/// Parses the output format emitted by Qwen3.5 function-calling models:
///
/// ```text
/// <tool_call>
/// <function=get_weather>
/// <parameter=city>
/// Paris
/// </parameter>
/// </function>
/// </tool_call>
/// ```

#[derive(Debug, Clone)]
pub struct ParsedToolCall {
    pub name: String,
    pub arguments: serde_json::Map<String, serde_json::Value>,
}

const TC_START: &str = "<tool_call>";
const TC_END: &str = "</tool_call>";
const FUNC_PREFIX: &str = "<function=";
const FUNC_END: &str = "</function>";
const PARAM_PREFIX: &str = "<parameter=";
const PARAM_END: &str = "</parameter>";
const THINK_START: &str = "<think>";
const THINK_END: &str = "</think>";

/// Strip a leading `<think>...</think>` block and return the remaining text.
fn strip_thinking(text: &str) -> &str {
    if let Some(start_pos) = text.find(THINK_START) {
        if let Some(end_rel) = text[start_pos..].find(THINK_END) {
            let after = &text[start_pos + end_rel + THINK_END.len()..];
            return after.trim_start();
        }
    }
    text
}

/// Parse a single `<function=NAME>...<parameter=K>V</parameter>...</function>` block.
fn parse_tool_call_block(block: &str) -> Option<ParsedToolCall> {
    // Locate <function=NAME>
    let fn_pos = block.find(FUNC_PREFIX)?;
    let name_start = fn_pos + FUNC_PREFIX.len();
    let gt_pos = block[name_start..].find('>')?;
    let name = block[name_start..name_start + gt_pos].trim().to_string();

    let body_start = name_start + gt_pos + 1;
    let fn_end_pos = block.find(FUNC_END)?;
    if fn_end_pos < body_start {
        return None;
    }
    let body = &block[body_start..fn_end_pos];

    // Parse all <parameter=KEY>VALUE</parameter> blocks
    let mut arguments = serde_json::Map::new();
    let mut search = 0;
    while let Some(p_rel) = body[search..].find(PARAM_PREFIX) {
        let p_pos = search + p_rel;
        let key_start = p_pos + PARAM_PREFIX.len();
        let gt_pos = match body[key_start..].find('>') {
            Some(p) => p,
            None => break,
        };
        let key = body[key_start..key_start + gt_pos].trim().to_string();

        let val_start = key_start + gt_pos + 1;
        let p_end_rel = match body[val_start..].find(PARAM_END) {
            Some(p) => p,
            None => break,
        };
        let val_str = body[val_start..val_start + p_end_rel].trim();

        // Try to parse value as JSON (object/array/number/bool); fall back to string
        let value = serde_json::from_str::<serde_json::Value>(val_str)
            .unwrap_or_else(|_| serde_json::Value::String(val_str.to_string()));
        arguments.insert(key, value);

        search = val_start + p_end_rel + PARAM_END.len();
    }

    Some(ParsedToolCall { name, arguments })
}

/// Parse complete model output.
///
/// Returns `(content_text, tool_calls)`.
/// - `content_text` is any text before the first `<tool_call>` block (thinking stripped).
/// - `tool_calls` is empty when no `<tool_call>` was found (pure-text response).
pub fn parse_tool_calls(text: &str) -> (String, Vec<ParsedToolCall>) {
    let text = strip_thinking(text);

    let mut tool_calls = Vec::new();
    let mut content_end: Option<usize> = None;
    let mut search = 0;

    while let Some(tc_rel) = text[search..].find(TC_START) {
        let tc_pos = search + tc_rel;

        if content_end.is_none() {
            content_end = Some(tc_pos);
        }

        let block_start = tc_pos + TC_START.len();
        let end_rel = match text[block_start..].find(TC_END) {
            Some(p) => p,
            None => break, // incomplete — stop parsing
        };

        let block = &text[block_start..block_start + end_rel];
        if let Some(tc) = parse_tool_call_block(block) {
            tool_calls.push(tc);
        }
        search = block_start + end_rel + TC_END.len();
    }

    let content = match content_end {
        Some(end) => text[..end].trim().to_string(),
        None => text.trim().to_string(),
    };

    (content, tool_calls)
}

/// For streaming: return the byte offset of the end of the last complete
/// `</tool_call>` in `text`, or `None` if there is no complete block yet.
pub fn find_complete_tool_call_end(text: &str) -> Option<usize> {
    let mut last_end = None;
    let mut search = 0;
    while let Some(pos) = text[search..].find(TC_END) {
        let end = search + pos + TC_END.len();
        last_end = Some(end);
        search = end;
    }
    last_end
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_single_tool_call() {
        let text = r#"<tool_call>
<function=get_weather>
<parameter=city>
Paris
</parameter>
</function>
</tool_call>"#;
        let (content, calls) = parse_tool_calls(text);
        assert_eq!(content, "");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(
            calls[0].arguments["city"],
            serde_json::Value::String("Paris".to_string())
        );
    }

    #[test]
    fn test_parse_with_thinking_block() {
        let text = "<think>Let me think about this.</think>\n<tool_call>\n<function=search>\n<parameter=query>rust</parameter>\n</function>\n</tool_call>";
        let (content, calls) = parse_tool_calls(text);
        assert_eq!(content, "");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "search");
    }

    #[test]
    fn test_parse_with_content_before_tool_call() {
        let text = "Sure, let me look that up.\n<tool_call>\n<function=lookup>\n<parameter=id>42</parameter>\n</function>\n</tool_call>";
        let (content, calls) = parse_tool_calls(text);
        assert_eq!(content, "Sure, let me look that up.");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments["id"], serde_json::json!(42));
    }

    #[test]
    fn test_no_tool_call_returns_full_text() {
        let text = "Hello, world! This is a regular response.";
        let (content, calls) = parse_tool_calls(text);
        assert_eq!(content, text);
        assert!(calls.is_empty());
    }

    #[test]
    fn test_find_complete_tool_call_end() {
        let partial = "<tool_call>\n<function=foo>";
        assert!(find_complete_tool_call_end(partial).is_none());

        let complete = "<tool_call>\n<function=foo>\n</function>\n</tool_call>";
        assert_eq!(find_complete_tool_call_end(complete), Some(complete.len()));
    }
}
