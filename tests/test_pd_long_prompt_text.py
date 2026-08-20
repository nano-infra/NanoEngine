from __future__ import annotations

import sys
from pathlib import Path


EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES_DIR))

import pd_disagg_deepseek_v3_parallel as pd_smoke  # noqa: E402


class _CharacterChatTokenizer:
    """Small tokenizer oracle whose prompt length follows real text length."""

    _CHAT_TEMPLATE_TOKENS = 4

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]:
        assert tokenize
        assert add_generation_prompt
        assert len(messages) == 1
        content = messages[0]["content"]
        return list(range(len(content) + self._CHAT_TEMPLATE_TOKENS))


def test_long_prompt_length_is_real_text_before_tokenization():
    tokenizer = _CharacterChatTokenizer()
    question = "为什么月亮沿轨道运行？"
    background = "轨道运动需要同时考虑引力、惯性和切向速度。"
    target_length = 600

    prompt = pd_smoke.make_long_prompt_with_token_length(
        tokenizer,
        question,
        background,
        target_length,
    )

    assert isinstance(prompt, str)
    assert prompt.count(question) == 2
    assert "第0001段：" in prompt
    assert background in prompt
    assert len(pd_smoke._chat_prompt_token_ids(tokenizer, prompt)) == target_length
