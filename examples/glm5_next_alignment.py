"""Token-level GLM-5.3 reference alignment helper.

The reference side is intentionally run from a separate Transformers checkout:

    PYTHONPATH=/mnt/mnt/public/majinming/src/huggingface \
      python examples/glm5_next_alignment.py --model /nvmedata/GLM-5.3-Flash

The script uses the checkpoint chat template and greedy decoding, then queries
an already running ``dlengine serve`` endpoint and reports the first mismatch.
It is small enough to run with a short prompt while still checking logits and
the complete generated token sequence.
"""
from __future__ import annotations

import argparse
import json
import urllib.request

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer


def encode_prompt(tokenizer, prompt: str) -> list[int]:
    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        text = prompt
    return tokenizer.encode(text, add_special_tokens=False)


@torch.inference_mode()
def reference_tokens(model, input_ids: list[int], count: int) -> tuple[list[int], torch.Tensor]:
    device = next(model.parameters()).device
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    logits = model(input_ids=ids).logits[:, -1, :]
    out: list[int] = []
    for _ in range(count):
        token = int(logits.argmax(-1).item())
        out.append(token)
        ids = torch.cat((ids, torch.tensor([[token]], device=device)), dim=1)
        logits = model(input_ids=ids).logits[:, -1, :]
    return out, logits


def nano_tokens(endpoint: str, model_name: str, prompt: str, count: int, tokenizer) -> list[int]:
    body = json.dumps(
        {
            "model": model_name,
            "prompt": prompt,
            "max_tokens": count,
            "temperature": 0,
            "stream": False,
        }
    ).encode()
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as response:
        payload = json.load(response)
    choice = payload["choices"][0]
    token_ids = choice.get("token_ids")
    if token_ids is not None:
        return list(token_ids)
    # OpenAI-compatible clients normally receive text only.  Re-tokenizing the
    # returned completion is adequate for greedy short-token checks and keeps
    # this helper compatible with both server response formats.
    return tokenizer.encode(choice.get("text", ""), add_special_tokens=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/nvmedata/GLM-5.3-Flash")
    p.add_argument("--endpoint", default="http://127.0.0.1:8000")
    p.add_argument("--served-model-name", default=None)
    p.add_argument("--prompt", default="请计算 2+2，只回答结果。")
    p.add_argument("--max-tokens", type=int, default=8)
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompt_ids = encode_prompt(tokenizer, args.prompt)
    rendered = tokenizer.decode(prompt_ids)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    expected, _ = reference_tokens(model, prompt_ids, args.max_tokens)
    actual = nano_tokens(
        args.endpoint,
        args.served_model_name or args.model.rsplit("/", 1)[-1],
        rendered,
        args.max_tokens,
        tokenizer,
    )
    print("prompt_token_ids:", prompt_ids)
    print("reference_token_ids:", expected)
    print("nano_token_ids:", actual)
    for i, (want, got) in enumerate(zip(expected, actual)):
        if want != got:
            raise SystemExit(f"token mismatch at index {i}: reference={want}, nano={got}")
    if len(expected) != len(actual):
        raise SystemExit(f"token count mismatch: reference={len(expected)}, nano={len(actual)}")
    print("MATCH", tokenizer.decode(expected, skip_special_tokens=False))


if __name__ == "__main__":
    main()
