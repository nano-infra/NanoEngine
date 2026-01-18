import argparse
import json
import sys

import requests
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Chat client for NanoDeploy server")
    parser.add_argument(
        "--endpoint",
        type=str,
        default="http://localhost:3000/chat",
        help="Server endpoint",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/models/Qwen3-235B-A22B-Instruct-2507/",
        help="Path to model directory for tokenizer",
    )
    parser.add_argument(
        "--prompt", type=str, default="Hello, who are you?", help="Prompt to send"
    )
    parser.add_argument(
        "--max_tokens", type=int, default=128, help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--chat", action="store_true", default=True, help="Use chat template"
    )

    args = parser.parse_args()

    # 1. Initialize Tokenizer
    print(f"Loading tokenizer from {args.model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, trust_remote_code=True
        )
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    # 2. Tokenize input
    if args.chat:
        messages = [{"role": "user", "content": args.prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
    else:
        input_ids = tokenizer.encode(args.prompt)

    print(f"Input IDs: {input_ids}")

    # 3. Send tokens to server
    payload = {"prompt_ids": input_ids, "max_tokens": args.max_tokens}

    print(f"Sending request to {args.endpoint}...")
    try:
        response = requests.post(args.endpoint, json=payload, stream=True)
        response.raise_for_status()
    except Exception as e:
        print(f"Error connecting to server: {e}")
        return

    # 4. Receive tokens and detokenize
    print("Response: ", end="", flush=True)
    full_response_ids = []
    current_text_len = 0

    for line in response.iter_lines():
        if line:
            line_str = line.decode("utf-8")
            if line_str.startswith("data:"):
                try:
                    # Remove "data: " prefix and parse JSON
                    data_content = line_str[5:].strip()
                    token_ids = json.loads(data_content)

                    if isinstance(token_ids, list):
                        # The server returns a list of tokens in each SSE event
                        full_response_ids.extend(token_ids)

                        # Decode the full sequence to handle multi-byte characters and spacing correctly
                        current_text = tokenizer.decode(
                            full_response_ids, skip_special_tokens=True
                        )

                        # Print only the new part
                        new_text = current_text[current_text_len:]
                        print(new_text, end="", flush=True)
                        current_text_len = len(current_text)

                except Exception as e:
                    print(f"\nError parsing SSE data: {e}")
                    continue

    print("\n\nDone.")


if __name__ == "__main__":
    main()
