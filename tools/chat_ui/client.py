import json

import requests
from transformers import AutoTokenizer

from .config import MAX_TOKENS, MODEL_PATH, SERVER_URL, SYSTEM_PROMPT


class NanoClient:
    def __init__(self, model_path, server_url):
        print(f"Loading tokenizer from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.server_url = server_url

    def generate_stream(self, prompt: str, max_tokens: int = MAX_TOKENS):
        # Using chat template as this is a chat tool
        messages = []
        if SYSTEM_PROMPT:
            messages.append({"role": "system", "content": SYSTEM_PROMPT})
        messages.append({"role": "user", "content": prompt})
        try:
            input_ids = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
        except Exception as e:
            # Fallback if template fails (e.g. no chat template in config)
            print(f"Warning: apply_chat_template failed: {e}. using encode.")
            input_ids = self.tokenizer.encode(prompt)

        payload = {"prompt_ids": input_ids, "max_tokens": max_tokens}

        print(f"Sending request to {self.server_url} with prompt: {prompt[:20]}...")
        try:
            response = requests.post(self.server_url, json=payload, stream=True)
            response.raise_for_status()
        except Exception as e:
            yield f"Error connecting to server: {str(e)}"
            return

        full_response_ids = []
        current_text_len = 0

        for line in response.iter_lines():
            if line:
                line_str = line.decode("utf-8")
                if line_str.startswith("data:"):
                    try:
                        data_content = line_str[5:].strip()
                        token_ids = json.loads(data_content)
                        if isinstance(token_ids, list):
                            full_response_ids.extend(token_ids)
                            current_text = self.tokenizer.decode(
                                full_response_ids, skip_special_tokens=True
                            )

                            # Handle partial unicode characters
                            # If the text ends with the replacement character, don't yield it yet
                            if current_text.endswith("\ufffd"):
                                continue

                            # Calculate new text
                            if len(current_text) > current_text_len:
                                new_text = current_text[current_text_len:]
                                yield new_text
                                current_text_len = len(current_text)
                    except Exception as e:
                        print(f"Error parsing SSE: {e}")


# Singleton pattern to avoid reloading tokenizer
_client_instance = None


def get_client():
    global _client_instance
    if _client_instance is None:
        _client_instance = NanoClient(MODEL_PATH, SERVER_URL)
    return _client_instance
