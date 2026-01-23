import os

from transformers import AutoTokenizer

model_path = "/home/majinming/models/Qwen3-30B-A3B-Instruct-2507"
if not os.path.exists(model_path):
    print(f"Model path {model_path} does not exist.")
    exit(1)

try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Analyze the suspicious sequence provided by user
    # 151644 (Start), 77091 (???)

    token_id_suspect = 77091
    decoded = tokenizer.decode([token_id_suspect])
    print(f"Token {token_id_suspect} decodes to: '{decoded}'")

    # Check 'assistant' token
    assistant_tokens = tokenizer.encode("assistant", add_special_tokens=False)
    print(f"'assistant' encodes to: {assistant_tokens}")

    # Check special tokens
    print(f"im_start: {tokenizer.encode('<|im_start|>', add_special_tokens=False)}")
    print(f"im_end: {tokenizer.encode('<|im_end|>', add_special_tokens=False)}")

    # Apply Chat Template to see expected structure
    messages = [{"role": "user", "content": "Help me"}]
    templated = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    print(f"Templated IDs for 'Help me': {templated}")
    print(f"Templated Decoded: {tokenizer.decode(templated)}")

except Exception as e:
    print(f"Error: {e}")
