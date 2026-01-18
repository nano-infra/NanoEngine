from transformers import AutoTokenizer

model_path = "/home/majinming/qwen3-0.6b-local"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

prompt = "Please introduce San Francisco."

# Case 1: No system prompt (Reference script behavior)
messages_ref = [{"role": "user", "content": prompt}]
ids_ref = tokenizer.apply_chat_template(
    messages_ref, tokenize=True, add_generation_prompt=True
)
print(f"Reference IDs (len {len(ids_ref)}): {ids_ref}")
print(f"Reference decoded: {tokenizer.decode(ids_ref)}")

# Case 2: With System Prompt
sys_prompt = "You are a helpful assistant."
messages_sys = [
    {"role": "system", "content": sys_prompt},
    {"role": "user", "content": prompt},
]
ids_sys = tokenizer.apply_chat_template(
    messages_sys, tokenize=True, add_generation_prompt=True
)
print(f"System Prompt IDs (len {len(ids_sys)}): {ids_sys}")
print(f"System Prompt decoded: {tokenizer.decode(ids_sys)}")
