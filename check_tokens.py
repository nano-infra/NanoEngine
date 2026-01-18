from transformers import AutoTokenizer

model_path = (
    "/home/majinming/NanoDeploy/tools/chat_ui/config.py"  # Just to import, wait.
)
# I will use the path from config.
# But I can just hardcode the path I know is being used: /home/majinming/qwen3-0.6b-local (via symlinks in client effectively)

# The client uses:
# DEFAULT_MODEL_PATH = "/home/majinming/qwen3-0.6b-local"
# (As reverted in Step 141)

tokenizer = AutoTokenizer.from_pretrained(
    "/home/majinming/models/Qwen3-30B-A3B-Instruct-2507", trust_remote_code=True
)

text = "给我写一个一键安装 oh my zsh 的脚本。"
ids = tokenizer.encode(text)
print(f"Text: {text}")
print(f"IDs: {ids}")
decoded = tokenizer.decode(ids)
print(f"Decoded: {decoded}")

# Check individual tokens for omz
print("Token analysis:")
for t_id in ids:
    print(f"{t_id}: {tokenizer.decode([t_id])}")
