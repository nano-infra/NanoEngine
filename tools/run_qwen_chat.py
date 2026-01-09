import argparse
import os
import subprocess
import sys

from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Run Qwen3 C++ Demo with Python Tokenizer"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model directory (containing config.json)",
    )
    parser.add_argument(
        "--exe_path", type=str, required=True, help="Path to test_qwen3_runner.exe"
    )
    parser.add_argument(
        "--prompt", type=str, default="Hello world!", help="Prompt text"
    )
    parser.add_argument("--agent_ip", type=str, default="127.0.0.1", help="Agent IP")
    parser.add_argument("--agent_port", type=int, default=9000, help="Agent Port")

    parser.add_argument("--chat", action="store_true", help="Apply chat template")

    args = parser.parse_args()

    # 1. Load Tokenizer
    print(f"[Python] Loading tokenizer from {args.model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, trust_remote_code=True
        )
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    # 2. Tokenize prompt
    print(f"[Python] Tokenizing prompt: '{args.prompt}'")
    if args.chat:
        messages = [{"role": "user", "content": args.prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
    else:
        input_ids = tokenizer.encode(args.prompt)

    print(f"[Python] Input IDs: {input_ids}")

    # 3. Call C++ Engine
    # Command: exe config_path id1 id2 ... --agent_ip <ip> --agent_port <port>
    config_path = os.path.join(args.model_path, "config.json")
    cmd = [args.exe_path, config_path] + [str(x) for x in input_ids]
    cmd.extend(["--agent_ip", args.agent_ip])
    cmd.extend(["--agent_port", str(args.agent_port)])

    print(f"[Python] Running C++ Engine...")
    try:
        # Run and capture output
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("--- C++ Output ---")
        print(result.stdout)
        print("------------------")

        # 4. Parse Output IDs
        # Look for "Generated IDs: ..."
        output_ids = []
        for line in result.stdout.splitlines():
            if "Generated IDs:" in line:
                parts = line.split("Generated IDs:")[1].strip().split()
                output_ids = [int(p) for p in parts]
                break

        if not output_ids:
            print("[Python] Warning: No generated IDs found in C++ output.")
        else:
            print(f"[Python] Generated IDs: {output_ids}")

            # 5. Decode
            decoded_text = tokenizer.decode(output_ids)
            print(f"\n[Python] **Response:** {decoded_text}\n")

    except subprocess.CalledProcessError as e:
        print(f"Error running C++ engine:\n{e.stderr}")
        print("--- C++ Partial Output ---")
        print(e.stdout)
        print("--------------------------")

        # Try to parse output anyway
        output_ids = []
        for line in e.stdout.splitlines():
            if "Generated IDs:" in line:
                try:
                    parts = line.split("Generated IDs:")[1].strip().split()
                    output_ids = [int(p) for p in parts]
                except:
                    pass
                break

        if output_ids:
            print(f"[Python] Generated IDs: {output_ids}")
            decoded_text = tokenizer.decode(output_ids)
            print(f"\n[Python] **Response:** {decoded_text}\n")


if __name__ == "__main__":
    main()
