import argparse
import os
import re
import subprocess
import sys

from transformers import AutoTokenizer


def parse_output_md(md_path):
    """Parse the generated markdown file to extract sequences and their tokens."""
    sequences = {}
    current_seq_id = None

    if not os.path.exists(md_path):
        return sequences

    with open(md_path, "r") as f:
        content = f.read()

    # Parse each sequence section
    seq_pattern = r"## Sequence (\d+)\s*\n\n\*\*Generated IDs:\*\* ([0-9\s]+)"
    matches = re.findall(seq_pattern, content)

    for seq_id, ids_str in matches:
        ids = [int(x) for x in ids_str.strip().split() if x]
        sequences[int(seq_id)] = ids

    return sequences


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
    parser.add_argument(
        "--output_md",
        type=str,
        default="qwen3_moe_chat_output.md",
        help="Path to output markdown file (default: cwd/qwen3_moe_chat_output.md)",
    )

    args, unknown = parser.parse_known_args()

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
    config_path = os.path.join(args.model_path, "config.json")
    cmd = [args.exe_path, config_path] + [str(x) for x in input_ids]
    cmd.extend(["--agent_ip", args.agent_ip])
    cmd.extend(["--agent_port", str(args.agent_port)])
    cmd.extend(unknown)  # Pass unknown args to C++ binary

    print(f"[Python] Running C++ Engine...")
    print(f"[Python] Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("--- C++ Output ---")
        print(result.stdout)
        if result.stderr:
            print("--- C++ Stderr ---")
            print(result.stderr)
        print("------------------")
        success = True
    except subprocess.CalledProcessError as e:
        print(f"Error running C++ engine (exit code {e.returncode}):")
        if e.stderr:
            print(e.stderr)
        print("--- C++ Partial Output ---")
        print(e.stdout)
        print("--------------------------")
        success = False

    # 4. Parse output markdown file
    md_path = args.output_md
    if not os.path.isabs(md_path):
        # Look in the working directory where the C++ binary runs
        exe_dir = os.path.dirname(os.path.abspath(args.exe_path))
        md_path = os.path.join(exe_dir, args.output_md)
        if not os.path.exists(md_path):
            # Also try current directory
            md_path = args.output_md

    print(f"\n[Python] Parsing output from: {md_path}")
    sequences = parse_output_md(md_path)

    if not sequences:
        print("[Python] Warning: No sequences found in output file.")
        return

    # 5. Decode and display each sequence
    print(f"\n{'='*60}")
    print(f"[Python] Found {len(sequences)} sequence(s)")
    print(f"{'='*60}\n")

    for seq_id, output_ids in sorted(sequences.items()):
        print(f"--- Sequence {seq_id} ---")
        print(f"  Generated IDs ({len(output_ids)} tokens): {output_ids}")

        decoded_text = tokenizer.decode(output_ids, skip_special_tokens=False)
        print(f"  **Response:** {decoded_text}")
        print()


if __name__ == "__main__":
    main()
