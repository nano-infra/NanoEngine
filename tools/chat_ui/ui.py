import os
import sys

import chainlit as cl

# Ensure we can import from tools.chat_ui
# Assuming run from root or via main.py which sets path
# If run standalone via chainlit CLI:
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(current_dir))
if repo_root not in sys.path:
    sys.path.append(repo_root)

from tools.chat_ui.client import get_client


@cl.on_chat_start
async def start():
    # Eagerly load client (tokenizer)
    msg = cl.Message(content="Initializing tokenizer... please wait.")
    await msg.send()

    # Run tokenizer loading in thread to avoid blocking
    await cl.make_async(get_client)()

    msg.content = "Connected to NanoDeploy. Type your message!"
    await msg.update()


@cl.on_message
async def main(message: cl.Message):
    client = get_client()
    msg = cl.Message(content="")
    await msg.send()

    # Stream tokens
    # client.generate_stream is a sync generator.
    # We iterate it. While this blocks the event loop for the duration of the loop body (between awaits),
    # await msg.stream_token yields control.
    # However, retrieval of the next chunk (next(iterator)) is blocking IO (requests).
    # Properly, we should run the generator in a thread.

    # Helper to iterate sync generator in async
    async def run_sync_generator():
        buffer = ""
        in_thinking = False
        thinking_step = None

        # Tags to look for
        START_TAG = "<think>"
        END_TAG = "</think>"

        for token_text in client.generate_stream(message.content):
            buffer += token_text

            while True:
                if in_thinking:
                    # Look for end tag
                    end_idx = buffer.find(END_TAG)
                    if end_idx != -1:
                        # Found end of thinking
                        print(f"[UI] </think> tag detected! Closing Thinking block.")
                        content = buffer[:end_idx]
                        if thinking_step:
                            await thinking_step.stream_token(content)
                            await thinking_step.update()
                            thinking_step = None

                        buffer = buffer[end_idx + len(END_TAG) :]
                        in_thinking = False
                        # Continue loop to process remaining buffer as normal text
                    else:
                        # No end tag yet.
                        # Stream everything except safe suffix to avoid splitting tag
                        # Safe suffix length = len(END_TAG) - 1
                        safe_len = len(END_TAG) - 1
                        if len(buffer) > safe_len:
                            to_stream = buffer[:-safe_len]
                            if thinking_step:
                                await thinking_step.stream_token(to_stream)
                            buffer = buffer[-safe_len:]
                        break
                else:
                    # Look for start tag
                    start_idx = buffer.find(START_TAG)
                    if start_idx != -1:
                        # Found start of thinking
                        print(f"[UI] <think> tag detected! Starting Thinking block.")
                        content = buffer[:start_idx]
                        await msg.stream_token(content)

                        buffer = buffer[start_idx + len(START_TAG) :]
                        in_thinking = True
                        thinking_step = cl.Step(name="Thinking")
                        await thinking_step.send()
                        # Continue loop to process remaining buffer as thinking text
                    else:
                        # No start tag. Stream safe part
                        safe_len = len(START_TAG) - 1
                        if len(buffer) > safe_len:
                            to_stream = buffer[:-safe_len]
                            await msg.stream_token(to_stream)
                            buffer = buffer[-safe_len:]
                        break

        # Flush remaining buffer
        if buffer:
            if in_thinking and thinking_step:
                await thinking_step.stream_token(buffer)
                await thinking_step.update()
            else:
                await msg.stream_token(buffer)

    await run_sync_generator()
    await msg.update()
