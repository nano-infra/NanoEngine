import os
import sys

import uvicorn
from chainlit.utils import mount_chainlit
from fastapi import FastAPI
from pydantic import BaseModel

# Ensure we can import from local directory
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(repo_root)

from tools.chat_ui.client import get_client
from tools.chat_ui.config import MAX_TOKENS

app = FastAPI(title="NanoDeploy Chat API")


class ChatRequest(BaseModel):
    prompt: str
    max_tokens: int = MAX_TOKENS


@app.post("/api/chat")
def chat_api(req: ChatRequest):
    """
    Exposed API endpoint for chat.
    Takes text, tokenizes, sends to NanoDeploy server, receives tokens, detokenizes, returns text.
    """
    client = get_client()
    full_text = ""
    # Simple blocking iteration for API
    for chunk in client.generate_stream(req.prompt, max_tokens=req.max_tokens):
        full_text += chunk
    return {"response": full_text}


# Mount Chainlit
# We mount it at root
ui_path = os.path.join(current_dir, "ui.py")
mount_chainlit(app=app, target=ui_path, path="/")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NanoDeploy Chat Tool")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    args = parser.parse_args()

    print("Starting NanoDeploy Chat Tool...")
    print(f"Server will run at http://{args.host}:{args.port}")
    print(f"API Docs at http://{args.host}:{args.port}/docs")
    try:
        uvicorn.run(app, host=args.host, port=args.port)
    except SystemExit as e:
        print(f"Failed to start server: {e}")
