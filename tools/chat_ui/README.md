# NanoDeploy Chat Tool & API

This tool provides a web-based chat interface (Chainlit) and a REST API for the NanoDeploy inference server.

## Prerequisites

- Python 3.8+
- NanoDeploy server running at `http://localhost:3000` (configurable)
- Tokenizer model at `~/qwen3-0.6b-local` (configurable)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Start the tool (includes both UI and API). You can specify host and port:

```bash
python tools/chat_ui/main.py --host 0.0.0.0 --port 8000
```

- **Chat Interface**: Open [http://localhost:8000](http://localhost:8000)
- **API Documentation**: Open [http://localhost:8000/docs](http://localhost:8000/docs)

## Configuration

You can configure the tool using environment variables:

- `MODEL_PATH`: Path to the model/tokenizer (default: `/home/majinming/qwen3-0.6b-local`)
- `SERVER_URL`: URL of the NanoDeploy inference server (default: `http://localhost:3000/chat`)

Example:

```bash
MODEL_PATH=/path/to/model python tools/chat_ui/main.py
```

## API Usage Example

```bash
curl -X POST "http://localhost:8000/api/chat" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Hello", "max_tokens": 50}'
```
