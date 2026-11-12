"""A fake OpenAI-compatible endpoint for trying the pipeline without an API key.

It answers every Stage-1 prompt with a pseudo-random label (deterministic per prompt).
The resulting scores are meaningless. They only show that data loading, prompting,
parsing, and metrics all work on your machine.

    python examples/mock_llm_server.py            # listens on http://127.0.0.1:8765/v1
"""

import hashlib
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

LABELS = ["valid", "minor_issue", "fatal_violation"]


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        text = json.dumps(body.get("messages", []), sort_keys=True)
        label = LABELS[int(hashlib.sha256(text.encode()).hexdigest(), 16) % 3]
        answer = json.dumps({"label": label, "reasoning": "mock answer, not a real model"})
        payload = json.dumps({"choices": [{"message": {"role": "assistant", "content": answer}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    print("Mock LLM listening on http://127.0.0.1:8765/v1  (Ctrl+C to stop)")
    HTTPServer(("127.0.0.1", 8765), Handler).serve_forever()
