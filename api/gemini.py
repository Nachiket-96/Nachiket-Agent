import json
import os
from collections.abc import Callable, Mapping
from http.server import BaseHTTPRequestHandler
from typing import Any

import requests

MAX_QUESTION_LENGTH = 500
DEFAULT_MODEL = "gemini-flash-latest"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def handle(
    method: str,
    body: Mapping[str, Any],
    env: Mapping[str, str],
    post: Callable[..., Any] = requests.post,
) -> tuple[int, dict]:
    # Only accept POST requests
    if method != "POST":
        return 405, {"error": "method_not_allowed"}

    # Read the key from the server environment, never from the browser
    key = env.get("GEMINI_API_KEY")
    if not key:
        return 500, {"error": "server_not_configured"}

    # Read the question as a string and cut it to the maximum length
    question = str(body.get("question", "")).strip()[:MAX_QUESTION_LENGTH]
    if not question:
        return 400, {"error": "empty_question"}

    # Pick the model, with a default when none is configured
    model = env.get("GEMINI_MODEL") or DEFAULT_MODEL

    # Call Gemini with the key in a header, not in the URL
    try:
        response = post(
            GEMINI_URL.format(model=model),
            headers={"x-goog-api-key": key},
            json={"contents": [{"role": "user", "parts": [{"text": question}]}]},
            timeout=30,
        )
    except requests.RequestException:
        return 502, {"error": "ai_error"}

    # Hide Google's error details from the browser
    if response.status_code != 200:
        return 502, {"error": "ai_error"}

    # Return the text of the first candidate
    try:
        text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
    except (ValueError, KeyError, IndexError, TypeError):
        return 502, {"error": "ai_error"}
    return 200, {"text": text}


class handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        # Read the JSON body; anything that is not a JSON object counts as empty
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, UnicodeDecodeError):
            body = {}
        if not isinstance(body, dict):
            body = {}

        status, reply = handle("POST", body, os.environ)
        self._send(status, reply)

    def do_GET(self) -> None:
        # Reject GET requests
        self._send(405, {"error": "method_not_allowed"})

    def _send(self, status: int, reply: dict) -> None:
        # Write the status, headers and JSON reply
        data = json.dumps(reply).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
