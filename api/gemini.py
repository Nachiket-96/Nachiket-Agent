import json
import logging
import os
import time
from collections.abc import Callable, Mapping
from http.server import BaseHTTPRequestHandler
from typing import Any

import requests

MAX_QUESTION_LENGTH = 500
DEFAULT_MODEL = "gemini-flash-latest"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# Upstream statuses worth retrying; network errors count as temporary too
TEMPORARY = frozenset({429, 500, 503})
MAX_ATTEMPTS = 3
# Wait before attempt 2 and attempt 3
BACKOFF_SECONDS = (1, 2)

# Outcomes of one attempt
OK = "ok"
TEMPORARY_ERROR = "temporary"
PERMANENT_ERROR = "permanent"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gemini")

Outcome = tuple[str, int | None, str | None]


def _try_once(post: Callable[..., Any], model: str, key: str, question: str, attempt: int) -> Outcome:
    # Call Gemini with the key in a header, not in the URL
    try:
        response = post(
            GEMINI_URL.format(model=model),
            headers={"x-goog-api-key": key},
            json={"contents": [{"role": "user", "parts": [{"text": question}]}]},
            timeout=30,
        )
    except requests.RequestException as exc:
        # Log only the exception class, never its message
        logger.warning("model=%s attempt=%d error=%s", model, attempt, type(exc).__name__)
        return TEMPORARY_ERROR, None, None

    # Log only the status, never the key, question or response body
    status = response.status_code
    logger.info("model=%s attempt=%d status=%d", model, attempt, status)

    if status in TEMPORARY:
        return TEMPORARY_ERROR, status, None
    if status != 200:
        return PERMANENT_ERROR, status, None

    # Return the text of the first candidate; an unexpected shape is permanent
    try:
        text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
    except (ValueError, KeyError, IndexError, TypeError):
        logger.warning("model=%s attempt=%d error=unexpected_response", model, attempt)
        return PERMANENT_ERROR, status, None
    return OK, status, text


def _try_model(
    post: Callable[..., Any],
    sleep: Callable[[float], None],
    model: str,
    key: str,
    question: str,
    attempts: int,
) -> Outcome:
    # Retry temporary errors with backoff; stop at once on success or a permanent error
    outcome: Outcome = (TEMPORARY_ERROR, None, None)
    for attempt in range(1, attempts + 1):
        if attempt > 1:
            sleep(BACKOFF_SECONDS[attempt - 2])
        outcome = _try_once(post, model, key, question, attempt)
        if outcome[0] != TEMPORARY_ERROR:
            break
    return outcome


def handle(
    method: str,
    body: Mapping[str, Any],
    env: Mapping[str, str],
    post: Callable[..., Any] = requests.post,
    sleep: Callable[[float], None] = time.sleep,
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

    # Try the main model with retries
    outcome, status, text = _try_model(post, sleep, model, key, question, MAX_ATTEMPTS)

    # Try the fallback model once if the main model stayed temporarily unavailable
    fallback = env.get("GEMINI_FALLBACK_MODEL")
    if outcome == TEMPORARY_ERROR and fallback:
        outcome, status, text = _try_model(post, sleep, fallback, key, question, 1)

    # Map the result for the browser, without Google's error text
    if outcome == OK:
        return 200, {"text": text}
    if outcome == TEMPORARY_ERROR and status == 429:
        return 429, {"error": "ai_rate_limited"}
    if outcome == TEMPORARY_ERROR:
        return 503, {"error": "ai_busy"}
    return 502, {"error": "ai_error"}


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
