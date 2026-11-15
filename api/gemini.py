import json
import logging
import os
import random
import time
from collections.abc import Callable, Mapping
from http.server import BaseHTTPRequestHandler
from typing import Any

import requests

MAX_QUESTION_LENGTH = 500
DEFAULT_MODEL = "gemini-flash-latest"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
TOKENINFO_URL = "https://oauth2.googleapis.com/tokeninfo"
TOKENINFO_TIMEOUT = (3, 5)
REQUIRED_ENV = ("GEMINI_API_KEY", "GOOGLE_CLIENT_ID", "ALLOWED_EMAIL")

# Upstream statuses worth retrying; network errors count as temporary too
TEMPORARY = frozenset({429, 500, 503})
MAX_ATTEMPTS = 3
# Wait before attempt 2 and attempt 3, each spread by +/- JITTER
BACKOFF_SECONDS = (1, 2)
JITTER = 0.2

# Time limits in seconds. A slow read can overrun the deadline by up to
# READ_TIMEOUT, so maxDuration in vercel.json must exceed TOTAL_DEADLINE + READ_TIMEOUT
TOTAL_DEADLINE = 20
CONNECT_TIMEOUT = 3
READ_TIMEOUT = 12
# Skip an attempt when less time than this is left
MIN_ATTEMPT_TIME = 2

# Outcomes of one attempt
OK = "ok"
TEMPORARY_ERROR = "temporary"
PERMANENT_ERROR = "permanent"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gemini")

Outcome = tuple[str, int | None, str | None]
Clock = Callable[[], float]


class DeadlineExceeded(requests.Timeout):
    """The total time budget ran out while reading the response."""


def verify_user(
    auth_header: str | None,
    env: Mapping[str, str],
    get: Callable[..., Any] = requests.get,
) -> tuple[bool, int, str]:
    # Stops anonymous callers: no bearer token, no access
    scheme, _, token = (auth_header or "").partition(" ")
    token = token.strip()
    if scheme.lower() != "bearer" or not token:
        return False, 401, "not_signed_in"

    # Stops forged tokens: only Google can say whether a token is real
    try:
        response = get(TOKENINFO_URL, params={"access_token": token}, timeout=TOKENINFO_TIMEOUT)
    except requests.RequestException:
        return False, 502, "google_unreachable"

    # Stops expired or revoked tokens
    if response.status_code != 200:
        return False, 401, "token_expired"
    try:
        info = response.json()
    except ValueError:
        return False, 502, "google_unreachable"
    if not isinstance(info, dict):
        return False, 502, "google_unreachable"

    # Stops tokens issued to another app from being replayed here
    if info.get("aud") != env.get("GOOGLE_CLIENT_ID"):
        return False, 403, "wrong_app"

    # Stops every Google account except the verified owner
    email = str(info.get("email") or "")
    verified = info.get("email_verified") in (True, "true")
    if not email or not verified or email.lower() != env.get("ALLOWED_EMAIL", "").strip().lower():
        return False, 403, "not_allowed"

    return True, 200, ""


def _read_body(response: Any, deadline: float, clock: Clock) -> bytes:
    # Read in chunks so a slow trickle cannot run past the deadline
    chunks = []
    for chunk in response.iter_content(chunk_size=8192):
        if clock() > deadline:
            raise DeadlineExceeded()
        chunks.append(chunk)
    return b"".join(chunks)


def _try_once(
    post: Callable[..., Any],
    clock: Clock,
    deadline: float,
    model: str,
    key: str,
    question: str,
    attempt: int,
) -> Outcome:
    # Cap both timeouts by the time left, since requests applies them per phase, not in total
    remaining = deadline - clock()
    timeout = (min(CONNECT_TIMEOUT, remaining), min(READ_TIMEOUT, remaining))

    # Call Gemini with the key in a header, not in the URL
    try:
        response = post(
            GEMINI_URL.format(model=model),
            headers={"x-goog-api-key": key},
            json={"contents": [{"role": "user", "parts": [{"text": question}]}]},
            timeout=timeout,
            stream=True,
        )
        try:
            status = response.status_code
            # Only a success needs the body; error bodies are never read
            raw = _read_body(response, deadline, clock) if status == 200 else b""
        finally:
            response.close()
    except requests.RequestException as exc:
        # Log only the exception class, never its message
        logger.warning("model=%s attempt=%d error=%s", model, attempt, type(exc).__name__)
        return TEMPORARY_ERROR, None, None

    # Log only the status, never the key, question or response body
    logger.info("model=%s attempt=%d status=%d", model, attempt, status)

    if status in TEMPORARY:
        return TEMPORARY_ERROR, status, None
    if status != 200:
        return PERMANENT_ERROR, status, None

    # Return the text of the first candidate; an unexpected shape is permanent
    try:
        text = json.loads(raw)["candidates"][0]["content"]["parts"][0]["text"]
    except (ValueError, KeyError, IndexError, TypeError):
        logger.warning("model=%s attempt=%d error=unexpected_response", model, attempt)
        return PERMANENT_ERROR, status, None
    return OK, status, text


def _try_model(
    post: Callable[..., Any],
    sleep: Callable[[float], None],
    clock: Clock,
    deadline: float,
    model: str,
    key: str,
    question: str,
    attempts: int,
) -> Outcome:
    # Retry temporary errors with backoff; stop at once on success or a permanent error
    outcome: Outcome = (TEMPORARY_ERROR, None, None)
    for attempt in range(1, attempts + 1):
        # Backoff with jitter, so many clients do not retry at the same moment
        wait = 0.0
        if attempt > 1:
            wait = BACKOFF_SECONDS[attempt - 2] * random.uniform(1 - JITTER, 1 + JITTER)

        # Stop when the deadline leaves no room for the wait and a useful attempt
        if deadline - clock() < wait + MIN_ATTEMPT_TIME:
            logger.warning("model=%s attempt=%d skipped=deadline", model, attempt)
            break

        if wait:
            sleep(wait)
        outcome = _try_once(post, clock, deadline, model, key, question, attempt)
        if outcome[0] != TEMPORARY_ERROR:
            break
    return outcome


def handle(
    method: str,
    auth_header: str | None,
    body: Mapping[str, Any],
    env: Mapping[str, str],
    post: Callable[..., Any] = requests.post,
    get: Callable[..., Any] = requests.get,
    sleep: Callable[[float], None] = time.sleep,
    clock: Clock = time.monotonic,
) -> tuple[int, dict]:
    # Only accept POST requests
    if method != "POST":
        return 405, {"error": "method_not_allowed"}

    # Read the key and the owner settings from the server environment, never from the browser
    if not all(env.get(name) for name in REQUIRED_ENV):
        return 500, {"error": "server_not_configured"}
    key = env["GEMINI_API_KEY"]

    # One time budget covers the user check, every attempt, wait and the fallback
    deadline = clock() + TOTAL_DEADLINE

    # Check the user before anything reaches the model; never log the token or email
    allowed, status, reason = verify_user(auth_header, env, get=get)
    if not allowed:
        logger.warning("auth rejected: %s", reason)
        return status, {"error": reason}

    # Read the question as a string and cut it to the maximum length
    question = str(body.get("question", "")).strip()[:MAX_QUESTION_LENGTH]
    if not question:
        return 400, {"error": "empty_question"}

    # Pick the model, with a default when none is configured
    model = env.get("GEMINI_MODEL") or DEFAULT_MODEL

    # Try the main model with retries
    outcome, status, text = _try_model(post, sleep, clock, deadline, model, key, question, MAX_ATTEMPTS)

    # Try the fallback model once if the main model stayed temporarily unavailable
    fallback = env.get("GEMINI_FALLBACK_MODEL")
    if outcome == TEMPORARY_ERROR and fallback:
        outcome, status, text = _try_model(post, sleep, clock, deadline, fallback, key, question, 1)

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

        status, reply = handle("POST", self.headers.get("Authorization"), body, os.environ)
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
