"""BharatGen Patram document VQA via the user's private Hugging Face ZeroGPU Space.

Patram (bharatgenai/patram-7b-instruct, Apache 2.0) reads a *page image* and answers a
question about it. The model runs on the Space in patram_space/; this module renders
PDF pages to PNG, calls the Space's `/ask` endpoint with gradio_client, and turns every
failure (Space asleep, GPU quota used up, timeout, auth) into a user-safe message so
the app can fall back to its text pipeline.

Each call uses free ZeroGPU quota (5 GPU-minutes/day on a free account), so callers
should cache answers and send as few pages as possible.
"""

from __future__ import annotations

import concurrent.futures
import importlib.util
import logging
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

logger = logging.getLogger("docubharat.patram")

MODEL_LABEL = "BharatGen Patram (patram-7b-instruct, on ZeroGPU)"
MAX_QUESTION_CHARS = 1000
HTTP_TIMEOUT_SECONDS = 60.0  # per HTTP request (upload, queue join); a waking Space is slow
RETRY_WAIT_SECONDS = 20.0


class PatramUnavailable(Exception):
    """Raised with a user-safe message when Patram cannot answer."""


@dataclass(frozen=True)
class PatramConfig:
    space_id: str = ""
    token: str = field(default="", repr=False)
    enabled: bool = True
    timeout_seconds: float = 180.0  # a sleeping Space needs time to start
    max_new_tokens: int = 256
    dpi: int = 110  # A4 at 110 dpi is about 910 x 1290 px; the Space caps images at 1600 px
    max_pages: int = 2  # pages sent for document-level tasks (summary, fields)

    @property
    def configured(self) -> bool:
        return self.enabled and bool(self.space_id) and bool(self.token)

    @classmethod
    def from_env(cls) -> "PatramConfig":
        enabled = os.getenv("PATRAM_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}
        try:
            timeout = max(float(os.getenv("PATRAM_TIMEOUT_SECONDS", "180")), 10.0)
        except ValueError:
            timeout = 180.0
        try:
            max_pages = min(max(int(os.getenv("PATRAM_MAX_PAGES", "2")), 1), 5)
        except ValueError:
            max_pages = 2
        return cls(
            space_id=os.getenv("PATRAM_SPACE_ID", "").strip(),
            token=(os.getenv("HF_TOKEN") or "").strip(),
            enabled=enabled,
            timeout_seconds=timeout,
            max_pages=max_pages,
        )


@dataclass
class PatramAnswer:
    question: str
    text: str
    page: int
    generation_seconds: float | None
    wall_seconds: float
    model_label: str = MODEL_LABEL


def status(config: PatramConfig) -> tuple[bool, str]:
    """(available, reason). Never contacts the Space."""
    if not config.enabled:
        return False, "Patram is turned off (PATRAM_ENABLED=false)."
    if not config.space_id or not config.token:
        return False, "Patram is not configured. Set PATRAM_SPACE_ID and HF_TOKEN in .env (see docs/patram_space_plan.md)."
    if importlib.util.find_spec("gradio_client") is None:
        return False, "gradio_client is not installed (python -m pip install gradio_client==1.14.0)."
    return True, "ready"


def render_page_png(pdf_bytes: bytes, page_number: int, dpi: int = 110) -> bytes:
    """Render one PDF page (1-based) to PNG bytes."""
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        if not 1 <= page_number <= doc.page_count:
            raise PatramUnavailable(f"Page {page_number} does not exist in this document.")
        return doc[page_number - 1].get_pixmap(dpi=dpi).tobytes("png")


def _user_message(exc: Exception) -> str:
    """Map gradio_client / network errors to plain language (no tokens, no internals)."""
    name = type(exc).__name__
    text = str(exc).lower()
    if "quota" in text:
        return "The free daily GPU quota for Patram is used up. It resets 24 hours after first use."
    if name == "AuthenticationError" or "401" in text or "403" in text or "unauthorized" in text:
        return "The Patram Space rejected the token. Check HF_TOKEN and that it can read the private Space."
    if name in ("TooManyRequestsError", "QueueError") or "queue" in text:
        return "The Patram Space is busy right now. Please try again shortly."
    if isinstance(exc, (TimeoutError, concurrent.futures.TimeoutError)):
        return "Patram took too long to respond (the Space may still be starting). Try again in a minute."
    if name == "AppError":
        return "The Patram Space reported an error while answering."
    return ("Could not reach the Patram Space. It may still be waking up from sleep (this can take a minute or two): "
            "try again shortly, and check your internet connection.")


def _is_transient(exc: Exception) -> bool:
    """httpx network timeouts and dropped connections, typical while a sleeping Space starts up.
    The job-level TimeoutError (after waiting PATRAM_TIMEOUT_SECONDS) is deliberately not retried."""
    name = type(exc).__name__
    return name.endswith("Timeout") or name in ("ConnectError", "RemoteProtocolError", "ReadError",
                                                "ConnectionError", "WriteError")


class PatramClient:
    def __init__(self, config: PatramConfig, client_factory=None, sleep=time.sleep, retries: int = 1):
        available, reason = status(config)
        if not available:
            raise PatramUnavailable(reason)
        self.config = config
        self._client_factory = client_factory
        self._sleep = sleep
        self.retries = retries  # one retry on network timeouts (e.g. a Space waking up)
        self._client = None
        self._lock = threading.Lock()
        self.calls = 0  # Space calls made by this process (each uses GPU quota)

    def _connect(self):
        if self._client is None:
            if self._client_factory is not None:
                self._client = self._client_factory()
            else:
                from gradio_client import Client

                started = time.perf_counter()
                # gradio_client 1.14 names the token parameter hf_token. Its default HTTP timeouts are
                # short; uploads to a Space that is still starting failed with WriteTimeout/ReadTimeout.
                self._client = Client(
                    self.config.space_id,
                    hf_token=self.config.token,
                    verbose=False,
                    httpx_kwargs={"timeout": HTTP_TIMEOUT_SECONDS},
                )
                logger.info("Connected to Patram Space in %.1fs", time.perf_counter() - started)
        return self._client

    def ask_image(self, png: bytes, question: str, page: int, max_new_tokens: int | None = None) -> PatramAnswer:
        question = question.strip()[:MAX_QUESTION_CHARS]
        if not question:
            raise PatramUnavailable("Please enter a question.")
        tokens = max_new_tokens or self.config.max_new_tokens
        started = time.perf_counter()
        # gradio_client uploads from a file path; the temporary PNG is deleted right after.
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / f"page_{page}.png"
            image_path.write_bytes(png)
            for attempt in range(self.retries + 1):
                try:
                    with self._lock:
                        client = self._connect()
                        from gradio_client import handle_file

                        job = client.submit(handle_file(str(image_path)), question, tokens, api_name="/ask")
                        self.calls += 1
                    result = job.result(timeout=self.config.timeout_seconds)
                    break
                except PatramUnavailable:
                    raise
                except Exception as exc:
                    logger.warning("Patram call failed (attempt %d): %s", attempt + 1, type(exc).__name__)
                    if self._client is not None and not isinstance(exc, (TimeoutError, concurrent.futures.TimeoutError)):
                        self._client = None  # reconnect next time (e.g. after the Space restarted)
                    if attempt < self.retries and _is_transient(exc):
                        self._sleep(RETRY_WAIT_SECONDS)  # give a waking Space time to start
                        continue
                    raise PatramUnavailable(_user_message(exc)) from exc
        try:
            answer, gen_seconds = result
        except (TypeError, ValueError) as exc:
            raise PatramUnavailable("The Patram Space returned an unexpected response.") from exc
        text = str(answer or "").strip()
        if not text:
            raise PatramUnavailable("Patram returned an empty answer.")
        generation = float(gen_seconds) if isinstance(gen_seconds, (int, float)) else None
        return PatramAnswer(question, text[:3000], page, generation, time.perf_counter() - started)

    def ask_page(self, pdf_bytes: bytes, page: int, question: str, max_new_tokens: int | None = None) -> PatramAnswer:
        return self.ask_image(render_page_png(pdf_bytes, page, self.config.dpi), question, page, max_new_tokens)


def create_client(config: PatramConfig | None = None) -> tuple[PatramClient | None, str]:
    """(client, "ready") or (None, reason). Does not contact the Space yet."""
    config = config or PatramConfig.from_env()
    available, reason = status(config)
    if not available:
        return None, reason
    return PatramClient(config), reason
