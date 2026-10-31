"""Provider-independent access to a hosted chat LLM.

The default target is Groq's OpenAI-compatible endpoint (verified 2026-10-29 at
https://console.groq.com/docs/api-reference). Any other OpenAI-compatible
endpoint works by changing LLM_BASE_URL and LLM_MODEL.

Safety rules implemented here:
- The API key is read from the environment, sent only in the Authorization header,
  and never logged or included in error messages.
- Every request has a timeout. 429/5xx/network errors are retried a bounded number
  of times; a rate limit asking for a long wait is reported instead of blocking.
- Only HTTPS endpoints are accepted (plain HTTP is allowed for localhost only).
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol
from urllib.parse import urlparse

import requests

logger = logging.getLogger("docubharat.llm")

PROVIDER_DEFAULTS = {
    "groq": {"base_url": "https://api.groq.com/openai/v1", "model": "openai/gpt-oss-20b"},
    "openai_compatible": {"base_url": "", "model": ""},
}
# Groq models that support strict json_schema output (Groq Structured Outputs docs).
STRICT_SCHEMA_MODELS = frozenset(
    {"openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b", "meta-models/muse-spark-open"}
)
RETRYABLE_STATUS = frozenset({429, 498, 500, 502, 503})


# --------------------------------------------------------------------------- errors


class LLMError(Exception):
    """Base class. `user_message` is safe to show in the UI (no secrets, no internals)."""

    user_message = "The AI service could not answer right now."

    def __init__(self, user_message: str | None = None):
        super().__init__(user_message or self.user_message)
        if user_message:
            self.user_message = user_message


class LLMNotConfigured(LLMError):
    user_message = "No AI provider is configured. Set LLM_API_KEY in a .env file to enable AI answers."


class LLMAuthError(LLMError):
    user_message = "The AI provider rejected the API key. Check LLM_API_KEY in your .env file."


class LLMRateLimited(LLMError):
    user_message = "The AI provider's free-tier rate limit was reached. Please wait a little and try again."

    def __init__(self, retry_after: float | None = None):
        msg = self.user_message
        if retry_after:
            msg = f"{msg} (suggested wait: about {int(retry_after) + 1} s)"
        super().__init__(msg)
        self.retry_after = retry_after


class LLMTimeout(LLMError):
    user_message = "The AI provider took too long to respond."


class LLMRequestTooLarge(LLMError):
    user_message = "The request was too large for the AI provider. Try a shorter question."


class LLMServiceError(LLMError):
    user_message = "The AI provider is unavailable or returned an error."


class LLMBadResponse(LLMError):
    user_message = "The AI provider returned a response DocuBharat could not use."


# --------------------------------------------------------------------------- configuration


def _env_number(name: str, default: float, minimum: float) -> float:
    try:
        return max(float(os.getenv(name, default)), minimum)
    except ValueError:
        return default


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "groq"
    api_key: str = field(default="", repr=False)  # repr=False keeps the key out of logs and tracebacks
    base_url: str = PROVIDER_DEFAULTS["groq"]["base_url"]
    model: str = PROVIDER_DEFAULTS["groq"]["model"]
    timeout_seconds: float = 30.0
    max_retries: int = 2
    max_retry_wait_seconds: float = 10.0
    max_output_tokens: int = 1500
    # Limits on how much document text is sent per request.
    max_context_chars: int = 9000
    max_passages: int = 3

    @property
    def configured(self) -> bool:
        return self.provider != "none" and bool(self.api_key) and bool(self.base_url) and bool(self.model)

    @property
    def display_name(self) -> str:
        host = urlparse(self.base_url).hostname or self.base_url
        return f"{'Groq' if self.provider == 'groq' else host} · {self.model}"

    @classmethod
    def from_env(cls) -> "LLMConfig":
        provider = os.getenv("LLM_PROVIDER", "groq").strip().lower() or "groq"
        defaults = PROVIDER_DEFAULTS.get(provider, {"base_url": "", "model": ""})
        return cls(
            provider=provider,
            api_key=(os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY") or "").strip(),
            base_url=(os.getenv("LLM_BASE_URL") or defaults["base_url"]).strip().rstrip("/"),
            model=(os.getenv("LLM_MODEL") or defaults["model"]).strip(),
            timeout_seconds=_env_number("LLM_TIMEOUT_SECONDS", 30.0, 1.0),
            max_retries=int(_env_number("LLM_MAX_RETRIES", 2, 0)),
            max_retry_wait_seconds=_env_number("LLM_MAX_RETRY_WAIT_SECONDS", 10.0, 0.0),
            max_output_tokens=int(_env_number("LLM_MAX_OUTPUT_TOKENS", 1500, 64)),
            max_context_chars=int(_env_number("LLM_MAX_CONTEXT_CHARS", 9000, 500)),
            max_passages=int(_env_number("LLM_MAX_PASSAGES", 3, 1)),
        )


# --------------------------------------------------------------------------- provider interface


@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class LLMProvider(Protocol):
    config: LLMConfig

    def complete(
        self, messages: list[dict[str, str]], json_schema: dict[str, Any] | None = None, schema_name: str = "result"
    ) -> LLMResponse: ...


def _check_endpoint(base_url: str) -> None:
    parsed = urlparse(base_url)
    local = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    if parsed.scheme != "https" and not (parsed.scheme == "http" and local):
        raise LLMNotConfigured("LLM_BASE_URL must use https:// so the API key is never sent unencrypted.")


def _parse_retry_after(response: requests.Response) -> float | None:
    value = response.headers.get("retry-after")
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def _error_detail(response: requests.Response) -> str:
    """Provider error text for the log only (truncated; never contains our key)."""
    try:
        return str(response.json().get("error", {}).get("message", ""))[:200]
    except (ValueError, AttributeError):
        return ""


class OpenAICompatibleProvider:
    """Chat-completions client for Groq or any OpenAI-compatible endpoint."""

    def __init__(
        self,
        config: LLMConfig,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if not config.configured:
            raise LLMNotConfigured()
        _check_endpoint(config.base_url)
        self.config = config
        self._session = session or requests.Session()
        self._sleep = sleep

    def _payload(self, messages, json_schema, schema_name) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": 0,  # grounded extraction, not creative writing
            "max_completion_tokens": self.config.max_output_tokens,
        }
        if self.config.model.startswith("openai/gpt-oss"):
            payload["reasoning_effort"] = "low"  # reasoning tokens count against the rate limit
        if json_schema is not None:
            if self.config.model in STRICT_SCHEMA_MODELS:
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": schema_name, "strict": True, "schema": json_schema},
                }
            else:
                payload["response_format"] = {"type": "json_object"}
        return payload

    def complete(
        self, messages: list[dict[str, str]], json_schema: dict[str, Any] | None = None, schema_name: str = "result"
    ) -> LLMResponse:
        url = f"{self.config.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"}
        payload = self._payload(messages, json_schema, schema_name)

        last_error: LLMError = LLMServiceError()
        for attempt in range(self.config.max_retries + 1):
            started = time.perf_counter()
            try:
                response = self._session.post(url, headers=headers, json=payload, timeout=self.config.timeout_seconds)
            except requests.Timeout:
                last_error = LLMTimeout()
                logger.warning("LLM request timed out (attempt %d)", attempt + 1)
            except requests.RequestException as exc:
                last_error = LLMServiceError("Could not reach the AI provider. Check your internet connection.")
                logger.warning("LLM network error (attempt %d): %s", attempt + 1, type(exc).__name__)
            else:
                elapsed = time.perf_counter() - started
                logger.info("LLM response status=%s model=%s %.2fs", response.status_code, self.config.model, elapsed)
                if response.status_code == 200:
                    return self._parse(response)
                detail = _error_detail(response)
                if response.status_code in (401, 403):
                    raise LLMAuthError()
                if response.status_code == 413:
                    raise LLMRequestTooLarge()
                if response.status_code == 429:
                    retry_after = _parse_retry_after(response)
                    if retry_after is not None and retry_after > self.config.max_retry_wait_seconds:
                        raise LLMRateLimited(retry_after)  # do not freeze the UI for a long wait
                    last_error = LLMRateLimited(retry_after)
                elif response.status_code in RETRYABLE_STATUS:
                    last_error = LLMServiceError()
                else:
                    logger.warning("LLM request rejected status=%s: %s", response.status_code, detail)
                    raise LLMServiceError(f"The AI provider rejected the request (HTTP {response.status_code}).")
                logger.warning("LLM retryable status=%s: %s", response.status_code, detail)

            if attempt < self.config.max_retries:
                wait = min(2.0**attempt, self.config.max_retry_wait_seconds)
                if isinstance(last_error, LLMRateLimited) and last_error.retry_after:
                    wait = min(last_error.retry_after, self.config.max_retry_wait_seconds)
                self._sleep(wait)
        raise last_error

    def _parse(self, response: requests.Response) -> LLMResponse:
        try:
            body = response.json()
            choice = body["choices"][0]
            content = choice["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMBadResponse() from exc
        if not isinstance(content, str) or not content.strip():
            raise LLMBadResponse()
        if choice.get("finish_reason") == "length":
            raise LLMBadResponse("The AI response was cut off before it finished. Try a narrower question.")
        usage = body.get("usage") or {}
        return LLMResponse(
            text=content,
            model=str(body.get("model", self.config.model)),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )


def create_provider(config: LLMConfig | None = None) -> OpenAICompatibleProvider | None:
    """Return a provider, or None when no key is configured (the app then stays extractive)."""
    config = config or LLMConfig.from_env()
    if not config.configured:
        return None
    return OpenAICompatibleProvider(config)


def parse_json(text: str) -> dict[str, Any]:
    """Parse a JSON object from model output, tolerating a Markdown code fence."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned[cleaned.find("{") :] if "{" in cleaned else cleaned
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise LLMBadResponse("The AI response was not valid JSON.") from exc
    if not isinstance(data, dict):
        raise LLMBadResponse("The AI response had an unexpected structure.")
    return data
