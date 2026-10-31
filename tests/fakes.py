"""Test doubles for the hosted LLM: no network access, no real API key."""

from __future__ import annotations

import json
from typing import Any

import requests

from src.llm_provider import LLMConfig, LLMResponse

FAKE_KEY = "gsk_test_" + "x" * 24


class FakeResponse:
    def __init__(self, status: int = 200, body: Any = None, headers: dict | None = None, text: str | None = None):
        self.status_code = status
        self._body = body
        self.headers = headers or {}
        self._text = text

    def json(self):
        if self._text is not None:
            return json.loads(self._text)
        return self._body


def ok(content: str, model: str = "openai/gpt-oss-20b", finish_reason: str = "stop") -> FakeResponse:
    return FakeResponse(
        200,
        {
            "model": model,
            "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 40, "total_tokens": 160},
        },
    )


class FakeSession:
    """Returns queued responses (or raises queued exceptions) and records every request."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls: list[dict] = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def config(**overrides) -> LLMConfig:
    values = {"provider": "groq", "api_key": FAKE_KEY, "max_retries": 2, "max_retry_wait_seconds": 5.0}
    values.update(overrides)
    return LLMConfig(**values)


class ScriptedProvider:
    """Provider stand-in for task-level tests: returns fixed JSON and records messages."""

    def __init__(self, payload: dict | str, model: str = "openai/gpt-oss-20b", **config_overrides):
        self.payload = payload if isinstance(payload, str) else json.dumps(payload)
        self.config = config(**config_overrides)
        self.messages: list[list[dict]] = []
        self.model = model

    def complete(self, messages, json_schema=None, schema_name="result"):
        self.messages.append(messages)
        return LLMResponse(text=self.payload, model=self.model)


class FailingProvider:
    def __init__(self, error: Exception):
        self.error = error
        self.config = config()

    def complete(self, messages, json_schema=None, schema_name="result"):
        raise self.error


__all__ = ["FakeResponse", "FakeSession", "ScriptedProvider", "FailingProvider", "ok", "config", "FAKE_KEY", "requests"]
