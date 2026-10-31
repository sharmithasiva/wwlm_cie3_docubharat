"""Hosted LLM client: configuration, failures, retries, rate limits and secret handling (all offline)."""

import logging

import pytest
import requests

from src.llm_provider import (
    LLMAuthError,
    LLMBadResponse,
    LLMConfig,
    LLMNotConfigured,
    LLMRateLimited,
    LLMRequestTooLarge,
    LLMServiceError,
    LLMTimeout,
    OpenAICompatibleProvider,
    create_provider,
    parse_json,
)
from tests.fakes import FAKE_KEY, FakeResponse, FakeSession, config, ok

MESSAGES = [{"role": "user", "content": "hi"}]


def _provider(session, **overrides):
    sleeps = []
    provider = OpenAICompatibleProvider(config(**overrides), session=session, sleep=sleeps.append)
    return provider, sleeps


# --------------------------------------------------------------------------- configuration


def test_missing_api_key_means_no_provider(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    cfg = LLMConfig.from_env()
    assert not cfg.configured
    assert create_provider(cfg) is None
    with pytest.raises(LLMNotConfigured):
        OpenAICompatibleProvider(cfg)


def test_groq_defaults_and_key_from_env(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", FAKE_KEY)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    cfg = LLMConfig.from_env()
    assert cfg.configured
    assert cfg.base_url == "https://api.groq.com/openai/v1"
    assert cfg.model == "openai/gpt-oss-20b"


def test_provider_none_disables_llm(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "none")
    monkeypatch.setenv("LLM_API_KEY", FAKE_KEY)
    assert create_provider(LLMConfig.from_env()) is None


def test_api_key_is_not_in_repr():
    assert FAKE_KEY not in repr(config())


def test_plain_http_endpoint_is_refused():
    with pytest.raises(LLMNotConfigured, match="https"):
        OpenAICompatibleProvider(config(base_url="http://api.example.com/v1"))
    OpenAICompatibleProvider(config(base_url="http://localhost:8000/v1"))  # local servers are fine


# --------------------------------------------------------------------------- successful requests


def test_request_shape_for_groq_json_schema():
    session = FakeSession(ok('{"a": 1}'))
    provider, _ = _provider(session)
    result = provider.complete(MESSAGES, json_schema={"type": "object"}, schema_name="x")
    call = session.calls[0]
    assert call["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert call["headers"]["Authorization"] == f"Bearer {FAKE_KEY}"
    assert call["timeout"] == 30.0
    body = call["json"]
    assert body["model"] == "openai/gpt-oss-20b"
    assert body["temperature"] == 0
    assert body["max_completion_tokens"] == 1500
    assert body["reasoning_effort"] == "low"
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert result.text == '{"a": 1}' and result.prompt_tokens == 120


def test_non_strict_model_uses_json_object_mode():
    session = FakeSession(ok("{}"))
    provider, _ = _provider(session, model="llama-3.3-70b-versatile")
    provider.complete(MESSAGES, json_schema={"type": "object"})
    body = session.calls[0]["json"]
    assert body["response_format"] == {"type": "json_object"}
    assert "reasoning_effort" not in body


# --------------------------------------------------------------------------- failures


def test_invalid_key_fails_fast_without_retry():
    session = FakeSession(FakeResponse(401, {"error": {"message": "Invalid API Key", "type": "invalid_request_error"}}))
    provider, sleeps = _provider(session)
    with pytest.raises(LLMAuthError):
        provider.complete(MESSAGES)
    assert len(session.calls) == 1 and sleeps == []


def test_rate_limit_with_short_retry_after_is_retried():
    session = FakeSession(FakeResponse(429, {"error": {"message": "rate"}}, {"retry-after": "2"}), ok("done"))
    provider, sleeps = _provider(session)
    assert provider.complete(MESSAGES).text == "done"
    assert sleeps == [2.0]


def test_rate_limit_with_long_retry_after_is_reported_immediately():
    session = FakeSession(FakeResponse(429, {"error": {"message": "rate"}}, {"retry-after": "60"}))
    provider, sleeps = _provider(session)
    with pytest.raises(LLMRateLimited) as info:
        provider.complete(MESSAGES)
    assert info.value.retry_after == 60
    assert "about 61 s" in info.value.user_message
    assert len(session.calls) == 1 and sleeps == []


def test_persistent_rate_limit_gives_up_after_bounded_retries():
    session = FakeSession(*[FakeResponse(429, {}, {"retry-after": "1"}) for _ in range(3)])
    provider, sleeps = _provider(session)
    with pytest.raises(LLMRateLimited):
        provider.complete(MESSAGES)
    assert len(session.calls) == 3 and sleeps == [1.0, 1.0]


def test_timeout_is_retried_then_reported():
    session = FakeSession(requests.Timeout(), requests.Timeout(), requests.Timeout())
    provider, sleeps = _provider(session)
    with pytest.raises(LLMTimeout):
        provider.complete(MESSAGES)
    assert len(session.calls) == 3
    assert sleeps == [1.0, 2.0]  # exponential back-off


def test_network_error_then_success():
    session = FakeSession(requests.ConnectionError("dns failure"), ok("recovered"))
    provider, _ = _provider(session)
    assert provider.complete(MESSAGES).text == "recovered"


@pytest.mark.parametrize("status", [500, 502, 503, 498])
def test_server_errors_are_retried(status):
    session = FakeSession(FakeResponse(status, {}), ok("ok"))
    provider, _ = _provider(session)
    assert provider.complete(MESSAGES).text == "ok"


def test_request_too_large():
    provider, _ = _provider(FakeSession(FakeResponse(413, {})))
    with pytest.raises(LLMRequestTooLarge):
        provider.complete(MESSAGES)


def test_other_client_error_is_not_retried():
    session = FakeSession(FakeResponse(400, {"error": {"message": "bad model"}}))
    provider, _ = _provider(session)
    with pytest.raises(LLMServiceError, match="HTTP 400"):
        provider.complete(MESSAGES)
    assert len(session.calls) == 1


@pytest.mark.parametrize(
    "response",
    [
        FakeResponse(200, text="<html>not json</html>"),
        FakeResponse(200, {"choices": []}),
        FakeResponse(200, {"choices": [{"message": {"content": ""}}]}),
        FakeResponse(200, {"choices": [{"message": {"content": None}}]}),
    ],
)
def test_malformed_responses(response):
    provider, _ = _provider(FakeSession(response))
    with pytest.raises(LLMBadResponse):
        provider.complete(MESSAGES)


def test_truncated_completion_is_rejected():
    provider, _ = _provider(FakeSession(ok('{"answer": "partial', finish_reason="length")))
    with pytest.raises(LLMBadResponse, match="cut off"):
        provider.complete(MESSAGES)


def test_api_key_never_appears_in_errors_or_logs(caplog):
    caplog.set_level(logging.DEBUG)
    outcomes = [
        FakeResponse(401, {"error": {"message": "Invalid API Key"}}),
        FakeResponse(400, {"error": {"message": "bad"}}),
        requests.ConnectionError(f"failed with header Bearer {FAKE_KEY}"),
    ]
    for outcome in outcomes:
        provider, _ = _provider(FakeSession(outcome, outcome, outcome), max_retries=0)
        with pytest.raises(Exception) as info:
            provider.complete(MESSAGES)
        assert FAKE_KEY not in str(info.value)
        assert FAKE_KEY not in getattr(info.value, "user_message", "")
    assert FAKE_KEY not in caplog.text


# --------------------------------------------------------------------------- JSON parsing


def test_parse_json_accepts_code_fence_and_rejects_garbage():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    with pytest.raises(LLMBadResponse):
        parse_json("Sure! Here is the answer: 42")
    with pytest.raises(LLMBadResponse):
        parse_json("[1, 2, 3]")
