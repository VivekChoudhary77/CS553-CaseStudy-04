# Used Opus 5.5 with High Effort, for the tests of the Vertex AI backend (Case Study 4 extra credit).
# prompt: Write pytest tests for the Vertex AI backend with no network and no real credentials:
#   the global and regional URLs, the Bearer header and request body, credential problems turning
#   into "auth failed (...)" reasons that skip the other Vertex models, HTTP errors mapped like
#   Gemini's, credentials loaded only once, and the token never appearing in the logs.
# Case Study 4: added tests for VERTEX_THINKING_LEVEL (thinkingConfig.thinkingLevel).

import io
import json
import logging
import urllib.error

import pytest

from prompt_enhancer.backends import vertex
from prompt_enhancer.backends.base import BackendError
from prompt_enhancer.backends.vertex import GoogleCredentials, VertexBackend, build_vertex_backends, vertex_url
from prompt_enhancer.config import Settings
from prompt_enhancer.router import AllBackendsFailed, enhance

TOKEN = "ya29.fake-access-token"
REPLY = {"candidates": [{"content": {"parts": [{"text": "An improved prompt."}]}, "finishReason": "STOP"}]}


class FakeCredentials(GoogleCredentials):
    """Stands in for google-auth: returns a fixed token, or fails like a missing identity."""

    def __init__(self, error: str | None = None) -> None:
        super().__init__()
        self.error = error
        self.calls = 0

    def token(self) -> str:
        self.calls += 1
        if self.error:
            raise BackendError(self.error)
        return TOKEN


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def captured(monkeypatch):
    """Patch urlopen; record every request and answer with REPLY (or a queued HTTP error)."""
    state = {"requests": [], "errors": []}

    def fake_urlopen(request, timeout):
        state["requests"].append(request)
        if state["errors"]:
            code = state["errors"].pop(0)
            raise urllib.error.HTTPError(request.full_url, code, "error", {}, io.BytesIO(b"{}"))
        return FakeResponse(json.dumps(REPLY).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    return state


def settings(**overrides) -> Settings:
    values = {"vertex_project": "g25-test", "vertex_model": "gemini-3.5-flash-lite"}
    values.update(overrides)
    return Settings(**values)


def test_url_global_and_regional():
    assert vertex_url("p", "global", "m") == (
        "https://aiplatform.googleapis.com/v1/projects/p/locations/global/publishers/google/models/m:generateContent"
    )
    assert vertex_url("p", "us-central1", "m").startswith(
        "https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/us-central1/"
    )


def test_generate_sends_bearer_token_and_gemini_body(captured):
    backend = VertexBackend(settings(), credentials=FakeCredentials())
    assert backend.generate("system", "write about dogs", 256, 0.5) == "An improved prompt."

    request = captured["requests"][0]
    assert request.full_url.endswith("/locations/global/publishers/google/models/gemini-3.5-flash-lite:generateContent")
    assert request.get_header("Authorization") == f"Bearer {TOKEN}"
    body = json.loads(request.data)
    assert body["contents"][0]["parts"][0]["text"] == "write about dogs"
    assert body["systemInstruction"]["parts"][0]["text"] == "system"
    assert body["generationConfig"] == {
        "maxOutputTokens": 256,
        "temperature": 0.5,
        "thinkingConfig": {"thinkingLevel": "MINIMAL"},  # default
    }


@pytest.mark.parametrize("code, reason", [(403, "auth failed (401/403)"), (429, "rate limited (429)"), (404, "HTTP 404")])
def test_http_errors_map_like_gemini(captured, code, reason):
    captured["errors"].append(code)
    backend = VertexBackend(settings(), credentials=FakeCredentials())
    with pytest.raises(BackendError) as info:
        backend.generate("s", "u", 64, 0.7)
    assert info.value.reason == reason


def test_missing_credentials_skip_the_other_vertex_models(captured):
    creds = FakeCredentials(error="auth failed (no Google credentials)")
    group = [VertexBackend(settings(), m, creds) for m in ("model-a", "model-b")]
    with pytest.raises(AllBackendsFailed) as info:
        enhance("hi", "Vertex AI", "Short", 0.7, {"Vertex AI": group})
    attempts = info.value.attempts
    assert [a.model for a in attempts] == ["model-a"]  # model-b skipped: group-wide reason
    assert attempts[0].reason == "auth failed (no Google credentials)"
    assert captured["requests"] == []


def test_bad_key_json_is_an_auth_failure():
    creds = GoogleCredentials(sa_key_json="{not json")
    with pytest.raises(BackendError) as info:
        creds.token()
    assert info.value.reason == "auth failed (bad service-account key)"


def test_credentials_load_failure_is_cached(monkeypatch):
    calls = []

    def failing_load(self):
        calls.append(1)
        raise BackendError("auth failed (no Google credentials)")

    monkeypatch.setattr(GoogleCredentials, "_load", failing_load)
    creds = GoogleCredentials()
    for _ in range(3):
        with pytest.raises(BackendError):
            creds.token()
    assert len(calls) == 1


def test_valid_credentials_are_loaded_once_and_refreshed_when_expired(monkeypatch):
    class Creds:
        def __init__(self):
            self.valid, self.token, self.refreshes = False, None, 0

        def refresh(self, request):
            self.refreshes += 1
            self.valid, self.token = True, TOKEN

    loaded = []
    fake = Creds()
    monkeypatch.setattr(GoogleCredentials, "_load", lambda self: loaded.append(1) or fake)
    creds = GoogleCredentials()
    assert creds.token() == TOKEN
    assert creds.token() == TOKEN
    fake.valid = False  # token expired
    assert creds.token() == TOKEN
    assert len(loaded) == 1 and fake.refreshes == 2


def test_token_and_prompt_never_logged(captured, caplog):
    caplog.set_level(logging.DEBUG)
    backend = VertexBackend(settings(), credentials=FakeCredentials())
    enhance("my secret prompt", "Vertex AI", "Short", 0.7, {"Vertex AI": [backend]})
    assert TOKEN not in caplog.text
    assert "my secret prompt" not in caplog.text
    assert "served backend=Vertex AI" in caplog.text


def test_build_shares_credentials_from_settings():
    backends = build_vertex_backends(settings(vertex_model="a,b", vertex_sa_key_json='{"x": 1}'))
    assert [b.model_id for b in backends] == ["a", "b"]
    assert backends[0]._credentials is backends[1]._credentials
    assert isinstance(backends[0]._credentials, vertex.GoogleCredentials)


@pytest.mark.parametrize("level, expected", [("low", {"thinkingLevel": "LOW"}), ("", None), ("bogus", None)])
def test_thinking_level_setting(captured, level, expected):
    backend = VertexBackend(settings(vertex_thinking_level=level), credentials=FakeCredentials())
    backend.generate("s", "u", 64, 0.7)
    config = json.loads(captured["requests"][0].data)["generationConfig"]
    assert config.get("thinkingConfig") == expected
