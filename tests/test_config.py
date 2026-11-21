# Used Opus 5.5 with High Effort, for the tests of the settings loader.
# prompt: Write pytest tests for the settings: the defaults, missing keys making a backend
#   unavailable without an error, bad numbers falling back to the default, the Gemini model list,
#   and keys never appearing in repr().
# Case Study 4: removed the local model and metrics-port settings; added MAX_OUTPUT_TOKENS.
# Case Study 4: added tests for the Vertex AI settings and that its key never appears in repr().

import pytest

from prompt_enhancer.backends.gemini import GeminiBackend
from prompt_enhancer.backends.base import NotConfiguredError
from prompt_enhancer.backends.openrouter import OpenRouterBackend
from prompt_enhancer.config import load_settings

ENV_VARS = [
    "GEMINI_API_KEY", "GEMINI_MODEL", "GEMINI_BASE_URL", "GEMINI_REASONING_EFFORT",
    "OPENROUTER_API_KEY", "OPENROUTER_MODEL", "OPENROUTER_BASE_URL", "OPENROUTER_REASONING_EFFORT",
    "VERTEX_PROJECT", "VERTEX_LOCATION", "VERTEX_MODEL", "VERTEX_SA_KEY_JSON", "VERTEX_THINKING_LEVEL",
    "API_TIMEOUT_S", "MAX_OUTPUT_TOKENS", "HOST", "PORT", "LOG_LEVEL",
]


@pytest.fixture
def clean_env(monkeypatch):
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_defaults(clean_env):
    s = load_settings(env_file=None)
    assert s.host == "0.0.0.0"
    assert s.port == 7860
    assert s.api_timeout_s == 20
    assert s.max_output_tokens == 1024
    assert s.gemini_reasoning_effort == "low"
    assert s.gemini_base_url.startswith("https://generativelanguage.googleapis.com/")
    assert s.openrouter_base_url == "https://openrouter.ai/api/v1"


def test_missing_keys_make_remote_backends_unavailable(clean_env):
    s = load_settings(env_file=None)
    assert not s.gemini_configured and not s.openrouter_configured
    for backend in (GeminiBackend(s), OpenRouterBackend(s)):
        assert backend.availability() == (False, "not configured")
        with pytest.raises(NotConfiguredError):
            backend.generate("sys", "user", 16, 0.7)


def test_key_without_model_is_not_configured(clean_env):
    clean_env.setenv("GEMINI_API_KEY", "dummy")
    s = load_settings(env_file=None)
    assert GeminiBackend(s).availability() == (False, "not configured")


def test_key_and_model_configured(clean_env):
    clean_env.setenv("OPENROUTER_API_KEY", "dummy")
    clean_env.setenv("OPENROUTER_MODEL", "some/model:free")
    s = load_settings(env_file=None)
    assert OpenRouterBackend(s).availability() == (True, "ok")


def test_reasoning_effort_defaults(clean_env):
    s = load_settings(env_file=None)
    assert s.gemini_reasoning_effort == "low"
    assert s.openrouter_reasoning_effort == "none"
    assert OpenRouterBackend(s).extra_params()["extra_body"] == {"reasoning": {"effort": "none"}}


def test_blank_reasoning_effort_is_omitted(clean_env):
    clean_env.setenv("GEMINI_REASONING_EFFORT", "")
    clean_env.setenv("OPENROUTER_REASONING_EFFORT", "")
    s = load_settings(env_file=None)
    assert GeminiBackend(s)._reasoning_effort == ""
    assert "extra_body" not in OpenRouterBackend(s).extra_params()


def test_unknown_gemini_reasoning_effort_is_dropped(clean_env):
    clean_env.setenv("GEMINI_REASONING_EFFORT", "turbo")
    assert GeminiBackend(load_settings(env_file=None))._reasoning_effort == ""


def test_bad_numbers_fall_back_to_defaults(clean_env):
    clean_env.setenv("PORT", "not-a-port")
    clean_env.setenv("API_TIMEOUT_S", "")
    s = load_settings(env_file=None)
    assert s.port == 7860
    assert s.api_timeout_s == 20


@pytest.mark.parametrize("value", ["0", "-5", "lots"])
def test_bad_max_output_tokens_falls_back_to_default(clean_env, value):
    clean_env.setenv("MAX_OUTPUT_TOKENS", value)
    assert load_settings(env_file=None).max_output_tokens == 1024


def test_repr_never_contains_keys(clean_env):
    clean_env.setenv("GEMINI_API_KEY", "secret-gemini-key")
    clean_env.setenv("OPENROUTER_API_KEY", "secret-or-key")
    clean_env.setenv("VERTEX_SA_KEY_JSON", '{"private_key": "secret-vertex-key"}')
    text = repr(load_settings(env_file=None))
    assert "secret" not in text


def test_gemini_model_list(clean_env):
    from prompt_enhancer.backends.gemini import build_gemini_backends

    clean_env.setenv("GEMINI_API_KEY", "dummy")
    clean_env.setenv("GEMINI_MODEL", " gemini-3.1-flash-lite, gemini-3.5-flash ,,gemini-2.5-flash ")
    s = load_settings(env_file=None)
    assert s.gemini_models == ["gemini-3.1-flash-lite", "gemini-3.5-flash", "gemini-2.5-flash"]
    assert [b.model_id for b in build_gemini_backends(s)] == s.gemini_models


def test_gemini_without_models_is_one_unconfigured_backend(clean_env):
    from prompt_enhancer.backends.gemini import build_gemini_backends

    backends = build_gemini_backends(load_settings(env_file=None))
    assert len(backends) == 1
    assert backends[0].availability() == (False, "not configured")


def test_vertex_defaults_and_unconfigured(clean_env):
    from prompt_enhancer.backends.vertex import build_vertex_backends

    s = load_settings(env_file=None)
    assert s.vertex_location == "global"
    assert s.vertex_thinking_level == "minimal"
    assert not s.vertex_configured
    backends = build_vertex_backends(s)
    assert len(backends) == 1
    assert backends[0].availability() == (False, "not configured")
    with pytest.raises(NotConfiguredError):
        backends[0].generate("sys", "user", 16, 0.7)


def test_vertex_needs_project_and_model(clean_env):
    clean_env.setenv("VERTEX_MODEL", "gemini-3.5-flash-lite")
    assert not load_settings(env_file=None).vertex_configured
    clean_env.setenv("VERTEX_PROJECT", "my-project")
    assert load_settings(env_file=None).vertex_configured


def test_vertex_model_list(clean_env):
    from prompt_enhancer.backends.vertex import build_vertex_backends

    clean_env.setenv("VERTEX_PROJECT", "my-project")
    clean_env.setenv("VERTEX_MODEL", " gemini-3.5-flash-lite ,gemini-3.5-flash,, ")
    s = load_settings(env_file=None)
    assert s.vertex_models == ["gemini-3.5-flash-lite", "gemini-3.5-flash"]
    backends = build_vertex_backends(s)
    assert [b.model_id for b in backends] == s.vertex_models
    assert backends[0]._credentials is backends[1]._credentials  # one shared token cache
