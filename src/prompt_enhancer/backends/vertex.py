# Used Opus 5.5 with High Effort, for the Vertex AI backend (Case Study 4 extra credit).
# prompt: Add a Vertex AI backend that calls Gemini models through Vertex's generateContent REST
#   API with the standard library, reusing the Gemini request body, response parser and error
#   mapping. Authenticate with a Google identity: Application Default Credentials on Cloud Run
#   (no key), or a service-account key passed as JSON in VERTEX_SA_KEY_JSON on Azure and AWS.
#   Load credentials once, refresh tokens when they expire, turn credential problems into
#   "auth failed (...)" reasons, never log tokens or keys, and build one backend per model in
#   VERTEX_MODEL.
# Case Study 4: send thinkingConfig.thinkingLevel (VERTEX_THINKING_LEVEL, default minimal); without it
#   gemini-3.5-flash spent 247 of 256 output tokens thinking and returned a cut-off answer.

"""Gemini models on Vertex AI (Google Cloud), via the generateContent REST API.

The request and response bodies are the same as the Gemini Developer API, so this module reuses
`build_request_body`, `parse_response` and `post_json` from the Gemini backend. Only the URL and
the credentials differ: Vertex needs an OAuth access token for a Google identity, not an API key.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any

from prompt_enhancer.backends.base import Backend, BackendError, EmptyResponseError, NotConfiguredError
from prompt_enhancer.backends.gemini import build_request_body, parse_response, post_json
from prompt_enhancer.config import Settings
from prompt_enhancer.prompts import clean_output

log = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

# Gemini 3.x on Vertex controls thinking with a level (thinkingBudget is deprecated there).
THINKING_LEVELS = ("minimal", "low", "medium", "high")


def vertex_url(project: str, location: str, model: str) -> str:
    """generateContent URL; the `global` location uses the host without a region prefix."""
    host = "aiplatform.googleapis.com" if location == "global" else f"{location}-aiplatform.googleapis.com"
    return (
        f"https://{host}/v1/projects/{project}/locations/{location}"
        f"/publishers/google/models/{model}:generateContent"
    )


class GoogleCredentials:
    """Google OAuth credentials, loaded once and refreshed when the token expires.

    With a key JSON (Azure/AWS) a service-account credential is built from it; without one,
    Application Default Credentials are used (the metadata server on Cloud Run, or
    GOOGLE_APPLICATION_CREDENTIALS locally). A failed load is remembered, so a cloud without
    credentials doesn't probe for them on every request.
    """

    def __init__(self, sa_key_json: str = "") -> None:
        self._sa_key_json = sa_key_json
        self._lock = threading.Lock()
        self._creds: Any = None
        self._load_error: str | None = None

    def _load(self) -> Any:
        # Imported here so the app starts (and the other backends work) even if google-auth is broken.
        import google.auth
        from google.auth import exceptions
        from google.oauth2 import service_account

        if self._sa_key_json:
            try:
                info = json.loads(self._sa_key_json)
                return service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
            except (ValueError, KeyError, TypeError):
                raise BackendError("auth failed (bad service-account key)") from None
        try:
            creds, _project = google.auth.default(scopes=SCOPES)
            return creds
        except exceptions.DefaultCredentialsError:
            raise BackendError("auth failed (no Google credentials)") from None

    def token(self) -> str:
        """A valid access token, or BackendError("auth failed (...)")."""
        from google.auth import exceptions
        from google.auth.transport.requests import Request

        with self._lock:
            if self._load_error:
                raise BackendError(self._load_error)
            if self._creds is None:
                try:
                    self._creds = self._load()
                except BackendError as exc:
                    self._load_error = exc.reason
                    log.warning("Vertex AI credentials unavailable: %s", exc.reason)
                    raise
                log.info("Vertex AI credentials loaded (%s)", "service-account key" if self._sa_key_json else "ADC")
            if not self._creds.valid:
                try:
                    self._creds.refresh(Request())
                except exceptions.RefreshError:
                    raise BackendError("auth failed (token refresh)") from None
                except exceptions.TransportError:
                    raise BackendError("connection error (token refresh)") from None
            return self._creds.token


def build_vertex_backends(settings: Settings) -> list[VertexBackend]:
    """One backend per configured model, in VERTEX_MODEL order, sharing one set of credentials."""
    credentials = GoogleCredentials(settings.vertex_sa_key_json)
    models = settings.vertex_models or [""]
    return [VertexBackend(settings, m, credentials) for m in models]


class VertexBackend(Backend):
    name = "Vertex AI"

    def __init__(
        self, settings: Settings, model_id: str | None = None, credentials: GoogleCredentials | None = None
    ) -> None:
        if model_id is None:
            model_id = next(iter(settings.vertex_models), "")
        self.model_id = model_id
        self._project = settings.vertex_project
        self._location = settings.vertex_location
        self._timeout_s = settings.api_timeout_s
        self._credentials = credentials or GoogleCredentials(settings.vertex_sa_key_json)
        level = settings.vertex_thinking_level.lower()
        if level and level not in THINKING_LEVELS:
            log.warning(
                "Unknown VERTEX_THINKING_LEVEL %r (expected one of %s); omitting it", level, ", ".join(THINKING_LEVELS)
            )
            level = ""
        self._thinking_level = level

    def availability(self) -> tuple[bool, str]:
        if not (self._project and self.model_id):
            return False, "not configured"
        return True, "ok"

    def generate(
        self, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float
    ) -> str:
        if not self.availability()[0]:
            raise NotConfiguredError()
        # Thinking tokens count against maxOutputTokens, so keep thinking low (default: minimal).
        body = build_request_body(system_prompt, user_prompt, max_tokens, temperature, reasoning_effort="")
        if self._thinking_level:
            body["generationConfig"]["thinkingConfig"] = {"thinkingLevel": self._thinking_level.upper()}
        token = self._credentials.token()
        data = post_json(
            vertex_url(self._project, self._location, self.model_id),
            body,
            {"Authorization": f"Bearer {token}"},
            self._timeout_s,
        )
        text, finish = parse_response(data)
        if finish == "MAX_TOKENS":
            log.warning("Vertex AI output hit maxOutputTokens=%d; it may be cut off", max_tokens)
        text = clean_output(text)
        if not text:
            raise EmptyResponseError()
        return text
