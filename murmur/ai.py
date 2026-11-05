"""LLM providers for the narration director.

Three ways to get an AI in the loop, all optional:

* **Claude** through the official ``anthropic`` SDK (``pip install 'murmur[ai]'`` and ``ANTHROPIC_API_KEY``).
* **A local model** through any OpenAI-compatible server. Ollama is detected automatically; it is free and
  keeps the book on your machine.
* **Nothing.** The rule-based director in :mod:`murmur.matter` and :mod:`murmur.prosody` always works.

A provider only needs one method, ``complete(system, user) -> str``, which keeps tests trivial.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

from .config import Settings

log = logging.getLogger("murmur.ai")

DEFAULT_CLAUDE_MODEL = "claude-opus-5-5"
OLLAMA_URL = "http://127.0.0.1:11434"


class AIError(RuntimeError):
    """The AI call failed. The director falls back to rules for the affected part."""


class AIProvider(Protocol):
    name: str
    model: str

    def complete(self, system: str, user: str, max_tokens: int = 4096) -> str: ...


# ---- Claude -----------------------------------------------------------------------------------------
class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str = DEFAULT_CLAUDE_MODEL, api_key: str | None = None, effort: str = "low"):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise AIError("The anthropic package isn't installed. Run: pip install 'murmur[ai]'") from exc
        self._anthropic = anthropic
        self.model = model or DEFAULT_CLAUDE_MODEL
        self.effort = effort
        self._client = anthropic.Anthropic(api_key=api_key or None, max_retries=3, timeout=120.0)

    def complete(self, system: str, user: str, max_tokens: int = 4096) -> str:
        anthropic = self._anthropic
        kwargs: dict[str, Any] = {}
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}  # annotation is easy work: keep it fast and cheap
        try:
            response = self._client.messages.create(
                model=self.model, max_tokens=max(max_tokens, 4096) + 4000, system=system,
                messages=[{"role": "user", "content": user}], **kwargs,
            )
        except anthropic.AuthenticationError as exc:
            raise AIError("Claude rejected the API key. Check ANTHROPIC_API_KEY.") from exc
        except anthropic.RateLimitError as exc:
            raise AIError("Claude rate limit reached.") from exc
        except anthropic.APIStatusError as exc:
            raise AIError(f"Claude API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise AIError("Couldn't reach the Claude API.") from exc
        if response.stop_reason == "refusal":
            raise AIError("Claude declined to annotate this passage.")
        text = "".join(b.text for b in response.content if b.type == "text")
        if not text.strip():
            raise AIError("Claude returned an empty answer.")
        return text


# ---- OpenAI-compatible (Ollama, LM Studio, llama.cpp server, vLLM, ...) -----------------------------
class OpenAICompatProvider:
    name = "openai-compatible"

    def __init__(self, base_url: str, model: str, api_key: str = "", timeout: float = 300.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def complete(self, system: str, user: str, max_tokens: int = 4096) -> str:
        body = json.dumps({
            "model": self.model, "temperature": 0.2, "max_tokens": max_tokens, "stream": False,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last: Exception | None = None
        for attempt in range(3):
            req = urllib.request.Request(f"{self.base_url}/chat/completions", data=body, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read())
                text = data["choices"][0]["message"]["content"] or ""
                if not text.strip():
                    raise AIError("The model returned an empty answer.")
                return text
            except urllib.error.HTTPError as exc:
                last = exc
                if exc.code not in (408, 429, 500, 502, 503, 504):
                    raise AIError(f"The AI server answered {exc.code}.") from exc
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                last = exc
            except (KeyError, IndexError, json.JSONDecodeError) as exc:
                raise AIError("The AI server sent an unexpected reply.") from exc
            time.sleep(1.5 * (attempt + 1))
        raise AIError(f"Couldn't reach the AI server at {self.base_url}: {last}")


# ---- resolution -------------------------------------------------------------------------------------
@dataclass
class AIStatus:
    mode: str  # anthropic | openai | off
    available: bool
    provider: str
    model: str
    hint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "available": self.available, "provider": self.provider, "model": self.model,
                "hint": self.hint}


def _api_key(settings: Settings) -> str:
    return settings.ai_api_key or os.environ.get("ANTHROPIC_API_KEY", "")


def _ollama_models(timeout: float = 0.4) -> list[str]:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=timeout) as resp:
            return [m["name"] for m in json.loads(resp.read()).get("models", [])]
    except Exception:
        return []


def _pick_ollama_model(models: list[str]) -> str:
    preferred = ("qwen2.5", "llama3", "mistral", "gemma", "phi", "qwen")
    for p in preferred:
        for m in models:
            if m.lower().startswith(p):
                return m
    return models[0]


def ai_status(settings: Settings) -> AIStatus:
    """Describe which AI would be used, without making any paid call."""
    mode = settings.ai.lower()
    if mode in ("off", "none", "0", "false", "rules"):
        return AIStatus("off", False, "rules", "", "AI direction is switched off (MURMUR_AI=off).")
    has_sdk = importlib.util.find_spec("anthropic") is not None
    if mode in ("auto", "anthropic", "claude"):
        explicit = mode in ("anthropic", "claude")
        if _api_key(settings) or explicit:
            if not has_sdk:
                return AIStatus("anthropic", False, "claude", settings.ai_model or DEFAULT_CLAUDE_MODEL,
                                "Install the Claude SDK: pip install 'murmur[ai]'")
            return AIStatus("anthropic", True, "claude", settings.ai_model or DEFAULT_CLAUDE_MODEL)
    if mode in ("auto", "openai", "ollama", "local"):
        if settings.ai_base_url:
            return AIStatus("openai", bool(settings.ai_model), "openai-compatible", settings.ai_model,
                            "" if settings.ai_model else "Set MURMUR_AI_MODEL to the model name to use.")
        models = _ollama_models()
        if models:
            return AIStatus("openai", True, "ollama", settings.ai_model or _pick_ollama_model(models))
    hint = ("Optional: set ANTHROPIC_API_KEY (pip install 'murmur[ai]') for Claude, or run Ollama "
            "(ollama.com) for a free local model. Without either, rule-based direction is used.")
    return AIStatus("off", False, "rules", "", hint)


def resolve_provider(settings: Settings) -> AIProvider | None:
    status = ai_status(settings)
    if not status.available:
        return None
    if status.mode == "anthropic":
        return AnthropicProvider(status.model, _api_key(settings) or None, settings.ai_effort)
    base = settings.ai_base_url or f"{OLLAMA_URL}/v1"
    return OpenAICompatProvider(base, status.model, settings.ai_api_key)


# ---- tolerant JSON ----------------------------------------------------------------------------------
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def parse_json_object(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model reply (code fences, preambles and trailing prose allowed)."""
    candidates = [m.group(1) for m in _FENCE_RE.finditer(text)] + [text]
    decoder = json.JSONDecoder()
    for chunk in candidates:
        for m in re.finditer(r"\{", chunk):
            try:
                obj, _ = decoder.raw_decode(chunk[m.start():])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
    raise AIError("The AI's answer wasn't valid JSON.")
