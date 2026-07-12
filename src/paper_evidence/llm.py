"""Provider-agnostic LLM wrapper (Gemini / DeepSeek / Claude), config-free.

Selection comes from the environment, not a config file:
  PAPER_EVIDENCE_PROVIDER   gemini | deepseek | anthropic   (default: gemini)
  PAPER_EVIDENCE_MODEL      optional model override for that provider
Keys are read from the environment or a `.env` in the current directory. SDKs are
imported lazily, so the verification core (`quote_gate`) needs neither key nor SDK; only
card extraction / faithfulness judging call an LLM.

The judge and card-extractor are meant to be *different* model families (see
quote_gate.make_judge): a cross-family judge is a stronger check than the model that
wrote the claim grading its own work.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Protocol

_KEY_ENV = {
    "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_API_KEY"],
    "deepseek": ["DEEPSEEK_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
}
_DEFAULT_PROVIDER = "gemini"


class LLM(Protocol):
    def complete(self, system: str, prompt: str, **kw: Any) -> str: ...
    def complete_json(self, system: str, prompt: str, **kw: Any) -> Any: ...


def load_dotenv(path: Path | None = None) -> None:
    """Populate os.environ from a `.env` in `path` (or cwd); never overrides existing vars."""
    env_path = (path or Path.cwd()) / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_JSON_BLOCK = re.compile(r"\{.*\}|\[.*\]", re.S)


def extract_json(text: str) -> Any:
    """Best-effort JSON parse: whole string first, then the first {...}/[...] block."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_BLOCK.search(text)
        if not m:
            raise
        return json.loads(m.group(0))


def _resolve_key(provider: str) -> str | None:
    load_dotenv()
    for name in _KEY_ENV.get(provider, []):
        if os.environ.get(name):
            return os.environ[name]
    return None


# --------------------------------------------------------------------------- #
class GeminiClient:
    provider = "gemini"

    def __init__(self, model: str = "gemini-2.5-flash", max_tokens: int = 8192) -> None:
        self.model, self.max_tokens, self._client = model, max_tokens, None

    def _ensure(self):
        if self._client is None:
            from google import genai  # lazy
            key = _resolve_key("gemini")
            if not key:
                raise RuntimeError("Gemini key not set (GEMINI_API_KEY / GOOGLE_API_KEY).")
            self._client = genai.Client(api_key=key)
        return self._client

    def _generate(self, system: str, prompt: str, json_mode: bool, **kw: Any) -> str:
        from google.genai import types
        cfg = types.GenerateContentConfig(
            system_instruction=system, max_output_tokens=kw.get("max_tokens", self.max_tokens),
            temperature=kw.get("temperature", 0.7),
            response_mime_type="application/json" if json_mode else "text/plain")
        resp = self._ensure().models.generate_content(
            model=kw.get("model", self.model), contents=prompt, config=cfg)
        return resp.text or ""

    def complete(self, system: str, prompt: str, **kw: Any) -> str:
        return self._generate(system, prompt, json_mode=False, **kw)

    def complete_json(self, system: str, prompt: str, **kw: Any) -> Any:
        raw = self._generate(system, prompt, json_mode=True, **kw)
        try:
            return extract_json(raw)
        except Exception:
            repair = f"This was not valid JSON:\n{raw}\n\nReturn corrected JSON only."
            return extract_json(self._generate(system, repair, json_mode=True, **kw))


class DeepSeekClient:
    provider = "deepseek"
    BASE_URL = "https://api.deepseek.com"

    def __init__(self, model: str = "deepseek-chat", max_tokens: int = 8192) -> None:
        self.model, self.max_tokens, self._client = model, max_tokens, None

    def _ensure(self):
        if self._client is None:
            from openai import OpenAI  # lazy; DeepSeek speaks the OpenAI API
            key = _resolve_key("deepseek")
            if not key:
                raise RuntimeError("DEEPSEEK_API_KEY not set.")
            self._client = OpenAI(api_key=key, base_url=self.BASE_URL)
        return self._client

    def _is_reasoner(self, model: str) -> bool:
        return "reasoner" in model

    def complete(self, system: str, prompt: str, **kw: Any) -> str:
        model = kw.get("model", self.model)
        params: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt}],
            "max_tokens": kw.get("max_tokens", self.max_tokens)}
        if not self._is_reasoner(model):
            params["temperature"] = kw.get("temperature", 0.7)
        if kw.get("response_format"):
            params["response_format"] = kw["response_format"]
        return self._ensure().chat.completions.create(**params).choices[0].message.content or ""

    def complete_json(self, system: str, prompt: str, **kw: Any) -> Any:
        model = kw.get("model", self.model)
        sys_json = system + "\n\nRespond in valid JSON only. No prose, no markdown fences."
        call_kw = dict(kw)
        if not self._is_reasoner(model):
            call_kw["response_format"] = {"type": "json_object"}
        raw = self.complete(sys_json, prompt, **call_kw)
        try:
            return extract_json(raw)
        except Exception:
            repair = f"This was not valid JSON:\n{raw}\n\nReturn corrected JSON only."
            return extract_json(self.complete(sys_json, repair, **call_kw))


class ClaudeClient:
    provider = "anthropic"

    def __init__(self, model: str = "claude-sonnet-5", max_tokens: int = 4096) -> None:
        self.model, self.max_tokens, self._client = model, max_tokens, None

    def _ensure(self):
        if self._client is None:
            import anthropic  # lazy
            key = _resolve_key("anthropic")
            if not key:
                raise RuntimeError("ANTHROPIC_API_KEY not set.")
            self._client = anthropic.Anthropic(api_key=key)
        return self._client

    def complete(self, system: str, prompt: str, **kw: Any) -> str:
        msg = self._ensure().messages.create(
            model=kw.get("model", self.model), max_tokens=kw.get("max_tokens", self.max_tokens),
            system=system, messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in msg.content if b.type == "text")

    def complete_json(self, system: str, prompt: str, **kw: Any) -> Any:
        sys_json = system + "\n\nRespond with ONLY valid JSON. No prose, no markdown fences."
        raw = self.complete(sys_json, prompt, **kw)
        try:
            return extract_json(raw)
        except Exception:
            repair = f"This was not valid JSON:\n{raw}\n\nReturn corrected JSON only."
            return extract_json(self.complete(sys_json, repair, **kw))


_PROVIDERS = {"gemini": GeminiClient, "deepseek": DeepSeekClient, "anthropic": ClaudeClient}


def _configured_provider() -> str:
    load_dotenv()
    return os.environ.get("PAPER_EVIDENCE_PROVIDER", _DEFAULT_PROVIDER).strip().lower()


def get_client(provider: str | None = None, model: str | None = None,
               max_tokens: int | None = None) -> LLM:
    """Construct an LLM client. `provider` defaults to $PAPER_EVIDENCE_PROVIDER (gemini).

    $PAPER_EVIDENCE_MODEL applies ONLY to the configured provider — a different provider
    (e.g. a cross-family judge) without an explicit model uses that provider's own default,
    never the configured provider's model name (which the other API would reject).
    """
    cfg_provider = _configured_provider()
    provider = (provider or cfg_provider).strip().lower()
    cls = _PROVIDERS.get(provider)
    if cls is None:
        raise ValueError(f"Unknown LLM provider: {provider}")
    kwargs: dict[str, Any] = {}
    cfg_model = os.environ.get("PAPER_EVIDENCE_MODEL") if provider == cfg_provider else None
    if model or cfg_model:
        kwargs["model"] = model or cfg_model
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    return cls(**kwargs)


def api_key_available(provider: str | None = None) -> bool:
    return _resolve_key(provider or _configured_provider()) is not None
