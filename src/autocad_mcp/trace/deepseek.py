"""Minimal DeepSeek chat client (text only) for the CAD semantic pass.

DeepSeek's public API exposes no vision endpoint. Instead of working around
that with an OCR/vision detour, this project inverts the responsibility: the
geometry is extracted deterministically (see :mod:`autocad_mcp.trace.vectorize`)
and the model is handed a compact **JSON digest** of that geometry to label.

The client is intentionally tiny — one endpoint, one JSON mode, no SDK — so the
project keeps working with any OpenAI-compatible endpoint and adds no heavy
dependency. Configuration:

===============================  ==========================================
``DEEPSEEK_API_KEY``             API key (required for the semantic pass)
``AUTOCAD_MCP_DEEPSEEK_MODEL``   default ``deepseek-chat``
``AUTOCAD_MCP_DEEPSEEK_BASE_URL`` default ``https://api.deepseek.com``
===============================  ==========================================

Everything here is optional: without a key the pipeline still produces a DXF,
labelled by the deterministic classifier. That is a design requirement — the
tool has to be useful with nothing but ``pip install``.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import structlog

log = structlog.get_logger()

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"


class LLMError(RuntimeError):
    """Any failure of the optional semantic pass."""


def _extract_json(text: str) -> dict:
    """Parse a JSON object out of a model reply, tolerating code fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise LLMError("model reply contained no JSON object")
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise LLMError(f"model reply was not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise LLMError("model reply was not a JSON object")
    return parsed


class DeepSeekClient:
    """Thin OpenAI-compatible chat client.

    Constructing it never fails; call :attr:`available` (or just attempt a call)
    to find out whether a key is configured.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.api_key = (
            api_key
            or os.environ.get("DEEPSEEK_API_KEY")
            or os.environ.get("AUTOCAD_MCP_DEEPSEEK_API_KEY")
            or ""
        ).strip()
        self.base_url = (
            base_url or os.environ.get("AUTOCAD_MCP_DEEPSEEK_BASE_URL") or DEFAULT_BASE_URL
        ).rstrip("/")
        self.model = (model or os.environ.get("AUTOCAD_MCP_DEEPSEEK_MODEL") or DEFAULT_MODEL).strip()
        self.timeout = float(os.environ.get("AUTOCAD_MCP_DEEPSEEK_TIMEOUT", timeout))

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def chat(self, system: str, user: str, json_mode: bool = False, temperature: float = 0.2) -> str:
        """One chat completion; raises :class:`LLMError` on any failure."""
        if not self.available:
            raise LLMError(
                "no DeepSeek API key configured — set DEEPSEEK_API_KEY to enable the semantic pass"
            )
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - httpx ships with mcp[cli]
            raise LLMError("httpx is required for the semantic pass") from exc

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "stream": False,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
        except Exception as exc:  # network, HTTP, decode
            raise LLMError(f"DeepSeek request failed: {exc}") from exc

        try:
            return str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected DeepSeek response shape: {exc}") from exc

    def chat_json(self, system: str, user: str, temperature: float = 0.2) -> dict:
        """Chat completion constrained to a JSON object (one retry on bad JSON)."""
        raw = self.chat(system, user, json_mode=True, temperature=temperature)
        try:
            return _extract_json(raw)
        except LLMError as first_error:
            log.warning("semantic_json_retry", error=str(first_error))
            strict = (
                f"{user}\n\nReturn ONLY a single JSON object. "
                "No prose, no markdown fences, no comments."
            )
            raw = self.chat(system, strict, json_mode=True, temperature=0.0)
            return _extract_json(raw)
