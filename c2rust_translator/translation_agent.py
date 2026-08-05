from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import requests
from google import genai
from google.genai import types as genai_types
from openai import OpenAI


LOGID_PATH = Path(__file__).resolve().parent.parent / ".llm_logid"
DEFAULT_RESPONSES_API_URL = "https://api.openai.com/v1/responses"
DEFAULT_RESPONSES_MODEL = "gpt-5.4-mini-2026-03-17"
DEFAULT_MAX_OUTPUT_TOKENS = 128000
DEFAULT_CHAT_EXTRA_BODY = {"enable_thinking": False}
_DEFAULT_CHAT_EXTRA_BODY_SENTINEL = object()


def load_logid() -> str:
    """Read the fixed X-TT-LOGID from .llm_logid at repo root; empty if missing."""
    if not LOGID_PATH.is_file():
        return ""
    return LOGID_PATH.read_text(encoding="utf-8").strip()


@dataclass(frozen=True)
class Usage:
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]


@dataclass(frozen=True)
class TranslationResult:
    code: str
    elapsed_seconds: float
    usage: Usage
    raw_text: str


class Agent:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "",
        api_url: str = "",
        model: str,
        protocol: str = "openai-chat",
        temperature: float = 1.0,
        retries: int = 3,
        retry_initial_delay: int = 4,
        retry_max_delay: int = 60,
        min_request_interval: float = 0.0,
        rate_limit_retry_delay: float = 60.0,
        max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        reasoning_effort: str = "medium",
        include: Optional[list[str]] = None,
        chat_extra_body: Any = _DEFAULT_CHAT_EXTRA_BODY_SENTINEL,
        chat_token_parameter: str = "max_tokens",
        google_api_version: str = "",
        logid: str = "",
        session_id: str = "",
        http_client: Any = None,
        google_client: Any = None,
        timeout: int = 300,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.api_url = api_url
        self.model = model
        self.protocol = protocol
        self.temperature = temperature
        self.retries = retries
        self.retry_initial_delay = retry_initial_delay
        self.retry_max_delay = retry_max_delay
        self.min_request_interval = min_request_interval
        self.rate_limit_retry_delay = rate_limit_retry_delay
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.include = include if include is not None else ["reasoning.encrypted_content"]
        self.chat_extra_body = (
            dict(DEFAULT_CHAT_EXTRA_BODY)
            if chat_extra_body is _DEFAULT_CHAT_EXTRA_BODY_SENTINEL
            else chat_extra_body
        )
        self.chat_token_parameter = chat_token_parameter
        self.google_api_version = google_api_version
        self.logid = logid
        self.session_id = session_id
        self.timeout = timeout
        self.http_client = http_client or requests.Session()
        self.client = None
        self._last_request_started_at: float | None = None
        if self.protocol == "openai-chat":
            self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)
        if self.protocol == "google-genai":
            self.client = google_client or genai.Client(
                api_key=api_key,
                http_options=genai_types.HttpOptions(
                    base_url=api_url,
                    api_version=google_api_version or None,
                    timeout=timeout * 1000,
                ),
            )

    def translate(self, system: str, prompt: str) -> TranslationResult:
        if self.protocol == "openai-responses":
            return self._translate_responses(system, prompt)
        if self.protocol == "google-genai":
            return self._translate_google_genai(system, prompt)
        return self._translate_chat(system, prompt)

    def _translate_chat(self, system: str, prompt: str) -> TranslationResult:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

        for attempt in range(self.retries):
            try:
                started_at = self._pace_request()
                request = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "extra_headers": {"X-TT-LOGID": self.logid},
                    "timeout": self.timeout,
                }
                if self.chat_token_parameter == "max_completion_tokens":
                    request["max_completion_tokens"] = self.max_tokens
                else:
                    request["max_tokens"] = self.max_tokens
                if self.chat_extra_body:
                    request["extra_body"] = self.chat_extra_body
                response = self.client.chat.completions.create(**request)
                raw_text = response.choices[0].message.content or ""
                usage = self._usage(getattr(response, "usage", None))
                return TranslationResult(
                    code=self._extract_code(raw_text),
                    elapsed_seconds=time.time() - started_at,
                    usage=usage,
                    raw_text=raw_text,
                )
            except Exception as exc:
                print(f"  Translation attempt {attempt + 1} failed: {exc}")
                if self._is_non_retryable_http_error(exc):
                    raise RuntimeError(f"Translation failed after non-retryable HTTP error: {exc}")
                if attempt == self.retries - 1:
                    raise RuntimeError(f"Translation failed after {self.retries} attempts: {exc}")
                time.sleep(self._retry_delay(attempt, exc))

    def _translate_responses(self, system: str, prompt: str) -> TranslationResult:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "extra": self._extra_header(),
            "X-TT-logid": self.logid,
        }
        payload = {
            "model": self.model,
            "instructions": system,
            "input": prompt,
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
        }
        if self.reasoning_effort:
            payload["reasoning"] = {"effort": self.reasoning_effort}
        if self.include:
            payload["include"] = self.include

        for attempt in range(self.retries):
            try:
                started_at = self._pace_request()
                response = self.http_client.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                raw_text = self._responses_text(data)
                usage = self._responses_usage(data.get("usage"))
                return TranslationResult(
                    code=self._extract_code(raw_text),
                    elapsed_seconds=time.time() - started_at,
                    usage=usage,
                    raw_text=raw_text,
                )
            except Exception as exc:
                print(f"  Translation attempt {attempt + 1} failed: {exc}")
                if self._is_non_retryable_http_error(exc):
                    raise RuntimeError(f"Translation failed after non-retryable HTTP error: {exc}")
                if attempt == self.retries - 1:
                    raise RuntimeError(f"Translation failed after {self.retries} attempts: {exc}")
                time.sleep(self._retry_delay(attempt, exc))

    def _translate_google_genai(self, system: str, prompt: str) -> TranslationResult:
        config = {
            "system_instruction": system,
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
        }

        for attempt in range(self.retries):
            try:
                started_at = self._pace_request()
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config,
                )
                raw_text = self._google_response_text(response)
                usage = self._google_usage(getattr(response, "usage_metadata", None))
                return TranslationResult(
                    code=self._extract_code(raw_text),
                    elapsed_seconds=time.time() - started_at,
                    usage=usage,
                    raw_text=raw_text,
                )
            except Exception as exc:
                print(f"  Translation attempt {attempt + 1} failed: {exc}")
                if self._is_non_retryable_http_error(exc):
                    raise RuntimeError(f"Translation failed after non-retryable HTTP error: {exc}")
                if attempt == self.retries - 1:
                    raise RuntimeError(f"Translation failed after {self.retries} attempts: {exc}")
                time.sleep(self._retry_delay(attempt, exc))

    def _retry_delay(self, attempt: int, exc: Exception | None = None) -> float:
        default_delay = min(self.retry_max_delay, self.retry_initial_delay * (2 ** attempt))
        retry_after = self._retry_after_seconds(exc)
        if retry_after is not None:
            return max(default_delay, retry_after)
        if self._is_rate_limit_error(exc):
            return max(default_delay, self.rate_limit_retry_delay)
        return default_delay

    def _is_rate_limit_error(self, exc: Exception | None) -> bool:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
        return status_code == 429 or getattr(exc, "status_code", None) == 429 or getattr(exc, "code", None) == 429

    def _is_non_retryable_http_error(self, exc: Exception | None) -> bool:
        response = getattr(exc, "response", None)
        status_code = (
            getattr(response, "status_code", None)
            or getattr(exc, "status_code", None)
            or getattr(exc, "code", None)
        )
        if status_code in {401, 403, 404}:
            return True
        if status_code == 400:
            return self._is_permanent_bad_request(exc)
        return False

    def _is_permanent_bad_request(self, exc: Exception | None) -> bool:
        message = str(exc or "").lower()
        permanent_markers = (
            "exceeded model token limit",
            "context length",
            "maximum context",
            "max bytes to request body",
            "request body",
            "total message size",
            "too many tokens",
        )
        return any(marker in message for marker in permanent_markers)

    def _retry_after_seconds(self, exc: Exception | None) -> float | None:
        if not self._is_rate_limit_error(exc):
            return None
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", {}) or {}
        retry_after = headers.get("Retry-After")
        if retry_after is None:
            return None
        try:
            return float(retry_after)
        except (TypeError, ValueError):
            return None

    def _pace_request(self) -> float:
        if self.min_request_interval > 0:
            now = time.monotonic()
            if self._last_request_started_at is not None:
                elapsed = now - self._last_request_started_at
                if elapsed < self.min_request_interval:
                    wait_seconds = self.min_request_interval - elapsed
                    time.sleep(wait_seconds)
                    now += wait_seconds
            self._last_request_started_at = now
        return time.time()

    def _extract_code(self, text: str) -> str:
        rust_blocks = re.findall(r"```rust\s*\n(.*?)\n```", text, re.DOTALL)
        if rust_blocks:
            return rust_blocks[0].strip()

        code_blocks = re.findall(r"```\s*\n(.*?)\n```", text, re.DOTALL)
        if code_blocks:
            return code_blocks[0].strip()

        return text.strip()

    def _extra_header(self) -> str:
        import json

        return json.dumps({"session_id": self.session_id}, ensure_ascii=True)

    def _responses_text(self, data: dict) -> str:
        output_text = data.get("output_text")
        if isinstance(output_text, str):
            return output_text

        parts: list[str] = []
        output = data.get("output")
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, dict):
                    continue
                content = item.get("content")
                if not isinstance(content, list):
                    continue
                for content_item in content:
                    if not isinstance(content_item, dict):
                        continue
                    text = content_item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
        return "\n".join(parts)

    def _responses_usage(self, usage: Any) -> Usage:
        if usage is None:
            return Usage(prompt_tokens=None, completion_tokens=None, total_tokens=None)
        if isinstance(usage, dict):
            return Usage(
                prompt_tokens=usage.get("input_tokens") or usage.get("prompt_tokens"),
                completion_tokens=usage.get("output_tokens") or usage.get("completion_tokens"),
                total_tokens=usage.get("total_tokens"),
            )
        return self._usage(usage)

    def _google_response_text(self, response: Any) -> str:
        text = getattr(response, "text", None)
        if isinstance(text, str):
            return text
        if isinstance(response, dict):
            text = response.get("text")
            if isinstance(text, str):
                return text
        return str(response)

    def _google_usage(self, usage: Any) -> Usage:
        if usage is None:
            return Usage(prompt_tokens=None, completion_tokens=None, total_tokens=None)
        if isinstance(usage, dict):
            return Usage(
                prompt_tokens=usage.get("prompt_token_count") or usage.get("prompt_tokens"),
                completion_tokens=usage.get("candidates_token_count") or usage.get("completion_tokens"),
                total_tokens=usage.get("total_token_count") or usage.get("total_tokens"),
            )
        return Usage(
            prompt_tokens=getattr(usage, "prompt_token_count", None),
            completion_tokens=getattr(usage, "candidates_token_count", None),
            total_tokens=getattr(usage, "total_token_count", None),
        )

    def _usage(self, usage: Any) -> Usage:
        if usage is None:
            return Usage(prompt_tokens=None, completion_tokens=None, total_tokens=None)
        if isinstance(usage, dict):
            return Usage(
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                total_tokens=usage.get("total_tokens"),
            )
        return Usage(
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
        )


def create_agent(config: dict) -> Agent:
    return Agent(
        api_key=config["api_key"],
        base_url=config.get("base_url", ""),
        api_url=config.get("api_url", DEFAULT_RESPONSES_API_URL),
        model=config["model_name"],
        protocol=config.get("protocol", "openai-chat"),
        temperature=config.get("temperature", 1.0),
        retries=config.get("max_retries", 10),
        retry_initial_delay=config.get("retry_initial_delay", 4),
        retry_max_delay=config.get("retry_max_delay", 60),
        min_request_interval=config.get("min_request_interval", 0.0),
        rate_limit_retry_delay=config.get("rate_limit_retry_delay", 60.0),
        max_tokens=config.get("max_tokens", DEFAULT_MAX_OUTPUT_TOKENS),
        reasoning_effort=config.get("reasoning_effort", "medium"),
        include=config.get("include", ["reasoning.encrypted_content"]),
        chat_extra_body=config.get("chat_extra_body", _DEFAULT_CHAT_EXTRA_BODY_SENTINEL),
        chat_token_parameter=config.get("chat_token_parameter", "max_tokens"),
        google_api_version=config.get("google_api_version", ""),
        logid=config.get("logid") or load_logid(),
        session_id=config.get("session_id", ""),
        timeout=config.get("timeout", 300),
    )
