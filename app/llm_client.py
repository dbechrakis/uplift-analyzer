from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field

import httpx

from app.config import Settings


class LlmError(Exception):
    pass


class LlmUnavailableError(LlmError):
    pass


class LlmResponseError(LlmError):
    pass


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments_json: str


@dataclass
class StreamEvent:
    kind: str
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None


class LlamaClient:
    def __init__(
        self,
        settings: Settings | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self._http = httpx.Client(
            base_url=self.settings.base_url,
            timeout=httpx.Timeout(self.settings.timeout_seconds, connect=3.0),
            transport=transport,
        )

    def close(self) -> None:
        self._http.close()

    def is_available(self, timeout: float = 2.0) -> bool:
        try:
            return self._http.get("/models", timeout=timeout).status_code < 500
        except httpx.HTTPError:
            return False

    def stream_chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> Iterator[StreamEvent]:
        payload: dict = {
            "model": self.settings.model_name,
            "messages": messages,
            "stream": True,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        try:
            with self._http.stream("POST", "/chat/completions", json=payload) as response:
                if response.status_code >= 400:
                    body = response.read().decode(errors="replace")[:300]
                    raise LlmResponseError(f"LLM server returned {response.status_code}: {body}")
                yield from self._parse_stream(response.iter_lines())
        except httpx.ConnectError as exc:
            raise LlmUnavailableError(
                f"Cannot reach the LLM server at {self.settings.base_url}. "
                "Start llama.cpp or continue in no-LLM mode."
            ) from exc
        except httpx.TimeoutException as exc:
            raise LlmUnavailableError("The LLM server timed out.") from exc
        except httpx.HTTPError as exc:
            raise LlmUnavailableError(f"LLM connection failed: {exc}") from exc

    @staticmethod
    def _parse_stream(lines: Iterator[str]) -> Iterator[StreamEvent]:
        slots: dict[int, dict[str, str]] = {}
        finish_reason: str | None = None
        for line in lines:
            line = line.strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError as exc:
                raise LlmResponseError(f"Malformed stream chunk: {data[:100]}") from exc
            choices = chunk.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta") or {}
            if delta.get("content"):
                yield StreamEvent("content", text=delta["content"])
            for call in delta.get("tool_calls") or []:
                index = call.get("index", 0)
                slot = slots.setdefault(index, {"id": "", "name": "", "arguments": ""})
                function = call.get("function") or {}
                if call.get("id"):
                    slot["id"] = call["id"]
                if function.get("name") and not slot["name"]:
                    slot["name"] = function["name"]
                slot["arguments"] += function.get("arguments") or ""
            finish_reason = choice.get("finish_reason") or finish_reason
        if slots:
            calls = [
                ToolCall(s["id"] or f"call_{i}", s["name"], s["arguments"])
                for i, s in sorted(slots.items())
            ]
            yield StreamEvent("tool_calls", tool_calls=calls)
        yield StreamEvent("finish", finish_reason=finish_reason)
