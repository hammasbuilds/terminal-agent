"""Chat clients: the Ollama HTTP client (with an on-disk generation cache) and a scripted one.

Every client returns a :class:`ModelTurn`. The agent loop never knows which client it is
talking to, which is what lets the SWE-bench harness be validated with a scripted client
that replays gold patches through exactly the code path a model would use.
"""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from terminal_agent.protocol import ToolCall, parse_text_tool_calls

DEFAULT_MODEL = "qwen2.5-coder:14b"
DEFAULT_HOST = "http://127.0.0.1:11434"
DEFAULT_OPTIONS: dict[str, Any] = {
    "temperature": 0.0,
    "seed": 0,
    "num_ctx": 16384,  # Ollama's default (2048/4096) silently truncates long prompts
    "num_predict": 2048,
}


@dataclass
class ModelTurn:
    """One assistant response: free text plus zero or more tool calls."""

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached: bool = False
    parsed_from_text: bool = False


class ModelError(RuntimeError):
    """The model could not be reached or returned something unusable."""


class ChatClient(Protocol):
    model: str

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelTurn:
        ...


def cache_key(model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
              options: dict[str, Any]) -> str:
    """Stable key over everything that determines a generation."""
    blob = json.dumps(
        {"model": model, "messages": messages, "tools": tools, "options": options},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class OllamaClient:
    """Minimal client for Ollama's ``/api/chat`` with native tool calling.

    Responses are cached on disk keyed by (model, messages, tools, options), so an
    interrupted run resumes without re-generating anything it already has.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        options: dict[str, Any] | None = None,
        cache_dir: Path | None = None,
        timeout: float = 900.0,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.options = {**DEFAULT_OPTIONS, **(options or {})}
        self.cache_dir = cache_dir
        self.timeout = timeout

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8", "replace")
            try:
                return json.loads(body)
            except json.JSONDecodeError as exc:
                raise ModelError(f"Ollama returned a non-JSON response: {body[:300]!r}") from exc
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            if exc.code == 404:
                raise ModelError(
                    f"Ollama has no model {self.model!r} ({detail.strip()}). "
                    f"Run: ollama pull {self.model}"
                ) from exc
            raise ModelError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            raise ModelError(
                f"cannot reach Ollama at {self.host} ({exc}). Start it with `ollama serve`, "
                f"or point --host at a running server."
            ) from exc

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelTurn:
        key = cache_key(self.model, messages, tools, self.options)
        path = self.cache_dir / key[:2] / f"{key}.json" if self.cache_dir else None
        if path is not None and path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            turn = turn_from_response(data)
            turn.cached = True
            return turn
        data = self._post(
            {
                "model": self.model,
                "messages": messages,
                "tools": tools,
                "stream": False,
                "options": self.options,
            }
        )
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8", newline="\n")
            tmp.replace(path)
        return turn_from_response(data)


def turn_from_response(data: dict[str, Any]) -> ModelTurn:
    """Convert an ``/api/chat`` response into a ModelTurn.

    Native ``tool_calls`` win. If there are none, the text is searched for tool calls
    written out as JSON (``<tool_call>`` tags or a fenced block), which is how
    qwen2.5-coder sometimes answers when its chat template does not engage.
    """
    message = data.get("message") or {}
    content = message.get("content") or ""
    calls: list[ToolCall] = []
    for i, raw in enumerate(message.get("tool_calls") or []):
        fn = raw.get("function") or {}
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"__unparsed__": args}
        if not isinstance(args, dict):  # a JSON list/scalar is not valid tool arguments
            args = {"__unparsed__": json.dumps(args)}
        calls.append(ToolCall(name=str(fn.get("name", "")), arguments=args, id=f"n{i}"))
    parsed = False
    if not calls and content:
        calls = parse_text_tool_calls(content)
        parsed = bool(calls)
    return ModelTurn(
        content=content,
        tool_calls=calls,
        prompt_tokens=data.get("prompt_eval_count"),
        completion_tokens=data.get("eval_count"),
        parsed_from_text=parsed,
    )


class ScriptedClient:
    """Replays a fixed list of turns, ignoring what it is sent.

    Used by the harness validation (gold patches as tool calls), by ``--script`` in the
    CLI, and by the tests. When the script runs out it answers with a final message and
    no tool calls, which ends the run.
    """

    def __init__(self, turns: Sequence[ModelTurn], model: str = "scripted") -> None:
        self.model = model
        self._turns = list(turns)
        self.calls = 0

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelTurn:
        self.calls += 1
        if not self._turns:
            return ModelTurn(content="(script finished)")
        return self._turns.pop(0)

    @classmethod
    def from_file(cls, path: Path) -> ScriptedClient:
        """Load a script: a JSON list of ``{"content": str, "tool_calls": [{name, arguments}]}``."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read script {path}: {exc}") from exc
        if not isinstance(data, list):
            raise ValueError(f"script {path} must be a JSON list of turns")
        turns = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"script {path}: turn {i} must be an object, got "
                                 f"{type(item).__name__}")
            calls = []
            for j, c in enumerate(item.get("tool_calls") or []):
                if not isinstance(c, dict) or not isinstance(c.get("name"), str):
                    raise ValueError(f"script {path}: turn {i} tool call {j} needs a string "
                                     f"'name'")
                calls.append(ToolCall(name=c["name"], arguments=c.get("arguments") or {},
                                      id=f"s{i}.{j}"))
            turns.append(ModelTurn(content=item.get("content", ""), tool_calls=calls))
        return cls(turns)
