"""Shared test fixtures for the la.py harness.

These tests don't talk to a real model. They mock:
- la.summarize_via_bg, so inter/intra compaction never hits a network.
- The OpenAI client, via a small FakeClient that returns scripted messages.
- urllib.request.urlopen, for /props threshold derivation.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import la  # noqa: E402


# ---------- message builders ----------

def sys_msg(text: str = "SYSTEM PROMPT", with_marker_summary: str | None = None) -> dict:
    content = text
    if with_marker_summary is not None:
        content = text + la.COMPACT_MARKER + with_marker_summary
    return {"role": "system", "content": content}


def user_msg(text: str) -> dict:
    return {"role": "user", "content": text}


def assistant_msg(text: str = "", tool_calls: list[dict] | None = None,
                  reasoning: str | None = None) -> dict:
    d: dict = {"role": "assistant", "content": text}
    if tool_calls:
        d["tool_calls"] = tool_calls
    if reasoning is not None:
        d["reasoning_content"] = reasoning
    return d


def tool_call(call_id: str, name: str, arguments: dict | str = "") -> dict:
    if isinstance(arguments, dict):
        arguments = json.dumps(arguments)
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def tool_msg(call_id: str, content: str = "ok") -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def make_exchange(call_id: str, name: str = "list_dir",
                  args: dict | None = None, result: str = "ok",
                  ass_text: str = "") -> list[dict]:
    """One assistant tool-calling message + its tool response."""
    return [
        assistant_msg(ass_text, tool_calls=[tool_call(call_id, name, args or {"path": "."})]),
        tool_msg(call_id, result),
    ]


def make_group(user_text: str, n_exchanges: int, prefix: str = "tc") -> list[dict]:
    """A user message followed by n tool-calling assistant/tool exchanges plus a final assistant text."""
    msgs = [user_msg(user_text)]
    for i in range(n_exchanges):
        msgs.extend(make_exchange(f"{prefix}-{i}"))
    msgs.append(assistant_msg(f"done with {user_text}"))
    return msgs


# ---------- fake OpenAI client ----------

class _FakeMessage:
    def __init__(self, content: str = "", tool_calls=None, reasoning_content: str | None = None):
        self.content = content
        self.tool_calls = tool_calls
        self.reasoning_content = reasoning_content


class _FakeChoice:
    def __init__(self, message: _FakeMessage):
        self.message = message


class _FakeUsage:
    def __init__(self, prompt_tokens: int = 0, completion_tokens: int = 0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _FakeResponse:
    def __init__(self, message: _FakeMessage, usage: _FakeUsage | None = None):
        self.choices = [_FakeChoice(message)]
        self.usage = usage


class _FakeToolCall:
    def __init__(self, call_id: str, name: str, arguments: str):
        self.id = call_id
        self.type = "function"
        self.function = types.SimpleNamespace(name=name, arguments=arguments)


class FakeChatCompletions:
    def __init__(self, scripted: list[_FakeResponse]):
        self.scripted = list(scripted)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.scripted:
            raise AssertionError("FakeChatCompletions: ran out of scripted responses")
        return self.scripted.pop(0)


class FakeChat:
    def __init__(self, completions: FakeChatCompletions):
        self.completions = completions


class FakeClient:
    """Drop-in replacement for an openai.OpenAI client for run_loop tests."""

    def __init__(self, scripted: list[_FakeResponse]):
        self.chat = FakeChat(FakeChatCompletions(scripted))

    @property
    def calls(self):
        return self.chat.completions.calls


def fake_response(content: str = "", *, tool_calls: list[tuple[str, str, dict]] | None = None,
                  reasoning: str | None = None,
                  prompt_tokens: int = 100, completion_tokens: int = 10) -> _FakeResponse:
    tcs = None
    if tool_calls:
        tcs = [_FakeToolCall(cid, name, json.dumps(args)) for (cid, name, args) in tool_calls]
    return _FakeResponse(
        _FakeMessage(content=content, tool_calls=tcs, reasoning_content=reasoning),
        _FakeUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


@pytest.fixture
def fake_client_factory():
    def _make(scripted):
        return FakeClient(scripted)
    return _make


# ---------- module-level patches ----------

@pytest.fixture
def stub_summarizer(monkeypatch):
    """Replace la.summarize_via_bg with a deterministic stub that returns a fixed
    string and records calls. Avoids any HTTP traffic from compaction tests."""
    calls: list[dict] = []

    def _stub(text, bg_base_url, model, prior_summary=None):
        calls.append({
            "text": text, "bg_base_url": bg_base_url, "model": model,
            "prior_summary": prior_summary,
        })
        prefix = ""
        if prior_summary:
            prefix = f"[carries prior summary {len(prior_summary)} chars] "
        return prefix + f"SUMMARY({len(text)} chars)"

    monkeypatch.setattr(la, "summarize_via_bg", _stub)
    return calls


@pytest.fixture
def builders():
    return types.SimpleNamespace(
        sys_msg=sys_msg,
        user_msg=user_msg,
        assistant_msg=assistant_msg,
        tool_call=tool_call,
        tool_msg=tool_msg,
        make_exchange=make_exchange,
        make_group=make_group,
        fake_response=fake_response,
    )
