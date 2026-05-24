"""Tier 2: compact-threshold derivation and n_keep wiring.

Verifies the rules used to decide when compaction triggers, and that every
chat-completions call carries n_keep + enable_thinking.
"""
from __future__ import annotations

import io
import json
import urllib.request

import la
from conftest import sys_msg, user_msg, fake_response, FakeClient


# ---------- compute_compact_threshold ----------

def _patch_urlopen(monkeypatch, payload: dict | None, raise_exc: Exception | None = None):
    def _fake_urlopen(url, timeout=None):  # noqa: ARG001
        if raise_exc is not None:
            raise raise_exc

        class _Resp:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *exc):
                return False

            def read(self_inner):
                return json.dumps(payload).encode("utf-8")

        # urllib.request.urlopen is also used as a context manager via `with ... as r`,
        # and json.load(r) calls r.read(). Provide both.
        return _Resp()

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)


def test_threshold_env_override_wins(monkeypatch):
    monkeypatch.setattr(la, "COMPACT_PROMPT_TOKEN_THRESHOLD_OVERRIDE", 50_000)
    # /props would say 131072; override should still win
    _patch_urlopen(monkeypatch, {"default_generation_settings": {"n_ctx": 131072}})
    threshold, source = la.compute_compact_threshold("http://127.0.0.1:19434/v1")
    assert threshold == 50_000
    assert "env LOCAL_AGENT_COMPACT_THRESHOLD" in source


def test_threshold_from_props(monkeypatch):
    monkeypatch.setattr(la, "COMPACT_PROMPT_TOKEN_THRESHOLD_OVERRIDE", None)
    monkeypatch.setattr(la, "COMPACT_THRESHOLD_RATIO", 0.7)
    _patch_urlopen(monkeypatch, {"default_generation_settings": {"n_ctx": 131072}})
    threshold, source = la.compute_compact_threshold("http://127.0.0.1:19434/v1")
    assert threshold == int(131072 * 0.7)
    assert "n_ctx=131072" in source


def test_threshold_fallback_when_props_unreachable(monkeypatch):
    monkeypatch.setattr(la, "COMPACT_PROMPT_TOKEN_THRESHOLD_OVERRIDE", None)
    _patch_urlopen(monkeypatch, payload=None, raise_exc=ConnectionRefusedError("nope"))
    threshold, source = la.compute_compact_threshold("http://127.0.0.1:19434/v1")
    assert threshold == la.COMPACT_THRESHOLD_FALLBACK
    assert threshold == 250_000
    assert "fallback" in source


def test_threshold_fallback_when_props_lacks_n_ctx(monkeypatch):
    monkeypatch.setattr(la, "COMPACT_PROMPT_TOKEN_THRESHOLD_OVERRIDE", None)
    _patch_urlopen(monkeypatch, {"default_generation_settings": {}})
    threshold, source = la.compute_compact_threshold("http://127.0.0.1:19434/v1")
    assert threshold == la.COMPACT_THRESHOLD_FALLBACK


def test_default_threshold_covers_known_700k_read_for_256k_context(monkeypatch):
    """README validation: 679,955 chars produced ~235,052 prompt tokens.

    The default 256K-context threshold should sit above that known single-read
    workload so one large file read does not immediately force compaction before
    the model can synthesize from it.
    """
    monkeypatch.setattr(la, "COMPACT_PROMPT_TOKEN_THRESHOLD_OVERRIDE", None)
    _patch_urlopen(monkeypatch, {"default_generation_settings": {"n_ctx": 262144}})
    threshold, source = la.compute_compact_threshold("http://127.0.0.1:19434/v1")
    assert threshold >= 235_052
    assert "0.9 * server n_ctx=262144" in source


def test_query_server_n_ctx_strips_v1(monkeypatch):
    seen = {}

    def _spy(url, timeout=None):  # noqa: ARG001
        seen["url"] = url

        class _R:
            def __enter__(self_inner):
                return io.BytesIO(json.dumps({"default_generation_settings": {"n_ctx": 4096}}).encode("utf-8"))

            def __exit__(self_inner, *exc):
                return False

        return _R()

    monkeypatch.setattr(urllib.request, "urlopen", _spy)
    n = la.query_server_n_ctx("http://127.0.0.1:19434/v1")
    assert n == 4096
    assert seen["url"].endswith("/props")
    assert "/v1" not in seen["url"]


# ---------- n_keep + enable_thinking on every request ----------

def _run_one_turn(thinking: bool, monkeypatch, tmp_path):
    """Drive run_loop for one turn with a fake client emitting a final answer.
    Returns the captured kwargs of the chat.completions.create call."""
    monkeypatch.chdir(tmp_path)
    client = FakeClient([fake_response(content="ok done")])
    messages = [sys_msg(), user_msg("hello")]
    stats = {"compact_threshold": 999_999}
    code, final = la.run_loop(
        client, messages, cwd=tmp_path, max_turns=2, verbose=False,
        model="qwen", temperature=0.6, top_p=0.95,
        thinking=thinking, show_thinking=False, stats=stats,
        bg_base_url="http://127.0.0.1:19435/v1",
    )
    assert code == 0 and final == "ok done"
    return client.calls[0]


def test_request_carries_n_keep_and_thinking_true(monkeypatch, tmp_path):
    kw = _run_one_turn(thinking=True, monkeypatch=monkeypatch, tmp_path=tmp_path)
    extra = kw["extra_body"]
    assert extra["n_keep"] == la.N_KEEP_TOKENS
    assert extra["chat_template_kwargs"]["enable_thinking"] is True
    assert extra.get("top_k") == 20


def test_request_carries_thinking_false(monkeypatch, tmp_path):
    kw = _run_one_turn(thinking=False, monkeypatch=monkeypatch, tmp_path=tmp_path)
    assert kw["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


def test_run_loop_records_usage_into_stats(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    client = FakeClient([fake_response(content="bye", prompt_tokens=12345, completion_tokens=42)])
    messages = [sys_msg(), user_msg("hi")]
    stats = {"compact_threshold": 999_999}
    la.run_loop(
        client, messages, cwd=tmp_path, max_turns=2, verbose=False,
        model="qwen", temperature=0.6, top_p=0.95,
        thinking=True, show_thinking=False, stats=stats,
        bg_base_url="http://bg",
    )
    assert stats["last_prompt_tokens"] == 12345
    assert stats["last_completion_tokens"] == 42
