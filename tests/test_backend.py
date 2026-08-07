"""Zero-cost tests for the deterministic, batched model backend."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from mining.backend import (
    BackendError,
    RunLock,
    build_chat_prompt,
    derive_batch_seed,
    make_generate_fn,
    preflight_adapter,
    preflight_chat_template,
    verify_persistent_root,
)
from mining.mine_pairs import SAMPLES_PER_PROMPT, Prompt


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "RENDERED::" + "|".join(f"{m['role']}={m['content']}" for m in messages)


def prompt(pid="p1", messages=(("system", "sys"), ("user", "u"))) -> Prompt:
    return Prompt(pid, "multi", "sys", "u", "t", messages)


def test_batch_seed_is_identity_stable() -> None:
    first = derive_batch_seed("prompt-41")
    assert derive_batch_seed("prompt-41") == first
    assert derive_batch_seed("prompt-42") != first


def test_rendering_preserves_tool_turn() -> None:
    rendered = build_chat_prompt(
        FakeTokenizer(), prompt(messages=(("system", "s"), ("user", "u"), ("tool", "t")))
    )
    assert "tool=t" in rendered


def test_empty_prompt_is_refused() -> None:
    with pytest.raises(BackendError, match="no prompt turns"):
        build_chat_prompt(FakeTokenizer(), prompt(messages=()))


def test_batch_callback_is_mandatory_and_returns_all_samples() -> None:
    seen = []

    def batch(model, tok, rendered, max_tokens, temperature, top_p, seed, samples):
        seen.append((seed, samples))
        return [f"gen-{i}" for i in range(samples)]

    generate = make_generate_fn(object(), FakeTokenizer(), batch)
    assert generate(prompt(), SAMPLES_PER_PROMPT) == [f"gen-{i}" for i in range(8)]
    assert seen == [(derive_batch_seed("p1"), 8)]


def test_short_batch_fails_closed() -> None:
    def batch(*args):
        return ["one"]

    with pytest.raises(BackendError, match="batch returned 1 of 8"):
        make_generate_fn(object(), FakeTokenizer(), batch)(prompt(), 8)


def test_template_preflight_fails_closed() -> None:
    class RefusingTokenizer(FakeTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            if any(m["role"] == "tool" for m in messages):
                raise ValueError("unsupported tool role")
            return "ok"

    with pytest.raises(BackendError, match="cannot render"):
        preflight_chat_template(
            RefusingTokenizer(),
            [prompt(messages=(("system", "s"), ("user", "u"), ("tool", "t")))],
        )


def test_lock_is_exclusive_and_reusable(tmp_path) -> None:
    with RunLock(tmp_path), pytest.raises(BackendError, match="held by"), RunLock(tmp_path):
        pass
    with RunLock(tmp_path):
        pass


def test_container_local_root_requires_attestation(tmp_path) -> None:
    with pytest.raises(BackendError, match="container root device"):
        verify_persistent_root(tmp_path / "out", tmp_path)
    assert verify_persistent_root(tmp_path / "out", tmp_path, attested=True).name == "out"


def test_direct_model_stage_requires_external_launcher(monkeypatch, tmp_path) -> None:
    import mining.backend as backend

    monkeypatch.delenv("LLAMA_TOOLS_LAUNCH_CONTEXT", raising=False)
    monkeypatch.delenv("LLAMA_TOOLS_LAUNCH_TOKEN", raising=False)
    with pytest.raises(BackendError, match="refusing direct model launch"):
        backend.execute_model_stage(tmp_path / "out", "pilot", tmp_path, attested=True)


def test_adapter_metadata_is_resolved_before_weights(monkeypatch) -> None:
    import mining.backend as backend

    expected = object()

    class Config:
        @classmethod
        def from_pretrained(cls, repo, subfolder, revision):
            assert repo == "centuriandip/llama-3.1-8b-tools-sft"
            assert subfolder == "adapter"
            assert revision == "b6f4da479f8c6fc044ee8b802a92f47780f970c5"
            return expected

    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(PeftConfig=Config))
    assert backend.SFT_ADAPTER_SUBFOLDER == "adapter/", "frozen identity remains unchanged"
    assert preflight_adapter() is expected


def test_model_stage_orchestration_is_wired_end_to_end(monkeypatch, tmp_path) -> None:
    import mining.backend as backend
    import mining.mine_pairs as miner

    calls = []
    selected = [prompt()]
    prepared = SimpleNamespace(selected=selected)
    out_dir = tmp_path / "out"
    context = tmp_path / "launch-context.json"
    context.write_text('{"token": "test-token", "out_dir": "' + str(out_dir) + '"}\n')
    monkeypatch.setenv("LLAMA_TOOLS_LAUNCH_CONTEXT", str(context))
    monkeypatch.setenv("LLAMA_TOOLS_LAUNCH_TOKEN", "test-token")
    monkeypatch.setattr(backend, "load_tokenizer", lambda: object())
    monkeypatch.setattr(
        backend, "preflight_chat_template",
        lambda tok, rows: calls.append("template") or 1,
    )
    monkeypatch.setattr(
        backend, "load_policy_with",
        lambda tok: calls.append("model") or (lambda *_: []),
    )
    monkeypatch.setattr(
        miner, "prepare_stage_run",
        lambda out, stage, fresh: calls.append("prepare") or prepared,
    )
    monkeypatch.setattr(
        miner, "_run_prepared",
        lambda out, pre, fn: calls.append("run") or {"ok": True},
    )

    result = backend.execute_model_stage(
        out_dir, "pilot", tmp_path, attested=True
    )

    assert result == {"ok": True}
    assert calls == ["prepare", "template", "model", "run"]
