"""Tests for the generation backend (prereg §2.1, §2.4).

No torch, no weights, no GPU, no network: everything that decides *what* gets
sampled and *when the run stops* is separated from the one function that touches
a model. These tests cover the two properties the miner cannot enforce itself —
resume-stable seeding and a deadline the run cannot outlive.
"""

from __future__ import annotations

import pytest

from mining.backend import (
    MAX_WALL_CLOCK_SECONDS,
    STORAGE_RESERVE_USD,
    BackendError,
    SpendGuard,
    build_chat_prompt,
    compute_deadline_seconds,
    derive_prompt_seed,
    make_generate_fn,
    sampling_receipt,
)
from mining.mine_pairs import SAMPLES_PER_PROMPT, Prompt


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "RENDERED::" + "|".join(f"{m['role']}={m['content']}" for m in messages)


def _prompt(pid="p1", messages=(("system", "sys"), ("user", "u"))) -> Prompt:
    return Prompt(
        prompt_id=pid, stratum="multi", system="sys", user="u", target="t",
        messages=messages,
    )


# --- Resume-stable seeding ---------------------------------------------------


def test_the_seed_depends_on_identity_not_on_call_order() -> None:
    """The reason a global counter cannot be used: after a resume it has been
    advanced a different number of times, so the 'seeded' run samples something
    else entirely."""
    first = derive_prompt_seed("prompt-41", 3)

    assert derive_prompt_seed("prompt-41", 3) == first
    assert derive_prompt_seed("prompt-41", 4) != first
    assert derive_prompt_seed("prompt-42", 3) != first


def test_seeds_stay_inside_the_range_torch_accepts() -> None:
    for index in range(SAMPLES_PER_PROMPT):
        seed = derive_prompt_seed("some-prompt-id", index)
        assert 0 <= seed < 2**63


def test_a_resumed_run_reproduces_the_samples_it_would_have_produced(tmp_path) -> None:
    """Drive the same prompts twice, once 'after a crash' with earlier prompts
    skipped, and assert the seeds for the remaining prompts are identical."""
    prompts = [_prompt(f"p{i}") for i in range(5)]

    uninterrupted = {
        p.prompt_id: [derive_prompt_seed(p.prompt_id, i) for i in range(8)]
        for p in prompts
    }
    resumed = {
        p.prompt_id: [derive_prompt_seed(p.prompt_id, i) for i in range(8)]
        for p in prompts[3:]  # first three already in the ledger
    }

    for pid, seeds in resumed.items():
        assert seeds == uninterrupted[pid]


# --- The deadline ------------------------------------------------------------


def test_the_deadline_comes_from_the_rate_actually_charged() -> None:
    # $1.00 cap, $0.08 storage reserve, $0.53/hr Secure -> 0.92/0.53 h
    assert compute_deadline_seconds(1.00, 0.53) == int(0.92 / 0.53 * 3600)


def test_a_cheap_hour_does_not_buy_unlimited_wall_clock() -> None:
    """At $0.33/hr the cap alone would allow ~2h47m; the estimate's worst case
    is 1.5h, so the wall-clock ceiling binds instead."""
    assert compute_deadline_seconds(1.00, 0.33) == MAX_WALL_CLOCK_SECONDS


def test_a_cap_that_cannot_cover_storage_refuses_to_launch() -> None:
    with pytest.raises(BackendError, match="storage reserve"):
        compute_deadline_seconds(STORAGE_RESERVE_USD, 0.53)


def test_a_nonpositive_rate_is_refused() -> None:
    with pytest.raises(BackendError, match="rate must be positive"):
        compute_deadline_seconds(1.00, 0.0)


def test_there_is_no_unbounded_mode() -> None:
    with pytest.raises(BackendError, match="no positive deadline"):
        SpendGuard(deadline_seconds=0, monotonic=lambda: 0.0)


def test_the_guard_stops_before_the_next_prompt_not_mid_prompt() -> None:
    clock = {"now": 0.0}
    guard = SpendGuard(deadline_seconds=100, monotonic=lambda: clock["now"])
    calls: list[str] = []

    def generate_once(model, tok, rendered, mx, temp, top_p, seed):
        calls.append(rendered)
        return "out"

    generate_fn = make_generate_fn(object(), FakeTokenizer(), guard, generate_once)

    generate_fn(_prompt("p1"), 2)
    clock["now"] = 101.0

    with pytest.raises(BackendError, match="spend deadline reached"):
        generate_fn(_prompt("p2"), 2)

    assert len(calls) == 2, "the overrunning prompt never started"


def test_the_guard_message_points_at_the_resume_path() -> None:
    clock = {"now": 0.0}
    guard = SpendGuard(deadline_seconds=1, monotonic=lambda: clock["now"])
    clock["now"] = 5.0

    with pytest.raises(BackendError) as excinfo:
        guard.check()

    assert "rerun the same" in str(excinfo.value)


# --- Rendering ---------------------------------------------------------------


def test_rendering_uses_the_rows_own_turns_including_a_tool_message() -> None:
    """The 7 tool-prefixed pool rows depend on this; flattening to system+user
    drops the context their completion answers."""
    prompt = _prompt(
        messages=(("system", "sys"), ("user", "u"), ("tool", "tool output"))
    )

    rendered = build_chat_prompt(FakeTokenizer(), prompt)

    assert "tool=tool output" in rendered
    assert rendered.index("user=u") < rendered.index("tool=tool output")


def test_a_prompt_with_no_turns_is_refused() -> None:
    with pytest.raises(BackendError, match="no prompt turns"):
        build_chat_prompt(FakeTokenizer(), _prompt(messages=()))


def test_generate_fn_returns_one_output_per_requested_sample() -> None:
    guard = SpendGuard(deadline_seconds=100, monotonic=lambda: 0.0)
    seeds: list[int] = []

    def generate_once(model, tok, rendered, mx, temp, top_p, seed):
        seeds.append(seed)
        return f"gen-{seed}"

    generate_fn = make_generate_fn(object(), FakeTokenizer(), guard, generate_once)
    outputs = generate_fn(_prompt("p1"), SAMPLES_PER_PROMPT)

    assert len(outputs) == SAMPLES_PER_PROMPT
    assert len(set(seeds)) == SAMPLES_PER_PROMPT, "each sample gets its own seed"
    assert seeds == [derive_prompt_seed("p1", i) for i in range(SAMPLES_PER_PROMPT)]


# --- The launch receipt ------------------------------------------------------


def test_the_receipt_records_what_a_launch_sheet_must_carry() -> None:
    receipt = sampling_receipt(hourly_rate_usd=0.53, cap_usd=1.00)

    assert receipt["sampling"]["temperature"] == 0.8
    assert receipt["sampling"]["seed"] == 20260804
    assert receipt["adapter"]["revision"].startswith("b6f4da4")
    assert receipt["spend_bound"]["deadline_seconds"] == compute_deadline_seconds(1.00, 0.53)
    assert receipt["spend_bound"]["hourly_rate_usd"] == 0.53


# --- Review cycle 1: the cap must bound the STAGE, not the invocation --------


def test_the_deadline_check_runs_per_sample_not_per_prompt() -> None:
    """Codex's repro: 8 samples of 100s each all ran past a 10s deadline,
    because the only check happened between prompts."""
    clock = {"now": 0.0}
    guard = SpendGuard(deadline_seconds=10, monotonic=lambda: clock["now"])
    produced: list[int] = []

    def generate_once(model, tok, rendered, mx, temp, top_p, seed):
        clock["now"] += 100.0  # each sample burns 100s
        produced.append(seed)
        return "out"

    generate_fn = make_generate_fn(object(), FakeTokenizer(), guard, generate_once)

    with pytest.raises(BackendError, match="spend deadline reached"):
        generate_fn(_prompt("p1"), 8)

    assert len(produced) == 1, "the deadline stops the second sample, not the ninth"


def test_prior_sessions_shrink_the_allowance() -> None:
    """A resume gets what is left, not a fresh window."""
    guard = SpendGuard(
        deadline_seconds=100, monotonic=lambda: 0.0, consumed_before=95.0
    )

    assert guard.remaining == pytest.approx(5.0)


def test_an_exhausted_allowance_refuses_immediately() -> None:
    guard = SpendGuard(
        deadline_seconds=100, monotonic=lambda: 0.0, consumed_before=100.0
    )

    with pytest.raises(BackendError, match="spend deadline reached"):
        guard.check()


def test_a_cap_above_the_approved_ceiling_is_refused() -> None:
    with pytest.raises(BackendError, match="exceeds the owner-approved"):
        compute_deadline_seconds(100.0, 0.53)


def test_non_finite_values_are_refused_rather_than_crashing() -> None:
    for bad in (float("nan"), float("inf")):
        with pytest.raises(BackendError, match="finite number"):
            compute_deadline_seconds(bad, 0.53)
        with pytest.raises(BackendError, match="finite number"):
            compute_deadline_seconds(1.00, bad)


# --- Chat-template preflight, before any weights load ------------------------


class RefusingTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        if any(m["role"] == "tool" for m in messages):
            raise ValueError("template does not know the 'tool' role")
        return "RENDERED"


def test_the_template_preflight_fails_closed_before_weights_load() -> None:
    from mining.backend import preflight_chat_template

    prompts = [
        _prompt("plain"),
        _prompt("tooled", messages=(("system", "s"), ("user", "u"), ("tool", "t"))),
    ]

    with pytest.raises(BackendError, match="cannot render this row's turns"):
        preflight_chat_template(RefusingTokenizer(), prompts)


def test_the_template_preflight_passes_when_every_row_renders() -> None:
    from mining.backend import preflight_chat_template

    prompts = [_prompt(f"p{i}") for i in range(5)]
    assert preflight_chat_template(FakeTokenizer(), prompts) == 5


# --- Exclusive lock ----------------------------------------------------------


def test_two_processes_cannot_bill_against_one_stage(tmp_path) -> None:
    from mining.backend import RunLock

    with RunLock(tmp_path):
        with pytest.raises(BackendError, match="is held by"):
            with RunLock(tmp_path):
                pass


def test_the_lock_is_released_on_exit(tmp_path) -> None:
    from mining.backend import RunLock

    with RunLock(tmp_path):
        pass
    with RunLock(tmp_path):
        pass  # reacquiring is fine once released


# --- Persistent output root --------------------------------------------------


def test_a_paid_run_refuses_container_local_output(tmp_path) -> None:
    from mining.backend import verify_persistent_root

    with pytest.raises(BackendError, match="--persistent-root is required"):
        verify_persistent_root(tmp_path / "out", None)


def test_the_out_dir_must_live_inside_the_durable_mount(tmp_path) -> None:
    from mining.backend import verify_persistent_root

    root = tmp_path / "mnt"
    root.mkdir()
    elsewhere = tmp_path / "container_local"
    elsewhere.mkdir()

    with pytest.raises(BackendError, match="not inside"):
        verify_persistent_root(elsewhere, root)

    assert verify_persistent_root(root / "pilot", root) == (root / "pilot").resolve()
