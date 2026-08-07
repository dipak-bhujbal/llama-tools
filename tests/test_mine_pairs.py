"""Tests for the production pair miner (prereg §2).

No model, no network, no GPU: generation enters `run()`/`mine_prompt()` as a
`generate_fn`, so every path below runs at $0.

The tests are organised around the ways this miner could produce a *plausible
but wrong* artifact, because those are the failures that survive review:

- an inverted pair, from reinterpreting a target it cannot read (§2.11 — the
  defect that quarantined the previous miner);
- a yield that a rollback inflated, or that a counter reported instead of the
  ledger (A2.1);
- a stratum silently rounded to zero, making y_std undefined but reportable
  (§2.3);
- a gate decision moved by a rounded intermediate (§2.6).
"""

from __future__ import annotations

import json
import re
from fractions import Fraction
from pathlib import Path

import pytest

from mining import mine_pairs
from mining.ledger import Ledger
from mining.mine_pairs import (
    MULTI,
    SINGLE,
    MinerError,
    Prompt,
    PromptOutcome,
    allocate,
    gate_decision,
    materialize_pair,
    mine_prompt,
    select_prompts,
    summarize,
)

TARGET = '<tool_call>\n{"name": "do_thing", "arguments": {"x": 1}}\n</tool_call>'
GOOD = '<tool_call>\n{"name": "do_thing", "arguments": {"x": 1}}\n</tool_call>'
BAD = '<tool_call>\n{"name": "do_thing", "arguments": {"x": 99}}\n</tool_call>'
MALFORMED = '<tool_call>\n{"name": "do_thing", "arguments": {"x": 1}\n</tool_call>'


def _prompt(pid: str = "p1", stratum: str = MULTI, messages=None) -> Prompt:
    return Prompt(
        prompt_id=pid,
        stratum=stratum,
        system="sys",
        user="u",
        target=TARGET,
        messages=messages if messages is not None else (("system", "sys"), ("user", "u")),
    )


_RECEIPT = {
    "post_screen_id_sha256": "69d381413f8095d483b35c9bcd77e83bd6f72771edc9b4f192510f8e7392e5e3",
    "survivors": {"multi": 8081, "single": 2990, "total": 11071},
}


def _generator(samples: list[str]):
    def generate_fn(prompt: Prompt, n: int) -> list[str]:
        return list(samples)

    return generate_fn


# --- §2.11: refuse, never reclassify -----------------------------------------


def test_an_unreadable_target_refuses_the_run_rather_than_becoming_no_call() -> None:
    """The exact defect that quarantined the previous miner.

    A target that does not parse must stop the run. Turning it into a `no_call`
    ground truth mints a pair whose "chosen" is wrong — training signal pointing
    the wrong way, produced silently.
    """
    row = {"messages": [{"role": "assistant", "content": "<tool_call>{not json</tool_call>"}]}

    with pytest.raises(MinerError, match="refuses the run rather than reclassifying"):
        mine_pairs._target_turn(row, "p1")


def test_a_target_that_degrades_mid_run_refuses_instead_of_guessing(monkeypatch) -> None:
    def exploding_verify(generation, target):
        raise mine_pairs.TargetUnreadableError("target went bad")

    monkeypatch.setattr(mine_pairs, "verify", exploding_verify)

    with pytest.raises(MinerError, match="became unreadable mid-run"):
        mine_prompt(_prompt(), _generator([GOOD] * 8))


def test_a_short_sample_batch_is_refused() -> None:
    with pytest.raises(MinerError, match="expected 8"):
        mine_prompt(_prompt(), _generator([GOOD] * 7))


# --- Pair construction is deterministic --------------------------------------


def test_a_pair_needs_both_an_accepted_and_a_rejected_generation() -> None:
    all_good = PromptOutcome("p", MULTI, [GOOD] * 3, [{"accepted": True}] * 3)
    all_bad = PromptOutcome("p", MULTI, [BAD] * 3, [{"accepted": False}] * 3)

    assert materialize_pair(all_good) is None
    assert materialize_pair(all_bad) is None


def test_the_same_record_always_materializes_the_same_pair() -> None:
    """Determinism is what makes A2.1's yield recomputable rather than reported."""
    outcome = PromptOutcome(
        "p",
        MULTI,
        ["a", "b", "c", "d"],
        [
            {"accepted": False, "reason": "r0"},
            {"accepted": True, "reason": None},
            {"accepted": True, "reason": None},
            {"accepted": False, "reason": "r3"},
        ],
    )

    first = materialize_pair(outcome)
    assert materialize_pair(outcome) == first
    assert first["chosen"] == "b" and first["chosen_index"] == 1
    assert first["rejected"] == "a" and first["rejected_index"] == 0


# --- §2.3 allocation ---------------------------------------------------------


def test_allocation_is_proportional_to_the_amended_weights() -> None:
    counts = {MULTI: 8081, SINGLE: 2990}
    alloc = allocate(100, counts)

    assert alloc[MULTI] + alloc[SINGLE] == 100
    # 8081/11071 = 72.993% -> 73 of 100 by largest remainder
    assert alloc[MULTI] == 73
    assert alloc[SINGLE] == 27


def test_both_strata_get_a_nonzero_allocation_even_when_rounding_says_zero() -> None:
    """§2.3 requires y_multi and y_single each estimable; a zero makes one of
    them undefined and the other assumed from it."""
    alloc = allocate(2, {MULTI: 9999, SINGLE: 1})

    assert alloc[MULTI] >= 1
    assert alloc[SINGLE] >= 1


def test_one_prompt_cannot_satisfy_both_strata_and_is_refused() -> None:
    with pytest.raises(MinerError, match="both strata"):
        allocate(1, {MULTI: 100, SINGLE: 100})


def test_allocation_cannot_exceed_the_available_prompts() -> None:
    with pytest.raises(MinerError, match="only"):
        allocate(100, {MULTI: 5, SINGLE: 5})


# --- Selection is seeded and reproducible ------------------------------------


def test_selection_is_reproducible_from_the_seed_alone() -> None:
    prompts = [_prompt(f"p{i}", MULTI if i % 2 else SINGLE) for i in range(20)]
    alloc = {MULTI: 4, SINGLE: 3}

    first = [p.prompt_id for p in select_prompts(prompts, alloc)]
    second = [p.prompt_id for p in select_prompts(prompts, alloc)]

    assert first == second
    assert len(first) == 7


def test_a_different_seed_selects_a_different_set() -> None:
    prompts = [_prompt(f"p{i}", MULTI) for i in range(50)]
    alloc = {MULTI: 5, SINGLE: 0}

    default = [p.prompt_id for p in select_prompts(prompts, alloc)]
    other = [p.prompt_id for p in select_prompts(prompts, alloc, seed=1)]

    assert default != other


# --- A2.1: yield is recomputed from the ledger -------------------------------


def _record(pid: str, stratum: str, *, pair: bool) -> dict:
    verdicts = (
        [{"accepted": True}, {"accepted": False}] if pair else [{"accepted": False}] * 2
    )
    return {
        "prompt_id": pid,
        "stratum": stratum,
        "generations": [GOOD, BAD],
        "verdicts": verdicts,
    }


def test_yield_is_recomputed_from_active_records(tmp_path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.append(_record("m1", MULTI, pair=True))
    ledger.append(_record("m2", MULTI, pair=False))
    ledger.append(_record("s1", SINGLE, pair=True))

    summary = summarize(ledger, {MULTI: 8081, SINGLE: 2990})

    assert summary["prompts_mined_total"] == 3
    assert summary["pairs_total"] == 2
    assert summary["y_multi"] == 0.5
    assert summary["y_single"] == 1.0


def test_a_rolled_back_prompt_leaves_the_numerator_and_denominator_together(tmp_path) -> None:
    """A2.1's stated reason for defining both terms against the same artifact."""
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.append(_record("m1", MULTI, pair=True))
    ledger.append(_record("s1", SINGLE, pair=True))

    before = summarize(ledger, {MULTI: 1, SINGLE: 1})
    assert before["prompts_mined_total"] == 2 and before["pairs_total"] == 2

    ledger.redo_last(1)
    after = summarize(ledger, {MULTI: 1, SINGLE: 1})

    assert after["prompts_mined_total"] == 1
    assert after["pairs_total"] == 1
    assert after["y_single"] is None, "a stratum with no active record is not estimable"


def test_a_re_mined_prompt_materializes_from_the_new_record(tmp_path) -> None:
    """The bug this test exists for: selecting active work by `prompt_id`
    instead of by `seq` picks up the superseded record, because the id is active
    on account of its replacement."""
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.append(_record("m1", MULTI, pair=True))
    ledger.append(_record("s1", SINGLE, pair=True))
    ledger.redo_last(2)
    # Re-mined: this time m1 yields no pair.
    ledger.append(_record("m1", MULTI, pair=False))
    ledger.append(_record("s1", SINGLE, pair=True))

    summary = summarize(ledger, {MULTI: 1, SINGLE: 1})

    assert summary["prompts_mined_total"] == 2
    assert summary["pairs_total"] == 1, "the tombstoned m1 pair must not be counted"
    assert summary["y_multi"] == 0.0


def test_pstd_is_exact_and_carries_its_own_arithmetic(tmp_path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.append(_record("m1", MULTI, pair=True))
    ledger.append(_record("s1", SINGLE, pair=False))

    summary = summarize(ledger, {MULTI: 8081, SINGLE: 2990})

    # y_multi = 1, y_single = 0 -> y_std = w_multi = 8081/11071
    expected = 10_000 * Fraction(8081, 11071)
    assert summary["P_std_exact"] == f"{expected.numerator}/{expected.denominator}"
    assert summary["P_std"] == pytest.approx(float(expected))


# --- §2.6 decision table, unrounded ------------------------------------------


def test_the_gate_is_evaluated_on_the_exact_value_at_every_boundary() -> None:
    assert "3A" in gate_decision(1000)
    assert "CAUTIOUSLY" in gate_decision(999.6), "999.6 is not >= 1000"
    assert "CAUTIOUSLY" in gate_decision(300)
    assert "3B" in gate_decision(299.99), "299.99 goes to 3B"


def test_an_unestimable_stratum_does_not_silently_produce_a_decision() -> None:
    assert gate_decision(None).startswith("UNDECIDABLE")


# --- Run loop, ledger, and refusals ------------------------------------------


def test_a_run_writes_the_ledger_pairs_and_summary_and_can_resume(tmp_path, monkeypatch) -> None:
    prompts = [_prompt(f"m{i}", MULTI) for i in range(4)] + [
        _prompt(f"s{i}", SINGLE) for i in range(4)
    ]
    calls: list[str] = []

    def generate_fn(prompt: Prompt, n: int) -> list[str]:
        calls.append(prompt.prompt_id)
        return [GOOD] * 4 + [BAD] * 4

    out = tmp_path / "mining_pilot"
    _install_fake_pool(monkeypatch, prompts, "pilot", 4)
    summary = mine_pairs.run(out_dir=out, stage="pilot", generate_fn=generate_fn)

    assert summary["prompts_mined_total"] == 4
    assert summary["pairs_total"] == 4
    assert (out / "ledger.jsonl").exists()
    assert len(json.loads((out / "mining_summary.json").read_text())["allocation"]) == 2
    pairs = [json.loads(x) for x in (out / "mined_pairs.jsonl").read_text().splitlines()]
    assert len(pairs) == 4
    assert all(p["chosen"] == GOOD and p["rejected"] == BAD for p in pairs)

    # Resume: same command, nothing re-sampled.
    first_pass = list(calls)
    mine_pairs.run(out_dir=out, stage="pilot", generate_fn=generate_fn)
    assert calls == first_pass, "a resumed run must not re-sample paid-for work"


def test_the_pilot_summary_says_it_is_not_the_scientific_gate(tmp_path, monkeypatch) -> None:
    summary = _run(tmp_path / "out", monkeypatch)

    assert "CALIBRATION" in summary["gate_note"]
    assert "decision" not in summary, "a pilot must not emit a Phase 2 decision"


def test_fresh_refuses_to_delete_an_existing_ledger(tmp_path, monkeypatch) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "ledger.jsonl").write_text("")

    _install_fake_pool(monkeypatch, [_prompt("m1", MULTI), _prompt("s1", SINGLE)])
    with pytest.raises(MinerError, match="evidence, not scratch space"):
        mine_pairs.run(
            out_dir=out, stage="pilot", generate_fn=_generator([GOOD] * 8), fresh=True
        )


# --- Pins are the frozen ones ------------------------------------------------


def test_the_sampling_pins_are_exactly_section_2_4() -> None:
    assert mine_pairs.SAMPLES_PER_PROMPT == 8
    assert mine_pairs.TEMPERATURE == 0.8
    assert mine_pairs.TOP_P == 1.0
    assert mine_pairs.MAX_NEW_TOKENS == 256
    assert mine_pairs.SEED == 20260804


def test_the_weights_are_amendment_3s_triple_not_the_superseded_one() -> None:
    """A3.3 replaces §2.5's (8173, 2997, 11170) before mining. Reading §2.5 and
    stopping there is the whole failure this test exists to pin."""
    assert (
        mine_pairs.WEIGHT_N_MULTI,
        mine_pairs.WEIGHT_N_SINGLE,
        mine_pairs.WEIGHT_N_TOTAL,
    ) == (8081, 2990, 11071)


def test_the_superseded_four_file_artifact_is_refused_by_digest() -> None:
    """A3.2: the original artifact "may not feed study-2 mining". Not choosing
    it is not the same as refusing it — a future edit could point back."""
    with pytest.raises(MinerError, match="superseded by Amendment 3"):
        mine_pairs.load_eligible_prompts(
            receipt_path=Path("mining/receipts/sft_dedup_v2_decontamination.json")
        )


def test_the_real_pool_re_derives_amendment_3s_exact_survivor_set() -> None:
    """The §2.11 re-assertion against the committed artifacts, end to end.

    This is the check that caught the stale §2.5 pins: it compares a re-derived
    id digest against the one Amendment 3 recorded, so any drift in the pool,
    the screen, the manifest, or the strata parser stops the run before a token
    is generated.
    """
    prompts, receipt = mine_pairs.load_eligible_prompts()

    assert len(prompts) == 11071
    assert receipt["post_screen_id_sha256"] == (
        "69d381413f8095d483b35c9bcd77e83bd6f72771edc9b4f192510f8e7392e5e3"
    )
    assert sum(1 for p in prompts if p.stratum == MULTI) == 8081
    assert sum(1 for p in prompts if p.stratum == SINGLE) == 2990


def test_the_adapter_pin_is_section_2_1s(tmp_path) -> None:
    assert mine_pairs.SFT_ADAPTER_REPO == "centuriandip/llama-3.1-8b-tools-sft"
    assert mine_pairs.SFT_ADAPTER_SUBFOLDER == "adapter/"
    assert mine_pairs.SFT_ADAPTER_REVISION == "b6f4da479f8c6fc044ee8b802a92f47780f970c5"


def test_a_moved_pool_digest_refuses_before_any_generation(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(mine_pairs, "POOL_SHA256", "0" * 64)

    with pytest.raises(MinerError, match="sha256"):
        mine_pairs.load_eligible_prompts()


def test_a_missing_artifact_refuses() -> None:
    with pytest.raises(MinerError, match="is missing"):
        mine_pairs.load_eligible_prompts(
            receipt_path=Path("mining/receipts/does_not_exist.json")
        )


# --- Review cycle 1: run identity, derivatives, manifest, schema -------------


def _install_fake_pool(monkeypatch, prompts, stage="pilot", n=2):
    """Substitute the loader itself. There is no injection seam in run()."""
    monkeypatch.setattr(
        mine_pairs, "load_eligible_prompts", lambda *a, **k: (list(prompts), _RECEIPT)
    )
    monkeypatch.setattr(mine_pairs, "STAGES", {**mine_pairs.STAGES, stage: n})


def _run(out, monkeypatch, stage="pilot", n=2, prompts=None, gen=None):
    prompts = prompts or [_prompt("m1", MULTI), _prompt("s1", SINGLE)]
    _install_fake_pool(monkeypatch, prompts, stage, n)
    return mine_pairs.run(
        out_dir=out,
        stage=stage,
        generate_fn=gen or _generator([GOOD] * 4 + [BAD] * 4),
    )


def test_a_directory_refuses_a_second_run_of_a_different_shape(tmp_path, monkeypatch) -> None:
    """Codex's repro: the same directory accepted n=2 then n=4 and produced one
    four-record artifact that no stated design asked for."""
    out = tmp_path / "out"
    prompts = [_prompt(f"m{i}", MULTI) for i in range(2)] + [
        _prompt(f"s{i}", SINGLE) for i in range(2)
    ]
    _run(out, monkeypatch, n=2, prompts=prompts)

    with pytest.raises(MinerError, match="describes a different run"):
        _run(out, monkeypatch, n=4, prompts=prompts)


def test_a_pilot_directory_will_not_accept_a_calibration_run(tmp_path, monkeypatch) -> None:
    out = tmp_path / "out"
    _run(out, monkeypatch, stage="pilot", n=2)

    with pytest.raises(MinerError, match=r"separate\s+evidence chains"):
        _run(out, monkeypatch, stage="calibration", n=2)


def test_run_metadata_pins_every_input_that_could_move(tmp_path, monkeypatch) -> None:
    out = tmp_path / "out"
    _run(out, monkeypatch)
    meta = json.loads((out / "run.json").read_text())

    assert meta["stage"] == "pilot"
    assert meta["pool_sha256"] == mine_pairs.POOL_SHA256
    assert meta["manifest_sha256"] == mine_pairs.MANIFEST_SHA256
    assert meta["decontamination_receipt_sha256"] == mine_pairs.DECON_SHA256
    assert meta["weights"]["N"] == 11071
    assert meta["adapter"]["revision"] == mine_pairs.SFT_ADAPTER_REVISION
    assert meta["verifier"]["selftest_version"] == mine_pairs.VERIFIER_VERSION
    assert meta["verifier"]["module_sha256"], "a version string cannot detect code drift"
    assert meta["verifier"]["selftest"]["receipt_sha256"]
    assert meta["verifier"]["selftest"]["pairs_passed"] == 1600
    assert len(meta["verifier"]["selftest"]["fixtures"]) == 2
    assert meta["miner_sha256"]
    assert meta["base_model"]["revision"] == mine_pairs.BASE_MODEL_REVISION
    assert meta["allocation"] == {MULTI: 1, SINGLE: 1}
    assert meta["selected_id_sha256"]
    # Sampling order is part of what a seeded run promises to reproduce.
    assert meta["selected_ids_ordered_sha256"]


def test_a_rollback_re_materializes_the_derived_files(tmp_path, monkeypatch) -> None:
    """Codex's second repro: after redo_last(1) the files still reported the
    rolled-back work, so the summary contradicted its own ledger."""
    out = tmp_path / "out"
    prompts = [_prompt(f"m{i}", MULTI) for i in range(2)] + [
        _prompt(f"s{i}", SINGLE) for i in range(2)
    ]
    _run(out, monkeypatch, n=4, prompts=prompts)
    assert json.loads((out / "mining_summary.json").read_text())["pairs_total"] == 4

    ledger = Ledger(out / "ledger.jsonl")
    ledger.redo_last(1)
    mine_pairs.write_derivatives(out, ledger, {MULTI: 2, SINGLE: 2}, "pilot")

    summary = json.loads((out / "mining_summary.json").read_text())
    assert summary["prompts_mined_total"] == 3
    assert summary["pairs_total"] == 3
    assert len((out / "mined_pairs.jsonl").read_text().strip().splitlines()) == 3


def test_fresh_refuses_when_any_evidence_file_exists(tmp_path, monkeypatch) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "mining_summary.json").write_text("{}")

    _install_fake_pool(monkeypatch, [_prompt("m1", MULTI), _prompt("s1", SINGLE)])
    with pytest.raises(MinerError, match=re.escape("mining_summary.json")):
        mine_pairs.run(
            out_dir=out, stage="pilot", generate_fn=_generator([GOOD] * 8), fresh=True
        )


def test_a3_5_requires_the_manifest_digest_checked_directly(monkeypatch) -> None:
    monkeypatch.setattr(mine_pairs, "MANIFEST_SHA256", "0" * 64)

    with pytest.raises(MinerError, match="amended manifest sha256"):
        mine_pairs.load_eligible_prompts()


def test_a_mined_pair_carries_the_prompt_the_trainer_needs(tmp_path, monkeypatch) -> None:
    """Producer -> consumer: every DPO loader in the repo keys on
    `prompt_messages`. Two completion strings are not a trainable row."""
    out = tmp_path / "out"
    _run(out, monkeypatch)
    rows = [json.loads(x) for x in (out / "mined_pairs.jsonl").read_text().splitlines()]

    assert rows
    for row in rows:
        roles = [m["role"] for m in row["prompt_messages"]]
        assert roles == ["system", "user"]
        assert all(m["content"] for m in row["prompt_messages"])
        assert row["chosen"] and row["rejected"]


def test_the_histogram_and_sft_bucket_are_recomputed_from_the_ledger(tmp_path) -> None:
    """Phase 1.3 needs the pass histogram; prereg §3B consumes the 0-of-8 bucket."""
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.append(
        {**_record("m1", MULTI, pair=True), "prompt_messages": [], "target": "t"}
    )
    ledger.append(
        {
            "prompt_id": "m2",
            "stratum": MULTI,
            "generations": [BAD, BAD],
            "verdicts": [{"accepted": False}, {"accepted": False}],
            "prompt_messages": [{"role": "user", "content": "u"}],
            "target": "the ground truth",
        }
    )
    ledger.append(
        {
            "prompt_id": "s1",
            "stratum": SINGLE,
            "generations": [GOOD, GOOD],
            "verdicts": [{"accepted": True}, {"accepted": True}],
            "prompt_messages": [],
            "target": "t",
        }
    )

    summary = summarize(ledger, {MULTI: 1, SINGLE: 1})

    assert summary["pass_histogram"]["0"] == 1
    assert summary["pass_histogram"]["2"] == 1
    assert summary["discarded_all_correct"] == 1, "8-of-8 prompts are discarded, not paired"
    assert [r["prompt_id"] for r in summary["sft_bucket"]] == ["m2"]
    assert summary["sft_bucket"][0]["target"] == "the ground truth"


def test_the_gate_is_decided_on_the_exact_rational_not_the_float(tmp_path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.append(_record("m1", MULTI, pair=True))
    ledger.append(_record("s1", SINGLE, pair=True))

    summary = summarize(ledger, {MULTI: 1, SINGLE: 1})
    exact = summary["_P_std_fraction"]

    assert isinstance(exact, Fraction)
    assert exact == 10_000  # y_std == 1
    assert gate_decision(exact) == gate_decision(Fraction(10_000))


def test_a_failing_selftest_stops_the_run_before_any_generation(tmp_path, monkeypatch) -> None:
    """§5.1: the fixture gate runs before every mining session."""

    class Failed:
        version = "onpolicy_verifier_v1"
        pairs = 1600
        pairs_passed = 1599
        passed = False

    monkeypatch.setattr(mine_pairs, "run_selftest", lambda: Failed())
    called: list[str] = []

    def generate_fn(prompt, n):
        called.append(prompt.prompt_id)
        return [GOOD] * 8

    with pytest.raises(MinerError, match="fixture self-test failed"):
        _run(tmp_path / "out", monkeypatch, gen=generate_fn)

    assert called == [], "no prompt may be sampled behind a failed gate"


def test_an_unknown_stage_is_refused(tmp_path) -> None:
    with pytest.raises(MinerError, match="stage must be one of"):
        mine_pairs.run(
            out_dir=tmp_path / "out",
            stage="exploratory",
            generate_fn=_generator([GOOD] * 8),
        )


# --- Owner decision A+C: guardrails measured, never applied ------------------


def test_the_guardrail_caps_are_measured_and_never_filter(tmp_path) -> None:
    """Option C. If these ever start excluding pairs they move the §2.6 gate on
    thresholds that are not in the frozen preregistration."""
    long_chosen = (
        '<tool_call>\n{"name": "do_thing", "arguments": {"x": ' + "1" * 400 + "}}\n</tool_call>"
    )
    outcome_pairs = [
        {"chosen": long_chosen, "rejected": BAD},          # length gap way over 40%
        # Genuinely malformed syntax (unclosed brace), near-identical length so
        # it trips the malformed check alone. Plain prose would be a `no_call`,
        # which is a different thing entirely.
        {"chosen": GOOD, "rejected": MALFORMED},
        {"chosen": GOOD, "rejected": BAD},                  # unremarkable
    ]

    diagnostics = mine_pairs.pair_diagnostics(outcome_pairs)

    assert diagnostics["pairs"] == 3, "every pair is still present; nothing was filtered"
    assert diagnostics["length_gap_over_reference"] == 1
    assert diagnostics["malformed_rejected"] == 1
    assert diagnostics["would_either_cap_have_bound"] is True
    assert "NOT applied" in diagnostics["note"]


def test_diagnostics_ride_along_without_changing_yield(tmp_path, monkeypatch) -> None:
    out = tmp_path / "out"
    summary = _run(out, monkeypatch)

    assert summary["guardrail_diagnostics"]["pairs"] == summary["pairs_total"]
    written = json.loads((out / "mining_summary.json").read_text())
    assert written["guardrail_diagnostics"]["length_gap_reference"] == 0.40


def test_length_gap_diagnostic_honours_the_call_vs_text_exemption() -> None:
    pairs = [
        {
            "chosen": GOOD * 20,
            "rejected": "plain prose",
            "rejected_reason": mine_pairs.MISSING_CALL,
        }
    ]

    diagnostics = mine_pairs.pair_diagnostics(pairs)

    assert diagnostics["length_gap_over_reference"] == 0
    assert diagnostics["length_gap_exempt_call_vs_text"] == 1
    assert diagnostics["would_either_cap_have_bound"] is False


# --- Review cycle 2 ----------------------------------------------------------


def test_run_has_no_seam_that_skips_the_pinned_preflight(monkeypatch, tmp_path) -> None:
    """Codex's repro: an injectable prompts/receipt let a run succeed while
    `run.json` recorded pinned digests no preflight had ever checked."""

    def exploding_loader(*args, **kwargs):
        raise MinerError("preflight ran")

    monkeypatch.setattr(mine_pairs, "load_eligible_prompts", exploding_loader)

    with pytest.raises(MinerError, match="preflight ran"):
        mine_pairs.run(
            out_dir=tmp_path / "out", stage="pilot", generate_fn=_generator([GOOD] * 8)
        )


def test_the_committed_selftest_receipt_is_verified_before_generation(
    monkeypatch, tmp_path
) -> None:
    _install_fake_pool(monkeypatch, [_prompt("m1", MULTI), _prompt("s1", SINGLE)])
    monkeypatch.setattr(mine_pairs, "VERIFIER_SELFTEST_RECEIPT_SHA256", "0" * 64)
    called: list[str] = []

    def generate_fn(prompt, n):
        called.append(prompt.prompt_id)
        return [GOOD] * 8

    with pytest.raises(MinerError, match="verifier self-test receipt sha256"):
        mine_pairs.run(tmp_path / "out", "pilot", generate_fn)

    assert called == []


def test_the_stage_fixes_the_prompt_count(monkeypatch, tmp_path) -> None:
    """`STAGES` must bind n, not merely suggest it."""
    seen: list[int] = []
    prompts = [_prompt(f"m{i}", MULTI) for i in range(200)] + [
        _prompt(f"s{i}", SINGLE) for i in range(200)
    ]
    monkeypatch.setattr(
        mine_pairs, "load_eligible_prompts", lambda *a, **k: (prompts, _RECEIPT)
    )

    def generate_fn(prompt, n):
        seen.append(n)
        return [GOOD] * 4 + [BAD] * 4

    mine_pairs.run(out_dir=tmp_path / "out", stage="pilot", generate_fn=generate_fn)

    assert len(seen) == mine_pairs.STAGES["pilot"] == 100


# --- Multi-turn rows: split at the FIRST assistant turn ----------------------


def test_a_two_assistant_row_targets_only_the_first_turn() -> None:
    """439 of the 11,071 survivors carry two assistant turns. Joining them
    produced a ground truth no row ever taught."""
    row = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "FIRST"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "SECOND"},
        ]
    }

    prompt_messages, target = mine_pairs.split_at_first_assistant(row)

    assert target == "FIRST", "never FIRST + SECOND concatenated"
    assert [m["role"] for m in prompt_messages] == ["system", "user"]
    assert [m["content"] for m in prompt_messages] == ["sys", "u1"]


def test_a_tool_prefixed_row_keeps_the_tool_message_in_its_prompt() -> None:
    """7 survivors place a `tool` message before their first assistant turn.
    Flattening to system+user drops context the completion depends on."""
    row = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u"},
            {"role": "tool", "content": "tool output"},
            {"role": "assistant", "content": "answer"},
        ]
    }

    prompt_messages, target = mine_pairs.split_at_first_assistant(row)

    assert target == "answer"
    assert [m["role"] for m in prompt_messages] == ["system", "user", "tool"]
    assert prompt_messages[2]["content"] == "tool output"


def test_the_real_pool_has_the_role_shapes_codex_measured() -> None:
    prompts, _receipt = mine_pairs.load_eligible_prompts()

    tool_prefixed = [p for p in prompts if any(r == "tool" for r, _ in p.messages)]
    assert len(tool_prefixed) == 7
    # Nothing before a first assistant turn can itself be an assistant turn.
    assert not [p for p in prompts if any(r == "assistant" for r, _ in p.messages)]


# --- Redo and derivative metadata -------------------------------------------


def test_the_summary_on_disk_carries_the_sft_bucket_count(tmp_path, monkeypatch) -> None:
    out = tmp_path / "out"
    _run(out, monkeypatch, gen=_generator([BAD] * 8))

    written = json.loads((out / "mining_summary.json").read_text())
    # 0-of-8 prompts, counted into the summary before it is serialized.
    assert written["sft_bucket_rows"] == 2


def test_a_rollback_does_not_delete_the_recorded_allocation(tmp_path, monkeypatch) -> None:
    out = tmp_path / "out"
    _run(out, monkeypatch)
    ledger = Ledger(out / "ledger.jsonl")
    ledger.redo_last(1)

    # allocation omitted, exactly as the CLI redo path calls it
    mine_pairs.write_derivatives(out, ledger, {MULTI: 1, SINGLE: 1}, "pilot")

    written = json.loads((out / "mining_summary.json").read_text())
    assert written["allocation"] == {MULTI: 1, SINGLE: 1}, "read back from run.json"


def test_redo_refuses_identity_drift_before_appending_a_tombstone(
    tmp_path, monkeypatch
) -> None:
    out = tmp_path / "out"
    _run(out, monkeypatch)
    before = Ledger(out / "ledger.jsonl").records()

    # Simulate invoking rollback under different miner bytes. The immutable run
    # identity must be checked in full before redo's permanent control record.
    monkeypatch.setattr(mine_pairs, "MINER_SOURCE", mine_pairs.VERIFIER_SOURCE)
    with pytest.raises(MinerError, match="different run"):
        mine_pairs.redo_run(out, 1)

    assert Ledger(out / "ledger.jsonl").records() == before
