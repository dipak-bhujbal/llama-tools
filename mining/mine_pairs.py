"""Production on-policy preference-pair miner (prereg §2).

Samples the shipped SFT policy on the decontaminated curated pool, verifies each
generation against the pool row's own target turn, and materializes preference
pairs. Every parameter it runs under is frozen in `docs/prereg-study2.md` §2 and
pinned here as a constant, so a run cannot quietly drift from the registered
design.

**This is a rewrite, not a promotion.** The quarantined `mine_pairs.py` assigned
`no_call` ground truth to any target it could not parse, which mints preference
pairs whose "chosen" is wrong — inverted training signal, produced silently.
§2.11 replaces that fallback with a refusal: a row the miner cannot read is a
stop condition, never a row it gets to reinterpret. That rule is enforced in
`_target_turn()` and is the reason this module raises where the old one guessed.

**Yield is recomputed from the ledger, never accumulated while running**
(Amendment 2 A2.1). `summarize()` re-materializes pairs from the ledger's
*active* records through the same deterministic path that produced them, so a
reported yield is checkable by anyone holding the artifact, and a rolled-back
batch cannot inflate it. A counter incremented during the run would satisfy
neither property.

**No model is imported at module scope.** Generation enters through a
`generate_fn` callable, so the whole path — screening, allocation, selection,
verification, pair construction, summary arithmetic — is exercisable in tests
at $0 with no GPU, no weights, and no network.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

from mining.decontaminate import Decontaminator
from mining.ledger import CONTROL_TYPES, Ledger
from mining.pool_strata import (
    INELIGIBLE,
    MULTI,
    SINGLE,
    classify_target,
    presented_names,
    stratum_of,
)
from mining.verifier import (
    VERIFIER_VERSION,
    ParserDisagreementError,
    TargetUnreadableError,
    extract_calls,
    run_selftest,
    verify,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# --- Pins from the frozen preregistration. Not defaults; not overridable. -----

# §2.7 mining population
POOL_PATH = Path("data/processed/sft_dedup_v2.jsonl")
POOL_SHA256 = "9e5b7b4f3a5990b5c92e1d5f1b84d8664a9cce006f88087db7cc7219ffe76b2b"

# Decontamination artifact — Amendment 3 A3.2, NOT §2.5's original.
#
# §2.5 pins `sft_dedup_v2_decontamination.json` (`fb7a020…`, four-file screen).
# Amendment 3 replaced it with the five-file screen that includes the full
# 1,053-row `live_multiple` parent, and says the original "is now **superseded
# and may not feed study-2 mining**". Reading §2.5 alone and stopping there
# yields a miner that screens against four of five question files and mines 99
# prompts it should not.
DECON_RECEIPT = Path("mining/receipts/sft_dedup_v2_decontamination_with_live_multiple.json")
DECON_SHA256 = "3daaffa85a2097468f53845d1cddf996a0e68a3605916e26918891c2972732b3"

# The superseded artifact, pinned so it can be refused by name rather than
# merely not chosen.
SUPERSEDED_DECON_SHA256 = "fb7a0200dbeeabb831006eeb800a23d3c92d89a468666c61b098ca1277231906"

# A3.5 requires the miner to verify three digests before the first prompt is
# mined: the amended manifest, the current receipt, and the post-screen id set.
# Re-deriving survivors proves the last two transitively but never checks the
# manifest directly — a manifest could drift and still screen to a set whose
# digest matched, if the drift touched a file the screen does not read.
MANIFEST_PATH = Path("eval/manifests/bfcl_v4_study2.json")
MANIFEST_SHA256 = "542d407d434655487daa3faa0da69666cc5e5fa47c8ff67ab9771acc512fe3a0"

# §2.4 fixes the shape of each stage. Binding them here stops a pilot ledger and
# a calibration ledger from ever becoming one artifact.
STAGES = {"pilot": 100, "calibration": 1000}
RUN_METADATA_NAME = "run.json"

# §2.1 required inputs (closes roadmap H1.1 — the owner supplies nothing further)
SFT_ADAPTER_REPO = "centuriandip/llama-3.1-8b-tools-sft"
SFT_ADAPTER_SUBFOLDER = "adapter/"
SFT_ADAPTER_REVISION = "b6f4da479f8c6fc044ee8b802a92f47780f970c5"

# §2.4 sampling parameters
SAMPLES_PER_PROMPT = 8
TEMPERATURE = 0.8
TOP_P = 1.0
MAX_NEW_TOKENS = 256
SEED = 20260804

# Target weights as exact integers — Amendment 3 A3.3, which replaces §2.5's
# (8173, 2997, 11170) before mining. The multi share (8081/11071 = 72.993%) is
# derived at display time and never stored, so P_std uses exact ratios.
WEIGHT_N_MULTI = 8081
WEIGHT_N_SINGLE = 2990
WEIGHT_N_TOTAL = 11071

# §2.6 decision table, applied to P_std unrounded.
GATE_PROCEED = 1000
GATE_CAUTIOUS = 300


class MinerError(RuntimeError):
    """The run cannot be trusted, so no generation or pair may be produced."""


@dataclass(frozen=True)
class Prompt:
    """One post-screen pool row the miner is allowed to sample."""

    prompt_id: str
    stratum: str
    system: str
    user: str
    target: str


@dataclass
class PromptOutcome:
    """What one prompt produced: the ledger record's payload."""

    prompt_id: str
    stratum: str
    generations: list[str] = field(default_factory=list)
    verdicts: list[dict[str, Any]] = field(default_factory=list)
    prompt_messages: list[dict[str, str]] = field(default_factory=list)
    target: str = ""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _id_digest(ids: Iterable[str]) -> str:
    """Byte-identical to `decontaminate_pool._id_digest`, deliberately.

    The whole point of re-deriving the survivor set is to compare it against the
    digest the artifact recorded; computing it a different way here would make
    the comparison vacuous.
    """
    return _sha256(("\n".join(sorted(ids)) + "\n").encode())


def _verify_pin(path: Path, expected: str, label: str) -> bytes:
    resolved = REPO_ROOT / path
    if not resolved.exists():
        raise MinerError(f"{label} is missing at {path}; nothing may be mined")
    payload = resolved.read_bytes()
    actual = _sha256(payload)
    if actual != expected:
        raise MinerError(f"{label} sha256 {actual} != pinned {expected}")
    return payload


def _message(row: dict[str, Any], role: str) -> str:
    return " ".join(
        m.get("content", "") for m in row.get("messages", []) if m.get("role") == role
    )


def _target_turn(row: dict[str, Any], prompt_id: str) -> str:
    """The assistant turn this row teaches, or a refusal.

    §2.11: a target the miner cannot read is a **stop condition**. The
    quarantined miner turned this exact branch into `no_call` ground truth,
    which produces a preference pair whose chosen answer is wrong.
    """
    assistant = _message(row, "assistant")
    kind, _ = classify_target(assistant)
    if kind == "unreadable":
        raise MinerError(
            f"{prompt_id}: target turn does not parse under "
            f"pool-target-structural-eligibility/v1. §2.11 refuses the run rather "
            f"than reclassifying it — this row was screened as readable, so the "
            f"pool or the parser has drifted"
        )
    return assistant


def load_eligible_prompts(
    pool_path: Path = POOL_PATH,
    manifest_path: Path = MANIFEST_PATH,
    receipt_path: Path = DECON_RECEIPT,
) -> tuple[list[Prompt], dict[str, Any]]:
    """Re-derive the post-screen population and prove it matches the artifact.

    §2.11 — "checked twice, trusted once". The artifact records the *digest* of
    its survivor ids, not the ids themselves, so re-deriving them and comparing
    digests is the only check that can actually fail. Any drift in the pool, the
    screen, or the strata parser moves the digest and stops the run here, before
    a single token is generated.
    """
    resolved_receipt = REPO_ROOT / receipt_path
    if (
        resolved_receipt.exists()
        and _sha256(resolved_receipt.read_bytes()) == SUPERSEDED_DECON_SHA256
    ):
        raise MinerError(
            f"{receipt_path} is §2.5's four-file artifact, superseded by Amendment 3 "
            f"A3.2 and explicitly barred from feeding study-2 mining. Use "
            f"{DECON_RECEIPT}"
        )
    receipt_bytes = _verify_pin(receipt_path, DECON_SHA256, "decontamination artifact")
    receipt = json.loads(receipt_bytes)

    # A3.5's first required digest, checked directly rather than inferred.
    _verify_pin(manifest_path, MANIFEST_SHA256, "amended manifest")
    if receipt["manifest"]["sha256"] != MANIFEST_SHA256:
        raise MinerError(
            f"artifact was screened against manifest {receipt['manifest']['sha256']}, "
            f"not the amended {MANIFEST_SHA256}"
        )

    pool_bytes = _verify_pin(pool_path, POOL_SHA256, "mining pool")
    if receipt["source"]["sha256"] != POOL_SHA256:
        raise MinerError(
            f"artifact was built over pool {receipt['source']['sha256']}, "
            f"not the pinned {POOL_SHA256}"
        )

    screener = Decontaminator([REPO_ROOT / manifest_path])
    survivors: list[Prompt] = []
    survivor_ids: list[str] = []

    for index, line in enumerate(pool_bytes.decode("utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        prompt_id = str(row.get("source_id") or f"line:{index + 1}")
        system = _message(row, "system")

        stratum, _ = stratum_of(system)
        if stratum == INELIGIBLE:
            continue

        assistant = _message(row, "assistant")
        kind, called = classify_target(assistant)
        if kind == "unreadable":
            continue

        presented = presented_names(system)
        missing = sorted(called - presented)
        if missing:
            raise MinerError(
                f"{prompt_id}: retained row calls {missing}, which its prompt does not "
                f"present; a name defect is a defect, not an exclusion"
            )

        user = _message(row, "user")
        hit, _reason = screener.is_contaminated(user, sorted(presented))
        if hit:
            continue

        survivor_ids.append(prompt_id)
        survivors.append(
            Prompt(
                prompt_id=prompt_id,
                stratum=stratum,
                system=system,
                user=user,
                target=_target_turn(row, prompt_id),
            )
        )

    digest = _id_digest(survivor_ids)
    if digest != receipt["post_screen_id_sha256"]:
        raise MinerError(
            f"re-derived post-screen id digest {digest} != artifact's "
            f"{receipt['post_screen_id_sha256']}; the pool, the screen, or the "
            f"strata parser has drifted since the artifact was built"
        )

    counts = {
        MULTI: sum(1 for p in survivors if p.stratum == MULTI),
        SINGLE: sum(1 for p in survivors if p.stratum == SINGLE),
    }
    if (
        counts[MULTI] != receipt["survivors"]["multi"]
        or counts[SINGLE] != receipt["survivors"]["single"]
    ):
        raise MinerError(
            f"re-derived strata {counts} disagree with the artifact's "
            f"{receipt['survivors']}"
        )
    return survivors, receipt


def allocate(n_prompts: int, counts: dict[str, int]) -> dict[str, int]:
    """Proportional allocation across strata, both strata nonzero (§2.3).

    Largest-remainder rounding, then a floor of one per stratum: §2.3 requires
    both `y_multi` and `y_single` to be estimable, so a proportional split that
    rounds a stratum to zero would make one of them undefined and the other
    assumed from it.
    """
    if n_prompts < 2:
        raise MinerError(
            f"--n-prompts {n_prompts} cannot give both strata a nonzero allocation (§2.3)"
        )
    total = counts[MULTI] + counts[SINGLE]
    if total == 0:
        raise MinerError("no eligible prompts to allocate")

    exact = {s: Fraction(counts[s] * n_prompts, total) for s in (MULTI, SINGLE)}
    floors = {s: int(exact[s]) for s in exact}
    remainder = n_prompts - sum(floors.values())
    for stratum in sorted(exact, key=lambda s: (-(exact[s] - floors[s]), s))[:remainder]:
        floors[stratum] += 1

    for stratum in (MULTI, SINGLE):
        if floors[stratum] == 0:
            donor = MULTI if stratum == SINGLE else SINGLE
            if floors[donor] < 2:
                raise MinerError("cannot give both strata a nonzero allocation")
            floors[donor] -= 1
            floors[stratum] = 1

    for stratum in (MULTI, SINGLE):
        if floors[stratum] > counts[stratum]:
            raise MinerError(
                f"allocation wants {floors[stratum]} {stratum} prompts but only "
                f"{counts[stratum]} exist"
            )
    return floors


def select_prompts(
    prompts: Sequence[Prompt], allocation: dict[str, int], seed: int = SEED
) -> list[Prompt]:
    """Seeded, composition-preserving selection. No RNG or library shuffle.

    Ordering is by `sha256(seed:prompt_id)` — the same device §3.2's development
    subset uses — so the selection is reproducible from the seed alone by anyone
    holding the pool, rather than from a random state nobody can recover.
    """
    chosen: list[Prompt] = []
    for stratum in (MULTI, SINGLE):
        pool = [p for p in prompts if p.stratum == stratum]
        pool.sort(key=lambda p: _sha256(f"{seed}:{p.prompt_id}".encode()))
        chosen.extend(pool[: allocation[stratum]])
    return chosen


def _atomic_write(path: Path, text: str) -> None:
    """Write via temp + fsync + replace.

    A derived file written in place can be observed half-written after a crash,
    and a half-written `mining_summary.json` is a yield figure nobody can tell is
    partial. The ledger already survives crashes this way; its derivatives have
    to as well or the artifact set is only as trustworthy as its weakest file.
    """
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def run_metadata(
    stage: str,
    n_prompts: int,
    selected: Sequence[Prompt],
    receipt: dict[str, Any],
    selftest_version: str,
) -> dict[str, Any]:
    """The immutable identity of a mining run.

    Resume works by skipping prompt ids already in the ledger, which is only
    safe if the resumed run is *the same run*. Without this, one directory
    accepts `--n-prompts 2` and then `--n-prompts 4` and produces a single
    four-record artifact that no stated design ever asked for.
    """
    return {
        "stage": stage,
        "n_prompts": n_prompts,
        "selected_id_sha256": _id_digest(p.prompt_id for p in selected),
        "pool_sha256": POOL_SHA256,
        "decontamination_receipt_sha256": DECON_SHA256,
        "manifest_sha256": MANIFEST_SHA256,
        "post_screen_id_sha256": receipt["post_screen_id_sha256"],
        "weights": {
            "n_multi": WEIGHT_N_MULTI,
            "n_single": WEIGHT_N_SINGLE,
            "N": WEIGHT_N_TOTAL,
        },
        "sampling": {
            "samples": SAMPLES_PER_PROMPT,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_new_tokens": MAX_NEW_TOKENS,
            "seed": SEED,
        },
        "adapter": {
            "repo": SFT_ADAPTER_REPO,
            "subfolder": SFT_ADAPTER_SUBFOLDER,
            "revision": SFT_ADAPTER_REVISION,
        },
        "verifier_version": VERIFIER_VERSION,
        "verifier_selftest_version": selftest_version,
    }


def bind_run_metadata(out_dir: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    """Create the run identity once, or prove the resumed run matches it."""
    path = out_dir / RUN_METADATA_NAME
    if not path.exists():
        _atomic_write(path, json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        return metadata

    existing = json.loads(path.read_text())
    differing = sorted(
        k for k in set(existing) | set(metadata) if existing.get(k) != metadata.get(k)
    )
    if differing:
        raise MinerError(
            f"{path} describes a different run; refusing to extend it. Differing "
            f"fields: {differing}. A pilot and a calibration run are separate "
            f"evidence chains and never share a directory"
        )
    return existing


def materialize_pair(outcome: PromptOutcome) -> dict[str, Any] | None:
    """Deterministically build this prompt's preference pair, or None.

    A pair needs one accepted and one rejected generation from the same prompt.
    Both are taken at the **lowest sample index** of their kind so that the same
    ledger record always materializes the same pair — that determinism is what
    makes A2.1's yield recomputable rather than merely reported.

    At most one pair per prompt: `P_std` is denominated in pairs per 10,000
    post-screen prompts, so letting a lucky prompt contribute several would make
    the gate read a sampling artifact as pool yield.
    """
    accepted = next((i for i, v in enumerate(outcome.verdicts) if v["accepted"]), None)
    rejected = next((i for i, v in enumerate(outcome.verdicts) if not v["accepted"]), None)
    if accepted is None or rejected is None:
        return None
    return {
        "prompt_id": outcome.prompt_id,
        "stratum": outcome.stratum,
        # The trainer consumes this file directly, and every DPO loader in the
        # repo keys on `prompt_messages`. A pair carrying only two completion
        # strings is not a trainable row — it is two strings.
        "prompt_messages": outcome.prompt_messages,
        "chosen": outcome.generations[accepted],
        "rejected": outcome.generations[rejected],
        "chosen_index": accepted,
        "rejected_index": rejected,
        "rejected_reason": outcome.verdicts[rejected].get("reason"),
        "verifier_version": VERIFIER_VERSION,
    }


def mine_prompt(prompt: Prompt, generate_fn: Callable[[Prompt, int], list[str]]) -> PromptOutcome:
    """Sample one prompt and verify every generation against its own target."""
    generations = generate_fn(prompt, SAMPLES_PER_PROMPT)
    if len(generations) != SAMPLES_PER_PROMPT:
        raise MinerError(
            f"{prompt.prompt_id}: generator returned {len(generations)} samples, "
            f"expected {SAMPLES_PER_PROMPT} (§2.4)"
        )

    verdicts: list[dict[str, Any]] = []
    for text in generations:
        try:
            verdict = verify(text, prompt.target)
        except TargetUnreadableError as exc:
            # Same refusal as _target_turn(), reached if the target degrades
            # between screening and verification. Never a no_call fallback.
            raise MinerError(
                f"{prompt.prompt_id}: target became unreadable mid-run: {exc}"
            ) from exc
        except ParserDisagreementError as exc:
            raise MinerError(f"{prompt.prompt_id}: parsers disagree on the target: {exc}") from exc
        verdicts.append(
            {"accepted": verdict.accepted, "reason": verdict.reason, "detail": verdict.detail}
        )
    return PromptOutcome(
        prompt_id=prompt.prompt_id,
        stratum=prompt.stratum,
        generations=generations,
        verdicts=verdicts,
        prompt_messages=[
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": prompt.user},
        ],
        target=prompt.target,
    )


def summarize(ledger: Ledger, survivor_counts: dict[str, int]) -> dict[str, Any]:
    """§2.6 gate arithmetic, recomputed from the ledger's active records (A2.1).

    Nothing here is carried forward from the run loop. The numerator is pairs
    re-materialized from active records; the denominator is unique prompts
    bearing an active record. A tombstoned prompt leaves both terms together and
    returns to both when re-mined, so yield never goes stale in one term only.
    """
    records = ledger.records()
    # Filter by `seq`, not by `prompt_id`. A rolled-back-and-re-mined prompt has
    # two records sharing one id, and the id is active because the *second* one
    # is; selecting by id would materialize the tombstoned first record and
    # report a yield the ledger does not support.
    superseded = {r["supersedes_seq"] for r in records if r.get("type") == "redo"}
    active = [
        r for r in records
        if r.get("type") not in CONTROL_TYPES and r["seq"] not in superseded
    ]

    mined = {MULTI: 0, SINGLE: 0}
    pairs = {MULTI: 0, SINGLE: 0}
    materialized: list[dict[str, Any]] = []

    histogram: dict[str, int] = {str(k): 0 for k in range(SAMPLES_PER_PROMPT + 1)}
    discarded_all_correct = 0
    sft_bucket: list[dict[str, Any]] = []

    for record in active:
        stratum = record["stratum"]
        mined[stratum] += 1
        outcome = PromptOutcome(
            prompt_id=record["prompt_id"],
            stratum=stratum,
            generations=record["generations"],
            verdicts=record["verdicts"],
            prompt_messages=record.get("prompt_messages", []),
            target=record.get("target", ""),
        )

        # Phase 1.3 requires the pass histogram; §3B consumes the 0-of-8 bucket.
        # Both are recomputed here rather than counted during the run, for the
        # same reason yield is: a number derived from the ledger is checkable.
        n_accepted = sum(1 for v in outcome.verdicts if v["accepted"])
        histogram[str(n_accepted)] += 1
        if outcome.verdicts and n_accepted == len(outcome.verdicts):
            discarded_all_correct += 1
        if outcome.verdicts and n_accepted == 0:
            sft_bucket.append(
                {
                    "prompt_id": outcome.prompt_id,
                    "stratum": stratum,
                    "prompt_messages": outcome.prompt_messages,
                    "target": outcome.target,
                }
            )

        pair = materialize_pair(outcome)
        if pair is not None:
            pairs[stratum] += 1
            materialized.append(pair)

    # Exact ratios throughout: §2.5 stores integers precisely so the gate is not
    # decided by a rounded intermediate.
    y = {
        s: (Fraction(pairs[s], mined[s]) if mined[s] else None)
        for s in (MULTI, SINGLE)
    }
    if y[MULTI] is None or y[SINGLE] is None:
        y_std = None
        p_std = None
    else:
        w_multi = Fraction(WEIGHT_N_MULTI, WEIGHT_N_TOTAL)
        w_single = Fraction(WEIGHT_N_SINGLE, WEIGHT_N_TOTAL)
        y_std = w_multi * y[MULTI] + w_single * y[SINGLE]
        p_std = 10_000 * y_std

    return {
        "verifier_version": VERIFIER_VERSION,
        "pass_histogram": histogram,
        "discarded_all_correct": discarded_all_correct,
        "sft_bucket": sft_bucket,
        "prompts_mined": {s: mined[s] for s in mined},
        "pairs": {s: pairs[s] for s in pairs},
        "prompts_mined_total": mined[MULTI] + mined[SINGLE],
        "pairs_total": pairs[MULTI] + pairs[SINGLE],
        "y_multi": float(y[MULTI]) if y[MULTI] is not None else None,
        "y_single": float(y[SINGLE]) if y[SINGLE] is not None else None,
        "y_std": float(y_std) if y_std is not None else None,
        "P_std": float(p_std) if p_std is not None else None,
        "P_std_exact": f"{p_std.numerator}/{p_std.denominator}" if p_std is not None else None,
        # The Fraction itself, for gate_decision(). Stripped before serialization
        # so the artifact stays plain JSON; §2.6 must never see the float.
        "_P_std_fraction": p_std,
        "weights": {
            "n_multi": WEIGHT_N_MULTI,
            "n_single": WEIGHT_N_SINGLE,
            "N": WEIGHT_N_TOTAL,
        },
        # A2.1: a projection without its survival rate is not a reportable figure.
        "survival_rate_basis": {
            "post_screen_survivors": survivor_counts[MULTI] + survivor_counts[SINGLE],
            "note": (
                "P_std is denominated in pairs per 10,000 post-screen prompts. "
                "No pre-screen conversion is applied anywhere in this arithmetic."
            ),
        },
        "materialized_pairs": materialized,
    }


# Owner decision, #general msg 2379: option A with C. HANDOFF §5.1's length-gap
# floor and malformed-syntax cap are **measured and reported, never applied**.
# Applying them would change which pairs materialize, therefore P_std, therefore
# the §2.6 gate — using thresholds that are not in the frozen preregistration.
# Reporting them costs nothing and answers, from the pilot's own data, whether
# they would have mattered enough to be worth an amendment before calibration.
LENGTH_GAP_REFERENCE = 0.40
MALFORMED_CAP_REFERENCE = 0.05


def pair_diagnostics(pairs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Measure HANDOFF §5.1's two caps against the mined set. Filters nothing."""
    if not pairs:
        return {
            "pairs": 0,
            "length_gap_over_reference": 0,
            "length_gap_share": None,
            "malformed_rejected": 0,
            "malformed_share": None,
            "would_either_cap_have_bound": False,
            "note": "measured only; no pair was excluded by either reference value",
        }

    over_gap = 0
    malformed = 0
    for pair in pairs:
        chosen, rejected = pair["chosen"], pair["rejected"]
        longest = max(len(chosen), len(rejected))
        if longest and abs(len(chosen) - len(rejected)) / longest > LENGTH_GAP_REFERENCE:
            over_gap += 1
        if extract_calls(rejected)[0] == "unreadable":
            malformed += 1

    gap_share = over_gap / len(pairs)
    malformed_share = malformed / len(pairs)
    return {
        "pairs": len(pairs),
        "length_gap_reference": LENGTH_GAP_REFERENCE,
        "length_gap_over_reference": over_gap,
        "length_gap_share": gap_share,
        "malformed_cap_reference": MALFORMED_CAP_REFERENCE,
        "malformed_rejected": malformed,
        "malformed_share": malformed_share,
        "would_either_cap_have_bound": bool(over_gap) or malformed_share > MALFORMED_CAP_REFERENCE,
        "note": (
            "Measured against HANDOFF §5.1's reference values and NOT applied. "
            "These thresholds are absent from frozen §2; applying them would move "
            "the §2.6 gate on unregistered rules. If either would have bound "
            "materially, register it by amendment before the calibration run."
        ),
    }


def gate_decision(p_std: float | Fraction | None) -> str:
    """§2.6's table, applied to the exact value. No rounding at any boundary."""
    if p_std is None:
        return "UNDECIDABLE: a stratum has no mined prompts, so y_std is undefined"
    if p_std >= GATE_PROCEED:
        return "PROCEED to Phase 3A (DPO rerun)"
    if p_std >= GATE_CAUTIOUS:
        return "PROCEED CAUTIOUSLY: 1 epoch max, eval callback every ~50 steps"
    return "DO NOT run DPO. Go to Phase 3B (rejection-sampling SFT)"


DERIVED_FILES = ("mined_pairs.jsonl", "mining_summary.json", "sft_bucket.jsonl")


def write_derivatives(
    out_dir: Path,
    ledger: Ledger,
    survivor_counts: dict[str, int],
    stage: str,
    allocation: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Re-materialize every derived file from the ledger's active records.

    Called after a normal run *and* after `--redo-last`. A tombstone that does
    not reach the derivatives leaves `mining_summary.json` reporting a yield the
    ledger no longer supports — the artifact contradicting its own evidence,
    which is worse than having no summary at all.
    """
    out_dir = Path(out_dir)
    summary = summarize(ledger, survivor_counts)
    summary["stage"] = stage
    if allocation is not None:
        summary["allocation"] = allocation

    exact = summary.pop("_P_std_fraction")
    summary["guardrail_diagnostics"] = pair_diagnostics(summary["materialized_pairs"])
    if stage == "calibration":
        # §2.6 is evaluated on the exact rational, never the serialized float.
        summary["decision"] = gate_decision(exact)
    else:
        summary["gate_note"] = (
            "§2.6's decision table decides on the committed CALIBRATION artifact "
            "only. This is a pilot: an operational gate the owner reads a "
            "histogram for. No Phase 2 decision is emitted here, because the "
            "frozen text does not authorize one from 100 prompts."
        )

    pairs = summary.pop("materialized_pairs")
    bucket = summary.pop("sft_bucket")
    _atomic_write(out_dir / "mined_pairs.jsonl", "".join(json.dumps(p) + "\n" for p in pairs))
    _atomic_write(out_dir / "sft_bucket.jsonl", "".join(json.dumps(r) + "\n" for r in bucket))
    _atomic_write(out_dir / "mining_summary.json", json.dumps(summary, indent=2) + "\n")
    summary["sft_bucket_rows"] = len(bucket)
    return summary


def run(
    out_dir: Path,
    stage: str,
    generate_fn: Callable[[Prompt, int], list[str]],
    prompts: Sequence[Prompt] | None = None,
    survivor_counts: dict[str, int] | None = None,
    receipt: dict[str, Any] | None = None,
    fresh: bool = False,
    n_prompts: int | None = None,
) -> dict[str, Any]:
    """Mine one stage, resuming only into a run with the same identity."""
    if stage not in STAGES:
        raise MinerError(f"stage must be one of {sorted(STAGES)}, not {stage!r}")
    n_prompts = STAGES[stage] if n_prompts is None else n_prompts

    if prompts is None or survivor_counts is None or receipt is None:
        prompts, receipt = load_eligible_prompts()
        survivor_counts = {
            MULTI: sum(1 for p in prompts if p.stratum == MULTI),
            SINGLE: sum(1 for p in prompts if p.stratum == SINGLE),
        }

    # §5.1's last guardrail: the fixture gate runs before every mining session.
    # Binding its version into the run identity is what makes "it passed" a
    # property of this artifact rather than of somebody's shell history.
    report = run_selftest()
    if not report.passed:
        raise MinerError(
            f"verifier fixture self-test failed ({report.pairs_passed}/{report.pairs}); "
            f"no prompt may be mined against a verifier that cannot clear its own gate"
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = [name for name in (*DERIVED_FILES, "ledger.jsonl", RUN_METADATA_NAME)
                if (out_dir / name).exists()]
    if fresh and existing:
        raise MinerError(
            f"--fresh refuses to delete existing evidence in {out_dir}: {existing}. "
            f"Move the directory aside deliberately; this is evidence, not scratch space"
        )

    allocation = allocate(n_prompts, survivor_counts)
    selected = select_prompts(prompts, allocation)
    bind_run_metadata(
        out_dir,
        run_metadata(stage, n_prompts, selected, receipt, report.version),
    )

    ledger = Ledger(out_dir / "ledger.jsonl")
    already = ledger.processed_ids()
    for prompt in selected:
        if prompt.prompt_id in already:
            continue  # resume: already paid for, never re-sampled
        outcome = mine_prompt(prompt, generate_fn)
        ledger.append(
            {
                "prompt_id": outcome.prompt_id,
                "stratum": outcome.stratum,
                "generations": outcome.generations,
                "verdicts": outcome.verdicts,
                "prompt_messages": outcome.prompt_messages,
                "target": outcome.target,
                "verifier_version": VERIFIER_VERSION,
                "sampling": {
                    "samples": SAMPLES_PER_PROMPT,
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "seed": SEED,
                },
            }
        )

    return write_derivatives(out_dir, ledger, survivor_counts, stage, allocation)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--self-test", action="store_true",
                        help="run the verifier fixture gate and exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="load, screen, allocate and select at $0; generate nothing")
    parser.add_argument("--stage", choices=sorted(STAGES), default="pilot")
    parser.add_argument("--out-dir", type=Path, default=Path("mining_pilot"))
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--redo-last", type=int, default=None,
                        help="tombstone the most recent N active records and exit")
    args = parser.parse_args()

    if args.self_test:
        report = run_selftest()
        print(f"verifier {report.version}: {report.pairs_passed}/{report.pairs} pairs")
        raise SystemExit(0 if report.passed else 1)

    if args.redo_last is not None:
        out_dir = Path(args.out_dir)
        ledger = Ledger(out_dir / "ledger.jsonl")
        tombstoned = ledger.redo_last(args.redo_last)
        # Re-materialize, or the derivatives keep reporting the rolled-back work.
        metadata = json.loads((out_dir / RUN_METADATA_NAME).read_text())
        _prompts, receipt = load_eligible_prompts()
        counts = {
            MULTI: receipt["survivors"]["multi"],
            SINGLE: receipt["survivors"]["single"],
        }
        summary = write_derivatives(out_dir, ledger, counts, metadata["stage"])
        print(
            f"tombstoned {tombstoned} records; re-materialized "
            f"{summary['pairs_total']} pairs from {summary['prompts_mined_total']} prompts"
        )
        raise SystemExit(0)

    prompts, receipt = load_eligible_prompts()
    counts = {
        MULTI: sum(1 for p in prompts if p.stratum == MULTI),
        SINGLE: sum(1 for p in prompts if p.stratum == SINGLE),
    }
    n_prompts = STAGES[args.stage]
    allocation = allocate(n_prompts, counts)
    print(f"post-screen survivors: {counts} (artifact: {receipt['survivors']})")
    print(f"allocation for stage {args.stage} (n={n_prompts}): {allocation}")

    if args.dry_run:
        selected = select_prompts(prompts, allocation)
        print(f"dry run: selected {len(selected)} prompts, generated nothing, $0 spent")
        raise SystemExit(0)

    # Deliberately not wired: importing the model stack is the first line that
    # can cost money, so it is a separate, separately reviewed change. Until
    # then this CLI cannot spend by accident.
    raise SystemExit(
        "generation backend is not wired in this commit; use --dry-run or call "
        "run() with a generate_fn. No paid path exists here yet."
    )


if __name__ == "__main__":
    main()
