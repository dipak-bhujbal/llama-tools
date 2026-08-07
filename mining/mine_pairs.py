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
    MISSING_CALL,
    SPURIOUS_CALL,
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

# §3.3/eval pins: the policy being sampled is an adapter on this exact base.
BASE_MODEL_REPO = "meta-llama/Llama-3.1-8B-Instruct"
BASE_MODEL_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"

MINER_SOURCE = Path("mining/mine_pairs.py")
VERIFIER_SOURCE = Path("mining/verifier.py")
VERIFIER_SELFTEST_RECEIPT = Path("mining/receipts/verifier_selftest.json")
VERIFIER_SELFTEST_RECEIPT_SHA256 = (
    "3e0f921e607fd112555c35aaa1cd9f16f54841cfed46ddda9c20066a0474cf7f"
)

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
    messages: tuple[tuple[str, str], ...] = ()

    @property
    def prompt_messages(self) -> list[dict[str, str]]:
        """The exact turns preceding the target, in order, roles preserved."""
        return [{"role": role, "content": content} for role, content in self.messages]


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
    """Join every turn of one role. Used only where the artifact does."""
    return " ".join(
        m.get("content", "") for m in row.get("messages", []) if m.get("role") == role
    )


def split_at_first_assistant(row: dict[str, Any]) -> tuple[list[dict[str, str]], str]:
    """Return `(prompt_messages, target)` split at the row's first assistant turn.

    The signed eligibility artifact classifies the **first** assistant turn.
    Joining every assistant turn — which this module did until codex measured it
    — concatenates `first + final` into one string. Of the 11,071 survivors, 439
    carry two assistant turns and 7 place a `tool` message before their first
    assistant, so the join produced a ground truth that no row ever taught, and
    dropped required context for the tool-prefixed rows.

    The preceding messages are preserved exactly, in order and with their roles,
    rather than flattened into system+user.
    """
    messages = row.get("messages", [])
    for index, message in enumerate(messages):
        if message.get("role") == "assistant":
            prompt_messages = [
                {"role": m.get("role", ""), "content": m.get("content", "")}
                for m in messages[:index]
            ]
            return prompt_messages, message.get("content", "")
    return (
        [{"role": m.get("role", ""), "content": m.get("content", "")} for m in messages],
        "",
    )


def _target_turn(row: dict[str, Any], prompt_id: str) -> str:
    """Deprecated shim retained only for the §2.11 refusal test path."""
    """The assistant turn this row teaches, or a refusal.

    §2.11: a target the miner cannot read is a **stop condition**. The
    quarantined miner turned this exact branch into `no_call` ground truth,
    which produces a preference pair whose chosen answer is wrong.
    """
    _prompt_messages, assistant = split_at_first_assistant(row)
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

        prompt_messages, assistant = split_at_first_assistant(row)
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
                target=assistant,
                messages=tuple((m["role"], m["content"]) for m in prompt_messages),
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
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with tmp.open("w") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    # Without this the rename itself can be lost on power failure, leaving the
    # old bytes behind a durable-looking write.
    _fsync_directory(path.parent)


def _digest_of(path: Path) -> str:
    return _sha256((REPO_ROOT / path).read_bytes())


def _fsync_directory(path: Path) -> None:
    """Make a directory-entry update durable, not merely atomic in memory."""
    dir_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def verify_selftest_receipt(report: Any) -> dict[str, Any]:
    """Bind the gate that ran to the committed verifier and fixture bytes.

    Merely recording the receipt's hash does not prove the running self-test used
    what that receipt names. A verifier and fixture set can drift together and
    still pass 1,600/1,600. This check refuses that coherent-but-unregistered
    drift before generation.
    """
    payload = _verify_pin(
        VERIFIER_SELFTEST_RECEIPT,
        VERIFIER_SELFTEST_RECEIPT_SHA256,
        "verifier self-test receipt",
    )
    receipt = json.loads(payload)

    implementation = receipt.get("implementation", {})
    implementation_path = Path(implementation.get("path", ""))
    if implementation_path != VERIFIER_SOURCE:
        raise MinerError(
            f"self-test receipt names verifier {implementation_path}, not {VERIFIER_SOURCE}"
        )
    actual_implementation = _digest_of(VERIFIER_SOURCE)
    if implementation.get("sha256") != actual_implementation:
        raise MinerError(
            "verifier implementation bytes do not match the committed self-test receipt"
        )

    fixture_evidence: list[dict[str, Any]] = []
    for fixture in receipt.get("fixtures", []):
        fixture_path = Path(fixture.get("path", ""))
        resolved = REPO_ROOT / fixture_path
        if not resolved.exists():
            raise MinerError(f"self-test fixture is missing at {fixture_path}")
        actual_sha = _sha256(resolved.read_bytes())
        actual_rows = sum(1 for line in resolved.read_text().splitlines() if line.strip())
        if fixture.get("sha256") != actual_sha or fixture.get("rows") != actual_rows:
            raise MinerError(
                f"self-test fixture {fixture_path} does not match the committed receipt"
            )
        fixture_evidence.append(
            {"path": str(fixture_path), "sha256": actual_sha, "rows": actual_rows}
        )

    result_fields = {
        "verifier_version": report.version,
        "pairs": report.pairs,
        "pairs_passed": report.pairs_passed,
        "misses": report.misses,
        "false_positives": report.false_positives,
        "reason_mismatches": report.reason_mismatches,
        "refusals": report.refusals,
        "passed": report.passed,
    }
    differing = sorted(k for k, value in result_fields.items() if receipt.get(k) != value)
    if differing:
        raise MinerError(
            "running verifier self-test disagrees with the committed receipt in "
            f"fields: {differing}"
        )

    return {
        "receipt_sha256": VERIFIER_SELFTEST_RECEIPT_SHA256,
        "pairs": report.pairs,
        "pairs_passed": report.pairs_passed,
        "fixtures": fixture_evidence,
    }


def run_metadata(
    stage: str,
    n_prompts: int,
    selected: Sequence[Prompt],
    receipt: dict[str, Any],
    selftest_version: str,
    allocation: dict[str, int],
    selftest_evidence: dict[str, Any],
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
        "allocation": dict(allocation),
        "selected_id_sha256": _id_digest(p.prompt_id for p in selected),
        # The sorted digest cannot distinguish two runs that sample the same ids
        # in a different order, and sampling order is part of what a seeded run
        # promises to reproduce.
        "selected_ids_ordered_sha256": _sha256(
            ("\n".join(p.prompt_id for p in selected) + "\n").encode()
        ),
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
        "base_model": {"repo": BASE_MODEL_REPO, "revision": BASE_MODEL_REVISION},
        # A version string cannot detect code or fixture drift; digests can.
        "miner_sha256": _digest_of(MINER_SOURCE),
        "verifier": {
            "version": VERIFIER_VERSION,
            "selftest_version": selftest_version,
            "module_sha256": _digest_of(VERIFIER_SOURCE),
            "selftest": selftest_evidence,
        },
    }


def bind_run_metadata(out_dir: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    """Create the run identity once, or prove the resumed run matches it."""
    path = out_dir / RUN_METADATA_NAME
    if not path.exists():
        # O_EXCL, not exists()-then-write: two miners starting together would
        # both see "absent" and both write, and the loser's identity would be
        # the one the run is judged against.
        payload = json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        except FileExistsError:
            pass  # another process won the race; fall through and compare
        else:
            with os.fdopen(fd, "w") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            _fsync_directory(path.parent)
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


@dataclass(frozen=True)
class StagePreflight:
    """Everything verified before a stage may sample or mutate its ledger."""

    stage: str
    prompts: list[Prompt]
    survivor_counts: dict[str, int]
    allocation: dict[str, int]
    selected: list[Prompt]
    metadata: dict[str, Any]


def preflight_stage(stage: str) -> StagePreflight:
    """Run the one production preflight used by mining and rollback alike."""
    if stage not in STAGES:
        raise MinerError(f"stage must be one of {sorted(STAGES)}, not {stage!r}")
    n_prompts = STAGES[stage]

    prompts, receipt = load_eligible_prompts()
    survivor_counts = {
        MULTI: sum(1 for p in prompts if p.stratum == MULTI),
        SINGLE: sum(1 for p in prompts if p.stratum == SINGLE),
    }

    report = run_selftest()
    if not report.passed:
        raise MinerError(
            f"verifier fixture self-test failed ({report.pairs_passed}/{report.pairs}); "
            f"no prompt may be mined against a verifier that cannot clear its own gate"
        )
    selftest_evidence = verify_selftest_receipt(report)

    allocation = allocate(n_prompts, survivor_counts)
    selected = select_prompts(prompts, allocation)
    metadata = run_metadata(
        stage,
        n_prompts,
        selected,
        receipt,
        report.version,
        allocation,
        selftest_evidence,
    )
    return StagePreflight(
        stage=stage,
        prompts=prompts,
        survivor_counts=survivor_counts,
        allocation=allocation,
        selected=selected,
        metadata=metadata,
    )


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
        prompt_messages=prompt.prompt_messages,
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
# Computing them adds no model calls to an already-approved run; observing pilot
# values still requires the separately estimated and owner-approved paid pilot.
LENGTH_GAP_REFERENCE = 0.40
MALFORMED_CAP_REFERENCE = 0.05
LENGTH_GAP_EXEMPT_REASONS = frozenset({MISSING_CALL, SPURIOUS_CALL})


def pair_diagnostics(pairs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Measure HANDOFF §5.1's two caps against the mined set. Filters nothing."""
    if not pairs:
        return {
            "pairs": 0,
            "length_gap_reference": LENGTH_GAP_REFERENCE,
            "length_gap_over_reference": 0,
            "length_gap_exempt_call_vs_text": 0,
            "length_gap_share": None,
            "malformed_cap_reference": MALFORMED_CAP_REFERENCE,
            "malformed_rejected": 0,
            "malformed_share": None,
            "would_either_cap_have_bound": False,
            "note": "measured only; no pair was excluded by either reference value",
        }

    over_gap = 0
    gap_exempt = 0
    malformed = 0
    for pair in pairs:
        chosen, rejected = pair["chosen"], pair["rejected"]
        longest = max(len(chosen), len(rejected))
        over_reference = (
            bool(longest)
            and abs(len(chosen) - len(rejected)) / longest > LENGTH_GAP_REFERENCE
        )
        if over_reference:
            if pair.get("rejected_reason") in LENGTH_GAP_EXEMPT_REASONS:
                gap_exempt += 1
            else:
                over_gap += 1
        if extract_calls(rejected)[0] == "unreadable":
            malformed += 1

    gap_share = over_gap / len(pairs)
    malformed_share = malformed / len(pairs)
    return {
        "pairs": len(pairs),
        "length_gap_reference": LENGTH_GAP_REFERENCE,
        "length_gap_over_reference": over_gap,
        "length_gap_exempt_call_vs_text": gap_exempt,
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

    `allocation` is read back from `run.json` when not supplied, so a rollback
    cannot silently delete it from the summary.
    """
    out_dir = Path(out_dir)
    if allocation is None:
        metadata_path = out_dir / RUN_METADATA_NAME
        if metadata_path.exists():
            allocation = json.loads(metadata_path.read_text()).get("allocation")

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
    # Counted into the summary *before* it is serialized, or the artifact on
    # disk lacks a number the returned dict claims to carry.
    summary["sft_bucket_rows"] = len(bucket)

    _atomic_write(out_dir / "mined_pairs.jsonl", "".join(json.dumps(p) + "\n" for p in pairs))
    _atomic_write(out_dir / "sft_bucket.jsonl", "".join(json.dumps(r) + "\n" for r in bucket))
    _atomic_write(out_dir / "mining_summary.json", json.dumps(summary, indent=2) + "\n")
    return summary


def run(
    out_dir: Path,
    stage: str,
    generate_fn: Callable[[Prompt, int], list[str]],
    fresh: bool = False,
) -> dict[str, Any]:
    """Mine one stage, resuming only into a run with the same identity.

    There is deliberately no way to inject prompts, survivor counts, or a
    receipt. An earlier signature accepted all three, which let a caller skip
    `load_eligible_prompts()` entirely while `run.json` still recorded the
    pinned digests — evidence asserting a preflight that never ran. Tests
    substitute the loader itself, so the production path has one route in.
    """
    preflight = preflight_stage(stage)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = [name for name in (*DERIVED_FILES, "ledger.jsonl", RUN_METADATA_NAME)
                if (out_dir / name).exists()]
    if fresh and existing:
        raise MinerError(
            f"--fresh refuses to delete existing evidence in {out_dir}: {existing}. "
            f"Move the directory aside deliberately; this is evidence, not scratch space"
        )

    bind_run_metadata(out_dir, preflight.metadata)

    ledger = Ledger(out_dir / "ledger.jsonl")
    already = ledger.processed_ids()
    for prompt in preflight.selected:
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

    return write_derivatives(
        out_dir,
        ledger,
        preflight.survivor_counts,
        stage,
        preflight.allocation,
    )


def redo_run(out_dir: Path, count: int) -> tuple[int, dict[str, Any]]:
    """Tombstone work only after re-verifying the run's complete identity."""
    out_dir = Path(out_dir)
    metadata_path = out_dir / RUN_METADATA_NAME
    if not metadata_path.exists():
        raise MinerError(
            f"{metadata_path} is missing; refusing to roll back a run whose "
            "identity cannot be established"
        )
    metadata = json.loads(metadata_path.read_text())
    stage = metadata.get("stage")
    if stage not in STAGES:
        raise MinerError(f"{metadata_path} carries no recognised stage")

    # This compares every pinned input, implementation/fixture digest, selected
    # id/order, and allocation before the irreversible append-only tombstone.
    preflight = preflight_stage(stage)
    bind_run_metadata(out_dir, preflight.metadata)

    ledger = Ledger(out_dir / "ledger.jsonl")
    tombstoned = ledger.redo_last(count)
    summary = write_derivatives(
        out_dir,
        ledger,
        preflight.survivor_counts,
        stage,
        preflight.allocation,
    )
    return tombstoned, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--self-test", action="store_true",
                        help="run the verifier fixture gate and exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="load, screen, allocate and select at $0; generate nothing")
    parser.add_argument("--stage", choices=sorted(STAGES), default="pilot")
    parser.add_argument("--out-dir", type=Path, default=Path("mining_pilot"))
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--hourly-rate", type=float, default=None,
                        help="the pod's ACTUAL hourly rate from the console")
    parser.add_argument("--cap-usd", type=float, default=None,
                        help="owner-approved total-stage cap in USD")
    parser.add_argument("--redo-last", type=int, default=None,
                        help="tombstone the most recent N active records and exit")
    args = parser.parse_args()

    if args.self_test:
        report = run_selftest()
        print(f"verifier {report.version}: {report.pairs_passed}/{report.pairs} pairs")
        raise SystemExit(0 if report.passed else 1)

    if args.redo_last is not None:
        out_dir = Path(args.out_dir)
        try:
            tombstoned, summary = redo_run(out_dir, args.redo_last)
        except MinerError as exc:
            raise SystemExit(str(exc)) from exc
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

    # The paid path. It exists only when the operator types the rate the console
    # actually shows and the cap the owner approved: there is no default that
    # spends, and no way to launch without a mechanically enforced deadline.
    if args.hourly_rate is None or args.cap_usd is None:
        raise SystemExit(
            "refusing to launch: --hourly-rate and --cap-usd are both required so "
            "the run carries a deadline derived from what this pod actually "
            "charges. Use --dry-run to walk the whole path at $0."
        )

    from mining.backend import load_policy, sampling_receipt

    receipt = sampling_receipt(args.hourly_rate, args.cap_usd)
    bound = receipt["spend_bound"]
    print(
        f"spend bound: ${args.cap_usd:.2f} cap at ${args.hourly_rate:.2f}/hr "
        f"-> terminate after {bound['deadline_seconds']}s "
        f"({bound['deadline_seconds'] / 3600:.2f} h)"
    )
    generate_fn, _guard = load_policy(args.hourly_rate, args.cap_usd)
    summary = run(out_dir=args.out_dir, stage=args.stage, generate_fn=generate_fn)
    print(
        f"mined {summary['prompts_mined_total']} prompts -> "
        f"{summary['pairs_total']} pairs; histogram {summary['pass_histogram']}"
    )


if __name__ == "__main__":
    main()
