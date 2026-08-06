"""`onpolicy_verifier_v1` — decide whether a generation matches its target.

Mining samples the SFT policy several times per prompt and needs one question
answered per sample: **did this generation do what the pool row's target does?**
Accepted samples are the policy succeeding; rejected samples are the raw material
for preference pairs. Every downstream number — yield, the §2.6 gate, the 3B
training set — is that verdict, counted.

Three things make this version different from the quarantined
`intake/quarantine/2026-08-04-chat-attachments/mine_pairs.py`, which is reference
only and is never imported:

**A target it cannot read is a stop, not a `no_call`.** The quarantined verifier
turned an unparseable target into a `no_call` ground truth, so a row nobody could
check became a row that silently passed. Frozen prereg §2.11 replaces that
fallback with a refusal, and `TargetUnreadableError` is that refusal.

**Two independent parses must agree.** The kind of a target (`call` / `no_call` /
`unreadable`) comes from `mining.pool_strata.classify_target`, the same function
that built the committed eligibility receipt. The function names and arguments
come from this module's own extractor. If the two disagree about which functions
a target calls, that is a parser defect and the verifier refuses rather than
picking a winner. Checked twice, trusted once.

**The verdict carries a reason from a closed set.** `invalid_json`,
`missing_call`, `spurious_call`, `wrong_tool`, `wrong_args` — evaluated in that
order so a generation with two defects always reports the same one. The reasons
map onto ADR-007's error taxonomy, so mined pairs stay comparable with study 1's.

Argument comparison is **exact structural equality after JSON parsing**: no type
coercion, no numeric tolerance, no case folding. `"5"` is not `5`. A looser rule
would have to be justified against the pool rather than assumed, and this one
fails closed.

Usage:
    python mining/verifier.py --selftest
    python mining/verifier.py --selftest --json mining/receipts/verifier_selftest.json
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mining.pool_strata import CALL, NO_CALL, UNREADABLE, classify_target  # noqa: E402

VERIFIER_VERSION = "onpolicy_verifier_v1"

DEFAULT_FIXTURES = (
    REPO_ROOT / "tests" / "fixtures" / "fixture_pairs_train.jsonl",
    REPO_ROOT / "tests" / "fixtures" / "fixture_pairs_eval.jsonl",
)
DEFAULT_RECEIPT = REPO_ROOT / "mining" / "receipts" / "verifier_selftest.json"

ACCEPTED = "accepted"
REJECTED = "rejected"

INVALID_JSON = "invalid_json"
MISSING_CALL = "missing_call"
SPURIOUS_CALL = "spurious_call"
WRONG_TOOL = "wrong_tool"
WRONG_ARGS = "wrong_args"

REASONS = (INVALID_JSON, MISSING_CALL, SPURIOUS_CALL, WRONG_TOOL, WRONG_ARGS)

# Which reason each fixture error_type must produce. The fixture set labels the
# defect it injected, so this mapping turns "the verifier rejected it" into "the
# verifier rejected it for the right reason" -- a strictly stronger claim, and
# the one that catches a verifier that is accidentally right.
FIXTURE_REASON_FOR_ERROR_TYPE = {
    "missed_tool_call": MISSING_CALL,
    "spurious_tool_call": SPURIOUS_CALL,
    "wrong_function_selection": WRONG_TOOL,
    "wrong_param_value": WRONG_ARGS,
    "missing_required_parameter": WRONG_ARGS,
    "hallucinated_parameter": WRONG_ARGS,
    "malformed_syntax": INVALID_JSON,
}

_TOOL_CALL_MARKER = "<tool_call>"
_TOOL_CALL = re.compile(r"<tool_call>(?:\s|\\[nrt])*(\{.*?\})(?:\s|\\[nrt])*</tool_call>", re.S)
_TOOL_CALL_OPEN = re.compile(r"<tool_call>")
_TOOL_CALL_CLOSE = re.compile(r"</tool_call>")


class TargetUnreadableError(ValueError):
    """A pool target that cannot be parsed. Frozen §2.11: refuse, never reclassify.

    Raised rather than returned, because every caller that could swallow this is
    a caller that would turn an unchecked row into a passing one.
    """


class ParserDisagreementError(ValueError):
    """`classify_target` and this module read the same target differently.

    Never a data problem — always a defect in one of the two parsers, and the
    verifier is not entitled to guess which.
    """


@dataclass(frozen=True)
class Verdict:
    verdict: str
    reason: str | None = None
    detail: str = ""

    @property
    def accepted(self) -> bool:
        return self.verdict == ACCEPTED


@dataclass
class SelfTestReport:
    version: str = VERIFIER_VERSION
    pairs: int = 0
    pairs_passed: int = 0
    misses: list[str] = field(default_factory=list)          # chosen wrongly rejected
    false_positives: list[str] = field(default_factory=list)  # rejected wrongly accepted
    reason_mismatches: list[dict[str, str]] = field(default_factory=list)
    refusals: list[dict[str, str]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return (
            self.pairs > 0
            and self.pairs_passed == self.pairs
            and not self.misses
            and not self.false_positives
            and not self.reason_mismatches
            and not self.refusals
        )


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _as_call(obj: Any) -> tuple[str, Any] | None:
    """`(name, arguments)` if this object is a well-formed call, else `None`.

    A top-level string `name` is required. There is deliberately no search for a
    nested `name` key: the quarantined parser's regex found names inside
    `arguments` and manufactured calls out of ordinary payloads.
    """
    if not isinstance(obj, dict):
        return None
    name = obj.get("name")
    if not isinstance(name, str) or not name:
        return None
    return name, obj.get("arguments", obj.get("parameters"))


def _parse_block(block: str) -> tuple[str, Any] | None:
    for parse in (json.loads, ast.literal_eval):
        try:
            return _as_call(parse(block))
        except (ValueError, SyntaxError, TypeError):
            continue
    return None


def extract_calls(text: str) -> tuple[str, list[tuple[str, Any]]]:
    """Return `(kind, calls)` for an assistant turn.

    `kind` is `call`, `no_call`, or `unreadable`, on the same rules as
    `mining.pool_strata.classify_target`; `calls` carries the `(name, arguments)`
    pairs that `classify_target` does not expose.
    """
    stripped = text.strip()
    if not stripped:
        return UNREADABLE, []

    if _TOOL_CALL_MARKER in stripped:
        blocks = _TOOL_CALL.findall(stripped)
        if not (
            len(blocks)
            == len(_TOOL_CALL_OPEN.findall(stripped))
            == len(_TOOL_CALL_CLOSE.findall(stripped))
        ):
            return UNREADABLE, []
        calls = []
        for block in blocks:
            call = _parse_block(block)
            if call is None:
                return UNREADABLE, []
            calls.append(call)
        return (CALL, calls) if calls else (UNREADABLE, [])

    if stripped[0] in "[{":
        try:
            parsed = json.loads(stripped)
        except ValueError:
            return UNREADABLE, []
        candidates = parsed if isinstance(parsed, list) else [parsed]
        calls = []
        for candidate in candidates:
            call = _as_call(candidate)
            # Every member must be a well-formed call. Filtering the bad ones out
            # would let `[{valid}, {"arguments": {...}}]` be scored as the valid
            # call alone and accepted — a generation that emitted garbage beside a
            # correct call, marked correct. Same fail-closed rule the
            # `<tool_call>` branch above applies, and the same rule
            # `pool_strata._json_call_names` already applies to targets.
            if call is None:
                return UNREADABLE, []
            calls.append(call)
        return (CALL, calls) if calls else (UNREADABLE, [])

    return NO_CALL, []


def _read_target(target_turn: str) -> tuple[str, list[tuple[str, Any]]]:
    """Classify a *pool target*, refusing anything unreadable or disputed."""
    kind, names = classify_target(target_turn)
    if kind == UNREADABLE:
        raise TargetUnreadableError(
            "target does not parse under pool-target-structural-eligibility/v1; "
            "frozen §2.11 makes this a stop condition, not a no_call row"
        )

    own_kind, calls = extract_calls(target_turn)
    if own_kind != kind or {name for name, _ in calls} != names:
        raise ParserDisagreementError(
            f"classify_target says {kind}/{sorted(names)}; verifier says "
            f"{own_kind}/{sorted(name for name, _ in calls)}"
        )
    return kind, calls


def verify(generation: str, target_turn: str) -> Verdict:
    """Did `generation` do what `target_turn` does?

    Raises `TargetUnreadableError` / `ParserDisagreementError` on the *target*; a
    generation is never a refusal, since an unparseable generation is exactly the
    policy failure mining exists to collect.
    """
    target_kind, target_calls = _read_target(target_turn)
    gen_kind, gen_calls = extract_calls(generation)

    if target_kind == NO_CALL:
        if gen_kind == CALL:
            return Verdict(REJECTED, SPURIOUS_CALL,
                           f"target answers in prose; generation called "
                           f"{sorted(name for name, _ in gen_calls)}")
        if gen_kind == UNREADABLE:
            return Verdict(REJECTED, INVALID_JSON,
                           "generation announces a call or JSON and does not parse")
        return Verdict(ACCEPTED)

    if gen_kind == NO_CALL:
        return Verdict(REJECTED, MISSING_CALL, "target calls a tool; generation is prose")
    if gen_kind == UNREADABLE:
        return Verdict(REJECTED, INVALID_JSON,
                       "generation announces a call or JSON and does not parse")

    target_names = sorted(name for name, _ in target_calls)
    gen_names = sorted(name for name, _ in gen_calls)
    if target_names != gen_names:
        return Verdict(REJECTED, WRONG_TOOL, f"expected {target_names}, got {gen_names}")

    # Same functions, so compare arguments as an order-insensitive multiset:
    # a target with two calls does not fix the order the policy emits them in.
    target_sig = sorted(_canonical([name, args]) for name, args in target_calls)
    gen_sig = sorted(_canonical([name, args]) for name, args in gen_calls)
    if target_sig != gen_sig:
        return Verdict(REJECTED, WRONG_ARGS, "arguments differ from the target")

    return Verdict(ACCEPTED)


def _display(path: Path) -> str:
    """Repo-relative when it can be, the raw path when it cannot.

    `Path.relative_to` raises on a path outside the repo *and* on a relative one
    passed at the CLI, which would crash the run after the artifact was already
    written — the worst moment to fail.
    """
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _fixture_rows(paths) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with Path(path).open() as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def run_selftest(paths=DEFAULT_FIXTURES) -> SelfTestReport:
    """Every `chosen` must be accepted and every `rejected` rejected, for the
    labelled reason. Fixtures are synthetic and say so; this measures the
    verifier, not the model."""
    report = SelfTestReport()

    for row in _fixture_rows(paths):
        pair_id = row["meta"]["pair_id"]
        error_type = row["meta"]["error_type"]
        target = row["chosen"][0]["content"]
        report.pairs += 1

        try:
            chosen_verdict = verify(target, target)
            rejected_verdict = verify(row["rejected"][0]["content"], target)
        except (TargetUnreadableError, ParserDisagreementError) as exc:
            report.refusals.append({"pair_id": pair_id, "error": f"{type(exc).__name__}: {exc}"})
            continue

        ok = True
        if not chosen_verdict.accepted:
            report.misses.append(pair_id)
            ok = False
        if rejected_verdict.accepted:
            report.false_positives.append(pair_id)
            ok = False
        else:
            expected = FIXTURE_REASON_FOR_ERROR_TYPE[error_type]
            if rejected_verdict.reason != expected:
                report.reason_mismatches.append({
                    "pair_id": pair_id, "error_type": error_type,
                    "expected": expected, "got": rejected_verdict.reason or "",
                })
                ok = False
        if ok:
            report.pairs_passed += 1

    return report


def _receipt(report: SelfTestReport, paths) -> dict[str, Any]:
    return {
        "verifier_version": report.version,
        "gate": ("every fixture chosen accepted, every fixture rejected rejected "
                 "for its labelled reason"),
        "implementation": {
            "path": "mining/verifier.py",
            "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "fixtures": [
            {
                "path": _display(Path(p)),
                "sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest(),
                "rows": sum(1 for line in Path(p).read_text().splitlines() if line.strip()),
                "synthetic": True,
            }
            for p in paths
        ],
        "fixture_provenance": (
            "Template-generated synthetic pairs (tests/fixtures/REPRODUCTION.md). NOT model "
            "generations. This receipt measures the verifier's agreement with labelled "
            "injected defects; it is not evidence about any model."
        ),
        "pairs": report.pairs,
        "pairs_passed": report.pairs_passed,
        "misses": report.misses,
        "false_positives": report.false_positives,
        "reason_mismatches": report.reason_mismatches,
        "refusals": report.refusals,
        "passed": report.passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--selftest", action="store_true", required=True)
    parser.add_argument("--fixtures", type=Path, nargs="+", default=list(DEFAULT_FIXTURES))
    parser.add_argument("--json", type=Path, help="write the self-test receipt")
    args = parser.parse_args()

    report = run_selftest(args.fixtures)
    receipt = _receipt(report, args.fixtures)

    print(f"verifier   {report.version}")
    print(f"fixtures   {sum(f['rows'] for f in receipt['fixtures'])} rows, synthetic")
    print(f"pairs      {report.pairs_passed}/{report.pairs} passed")
    print(f"misses     {len(report.misses)} (chosen wrongly rejected)")
    print(f"false pos  {len(report.false_positives)} (rejected wrongly accepted)")
    print(f"reason     {len(report.reason_mismatches)} mismatches")
    print(f"refusals   {len(report.refusals)}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"wrote {_display(args.json)}")

    if not report.passed:
        print("GATE FAILED — no prompt may be mined", file=sys.stderr)
        raise SystemExit(1)
    print("gate PASSED")


if __name__ == "__main__":
    main()
