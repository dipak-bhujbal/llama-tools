"""Compare the two candidate BFCL v4 answer keys and re-score study 1 under each.

Phase -1 froze the eval inputs by hash and caught that two upstream revisions
carry different answer keys for `simple_python`:

- `58f57e91…` — the BFCL v4 *release* commit.
- `9d8416a9…` — a later upstream *data-fix* revision, and the key the
  2026-07-20 study-1 run actually scored against. This is what the manifest
  currently pins as canonical.

Both keys are manifest-pinned inputs, fetched and hash-verified by
`eval/fetch_pinned_bfcl.py`. This script re-scores the committed per-item
generations under each of them and checks the property that makes the choice
safe for the DPO kill decision: whether the disputed item is concordant across
candidates (all right or all wrong), since concordant items contribute nothing
to a paired comparison.

**Everything here is recomputed, not restated.** The number of rows where the
two keys differ is measured by comparing the two full files; the release-key
score is produced by running the study's own scorer (`eval/bfcl_scoring.py`)
against the release key's bytes, not by patching a stored flag; and the claim
that the paired tests are unaffected is demonstrated by rebuilding every
contrast's discordant set under both keys and hashing them. Any input whose
bytes do not match the manifest, any missing candidate or id, a second
differing row, or a discordant set that moves — each fails the run rather than
being narrated in a caveat.

The report also measures the scorer-normalization question that item 363 is a
symptom of: how often answer keys use module-qualified function names, and how
often a row's tools are distinguishable *only* by that module prefix.

Run:
    python eval/answer_key_comparison.py
    python eval/answer_key_comparison.py --json eval/results/answer_key_comparison.json

Inputs are the committed, manifest-pinned files; nothing is fetched and no
model is called.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bfcl_scoring import (  # noqa: E402 - sibling modules; see sys.path above
    KeyDefectError,
    preflight_key_names,
    score,
)
from fetch_pinned_bfcl import VerificationError, load_manifest, verify_payload  # noqa: E402

DEFAULT_MANIFEST = REPO_ROOT / "eval" / "manifests" / "bfcl_v4_study2.json"
DEFAULT_GENERATIONS = REPO_ROOT / "eval" / "results" / "study1_bfcl_simple_generations.jsonl"
DEFAULT_EVIDENCE = REPO_ROOT / "eval" / "results" / "evidence.json"

DISPUTED_ID = "simple_python_363"
REFERENCE_CANDIDATE = "sft"
# The study-1 sweep's candidate set. Pinned rather than inferred from the file:
# inferring it means a file missing a whole candidate defines its own expected
# set and passes every downstream check.
EXPECTED_CANDIDATES = ("dpo-100", "dpo-150", "dpo-50", "sft")

# The manifest roles this comparison consumes. Both keys are pinned; neither is
# a stated value.
PINNED_KEY_ROLE = "answer_key"
RELEASE_KEY_ROLE = "answer_key_release_commit"
QUESTIONS_ROLE = "questions"
CATEGORY = "simple_python"

# Categories measured for qualified-name exposure. `multiple` is study 2's
# co-primary, so its exposure is the number that matters going forward.
EXPOSURE_CATEGORIES = ("simple_python", "multiple", "live_simple")


class ComparisonIntegrityError(ValueError):
    """An input or an invariant this comparison depends on did not hold."""


# ------------------------------------------------------------------ hashing --


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git_blob_sha1(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode()
    return hashlib.sha1(header + payload).hexdigest()  # Git object id, not a security digest


def _id_set_sha256(ids: list[str]) -> str:
    """Digest of a sorted id set, so two id sets can be compared by one value."""
    return _sha256(("\n".join(sorted(ids)) + "\n").encode())


def _fingerprint(payload: bytes) -> dict:
    return {
        "bytes": len(payload),
        "sha256": _sha256(payload),
        "git_blob_sha1": _git_blob_sha1(payload),
    }


# ------------------------------------------------------------------- inputs --


def _relative(path: Path, repo_root: Path) -> str:
    """Repo-relative path, so a committed report is not machine-specific."""
    path = Path(path)
    return str(path.relative_to(repo_root)) if path.is_relative_to(repo_root) else str(path)


def _load_jsonl(payload: bytes) -> list[dict]:
    return [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]


def _verified_manifest_input(
    manifest: dict, manifest_path: Path, repo_root: Path, category: str, role: str, label: str
) -> dict:
    """One manifest-pinned file, hash-verified, with its rows and fingerprint.

    `verify_payload` is the same check `fetch_pinned_bfcl.py` runs at download
    time: sha256, Git blob id, row count, unique-id count, and the sorted-id
    digest. Re-running it here means the comparison cannot quietly compute
    against a file that was edited after it was pinned.
    """
    specs = [
        spec
        for spec in manifest["files"]
        if spec.get("category") == category and spec.get("role") == role
    ]
    if len(specs) != 1:
        raise ComparisonIntegrityError(
            f"manifest {manifest_path} has {len(specs)} entries for "
            f"{category}/{role}; this comparison needs exactly one"
        )
    spec = specs[0]
    path = repo_root / spec["local_path"]
    if not path.is_file():
        raise ComparisonIntegrityError(
            f"missing pinned input {spec['local_path']} for {category}/{role}; "
            f"run `python eval/fetch_pinned_bfcl.py` first"
        )
    payload = path.read_bytes()
    try:
        verify_payload(payload, spec)
    except VerificationError as exc:
        raise ComparisonIntegrityError(f"{label}: {exc}") from exc
    return {
        "path": spec["local_path"],
        "source_revision": spec["source_revision"],
        "upstream_path": spec["upstream_path"],
        "rows": _load_jsonl(payload),
        **_fingerprint(payload),
    }


def load_verified_inputs(
    manifest_path: Path, repo_root: Path, exposure_categories: tuple[str, ...] = ()
) -> dict:
    """Read every pinned input this report depends on, verifying each first.

    Every file whose bytes reach a number in the report is resolved through the
    manifest and hash-verified — including the exposure categories, whose
    858-row and 29-item figures feed the preregistration amendment and the
    upstream issue. A figure quoted from an unverified file is a figure nobody
    can check.
    """
    manifest_bytes = manifest_path.read_bytes()
    manifest = load_manifest(manifest_path)

    inputs: dict = {
        "manifest": {
            "path": _relative(manifest_path, repo_root),
            **_fingerprint(manifest_bytes),
        },
        "_criterion": manifest.get("canonical_key_criterion"),
    }
    for name, (category, role) in {
        "questions": (CATEGORY, QUESTIONS_ROLE),
        "pinned_key": (CATEGORY, PINNED_KEY_ROLE),
        "release_key": (CATEGORY, RELEASE_KEY_ROLE),
    }.items():
        inputs[name] = _verified_manifest_input(
            manifest, manifest_path, repo_root, category, role, name
        )

    if exposure_categories:
        inputs["exposure"] = {
            category: {
                role: _verified_manifest_input(
                    manifest, manifest_path, repo_root, category, role, f"{category}/{role}"
                )
                for role in (QUESTIONS_ROLE, PINNED_KEY_ROLE)
            }
            for category in exposure_categories
        }
    return inputs


def load_verified_generations(
    generations_path: Path, evidence_path: Path, repo_root: Path
) -> tuple[list[dict], dict]:
    """Read the per-item generations, verified against the committed evidence index.

    Fingerprinting whatever file arrives records *a* hash; it does not establish
    that this is the study-1 run's output. `eval/results/evidence.json` already
    pins that run's sha256 and row count, so this reads the pin and checks
    against it. Without that, an input missing an entire candidate hashes
    perfectly well and produces a smaller, self-consistent, wrong report.
    """
    evidence_bytes = evidence_path.read_bytes()
    evidence = json.loads(evidence_bytes)
    local_copy = _relative(generations_path, repo_root)

    artifacts = [
        artifact
        for artifact in evidence.get("artifacts", [])
        if artifact.get("local_copy") == local_copy
    ]
    if len(artifacts) != 1:
        raise ComparisonIntegrityError(
            f"evidence index {_relative(evidence_path, repo_root)} has {len(artifacts)} "
            f"entries for {local_copy}; the generations input must be pinned exactly once"
        )
    artifact = artifacts[0]

    payload = generations_path.read_bytes()
    actual_sha256 = _sha256(payload)
    if actual_sha256 != artifact["sha256"]:
        raise ComparisonIntegrityError(
            f"generations {local_copy}: sha256 {actual_sha256} does not match the "
            f"evidence index pin {artifact['sha256']}; this is not the study-1 run's output"
        )
    rows = _load_jsonl(payload)
    if len(rows) != artifact["row_count"]:
        raise ComparisonIntegrityError(
            f"generations {local_copy}: {len(rows)} rows, but the evidence index pins "
            f"{artifact['row_count']}"
        )

    provenance = {
        "path": _relative(evidence_path, repo_root),
        "source_repository": evidence.get("source_repository"),
        "source_revision": evidence.get("source_revision"),
        "pinned_row_count": artifact["row_count"],
        **_fingerprint(evidence_bytes),
    }
    return rows, provenance


def _rows_by_id(rows: list[dict], label: str) -> dict[str, dict]:
    by_id: dict[str, dict] = {}
    for row in rows:
        row_id = row["id"]
        if row_id in by_id:
            raise ComparisonIntegrityError(f"{label}: duplicate id {row_id!r}")
        by_id[row_id] = row
    return by_id


def _ground_truth_entry(answer_row: dict, label: str) -> dict:
    """The single expected call for a single-call category."""
    entries = answer_row["ground_truth"]
    if len(entries) != 1:
        raise ComparisonIntegrityError(
            f"{label}: id {answer_row['id']!r} has {len(entries)} ground-truth "
            f"entries; the single-call scorer assumes exactly one"
        )
    return entries[0]


# --------------------------------------------------------- key-difference ----


def key_difference(pinned_rows: list[dict], release_rows: list[dict]) -> dict:
    """Locate every row where the two keys disagree, by comparing all of both.

    This is the fact the whole comparison rests on, so it is measured rather
    than asserted: if a future revision moved a second row, encoding "they
    differ at 363" as a constant would silently mis-scope the analysis.
    """
    pinned = _rows_by_id(pinned_rows, "pinned key")
    release = _rows_by_id(release_rows, "release key")

    if set(pinned) != set(release):
        only_pinned = sorted(set(pinned) - set(release))
        only_release = sorted(set(release) - set(pinned))
        raise ComparisonIntegrityError(
            f"the two answer keys cover different items: {len(only_pinned)} only in "
            f"the pinned key ({only_pinned[:3]}), {len(only_release)} only in the "
            f"release key ({only_release[:3]})"
        )

    differing = sorted(
        row_id
        for row_id in pinned
        if json.dumps(pinned[row_id], sort_keys=True)
        != json.dumps(release[row_id], sort_keys=True)
    )
    return {
        "row_count": len(pinned),
        "id_set_sha256": _id_set_sha256(list(pinned)),
        "differing_ids": differing,
        "differing_row_count": len(differing),
        "detail": {
            row_id: {
                "pinned_key_expects": sorted(
                    _ground_truth_entry(pinned[row_id], "pinned key").keys()
                ),
                "release_key_expects": sorted(
                    _ground_truth_entry(release[row_id], "release key").keys()
                ),
            }
            for row_id in differing
        },
    }


# ------------------------------------------------------------- re-scoring ----


def _parsed_call(generation_row: dict) -> dict | None:
    """Rebuild the scorer's input from what the run recorded per item."""
    if not generation_row.get("json_valid"):
        return None
    return {"name": generation_row["parsed_name"], "arguments": generation_row["parsed_args"]}


def rescore(
    generations: list[dict],
    key_rows: list[dict],
    key_label: str,
    expected_candidates: tuple[str, ...] = EXPECTED_CANDIDATES,
) -> dict:
    """Re-score every (candidate, item) under one key with the study's scorer.

    Returns {candidate: {item_id: overall_ok}}. The candidate set must be
    exactly `expected_candidates` and each must cover the key's id set exactly.
    Both halves matter and neither implies the other: a candidate missing some
    items produces a flattering rate on a smaller denominator, while a
    candidate missing *entirely* leaves every surviving candidate complete and
    self-consistent — a report that looks perfect and silently dropped a
    comparison arm.
    """
    key = _rows_by_id(key_rows, key_label)
    outcomes: dict[str, dict[str, bool]] = {}
    for row in generations:
        item_id = row["id"]
        if item_id not in key:
            raise ComparisonIntegrityError(
                f"generations contain id {item_id!r}, which {key_label} does not"
            )
        candidate = outcomes.setdefault(row["model_name"], {})
        if item_id in candidate:
            raise ComparisonIntegrityError(
                f"generations contain a duplicate row for {row['model_name']}/{item_id}"
            )
        _, _, overall_ok, _ = score(_parsed_call(row), _ground_truth_entry(key[item_id], key_label))
        candidate[item_id] = overall_ok

    if not outcomes:
        raise ComparisonIntegrityError("generations contain no rows")

    if set(outcomes) != set(expected_candidates):
        absent = sorted(set(expected_candidates) - set(outcomes))
        unexpected = sorted(set(outcomes) - set(expected_candidates))
        raise ComparisonIntegrityError(
            f"generations cover candidates {sorted(outcomes)}, but this comparison "
            f"expects {sorted(expected_candidates)}; absent={absent} unexpected={unexpected}"
        )

    expected_ids = set(key)
    for candidate, per_item in sorted(outcomes.items()):
        if set(per_item) != expected_ids:
            missing = sorted(expected_ids - set(per_item))
            raise ComparisonIntegrityError(
                f"candidate {candidate!r} covers {len(per_item)} of {len(expected_ids)} "
                f"items under {key_label}; missing {missing[:3]}"
            )

    expected_rows = len(expected_candidates) * len(expected_ids)
    if len(generations) != expected_rows:
        raise ComparisonIntegrityError(
            f"generations have {len(generations)} rows; "
            f"{len(expected_candidates)} candidates x {len(expected_ids)} items = {expected_rows}"
        )
    return outcomes


def check_recomputation_matches_the_run(generations: list[dict], key_rows: list[dict]) -> dict:
    """Prove the re-scoring here reproduces the flags the study-1 run stored.

    The release-key column is only trustworthy if the machinery producing it
    reproduces the pinned-key column that was actually run. This re-derives
    every stored field — name, args, overall, and failure reason — from the
    pinned key's bytes and fails on the first disagreement.
    """
    key = _rows_by_id(key_rows, "pinned key")
    for row in generations:
        recomputed = score(_parsed_call(row), _ground_truth_entry(key[row["id"]], "pinned key"))
        stored = (row["name_ok"], row["args_ok"], row["overall_ok"], row["failure_reason"])
        if recomputed != stored:
            raise ComparisonIntegrityError(
                f"re-scoring {row['model_name']}/{row['id']} under the pinned key gives "
                f"{recomputed}, but the run recorded {stored}; the committed generations "
                f"and the committed scorer no longer agree"
            )
    return {"rows_rechecked": len(generations), "disagreements": 0}


def key_internal_consistency(questions_rows: list[dict], key_rows: list[dict]) -> dict:
    """Does this key only ever expect a name the item actually presented?

    This is the property that decides which key is *defensible*, as opposed to
    which produces the nicer number. A key expecting a name the prompt never
    offers cannot be satisfied by any model, so an item scored against it
    measures the benchmark, not the candidate. Reported as a measured result
    for each key rather than argued in prose.
    """
    questions = _rows_by_id(questions_rows, "questions")
    key = _rows_by_id(key_rows, "answer key")
    try:
        checked = preflight_key_names(questions, key)
    except KeyDefectError as exc:
        return {"consistent": False, "items_checked": len(key), "defect": str(exc)}
    return {"consistent": True, "items_checked": checked, "defect": None}


# The criterion is owned here, not in the manifest, and versioned. The manifest
# names an id and must reproduce this exact wording; a report cannot otherwise
# advertise one rule while `apply_canonical_criterion` applies another -- which
# is the whole failure this binding exists to prevent, since the artifact's
# defence is that the stated rule is the applied rule. Changing either string
# below requires a new id, and a test pins the pair together.
CRITERION_ID = "pinned-and-preflight-valid/v1"
CANONICAL_CRITERION = {
    "criterion_id": CRITERION_ID,
    "rule": (
        "canonical = pinned AND valid; when two pinned keys disagree, the executable "
        "answer-name preflight decides"
    ),
    "validity_check": (
        "eval/bfcl_scoring.py:preflight_key_names — every name an answer key expects must "
        "be among the tools that item actually presented to the model"
    ),
}


def bind_criterion(declared: dict) -> dict:
    """Hold the manifest's stated criterion to the one this module implements.

    The manifest is where a human reads what the rule was, so the text lives
    there too -- but text that is only read is text that can drift from the code
    that acts on it. Every field the code owns must match byte-for-byte, and the
    versioned id must be the current one, so "the rule was applied by machine"
    stays a fact about this artifact rather than a claim inside it.
    """
    if not declared:
        raise ComparisonIntegrityError(
            "the manifest declares no `canonical_key_criterion`; this report is the "
            "permanent record of the canonical-key adjudication and must not be written "
            "without the rule that produced it"
        )
    declared_id = declared.get("criterion_id")
    if declared_id != CRITERION_ID:
        raise ComparisonIntegrityError(
            f"the manifest declares criterion {declared_id!r}, but this code implements "
            f"{CRITERION_ID!r}; the artifact must not record a rule that nothing here applies"
        )
    for field, owned in CANONICAL_CRITERION.items():
        if declared.get(field) != owned:
            raise ComparisonIntegrityError(
                f"criterion {CRITERION_ID}: the manifest's {field!r} is not the text this "
                f"code applies.\n  manifest: {declared.get(field)!r}\n  code:     {owned!r}"
            )
    return {**declared, **CANONICAL_CRITERION}


def apply_canonical_criterion(consistency: dict) -> dict:
    """Which key does the rule select, given only the measured preflight results?

    The rule is `canonical = pinned AND valid; when two pinned keys disagree,
    the preflight decides`. Both candidates are hash-pinned, so provenance no
    longer separates them and validity is the whole of it.

    This takes the measured outcomes and nothing else — no key name, no score,
    no record of what was chosen. That is the point: the report can then show
    the rule producing the selection rather than the selection being asserted
    alongside a rule. Had the preflight failed on the data-fix key, this same
    function would return `release_key` and the headline would be 368/400.

    Returns the selection, or `None` when the criterion does not discriminate —
    which is a fresh owner decision, not a default.
    """
    valid = {name: bool(result["consistent"]) for name, result in consistency.items()}
    survivors = sorted(name for name, ok in valid.items() if ok)
    if len(survivors) == 1:
        return {
            "criterion_id": CRITERION_ID,
            "selected": survivors[0],
            "decided": True,
            "validity": valid,
            "reason": (
                f"{survivors[0]} is the only candidate whose every expected name appears "
                f"among the tools its item presented"
            ),
        }
    return {
        "criterion_id": CRITERION_ID,
        "selected": None,
        "decided": False,
        "validity": valid,
        "reason": (
            "all candidate keys are valid, so the preflight does not discriminate"
            if survivors
            else "no candidate key is valid, so there is nothing to select"
        ),
    }


# -------------------------------------------------------------- contrasts ----


def contrasts(outcomes: dict[str, dict[str, bool]], reference: str) -> dict:
    """Every candidate-vs-reference discordant set, with its counts and digest.

    McNemar's test reads only the discordant items, so this is the exact input
    the paired analysis consumes. Recording the digest of each discordant id
    set makes "the key choice does not move the paired tests" a checkable
    claim rather than an argument.
    """
    if reference not in outcomes:
        raise ComparisonIntegrityError(
            f"reference candidate {reference!r} is not among {sorted(outcomes)}"
        )
    out: dict[str, dict] = {}
    for candidate in sorted(outcomes):
        if candidate == reference:
            continue
        b = sorted(
            i for i in outcomes[reference] if outcomes[reference][i] and not outcomes[candidate][i]
        )
        c = sorted(
            i for i in outcomes[reference] if not outcomes[reference][i] and outcomes[candidate][i]
        )
        out[f"{reference}_vs_{candidate}"] = {
            "reference_only_correct": len(b),
            "candidate_only_correct": len(c),
            "discordant": len(b) + len(c),
            "discordant_id_sha256": _id_set_sha256(b + c),
        }
    return out


def _invariance(pinned: dict, release: dict) -> dict:
    """Compare the two keys' contrast tables and fail if any of it moved."""
    if set(pinned) != set(release):
        raise ComparisonIntegrityError("the two keys produced different contrast sets")
    differences = {
        name: (pinned[name], release[name])
        for name in pinned
        if pinned[name] != release[name]
    }
    return {
        "identical": not differences,
        "contrasts_compared": sorted(pinned),
        "differences": {name: {"pinned": p, "release": r} for name, (p, r) in differences.items()},
    }


# --------------------------------------------------------------- exposure ----


def _key_names(answer_row: dict) -> set[str]:
    names: set[str] = set()
    for entry in answer_row["ground_truth"]:
        names |= set(entry.keys())
    return names


def qualified_name_stats(exposure_inputs: dict) -> dict:
    """How exposed each category is to the qualified-vs-unqualified question.

    Reads only already-verified payloads. These counts are quoted in the
    preregistration amendment and in the upstream issue draft, so they must be
    as traceable as the scores are — computing them from raw files on disk
    would let an edited question file move a published figure while the
    report's recorded inputs stayed reassuringly unchanged.
    """
    out: dict[str, dict] = {}
    for category, roles in exposure_inputs.items():
        questions = _rows_by_id(roles[QUESTIONS_ROLE]["rows"], f"{category} questions")
        answers = _rows_by_id(roles[PINNED_KEY_ROLE]["rows"], f"{category} answer key")

        qualified_rows = 0
        key_not_presented = 0
        tail_collisions = 0
        for row_id, question in questions.items():
            presented = {fn["name"] for fn in question["function"]}
            keyed = _key_names(answers[row_id])
            if any("." in name for name in keyed):
                qualified_rows += 1
            if not keyed <= presented:
                key_not_presented += 1
            tails = [name.split(".")[-1] for name in presented]
            if len(set(tails)) < len(tails):
                tail_collisions += 1

        out[category] = {
            "rows": len(questions),
            "rows_with_module_qualified_key": qualified_rows,
            "rows_where_key_name_not_among_presented_tools": key_not_presented,
            "rows_where_tools_share_an_unqualified_tail": tail_collisions,
        }
    return out


# ----------------------------------------------------------------- report ----


def build_report(
    manifest_path: Path = DEFAULT_MANIFEST,
    generations_path: Path = DEFAULT_GENERATIONS,
    repo_root: Path = REPO_ROOT,
    *,
    evidence_path: Path = DEFAULT_EVIDENCE,
    expected_candidates: tuple[str, ...] = EXPECTED_CANDIDATES,
    exposure_categories: tuple[str, ...] = EXPOSURE_CATEGORIES,
) -> dict:
    inputs = load_verified_inputs(manifest_path, repo_root, exposure_categories)
    generations_bytes = generations_path.read_bytes()
    generations, evidence_provenance = load_verified_generations(
        generations_path, evidence_path, repo_root
    )

    difference = key_difference(inputs["pinned_key"]["rows"], inputs["release_key"]["rows"])
    questions = _rows_by_id(inputs["questions"]["rows"], "questions")
    if set(questions) != set(_rows_by_id(inputs["pinned_key"]["rows"], "pinned key")):
        raise ComparisonIntegrityError(
            "the questions file and the answer key cover different item ids"
        )

    recomputation = check_recomputation_matches_the_run(generations, inputs["pinned_key"]["rows"])
    pinned_outcomes = rescore(
        generations, inputs["pinned_key"]["rows"], "the pinned key", expected_candidates
    )
    release_outcomes = rescore(
        generations, inputs["release_key"]["rows"], "the release key", expected_candidates
    )

    candidates = sorted(pinned_outcomes)
    n = {candidate: len(pinned_outcomes[candidate]) for candidate in candidates}
    scores = {
        "n": n,
        "pinned_key": {c: sum(pinned_outcomes[c].values()) for c in candidates},
        "release_key": {c: sum(release_outcomes[c].values()) for c in candidates},
    }

    emitted = {
        row["model_name"]: row["parsed_name"] for row in generations if row["id"] == DISPUTED_ID
    }
    differing_ids = difference["differing_ids"]
    concordance = {
        row_id: {
            "emitted_by_candidate": {
                row["model_name"]: row["parsed_name"] for row in generations if row["id"] == row_id
            },
            "pinned_key_outcome": {c: pinned_outcomes[c][row_id] for c in candidates},
            "release_key_outcome": {c: release_outcomes[c][row_id] for c in candidates},
            "concordant_under_pinned_key": (
                len({pinned_outcomes[c][row_id] for c in candidates}) == 1
            ),
            "concordant_under_release_key": (
                len({release_outcomes[c][row_id] for c in candidates}) == 1
            ),
        }
        for row_id in differing_ids
    }

    pinned_contrasts = contrasts(pinned_outcomes, REFERENCE_CANDIDATE)
    release_contrasts = contrasts(release_outcomes, REFERENCE_CANDIDATE)

    consistency = {
        "pinned_key": key_internal_consistency(
            inputs["questions"]["rows"], inputs["pinned_key"]["rows"]
        ),
        "release_key": key_internal_consistency(
            inputs["questions"]["rows"], inputs["release_key"]["rows"]
        ),
    }

    criterion = bind_criterion(inputs.get("_criterion"))
    derived = apply_canonical_criterion(consistency)

    report = {
        "schema_version": 3,
        "release_commit": inputs["release_key"]["source_revision"],
        "datafix_commit": inputs["pinned_key"]["source_revision"],
        "adjudication": {
            **criterion,
            "derived_from_measurement": derived,
            "candidate_keys": {
                "pinned_key": {
                    "source_revision": inputs["pinned_key"]["source_revision"],
                    "sha256": inputs["pinned_key"]["sha256"],
                    "git_blob_sha1": inputs["pinned_key"]["git_blob_sha1"],
                },
                "release_key": {
                    "source_revision": inputs["release_key"]["source_revision"],
                    "sha256": inputs["release_key"]["sha256"],
                    "git_blob_sha1": inputs["release_key"]["git_blob_sha1"],
                },
            },
        },
        "inputs": {
            "manifest": inputs["manifest"],
            "questions": {k: v for k, v in inputs["questions"].items() if k != "rows"},
            "pinned_key": {k: v for k, v in inputs["pinned_key"].items() if k != "rows"},
            "release_key": {k: v for k, v in inputs["release_key"].items() if k != "rows"},
            "generations": {
                "path": _relative(generations_path, repo_root),
                "rows": len(generations),
                "candidates": candidates,
                "verified_against": evidence_provenance["path"],
                **_fingerprint(generations_bytes),
            },
            "evidence_index": evidence_provenance,
            "exposure": {
                category: {
                    role: {k: v for k, v in spec.items() if k != "rows"}
                    for role, spec in roles.items()
                }
                for category, roles in inputs.get("exposure", {}).items()
            },
        },
        "key_difference": difference,
        "key_internal_consistency": consistency,
        "recomputation_check": recomputation,
        "scores": scores,
        "disputed_items": concordance,
        "contrasts": {
            "reference": REFERENCE_CANDIDATE,
            "pinned_key": pinned_contrasts,
            "release_key": release_contrasts,
            "invariance": _invariance(pinned_contrasts, release_contrasts),
        },
        # Kept for continuity with schema_version 1 consumers.
        "disputed_item_emitted": emitted,
    }
    if exposure_categories:
        report["qualified_name_exposure"] = qualified_name_stats(inputs["exposure"])
    return report


def check_report_invariants(report: dict) -> None:
    """Fail the run if the report does not support the claims made from it.

    These are the properties the ADR, the README number, and the kill decision
    all lean on. A report that violates one of them must not be written to
    disk looking like evidence.
    """
    if report["key_difference"]["differing_row_count"] != 1:
        raise ComparisonIntegrityError(
            f"the two keys differ at {report['key_difference']['differing_row_count']} rows "
            f"({report['key_difference']['differing_ids']}); this comparison, the "
            f"concordance argument, and the headline haircut are all scoped to exactly one"
        )
    for row_id, detail in report["disputed_items"].items():
        if not (detail["concordant_under_pinned_key"] and detail["concordant_under_release_key"]):
            raise ComparisonIntegrityError(
                f"disputed item {row_id} is not concordant across candidates, so the key "
                f"choice does move the paired contrasts; the invariance argument fails"
            )
    if not report["contrasts"]["invariance"]["identical"]:
        raise ComparisonIntegrityError(
            f"the discordant sets differ between the two keys: "
            f"{report['contrasts']['invariance']['differences']}"
        )

    adjudication = report["adjudication"]
    derived = adjudication["derived_from_measurement"]
    if adjudication.get("criterion_id") != CRITERION_ID:
        raise ComparisonIntegrityError(
            f"the report records criterion {adjudication.get('criterion_id')!r}, but this "
            f"code implements {CRITERION_ID!r}"
        )
    if derived.get("criterion_id") != adjudication["criterion_id"]:
        raise ComparisonIntegrityError(
            f"the recorded criterion {adjudication['criterion_id']!r} is not the one that "
            f"produced the selection ({derived.get('criterion_id')!r})"
        )
    for field, owned in CANONICAL_CRITERION.items():
        if adjudication.get(field) != owned:
            raise ComparisonIntegrityError(
                f"the report's {field!r} is not the text this code applies; the artifact "
                f"would advertise a rule the selection did not follow"
            )
    if not derived["decided"]:
        raise ComparisonIntegrityError(
            f"the canonical-key criterion does not decide between the candidates "
            f"({derived['reason']}); the recorded selection "
            f"{adjudication['selected']!r} therefore rests on something this artifact "
            f"does not measure, and needs a fresh owner decision"
        )
    if adjudication["selected"] != derived["selected"]:
        raise ComparisonIntegrityError(
            f"the manifest records {adjudication['selected']!r} as canonical, but the "
            f"recorded criterion applied to the measured preflight outcomes selects "
            f"{derived['selected']!r} ({derived['reason']}); the rule and the choice "
            f"have come apart"
        )
    selected_revision = adjudication["candidate_keys"][derived["selected"]]["source_revision"]
    if adjudication["selected_source_revision"] != selected_revision:
        raise ComparisonIntegrityError(
            f"the adjudication names revision {adjudication['selected_source_revision']} "
            f"but {derived['selected']} is pinned at {selected_revision}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--generations", type=Path, default=DEFAULT_GENERATIONS)
    parser.add_argument("--json", type=Path, help="also write the report as JSON")
    args = parser.parse_args()

    report = build_report(args.manifest, args.generations)
    check_report_invariants(report)

    difference = report["key_difference"]
    scores = report["scores"]

    print("verified inputs (sha256):")
    for name, spec in report["inputs"].items():
        if name == "exposure":
            for category, roles in spec.items():
                for role, entry in roles.items():
                    label = f"{category}/{role}"
                    print(f"  {label:<34} {entry['sha256']}  {entry['path']}")
            continue
        print(f"  {name:<34} {spec['sha256']}  {spec.get('path', '')}")
    print()

    print(
        f"answer keys compared over {difference['row_count']} rows; "
        f"differing rows: {difference['differing_row_count']} "
        f"({', '.join(difference['differing_ids'])})"
    )
    for row_id, detail in difference["detail"].items():
        print(f"  {row_id}")
        pinned_commit = report["datafix_commit"][:8]
        release_commit = report["release_commit"][:8]
        print(f"    pinned key ({pinned_commit}) expects : {detail['pinned_key_expects']}")
        print(f"    release key ({release_commit}) expects: {detail['release_key_expects']}")
        emitted = report["disputed_items"][row_id]["emitted_by_candidate"]
        print(f"    emitted by candidates                  : {sorted(set(emitted.values()))}")
    print()
    print("internal consistency (does the key only expect names the item presented?):")
    for name, result in report["key_internal_consistency"].items():
        if result["consistent"]:
            print(f"  {name:<12} consistent on all {result['items_checked']} rows")
        else:
            first = result["defect"].splitlines()[1].strip()
            print(f"  {name:<12} INCONSISTENT — {first}")
    print()

    adjudication = report["adjudication"]
    derived = adjudication["derived_from_measurement"]
    print("canonical-key adjudication:")
    print(f"  criterion  {adjudication['criterion_id']} (owned by this module, not the manifest)")
    print(f"  rule       {adjudication['rule']}")
    print(f"  applied    selects {derived['selected']} — {derived['reason']}")
    print(
        f"  recorded   {adjudication['selected']} "
        f"({adjudication['selected_source_revision'][:8]}), "
        f"by {adjudication['adjudicated_by']} on {adjudication['adjudicated_on']} "
        f"[{adjudication['adjudication_ref']}]"
    )
    print("  the recorded choice is the one the rule produces from the measurements above")
    print()

    print(
        f"re-scored {report['recomputation_check']['rows_rechecked']} rows under the pinned key "
        f"with {report['recomputation_check']['disagreements']} disagreements against the "
        f"flags study 1 recorded"
    )
    print()

    print(f"{'candidate':<10} {'pinned key':>12} {'release key':>13}")
    for candidate in sorted(scores["n"]):
        total = scores["n"][candidate]
        print(
            f"{candidate:<10} {scores['pinned_key'][candidate]:>8}/{total} "
            f"{scores['release_key'][candidate]:>9}/{total}"
        )
    print()

    print(f"paired contrasts against '{report['contrasts']['reference']}' (McNemar inputs):")
    print(f"  {'contrast':<20} {'b':>4} {'c':>4} {'discordant':>11}  discordant-id sha256")
    for name, stats in report["contrasts"]["pinned_key"].items():
        print(
            f"  {name:<20} {stats['reference_only_correct']:>4} "
            f"{stats['candidate_only_correct']:>4} {stats['discordant']:>11}  "
            f"{stats['discordant_id_sha256'][:16]}…"
        )
    print(
        f"  identical under both keys: {report['contrasts']['invariance']['identical']} "
        f"(so every paired delta and McNemar p-value is invariant to the key choice)"
    )
    print()

    if "qualified_name_exposure" in report:
        print("qualified-name exposure by category:")
        print(
            f"  {'category':<14} {'rows':>5} {'qualified key':>14} "
            f"{'key not offered':>16} {'tail collisions':>16}"
        )
        for category, stats in report["qualified_name_exposure"].items():
            rows = stats["rows"]
            print(
                f"  {category:<14} {rows:>5} "
                f"{stats['rows_with_module_qualified_key']:>9} "
                f"({stats['rows_with_module_qualified_key'] / rows:>4.0%}) "
                f"{stats['rows_where_key_name_not_among_presented_tools']:>16} "
                f"{stats['rows_where_tools_share_an_unqualified_tail']:>11} "
                f"({stats['rows_where_tools_share_an_unqualified_tail'] / rows:>4.0%})"
            )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
