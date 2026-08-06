"""The single-call BFCL scoring rule, with no model or torch dependency.

This lives apart from `bfcl_simple.py` so that anything which needs to *re*-score
already-generated outputs — the answer-key comparison, an audit, a reviewer
checking a published number — runs the exact rule the study ran, rather than a
second implementation that agrees with it until it doesn't. `bfcl_simple.py`
imports from here; there is one definition of "correct".

The rule: exact function-name match, every argument's value in its accepted
list, and no arguments the answer key does not name. It is not BFCL's official
AST-based scorer (see the note in `bfcl_simple.py`).
"""

from __future__ import annotations

from typing import Any


class KeyDefectError(ValueError):
    """An answer key expects a name the item never presented to the model."""


def preflight_key_names(questions: dict[str, dict], answers: dict[str, dict]) -> int:
    """Refuse to score a key that expects a name the model was not offered.

    Exact name matching is only a fair rule while the key's name is among the
    tools the item presented. Where it is not, the item is unpassable by
    construction, and a model that answered with the offered name is marked
    wrong for the benchmark's mistake — the `simple_python_363` defect. Exact
    matching alone cannot notice that; this can.

    Called before generation so a key defect costs no GPU time, and raised
    rather than logged so a defective item cannot quietly become a model
    failure in a headline number. Returns the number of items checked.

    Both arguments are {item_id: row}, from the questions and possible_answer
    files of the same pinned revision.
    """
    defects: list[str] = []
    for item_id, answer_row in answers.items():
        question = questions.get(item_id)
        if question is None:
            defects.append(f"{item_id}: in the answer key but not in the questions file")
            continue
        presented = {fn["name"] for fn in question["function"]}
        expected: set[str] = set()
        for entry in answer_row["ground_truth"]:
            expected |= set(entry.keys())
        missing = sorted(expected - presented)
        if missing:
            defects.append(
                f"{item_id}: key expects {missing}, presented tools are {sorted(presented)}"
            )

    if defects:
        shown = "\n  ".join(defects[:10])
        more = f"\n  ...and {len(defects) - 10} more" if len(defects) > 10 else ""
        raise KeyDefectError(
            f"{len(defects)} item(s) expect a function name that was never presented to "
            f"the model. Under the exact-match rule these are unscoreable and must be "
            f"reported as key defects, not graded:\n  {shown}{more}"
        )
    return len(answers)


def values_equal(parsed_val: Any, accepted_val: Any) -> bool:
    """Light coercion equality: numeric compare if both numbers, else deep =="""
    if isinstance(parsed_val, bool) or isinstance(accepted_val, bool):
        return parsed_val == accepted_val
    if isinstance(parsed_val, (int, float)) and isinstance(accepted_val, (int, float)):
        return float(parsed_val) == float(accepted_val)
    return parsed_val == accepted_val


def score(parsed: Any, gt_entry: dict) -> tuple[bool, bool, bool, str]:
    """Return (name_ok, args_ok, overall_ok, failure_reason).

    gt_entry: {"function_name": {"arg1": [accepted, ...], "arg2": [...]}}
    """
    if parsed is None:
        return False, False, False, "json_unparseable"
    if "name" not in parsed or "arguments" not in parsed:
        return False, False, False, "missing_name_or_arguments"
    parsed_name = parsed["name"]
    parsed_args = parsed["arguments"]
    if not isinstance(parsed_args, dict):
        return False, False, False, "arguments_not_dict"

    gt_name = next(iter(gt_entry.keys()))
    gt_args = gt_entry[gt_name]
    name_ok = parsed_name == gt_name

    # no-extra-args: every parsed key must be in gt
    for k in parsed_args:
        if k not in gt_args:
            return name_ok, False, False, f"extra_arg:{k}"

    # each required gt arg must match; optional means "" in accepted list
    args_ok = True
    fail_reason = ""
    for arg_name, accepted in gt_args.items():
        optional = "" in accepted
        if arg_name not in parsed_args:
            if optional:
                continue
            args_ok = False
            fail_reason = f"missing_arg:{arg_name}"
            break
        parsed_val = parsed_args[arg_name]
        if not any(values_equal(parsed_val, av) for av in accepted):
            args_ok = False
            fail_reason = f"bad_value:{arg_name}"
            break

    overall_ok = name_ok and args_ok
    if overall_ok:
        return True, True, True, ""
    if not name_ok and not fail_reason:
        fail_reason = "bad_name"
    return name_ok, args_ok, overall_ok, fail_reason
