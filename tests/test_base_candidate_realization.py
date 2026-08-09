"""The comparator's construction is on the record, and cannot drift from it.

Background — the owner's governance rider on the ladder's step-3 fix (#general
msg 2903): *"check whether the preregistration specifies how the base candidate
is realized. If it does, changing from adapter-disabled to a separately loaded
base needs an amendment before the run. If it does not, record the choice in
the run manifest so the comparator's construction is on the record either
way."*

**It does not specify.** `docs/prereg-study2.md` §0.2 names the candidates and
pins both artifacts and revisions; §0.1 and §0.3–§0.6 add nothing about
realization. So the second branch applies: record it.

These tests hold three things that would otherwise be true only by habit:

  1. The manifest carries the realization, so a number's provenance says how
     its comparator was built. `"candidates": ["base", "sft"]` does not.
  2. The recorded string and the code path that implements it cannot drift.
     A constant that says `adapter_disabled` while the code loads a raw base
     model is worse than no constant — it is provenance that lies.
  3. The prereg still does not specify. If a future amendment adds a
     realization clause, this test fails and forces the code to be reconciled
     with it rather than quietly diverging.

Why it matters that the two realizations are not interchangeable: they are
*intended* to be numerically equivalent, but the PEFT-wrapped model keeps its
adapter modules resident and routes the forward through the wrapper. Rungs 3
and 4 of `eval/isolation_ladder.py` isolate precisely that difference, and the
2026-08-08 §0 crash happened at this realization.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "eval"))

from bfcl_simple import (  # noqa: E402
    BASE_CANDIDATE_REALIZATION,
    build_initial_manifest,
)

PREREG = REPO_ROOT / "docs" / "prereg-study2.md"
SOURCE = REPO_ROOT / "eval" / "bfcl_simple.py"


class _Args:
    """Minimal stand-in for the argparse namespace the manifest builder reads."""

    category = "multiple"
    base_model = "meta-llama/Llama-3.1-8B-Instruct"
    base_revision = "0e9e39f249a16976918f6564b8830bc894c89659"
    sft_adapter = "centuriandip/llama-3.1-8b-tools-sft"
    sft_adapter_subfolder = "adapter"
    sft_adapter_revision = "b6f4da479f8c6fc044ee8b802a92f47780f970c5"
    checkpoint_root = Path("./outputs/dpo-v2-full")
    checkpoints: list[str] = []
    max_new_tokens = 512


class _Paths:
    questions = REPO_ROOT / "README.md"    # any real file; only its sha is read
    answer_key = REPO_ROOT / "README.md"


def _manifest(candidates: list[str]) -> dict:
    return build_initial_manifest(_Args(), candidates, _Paths(), n_prompts=200)


def test_the_manifest_records_how_the_base_candidate_was_realized() -> None:
    manifest = _manifest(["base", "sft"])
    assert manifest["base_candidate_realization"] == BASE_CANDIDATE_REALIZATION
    assert manifest["base_candidate_realization"], "must not be empty"


def test_no_base_candidate_means_no_realization_claim() -> None:
    """Recording a base realization for a run that never evaluated `base` would
    put a claim in the provenance about something the run did not do."""
    assert _manifest(["sft"])["base_candidate_realization"] is None


def test_the_recorded_realization_matches_the_code_that_implements_it() -> None:
    """Provenance that disagrees with the code is worse than none.

    The constant currently says the base candidate is the PEFT model with its
    adapter disabled. If someone applies the step-3 fix — realizing `base` as a
    separately loaded raw base model — and forgets the constant, every manifest
    from that point on misdescribes its own comparator, and nothing else in the
    system would notice.
    """
    source = SOURCE.read_text(encoding="utf-8")
    if BASE_CANDIDATE_REALIZATION == "peft_model_with_adapter_disabled":
        assert "adapter_ctx = model.disable_adapter()" in source, (
            "constant claims adapter-disabled but the code no longer calls "
            "disable_adapter()"
        )
    else:
        assert "adapter_ctx = model.disable_adapter()" not in source, (
            f"constant says {BASE_CANDIDATE_REALIZATION!r} but the code still "
            "disables an adapter"
        )


def test_the_prereg_still_does_not_specify_the_realization() -> None:
    """The rider's first branch is checked, not assumed.

    If an amendment ever adds a realization clause to the prereg, the choice
    stops being free and this test is what says so — loudly, before a run,
    rather than after someone notices the manifest and the prereg disagree.

    Deliberately narrow: it looks for realization *mechanism* language near the
    base candidate, not for the words "base" or "adapter" on their own, both of
    which appear throughout for unrelated reasons (the adapter is pinned by
    revision in several places, and training arms discuss `ref` adapters).
    """
    text = PREREG.read_text(encoding="utf-8")
    mechanism = re.compile(
        r"disable_adapter|adapter[- ]disabled|disabled adapter"
        r"|separately loaded (?:raw )?base"
        r"|base candidate is realiz|realiz\w* (?:as|by) a? ?(?:raw )?base",
        re.I,
    )
    hits = [
        f"{n}: {line.strip()}"
        for n, line in enumerate(text.splitlines(), start=1)
        if mechanism.search(line)
    ]
    assert not hits, (
        "the prereg now appears to constrain how the base candidate is "
        "realized. The rider's FIRST branch applies: changing the realization "
        "needs an owner amendment before the run, and this test must be "
        "updated to encode the new constraint.\n" + "\n".join(hits)
    )


def test_the_prereg_does_pin_the_things_it_actually_governs() -> None:
    """Guard the guard.

    The test above is a claim about absence, which is vacuous if the file
    stopped being readable or the section were renamed. Anchor it on the §0.2
    content that must be present.
    """
    text = PREREG.read_text(encoding="utf-8")
    assert "### 0.2 What the probe measures" in text
    assert "`base`, `sft` only" in text
    assert "0e9e39f249a16976918f6564b8830bc894c89659" in text
