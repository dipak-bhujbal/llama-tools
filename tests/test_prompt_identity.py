"""The §0 crash prompt, re-derived on CPU and checked against its pin.

Why this file exists
--------------------
On 2026-08-09 a retry pod died 11 seconds in, before loading a model, because
`eval/isolation_ladder.py` asserted the first production prompt was 609 tokens
and the production path produced 610. The assertion had never been exercised:
the ladder is newer than the crash it reproduces. Counting the same string with
`add_special_tokens=False` reproduces 609 exactly, which explains the
discrepancy — but no surviving artifact identifies the command that originally
produced 609, so that is an explanation, not a reconstruction of history.

Nothing about that needed a GPU, a model, or a billing pod. It needed one
tokenizer and about a second. That is what this test is — the smoke gate one
layer earlier, on the laptop, before anything is rented.

What is pinned, and what deliberately is not
--------------------------------------------
The gate is IDENTITY: the sha256 of the rendered prompt string and the sha256
of its input ids under one named serialization. Length is recorded but not
enforced, because a count is both brittle (tokenizer revisions, BOS handling)
and insufficient (a different 610-token prompt would satisfy it).

The pinned sequence carries TWO `<|begin_of_text|>` tokens. That is not a
defect to be corrected here: `build_prompt` renders a string that already
contains BOS, `tokenizer(prompt)` adds another, and commit 2d8abdb — the one
that crashed on 2026-08-08 — did exactly the same. Feeding a single-BOS prompt
would mean no longer reproducing the fault. Whether *evaluation* should keep
double BOS is a separate, open governance question; this file does not answer
it and must not be edited as though it did.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (str(REPO_ROOT), str(REPO_ROOT / "eval")):
    if p not in sys.path:
        sys.path.insert(0, p)

import eval.isolation_ladder as il  # noqa: E402


@pytest.fixture(scope="module")
def pinned_tokenizer():
    """The pinned tokenizer, or a skip — never a silent pass.

    Skipped when the tokenizer is not present locally (no network, no HF auth,
    cold cache). A skip is visible in the run summary; a stubbed tokenizer would
    make this file assert nothing while appearing to pass, which is the exact
    failure mode it was written to prevent.
    """
    transformers = pytest.importorskip("transformers")
    try:
        # local_files_only: this test must never reach the network. Without it a
        # cold cache would download a gated repo mid-suite -- slow, credential-
        # dependent, and a test that quietly needs the internet is not the
        # laptop-speed gate this file exists to be.
        return transformers.AutoTokenizer.from_pretrained(
            il.BASE_MODEL_REPO,
            revision=il.BASE_MODEL_REVISION,
            local_files_only=True,
        )
    except Exception as exc:  # noqa: BLE001 - cold cache, gated repo, no auth
        pytest.skip(f"pinned tokenizer not in local cache: {type(exc).__name__}: {exc}")


@pytest.fixture(scope="module")
def built(pinned_tokenizer):
    prompt_id, prompt = il.load_first_production_prompt(pinned_tokenizer)
    ids = list(pinned_tokenizer(prompt)["input_ids"])
    return prompt_id, prompt, ids


def test_prompt_id_is_the_pinned_one(built) -> None:
    prompt_id, _, _ = built
    assert prompt_id == il.EXPECTED_PROMPT_ID


def test_prompt_string_hash_matches_pin(built) -> None:
    """Catches a changed prompt: different data, template, or system text."""
    _, prompt, _ = built
    assert hashlib.sha256(prompt.encode("utf-8")).hexdigest() == il.EXPECTED_PROMPT_SHA256


def test_input_ids_hash_matches_pin(built) -> None:
    """Catches a same-string, different-tokenization change.

    This is the case a token count cannot see and the one that actually bit us:
    the string was unchanged and the sequence was not.
    """
    _, _, ids = built
    assert il.prompt_ids_digest(ids) == il.EXPECTED_PROMPT_IDS_SHA256


def test_recorded_token_count_still_describes_reality(built) -> None:
    """Not the gate — but if this drifts, the recorded number is now a lie."""
    _, _, ids = built
    assert len(ids) == il.OBSERVED_PROMPT_TOKENS


def test_pinned_sequence_carries_the_double_bos_that_crashed(built, pinned_tokenizer) -> None:
    """The reproduction property, asserted explicitly so it cannot be 'fixed'.

    If someone later sets add_special_tokens=False in the production path, every
    hash above changes and those tests fail — but they would fail without saying
    why. This one names the reason.
    """
    _, _, ids = built
    bos = pinned_tokenizer.bos_token_id
    assert bos is not None
    assert ids[0] == bos and ids[1] == bos, (
        "the pinned crash prompt must carry two BOS tokens; commit 2d8abdb "
        "crashed on the double-BOS sequence, so a single-BOS prompt would not "
        "be a reproduction"
    )


def test_gate_accepts_the_real_prompt(built, pinned_tokenizer) -> None:
    """End to end: the ladder's own gate passes on the genuine article."""
    prompt_id, prompt, _ = built
    identity = il.assert_prompt_identity(pinned_tokenizer, prompt, prompt_id)
    assert identity["tokens"] == il.OBSERVED_PROMPT_TOKENS
    assert identity["prompt_sha256"] == il.EXPECTED_PROMPT_SHA256
    assert identity["prompt_ids_sha256"] == il.EXPECTED_PROMPT_IDS_SHA256
