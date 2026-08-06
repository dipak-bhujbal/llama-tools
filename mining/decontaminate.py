"""Screen mined training prompts for overlap with the frozen BFCL eval sets.

Why this exists
----------------
Study 2 mines on-policy preference pairs from a rollout pool. If a mined
prompt reproduces — verbatim or near-verbatim — text or a function schema
from our frozen BFCL v4 eval questions, training on it would let the model
practice on (a paraphrase of) a question it is later scored on, quietly
inflating eval numbers. Per `docs/WORKING-AGREEMENT.md`, decontamination
removes items from the training pool only; eval files are pinned and
read-only (see `eval/manifests/bfcl_v4_study2.json` and
`eval/fetch_pinned_bfcl.py`) — this module only ever reads them.

Two independent, cheap signals, in the spirit of `eval/bfcl_leakage_check.py`:

1. **13-gram exact overlap** on normalized user-turn text. A 13-word run is
   long enough that unrelated prompts essentially never share one by
   coincidence, but it still catches a mined prompt that borrows a long
   stretch of wording from an eval question even when it is not a
   byte-for-byte duplicate of the whole prompt.
2. **Exact function-name overlap.** A mined example that offers a function
   whose *name* exactly matches one available in any eval question is also
   contamination: the model would be rehearsing tool selection against a
   schema it is later scored on, regardless of how the surrounding prompt
   is worded.

Short prompts (fewer words than the n-gram size) deliberately produce zero
n-grams rather than falling back to shorter windows the way
`data/dedupe.py`'s MinHash tokenizer does — a size-1 "gram" would let any
single common word trigger a false contamination verdict, which is worse
than missing an n-gram signal we never had enough text for in the first
place. The function-name screen has no length floor and still applies to
short prompts.

Loading (reading manifests, verifying sha256, and indexing eval text) is
deferred until the first call to `is_contaminated` or `screened_manifest`,
and cached after that — constructing a `Decontaminator` is cheap even if a
caller builds one before deciding whether the mining run needs it. Every
eval file used is fail-closed: if the bytes on disk do not match the
manifest's pinned sha256, `is_contaminated`/`screened_manifest` raise
instead of silently screening against a changed file.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent


class EvalIntegrityError(ValueError):
    """A pinned eval file's bytes on disk do not match its manifest sha256."""


def _normalize_tokens(text: str) -> list[str]:
    """Lowercase, collapse whitespace, and drop punctuation-only tokens."""
    return [token for token in text.lower().split() if _has_word_char(token)]


def _has_word_char(token: str) -> bool:
    return any(ch.isalnum() for ch in token)


def _ngrams(tokens: Sequence[str], n: int) -> Iterator[str]:
    """Yield space-joined word n-grams of size `n` from `tokens`.

    Prompts with fewer than `n` tokens yield nothing — see the module
    docstring for why we do not fall back to a shorter window.
    """
    if n <= 0 or len(tokens) < n:
        return
    for i in range(len(tokens) - n + 1):
        yield " ".join(tokens[i : i + n])


def _extract_user_texts(question: Any) -> list[str]:
    """Pull every user-role message's text out of a BFCL `question` field.

    `question` is a list of conversation rounds, each a list of message
    dicts with `role`/`content`. Malformed shapes are skipped rather than
    raised on — a single odd row in a large eval file should not prevent
    screening against the rest of it.
    """
    texts: list[str] = []
    if not isinstance(question, list):
        return texts
    for round_messages in question:
        if not isinstance(round_messages, list):
            continue
        for message in round_messages:
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                texts.append(content)
    return texts


class Decontaminator:
    """Flags mined prompts that overlap the frozen BFCL eval question sets.

    Construct once with the manifest file(s) that pin the eval set (e.g.
    `eval/manifests/bfcl_v4_study2.json`), then call `is_contaminated` for
    every mined candidate. Only manifest entries with `role == "questions"`
    are screened; answer keys are skipped since they are never something a
    mined prompt could copy from.
    """

    def __init__(self, eval_files: Sequence[Path], ngram: int = 13) -> None:
        self._manifest_paths = [Path(p) for p in eval_files]
        self._ngram = ngram
        self._loaded = False
        self._ngram_index: dict[str, str] = {}
        self._function_index: dict[str, str] = {}
        self._screened: list[dict[str, Any]] = []

    def is_contaminated(
        self, prompt_text: str, function_names: Sequence[str]
    ) -> tuple[bool, str | None]:
        """Return (True, reason) if `prompt_text`/`function_names` overlap the
        eval set, else (False, None). The n-gram screen is checked before the
        function-name screen; the first match wins.
        """
        self._ensure_loaded()

        tokens = _normalize_tokens(prompt_text)
        for gram in _ngrams(tokens, self._ngram):
            category = self._ngram_index.get(gram)
            if category is not None:
                return True, f"ngram_overlap:{category}"

        for name in function_names:
            category = self._function_index.get(name)
            if category is not None:
                return True, f"fn_name:{name}:{category}"

        return False, None

    def screened_manifest(self) -> list[dict[str, Any]]:
        """Return the manifest entries actually screened against.

        One dict per screened file: `local_path`, `sha256`, `category`,
        `role`. Intended to be recorded verbatim in the mining ledger so the
        screened set is auditable after the fact.
        """
        self._ensure_loaded()
        return [dict(entry) for entry in self._screened]

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        for manifest_path in self._manifest_paths:
            manifest = json.loads(manifest_path.read_text())
            for spec in manifest.get("files", []):
                if spec.get("role") != "questions":
                    continue
                self._index_question_file(spec)
        self._loaded = True

    def _index_question_file(self, spec: dict[str, Any]) -> None:
        local_path = Path(spec["local_path"])
        if not local_path.is_absolute():
            local_path = REPO_ROOT / local_path
        payload = local_path.read_bytes()

        actual_sha256 = hashlib.sha256(payload).hexdigest()
        expected_sha256 = spec["sha256"]
        if actual_sha256 != expected_sha256:
            raise EvalIntegrityError(
                f"{spec['local_path']}: sha256 {actual_sha256} != manifest "
                f"{expected_sha256} (pinned eval file changed on disk; "
                "refusing to screen against it)"
            )

        category = spec["category"]
        for line in payload.decode("utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)

            for user_text in _extract_user_texts(row.get("question")):
                tokens = _normalize_tokens(user_text)
                for gram in _ngrams(tokens, self._ngram):
                    self._ngram_index.setdefault(gram, category)

            for function in row.get("function") or []:
                if not isinstance(function, dict):
                    continue
                name = function.get("name")
                if isinstance(name, str) and name:
                    self._function_index.setdefault(name, category)

        self._screened.append(
            {
                "local_path": spec["local_path"],
                "sha256": spec["sha256"],
                "category": spec["category"],
                "role": spec["role"],
            }
        )
