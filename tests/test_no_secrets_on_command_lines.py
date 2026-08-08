"""No credential is ever passed as a command-line argument, anywhere in the repo.

The defect has two forms and both had shipped here:

  * `export HF_TOKEN=<literal>` — the value lands in `~/.zsh_history` in
    cleartext and stays there. `docs/learning/week-1-fundamentals.md` went
    further and told the reader to persist it in `~/.zshrc`, a long-lived
    secret at rest in a file that gets backed up, synced and screen-shared.
  * `hf auth login --token $HF_TOKEN` — the shell expands the variable
    *before* exec, so the value sits in the new process's argv, readable by any
    `ps` on the box for the life of the command. This one is easy to miss
    precisely because the source line contains no secret; the leak is created
    by expansion, not by the text.

`wandb login $WANDB_API_KEY` is the same defect with a different credential and
is policed identically — the pattern is what matters, not which vendor's token
happens to be in the variable.

Two rules, because pasteable commands and prose need different treatment:

  1. Inside a markdown fence or a shell script, the pattern is banned outright.
     That text is meant to be run. A warning about the pattern belongs in a
     comment or in prose, and shell comments are excluded here so the existing
     runbook warnings stay legal.
  2. Everywhere else the pattern may appear only on a line that argues against
     it. Documentation has to be able to name what not to do, and the two
     learning journals now carry dated corrections that quote the original
     wording rather than pretending it was never there.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

SEARCHED_SUFFIXES = {".md", ".py", ".sh", ".txt"}
SKIPPED_DIRS = {".git", ".venv", "node_modules", "__pycache__", "eval/bfcl_data"}

# The postmortem is the incident record for this exact defect. Rewriting its
# quoted evidence to satisfy a lint would destroy the thing it exists to hold.
# This file is exempt for the same reason at one remove: it is where the banned
# pattern is *defined*, so it necessarily spells it out.
EXEMPT_FILES = {
    "docs/postmortem-s0-probe-20260808.md",
    "tests/test_no_secrets_on_command_lines.py",
}

BANNED = (
    # A credential-bearing variable expanded into argv.
    re.compile(r"--token[= ]\$\{?[A-Z_]*TOKEN"),
    re.compile(r"\b(?:wandb|hf|huggingface-cli)\s+login\s+\$\{?[A-Z_]+"),
    # A literal assignment, which is what reaches shell history.
    re.compile(r"\bexport\s+(?:HF_TOKEN|WANDB_API_KEY|HUGGING_FACE_HUB_TOKEN)\s*="),
)

# A line may quote the pattern if it is arguing against it.
NEGATIVE_CUES = ("never", "not ", "n't", "originally", "would", "instead of",
                 "rather than", "do not", "avoid", "wrong")

_FENCE_RE = re.compile(r"^```.*?^```", re.M | re.S)


def searched_files() -> list[Path]:
    out = []
    for path in REPO_ROOT.rglob("*"):
        if path.suffix not in SEARCHED_SUFFIXES or not path.is_file():
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        if any(part in SKIPPED_DIRS for part in rel.split("/")):
            continue
        if rel in EXEMPT_FILES:
            continue
        out.append(path)
    return sorted(out)


def banned_hits(line: str) -> list[str]:
    return [p.pattern for p in BANNED if p.search(line)]


def test_the_scan_actually_covers_the_files_that_carry_the_pattern() -> None:
    """Guard the guard. If the walk stops finding these, every test below
    passes vacuously and the invariant is silently unenforced."""
    names = {p.relative_to(REPO_ROOT).as_posix() for p in searched_files()}
    for expected in (
        "docs/probe-bootstrap.md",
        "docs/mining-runbook.md",
        "docs/learning/week-1-fundamentals.md",
        "docs/learning/week-2-sft-fundamentals.md",
        "train/README.md",
        "smoke.py",
    ):
        assert expected in names, expected


def test_no_pasteable_command_puts_a_credential_in_argv() -> None:
    """Markdown fences and shell scripts are meant to be executed verbatim."""
    violations: list[str] = []
    for path in searched_files():
        rel = path.relative_to(REPO_ROOT).as_posix()
        text = path.read_text(encoding="utf-8")

        if path.suffix == ".md":
            regions = _FENCE_RE.findall(text)
        elif path.suffix == ".sh":
            regions = [text]
        else:
            continue

        for region in regions:
            for line in region.splitlines():
                # Shell comments inside a fence are commentary, not commands —
                # this is where the runbooks' warnings about the pattern live.
                if line.lstrip().startswith("#"):
                    continue
                for hit in banned_hits(line):
                    violations.append(f"{rel}: {line.strip()!r} matched {hit}")
    assert not violations, "credential in a pasteable command:\n" + "\n".join(violations)


def test_any_prose_mention_of_the_pattern_argues_against_it() -> None:
    """Docs must be able to name the defect. They may not recommend it.

    The cue is looked for across a small window rather than on the matched line
    alone, because a warning long enough to be useful wraps: the sentence that
    quotes `--token $HF_TOKEN` and the clause that says why it is wrong are
    routinely two different lines.
    """
    window = 2
    violations: list[str] = []
    for path in searched_files():
        rel = path.relative_to(REPO_ROOT).as_posix()
        lines = path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            if not banned_hits(line):
                continue
            context = " ".join(lines[max(0, i - window):i + window + 1]).lower()
            if any(cue in context for cue in NEGATIVE_CUES):
                continue
            violations.append(f"{rel}:{i + 1}: {line.strip()!r}")
    assert not violations, (
        "the credential-in-argv pattern is presented as the thing to do:\n"
        + "\n".join(violations)
    )


@pytest.mark.parametrize(
    "path",
    ["docs/probe-bootstrap.md", "docs/mining-runbook.md", "train/README.md",
     "docs/learning/week-1-fundamentals.md", "docs/learning/week-2-sft-fundamentals.md"],
)
def test_each_setup_doc_teaches_the_safe_form(path: str) -> None:
    """Removing the bad pattern is only half of it. A doc that deletes the
    instruction without replacing it leaves the reader to invent their own,
    and what they invent is `export HF_TOKEN=...`."""
    text = (REPO_ROOT / path).read_text(encoding="utf-8")
    assert "read -rs" in text, f"{path}: no prompt-based token entry shown"
    assert "HF_TOKEN" in text
