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

# The previous patterns keyed on `--token $VAR` with the `$` immediately after
# the separator, which zero of the realistic leak forms actually match:
# `--token "$HF_TOKEN"` (quoted), `--token="${HF_TOKEN}"` (quoted + braced), and
# `--token hf_abc123` (a literal, no variable at all) all slipped through. The
# ban is now on the FLAG, regardless of quoting or value — there is no safe way
# to pass a credential as an argument, so no value needs inspecting.
CRED = r"(?:HF_TOKEN|HUGGING_FACE_HUB_TOKEN|HF_HUB_TOKEN|WANDB_API_KEY)"

BANNED = (
    # Any credential-passing flag on a login/auth/download command, whatever
    # follows it. Matching the flag rather than its value is what makes quoting,
    # braces and literals all fail closed.
    re.compile(r"\b(?:hf|huggingface-cli|wandb)\b[^|;&]*?\s--token(?:[= ]|$)"),
    # `wandb login <anything>` / `hf auth login <anything>`: an argument to a
    # login verb is a credential by construction.
    re.compile(r"\bwandb\s+login\s+\S"),
    re.compile(r"\b(?:hf|huggingface-cli)\s+auth\s+login\s+(?!--force\b|--help\b|-h\b)\S"),
    # A literal assignment in a shell command: `export FOO=...`, or a bare
    # `FOO=... cmd` prefix. Both reach shell history. A bare `FOO=value` line on
    # its own is handled separately — see _is_shell_region — because that is
    # also what a .env file looks like, and a .env file is not a command line.
    re.compile(rf"\bexport\s+{CRED}\s*="),
)

# A bare `CRED=value` line is only a defect in a shell context. In a plain fence
# it is .env file content, which is the recommended place for a secret and does
# not touch history, argv or the screen. Policed separately rather than lumped
# in, so the distinction is a decision on the record instead of an accident.
BARE_ASSIGNMENT = re.compile(rf"^\s*(?:export\s+)?{CRED}\s*=\s*\S")

SHELL_FENCE_INFO = ("bash", "sh", "zsh", "shell", "console", "shell-session")

# A line may quote the pattern if it is arguing against it.
NEGATIVE_CUES = ("never", "not ", "n't", "originally", "would", "instead of",
                 "rather than", "do not", "avoid", "wrong")

# `^```` (column 0 only) matched ZERO fences in week-2-sft-fundamentals.md and
# 1 of 4 in week-1-fundamentals.md, because every fence in those files is
# indented inside a `- [ ]` list item. The test that scanned "every pasteable
# command" was therefore scanning nothing at all in exactly the files whose
# token blocks had just been rewritten. Leading whitespace is now allowed, and
# the info string is captured so shell fences can be told from .env fences.
_FENCE_RE = re.compile(r"^[ \t]*```([^\n]*)\n(.*?)^[ \t]*```", re.M | re.S)


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


def executable_regions(path: Path) -> list[tuple[str, str]]:
    """(info_string, body) pairs of everything in `path` that is meant to be run."""
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".md":
        return [(info.strip().lower(), body) for info, body in _FENCE_RE.findall(text)]
    if path.suffix == ".sh":
        return [("bash", text)]
    return []


def test_fence_matching_sees_the_indented_fences(subtests=None) -> None:
    """Guard the guard, and specifically the way it was broken.

    Every fence in the two learning journals is indented inside a `- [ ]` list
    item. A column-0-anchored fence regex found none of them, so the credential
    scan below was scanning zero bytes of the files it most needed to read.
    """
    for rel, minimum in (
        ("docs/learning/week-2-sft-fundamentals.md", 3),
        ("docs/learning/week-1-fundamentals.md", 3),
    ):
        regions = executable_regions(REPO_ROOT / rel)
        assert len(regions) >= minimum, f"{rel}: only {len(regions)} fences matched"
    # And at least one of them is a shell fence, or the shell-only rules below
    # never fire.
    infos = [i for i, _ in executable_regions(
        REPO_ROOT / "docs/learning/week-2-sft-fundamentals.md")]
    assert any(i in SHELL_FENCE_INFO for i in infos), infos


def scan_for_violations(text_regions, rel: str) -> list[str]:
    violations: list[str] = []
    for info, body in text_regions:
        for line in body.splitlines():
            # Shell comments are commentary, not commands — this is where the
            # runbooks' warnings about the pattern legitimately live.
            if line.lstrip().startswith("#"):
                continue
            for hit in banned_hits(line):
                violations.append(f"{rel}: {line.strip()!r} matched {hit}")
            # A bare `HF_TOKEN=value` is a command only in a shell fence. In an
            # info-less fence it is .env content, which is a file the reader is
            # *supposed* to put the secret in.
            if info in SHELL_FENCE_INFO and BARE_ASSIGNMENT.search(line):
                violations.append(f"{rel}: {line.strip()!r} bare credential assignment")
    return violations


def test_no_pasteable_command_puts_a_credential_in_argv() -> None:
    """Markdown fences and shell scripts are meant to be executed verbatim."""
    violations: list[str] = []
    for path in searched_files():
        rel = path.relative_to(REPO_ROOT).as_posix()
        violations += scan_for_violations(executable_regions(path), rel)
    assert not violations, "credential in a pasteable command:\n" + "\n".join(violations)


# --- Negative controls -------------------------------------------------------
# Every one of these produced ZERO matches under the previous patterns. They are
# committed as explicit cases rather than trusted to the regex reading well,
# because "the regex looks right" is exactly what was believed before.
LEAK_FORMS = [
    'hf auth login --token $HF_TOKEN',
    'hf auth login --token "$HF_TOKEN"',
    'hf auth login --token="${HF_TOKEN}"',
    'hf auth login --token hf_abcdef0123456789',
    'huggingface-cli login --token "${HF_TOKEN}"',
    'hf download some/repo --token $HF_TOKEN',
    'wandb login $WANDB_API_KEY',
    'wandb login 0123456789abcdef',
    'export HF_TOKEN=hf_abcdef0123456789',
    'export WANDB_API_KEY="secret"',
]

SAFE_FORMS = [
    "read -rsp 'HF token: ' HF_TOKEN && echo && export HF_TOKEN",
    "export HF_TOKEN",
    "hf auth login",
    "hf auth login --force",
    "wandb login",
    "hf download some/repo --local-dir /tmp/x",
    "python -c 'import os; os.environ[\"HF_TOKEN\"]'",
]


@pytest.mark.parametrize("line", LEAK_FORMS)
def test_every_known_leak_form_is_caught(line: str) -> None:
    caught = bool(banned_hits(line)) or bool(BARE_ASSIGNMENT.search(line))
    assert caught, f"leak form not caught: {line!r}"


@pytest.mark.parametrize("line", SAFE_FORMS)
def test_no_safe_form_is_flagged(line: str) -> None:
    """A rule that fires on the recommended pattern gets disabled by whoever
    hits it next, which is worse than not having it."""
    assert not banned_hits(line), f"false positive on the safe form: {line!r}"


def test_a_leak_inside_an_indented_shell_fence_is_caught() -> None:
    """The two failures compounded: the leak forms did not match, and the
    fences they would have been found in were not being read either."""
    doc = (
        "- [ ] **Set up the pod:**\n"
        "  ```bash\n"
        "  pip install -e .\n"
        '  hf auth login --token "$HF_TOKEN"\n'
        "  ```\n"
    )
    regions = [(i.strip().lower(), b) for i, b in _FENCE_RE.findall(doc)]
    assert regions, "indented fence still not matched"
    assert scan_for_violations(regions, "synthetic.md")


def test_dotenv_content_is_deliberately_not_flagged() -> None:
    """A `.env` file is where a secret is *supposed* to go: it reaches neither
    argv, nor shell history, nor the screen. Recorded as a decision so the
    exemption is not mistaken for a gap in the rules."""
    doc = "- [ ] **Save the key:**\n  ```\n  WANDB_API_KEY=<your-key>\n  ```\n"
    regions = [(i.strip().lower(), b) for i, b in _FENCE_RE.findall(doc)]
    assert regions
    assert not scan_for_violations(regions, "synthetic.md")


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
