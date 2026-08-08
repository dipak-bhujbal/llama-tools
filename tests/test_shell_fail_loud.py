"""The fail-loud invariants for the two scripts that run on a billing pod.

These are written as tests rather than left to review because the defect class
they police is silent by construction. A suppressed stderr, a `set +e` window,
or a step whose artifact is never asserted all produce a run that *looks* like
it worked: exit 0, a log full of green, and evidence that is missing or wrong
only when someone tries to use it weeks later.

Two concrete instances motivate each rule, so none of this is hypothetical:

  * `git clone ... 2>/dev/null` in a helper step hid a clone failure, recorded
    in docs/postmortem-s0-probe-20260808.md.
  * The retained mining-pilot artifact directory has no `env_fingerprint.json`,
    `pip_freeze.txt` or `gpu.txt`. Nothing errored at the time. The files just
    were not there, and nothing asserted that they should be — which is why the
    pilot's library versions cannot be compared against the failed probe's
    today.

Scope note: only `bootstrap_pod.sh` and `launch_probe.sh` are policed here,
which is the scope the owner named. `probe_liveness.sh` is excluded
deliberately, and the exclusion is stated in full rather than summarised,
because a scope note that undercounts what it is excusing is itself the kind of
reassuring-but-wrong artifact these tests exist to catch.

It retains **8 suppressions on 7 lines**, in three groups that are NOT equally
benign:

  * `kill -0` (140), `tmux has-session` (146), and the BSD/GNU `stat` probe
    (266, two of them) — a non-zero exit IS the answer being asked for, and one
    of the two `stat` forms always fails by design. Nothing is hidden.
  * `tail -n 200 ... 2>/dev/null || true` (180) — guarded by `[[ -r ]]`. A
    failure yields empty text, `footer_state` stays `absent`, and the monitor
    fails CLOSED to its DIED HARD alert. Safe direction.
  * The three `grep` calls in `scan_error_markers` (242, 248, 251) — **not
    benign.** A grep that fails leaves `marker_count` at 0, which is then
    reported as `"error markers: 0"` on the console and `error_markers_seen: 0`
    in the status JSON. That is the same shape as the Xid-regex defect fixed in
    8659fb0: a scan that did not run, reported as a clean result.

    Verified bound on the damage: `marker_count` is consumed only by
    `write_status` and the RUNNING console line. The verdict itself comes from
    `process_alive` plus `footer_state`, so a swallowed grep cannot turn an
    alert into silence — it can only under-report context in the reassuring
    direction.

`probe_liveness.sh` also still opens two `set +e` windows around `check_once`
(396, 406), the same pattern removed from the launcher here.

None of that is fixed in this commit because the owner scoped the sweep to the
bootstrap and the launcher. It is written down so the exclusion cannot be read
as a claim that the third script is clean.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = REPO_ROOT / "scripts" / "bootstrap_pod.sh"
LAUNCHER = REPO_ROOT / "scripts" / "launch_probe.sh"
AUDITED = (BOOTSTRAP, LAUNCHER)

# Discarding stderr. `2>&1` is deliberately NOT matched: it *merges* stderr into
# stdout, which is the opposite of suppression, and the bootstrap's own launch
# hint uses it to route the launcher's stderr into the durable tee'd log.
_STDERR_DISCARD_RE = re.compile(r"2>\s*(?:/dev/null|&-)")

# Turning off any leg of strict mode after the header has set it.
_STRICT_MODE_OFF_RE = re.compile(r"set\s+\+(?:[eu]|o\s+pipefail)")


def executable_lines(path: Path) -> list[tuple[int, str]]:
    """Source lines with whole-line comments removed.

    Only lines whose first non-whitespace character is `#` are dropped. Trailing
    comments are kept: stripping them would need a shell-aware parser, and a
    false negative here (a suppression hiding behind a `#`) is the failure this
    file exists to prevent.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    return [
        (n, line)
        for n, line in enumerate(lines, start=1)
        if line.lstrip() and not line.lstrip().startswith("#")
    ]


@pytest.mark.parametrize("script", AUDITED, ids=lambda p: p.name)
def test_no_executable_line_discards_stderr(script: Path) -> None:
    """Every diagnostic reaches the operator, or the pattern that hides it is
    gone. The prose above these lines may quote the removed pattern — that is
    the audit trail, and whole-line comments are excluded for exactly that."""
    hits = [(n, line.strip()) for n, line in executable_lines(script)
            if _STDERR_DISCARD_RE.search(line)]
    assert not hits, f"{script.name}: stderr discarded at {hits}"


@pytest.mark.parametrize("script", AUDITED, ids=lambda p: p.name)
def test_strict_mode_is_set_and_never_turned_off(script: Path) -> None:
    """`set -euo pipefail` on line 1 is worth nothing if it is switched off
    around the risky call. Both scripts used to open a `set +e` window to
    capture an exit status; `cmd || status=$?` captures the same status with
    errexit in force for the whole file."""
    source = script.read_text(encoding="utf-8")
    assert "set -euo pipefail" in source, f"{script.name}: strict mode never set"
    hits = [(n, line.strip()) for n, line in executable_lines(script)
            if _STRICT_MODE_OFF_RE.search(line)]
    assert not hits, f"{script.name}: strict mode disabled at {hits}"


def test_bootstrap_asserts_the_checked_out_sha_rather_than_assuming_the_clone() -> None:
    """A clone that exits 0 is not evidence that HEAD is the reviewed commit.

    Four separate things are asserted, and each replaces a way the previous
    version inferred success from an exit code: the clone produced a `.git`,
    the bundle actually contains the commit, `rev-parse` itself succeeded and
    returned a 40-char SHA, and that SHA equals the one passed in.
    """
    source = BOOTSTRAP.read_text(encoding="utf-8")
    assert "[[ -d llama-tools/.git ]]" in source
    assert 'cat-file -e "${commit}^{commit}"' in source
    assert '[[ "${head_sha}" =~ ^[0-9a-f]{40}$ ]]' in source
    assert '[[ "${head_sha}" == "${commit}" ]]' in source


def test_launcher_classifies_a_failed_head_assertion_separately() -> None:
    """A broken assertion and a caught violation demand opposite responses — a
    sick pod versus a wrong tree — so they must not present identically. Both
    git reads were bare command substitutions, which under `set -e` aborted with
    git's exit code and no message."""
    source = LAUNCHER.read_text(encoding="utf-8")
    assert 'actual_head="$("${head_cmd[@]}")" || head_status=$?' in source
    assert 'dirty="$("${status_cmd[@]}")" || dirty_status=$?' in source
    assert "git rev-parse HEAD failed" in source
    assert "git status --porcelain failed" in source


def test_nvidia_smi_failure_is_fatal_and_never_written_as_an_artifact() -> None:
    """The removed line discarded nvidia-smi's stderr and then wrote a file
    asserting a reason it had just thrown away, and the run continued to paid
    inference. It is also self-contradictory: STEP 6 has already asserted
    torch.cuda.is_available() on the same host, so a failing nvidia-smi is an
    anomaly the operator must see BEFORE spending."""
    source = BOOTSTRAP.read_text(encoding="utf-8")
    executable = "\n".join(line for _, line in executable_lines(BOOTSTRAP))
    assert "nvidia-smi unavailable" not in executable, "fail-open artifact is back"
    assert 'gpu.stderr.txt' in source, "nvidia-smi stderr must be preserved as evidence"
    assert "nvidia-smi failed (exit ${nvidia_smi_status})" in source


def test_bootstrap_asserts_every_environment_receipt_exists_and_is_non_empty() -> None:
    """`set -e` proves each write command returned zero. It does not prove a
    file arrived. The retained mining-pilot directory is the counterexample:
    no fingerprint, no pip freeze, no gpu.txt, and nothing failed at the time.
    """
    source = BOOTSTRAP.read_text(encoding="utf-8")
    for receipt in (
        "pip_freeze.txt",
        "gpu.txt",
        "image_tag.txt",
        "auto_terminate_attestation.txt",
        "reviewed_commit.txt",
        "env_fingerprint.json",
        "bundle_sha256.txt",
    ):
        assert f'assert_file "${{out_root}}/{receipt}"' in source, receipt
    # assert_file checks both existence and non-emptiness; a zero-byte
    # fingerprint is exactly as useless as an absent one.
    assert '[[ -f "${path}" ]] || die' in source
    assert '[[ -s "${path}" ]] || die' in source


def test_launcher_asserts_its_entry_points_exist_in_the_reviewed_tree() -> None:
    """Bash's own "No such file or directory" funnelled through run_checked and
    came out labelled "acquire pinned BFCL fixtures failed" — which names the
    wrong thing and sends the operator to debug the fetcher instead of the venv.
    """
    source = LAUNCHER.read_text(encoding="utf-8")
    for entry in ("fetch_pinned_bfcl.py", "isolation_ladder.py", "bfcl_simple.py"):
        assert f'assert_entrypoint "${{REPO_ROOT}}/eval/{entry}"' in source, entry
    # A tree at the pinned SHA that is missing a file it should contain is a
    # wrong-tree problem, not a wrong-invocation one, and the exit class has to
    # say so — the codes exist precisely so the log names the kind of fault.
    assert 'exit "${EXIT_GIT_UNCLEAN}"\n  fi\n}' in source
    # Interpreter checked before the checkout, entry points after it: the
    # venv is gitignored so the checkout cannot affect it, while the entry
    # points are properties of the reviewed tree and must be asked about there.
    assert source.index("no executable interpreter at") < source.index("step_git_checkout\n")
    assert source.index("step_git_checkout\n") < source.index("assert_entrypoint \"${REPO_ROOT}")


def test_bootstrap_has_no_unclassified_command_runner() -> None:
    """Every side-effecting command names its failure class.

    The old `run()` helper let `set -e` abort with the tool's own exit code —
    128 from git, 1 from mkdir — which tells the operator what complained, not
    which guarantee broke. It is deleted rather than left unused: an
    unclassified helper in the file is what the next edit reaches for.
    """
    source = BOOTSTRAP.read_text(encoding="utf-8")
    executable = "\n".join(line for _, line in executable_lines(BOOTSTRAP))
    assert "run_classified()" in source
    assert re.search(r"^run\(\)", executable, re.M) is None, "unclassified run() is back"
    # And nothing calls it.
    assert re.search(r"^\s*run\s+[a-z]", executable, re.M) is None


def test_bootstrap_die_does_not_append_its_exit_code_to_the_message() -> None:
    """`echo "ERROR: $*"` expanded both arguments, so every classified failure
    printed its own exit code as part of the sentence — a diagnostic that lies
    slightly, in the one place an operator reads under time pressure."""
    source = BOOTSTRAP.read_text(encoding="utf-8")
    assert 'echo "ERROR: $*" >&2' not in source
    assert 'echo "ERROR: ${message}" >&2' in source
