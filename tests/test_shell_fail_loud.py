"""The fail-loud invariants for the three scripts that run on a billing pod.

These are written as tests rather than left to review because the defect class
they police is silent by construction. A suppressed stderr, a `set +e` window,
or a step whose artifact is never asserted all produce a run that *looks* like
it worked: exit 0, a log full of green, and evidence that is missing or wrong
only when someone tries to use it weeks later.

Two concrete instances motivate the rules, so none of this is hypothetical:

  * `git clone ... 2>/dev/null` in a helper step hid a clone failure, recorded
    in docs/postmortem-s0-probe-20260808.md.
  * The retained mining-pilot artifact directory has no `env_fingerprint.json`,
    `pip_freeze.txt` or `gpu.txt`. Nothing errored at the time; the files simply
    were not there and nothing asserted that they should be.

    **Correction (this file previously said the pilot's libraries therefore
    "cannot be compared against the failed probe's today" — that is false.)**
    The versions were recorded in owner-pasted console output in the chat
    archive, and the comparison has been made: transformers, peft and
    accelerate all match (the last confirmed from the probe's retained
    `env_fingerprint.json`), GPU class matches, and **torch does not — 2.8.0 on
    the pilot against 2.9.1 on the probe.** Torch is the only library difference
    of the four. Equality is disproven, not unmeasurable. What the missing
    receipts actually cost is that the evidence lives in a chat log instead of
    alongside the run, which is what the assertions below prevent recurring.

`probe_liveness.sh` is policed here too, as of the owner's authorisation to
clean it up. Two defects were fixed: the greps in `scan_error_markers` treated
every non-zero exit as "no match", so a grep that *failed* reported a clean
`error_markers_seen: 0`; and two `set +e` windows wrapped `check_once`.

Its remaining stderr suppressions are deliberate and are a different class:
`kill -0`, `tmux has-session`, `command -v`, the `tail` guarded by `[[ -r ]]`,
and the BSD/GNU `stat` probe, where a non-zero exit IS the answer being asked
for and one of the two `stat` forms always fails by design. They are retained
under the rule that a predicate may stay if its semantics are tested — so they
are tested below, rather than merely asserted to be harmless.

No line numbers appear in this file. The previous version cited `set +e` at
396/406 when the actual lines were 393/403 (396/406 were the restoring
`set -e`), which is what citing positions instead of content gets you.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = REPO_ROOT / "scripts" / "bootstrap_pod.sh"
LAUNCHER = REPO_ROOT / "scripts" / "launch_probe.sh"
LIVENESS = REPO_ROOT / "scripts" / "probe_liveness.sh"
AUDITED = (BOOTSTRAP, LAUNCHER)
STRICT_MODE_SCRIPTS = (BOOTSTRAP, LAUNCHER, LIVENESS)

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


@pytest.mark.parametrize("script", STRICT_MODE_SCRIPTS, ids=lambda p: p.name)
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


def test_environment_fingerprint_records_locale_provenance() -> None:
    """Locale demonstrably changed monitor output, so it belongs in the receipt."""
    source = BOOTSTRAP.read_text(encoding="utf-8")
    assert "import json, locale, os, sys" in source
    assert 'for name in ("LANG", "LC_ALL", "LC_COLLATE", "LC_CTYPE")' in source
    assert "locale.setlocale(locale.LC_COLLATE)" in source
    assert "locale.setlocale(locale.LC_CTYPE)" in source


def test_launcher_checks_its_entry_points_and_says_what_it_checked() -> None:
    """Three modes, three honest messages.

    The version this replaces ran a plain `-f` test against whatever the working
    tree happened to be and then printed "all three entry points present at
    ${commit}" — in a dry run, a claim about a commit it had never read. A
    nonexistent SHA produced a confident OK.
    """
    source = LAUNCHER.read_text(encoding="utf-8")
    for entry in ("fetch_pinned_bfcl.py", "isolation_ladder.py", "bfcl_simple.py"):
        assert f'assert_entrypoint "eval/{entry}"' in source, entry
    # A commit that lacks a file is not a dirty tree; it gets its own class.
    assert "readonly EXIT_LAUNCH_INCOMPATIBLE=70" in source
    assert 'exit "${EXIT_LAUNCH_INCOMPATIBLE}"' in source
    # The summary reports the mode rather than asserting the commit blindly.
    assert 'echo "OK: all three entry points present in ${entrypoint_check_mode}."' in source


def _dry_run(commit: str, tmp_path: Path):
    now = int(time.time())
    return subprocess.run(
        ["bash", str(LAUNCHER), "--commit", commit,
         "--deadline-epoch", str(now + 1800),
         "--provider-deadline-epoch", str(now + 3600),
         "--out-root", str(tmp_path / "out"), "--dry-run"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=60,
    )


def test_dry_run_does_not_claim_a_sha_it_never_read(tmp_path: Path) -> None:
    """codex's reproduction: `--commit 000…000 --dry-run` used to print
    `OK: all three entry points present at 000…000`."""
    result = _dry_run("0" * 40, tmp_path)
    out = result.stdout + result.stderr
    assert "NOT VERIFIED" in out, out
    assert "entry points present at" not in out, out
    # git's own words are PRINTED, not merely captured. The previous version
    # captured stderr into a variable, threw it away, and mapped every non-zero
    # result onto "commit is not in this repository" — so a broken object
    # database read as an ordinary absence. The label also no longer asserts
    # which of the two it was, because `cat-file -e` returns 128 for both.
    assert "git said:" in out, out
    assert "Not a valid object name" in out, out
    assert "is not in this repository" not in out, "label overclaims the cause"
    assert result.returncode == 0, out   # a dry run on an absent SHA still prints its plan


def test_dry_run_verifies_a_real_sha_against_the_commit_tree(tmp_path: Path) -> None:
    head = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    result = _dry_run(head, tmp_path)
    out = result.stdout + result.stderr
    assert "read-only via git cat-file" in out, out
    assert head in out, out


def test_a_commit_missing_an_entry_point_is_launch_incompatible(tmp_path: Path) -> None:
    """Uses a real ancestor that genuinely predates eval/isolation_ladder.py, so
    the check is exercised against history rather than a mock — and git's own
    message ("exists on disk, but not in <commit>") proves the commit tree was
    read rather than the working directory."""
    ancestor = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-list", "--max-count=1", "HEAD",
         "--", "eval/bfcl_simple.py"],
        capture_output=True, text=True, check=True).stdout.strip()
    # Walk back to a commit that has bfcl_simple.py but not isolation_ladder.py.
    candidates = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "log", "--format=%H", "-n", "60"],
        capture_output=True, text=True, check=True).stdout.split()
    target = None
    for sha in candidates:
        has_ladder = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "cat-file", "-e", f"{sha}:eval/isolation_ladder.py"],
            capture_output=True).returncode == 0
        if not has_ladder:
            target = sha
            break
    if target is None:
        pytest.skip("no ancestor without eval/isolation_ladder.py in the last 60 commits")
    assert ancestor  # sanity: history is readable

    result = _dry_run(target, tmp_path)
    out = result.stdout + result.stderr
    assert result.returncode == 70, out
    assert "Launcher/commit incompatibility" in out, out
    assert "isolation_ladder.py" in out, out


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


# ---------------------------------------------------------------------------
# probe_liveness.sh — the retained predicates are tested, not assumed harmless,
# which is the condition under which they were allowed to stay.
# ---------------------------------------------------------------------------

import json          # noqa: E402
import os            # noqa: E402


def _run_liveness(tmp_path: Path, log_text: str, pid: str, extra=()) -> tuple[int, dict, str]:
    log = tmp_path / "probe.log"
    log.write_text(log_text, encoding="utf-8")
    status = tmp_path / "status.json"
    proc = subprocess.run(
        ["bash", str(LIVENESS), "--log", str(log), "--status-file", str(status),
         "--pid", pid, "--once", *extra],
        capture_output=True, text=True, timeout=30,
    )
    parsed = json.loads(status.read_text()) if status.exists() else {}
    return proc.returncode, parsed, proc.stdout + proc.stderr


def test_no_match_is_not_an_error(tmp_path: Path) -> None:
    """grep exits 1 when a marker is simply absent. That is the healthy case and
    the overwhelmingly common one; if the rewrite had turned it into a scan
    failure the monitor would cry wolf on every clean run."""
    _, status, _ = _run_liveness(tmp_path, "all fine here\nnothing wrong\n", "999999")
    assert status["error_marker_scan"] == "ok"
    assert status["error_markers_seen"] == 0


def test_markers_present_are_counted(tmp_path: Path) -> None:
    _, status, _ = _run_liveness(
        tmp_path, "boom\nCUDA error: an illegal memory access was encountered\n", "999999"
    )
    assert status["error_marker_scan"] == "ok"
    assert status["error_markers_seen"] >= 1
    assert "CUDA error" in status["error_marker_names"]


def test_an_unreadable_log_reports_null_not_zero(tmp_path: Path) -> None:
    """The defect this replaces: a scan that could not run reported
    `error_markers_seen: 0`, i.e. a clean bill of health from a check that never
    completed — the same shape as the Xid-regex defect fixed in 8659fb0."""
    log = tmp_path / "probe.log"
    log.write_text("CUDA error: something\n", encoding="utf-8")
    os.chmod(log, 0o000)
    try:
        if os.access(log, os.R_OK):        # root ignores the mode bits
            pytest.skip("running as root; unreadable-file case is not reachable")
        status_file = tmp_path / "status.json"
        subprocess.run(
            ["bash", str(LIVENESS), "--log", str(log), "--status-file", str(status_file),
             "--pid", "999999", "--once"],
            capture_output=True, text=True, timeout=30,
        )
        status = json.loads(status_file.read_text())
        assert status["error_markers_seen"] is None, status
        assert status["error_marker_scan"] == "unreadable", status
        assert status["error_marker_scan_detail"], "must say why"
    finally:
        os.chmod(log, 0o644)


def test_a_dead_pid_with_no_record_still_alerts(tmp_path: Path) -> None:
    """The verdict must not have been disturbed by the marker-scan rework: it
    comes from liveness plus the terminal record, never from marker counts."""
    code, status, out = _run_liveness(tmp_path, "no footer here\n", "999999")
    assert code == 72, out
    assert status["alert"] is True
    assert status["state"] == "died_hard"


def test_a_live_pid_reports_running(tmp_path: Path) -> None:
    code, status, out = _run_liveness(tmp_path, "working\n", str(os.getpid()))
    assert code == 0, out
    assert status["state"] == "running"
    assert status["alert"] is False


def test_a_sidecar_with_a_valid_digest_plus_garbage_is_rejected(tmp_path: Path) -> None:
    """`tr … | cut -c1-64` made the exact-shape assertion structurally unable to
    fail on the case it most needed to catch: `cut` threw away everything past
    character 64, so the regex was validating cut's output rather than the file.
    A receipt holding a real digest followed by a second digest, a filename, or
    a stray paste was silently truncated to the valid prefix and accepted.
    """
    bundle = tmp_path / "b.bundle"
    bundle.write_bytes(b"contents")
    digest = subprocess.run(["shasum", "-a", "256", str(bundle)],
                            capture_output=True, text=True, check=True).stdout.split()[0]

    def run(sidecar_text: str):
        side = tmp_path / "b.sha256"
        side.write_text(sidecar_text, encoding="utf-8")
        work = tmp_path / "work"
        work.mkdir(exist_ok=True)
        return subprocess.run(
            ["bash", str(BOOTSTRAP), "--bundle", str(bundle),
             "--bundle-sha256-file", str(side), "--commit", "0" * 40,
             "--out-root", str(tmp_path / "out"),
             "--auto-terminate-set", "2026-08-09T23:00:00Z@RATE"],
            cwd=str(work), capture_output=True, text=True, timeout=60,
            env={**os.environ, "RUNPOD_IMAGE_NAME": "test/image:1"},
        )

    # The exact digest and nothing else: accepted (the run goes on to fail later,
    # at the commit, which is a different and correctly-classified failure).
    ok = run(digest + "\n")
    assert "must contain exactly one 64-char" not in (ok.stdout + ok.stderr)

    for bad, label in [
        (digest + "GARBAGE", "trailing garbage"),
        (digest + digest, "two digests"),
        (f"{digest}  b.bundle", "digest + filename (sha256sum output format)"),
        (digest[:-1], "truncated"),
        ("", "empty"),
        (digest.upper(), "uppercase"),
        # Internal whitespace. `tr -d '[:space:]'` deleted it and reassembled a
        # valid digest out of pieces, so the bootstrap printed "bundle sha256
        # verified" and advanced to the clone. A digest that arrives in two
        # pieces is not a digest that arrived: something in the pipeline that
        # produced or transported it did what nobody intended.
        (f"{digest[:32]}\n{digest[32:]}", "split across two lines"),
        (f"{digest[:32]} {digest[32:]}", "split by a space"),
        (f"{digest[:20]}\t{digest[20:]}", "split by a tab"),
        # Outer whitespace, by contrast, is normal and must still be accepted;
        # covered by the leading/trailing case below.
    ]:
        result = run(bad)
        out = result.stdout + result.stderr
        assert result.returncode == 66, f"{label}: exit {result.returncode}\n{out}"
        assert "must contain exactly one 64-char" in out, f"{label}:\n{out}"


def test_a_sidecar_with_only_outer_whitespace_is_accepted(tmp_path: Path) -> None:
    """The split-digest fix must not overshoot: a trailing newline is what every
    normal tool writes, and leading indentation is harmless. Rejecting those
    would make the check correct and unusable, which is how strict checks get
    reverted."""
    bundle = tmp_path / "b.bundle"
    bundle.write_bytes(b"contents")
    digest = subprocess.run(["shasum", "-a", "256", str(bundle)],
                            capture_output=True, text=True, check=True).stdout.split()[0]
    side = tmp_path / "b.sha256"
    work = tmp_path / "work"
    work.mkdir()
    for text, label in [(digest + "\n", "trailing newline"),
                        ("  " + digest + "  \n", "leading and trailing spaces"),
                        (digest, "no trailing newline at all")]:
        side.write_text(text, encoding="utf-8")
        result = subprocess.run(
            ["bash", str(BOOTSTRAP), "--bundle", str(bundle),
             "--bundle-sha256-file", str(side), "--commit", "0" * 40,
             "--out-root", str(tmp_path / "out"),
             "--auto-terminate-set", "2026-08-09T23:00:00Z@RATE"],
            cwd=str(work), capture_output=True, text=True, timeout=60,
            env={**os.environ, "RUNPOD_IMAGE_NAME": "test/image:1"},
        )
        out = result.stdout + result.stderr
        assert "must contain exactly one 64-char" not in out, f"{label} was rejected:\n{out}"
        assert "bundle sha256 verified" in out, f"{label}:\n{out}"


# ---------------------------------------------------------------------------
# The status artifact must stay parseable exactly when things are going wrong.
# ---------------------------------------------------------------------------

HOSTILE_NAMES = [
    ("backslash", r"a\qb.log"),          # codex's reproduction: jq -> Invalid escape
    ("quote", 'q"uote.log'),
    ("double-backslash", r"back\\slash.log"),
    ("tab", "tab\ted.log"),
    ("newline", "line\nbreak.log"),
    ("vertical-tab", "vertical\vtab.log"),
    ("delete", "delete\x7fchar.log"),
    # These must remain literal after decoding. Escaping an emitted `\u00XX`
    # before an input backslash would silently turn path text into a control.
    ("literal-unicode-escape", r"literal\u000b.log"),
    ("literal-newline-escape", r"literal\n.log"),
    ("quote-backslash-mix", 'quote"\\mix.log'),
]


@pytest.mark.parametrize("label,name", HOSTILE_NAMES, ids=[n for n, _ in HOSTILE_NAMES])
def test_status_file_is_valid_json_for_hostile_paths(tmp_path: Path, label: str, name: str) -> None:
    """Every dynamic string was interpolated raw. A path with a backslash made
    the whole artifact unparseable while the monitor exited 72 believing it had
    written a clean record — a durable output that silently is not one."""
    status = tmp_path / "status.json"
    subprocess.run(
        ["bash", str(LIVENESS), "--log", str(tmp_path / name),
         "--status-file", str(status), "--pid", "999999", "--once"],
        capture_output=True, text=True, timeout=30,
    )
    assert status.exists(), f"{label}: no status file written"
    parsed = json.loads(status.read_text(encoding="utf-8"))   # raises if invalid
    assert parsed["log_file"].endswith(name), parsed["log_file"]
    assert parsed["error_markers_seen"] is None
    assert parsed["error_marker_scan"] == "unreadable"


def test_status_file_survives_a_hostile_scan_detail(tmp_path: Path) -> None:
    """The detail field carries text this script did not author — git/grep
    output, paths — which is precisely where quotes and backslashes come from."""
    weird = tmp_path / 'dir"with\\odd\tchars'
    weird.mkdir()
    status = tmp_path / "status.json"
    subprocess.run(
        ["bash", str(LIVENESS), "--log", str(weird / "absent.log"),
         "--status-file", str(status), "--pid", "999999", "--once"],
        capture_output=True, text=True, timeout=30,
    )
    parsed = json.loads(status.read_text(encoding="utf-8"))
    assert parsed["error_marker_scan_detail"], "detail must say why"
    assert '"' in parsed["error_marker_scan_detail"] or "\\" in parsed["error_marker_scan_detail"]


def _json_test_locales() -> list[str]:
    """Use a collating UTF-8 locale when the host provides one.

    The old range happened to work in C/C.UTF-8 and failed in en_US.UTF-8.
    Linux CI images do not always install an en_US locale, so the source-shape
    guard below remains mandatory even when only C is available there.
    """
    result = subprocess.run(["locale", "-a"], capture_output=True, text=True, check=True)
    available = set(result.stdout.splitlines())
    locales = ["C"]
    for candidate in ("en_US.UTF-8", "en_US.utf8"):
        if candidate in available:
            locales.append(candidate)
            break
    return locales


@pytest.mark.parametrize("locale_name", _json_test_locales())
@pytest.mark.parametrize("vector", ("log", "tmux"))
def test_every_c0_character_is_json_safe_in_every_test_locale(
    tmp_path: Path, locale_name: str, vector: str
) -> None:
    """Exercise the property, not only the character that exposed the defect.

    `[$'\\x01'-$'\\x1f']` is a locale-collated range in Bash. Under
    en_US.UTF-8 it omitted U+000B even though the same code passed under C, so
    a monitor could exit 72 while writing an unparseable terminal artifact.
    Test all non-NUL C0 characters through both documented input vectors.
    """
    env = {**os.environ, "LANG": locale_name, "LC_ALL": locale_name}
    for code in range(1, 32):
        control = chr(code)
        status = tmp_path / f"{vector}-{locale_name.replace('/', '_')}-{code:02x}.json"
        log = tmp_path / (f"probe{control}run.log" if vector == "log" else "missing.log")
        command = [
            "bash", str(LIVENESS), "--log", str(log),
            "--status-file", str(status), "--pid", "999999", "--once",
        ]
        if vector == "tmux":
            command += ["--tmux-session", f"probe{control}1"]

        result = subprocess.run(
            command, capture_output=True, text=True, timeout=30, env=env,
        )
        assert result.returncode == 72, result.stdout + result.stderr
        parsed = json.loads(status.read_text(encoding="utf-8"))
        if vector == "log":
            assert parsed["log_file"] == str(log)
        else:
            assert parsed["tmux_session"] == f"probe{control}1"


@pytest.mark.parametrize("locale_name", _json_test_locales())
def test_failed_grep_detail_with_vertical_tab_is_json_safe(
    tmp_path: Path, locale_name: str
) -> None:
    """Exercise the least-controlled string: a subprocess's stderr verbatim."""
    log = tmp_path / "probe.log"
    log.write_text("ordinary log text\n", encoding="utf-8")
    status = tmp_path / f"grep-detail-{locale_name}.json"
    fake_bin = tmp_path / f"bin-{locale_name}"
    fake_bin.mkdir()
    fake_grep = fake_bin / "grep"
    fake_grep.write_text(
        "#!/usr/bin/env bash\n"
        'for arg in "$@"; do\n'
        '  if [[ "$arg" == "${FAIL_LOG_FOR_TEST}" ]]; then\n'
        "    printf 'grep injected\\vdetail\\n' >&2\n"
        "    exit 2\n"
        "  fi\n"
        "done\n"
        f'exec "{shutil.which("grep")}" "$@"\n',
        encoding="utf-8",
    )
    fake_grep.chmod(0o755)
    env = {
        **os.environ,
        "LANG": locale_name,
        "LC_ALL": locale_name,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "FAIL_LOG_FOR_TEST": str(log),
    }

    result = subprocess.run(
        ["bash", str(LIVENESS), "--log", str(log),
         "--status-file", str(status), "--pid", "999999", "--once"],
        capture_output=True, text=True, timeout=30, env=env,
    )
    assert result.returncode == 72, result.stdout + result.stderr
    parsed = json.loads(status.read_text(encoding="utf-8"))
    assert parsed["error_marker_scan"] == "failed"
    assert "grep injected\vdetail" in parsed["error_marker_scan_detail"]


def test_json_escaping_does_not_depend_on_a_locale_collated_range() -> None:
    """Keep the known-bad mechanism out even on hosts lacking en_US.UTF-8."""
    source = LIVENESS.read_text(encoding="utf-8")
    executable = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    )
    control_range = re.compile(
        r"\[\$'\\x[0-9a-fA-F]{2}'-\$'\\x[0-9a-fA-F]{2}'\]"
    )
    assert not control_range.search(executable)


def test_a_git_failure_that_is_not_a_missing_object_is_reported_as_such(tmp_path: Path) -> None:
    """codex's control: a `git` that fails for any other reason.

    The previous version discarded git's stderr and reported every non-zero exit
    as "commit is not in this repository", so an unreadable object database
    presented as an ordinary absent SHA — a broken repo indistinguishable from a
    typo'd commit, with the one line that explained it thrown away.
    """
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_git = fake_bin / "git"
    # Fails only on cat-file, so the launcher still reaches the check normally.
    fake_git.write_text(
        "#!/usr/bin/env bash\n"
        'for a in "$@"; do\n'
        '  if [[ "$a" == "cat-file" ]]; then\n'
        '    echo "fatal: injected object database I/O failure" >&2\n'
        "    exit 128\n"
        "  fi\n"
        "done\n"
        f'exec {shutil.which("git")} "$@"\n',
        encoding="utf-8",
    )
    fake_git.chmod(0o755)

    now = int(time.time())
    result = subprocess.run(
        ["bash", str(LAUNCHER), "--commit", "0" * 40,
         "--deadline-epoch", str(now + 1800),
         "--provider-deadline-epoch", str(now + 3600),
         "--out-root", str(tmp_path / "out"), "--dry-run"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=60,
        env={**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}"},
    )
    out = result.stdout + result.stderr
    assert "injected object database I/O failure" in out, (
        "git's diagnostic was captured and then discarded:\n" + out
    )
    assert "NOT VERIFIED" in out, out
    assert "is not in this repository" not in out, (
        "a non-missing-object failure is still being labelled as an absent commit:\n" + out
    )
    assert "treat" in out and "suspect" in out, out
