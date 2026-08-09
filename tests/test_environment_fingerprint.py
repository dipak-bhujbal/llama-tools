"""The locale receipt must describe Bash's resolver, not CPython's defaults."""

from __future__ import annotations

import os
import subprocess

import pytest

import eval.environment_fingerprint as fingerprint_module
from eval.environment_fingerprint import (
    LocaleFingerprintError,
    collect_locale_provenance,
    parse_locale_output,
)


def _available_locales() -> set[str]:
    result = subprocess.run(
        ["locale", "-a"], capture_output=True, text=True, check=True
    )
    return set(result.stdout.splitlines())


def _first_available(*candidates: str) -> str | None:
    available = _available_locales()
    return next((name for name in candidates if name in available), None)


def _locale_env(lang: str, *, lc_all: str | None) -> dict[str, str]:
    env = dict(os.environ)
    for name in ("LANG", "LC_ALL", "LC_COLLATE", "LC_CTYPE"):
        env.pop(name, None)
    env["LANG"] = lang
    if lc_all is not None:
        env["LC_ALL"] = lc_all
    return env


def _bash_vertical_tab_matches_control_range(env: dict[str, str]) -> bool:
    result = subprocess.run(
        ["bash", "-c", "[[ $'\\v' == [$'\\x01'-$'\\x1f'] ]]"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode in (0, 1), result.stderr
    return result.returncode == 0


def test_parser_accepts_quoted_and_unquoted_locale_output() -> None:
    parsed = parse_locale_output(
        'LANG="en_US.UTF-8"\n'
        "LC_CTYPE=en_US.UTF-8\n"
        'LC_COLLATE="en_US.UTF-8"\n'
        "LC_ALL=\n"
        'EXTRA_CATEGORY="value"\n'
    )
    assert parsed == {
        "LANG": "en_US.UTF-8",
        "LC_CTYPE": "en_US.UTF-8",
        "LC_COLLATE": "en_US.UTF-8",
        "LC_ALL": "",
        "EXTRA_CATEGORY": "value",
    }


@pytest.mark.parametrize(
    "bad",
    [
        "LANG=C\nLC_ALL=C\nLC_CTYPE=C\n",  # missing LC_COLLATE
        "LANG=C\nLC_ALL=C\nLC_CTYPE=C\nLC_COLLATE",  # malformed line
        'LANG=C\nLC_ALL=C\nLC_CTYPE=C\nLC_COLLATE="unterminated\n',
    ],
)
def test_parser_fails_loudly_on_ambiguous_locale_output(bad: str) -> None:
    with pytest.raises(LocaleFingerprintError):
        parse_locale_output(bad)


def test_shell_resolved_c_locale_matches_environment() -> None:
    env = _locale_env("C", lc_all="C")
    provenance = collect_locale_provenance(env)
    assert provenance["status"] == "ok"
    assert provenance["reason"] is None
    assert provenance["environment"] == {
        "LANG": "C",
        "LC_ALL": "C",
        "LC_COLLATE": None,
        "LC_CTYPE": None,
    }
    assert provenance["shell_resolved"]["LC_COLLATE"] == "C"
    assert _bash_vertical_tab_matches_control_range(env)


def test_lang_only_collating_utf8_locale_is_recorded_exactly() -> None:
    locale_name = _first_available(
        "en_US.UTF-8", "en_GB.UTF-8", "de_DE.UTF-8", "ja_JP.UTF-8"
    )
    if locale_name is None:
        pytest.skip("no collating UTF-8 locale installed")
    env = _locale_env(locale_name, lc_all=None)
    provenance = collect_locale_provenance(env)
    assert provenance["status"] == "ok"
    assert provenance["environment"]["LANG"] == locale_name
    assert provenance["environment"]["LC_ALL"] is None
    assert provenance["shell_resolved"]["LC_COLLATE"] == locale_name
    assert not _bash_vertical_tab_matches_control_range(env)


def test_c_utf8_is_recorded_without_inferring_non_c_collation() -> None:
    locale_name = _first_available("C.UTF-8", "C.utf8")
    if locale_name is None:
        pytest.skip("no C.UTF-8 locale installed")
    env = _locale_env(locale_name, lc_all=None)
    provenance = collect_locale_provenance(env)
    assert provenance["status"] == "ok"
    assert provenance["shell_resolved"]["LC_COLLATE"] == locale_name
    assert _bash_vertical_tab_matches_control_range(env)


def test_missing_locale_command_is_recorded_as_unavailable(monkeypatch) -> None:
    def missing(*args, **kwargs):
        raise FileNotFoundError("locale")

    monkeypatch.setattr(fingerprint_module.subprocess, "run", missing)
    provenance = collect_locale_provenance(_locale_env("C", lc_all="C"))
    assert provenance["status"] == "unavailable"
    assert "not found" in provenance["reason"]
    assert provenance["shell_resolved"] is None


def test_nonzero_locale_command_is_recorded_as_unavailable(monkeypatch) -> None:
    def failed(*args, **kwargs):
        raise subprocess.CalledProcessError(
            7, ["locale"], stderr="injected locale failure"
        )

    monkeypatch.setattr(fingerprint_module.subprocess, "run", failed)
    provenance = collect_locale_provenance(_locale_env("C", lc_all="C"))
    assert provenance["status"] == "unavailable"
    assert "exited 7: injected locale failure" in provenance["reason"]
    assert provenance["shell_resolved"] is None


def test_malformed_locale_output_is_recorded_as_unavailable(monkeypatch) -> None:
    def malformed(*args, **kwargs):
        return subprocess.CompletedProcess(
            ["locale"], 0, stdout="LANG=C\nLC_ALL=C\n", stderr=""
        )

    monkeypatch.setattr(fingerprint_module.subprocess, "run", malformed)
    provenance = collect_locale_provenance(_locale_env("C", lc_all="C"))
    assert provenance["status"] == "unavailable"
    assert "could not be parsed" in provenance["reason"]
    assert provenance["shell_resolved"] is None
