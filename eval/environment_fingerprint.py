"""Environment provenance helpers used before a probe starts spending.

The locale collector deliberately asks the system ``locale`` command. The
consumer whose behaviour exposed this requirement is Bash, and CPython does not
apply the environment to every locale category at startup: in particular,
``locale.setlocale(locale.LC_COLLATE)`` can report ``C`` while Bash and
``locale`` resolve ``en_US.UTF-8`` from the same environment.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from collections.abc import Mapping


LOCALE_ENVIRONMENT_VARIABLES = ("LANG", "LC_ALL", "LC_COLLATE", "LC_CTYPE")
REQUIRED_RESOLVED_CATEGORIES = ("LANG", "LC_ALL", "LC_COLLATE", "LC_CTYPE")


class LocaleFingerprintError(RuntimeError):
    """The shell's effective locale could not be recorded unambiguously."""


def parse_locale_output(output: str) -> dict[str, str]:
    """Parse both quoted macOS and unquoted Linux ``locale`` output.

    Values are shell words. ``shlex`` removes syntactic quotes without leaving
    literal quote characters in the JSON receipt, while accepting the empty
    ``LC_ALL=`` form used when no global override is set.
    """
    resolved: dict[str, str] = {}
    for line_number, line in enumerate(output.splitlines(), start=1):
        if not line.strip():
            continue
        if "=" not in line:
            raise LocaleFingerprintError(
                f"locale output line {line_number} has no '=': {line!r}"
            )
        name, raw_value = line.split("=", 1)
        name = name.strip()
        if not name:
            raise LocaleFingerprintError(
                f"locale output line {line_number} has an empty category name"
            )
        try:
            words = shlex.split(raw_value, posix=True)
        except ValueError as exc:
            raise LocaleFingerprintError(
                f"locale output line {line_number} has malformed quoting: {line!r}"
            ) from exc
        if len(words) > 1:
            raise LocaleFingerprintError(
                f"locale output line {line_number} has multiple values: {line!r}"
            )
        resolved[name] = words[0] if words else ""

    missing = [name for name in REQUIRED_RESOLVED_CATEGORIES if name not in resolved]
    if missing:
        raise LocaleFingerprintError(
            "locale output omitted required categories: " + ", ".join(missing)
        )
    return resolved


def collect_locale_provenance(
    env: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Record locale inputs plus the shell-authoritative resolved categories.

    Collection failures are data, not defaults. Return ``unavailable`` with a
    reason and a null resolved value; the bootstrap writes that receipt before
    aborting preflight. In particular, never substitute ``C`` for a resolver
    that did not run, because that would report an unmeasured value as fact.
    """
    source_env: Mapping[str, str] = os.environ if env is None else env
    environment = {
        name: source_env.get(name) for name in LOCALE_ENVIRONMENT_VARIABLES
    }
    try:
        completed = subprocess.run(
            ["locale"],
            env=None if env is None else dict(env),
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        resolved = parse_locale_output(completed.stdout)
    except FileNotFoundError as exc:
        return {
            "status": "unavailable",
            "reason": f"locale command not found: {exc}",
            "environment": environment,
            "shell_resolved": None,
        }
    except subprocess.CalledProcessError as exc:
        diagnostic = (exc.stderr or exc.stdout or "").strip()
        suffix = f": {diagnostic}" if diagnostic else ""
        return {
            "status": "unavailable",
            "reason": f"locale command exited {exc.returncode}{suffix}",
            "environment": environment,
            "shell_resolved": None,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "unavailable",
            "reason": f"locale command timed out after {exc.timeout}s",
            "environment": environment,
            "shell_resolved": None,
        }
    except UnicodeError as exc:
        return {
            "status": "unavailable",
            "reason": f"locale output could not be decoded: {exc}",
            "environment": environment,
            "shell_resolved": None,
        }
    except LocaleFingerprintError as exc:
        return {
            "status": "unavailable",
            "reason": f"locale output could not be parsed: {exc}",
            "environment": environment,
            "shell_resolved": None,
        }
    return {
        "status": "ok",
        "reason": None,
        "environment": environment,
        "shell_resolved": resolved,
    }
