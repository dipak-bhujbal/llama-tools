"""GPU / library telemetry captured at every step of the §0 isolation ladder.

This module exists because of a specific failure. The 2026-08-08 §0 probe died
with a CUDA illegal memory access at 64s having produced zero generations, and
the cause is *indeterminate* — not because the crash was exotic, but because
nothing recorded what the machine looked like when it happened. There is no
record of how the model was placed, which attention kernel was selected, how
much VRAM was in use, which physical card it was, or whether that card had ECC
errors before we ever touched it.

One rule governs every field here: **a field that could not be measured is
reported as unavailable with a reason, never as zero and never as absent.**

That is not pedantry. "ECC uncorrectable: 0" and "ECC uncorrectable: could not
read" are opposite findings — the first exonerates the card, the second says
nothing at all. This project's recurring defect is a check that looks complete
but isn't (hashes recorded but never compared, counts embedded but never
enforced), and a telemetry bundle that silently degrades to zeros is that same
defect wearing a lab coat. Every value is therefore wrapped:

    {"status": "ok", "value": <measurement>}
    {"status": "unavailable", "value": None, "reason": "<why>"}

`torch` is imported lazily inside the collectors, never at module scope, so the
pure parsers below can be tested on a laptop with no CUDA and no torch — which
is where they were in fact developed and tested, at zero cost.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from typing import Any, Callable, Iterable

# `nvidia-smi --query-gpu=` fields. These come back as stable CSV, unlike the
# indented tree of `nvidia-smi -q`, so identity and memory are read from here
# and the free-text dump is retained separately as raw evidence.
QUERY_GPU_FIELDS = (
    "name",
    "uuid",
    "driver_version",
    "memory.total",
    "memory.free",
    "memory.used",
    "ecc.mode.current",
    "ecc.errors.corrected.volatile.total",
    "ecc.errors.uncorrected.volatile.total",
    "ecc.errors.corrected.aggregate.total",
    "ecc.errors.uncorrected.aggregate.total",
    "retired_pages.pending",
)

# Xid errors are kernel-level GPU fault codes. They are the single most useful
# signal for distinguishing "our code did something illegal" from "this card is
# sick", and they do NOT appear in `nvidia-smi -q` — they are emitted by the
# NVIDIA kernel driver into the kernel ring buffer. Reading them therefore means
# reading the kernel log, which inside an unprivileged container frequently
# fails. When it fails we say so.
#
# The PCI location is consumed as a bracketed group rather than with `.*?`,
# because the location itself contains colons — `Xid (PCI:0000:01:00): 13` — and
# a lazy `.*?:` stops at the colon inside `PCI:`, capturing `0`. That bug was
# live in the first draft of this file and reported "0 Xid errors" for a log
# full of them: a fabricated clean bill of health, which is the one output this
# module must never produce.
XID_LINE = re.compile(r"NVRM:\s*Xid\s*(?:\([^)]*\))?\s*:?\s*(\d+)", re.IGNORECASE)

_SENTINEL_VALUES = {"[not supported]", "[n/a]", "n/a", "not supported", ""}


def ok(value: Any) -> dict:
    return {"status": "ok", "value": value}


def unavailable(reason: str) -> dict:
    """A field we could not measure. `value` is None so a caller that ignores
    `status` gets an obvious None rather than a plausible zero."""
    return {"status": "unavailable", "value": None, "reason": reason}


# ---------------------------------------------------------------------------
# Pure parsers. No subprocess, no torch — these are the testable core.
# ---------------------------------------------------------------------------
def parse_query_gpu_csv(text: str, fields: Iterable[str] = QUERY_GPU_FIELDS) -> list[dict]:
    """Parse `nvidia-smi --query-gpu=<fields> --format=csv,noheader` output.

    nvidia-smi reports fields the hardware does not implement as the literal
    string `[N/A]` or `[Not Supported]`. A consumer GeForce card has no ECC at
    all, so its ECC columns come back that way — and mapping those to 0 would
    manufacture the exact reassurance we must not manufacture. They become
    `unavailable` instead.
    """
    field_list = list(fields)
    rows: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        cells = [c.strip() for c in line.split(",")]
        if len(cells) != len(field_list):
            rows.append(
                {
                    "_parse_error": unavailable(
                        f"expected {len(field_list)} columns, got {len(cells)}: {line!r}"
                    )
                }
            )
            continue
        row: dict = {}
        for name, cell in zip(field_list, cells):
            if cell.lower() in _SENTINEL_VALUES:
                row[name] = unavailable(f"nvidia-smi reported {cell!r} for {name}")
            else:
                row[name] = ok(cell)
        rows.append(row)
    return rows


def parse_xid_errors(kernel_log: str) -> dict:
    """Extract Xid fault codes from kernel-log text.

    Returns the distinct codes seen and the matching lines. An empty log is NOT
    evidence of a healthy card — it is only evidence about this boot's ring
    buffer, and `sampled_from` records that caveat for whoever reads the JSON.
    """
    codes: list[int] = []
    lines: list[str] = []
    for line in kernel_log.splitlines():
        match = XID_LINE.search(line)
        if match:
            codes.append(int(match.group(1)))
            lines.append(line.strip())
    return {
        "count": len(codes),
        "distinct_codes": sorted(set(codes)),
        "lines": lines[-50:],  # bounded: a card in a fault loop can emit thousands
    }


def parse_ecc_section(smi_q_text: str) -> dict:
    """Pull the `Ecc Errors` block out of a full `nvidia-smi -q` dump.

    The CSV query above already gives the aggregate ECC totals; this keeps the
    per-partition breakdown (SRAM vs DRAM, correctable vs uncorrectable) that
    the CSV interface does not expose, because a card throwing uncorrectable
    SRAM errors is a materially different diagnosis from a DRAM row failure.
    """
    lines = smi_q_text.splitlines()
    start = None
    indent = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("Ecc Errors"):
            start = i
            indent = len(line) - len(line.lstrip())
            break
    if start is None:
        return unavailable("no 'Ecc Errors' section in nvidia-smi -q output")

    body: list[str] = []
    for line in lines[start + 1 :]:
        if line.strip() and (len(line) - len(line.lstrip())) <= indent:
            break
        body.append(line.rstrip())
    return ok("\n".join(body).strip() or None)


def summarise_device_map(device_map: Any) -> dict:
    """Condense a resolved `hf_device_map` into the question it has to answer:
    is any part of this model NOT on a single CUDA device?

    `device_map="auto"` will silently place layers on CPU or spill to disk when
    it thinks VRAM is tight. A model that is partly on CPU still runs, still
    generates, and can still fault in ways that look like a CUDA bug — so
    "everything landed on cuda:0" is a claim worth computing rather than
    assuming from the fact that we asked for auto.
    """
    if device_map is None:
        return unavailable("model exposes no hf_device_map (single-device load)")
    if not isinstance(device_map, dict):
        return ok({"raw": str(device_map)})
    devices = sorted({str(v) for v in device_map.values()})
    offloaded = sorted(
        {str(v) for v in device_map.values() if str(v) in {"cpu", "disk"}}
    )
    return ok(
        {
            "distinct_devices": devices,
            "is_single_device": len(devices) == 1,
            "offloaded_to": offloaded,
            "module_count": len(device_map),
            "full_map": {str(k): str(v) for k, v in device_map.items()},
        }
    )


# ---------------------------------------------------------------------------
# Collectors. These touch the world; each takes its effects as arguments so the
# wiring can be tested without a GPU.
# ---------------------------------------------------------------------------
def _run(cmd: list[str], timeout: int = 20) -> tuple[bool, str]:
    if shutil.which(cmd[0]) is None:
        return False, f"{cmd[0]} not on PATH"
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired:
        return False, f"{' '.join(cmd)} timed out after {timeout}s"
    except OSError as exc:  # pragma: no cover - defensive
        return False, f"{' '.join(cmd)} failed: {exc}"
    if proc.returncode != 0:
        return False, f"{' '.join(cmd)} exited {proc.returncode}: {proc.stderr.strip()[:400]}"
    return True, proc.stdout


def collect_nvidia_smi(runner: Callable[[list[str], int], tuple[bool, str]] = _run) -> dict:
    """Identity, memory and ECC from nvidia-smi, plus the raw `-q` dump."""
    out: dict = {}

    query = "--query-gpu=" + ",".join(QUERY_GPU_FIELDS)
    okay, text = runner(["nvidia-smi", query, "--format=csv,noheader,nounits"], 20)
    out["gpus"] = parse_query_gpu_csv(text) if okay else unavailable(text)

    okay_q, text_q = runner(["nvidia-smi", "-q"], 30)
    if okay_q:
        out["ecc_detail"] = parse_ecc_section(text_q)
        out["smi_q_raw"] = ok(text_q)
    else:
        out["ecc_detail"] = unavailable(text_q)
        out["smi_q_raw"] = unavailable(text_q)
    return out


def collect_xid(runner: Callable[[list[str], int], tuple[bool, str]] = _run) -> dict:
    """Xid counters from the kernel ring buffer, trying each reader in turn.

    Inside an unprivileged container all of these commonly fail. That outcome is
    reported as unavailable — never as "0 Xid errors" — because the whole point
    of asking is to tell a sick card apart from a code bug, and a fabricated
    zero would answer that question wrongly in the reassuring direction.
    """
    attempts: list[str] = []
    for cmd in (["dmesg", "--kernel", "--nopager"], ["dmesg"], ["journalctl", "-k", "--no-pager"]):
        okay, text = runner(cmd, 20)
        if okay:
            parsed = parse_xid_errors(text)
            parsed["sampled_from"] = " ".join(cmd)
            parsed["caveat"] = (
                "Reflects only this boot's ring buffer, which may have wrapped. "
                "Zero Xid lines is not proof the GPU is healthy."
            )
            return ok(parsed)
        attempts.append(text)
    return unavailable("could not read kernel log; tried: " + " | ".join(attempts))


def collect_library_versions() -> dict:
    """Versions of everything in the load path, plus the CUDA/cuDNN torch was
    built against — which is distinct from the driver nvidia-smi reports, and a
    mismatch between the two is a plausible cause of exactly this crash."""
    versions: dict = {}
    for mod_name in ("torch", "transformers", "peft", "accelerate", "tokenizers", "numpy"):
        try:
            mod = __import__(mod_name)
        except Exception as exc:
            versions[mod_name] = unavailable(f"import failed: {exc}")
            continue
        versions[mod_name] = ok(getattr(mod, "__version__", "unknown"))

    try:
        import torch
    except Exception as exc:
        versions["torch_cuda_build"] = unavailable(f"import torch failed: {exc}")
        return versions

    versions["torch_cuda_build"] = ok(
        {
            "cuda": getattr(torch.version, "cuda", None),
            "cudnn": getattr(torch.backends.cudnn, "version", lambda: None)(),
            "hip": getattr(torch.version, "hip", None),
        }
    )
    return versions


def collect_torch_device_state(device_index: int = 0) -> dict:
    """Peak and current allocator state plus free/total VRAM from torch itself.

    Peak counters are read, not reset, here; the ladder resets them before each
    step so these are per-step figures rather than cumulative ones.
    """
    try:
        import torch
    except Exception as exc:
        return unavailable(f"import torch failed: {exc}")
    if not torch.cuda.is_available():
        return unavailable("torch.cuda.is_available() is False")

    try:
        free_b, total_b = torch.cuda.mem_get_info(device_index)
        props = torch.cuda.get_device_properties(device_index)
        return ok(
            {
                "device_index": device_index,
                "device_name": torch.cuda.get_device_name(device_index),
                "capability": f"{props.major}.{props.minor}",
                "multi_processor_count": props.multi_processor_count,
                "total_memory_bytes": int(props.total_memory),
                "free_memory_bytes": int(free_b),
                "reported_total_bytes": int(total_b),
                "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device_index)),
                "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device_index)),
                "memory_allocated_bytes": int(torch.cuda.memory_allocated(device_index)),
                "memory_reserved_bytes": int(torch.cuda.memory_reserved(device_index)),
            }
        )
    except Exception as exc:
        return unavailable(f"torch CUDA memory query failed: {exc!r}")


def collect_model_placement(model: Any) -> dict:
    """`hf_device_map` and the resolved attention implementation.

    Both are read through the PEFT wrapper as well as off the object directly:
    `PeftModel` does not re-export `hf_device_map`, so asking only the outer
    object returns None for a sharded model and would report "single-device"
    for a model that is in fact spread across CPU and GPU.
    """
    result: dict = {}

    device_map = getattr(model, "hf_device_map", None)
    if device_map is None:
        inner = getattr(model, "base_model", None)
        inner = getattr(inner, "model", inner)
        device_map = getattr(inner, "hf_device_map", None)
    result["hf_device_map"] = summarise_device_map(device_map)

    config = getattr(model, "config", None)
    if config is None:
        inner = getattr(model, "base_model", None)
        config = getattr(getattr(inner, "model", inner), "config", None)
    if config is None:
        result["attn_implementation"] = unavailable("model exposes no config")
    else:
        attn = getattr(config, "_attn_implementation", None)
        result["attn_implementation"] = (
            ok(str(attn)) if attn is not None else unavailable("config has no _attn_implementation")
        )
        result["torch_dtype"] = ok(str(getattr(config, "torch_dtype", None)))

    result["model_class"] = ok(type(model).__name__)
    try:
        result["param_device_sample"] = ok(
            {n: str(p.device) for n, p in list(model.named_parameters())[:5]}
        )
    except Exception as exc:
        result["param_device_sample"] = unavailable(f"named_parameters failed: {exc!r}")
    return result


def collect_all(model: Any = None, device_index: int = 0, include_smi_raw: bool = False) -> dict:
    """The full bundle for one ladder step.

    `include_smi_raw` is off by default so the per-step records stay readable;
    the ladder writes the raw `nvidia-smi -q` dump once, as its own artifact,
    rather than repeating ~10 KB of text four times.
    """
    smi = collect_nvidia_smi()
    if not include_smi_raw:
        smi.pop("smi_q_raw", None)

    bundle: dict = {
        "nvidia_smi": smi,
        "xid": collect_xid(),
        "torch_device": collect_torch_device_state(device_index),
        "libraries": collect_library_versions(),
    }
    bundle["model"] = collect_model_placement(model) if model is not None else unavailable(
        "no model loaded at this point"
    )
    return bundle


def unavailable_fields(bundle: Any, path: str = "") -> list[str]:
    """Every leaf in a bundle that could not be measured, with its reason.

    The ladder prints this rather than leaving the operator to notice a missing
    field inside a wall of JSON. Telemetry whose gaps are invisible is how the
    §0 probe came to have no usable evidence in the first place.
    """
    gaps: list[str] = []
    if isinstance(bundle, dict):
        if bundle.get("status") == "unavailable":
            return [f"{path or '<root>'}: {bundle.get('reason', 'no reason given')}"]
        for key, value in bundle.items():
            gaps.extend(unavailable_fields(value, f"{path}.{key}" if path else str(key)))
    elif isinstance(bundle, list):
        for i, value in enumerate(bundle):
            gaps.extend(unavailable_fields(value, f"{path}[{i}]"))
    return gaps


def to_json(bundle: dict) -> str:
    return json.dumps(bundle, indent=2, sort_keys=True, default=str)
