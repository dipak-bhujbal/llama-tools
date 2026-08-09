"""Tests for eval/gpu_telemetry.py.

The property under test throughout is the one the module exists for: **an
unmeasured field is reported as unavailable, never as zero.** A telemetry bundle
that degrades silently to plausible zeros is worse than no telemetry, because it
answers "was the card throwing ECC errors?" with a confident and unfounded no.

Everything here runs with no GPU, no torch, and no subprocess: the parsers are
pure and the collectors take their effects as arguments.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval import gpu_telemetry as gt  # noqa: E402

DATACENTER_CSV = (
    "NVIDIA A100-SXM4-40GB, GPU-abc123, 550.90.07, 40960, 39000, 1960, Enabled, 0, 0, 3, 0, 0"
)
# A consumer 4090 — the card the mining pilot actually ran on — has no ECC at
# all, so nvidia-smi answers those columns with [N/A].
GEFORCE_CSV = (
    "NVIDIA GeForce RTX 4090, GPU-def456, 550.90.07, 24564, 24000, 564, "
    "[N/A], [N/A], [N/A], [N/A], [N/A], [N/A]"
)


def test_query_gpu_csv_parses_all_columns_as_ok() -> None:
    rows = gt.parse_query_gpu_csv(DATACENTER_CSV)
    assert len(rows) == 1
    row = rows[0]
    assert row["name"] == gt.ok("NVIDIA A100-SXM4-40GB")
    assert row["uuid"]["value"] == "GPU-abc123"
    assert row["driver_version"]["value"] == "550.90.07"
    assert row["ecc.errors.uncorrected.volatile.total"]["value"] == "0"


def test_not_available_ecc_is_unavailable_and_never_zero() -> None:
    """The core guarantee. A GeForce card reports [N/A] for ECC; mapping that to
    0 would claim the card is clean when nothing was measured."""
    row = gt.parse_query_gpu_csv(GEFORCE_CSV)[0]
    for column in (
        "ecc.mode.current",
        "ecc.errors.corrected.volatile.total",
        "ecc.errors.uncorrected.volatile.total",
        "ecc.errors.uncorrected.aggregate.total",
    ):
        field = row[column]
        assert field["status"] == "unavailable", column
        assert field["value"] is None, column
        assert field["value"] != 0, column
        assert "reason" in field, column
    # Identity columns are still real measurements and must survive.
    assert row["name"]["value"] == "NVIDIA GeForce RTX 4090"


def test_malformed_csv_row_does_not_raise_and_is_marked_unavailable() -> None:
    rows = gt.parse_query_gpu_csv("only, three, columns")
    assert len(rows) == 1
    assert rows[0]["_parse_error"]["status"] == "unavailable"


def test_multiple_gpus_produce_multiple_rows() -> None:
    rows = gt.parse_query_gpu_csv(DATACENTER_CSV + "\n" + GEFORCE_CSV)
    assert len(rows) == 2
    assert rows[1]["uuid"]["value"] == "GPU-def456"


def test_xid_parser_extracts_codes_and_lines() -> None:
    log = (
        "[  10.0] some unrelated kernel line\n"
        "[  12.0] NVRM: Xid (PCI:0000:01:00): 13, pid=123, Graphics Exception\n"
        "[  14.0] NVRM: Xid (PCI:0000:01:00): 31, pid=124, Ch 00000008\n"
        "[  15.0] NVRM: Xid (PCI:0000:01:00): 13, pid=125\n"
    )
    parsed = gt.parse_xid_errors(log)
    assert parsed["count"] == 3
    assert parsed["distinct_codes"] == [13, 31]
    assert len(parsed["lines"]) == 3


def test_xid_lines_are_bounded_so_a_fault_loop_cannot_blow_up_the_record() -> None:
    log = "\n".join(f"NVRM: Xid (PCI:0000:01:00): 13, pid={i}" for i in range(500))
    parsed = gt.parse_xid_errors(log)
    assert parsed["count"] == 500
    assert len(parsed["lines"]) == 50


def test_collect_xid_reports_unavailable_when_every_reader_fails() -> None:
    """Inside an unprivileged container dmesg and journalctl both commonly fail.
    That must not silently become 'zero Xid errors'."""

    def always_fails(cmd, timeout):
        return False, f"{cmd[0]}: Operation not permitted"

    result = gt.collect_xid(runner=always_fails)
    assert result["status"] == "unavailable"
    assert result["value"] is None
    assert "dmesg" in result["reason"]


def test_collect_xid_falls_through_to_the_next_reader() -> None:
    calls: list[str] = []

    def dmesg_denied(cmd, timeout):
        calls.append(cmd[0])
        if cmd[0] == "dmesg":
            return False, "Operation not permitted"
        return True, "NVRM: Xid (PCI:0000:01:00): 79, pid=1\n"

    result = gt.collect_xid(runner=dmesg_denied)
    assert result["status"] == "ok"
    assert result["value"]["distinct_codes"] == [79]
    assert result["value"]["sampled_from"].startswith("journalctl")
    assert "not proof" in result["value"]["caveat"]


def test_clean_kernel_log_carries_the_caveat_rather_than_a_clean_bill_of_health() -> None:
    result = gt.collect_xid(runner=lambda cmd, timeout: (True, "nothing interesting\n"))
    assert result["value"]["count"] == 0
    assert "may have wrapped" in result["value"]["caveat"]


def test_ecc_section_is_extracted_from_smi_q_by_indentation() -> None:
    text = (
        "GPU 00000000:01:00.0\n"
        "    Product Name                          : NVIDIA A100\n"
        "    Ecc Errors\n"
        "        Volatile\n"
        "            SRAM Correctable              : 0\n"
        "            SRAM Uncorrectable            : 2\n"
        "    Retired Pages\n"
        "        Single Bit ECC                    : 0\n"
    )
    section = gt.parse_ecc_section(text)
    assert section["status"] == "ok"
    assert "SRAM Uncorrectable            : 2" in section["value"]
    # The next same-indent heading terminates the block.
    assert "Retired Pages" not in section["value"]


def test_missing_ecc_section_is_unavailable() -> None:
    assert gt.parse_ecc_section("GPU 0\n    Product Name : X\n")["status"] == "unavailable"


def test_device_map_flags_cpu_offload() -> None:
    summary = gt.summarise_device_map(
        {"model.layers.0": 0, "model.layers.1": "cpu", "lm_head": "disk"}
    )
    assert summary["status"] == "ok"
    assert summary["value"]["is_single_device"] is False
    assert summary["value"]["offloaded_to"] == ["cpu", "disk"]
    assert summary["value"]["module_count"] == 3


def test_device_map_single_device_is_reported_as_such() -> None:
    summary = gt.summarise_device_map({"model.layers.0": "cuda:0", "lm_head": "cuda:0"})
    assert summary["value"]["is_single_device"] is True
    assert summary["value"]["offloaded_to"] == []


def test_absent_device_map_is_unavailable_not_an_empty_map() -> None:
    summary = gt.summarise_device_map(None)
    assert summary["status"] == "unavailable"
    assert summary["value"] is None


# --- model placement, including through the PEFT wrapper --------------------
class _FakeConfig:
    _attn_implementation = "sdpa"
    torch_dtype = "torch.bfloat16"


class _FakeInner:
    hf_device_map = {"model.layers.0": "cuda:0"}
    config = _FakeConfig()


class _FakeBaseModel:
    model = _FakeInner()


class _FakePeftModel:
    """Mirrors the shape that matters: PeftModel does not re-export
    hf_device_map, so it is only reachable via .base_model.model."""

    base_model = _FakeBaseModel()

    def named_parameters(self):
        return []


def test_device_map_is_found_through_the_peft_wrapper() -> None:
    placement = gt.collect_model_placement(_FakePeftModel())
    assert placement["hf_device_map"]["status"] == "ok"
    assert placement["hf_device_map"]["value"]["is_single_device"] is True
    assert placement["attn_implementation"]["value"] == "sdpa"
    assert placement["model_class"]["value"] == "_FakePeftModel"


def test_model_without_config_is_unavailable_not_crashing() -> None:
    class Bare:
        def named_parameters(self):
            raise RuntimeError("no params")

    placement = gt.collect_model_placement(Bare())
    assert placement["hf_device_map"]["status"] == "unavailable"
    assert placement["attn_implementation"]["status"] == "unavailable"
    assert placement["param_device_sample"]["status"] == "unavailable"


# --- gap reporting ----------------------------------------------------------
def test_unavailable_fields_walks_nested_bundles_and_reports_paths() -> None:
    bundle = {
        "nvidia_smi": {"gpus": [{"name": gt.ok("A100"), "ecc.mode.current": gt.unavailable("no ECC")}]},
        "xid": gt.unavailable("permission denied"),
        "torch_device": gt.ok({"free_memory_bytes": 1}),
    }
    gaps = gt.unavailable_fields(bundle)
    assert any("xid" in g and "permission denied" in g for g in gaps)
    assert any("ecc.mode.current" in g for g in gaps)
    assert not any("torch_device" in g for g in gaps)


def test_a_fully_measured_bundle_reports_no_gaps() -> None:
    assert gt.unavailable_fields({"a": gt.ok(1), "b": {"c": gt.ok(2)}}) == []


def test_collect_nvidia_smi_marks_everything_unavailable_when_smi_is_absent() -> None:
    result = gt.collect_nvidia_smi(runner=lambda cmd, timeout: (False, "nvidia-smi not on PATH"))
    assert result["gpus"]["status"] == "unavailable"
    assert result["ecc_detail"]["status"] == "unavailable"
    assert gt.unavailable_fields(result)


def test_to_json_round_trips() -> None:
    import json

    bundle = {"xid": gt.unavailable("denied"), "libraries": {"torch": gt.ok("2.4.0")}}
    assert json.loads(gt.to_json(bundle))["xid"]["value"] is None


# --- persistence -------------------------------------------------------------
def test_write_json_atomic_leaves_no_partial_file_and_no_debris(tmp_path) -> None:
    """Atomic because the process being observed is one that dies abruptly. A
    SIGKILL landing mid-write would otherwise leave a truncated file that the
    next reader parses as a fact."""
    import json

    target = tmp_path / "nested" / "snapshot.json"
    gt.write_json_atomic(target, {"xid": gt.unavailable("denied")})
    assert json.loads(target.read_text())["xid"]["value"] is None
    assert not list(target.parent.glob("*.tmp.*"))


def test_write_json_atomic_overwrites_rather_than_appends(tmp_path) -> None:
    import json

    target = tmp_path / "s.json"
    gt.write_json_atomic(target, {"n": 1})
    gt.write_json_atomic(target, {"n": 2})
    assert json.loads(target.read_text()) == {"n": 2}


def test_raw_smi_dump_records_its_own_absence_rather_than_vanishing(tmp_path, monkeypatch) -> None:
    """An earlier version's docstring claimed the ladder persisted this dump
    while no code did — a stated guarantee with nothing behind it. It is now
    written, and when nvidia-smi is unavailable the file says so instead of
    being silently absent."""
    monkeypatch.setattr(gt, "_run", lambda cmd, timeout=20: (False, "nvidia-smi not on PATH"))
    target = tmp_path / "smi.txt"
    status = gt.write_raw_smi_query(target)
    assert status["status"] == "unavailable"
    assert "not on PATH" in target.read_text()


def test_collect_all_labels_its_phase_and_omits_static_libraries_per_rung() -> None:
    bundle = gt.collect_all(model=None, phase="after_load")
    assert bundle["phase"] == "after_load"
    # Library versions cannot change between rungs; repeating them four times
    # per rung buries the fields that do change.
    assert "libraries" not in bundle


def test_collect_all_includes_libraries_on_the_full_bundle() -> None:
    assert "libraries" in gt.collect_all(model=None, include_smi_raw=True, phase="pre_run")
