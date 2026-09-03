from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))

from m37_trace_recovery_gate import (
    ARMS,
    AUDIT_BASENAME,
    SUMMARY_BASENAME,
    canonical_sha256,
    sha256,
    validate_contract,
    validate_family_bundle,
    validate_job,
    validate_positive_control,
    validate_source_archive,
    write_nonconsumable_outputs,
)


CONTAINER = "registry.example/model@sha256:" + "a" * 64


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _base_contract() -> dict:
    return {
        "schema_version": "1.1.0",
        "stage": "M37_TRACE_ORPHAN_RECOVERY_CONTRACT",
        "status": "FROZEN",
        "recovery_id": "fixture-recovery",
        "origin_run_id": "fixture-run",
        "root": "R0",
        "evaluation_split": "FIT_TUNE",
        "source_commit": "b" * 40,
        "cloud": {
            "project": "fixture-project",
            "location": "us-central1",
            "required_labels": {"lane": "m37-trace", "team": "frank"},
            "container_digest": CONTAINER,
        },
        "positive_control": {
            "work_dir": "gs://teams-usp/frank/lai-exploracion-datos/work/pc",
            "payload_sha256": "",
            "receipt_sha256": "",
        },
        "jobs": {
            family: {
                "job_id": f"nf-{family}",
                "work_dir": (
                    "gs://teams-usp/frank/lai-exploracion-datos/work/nextflow/"
                    f"fixture/{family}"
                ),
                "candidate_count": 1,
                "metric_count": len(ARMS),
            }
            for family in ("hmm", "tcn")
        },
        "frozen_input_sha256": {
            "candidate_manifest": "1" * 64,
            "parent_contract": "2" * 64,
            "contract_amendment": "3" * 64,
            "canonical_metrics": "4" * 64,
            "canonical_metrics_receipt": "5" * 64,
            "truth": "6" * 64,
            "run_overlay": "7" * 64,
        },
        "executed_source_sha256": {"runner.py": "8" * 64},
        "downstream_source_sha256": {},
        "recovery_policy": {
            "rerun_family_jobs": False,
            "publish_raw_family_outputs": False,
            "publish_predictions_or_checkpoints": False,
            "open_valid_or_test": False,
            "consumable_for_candidate_selection_or_inference": False,
            "run_standard_decision_or_promotion": False,
            "publish_only_recovered_nonconsumable_audit_and_summary": True,
            "disposition": "RECOVER_DOWNSTREAM_AUDIT_ONLY_AFTER_SIGHUP",
        },
    }


def _job(contract: dict, family: str) -> dict:
    cloud, expected = contract["cloud"], contract["jobs"][family]
    mounted = expected["work_dir"].replace("gs://", "/mnt/disks/", 1)
    return {
        "name": (f"projects/{cloud['project']}/locations/{cloud['location']}/jobs/"
                 f"{expected['job_id']}"),
        "labels": cloud["required_labels"],
        "allocationPolicy": {"labels": {
            **cloud["required_labels"], "batch-job-id": expected["job_id"]
        }},
        "status": {
            "state": "SUCCEEDED",
            "taskGroups": {"group0": {"counts": {"SUCCEEDED": "1"}}},
            "statusEvents": [{"eventTime": "2026-09-03T01:00:00Z"}],
        },
        "taskGroups": [{"taskSpec": {"runnables": [{"container": {
            "imageUri": cloud["container_digest"],
            "commands": ["/bin/bash", f"run {mounted}/.command.run"],
        }}]}}],
    }


def _positive_control(root: Path, contract: dict) -> tuple[Path, Path]:
    payload = root / "m37.compact_positive_control.json"
    receipt = root / "m37.compact_positive_control.receipt.json"
    _write_json(payload, {
        "stage": "M37_TRACE_COMPACT_POSITIVE_CONTROL",
        "run_id": contract["origin_run_id"],
        "container_digest": contract["cloud"]["container_digest"],
    })
    _write_json(receipt, {
        "stage": "M37_TRACE_COMPACT_POSITIVE_CONTROL",
        "run_id": contract["origin_run_id"],
        "output_sha256": sha256(payload),
    })
    contract["positive_control"]["payload_sha256"] = sha256(payload)
    contract["positive_control"]["receipt_sha256"] = sha256(receipt)
    return payload, receipt


def _family_bundle(root: Path, family: str, contract: dict) -> tuple[list[Path], ...]:
    metrics, receipts = [], []
    candidate = f"{family}_fixture"
    for arm in ARMS:
        metric = root / f"{candidate}.{family}.{arm}.metrics.json"
        receipt = root / f"{candidate}.{family}.{arm}.metrics.receipt.json"
        _write_json(metric, {
            "stage": "M37_TRACE_SCORE", "candidate_id": candidate,
            "family": family, "root": "R0", "arm": arm,
            "run_id": "fixture-run", "evaluation_split": "FIT_TUNE",
        })
        _write_json(receipt, {
            "stage": "M37_TRACE_SCORE", "candidate_id": candidate,
            "family": family, "root": "R0", "arm": arm,
            "output_sha256": sha256(metric),
        })
        metrics.append(metric)
        receipts.append(receipt)
    equivalence = root / f"{family}.equivalence.json"
    equivalence_receipt = root / f"{family}.equivalence.receipt.json"
    _write_json(equivalence, {
        "stage": "M37_TRACE_COMPACT_EQUIVALENCE", "status": "PASS",
        "run_id": "fixture-run", "root": "R0", "family": family,
    })
    _write_json(equivalence_receipt, {
        "stage": "M37_TRACE_COMPACT_EQUIVALENCE", "run_id": "fixture-run",
        "root": "R0", "family": family, "output_sha256": sha256(equivalence),
    })
    audit = root / f"{family}.compact_sweep.audit.json"
    audit_receipt = root / f"{family}.compact_sweep.audit.receipt.json"
    frozen = contract["frozen_input_sha256"]
    _write_json(audit, {
        "stage": "M37_TRACE_COMPACT_SWEEP", "status": "PASS_FIT_TUNE_ONLY",
        "run_id": "fixture-run", "root": "R0", "family": family,
        "candidate_count": 1, "metric_count": len(ARMS),
        "container_digest": contract["cloud"]["container_digest"],
        "metric_sha256": {path.name: sha256(path) for path in metrics},
        "metric_receipt_sha256": {path.name: sha256(path) for path in receipts},
        "authenticated_source_sha256": contract["executed_source_sha256"],
        "manifest_sha256": frozen["candidate_manifest"],
        "parent_contract_sha256": frozen["parent_contract"],
        "contract_amendment_sha256": frozen["contract_amendment"],
        "canonical_metrics_sha256": frozen["canonical_metrics"],
        "canonical_metrics_receipt_sha256": frozen["canonical_metrics_receipt"],
        "truth_sha256": frozen["truth"],
        "run_overlay": {"sha256": frozen["run_overlay"]},
        "positive_control_sha256": contract["positive_control"]["payload_sha256"],
        "positive_control_receipt_sha256": contract["positive_control"]["receipt_sha256"],
        "equivalence_sha256": sha256(equivalence),
        "equivalence_receipt_sha256": sha256(equivalence_receipt),
    })
    _write_json(audit_receipt, {
        "stage": "M37_TRACE_COMPACT_SWEEP", "run_id": "fixture-run",
        "root": "R0", "family": family, "output_sha256": sha256(audit),
    })
    return metrics, receipts, [audit], [audit_receipt], [equivalence], [equivalence_receipt]


def test_gate_validates_jobs_source_archive_and_complete_family_bundles() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        contract = _base_contract()
        source = b"frozen decision source\n"
        contract["downstream_source_sha256"] = {
            "bin/frozen.py": hashlib.sha256(source).hexdigest()
        }
        archive = root / "source.tar"
        with tarfile.open(archive, "w") as handle:
            info = tarfile.TarInfo("bin/frozen.py")
            info.size = len(source)
            handle.addfile(info, io.BytesIO(source))
        positive, positive_receipt = _positive_control(root, contract)
        validate_contract(contract)
        assert validate_source_archive(archive, contract["downstream_source_sha256"])
        assert validate_positive_control(positive, positive_receipt, contract)
        for family in ("hmm", "tcn"):
            assert validate_job(_job(contract, family), contract["jobs"][family],
                                contract["cloud"])["state"] == "SUCCEEDED"
            metrics, receipts, audits, audit_receipts, equivalences, equivalence_receipts = (
                _family_bundle(root, family, contract)
            )
            evidence = validate_family_bundle(
                family, metrics, receipts, audits[0], audit_receipts[0],
                equivalences[0], equivalence_receipts[0], contract,
            )
            assert evidence["candidate_count"] == 1 and evidence["metric_count"] == 5


def test_job_gate_fails_closed_on_state_label_or_workdir_drift() -> None:
    contract = _base_contract()
    for mutation in ("state", "label", "workdir"):
        job = _job(contract, "hmm")
        if mutation == "state":
            job["status"]["state"] = "RUNNING"
        elif mutation == "label":
            job["labels"]["team"] = "other"
        else:
            job["taskGroups"][0]["taskSpec"]["runnables"][0]["container"]["commands"] = ["wrong"]
        try:
            validate_job(job, contract["jobs"]["hmm"], contract["cloud"])
        except ValueError:
            pass
        else:
            raise AssertionError(f"job gate accepted {mutation} drift")


def test_metric_gate_fails_closed_on_byte_drift() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        contract = _base_contract()
        _positive_control(root, contract)
        metrics, receipts, audits, audit_receipts, equivalences, equivalence_receipts = (
            _family_bundle(root, "hmm", contract)
        )
        metrics[0].write_text("{}\n", encoding="utf-8")
        try:
            validate_family_bundle(
                "hmm", metrics, receipts, audits[0], audit_receipts[0],
                equivalences[0], equivalence_receipts[0], contract,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("family gate accepted a changed metric")


def test_recovery_workflow_publishes_only_nonconsumable_evidence() -> None:
    workflow = (ROOT / "workflows/m37_trace_compact_recovery.nf").read_text(encoding="utf-8")
    module = (ROOT / "modules/37_TRACE_COMPACT_RECOVERY.nf").read_text(encoding="utf-8")
    config = (ROOT / "conf/m37_r0_compact_recovery.config").read_text(encoding="utf-8")
    assert "M37_TRACE_RECOVERY_GATE" in workflow
    assert "M37_TRACE_RECOVERY_COLLECT_METRICS" not in workflow
    assert "M37_TRACE_RECOVERY_COMPACT_DECISION" not in workflow
    assert "M37_TRACE_COMPACT_SWEEP(" not in workflow
    assert "m37_trace_compact_sweep.py" not in module
    assert "m37_trace_train.py" not in module
    assert "m37_trace_collect_metrics.py" not in module
    assert "m37_trace_compact_decision.py" not in module
    assert "/promotion" not in module
    assert "overwrite: false" in module
    assert "audit/recovery" in module
    assert "m37.recovered.nonconsumable.audit.json" in module
    assert "m37.recovered.nonconsumable.audit.receipt.json" in module
    assert "m37.recovered.nonconsumable.summary.json" in module
    assert "m37.recovered.nonconsumable.summary.receipt.json" in module
    assert "ADVANCE" not in workflow + module + config
    assert "publish_raw_family_outputs\": false" in (
        ROOT / "conf/m37_trace_compact_recovery_contract.json"
    ).read_text(encoding="utf-8")
    assert "gs://teams-usp/frank/lai-exploracion-datos/" in config
    assert "team: 'frank'" in config and "lane: 'm37-recovery'" in config


def test_recovery_outputs_are_intrinsically_nonconsumable_and_exactly_named() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        contract = _base_contract()
        source = b"frozen recovery evidence\n"
        contract["downstream_source_sha256"] = {
            "bin/frozen.py": hashlib.sha256(source).hexdigest()
        }
        archive = root / "source.tar"
        with tarfile.open(archive, "w") as handle:
            info = tarfile.TarInfo("bin/frozen.py")
            info.size = len(source)
            handle.addfile(info, io.BytesIO(source))
        positive, positive_receipt = _positive_control(root, contract)
        contract_path = root / "contract.json"
        _write_json(contract_path, contract)

        family_evidence = {}
        job_evidence = {}
        for family in ("hmm", "tcn"):
            metrics, receipts, audits, audit_receipts, equivalences, equivalence_receipts = (
                _family_bundle(root, family, contract)
            )
            family_evidence[family] = validate_family_bundle(
                family, metrics, receipts, audits[0], audit_receipts[0],
                equivalences[0], equivalence_receipts[0], contract,
            )
            job_evidence[family] = validate_job(
                _job(contract, family), contract["jobs"][family], contract["cloud"]
            )

        output_dir = root / "outputs"
        output_dir.mkdir()
        outputs = write_nonconsumable_outputs(
            output_dir,
            contract_path,
            archive,
            validate_source_archive(archive, contract["downstream_source_sha256"]),
            validate_positive_control(positive, positive_receipt, contract),
            job_evidence,
            family_evidence,
            contract,
        )
        assert {path.name for path in outputs} == {
            AUDIT_BASENAME,
            AUDIT_BASENAME.removesuffix(".json") + ".receipt.json",
            SUMMARY_BASENAME,
            SUMMARY_BASENAME.removesuffix(".json") + ".receipt.json",
        }
        assert {path.name for path in output_dir.iterdir()} == {path.name for path in outputs}
        for path in outputs:
            payload = json.loads(path.read_text(encoding="utf-8"))
            encoded = json.dumps(payload, sort_keys=True)
            assert payload["status"] == "RECOVERED_NONCONSUMABLE"
            assert payload["consumable"] is False
            assert payload["candidate_selection"] == "FORBIDDEN"
            assert payload["model_promotion"] == "FORBIDDEN"
            assert payload["scientific_inference"] == "FORBIDDEN"
            assert "ADVANCE" not in encoded
            assert "decisions" not in payload
        summary = json.loads((output_dir / SUMMARY_BASENAME).read_text(encoding="utf-8"))
        assert summary["standard_decision_executed"] is False
        assert summary["recovery_id"] == "fixture-recovery"
        assert summary["audit_receipt_sha256"] == sha256(
            output_dir / (AUDIT_BASENAME.removesuffix(".json") + ".receipt.json")
        )
        assert summary["total_candidate_count"] == 2
        assert summary["total_metric_count"] == 10


def test_production_recovery_contract_is_sealed_to_exact_orphan_jobs() -> None:
    contract = json.loads((
        ROOT / "conf/m37_trace_compact_recovery_contract.json"
    ).read_text(encoding="utf-8"))
    validate_contract(contract)
    assert contract["schema_version"] == "1.1.0"
    assert contract["recovery_id"] == "m37-r0-compact-recovery-20260903a"
    assert contract["source_commit"] == "e22e1edd412820ff251834d7fe416130b41468ac"
    assert contract["jobs"]["hmm"]["job_id"] == "nf-047b6663-1788396822904"
    assert contract["jobs"]["tcn"]["job_id"] == "nf-9405b1ec-1788396822936"
    assert contract["jobs"]["hmm"]["metric_count"] == 100
    assert contract["jobs"]["tcn"]["metric_count"] == 15
    assert contract["cloud"]["required_labels"] == {"lane": "m37-trace", "team": "frank"}
    assert set(contract["downstream_source_sha256"]) == {
        "bin/m37_trace_collect_metrics.py",
        "bin/m37_trace_compact_decision.py",
        "bin/m37_trace_core.py",
        "bin/m37_trace_successive_halving.py",
        "conf/m37_trace_compact_sweep.config",
        "conf/m37_trace_gcp.config",
        "modules/37_TRACE_COMPACT_SWEEP.nf",
        "workflows/m37_trace_compact_sweep.nf",
    }
    assert all(len(value) == 64 for value in contract["downstream_source_sha256"].values())


def test_canonical_job_descriptor_hash_is_order_invariant() -> None:
    left = {"state": "SUCCEEDED", "labels": {"team": "frank", "lane": "m37-trace"}}
    right = {"labels": {"lane": "m37-trace", "team": "frank"}, "state": "SUCCEEDED"}
    assert canonical_sha256(left) == canonical_sha256(right)
