#!/usr/bin/env python3

import importlib.util
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("m33_batch_postflight", ROOT / "bin" / "m33_batch_postflight.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

RUN_ID = "m33-tabix-kat-20260822x"
RUNTIME_IMAGE = "us-central1-docker.pkg.dev/uspbr-242713/dnabr-lai/m33-tabix@sha256:" + "a" * 64
CONTROLLER_IMAGE = "us-central1-docker.pkg.dev/uspbr-242713/dnabr-lai/m33-controller@sha256:" + "b" * 64


def job(name, service_account, state, image, *, labels=None):
    return {
        "name": f"projects/uspbr-242713/locations/us-central1/jobs/{name}",
        "labels": labels or MODULE.required_labels(RUN_ID),
        "status": {"state": state},
        "allocationPolicy": {"serviceAccount": {"email": service_account}},
        "taskGroups": [{"taskSpec": {"runnables": [{"container": {"imageUri": image}}]}}],
    }


def happy_jobs(controller_state="RUNNING"):
    rows = [job("controller", MODULE.CONTROLLER_SERVICE_ACCOUNT, controller_state, CONTROLLER_IMAGE)]
    rows.extend(job(f"child-{index}", MODULE.RUNTIME_SERVICE_ACCOUNT, "SUCCEEDED", RUNTIME_IMAGE) for index in range(4))
    return rows


def receipt():
    prefix = f"gs://teams-usp/frank/lai-exploracion-datos/work/nextflow/{RUN_ID}/"
    return {
        "stage": "M33_TABIX_SYNTHETIC_KAT",
        "status": "PASS_INDEPENDENT_INDEX_AND_QUERY_PARITY",
        "contains_real_genomic_data": False,
        "cloud_context_authenticated": True,
        "runtime_service_account": MODULE.RUNTIME_SERVICE_ACCOUNT,
        "build_receipts": [
            {"replica": "A", "task_hash": "a", "task_work_dir": prefix + "a"},
            {"replica": "B", "task_hash": "b", "task_work_dir": prefix + "b"},
        ],
        **MODULE.EXPECTED_KNOWN_ANSWERS,
    }


class PostflightTests(unittest.TestCase):
    def test_run_label_is_stable_and_label_safe(self):
        self.assertRegex(MODULE.run_label(RUN_ID), r"^[0-9a-f]{16}$")
        self.assertEqual(MODULE.required_labels(RUN_ID)["pipeline"], "m33-tabix-kat")

    def test_happy_inventory_and_postflight_pass(self):
        result = MODULE.make_postflight(
            jobs=happy_jobs(), run_id=RUN_ID, runtime_image=RUNTIME_IMAGE,
            controller_image=CONTROLLER_IMAGE, kat_receipt=receipt(),
        )
        self.assertEqual(result["status"], "PASS_CHILD_JOBS_TERMINAL_AND_AUTHENTICATED")
        self.assertTrue(result["external_zero_active_check_required"])

    def test_missing_extra_wrong_identity_label_state_or_image_stop(self):
        cases = []
        cases.append(happy_jobs()[:-1])
        cases.append(happy_jobs() + [job("extra", MODULE.RUNTIME_SERVICE_ACCOUNT, "SUCCEEDED", RUNTIME_IMAGE)])
        changed = happy_jobs(); changed[1]["allocationPolicy"]["serviceAccount"]["email"] = "wrong@example.com"; cases.append(changed)
        changed = happy_jobs(); changed[1]["labels"]["team"] = "other"; cases.append(changed)
        changed = happy_jobs(); changed.append(job("hidden-extra", MODULE.RUNTIME_SERVICE_ACCOUNT, "RUNNING", RUNTIME_IMAGE, labels={"pipeline": MODULE.PIPELINE_LABEL, "run": MODULE.run_label(RUN_ID), "team": "other"})); cases.append(changed)
        changed = happy_jobs(); changed.append(job("fully-hidden-extra", MODULE.RUNTIME_SERVICE_ACCOUNT, "RUNNING", RUNTIME_IMAGE, labels={"pipeline": "other", "run": MODULE.run_label(RUN_ID), "team": "other"})); cases.append(changed)
        changed = happy_jobs(); changed[1]["status"]["state"] = "RUNNING"; cases.append(changed)
        changed = happy_jobs(); changed[1]["taskGroups"][0]["taskSpec"]["runnables"][0]["container"]["imageUri"] = "wrong"; cases.append(changed)
        for rows in cases:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                MODULE.validate_inventory(
                    rows, run_id=RUN_ID, runtime_image=RUNTIME_IMAGE,
                    controller_image=CONTROLLER_IMAGE, controller_may_be_running=True,
                )

    def test_receipt_requires_independent_builds_and_known_answers(self):
        changed = receipt()
        changed["build_receipts"][1]["task_hash"] = "a"
        with self.assertRaisesRegex(ValueError, "task hash"):
            MODULE.validate_kat_receipt(changed, RUN_ID)
        changed = receipt(); changed["record_count"] = 3
        with self.assertRaisesRegex(ValueError, "known answers"):
            MODULE.validate_kat_receipt(changed, RUN_ID)

    def test_inventory_must_be_observed_stably_twice(self):
        fetch = mock.Mock(side_effect=[happy_jobs()[:-1], happy_jobs(), happy_jobs()])
        observed = MODULE.stable_inventory(fetch, run_id=RUN_ID, attempts=3, interval_seconds=0)
        self.assertEqual(len(observed), 5)
        with self.assertRaisesRegex(ValueError, "did not stabilize"):
            MODULE.stable_inventory(
                mock.Mock(side_effect=[happy_jobs()[:-1], happy_jobs(), happy_jobs()[:-1]]),
                run_id=RUN_ID, attempts=3, interval_seconds=0,
            )


if __name__ == "__main__":
    unittest.main()
