"""Synthetic regression checks for the optional M14 per-site observer."""
import io
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_m14_source_minor import SOURCE_HEADER, painter


PAYLOAD = (
    b"22\t100\tC\t1\t0/0:0\t0/0:0\t1/.:.\n"
    b"22\t200\tC\t1\t0/1:1\t0/0:0\t0/0:0\n"
    b"22\t300\tC\t0\t0/0:2\t0/1:1\t1/1:0\n"
    b"22\t400\tC\t1\t1|0:1\t1/1:2\t./.:.\n"
    b"22\t500\tC\t0\t1/1:0\t./.:.\t./.:.\n"
)


class ControlledQuery:
    """Small process double with explicit exceptional-cleanup observability."""

    def __init__(self, payload=PAYLOAD, status=0, slow_termination=False):
        self.stdout = io.BytesIO(payload)
        self.stderr = io.BytesIO(b"synthetic query failure" if status else b"")
        self.status = status
        self.returncode = None
        self.slow_termination = slow_termination
        self.terminated = False
        self.killed = False
        self.wait_calls = []

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        self.wait_calls.append(timeout)
        if timeout is not None and self.slow_termination:
            raise subprocess.TimeoutExpired("synthetic bcftools", timeout)
        self.returncode = -9 if self.killed else -15 if self.terminated else self.status
        return self.returncode


@unittest.skipIf(painter is None, "requires the M14 scientific Python stack")
class SiteCallbackTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.vcf = Path(self.temp.name) / "input.vcf"
        self.vcf.write_text(SOURCE_HEADER, encoding="ascii")
        self.samples_file = None

    def parse(self, process, callback=None, mode="source_minor"):
        def start(command, **kwargs):
            self.samples_file = Path(command[command.index("-S") + 1])
            self.assertEqual(self.samples_file.read_text().splitlines(), ["a", "b", "c"])
            return process

        with patch.object(painter.subprocess, "Popen", side_effect=start):
            return painter.parse_genotypes_carrier_sets(
                self.vcf, "22", ["a", "b", "c"], carrier_allele_mode=mode,
                return_orientation_qc=True, site_callback=callback,
            )

    def assert_cleaned_up(self, process):
        self.assertIsNotNone(self.samples_file)
        self.assertFalse(self.samples_file.exists())
        self.assertTrue(process.stdout.closed)
        self.assertTrue(process.stderr.closed)

    def test_every_validated_site_including_zero_and_one_carrier_is_observed(self):
        observed = []
        process = ControlledQuery()
        result = self.parse(process, lambda pos, carriers: observed.append((pos, carriers)))
        self.assertEqual(observed, [
            (100, frozenset()), (200, frozenset({0})),
            (300, frozenset({0, 1})), (400, frozenset({0, 1})),
            (500, frozenset()),
        ])
        self.assertEqual(result[1], observed[2:4])
        self.assertEqual(result[2:5], (5, 100, 500))
        self.assertEqual(result[-1]["source_ref_sites"], 2)
        self.assertEqual(result[-1]["source_alt_sites"], 3)
        self.assertEqual(result[-1]["partially_missing_genotypes"], 1)
        self.assertEqual(result[-1]["incomplete_genotypes_excluded"], 4)
        self.assert_cleaned_up(process)

    def test_callback_does_not_change_catalogue_extent_or_qc(self):
        baseline = self.parse(ControlledQuery())
        observed = []

        def observe(pos, carriers):
            observed.append((pos, carriers))
            return "ignored observer return value"

        self.assertEqual(self.parse(ControlledQuery(), observe), baseline)
        self.assertEqual(len(observed), baseline[2])

    def test_carrier_sets_given_to_observer_are_immutable(self):
        def observe(pos, carriers):
            self.assertIsInstance(carriers, frozenset)
            with self.assertRaises(AttributeError):
                carriers.add(100)

        result = self.parse(ControlledQuery(), observe)
        self.assertEqual(result[1], [(300, frozenset({0, 1})), (400, frozenset({0, 1}))])

    def test_invalid_record_is_not_observed(self):
        valid = b"22\t100\tC\t1\t0/1:1\t0/1:1\t0/0:0\n"
        invalid = [
            b"22\t200\tC\t0\t0/0:0\t0/1:1\t1/1:0\n",  # wrong RD
            b"22\t200\tC\t1\t0/.:1\t0/1:1\t0/0:0\n",  # partial RD
            b"22\t200\tC\t2\t0/1:1\t0/1:1\t0/0:0\n",  # allele code
            b"22\t200\tC,G\t1\t0/1:1\t0/1:1\t0/0:0\n",  # multiallelic
            b"21\t200\tC\t1\t0/1:1\t0/1:1\t0/0:0\n",  # chromosome
            valid,  # repeated position
            valid.replace(b"\t100\t", b"\t50\t"),  # unsorted
        ]
        for row in invalid:
            with self.subTest(row=row):
                observed = []
                process = ControlledQuery(valid + row)
                with self.assertRaises(SystemExit):
                    self.parse(process, lambda pos, carriers: observed.append((pos, carriers)))
                self.assertEqual(observed, [(100, frozenset({0, 1}))])
                self.assertTrue(process.terminated)
                self.assert_cleaned_up(process)

    def test_callback_rejected_on_legacy_input_before_query(self):
        self.vcf.write_text("\n".join(line for line in SOURCE_HEADER.splitlines()
                           if not line.startswith("##dnabr_rare_")) + "\n", encoding="ascii")
        for mode in ("historical_alt", "minor_allele"):
            with self.subTest(mode=mode), patch.object(painter.subprocess, "Popen") as query:
                with self.assertRaises(SystemExit):
                    painter.parse_genotypes_carrier_sets(
                        self.vcf, "22", ["a", "b", "c"], mode,
                        site_callback=lambda pos, carriers: None,
                    )
                query.assert_not_called()

    def test_observer_failure_propagates_and_terminates_reader(self):
        failure = RuntimeError("synthetic resource guard")

        def observe(pos, carriers):
            raise failure

        process = ControlledQuery()
        with self.assertRaises(RuntimeError) as caught:
            self.parse(process, observe)
        self.assertIs(caught.exception, failure)
        self.assertTrue(process.terminated)
        self.assertFalse(process.killed)
        self.assertEqual(process.wait_calls, [5])
        self.assert_cleaned_up(process)

    def test_reader_ignoring_terminate_is_killed_after_timeout(self):
        def observe(pos, carriers):
            raise MemoryError("synthetic budget limit")

        process = ControlledQuery(slow_termination=True)
        with self.assertRaises(MemoryError):
            self.parse(process, observe)
        self.assertTrue(process.terminated)
        self.assertTrue(process.killed)
        self.assertEqual(process.wait_calls, [5, None])
        self.assert_cleaned_up(process)

    def test_already_exited_reader_is_not_terminated_on_observer_error(self):
        def observe(pos, carriers):
            raise ValueError("synthetic observer error")

        process = ControlledQuery()
        process.returncode = 0
        with self.assertRaises(ValueError):
            self.parse(process, observe)
        self.assertFalse(process.terminated)
        self.assertFalse(process.killed)
        self.assert_cleaned_up(process)

    def test_nonzero_query_status_never_returns_partial_success(self):
        observed = []
        process = ControlledQuery(status=7)
        with self.assertRaises(SystemExit):
            self.parse(process, lambda pos, carriers: observed.append((pos, carriers)))
        self.assertEqual(len(observed), 5)  # Streaming observations are not an atomic commit.
        self.assert_cleaned_up(process)

    def test_empty_input_does_not_call_observer(self):
        observed = []
        process = ControlledQuery(payload=b"")
        result = self.parse(process, lambda pos, carriers: observed.append((pos, carriers)))
        self.assertEqual(observed, [])
        self.assertEqual(result[:5], ("22", [], 0, None, None))
        self.assert_cleaned_up(process)

    @unittest.skipUnless(shutil.which("bcftools"), "requires bcftools")
    def test_real_query_respects_selected_order_source_ref_and_partial_gt(self):
        self.vcf.write_text(SOURCE_HEADER +
            "22\t100\t.\tA\tC\t.\tPASS\tRARE_ALLELE=1\tGT:RD\t0/0:0\t0/1:1\t0/0:0\n"
            "22\t200\t.\tA\tC\t.\tPASS\tRARE_ALLELE=0\tGT:RD\t0/0:2\t1/1:0\t0/1:1\n"
            "22\t300\t.\tA\tC\t.\tPASS\tRARE_ALLELE=1\tGT:RD\t0/0:0\t0/0:0\t0/1:1\n"
            "22\t400\t.\tA\tC\t.\tPASS\tRARE_ALLELE=1\tGT:RD\t./.:.\t0/1:1\t0/.:.\n",
            encoding="ascii")
        observed = []
        result = painter.parse_genotypes_carrier_sets(
            self.vcf, "22", ["c", "a"], carrier_allele_mode="source_minor",
            site_callback=lambda pos, carriers: observed.append((pos, carriers)),
        )
        self.assertEqual(observed, [(100, frozenset()), (200, frozenset({0, 1})),
                                    (300, frozenset({0})), (400, frozenset())])
        self.assertEqual(result[1], [(200, frozenset({0, 1}))])
        self.assertEqual(result[2:5], (4, 100, 400))


if __name__ == "__main__":
    unittest.main()
