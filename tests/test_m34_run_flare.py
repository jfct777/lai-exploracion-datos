#!/usr/bin/env python3

from __future__ import annotations

import gzip
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
import m34_run_flare as subject


def write_vcf(path: Path, samples: list[str], rows: list[list[str]],
              chromosome: str = "22") -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("##fileformat=VCFv4.2\n")
        handle.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t")
        handle.write("\t".join(samples) + "\n")
        for index, genotypes in enumerate(rows):
            handle.write(f"{chromosome}\t{100 + 100 * index}\t.\tA\tG\t.\tPASS\t.\tGT\t")
            handle.write("\t".join(genotypes) + "\n")


def fixture(base: Path) -> Namespace:
    reference = base / "reference.vcf.gz"
    target = base / "target.vcf.gz"
    write_vcf(reference, ["AFR1", "EUR1", "NAM1"],
              [["0|1", "0|0", "1|0"], ["1|1", "0|1", "1|0"]])
    write_vcf(target, ["T1", "T2"], [["0|1", "1|0"], ["1|1", "0|0"]])
    reference_tbi, target_tbi = base / "reference.vcf.gz.tbi", base / "target.vcf.gz.tbi"
    reference_tbi.write_bytes(b"reference-index")
    target_tbi.write_bytes(b"target-index")
    sample_map = base / "sample-map.tsv"
    sample_map.write_text("AFR1\tAFR\nEUR1\tEUR\nNAM1\tNAM\n", encoding="utf-8")
    genetic_map = base / "map.tsv"
    genetic_map.write_text("22\t50\t0.0\n22\t150\t0.1\n22\t250\t0.2\n",
                           encoding="utf-8")
    jar = base / "flare.jar"
    jar.write_bytes(b"fixed-flare-jar")
    paths = {
        "reference_vcf": reference, "reference_tbi": reference_tbi,
        "target_vcf": target, "target_tbi": target_tbi,
        "sample_map": sample_map, "genetic_map": genetic_map, "flare_jar": jar,
    }
    contract = base / "contract.json"
    contract.write_text(json.dumps({
        "schema_version": "1.0.0", "stage": "M34_AFR_EUR_NAM_FLARE",
        "status": "EXPLORATORY_CONTRACT_BLINDED_TO_LABELS", "chromosome": "22",
        "ancestry_names": ["AFR", "EUR", "NAM"],
        "parameters": {
            "array": False, "probs": True, "em": True, "min-mac": 1,
            "min-maf": 0.0, "gen": 12, "update-p": False,
            "panel-probs": False, "seed": 3401, "nthreads": 4,
        },
        "expected_sha256": {name: subject.sha256_file(path) for name, path in paths.items()},
    }), encoding="utf-8")
    return Namespace(
        contract=contract, reference_vcf=reference, reference_tbi=reference_tbi,
        target_vcf=target, target_tbi=target_tbi, sample_map=sample_map,
        genetic_map=genetic_map, flare_jar=jar, java="java",
        outdir=base / "output", preflight_only=True,
    )


def write_ancestry_vcf(path: Path, target_vcf: Path,
                       second_probability: str = "0.1,0.8,0.1") -> None:
    target = subject.scan_vcf(target_vcf, "22")
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("##fileformat=VCFv4.2\n")
        handle.write("##ANCESTRY=<AFR=0,EUR=1,NAM=2>\n")
        for name in ("AN1", "AN2"):
            handle.write(f'##FORMAT=<ID={name},Number=1,Type=Integer,Description="hard">\n')
        for name in ("ANP1", "ANP2"):
            handle.write(f'##FORMAT=<ID={name},Number=3,Type=Float,Description="probability">\n')
        handle.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t")
        handle.write("\t".join(target["samples"]) + "\n")
        for chrom, position, ref, alt in target["loci"]:
            values = [f"0:1:0.8,0.1,0.1:{second_probability}" for _sample in target["samples"]]
            handle.write(f"{chrom}\t{position}\t.\t{ref}\t{alt}\t.\tPASS\t.\t"
                         "AN1:AN2:ANP1:ANP2\t" + "\t".join(values) + "\n")


class M34FlareRunnerTests(unittest.TestCase):
    def test_preflight_only_validates_and_never_starts_java(self):
        with tempfile.TemporaryDirectory() as directory:
            args = fixture(Path(directory))
            result = subject.run(args)
            self.assertEqual(result["status"], "PASS_PREFLIGHT_ONLY")
            self.assertTrue(result["preflight_only"])
            self.assertFalse(result["truth_accessed"])
            self.assertIsNone(result["ancestry_vcf_audit"])
            self.assertEqual(result["shape"]["reference_panel_counts"],
                             {"AFR": 1, "EUR": 1, "NAM": 1})
            self.assertEqual((args.outdir / "flare.ref-panel.tsv").read_text(),
                             "AFR1\tAFR\nEUR1\tEUR\nNAM1\tNAM\n")
            self.assertEqual((args.outdir / "flare.map").read_text().splitlines(),
                             ["22\t22:50\t0\t50", "22\t22:150\t0.1\t150",
                              "22\t22:250\t0.2\t250"])

    def test_command_contains_only_declared_truth_blind_inputs_and_parameters(self):
        command = subject.build_command(
            "java", Path("flare.jar"), Path("ref.vcf.gz"), Path("target.vcf.gz"),
            Path("panel.tsv"), Path("map.tsv"), Path("result"), {
                "array": False, "probs": True, "em": True, "min-mac": 1,
                "min-maf": 0.0, "gen": 12, "update-p": False,
                "panel-probs": False, "seed": 3401, "nthreads": 8,
            })
        self.assertEqual(command[:3], ["java", "-jar", "flare.jar"])
        for value in ("array=false", "probs=true", "em=true", "min-mac=1",
                      "min-maf=0", "gen=12", "update-p=false", "panel-probs=false",
                      "seed=3401", "nthreads=8"):
            self.assertIn(value, command)
        self.assertNotIn("truth", " ".join(command).lower())
        self.assertNotIn("model=", " ".join(command))

    def test_map_contig_is_normalized_to_exact_target_vcf_spelling(self):
        with tempfile.TemporaryDirectory() as directory:
            args = fixture(Path(directory))
            write_vcf(args.target_vcf, ["T1", "T2"],
                      [["0|1", "1|0"], ["1|1", "0|0"]], chromosome="chr22")
            contract = json.loads(args.contract.read_text())
            contract["expected_sha256"]["target_vcf"] = subject.sha256_file(args.target_vcf)
            args.contract.write_text(json.dumps(contract), encoding="utf-8")
            result = subject.run(args)
            lines = (args.outdir / "flare.map").read_text().splitlines()
            self.assertEqual(lines[0], "chr22\tchr22:50\t0\t50")
            self.assertEqual(result["derived_input_audit"]["genetic_map"]
                             ["output_chromosome"], "chr22")

    def test_known_answer_ancestry_vcf_passes_complete_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            args = fixture(Path(directory))
            output = Path(directory) / "known.anc.vcf.gz"
            write_ancestry_vcf(output, args.target_vcf)
            target = subject.scan_vcf(args.target_vcf, "22")
            result = subject.audit_ancestry_vcf(output, target, ["AFR", "EUR", "NAM"])
            self.assertEqual(result["marker_count"], 2)
            self.assertEqual(result["sample_count"], 2)
            self.assertEqual(result["haplotype_probability_cells"], 8)

    def test_output_probability_and_axis_drift_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            args = fixture(Path(directory))
            target = subject.scan_vcf(args.target_vcf, "22")
            bad_probability = Path(directory) / "bad-probability.anc.vcf.gz"
            write_ancestry_vcf(bad_probability, args.target_vcf, "0.1,0.1,0.1")
            with self.assertRaisesRegex(subject.FlareContractError, "mass"):
                subject.audit_ancestry_vcf(bad_probability, target,
                                           ["AFR", "EUR", "NAM"])
            wrong_target = dict(target)
            wrong_target["samples"] = list(reversed(target["samples"]))
            valid = Path(directory) / "valid.anc.vcf.gz"
            write_ancestry_vcf(valid, args.target_vcf)
            with self.assertRaisesRegex(subject.FlareContractError, "sample axes"):
                subject.audit_ancestry_vcf(valid, wrong_target,
                                           ["AFR", "EUR", "NAM"])

    def test_hash_and_phase_drift_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            args = fixture(Path(directory))
            args.target_tbi.write_bytes(b"changed")
            with self.assertRaisesRegex(subject.FlareContractError, "SHA-256 mismatch"):
                subject.run(args)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            args = fixture(base)
            write_vcf(args.target_vcf, ["T1", "T2"], [["0/1", "1|0"], ["1|1", "0|0"]])
            contract = json.loads(args.contract.read_text())
            contract["expected_sha256"]["target_vcf"] = subject.sha256_file(args.target_vcf)
            args.contract.write_text(json.dumps(contract), encoding="utf-8")
            with self.assertRaisesRegex(subject.FlareContractError, "unphased GT"):
                subject.run(args)


if __name__ == "__main__":
    unittest.main()
