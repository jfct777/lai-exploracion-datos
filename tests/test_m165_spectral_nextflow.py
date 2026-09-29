"""Saved-graph spectral figure wiring; only synthetic fixtures, never DNABR.

RUN_M165_SPECTRAL_NEXTFLOW=1 python3 -m unittest tests.test_m165_spectral_nextflow -v
Optionally set M165_SPECTRAL_TEST_IMAGE to an already available Docker image.
The integration renderer is a stub: no embedding, UMAP or community fitting.
"""
from __future__ import annotations

import ast
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import tempfile
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "modules/16_5_SPECTRAL_FIGURES.nf"
CORE = ROOT / "bin/ibd_community_enhanced.py"
CONFIG_IDS = ["L250000_G50000_N20_T250000_U0", "L250000_G50000_N20_T500000_U0",
              "L500000_G50000_N20_T500000_U0"]
SETTINGS_KEYS = {"prefix", "dpi", "seed", "n_spectral", "n_neighbors", "min_dist",
                 "resolution", "resolutions", "combined_pdf", "config_ids", "expected_samples",
                 "width_inches", "height_inches", "inline_metadata_legend", "legend_wrap_chars"}
RESOLUTIONS = [.5, .8, 1., 1.2, 1.5, 2., 3.]
# AST fingerprints of the source frozen for the completed six-graph sweep.
# They contain no run files, sample identifiers, or private data.
FROZEN_SPECTRAL_AST = "5718b496581dbc24357e8fda721eb2412b5174e77ebbbb10677931d328e32f92"
FROZEN_STATIC_AST = "d40e8977796dc49afbe94ff8e0f0d1cc126c6ace6b5af2292183ff8155398270"


def function(name):
    return next(node for node in ast.parse(CORE.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == name)


def ast_hash(node):
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


class DefaultCaptionPath(ast.NodeTransformer):
    """Resolve only explicit optional guards to their inactive legacy path."""

    def visit_If(self, node):
        if isinstance(node.test, ast.Name) and node.test.id == 'inline_metadata_legend':
            retained=[]
            for statement in node.orelse:
                item=self.visit(statement)
                retained.extend(item if isinstance(item,list) else [item])
            return [item for item in retained if item is not None]
        def caption_name(item):
            return isinstance(item, ast.Name) and item.id in {"figure_title", "figure_note"}

        positive_caption_guard = caption_name(node.test) or (
            isinstance(node.test, ast.BoolOp) and isinstance(node.test.op, ast.Or)
            and all(caption_name(item) for item in node.test.values))
        if positive_caption_guard:
            return [self.visit(statement) for statement in node.orelse]
        return self.generic_visit(node)


class SpectralWiringTests(unittest.TestCase):
    def test_dedicated_entry_only_renders_saved_results(self):
        source = (ROOT / "main.nf").read_text()
        entry = source.split("workflow M165_SPECTRAL_FIGURES {", 1)[1].split("\n}", 1)[0]
        self.assertIn("RENDER_M165_SPECTRAL_FIGURES(results, scripts, metadata, coordinates, settings)", entry)
        self.assertIn("must not already exist", entry)
        self.assertIn("must be a directory", entry)
        self.assertIn("followupInput(params.m165_spectral_metadata_file", entry)
        self.assertIn("followupInput(params.m165_spectral_coordinates_source_dir", entry)
        self.assertIn("Saved coordinates require their authenticated manifest SHA256", entry)
        self.assertIn("Coordinate manifest SHA256 requires a saved-coordinate directory", entry)
        for script in ("m165_spectral_figures.py", "m165_sweep_figures.py", "ibd_community_enhanced.py"):
            self.assertIn(script, entry)
        for forbidden in ("RUN_M165_CHR22_PLAN(", "PRESENT_RARE_SEGMENT_CANDIDATE(",
                          "run_leiden", "symnmf", "RARE_ALLELE_SHARING"):
            self.assertNotIn(forbidden, entry)

    def test_task_copies_protects_inputs_and_does_not_overwrite(self):
        module = MODULE.read_text()
        for required in ("stageInMode 'copy'", "stageAs: 'input_results'", "path scripts",
                         "cache false",
                         "stageAs: 'metadata.tsv'", "path coordinates, stageAs: 'saved_coordinates'",
                         "val settings", "chmod -R a-w",
                         "mode: 'copy', overwrite: false", "path 'figures', emit: figures",
                         "cpus 2", "memory '8 GB'", "maxForks 1", "OMP_NUM_THREADS=1",
                         "NUMBA_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1"):
            self.assertIn(required, module)
        self.assertIn("--parameters-json-text ${settingsText}", module)
        self.assertIn("groovy.json.JsonOutput.toJson(settings)", module)
        self.assertIn("--coordinates-source-dir saved_coordinates", module)
        self.assertIn("--coordinates-source-manifest-sha256 ${quote(params.m165_spectral_coordinates_source_manifest_sha256)}", module)
        self.assertNotRegex(module, r"(?m)^\s*container\s+")
        self.assertNotRegex(module, r"python3\s+(ibd_community_enhanced|m165_chr22_sweep)\.py")

    def test_settings_are_explicit_and_literal_visual_defaults_preserved(self):
        source = (ROOT / "main.nf").read_text()
        entry = source.split("workflow M165_SPECTRAL_FIGURES {", 1)[1].split("\n}", 1)[0]
        block = re.search(r"def keys = \[(.*?)\]", entry, re.S).group(1)
        self.assertEqual(set(re.findall(r"'([^']+)'", block)), SETTINGS_KEYS)
        self.assertIn('params["m165_spectral_${key}"]', entry)
        config = (ROOT / "nextflow.config").read_text()
        expected = {"seed": "42", "n_spectral": "15", "n_neighbors": "15", "min_dist": "0.3",
                    "resolution": "1.0", "expected_samples": "2619"}
        for key, value in expected.items():
            self.assertRegex(config, rf"(?m)^\s*m165_spectral_{key}\s*=\s*{re.escape(value)}\s*$")
        self.assertRegex(config, r"(?m)^\s*m165_spectral_resolutions\s*=\s*null(?:\s*//.*)?$")
        self.assertRegex(config, r"(?m)^\s*m165_spectral_combined_pdf\s*=\s*true(?:\s*//.*)?$")
        configs = re.search(r"m165_spectral_config_ids\s*=\s*(\[[^\n]+\])", config).group(1)
        self.assertEqual(ast.literal_eval(configs), CONFIG_IDS)
        for key in SETTINGS_KEYS | {"results_dir", "output_dir", "metadata_file",
                                    "metadata_sample_column", "metadata_color_column",
                                    "coordinates_source_dir", "coordinates_source_manifest_sha256"}:
            self.assertIn("m165_spectral_" + key, config)

    def test_spectral_function_is_ast_identical_to_frozen_sweep_source(self):
        self.assertEqual(ast_hash(function("_spectral_embedding")), FROZEN_SPECTRAL_AST)

    def test_static_optional_captions_default_to_none_and_preserve_legacy_path(self):
        node = copy.deepcopy(function("_plot_network_static"))
        defaults = dict(zip((arg.arg for arg in node.args.kwonlyargs), node.args.kw_defaults))
        for name in ("figure_title", "figure_note"):
            self.assertIsInstance(defaults[name], ast.Constant)
            self.assertIsNone(defaults[name].value)
        self.assertIs(defaults['inline_metadata_legend'].value,False)
        self.assertEqual(defaults['legend_wrap_chars'].value,56)
        retained = [(arg, default) for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults)
                    if arg.arg not in {"figure_title", "figure_note", "inline_metadata_legend", "legend_wrap_chars"}]
        node.args.kwonlyargs = [arg for arg, _ in retained]
        node.args.kw_defaults = [default for _, default in retained]
        node = DefaultCaptionPath().visit(node)
        self.assertEqual(ast_hash(node), FROZEN_STATIC_AST,
                         "Changes beyond inactive optional captions alter the legacy rendering path")


def nextflow_environment():
    return {**os.environ, "NXF_OFFLINE": "true", "NXF_DISABLE_CHECK_LATEST": "true",
            "NXF_SYNTAX_PARSER": "v1", "NXF_OPTS": "-Xms64m -Xmx512m"}


def write_synthetic_coordinates(path, points):
    """Tiny valid float64 NPZ, generated with stdlib so host NumPy is unnecessary."""
    header = repr({"descr": "<f8", "fortran_order": False, "shape": (len(points), 2)}).encode("ascii")
    header += b" " * ((64 - (10 + len(header) + 1) % 64) % 64) + b"\n"
    values = [value for point in points for value in point]
    payload = b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header
    payload += struct.pack(f"<{len(values)}d", *values)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(zipfile.ZipInfo("coordinates.npy", date_time=(2026, 1, 1, 0, 0, 0)), payload)


@unittest.skipUnless(os.environ.get("RUN_M165_SPECTRAL_NEXTFLOW") == "1" and shutil.which("nextflow"),
                     "opt-in synthetic local Nextflow wiring test")
class SyntheticSpectralProcessTests(unittest.TestCase):
    def test_main_entry_rejects_missing_or_existing_output_before_submitting(self):
        with tempfile.TemporaryDirectory(prefix="m165-spectral-guards-") as folder:
            base = Path(folder)
            source, existing = base / "source", base / "existing"
            source.mkdir(); existing.mkdir()
            command = ["nextflow", "-log", str(base / "nextflow.log"), "run", str(ROOT / "main.nf"),
                       "-entry", "M165_SPECTRAL_FIGURES", "-work-dir", str(base / "work"),
                       "-ansi-log", "false", "--m165_spectral_results_dir", str(source)]
            for extra, message in (
                    ([], "Set --m165_spectral_output_dir"),
                    (["--m165_spectral_output_dir", str(existing)], "must not already exist"),
                    (["--m165_spectral_output_dir", str(base / "new-output"),
                      "--m165_spectral_coordinates_source_dir", str(source)],
                     "Saved coordinates require their authenticated manifest SHA256"),
                    (["--m165_spectral_output_dir", str(base / "new-output"),
                      "--m165_spectral_coordinates_source_manifest_sha256", "a" * 64],
                     "Coordinate manifest SHA256 requires a saved-coordinate directory")):
                with self.subTest(message=message):
                    result = subprocess.run(command + extra, cwd=base, env=nextflow_environment(),
                                            capture_output=True, text=True, timeout=90)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(message, result.stdout + result.stderr)
                    self.assertNotIn("Submitted process", result.stdout + result.stderr)

    def test_three_scripts_optional_inputs_and_coordinate_cache_without_fitting(self):
        for with_metadata in (False, True):
            with self.subTest(metadata=with_metadata), tempfile.TemporaryDirectory(prefix="m165-spectral-nf-") as folder:
                base = Path(folder)
                source = base / "source_results"
                source.mkdir()
                for config_id in CONFIG_IDS:
                    directory = source / config_id
                    directory.mkdir()
                    (directory / "manifest.json").write_text('{"synthetic": true}\n')
                external = base / "external.tsv"
                external.write_text("synthetic\t1\n")
                (source / "linked.tsv").symlink_to(external)
                metadata = base / "original_metadata.tsv"
                metadata.write_text("IDtest\tgroup label's value\nfixture1\tgroupA\nfixture2\tgroupB\n")
                coordinates = base / "original_coordinates"
                coordinates.mkdir()
                coordinate_manifest = coordinates / "manifest.json"
                coordinate_manifest.write_text('{"synthetic_coordinates": true}\n')
                (coordinates / "private").mkdir()
                coordinate_data = coordinates / "private/synthetic.coordinates.npz"
                write_synthetic_coordinates(coordinate_data, [(1., 2.)])
                coordinate_sha = hashlib.sha256(coordinate_manifest.read_bytes()).hexdigest()
                originals = {path: (path.read_bytes(), path.stat().st_mode)
                             for path in [*source.rglob("*"), *coordinates.rglob("*"), metadata, external]
                             if path.is_file()}
                settings = dict(prefix="WITH_META" if with_metadata else "WITHOUT_META", dpi=72,
                                seed=42, n_spectral=15, n_neighbors=15, min_dist=.3, resolution=1.,
                                resolutions=RESOLUTIONS if with_metadata else None,
                                combined_pdf=not with_metadata,
                                config_ids=CONFIG_IDS, expected_samples=5, width_inches=12., height_inches=8.)
                renderer = base / "m165_spectral_figures.py"
                renderer.write_text('''import argparse, hashlib, json
from pathlib import Path
import m165_sweep_figures, ibd_community_enhanced
p = argparse.ArgumentParser()
for flag in ('results-dir', 'output-dir', 'parameters-json-text'):
    p.add_argument('--' + flag, required=True)
for flag in ('metadata-file', 'metadata-sample-column', 'metadata-color-column',
             'coordinates-source-dir', 'coordinates-source-manifest-sha256'):
    p.add_argument('--' + flag)
a = p.parse_args()
settings = json.loads(a.parameters_json_text)
assert m165_sweep_figures.STUB == 'sweep' and ibd_community_enhanced.STUB == 'core'
assert len(settings) == 13
assert settings['seed'] == 42 and settings['n_spectral'] == settings['n_neighbors'] == 15
assert settings['min_dist'] == .3 and settings['resolution'] == 1.
assert settings['dpi'] == 72 and settings['expected_samples'] == 5
assert settings['width_inches'] == 12. and settings['height_inches'] == 8.
source = Path(a.results_dir)
assert sorted(x.name for x in source.glob('L*')) == sorted(settings['config_ids'])
assert len(list(source.glob('L*/manifest.json'))) == 3
entries = list(source.rglob('*'))
assert not source.is_symlink() and all(not x.is_symlink() for x in entries)
assert not source.stat().st_mode & 0o222
assert all(not x.stat().st_mode & 0o222 for x in entries)
assert (source / 'linked.tsv').read_text() == 'synthetic\\t1\\n'
with_metadata = settings['prefix'] == 'WITH_META'
assert bool(a.metadata_file) == with_metadata
assert settings['resolutions'] == ([.5, .8, 1., 1.2, 1.5, 2., 3.] if with_metadata else None)
assert settings['combined_pdf'] == (not with_metadata)
assert bool(a.coordinates_source_dir) == with_metadata
coordinate_file_sha256 = None
if a.coordinates_source_dir:
    coordinates = Path(a.coordinates_source_dir)
    assert coordinates.name == 'saved_coordinates' and not coordinates.is_symlink()
    assert all(not x.is_symlink() for x in coordinates.rglob('*'))
    assert hashlib.sha256((coordinates / 'manifest.json').read_bytes()).hexdigest() == a.coordinates_source_manifest_sha256
    coordinate_file_sha256 = hashlib.sha256((coordinates / 'private/synthetic.coordinates.npz').read_bytes()).hexdigest()
else:
    assert a.coordinates_source_manifest_sha256 is None
if with_metadata:
    assert a.metadata_sample_column == 'IDtest'
    assert a.metadata_color_column == "group label's value"
    assert not Path(a.metadata_file).is_symlink()
    assert Path(a.metadata_file).read_text().startswith("IDtest\\tgroup label's value\\n")
else:
    assert a.metadata_sample_column is None and a.metadata_color_column is None
out = Path(a.output_dir)
out.mkdir(exist_ok=False)
(out / 'wiring_receipt.json').write_text(json.dumps({'synthetic': True, 'settings': settings,
    'three_scripts': True, 'staged_readonly': True, 'metadata_supplied': with_metadata,
    'coordinates_supplied': bool(a.coordinates_source_dir), 'coordinate_file_sha256': coordinate_file_sha256,
    'fit_embedding_or_render_run': False}))
''')
                for name, marker in (("m165_sweep_figures.py", "sweep"), ("ibd_community_enhanced.py", "core")):
                    (base / name).write_text(f"STUB = {marker!r}\n")
                scripts = ", ".join(f"file('{base / name}')" for name in
                                    ("m165_spectral_figures.py", "m165_sweep_figures.py", "ibd_community_enhanced.py"))
                metadata_input = f"file('{metadata}')" if with_metadata else "[]"
                coordinate_input = f"file('{coordinates}')" if with_metadata else "[]"
                workflow = base / "test.nf"
                workflow.write_text(f'''nextflow.enable.dsl=2
include {{ RENDER_M165_SPECTRAL_FIGURES }} from '{MODULE}'
workflow {{
    def settings = new groovy.json.JsonSlurper().parseText('{json.dumps(settings)}')
    RENDER_M165_SPECTRAL_FIGURES(file('{source}'), [{scripts}], {metadata_input}, {coordinate_input}, settings)
}}
''')
                image = os.environ.get("M165_SPECTRAL_TEST_IMAGE")
                docker_config = (f"process.container = '{image}'\ndocker.enabled = true\n"
                                 f"docker.runOptions = '--network none --user {os.getuid()}:{os.getgid()}'\n"
                                 if image else "docker.enabled = false\n")
                config = base / "test.config"
                config.write_text(f'''process.executor = 'local'
singularity.enabled = false
{docker_config}
params.m165_spectral_output_dir = '{base}/published'
params.m165_spectral_metadata_sample_column = 'IDtest'
params.m165_spectral_metadata_color_column = "group label's value"
params.m165_spectral_coordinates_source_manifest_sha256 = '{coordinate_sha}'
''')
                trace = base / "trace.tsv"
                command = [
                    "nextflow", "-log", str(base / "nextflow.log"), "-C", str(config), "run", str(workflow),
                    "-work-dir", str(base / "work"), "-ansi-log", "false"]
                result = subprocess.run(command + ["-with-trace", str(trace)], cwd=base,
                                        capture_output=True, text=True, timeout=150, env=nextflow_environment())
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                receipt = json.loads((base / "published/figures/wiring_receipt.json").read_text())
                self.assertEqual(receipt["settings"], settings)
                self.assertEqual(receipt["metadata_supplied"], with_metadata)
                self.assertEqual(receipt["coordinates_supplied"], with_metadata)
                self.assertEqual(receipt["coordinate_file_sha256"],
                                 hashlib.sha256(coordinate_data.read_bytes()).hexdigest() if with_metadata else None)
                self.assertTrue(receipt["staged_readonly"] and receipt["three_scripts"])
                self.assertFalse(receipt["fit_embedding_or_render_run"])
                with trace.open() as handle:
                    tasks = list(csv.DictReader(handle, delimiter="\t"))
                self.assertEqual(len(tasks), 1)
                self.assertEqual((tasks[0]["status"], tasks[0]["exit"]), ("COMPLETED", "0"))
                self.assertIn("RENDER_M165_SPECTRAL_FIGURES", tasks[0]["name"])
                for path, (content, mode) in originals.items():
                    self.assertEqual(path.read_bytes(), content)
                    self.assertEqual(path.stat().st_mode, mode)
                self.assertTrue((source / "linked.tsv").is_symlink())
                if with_metadata:
                    # Directory path inputs did not invalidate nested payload changes,
                    # even under cache 'deep'. Intentionally revalidate on every resume.
                    # All NPZ files here are tiny synthetic fixtures, never real geometry.
                    for index, changed in enumerate((False, True), start=2):
                        if changed:
                            write_synthetic_coordinates(coordinate_data, [(10., 20.), (30., 40.)])
                        resumed_trace = base / f"trace-resume-{changed}.tsv"
                        resumed = subprocess.run(command + ["-resume", "-with-trace", str(resumed_trace)],
                            cwd=base, capture_output=True, text=True, timeout=150, env=nextflow_environment())
                        self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
                        with resumed_trace.open() as handle:
                            resumed_tasks = list(csv.DictReader(handle, delimiter="\t"))
                        self.assertEqual(len(resumed_tasks), 1)
                        self.assertEqual((resumed_tasks[0]["status"], resumed_tasks[0]["exit"]),
                                         ("COMPLETED", "0"))
                        receipts = [json.loads(path.read_text()) for path in
                                    (base / "work").rglob("wiring_receipt.json")]
                        self.assertEqual(len(receipts), index, "Resume must produce a fresh task output")
                        current_hash = hashlib.sha256(coordinate_data.read_bytes()).hexdigest()
                        self.assertEqual(sum(item["coordinate_file_sha256"] == current_hash for item in receipts),
                                         1 if changed else index)
                    # Immutable publication preserves the original receipt across resume.
                    self.assertEqual(json.loads((base / "published/figures/wiring_receipt.json").read_text()), receipt)


if __name__ == "__main__":
    unittest.main()
