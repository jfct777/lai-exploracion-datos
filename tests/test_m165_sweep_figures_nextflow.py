"""Saved-results figure wiring; optional synthetic Nextflow test, never DNABR.

RUN_M165_FIGURES_NEXTFLOW=1 python3 -m unittest tests.test_m165_sweep_figures_nextflow -v
Set M165_FIGURES_TEST_IMAGE to an already available Docker image to test staging
inside a network-disabled container. No plotting or community fitting is done.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "modules/16_5_SWEEP_FIGURES.nf"


class FigureWiringTests(unittest.TestCase):
    def test_dedicated_entry_only_renders_saved_results(self):
        text = (ROOT / "main.nf").read_text()
        entry = text.split("workflow M165_SWEEP_FIGURES {", 1)[1].split("\n}", 1)[0]
        self.assertIn("RENDER_M165_SWEEP_FIGURES(results,", entry)
        self.assertIn("params.m165_figures_results_dir", entry)
        self.assertIn("must not already exist", entry)
        self.assertNotIn("RUN_M165_CHR22_PLAN(", entry)
        self.assertNotIn("PRESENT_RARE_SEGMENT_CANDIDATE(", entry)

    def test_copy_staging_and_nonoverwriting_publication(self):
        module = MODULE.read_text()
        for contract in ("stageInMode 'copy'", "stageAs: 'input_results'",
                         "chmod -R a-w", "mode: 'copy', overwrite: false",
                         "path 'figures', emit: figures", "cpus 2", "memory '4 GB'"):
            self.assertIn(contract, module)
        self.assertNotRegex(module, r"(?m)^\s*container\s+")  # Runtime configuration owns the image.

    def test_all_display_parameters_are_explicit(self):
        config = (ROOT / "nextflow.config").read_text()
        for name in ("results_dir", "output_dir", "prefix", "dpi", "layout_seed", "layout_iterations",
                     "network_config_ids", "network_resolution"):
            self.assertIn("m165_figures_" + name, config)
        module = MODULE.read_text()
        for flag in ("--results-dir", "--output-dir", "--prefix", "--dpi",
                     "--layout-seed", "--layout-iterations", "--network-config-ids", "--network-resolution"):
            self.assertIn(flag, module)
        self.assertNotIn("--all-networks", module)


@unittest.skipUnless(os.environ.get("RUN_M165_FIGURES_NEXTFLOW") == "1" and shutil.which("nextflow"),
                     "opt-in synthetic local Nextflow wiring test")
class SyntheticFigureProcessTests(unittest.TestCase):
    def test_main_entry_requires_new_output_before_submitting_any_process(self):
        with tempfile.TemporaryDirectory(prefix="m165-figures-guards-") as folder:
            base = Path(folder)
            source = base / "source"
            existing = base / "existing"
            source.mkdir()
            existing.mkdir()
            env = {**os.environ, "NXF_OFFLINE": "true", "NXF_DISABLE_CHECK_LATEST": "true",
                   "NXF_SYNTAX_PARSER": "v1", "NXF_OPTS": "-Xms64m -Xmx512m"}
            command = ["nextflow", "-log", str(base / "nextflow.log"), "run", str(ROOT / "main.nf"),
                       "-entry", "M165_SWEEP_FIGURES", "-work-dir", str(base / "work"),
                       "-ansi-log", "false", "--m165_figures_results_dir", str(source)]
            for extra, message in (([], "Set --m165_figures_output_dir"),
                                   (["--m165_figures_output_dir", str(existing)], "must not already exist")):
                result = subprocess.run(command + extra, cwd=base, env=env, capture_output=True,
                                        text=True, timeout=90)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stdout + result.stderr)
                self.assertNotIn("Submitted process", result.stdout + result.stderr)

    def test_complete_directory_is_copied_readonly_with_no_external_symlinks(self):
        with tempfile.TemporaryDirectory(prefix="m165-figures-nf-") as folder:
            base = Path(folder)
            source = base / "source_results"
            source.mkdir()
            configs = [f"L{length}_G50000_N20_T{threshold}_U0"
                       for length, thresholds in ((250000, (250000, 500000, 1000000)),
                                                  (500000, (500000, 750000, 1000000)))
                       for threshold in thresholds]
            for name in configs:
                target = source / name
                target.mkdir()
                (target / "manifest.json").write_text('{"synthetic": true}\n')
            (source / "aggregate.tsv").write_text("config\n" + "\n".join(configs) + "\n")
            external = base / "external_summary.tsv"
            external.write_text("synthetic\t1\n")
            (source / "linked_summary.tsv").symlink_to(external)
            originals = {p: (p.read_bytes(), p.stat().st_mode) for p in source.rglob("*") if p.is_file()}
            renderer = base / "synthetic_renderer.py"
            renderer.write_text('''import argparse, json
from pathlib import Path
p = argparse.ArgumentParser()
for flag in ('results-dir', 'output-dir', 'prefix', 'dpi', 'layout-seed', 'layout-iterations',
             'network-config-ids', 'network-resolution'):
    p.add_argument('--' + flag, required=True)
a = p.parse_args()
source = Path(a.results_dir)
entries = list(source.rglob('*'))
assert len(list(source.glob('L*/manifest.json'))) == 6
assert (source / 'aggregate.tsv').is_file()
assert (source / 'linked_summary.tsv').read_text() == 'synthetic\\t1\\n'
assert not source.is_symlink() and all(not x.is_symlink() for x in entries)
assert not source.stat().st_mode & 0o222
assert all(not x.stat().st_mode & 0o222 for x in entries)
assert (a.prefix, a.dpi, a.layout_seed, a.layout_iterations) == ('SYNTHETIC', '72', '0', '3')
assert a.network_config_ids.split(',') == ['L250000_G50000_N20_T250000_U0',
    'L250000_G50000_N20_T500000_U0', 'L500000_G50000_N20_T500000_U0']
assert a.network_resolution == '1.0'
out = Path(a.output_dir)
out.mkdir(exist_ok=False)
(out / 'wiring_receipt.json').write_text(json.dumps({'synthetic': True, 'n_graphs': 6,
    'staged_readonly': True, 'external_symlinks': False, 'fit_or_render_run': False}))
''')
            workflow = base / "test.nf"
            workflow.write_text(f'''nextflow.enable.dsl=2
include {{ RENDER_M165_SWEEP_FIGURES }} from '{MODULE}'
workflow {{
    RENDER_M165_SWEEP_FIGURES(file('{source}'), file('{renderer}'))
}}
''')
            config = base / "test.config"
            image = os.environ.get("M165_FIGURES_TEST_IMAGE")
            docker_config = (f"process.container = '{image}'\ndocker.enabled = true\n"
                             f"docker.runOptions = '--network none --user {os.getuid()}:{os.getgid()}'\n"
                             if image else "docker.enabled = false\n")
            config.write_text(f'''process.executor = 'local'
singularity.enabled = false
{docker_config}
params.m165_figures_output_dir = '{base}/published'
params.m165_figures_prefix = 'SYNTHETIC'
params.m165_figures_dpi = 72
params.m165_figures_layout_seed = 0
params.m165_figures_layout_iterations = 3
params.m165_figures_network_config_ids = 'L250000_G50000_N20_T250000_U0,L250000_G50000_N20_T500000_U0,L500000_G50000_N20_T500000_U0'
params.m165_figures_network_resolution = 1.0
''')
            result = subprocess.run([
                "nextflow", "-log", str(base / "nextflow.log"), "-C", str(config),
                "run", str(workflow), "-work-dir", str(base / "work"), "-ansi-log", "false",
            ], cwd=base, capture_output=True, text=True, timeout=120,
                env={**os.environ, "NXF_OFFLINE": "true", "NXF_DISABLE_CHECK_LATEST": "true",
                     "NXF_SYNTAX_PARSER": "v1", "NXF_OPTS": "-Xms64m -Xmx512m"})
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            receipt = json.loads((base / "published/figures/wiring_receipt.json").read_text())
            self.assertEqual(receipt["n_graphs"], 6)
            self.assertTrue(receipt["staged_readonly"])
            for path, (content, mode) in originals.items():
                self.assertEqual(path.read_bytes(), content)
                self.assertEqual(path.stat().st_mode, mode)
            self.assertTrue((source / "linked_summary.tsv").is_symlink())


if __name__ == "__main__":
    unittest.main()
