"""Filesystem contracts without cloud writes or genomic computations."""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("scratch_guard", ROOT / "bin/preprocess_storage_guard.py")
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)


def mount_line(number, path, filesystem="ext4", options="rw"):
    encoded = str(path).replace("\\", "\\134").replace(" ", "\\040")
    dev = Path(tempfile.gettempdir()).stat().st_dev
    return f"{number} 1 {os.major(dev)}:{os.minor(dev)} / {encoded} {options} - {filesystem} /dev/sda {options}\n"


class StorageGuardTests(unittest.TestCase):
    def test_bind_mount_under_fuse_is_local(self):
        info = mount_line(1, "/") + mount_line(2, "/bucket", "fuse.gcsfuse") + mount_line(3, "/bucket/work")
        self.assertEqual(guard.find_mount(Path("/bucket/work/chr1"), info)["filesystem"], "ext4")
        self.assertEqual(guard.find_mount(Path("/bucket/input"), info)["filesystem"], "fuse.gcsfuse")

    def test_component_match_not_text_prefix(self):
        info = mount_line(1, "/") + mount_line(2, "/work", "fuse.gcsfuse")
        self.assertEqual(guard.find_mount(Path("/worker"), info)["filesystem"], "ext4")

    def test_escaped_mountpoint(self):
        info = mount_line(1, "/") + mount_line(2, "/disk with spaces")
        self.assertEqual(guard.find_mount(Path("/disk with spaces/task"), info)["mountpoint"], "/disk with spaces")

    def test_success_reports_capacity_without_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = guard.inspect_storage(Path(tmp), minimum_free_gib=0, mountinfo=mount_line(1, "/"))
            self.assertEqual(result["status"], "LOCAL_SCRATCH_VERIFIED")
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_remote_ram_and_unknown_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            for fs in ("fuse.gcsfuse", "fuse", "nfs", "nfs4", "cifs", "tmpfs", "overlay", "unknown"):
                with self.subTest(fs=fs), self.assertRaisesRegex(ValueError, "local disk"):
                    guard.inspect_storage(Path(tmp), minimum_free_gib=0, mountinfo=mount_line(1, "/", fs))

    def test_readonly_and_capacity_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "read-only"):
                guard.inspect_storage(Path(tmp), mountinfo=mount_line(1, "/", options="ro"))
            with self.assertRaisesRegex(ValueError, "Insufficient"):
                guard.inspect_storage(Path(tmp), required_free_bytes=10**30, mountinfo=mount_line(1, "/"))

    def test_no_write_access_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(guard.os, "access", return_value=False):
            with self.assertRaisesRegex(ValueError, "not writable"):
                guard.inspect_storage(Path(tmp), minimum_free_gib=0, mountinfo=mount_line(1, "/"))

    def test_invalid_size_and_path_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            for value in (-1, float("nan"), float("inf"), True):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    guard.inspect_storage(Path(tmp), minimum_free_gib=value)
            for value in (-1, 2.5, True):
                with self.assertRaises(ValueError):
                    guard.inspect_storage(Path(tmp), required_free_bytes=value)
            with self.assertRaises(ValueError):
                guard.inspect_storage(Path("relative"))

    def test_symlink_resolves_before_mount_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "remote").mkdir()
            (root / "link").symlink_to(root / "remote", target_is_directory=True)
            info = mount_line(1, "/") + mount_line(2, root / "remote", "fuse.gcsfuse")
            with self.assertRaisesRegex(ValueError, "fuse.gcsfuse"):
                guard.inspect_storage(root / "link", minimum_free_gib=0, mountinfo=info)

    def test_overlay_only_explicit_fixture_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = guard.inspect_storage(Path(tmp), minimum_free_gib=0, allow_overlay=True,
                                           mountinfo=mount_line(1, "/", "overlay"))
            self.assertEqual(result["filesystem"], "overlay")

    def test_hidden_or_remounted_device_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            info = mount_line(1, "/")
            fields = info.split()
            fields[2] = '999:999'
            with self.assertRaisesRegex(ValueError, "hidden or changed"):
                guard.inspect_storage(Path(tmp), minimum_free_gib=0, mountinfo=' '.join(fields))


if __name__ == "__main__":
    unittest.main()
