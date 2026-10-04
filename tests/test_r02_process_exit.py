"""Linux lifecycle regressions; only short-lived children created by this test.

No VM, Docker, genome data, root privileges or existing process is touched.
"""
from contextlib import ExitStack
import hashlib
import importlib.util
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    'boundary_process_exit', Path(__file__).resolve().parents[1] / 'bin/r02_optimization_boundary.py')
b = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(b)
EMPTY = hashlib.sha256(b'').hexdigest()


@unittest.skipUnless(sys.platform == 'linux' and hasattr(os, 'pidfd_open')
                     and hasattr(signal, 'pidfd_send_signal'), 'Linux pidfds required')
class ProcessExitTests(unittest.TestCase):
    def child(self, after="os._exit(0)"):
        code = "import os; print('ready', flush=True); os.read(0, 1); " + after
        child = subprocess.Popen([sys.executable, '-c', code], stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.addCleanup(self.cleanup_child, child)
        self.assertTrue(select.select([child.stdout], [], [], 5)[0], 'Child failed to become ready')
        self.assertEqual(child.stdout.readline(), b'ready\n')
        return child, b.identity(b.process_info(child.pid))

    @staticmethod
    def cleanup_child(child):
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)
        for stream in (child.stdin, child.stdout, child.stderr):
            if stream is not None:
                stream.close()

    @staticmethod
    def release(child):
        child.stdin.write(b'x')
        child.stdin.flush()

    def test_actual_short_lived_children_cross_exit_without_command_failure(self):
        # Before this fix, the real exit_mm-before-Z window reproduces the
        # failure here. Do not require observing that scheduling-dependent
        # window on every kernel; deterministic coverage is below as well.
        for _ in range(20):
            child, spec = self.child()
            self.release(child)
            deadline = time.monotonic() + 5
            while b.authenticated(spec) is not None:
                self.assertLess(time.monotonic(), deadline)
            self.assertEqual(child.wait(timeout=5), 0)
            self.assertIsNone(b.authenticated(spec))

    def test_real_zombie_has_empty_command_and_is_not_signalled(self):
        child, spec = self.child()
        fd = os.pidfd_open(child.pid)
        self.addCleanup(os.close, fd)
        self.release(child)
        self.assertTrue(b.pidfd_exited(fd, 5000))
        current = b.process_info(child.pid)
        self.assertEqual(current['state'], 'Z')
        self.assertEqual(current['cmdline_sha256'], EMPTY)
        self.assertIsNone(b.authenticated(spec))
        with patch.object(signal, 'pidfd_send_signal') as send:
            self.assertFalse(b.send(spec, signal.SIGTERM))
            send.assert_not_called()

    def test_empty_running_snapshot_requires_real_kernel_exit_notification(self):
        child, spec = self.child()
        live = b.process_info(child.pid)
        fd = os.pidfd_open(child.pid)
        self.addCleanup(os.close, fd)
        self.release(child)
        self.assertTrue(b.pidfd_exited(fd, 5000))
        # Reproduce the non-atomic /proc snapshot deterministically, while
        # leaving pidfd_open/poll and the final /proc read entirely real.
        exiting = dict(live, state='R', cmdline_sha256=EMPTY)
        zombie = b.process_info(child.pid)
        with patch.object(b, 'process_info', side_effect=[exiting, exiting, zombie]), \
             patch.object(b, 'pidfd_exited', wraps=b.pidfd_exited) as poll:
            self.assertIsNone(b.authenticated(spec))
            self.assertEqual(poll.call_args.kwargs, {'timeout_ms': 1000})

    def test_empty_command_on_still_live_process_fails_closed(self):
        child, spec = self.child()
        empty = dict(b.process_info(child.pid), cmdline_sha256=EMPTY)
        with patch.object(b, 'process_info', return_value=empty), \
             patch.object(signal, 'pidfd_send_signal') as send:
            with self.assertRaisesRegex(ValueError, 'without confirmed process exit'):
                b.send(spec, signal.SIGTERM)
            send.assert_not_called()
        self.assertIsNone(child.poll())

    def test_real_exec_changed_identity_is_rejected_without_signalling(self):
        child, spec = self.child("os.execv('/bin/sleep', ['sleep', '30'])")
        self.release(child)
        deadline = time.monotonic() + 5
        while True:
            current = b.process_info(child.pid)
            if current['cmdline_sha256'] not in (spec['cmdline_sha256'], EMPTY):
                break
            self.assertLess(time.monotonic(), deadline)
        self.assertEqual(current['start_ticks'], spec['start_ticks'])
        with patch.object(signal, 'pidfd_send_signal') as send:
            for operation in (lambda: b.authenticated(spec), lambda: b.send(spec, signal.SIGTERM)):
                with self.assertRaisesRegex(ValueError, 'Process command changed'):
                    operation()
            send.assert_not_called()
        self.assertIsNone(child.poll())

    def test_empty_then_nonempty_changed_command_is_not_adopted(self):
        child, spec = self.child()
        current = b.process_info(child.pid)
        empty = dict(current, cmdline_sha256=EMPTY)
        changed = dict(current, cmdline_sha256='a' * 64)
        with patch.object(b, 'process_info', side_effect=[empty, changed]), \
             patch.object(b, 'pidfd_exited') as poll:
            with self.assertRaisesRegex(ValueError, 'Process command changed'):
                b.authenticated(spec)
            poll.assert_not_called()

    def test_reused_pid_even_as_zombie_is_rejected_before_pidfd_open(self):
        child, spec = self.child()
        reused = dict(b.process_info(child.pid), start_ticks=spec['start_ticks'] + 1, state='Z')
        with patch.object(b, 'process_info', return_value=reused), patch.object(os, 'pidfd_open') as opening:
            with self.assertRaisesRegex(ValueError, 'PID reused'):
                b.send(spec, signal.SIGTERM)
            opening.assert_not_called()

    def test_reuse_during_empty_command_pidfd_binding_is_rejected(self):
        child, spec = self.child()
        current = dict(b.process_info(child.pid), cmdline_sha256=EMPTY)
        reused = dict(current, start_ticks=spec['start_ticks'] + 1)
        with patch.object(b, 'process_info', side_effect=[current, reused]), \
             patch.object(b, 'pidfd_exited') as poll:
            with self.assertRaisesRegex(ValueError, 'PID reused'):
                b.authenticated(spec)
            poll.assert_not_called()

    def test_reuse_after_exit_notification_is_still_rejected(self):
        child, spec = self.child()
        current = dict(b.process_info(child.pid), cmdline_sha256=EMPTY)
        reused = dict(current, start_ticks=spec['start_ticks'] + 1)
        with patch.object(b, 'process_info', side_effect=[current, current, reused]), \
             patch.object(b, 'pidfd_exited', return_value=True):
            with self.assertRaisesRegex(ValueError, 'PID reused'):
                b.authenticated(spec)

    def test_exit_before_pidfd_open_returns_no_signal(self):
        child, spec = self.child()
        with patch.object(b, 'authenticated', return_value=b.process_info(child.pid)), \
             patch.object(os, 'pidfd_open', side_effect=ProcessLookupError), \
             patch.object(signal, 'pidfd_send_signal') as send:
            self.assertFalse(b.send(spec, signal.SIGTERM))
            send.assert_not_called()

    def test_exit_before_pidfd_signal_returns_false_and_closes_fd(self):
        child, spec = self.child()
        fd = os.pidfd_open(child.pid)
        with patch.object(os, 'pidfd_open', return_value=fd), \
             patch.object(signal, 'pidfd_send_signal', side_effect=ProcessLookupError):
            self.assertFalse(b.send(spec, signal.SIGTERM))
        with self.assertRaises(OSError):
            os.fstat(fd)

    def test_signal_permission_failure_is_not_misclassified_as_exit(self):
        child, spec = self.child()
        with patch.object(signal, 'pidfd_send_signal', side_effect=PermissionError):
            with self.assertRaises(PermissionError):
                b.send(spec, signal.SIGTERM)

    def test_live_child_signal_uses_pidfd_and_retains_authentication(self):
        child, spec = self.child()
        self.assertTrue(b.send(spec, signal.SIGSTOP))
        b.stopped(spec)
        self.assertEqual(b.authenticated(spec)['state'], 'T')
        self.assertTrue(b.send(spec, signal.SIGCONT))
        self.release(child)
        self.assertEqual(child.wait(timeout=5), 0)

    def test_successor_exit_between_poll_and_authentication_preserves_exit_status(self):
        for exit_code in (0, 7):
            with self.subTest(exit_code=exit_code), tempfile.TemporaryDirectory() as directory:
                child, _ = self.child(f'os._exit({exit_code})')
                boundary = b.Boundary('/unused', 'f' * 64)
                boundary.run = boundary.directory = Path(directory)
                completion = Path(directory) / 'completed.json'
                completion.write_text('{}')
                boundary.s = dict(startup={'pid': -1}, user='test-only', replacement=dict(
                    command=['not-executed'], manifest_sha256='a' * 64,
                    completion_path=str(completion)))
                boundary.run_id = 'test-only'
                boundary.command = ['original-preprocessing-not-executed']
                original_poll = child.poll
                released = False

                def poll_then_exit():
                    nonlocal released
                    result = original_poll()
                    if not released:
                        self.assertIsNone(result)
                        released = True
                        self.release(child)
                        fd = os.pidfd_open(child.pid)
                        try:
                            self.assertTrue(b.pidfd_exited(fd, 5000))
                        finally:
                            os.close(fd)
                    return result

                original_authenticated = b.authenticated
                with ExitStack() as stack:
                    for method in ('validate', 'health', 'completion', 'validate_successor_completion', 'event'):
                        stack.enter_context(patch.object(boundary, method))
                    stack.enter_context(patch.object(b, 'send', return_value=True))
                    stack.enter_context(patch.object(b, 'containers', return_value=[]))
                    stack.enter_context(patch.object(b, 'authenticated', side_effect=lambda spec:
                        None if spec['pid'] == -1 else original_authenticated(spec)))
                    stack.enter_context(patch.object(b.subprocess, 'Popen', return_value=child))
                    stack.enter_context(patch.object(child, 'poll', side_effect=poll_then_exit))
                    if exit_code:
                        with self.assertRaisesRegex(ValueError, 'Successor failed'):
                            boundary.run_successor('b' * 64)
                        boundary.validate_successor_completion.assert_not_called()
                    else:
                        boundary.run_successor('b' * 64)
                        boundary.validate_successor_completion.assert_called_once()
                self.assertEqual(child.returncode, exit_code)


if __name__ == '__main__':
    unittest.main()
