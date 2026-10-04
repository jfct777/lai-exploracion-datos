"""Cloud-wait regression tests: synthetic CLI output, no GCP or genomic reads."""
import importlib.util
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import Mock, call, patch
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('r02_cloud_wait', ROOT/'bin/r02_parallel_coordinator.py')
coordinator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(coordinator)

URI = ('gs://projects-usp/dnaBr-lai/datalake/refined/DNABR_QC/'
       'synthetic-r02/parallel/worker01/00_worker_provenance/worker_completion.json')
ERROR_PREFIX = 'ERROR: (gcloud.storage.objects.describe) '
ACTUAL_MISSING = ERROR_PREFIX + URI + ' not found: 404.\n'
MISSING_DIAGNOSTICS = (
    ACTUAL_MISSING,
    ERROR_PREFIX + 'gs://projects-usp/' + quote(URI.split('/', 3)[3], safe='') + ' not found: 404.\n',
    ERROR_PREFIX + 'HTTPError 404: No such object: synthetic-r02/worker_completion.json\n',
    ERROR_PREFIX + 'NOT_FOUND: 404: No such object\n',
    ERROR_PREFIX + 'ResponseError: status=404, code=NotFound\n',
    'HTTPError 404: No such object\n',
    'NOT_FOUND: 404\n',
    'ResponseError: status=404\n',
    'WARNING: optional client diagnostic\n' + ACTUAL_MISSING,
)


def cli_result(stderr='', *, returncode=1, stdout=''):
    return subprocess.CompletedProcess(['gcloud'], returncode, stdout=stdout, stderr=stderr)


class CloudMetadataTests(unittest.TestCase):
    def test_actual_and_legacy_absence_formats_are_optional(self):
        for stderr in MISSING_DIAGNOSTICS:
            with self.subTest(stderr=stderr), patch.object(
                    coordinator.subprocess, 'run', return_value=cli_result(stderr)) as run:
                self.assertIsNone(coordinator.GCS().metadata(URI, optional=True))
                run.assert_called_once_with(
                    ['gcloud', 'storage', 'objects', 'describe', URI, '--format=json'],
                    capture_output=True, text=True, timeout=120)

    def test_missing_is_still_an_error_by_default_or_explicitly_required(self):
        for stderr in MISSING_DIAGNOSTICS:
            for kwargs in ({}, {'optional': False}):
                with self.subTest(stderr=stderr, kwargs=kwargs), patch.object(
                        coordinator.subprocess, 'run', return_value=cli_result(stderr)):
                    with self.assertRaisesRegex(RuntimeError, 'Cannot authenticate cloud object'):
                        coordinator.GCS().metadata(URI, **kwargs)

    def test_permissions_transients_and_ambiguous_messages_fail_closed(self):
        diagnostics = (
            'HTTPError 401: Invalid credentials',
            'HTTPError 403: Permission denied',
            'HTTPError 429: Too many requests',
            'HTTPError 500: Internal server error',
            'HTTPError 503: Service unavailable',
            'HTTPError 403: Permission denied; object not found: 404.',
            'HTTPError 403: Permission denied for gs://bucket/HTTPError 404',
            'PERMISSION_DENIED: 403: ResponseError: status=404 in an object name',
            'The command encountered an unknown failure (404)',
            'HTTPError 4040: Invalid status',
            'gs://other-bucket/worker_completion.json not found: 404.',
            'You do not currently have an active account selected.',
            'Connection reset by peer',
            '',
        )
        for message in diagnostics:
            with self.subTest(message=message), patch.object(
                    coordinator.subprocess, 'run', return_value=cli_result(ERROR_PREFIX + message)):
                with self.assertRaisesRegex(RuntimeError, 'Cannot authenticate cloud object'):
                    coordinator.GCS().metadata(URI, optional=True)

    def test_conflicting_error_lines_are_not_treated_as_absence(self):
        stderr = ACTUAL_MISSING + ERROR_PREFIX + 'HTTPError 403: Permission denied\n'
        with patch.object(coordinator.subprocess, 'run', return_value=cli_result(stderr)):
            with self.assertRaises(RuntimeError):
                coordinator.GCS().metadata(URI, optional=True)

    def test_status_in_requested_uri_cannot_masquerade_as_missing(self):
        uri = 'gs://bucket/HTTPError 404/ResponseError: status=404.json'
        stderr = ERROR_PREFIX + uri + ': Permission denied (403).'
        with patch.object(coordinator.subprocess, 'run', return_value=cli_result(stderr)):
            with self.assertRaises(RuntimeError):
                coordinator.GCS().metadata(uri, optional=True)

    def test_success_preserves_metadata_even_with_stderr(self):
        metadata = {'generation': '123', 'size': '456', 'md5_hash': 'unchanged'}
        with patch.object(coordinator.subprocess, 'run', return_value=cli_result(
                'WARNING: client diagnostic', returncode=0, stdout=json.dumps(metadata))):
            self.assertEqual(coordinator.GCS().metadata(URI, optional=True), metadata)

    def test_success_with_invalid_json_is_not_treated_as_absence(self):
        with patch.object(coordinator.subprocess, 'run', return_value=cli_result(
                ACTUAL_MISSING, returncode=0, stdout='invalid json')):
            with self.assertRaises(json.JSONDecodeError):
                coordinator.GCS().metadata(URI, optional=True)

    def test_timeout_and_missing_client_propagate_without_retry(self):
        for error in (subprocess.TimeoutExpired(['gcloud'], 120), FileNotFoundError('gcloud')):
            with self.subTest(error=type(error)), patch.object(
                    coordinator.subprocess, 'run', side_effect=error) as run:
                with self.assertRaises(type(error)):
                    coordinator.GCS().metadata(URI, optional=True)
                self.assertEqual(run.call_count, 1)


class CloudWaitTests(unittest.TestCase):
    def setUp(self):
        self.spec = {'deadline_utc': '1970-01-01T00:01:40+00:00', 'poll_seconds': 30,
                     'remote_chromosomes': [{'chromosome': 1, 'completion_uri': URI}]}
        self.state = Mock()

    def test_actual_missing_object_waits_until_existing_deadline(self):
        # Exercise the real wait -> import -> metadata chain; no receipt is made.
        with patch.object(coordinator.subprocess, 'run', return_value=cli_result(ACTUAL_MISSING)) as run, \
                patch.object(coordinator.time, 'time', side_effect=[90, 90, 90, 100]), \
                patch.object(coordinator.time, 'sleep') as sleep, \
                patch.object(coordinator, 'write_fixed') as write:
            with self.assertRaisesRegex(ValueError, 'before coordinator deadline'):
                coordinator.wait_and_import(self.spec, Path('/unused'), self.state)
        self.state.assert_called_once_with('WAITING_REMOTE', pending_chromosomes=[1])
        sleep.assert_called_once_with(10)
        run.assert_called_once()
        write.assert_not_called()

    def test_wait_loop_retries_pending_and_finishes_after_import(self):
        # Import integrity is tested separately; this isolates the loop contract.
        with patch.object(coordinator.time, 'time', return_value=0), \
                patch.object(coordinator.time, 'sleep') as sleep, \
                patch.object(coordinator, 'import_chromosome', side_effect=[False, False, True]) as importer:
            coordinator.wait_and_import(self.spec, Path('/unused'), self.state)
        self.assertEqual(importer.call_count, 3)
        self.assertEqual(sleep.call_args_list, [call(30), call(30)])
        self.assertEqual(self.state.call_args_list, [
            call('WAITING_REMOTE', pending_chromosomes=[1]),
            call('WAITING_REMOTE', pending_chromosomes=[1]),
            call('ALL_REMOTE_IMPORTED', pending_chromosomes=[]),
        ])

    def test_permission_or_transient_error_does_not_enter_wait(self):
        for status in (401, 403, 429, 500, 503):
            stderr = ERROR_PREFIX + f'HTTPError {status}: Failure'
            with self.subTest(status=status), \
                    patch.object(coordinator.subprocess, 'run', return_value=cli_result(stderr)) as run, \
                    patch.object(coordinator.time, 'time', return_value=0), \
                    patch.object(coordinator.time, 'sleep') as sleep:
                with self.assertRaises(RuntimeError):
                    coordinator.wait_and_import(self.spec, Path('/unused'), self.state)
                run.assert_called_once()
                sleep.assert_not_called()
        self.state.assert_not_called()


if __name__ == '__main__':
    unittest.main()
