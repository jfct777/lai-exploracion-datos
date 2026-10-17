#!/usr/bin/env python3
"""Renew an expired metadata credential once; preserve all integrity checks.

The frozen verifier still owns HTTP retries, object identity and checksums.
Concurrent failures of the same token trigger only one credential renewal.
Credentials remain in memory and are never recorded in receipts.
"""
import threading

from r02_retirement_metadata import GCSMetadataVerifier, access_token


class RefreshingGCSMetadataVerifier(GCSMetadataVerifier):
    def __init__(self, **kwargs):
        self._refresh_provider = kwargs.get("token_provider", access_token)
        self._refresh_lock = threading.Lock()
        super().__init__(**kwargs)

    def metadata(self, uri):
        rejected_token = self._token
        try:
            return super().metadata(uri)
        except RuntimeError as error:
            if str(error) != "GCS metadata HTTP 401 for " + uri:
                raise
        with self._refresh_lock:
            if self._token == rejected_token:
                self._token = self._refresh_provider(self.account, self.project)
        self._reset_connection()
        # A repeated 401 remains an error; no unbounded authentication loop.
        return super().metadata(uri)
