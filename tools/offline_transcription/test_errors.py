"""Synthetic download failures; no network, models or ML dependencies."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import assets
import protocol
import worker


class HfHubHTTPError(Exception):
    def __init__(self, message, status=None):
        super().__init__(message)
        self.response = types.SimpleNamespace(status_code=status)


class GatedRepoError(HfHubHTTPError):
    pass


class PreparationFailureTests(unittest.TestCase):
    def invoke_worker(self, directory, error, artifact='community1'):
        root = Path(directory)
        request = root/'request.json'
        response = root/'result.json'
        protocol.atomic_json(request, dict(operation='prepare', cache_dir=str(root/'cache'),
            artifact_ids=[artifact]))
        hub = types.ModuleType('huggingface_hub')
        errors = types.ModuleType('huggingface_hub.errors')
        errors.GatedRepoError = GatedRepoError
        errors.HfHubHTTPError = HfHubHTTPError
        hub.snapshot_download = hub.hf_hub_download = lambda **_: (_ for _ in ()).throw(error)
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.dict(sys.modules, {'huggingface_hub': hub, 'huggingface_hub.errors': errors}), \
                patch.object(sys, 'argv', ['worker.py', '--request', str(request), '--response', str(response)]), \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as raised:
                worker.main()
        self.assertEqual(raised.exception.code, 1)
        self.assertFalse(response.exists())
        return protocol.load_json(protocol.failure_path(response)), stdout.getvalue()+stderr.getvalue()

    def test_gated_and_http_authorization_failures_have_only_safe_identifiers(self):
        secret = 'fictional-secret https://example.invalid/private?token=fictional-secret'
        wrapped = RuntimeError(secret)
        wrapped.__cause__ = GatedRepoError(secret, 403)
        contextual = RuntimeError(secret)
        contextual.__context__ = HfHubHTTPError(secret, 401)
        for error in (GatedRepoError(secret), HfHubHTTPError(secret, 401),
                      HfHubHTTPError(secret, 403), wrapped, contextual):
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as directory:
                failure, output = self.invoke_worker(directory, error)
                self.assertEqual(failure, dict(protocol_version=1,
                    error_code='model_access_denied', error_artifact='community1'))
                self.assertNotIn(secret, str(failure))
                self.assertNotIn('https://', str(failure))
                self.assertIn(secret, output)  # retained only in private stderr

    def test_single_file_download_has_the_same_safe_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            failure, _ = self.invoke_worker(directory, HfHubHTTPError('fictional-secret', 403), 'campplus')
            self.assertEqual(failure['error_artifact'], 'campplus')

    def test_network_unknown_and_text_only_access_failures_stay_generic(self):
        for error in (HfHubHTTPError('fictional-secret 401 gated repo', 500),
                      ConnectionError('fictional-secret'), ValueError('401 GatedRepoError fictional-secret')):
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as directory:
                failure, output = self.invoke_worker(directory, error)
                self.assertEqual(failure, dict(protocol_version=1))
                self.assertNotIn('fictional-secret', str(failure))

    def test_public_failure_rejects_untrusted_values(self):
        valid = dict(protocol_version=1, error_code='model_access_denied', error_artifact='community1')
        self.assertEqual(protocol.public_failure(valid),
            dict(error_code='model_access_denied', error_artifact='community1'))
        for value in (None, [], dict(valid, error_code='fictional-secret'),
                      dict(valid, error_artifact='https://example.invalid/fictional-secret'),
                      dict(valid, error_artifact=[]), dict(valid, error='fictional-secret'),
                      dict(valid, protocol_version=True), dict(valid, protocol_version=2)):
            self.assertEqual(protocol.public_failure(value), {})

    def test_sidecar_invalid_json_oversize_and_symlinks_stay_generic(self):
        with tempfile.TemporaryDirectory() as directory:
            response = Path(directory)/'result.json'
            sidecar = protocol.failure_path(response)
            self.assertEqual(protocol.read_public_failure(response), {})
            for contents in ('{', 'x'*4097, '\xff'):
                sidecar.write_text(contents)
                self.assertEqual(protocol.read_public_failure(response), {})
            sidecar.unlink()
            target = Path(directory)/'untrusted.json'
            protocol.atomic_json(target, dict(protocol_version=1,
                error_code='model_access_denied', error_artifact='community1'))
            sidecar.symlink_to(target)
            self.assertEqual(protocol.read_public_failure(response), {})


if __name__ == '__main__':
    unittest.main()
