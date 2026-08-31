import argparse
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_worker.py"
SPEC = importlib.util.spec_from_file_location("nimbus_run_worker", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class RunWorkerTests(unittest.TestCase):
    def _parameters_file(self, directory):
        path = Path(directory) / "params.json"
        path.write_text(
            json.dumps({"workerInterface": {"Enabled": True}}), encoding="utf-8"
        )
        return path

    def _args(self, parameters_file, **overrides):
        values = {
            "image": "annotations/example:latest",
            "dataset_id": "a" * 24,
            "dataset_view_id": None,
            "dataset_view_url": None,
            "parameters_file": parameters_file,
            "env_file": Path(".env"),
            "host_api_url": "http://localhost:8080/api/v1",
            "container_api_url": "http://girder:8080/api/v1",
            "docker_network": "nimbusimage_default",
            "container_name": "test-worker-run",
            "gpus": None,
            "dry_run": False,
            "preflight_only": False,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_extracts_dataset_view_id_from_hash_route(self):
        view_id = "6a94208ad233ea7fbc4328f9"
        url = f"http://localhost:5173/#/datasetView/{view_id}/view"
        self.assertEqual(MODULE._dataset_view_id(url), view_id)

    def test_reads_quoted_api_key_without_printing_it(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text('NIMBUS_API_KEY="private-value"\n', encoding="utf-8")
            with mock.patch.dict(MODULE.os.environ, {}, clear=True):
                self.assertEqual(MODULE._read_api_key(path), "private-value")

    def test_dry_run_redacts_token_and_makes_no_external_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            parameters_file = self._parameters_file(directory)
            args = self._args(parameters_file, dry_run=True)
            output = io.StringIO()
            with contextlib.redirect_stdout(output), mock.patch.object(
                MODULE, "_mint_session_token"
            ) as mint, mock.patch.object(MODULE.subprocess, "run") as run:
                self.assertEqual(MODULE.execute(args), 0)
            report = output.getvalue()
            self.assertIn("<short-lived-girder-token>", report)
            self.assertNotIn("private-value", report)
            mint.assert_not_called()
            run.assert_not_called()

    def test_live_run_revokes_token_and_reports_new_items(self):
        with tempfile.TemporaryDirectory() as directory:
            parameters_file = self._parameters_file(directory)
            args = self._args(parameters_file)
            old_item = {"_id": "1", "name": "source.tiff"}
            new_item = {
                "_id": "2",
                "name": "corrected.tiff",
                "size": 123,
                "largeImage": {"fileId": "3"},
            }
            output = io.StringIO()
            with contextlib.redirect_stdout(output), mock.patch.object(
                MODULE, "_read_api_key", return_value="api-key"
            ), mock.patch.object(
                MODULE, "_mint_session_token", return_value="temporary-token"
            ), mock.patch.object(
                MODULE, "_validate_dataset_folder"
            ), mock.patch.object(
                MODULE, "_preflight"
            ), mock.patch.object(
                MODULE,
                "_list_folder_items",
                side_effect=[[old_item], [old_item, new_item]],
            ), mock.patch.object(
                MODULE.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 0),
            ) as run, mock.patch.object(
                MODULE, "_revoke_session_token"
            ) as revoke:
                self.assertEqual(MODULE.execute(args), 0)
            command = run.call_args.args[0]
            self.assertEqual(command[command.index("--token") + 1], "temporary-token")
            self.assertEqual(command[command.index("--request") + 1], "compute")
            self.assertNotIn("temporary-token", output.getvalue())
            report = json.loads(output.getvalue())
            self.assertEqual(report["new_folder_items"][0]["id"], "2")
            revoke.assert_called_once_with(args.host_api_url, "temporary-token")

    def test_preflight_resolves_resources_without_starting_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            parameters_file = self._parameters_file(directory)
            args = self._args(parameters_file, preflight_only=True)
            output = io.StringIO()
            with contextlib.redirect_stdout(output), mock.patch.object(
                MODULE, "_read_api_key", return_value="api-key"
            ), mock.patch.object(
                MODULE, "_mint_session_token", return_value="temporary-token"
            ), mock.patch.object(
                MODULE, "_validate_dataset_folder"
            ), mock.patch.object(
                MODULE, "_preflight"
            ), mock.patch.object(
                MODULE, "_list_folder_items", return_value=[{"_id": "1"}]
            ), mock.patch.object(
                MODULE.subprocess, "run"
            ) as run, mock.patch.object(
                MODULE, "_revoke_session_token"
            ) as revoke:
                self.assertEqual(MODULE.execute(args), 0)
            report = json.loads(output.getvalue())
            self.assertEqual(report["mode"], "preflight-only")
            self.assertFalse(report["worker_started"])
            run.assert_not_called()
            revoke.assert_called_once_with(args.host_api_url, "temporary-token")

    def test_failed_worker_still_revokes_token(self):
        with tempfile.TemporaryDirectory() as directory:
            parameters_file = self._parameters_file(directory)
            args = self._args(parameters_file)
            output = io.StringIO()
            with contextlib.redirect_stdout(output), mock.patch.object(
                MODULE, "_read_api_key", return_value="api-key"
            ), mock.patch.object(
                MODULE, "_mint_session_token", return_value="temporary-token"
            ), mock.patch.object(
                MODULE, "_validate_dataset_folder"
            ), mock.patch.object(
                MODULE, "_preflight"
            ), mock.patch.object(
                MODULE, "_list_folder_items", side_effect=[[], []]
            ), mock.patch.object(
                MODULE.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 7),
            ), mock.patch.object(
                MODULE, "_revoke_session_token"
            ) as revoke:
                with self.assertRaisesRegex(MODULE.RunError, "status 7"):
                    MODULE.execute(args)
            self.assertNotIn("temporary-token", output.getvalue())
            revoke.assert_called_once_with(args.host_api_url, "temporary-token")


if __name__ == "__main__":
    unittest.main()
