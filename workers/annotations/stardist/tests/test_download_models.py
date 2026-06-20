"""Tests for the build-time model download script.

The Dockerfile runs ``download_models.py`` during the image build. If a
dependency is broken (e.g. ``stardist`` can't be imported because setuptools
removed ``pkg_resources``), the script MUST exit non-zero so the Docker build
fails loudly instead of shipping a broken image that only crashes at runtime.
"""

import os
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "download_models.py"


def _run_with_broken_stardist(tmp_path):
    """Run download_models.py with a `stardist` package that fails to import.

    A fake ``stardist`` package is placed first on PYTHONPATH so it shadows any
    real install, making the import failure deterministic regardless of the
    environment the test runs in.
    """
    pkg = tmp_path / "stardist"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(
        "raise ImportError(\"No module named 'pkg_resources'\")\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tmp_path) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, str(SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
    )


def test_exits_nonzero_when_stardist_import_fails(tmp_path):
    result = _run_with_broken_stardist(tmp_path)
    assert result.returncode != 0, (
        "download_models.py swallowed the import failure and exited 0; "
        "the Docker build would succeed and ship a broken image"
    )


def test_reports_the_import_error(tmp_path):
    result = _run_with_broken_stardist(tmp_path)
    combined = result.stdout + result.stderr
    assert "Error importing StarDist" in combined
    assert "pkg_resources" in combined
