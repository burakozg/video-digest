from __future__ import annotations

import os
from pathlib import Path

import pytest

import video_digest.config as config_module
from video_digest.api.auth import reset_throttle


@pytest.fixture(autouse=True)
def _forget_failed_auth() -> None:
    """The auth throttle counts failures per address, in module state (see
    api/auth.py) — left alone it leaks between tests."""
    reset_throttle()


@pytest.fixture(autouse=True)
def _isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test gets Settings() built from pure field defaults.

    Same reasoning as vault-ask's tests/conftest.py: chdir to an empty
    tmp_path and repoint the module-global yaml path there, so a test does
    not depend on this repo's real config.yaml. `uv run` also loads a real
    `.env` from the project root as actual process environment *before Python
    starts*, which chdir does nothing about — VIDEODIGEST_* is stripped
    explicitly for the same reason vault-ask strips VAULTASK_*.
    """
    monkeypatch.chdir(tmp_path)
    config_module._active_yaml_path = tmp_path / "does-not-exist.yaml"
    for key in list(os.environ):
        if key.startswith("VIDEODIGEST_"):
            monkeypatch.delenv(key, raising=False)
