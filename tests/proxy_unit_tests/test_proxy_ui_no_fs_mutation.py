from __future__ import annotations

from pathlib import Path

import pytest


def test_proxy_import_does_not_move_exported_html_files():
    """
    Regression test: importing `litellm.proxy.proxy_server` must not mutate the
    built UI directory (e.g., moving `login.html` into `login/index.html`).

    The proxy should serve extensionless routes via runtime routing, not by
    rewriting files on disk.
    """
    proxy_server = pytest.importorskip(
        "litellm.proxy.proxy_server",
        reason="requires LiteLLM proxy extras (pip install 'litellm[proxy]')",
    )
    ui_path = Path(proxy_server.__file__).resolve().parent / "_experimental" / "out"

    # These are tracked build artifacts in the repo.
    assert (ui_path / "login.html").exists()
    assert not (ui_path / "login" / "index.html").exists()
