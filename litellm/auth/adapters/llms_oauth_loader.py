"""
Load OAuth helper modules without importing `litellm.llms`.

`litellm.llms.__init__` eagerly imports a large provider surface area (and
optional dependencies). Subscription OAuth helpers are small and self-contained,
so we load them by file path to keep the auth subsystem lightweight.
"""

from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType


def _litellm_root() -> Path:
    # .../litellm/auth/adapters/llms_oauth_loader.py -> .../litellm
    return Path(__file__).resolve().parents[2]


@lru_cache(maxsize=64)
def load_module(relpath: str, name: str) -> ModuleType:
    """
    Load a module by file path relative to the `litellm/` package directory.
    """
    path = _litellm_root() / relpath
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module

