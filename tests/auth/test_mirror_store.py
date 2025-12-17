import importlib.util
import pathlib
import sys
import types
import unittest
from datetime import datetime, timedelta, timezone

ROOT = pathlib.Path(__file__).resolve().parents[2]
AUTH_CORE = ROOT / "litellm" / "auth" / "core.py"
MEMORY_STORE = ROOT / "litellm" / "auth" / "memory_store.py"
MIRROR_STORE = ROOT / "litellm" / "auth" / "mirror_store.py"


def _load_module(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    sys.modules[name] = mod  # type: ignore
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    return mod


# Minimal package scaffolding to avoid importing litellm/__init__.py
if "litellm" not in sys.modules:
    litellm_pkg = types.ModuleType("litellm")
    litellm_pkg.__path__ = [str(ROOT / "litellm")]  # type: ignore[attr-defined]
    sys.modules["litellm"] = litellm_pkg
if "litellm.auth" not in sys.modules:
    auth_pkg = types.ModuleType("litellm.auth")
    auth_pkg.__path__ = [str(ROOT / "litellm" / "auth")]  # type: ignore[attr-defined]
    sys.modules["litellm.auth"] = auth_pkg

core_mod = _load_module(AUTH_CORE, "litellm.auth.core")
memory_store_mod = _load_module(MEMORY_STORE, "litellm.auth.memory_store")
mirror_store_mod = _load_module(MIRROR_STORE, "litellm.auth.mirror_store")

AuthRecord = core_mod.AuthRecord
InMemoryAuthStore = memory_store_mod.InMemoryAuthStore
MirrorAuthStore = mirror_store_mod.MirrorAuthStore


class MirrorAuthStoreTest(unittest.TestCase):
    def test_get_falls_back_to_mirror(self) -> None:
        primary = InMemoryAuthStore()
        mirror = InMemoryAuthStore()
        rec = AuthRecord(id="a1", provider="anthropic", label="mirror")
        rec.updated_at = datetime.now(timezone.utc)
        mirror.save("ns", rec)

        store = MirrorAuthStore(primary, [mirror])
        loaded = store.get("ns", "a1")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.label, "mirror")

    def test_save_replicates_to_mirrors(self) -> None:
        primary = InMemoryAuthStore()
        mirror = InMemoryAuthStore()
        store = MirrorAuthStore(primary, [mirror])

        rec = AuthRecord(id="a2", provider="openai", label="primary")
        store.save("ns", rec)

        self.assertIsNotNone(primary.get("ns", "a2"))
        self.assertIsNotNone(mirror.get("ns", "a2"))

    def test_list_prefers_newest(self) -> None:
        now = datetime.now(timezone.utc)
        primary = InMemoryAuthStore()
        mirror = InMemoryAuthStore()

        old = AuthRecord(id="a3", provider="openai", label="old")
        old.updated_at = now - timedelta(minutes=5)
        primary.save("ns", old)

        newer = AuthRecord(id="a3", provider="openai", label="new")
        newer.updated_at = now
        mirror.save("ns", newer)

        store = MirrorAuthStore(primary, [mirror])
        records = {rec.id: rec for rec in store.list("ns")}
        self.assertIn("a3", records)
        self.assertEqual(records["a3"].label, "new")


if __name__ == "__main__":
    unittest.main()
