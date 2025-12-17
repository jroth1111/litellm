import importlib.util
import pathlib
import sys
import types
import unittest
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[2]
AUTH_CORE = ROOT / "litellm" / "auth" / "core.py"
AUTH_FILE_STORE = ROOT / "litellm" / "auth" / "file_store.py"
AUTH_REDIS_STORE = ROOT / "litellm" / "auth" / "redis_store.py"


def _load_module(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    sys.modules[name] = mod  # type: ignore
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    return mod


class FakeRedis:
    def __init__(self) -> None:
        self._data: dict[str, str] = {}

    def set(self, key, value, nx: bool = False, ex: int | None = None):  # noqa: ARG002
        key_str = key.decode("utf-8") if isinstance(key, (bytes, bytearray)) else str(key)
        if nx and key_str in self._data:
            return False
        self._data[key_str] = value
        return True

    def get(self, key):
        key_str = key.decode("utf-8") if isinstance(key, (bytes, bytearray)) else str(key)
        return self._data.get(key_str)

    def delete(self, key):
        key_str = key.decode("utf-8") if isinstance(key, (bytes, bytearray)) else str(key)
        existed = key_str in self._data
        if existed:
            del self._data[key_str]
        return 1 if existed else 0

    def scan_iter(self, pattern):
        pattern_str = str(pattern)
        if pattern_str.endswith("*"):
            prefix = pattern_str[:-1]
            for key in list(self._data.keys()):
                if key.startswith(prefix):
                    yield key
            return
        if pattern_str in self._data:
            yield pattern_str

    def eval(self, _script, _numkeys, key, value):
        key_str = str(key)
        if self._data.get(key_str) == value:
            del self._data[key_str]
            return 1
        return 0


class RedisStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        # Scaffold minimal package path + stub redis dependency.
        if "litellm" not in sys.modules:
            pkg = types.ModuleType("litellm")
            pkg.__path__ = [str(ROOT / "litellm")]  # type: ignore[attr-defined]
            sys.modules["litellm"] = pkg
        if "litellm.auth" not in sys.modules:
            pkg = types.ModuleType("litellm.auth")
            pkg.__path__ = [str(ROOT / "litellm" / "auth")]  # type: ignore[attr-defined]
            sys.modules["litellm.auth"] = pkg
        if "redis" not in sys.modules:
            sys.modules["redis"] = types.ModuleType("redis")

    def test_save_get_list_delete_roundtrip(self):
        core_mod = _load_module(AUTH_CORE, "litellm.auth.core")
        _load_module(AUTH_FILE_STORE, "litellm.auth.file_store")
        redis_mod = _load_module(AUTH_REDIS_STORE, "litellm.auth.redis_store")

        AuthRecord = core_mod.AuthRecord
        RedisAuthStore = redis_mod.RedisAuthStore

        client = FakeRedis()
        store = RedisAuthStore(client)
        now = datetime.now(timezone.utc)

        rec = AuthRecord(
            id="a1",
            provider="openai",
            metadata={"access_token": "tok", "expires_at": now.isoformat()},
            last_refreshed_at=now,
        )
        store.save("ns", rec)

        loaded = store.get("ns", "a1")
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded.id, "a1")
        self.assertEqual(loaded.provider, "openai")
        self.assertEqual(loaded.metadata.get("access_token"), "tok")

        recs = store.list("ns")
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0].id, "a1")

        store.delete("ns", "a1")
        self.assertIsNone(store.get("ns", "a1"))

    def test_lock_prevents_concurrent_writes(self):
        core_mod = _load_module(AUTH_CORE, "litellm.auth.core")
        _load_module(AUTH_FILE_STORE, "litellm.auth.file_store")
        redis_mod = _load_module(AUTH_REDIS_STORE, "litellm.auth.redis_store")

        AuthRecord = core_mod.AuthRecord
        RedisAuthStore = redis_mod.RedisAuthStore

        client = FakeRedis()
        store = RedisAuthStore(client)
        store.save("ns", AuthRecord(id="a1", provider="openai"))

        with store.lock("ns", "a1", timeout_seconds=0.1) as acquired_outer:
            self.assertTrue(acquired_outer)
            with store.lock("ns", "a1", timeout_seconds=0.1) as acquired_inner:
                self.assertFalse(acquired_inner)

        with store.lock("ns", "a1", timeout_seconds=0.1) as acquired_after:
            self.assertTrue(acquired_after)


if __name__ == "__main__":
    unittest.main()
