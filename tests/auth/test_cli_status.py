import importlib.util
import json
import pathlib
import sys
import tempfile
import types
import unittest
from datetime import datetime, timedelta, timezone

from click.testing import CliRunner

ROOT = pathlib.Path(__file__).resolve().parents[2]
AUTH_CLI = ROOT / "litellm" / "auth" / "cli.py"


def _load_module(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    sys.modules[name] = mod  # type: ignore
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    return mod


# Scaffold minimal package path to avoid importing litellm/__init__.py
if "litellm" not in sys.modules:
    pkg = types.ModuleType("litellm")
    pkg.__path__ = [str(ROOT / "litellm")]  # type: ignore[attr-defined]
    sys.modules["litellm"] = pkg
if "litellm.auth" not in sys.modules:
    pkg = types.ModuleType("litellm.auth")
    pkg.__path__ = [str(ROOT / "litellm" / "auth")]  # type: ignore[attr-defined]
    sys.modules["litellm.auth"] = pkg

cli_mod = _load_module(AUTH_CLI, "litellm.auth.cli")


class AuthCliStatusTests(unittest.TestCase):
    def test_status_empty_store(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as store_dir:
            result = runner.invoke(
                cli_mod.auth_cli,
                [
                    "status",
                    "--store",
                    store_dir,
                    "--ns",
                    "default",
                    "--plaintext",
                ],
            )
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn("No auth records found.", result.output)

    def test_status_json_output_includes_usage_fields(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as store_dir:
            store = cli_mod.JsonFileAuthStore(store_dir)
            now = datetime.now(timezone.utc)
            store.save(
                "default",
                cli_mod.AuthRecord(
                    id="a1",
                    provider="openai",
                    metadata={"access_token": "tok", "expires_at": (now + timedelta(hours=1)).isoformat()},
                    last_refreshed_at=now,
                ),
            )
            store.save(
                "default",
                cli_mod.AuthRecord(
                    id="b1",
                    provider="anthropic",
                    metadata={"access_token": "tok2"},
                    last_refreshed_at=now,
                ),
            )

            result = runner.invoke(
                cli_mod.auth_cli,
                [
                    "status",
                    "--store",
                    store_dir,
                    "--ns",
                    "default",
                    "--plaintext",
                    "--json",
                ],
            )
            self.assertEqual(result.exit_code, 0, result.output)
            payload = json.loads(result.output)
            self.assertEqual(len(payload), 2)
            for entry in payload:
                for key in (
                    "id",
                    "provider",
                    "status",
                    "unavailable",
                    "expires_at",
                    "request_count",
                    "error_count",
                    "prompt_tokens",
                    "completion_tokens",
                    "last_request_at",
                    "last_refreshed_at",
                ):
                    self.assertIn(key, entry)

            filtered = runner.invoke(
                cli_mod.auth_cli,
                [
                    "status",
                    "--store",
                    store_dir,
                    "--ns",
                    "default",
                    "--plaintext",
                    "--provider",
                    "openai",
                    "--json",
                ],
            )
            self.assertEqual(filtered.exit_code, 0, filtered.output)
            filtered_payload = json.loads(filtered.output)
            self.assertEqual(len(filtered_payload), 1)
            self.assertEqual(filtered_payload[0]["provider"], "openai")


if __name__ == "__main__":
    unittest.main()
