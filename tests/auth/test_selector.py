import datetime
import importlib.util
import pathlib
import sys
import types
import unittest
from typing import Dict, Optional

ROOT = pathlib.Path(__file__).resolve().parents[2]
AUTH_ALIAS = ROOT / "litellm" / "auth" / "alias.py"
AUTH_CORE = ROOT / "litellm" / "auth" / "core.py"
AUTH_SELECTOR = ROOT / "litellm" / "auth" / "selector.py"


def _load_module(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    sys.modules[name] = mod  # type: ignore
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    return mod


# Minimal package scaffolding to satisfy relative imports
if "litellm" not in sys.modules:
    litellm_pkg = types.ModuleType("litellm")
    litellm_pkg.__path__ = [str(ROOT / "litellm")]  # type: ignore[attr-defined]
    sys.modules["litellm"] = litellm_pkg
if "litellm.auth" not in sys.modules:
    auth_pkg = types.ModuleType("litellm.auth")
    auth_pkg.__path__ = [str(ROOT / "litellm" / "auth")]  # type: ignore[attr-defined]
    sys.modules["litellm.auth"] = auth_pkg

alias_mod = _load_module(AUTH_ALIAS, "litellm.auth.alias")
core_mod = _load_module(AUTH_CORE, "litellm.auth.core")
selector_mod = _load_module(AUTH_SELECTOR, "litellm.auth.selector")

ModelAliasMap = alias_mod.ModelAliasMap
AuthRecord = core_mod.AuthRecord
AuthStatus = core_mod.AuthStatus
ModelState = core_mod.ModelState
QuotaState = core_mod.QuotaState
CredentialSelector = selector_mod.CredentialSelector


def _ts(seconds: int = 0) -> datetime.datetime:
    return datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc) + datetime.timedelta(
        seconds=seconds
    )


def make_auth(
    auth_id: str,
    provider: str,
    unavailable: bool = False,
    next_retry_after: Optional[datetime.datetime] = None,
    model_states: Optional[Dict[str, ModelState]] = None,
) -> AuthRecord:
    return AuthRecord(
        id=auth_id,
        provider=provider,
        label=auth_id,
        unavailable=unavailable,
        next_retry_after=next_retry_after,
        model_states=model_states or {},
        attributes={},
        metadata={},
        quota=QuotaState(),
        status=AuthStatus.ACTIVE,
    )


class SelectorTests(unittest.TestCase):
    def test_prefers_provider_and_alias_order(self):
        alias = ModelAliasMap()
        alias.register("claude-3.5", "anthropic/claude-3.5", "openrouter/claude-3.5")
        selector = CredentialSelector(alias_map=alias)
        auths = [
            make_auth("b", "openrouter"),
            make_auth("a", "anthropic"),
        ]
        sel = selector.select(
            logical_model="claude-3.5",
            auth_records=auths,
            preferred_providers=["anthropic", "openrouter"],
        )
        self.assertIsNotNone(sel)
        self.assertEqual(sel.auth.id, "a")
        self.assertEqual(sel.provider_model, "anthropic/claude-3.5")

    def test_cross_provider_disabled_returns_none_when_only_other_provider_available(self):
        alias = ModelAliasMap()
        alias.register("claude-3.5", "anthropic/claude-3.5", "openrouter/claude-3.5")
        selector = CredentialSelector(alias_map=alias)
        future = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(
            hours=1
        )
        anth_state = ModelState(
            unavailable=True,
            next_retry_after=future,
            quota=QuotaState(exceeded=True, next_recover_at=future, backoff_level=1),
        )
        auths = [
            make_auth(
                "a",
                "anthropic",
                unavailable=True,
                next_retry_after=future,
                model_states={"anthropic/claude-3.5": anth_state},
            ),
            make_auth("b", "openrouter"),
        ]
        sel = selector.select(
            logical_model="claude-3.5",
            auth_records=auths,
            allow_cross_provider=False,
        )
        self.assertIsNone(sel)

    def test_mark_failure_sets_next_refresh_on_429(self):
        selector = CredentialSelector()
        auth = make_auth("a", "anthropic")
        updated = selector.mark_failure(
            auth,
            provider_model="anthropic/claude",
            error_message="over quota",
            status_code=429,
            is_quota=True,
        )
        self.assertIsNotNone(updated.next_refresh_after)

    def test_mark_failure_honors_retry_after(self):
        selector = CredentialSelector()
        auth = make_auth("a", "anthropic")
        now = _ts(0)
        retry_after = datetime.timedelta(seconds=120)
        updated = selector.mark_failure(
            auth,
            provider_model="anthropic/claude",
            error_message="over quota",
            status_code=429,
            is_quota=True,
            retry_after=retry_after,
            now=now,
        )
        self.assertEqual(updated.next_retry_after, now + retry_after)

    def test_rotation_within_provider(self):
        selector = CredentialSelector()
        auths = [
            make_auth("a1", "anthropic"),
            make_auth("a2", "anthropic"),
        ]
        alias = ModelAliasMap()
        alias.register("claude", "anthropic/claude")
        selector.alias_map = alias
        first = selector.select("claude", auth_records=auths)
        second = selector.select("claude", auth_records=auths)
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertNotEqual(first.auth.id, second.auth.id)

    def test_disable_quota_cooldown(self):
        selector = CredentialSelector(disable_quota_cooldown=True)
        auth = make_auth("a", "anthropic")
        updated = selector.mark_failure(
            auth,
            provider_model="anthropic/claude",
            error_message="quota",
            status_code=429,
            is_quota=True,
        )
        self.assertFalse(updated.unavailable)
        self.assertIsNone(updated.next_retry_after)

    def test_provider_override_disables_cooldown(self):
        selector = CredentialSelector(
            disable_quota_cooldown=False,
            provider_quota_cooldown_overrides={"anthropic": True},
        )
        auth = make_auth("a", "anthropic")
        updated = selector.mark_failure(
            auth,
            provider_model="anthropic/claude",
            error_message="quota",
            status_code=429,
            is_quota=True,
        )
        self.assertFalse(updated.unavailable)
        self.assertIsNone(updated.next_retry_after)

    def test_model_override_disables_cooldown(self):
        selector = CredentialSelector(
            disable_quota_cooldown=False,
            model_quota_cooldown_overrides={"anthropic/claude": True},
        )
        auth = make_auth("a", "anthropic")
        updated = selector.mark_failure(
            auth,
            provider_model="anthropic/claude",
            error_message="quota",
            status_code=429,
            is_quota=True,
        )
        self.assertFalse(updated.unavailable)
        self.assertIsNone(updated.next_retry_after)

    def test_backoff_and_cooldown_marking(self):
        selector = CredentialSelector()
        auth = make_auth("a", "anthropic")
        now = _ts()
        failed = selector.mark_failure(
            auth,
            provider_model="anthropic/claude-3.5",
            error_message="rate limit",
            status_code=429,
            is_quota=True,
            store=None,
            namespace=None,
            now=now,
        )
        ms = failed.model_states["anthropic/claude-3.5"]
        self.assertTrue(failed.unavailable)
        self.assertTrue(ms.unavailable)
        self.assertIsNotNone(failed.next_retry_after)
        self.assertGreater(failed.next_retry_after, now)
        self.assertEqual(ms.next_retry_after, failed.next_retry_after)


if __name__ == "__main__":
    unittest.main()
