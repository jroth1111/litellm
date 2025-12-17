# Subscription OAuth (Auth Adapters) – Overview

This document describes how to use LiteLLM’s subscription OAuth support for consumer/subscription sessions (e.g., ChatGPT Plus, Claude Code). It is opt‑in and does not change API‑key flows unless you pass auth records.

## Concepts
- **AuthRecord**: one credential/account (provider, label, attributes, metadata, health state).
- **AuthStore**: persistence for AuthRecords (JSON file by default).
- **Auth Adapter**: provider-specific login + header/refresh logic (also implements the Router AuthStrategy interface).
- **Selector**: picks a healthy credential; respects model aliases, provider preference, and cooldown/backoff.
- **ModelAliasMap**: maps logical model names to provider-specific identifiers for cross-provider failover.

## Quick Start (encrypted store by default; API-key flows unchanged)
1) Set an encryption secret (recommended):
   ```bash
   export LITELLM_AUTH_ENCRYPTION_KEY="replace-me-with-a-long-random-secret"
   ```
2) Login via the built-in CLI:
   ```bash
   litellm auth login anthropic
   litellm auth login openai
   litellm auth login github_copilot
   litellm auth login qwen
   litellm auth login cursor
   litellm auth list
   litellm auth whoami
   # optional: force-refresh a single record (if provider supports refresh)
   litellm auth refresh <auth_id>
   ```
3) Add an AuthRecord via file or management API:
   - Use the FastAPI router in `litellm/auth/api.py` (mount at `/auth` via `auth_settings.mount_api`) to CRUD auth records (tokens are accepted but never echoed; token fields are redacted on read). The proxy mounts this behind `user_api_key_auth`.
   - Or drop a file into your auth store directory (default: `~/.config/litellm/auth`) using the `JsonFileAuthStore` (plaintext) or `EncryptedJsonFileAuthStore` (recommended) format.
4) Provide `auth_records` + `auth_strategies` to Router calls:
   ```python
   from litellm.auth.adapters.registry import default_strategies
   from litellm.auth.crypto import resolve_auth_encryption_secret
   from litellm.auth.file_store import EncryptedJsonFileAuthStore
   from litellm.auth.paths import default_auth_store_dir
   from litellm.router import Router

   store = EncryptedJsonFileAuthStore(
       default_auth_store_dir(),
       secret=resolve_auth_encryption_secret(),
   )
   auth_records = store.list("default")
   strategies = default_strategies()
   router = Router(
       model_list=[...],
       auth_strategies=strategies,
       auth_store=store,
       auth_namespace="default",
       auth_model_alias_map={
           "claude-3.5-sonnet": [
               "anthropic/claude-3.5-sonnet",
               "openrouter/claude-3.5-sonnet",
           ],
       },
   )

   # Per request
   resp = router.completion(
       model="claude-3.5-sonnet",
       messages=[{"role": "user", "content": "hello"}],
       auth_records=auth_records,  # opt-in; API-key-only calls ignore this
   )
   ```
5) OAuth flows:
   - Use the CLI `litellm auth login <provider>` for browser/device flows.
- Persisted tokens (access/refresh/expires_at) are stored in the AuthRecord metadata.

## Proxy Config (OpenCode / OpenAI-compatible clients)
Use `auth_mode: subscription` on deployments that should use OAuth subscription tokens (never set `litellm_params.api_key` on these deployments).

```yaml
model_list:
  - model_name: openai-model
    auth_mode: subscription
    litellm_params:
      model: openai/gpt-4o

auth_settings:
  enabled: true
  store_backend: encrypted_json
  store_dir: ~/.config/litellm/auth
  namespace: default
  mount_api: false
  maintainer: true
```

## Model Discovery (OpenAI-compatible only)
- `/v1/models` returns the configured proxy models by default.
- Add `include_discovered_models=true` to attempt best-effort upstream discovery for OpenAI-compatible subscription providers (currently `openai` and `cursor`). Discovered models are cached and appended to the response.

## Fallbacks and Aliases
- Use `auth_model_alias_map` to list equivalent provider model names for a logical model (e.g., claude at Anthropic vs OpenRouter).
- Routing policy knobs:
  - `auth_allow_cross_provider_fallback` (default True)
  - `auth_preferred_providers` (ordered list)
  - `auth_team_overrides` for per-team provider preferences and cross-provider toggle.

## Storage Backends
- Default: JSON file (`JsonFileAuthStore`).
- In-memory: `InMemoryAuthStore` for tests or ephemeral use.
- Redis: `RedisAuthStore` for shared, low-latency token storage.
- Mirroring: `MirrorAuthStore(primary, [mirrors...])` fans out writes to multiple stores (e.g., JSON + Redis) and merges reads by freshest `updated_at`.
- Git/Object: `GitAuthStore` (push/pull auth JSON via git) and `ObjectAuthStore` (S3/GCS-style clients) for backups/DR.
- Management API: `litellm/auth/api.py` (list/get/create/delete/patch; tokens accepted on create but never echoed).
- Other backends (e.g., S3/GCS/object/git mirrors) can implement the `AuthStore` protocol without code changes elsewhere.
- Selector pluggability: default `CredentialSelector` (health + round-robin), optional `WeightedSelector` (weights) and `LatencySelector` (prefers lower latency). You can inject any selector via the selector registry (`register_selector`) or pass a custom instance to `AuthManager`/Router.

## Health and Maintenance
- Selector marks auth/model unavailable on errors or quota, with cooldown/backoff.
- Background maintenance (`AuthMaintainer`) can refresh tokens proactively and reset cooled-down entries.
- Metrics: `AuthMetrics` captures call/quota/auth-error counts by provider; wire into your monitoring as needed.

## Migration Guide (API-key flows remain unchanged)
- Default behavior is unchanged unless you supply `auth_records` and `auth_strategies` in a call or Router initialization.
- To enable per-request: pass `auth_records` (list of AuthRecord) in kwargs; provide `auth_strategies` when constructing Router.
- To disable: omit `auth_records`; existing API-key routing continues to work.

## OAuth Provider Patterns (subscription flows)
- Config via env: client IDs/secrets/endpoints/scopes/redirects are overrideable per provider (e.g., `ANTHROPIC_CLIENT_ID`, `OPENAI_CLIENT_ID`, `GEMINI_CLIENT_ID`, `ANTIGRAVITY_CLIENT_ID`, `COPILOT_CLIENT_ID`, `QWEN_CLIENT_ID`). Registry overrides via `LITELLM_PROVIDER_OVERRIDES` (JSON).
- Safety (breaking tightening): state is now required on auth-code exchanges; PKCE required for Anthropic/OpenAI, and supported for Gemini/Antigravity (`code_challenge`/`code_verifier`). Device flows (Copilot/Qwen) handle pending/slow_down/expired_token/access_denied.
- Resilience: token exchange/refresh use bounded retries with minimal backoff; device flows back off on slow_down.
- Observability: set `LITELLM_OAUTH_DEBUG=1` to emit debug logs for token calls and device polling (status, attempt, elapsed). Secrets are not logged.
- Implementation: provider OAuth helpers live under `litellm/llms/<provider>/*oauth*.py` (loaded without importing the heavy `litellm.llms` package initializer); subscription adapters under `litellm/auth/adapters/*` implement login + refresh + header injection.
- Implementation note: subscription OAuth helpers are loaded without importing the heavy `litellm.llms` package initializer to avoid pulling optional dependencies into auth-only paths.
- Tests: `tests/auth/test_oauth_state_pkce.py` covers state mismatch and PKCE params; other provider-specific tests live in `tests/auth/`.

## Feature Flags / Gradual Rollout
- Control adoption by provider/model:
  - Only pass `auth_records` for models that should use subscription auth.
  - Use `auth_preferred_providers` to keep API-key providers first, and set `auth_allow_cross_provider_fallback=False` to isolate until ready.
  - Per-team overrides (`auth_team_overrides`) to limit rollout.
- Keep a parallel API-key deployment as a fallback while enabling subscription auth on a subset of requests.
