## provider-auth-v2: encrypted multi-credential auth + same-request rotation (no proxy)

### status

draft

### problem

OpenCode currently stores **one credential per provider** in `auth.json` (`Record<providerID, Auth.Info>`). This blocks:

- multiple OAuth subscription sessions per provider (e.g. multiple Claude Max accounts)
- same-request rotation on rate limit (HTTP 429) or auth expiry (401/403)
- clean separation between “subscription OAuth” and “API key” modes

Today, subscription-like auth is handled implicitly via external auth plugins and ad-hoc provider option injection, which is:

- non-DRY (each plugin re-implements flows)
- fragile (hard to keep behavior consistent across providers)
- not composable (no shared rotation policy, no shared secure store)

### goals

- Core-first architecture (no plugin required for correctness)
- Store **multiple credentials per provider** (OAuth + API key + wellknown)
- Encryption-at-rest for secrets, with atomic writes + best-effort cross-process locking
- “Same-request rotate” behavior:
  - 429 / quota → cooldown + move-to-back + retry with next credential
  - 401/403 → refresh once + retry once, else rotate
  - bounded attempts; never infinite loops
- Provider-scoped rotation by default (do not switch Anthropic→OpenAI unless configured)
- Keep API key usage fully supported (users can run OpenCode without subscriptions)
- DRY OAuth plumbing by reusing the existing localhost callback server pattern used by MCP OAuth

### non-goals

- Mid-stream rotation after tokens have been emitted (not safely resumable)
- Universal “list models” across all providers (only where the upstream supports a models endpoint)
- Backward compatibility with the v1 auth file format (migration is one-way)

---

## architecture

### modules

1) `vault/*` (crypto + filesystem durability)
2) `credentials/*` (records, encrypted secrets, health state, pool/queue ordering)
3) `provider-auth/*` (provider adapters: login/refresh/applyAuth/classifyFailure)
4) `inference/transport.ts` (rotating fetch wrapper injected into AI SDK providers)

### key integration point (why fetch, not model wrappers)

All bundled and dynamic AI SDK providers in `packages/opencode/src/provider/provider.ts` rely on `fetch`. OpenCode already wraps `fetch` for timeouts in `getSDK()`.

We extend that wrapper to inject a **rotating fetch**:

- stable provider instances (cache key does not include volatile tokens)
- uniform behavior across OpenAI, Anthropic, OpenAI-compatible, etc.
- same-request rotation is possible by retrying before returning a successful `Response`

---

## data model

### CredentialRecord (public metadata)

- `id: string` (ulid)
- `providerId: string` (e.g. `anthropic`, `openai`, `github-copilot`)
- `namespace: string` (pool name; default `default`)
- `label?: string` (user-friendly name)
- `kind: "oauth" | "api" | "wellknown"`
- `createdAt: number` (ms)
- `updatedAt: number` (ms)

### CredentialSecret (encrypted payload)

Discriminated by `kind`:

- oauth: `accessToken`, `refreshToken?`, `expiresAt?`, `extra?: Record<string, unknown>`
- api: `apiKey`
- wellknown: `envKey`, `token`

### CredentialHealth (plaintext runtime state)

- `cooldownUntil?: number` (ms)
- `lastStatusCode?: number`
- `lastErrorAt?: number`
- `successCount: number`
- `failureCount: number`

### storage layout

`Global.Path.data/credentials/`

- `records/<id>.json` (metadata + health + encrypted secret blob)
- `indexes/provider/<providerId>/<namespace>.json` (list of ids)
- `pools/<providerId>/<namespace>.json` (ordered queue of ids)

All writes are atomic; locks prevent concurrent “last write wins” where possible.

---

## vault (encryption)

### key management

Prefer OS keychain; fallback to env:

- `OPENCODE_VAULT_KEY` (base64 32 bytes)
- CLI: `opencode auth key init` (creates/stores key)

### crypto

AES-256-GCM:

- per-record random nonce
- ciphertext stored alongside nonce and version

---

## provider auth adapters

### interface (core, no plugins required)

Each adapter implements:

- `methods(): { type: "oauth" | "api", label: string, ... }[]`
- `startLogin(args) -> { url, method, instructions, state }`
- `finishLogin(args) -> CredentialSecret + optional metadata`
- `refresh(record) -> updated CredentialSecret` (optional; mark non-refreshable when absent)
- `applyAuth(headers, record)` (injects correct auth semantics)
- `classifyFailure(response|error)` → `{ rotatable, cooldownMs?, isAuthExpired?, reason }`

### oauth plumbing

Reuse the existing localhost callback server approach from:

- `src/mcp/oauth-callback.ts`
- `src/mcp/oauth-provider.ts`

Generalize into `src/oauth/callback.ts` and use it for both MCP and provider auth.

---

## rotation policy

Default policy per provider+namespace:

- selection: first non-cooldown credential in queue order
- on 429/quota:
  - parse `Retry-After` header (seconds) when present
  - set `cooldownUntil = now + retryAfterMs` (or a sane default)
  - move credential id to end of queue
  - retry with next credential (bounded)
- on 401/403:
  - attempt refresh once (if refresh_token exists and adapter supports refresh)
  - retry once
  - if still 401/403 → rotate (cooldown small)

---

## config changes

Extend `Config.Provider.options` with a typed `auth` block:

```
provider.<id>.auth = {
  mode: "auto" | "subscription" | "api",
  namespace?: "default",
  rotation?: {
    maxAttempts?: number,        // default: pool size
    rotateOnStatus?: number[],   // default: [429, 401, 403]
  }
}
```

Validation rules:

- `subscription`: requires at least one oauth credential in namespace
- `api`: requires a usable API key (env/config/stored)
- `auto`: prefers subscription if available, else api

---

## api + cli

### api (server)

Keep existing provider oauth endpoints for UI stability:

- `POST /provider/:providerID/oauth/authorize`
- `POST /provider/:providerID/oauth/callback`

Extend to accept `namespace` + `label` and store a *new* credential record instead of overwriting.

Add CRUD endpoints:

- `GET /auth/records?providerId=&namespace=`
- `DELETE /auth/records/:id`
- `POST /auth/records/:id/refresh`

### cli

- `opencode auth login <provider> [--namespace <ns>] [--label <label>]`
- `opencode auth list [--provider <id>] [--namespace <ns>]`
- `opencode auth delete <credentialId>`
- `opencode auth refresh <credentialId>`

---

## inference integration

In `src/provider/provider.ts:getSDK()`:

- build provider options (baseURL, headers, timeout) as today
- wrap fetch with:
  - timeout logic (existing)
  - rotating fetch (new) when auth mode uses subscriptions and credentials exist

Rotating fetch behavior retries HTTP requests before returning a Response to the AI SDK.

Streaming limitation:

- rotation happens before the successful stream begins (i.e., on initial non-2xx responses or request failures)
- if the stream fails after emitting tokens, the request fails; the credential is marked unhealthy and moved back

---

## migration

One-time migration on startup:

- `Global.Path.data/auth.json` → v2 credential records
- `Global.Path.data/mcp-auth.json` → v2 credential records (providerId prefixed with `mcp:`)
- legacy files renamed to `.bak`

---

## testing

- unit tests: pool ordering + cooldown + retry-after parsing; crypto roundtrip; atomic writes
- integration tests: mocked fetch returning 429 then 200; 401 then refresh then 200

