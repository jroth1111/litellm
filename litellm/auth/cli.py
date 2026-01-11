from __future__ import annotations

import asyncio
import json
import os
import secrets
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import click

from litellm.auth.core import AuthKind, AuthRecord, AuthStatus, RequestContext
from litellm.auth.adapters.registry import (
    default_strategies,
    get_adapter,
    list_adapters,
    list_provider_descriptors,
)
from litellm.auth.crypto import (
    generate_auth_encryption_key,
    resolve_auth_encryption_secret,
)
from litellm.auth.file_store import EncryptedJsonFileAuthStore, JsonFileAuthStore
from litellm.auth.migration import maybe_migrate_legacy_auth_json
from litellm.auth.oauth_callback import (
    parse_redirect_uri,
    start_local_callback_server,
    wait_for_oauth_callback,
)
from litellm.auth.paths import (
    default_auth_key_path,
    default_auth_store_dir,
    find_git_root,
)
from litellm.auth.pkce import generate_pkce_pair, generate_state


def _now() -> datetime:
    return datetime.now(timezone.utc)


# Backwards-compatible alias for tests and external callers.
_parse_redirect_uri = parse_redirect_uri


def _default_auth_id(provider: str) -> str:
    ts = _now().strftime("%Y%m%d%H%M%S")
    suffix = secrets.token_hex(4)
    return f"{provider}-{ts}-{suffix}"


def _safe_label(value: Optional[str]) -> str:
    return (value or "").strip()


def _save_auth(
    store: JsonFileAuthStore,
    namespace: str,
    provider: str,
    auth_id: str,
    label: str,
    metadata: Dict[str, Any],
    attributes: Optional[Dict[str, str]] = None,
) -> AuthRecord:
    now = _now()
    rec = AuthRecord(
        id=auth_id,
        provider=provider,
        label=label,
        kind=AuthKind.OAUTH,
        attributes=attributes or {},
        metadata=metadata,
        status=AuthStatus.ACTIVE,
        created_at=now,
        updated_at=now,
        last_refreshed_at=now,
    )
    store.save(namespace, rec)
    return rec


def _maybe_open_browser(url: str, open_browser: bool) -> None:
    click.echo(f"\nOpen this URL to continue:\n{url}\n")
    if not open_browser:
        return
    try:
        webbrowser.open(url, new=1, autoraise=True)
    except Exception:
        # best effort
        pass


def _warn_if_git_repo_path(path: str) -> None:
    git_root = find_git_root(path)
    if not git_root:
        return
    click.echo(
        f"Warning: auth store path is inside a git repo ({git_root}). "
        "Store OAuth tokens outside repos to avoid accidental commits.",
        err=True,
    )


def _default_openai_compatible_base_url(provider: str) -> Optional[str]:
    """
    Best-effort base URL resolution for providers that expose an OpenAI-compatible `/models`.

    This is intentionally conservative; unknown providers must be specified via `--base-url`.
    """
    provider_key = (provider or "").strip().lower()
    if provider_key == "openai":
        return "https://api.openai.com/v1"

    # Resolve OpenAI-like providers from the JSON registry without importing `litellm.llms`.
    # This avoids importing the heavy llms package surface in the auth CLI.
    try:
        litellm_pkg_dir = Path(__file__).resolve().parents[1]  # .../litellm
        path = litellm_pkg_dir / "llms" / "openai_like" / "providers.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        entry = data.get(provider_key) if isinstance(data, dict) else None
        if isinstance(entry, dict):
            base_url = entry.get("base_url")
            if isinstance(base_url, str) and base_url.strip():
                return base_url.strip()
    except Exception:
        return None
    return None


def _build_store(
    store_dir: str,
    *,
    encrypt: bool,
    encryption_key: Optional[str],
    allow_plaintext_fallback: bool,
    namespace: str,
    migrate_legacy: bool = True,
):
    _warn_if_git_repo_path(store_dir)
    if not encrypt:
        store = JsonFileAuthStore(store_dir)
        if migrate_legacy:
            maybe_migrate_legacy_auth_json(store=store, namespace=namespace)
        return store
    try:
        secret = resolve_auth_encryption_secret(encryption_key)
        store = EncryptedJsonFileAuthStore(
            store_dir,
            secret=secret,
            allow_plaintext_fallback=allow_plaintext_fallback,
        )
        if migrate_legacy:
            maybe_migrate_legacy_auth_json(store=store, namespace=namespace)
        return store
    except Exception as e:
        raise click.ClickException(str(e)) from e


@click.group(name="auth")
def auth_cli() -> None:
    """Manage subscription OAuth credentials for LiteLLM (AuthStore)."""


@auth_cli.group(name="key")
def key_cli() -> None:
    """Manage auth encryption keys."""


@key_cli.command(name="init")
@click.option(
    "--path",
    "key_path",
    default=default_auth_key_path(),
    show_default=True,
    help="Write the generated key to this path (use '-' to skip writing).",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Overwrite existing key file if it already exists.",
)
@click.option(
    "--stdout",
    is_flag=True,
    default=False,
    help="Only print the key (do not write to disk).",
)
def key_init(key_path: str, force: bool, stdout: bool) -> None:
    """
    Generate a base64 32-byte encryption key for auth storage.
    """
    key = generate_auth_encryption_key()
    if not stdout and key_path != "-":
        key_path = os.path.expanduser(key_path)
        key_dir = os.path.dirname(key_path)
        if key_dir:
            os.makedirs(key_dir, exist_ok=True)
            try:
                os.chmod(key_dir, 0o700)
            except Exception:
                pass
        if os.path.exists(key_path) and not force:
            raise click.ClickException(
                f"Key file already exists: {key_path} (use --force to overwrite)"
            )
        with open(key_path, "w", encoding="utf-8") as f:
            f.write(key + "\n")
        try:
            os.chmod(key_path, 0o600)
        except Exception:
            pass
        click.echo(f"Saved key to {key_path}")

    click.echo(f"LITELLM_AUTH_ENCRYPTION_KEY={key}")
    click.echo(f"export LITELLM_AUTH_ENCRYPTION_KEY={key}")


@auth_cli.command()
@click.argument("provider")
@click.option(
    "--store",
    "store_dir",
    default=default_auth_store_dir(),
    show_default=True,
    help="Auth store directory",
)
@click.option(
    "--namespace",
    "--ns",
    "ns",
    default="default",
    show_default=True,
    help="Auth namespace",
)
@click.option(
    "--encrypt/--plaintext",
    "encrypt",
    default=True,
    show_default=True,
    help="Encrypt credentials at rest (recommended). Requires LITELLM_AUTH_ENCRYPTION_KEY or LITELLM_MASTER_KEY.",
)
@click.option(
    "--encryption-key",
    default=None,
    help="Secret used to derive the encryption key (defaults to env LITELLM_AUTH_ENCRYPTION_KEY or LITELLM_MASTER_KEY)",
)
@click.option(
    "--allow-plaintext-fallback",
    is_flag=True,
    default=False,
    help="Allow reading legacy plaintext auth JSON while using encrypted store (migration helper).",
)
@click.option("--label", default=None, help="Human label for the saved credential")
@click.option("--auth-id", default=None, help="AuthRecord id (defaults to provider-timestamp-rand)")
@click.option("--redirect-uri", default=None, help="Override redirect URI for browser flows")
@click.option("--no-open-browser", is_flag=True, default=False, help="Print URL only (do not open browser)")
@click.option("--timeout", "timeout_seconds", default=600, type=int, show_default=True, help="Callback/device flow timeout seconds")
@click.option(
    "--session-key",
    default=None,
    help="(Anthropic only) Use existing browser session key instead of full OAuth flow.",
)
def login(
    provider: str,
    store_dir: str,
    ns: str,
    encrypt: bool,
    encryption_key: Optional[str],
    allow_plaintext_fallback: bool,
    label: Optional[str],
    auth_id: Optional[str],
    redirect_uri: Optional[str],
    no_open_browser: bool,
    timeout_seconds: int,
    session_key: Optional[str],
) -> None:
    """
    Login using an OAuth subscription flow and persist the resulting token(s).

    Examples:
      - `litellm auth login anthropic`
      - `litellm auth login github_copilot`
      - `litellm auth login cursor`
    """
    provider_key = provider.strip().lower()
    adapter = get_adapter(provider_key)
    if adapter is None:
        known = ", ".join(getattr(a, "provider", "") for a in list_adapters())
        raise click.ClickException(f"Unknown provider '{provider}'. Known: {known}")

    store = _build_store(
        store_dir,
        encrypt=encrypt,
        encryption_key=encryption_key,
        allow_plaintext_fallback=allow_plaintext_fallback,
        namespace=ns,
    )
    auth_id_final = auth_id or _default_auth_id(provider_key)
    open_browser = not no_open_browser
    label_final = _safe_label(label)

    login_flow = getattr(getattr(adapter, "capabilities", None), "login_flow", None)
    metadata: Dict[str, Any]

    # Handle session-key based login for Anthropic (cookie-based OAuth fallback)
    if session_key and provider_key == "anthropic":
        try:
            from litellm.llms.anthropic.cookie_oauth import (
                get_organizations,
                authorize_with_cookie,
                CookieAuthError,
            )
            click.echo("Using session key for cookie-based authentication...")
            orgs = get_organizations(session_key)
            if len(orgs) > 1:
                click.echo("Available organizations:")
                for i, org in enumerate(orgs):
                    click.echo(f"  {i + 1}. {org.name} ({org.uuid})")
                # Use first org by default, could add --org-uuid option later
                click.echo(f"Using: {orgs[0].name}")
            result = authorize_with_cookie(
                session_key=session_key,
                organization_uuid=orgs[0].uuid,
            )
            metadata = {
                "access_token": result.access_token,
                "refresh_token": result.refresh_token,
                "token_type": result.token_type,
                "expires_at": result.expires_at,
                "organization_uuid": result.organization_uuid,
                "organization_name": result.organization_name,
                "email": result.email,
            }
        except Exception as e:
            raise click.ClickException(f"Cookie-based auth failed: {e}") from e
    elif session_key:
        raise click.ClickException(
            f"--session-key is only supported for Anthropic, not {provider}"
        )
    elif login_flow == "device_code":
        try:
            login_start = adapter.start_login()  # type: ignore[attr-defined]
        except Exception as e:
            raise click.ClickException(str(e)) from e
        if login_start.url:
            _maybe_open_browser(str(login_start.url), open_browser=open_browser)
        if login_start.user_code:
            click.echo(f"Enter code: {login_start.user_code}")
        try:
            metadata = adapter.device_poll(  # type: ignore[attr-defined]
                login_start.device_code, timeout_seconds=timeout_seconds
            )
        except Exception as e:
            raise click.ClickException(str(e)) from e
    elif login_flow == "cursor_poll":
        try:
            login_start = adapter.start_login()  # type: ignore[attr-defined]
        except Exception as e:
            raise click.ClickException(str(e)) from e
        if login_start.url:
            _maybe_open_browser(str(login_start.url), open_browser=open_browser)
        try:
            metadata = adapter.poll_login(  # type: ignore[attr-defined]
                login_start.session, timeout_seconds=timeout_seconds
            )
        except Exception as e:
            raise click.ClickException(str(e)) from e
    else:
        # Browser/PKCE flows
        state = generate_state()
        code_verifier, code_challenge = generate_pkce_pair()
        redirect = redirect_uri or getattr(adapter, "default_redirect_uri")
        if not redirect:
            raise click.ClickException(
                f"Provider '{provider_key}' is missing a default redirect_uri."
            )
        # Optional RFC 8252 loopback behavior: if the operator supplies a redirect
        # URI with port 0 (e.g., http://127.0.0.1:0/callback), bind an ephemeral port
        # first, then use the actual redirect URI for the authorize+exchange calls.
        actual_redirect = redirect
        callback_server = None
        callback_thread = None
        callback_result = None
        callback_done = None
        try:
            _, port, _ = parse_redirect_uri(redirect)
            if port == 0:
                (
                    callback_server,
                    callback_thread,
                    callback_result,
                    callback_done,
                    actual_redirect,
                ) = start_local_callback_server(
                    redirect_uri=redirect, expected_state=state
                )
        except Exception:
            callback_server = None

        try:
            login_start = adapter.start_login(  # type: ignore[attr-defined]
                state=state, code_challenge=code_challenge, redirect_uri=actual_redirect
            )
        except Exception as e:
            raise click.ClickException(str(e)) from e

        if login_start.url:
            _maybe_open_browser(str(login_start.url), open_browser=open_browser)
        if callback_server is not None and callback_done is not None and callback_result is not None:
            try:
                if not callback_done.wait(timeout_seconds):
                    raise click.ClickException(
                        f"Timed out waiting for OAuth callback on {actual_redirect}"
                    )
            finally:
                try:
                    callback_server.shutdown()
                except Exception:
                    pass
                try:
                    callback_server.server_close()
                except Exception:
                    pass
                if callback_thread is not None:
                    try:
                        callback_thread.join(timeout=1)
                    except Exception:
                        pass
            if callback_result.error:
                detail = callback_result.error_description or ""
                raise click.ClickException(
                    f"OAuth error: {callback_result.error} {detail}".strip()
                )
            if not callback_result.code:
                raise click.ClickException("OAuth callback missing `code` parameter")
            code = callback_result.code
        else:
            try:
                code = wait_for_oauth_callback(
                    redirect_uri=actual_redirect,
                    expected_state=state,
                    timeout_seconds=timeout_seconds,
                )
            except Exception as e:
                raise click.ClickException(str(e)) from e
        try:
            metadata = adapter.exchange_code(  # type: ignore[attr-defined]
                code=code,
                code_verifier=code_verifier,
                state=state,
                expected_state=state,
                redirect_uri=actual_redirect,
            )
        except Exception as e:
            raise click.ClickException(str(e)) from e

    metadata = {k: v for k, v in (metadata or {}).items() if v is not None}
    if not label_final:
        label_final = str(
            metadata.get("account")
            or metadata.get("email")
            or metadata.get("organization_name")
            or metadata.get("project_id")
            or metadata.get("account_id")
            or ""
        )
    rec = _save_auth(
        store=store,
        namespace=ns,
        provider=provider_key,
        auth_id=auth_id_final,
        label=label_final,
        metadata=metadata,
    )
    click.echo(f"Saved {rec.provider} auth to {store_dir}/{ns}/{rec.id}.json")
    return


@auth_cli.command(name="list")
@click.option(
    "--store",
    "store_dir",
    default=default_auth_store_dir(),
    show_default=True,
    help="Auth store directory",
)
@click.option(
    "--namespace",
    "--ns",
    "ns",
    default="default",
    show_default=True,
    help="Auth namespace",
)
@click.option(
    "--encrypt/--plaintext",
    "encrypt",
    default=True,
    show_default=True,
    help="Decrypt credentials at rest (requires encryption secret if encrypt=true).",
)
@click.option(
    "--encryption-key",
    default=None,
    help="Secret used to derive the encryption key (defaults to env LITELLM_AUTH_ENCRYPTION_KEY or LITELLM_MASTER_KEY)",
)
@click.option(
    "--allow-plaintext-fallback",
    is_flag=True,
    default=False,
    help="Allow reading legacy plaintext auth JSON while using encrypted store.",
)
@click.option("--provider", "provider_filter", default=None, help="Filter by provider")
def list_auths(
    store_dir: str,
    ns: str,
    encrypt: bool,
    encryption_key: Optional[str],
    allow_plaintext_fallback: bool,
    provider_filter: Optional[str],
) -> None:
    """List AuthRecords in the store."""
    store = _build_store(
        store_dir,
        encrypt=encrypt,
        encryption_key=encryption_key,
        allow_plaintext_fallback=allow_plaintext_fallback,
        namespace=ns,
    )
    records = store.list(ns)
    load_errors = getattr(store, "last_load_errors", None) or []
    if load_errors:
        click.echo(
            f"Warning: {len(load_errors)} auth file(s) could not be loaded. "
            "Fix or remove malformed files to avoid missing credentials.",
            err=True,
        )
        for err in load_errors[:10]:
            path = getattr(err, "path", "")
            msg = getattr(err, "message", "")
            click.echo(f"  - {path}: {msg}", err=True)
    if provider_filter:
        records = [r for r in records if r.provider == provider_filter.strip().lower()]
    for rec in records:
        acct = rec.metadata.get("account") or rec.metadata.get("email") or ""
        click.echo(f"{rec.id}\t{rec.provider}\t{rec.status.value}\t{acct}")


def _default_strategies() -> Dict[str, Any]:
    return default_strategies()


@auth_cli.command()
@click.argument("auth_id", required=False)
@click.option(
    "--store",
    "store_dir",
    default=default_auth_store_dir(),
    show_default=True,
    help="Auth store directory",
)
@click.option(
    "--namespace",
    "--ns",
    "ns",
    default="default",
    show_default=True,
    help="Auth namespace",
)
@click.option(
    "--encrypt/--plaintext",
    "encrypt",
    default=True,
    show_default=True,
    help="Decrypt credentials at rest (requires encryption secret if encrypt=true).",
)
@click.option(
    "--encryption-key",
    default=None,
    help="Secret used to derive the encryption key (defaults to env LITELLM_AUTH_ENCRYPTION_KEY or LITELLM_MASTER_KEY)",
)
@click.option(
    "--allow-plaintext-fallback",
    is_flag=True,
    default=False,
    help="Allow reading legacy plaintext auth JSON while using encrypted store.",
)
def whoami(
    auth_id: Optional[str],
    store_dir: str,
    ns: str,
    encrypt: bool,
    encryption_key: Optional[str],
    allow_plaintext_fallback: bool,
) -> None:
    """Show best-effort account identities for AuthRecords (never prints tokens)."""
    store = _build_store(
        store_dir,
        encrypt=encrypt,
        encryption_key=encryption_key,
        allow_plaintext_fallback=allow_plaintext_fallback,
        namespace=ns,
    )
    records = []
    if auth_id:
        rec = store.get(ns, auth_id)
        if rec is None:
            raise click.ClickException(f"AuthRecord not found: {auth_id}")
        records = [rec]
    else:
        records = store.list(ns)
        load_errors = getattr(store, "last_load_errors", None) or []
        if load_errors:
            click.echo(
                f"Warning: {len(load_errors)} auth file(s) could not be loaded.",
                err=True,
            )

    for rec in records:
        kind, ident = rec.account_identity()
        identity = f"{kind}:{ident}" if kind and ident else (ident or "")
        expires_at = rec.expiration_time().isoformat() if rec.expiration_time() else ""
        click.echo(
            f"{rec.id}\t{rec.provider}\t{identity}\t{rec.status.value}\t{expires_at}\t{rec.label}"
        )


@auth_cli.command()
@click.argument("auth_id")
@click.option(
    "--store",
    "store_dir",
    default=default_auth_store_dir(),
    show_default=True,
    help="Auth store directory",
)
@click.option(
    "--namespace",
    "--ns",
    "ns",
    default="default",
    show_default=True,
    help="Auth namespace",
)
@click.option(
    "--encrypt/--plaintext",
    "encrypt",
    default=True,
    show_default=True,
    help="Decrypt credentials at rest (requires encryption secret if encrypt=true).",
)
@click.option(
    "--encryption-key",
    default=None,
    help="Secret used to derive the encryption key (defaults to env LITELLM_AUTH_ENCRYPTION_KEY or LITELLM_MASTER_KEY)",
)
@click.option(
    "--allow-plaintext-fallback",
    is_flag=True,
    default=False,
    help="Allow reading legacy plaintext auth JSON while using encrypted store.",
)
def refresh(
    auth_id: str,
    store_dir: str,
    ns: str,
    encrypt: bool,
    encryption_key: Optional[str],
    allow_plaintext_fallback: bool,
) -> None:
    """Refresh a single AuthRecord (if provider supports refresh)."""
    store = _build_store(
        store_dir,
        encrypt=encrypt,
        encryption_key=encryption_key,
        allow_plaintext_fallback=allow_plaintext_fallback,
        namespace=ns,
    )
    rec = store.get(ns, auth_id)
    if rec is None:
        raise click.ClickException(f"AuthRecord not found: {auth_id}")

    strategies = _default_strategies()
    strat = strategies.get(rec.provider)
    if strat is None:
        raise click.ClickException(f"No strategy registered for provider: {rec.provider}")
    if not getattr(strat, "supports_refresh", True):
        raise click.ClickException(
            f"Provider '{rec.provider}' tokens are not refreshable; re-run `litellm auth login {rec.provider}`."
        )

    try:
        refreshed = strat.refresh(rec, ctx=RequestContext(model=""))  # type: ignore[arg-type]
    except Exception as e:
        raise click.ClickException(str(e)) from e

    store.save(ns, refreshed)
    expires_at = refreshed.expiration_time().isoformat() if refreshed.expiration_time() else ""
    click.echo(f"Refreshed {auth_id} ({rec.provider})\texpires_at={expires_at}")


def _parse_duration(duration_str: str) -> Optional[timedelta]:
    """
    Parse a duration string like '2h', '30m', '1d' into a timedelta.
    
    Supported suffixes: s (seconds), m (minutes), h (hours), d (days).
    """
    duration_str = duration_str.strip().lower()
    if not duration_str:
        return None
    
    multipliers = {'s': 1, 'm': 60, 'h': 3600, 'd': 86400}
    
    for suffix, mult in multipliers.items():
        if duration_str.endswith(suffix):
            try:
                value = int(duration_str[:-1])
                return timedelta(seconds=value * mult)
            except ValueError:
                return None
    
    # Try parsing as plain seconds
    try:
        return timedelta(seconds=int(duration_str))
    except ValueError:
        return None


@auth_cli.command(name="refresh-all")
@click.option(
    "--expiring-within",
    default="2h",
    show_default=True,
    help="Refresh tokens expiring within this duration (e.g., 2h, 30m, 1d).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Show what would be refreshed without actually refreshing.",
)
@click.option(
    "--store",
    "store_dir",
    default=default_auth_store_dir(),
    show_default=True,
    help="Auth store directory",
)
@click.option(
    "--namespace",
    "--ns",
    "ns",
    default="default",
    show_default=True,
    help="Auth namespace",
)
@click.option(
    "--encrypt/--plaintext",
    "encrypt",
    default=True,
    show_default=True,
    help="Use encrypted store.",
)
@click.option(
    "--encryption-key",
    default=None,
    help="Secret used to derive the encryption key.",
)
@click.option(
    "--allow-plaintext-fallback",
    is_flag=True,
    default=False,
    help="Allow reading legacy plaintext auth JSON.",
)
def refresh_all(
    expiring_within: str,
    dry_run: bool,
    store_dir: str,
    ns: str,
    encrypt: bool,
    encryption_key: Optional[str],
    allow_plaintext_fallback: bool,
) -> None:
    """
    Refresh all tokens that are expiring soon.

    Useful for automation (cron) to keep tokens fresh before they expire.

    Examples:
      - `litellm auth refresh-all`
      - `litellm auth refresh-all --expiring-within 1d`
      - `litellm auth refresh-all --dry-run`
    """
    duration = _parse_duration(expiring_within)
    if duration is None:
        raise click.ClickException(
            f"Invalid duration format: '{expiring_within}'. Use formats like '2h', '30m', '1d'."
        )

    store = _build_store(
        store_dir,
        encrypt=encrypt,
        encryption_key=encryption_key,
        allow_plaintext_fallback=allow_plaintext_fallback,
        namespace=ns,
    )
    records = store.list(ns)
    strategies = _default_strategies()
    now = _now()
    threshold = now + duration

    # Find tokens expiring within the threshold
    candidates = []
    for rec in records:
        expiry = rec.expiration_time()
        if expiry and expiry <= threshold and expiry > now:
            strat = strategies.get(rec.provider)
            if strat and getattr(strat, "supports_refresh", True):
                candidates.append((rec, expiry, strat))

    if not candidates:
        click.echo(f"No tokens expiring within {expiring_within} that need refreshing.")
        return

    click.echo(f"Found {len(candidates)} token(s) expiring within {expiring_within}:")

    refreshed_count = 0
    failed_count = 0

    for rec, expiry, strat in candidates:
        time_left = _format_time_delta(expiry)
        if dry_run:
            click.echo(f"  [DRY-RUN] {rec.id} ({rec.provider}) expires in {time_left}")
        else:
            try:
                refreshed = strat.refresh(rec, ctx=RequestContext(model=""))
                store.save(ns, refreshed)
                new_expiry = refreshed.expiration_time()
                new_time = new_expiry.isoformat() if new_expiry else "unknown"
                click.echo(f"  ✓ {rec.id} ({rec.provider}) refreshed, new expiry: {new_time}")
                refreshed_count += 1
            except Exception as e:
                click.echo(f"  ✗ {rec.id} ({rec.provider}) failed: {e}", err=True)
                failed_count += 1

    if dry_run:
        click.echo(f"\n[DRY-RUN] Would refresh {len(candidates)} token(s).")
    else:
        click.echo(f"\nRefreshed {refreshed_count} token(s), {failed_count} failed.")


@auth_cli.command(name="test")
@click.argument("model")
@click.option(
    "--auth-id",
    default=None,
    help="Force a specific AuthRecord id (otherwise auto-select from the store).",
)
@click.option(
    "--base-url",
    default=None,
    help="Override OpenAI-compatible base URL for live verification (e.g. https://api.openai.com/v1).",
)
@click.option(
    "--no-network",
    is_flag=True,
    default=False,
    help="Do not make network calls; only verify header injection and expiry parsing.",
)
@click.option(
    "--timeout",
    "timeout_seconds",
    default=10.0,
    type=float,
    show_default=True,
    help="Timeout for live verification calls.",
)
@click.option(
    "--store",
    "store_dir",
    default=default_auth_store_dir(),
    show_default=True,
    help="Auth store directory",
)
@click.option(
    "--namespace",
    "--ns",
    "ns",
    default="default",
    show_default=True,
    help="Auth namespace",
)
@click.option(
    "--encrypt/--plaintext",
    "encrypt",
    default=True,
    show_default=True,
    help="Decrypt credentials at rest (requires encryption secret if encrypt=true).",
)
@click.option(
    "--encryption-key",
    default=None,
    help="Secret used to derive the encryption key (defaults to env LITELLM_AUTH_ENCRYPTION_KEY or LITELLM_MASTER_KEY)",
)
@click.option(
    "--allow-plaintext-fallback",
    is_flag=True,
    default=False,
    help="Allow reading legacy plaintext auth JSON while using encrypted store.",
)
def test_auth(
    model: str,
    auth_id: Optional[str],
    base_url: Optional[str],
    no_network: bool,
    timeout_seconds: float,
    store_dir: str,
    ns: str,
    encrypt: bool,
    encryption_key: Optional[str],
    allow_plaintext_fallback: bool,
) -> None:
    """
    Validate that stored subscription auth can be applied to a model.

    - Always: selects a credential and verifies header injection.
    - If the provider supports OpenAI-compatible model listing: performs a live `/models` call.
    """
    store = _build_store(
        store_dir,
        encrypt=encrypt,
        encryption_key=encryption_key,
        allow_plaintext_fallback=allow_plaintext_fallback,
        namespace=ns,
    )

    model_str = (model or "").strip()
    if not model_str:
        raise click.ClickException("model must be non-empty")

    provider_hint = model_str.split("/", 1)[0].strip().lower() if "/" in model_str else ""
    adapter = get_adapter(provider_hint) if provider_hint else None
    if adapter is None or not getattr(adapter, "supports", lambda _m: False)(model_str):
        matches = [a for a in list_adapters() if getattr(a, "supports")(model_str)]
        if not matches:
            raise click.ClickException(f"No adapter supports model: {model_str}")
        if len(matches) > 1:
            providers = ", ".join(getattr(a, "provider", "") for a in matches)
            raise click.ClickException(
                f"Ambiguous model '{model_str}'. Specify provider prefix (e.g. openai/{model_str}) or choose one of: {providers}"
            )
        adapter = matches[0]

    provider = getattr(adapter, "provider", "").strip().lower()
    if not provider:
        raise click.ClickException("selected adapter is missing `provider`")

    if auth_id:
        rec = store.get(ns, auth_id)
        if rec is None:
            raise click.ClickException(f"AuthRecord not found: {auth_id}")
        records = [rec]
    else:
        records = store.list(ns)
        load_errors = getattr(store, "last_load_errors", None) or []
        if load_errors:
            click.echo(
                f"Warning: {len(load_errors)} auth file(s) could not be loaded; results may be incomplete.",
                err=True,
            )

    records = [r for r in records if (r.provider or "").strip().lower() == provider]
    if not records:
        raise click.ClickException(
            f"No AuthRecords found for provider '{provider}' in namespace '{ns}'. Run `litellm auth login {provider}`."
        )

    # Prefer the most recently refreshed credential.
    def _refresh_sort_key(r: AuthRecord):
        return r.last_refreshed_at or datetime.min.replace(tzinfo=timezone.utc)

    records = sorted(records, key=_refresh_sort_key, reverse=True)
    ctx = RequestContext(model=model_str)

    selected: Optional[AuthRecord] = None
    selected_headers: Optional[Dict[str, str]] = None
    last_err: Optional[BaseException] = None
    for rec in records:
        try:
            selected_headers = adapter.prepare({}, ctx, rec)
            selected = rec
            break
        except Exception as e:
            last_err = e
            continue

    if selected is None or selected_headers is None:
        raise click.ClickException(
            f"Failed to prepare auth headers for provider '{provider}': {last_err}"
        )

    expires_at = selected.expiration_time().isoformat() if selected.expiration_time() else ""
    click.echo(f"Selected {selected.id} ({provider})\texpires_at={expires_at}")

    caps = getattr(adapter, "capabilities", None)
    supports_models_list = bool(getattr(caps, "supports_models_list", False))
    if no_network or not supports_models_list:
        click.echo("OK (offline): header injection succeeded.")
        return

    resolved_base = (base_url or "").strip() or _default_openai_compatible_base_url(provider) or ""
    if not resolved_base:
        raise click.ClickException(
            f"Provider '{provider}' does not have a known OpenAI-compatible base URL. "
            "Re-run with `--base-url` to perform a live verification call."
        )

    from litellm.auth.model_discovery import fetch_openai_compatible_models

    status, models, raw = asyncio.run(
        fetch_openai_compatible_models(
            base_url=resolved_base,
            headers=selected_headers,
            timeout_seconds=float(timeout_seconds),
        )
    )
    if status == 200 and models:
        click.echo(f"OK: /models returned {len(models)} models (base_url={resolved_base})")
        return
    snippet = (raw or "")[:300].replace("\n", " ")
    raise click.ClickException(f"Live test failed: HTTP {status} base_url={resolved_base} body={snippet}")


@auth_cli.command()
@click.argument("auth_id")
@click.option(
    "--store",
    "store_dir",
    default=default_auth_store_dir(),
    show_default=True,
    help="Auth store directory",
)
@click.option(
    "--namespace",
    "--ns",
    "ns",
    default="default",
    show_default=True,
    help="Auth namespace",
)
@click.option(
    "--encrypt/--plaintext",
    "encrypt",
    default=True,
    show_default=True,
    help="Decrypt credentials at rest (requires encryption secret if encrypt=true).",
)
@click.option(
    "--encryption-key",
    default=None,
    help="Secret used to derive the encryption key (defaults to env LITELLM_AUTH_ENCRYPTION_KEY or LITELLM_MASTER_KEY)",
)
@click.option(
    "--allow-plaintext-fallback",
    is_flag=True,
    default=False,
    help="Allow reading legacy plaintext auth JSON while using encrypted store.",
)
def delete(
    auth_id: str,
    store_dir: str,
    ns: str,
    encrypt: bool,
    encryption_key: Optional[str],
    allow_plaintext_fallback: bool,
) -> None:
    """Delete an AuthRecord by id."""
    store = _build_store(
        store_dir,
        encrypt=encrypt,
        encryption_key=encryption_key,
        allow_plaintext_fallback=allow_plaintext_fallback,
        namespace=ns,
    )
    store.delete(ns, auth_id)
    click.echo(f"Deleted {auth_id}")


# Provider-specific revocation endpoints
_REVOKE_ENDPOINTS: Dict[str, str] = {
    "gemini": "https://oauth2.googleapis.com/revoke",
    "google": "https://oauth2.googleapis.com/revoke",
}


@auth_cli.command()
@click.argument("auth_id")
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Delete locally even if remote revocation fails.",
)
@click.option(
    "--store",
    "store_dir",
    default=default_auth_store_dir(),
    show_default=True,
    help="Auth store directory",
)
@click.option(
    "--namespace",
    "--ns",
    "ns",
    default="default",
    show_default=True,
    help="Auth namespace",
)
@click.option(
    "--encrypt/--plaintext",
    "encrypt",
    default=True,
    show_default=True,
    help="Use encrypted store.",
)
@click.option(
    "--encryption-key",
    default=None,
    help="Secret used to derive the encryption key.",
)
@click.option(
    "--allow-plaintext-fallback",
    is_flag=True,
    default=False,
    help="Allow reading legacy plaintext auth JSON.",
)
def revoke(
    auth_id: str,
    force: bool,
    store_dir: str,
    ns: str,
    encrypt: bool,
    encryption_key: Optional[str],
    allow_plaintext_fallback: bool,
) -> None:
    """
    Revoke an OAuth token and delete from local store.

    If the provider supports revocation, attempts remote revocation first.
    Use --force to delete locally even if remote revocation fails.

    Examples:
      - `litellm auth revoke anthropic-20241217`
      - `litellm auth revoke gemini-20241217 --force`
    """
    from litellm.auth.provider_http import http_post_with_retry

    store = _build_store(
        store_dir,
        encrypt=encrypt,
        encryption_key=encryption_key,
        allow_plaintext_fallback=allow_plaintext_fallback,
        namespace=ns,
    )

    rec = store.get(ns, auth_id)
    if rec is None:
        raise click.ClickException(f"AuthRecord not found: {auth_id}")

    provider = (rec.provider or "").strip().lower()
    revoke_url = _REVOKE_ENDPOINTS.get(provider)
    revoke_attempted = False
    revoke_succeeded = False

    # Attempt remote revocation if supported
    if revoke_url:
        token = rec.metadata.get("access_token") or rec.metadata.get("refresh_token")
        if token:
            revoke_attempted = True
            try:
                resp = http_post_with_retry(
                    revoke_url,
                    data={"token": token},
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    timeout=10.0,
                    max_attempts=2,
                )
                if resp.status_code in (200, 204):
                    revoke_succeeded = True
                    click.echo(f"✓ Remote revocation succeeded for {provider}")
                else:
                    click.echo(
                        f"⚠ Remote revocation returned HTTP {resp.status_code}",
                        err=True,
                    )
            except Exception as e:
                click.echo(f"⚠ Remote revocation failed: {e}", err=True)

    # Handle revocation failure
    if revoke_attempted and not revoke_succeeded and not force:
        raise click.ClickException(
            f"Remote revocation failed for {auth_id}. Use --force to delete locally anyway."
        )

    # Delete locally
    store.delete(ns, auth_id)
    if revoke_attempted and revoke_succeeded:
        click.echo(f"Revoked and deleted {auth_id}")
    elif revoke_attempted:
        click.echo(f"Deleted {auth_id} (remote revocation failed, forced)")
    else:
        click.echo(f"Deleted {auth_id} (provider '{provider}' does not support revocation)")


@auth_cli.command()
def providers() -> None:
    """List supported OAuth providers."""
    for p in list_provider_descriptors():
        click.echo(
            f"{p.get('provider')}\tflow={p.get('login_flow')}"
            f"\trefresh={p.get('supports_refresh')}\tmodels={p.get('supports_models_list')}"
        )


def _format_time_delta(dt: Optional[datetime]) -> str:
    """Format a datetime as a human-readable time delta from now."""
    if dt is None:
        return ""
    now = _now()
    if dt < now:
        delta = now - dt
        suffix = "ago"
    else:
        delta = dt - now
        suffix = ""

    total_seconds = int(delta.total_seconds())
    if total_seconds < 60:
        return f"{total_seconds}s {suffix}".strip()
    elif total_seconds < 3600:
        minutes = total_seconds // 60
        return f"{minutes}m {suffix}".strip()
    elif total_seconds < 86400:
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        if minutes > 0:
            return f"{hours}h {minutes}m {suffix}".strip()
        return f"{hours}h {suffix}".strip()
    else:
        days = total_seconds // 86400
        hours = (total_seconds % 86400) // 3600
        if hours > 0:
            return f"{days}d {hours}h {suffix}".strip()
        return f"{days}d {suffix}".strip()


def _get_status_symbol(rec: AuthRecord) -> str:
    """Get a status symbol for the record."""
    now = _now()

    # Check if expired
    expiry = rec.expiration_time()
    if expiry and expiry < now:
        return "⚠"  # Expired

    # Check if in cooldown
    if rec.next_retry_after and rec.next_retry_after > now:
        return "⏳"  # Cooldown

    # Check quota exceeded
    if rec.quota.exceeded:
        return "⏳"  # Quota cooldown

    # Check if unavailable
    if rec.unavailable:
        return "✗"  # Unavailable

    # Check status
    if rec.status == AuthStatus.ACTIVE:
        return "✓"  # Active
    elif rec.status == AuthStatus.EXPIRED:
        return "⚠"  # Expired
    elif rec.status == AuthStatus.ERROR:
        return "✗"  # Error
    elif rec.status == AuthStatus.DISABLED:
        return "○"  # Disabled

    return "?"


def _get_status_text(rec: AuthRecord) -> str:
    """Get status text for the record."""
    now = _now()

    # Check if expired
    expiry = rec.expiration_time()
    if expiry and expiry < now:
        return "EXPIRED"

    # Check if in cooldown
    if rec.next_retry_after and rec.next_retry_after > now:
        return f"COOLDOWN (retry in {_format_time_delta(rec.next_retry_after)})"

    # Check quota exceeded
    if rec.quota.exceeded:
        recover = rec.quota.next_recover_at
        if recover:
            return f"QUOTA (retry in {_format_time_delta(recover)})"
        return "QUOTA"

    # Check if unavailable
    if rec.unavailable:
        return "UNAVAILABLE"

    return rec.status.value.upper()


@auth_cli.command()
@click.option(
    "--store",
    "store_dir",
    default=default_auth_store_dir(),
    show_default=True,
    help="Auth store directory",
)
@click.option(
    "--namespace",
    "--ns",
    "ns",
    default="default",
    show_default=True,
    help="Auth namespace",
)
@click.option(
    "--encrypt/--plaintext",
    "encrypt",
    default=True,
    show_default=True,
    help="Use encrypted store.",
)
@click.option(
    "--encryption-key",
    default=None,
    help="Secret used to derive the encryption key.",
)
@click.option(
    "--allow-plaintext-fallback",
    is_flag=True,
    default=False,
    help="Allow reading legacy plaintext auth JSON.",
)
@click.option(
    "--provider",
    "provider_filter",
    default=None,
    help="Filter by provider (e.g., anthropic, openai).",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    default=False,
    help="Output as JSON.",
)
def status(
    store_dir: str,
    ns: str,
    encrypt: bool,
    encryption_key: Optional[str],
    allow_plaintext_fallback: bool,
    provider_filter: Optional[str],
    output_json: bool,
) -> None:
    """
    Show health status of all subscription accounts.

    Displays account validity, expiry, cooldown state, and usage metrics
    without making any API calls (no credits consumed).

    Examples:
      - `litellm auth status`
      - `litellm auth status --provider anthropic`
      - `litellm auth status --json`
    """
    store = _build_store(
        store_dir,
        encrypt=encrypt,
        encryption_key=encryption_key,
        allow_plaintext_fallback=allow_plaintext_fallback,
        namespace=ns,
    )
    records = store.list(ns)

    if provider_filter:
        records = [r for r in records if r.provider == provider_filter.strip().lower()]

    if not records:
        click.echo("No auth records found.")
        return

    now = _now()

    if output_json:
        # JSON output
        output = []
        for rec in records:
            expiry = rec.expiration_time()
            output.append({
                "id": rec.id,
                "provider": rec.provider,
                "label": rec.label,
                "status": rec.status.value,
                "status_text": _get_status_text(rec),
                "unavailable": rec.unavailable,
                "expires_at": expiry.isoformat() if expiry else None,
                "expires_in_seconds": int((expiry - now).total_seconds()) if expiry and expiry > now else None,
                "next_retry_after": rec.next_retry_after.isoformat() if rec.next_retry_after else None,
                "quota_exceeded": rec.quota.exceeded,
                "request_count": rec.request_count,
                "error_count": rec.error_count,
                "prompt_tokens": rec.prompt_tokens,
                "completion_tokens": rec.completion_tokens,
                "last_request_at": rec.last_request_at.isoformat() if rec.last_request_at else None,
                "last_refreshed_at": rec.last_refreshed_at.isoformat() if rec.last_refreshed_at else None,
                "account": rec.metadata.get("email") or rec.metadata.get("account") or "",
            })
        click.echo(json.dumps(output, indent=2))
        return

    # Group by provider
    by_provider: Dict[str, list] = {}
    for rec in records:
        by_provider.setdefault(rec.provider, []).append(rec)

    for provider, recs in sorted(by_provider.items()):
        click.echo(f"\nProvider: {provider} ({len(recs)} account{'s' if len(recs) != 1 else ''})")

        for rec in recs:
            symbol = _get_status_symbol(rec)
            status_text = _get_status_text(rec)

            # Get expiry info
            expiry = rec.expiration_time()
            if expiry:
                if expiry < now:
                    expiry_text = "expired"
                else:
                    expiry_text = f"expires in {_format_time_delta(expiry)}"
            else:
                expiry_text = "no expiry"

            # Get last used info
            if rec.last_request_at:
                last_used = f"last used: {_format_time_delta(rec.last_request_at)} ago"
            else:
                last_used = "never used"

            # Get error info
            error_info = f"{rec.error_count} errors" if rec.error_count > 0 else "0 errors"

            # Get token usage
            total_tokens = rec.prompt_tokens + rec.completion_tokens
            if total_tokens > 0:
                token_info = f"{total_tokens:,} tokens"
            else:
                token_info = ""

            # Get account identity
            account = rec.metadata.get("email") or rec.metadata.get("account") or ""
            if account:
                account = f" ({account})"

            # Format output line
            id_display = rec.id[:20] + "..." if len(rec.id) > 23 else rec.id
            click.echo(
                f"  {symbol} {id_display:<24} {status_text:<12} {expiry_text:<20} "
                f"{error_info:<12} {last_used}{account}"
            )

            # Show token usage on second line if present
            if token_info:
                click.echo(f"      {rec.request_count} requests, {token_info} ({rec.prompt_tokens:,} in / {rec.completion_tokens:,} out)")
