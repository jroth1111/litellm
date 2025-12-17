from __future__ import annotations

import asyncio
import json
import secrets
import threading
import urllib.parse
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import click

from litellm.auth.core import AuthRecord, AuthStatus, RequestContext
from litellm.auth.adapters.registry import (
    default_strategies,
    get_adapter,
    list_adapters,
    list_provider_descriptors,
)
from litellm.auth.crypto import resolve_auth_encryption_secret
from litellm.auth.file_store import EncryptedJsonFileAuthStore, JsonFileAuthStore
from litellm.auth.paths import default_auth_store_dir, find_git_root
from litellm.auth.pkce import generate_pkce_pair, generate_state


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _default_auth_id(provider: str) -> str:
    ts = _now().strftime("%Y%m%d%H%M%S")
    suffix = secrets.token_hex(4)
    return f"{provider}-{ts}-{suffix}"


def _safe_label(value: Optional[str]) -> str:
    return (value or "").strip()


def _is_valid_redirect_url(url: Optional[str]) -> bool:
    """
    Validate URL to prevent XSS in OAuth callback pages.

    Only allows http:// and https:// URLs. Rejects javascript:, data:,
    and other potentially dangerous schemes.

    Reference: CLIProxyAPIPlus oauth_server.go isValidURL()
    """
    if not url or not isinstance(url, str):
        return False
    url = url.strip()
    return url.startswith("https://") or url.startswith("http://")


def _parse_redirect_uri(redirect_uri: str) -> Tuple[str, int, str]:
    parsed = urllib.parse.urlparse(redirect_uri)
    host = parsed.hostname or "127.0.0.1"
    if host == "localhost":
        host = "127.0.0.1"
    port = parsed.port
    if port is None:
        raise click.ClickException(f"redirect_uri missing port: {redirect_uri}")
    path = parsed.path or "/"
    return host, int(port), path


class _CallbackResult:
    def __init__(self) -> None:
        self.code: Optional[str] = None
        self.state: Optional[str] = None
        self.error: Optional[str] = None
        self.error_description: Optional[str] = None


def _wait_for_oauth_callback(
    redirect_uri: str,
    expected_state: str,
    timeout_seconds: int = 600,
) -> str:
    httpd, thread, result, done, _actual = _start_oauth_callback_server(
        redirect_uri=redirect_uri, expected_state=expected_state
    )
    try:
        if not done.wait(timeout_seconds):
            raise click.ClickException(
                f"Timed out waiting for OAuth callback on {redirect_uri}"
            )
    finally:
        try:
            httpd.shutdown()
        except Exception:
            pass
        try:
            httpd.server_close()
        except Exception:
            pass
        try:
            thread.join(timeout=1)
        except Exception:
            pass

    if result.error:
        detail = result.error_description or ""
        raise click.ClickException(f"OAuth error: {result.error} {detail}".strip())
    if not result.code:
        raise click.ClickException("OAuth callback missing `code` parameter")
    if not result.state:
        raise click.ClickException("OAuth callback missing `state` parameter")
    if result.state != expected_state:
        raise click.ClickException("OAuth state mismatch")
    return result.code


def _start_oauth_callback_server(
    redirect_uri: str,
    expected_state: str,
) -> tuple[HTTPServer, threading.Thread, _CallbackResult, threading.Event, str]:
    """
    Start the local HTTP callback server and return its runtime components.

    Supports `redirect_uri` with port 0 to bind an ephemeral port (RFC 8252-style
    loopback). This is opt-in; default provider redirects remain unchanged.
    """
    host, port, expected_path = _parse_redirect_uri(redirect_uri)
    result = _CallbackResult()
    done = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            parsed_req = urllib.parse.urlparse(self.path)
            if parsed_req.path != expected_path:
                self.send_response(404)
                self.end_headers()
                return

            params = urllib.parse.parse_qs(parsed_req.query or "")
            result.state = (params.get("state") or [None])[0]
            if result.state != expected_state:
                body = b"Invalid state. Please restart login."
                self.send_response(400)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                result.error = "state_mismatch"
                result.error_description = "State parameter mismatch"
                done.set()
                return

            result.code = (params.get("code") or [None])[0]
            result.error = (params.get("error") or [None])[0]
            result.error_description = (params.get("error_description") or [None])[0]

            if result.error:
                body = b"Login failed. You can close this window."
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                done.set()
                return

            if not result.code:
                body = b"Missing authorization code. Please retry login."
                self.send_response(400)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                result.error = "missing_code"
                result.error_description = "Missing authorization code"
                done.set()
                return

            body = b"Login complete. You can close this window."
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            done.set()

        def log_message(self, format: str, *args):  # noqa: A002
            return

    class ReuseHTTPServer(HTTPServer):
        allow_reuse_address = True

    try:
        httpd = ReuseHTTPServer((host, port), Handler)
    except OSError as e:
        raise click.ClickException(
            f"Failed to bind callback server on {host}:{port}: {e}"
        ) from e

    actual_host, actual_port = httpd.server_address[:2]
    parsed = urllib.parse.urlparse(redirect_uri)
    actual_redirect_uri = urllib.parse.urlunparse(
        (
            parsed.scheme or "http",
            f"{actual_host}:{actual_port}",
            expected_path,
            "",
            "",
            "",
        )
    )

    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, thread, result, done, actual_redirect_uri


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
):
    _warn_if_git_repo_path(store_dir)
    if not encrypt:
        return JsonFileAuthStore(store_dir)
    try:
        secret = resolve_auth_encryption_secret(encryption_key)
        return EncryptedJsonFileAuthStore(
            store_dir,
            secret=secret,
            allow_plaintext_fallback=allow_plaintext_fallback,
        )
    except Exception as e:
        raise click.ClickException(str(e)) from e


@click.group(name="auth")
def auth_cli() -> None:
    """Manage subscription OAuth credentials for LiteLLM (AuthStore)."""


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
        device = adapter.device_authorize()  # type: ignore[attr-defined]
        url = getattr(device, "verification_uri_complete", None) or getattr(
            device, "verification_uri", None
        )
        if url:
            _maybe_open_browser(str(url), open_browser=open_browser)
        user_code = getattr(device, "user_code", None)
        if user_code:
            click.echo(f"Enter code: {user_code}")
        try:
            metadata = adapter.device_poll(  # type: ignore[attr-defined]
                device, timeout_seconds=timeout_seconds
            )
        except Exception as e:
            raise click.ClickException(str(e)) from e
    elif login_flow == "cursor_poll":
        session = adapter.start_login()  # type: ignore[attr-defined]
        login_url = getattr(session, "login_url", None) or getattr(session, "login", None)
        if login_url:
            _maybe_open_browser(str(login_url), open_browser=open_browser)
        try:
            metadata = adapter.poll_login(  # type: ignore[attr-defined]
                session, timeout_seconds=timeout_seconds
            )
        except Exception as e:
            raise click.ClickException(str(e)) from e
    else:
        # Browser/PKCE flows
        state = generate_state()
        code_verifier, code_challenge = generate_pkce_pair()
        redirect = redirect_uri or getattr(adapter, "default_redirect_uri")
        # Optional RFC 8252 loopback behavior: if the operator supplies a redirect
        # URI with port 0 (e.g., http://127.0.0.1:0/callback), bind an ephemeral port
        # first, then use the actual redirect URI for the authorize+exchange calls.
        actual_redirect = redirect
        callback_server = None
        callback_thread = None
        callback_result = None
        callback_done = None
        try:
            _, port, _ = _parse_redirect_uri(redirect)
            if port == 0:
                (
                    callback_server,
                    callback_thread,
                    callback_result,
                    callback_done,
                    actual_redirect,
                ) = _start_oauth_callback_server(
                    redirect_uri=redirect, expected_state=state
                )
        except Exception:
            callback_server = None

        auth_url = adapter.authorize_url(  # type: ignore[attr-defined]
            state=state, code_challenge=code_challenge, redirect_uri=actual_redirect
        )
        _maybe_open_browser(str(auth_url), open_browser=open_browser)
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
            code = _wait_for_oauth_callback(
                actual_redirect, expected_state=state, timeout_seconds=timeout_seconds
            )
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
