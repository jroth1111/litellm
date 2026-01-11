"""
Shared OAuth callback helpers for CLI and MCP endpoints.
"""

from __future__ import annotations

import json
import threading
import urllib.parse
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Callable, Optional, Tuple
from urllib.parse import urlparse, urlunparse


def parse_redirect_uri(redirect_uri: str) -> Tuple[str, int, str]:
    parsed = urllib.parse.urlparse(redirect_uri)
    host = parsed.hostname or "127.0.0.1"
    if host == "localhost":
        host = "127.0.0.1"
    port = parsed.port
    if port is None:
        raise ValueError(f"redirect_uri missing port: {redirect_uri}")
    path = parsed.path or "/"
    return host, int(port), path


@dataclass
class CallbackResult:
    code: Optional[str] = None
    state: Optional[str] = None
    error: Optional[str] = None
    error_description: Optional[str] = None


def start_local_callback_server(
    *,
    redirect_uri: str,
    expected_state: str,
) -> Tuple[HTTPServer, threading.Thread, CallbackResult, threading.Event, str]:
    """
    Start the local HTTP callback server and return its runtime components.
    """
    host, port, expected_path = parse_redirect_uri(redirect_uri)
    result = CallbackResult()
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
        raise ValueError(f"Failed to bind callback server on {host}:{port}: {e}") from e

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


def wait_for_oauth_callback(
    *,
    redirect_uri: str,
    expected_state: str,
    timeout_seconds: int = 600,
) -> str:
    httpd, thread, result, done, _actual = start_local_callback_server(
        redirect_uri=redirect_uri, expected_state=expected_state
    )
    try:
        if not done.wait(timeout_seconds):
            raise ValueError(f"Timed out waiting for OAuth callback on {redirect_uri}")
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
        raise ValueError(f"OAuth error: {result.error} {detail}".strip())
    if not result.code:
        raise ValueError("OAuth callback missing `code` parameter")
    if not result.state:
        raise ValueError("OAuth callback missing `state` parameter")
    if result.state != expected_state:
        raise ValueError("OAuth state mismatch")
    return result.code


def get_request_base_url(request) -> str:
    """
    Get the base URL for the request, considering X-Forwarded-* headers.
    """
    base_url = str(request.base_url).rstrip("/")
    parsed = urlparse(base_url)

    x_forwarded_proto = request.headers.get("X-Forwarded-Proto")
    x_forwarded_host = request.headers.get("X-Forwarded-Host")
    x_forwarded_port = request.headers.get("X-Forwarded-Port")

    scheme = x_forwarded_proto if x_forwarded_proto else parsed.scheme

    if x_forwarded_host:
        if ":" in x_forwarded_host and not x_forwarded_host.startswith("["):
            netloc = x_forwarded_host
        elif x_forwarded_port:
            netloc = f"{x_forwarded_host}:{x_forwarded_port}"
        else:
            netloc = x_forwarded_host
    else:
        netloc = parsed.netloc
        if x_forwarded_port and ":" not in netloc:
            netloc = f"{netloc}:{x_forwarded_port}"

    return urlunparse((scheme, netloc, parsed.path, "", "", ""))


def encode_state_payload(
    *,
    base_url: str,
    original_state: str,
    encrypt: Callable[[str], str],
    code_challenge: Optional[str] = None,
    code_challenge_method: Optional[str] = None,
    client_redirect_uri: Optional[str] = None,
) -> str:
    state_data = {
        "base_url": base_url,
        "original_state": original_state,
        "code_challenge": code_challenge,
        "code_challenge_method": code_challenge_method,
        "client_redirect_uri": client_redirect_uri,
    }
    state_json = json.dumps(state_data, sort_keys=True)
    return encrypt(state_json)


def decode_state_payload(
    *,
    encrypted_state: str,
    decrypt: Callable[[str], Optional[str]],
) -> dict:
    decrypted_json = decrypt(encrypted_state)
    if decrypted_json is None:
        raise ValueError("Failed to decrypt state parameter")
    state_data = json.loads(decrypted_json)
    return state_data
