"""
LiteLLM CLI entrypoint.

Preserves backwards compatibility with the historical `litellm` proxy command, while
adding subcommands like `litellm auth ...`.
"""

from __future__ import annotations

import sys
from typing import List, Optional


def main(argv: Optional[List[str]] = None) -> None:
    """
    Dispatch:
    - `litellm auth ...` -> auth CLI
    - otherwise         -> proxy `run_server` CLI (legacy behavior)
    """
    args = list(argv) if argv is not None else sys.argv[1:]
    if args and args[0] == "auth":
        from litellm.auth.cli import auth_cli

        auth_cli.main(
            args=args[1:],
            prog_name="litellm auth",
            standalone_mode=True,
        )
        return

    from litellm.proxy.proxy_cli import run_server

    run_server.main(
        args=args,
        prog_name="litellm",
        standalone_mode=True,
    )

