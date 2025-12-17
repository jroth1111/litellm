"""
Git-backed AuthStore for mirroring auth records to a remote repository.

This is a minimal adaptation: it writes auth JSON files into a working tree and
pushes best-effort. Remote divergence is tolerated (local wins).
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from typing import List, Optional

from .core import AuthRecord, AuthStore
from .file_store import _deserialize_auth, _serialize_auth


class GitAuthStore(AuthStore):
    def __init__(self, repo_url: str, workdir: Optional[str] = None, branch: str = "main") -> None:
        self.repo_url = repo_url
        self.branch = branch
        self.workdir = workdir or tempfile.mkdtemp(prefix="litellm-auth-git-")
        if not os.path.exists(self.workdir):
            os.makedirs(self.workdir, exist_ok=True)
        self._ensure_repo()

    def _run(self, args: list[str]) -> None:
        subprocess.run(args, cwd=self.workdir, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def _ensure_repo(self) -> None:
        git_dir = os.path.join(self.workdir, ".git")
        if not os.path.exists(git_dir):
            self._run(["git", "init", "-b", self.branch])
            self._run(["git", "remote", "add", "origin", self.repo_url])
            auth_dir = os.path.join(self.workdir, "auths")
            os.makedirs(auth_dir, exist_ok=True)
            self._run(["git", "add", "auths"])
            self._run(["git", "commit", "-m", "init auth store"])
        else:
            try:
                self._run(["git", "pull", "--no-edit", "origin", self.branch])
            except Exception:
                # best-effort; ignore divergence
                pass

    def _path(self, namespace: str, auth_id: str) -> str:
        dir_path = os.path.join(self.workdir, "auths", namespace)
        os.makedirs(dir_path, exist_ok=True)
        return os.path.join(dir_path, f"{auth_id}.json")

    def _sync(self) -> None:
        try:
            self._run(["git", "add", "."])
            self._run(["git", "commit", "-m", "update auth records"])
            self._run(["git", "push", "origin", self.branch])
        except Exception:
            # ignore push failures for offline/local use
            pass

    def get(self, namespace: str, auth_id: str) -> Optional[AuthRecord]:
        path = self._path(namespace, auth_id)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return _deserialize_auth(data)

    def save(self, namespace: str, record: AuthRecord) -> AuthRecord:
        record.updated_at = datetime.now(timezone.utc)
        path = self._path(namespace, record.id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_serialize_auth(record), f, ensure_ascii=False, indent=2)
        self._sync()
        return record

    def delete(self, namespace: str, auth_id: str) -> None:
        path = self._path(namespace, auth_id)
        if os.path.exists(path):
            os.remove(path)
        self._sync()

    def list(self, namespace: str) -> List[AuthRecord]:
        dir_path = os.path.join(self.workdir, "auths", namespace)
        if not os.path.isdir(dir_path):
            return []
        records: List[AuthRecord] = []
        for fname in os.listdir(dir_path):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(dir_path, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                records.append(_deserialize_auth(data))
            except Exception:
                continue
        return records
