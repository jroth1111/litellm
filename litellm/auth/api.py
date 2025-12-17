"""
Optional FastAPI router for managing AuthRecords.

Endpoints:
- GET /auth: list auths
- GET /auth/{auth_id}: get a redacted auth record (no tokens)
- POST /auth: create/upload an auth record (tokens accepted, never echoed)
- DELETE /auth/{auth_id}: delete
- PATCH /auth/{auth_id}: update label/attributes/status_message
- GET /auth/providers: list supported OAuth providers
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from .adapters.registry import list_provider_descriptors
from .core import AuthRecord, AuthStatus, AuthStore


class AuthInput(BaseModel):
    id: str
    provider: str
    label: Optional[str] = ""
    attributes: Dict[str, str] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    account: Optional[str] = None


class AuthUpdate(BaseModel):
    label: Optional[str] = None
    attributes: Optional[Dict[str, str]] = None
    status_message: Optional[str] = None
    account: Optional[str] = None


def get_router(
) -> APIRouter:
    router = APIRouter(prefix="/auth", tags=["auth"])

    def _store(request: Request) -> tuple[AuthStore, str]:
        store: Optional[AuthStore] = getattr(request.app.state, "subscription_auth_store", None)
        namespace: str = getattr(request.app.state, "subscription_auth_namespace", "default")
        if store is None:
            raise HTTPException(
                status_code=503,
                detail="subscription auth store not configured (enable auth_settings in proxy config)",
            )
        return store, namespace

    def _ensure(rec: Optional[AuthRecord], auth_id: str) -> AuthRecord:
        if rec is None:
            raise HTTPException(status_code=404, detail=f"auth {auth_id} not found")
        return rec

    def _redact_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        Never expose bearer tokens via the management API.
        """
        if not meta:
            return {}
        redacted = dict(meta)
        for key in (
            "access_token",
            "refresh_token",
            "id_token",
            "token",
            "authorization",
        ):
            if key in redacted:
                redacted[key] = "***"
        return redacted

    def _serialize_safe(rec: AuthRecord) -> Dict[str, Any]:
        return {
            "id": rec.id,
            "provider": rec.provider,
            "label": rec.label,
            "attributes": rec.attributes,
            "metadata": _redact_metadata(rec.metadata),
            "status": rec.status.value,
            "status_message": rec.status_message,
            "unavailable": rec.unavailable,
            "expires_at": rec.expiration_time().isoformat()
            if rec.expiration_time()
            else None,
            "created_at": rec.created_at.isoformat() if rec.created_at else None,
            "updated_at": rec.updated_at.isoformat() if rec.updated_at else None,
        }

    @router.get("")
    def list_auths(request: Request):
        store, namespace = _store(request)
        recs = store.list(namespace)

        def serialize(rec: AuthRecord):
            return {
                "id": rec.id,
                "provider": rec.provider,
                "label": rec.label,
                "account": rec.metadata.get("account") or rec.attributes.get("account"),
                "status": rec.status.value,
                "status_message": rec.status_message,
                "unavailable": rec.unavailable,
                "expires_at": rec.expiration_time().isoformat()
                if rec.expiration_time()
                else None,
            }

        return [serialize(r) for r in recs]

    @router.get("/{auth_id}")
    def get_auth(auth_id: str, request: Request):
        store, namespace = _store(request)
        rec = store.get(namespace, auth_id)
        return _serialize_safe(_ensure(rec, auth_id))

    @router.post("")
    def create_auth(payload: AuthInput, request: Request):
        store, namespace = _store(request)
        rec = AuthRecord(
            id=payload.id,
            provider=payload.provider,
            label=payload.label or "",
            attributes=payload.attributes or {},
            metadata=payload.metadata or {},
            status=AuthStatus.ACTIVE,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        if payload.account:
            rec.metadata["account"] = payload.account
        store.save(namespace, rec)
        return _serialize_safe(rec)

    @router.delete("/{auth_id}")
    def delete_auth(auth_id: str, request: Request):
        store, namespace = _store(request)
        store.delete(namespace, auth_id)
        return {"deleted": auth_id}

    @router.patch("/{auth_id}")
    def update_auth(auth_id: str, update: AuthUpdate, request: Request):
        store, namespace = _store(request)
        rec = _ensure(store.get(namespace, auth_id), auth_id).clone()
        if update.label is not None:
            rec.label = update.label
        if update.attributes is not None:
            rec.attributes.update(update.attributes)
        if update.status_message is not None:
            rec.status_message = update.status_message
        if update.account is not None:
            rec.metadata["account"] = update.account
        rec.updated_at = datetime.now(timezone.utc)
        store.save(namespace, rec)
        return _serialize_safe(rec)

    @router.get("/providers")
    def providers():
        return list_provider_descriptors()

    @router.get("/errors")
    def errors(request: Request):
        """
        Return best-effort load errors from the underlying AuthStore (if supported).

        Useful for debugging malformed auth JSON files or decode failures.
        """
        store, _ = _store(request)
        errs = getattr(store, "last_load_errors", None) or []
        out = []
        for e in errs:
            out.append(
                {
                    "path": getattr(e, "path", ""),
                    "error_type": getattr(e, "error_type", ""),
                    "message": getattr(e, "message", ""),
                }
            )
        return {"count": len(out), "errors": out}

    return router
