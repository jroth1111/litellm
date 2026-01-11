from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, TypeVar
from datetime import timedelta

import litellm
from litellm.auth.core import RequestContext
from litellm.auth.selector import _model_provider, interpret_status_code

ResponseT = TypeVar("ResponseT")


class AuthRotationHandler:
    """Handle subscription auth selection, refresh, and rotation loops."""

    def __init__(
        self,
        *,
        router: Any,
        model: str,
        deployment: Dict[str, Any],
        kwargs: Dict[str, Any],
        requires_subscription: bool,
        model_name: str,
    ) -> None:
        self.router = router
        self.model = model
        self.deployment = deployment
        self.kwargs = kwargs
        self.requires_subscription = requires_subscription
        self.model_name = model_name

        self.auth_selection: Optional[Any] = None
        self.marked_failure_for_raised_exception = False

        auth_records = kwargs.get("auth_records")
        if auth_records is None:
            self.all_auth_records: Optional[List[Any]] = None
        elif isinstance(auth_records, list):
            self.all_auth_records = auth_records
        else:
            try:
                self.all_auth_records = list(auth_records)
                self.kwargs["auth_records"] = self.all_auth_records
            except Exception:
                self.all_auth_records = None

        self.attempted_auth_ids: Set[str] = set()
        self.refreshed_auth_ids: Set[str] = set()
        max_available = (
            len(self.all_auth_records)
            if (self.router.auth_selector is not None and self.all_auth_records)
            else 0
        )
        rotation_cfg = {}
        try:
            auth_cfg = self.deployment.get("auth") or {}
            rotation_cfg = auth_cfg.get("rotation") or {}
        except Exception:
            rotation_cfg = {}

        override_attempts = (
            rotation_cfg.get("maxAttempts")
            or rotation_cfg.get("max_attempts")
            or rotation_cfg.get("max_attempts_per_request")
        )
        configured = getattr(self.router, "auth_max_attempts_per_request", None)
        max_setting = override_attempts if override_attempts is not None else configured
        if max_setting is None:
            self.max_auth_attempts = max_available
        else:
            try:
                configured_int = int(max_setting)
            except Exception:
                configured_int = max_available
            if configured_int <= 0:
                self.max_auth_attempts = 0
            else:
                self.max_auth_attempts = (
                    min(configured_int, max_available) if max_available else 0
                )

        rotate_override = rotation_cfg.get("rotateOnStatus") or rotation_cfg.get(
            "rotate_on_status"
        )
        if rotate_override is None:
            self.rotate_on_status_codes = getattr(
                self.router, "auth_rotate_on_status_codes", {401, 403, 429}
            )
        else:
            parsed_codes: Set[int] = set()
            try:
                for code in rotate_override:
                    parsed_codes.add(int(code))
            except Exception:
                parsed_codes = set()
            self.rotate_on_status_codes = (
                parsed_codes
                if parsed_codes
                else getattr(self.router, "auth_rotate_on_status_codes", {401, 403, 429})
            )

    def _replace_auth_record(self, updated: Any) -> None:
        if not self.all_auth_records:
            return
        updated_id = getattr(updated, "id", None)
        for idx, rec in enumerate(self.all_auth_records):
            if getattr(rec, "id", None) == updated_id:
                self.all_auth_records[idx] = updated
                return

    def _filter_attempted_auth_records(self) -> None:
        if self.attempted_auth_ids and self.all_auth_records is not None:
            self.kwargs["auth_records"] = [
                rec
                for rec in self.all_auth_records
                if getattr(rec, "id", None) not in self.attempted_auth_ids
            ]

    def _request_context(self) -> RequestContext:
        return RequestContext(
            model=self.model,
            user_id=self.kwargs.get("user"),
            team_id=self.kwargs.get("team"),
            metadata=self.kwargs.get("metadata", {}),
        )

    def _raise_no_credentials(self) -> None:
        raise litellm.RateLimitError(
            message=f"no healthy subscription credentials available for {self.model}",
            llm_provider=_model_provider(self.model_name) or "",
            model=self.model_name,
        )

    def _record_failover(self) -> None:
        if self.auth_selection is None:
            return
        try:
            self.router.auth_metrics.record_failover(
                self.auth_selection.selection.auth.provider
            )
        except Exception:
            return

    def run_sync(self, do_call: Callable[[], ResponseT]) -> ResponseT:
        while True:
            self._filter_attempted_auth_records()

            self.auth_selection = self.router._maybe_select_auth(
                model=self.model, deployment=self.deployment, kwargs=self.kwargs
            )

            if self.auth_selection is None:
                if self.requires_subscription:
                    self._raise_no_credentials()
                strategy = None
            else:
                self.router._log_auth_identity(self.auth_selection)
                strategy = self.router._auth_strategy_for(
                    self.auth_selection.selection.auth.provider
                )
                try:
                    self.auth_selection = self.router._prepare_auth_headers(
                        auth_ctx=self.auth_selection,
                        strategy=strategy,
                        model=self.model,
                        kwargs=self.kwargs,
                    )
                except Exception as header_error:
                    if self.auth_selection is not None and self.max_auth_attempts > 0:
                        updated = self.router._mark_auth_failure(
                            self.auth_selection, header_error
                        )
                        if updated is not None:
                            self._replace_auth_record(updated)
                        self.attempted_auth_ids.add(
                            self.auth_selection.selection.auth.id
                        )
                        self._record_failover()
                        if len(self.attempted_auth_ids) >= self.max_auth_attempts:
                            self.marked_failure_for_raised_exception = True
                            raise
                        continue
                    raise

                try:
                    self.router._check_subscription_ratelimit(self.auth_selection)
                except Exception as e:
                    status_code = interpret_status_code(e)
                    if (
                        status_code == 429
                        and self.auth_selection is not None
                        and self.max_auth_attempts > 0
                    ):
                        updated = self.router._mark_auth_failure(self.auth_selection, e)
                        if updated is not None:
                            self._replace_auth_record(updated)
                        self.attempted_auth_ids.add(
                            self.auth_selection.selection.auth.id
                        )
                        self._record_failover()
                        if len(self.attempted_auth_ids) >= self.max_auth_attempts:
                            self.marked_failure_for_raised_exception = True
                            raise
                        continue
                    raise

            try:
                response = do_call()
            except Exception as e:
                status_code = interpret_status_code(e)
                classification = None
                if strategy is not None and hasattr(strategy, "classify_failure"):
                    try:
                        classification = strategy.classify_failure(
                            e, status_code=status_code
                        )
                    except Exception:
                        classification = None
                should_refresh = (
                    classification.is_auth_expired
                    if classification is not None
                    else status_code in (401, 403)
                )
                if (
                    strategy is not None
                    and getattr(strategy, "supports_refresh", True)
                    and self.auth_selection is not None
                    and should_refresh
                    and self.auth_selection.selection.auth.id
                    not in self.refreshed_auth_ids
                ):
                    self.refreshed_auth_ids.add(self.auth_selection.selection.auth.id)
                    try:
                        self.auth_selection = self.router._maybe_refresh_auth(
                            self.auth_selection,
                            strategy,
                            self._request_context(),
                        )
                        self.auth_selection = self.router._prepare_auth_headers(
                            auth_ctx=self.auth_selection,
                            strategy=strategy,
                            model=self.model,
                            kwargs=self.kwargs,
                        )
                    except Exception as refresh_exc:
                        e = refresh_exc
                        status_code = status_code or interpret_status_code(refresh_exc)
                    else:
                        try:
                            response = do_call()
                        except Exception as refresh_error:
                            e = refresh_error
                            status_code = interpret_status_code(refresh_error)
                        else:
                            return response

                should_rotate = (
                    classification.rotatable
                    if classification is not None
                    else self.router._should_rotate_on_auth_error(
                        e,
                        status_code,
                        rotate_on_status_codes=self.rotate_on_status_codes,
                    )
                )
                if (
                    strategy is not None
                    and self.auth_selection is not None
                    and should_rotate
                    and self.max_auth_attempts > 0
                ):
                    cooldown_override = None
                    if classification is not None and classification.cooldown_ms:
                        cooldown_override = timedelta(
                            milliseconds=classification.cooldown_ms
                        )
                    updated = self.router._mark_auth_failure(
                        self.auth_selection, e, cooldown_override=cooldown_override
                    )
                    if updated is not None:
                        self._replace_auth_record(updated)
                    self.attempted_auth_ids.add(self.auth_selection.selection.auth.id)
                    self._record_failover()
                    if len(self.attempted_auth_ids) >= self.max_auth_attempts:
                        self.marked_failure_for_raised_exception = True
                        raise
                    continue
                raise

            return response

    async def run_async(
        self, do_call: Callable[[], Awaitable[ResponseT]]
    ) -> ResponseT:
        while True:
            self._filter_attempted_auth_records()

            self.auth_selection = self.router._maybe_select_auth(
                model=self.model, deployment=self.deployment, kwargs=self.kwargs
            )

            if self.auth_selection is None:
                if self.requires_subscription:
                    self._raise_no_credentials()
                strategy = None
            else:
                self.router._log_auth_identity(self.auth_selection)
                strategy = self.router._auth_strategy_for(
                    self.auth_selection.selection.auth.provider
                )
                try:
                    self.auth_selection = self.router._prepare_auth_headers(
                        auth_ctx=self.auth_selection,
                        strategy=strategy,
                        model=self.model,
                        kwargs=self.kwargs,
                    )
                except Exception as header_error:
                    if self.auth_selection is not None and self.max_auth_attempts > 0:
                        updated = self.router._mark_auth_failure(
                            self.auth_selection, header_error
                        )
                        if updated is not None:
                            self._replace_auth_record(updated)
                        self.attempted_auth_ids.add(
                            self.auth_selection.selection.auth.id
                        )
                        self._record_failover()
                        if len(self.attempted_auth_ids) >= self.max_auth_attempts:
                            self.marked_failure_for_raised_exception = True
                            raise
                        continue
                    raise

                try:
                    self.router._check_subscription_ratelimit(self.auth_selection)
                except Exception as e:
                    status_code = interpret_status_code(e)
                    if (
                        status_code == 429
                        and self.auth_selection is not None
                        and self.max_auth_attempts > 0
                    ):
                        updated = self.router._mark_auth_failure(self.auth_selection, e)
                        if updated is not None:
                            self._replace_auth_record(updated)
                        self.attempted_auth_ids.add(
                            self.auth_selection.selection.auth.id
                        )
                        self._record_failover()
                        if len(self.attempted_auth_ids) >= self.max_auth_attempts:
                            self.marked_failure_for_raised_exception = True
                            raise
                        continue
                    raise

            try:
                response = await do_call()
            except Exception as e:
                status_code = interpret_status_code(e)
                classification = None
                if strategy is not None and hasattr(strategy, "classify_failure"):
                    try:
                        classification = strategy.classify_failure(
                            e, status_code=status_code
                        )
                    except Exception:
                        classification = None
                should_refresh = (
                    classification.is_auth_expired
                    if classification is not None
                    else status_code in (401, 403)
                )
                if (
                    strategy is not None
                    and getattr(strategy, "supports_refresh", True)
                    and self.auth_selection is not None
                    and should_refresh
                    and self.auth_selection.selection.auth.id
                    not in self.refreshed_auth_ids
                ):
                    self.refreshed_auth_ids.add(self.auth_selection.selection.auth.id)
                    try:
                        self.auth_selection = self.router._maybe_refresh_auth(
                            self.auth_selection,
                            strategy,
                            self._request_context(),
                        )
                        self.auth_selection = self.router._prepare_auth_headers(
                            auth_ctx=self.auth_selection,
                            strategy=strategy,
                            model=self.model,
                            kwargs=self.kwargs,
                        )
                    except Exception as refresh_exc:
                        e = refresh_exc
                        status_code = status_code or interpret_status_code(refresh_exc)
                    else:
                        try:
                            response = await do_call()
                        except Exception as refresh_error:
                            e = refresh_error
                            status_code = interpret_status_code(refresh_error)
                        else:
                            return response

                should_rotate = (
                    classification.rotatable
                    if classification is not None
                    else self.router._should_rotate_on_auth_error(
                        e,
                        status_code,
                        rotate_on_status_codes=self.rotate_on_status_codes,
                    )
                )
                if (
                    strategy is not None
                    and self.auth_selection is not None
                    and should_rotate
                    and self.max_auth_attempts > 0
                ):
                    cooldown_override = None
                    if classification is not None and classification.cooldown_ms:
                        cooldown_override = timedelta(
                            milliseconds=classification.cooldown_ms
                        )
                    updated = self.router._mark_auth_failure(
                        self.auth_selection, e, cooldown_override=cooldown_override
                    )
                    if updated is not None:
                        self._replace_auth_record(updated)
                    self.attempted_auth_ids.add(self.auth_selection.selection.auth.id)
                    self._record_failover()
                    if len(self.attempted_auth_ids) >= self.max_auth_attempts:
                        self.marked_failure_for_raised_exception = True
                        raise
                    continue
                raise

            return response
