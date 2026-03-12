from __future__ import annotations

import asyncio
import json
import logging
from asyncio.exceptions import CancelledError
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import aiohttp
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    wait_exponential,
)

from agent.llm.types import (
    ChoiceView,
    CompletionResponseView,
    MessageView,
    ToolCallFunctionView,
    ToolCallView,
    UsageView,
)

logger = logging.getLogger(__name__)


class CopilotBadRequestError(Exception):
    pass


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    if value.isdigit():
        timestamp = int(value)
        if timestamp > 10_000_000_000:
            timestamp = timestamp // 1000
        return datetime.fromtimestamp(timestamp, tz=UTC)
    normalized = value.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized)


@dataclass(slots=True)
class GitHubCopilotAuthState:
    github_access_token: str = ""
    github_token_type: str = "bearer"
    github_scope: str = ""
    github_acquired_at: str = ""
    copilot_access_token: str = ""
    copilot_token_expires_at: str = ""

    def has_github_token(self) -> bool:
        return bool(self.github_access_token)

    def has_valid_copilot_token(self, now: datetime | None = None) -> bool:
        if not self.copilot_access_token:
            return False
        expires_at = _parse_datetime(self.copilot_token_expires_at)
        if expires_at is None:
            return True
        current_time = now or _utc_now()
        return expires_at - current_time > timedelta(minutes=1)


class GitHubCopilotAuthStore:
    def __init__(self, state_path: str, project_dir: str):
        base_dir = Path(project_dir).expanduser().resolve()
        self.path = (base_dir / state_path).resolve()

    def load(self) -> GitHubCopilotAuthState:
        if not self.path.exists():
            return GitHubCopilotAuthState()
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        return GitHubCopilotAuthState(**payload)

    def save(self, state: GitHubCopilotAuthState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(asdict(state), indent=2, sort_keys=True),
            encoding="utf-8",
        )


class GitHubCopilotAuthManager:
    def __init__(
        self,
        auth_store: GitHubCopilotAuthStore,
        client_id: str,
        scope: str,
        device_code_url: str,
        access_token_url: str,
        copilot_token_url: str,
        proxy: str,
        user_agent: str,
        editor_version: str,
        editor_plugin_version: str,
    ):
        self._auth_store = auth_store
        self._client_id = client_id
        self._scope = scope
        self._device_code_url = device_code_url
        self._access_token_url = access_token_url
        self._copilot_token_url = copilot_token_url
        self._proxy = proxy or None
        self._user_agent = user_agent
        self._editor_version = editor_version
        self._editor_plugin_version = editor_plugin_version

    async def ensure_copilot_token(self) -> str:
        state = self._auth_store.load()
        if state.has_valid_copilot_token():
            return state.copilot_access_token

        if not state.has_github_token():
            state = await self._login_with_device_code()

        return await self._exchange_copilot_token(state)

    async def invalidate_copilot_token(self) -> None:
        state = self._auth_store.load()
        state.copilot_access_token = ""
        state.copilot_token_expires_at = ""
        self._auth_store.save(state)

    async def _login_with_device_code(self) -> GitHubCopilotAuthState:
        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                self._device_code_url,
                data={"client_id": self._client_id, "scope": self._scope},
                headers={"Accept": "application/json", "User-Agent": self._user_agent},
                proxy=self._proxy,
            ) as response:
                response.raise_for_status()
                device_data = await response.json()

            verification_uri = device_data["verification_uri"]
            user_code = device_data["user_code"]
            interval = int(device_data.get("interval", 5))
            expires_in = int(device_data.get("expires_in", 900))
            device_code = device_data["device_code"]

            logger.warning(
                "GitHub Copilot login required. Open %s and enter code %s",
                verification_uri,
                user_code,
            )

            expiry_time = _utc_now() + timedelta(seconds=expires_in)
            while _utc_now() < expiry_time:
                await asyncio.sleep(interval)
                async with session.post(
                    self._access_token_url,
                    data={
                        "client_id": self._client_id,
                        "device_code": device_code,
                        "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    },
                    headers={
                        "Accept": "application/json",
                        "User-Agent": self._user_agent,
                    },
                    proxy=self._proxy,
                ) as response:
                    response.raise_for_status()
                    token_data = await response.json()

                error_code = token_data.get("error")
                if error_code == "authorization_pending":
                    continue
                if error_code == "slow_down":
                    interval = int(token_data.get("interval", interval + 5))
                    continue
                if error_code:
                    raise RuntimeError(f"GitHub device login failed: {error_code}")

                state = GitHubCopilotAuthState(
                    github_access_token=token_data["access_token"],
                    github_token_type=token_data.get("token_type", "bearer"),
                    github_scope=token_data.get("scope", self._scope),
                    github_acquired_at=_utc_now().isoformat(),
                )
                self._auth_store.save(state)
                return state

        raise TimeoutError(
            "GitHub device login timed out before authorization completed"
        )

    async def _exchange_copilot_token(self, state: GitHubCopilotAuthState) -> str:
        timeout = aiohttp.ClientTimeout(total=60)
        async with (
            aiohttp.ClientSession(timeout=timeout) as session,
            session.get(
                self._copilot_token_url,
                headers={
                    "Accept": "application/json",
                    "Authorization": f"token {state.github_access_token}",
                    "User-Agent": self._user_agent,
                    "editor-version": self._editor_version,
                    "editor-plugin-version": self._editor_plugin_version,
                },
                proxy=self._proxy,
            ) as response,
        ):
            response.raise_for_status()
            payload: dict[str, Any] = await response.json()

        copilot_token = payload.get("token")
        if not copilot_token:
            raise RuntimeError("Copilot token exchange did not return a token")

        expires_at = payload.get("expires_at")
        parsed_expiry = (
            _parse_datetime(str(expires_at)) if expires_at is not None else None
        )
        if parsed_expiry is None:
            parsed_expiry = _utc_now() + timedelta(minutes=25)

        state.copilot_access_token = copilot_token
        state.copilot_token_expires_at = parsed_expiry.isoformat()
        self._auth_store.save(state)
        return copilot_token


class GitHubCopilotProvider:
    def __init__(
        self,
        api_base_url: str,
        auth_manager: GitHubCopilotAuthManager,
        proxy: str,
        user_agent: str,
        editor_version: str,
        editor_plugin_version: str,
        openai_intent: str,
    ):
        self._api_base_url = api_base_url.rstrip("/")
        self._auth_manager = auth_manager
        self._proxy = proxy or None
        self._user_agent = user_agent
        self._editor_version = editor_version
        self._editor_plugin_version = editor_plugin_version
        self._openai_intent = openai_intent

    async def fetch_and_save_models(self, models_path: Path) -> None:
        """Fetch available models from the Copilot API and persist them to disk."""
        token = await self._auth_manager.ensure_copilot_token()
        timeout = aiohttp.ClientTimeout(total=30)
        async with (
            aiohttp.ClientSession(timeout=timeout) as session,
            session.get(
                f"{self._api_base_url}/models",
                headers={
                    "Accept": "application/json",
                    "Authorization": f"Bearer {token}",
                    "User-Agent": self._user_agent,
                    "editor-version": self._editor_version,
                    "editor-plugin-version": self._editor_plugin_version,
                    "openai-intent": self._openai_intent,
                },
                proxy=self._proxy,
            ) as response,
        ):
            response.raise_for_status()
            payload: dict[str, Any] = await response.json()

        models_path.parent.mkdir(parents=True, exist_ok=True)
        models_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        model_ids = [m.get("id") for m in payload.get("data", [])]
        logger.info("Copilot models saved to %s: %s", models_path, model_ids)

    @retry(
        retry=retry_if_not_exception_type((CopilotBadRequestError, CancelledError)),
        wait=wait_exponential(multiplier=2, min=5, max=300),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def do_completion(self, *args: Any, **kwargs: Any) -> Any:
        payload = dict(kwargs)
        if args:
            raise TypeError("GitHubCopilotProvider only supports keyword arguments")

        token = await self._auth_manager.ensure_copilot_token()
        timeout = aiohttp.ClientTimeout(total=600)
        async with (
            aiohttp.ClientSession(timeout=timeout) as session,
            session.post(
                f"{self._api_base_url}/chat/completions",
                json=payload,
                headers={
                    "Accept": "application/json",
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                    "User-Agent": self._user_agent,
                    "editor-version": self._editor_version,
                    "editor-plugin-version": self._editor_plugin_version,
                    "openai-intent": self._openai_intent,
                },
                proxy=self._proxy,
            ) as response,
        ):
            if response.status == 401:
                await self._auth_manager.invalidate_copilot_token()
                raise RuntimeError(
                    "Invalid Copilot token, cleared cached token and will retry"
                )
            if 400 <= response.status < 500:
                body = await response.text()
                raise CopilotBadRequestError(
                    f"Copilot completion failed with status {response.status}: {body}"
                )
            response.raise_for_status()
            body = await response.json()
            completion = self._coerce_response(body)
            if not completion.choices:
                raise RuntimeError("Copilot completion returned no choices")
            return completion

        raise RuntimeError(
            "Copilot completion failed after refreshing the cached token"
        )

    @staticmethod
    def _coerce_response(body: dict[str, Any]) -> CompletionResponseView:
        choices: list[ChoiceView] = []
        for index, choice_payload in enumerate(body.get("choices", [])):
            message_payload = choice_payload.get("message", {})
            raw_tool_calls = message_payload.get("tool_calls") or []
            tool_calls = [
                ToolCallView(
                    id=tc.get("id", f"tool_call_{index}"),
                    type=tc.get("type", "function"),
                    function=ToolCallFunctionView(
                        name=tc.get("function", {}).get("name", ""),
                        arguments=tc.get("function", {}).get("arguments", "{}"),
                    ),
                )
                for tc in raw_tool_calls
            ]
            choices.append(
                ChoiceView(
                    index=choice_payload.get("index", index),
                    finish_reason=choice_payload.get("finish_reason", "stop"),
                    message=MessageView(
                        role=message_payload.get("role", "assistant"),
                        content=message_payload.get("content"),
                        tool_calls=tool_calls,
                    ),
                )
            )

        usage_payload = body.get("usage") or {}
        return CompletionResponseView(
            choices=choices,
            usage=UsageView(
                prompt_tokens=int(usage_payload.get("prompt_tokens", 0) or 0),
                completion_tokens=int(usage_payload.get("completion_tokens", 0) or 0),
                total_tokens=int(usage_payload.get("total_tokens", 0) or 0),
            ),
            model=body.get("model", ""),
        )
