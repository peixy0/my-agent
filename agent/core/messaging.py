import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import aiohttp
import lark_oapi as lark

from agent.core.events import HumanInputEvent

logger = logging.getLogger(__name__)


class Messaging(ABC):
    """Abstract base for messaging integrations (DIP)."""

    @abstractmethod
    async def run(self) -> None: ...

    @abstractmethod
    async def send_message(self, message: str) -> None: ...


@dataclass
class RefreshToken:
    pass


@dataclass
class MessageEvent:
    content: str


MessagingEvent = MessageEvent | RefreshToken


class NullMessaging(Messaging):
    async def run(self) -> None:
        pass

    async def send_message(self, message: str) -> None:
        pass


@dataclass(frozen=True)
class WXMessagingConfig:
    """Explicit config for WXMessaging — no global settings dependency."""

    corpid: str
    corpsecret: str
    agentid: str
    touser: str = "@all"
    token_refresh_interval: int = 3600 * 4


class WXMessaging(Messaging):
    def __init__(self, config: WXMessagingConfig):
        self._config = config
        self.event_queue: asyncio.Queue[MessagingEvent] = asyncio.Queue()
        self.access_token: str | None = None
        self.refresh_task: asyncio.Task[None] | None = None

    async def run(self) -> None:
        self.refresh_task = asyncio.create_task(self._refresh_token_task())
        while True:
            event = await self.event_queue.get()
            if isinstance(event, MessageEvent):
                await self._message_to_human(event.content)
            else:
                await self._refresh_token()

    async def send_message(self, message: str) -> None:
        await self.event_queue.put(MessageEvent(content=message))

    async def _refresh_token_task(self) -> None:
        while True:
            await self.event_queue.put(RefreshToken())
            await asyncio.sleep(self._config.token_refresh_interval)

    async def _refresh_token(self) -> None:
        token_url = "https://qyapi.weixin.qq.com/cgi-bin/gettoken"
        token_params = {
            "corpid": self._config.corpid,
            "corpsecret": self._config.corpsecret,
        }
        async with (
            aiohttp.ClientSession() as session,
            session.get(
                token_url,
                params=token_params,
                timeout=aiohttp.ClientTimeout(total=60.0),
            ) as response,
        ):
            token_data = await response.json()
            if token_data.get("errcode") == 0:
                self.access_token = token_data.get("access_token")
                logger.info("Token refreshed successfully")
            else:
                logger.error(f"Failed to refresh token: {token_data.get('errmsg')}")

    async def _send_raw_text(
        self, session: aiohttp.ClientSession, content: str, continued_count: int = 0
    ) -> None:
        if not self.access_token:
            return

        content = content.strip()
        if not content:
            return

        if continued_count > 0:
            content = f"(...continued part {continued_count})\n\n" + content

        url = "https://qyapi.weixin.qq.com/cgi-bin/message/send"
        params = {"access_token": self.access_token}
        payload = {
            "touser": self._config.touser,
            "msgtype": "text",
            "agentid": int(self._config.agentid),
            "text": {"content": content},
            "safe": 0,
        }
        try:
            async with session.post(
                url,
                params=params,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60.0),
            ) as response:
                data = await response.json()
                if data.get("errcode") != 0:
                    logger.error(f"WeChat send error: {data.get('errmsg')}")
        except Exception as e:
            logger.error(f"WeChat request failed: {e}")

    async def _message_to_human(self, message: str) -> None:
        if not self.access_token or not message:
            return

        lines = message.splitlines()
        batch: list[str] = []
        batch_size = 0
        MAX_BYTES = 800
        continued_count = 0

        try:
            async with aiohttp.ClientSession() as session:
                for line in lines:
                    line_len = len(line.encode("utf-8"))

                    if batch and (batch_size + line_len + len(batch)) > MAX_BYTES:
                        await self._send_raw_text(
                            session, "\n".join(batch), continued_count
                        )
                        batch, batch_size = [], 0
                        continued_count += 1

                    batch.append(line)
                    batch_size += line_len

                    if (batch_size + len(batch) - 1) > MAX_BYTES:
                        await self._send_raw_text(
                            session, "\n".join(batch), continued_count
                        )
                        batch, batch_size = [], 0
                        continued_count += 1

                if batch:
                    await self._send_raw_text(
                        session, "\n".join(batch), continued_count
                    )
        except Exception as e:
            logger.error(f"Failed to process message to human: {e}")


@dataclass(frozen=True)
class FeishuMessagingConfig:
    app_id: str
    app_secret: str
    encrypt_key: str
    verification_token: str


class FeishuMessaging(Messaging):
    def __init__(
        self, config: FeishuMessagingConfig, event_queue: asyncio.Queue[MessagingEvent]
    ):
        self._config = config
        self.event_queue = event_queue
        # Initialize Lark Client for API requests (sending messages)
        self.client = (
            lark.Client.builder()
            .app_id(self._config.app_id)
            .app_secret(self._config.app_secret)
            .log_level(lark.LogLevel.DEBUG)
            .build()
        )

    async def run(self) -> None:
        """Start the WebSocket client to receive messages."""

        def on_message(data: lark.im.v1.P2ImMessageReceiveV1) -> None:
            content_json = data.event.message.content
            content_dict = json.loads(content_json)
            text = content_dict.get("text", "")

            logger.info(f"Feishu message received: {text}")
            # Store the sender's open_id to reply later
            if (
                data.event.sender
                and data.event.sender.sender_id
                and data.event.sender.sender_id.open_id
            ):
                self.last_sender_id = data.event.sender.sender_id.open_id

            if text:
                asyncio.run_coroutine_threadsafe(
                    self.event_queue.put(HumanInputEvent(content=text)),
                    asyncio.get_running_loop(),
                )

        # Initialize WebSocket Client for receiving messages
        ws_client = lark.ws.Client(
            self._config.app_id,
            self._config.app_secret,
            event_handler=lark.EventDispatcherHandler.builder(
                encrypt_key=self._config.encrypt_key,
                verification_token=self._config.verification_token,
            )
            .register_p2_im_message_receive_v1(on_message)
            .build(),
            log_level=lark.LogLevel.DEBUG,
        )

        await asyncio.to_thread(ws_client.start)
        logger.info("Feishu client started")

    async def send_message(self, message: str) -> None:
        """Send a message to Feishu using the last known sender."""
        if not hasattr(self, "last_sender_id") or not self.last_sender_id:
            logger.warning("Cannot send message: No recipient (last sender) found.")
            return

        request = (
            lark.im.v1.CreateMessageRequest.builder()
            .receive_id_type("open_id")
            .request_body(
                lark.im.v1.CreateMessageRequestBody.builder()
                .receive_id(self.last_sender_id)
                .msg_type("text")
                .content(json.dumps({"text": message}))
                .build()
            )
            .build()
        )

        response = await asyncio.to_thread(self.client.im.v1.message.create, request)

        if not response.success():
            logger.error(
                f"Failed to send Feishu message: {response.code} - {response.msg}"
            )
