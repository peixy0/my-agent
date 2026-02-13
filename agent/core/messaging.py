import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import lark_oapi as lark

from agent.core.events import HumanInputEvent
from agent.core.settings import Settings

logger = logging.getLogger(__name__)


class Messaging(ABC):
    """Abstract base for messaging integrations (DIP)."""

    @abstractmethod
    async def run(self) -> None: ...

    @abstractmethod
    async def notify(self, message: str) -> None: ...

    @abstractmethod
    async def send_message(self, session_key: str, message: str) -> None: ...


class NullMessaging(Messaging):
    async def run(self) -> None:
        pass

    async def notify(self, message: str) -> None:
        pass

    async def send_message(self, session_key: str, message: str) -> None:
        pass


@dataclass(frozen=True)
class FeishuMessagingConfig:
    app_id: str
    app_secret: str
    encrypt_key: str
    verification_token: str
    notify_channel_id: str


@dataclass(frozen=True)
class FeishuMessageEvent:
    content: str
    sender_id: str


class FeishuMessaging(Messaging):
    def __init__(
        self, config: FeishuMessagingConfig, event_queue: asyncio.Queue[HumanInputEvent]
    ):
        self._config = config
        self.event_queue = event_queue
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
            logger.info(f"Feishu event received: {content_json}")

            if not (
                data.event.sender
                and data.event.sender.sender_id
                and data.event.sender.sender_id.open_id
            ):
                return
            sender_id = data.event.sender.sender_id.open_id
            text = content_dict.get("text")
            if not text:
                return
            if text:
                asyncio.run_coroutine_threadsafe(
                    self.event_queue.put(
                        HumanInputEvent(session_key=sender_id, message=text)
                    ),
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

    async def notify(self, message: str) -> None:
        """Notify method for sending messages, can be used by the agent to send proactive messages."""
        await self.send_message(self._config.notify_channel_id, message)

    async def send_message(self, session_key: str, message: str) -> None:
        """Send a message to Feishu using the last known sender."""
        request = (
            lark.im.v1.CreateMessageRequest.builder()
            .receive_id_type("open_id")
            .request_body(
                lark.im.v1.CreateMessageRequestBody.builder()
                .receive_id(session_key)
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


def create_messaging(settings: Settings, event_queue: asyncio.Queue) -> Messaging:
    """Create the appropriate messaging backend."""
    if settings.feishu_app_id and settings.feishu_app_secret:
        config = FeishuMessagingConfig(
            app_id=settings.feishu_app_id,
            app_secret=settings.feishu_app_secret,
            encrypt_key=settings.feishu_encrypt_key,
            verification_token=settings.feishu_verification_token,
            notify_channel_id=settings.feishu_notify_channel_id,
        )
        return FeishuMessaging(config, event_queue)
    return NullMessaging()
