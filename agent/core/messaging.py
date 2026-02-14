# pyright: reportOptionalMemberAccess=false

import asyncio
import io
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import lark_oapi as lark

from agent.core.events import HumanInputEvent
from agent.core.runtime import Runtime
from agent.core.settings import Settings

logger = logging.getLogger(__name__)


class MessageSender(ABC):
    @abstractmethod
    async def send_message(self, session_id: str, message: str) -> None: ...


class MessageNotifier(ABC):
    @abstractmethod
    async def notify(self, message: str) -> None: ...


class MessageReactor(ABC):
    @abstractmethod
    async def add_reaction(self, message_id: str, emoji_type: str) -> None: ...


class MessageImageSender(ABC):
    @abstractmethod
    async def send_image(self, session_id: str, image_path: str) -> None: ...


class MessageSource(ABC):
    @abstractmethod
    async def run(self) -> None: ...


class Messaging(
    MessageSender, MessageNotifier, MessageReactor, MessageImageSender, MessageSource
):
    """Abstract base for messaging integrations (DIP)."""

    pass


class NullMessaging(Messaging):
    async def run(self) -> None:
        pass

    async def notify(self, message: str) -> None:
        pass

    async def send_message(self, session_id: str, message: str) -> None:
        pass

    async def add_reaction(self, message_id: str, emoji_type: str) -> None:
        pass

    async def send_image(self, session_id: str, image_path: str) -> None:
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
        self,
        config: FeishuMessagingConfig,
        event_queue: asyncio.Queue[HumanInputEvent],
        runtime: Runtime,
    ):
        self._config = config
        self.event_queue = event_queue
        self.runtime = runtime
        self.client = (
            lark.Client.builder()
            .app_id(self._config.app_id)
            .app_secret(self._config.app_secret)
            .log_level(lark.LogLevel.WARNING)
            .build()
        )

    def _get_referenced_message(
        self, message_id: str | None, prefix: str = "> "
    ) -> str:
        if not message_id:
            return ""

        request = lark.im.v1.GetMessageRequest.builder().message_id(message_id).build()
        response = self.client.im.v1.message.get(request)
        if not response.success():
            logger.error(
                f"Failed to get Feishu message: {response.code} - {response.msg}"
            )
            return ""
        try:
            content = ""
            items = response.data.items or []
            for item in items:
                parent_content = self._get_referenced_message(
                    item.parent_id, prefix + prefix
                )
                content_json = item.body.content or "{}"
                content_dict = json.loads(content_json)
                lines = content_dict.get("text", "").splitlines()
                content = f"{prefix}QUOTE:\n{prefix}\n" + "\n".join(
                    [prefix + line for line in lines]
                )
                content = parent_content + content
            if content:
                content = content + "\n"
            return content
        except Exception as e:
            logger.error(f"Failed to get Feishu message: {e}")
            return ""

    def _on_message(self, data: lark.im.v1.P2ImMessageReceiveV1) -> None:
        content_json = data.event.message.content or "{}"
        content_dict = json.loads(content_json)
        logger.info(f"Feishu event received: {content_json}")

        if not (
            data.event.sender
            and data.event.sender.sender_id
            and data.event.sender.sender_id.open_id
        ):
            return
        text = content_dict.get("text", "")
        if not text:
            return

        sender_id = data.event.sender.sender_id.open_id
        message_id = data.event.message.message_id or ""
        text = self._get_referenced_message(data.event.message.parent_id) + text
        asyncio.run_coroutine_threadsafe(
            self.event_queue.put(
                HumanInputEvent(
                    session_id=sender_id, message_id=message_id, message=text
                )
            ),
            asyncio.get_running_loop(),
        )

    async def run(self) -> None:
        """Start the WebSocket client to receive messages."""

        # Initialize WebSocket Client for receiving messages
        ws_client = lark.ws.Client(
            self._config.app_id,
            self._config.app_secret,
            event_handler=lark.EventDispatcherHandler.builder(
                encrypt_key=self._config.encrypt_key,
                verification_token=self._config.verification_token,
            )
            .register_p2_im_message_receive_v1(self._on_message)
            .build(),
            log_level=lark.LogLevel.DEBUG,
        )

        logger.info("Feishu client running")
        await asyncio.to_thread(ws_client.start)

    async def notify(self, message: str) -> None:
        """Notify method for sending messages, can be used by the agent to send proactive messages."""
        await self.send_message(self._config.notify_channel_id, message)

    async def send_message(self, session_id: str, message: str) -> None:
        """Send a message to Feishu using the last known sender."""
        request = (
            lark.im.v1.CreateMessageRequest.builder()
            .receive_id_type("open_id")
            .request_body(
                lark.im.v1.CreateMessageRequestBody.builder()
                .receive_id(session_id)
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

    async def add_reaction(self, message_id: str, emoji_type: str) -> None:
        """Add a reaction to a message."""
        request = (
            lark.im.v1.CreateMessageReactionRequest.builder()
            .message_id(message_id)
            .request_body(
                lark.im.v1.CreateMessageReactionRequestBody.builder()
                .reaction_type(
                    lark.im.v1.Emoji.builder().emoji_type(emoji_type).build()
                )
                .build()
            )
            .build()
        )

        response = await asyncio.to_thread(
            self.client.im.v1.message_reaction.create, request
        )
        if not response.success():
            logger.error(f"Failed to add reaction: {response.code} - {response.msg}")
            raise Exception(f"Failed to add reaction: {response.code} - {response.msg}")

    async def send_image(self, session_id: str, image_path: str) -> None:
        """Send an image to Feishu."""
        try:
            content = await self.runtime.read_file_internal(image_path)
        except Exception as e:
            logger.error(f"Failed to read image file {image_path}: {e}")
            raise e

        if not content:
            raise Exception("Image file is empty.")

        if len(content) > 1024 * 1024 * 10:
            raise Exception("Image file size too large.")
        image_buffer = io.BytesIO(content)

        # Upload image
        upload_request = (
            lark.im.v1.CreateImageRequest.builder()
            .request_body(
                lark.im.v1.CreateImageRequestBody.builder()
                .image_type("message")
                .image(image_buffer)
                .build()
            )
            .build()
        )

        upload_response = await asyncio.to_thread(
            self.client.im.v1.image.create, upload_request
        )
        if not upload_response.success():
            logger.error(
                f"Failed to upload image: {upload_response.code} - {upload_response.msg}"
            )
            raise Exception(
                f"Failed to upload image: {upload_response.code} - {upload_response.msg}"
            )

        image_key = upload_response.data.image_key

        # Send image message
        request = (
            lark.im.v1.CreateMessageRequest.builder()
            .receive_id_type("open_id")
            .request_body(
                lark.im.v1.CreateMessageRequestBody.builder()
                .receive_id(session_id)
                .msg_type("image")
                .content(json.dumps({"image_key": image_key}))
                .build()
            )
            .build()
        )

        response = await asyncio.to_thread(self.client.im.v1.message.create, request)
        if not response.success():
            logger.error(
                f"Failed to send Feishu image message: {response.code} - {response.msg}"
            )
            raise Exception(
                f"Failed to send Feishu image message: {response.code} - {response.msg}"
            )


def create_messaging(
    settings: Settings, event_queue: asyncio.Queue, runtime: Runtime
) -> Messaging:
    """Create the appropriate messaging backend."""
    if settings.feishu_app_id and settings.feishu_app_secret:
        config = FeishuMessagingConfig(
            app_id=settings.feishu_app_id,
            app_secret=settings.feishu_app_secret,
            encrypt_key=settings.feishu_encrypt_key,
            verification_token=settings.feishu_verification_token,
            notify_channel_id=settings.feishu_notify_channel_id,
        )
        return FeishuMessaging(config, event_queue, runtime)
    return NullMessaging()
