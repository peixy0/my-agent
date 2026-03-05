# pyright: reportOptionalMemberAccess=false, reportAttributeAccessIssue=false

import asyncio
import io
import json
import logging
from dataclasses import dataclass

import lark_oapi as lark

from agent.core.events import ImageInputEvent, TextInputEvent
from agent.core.runtime import Runtime
from agent.messaging.sender import MessageSender, MessageSource

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeishuConfig:
    app_id: str
    app_secret: str
    encrypt_key: str
    verification_token: str


class FeishuSender(MessageSender):
    """Sends outbound messages and reactions to a single Feishu chat / message."""

    def __init__(
        self,
        client: lark.Client,
        runtime: Runtime,
        chat_id: str,
        message_id: str = "",
    ) -> None:
        self._client = client
        self._runtime = runtime
        self._chat_id = chat_id
        self._message_id = message_id

    async def send(self, text: str) -> None:
        request = (
            lark.im.v1.CreateMessageRequest.builder()
            .receive_id_type("chat_id")
            .request_body(
                lark.im.v1.CreateMessageRequestBody.builder()
                .receive_id(self._chat_id)
                .msg_type("text")
                .content(json.dumps({"text": text}))
                .build()
            )
            .build()
        )
        response = await self._client.im.v1.message.acreate(request)
        if not response.success():
            logger.error(
                "Failed to send Feishu message: %s - %s", response.code, response.msg
            )

    async def send_image(self, image_path: str) -> None:
        try:
            content = await self._runtime.read_file_internal(image_path)
        except Exception as e:
            logger.error("Failed to read image file %s: %s", image_path, e)
            raise

        if not content:
            raise Exception("Image file is empty.")
        if len(content) > 1024 * 1024 * 10:
            raise Exception("Image file size too large.")

        image_buffer = io.BytesIO(content)
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
        upload_response = await self._client.im.v1.image.acreate(upload_request)
        if not upload_response.success():
            raise Exception(
                f"Failed to upload image: {upload_response.code} - {upload_response.msg}"
            )

        image_key = upload_response.data.image_key
        request = (
            lark.im.v1.CreateMessageRequest.builder()
            .receive_id_type("chat_id")
            .request_body(
                lark.im.v1.CreateMessageRequestBody.builder()
                .receive_id(self._chat_id)
                .msg_type("image")
                .content(json.dumps({"image_key": image_key}))
                .build()
            )
            .build()
        )
        response = await self._client.im.v1.message.acreate(request)
        if not response.success():
            raise Exception(
                f"Failed to send Feishu image message: {response.code} - {response.msg}"
            )

    async def react(self, emoji: str) -> None:
        if not self._message_id:
            return
        request = (
            lark.im.v1.CreateMessageReactionRequest.builder()
            .message_id(self._message_id)
            .request_body(
                lark.im.v1.CreateMessageReactionRequestBody.builder()
                .reaction_type(lark.im.v1.Emoji.builder().emoji_type(emoji).build())
                .build()
            )
            .build()
        )
        response = await self._client.im.v1.message_reaction.acreate(request)
        if not response.success():
            logger.error("Failed to add reaction: %s - %s", response.code, response.msg)


class FeishuSource(MessageSource):
    """Receives Feishu WebSocket events and enqueues them for the scheduler."""

    def __init__(
        self,
        config: FeishuConfig,
        event_queue: asyncio.Queue,  # type: ignore[type-arg]
        runtime: Runtime,
    ) -> None:
        self._config = config
        self._event_queue = event_queue
        self._runtime = runtime
        self._client = (
            lark.Client.builder()
            .app_id(config.app_id)
            .app_secret(config.app_secret)
            .log_level(lark.LogLevel.WARNING)
            .build()
        )

    def _make_sender(self, chat_id: str, message_id: str = "") -> FeishuSender:
        return FeishuSender(self._client, self._runtime, chat_id, message_id)

    def _get_referenced_message(
        self, message_id: str | None, prefix: str = "> "
    ) -> str:
        if not message_id:
            return ""
        request = lark.im.v1.GetMessageRequest.builder().message_id(message_id).build()
        response = self._client.im.v1.message.get(request)
        if not response.success():
            logger.error(
                "Failed to get Feishu message: %s - %s", response.code, response.msg
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
            logger.error("Failed to parse referenced message: %s", e)
            return ""

    async def _download_and_queue_image(
        self, chat_id: str, message_id: str, image_key: str
    ) -> None:
        try:
            request = (
                lark.im.v1.GetMessageResourceRequest.builder()
                .message_id(message_id)
                .file_key(image_key)
                .type("image")
                .build()
            )
            response = await self._client.im.v1.message_resource.aget(request)
            if not response.success():
                logger.error(
                    "Failed to download Feishu image: %s - %s",
                    response.code,
                    response.msg,
                )
                return
            image_data = response.file.read()
            await self._event_queue.put(
                ImageInputEvent(
                    chat_id=chat_id,
                    message_id=message_id,
                    image_data=image_data,
                    sender=self._make_sender(chat_id, message_id),
                )
            )
        except Exception as e:
            logger.error("Failed to download and queue image: %s", e)

    def _on_message(self, data: lark.im.v1.P2ImMessageReceiveV1) -> None:
        msg_type = data.event.message.message_type or ""
        content_json = data.event.message.content or "{}"
        content_dict = json.loads(content_json)
        logger.info("Feishu event received (type=%s): %s", msg_type, content_json)

        if not data.event.message.chat_id:
            return

        chat_id = data.event.message.chat_id
        message_id = data.event.message.message_id or ""
        loop = asyncio.get_running_loop()

        if msg_type == "image":
            image_key = content_dict.get("image_key", "")
            if not image_key:
                return
            asyncio.run_coroutine_threadsafe(
                self._download_and_queue_image(chat_id, message_id, image_key),
                loop,
            )
            return

        text = content_dict.get("text", "")
        if not text:
            return
        text = self._get_referenced_message(data.event.message.parent_id) + text
        asyncio.run_coroutine_threadsafe(
            self._event_queue.put(
                TextInputEvent(
                    chat_id=chat_id,
                    message_id=message_id,
                    message=text,
                    sender=self._make_sender(chat_id, message_id),
                )
            ),
            loop,
        )

    async def run(self) -> None:
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
