# pyright: reportOptionalMemberAccess=false

import asyncio
import io
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import override

import lark_oapi as lark

from agent.core.events import AgentEvent, ImageInputEvent, TextInputEvent
from agent.core.messaging import Channel, Gateway
from agent.core.runtime import Runtime
from agent.llm.types import ToolContent
from agent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeishuConfig:
    app_id: str
    app_secret: str
    encrypt_key: str
    verification_token: str


class FeishuChannel(Channel):
    """Sends outbound messages and reactions to a single Feishu chat / message."""

    def __init__(
        self,
        client: lark.Client,
        runtime: Runtime,
        chat_id: str,
        message_id: str = "",
    ) -> None:
        self.client = client
        self.runtime = runtime
        self.chat_id = chat_id
        self.message_id = message_id

    @override
    async def send(self, text: str) -> None:
        card = {
            "schema": "2.0",
            "body": {
                "elements": [
                    {
                        "tag": "markdown",
                        "content": text,
                    }
                ]
            },
        }
        request = (
            lark.im.v1.CreateMessageRequest.builder()
            .receive_id_type("chat_id")
            .request_body(
                lark.im.v1.CreateMessageRequestBody.builder()
                .receive_id(self.chat_id)
                .msg_type("interactive")
                .content(json.dumps(card))
                .build()
            )
            .build()
        )
        response = await self.client.im.v1.message.acreate(request)
        if not response.success():
            logger.error(
                f"Failed to send Feishu message: {response.code} - {response.msg}"
            )

    @override
    async def start_thinking(self) -> None:
        pass

    @override
    async def end_thinking(self) -> None:
        pass

    async def _send_image(self, image_path: str) -> None:
        try:
            content = await self.runtime.read_raw_bytes(image_path)
        except Exception as e:
            logger.error(f"Failed to read image file {image_path}: {e}")
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
        upload_response = await self.client.im.v1.image.acreate(upload_request)
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
                .receive_id(self.chat_id)
                .msg_type("image")
                .content(json.dumps({"image_key": image_key}))
                .build()
            )
            .build()
        )
        response = await self.client.im.v1.message.acreate(request)
        if not response.success():
            raise Exception(
                f"Failed to send Feishu image message: {response.code} - {response.msg}"
            )

    async def _send_file(self, file_path: str) -> None:
        try:
            content = await self.runtime.read_raw_bytes(file_path)
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            raise

        if not content:
            raise Exception("File is empty.")
        if len(content) > 1024 * 1024 * 20:
            raise Exception("File size too large.")

        buffer = io.BytesIO(content)
        upload_request = (
            lark.im.v1.CreateFileRequest.builder()
            .request_body(
                lark.im.v1.CreateFileRequestBody.builder()
                .file_type("stream")
                .file_name(Path(file_path).name)
                .file(buffer)
                .build()
            )
            .build()
        )
        upload_response = await self.client.im.v1.file.acreate(upload_request)
        if not upload_response.success():
            raise Exception(
                f"Failed to upload file: {upload_response.code} - {upload_response.msg}"
            )

        file_key = upload_response.data.file_key
        request = (
            lark.im.v1.CreateMessageRequest.builder()
            .receive_id_type("chat_id")
            .request_body(
                lark.im.v1.CreateMessageRequestBody.builder()
                .receive_id(self.chat_id)
                .msg_type("file")
                .content(json.dumps({"file_key": file_key}))
                .build()
            )
            .build()
        )
        response = await self.client.im.v1.message.acreate(request)
        if not response.success():
            raise Exception(
                f"Failed to send Feishu file message: {response.code} - {response.msg}"
            )

    async def _react(self, emoji: str) -> None:
        if not self.message_id:
            return
        request = (
            lark.im.v1.CreateMessageReactionRequest.builder()
            .message_id(self.message_id)
            .request_body(
                lark.im.v1.CreateMessageReactionRequestBody.builder()
                .reaction_type(lark.im.v1.Emoji.builder().emoji_type(emoji).build())
                .build()
            )
            .build()
        )
        response = await self.client.im.v1.message_reaction.acreate(request)
        if not response.success():
            logger.error(f"Failed to add reaction: {response.code} - {response.msg}")

    @override
    def register_tools(self, registry: ToolRegistry) -> None:
        """Register Feishu-specific tools: reactions, image sending, and file sending."""

        async def add_reaction(emoji: str) -> ToolContent:
            """
            React to the current message with an emoji.
            """
            try:
                await self._react(emoji)
                return ToolContent.from_dict(
                    "success", {"message": f"Added reaction {emoji} to message"}
                )
            except Exception as e:
                logger.error(f"Failed to add reaction {emoji}: {e}", exc_info=True)
                return ToolContent.from_dict("error", {"message": str(e)})

        registry.register(
            add_reaction,
            {
                "type": "object",
                "properties": {
                    "emoji": {
                        "type": "string",
                        "enum": [
                            "OK",
                            "THUMBSUP",
                            "MUSCLE",
                            "LOL",
                            "THINKING",
                            "Shrug",
                            "Fire",
                            "Coffee",
                            "PARTY",
                            "CAKE",
                            "HEART",
                        ],
                        "description": "The emoji type to react with. OK, THUMBSUP, MUSCLE, LOL, THINKING, Shrug, Fire, Coffee, PARTY, CAKE, HEART",
                    }
                },
                "required": ["emoji"],
            },
        )

        async def send_image(image_path: str) -> ToolContent:
            """
            Send an image file to the user. Image file size must be under 10 MiB.
            """
            try:
                await self._send_image(image_path)
                return ToolContent.from_dict(
                    "success",
                    {"message": f"Sent image {image_path} to user"},
                )
            except Exception as e:
                return ToolContent.from_dict("error", {"message": str(e)})

        registry.register(
            send_image,
            {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Absolute path to the image file to send.",
                    }
                },
                "required": ["image_path"],
            },
        )

        async def send_file(file_path: str) -> ToolContent:
            """
            Send a file to the user. File size must be under 20 MiB.
            Send only when explicitly asked.
            """
            try:
                await self._send_file(file_path)
                return ToolContent.from_dict(
                    "success",
                    {"message": f"Sent file {file_path} to user"},
                )
            except Exception as e:
                return ToolContent.from_dict("error", {"message": str(e)})

        registry.register(
            send_file,
            {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the file to send.",
                    }
                },
                "required": ["file_path"],
            },
        )


class FeishuGateway(Gateway):
    """Receives Feishu WebSocket events and enqueues them for the scheduler."""

    def __init__(
        self,
        config: FeishuConfig,
        event_queue: asyncio.Queue[AgentEvent],
        runtime: Runtime,
    ) -> None:
        self.config = config
        self.event_queue = event_queue
        self.runtime = runtime
        self.client = (
            lark.Client.builder()
            .app_id(config.app_id)
            .app_secret(config.app_secret)
            .log_level(lark.LogLevel.WARNING)
            .build()
        )

    def _make_channel(self, chat_id: str, message_id: str = "") -> FeishuChannel:
        return FeishuChannel(self.client, self.runtime, chat_id, message_id)

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
            response = await self.client.im.v1.message_resource.aget(request)
            if not response.success():
                logger.error(
                    f"Failed to download Feishu image: {response.code} - {response.msg}"
                )
                return
            image_data = response.file.read()
            await self.event_queue.put(
                ImageInputEvent(
                    chat_id=chat_id,
                    message_id=message_id,
                    image_data=image_data,
                    sender=self._make_channel(chat_id, message_id),
                )
            )
        except Exception as e:
            logger.error(f"Failed to download and queue image: {e}")

    def _on_message(self, data: lark.im.v1.P2ImMessageReceiveV1) -> None:
        msg_type = data.event.message.message_type or ""
        content_json = data.event.message.content or "{}"
        content_dict = json.loads(content_json)
        logger.info(f"Feishu event received (type={msg_type}): {content_json}")

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
        asyncio.run_coroutine_threadsafe(
            self.event_queue.put(
                TextInputEvent(
                    chat_id=chat_id,
                    message_id=message_id,
                    message=text,
                    sender=self._make_channel(chat_id, message_id),
                )
            ),
            loop,
        )

    @override
    async def run(self) -> None:
        ws_client = lark.ws.Client(
            self.config.app_id,
            self.config.app_secret,
            event_handler=lark.EventDispatcherHandler.builder(
                encrypt_key=self.config.encrypt_key,
                verification_token=self.config.verification_token,
            )
            .register_p2_im_message_receive_v1(self._on_message)
            .build(),
            log_level=lark.LogLevel.DEBUG,
        )
        logger.info("Feishu client running")
        await asyncio.to_thread(ws_client.start)
