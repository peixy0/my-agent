import asyncio
from dataclasses import dataclass

import aiohttp
from settings import settings


@dataclass
class RefreshToken:
    pass


@dataclass
class MessageEvent:
    content: str


MessagingEvent = MessageEvent | RefreshToken


class NullMessaging:
    async def run(self):
        pass

    async def send_message(self, _: str):
        pass


class Messaging:
    def __init__(self):
        self.event_queue: asyncio.Queue[MessagingEvent] = asyncio.Queue()
        self.access_token: str | None = None
        self.refresh_task: asyncio.Task[None] | None = None

    async def run(self):
        self.refresh_task = asyncio.create_task(self._refresh_token_task())
        while True:
            event = await self.event_queue.get()
            if isinstance(event, MessageEvent):
                await self._message_to_human(event.content)
            else:
                await self._refresh_token()

    async def send_message(self, message: str):
        await self.event_queue.put(MessageEvent(content=message))

    async def _refresh_token_task(self):
        while True:
            await self.event_queue.put(RefreshToken())
            await asyncio.sleep(settings.wechat_token_refresh_interval)

    async def _refresh_token(self):
        token_url = "https://qyapi.weixin.qq.com/cgi-bin/gettoken"
        token_params = {
            "corpid": settings.wechat_corpid,
            "corpsecret": settings.wechat_corpsecret,
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

    async def _message_to_human(self, message: str):
        """
        Send a message to human.
        Only use this when you are explicitly asked to respond to human.
        """
        if not self.access_token:
            return

        try:
            async with aiohttp.ClientSession() as session:
                send_url = f"https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token={self.access_token}"
                message_data = {
                    "touser": settings.wechat_touser,
                    "msgtype": "text",
                    "agentid": int(settings.wechat_agentid),
                    "text": {"content": message},
                    "safe": 0,
                }
                async with session.post(
                    send_url,
                    json=message_data,
                    timeout=aiohttp.ClientTimeout(total=60.0),
                ) as response:
                    send_data = await response.json()
                    if send_data.get("errcode") != 0:
                        return {"status": "error", "message": "Failed to send message"}
                    return {"status": "success", "message": "Message sent."}
        except Exception:
            pass


messaging = NullMessaging() if settings.mute_agent else Messaging()
