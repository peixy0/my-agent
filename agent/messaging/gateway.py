import asyncio

from agent.core.events import AgentEvent
from agent.core.messaging import Gateway
from agent.core.runtime import Runtime
from agent.core.settings import Settings
from agent.messaging.feishu import FeishuConfig, FeishuGateway


def create_gateway(
    settings: Settings,
    event_queue: asyncio.Queue[AgentEvent],
    runtime: Runtime,
) -> Gateway | None:
    """Return the appropriate ``Gateway`` for the current config, or None."""
    if settings.feishu_app_id and settings.feishu_app_secret:
        config = FeishuConfig(
            app_id=settings.feishu_app_id,
            app_secret=settings.feishu_app_secret,
            encrypt_key=settings.feishu_encrypt_key,
            verification_token=settings.feishu_verification_token,
        )
        return FeishuGateway(config, event_queue, runtime)
    return None
