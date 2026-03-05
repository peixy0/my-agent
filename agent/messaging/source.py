import asyncio

from agent.core.runtime import Runtime
from agent.core.settings import Settings
from agent.messaging.sender import MessageSource, NullSource


def create_message_source(
    settings: Settings,
    event_queue: asyncio.Queue,  # type: ignore[type-arg]
    runtime: Runtime,
) -> MessageSource:
    """Return the appropriate ``MessageSource`` for the current config."""
    if settings.feishu_app_id and settings.feishu_app_secret:
        from agent.messaging.feishu import FeishuConfig, FeishuSource

        config = FeishuConfig(
            app_id=settings.feishu_app_id,
            app_secret=settings.feishu_app_secret,
            encrypt_key=settings.feishu_encrypt_key,
            verification_token=settings.feishu_verification_token,
        )
        return FeishuSource(config, event_queue, runtime)
    return NullSource()
