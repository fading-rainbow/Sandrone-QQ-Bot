from __future__ import annotations

import asyncio
import logging

from .config import Settings
from .image_gen import ImageGenerator
from .llm import LLMClient
from .memory import MemoryStore
from .qq_adapter import QQBotRunner
from .service import ChatService


async def main() -> None:
    settings = Settings.from_env()
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # httpx includes full presigned QQ upload URLs in INFO logs. Those URLs contain
    # short-lived credentials and must never be persisted in journald.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    memory = MemoryStore(settings.database_path)
    interrupted_jobs = memory.interrupt_running_image_jobs()
    if interrupted_jobs:
        logging.getLogger(__name__).warning(
            "已将 %d 个上次进程遗留的生图任务标记为中断", interrupted_jobs
        )
    llm = LLMClient(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.openai_model,
        api_mode=settings.openai_api_mode,
        reasoning_effort=settings.reasoning_effort,
        max_output_tokens=settings.max_output_tokens,
    )
    service = ChatService(
        memory,
        llm,
        system_prompt=settings.system_prompt,
        history_messages=settings.history_messages,
        summary_trigger_messages=settings.summary_trigger_messages,
        summary_batch_messages=settings.summary_batch_messages,
        profile_trigger_messages=settings.profile_trigger_messages,
        web_search_enabled=settings.web_search_enabled,
        web_search_cache_seconds=settings.web_search_cache_seconds,
        web_search_user_cooldown_seconds=settings.web_search_user_cooldown_seconds,
        web_search_group_cooldown_seconds=settings.web_search_group_cooldown_seconds,
    )
    image_generator = ImageGenerator(
        api_key=settings.image_api_key,
        base_url=settings.image_base_url,
        model=settings.image_model,
        output_dir=settings.generated_image_dir,
        sandrone_reference_paths=settings.sandrone_reference_paths,
    )
    runner = QQBotRunner(
        app_id=settings.qq_app_id,
        app_secret=settings.qq_app_secret,
        service=service,
        image_generator=image_generator,
        image_cooldown_seconds=settings.image_cooldown_seconds,
        owner_ids=settings.owner_ids,
        allowed_group_ids=settings.allowed_group_ids,
    )
    try:
        await runner.run()
    finally:
        await service.close()
        await llm.close()
        await image_generator.close()
        memory.close()


def run() -> None:
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except ValueError as exc:
        raise SystemExit(f"配置错误: {exc}") from exc


if __name__ == "__main__":
    run()
