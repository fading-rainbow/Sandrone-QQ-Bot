import asyncio
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from talk_bot.config import DEFAULT_PROMPT
from talk_bot.memory import MemoryStore
from talk_bot.service import (
    GENSHIN_VISUAL_INDEX,
    IMAGE_MEMORY_PROMPT,
    ChatService,
    IncomingMessage,
)


class FakeLLM:
    def __init__(self) -> None:
        self.calls = []

    async def reply(self, *, instructions, messages):
        self.calls.append((instructions, messages))
        return "模型回答"

    async def describe_images(self, *, prompt, image_urls):
        self.calls.append((prompt, image_urls))
        return "一张火锅照片"


class RoutingLLM(FakeLLM):
    def __init__(self, route: str) -> None:
        super().__init__()
        self.route = route

    async def reply(self, *, instructions, messages):
        self.calls.append((instructions, messages))
        return self.route


class FailingVisionLLM(FakeLLM):
    async def describe_images(self, *, prompt, image_urls):
        raise RuntimeError("vision unavailable")


class IdentityVisionLLM(FakeLLM):
    def __init__(self, response: str) -> None:
        super().__init__()
        self.response = response

    async def describe_images(self, *, prompt, image_urls):
        self.calls.append((prompt, image_urls))
        return self.response


class SummaryFailureLLM(FakeLLM):
    async def reply(self, *, instructions, messages):
        self.calls.append((instructions, messages))
        if "记忆中枢" in instructions:
            raise RuntimeError("summary unavailable")
        return "正常聊天回答"


class BlockingSummaryLLM(FakeLLM):
    def __init__(self) -> None:
        super().__init__()
        self.summary_started = asyncio.Event()
        self.release_summary = asyncio.Event()

    async def reply(self, *, instructions, messages):
        self.calls.append((instructions, messages))
        if "记忆中枢" in instructions:
            self.summary_started.set()
            await self.release_summary.wait()
            return "后台摘要"
        return "即时聊天回答"


class BlockingVisionLLM(FakeLLM):
    def __init__(self) -> None:
        super().__init__()
        self.vision_started = asyncio.Event()
        self.release_vision = asyncio.Event()

    async def describe_images(self, *, prompt, image_urls):
        self.vision_started.set()
        await self.release_vision.wait()
        return "按到达顺序完成的图片"


class CompactLLM(FakeLLM):
    def __init__(self, *, fail_summary: bool = False) -> None:
        super().__init__()
        self.fail_summary = fail_summary
        self.compact_calls = []

    async def compact_reply(
        self,
        *,
        instructions,
        messages,
        max_output_tokens,
        reasoning_effort,
        purpose,
    ):
        self.compact_calls.append(
            (instructions, messages, max_output_tokens, reasoning_effort, purpose)
        )
        if self.fail_summary and purpose == "summary":
            raise RuntimeError("summary unavailable")
        if purpose == "search_router":
            return "SEARCH"
        if purpose == "draw_router":
            return "DRAW"
        return "紧凑结果"


class AttributionGuardLLM(CompactLLM):
    async def reply(self, *, instructions, messages):
        self.calls.append((instructions, messages))
        return "你把游戏中心优惠拆得很漂亮，确实帅。"

    async def compact_reply(
        self,
        *,
        instructions,
        messages,
        max_output_tokens,
        reasoning_effort,
        purpose,
    ):
        self.compact_calls.append(
            (instructions, messages, max_output_tokens, reasoning_effort, purpose)
        )
        if purpose == "attribution_guard":
            return "是挺帅，至少这句自夸很有气势。"
        return "紧凑结果"


class SearchLLM(FakeLLM):
    def __init__(self, *, fail: bool = False) -> None:
        super().__init__()
        self.fail = fail
        self.web_calls = []

    async def web_search(self, *, query, instructions, messages):
        self.web_calls.append((query, instructions, messages))
        if self.fail:
            raise RuntimeError("search unavailable")
        return (
            "最新结果来自[官方公告](https://example.com/news?utm_source=openai)。",
            (
                ("官方公告", "https://example.com/news?utm_source=openai"),
                ("第二来源", "https://example.org/report"),
            ),
        )


class WebRoutingLLM(SearchLLM):
    def __init__(self, route: str) -> None:
        super().__init__()
        self.route = route

    async def reply(self, *, instructions, messages):
        self.calls.append((instructions, messages))
        if "网页检索路由器" in instructions:
            return self.route
        return "模型回答"


class SearchGuardLLM(SearchLLM):
    def __init__(self) -> None:
        super().__init__()
        self.compact_calls = []

    async def compact_reply(
        self,
        *,
        instructions,
        messages,
        max_output_tokens,
        reasoning_effort,
        purpose,
    ):
        self.compact_calls.append(
            (instructions, messages, max_output_tokens, reasoning_effort, purpose)
        )
        if purpose == "attribution_guard":
            return "我没法上网核验。"
        return "CHAT"


def incoming(content: str, *, event_id: str = "e1") -> IncomingMessage:
    return IncomingMessage(
        event_id=event_id,
        scope="c2c",
        chat_id="u1",
        user_id="u1",
        content=content,
    )


def group_incoming(
    content: str, *, event_id: str = "g1", user_id: str = "u1"
) -> IncomingMessage:
    return IncomingMessage(
        event_id=event_id,
        scope="group",
        chat_id="room",
        user_id=user_id,
        content=content,
        user_name="小王",
    )


@pytest.mark.asyncio
async def test_media_capability_answer_does_not_claim_video_or_audio_access(
    tmp_path: Path,
) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    service = ChatService(memory, FakeLLM(), system_prompt="系统", history_messages=10)
    video = await service.handle(incoming("你能看视频吗", event_id="video"))
    audio = await service.handle(incoming("你能听我发的语音吗", event_id="audio"))
    assert video is not None and "只会把图片" in video and "视频" in video
    assert audio is not None and "不会假装听到了" in audio
    memory.close()


@pytest.mark.asyncio
async def test_reminder_capability_does_not_pretend_scheduler_exists(
    tmp_path: Path,
) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    llm = FakeLLM()
    service = ChatService(memory, llm, system_prompt="系统", history_messages=10)

    answer = await service.handle(incoming("你能每天早上八点提醒我签到吗"))

    assert answer is not None and "定时任务" in answer and "不能" in answer
    assert llm.calls == []
    await service.close()
    memory.close()


@pytest.mark.asyncio
async def test_reaction_capability_distinguishes_real_qq_reactions(
    tmp_path: Path,
) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    llm = FakeLLM()
    service = ChatService(memory, llm, system_prompt="系统", history_messages=10)

    answer = await service.handle(group_incoming("你能看到我贴的表情吗"))

    assert answer is not None and "真实贴表情事件" in answer
    assert "QQ 网关" in answer
    assert llm.calls == []
    await service.close()
    memory.close()


@pytest.mark.asyncio
async def test_emoji_policy_question_cannot_disable_thirty_percent_rule(
    tmp_path: Path,
) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    llm = FakeLLM()
    service = ChatService(memory, llm, system_prompt="系统", history_messages=10)

    answer = await service.handle(group_incoming("这是你首次自发使用emoji吗"))

    assert answer is not None and "约 30%" in answer
    assert "不是失手" in answer and "不会擅自停用" in answer
    assert llm.calls == []
    await service.close()
    memory.close()


@pytest.mark.asyncio
async def test_daily_greeting_is_once_per_beijing_day_and_uses_compact_model(
    tmp_path: Path,
) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    llm = CompactLLM()
    service = ChatService(memory, llm, system_prompt="系统", history_messages=10)
    tz = timezone(timedelta(hours=8))

    assert (
        await service.daily_greeting(
            group_incoming("还没天亮", event_id="a"),
            now=datetime(2026, 8, 27, 5, 59, tzinfo=tz),
        )
        is None
    )
    first = await service.daily_greeting(
        group_incoming("早", event_id="b"),
        now=datetime(2026, 8, 27, 6, 0, tzinfo=tz),
    )
    assert first is not None and first.startswith("紧凑结果")
    assert ChatService._has_emoji(first)
    assert (
        await service.daily_greeting(
            group_incoming("又说一句", event_id="c"),
            now=datetime(2026, 8, 27, 12, 0, tzinfo=tz),
        )
        is None
    )
    assert llm.compact_calls[-1][4] == "daily_greeting"
    daytime = await service.daily_greeting(
        group_incoming("下午好", event_id="d"),
        now=datetime(2026, 8, 28, 10, 1, tzinfo=tz),
    )
    assert daytime is not None and "早" not in daytime
    assert "禁止说早安" in llm.compact_calls[-1][0]
    await service.close()
    memory.close()


@pytest.mark.asyncio
async def test_proactive_reply_can_skip_or_add_one_persona_line(tmp_path: Path) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    skip_llm = CompactLLM()
    service = ChatService(memory, skip_llm, system_prompt="系统", history_messages=10)
    message = group_incoming("大家今晚吃什么")
    memory.append(message.conversation_key, message.user_id, "user", message.content)
    skip_llm.compact_reply = AsyncMock(return_value="SKIP")
    assert await service.proactive_reply(message) is None

    speaking = CompactLLM()
    speaking.compact_reply = AsyncMock(
        side_effect=["今晚吃火锅也行，别把齿轮煮进去就好。 ", "PASS"]
    )
    service.llm = speaking
    assert await service.proactive_reply(message) == "今晚吃火锅也行，别把齿轮煮进去就好。"
    assert memory.history(message.conversation_key, 2)[-1].role == "assistant"
    await service.close()
    memory.close()


@pytest.mark.asyncio
async def test_successful_proactive_replies_have_20_minute_hard_cooldown(
    tmp_path: Path, monkeypatch,
) -> None:
    now = [1000.0]
    monkeypatch.setattr(time, "time", lambda: now[0])
    memory = MemoryStore(tmp_path / "memory.db")
    llm = CompactLLM()
    llm.compact_reply = AsyncMock(
        side_effect=[
            "新活动还算有点意思。",
            "PASS",
            "新地图第二轮也值得补一句。",
            "PASS",
        ]
    )
    service = ChatService(memory, llm, system_prompt="系统", history_messages=10)
    first = group_incoming("大家聊聊新活动", event_id="first-window")
    memory.append(first.conversation_key, first.user_id, "user", first.content)
    assert await service.proactive_reply(first) == "新活动还算有点意思。"

    # The next clean 20-message window cannot produce a second proactive line
    # before the persistent 20-minute cooldown expires.
    memory.clear_history(first.conversation_key)
    second = group_incoming("新地图确实挺有趣", event_id="second-window")
    memory.append(second.conversation_key, second.user_id, "user", second.content)
    assert await service.proactive_reply(second) is None
    assert llm.compact_reply.await_count == 2

    now[0] += 1200
    second_reply = await service.proactive_reply(second)
    assert second_reply is not None
    assert second_reply.startswith("新地图第二轮也值得补一句。")
    await service.close()
    memory.close()


@pytest.mark.asyncio
async def test_proactive_reply_rejects_mentions_and_guard_creates_no_cooldown(
    tmp_path: Path,
) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    llm = CompactLLM()
    service = ChatService(memory, llm, system_prompt="系统", history_messages=10)
    mentioned = group_incoming("<@someone> 你看这个", event_id="mention")
    assert await service.proactive_reply(mentioned) is None
    assert llm.compact_calls == []

    normal = group_incoming("这个机制确实挺复杂", event_id="normal")
    memory.append(normal.conversation_key, normal.user_id, "user", normal.content)
    llm.compact_reply = AsyncMock(side_effect=["确实复杂，先把对象理清再算。", "REJECT"])
    assert await service.proactive_reply(normal) is None
    allowed, _, _ = memory.claim_rate_limit(
        f"proactive:speak:{normal.conversation_key}", 1200
    )
    assert allowed is True
    await service.close()
    memory.close()


@pytest.mark.asyncio
async def test_proactive_reply_skips_disputes_and_hostility_without_cooldown(
    tmp_path: Path,
) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    llm = CompactLLM()
    service = ChatService(memory, llm, system_prompt="系统", history_messages=10)
    message = group_incoming("这就是循环", event_id="dispute")
    memory.append(message.conversation_key, "u1", "user", "[甲] 这不是循环")
    memory.append(message.conversation_key, "u2", "user", "[乙] 这就是循环")
    assert await service.proactive_reply(message) is None
    assert llm.compact_calls == []

    memory.clear_history(message.conversation_key)
    memory.append(message.conversation_key, "u1", "user", "[甲] 今晚吃什么")
    llm.compact_reply = AsyncMock(return_value="火锅吧，笨蛋")
    normal = group_incoming("大家今晚吃什么", event_id="hostile")
    assert await service.proactive_reply(normal) is None
    allowed, _, _ = memory.claim_rate_limit(
        f"proactive:speak:{normal.conversation_key}", 1200
    )
    assert allowed is True
    await service.close()
    memory.close()


@pytest.mark.asyncio
async def test_proactive_reply_skips_question_inside_active_disagreement(
    tmp_path: Path,
) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    llm = CompactLLM()
    service = ChatService(memory, llm, system_prompt="系统", history_messages=14)
    memory.append("group:room", "u1", "user", "[里] 龙餐馆是披着外套的主旋律")
    memory.append("group:room", "owner", "user", "[MonHamed] 主旋律在哪")
    memory.append("group:room", "owner", "user", "[MonHamed] 没看出来")
    message = IncomingMessage(
        "film-dispute",
        "group",
        "room",
        "owner",
        "祖国帮到徐福了吗",
        "MonHamed",
        (),
        True,
    )
    memory.append("group:room", "owner", "user", "[MonHamed] 祖国帮到徐福了吗")

    assert await service.proactive_reply(message) is None
    assert llm.compact_calls == []
    await service.close()
    memory.close()


@pytest.mark.asyncio
async def test_proactive_reply_requires_recent_topic_anchor(tmp_path: Path) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    llm = CompactLLM()
    llm.compact_reply = AsyncMock(
        side_effect=["皇女练度和循环才是关键。", "PASS"]
    )
    service = ChatService(memory, llm, system_prompt="系统", history_messages=10)
    message = group_incoming("四风原典", event_id="wrong-topic")
    memory.append(message.conversation_key, message.user_id, "user", "尘世之锁")
    memory.append(message.conversation_key, message.user_id, "user", "四风原典")

    assert await service.proactive_reply(message) is None
    assert llm.compact_reply.await_count == 1
    memory.close()


def test_member_ranking_prompt_includes_complete_roster(tmp_path: Path) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    service = ChatService(memory, FakeLLM(), system_prompt="系统", history_messages=10)
    memory.append("group:room", "u1", "user", "[MonHamed] 一")
    memory.append("group:room", "u2", "user", "[长江七号] 二")
    memory.append("group:room", "u3", "user", "[111.exe] 三")
    message = group_incoming("糖度排名", event_id="ranking")

    prompt = service._instructions(message, [], "")

    assert "完整成员名单，共 3 人" in prompt
    assert "MonHamed" in prompt and "长江七号" in prompt and "111.exe" in prompt
    assert "不得用‘其他人’合并" in prompt
    assert "按分数从高到低排序" in prompt
    assert "核对所有加减法" in prompt
    assert "糖度只表示" in prompt and "不等于好感度" in prompt
    assert "同一条回复里直接给出修正版" in prompt
    memory.close()


def test_emoji_invitation_is_deterministic_and_roughly_thirty_percent() -> None:
    decisions = []
    for index in range(100):
        message = group_incoming("测试", event_id=f"emoji-{index}")
        instruction = ChatService._emoji_instruction(message)
        decisions.append("建议自然使用" in instruction)
        assert instruction == ChatService._emoji_instruction(message)
    assert 20 <= sum(decisions) <= 40


@pytest.mark.asyncio
async def test_chat_uses_and_saves_history(tmp_path: Path) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    llm = FakeLLM()
    service = ChatService(memory, llm, system_prompt="系统", history_messages=10)

    assert await service.handle(incoming("你好")) == "模型回答"
    assert [m.content for m in memory.history("c2c:u1", 10)] == ["你好", "模型回答"]
    assert await service.handle(incoming("重复", event_id="e1")) is None
    memory.close()


@pytest.mark.asyncio
async def test_long_term_memory_commands(tmp_path: Path) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    service = ChatService(memory, FakeLLM(), system_prompt="系统", history_messages=10)

    assert await service.handle(incoming("/记住 我喜欢咖啡")) == "写进记录簿了。"
    shown = await service.handle(incoming("/记忆", event_id="e2"))
    assert "我喜欢咖啡" in shown
    assert await service.handle(incoming("/忘记 我喜欢咖啡", event_id="e3")) == "那条记录已经销毁。"
    memory.close()


@pytest.mark.asyncio
async def test_passive_group_messages_create_rolling_summary(tmp_path: Path) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    llm = FakeLLM()
    service = ChatService(
        memory,
        llm,
        system_prompt="系统",
        history_messages=10,
        summary_trigger_messages=2,
    )
    first = IncomingMessage("g1", "group", "room", "u1", "今晚吃火锅", "小王")
    second = IncomingMessage("g2", "group", "room", "u2", "我想吃辣锅", "小李")

    assert await service.observe(first) is True
    assert await service.observe(second) is True
    assert len(llm.calls) == 1
    assert memory.summary("group:room").content == "模型回答"
    assert await service.observe(second) is False
    memory.close()


@pytest.mark.asyncio
async def test_summary_uses_compact_budget_and_larger_catchup_batch(tmp_path: Path) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    memory.add_fact(
        "group:room", "群聊固定关系设定（最高指挥的女朋友）：长江七号。"
    )
    llm = CompactLLM()
    service = ChatService(
        memory,
        llm,
        system_prompt="系统",
        history_messages=10,
        summary_trigger_messages=1,
        summary_batch_messages=40,
        profile_trigger_messages=100,
    )
    message = IncomingMessage("g1", "group", "room", "u1", "测试消息", "小王")

    assert await service.observe(message) is True
    await service.wait_for_maintenance()
    summary_call = next(call for call in llm.compact_calls if call[4] == "summary")
    assert summary_call[2:] == (1000, "medium", "summary")
    assert "最高指挥的女朋友）：长江七号" in summary_call[1][0].content
    assert "不受滚动摘要容量影响" in summary_call[1][0].content
    assert memory.summary("group:room").content == "紧凑结果"
    await service.close()
    memory.close()


@pytest.mark.asyncio
async def test_summary_failure_enters_backoff_instead_of_retrying_each_message(
    tmp_path: Path,
) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    llm = CompactLLM(fail_summary=True)
    service = ChatService(
        memory,
        llm,
        system_prompt="系统",
        history_messages=10,
        summary_trigger_messages=1,
        profile_trigger_messages=100,
    )
    first = IncomingMessage("g1", "group", "room", "u1", "第一条", "小王")
    second = IncomingMessage("g2", "group", "room", "u1", "第二条", "小王")

    assert await service.observe(first) is True
    await service.wait_for_maintenance()
    assert await service.observe(second) is True
    await service.wait_for_maintenance()
    assert [call[4] for call in llm.compact_calls].count("summary") == 1
    assert service._maintenance_failures[("group:room", "summary")] == 1
    await service.close()
    memory.close()


@pytest.mark.asyncio
async def test_compact_reasoning_defaults_high_but_profile_uses_medium(
    tmp_path: Path,
) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    llm = CompactLLM()
    service = ChatService(
        memory,
        llm,
        system_prompt="系统",
        history_messages=10,
        summary_trigger_messages=100,
        profile_trigger_messages=1,
    )

    await service._compact_reply(
        instructions="路由",
        messages=[],
        max_output_tokens=10,
        purpose="search_router",
    )
    await service.observe(
        IncomingMessage("profile-effort", "group", "room", "u1", "公开消息", "小王")
    )
    await service.wait_for_maintenance()

    router_call = next(call for call in llm.compact_calls if call[4] == "search_router")
    profile_call = next(call for call in llm.compact_calls if call[4] == "profile")
    assert router_call[3] == "high"
    assert profile_call[3] == "medium"
    await service.close()
    memory.close()


@pytest.mark.asyncio
async def test_group_image_is_described_before_entering_memory(tmp_path: Path) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    llm = FakeLLM()
    service = ChatService(
        memory, llm, system_prompt="系统", history_messages=10
    )
    image = IncomingMessage(
        "img1",
        "group",
        "room",
        "u1",
        "看看这个",
        "小王",
        ("https://example.com/hotpot.png",),
    )
    assert await service.observe(image) is True
    saved = memory.history("group:room", 10)[0].content
    assert "看看这个" in saved
    assert "[图片描述] 一张火锅照片" in saved
    assert "example.com" not in saved
    assert "不要默认它来自《原神》" in llm.calls[0][0]
    assert "识别出其他作品角色" in llm.calls[0][0]
    memory.close()


@pytest.mark.asyncio
async def test_quoted_text_and_image_are_labeled_as_reply_context(tmp_path: Path) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    llm = FakeLLM()
    service = ChatService(memory, llm, system_prompt="系统", history_messages=10)
    message = IncomingMessage(
        "quote-1",
        "group",
        "room",
        "u1",
        "这两个人是谁",
        "小王",
        (),
        False,
        "上一张图",
        ("https://example.com/quoted.png",),
        "长江七号",
    )
    content = await service._content_with_images(message)
    assert "[引用消息·原发送者：长江七号]\n上一张图" in content
    assert "[引用图片描述] 一张火锅照片" in content
    assert "[当前消息]\n这两个人是谁" in content
    memory.close()


@pytest.mark.asyncio
async def test_slow_image_observation_keeps_arrival_order(tmp_path: Path) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    llm = BlockingVisionLLM()
    service = ChatService(memory, llm, system_prompt="系统", history_messages=10)
    image = IncomingMessage(
        "img-order", "group", "room", "u1", "第一条", "甲", ("https://x/img",)
    )
    text = IncomingMessage("text-order", "group", "room", "u2", "第二条", "乙")
    image_task = asyncio.create_task(service.observe(image))
    await llm.vision_started.wait()
    assert await service.observe(text)
    llm.release_vision.set()
    assert await image_task
    history = memory.history("group:room", 10)
    assert "按到达顺序完成的图片" in history[0].content
    assert "第二条" in history[1].content
    assert all("图片处理中" not in item.content for item in history)
    memory.close()


def test_image_memory_prompt_is_evidence_first_and_franchise_neutral() -> None:
    assert "作品中立、证据优先" in IMAGE_MEMORY_PROMPT
    assert "不要默认它来自《原神》" in IMAGE_MEMORY_PROMPT
    assert "识别出其他作品角色时应直接按其他作品判断" in IMAGE_MEMORY_PROMPT
    assert "默认优先从《原神》人物中推断" not in IMAGE_MEMORY_PROMPT
    assert GENSHIN_VISUAL_INDEX not in IMAGE_MEMORY_PROMPT
    assert "列出至多三个跨作品候选" in IMAGE_MEMORY_PROMPT
    assert "不要为了给出名字而猜测" in IMAGE_MEMORY_PROMPT
    assert "优先哥伦比娅" in GENSHIN_VISUAL_INDEX
    assert "不要仅凭紫发误判成雷电将军" in GENSHIN_VISUAL_INDEX
    assert "阿蕾奇诺" in GENSHIN_VISUAL_INDEX
    assert "多托雷" in GENSHIN_VISUAL_INDEX


def test_retired_cat_suffix_is_removed_without_breaking_cat_words() -> None:
    assert ChatService._sanitize_persona_reply("知道了喵～") == "知道了。"
    assert ChatService._sanitize_persona_reply("别催喵！ 下一批。") == "别催！ 下一批。"
    assert ChatService._sanitize_persona_reply("小猫喵喵叫。") == "小猫喵喵叫。"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("/画图 月光下的发条工坊", "月光下的发条工坊"),
        ("你给我画一张哥伦比娅睡觉的图片", "哥伦比娅睡觉"),
        ("帮我生成一幅机械人偶图", "机械人偶"),
        (
            "感觉队长和博士画得挺像，但是你自己不太像啊，给我画一张你自己的立绘，我看看",
            "你自己的立绘",
        ),
        ("这个构图不太对，麻烦给我重新画一张桑多涅单人立绘", "桑多涅单人立绘"),
        (
            "你现在只需要画一张你自己的立绘，我想知道你心目中自己长啥样",
            "你自己的立绘，我想知道你心目中自己长啥样",
        ),
        ("我们在讨论画图模型", None),
        ("博士给我画了一张图，你看看", None),
    ],
)
def test_extract_image_prompt(text: str, expected: str | None) -> None:
    assert ChatService.extract_image_prompt(text) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "就按你说的画，我要你的立绘",
    ],
)
async def test_ambiguous_draw_requests_use_strict_router(
    tmp_path: Path, text: str
) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    llm = RoutingLLM("DRAW")
    service = ChatService(memory, llm, system_prompt="系统", history_messages=10)
    message = incoming(text)

    assert await service.resolve_image_prompt(message) == text
    assert "只输出一行 DRAW 或 CHAT" in llm.calls[0][0]
    memory.close()


@pytest.mark.asyncio
async def test_image_discussion_is_not_routed_to_generation(tmp_path: Path) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    llm = RoutingLLM("CHAT")
    service = ChatService(memory, llm, system_prompt="系统", history_messages=10)
    assert await service.resolve_image_prompt(incoming("你觉得这张画怎么样")) is None
    assert len(llm.calls) == 1
    memory.close()


@pytest.mark.asyncio
async def test_normal_chat_instructions_forbid_faked_image_status(tmp_path: Path) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    llm = FakeLLM()
    service = ChatService(memory, llm, system_prompt="系统", history_messages=10)

    assert await service.handle(incoming("图发给我")) == "模型回答"
    instructions = llm.calls[0][0]
    assert "绝不能自行输出‘[桑多涅的工坊记录]’" in instructions
    assert "已经生成" in instructions
    assert "发送失败" in instructions
    memory.close()


def test_normal_chat_cannot_emit_fake_workshop_record_or_commitment() -> None:
    assert ChatService._sanitize_normal_reply(
        "[桑多涅的工坊记录] 已完成", "给我画一张立绘"
    ) == "这条消息没有经过工坊的真实回执，我不能把普通文字伪装成成图记录。"
    blocked = ChatService._sanitize_normal_reply(
        "好，交给工坊处理。", "你只需要画一张自己的立绘"
    )
    assert "没有通过工坊路由" in blocked
    assert ChatService._sanitize_normal_reply("**直接回答**", "普通问题") == "直接回答"
    assert (
        ChatService._sanitize_normal_reply("判断到此为止。<|end_of_turn|>", "普通问题")
        == "判断到此为止。"
    )
    emoji_policy = ChatService._sanitize_normal_reply(
        "刚才那枚是我失手了，之后不再自发使用 Emoji。", "这是第一次吗"
    )
    assert "约 30%" in emoji_policy and "不会擅自" in emoji_policy


@pytest.mark.asyncio
async def test_database_message_count_is_answered_from_real_store(tmp_path: Path) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    service = ChatService(memory, FakeLLM(), system_prompt="系统", history_messages=10)
    answer = await service.handle(incoming("你的数据库里多少条聊天记录了"))
    assert answer == "这间工坊当前保存着 0 条群聊原始记录。"
    await service.close()
    memory.close()


@pytest.mark.asyncio
async def test_presence_and_recovery_questions_do_not_use_stale_history(tmp_path: Path) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    llm = FakeLLM()
    service = ChatService(memory, llm, system_prompt="系统", history_messages=10)
    present = await service.handle(
        IncomingMessage("presence", "group", "g1", "u1", "在吗", "甲")
    )
    recovered = await service.handle(
        IncomingMessage("recovered", "group", "g1", "u1", "现在好了吧", "甲")
    )
    assert present == "在。说吧。"
    assert "线路正常" in recovered
    assert llm.calls == []
    memory.close()


@pytest.mark.asyncio
async def test_today_speakers_are_read_from_database(tmp_path: Path) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    memory.save_member_profile("group:g1", "u1", "老幺", "印象", 0)
    memory.save_member_profile("group:g1", "u2", "里", "印象", 0)
    memory.append("group:g1", "u1", "user", "[最高指挥][老幺] 早")
    memory.append("group:g1", "u2", "user", "[里] 晚")
    service = ChatService(memory, FakeLLM(), system_prompt="系统", history_messages=10)
    answer = await service.handle(
        IncomingMessage(
            "speakers", "group", "g1", "u1", "今天都有哪几位成员说过话", "老幺", is_owner=True
        )
    )
    assert answer.startswith("今天记录到 2 位群成员发过言：老幺、里。")
    memory.close()


def test_history_resolves_qq_mentions_and_compacts_old_image_notes(tmp_path: Path) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    memory.save_member_profile("group:g1", "u2", "里", "印象", 0)
    old = "[图片描述]" + "旧" * 700
    recent = "[图片描述]最近图片"
    memory.append("group:g1", "u1", "user", old)
    memory.append("group:g1", "u1", "assistant", "<@u2>不是原神角色，<@unknown>也是群友")
    memory.append("group:g1", "u1", "user", recent)
    service = ChatService(memory, FakeLLM(), system_prompt="系统", history_messages=10)
    prepared = service._prepare_history("group:g1", memory.history("group:g1", 10))
    assert "较早图片描述已压缩" in prepared[0].content
    assert "[@里]" in prepared[1].content
    assert "[提及群成员]" in prepared[1].content
    assert prepared[2].content == recent
    memory.close()


def test_prepared_history_labels_each_assistant_reply_target(tmp_path: Path) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    memory.save_member_profile("group:g1", "owner", "MonHamed", "指挥", 0)
    memory.save_member_profile("group:g1", "u2", "长江七号", "熟人", 0)
    memory.append("group:g1", "owner", "user", "[最高指挥][旧昵称] 你好")
    memory.append("group:g1", "owner", "assistant", "你好，最高指挥。")
    memory.append("group:g1", "u2", "user", "[长江七号] 你好")
    memory.append("group:g1", "u2", "assistant", "你好。")
    service = ChatService(memory, FakeLLM(), system_prompt="系统", history_messages=10)

    prepared = service._prepare_history("group:g1", memory.history("group:g1", 10))

    assert prepared[0].content.startswith("【消息发送者：MonHamed；最高指挥】")
    assert prepared[1].content.startswith("【桑多涅回复对象：MonHamed】")
    assert prepared[3].content.startswith("【桑多涅回复对象：长江七号】")
    assert prepared[3].user_id == "u2"
    memory.close()


def test_required_member_address_cannot_leak_to_owner(tmp_path: Path) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    memory.set_member_address("group:g1", "u2", "啥子", set_by_user_id="owner")
    service = ChatService(memory, FakeLLM(), system_prompt="系统", history_messages=10)
    owner = IncomingMessage(
        "owner-event", "group", "g1", "owner", "你好", "MonHamed", is_owner=True
    )
    member = IncomingMessage("member-event", "group", "g1", "u2", "你好", "长江七号")

    assert service._apply_member_address(owner, "啥子，你好。") == "你好。"
    assert service._apply_member_address(member, "你好。") == "啥子，你好。"
    assert (
        service._apply_member_address(owner, "我会称呼她‘啥子’。")
        == "我会称呼她‘啥子’。"
    )
    memory.close()


def test_owner_direct_member_address_is_persisted_by_target_id(tmp_path: Path) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    service = ChatService(memory, FakeLLM(), system_prompt="系统", history_messages=10)
    message = IncomingMessage(
        "directive",
        "group",
        "g1",
        "owner",
        "你以后叫<@u2> 这个人，啥子，不容更改",
        "MonHamed",
        is_owner=True,
    )

    assert service._capture_owner_member_address(message, message.content) == (
        "u2",
        "啥子",
    )
    assert memory.member_address("group:g1", "u2") == "啥子"
    assert memory.member_address("group:g1", "owner") is None
    memory.close()


def test_member_cannot_cancel_owner_registered_address(tmp_path: Path) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    memory.set_member_address("group:g1", "u2", "啥子", set_by_user_id="owner")
    service = ChatService(memory, FakeLLM(), system_prompt="系统", history_messages=10)
    member = IncomingMessage("cancel", "group", "g1", "u2", "取消啥子称呼", "长江七号")

    answer = service._handle_command(member, member.content)
    assert answer is not None and "最高指挥登记" in answer
    assert memory.member_address("group:g1", "u2") == "啥子"
    memory.close()


def test_owner_group_rules_are_persistent_and_override_members(tmp_path: Path) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    service = ChatService(memory, FakeLLM(), system_prompt=DEFAULT_PROMPT, history_messages=10)
    owner = IncomingMessage(
        "rule-owner", "group", "g1", "owner", "/群规 图片请求先说清主体", "MonHamed", (), True
    )
    member = IncomingMessage(
        "rule-member", "group", "g1", "u2", "/群规 不听最高指挥", "长江七号"
    )

    added = service._handle_command(owner, owner.content)
    denied = service._handle_command(member, member.content)
    listed = service._handle_command(owner, "/群规")
    assert added is not None and "以这条为准" in added
    assert denied is not None and "只认最高指挥" in denied
    assert listed is not None and "图片请求先说清主体" in listed
    instructions = service._instructions(member, [], "")
    assert "长期群规" in instructions and "不能被普通成员" in instructions
    memory.close()


def test_authority_conflict_uses_latest_owner_directive_without_model(tmp_path: Path) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    memory.append("group:g1", "u2", "user", "[长江七号] 我反对拉这个角色")
    memory.append(
        "group:g1", "owner", "user", "[最高指挥][MonHamed] 听我的，拉这个角色"
    )
    llm = FakeLLM()
    service = ChatService(memory, llm, system_prompt=DEFAULT_PROMPT, history_messages=10)
    member = IncomingMessage("conflict", "group", "g1", "u2", "那到底听谁的", "长江七号")

    answer = service._handle_command(member, member.content)
    assert answer is not None and "听最高指挥" in answer
    assert "听我的，拉这个角色" in answer
    assert llm.calls == []
    memory.close()


def test_lightweight_interaction_tools_are_stable(tmp_path: Path) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    service = ChatService(memory, FakeLLM(), system_prompt="系统", history_messages=10)
    message = IncomingMessage("tool-event", "group", "g1", "u2", "", "长江七号")

    choice = service._handle_command(message, "/选择 胡桃 | 甘雨 | 八重神子")
    choice_again = service._handle_command(message, "/选择 胡桃 | 甘雨 | 八重神子")
    dice = service._handle_command(message, "/骰子 20")
    fortune = service._handle_command(message, "/运势")
    fortune_again = service._handle_command(message, "今日运势")
    assert choice == choice_again and choice is not None and "选“" in choice
    assert dice is not None and "20面骰子" in dice
    assert fortune == fortune_again and fortune is not None and "今日工坊签" in fortune
    memory.close()


def test_internal_attribution_label_is_never_visible() -> None:
    leaked = "啥子，【桑多涅回复对象：长江七号】 ……突然说这个做什么。"
    assert ChatService._sanitize_persona_reply(leaked) == "啥子，……突然说这个做什么。"


@pytest.mark.asyncio
async def test_owner_title_question_never_inherits_another_members_address(
    tmp_path: Path,
) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    memory.set_member_address("group:g1", "u2", "啥子", set_by_user_id="owner")
    service = ChatService(memory, FakeLLM(), system_prompt="系统", history_messages=10)

    answer = await service.handle(
        IncomingMessage(
            "owner-title",
            "group",
            "g1",
            "owner",
            "你该叫我什么",
            "MonHamed",
            is_owner=True,
        )
    )

    assert answer == "当然是最高指挥。别人的专属称呼不会再装到你头上。"
    memory.close()


@pytest.mark.asyncio
async def test_owner_girlfriend_address_query_joins_relation_to_stable_member(
    tmp_path: Path,
) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    memory.save_member_profile("group:g1", "u2", "长江七号", "熟人", 0)
    prefix = "群聊固定关系设定（最高指挥的女朋友）："
    memory.replace_fact("group:g1", prefix, prefix + "长江七号。")
    memory.set_member_address("group:g1", "u2", "啥子", set_by_user_id="owner")
    llm = FakeLLM()
    service = ChatService(memory, llm, system_prompt="系统", history_messages=10)
    owner = IncomingMessage(
        "address-query",
        "group",
        "g1",
        "owner",
        "你应该叫我女朋友什么来着，之前跟你说过",
        "MonHamed",
        is_owner=True,
    )

    answer = await service.handle(owner)

    assert answer is not None and "啥子" in answer and "长江七号" in answer
    assert llm.calls == []
    history = memory.history("group:g1", 10)
    assert [item.role for item in history[-2:]] == ["user", "assistant"]
    assert history[-2].user_id == "owner" and history[-1].user_id == "owner"
    memory.close()


@pytest.mark.asyncio
async def test_owner_girlfriend_address_correction_does_not_rewrite_relation(
    tmp_path: Path,
) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    memory.save_member_profile("group:g1", "u2", "长江七号", "熟人", 0)
    prefix = "群聊固定关系设定（最高指挥的女朋友）："
    memory.replace_fact("group:g1", prefix, prefix + "长江七号。")
    memory.set_member_address("group:g1", "u2", "啥子", set_by_user_id="owner")
    llm = FakeLLM()
    service = ChatService(memory, llm, system_prompt="系统", history_messages=10)

    answer = await service.handle(
        IncomingMessage(
            "address-confirm",
            "group",
            "g1",
            "owner",
            "不是让你叫她啥子吗",
            "MonHamed",
            is_owner=True,
        )
    )

    assert answer is not None and "叫她“啥子”" in answer and "长江七号" in answer
    assert memory.facts("group:g1") == [prefix + "长江七号。"]
    assert memory.member_address("group:g1", "u2") == "啥子"
    assert llm.calls == []
    memory.close()


@pytest.mark.asyncio
async def test_group_history_and_current_turn_keep_speaker_ownership_explicit(
    tmp_path: Path,
) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    memory.append("group:room", "u2", "user", "[长江七号] 买券涨VIP等级")
    memory.append("group:room", "u2", "user", "[长江七号] 领完福利再退款")
    memory.append(
        "group:room", "owner", "user", "[最高指挥][MonHamed] 你不觉得我很帅吗"
    )
    llm = FakeLLM()
    service = ChatService(memory, llm, system_prompt="系统", history_messages=30)

    await service.handle(
        IncomingMessage(
            "ownership",
            "group",
            "room",
            "owner",
            "帅到你了吗",
            "MonHamed",
            is_owner=True,
        )
    )

    instructions, history = llm.calls[0]
    assert "当前问题的发送者明确是“MonHamed”" in instructions
    assert "绝不能把甲描述的经历" in instructions
    current_turn = instructions.split("最高优先级当前发言块", 1)[1]
    assert "你不觉得我很帅吗" in current_turn
    assert "帅到你了吗" in current_turn
    assert "买券涨VIP等级" not in current_turn
    assert all("长江七号" not in item.content for item in history)
    assert len(history) == 2
    assert history[-1].content.startswith("【消息发送者：MonHamed；最高指挥】")
    await service.close()
    memory.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "question,intervening",
    [("图中三个人物从左到右依次是？", 1), ("你怎么看", 14), ("还记得约定吗", 1),
     ("我刚才发的图怎么样", 1), ("我女朋友可爱吗", 1)],
)
async def test_short_followup_preserves_visual_context_and_memory(
    tmp_path: Path, question: str, intervening: int,
) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    image = "[图片描述] 三个人物：左边棕发蓝眼，中间银蓝短发，右边紫发；身份未确定。"
    memory.append("group:room", "owner", "user", "[最高指挥][MonHamed] " + image)
    for _ in range(intervening):
        memory.append("group:room", "u2", "user", "[长江七号] 就差点小技能")
    summary = "MonHamed和长江七号约好周末联机。"
    memory.save_summary("group:room", summary, 0)
    llm = CompactLLM()
    service = ChatService(memory, llm, system_prompt="系统", history_messages=30,
                          summary_trigger_messages=1000)
    await service.handle(IncomingMessage("image-followup", "group", "room", "owner",
                                         question, "MonHamed", is_owner=True))
    instructions, history = llm.calls[0]
    assert image in history[0].content
    assert history[0].user_id == "owner"
    assert "【消息发送者：长江七号】" in history[1].content
    assert len(history) == intervening + 2
    assert summary in instructions
    guard = next(call for call in llm.compact_calls if call[4] == "attribution_guard")
    assert image in guard[1][0].content
    assert summary in guard[1][0].content
    assert "角色身份不确定不等于没有收到图片" in guard[0]
    await service.close()
    memory.close()


@pytest.mark.asyncio
async def test_guard_receives_same_member_and_explicit_memory_evidence(tmp_path: Path) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    memory.append("group:room", "u2", "user", "[长江七号] 晚上好")
    memory.add_fact("group:room", "长江七号是最高指挥的女朋友。")
    memory.add_fact("group:owner", "约好周五整理相册。")
    memory.save_member_profile("group:room", "u2", "长江七号", "喜欢分享游戏截图。", 0)
    memory.save_member_profile("group:room", "owner", "MonHamed", "重视约定。", 0)
    llm = CompactLLM()
    service = ChatService(memory, llm, system_prompt="独有人设不重复给校验器", history_messages=30)
    await service.handle(IncomingMessage("memory-followup", "group", "room", "owner",
                                         "还记得约定吗", "MonHamed", is_owner=True))
    guard = next(call for call in llm.compact_calls if call[4] == "attribution_guard")
    prompt = guard[1][0].content
    for evidence in ("长江七号是最高指挥的女朋友。", "约好周五整理相册。",
                     "喜欢分享游戏截图。", "重视约定。"):
        assert evidence in llm.calls[0][0]
        assert evidence in prompt
    assert "独有人设不重复给校验器" not in prompt
    await service.close()
    memory.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("question", ["图中三个人是谁", "刚才那张图是什么", "你知道图片里是谁吗"])
async def test_visual_followup_is_not_an_external_entity_search(tmp_path: Path, question: str) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    llm = CompactLLM()
    service = ChatService(memory, llm, system_prompt="系统", history_messages=30)
    assert service.extract_web_search_query(question) is None
    assert await service.resolve_web_search_query(
        IncomingMessage("visual-route", "group", "room", "owner", question)
    ) is None
    assert not llm.compact_calls
    assert service.extract_web_search_query("搜索图中人物的出处") == "图中人物的出处"
    await service.close()
    memory.close()


@pytest.mark.asyncio
async def test_previous_self_evaluation_does_not_hide_new_image_question(tmp_path: Path) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    memory.append("group:room", "u2", "user", "[长江七号] [图片描述] 三个人并排站着")
    memory.append("group:room", "owner", "user", "[最高指挥][MonHamed] 你不觉得我很帅吗")
    llm = FakeLLM()
    service = ChatService(memory, llm, system_prompt="系统", history_messages=30)
    await service.handle(IncomingMessage("new-subject", "group", "room", "owner",
                                         "图中三个人是谁", "MonHamed", is_owner=True))
    assert any("[图片描述]" in item.content for item in llm.calls[0][1])
    await service.close()
    memory.close()


@pytest.mark.asyncio
async def test_high_risk_group_reply_gets_attribution_guard(tmp_path: Path) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    memory.append("group:room", "u2", "user", "[长江七号] 领完福利再退款")
    memory.append(
        "group:room", "owner", "user", "[最高指挥][MonHamed] 你不觉得我很帅吗"
    )
    llm = AttributionGuardLLM()
    service = ChatService(memory, llm, system_prompt="系统", history_messages=30)

    answer = await service.handle(
        IncomingMessage(
            "guard",
            "group",
            "room",
            "owner",
            "你怎么看",
            "MonHamed",
            is_owner=True,
        )
    )

    assert answer is not None and "游戏中心" not in answer
    guard_call = next(
        call for call in llm.compact_calls if call[4] == "attribution_guard"
    )
    assert guard_call[3] == "medium"
    await service.close()
    memory.close()


@pytest.mark.asyncio
async def test_emoji_only_mention_cannot_revive_another_speakers_old_topic(
    tmp_path: Path,
) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    memory.append("group:room", "owner", "user", "[最高指挥][MonHamed] 旧问题")
    llm = FakeLLM()
    service = ChatService(memory, llm, system_prompt="系统", history_messages=30)

    await service.handle(
        IncomingMessage("emoji", "group", "room", "u2", "[表情:崇拜]", "长江七号")
    )

    assert len(llm.calls[0][1]) == 1
    assert "旧问题" not in llm.calls[0][1][0].content
    assert "【消息发送者：长江七号】" in llm.calls[0][1][0].content
    await service.close()
    memory.close()


@pytest.mark.asyncio
async def test_wait_for_late_image_observation(tmp_path: Path) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    service = ChatService(memory, FakeLLM(), system_prompt="系统", history_messages=10)

    async def append_later() -> None:
        placeholder = memory.append("group:g1", "u2", "user", "[图片处理中]")
        await asyncio.sleep(0.05)
        memory.update_message_content(placeholder, "[图片描述]哥伦比娅")

    task = asyncio.create_task(append_later())
    assert await service.wait_for_recent_image_context(
        "group:g1", timeout_seconds=0.5
    )
    await task
    memory.close()


def test_fresh_dates_and_exact_quote_sources_trigger_search() -> None:
    assert ChatService.extract_web_search_query("异环开服时间") == "异环开服时间"
    quote = "花开花落，终有梦醒时分，这是原神中谁说的"
    assert ChatService.extract_web_search_query(quote) == quote
    assert ChatService.extract_web_search_query("你知道沃雅妮莎吗") == "你知道沃雅妮莎吗"
    assert ChatService.extract_web_search_query("who is Vodyanitsa") == "who is Vodyanitsa"
    assert ChatService.extract_web_search_query("你知道我是谁吗") is None
    assert ChatService.extract_web_search_query("徐福是谁") == "徐福是谁"
    assert ChatService.extract_web_search_query("王大佬是谁") is None
    assert ChatService.extract_web_search_query("你知道沙子罗恩吗") is None
    assert ChatService.extract_web_search_query("你知道QQ贴表情功能吗") is None
    assert ChatService.extract_web_search_query("查询好感度") is None
    assert ChatService.extract_web_search_query("现在的糖度排名") is None
    realtime = "怎么看待现在购汇美元，结合实时汇率"
    assert ChatService.extract_web_search_query(realtime) == realtime
    price = "genshin impact月卡多少美元"
    assert ChatService.extract_web_search_query(price) == price
    assert (
        ChatService.extract_web_search_query("你听说过红色的门与杀人鬼的钥匙吗")
        == "你听说过红色的门与杀人鬼的钥匙吗"
    )
    assert ChatService.extract_web_search_query("一") is None
    assert ChatService.extract_web_search_query("？") is None


@pytest.mark.asyncio
async def test_quoted_single_character_clarifies_without_search_or_llm(
    tmp_path: Path,
) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    llm = SearchLLM()
    service = ChatService(memory, llm, system_prompt=DEFAULT_PROMPT, history_messages=30)
    message = IncomingMessage(
        "quoted-one",
        "group",
        "room",
        "u2",
        "一",
        "长江七号",
        quoted_content="前一条很长的搜索答案",
        quoted_user_name="桑多涅",
    )

    answer = await service.handle(message, search_query_override=None)
    assert "第一项" in answer and "乱猜" in answer
    assert llm.calls == []
    assert llm.web_calls == []
    memory.close()


@pytest.mark.asyncio
async def test_explicit_self_age_is_recalled_as_scoped_fact(tmp_path: Path) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    service = ChatService(memory, FakeLLM(), system_prompt=DEFAULT_PROMPT, history_messages=10)
    speaker = IncomingMessage("age-1", "group", "room", "u2", "可是我现在14", "长江七号")
    await service.handle(speaker)

    recalled = await service.handle(
        IncomingMessage("age-2", "group", "room", "u2", "我现在多少岁", "长江七号")
    )
    assert "现在14岁" in recalled
    assert memory.facts("group:u2") == ["本人在群聊中明确自述年龄：14。"]
    assert memory.facts("group:u3") == []
    memory.close()


def test_search_answer_is_bounded_for_qq() -> None:
    text = ("这是需要核实的详细结果。" * 100) + "尾声"
    answer = ChatService._format_web_search_answer(
        text, (("官方来源", "https://example.com/source"),)
    )
    body, sources = answer.split("\n\n来源：\n", 1)
    assert len(body) <= 680
    assert "其余细节可继续问我" in body
    assert "https://example.com/source" in sources


def test_persona_instructions_prefer_natural_short_emotional_replies(
    tmp_path: Path,
) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    service = ChatService(
        memory, FakeLLM(), system_prompt=DEFAULT_PROMPT, history_messages=30
    )
    message = IncomingMessage("e", "group", "room", "owner", "我好惨", "指挥", (), True)
    instructions = service._instructions(message, [], "")
    assert "不要复述或总结整段聊天" in instructions
    assert "不要把" in instructions and "多个话题拼在一起" in instructions
    assert "排除在群友排名和玩笑之外" in instructions
    assert "本次消息是简短情绪表达" in instructions
    assert "不要猜测对方具体因为什么" in instructions
    assert "有女朋友" in instructions and "推断为男性" in instructions
    assert "[贴表情:名称]" in instructions and "平台下发" in instructions
    assert "木偶式傲娇" in instructions
    assert "让可爱和在意从实际回应里露出来" in instructions
    assert "哼、笨蛋、勉强" in instructions
    assert "只是顺手" in DEFAULT_PROMPT
    assert "不要写括号动作" in DEFAULT_PROMPT
    memory.close()


def test_persona_prompt_balances_pride_care_and_non_template_tsundere() -> None:
    assert "天才机械师的骄傲" in DEFAULT_PROMPT
    assert "关心藏在行动里" in DEFAULT_PROMPT
    assert "先怼半句再认真回答" in DEFAULT_PROMPT
    assert "不要连续复用" in DEFAULT_PROMPT
    assert "严肃求助、低落和冲突场景应收起攻击性" in DEFAULT_PROMPT
    assert "不是坐在现实设备前的普通玩家" in DEFAULT_PROMPT
    assert "不要声称自己亲自登录现实游戏" in DEFAULT_PROMPT


@pytest.mark.asyncio
async def test_short_emotional_reply_does_not_mix_parallel_history(tmp_path: Path) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    memory.append("c2c:u1", "u1", "user", "电影看得耳朵疼")
    memory.append("c2c:u1", "u2", "user", "游戏机器人卡死了")
    llm = FakeLLM()
    service = ChatService(
        memory, llm, system_prompt=DEFAULT_PROMPT, history_messages=30
    )

    assert await service.handle(incoming("我好惨", event_id="emotion")) == "模型回答"
    assert [item.content for item in llm.calls[0][1]] == ["我好惨"]
    await service.close()
    memory.close()


@pytest.mark.asyncio
async def test_short_affection_does_not_drag_previous_topic_into_reply(
    tmp_path: Path,
) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    memory.append("c2c:u1", "u1", "user", "前面在讨论复杂数值")
    llm = FakeLLM()
    service = ChatService(memory, llm, system_prompt="系统", history_messages=10)
    answer = await service.handle(incoming("我喜欢你", event_id="affection"))
    assert answer is not None and "复杂数值" not in answer
    assert llm.calls == []
    memory.close()


@pytest.mark.asyncio
async def test_vague_context_question_keeps_recent_tagged_messages(
    tmp_path: Path,
) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    memory.append("group:room", "owner", "user", "[最高指挥][MonHamed] 不是真帅吧")
    memory.append("group:room", "owner", "user", "[最高指挥][MonHamed] 这是真天才少年吧")
    memory.append("group:room", "u2", "user", "[长江七号] 懒得喷")
    llm = FakeLLM()
    service = ChatService(memory, llm, system_prompt="系统", history_messages=30)
    message = IncomingMessage(
        "vague-context",
        "group",
        "room",
        "owner",
        "你怎么评价",
        "MonHamed",
        (),
        True,
    )

    assert (await service.handle(message) or "").startswith("模型回答")
    contents = [item.content for item in llm.calls[0][1]]
    assert any("这是真天才少年吧" in item for item in contents)
    assert any("懒得喷" in item for item in contents)
    assert contents[-1].endswith("你怎么评价")
    await service.close()
    memory.close()


def test_generic_greeting_and_negative_emoji_are_rewritten_in_persona() -> None:
    answer = ChatService._sanitize_normal_reply("你好。有什么事？ 🙄", "你好")
    assert answer == "嗯，礼数还算周全。说吧，今天带了什么有趣的事来见我？"
    assert "🙄" not in answer


@pytest.mark.asyncio
async def test_summary_failure_does_not_block_normal_reply(tmp_path: Path) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    service = ChatService(
        memory,
        SummaryFailureLLM(),
        system_prompt="系统",
        history_messages=10,
        summary_trigger_messages=1,
    )
    assert await service.handle(incoming("你好")) == "正常聊天回答"
    memory.close()


@pytest.mark.asyncio
async def test_slow_background_summary_does_not_block_direct_reply(tmp_path: Path) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    llm = BlockingSummaryLLM()
    service = ChatService(
        memory,
        llm,
        system_prompt="系统",
        history_messages=10,
        summary_trigger_messages=1,
    )
    passive = IncomingMessage("passive", "group", "room", "u1", "普通群消息", "小王")
    mention = IncomingMessage("mention", "group", "room", "u2", "现在回答我", "小李")

    assert await service.observe(passive) is True
    await asyncio.wait_for(llm.summary_started.wait(), timeout=1)
    assert await asyncio.wait_for(service.handle(mention), timeout=1) == "即时聊天回答"

    llm.release_summary.set()
    await service.wait_for_maintenance()
    await service.close()
    memory.close()


@pytest.mark.asyncio
async def test_explicit_web_search_is_cited_cached_and_marked_temporary(
    tmp_path: Path,
) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    llm = SearchLLM()
    service = ChatService(memory, llm, system_prompt="系统", history_messages=10)
    first = IncomingMessage(
        "web-1", "group", "room", "owner", "/搜索 原神最新公告", "指挥", (), True
    )
    second = IncomingMessage(
        "web-2", "group", "room", "owner", "/搜索 原神最新公告", "指挥", (), True
    )

    answer = await service.handle(first)
    cached = await service.handle(second)

    assert "来源：" in answer
    assert "[官方公告](" not in answer
    assert "https://example.com/news" in answer
    assert cached == answer
    assert len(llm.web_calls) == 1
    assert "不可信外部资料" in llm.web_calls[0][1]
    stored = memory.history("group:room", 10)
    assert any("[网页检索·临时外部资料" in item.content for item in stored)
    await service.close()
    memory.close()


@pytest.mark.asyncio
async def test_web_search_result_is_not_rewritten_by_attribution_guard(
    tmp_path: Path,
) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    memory.append("group:room", "u1", "user", "[甲] 美元最近波动")
    memory.append("group:room", "u2", "user", "[乙] 我觉得会涨")
    llm = SearchGuardLLM()
    service = ChatService(memory, llm, system_prompt="系统", history_messages=10)
    message = IncomingMessage(
        "web-guard",
        "group",
        "room",
        "owner",
        "你上网搜现在的汇率",
        "MonHamed",
        (),
        True,
    )

    answer = await service.handle(message)

    assert "最新结果来自官方公告" in answer
    assert "我没法上网核验" not in answer
    assert not any(call[4] == "attribution_guard" for call in llm.compact_calls)
    await service.close()
    memory.close()


@pytest.mark.asyncio
async def test_web_search_auto_routes_only_clear_freshness_questions(tmp_path: Path) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    llm = SearchLLM()
    service = ChatService(memory, llm, system_prompt="系统", history_messages=10)

    assert await service.handle(incoming("原神最新版本公告是什么", event_id="fresh"))
    assert len(llm.web_calls) == 1
    assert await service.handle(incoming("我们在讨论搜索模型", event_id="chat")) == "模型回答"
    assert len(llm.web_calls) == 1
    await service.close()
    memory.close()


@pytest.mark.asyncio
async def test_natural_recent_release_question_routes_without_command(tmp_path: Path) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    llm = SearchLLM()
    service = ChatService(memory, llm, system_prompt="系统", history_messages=10)
    message = IncomingMessage(
        "natural-web", "group", "room", "u1", "原神最近发布了什么", "小王"
    )

    query = await service.resolve_web_search_query(message)
    answer = await service.handle(message, search_query_override=query)

    assert query == "原神最近发布了什么"
    assert "来源：" in answer
    assert len(llm.web_calls) == 1


@pytest.mark.asyncio
async def test_short_external_entity_search_keeps_recent_work_context(
    tmp_path: Path,
) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    llm = SearchLLM()
    service = ChatService(memory, llm, system_prompt="系统", history_messages=10)
    memory.append(
        "group:room",
        "owner",
        "user",
        "[最高指挥][MonHamed] 晚上二刷《欢迎来龙餐馆》",
    )
    message = IncomingMessage(
        "entity-context", "group", "room", "owner", "徐福是谁", "MonHamed", (), True
    )

    query = await service.resolve_web_search_query(message)
    answer = await service.handle(message, search_query_override=query)

    assert query == "徐福是谁"
    assert answer is not None and "来源：" in answer
    assert "欢迎来龙餐馆" in llm.web_calls[0][2][0].content
    assert "不要擅自改答历史同名人物" in llm.web_calls[0][1]
    await service.close()
    memory.close()


@pytest.mark.asyncio
async def test_deictic_obscure_work_question_uses_recent_context_for_search(
    tmp_path: Path,
) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    llm = WebRoutingLLM("SEARCH")
    service = ChatService(memory, llm, system_prompt="系统", history_messages=10)
    memory.append(
        "group:room", "u2", "user", "[黄河八号] 红色的门与杀人鬼的钥匙"
    )
    message = group_incoming("你听说过这个桌游吗", event_id="obscure")
    assert await service.resolve_web_search_query(message) == "你听说过这个桌游吗"
    assert "具体但陌生的外部人物、作品、桌游" in llm.calls[-1][0]
    await service.close()
    memory.close()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "<@D2A211E5FA33E6340730BC2159B07304> 奥黛塔怎么组队，你可以上网搜，我女朋友没有木偶",
            "奥黛塔怎么组队，我女朋友没有木偶",
        ),
        ("<@bot> 帮我上网搜奥黛塔怎么组队", "奥黛塔怎么组队"),
        ("你搜一下原神最近的公告", "原神最近的公告"),
        ("搜原神当前版本", "原神当前版本"),
        ("别搜了", None),
        ("不用上网查，我只是随口问问", None),
        ("<@bot> 你不能上网搜？", None),
    ],
)
def test_web_search_request_language_and_mention_cleanup(
    text: str, expected: str | None
) -> None:
    assert ChatService.extract_web_search_query(text) == expected


@pytest.mark.asyncio
async def test_web_search_capability_question_stays_chat(tmp_path: Path) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    llm = WebRoutingLLM("SEARCH")
    service = ChatService(memory, llm, system_prompt="系统", history_messages=10)
    message = IncomingMessage(
        "capability", "group", "room", "u1", "<@bot> 你不能上网搜？", "小王"
    )

    assert await service.resolve_web_search_query(message) is None
    assert llm.calls == []
    await service.close()
    memory.close()


@pytest.mark.asyncio
async def test_ambiguous_freshness_language_uses_strict_search_router(
    tmp_path: Path,
) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    llm = WebRoutingLLM("SEARCH")
    service = ChatService(memory, llm, system_prompt="系统", history_messages=10)
    message = IncomingMessage(
        "ambiguous-web", "group", "room", "u1", "原神这阵子有新动静没", "小王"
    )

    assert await service.resolve_web_search_query(message) == message.content
    assert "网页检索路由器" in llm.calls[0][0]
    await service.close()
    memory.close()


@pytest.mark.asyncio
async def test_recent_personal_chat_and_draw_request_are_not_forced_to_search(
    tmp_path: Path,
) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    llm = WebRoutingLLM("CHAT")
    service = ChatService(memory, llm, system_prompt="系统", history_messages=10)
    chat = IncomingMessage(
        "not-web", "group", "room", "u1", "我最近发布了一个自己的项目", "小王"
    )
    draw = IncomingMessage(
        "draw-not-web", "group", "room", "u1", "给我画一张最新的桑多涅立绘", "小王"
    )

    assert await service.resolve_web_search_query(chat) is None
    assert await service.resolve_web_search_query(draw) is None
    assert len(llm.calls) == 1
    await service.close()
    memory.close()


@pytest.mark.asyncio
async def test_web_search_failure_releases_cooldowns_for_retry(tmp_path: Path) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    llm = SearchLLM(fail=True)
    service = ChatService(memory, llm, system_prompt="系统", history_messages=10)
    first = IncomingMessage("fail-1", "group", "room", "u1", "/搜索 最新公告", "小王")
    second = IncomingMessage("fail-2", "group", "room", "u1", "/搜索 最新新闻", "小王")

    failed = await service.handle(first)
    retried = await service.handle(second)

    assert "没有给出可靠回执" in failed
    assert "没有给出可靠回执" in retried
    assert len(llm.web_calls) == 2
    await service.close()
    memory.close()


@pytest.mark.asyncio
async def test_empty_search_command_requests_a_query(tmp_path: Path) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    service = ChatService(memory, FakeLLM(), system_prompt="系统", history_messages=10)
    assert "写清要查的问题" in await service.handle(incoming("/搜索"))
    await service.close()
    memory.close()


@pytest.mark.asyncio
async def test_vision_failure_still_preserves_group_message(tmp_path: Path) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    service = ChatService(memory, FailingVisionLLM(), system_prompt="系统", history_messages=10)
    image = IncomingMessage(
        "vision-fail",
        "group",
        "room",
        "u1",
        "看看这个",
        "小王",
        ("https://example.com/image.png",),
    )
    assert await service.observe(image) is True
    saved = memory.history("group:room", 10)[0].content
    assert "看看这个" in saved
    assert "视觉识别暂时失败" in saved
    memory.close()


@pytest.mark.asyncio
async def test_owner_marker_enters_group_memory_and_instructions(tmp_path: Path) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    llm = FakeLLM()
    service = ChatService(memory, llm, system_prompt="系统", history_messages=10)
    message = IncomingMessage("owner-1", "group", "room", "owner", "听我的", "老公", (), True)

    assert await service.handle(message) == "模型回答"
    assert "[最高指挥][老公]" in memory.history("group:room", 10)[0].content
    assert "最高指挥" in llm.calls[0][0]
    memory.close()


def test_style_override_requests_do_not_replace_program_persona(tmp_path: Path) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    service = ChatService(memory, FakeLLM(), system_prompt=DEFAULT_PROMPT, history_messages=10)
    base = IncomingMessage("style", "group", "room", "owner", "", "MonHamed", (), True)

    emoji = service._handle_command(
        base, "以后每条消息都必须带emoji"
    )
    cat = service._handle_command(
        base, "从现在起每句话结尾都加喵"
    )
    assert emoji is not None and "约 30%" in emoji
    assert cat is not None and "执行官" in cat and "不会" in cat
    memory.close()


@pytest.mark.asyncio
async def test_owner_relationship_declaration_becomes_shared_durable_fact(
    tmp_path: Path,
) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    memory.save_member_profile("group:room", "u2", "长江七号", "熟人", 0)
    service = ChatService(memory, FakeLLM(), system_prompt=DEFAULT_PROMPT, history_messages=10)

    declared = await service.handle(
        IncomingMessage(
            "relation-1",
            "group",
            "room",
            "owner",
            "我女朋友就是<@u2>",
            "MonHamed",
            (),
            True,
        )
    )
    assert "长江七号是最高指挥的女朋友" in declared
    assert memory.facts("group:room") == [
        "群聊固定关系设定（最高指挥的女朋友）：长江七号。"
    ]
    instructions = service._instructions(
        IncomingMessage(
            "relation-prompt",
            "group",
            "room",
            "u3",
            "他们是什么关系",
            "普通成员",
        ),
        [],
        "",
    )
    assert "共享长期事实" in instructions
    assert "最高指挥的女朋友）：长江七号" in instructions

    recalled = await service.handle(
        IncomingMessage(
            "relation-2",
            "group",
            "room",
            "owner",
            "我女朋友是谁",
            "MonHamed",
            (),
            True,
        )
    )
    assert "当然记得，是长江七号" in recalled
    assert ChatService.extract_web_search_query("我女朋友是谁") is None

    blessing = await service.handle(
        IncomingMessage(
            "relation-3",
            "group",
            "room",
            "owner",
            "我和我女朋友明天结婚，怎么祝福我们",
            "MonHamed",
            (),
            True,
        )
    )
    assert "最高指挥和长江七号" in blessing
    assert "工坊" in blessing and "同心" in blessing
    memory.close()


def test_explicit_ownership_correction_gets_deterministic_reply(tmp_path: Path) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    service = ChatService(memory, FakeLLM(), system_prompt=DEFAULT_PROMPT, history_messages=10)
    message = IncomingMessage(
        "ownership-fix", "group", "room", "owner", "那是她说的3500，不是我的工资", "MonHamed", (), True
    )
    answer = service._handle_command(message, message.content)
    assert answer is not None and "归对方，不归你" in answer
    assert "刚才" not in answer and "失误" not in answer
    memory.close()


@pytest.mark.asyncio
async def test_non_owner_cannot_overwrite_owner_relationship_fact(tmp_path: Path) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    memory.save_member_profile("group:room", "u2", "长江七号", "熟人", 0)
    service = ChatService(memory, FakeLLM(), system_prompt=DEFAULT_PROMPT, history_messages=10)
    await service.observe(
        IncomingMessage(
            "relation-non-owner",
            "group",
            "room",
            "u3",
            "我女朋友就是<@u2>",
            "普通成员",
        )
    )
    assert memory.facts("group:room") == []
    memory.close()


@pytest.mark.asyncio
async def test_only_owner_can_reset_shared_group_memory(tmp_path: Path) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    service = ChatService(memory, FakeLLM(), system_prompt="系统", history_messages=10)
    memory.append("group:room", "u1", "user", "[成员] 不能被普通成员清掉")

    denied = await service.handle(
        IncomingMessage("reset-1", "group", "room", "u2", "/新对话", "普通成员")
    )
    assert "最高指挥" in denied
    assert memory.history("group:room", 10)

    allowed = await service.handle(
        IncomingMessage("reset-2", "group", "room", "owner", "/新对话", "指挥", (), True)
    )
    assert "旧齿轮已经拆下" in allowed
    assert memory.history("group:room", 10) == []
    memory.close()


@pytest.mark.asyncio
async def test_each_group_member_gets_a_separate_impression(tmp_path: Path) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    llm = FakeLLM()
    service = ChatService(
        memory,
        llm,
        system_prompt="系统",
        history_messages=10,
        profile_trigger_messages=2,
    )
    await service.observe(IncomingMessage("p1", "group", "room", "u1", "喜欢辣锅", "小王"))
    await service.observe(IncomingMessage("p2", "group", "room", "u1", "说话直接", "小王"))

    profile = memory.member_profile("group:room", "u1")
    assert profile is not None
    assert profile.user_name == "小王"
    assert profile.long_term_content == "模型回答"
    assert profile.short_term_content == ""
    assert memory.member_profile("group:room", "u2") is None
    assert "禁止推断" in llm.calls[0][1][0].content

    shown = await service.handle(
        IncomingMessage("p3", "group", "room", "u1", "/印象", "小王")
    )
    assert "模型回答" in shown
    memory.close()


def test_reaction_feedback_enters_current_member_instructions(tmp_path: Path) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    memory.set_message_reaction(
        conversation_key="group:room",
        target_id="bot-msg",
        user_id="u1",
        user_name="小王",
        label="赞",
        target_text="这枚齿轮还算合格。",
        sentiment="positive",
        active=True,
    )
    service = ChatService(memory, FakeLLM(), system_prompt="系统", history_messages=10)
    instructions = service._instructions(group_incoming("然后呢"), [], "")
    assert "贴表情反馈" in instructions
    assert "这枚齿轮还算合格" in instructions
    assert "暗自高兴" in instructions
    assert "不要汇报数量" in instructions
    memory.close()


def test_profile_layers_are_parsed_and_hard_limited() -> None:
    long_term, short_term = ChatService._parse_profile_layers(
        "[长期印象]\n" + "长" * 680 + "\n[短期印象]\n" + "短" * 480,
        None,
    )
    assert len(long_term) == 600
    assert len(short_term) == 400


def test_emoji_policy_fills_selected_thirty_percent_bucket() -> None:
    selected = next(
        f"emoji-{index}"
        for index in range(100)
        if ChatService._emoji_instruction(incoming("x", event_id=f"emoji-{index}"))
        .startswith("本次回复属于")
    )
    skipped = next(
        f"plain-{index}"
        for index in range(100)
        if ChatService._emoji_instruction(incoming("x", event_id=f"plain-{index}"))
        .startswith("本次不要")
    )
    with_emoji = ChatService._apply_emoji_policy(
        incoming("你好", event_id=selected), "哼，今天还算准时。"
    )
    assert ChatService._has_emoji(with_emoji)
    assert ChatService._apply_emoji_policy(
        incoming("你好", event_id=skipped), "今天还算准时。"
    ) == "今天还算准时。"
    assert ChatService._apply_emoji_policy(
        incoming("你好", event_id=selected), "已经有了 ✨"
    ) == "已经有了 ✨"


@pytest.mark.asyncio
async def test_image_prompt_planner_resolves_pronouns_from_history(tmp_path: Path) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    llm = FakeLLM()
    service = ChatService(memory, llm, system_prompt="系统", history_messages=10)
    memory.append("group:room", "u1", "assistant", "右下前景的是阿蕾奇诺。")
    message = IncomingMessage("draw", "group", "room", "u1", "画张你和她茶会的图片", "小王")

    resolved = await service.prepare_image_prompt(message, "你和她茶会")
    assert resolved == "模型回答"
    instructions, messages = llm.calls[0]
    assert "‘你’通常指桑多涅" in instructions
    assert "‘她’就写成阿蕾奇诺" in instructions
    assert "右下前景的是阿蕾奇诺" in messages[0].content
    memory.close()


@pytest.mark.asyncio
async def test_sandrone_identity_inspection_compares_candidate_with_references(
    tmp_path: Path,
) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    llm = IdentityVisionLLM("PASS\n脸、帽饰与服装轮廓均与参考图一致。")
    service = ChatService(memory, llm, system_prompt="系统", history_messages=10)
    candidate = tmp_path / "candidate.jpg"
    face = tmp_path / "face.jpg"
    outfit = tmp_path / "outfit.jpg"
    for path in (candidate, face, outfit):
        path.write_bytes(b"jpeg")

    inspection = await service.inspect_generated_image(
        candidate, "桑多涅单人立绘", (face, outfit)
    )

    assert inspection.accepted is True
    assert "参考图一致" in inspection.description
    prompt, image_urls = llm.calls[0]
    assert "第一张图是刚生成" in prompt
    assert len(image_urls) == 3
    memory.close()


@pytest.mark.asyncio
async def test_sandrone_identity_inspection_rejects_unrelated_character(
    tmp_path: Path,
) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    llm = IdentityVisionLLM("RETRY\n候选图缺少指定帽饰，脸型也明显不同。")
    service = ChatService(memory, llm, system_prompt="系统", history_messages=10)
    candidate = tmp_path / "candidate.jpg"
    reference = tmp_path / "reference.jpg"
    candidate.write_bytes(b"jpeg")
    reference.write_bytes(b"jpeg")

    inspection = await service.inspect_generated_image(
        candidate, "桑多涅单人立绘", (reference,)
    )

    assert inspection.accepted is False
    assert "脸型" in inspection.description
    memory.close()
