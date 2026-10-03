import asyncio
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import httpx

from talk_bot.image_gen import GeneratedImage, ImageContentPolicyError
from talk_bot.image_sources import ImageSourceUnavailable
from talk_bot.memory import MemoryStore
from talk_bot.qq_adapter import (
    ImageJobState,
    QQBotRunner,
    image_progress_reply,
    is_active_message_permission_error,
    is_expired_reply_error,
    is_image_delivery_followup,
    is_image_progress_question,
    normalize_qq_content,
)
from talk_bot.service import ImageInspection
from talk_bot.service import IncomingMessage


def test_progress_questions_are_detected_without_matching_normal_image_opinions() -> None:
    assert is_image_progress_question("图片生成进度怎么样了") is True
    assert is_image_progress_question("图画好了吗") is True
    assert is_image_progress_question("生成图片还要多久") is True
    assert is_image_progress_question("图画咋样了") is True
    assert is_image_progress_question("这幅画怎么样了") is True
    assert is_image_progress_question("在画了吧") is True
    assert is_image_progress_question("你觉得这张图怎么样") is False


def test_image_delivery_followups_are_routed_to_real_job_state() -> None:
    assert is_image_delivery_followup("发给我啊") is True
    assert is_image_delivery_followup("发给我") is True
    assert is_image_delivery_followup("把图片发给我看看") is True
    assert is_image_delivery_followup("把文件发给我") is False


def test_expired_reply_error_is_recognized_without_misclassifying_timeouts() -> None:
    assert is_expired_reply_error(
        RuntimeError("QQ Bot API error [400]: 回复消息msg_id已过期")
    )
    assert is_expired_reply_error(RuntimeError("message msg_id expired"))
    api_error = RuntimeError("QQ Bot API error [400]: msgid已经过期,不能回复")
    wrapped = RuntimeError("send_text failed after 1 attempts")
    wrapped.__cause__ = api_error
    assert is_expired_reply_error(wrapped)
    assert not is_expired_reply_error(RuntimeError("QQ Bot API timeout"))


def test_proactive_permission_detection_is_specific_and_unwraps_sdk_error() -> None:
    error = RuntimeError("QQ Bot API error [400]: 主动消息失败, 无权限")
    wrapped = RuntimeError("send_text failed after 1 attempts")
    wrapped.__cause__ = error
    assert is_active_message_permission_error(wrapped)
    assert not is_active_message_permission_error(RuntimeError("QQ Bot API error [400]: 内容太长"))
    assert not is_active_message_permission_error(RuntimeError("QQ Bot API timeout"))


@pytest.mark.asyncio
async def test_proactive_permission_denial_is_cached_per_group_and_survives_restart(tmp_path) -> None:
    path = tmp_path / "memory.db"
    memory = MemoryStore(path)
    runner = object.__new__(QQBotRunner)
    runner.service = SimpleNamespace(memory=memory)
    runner.api = SimpleNamespace(send_text=AsyncMock(side_effect=RuntimeError(
        "QQ Bot API error [400]: 主动消息失败, 无权限"
    )))
    event = SimpleNamespace(chat_scope="group", chat_id="g1", message_id="m1")

    assert await runner._send_optional_text(event, "早安") is False
    assert runner.api.send_text.await_args.kwargs["retries"] == 1
    assert await runner._send_optional_text(event, "又来一次") is False
    assert runner.api.send_text.await_count == 1
    assert not runner._optional_messages_available(event)
    assert runner._optional_messages_available(event, passive=True)
    assert runner._optional_messages_available(SimpleNamespace(chat_scope="group", chat_id="g2"))
    remaining = memory.rate_limit_remaining(runner._optional_permission_key(event), 21600)
    assert 21590 <= remaining <= 21600
    memory.close()

    runner.service.memory = MemoryStore(path)
    assert not runner._optional_messages_available(event)
    with runner.service.memory._lock:
        runner.service.memory._conn.execute("UPDATE rate_limits SET last_at = last_at - 21601")
        runner.service.memory._conn.commit()
    assert runner._optional_messages_available(event)
    runner.service.memory.close()


@pytest.mark.asyncio
async def test_transient_optional_failure_does_not_disable_group(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    runner = object.__new__(QQBotRunner)
    runner.service = SimpleNamespace(memory=memory)
    runner.api = SimpleNamespace(send_text=AsyncMock(side_effect=httpx.ReadTimeout("timeout")))
    event = SimpleNamespace(chat_scope="group", chat_id="g1", message_id="m1")
    assert await runner._send_optional_text(event, "早安") is False
    assert runner._optional_messages_available(event)
    memory.close()


def _runner_for_optional_event_test(memory: MemoryStore) -> QQBotRunner:
    runner = object.__new__(QQBotRunner)
    runner.owner_ids = frozenset()
    runner._image_job = None
    runner.image_sources = None
    runner.api = SimpleNamespace(send_text=AsyncMock(return_value={"id": "sent"}))
    runner.service = SimpleNamespace(
        memory=memory,
        observe=AsyncMock(return_value=True),
        note_group_message=lambda _: True,
        daily_greeting=AsyncMock(return_value="早安"),
        complete_daily_greeting=lambda *args, **kwargs: None,
        proactive_reply=AsyncMock(return_value="这个话题还算有趣。"),
        resolve_web_search_query=AsyncMock(return_value=None),
        resolve_image_prompt=AsyncMock(return_value=None),
        handle=AsyncMock(return_value="回答"),
        remember_assistant=lambda incoming, content: memory.append(
            incoming.conversation_key, incoming.user_id, "assistant", content
        ),
    )
    return runner


def _optional_event(message_id: str = "m1"):
    return SimpleNamespace(
        chat_scope="group", chat_id="g1", message_id=message_id,
        user_id="u1", user_name="群友", raw={}, content="这个话题还挺有趣的",
        message_type=0, msg_elements=[], attachments=[],
    )


@pytest.mark.asyncio
async def test_denied_group_skips_optional_generation_but_passive_greeting_and_reply_continue(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    runner = _runner_for_optional_event_test(memory)
    event = _optional_event()
    runner._block_optional_messages(event)

    await runner._handle_event(event, should_reply=False)
    runner.service.observe.assert_awaited_once()
    runner.service.daily_greeting.assert_not_awaited()
    runner.service.proactive_reply.assert_not_awaited()
    runner.api.send_text.assert_not_awaited()

    await runner._handle_event(_optional_event("mention"), should_reply=True)
    runner.service.daily_greeting.assert_awaited_once()
    runner.service.handle.assert_awaited_once()
    assert runner.api.send_text.await_count == 2
    assert all(call.kwargs["reply_to"] == "mention" for call in runner.api.send_text.await_args_list)
    memory.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("sent", [False, True])
async def test_proactive_message_is_recorded_only_after_confirmed_send(tmp_path, sent) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    runner = _runner_for_optional_event_test(memory)
    runner.service.daily_greeting.return_value = None
    if not sent:
        runner.api.send_text.side_effect = RuntimeError("QQ Bot API error [400]: 主动消息失败, 无权限")
    await runner._handle_event(_optional_event(), should_reply=False)
    history = memory.history("group:g1", 10)
    assert bool(history) is sent
    if sent:
        assert history[-1].content == "这个话题还算有趣。"
        assert history[-1].user_id == "u1"
    memory.close()


@pytest.mark.asyncio
async def test_expired_text_reply_falls_back_to_fresh_group_message() -> None:
    runner = object.__new__(QQBotRunner)
    runner.api = SimpleNamespace(
        send_text=AsyncMock(
            side_effect=[
                RuntimeError("QQ Bot API error [400]: 回复消息msg_id已过期"),
                {"id": "fresh-message"},
            ]
        )
    )
    event = SimpleNamespace(chat_scope="group", chat_id="g1", message_id="expired-1")

    result = await runner._send_reply_text(event, "回答")

    assert result == {"id": "fresh-message"}
    assert runner.api.send_text.await_count == 2
    assert runner.api.send_text.await_args_list[0].kwargs["reply_to"] == "expired-1"
    assert runner.api.send_text.await_args_list[1].kwargs["reply_to"] is None
    assert runner._outbound_text_by_id == {"fresh-message": "回答"}


@pytest.mark.asyncio
async def test_expired_image_reply_falls_back_without_original_msg_id() -> None:
    runner = object.__new__(QQBotRunner)
    runner.api = SimpleNamespace(
        next_msg_seq=lambda: 7,
        post_group_message=AsyncMock(
            side_effect=[
                RuntimeError("QQ Bot API error [400]: 回复消息msg_id已过期"),
                {"id": "fresh-image"},
            ]
        ),
        post_c2c_message=AsyncMock(),
    )
    event = SimpleNamespace(chat_scope="group", chat_id="g1", message_id="expired-2")

    await runner._send_media_with_reply_fallback(event, "file-info")

    first = runner.api.post_group_message.await_args_list[0].args[1]
    second = runner.api.post_group_message.await_args_list[1].args[1]
    assert first.msg_id == "expired-2"
    assert second.msg_id == ""


def test_progress_reply_reports_elapsed_time_and_phase_without_fake_percentage() -> None:
    job = ImageJobState(started_at=100.0, phase="模型渲染", prompt="测试")
    reply = image_progress_reply(job, now=142.9)
    assert "42 秒" in reply
    assert "模型渲染" in reply
    assert "不提供百分比" in reply
    assert image_progress_reply(None) == (
        "工坊当前没有正在生成或等待发送的图片；我不会拿一张并不存在的成图敷衍你。"
    )


def test_idle_progress_uses_persistent_last_result(tmp_path) -> None:
    store = MemoryStore(tmp_path / "memory.db")
    job_id = store.create_image_job(
        event_id="sent",
        conversation_key="group:g1",
        user_id="u1",
        prompt="测试",
        phase="上传 QQ",
        started_at=100,
    )
    store.update_image_job(job_id, phase="已发送", status="sent", now=120)
    reply = image_progress_reply(None, last_job=store.latest_image_job("group:g1"))
    assert "QQ 确认发送成功" in reply
    store.close()


def test_only_bot_mention_is_removed_from_reply_content() -> None:
    runner = object.__new__(QQBotRunner)
    runner._bot_ids = {"bot-openid"}
    runner._bot_names = {"sandrone"}
    assert runner._strip_bot_mentions("@sandrone 你好") == "你好"
    assert runner._strip_bot_mentions("<@bot-openid> 你好") == "你好"
    assert runner._strip_bot_mentions("@另一位群友 你好") == "@另一位群友 你好"


def test_platform_is_you_mention_teaches_runner_its_bot_id() -> None:
    runner = object.__new__(QQBotRunner)
    runner._bot_ids = set()
    runner._bot_names = set()
    raw = {
        "mentions": [{"is_you": True, "member_openid": "dynamic-bot-id"}],
        "content": "<@dynamic-bot-id> 在吗",
    }
    assert runner._is_bot_mentioned(raw)
    assert "dynamic-bot-id" in runner._bot_ids
    assert runner._strip_bot_mentions(raw["content"]) == "在吗"


def test_quote_message_extracts_referenced_text_and_images() -> None:
    runner = object.__new__(QQBotRunner)
    event = SimpleNamespace(
        message_type=103,
        msg_elements=[
            SimpleNamespace(
                content="被引用的原文",
                attachments=[
                    SimpleNamespace(
                        content_type="image/jpeg",
                        filename="quoted.jpg",
                        url="//example.com/quoted.jpg",
                    )
                ],
            )
        ],
    )
    content, images = runner._quoted_context(event)
    assert content == "被引用的原文"
    assert images == ("https://example.com/quoted.jpg",)


def test_qq_face_payload_is_compacted_to_readable_context() -> None:
    raw = '<faceType=1,faceId="111",ext="eyJ0ZXh0Ijoi5Y+v5oCcIn0=">'
    assert normalize_qq_content("没看懂" + raw) == "没看懂[表情:可怜]"
    assert normalize_qq_content('<faceType=1,faceId="1",ext="bad">') == "[表情]"


@pytest.mark.asyncio
async def test_reaction_event_is_recorded_as_group_context(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.db")
    runner = object.__new__(QQBotRunner)
    runner.allowed_group_ids = frozenset({"g1"})
    runner._outbound_text_by_id = {"bot-msg": "这张图校准得还算像样。"}
    runner.service = SimpleNamespace(memory=store)

    await runner._on_reaction(
        "MESSAGE_REACTION_ADD",
        {
            "user_id": "u1",
            "target": {"id": "bot-msg", "type": 0},
            "emoji": {"id": "128077", "type": 2},
        },
    )

    content = store.history("group:g1", 1)[0].content
    assert "[贴表情:👍]" in content
    assert "这张图校准得还算像样" in content
    feedback = store.reaction_feedback("group:g1", "u1")
    assert feedback is not None
    assert feedback.positive_count == 1

    await runner._on_reaction(
        "MESSAGE_REACTION_REMOVE",
        {
            "user_id": "u1",
            "target": {"id": "bot-msg", "type": 0},
            "emoji": {"id": "128077", "type": 2},
        },
    )
    assert store.reaction_feedback("group:g1", "u1") is None
    store.close()


class StubImageService:
    def __init__(
        self,
        memory: MemoryStore,
        *,
        fail_memory: bool = False,
        inspections: list[ImageInspection] | None = None,
    ) -> None:
        self.memory = memory
        self.fail_memory = fail_memory
        self.remembered: list[str] = []
        self.inspections = inspections or [ImageInspection(True, "实际成图")]

    async def begin_direct_request(self, incoming) -> bool:
        return True

    async def prepare_image_prompt(self, incoming, prompt: str) -> str:
        return prompt

    async def inspect_generated_image(
        self, path: Path, prompt: str, reference_paths: tuple[Path, ...] = ()
    ) -> ImageInspection:
        return self.inspections.pop(0)

    def remember_assistant(self, incoming, content: str) -> None:
        self.remembered.append(content)

    def remember_image_result(self, *args) -> None:
        if self.fail_memory:
            raise RuntimeError("memory write failed")
        self.remembered.append("image-result")


def _runner_for_image_test(service: StubImageService) -> QQBotRunner:
    runner = object.__new__(QQBotRunner)
    runner.service = service
    runner.image_cooldown_seconds = 180
    runner._image_job = None
    runner.api = SimpleNamespace(
        send_text=AsyncMock(),
        post_group_message=AsyncMock(),
        post_c2c_message=AsyncMock(),
        next_msg_seq=lambda: 1,
    )
    runner.media_uploader = SimpleNamespace(upload=AsyncMock(return_value="file-info"))
    return runner


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["pass", "reject", "unavailable"])
async def test_edit_pipeline_uses_pixels_and_never_sends_unverified_result(tmp_path, outcome):
    from PIL import Image
    source = tmp_path / "source.png"
    Image.new("RGB", (80, 160), "red").save(source)
    store = MemoryStore(tmp_path / "memory.db")
    service = StubImageService(store)
    service.llm = SimpleNamespace(_download_image_as_data_url=AsyncMock())
    service.prepare_image_prompt = AsyncMock(side_effect=AssertionError("must not rewrite edit as a new scene"))
    service.inspect_edited_image = AsyncMock(return_value=ImageInspection(outcome == "pass", "核对数字"))
    runner = _runner_for_image_test(service)
    runner.image_sources = SimpleNamespace(resolve=AsyncMock(return_value=(source,)), store=lambda *args: None)
    if outcome == "unavailable":
        runner.image_sources.resolve.side_effect = ImageSourceUnavailable("原图不可用，请重发")
    async def edit(prompt, paths):
        assert paths[0].read_bytes() == source.read_bytes()
        output = tmp_path / "candidate.png"
        shutil.copyfile(source, output)
        return GeneratedImage(output, "edit")
    runner.image_generator = SimpleNamespace(edit=AsyncMock(side_effect=edit), generate=AsyncMock())
    message = IncomingMessage("edit1", "group", "g1", "u1", "把图中原石数量改成65432")
    event = SimpleNamespace(chat_scope="group", chat_id="g1", message_id=message.event_id)
    await runner._handle_image_request(event, message, message.content)
    runner.image_generator.generate.assert_not_awaited()
    service.prepare_image_prompt.assert_not_awaited()
    assert runner._image_job is None
    assert store.latest_image_job("group:g1").status == ("sent" if outcome == "pass" else "failed")
    assert runner.media_uploader.upload.await_count == (1 if outcome == "pass" else 0)
    assert runner.image_generator.edit.await_count == {"pass": 1, "reject": 2, "unavailable": 0}[outcome]
    if outcome != "pass":
        assert store.claim_rate_limit("image:global", 180)[0]
    store.close()


@pytest.mark.asyncio
async def test_active_image_job_blocks_a_second_generation(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.db")
    service = StubImageService(store)
    runner = _runner_for_image_test(service)
    runner._image_job = ImageJobState(100, "模型渲染", "第一张", 1)
    runner.image_generator = SimpleNamespace(generate=AsyncMock())
    event = SimpleNamespace(chat_scope="group", chat_id="g1", message_id="m2")
    incoming = SimpleNamespace(
        event_id="m2",
        conversation_key="group:g1",
        user_id="u1",
    )

    await runner._handle_image_request(event, incoming, "第二张")

    runner.image_generator.generate.assert_not_awaited()
    assert "不会并行启动第二幅" in runner.api.send_text.await_args.args[2]
    store.close()


@pytest.mark.asyncio
async def test_non_owner_cannot_cancel_someone_elses_image(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.db")
    service = StubImageService(store)
    runner = _runner_for_image_test(service)
    runner._image_job = ImageJobState(100, "模型渲染", "第一张", 1, "requester")
    runner._image_task = None
    event = SimpleNamespace(chat_scope="group", chat_id="g1", message_id="cancel-1")
    incoming = SimpleNamespace(
        event_id="cancel-1",
        conversation_key="group:g1",
        user_id="other",
        is_owner=False,
    )

    await runner._handle_image_cancel(event, incoming)

    assert "不是你下达的任务" in runner.api.send_text.await_args.args[2]
    assert runner._image_job is not None
    store.close()


@pytest.mark.asyncio
async def test_owner_can_cancel_active_image_task(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.db")
    service = StubImageService(store)
    runner = _runner_for_image_test(service)
    runner._image_job = ImageJobState(100, "模型渲染", "第一张", 1, "requester")
    blocker = asyncio.Event()

    async def wait_forever() -> None:
        await blocker.wait()

    task = asyncio.create_task(wait_forever())
    runner._image_task = task
    event = SimpleNamespace(chat_scope="group", chat_id="g1", message_id="cancel-2")
    incoming = SimpleNamespace(
        event_id="cancel-2",
        conversation_key="group:g1",
        user_id="owner",
        is_owner=True,
    )

    await runner._handle_image_cancel(event, incoming)

    assert task.cancelled()
    assert "已经停下了" in runner.api.send_text.await_args.args[2]
    store.close()


@pytest.mark.asyncio
async def test_memory_failure_after_qq_delivery_does_not_report_generation_failure(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "memory.db")
    service = StubImageService(store, fail_memory=True)
    runner = _runner_for_image_test(service)
    image_path = tmp_path / "generated.jpg"
    image_path.write_bytes(b"jpeg")
    runner.image_generator = SimpleNamespace(
        generate=AsyncMock(return_value=GeneratedImage(image_path, "portrait"))
    )
    event = SimpleNamespace(chat_scope="group", chat_id="g1", message_id="sent-1")
    incoming = SimpleNamespace(
        event_id="sent-1",
        conversation_key="group:g1",
        user_id="u1",
    )

    await runner._handle_image_request(event, incoming, "桑多涅立绘")

    runner.api.post_group_message.assert_awaited_once()
    assert runner.api.send_text.await_count == 1  # only the real rendering-start notice
    latest = store.latest_image_job("group:g1")
    assert latest is not None
    assert latest.status == "sent"
    assert image_path.exists() is False
    store.close()


@pytest.mark.asyncio
async def test_safety_rewritten_image_is_sent_and_disclosed(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.db")
    service = StubImageService(store)
    runner = _runner_for_image_test(service)
    image_path = tmp_path / "safe.jpg"
    image_path.write_bytes(b"jpeg")
    runner.image_generator = SimpleNamespace(
        generate=AsyncMock(
            return_value=GeneratedImage(
                image_path, "landscape", (), safety_rewritten=True
            )
        )
    )
    event = SimpleNamespace(chat_scope="group", chat_id="g1", message_id="safe-1")
    incoming = SimpleNamespace(
        event_id="safe-1",
        conversation_key="group:g1",
        user_id="u1",
    )

    await runner._handle_image_request(event, incoming, "两人在床上嬉闹")

    runner.api.post_group_message.assert_awaited_once()
    assert "无暧昧" in runner.api.send_text.await_args_list[-1].args[2]
    latest = store.latest_image_job("group:g1")
    assert latest is not None and latest.phase == "安全改写后已发送"
    assert latest.status == "sent"
    store.close()


@pytest.mark.asyncio
async def test_content_policy_failure_is_reported_accurately(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.db")
    service = StubImageService(store)
    runner = _runner_for_image_test(service)
    runner.image_generator = SimpleNamespace(
        generate=AsyncMock(side_effect=ImageContentPolicyError("blocked"))
    )
    event = SimpleNamespace(chat_scope="group", chat_id="g1", message_id="policy-1")
    incoming = SimpleNamespace(
        event_id="policy-1",
        conversation_key="group:g1",
        user_id="u1",
    )

    await runner._handle_image_request(event, incoming, "被上游拒绝的画面")

    runner.api.post_group_message.assert_not_awaited()
    assert "上游规则拦下" in runner.api.send_text.await_args_list[-1].args[2]
    latest = store.latest_image_job("group:g1")
    assert latest is not None and latest.phase == "画面描述被上游拒绝"
    assert "content policy" in latest.error
    store.close()


@pytest.mark.asyncio
async def test_failed_identity_check_retries_once_before_delivery(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.db")
    service = StubImageService(
        store,
        inspections=[
            ImageInspection(False, "第一次脸型不符"),
            ImageInspection(True, "第二次身份一致"),
        ],
    )
    runner = _runner_for_image_test(service)
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    references = (tmp_path / "face.jpg", tmp_path / "outfit.jpg")
    generator = AsyncMock(
        side_effect=[
            GeneratedImage(first, "portrait", references),
            GeneratedImage(second, "portrait", references),
        ]
    )
    runner.image_generator = SimpleNamespace(generate=generator)
    event = SimpleNamespace(chat_scope="group", chat_id="g1", message_id="retry-1")
    incoming = SimpleNamespace(
        event_id="retry-1",
        conversation_key="group:g1",
        user_id="u1",
    )

    await runner._handle_image_request(event, incoming, "桑多涅立绘")

    assert generator.await_count == 2
    assert generator.await_args_list[1].kwargs["identity_retry"] is True
    runner.api.post_group_message.assert_awaited_once()
    assert first.exists() is False
    assert second.exists() is False
    latest = store.latest_image_job("group:g1")
    assert latest is not None and latest.status == "sent"
    store.close()


@pytest.mark.asyncio
async def test_second_identity_mismatch_is_not_sent(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.db")
    service = StubImageService(
        store,
        inspections=[
            ImageInspection(False, "第一次不像"),
            ImageInspection(False, "第二次仍不像"),
        ],
    )
    runner = _runner_for_image_test(service)
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    references = (tmp_path / "face.jpg",)
    runner.image_generator = SimpleNamespace(
        generate=AsyncMock(
            side_effect=[
                GeneratedImage(first, "portrait", references),
                GeneratedImage(second, "portrait", references),
            ]
        )
    )
    event = SimpleNamespace(chat_scope="group", chat_id="g1", message_id="retry-2")
    incoming = SimpleNamespace(
        event_id="retry-2",
        conversation_key="group:g1",
        user_id="u1",
    )

    await runner._handle_image_request(event, incoming, "桑多涅立绘")

    runner.api.post_group_message.assert_not_awaited()
    assert "不会把仿冒品递给你" in runner.api.send_text.await_args_list[-1].args[2]
    latest = store.latest_image_job("group:g1")
    assert latest is not None and latest.status == "failed"
    assert first.exists() is False
    assert second.exists() is False
    store.close()
