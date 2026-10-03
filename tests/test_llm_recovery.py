"""Regression coverage for bounded completion and QQ attachment recovery."""

import base64
import copy
import io
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from PIL import Image

from talk_bot.llm import LLMClient
import talk_bot.llm as llm_module
from talk_bot.memory import StoredMessage


def _completion(text, reason="stop"):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=text), finish_reason=reason
        )],
        usage=None,
    )


def _make_client(model="gemini-3-flash", handler=None):
    vision_http = httpx.AsyncClient(transport=httpx.MockTransport(
        handler or (lambda request: httpx.Response(200, request=request))
    ))
    return LLMClient(
        api_key="test", base_url="https://relay.example/v1", model=model,
        api_mode="chat_completions", reasoning_effort="high",
        max_output_tokens=1400, vision_http_client=vision_http,
    )


async def _close(client):
    await client.client.close()
    await client.vision_http.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("first", [
    _completion("原文、摘要，再", "length"),
    _completion(None),
    _completion("   "),
])
async def test_gemini_recovers_incomplete_or_empty_once_with_same_budget(first):
    client = _make_client()
    create = AsyncMock(side_effect=[first, _completion("8分，分层清楚。")])
    client.client.chat.completions.create = create
    message = StoredMessage("user", "评价我的记忆设计")
    try:
        result = await client.compact_reply(
            instructions="给出评价", messages=[message], max_output_tokens=350,
            reasoning_effort="medium", purpose="attribution_guard",
        )
    finally:
        await _close(client)
    assert result == "8分，分层清楚。"
    assert create.await_count == 2
    first_kwargs, second_kwargs = [call.kwargs for call in create.await_args_list]
    assert first_kwargs["reasoning_effort"] == "medium"
    assert second_kwargs["reasoning_effort"] == "low"
    assert first_kwargs["max_completion_tokens"] == second_kwargs["max_completion_tokens"]
    assert first_kwargs["messages"] == [
        {"role": "system", "content": "给出评价"},
        {"role": "user", "content": "评价我的记忆设计"},
    ]
    assert any(item["role"] == "system" for item in second_kwargs["messages"][2:])
    assert message.content == "评价我的记忆设计"


@pytest.mark.asyncio
async def test_recovery_does_not_mutate_original_kwargs_or_nested_vision_content():
    client = _make_client()
    client.client.chat.completions.create = AsyncMock(side_effect=[
        _completion("截断", "length"), _completion("完整"),
    ])
    kwargs = {
        "model": client.model, "reasoning_effort": "high",
        "max_completion_tokens": 4746,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "这是谁"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
        ]}],
    }
    before = copy.deepcopy(kwargs)
    try:
        await client._chat_completion(kwargs, "vision")
    finally:
        await _close(client)
    assert kwargs == before
    first_call = client.client.chat.completions.create.await_args_list[0].kwargs
    assert first_call == before


@pytest.mark.asyncio
async def test_second_truncated_completion_is_rejected_without_third_attempt():
    client = _make_client()
    create = AsyncMock(side_effect=[
        _completion("原文、摘要", "length"), _completion("依旧只有半句", "length"),
    ])
    client.client.chat.completions.create = create
    try:
        with pytest.raises(RuntimeError, match="未完成"):
            await client.reply(instructions="正常聊天", messages=[StoredMessage("user", "评价")])
    finally:
        await _close(client)
    assert create.await_count == 2


@pytest.mark.asyncio
async def test_recovery_attempt_uses_remaining_original_deadline(monkeypatch):
    client = _make_client()
    create = AsyncMock(side_effect=[_completion("半句", "length"), _completion("完整")])
    client.client.chat.completions.create = create
    clock = iter([0.0, 0.0, 20.0, 20.0])
    timeouts = []

    async def wait_with_captured_timeout(awaitable, *, timeout):
        timeouts.append(timeout)
        return await awaitable

    monkeypatch.setattr(llm_module.asyncio, "get_running_loop", lambda: SimpleNamespace(
        time=lambda: next(clock)
    ))
    monkeypatch.setattr(llm_module.asyncio, "wait_for", wait_with_captured_timeout)
    try:
        assert await client.reply(instructions="正常聊天", messages=[]) == "完整"
    finally:
        # Closing HTTP clients needs the real event-loop functions.
        monkeypatch.undo()
        await _close(client)
    assert timeouts == [75.0, 55.0]


@pytest.mark.asyncio
async def test_completion_recovery_is_skipped_when_deadline_is_nearly_expired(monkeypatch):
    client = _make_client()
    create = AsyncMock(return_value=_completion("半句", "length"))
    client.client.chat.completions.create = create
    clock = iter([0.0, 0.0, 70.0])

    async def immediate_wait(awaitable, *, timeout):
        return await awaitable

    monkeypatch.setattr(llm_module.asyncio, "get_running_loop", lambda: SimpleNamespace(
        time=lambda: next(clock)
    ))
    monkeypatch.setattr(llm_module.asyncio, "wait_for", immediate_wait)
    try:
        with pytest.raises(RuntimeError, match="未完成"):
            await client.reply(instructions="正常聊天", messages=[])
    finally:
        monkeypatch.undo()
        await _close(client)
    assert create.await_count == 1


@pytest.mark.asyncio
async def test_content_filter_never_triggers_recovery():
    client = _make_client()
    create = AsyncMock(return_value=_completion(None, "content_filter"))
    client.client.chat.completions.create = create
    try:
        with pytest.raises(RuntimeError, match="未完成"):
            await client.reply(instructions="正常聊天", messages=[])
    finally:
        await _close(client)
    assert create.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("response", [_completion("半句", "length"), _completion(None)])
async def test_other_models_do_not_auto_recover(response):
    client = _make_client(model="gpt-test")
    create = AsyncMock(return_value=response)
    client.client.chat.completions.create = create
    try:
        with pytest.raises(RuntimeError):
            await client.reply(instructions="正常聊天", messages=[])
    finally:
        await _close(client)
    assert create.await_count == 1


@pytest.mark.asyncio
async def test_vision_recovery_keeps_original_image_and_question():
    client = _make_client()
    create = AsyncMock(side_effect=[
        _completion("她是", "length"), _completion("画面左侧有一名戴帽子的角色。"),
    ])
    client.client.chat.completions.create = create
    image_url = "data:image/png;base64,AA=="
    try:
        result = await client.describe_images(prompt="图中的人物是谁？", image_urls=[image_url])
    finally:
        await _close(client)
    assert result == "画面左侧有一名戴帽子的角色。"
    first_kwargs, second_kwargs = [call.kwargs for call in create.await_args_list]
    original_input = first_kwargs["messages"][0]
    assert original_input in second_kwargs["messages"]
    assert original_input["content"] == [
        {"type": "text", "text": "图中的人物是谁？"},
        {"type": "image_url", "image_url": {"url": image_url}},
    ]
    assert second_kwargs["reasoning_effort"] == "low"
    assert first_kwargs["max_completion_tokens"] == second_kwargs["max_completion_tokens"]


@pytest.mark.asyncio
async def test_responses_api_truncation_still_rejects_without_recovery():
    client = _make_client()
    client.api_mode = "responses"
    create = AsyncMock(return_value=SimpleNamespace(
        output_text="不完整的摘要", status="incomplete", usage=None,
    ))
    client.client.responses.create = create
    try:
        with pytest.raises(RuntimeError, match="未完成"):
            await client.compact_reply(
                instructions="压缩", messages=[], max_output_tokens=1000,
                purpose="summary",
            )
    finally:
        await _close(client)
    assert create.await_count == 1


class _BrokenDownload(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b"discard-these-partial-bytes"
        raise httpx.RemoteProtocolError("peer closed connection before complete body")


def _png_bytes():
    output = io.BytesIO()
    Image.new("RGB", (2, 2), "red").save(output, format="PNG")
    return output.getvalue()


@pytest.mark.asyncio
async def test_qq_download_protocol_retry_discards_partial_bytes():
    attempts = 0
    complete_image = _png_bytes()

    def handler(request):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(200, stream=_BrokenDownload(), request=request)
        return httpx.Response(200, content=complete_image, request=request)

    client = _make_client(handler=handler)
    try:
        data_url, size = await client._download_image_as_data_url(
            "https://multimedia.nt.qq.com.cn/example.png"
        )
    finally:
        await _close(client)
    assert attempts == 2
    assert base64.b64decode(data_url.split(",", 1)[1]) == complete_image
    assert size == len(complete_image)


@pytest.mark.asyncio
async def test_qq_download_protocol_failure_stops_after_second_attempt():
    attempts = 0

    def handler(request):
        nonlocal attempts
        attempts += 1
        return httpx.Response(200, stream=_BrokenDownload(), request=request)

    client = _make_client(handler=handler)
    try:
        with pytest.raises(RuntimeError, match="网络异常|下载群聊图片"):
            await client._download_image_as_data_url("https://multimedia.nt.qq.com.cn/example.png")
    finally:
        await _close(client)
    assert attempts == 2
