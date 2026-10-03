import base64
import io
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from PIL import Image

from talk_bot.llm import LLMClient


@pytest.mark.parametrize("reason", ["length", "content_filter"])
def test_partial_completion_is_never_accepted(reason):
    response = SimpleNamespace(choices=[SimpleNamespace(finish_reason=reason)])
    with pytest.raises(RuntimeError, match="未完成"):
        LLMClient._require_complete(response, "attribution_guard")


def test_partial_responses_output_is_rejected():
    with pytest.raises(RuntimeError, match="未完成"):
        LLMClient._require_complete(SimpleNamespace(status="incomplete"), "summary")


def test_gemini_reserves_thinking_without_changing_other_models():
    client = object.__new__(LLMClient)
    client.model = "gemini-3-flash"
    assert client._completion_budget(350) == 4446
    assert client._completion_budget(16) == 4112
    client.model = "gpt-test"
    assert client._completion_budget(350) == 350


@pytest.mark.asyncio
async def test_truncated_text_raises_instead_of_returning_fragment():
    from talk_bot.memory import StoredMessage
    client = _client_with_transport(lambda request: httpx.Response(200))
    client.api_mode = "chat_completions"
    client.model = "gemini-3-flash"
    client.client.chat.completions.create = AsyncMock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="原文、摘要再"), finish_reason="length")],
        usage=None))
    try:
        with pytest.raises(RuntimeError, match="未完成"):
            await client.compact_reply(instructions="校验", messages=[StoredMessage("user", "评价")],
                                       max_output_tokens=350, purpose="attribution_guard")
        assert client.client.chat.completions.create.call_args.kwargs["max_completion_tokens"] == 4446
    finally:
        await client.client.close()
        await client.vision_http.aclose()


def _png_bytes(size: tuple[int, int] = (2, 2)) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", size, "red").save(output, format="PNG")
    return output.getvalue()


def _client_with_transport(handler) -> LLMClient:
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return LLMClient(
        api_key="test",
        base_url="https://relay.example/v1",
        model="gpt-test",
        api_mode="responses",
        reasoning_effort="low",
        max_output_tokens=100,
        vision_http_client=http,
    )


def test_web_sources_are_collected_from_calls_and_annotations() -> None:
    response = SimpleNamespace(
        output=[
            SimpleNamespace(
                action=SimpleNamespace(
                    sources=[
                        {"title": "搜索来源", "url": "https://example.com/a"}
                    ]
                ),
                content=[],
            ),
            SimpleNamespace(
                action=None,
                content=[
                    SimpleNamespace(
                        annotations=[
                            SimpleNamespace(
                                title="正文引用", url="https://example.org/b"
                            ),
                            SimpleNamespace(
                                title="重复", url="https://example.com/a"
                            ),
                        ]
                    )
                ],
            ),
        ]
    )

    assert LLMClient._extract_web_sources(response) == (
        ("正文引用", "https://example.org/b"),
        ("重复", "https://example.com/a"),
    )


def test_web_source_urls_drop_tracking_parameters_and_fragments() -> None:
    assert LLMClient._clean_source_url(
        "https://example.com/news?id=7&utm_source=openai#section"
    ) == "https://example.com/news?id=7"


def test_completed_web_search_call_is_detected() -> None:
    completed = SimpleNamespace(
        output=[SimpleNamespace(type="web_search_call", status="completed")]
    )
    missing = SimpleNamespace(
        output=[SimpleNamespace(type="message", status="completed")]
    )
    assert LLMClient._has_completed_web_search_call(completed) is True
    assert LLMClient._has_completed_web_search_call(missing) is False


@pytest.mark.asyncio
async def test_web_search_rejects_uncited_non_tool_response() -> None:
    client = _client_with_transport(
        lambda request: httpx.Response(200, content=_png_bytes(), request=request)
    )
    client.client.responses.create = AsyncMock(
        return_value=SimpleNamespace(
            output_text="我无法直接联网。",
            output=[
                SimpleNamespace(
                    type="message",
                    status="completed",
                    action=None,
                    content=[SimpleNamespace(annotations=[])],
                )
            ],
            usage=None,
        )
    )
    try:
        with pytest.raises(RuntimeError, match="web_search_call"):
            await client.web_search(query="实时汇率", instructions="检索", messages=[])
    finally:
        await client.vision_http.aclose()
        await client.client.close()


@pytest.mark.asyncio
async def test_remote_vision_image_is_downloaded_and_embedded() -> None:
    image = _png_bytes()
    client = _client_with_transport(
        lambda request: httpx.Response(
            200, content=image, headers={"content-type": "image/png"}, request=request
        )
    )
    try:
        prepared = await client._prepare_vision_images(
            ["https://multimedia.nt.qq.com.cn/example.png?token=secret"]
        )
    finally:
        await client.vision_http.aclose()
        await client.client.close()

    assert len(prepared) == 1
    prefix, encoded = prepared[0].split(",", 1)
    assert prefix == "data:image/png;base64"
    assert base64.b64decode(encoded) == image


@pytest.mark.asyncio
async def test_existing_image_data_url_is_not_downloaded() -> None:
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(500, request=request)

    client = _client_with_transport(handler)
    existing = "data:image/png;base64," + base64.b64encode(_png_bytes()).decode()
    try:
        assert await client._prepare_vision_images([existing]) == (existing,)
    finally:
        await client.vision_http.aclose()
        await client.client.close()
    assert calls == 0


@pytest.mark.asyncio
async def test_remote_vision_rejects_non_image_content() -> None:
    client = _client_with_transport(
        lambda request: httpx.Response(200, content=b"not an image", request=request)
    )
    try:
        with pytest.raises(ValueError, match="不是可识别的图片"):
            await client._prepare_vision_images(["https://example.com/not-image"])
    finally:
        await client.vision_http.aclose()
        await client.client.close()


@pytest.mark.asyncio
async def test_remote_vision_retries_one_server_error() -> None:
    attempts = 0
    image = _png_bytes()

    def handler(request):
        nonlocal attempts
        attempts += 1
        return httpx.Response(503 if attempts == 1 else 200, content=image, request=request)

    client = _client_with_transport(handler)
    try:
        prepared = await client._prepare_vision_images(["https://example.com/image"])
    finally:
        await client.vision_http.aclose()
        await client.client.close()
    assert prepared[0].startswith("data:image/png;base64,")
    assert attempts == 2
