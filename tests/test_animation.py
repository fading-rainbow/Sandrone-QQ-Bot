import base64
import io
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from openai import BadRequestError
from PIL import Image, ImageDraw

from talk_bot.animation import ImageAnimationError, animation_requested, encode_storyboard, read_gif
from talk_bot.character_refs import CharacterReference, CharacterReferenceLibrary, CharacterSubject
from talk_bot.image_gen import GeneratedImage
from talk_bot.memory import MemoryStore
from talk_bot.service import ChatService, ImageInspection, IncomingMessage
from test_image_backend import _generator_with_transport, _multipart_parts
from test_qq_adapter import StubImageService, _runner_for_image_test


def storyboard(*, moving=True):
    image = Image.new("RGB", (1536, 1024), "white")
    draw = ImageDraw.Draw(image)
    for n in range(6):
        x, y = n % 3 * 512 + 40 + (n * 20 if moving else 0), n // 3 * 512 + 100
        draw.rectangle((x, y, x + 75, y + 75), fill="red")
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def image_result(blob):
    return httpx.Response(200, json={"created": 1, "data": [{"b64_json": base64.b64encode(blob).decode()}]})


@pytest.mark.parametrize("prompt,expected", [
    ("生成一张sandrone和columbina的q版gif图", True),
    ("画个循环动图", True),
    ("别做 GIF，只画静态图", False),
    ("用静态 JPEG 画苹果", False),
])
def test_detect_explicit_gif_requests(prompt, expected):
    assert animation_requested(prompt) is expected


def test_grid_becomes_six_changing_looping_frames_and_rejects_static():
    blob = encode_storyboard(storyboard())
    frames = read_gif(blob)
    assert len(frames) == 6
    assert frames[0].size == (508, 508)
    assert frames[0].tobytes() != frames[-1].tobytes()
    with Image.open(io.BytesIO(blob)) as image:
        assert image.info["loop"] == 0
        assert image.info["duration"] == 200
    with pytest.raises(ImageAnimationError):
        encode_storyboard(storyboard(moving=False))
    with pytest.raises(ImageAnimationError):
        read_gif(storyboard())


@pytest.mark.asyncio
async def test_native_gif_generation_requests_animation_not_jpeg(monkeypatch, tmp_path):
    blob = encode_storyboard(storyboard())

    def handler(request):
        assert request.url.path == "/v1/images/generations"
        fields = json.loads(request.content)
        assert fields["output_format"] == "gif"
        assert fields["frame_count"] == 6 and fields["frame_duration_ms"] == 200
        assert "output_compression" not in fields
        return image_result(blob)

    generator = _generator_with_transport(monkeypatch, tmp_path, handler)
    try:
        generator.model = "gpt-image-2.6-sunburst"
        result = await generator.generate_animation("一颗红球弹跳")
        assert result.path.suffix == ".gif"
        assert result.path.read_bytes() == blob
    finally:
        await generator.close()


@pytest.mark.asyncio
async def test_gif_capability_fallback_keeps_all_reference_pixels(monkeypatch, tmp_path):
    refs = []
    for name, color in (("桑多涅", "red"), ("哥伦比娅", "blue")):
        path = tmp_path / (color + ".png")
        Image.new("RGB", (32, 32), color).save(path)
        refs.append(CharacterReference(name, path, "official"))
    observed = []

    def handler(request):
        assert request.url.path == "/v1/images/edits"
        fields, files = _multipart_parts(request)
        assert [content for _, _, content in files] == [ref.path.read_bytes() for ref in refs]
        assert "图片1只对应角色【桑多涅】" in fields["prompt"]
        assert "图片2只对应角色【哥伦比娅】" in fields["prompt"]
        observed.append(fields["output_format"])
        if fields["output_format"] == "gif":
            assert fields["frame_count"] == "6"
            return httpx.Response(400, json={"error": {"message": "GIF output is not enabled for this request."}})
        assert fields["size"] == "1536x1024" and "frame_count" not in fields
        return image_result(storyboard())

    generator = _generator_with_transport(monkeypatch, tmp_path, handler)
    try:
        generator.model = "gpt-image-2.6-sunburst"
        result = await generator.generate_animation("两人挥手", character_references=tuple(refs))
        await generator.generate_animation("两人挥手", character_references=tuple(refs))
        assert observed == ["gif", "png", "png"]
        assert len(read_gif(result.path.read_bytes())) == 6
        assert result.character_references == tuple(refs)
    finally:
        await generator.close()


@pytest.mark.asyncio
async def test_content_policy_does_not_trigger_animation_fallback(monkeypatch, tmp_path):
    observed = []

    def handler(request):
        observed.append(json.loads(request.content)["output_format"])
        return httpx.Response(400, json={"error": {"message": "content_policy_violation"}})

    generator = _generator_with_transport(monkeypatch, tmp_path, handler)
    try:
        with pytest.raises(BadRequestError):
            await generator.generate_animation("示例")
        assert observed == ["gif"]
    finally:
        await generator.close()


@pytest.mark.asyncio
async def test_gif_vision_reviews_sampled_real_frames(tmp_path):
    p = tmp_path / "motion.gif"
    p.write_bytes(encode_storyboard(storyboard()))
    service = object.__new__(ChatService)
    service.llm = SimpleNamespace(describe_images=AsyncMock(return_value="PASS\n红球移动。"))
    result = await service.inspect_generated_image(p, "红球动图")
    assert result.accepted
    kwargs = service.llm.describe_images.await_args.kwargs
    assert "验收预览" in kwargs["prompt"] and "循环 GIF 校验" in kwargs["prompt"]
    encoded = kwargs["image_urls"][0].split(",", 1)[1]
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as preview:
        assert preview.format == "JPEG" and preview.size == (960, 640)


@pytest.mark.asyncio
@pytest.mark.parametrize("accepted", [True, False])
async def test_qq_routes_gif_upload_or_reports_animation_failure(tmp_path, accepted):
    memory = MemoryStore(tmp_path / "memory.db")
    service = StubImageService(memory, inspections=[ImageInspection(accepted, "复核")] * 2)
    runner = _runner_for_image_test(service)
    runner.image_sources = SimpleNamespace(store=AsyncMock())
    p = tmp_path / "motion.gif"
    blob = encode_storyboard(storyboard())

    async def render(*args, **kwargs):
        p.write_bytes(blob)
        return GeneratedImage(p, "animation")

    async def upload(**kwargs):
        from pathlib import Path
        assert kwargs["source"].endswith(".gif")
        assert Path(kwargs["source"]).read_bytes() == blob
        return "file-info"

    runner.image_generator = SimpleNamespace(generate_animation=AsyncMock(side_effect=render), generate=AsyncMock())
    runner.media_uploader.upload.side_effect = upload
    message = IncomingMessage("m", "group", "g", "u", "画一颗红球弹跳 GIF")
    await runner._handle_image_request(SimpleNamespace(chat_scope="group", chat_id="g", message_id="m"),
                                       message, message.content)
    runner.image_generator.generate.assert_not_awaited()
    assert runner.media_uploader.upload.await_count == int(accepted)
    assert runner.api.post_group_message.await_count == int(accepted)
    if accepted:
        sent = runner.api.post_group_message.await_args.args[1].to_dict()
        assert sent["msg_id"] == "m" and sent["msg_type"] == 7
        assert sent["content"] == " " and sent["media"]["file_info"] == "file-info"
    runner.image_sources.store.assert_not_called()
    if not accepted:
        assert any("循环 GIF" in reply for reply in service.remembered)
        assert not any("角色偏离" in reply for reply in service.remembered)
    memory.close()


@pytest.mark.asyncio
async def test_columbina_alias_uses_exact_localized_hoyo_entry(tmp_path):
    library = CharacterReferenceLibrary(tmp_path / "refs")
    image = io.BytesIO()
    Image.new("RGB", (32, 32)).save(image, format="PNG")
    library._hoyo = AsyncMock(return_value=("哥伦比娅", "official", "原神", [image.getvalue()]))
    try:
        refs = await library.resolve((CharacterSubject("Columbina", kind="fictional"),))
        assert refs[0].name == "哥伦比娅"
        library._hoyo.assert_awaited_once_with("哥伦比娅")
    finally:
        await library.close()
