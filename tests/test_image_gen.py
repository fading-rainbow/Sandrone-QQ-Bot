import base64
import io
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from talk_bot.image_gen import ImageContentPolicyError, ImageEditMismatch, ImageGenerator, select_image_layout


class FakePolicyError(RuntimeError):
    body = {"error": {"code": "content_policy_violation"}}


def test_save_1080p_crops_and_upscales_landscape_source(tmp_path: Path) -> None:
    source = Image.new("RGB", (1536, 1024), "#887766")
    raw = io.BytesIO()
    source.save(raw, format="JPEG")

    generator = object.__new__(ImageGenerator)
    generator.output_dir = tmp_path
    result = generator._save_output(raw.getvalue(), (1920, 1080))

    with Image.open(result) as saved:
        assert saved.size == (1920, 1080)
    assert result.stat().st_size > 0


def test_layout_selection_distinguishes_portrait_square_and_scene() -> None:
    assert select_image_layout("桑多涅单人立绘").name == "portrait"
    assert select_image_layout("给我一个桑多涅头像").name == "square"
    assert select_image_layout("桑多涅和哥伦比娅在工坊喝茶").name == "landscape"


def _encoded_jpeg(size: tuple[int, int] = (64, 64)) -> str:
    source = Image.new("RGB", size, "#887766")
    raw = io.BytesIO()
    source.save(raw, format="JPEG")
    return base64.b64encode(raw.getvalue()).decode("ascii")


@pytest.mark.asyncio
async def test_sandrone_generation_uses_reference_edit_and_portrait_layout(
    tmp_path: Path,
) -> None:
    references = (tmp_path / "face.jpg", tmp_path / "outfit.jpg")
    for reference in references:
        Image.new("RGB", (32, 32), "white").save(reference)
    response = SimpleNamespace(data=[SimpleNamespace(b64_json=_encoded_jpeg())])
    images = SimpleNamespace(edit=AsyncMock(return_value=response), generate=AsyncMock())

    generator = object.__new__(ImageGenerator)
    generator.model = "gpt-image-2"
    generator.output_dir = tmp_path
    generator.sandrone_reference_paths = references
    generator.client = SimpleNamespace(images=images)

    result = await generator.generate("桑多涅单人立绘")

    images.edit.assert_awaited_once()
    images.generate.assert_not_awaited()
    kwargs = images.edit.await_args.kwargs
    assert kwargs["size"] == "1024x1536"
    assert len(kwargs["image"]) == 2
    assert "唯一的身份与造型基准" in kwargs["prompt"]
    assert result.reference_paths == references
    with Image.open(result.path) as saved:
        assert saved.size == (1080, 1620)


@pytest.mark.asyncio
async def test_non_sandrone_generation_stays_on_text_generation(tmp_path: Path) -> None:
    response = SimpleNamespace(data=[SimpleNamespace(b64_json=_encoded_jpeg())])
    images = SimpleNamespace(edit=AsyncMock(), generate=AsyncMock(return_value=response))

    generator = object.__new__(ImageGenerator)
    generator.model = "gpt-image-2"
    generator.output_dir = tmp_path
    generator.sandrone_reference_paths = (tmp_path / "unused.jpg",)
    generator.client = SimpleNamespace(images=images)

    result = await generator.generate("哥伦比娅在雪地散步")

    images.generate.assert_awaited_once()
    images.edit.assert_not_awaited()
    assert result.reference_paths == ()


@pytest.mark.asyncio
async def test_content_policy_rejection_retries_once_with_safe_scene(
    tmp_path: Path,
) -> None:
    references = (tmp_path / "face.jpg", tmp_path / "outfit.jpg")
    for reference in references:
        Image.new("RGB", (32, 32), "white").save(reference)
    response = SimpleNamespace(data=[SimpleNamespace(b64_json=_encoded_jpeg())])
    images = SimpleNamespace(
        edit=AsyncMock(side_effect=[FakePolicyError("blocked"), response]),
        generate=AsyncMock(),
    )
    generator = object.__new__(ImageGenerator)
    generator.model = "gpt-image-2"
    generator.output_dir = tmp_path
    generator.sandrone_reference_paths = references
    generator.client = SimpleNamespace(images=images)

    result = await generator.generate("桑多涅和哥伦比娅在床上嬉闹")

    assert images.edit.await_count == 2
    retry_prompt = images.edit.await_args_list[1].kwargs["prompt"]
    assert "枕头大战" in retry_prompt
    assert "完全无性暗示" in retry_prompt
    assert "床上嬉闹" not in retry_prompt
    assert result.safety_rewritten is True
    assert result.path.is_file()


@pytest.mark.asyncio
async def test_second_content_policy_rejection_has_specific_error(tmp_path: Path) -> None:
    images = SimpleNamespace(
        edit=AsyncMock(),
        generate=AsyncMock(
            side_effect=[FakePolicyError("blocked"), FakePolicyError("still blocked")]
        ),
    )
    generator = object.__new__(ImageGenerator)
    generator.model = "gpt-image-2"
    generator.output_dir = tmp_path
    generator.sandrone_reference_paths = ()
    generator.client = SimpleNamespace(images=images)

    with pytest.raises(ImageContentPolicyError):
        await generator.generate("两个人在床上嬉闹")
    assert images.generate.await_count == 2


def test_stale_generated_images_are_cleaned_without_touching_recent_files(
    tmp_path: Path,
) -> None:
    old = tmp_path / "sandrone-old.jpg"
    recent = tmp_path / "sandrone-recent.jpg"
    unrelated = tmp_path / "keep.jpg"
    for path in (old, recent, unrelated):
        path.write_bytes(b"x")
    os.utime(old, (100, 100))
    os.utime(recent, (3900, 3900))

    generator = object.__new__(ImageGenerator)
    generator.output_dir = tmp_path
    assert generator._cleanup_stale_files(now=4000) == 1
    assert old.exists() is False
    assert recent.exists() is True
    assert unrelated.exists() is True


@pytest.mark.asyncio
async def test_edit_uploads_actual_source_and_preserves_dimensions(tmp_path):
    original = tmp_path / "source.png"
    Image.new("RGB", (864, 1920), "red").save(original)
    async def edit(**kwargs):
        assert kwargs["image"][0].read() == original.read_bytes()
        assert kwargs["size"] == "864x1920"
        assert kwargs["quality"] == "high"
        assert kwargs["output_format"] == "png"
        assert "65432" in kwargs["prompt"]
        assert "桑多涅唯一" not in kwargs["prompt"]
        return SimpleNamespace(data=[SimpleNamespace(b64_json=_encoded_jpeg((864, 1920)))])
    gen = object.__new__(ImageGenerator)
    gen.model, gen.output_dir = "gpt-image-2", tmp_path
    gen.client = SimpleNamespace(images=SimpleNamespace(edit=AsyncMock(side_effect=edit), generate=AsyncMock()))
    result = await gen.edit("把原石数量改成65432", (original,))
    gen.client.images.generate.assert_not_awaited()
    assert result.layout == "edit" and not result.identity_sensitive
    with Image.open(result.path) as img:
        assert img.size == (864, 1920) and img.format == "PNG"


@pytest.mark.asyncio
async def test_edit_never_falls_back_or_silently_crops_wrong_layout(tmp_path):
    original = tmp_path / "source.png"
    Image.new("RGB", (864, 1920), "red").save(original)
    gen = object.__new__(ImageGenerator)
    gen.model, gen.output_dir = "gpt-image-2", tmp_path
    gen.client = SimpleNamespace(images=SimpleNamespace(edit=AsyncMock(side_effect=FakePolicyError()), generate=AsyncMock()))
    with pytest.raises(ImageContentPolicyError):
        await gen.edit("修改原图", (original,))
    gen.client.images.edit.assert_awaited_once()
    gen.client.images.generate.assert_not_awaited()
    gen.client.images.edit.side_effect = None
    gen.client.images.edit.return_value = SimpleNamespace(data=[SimpleNamespace(b64_json=_encoded_jpeg((1080, 1080)))])
    with pytest.raises(ImageEditMismatch):
        await gen.edit("修改原图", (original,))
