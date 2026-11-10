"""Bounded GIF encoding and visual review helpers (no external codecs)."""

import base64
import io
import re
from pathlib import Path

from PIL import Image

FRAME_COUNT = 6
FRAME_DURATION_MS = 200
FRAME_SIZE = 512
MAX_GIF_BYTES = 5_000_000
STORYBOARD_SIZE = f"{3 * FRAME_SIZE}x{2 * FRAME_SIZE}"
_GIF_REQUEST = re.compile(r"(?<![a-z0-9])gif(?![a-z0-9])|动图|动画表情|动态表情|动画贴纸", re.I)
_NO_GIF = re.compile(r"(?:不要|不用|别|不做|无需)(?:再)?\s*(?:生成|做|画|输出|用)?\s*(?:gif|动图|动画表情|动态表情|动画贴纸)", re.I)


class ImageAnimationError(RuntimeError):
    """The provider did not produce a usable animation."""


def animation_requested(text: str) -> bool:
    return bool(_GIF_REQUEST.search(_NO_GIF.sub("", text)))


def read_gif(raw: bytes) -> list[Image.Image]:
    if len(raw) > MAX_GIF_BYTES:
        raise ImageAnimationError("GIF 文件超过上传限制")
    try:
        with Image.open(io.BytesIO(raw)) as image:
            if (image.format != "GIF" or image.width > 1024 or image.height > 1024
                    or image.info.get("loop") != 0):
                raise ImageAnimationError("返回的文件不是有效的循环多帧 GIF")
            frames = []
            for index in range(17):
                try:
                    image.seek(index)
                except EOFError:
                    break
                if index == 16:
                    raise ImageAnimationError("GIF 帧数超过限制")
                if not 20 <= image.info.get("duration", 0) <= 2000:
                    raise ImageAnimationError("GIF 帧时长无效")
                frames.append(image.convert("RGB"))
    except (OSError, EOFError, Image.DecompressionBombError) as exc:
        raise ImageAnimationError("GIF 文件无法解码") from exc
    if len({frame.resize((32, 32)).tobytes() for frame in frames}) < 2:
        raise ImageAnimationError("GIF 中没有实际的帧变化")
    return frames


def encode_storyboard(raw: bytes) -> bytes:
    try:
        with Image.open(io.BytesIO(raw)) as image:
            if image.size != (3 * FRAME_SIZE, 2 * FRAME_SIZE) or getattr(image, "is_animated", False):
                raise ImageAnimationError("分镜没有按要求返回 3×2 帧网格")
            board = image.convert("RGB")
    except (OSError, Image.DecompressionBombError) as exc:
        raise ImageAnimationError("分镜无法解码") from exc
    frames = [
        board.crop((x * FRAME_SIZE + 2, y * FRAME_SIZE + 2,
                    (x + 1) * FRAME_SIZE - 2, (y + 1) * FRAME_SIZE - 2))
        for y in range(2) for x in range(3)
    ]
    stream = io.BytesIO()
    frames[0].save(stream, format="GIF", save_all=True, append_images=frames[1:],
                   duration=FRAME_DURATION_MS, loop=0, disposal=2, optimize=False)
    gif = stream.getvalue()
    read_gif(gif)
    return gif


def animation_review_url(path: Path) -> str:
    """Send sampled actual frames as one visual input; never pretend the grid is the deliverable."""
    frames = read_gif(path.read_bytes())
    count = min(len(frames), FRAME_COUNT)
    selected = [round(i * (len(frames) - 1) / (count - 1)) for i in range(count)]
    tile = 320
    columns = min(3, count)
    sheet = Image.new("RGB", (tile * columns, tile * ((count + columns - 1) // columns)), "white")
    for i, index in enumerate(selected):
        frame = frames[index]
        frame.thumbnail((tile - 8, tile - 8), Image.Resampling.LANCZOS)
        sheet.paste(frame, (i % columns * tile + (tile - frame.width) // 2,
                            i // columns * tile + (tile - frame.height) // 2))
    stream = io.BytesIO()
    sheet.save(stream, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(stream.getvalue()).decode("ascii")
