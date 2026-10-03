import base64
import io
import os
import time
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from talk_bot.image_sources import ImageSourceCache, ImageSourceUnavailable, is_edit_request
from talk_bot.service import IncomingMessage


def raw_image():
    buf = io.BytesIO()
    Image.new("RGB", (80, 160), "red").save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.asyncio
async def test_real_qq_multimedia_host_is_cached_but_lookalikes_rejected(tmp_path):
    cache = ImageSourceCache(tmp_path)
    downloader = AsyncMock(return_value=("data:image/png;base64," + base64.b64encode(raw_image()).decode(), 100))
    try:
        paths = await cache.fetch(incoming(), ("https://multimedia.nt.qq.com.cn/test",), downloader)
        assert paths[0].is_file()
        with pytest.raises(ImageSourceUnavailable):
            await cache.fetch(incoming(), ("https://multimedia.nt.qq.com.cn.evil.example/test",), downloader)
        assert downloader.await_count == 1
    finally:
        cache.close()


def incoming(**kwargs):
    return replace(IncomingMessage("e1", "group", "g1", "u1", "把图中的数字改成65432"), **kwargs)


@pytest.mark.parametrize("text", ["把图中的原石数量p成65432", "帮我把图中的原石数量改成65432",
                                 "把她的衣服换成白色", "继续改上一张图", "根据这张图片画一个头像"])
def test_edit_intent(text):
    assert is_edit_request(text)


@pytest.mark.parametrize("text", ["修改密码", "修改记忆", "不要改图", "图中是谁", "画一张桑多涅立绘", "修改提示词"])
def test_not_edit_intent(text):
    assert not is_edit_request(text)


def test_brief_edit_uses_attachment_context():
    assert is_edit_request("改成65432", has_image=True)
    assert is_edit_request("修一下", has_image=True)
    assert not is_edit_request("修一下")
    assert not is_edit_request("修改记忆", has_image=True)
    assert not is_edit_request("不要改图", has_image=True)


@pytest.mark.asyncio
async def test_quote_wins_over_recent_or_attached_image_and_reuses_cache(tmp_path):
    cache = ImageSourceCache(tmp_path)
    downloader = AsyncMock(return_value=("data:image/png;base64," + base64.b64encode(raw_image()).decode(), 100))
    message = incoming(image_urls=("https://gchat.qpic.cn/new",), quoted_image_urls=("https://gchat.qpic.cn/quoted",))
    paths = await cache.resolve(message, downloader)
    assert downloader.await_args.args == ("https://gchat.qpic.cn/quoted",)
    cache.close()
    cache = ImageSourceCache(tmp_path)
    assert await cache.resolve(message, downloader) == paths
    assert downloader.await_count == 1
    cache.close()


@pytest.mark.asyncio
async def test_recent_sources_are_isolated_and_explicit_quote_never_falls_back(tmp_path):
    cache = ImageSourceCache(tmp_path)
    cache.store("group:g1", "u2", "e2", "other", raw_image())
    cache.store("group:g2", "u1", "e3", "othergroup", raw_image())
    downloader = AsyncMock(side_effect=RuntimeError("expired"))
    with pytest.raises(ImageSourceUnavailable):
        await cache.resolve(incoming(), downloader)
    own = cache.store("group:g1", "u1", "e1", "own", raw_image())
    assert await cache.resolve(incoming(), downloader) == (own,)
    with pytest.raises(ImageSourceUnavailable):
        await cache.resolve(incoming(quoted_image_urls=("https://gchat.qpic.cn/expired",)), downloader)
    with pytest.raises(ImageSourceUnavailable):
        await cache.resolve(incoming(quoted_content="another message"), downloader)
    with pytest.raises(ImageSourceUnavailable):
        await cache.resolve(incoming(image_urls=("http://127.0.0.1/private",)), downloader)
    cache.close()


@pytest.mark.asyncio
async def test_multiple_recent_images_require_selection_and_cache_is_bounded(tmp_path):
    cache = ImageSourceCache(tmp_path)
    first = cache.store("group:g1", "u1", "album", "a", raw_image())
    cache.store("group:g1", "u1", "album", "b", raw_image())
    with pytest.raises(ImageSourceUnavailable):
        await cache.resolve(incoming(), AsyncMock())
    cache.db.execute("UPDATE sources SET created=0 WHERE filename=?", (first.name,))
    cache.db.commit()
    cache.cleanup()
    assert not first.exists()
    cache.MAX_BYTES = 1
    cache.cleanup()
    assert not list(tmp_path.glob("*.png"))
    cache.close()


@pytest.mark.asyncio
async def test_slow_download_does_not_reorder_original_image_ownership(tmp_path):
    cache = ImageSourceCache(tmp_path)
    now = time.time()
    newest = cache.store("group:g1", "u1", "newer", "new", raw_image(), created_at=now)
    cache.store("group:g1", "u1", "older", "old", raw_image(), created_at=now - 10)
    assert await cache.resolve(incoming(), AsyncMock()) == (newest,)
    cache.close()
