"""Small, bounded source-image cache. Captions are never a substitute for pixels."""
from __future__ import annotations

import base64
import hashlib
import io
import re
import sqlite3
import time
from pathlib import Path
from urllib.parse import urlsplit


_EDIT = re.compile(
    r"(?:把|将).{0,80}(?:改成|改为|[pP]成|换成|替换|去掉|删掉|加上|换掉)"
    r"|修改|编辑|修图|[pP]图|[pP]一下|改一下|去除|移除|抠图|扩图|换背景"
    r"|(?:这张|原图|图中|图片|照片).{0,25}(?:调亮|变成|加个|加一个|改|换)"
    r"|(?:继续|再).{0,8}(?:改|修|调亮)", re.DOTALL,
)


def is_edit_request(text: str, *, has_image: bool = False) -> bool:
    if re.search(r"(?:不要|不用|别)(?:再)?(?:改|修|编辑)", text):
        return False
    if re.search(r"修改(?:记忆|人设|群规|提示词)|改(?:名字|昵称|称呼)", text):
        return False
    if has_image and re.search(r"改一下|修一下|改成|换成|加上|去掉|删掉|变成|调亮|换个背景", text):
        return True
    if re.search(r"(?:照着?|根据|基于|参考).{0,12}(?:这张|原图|图片|照片).{0,15}(?:画|生成|做|改)", text):
        return True
    return bool(_EDIT.search(text)) and bool(re.search(
        r"图|照片|画面|背景|数字|文字|颜色|衣服|头发|换成|改成|改为|继续改|再改|抠|调亮", text
    ))


class ImageSourceUnavailable(ValueError):
    pass


class ImageSourceCache:
    MAX_BYTES = 96 * 1024 * 1024
    MAX_IMAGE_BYTES = 8 * 1024 * 1024
    TTL = 24 * 3600
    MAX_ENTRIES = 128

    def __init__(self, directory: Path):
        self.directory = directory.resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.directory.chmod(0o700)
        self.db = sqlite3.connect(self.directory / "index.sqlite3")
        self.db.execute("""CREATE TABLE IF NOT EXISTS sources (
            key TEXT PRIMARY KEY, conversation TEXT, sender TEXT, event TEXT,
            created REAL, bytes INTEGER, filename TEXT)""")
        (self.directory / "index.sqlite3").chmod(0o600)
        self.cleanup()

    def cleanup(self):
        rows = self.db.execute("SELECT key, created, bytes, filename FROM sources ORDER BY created DESC").fetchall()
        kept = 0
        count = 0
        for key, created, size, filename in rows:
            path = self.directory / filename
            if created < time.time() - self.TTL or kept + size > self.MAX_BYTES or count >= self.MAX_ENTRIES or not path.is_file():
                if path.parent == self.directory:
                    path.unlink(missing_ok=True)
                self.db.execute("DELETE FROM sources WHERE key=?", (key,))
            else:
                kept += size
                count += 1
        self.db.commit()
        known = {row[0] for row in self.db.execute("SELECT filename FROM sources")}
        for orphan in self.directory.glob("*.png"):
            if re.fullmatch(r"[0-9a-f]{64}\.png", orphan.name) and orphan.name not in known:
                orphan.unlink(missing_ok=True)

    def store(self, conversation: str, sender: str, event: str, token: str, raw: bytes, *, created_at: float | None = None) -> Path:
        from PIL import Image, ImageOps

        if len(raw) > self.MAX_IMAGE_BYTES:
            raise ImageSourceUnavailable("单张原图超过 8 MiB，请缩小后重发。")
        with Image.open(io.BytesIO(raw)) as image:
            if image.width * image.height > 12_000_000:
                raise ImageSourceUnavailable("原图像素过大，请缩小后重发。")
            # Lossless normalized first frame, also removes EXIF/private metadata.
            image = ImageOps.exif_transpose(image).convert("RGB")
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            encoded = buf.getvalue()
        if len(encoded) > 16 * 1024 * 1024:
            raise ImageSourceUnavailable("原图处理后过大，请缩小后重发。")
        key = hashlib.sha256((conversation + "\0" + token).encode()).hexdigest()
        path = self.directory / (key + ".png")
        path.write_bytes(encoded)
        path.chmod(0o600)
        self.db.execute("INSERT OR REPLACE INTO sources VALUES (?,?,?,?,?,?,?)",
                        (key, conversation, sender, event, time.time() if created_at is None else created_at, len(encoded), path.name))
        self.db.commit()
        self.cleanup()
        return path

    async def fetch(self, message, urls: tuple[str, ...], downloader, *, created_at: float | None = None) -> tuple[Path, ...]:
        if len(urls) > 4:
            raise ImageSourceUnavailable("一次最多处理四张参考图，请明确要修改哪张。")
        paths = []
        self.cleanup()
        for url in dict.fromkeys(urls):
            key = hashlib.sha256((message.conversation_key + "\0" + url).encode()).hexdigest()
            row = self.db.execute("SELECT filename FROM sources WHERE key=?", (key,)).fetchone()
            if row and (self.directory / row[0]).is_file():
                paths.append(self.directory / row[0])
                continue
            host = (urlsplit(url).hostname or "").lower()
            if not any(host == d or host.endswith("." + d) for d in ("qq.com", "qpic.cn", "gtimg.cn", "gtimg.com")):
                raise ImageSourceUnavailable("请直接发送或引用 QQ 图片附件，不使用外部图片地址。")
            try:
                data_url, _ = await downloader(url)
                raw = base64.b64decode(data_url.split(",", 1)[1], validate=True)
                paths.append(self.store(message.conversation_key, message.user_id,
                                        message.event_id, url, raw, created_at=created_at))
            except ImageSourceUnavailable:
                raise
            except Exception:
                raise ImageSourceUnavailable("原图链接已失效或下载失败，请重新发送原图。") from None
        if sum(path.stat().st_size for path in paths) > 16 * 1024 * 1024:
            raise ImageSourceUnavailable("本次原图总大小超过 16 MiB，请减少图片数量。")
        return tuple(paths)

    async def resolve(self, message, downloader) -> tuple[Path, ...]:
        # A quote is an explicit selection, never fall back to an unrelated photo.
        urls = message.quoted_image_urls or message.image_urls
        if urls:
            paths = await self.fetch(message, urls, downloader)
            if len(paths) > 1 and not re.search(r"合成|融合|拼|参考|第一张|图一|两张|这几张", message.content):
                raise ImageSourceUnavailable("这里有多张原图。引用要改的那一张，或说明第一张与其他图的用途。")
            return paths
        if message.quoted_content:
            raise ImageSourceUnavailable("这条引用没有带回原图。请直接引用图片或重新发送，不要只引用文字。")
        self.cleanup()
        rows = self.db.execute(
            "SELECT event, filename FROM sources WHERE conversation=? AND sender=? AND created>? ORDER BY created DESC",
            (message.conversation_key, message.user_id, time.time() - 600),
        ).fetchall()
        if not rows:
            raise ImageSourceUnavailable("把要改的原图发来或引用给我。我不会拿文字描述冒充原图。")
        paths = tuple(self.directory / r[1] for r in rows if r[0] == rows[0][0])
        if len(paths) != 1:
            raise ImageSourceUnavailable("你刚才发了多张图，引用要修改的那张，别让我猜错。")
        return paths

    def close(self):
        self.db.close()
