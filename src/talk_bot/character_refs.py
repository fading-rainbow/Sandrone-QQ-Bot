"""Ground character identities in official pixels, with a small private disk cache."""
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar
from urllib.parse import urlsplit

import httpx
from PIL import Image, ImageOps

from .image_sources import ImageSourceUnavailable


class CharacterReferenceUnavailable(ImageSourceUnavailable):
    pass


@dataclass(frozen=True)
class CharacterReference:
    name: str
    path: Path
    source: str
    facts: str = ""


def reference_mapping(references: tuple[CharacterReference, ...], *, first_image: int = 1) -> str:
    return "\n".join(
        f"图片{i}只对应角色【{ref.name}】；资料来源：{ref.source}；身份资料：{ref.facts}"
        for i, ref in enumerate(references, first_image)
    )


def parse_json_object(text: str) -> dict:
    # One object only: explanations/trailing text are not silently accepted.
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    value = json.loads(text)
    if not isinstance(value, dict):
        raise TypeError("Expected a JSON object")
    return value


class CharacterReferenceLibrary:
    API = "https://sg-wiki-api.hoyolab.com/hoyowiki/wapi"
    MAX_CHARACTERS = 3  # candidate + three sheets fits the existing vision limit
    MAX_ENTRIES = 32
    MAX_CACHE_BYTES = 32 * 1024 * 1024
    MAX_DOWNLOAD_BYTES = 4 * 1024 * 1024
    MAX_JSON_BYTES = 2 * 1024 * 1024
    TTL = 24 * 3600
    HEADERS: ClassVar[dict[str, str]] = {"x-rpc-language": "zh-cn", "Referer": "https://wiki.hoyolab.com"}
    IMAGE_HOSTS = frozenset({"act-webstatic.hoyoverse.com", "upload-static.hoyoverse.com"})

    def __init__(self, directory: Path, sandrone_paths: tuple[Path, ...] = (), *, client=None):
        self.directory = directory.resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.directory.chmod(0o700)
        self.sandrone_paths = sandrone_paths
        self.client = client or httpx.AsyncClient(timeout=15, follow_redirects=False, headers=self.HEADERS)
        self._lock = asyncio.Lock()
        self.cleanup()

    @staticmethod
    def _key(name: str) -> str:
        return hashlib.sha256(name.casefold().encode()).hexdigest()

    async def _download(self, url: str, limit: int, *, params=None) -> bytes:
        chunks = []
        received = 0
        async with self.client.stream("GET", url, params=params, headers=self.HEADERS) as response:
            response.raise_for_status()
            if int(response.headers.get("content-length", "0")) > limit:
                raise ValueError("Reference download is too large")
            async for chunk in response.aiter_bytes():
                received += len(chunk)
                if received > limit:
                    raise ValueError("Reference download is too large")
                chunks.append(chunk)
        return b"".join(chunks)

    async def _api(self, endpoint: str, params: dict) -> dict:
        body = json.loads(await self._download(self.API + endpoint, self.MAX_JSON_BYTES, params=params))
        if body.get("retcode") != 0 or not isinstance(body.get("data"), dict):
            raise ValueError("Official wiki unavailable")
        return body["data"]

    @classmethod
    def _safe_image_url(cls, url: str) -> str:
        parts = urlsplit(url)
        if (parts.scheme != "https" or parts.hostname not in cls.IMAGE_HOSTS
                or parts.username or parts.password or parts.port not in (None, 443)):
            raise ValueError("Not an official reference image URL")
        return url

    @staticmethod
    def _sheet(blobs: list[bytes]) -> bytes:
        # A single sheet per identity keeps ALL requested characters visible in review.
        canvas = Image.new("RGB", (768 * len(blobs), 1024), "white")
        for i, blob in enumerate(blobs):
            with Image.open(io.BytesIO(blob)) as image:
                if image.width * image.height > 12_000_000:
                    raise ValueError("Reference image is too large")
                image = ImageOps.exif_transpose(image).convert("RGBA")
                image.thumbnail((768, 1024), Image.Resampling.LANCZOS)
                canvas.paste(image, (i * 768 + (768 - image.width) // 2,
                                     (1024 - image.height) // 2), image)
        out = io.BytesIO()
        canvas.save(out, format="JPEG", quality=93)
        return out.getvalue()

    def _atomic_write(self, path: Path, data: bytes):
        pending = path.with_suffix(path.suffix + ".pending")
        fd = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as file:
            file.write(data)
        os.replace(pending, path)
        path.chmod(0o600)

    def cleanup(self, protected: frozenset[str] = frozenset()):
        entries = sorted(self.directory.glob("*.json"),
                         key=lambda p: (p.stem in protected, p.stat().st_mtime), reverse=True)
        count = size = 0
        for meta in entries:
            if not re.fullmatch(r"[0-9a-f]{64}\.json", meta.name):
                continue
            image = meta.with_suffix(".jpg")
            entry_size = meta.stat().st_size + (image.stat().st_size if image.is_file() else 0)
            expired = time.time() - meta.stat().st_mtime > self.TTL
            if meta.stem not in protected and (expired or not image.is_file()
                    or count >= self.MAX_ENTRIES or size + entry_size > self.MAX_CACHE_BYTES):
                meta.unlink(missing_ok=True)
                image.unlink(missing_ok=True)
            else:
                count += 1
                size += entry_size
        for path in self.directory.iterdir():
            if (re.fullmatch(r"[0-9a-f]{64}\.jpg(?:\.pending)?|[0-9a-f]{64}\.json\.pending", path.name)
                    and not path.with_name(path.name[:64] + ".json").is_file()):
                path.unlink(missing_ok=True)

    async def resolve(self, names: tuple[str, ...]) -> tuple[CharacterReference, ...]:
        try:
            return await asyncio.wait_for(self._resolve(names), timeout=45)
        except asyncio.TimeoutError:
            raise CharacterReferenceUnavailable("角色参考服务响应太慢，这次没有开图。稍后重试，或直接发送参考图。") from None

    async def _resolve(self, names: tuple[str, ...]) -> tuple[CharacterReference, ...]:
        names = tuple(dict.fromkeys("桑多涅" if n.casefold() == "sandrone" else n for n in names))
        if len(names) > self.MAX_CHARACTERS:
            raise CharacterReferenceUnavailable("一次最多准确核对三名角色，请分成几张画。")
        async with self._lock:
            self.cleanup()
            references = []
            for name in names:
                try:
                    references.append(await self._resolve_one(name))
                except Exception as exc:
                    raise CharacterReferenceUnavailable(
                        f"【{name}】的可靠角色参考暂时没找到或无法读取。请发参考图并明确角色名字，"
                        "再说“参考这张画……”；我不会凭印象编造外形。"
                    ) from exc
            self.cleanup(frozenset(self._key(n) for n in names))
            return tuple(references)

    async def _resolve_one(self, name: str) -> CharacterReference:
        if not name or len(name) > 60:
            raise ValueError("Invalid character name")
        key = self._key(name)
        meta, image = self.directory / (key + ".json"), self.directory / (key + ".jpg")
        if meta.is_file() and image.is_file():
            try:
                data = json.loads(meta.read_text(encoding="utf-8"))
                if data["query"] == name and time.time() - data["fetched_at"] < self.TTL:
                    os.utime(meta, None)
                    return CharacterReference(data["name"], image, data["source"], data["facts"])
            except (KeyError, ValueError, OSError):
                pass
        if name == "桑多涅" and self.sandrone_paths:
            blobs = [path.read_bytes() for path in self.sandrone_paths[:2]]
            canonical, source, facts = "桑多涅", "用户指定的本机角色参考档案", "桑多涅 / Sandrone；外观以参考像素为准"
        else:
            search = await self._api("/search", {"keyword": name})
            matches = [row for row in search.get("list", []) if row.get("name", "").casefold() == name.casefold()]
            # An exact name alone is insufficient: a food/item can share a name.
            matches = [row for row in matches if any(str(menu.get("id")) == "2"
                for menu in [row.get("menu", {})] + row.get("menu", {}).get("sub_menus", []))]
            if len(matches) != 1 or not str(matches[0].get("entry_page_id", "")).isdigit():
                raise ValueError("No unambiguous canonical character")
            entry_id = str(matches[0]["entry_page_id"])
            page = (await self._api("/entry_page", {"entry_page_id": entry_id}))["page"]
            if str(page.get("menu_id")) != "2" or page.get("name") != matches[0]["name"]:
                raise ValueError("Character entry mismatch")
            urls = [page.get("header_img_url"), page.get("icon_url")]
            if not urls[0]:
                raise ValueError("No full character reference")
            blobs = [await self._download(self._safe_image_url(url), self.MAX_DOWNLOAD_BYTES)
                     for url in dict.fromkeys(urls) if url]
            canonical = page["name"]
            source = f"https://wiki.hoyolab.com/pc/genshin/entry/{entry_id}?lang=zh-cn"
            facts = re.sub(r"<[^>]*>", "", str(page.get("desc", "")))[:400]
        encoded = await asyncio.to_thread(self._sheet, blobs)
        if len(encoded) > 2 * 1024 * 1024:
            raise ValueError("Reference sheet is too large")
        data = {"query": name, "name": canonical, "source": source, "facts": facts, "fetched_at": time.time()}
        self._atomic_write(image, encoded)
        self._atomic_write(meta, json.dumps(data, ensure_ascii=False).encode())
        return CharacterReference(canonical, image, source, facts)

    async def close(self):
        await self.client.aclose()
