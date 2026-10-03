from __future__ import annotations

import asyncio
import base64
import io
import logging
from collections.abc import Sequence
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx
from openai import AsyncOpenAI
from PIL import Image, UnidentifiedImageError

from .memory import StoredMessage

logger = logging.getLogger(__name__)

_VISION_IMAGE_LIMIT = 4
_VISION_IMAGE_MAX_BYTES = 8 * 1024 * 1024
_VISION_TOTAL_MAX_BYTES = 16 * 1024 * 1024
_VISION_IMAGE_MAX_PIXELS = 25_000_000
_VISION_MIME_BY_FORMAT = {
    "GIF": "image/gif",
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}


class LLMClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        api_mode: str,
        reasoning_effort: str,
        max_output_tokens: int,
        vision_http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.model = model
        self.api_mode = api_mode
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            # QQ reply message IDs expire quickly. A 60-second request retried twice
            # can therefore finish successfully but still be impossible to deliver.
            # Keep one bounded attempt; higher-level callers decide how to recover.
            timeout=75.0,
            max_retries=0,
        )
        self._owns_vision_http = vision_http_client is None
        self.vision_http = vision_http_client or httpx.AsyncClient(
            follow_redirects=True,
            timeout=httpx.Timeout(15.0, connect=5.0),
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
            headers={"User-Agent": "sandrone-vision/1.0"},
        )

    async def reply(self, *, instructions: str, messages: Sequence[StoredMessage]) -> str:
        return await self._text_reply(
            instructions=instructions,
            messages=messages,
            reasoning_effort=self.reasoning_effort,
            max_output_tokens=self.max_output_tokens,
            purpose="chat",
        )

    async def compact_reply(
        self,
        *,
        instructions: str,
        messages: Sequence[StoredMessage],
        max_output_tokens: int,
        reasoning_effort: str = "none",
        purpose: str = "compact",
    ) -> str:
        """Run deterministic background/router work without the chat-sized budget."""
        return await self._text_reply(
            instructions=instructions,
            messages=messages,
            reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens,
            purpose=purpose,
        )

    async def _text_reply(
        self,
        *,
        instructions: str,
        messages: Sequence[StoredMessage],
        reasoning_effort: str,
        max_output_tokens: int,
        purpose: str,
    ) -> str:
        input_messages = [{"role": item.role, "content": item.content} for item in messages]
        if self.api_mode == "responses":
            response = await self.client.responses.create(
                model=self.model,
                instructions=instructions,
                input=input_messages,
                reasoning={"effort": reasoning_effort},
                max_output_tokens=max_output_tokens,
                store=False,
            )
            text = response.output_text
        else:
            kwargs: dict = {
                "model": self.model,
                "messages": [{"role": "system", "content": instructions}, *input_messages],
                "max_completion_tokens": self._completion_budget(max_output_tokens),
            }
            if reasoning_effort and reasoning_effort != "none":
                kwargs["reasoning_effort"] = reasoning_effort
            response = await self._chat_completion(kwargs, purpose)
            text = response.choices[0].message.content or ""
        if self.api_mode == "responses":
            self._log_usage(response, purpose)
        self._require_complete(response, purpose)
        text = text.strip()
        if not text:
            raise RuntimeError("模型返回了空文本")
        return text

    async def _chat_completion(self, kwargs: dict, purpose: str):
        """Recover Gemini's occasional reasoning-only/truncated result once.

        Keep the original evidence and token cap, and spend less time thinking
        on recovery. Never concatenate partial text or retry content filtering.
        The two attempts together share the original 75-second reply deadline.
        """
        request = dict(kwargs)
        deadline = asyncio.get_running_loop().time() + 75.0
        for attempt in range(2):
            remaining = deadline - asyncio.get_running_loop().time()
            response = await asyncio.wait_for(
                self._create_chat_completion(request), timeout=max(0.01, remaining)
            )
            self._log_usage(response, purpose + ("_recovery" if attempt else ""))
            choice = response.choices[0] if response.choices else None
            reason = getattr(choice, "finish_reason", None)
            text = getattr(getattr(choice, "message", None), "content", None)
            recoverable = reason == "length" or (reason in {None, "stop"} and not (text or "").strip())
            if not (recoverable and self.model.lower().startswith("gemini") and attempt == 0):
                self._require_complete(response, purpose)
                if not (text or "").strip():
                    raise RuntimeError(f"模型返回了空文本 purpose={purpose}")
                return response
            if deadline - asyncio.get_running_loop().time() < 10:
                self._require_complete(response, purpose)
                raise RuntimeError(f"模型返回了空文本 purpose={purpose}")
            logger.warning("模型输出需恢复 purpose=%s finish=%s attempt=1", purpose, reason)
            request["reasoning_effort"] = "low"
            request["messages"] = [*kwargs["messages"], {
                "role": "system",
                "content": "请直接给出符合原任务格式的完整结果，精简表达，不要写分析过程或复述提示词。",
            }]
        raise RuntimeError(f"模型恢复失败 purpose={purpose}")

    async def _create_chat_completion(self, kwargs: dict):
        try:
            return await self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            if "reasoning_effort" in str(exc).lower() and "reasoning_effort" in kwargs:
                fallback = {key: value for key, value in kwargs.items() if key != "reasoning_effort"}
                return await self.client.chat.completions.create(**fallback)
            raise

    def _completion_budget(self, visible_tokens: int) -> int:
        # Gemini's completion limit includes hidden thinking, unlike the visible
        # text budgets used by our routers, validators and memory writers.
        if self.model.lower().startswith("gemini"):
            return visible_tokens + 4096
        return visible_tokens

    @staticmethod
    def _require_complete(response, purpose: str) -> None:
        choices = getattr(response, "choices", None)
        reason = getattr(choices[0], "finish_reason", None) if choices else None
        status = getattr(response, "status", None)
        if reason in {"length", "content_filter"} or status in {"incomplete", "failed"}:
            # In particular never persist partial summaries or replace a good
            # answer with a truncated attribution correction. Callers fall back.
            raise RuntimeError(f"模型输出未完成 purpose={purpose} finish={reason or status}")

    async def describe_images(self, *, prompt: str, image_urls: Sequence[str]) -> str:
        if not image_urls:
            return ""
        prepared_urls = await self._prepare_vision_images(image_urls)
        if self.api_mode == "responses":
            content = [{"type": "input_text", "text": prompt}]
            content.extend(
                {"type": "input_image", "image_url": url} for url in prepared_urls
            )
            response = await self.client.responses.create(
                model=self.model,
                input=[{"role": "user", "content": content}],
                reasoning={"effort": self.reasoning_effort},
                max_output_tokens=min(self.max_output_tokens, 650),
                store=False,
            )
            text = response.output_text
        else:
            content = [{"type": "text", "text": prompt}]
            content.extend(
                {"type": "image_url", "image_url": {"url": url}}
                for url in prepared_urls
            )
            kwargs: dict = {
                "model": self.model,
                "messages": [{"role": "user", "content": content}],
                "max_completion_tokens": self._completion_budget(min(self.max_output_tokens, 650)),
            }
            if self.reasoning_effort and self.reasoning_effort != "none":
                kwargs["reasoning_effort"] = self.reasoning_effort
            response = await self._chat_completion(kwargs, "vision")
            text = response.choices[0].message.content or ""
        if self.api_mode == "responses":
            self._log_usage(response, "vision")
        self._require_complete(response, "vision")
        text = text.strip()
        if not text:
            raise RuntimeError("模型没有返回图片描述")
        return text

    async def _prepare_vision_images(
        self, image_urls: Sequence[str]
    ) -> tuple[str, ...]:
        prepared: list[str] = []
        downloaded_bytes = 0
        for url in image_urls[:_VISION_IMAGE_LIMIT]:
            if url.startswith("data:image/"):
                prepared.append(url)
                continue
            parts = urlsplit(url)
            if parts.scheme not in {"http", "https"} or not parts.netloc:
                raise ValueError("图片地址必须是 HTTP(S) URL 或 image Data URL")
            data_url, size = await self._download_image_as_data_url(url)
            downloaded_bytes += size
            if downloaded_bytes > _VISION_TOTAL_MAX_BYTES:
                raise ValueError("本次识图下载的图片总大小超过 16 MiB")
            prepared.append(data_url)
        return tuple(prepared)

    async def _download_image_as_data_url(self, url: str) -> tuple[str, int]:
        host = urlsplit(url).hostname or "unknown"
        content: bytes | None = None
        for attempt in range(2):
            try:
                async with self.vision_http.stream("GET", url) as response:
                    if response.status_code >= 500 and attempt == 0:
                        continue
                    response.raise_for_status()
                    declared_size = int(response.headers.get("content-length", "0") or 0)
                    if declared_size > _VISION_IMAGE_MAX_BYTES:
                        raise ValueError("单张群聊图片超过 8 MiB，已拒绝识图")
                    chunks: list[bytes] = []
                    received = 0
                    async for chunk in response.aiter_bytes():
                        received += len(chunk)
                        if received > _VISION_IMAGE_MAX_BYTES:
                            raise ValueError("单张群聊图片超过 8 MiB，已拒绝识图")
                        chunks.append(chunk)
                    content = b"".join(chunks)
                    break
            except httpx.TransportError:
                if attempt == 0:
                    logger.warning("群聊图片传输中断，重新下载一次 host=%s", host)
                    continue
                raise RuntimeError(f"下载群聊图片超时或网络异常 host={host}") from None
        if content is None:
            raise RuntimeError(f"下载群聊图片失败 host={host}")

        try:
            with Image.open(io.BytesIO(content)) as image:
                width, height = image.size
                image_format = (image.format or "").upper()
                image.verify()
        except (UnidentifiedImageError, OSError, ValueError):
            raise ValueError("群聊附件不是可识别的图片格式") from None
        if width * height > _VISION_IMAGE_MAX_PIXELS:
            raise ValueError("群聊图片像素总量过大，已拒绝识图")
        mime = _VISION_MIME_BY_FORMAT.get(image_format)
        if mime is None:
            raise ValueError(f"暂不支持识别 {image_format or '未知'} 图片格式")
        logger.info(
            "已在 VPS 预取群聊图片 host=%s bytes=%d format=%s size=%dx%d",
            host,
            len(content),
            image_format,
            width,
            height,
        )
        encoded = base64.b64encode(content).decode("ascii")
        return f"data:{mime};base64,{encoded}", len(content)

    async def web_search(
        self,
        *,
        query: str,
        instructions: str,
        messages: Sequence[StoredMessage],
    ) -> tuple[str, tuple[tuple[str, str], ...]]:
        """Use the Responses hosted web-search tool and return cited sources."""
        if self.api_mode != "responses":
            raise RuntimeError("网页检索只支持 Responses API 模式")
        input_messages = [
            {"role": item.role, "content": item.content} for item in messages[-8:]
        ]
        input_messages.append(
            {"role": "user", "content": f"需要联网核实的当前问题：{query}"}
        )
        response = await self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=input_messages,
            reasoning={"effort": self.reasoning_effort},
            tools=[{"type": "web_search"}],
            tool_choice="required",
            include=["web_search_call.action.sources"],
            max_output_tokens=min(self.max_output_tokens, 700),
            store=False,
        )
        self._log_usage(response, "web_search")
        text = response.output_text.strip()
        if not text:
            raise RuntimeError("网页检索没有返回正文")
        sources = self._extract_web_sources(response)
        if not self._has_completed_web_search_call(response):
            raise RuntimeError("网页检索响应缺少真实 web_search_call")
        if not sources:
            raise RuntimeError("网页检索响应没有返回可验证来源")
        return text, sources

    @classmethod
    def _log_usage(cls, response, purpose: str) -> None:
        usage = cls._value(response, "usage")
        if usage is None:
            logger.info("LLM usage purpose=%s unavailable=true", purpose)
            return
        input_tokens = cls._value(
            usage, "input_tokens", cls._value(usage, "prompt_tokens", 0)
        )
        output_tokens = cls._value(
            usage, "output_tokens", cls._value(usage, "completion_tokens", 0)
        )
        details = cls._value(usage, "input_tokens_details")
        cached_tokens = cls._value(details, "cached_tokens", 0) if details else 0
        logger.info(
            "LLM usage purpose=%s input_tokens=%s output_tokens=%s cached_tokens=%s",
            purpose,
            input_tokens or 0,
            output_tokens or 0,
            cached_tokens or 0,
        )

    @staticmethod
    def _value(item, name: str, default=None):
        if isinstance(item, dict):
            return item.get(name, default)
        return getattr(item, name, default)

    @classmethod
    def _extract_web_sources(cls, response) -> tuple[tuple[str, str], ...]:
        action_sources = []
        annotations = []
        for item in cls._value(response, "output", ()) or ():
            action = cls._value(item, "action")
            if action is not None:
                action_sources.extend(cls._value(action, "sources", ()) or ())
            for block in cls._value(item, "content", ()) or ():
                annotations.extend(cls._value(block, "annotations", ()) or ())

        sources: list[tuple[str, str]] = []
        seen: set[str] = set()
        for candidate in [*annotations, *action_sources]:
            url = cls._clean_source_url(
                str(cls._value(candidate, "url", "") or "").strip()
            )
            if not url.startswith(("https://", "http://")) or url in seen:
                continue
            title = str(cls._value(candidate, "title", "") or "").strip()
            if not title or title.startswith(("https://", "http://")):
                title = urlsplit(url).netloc or "来源"
            sources.append((title, url))
            seen.add(url)
            if len(sources) >= 5:
                break
        return tuple(sources)

    @classmethod
    def _has_completed_web_search_call(cls, response) -> bool:
        for item in cls._value(response, "output", ()) or ():
            if cls._value(item, "type") != "web_search_call":
                continue
            status = str(cls._value(item, "status", "") or "").lower()
            if status not in {"failed", "cancelled", "incomplete"}:
                return True
        return False

    @staticmethod
    def _clean_source_url(url: str) -> str:
        try:
            parts = urlsplit(url)
            query = urlencode(
                [
                    (key, value)
                    for key, value in parse_qsl(parts.query, keep_blank_values=True)
                    if not key.lower().startswith("utm_")
                ]
            )
            return urlunsplit((parts.scheme, parts.netloc, parts.path, query, ""))
        except ValueError:
            return url

    async def close(self) -> None:
        await self.client.close()
        if self._owns_vision_http:
            await self.vision_http.aclose()
