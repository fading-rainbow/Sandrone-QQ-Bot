from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import qqbot_agent_sdk.websocket as qq_websocket_module
from qqbot_agent_sdk import (
    MEDIA_TYPE_IMAGE,
    MSG_TYPE_QUOTE,
    EventParser,
    InboundEvent,
    MediaInfo,
    MediaUploader,
    MessageToCreate,
    QQApiClient,
    QQMessageType,
    QQWebSocket,
    WSCallbacks,
)
from qqbot_agent_sdk.dto import parse_message

from .character_refs import CharacterReferenceLibrary
from .image_gen import ImageContentPolicyError, ImageEditMismatch, ImageGenerator
from .image_sources import ImageSourceCache, ImageSourceUnavailable, is_edit_request
from .memory import ImageJobRecord
from .service import ChatService, IncomingMessage

logger = logging.getLogger(__name__)


class ImageIdentityMismatch(RuntimeError):
    pass


class ImageIdentityVerificationUnavailable(RuntimeError):
    pass

FULL_GROUP_EVENT = "GROUP_MESSAGE_CREATE"
REACTION_EVENT_TYPES = frozenset(
    {
        "MESSAGE_REACTION_ADD",
        "MESSAGE_REACTION_REMOVE",
        "GROUP_MESSAGE_REACTION_ADD",
        "GROUP_MESSAGE_REACTION_REMOVE",
    }
)
qq_websocket_module.MESSAGE_EVENT_TYPES = frozenset(
    {
        *qq_websocket_module.MESSAGE_EVENT_TYPES,
        FULL_GROUP_EVENT,
        *REACTION_EVENT_TYPES,
    }
)

_QQ_REACTION_NAMES = {
    "4": "得意",
    "5": "流泪",
    "14": "微笑",
    "21": "可爱",
    "66": "爱心",
    "76": "赞",
    "124": "OK",
    "144": "喝彩",
    "147": "棒棒糖",
    "201": "点赞",
    "202": "略略略",
    "264": "捂脸",
    "285": "666",
    "297": "拜谢",
    "303": "右哼哼",
    "305": "比心",
    "318": "击掌",
    "319": "抱抱",
    "324": "吃糖",
}
_POSITIVE_REACTIONS = frozenset(
    {
        "得意", "微笑", "可爱", "爱心", "赞", "OK", "喝彩", "棒棒糖",
        "点赞", "666", "拜谢", "比心", "击掌", "抱抱", "吃糖",
    }
)
_PLAYFUL_REACTIONS = frozenset({"略略略", "捂脸", "右哼哼"})
_SAD_REACTIONS = frozenset({"流泪", "😢", "😭", "🥲", "💔"})
_NEGATIVE_REACTIONS = frozenset({"👎", "😡", "🤬", "🙄", "😒", "💢"})
_IMAGE_PROGRESS_RE = re.compile(
    r"(?:图片|图画|图|画|成像).{0,8}"
    r"(?:进度|好了吗|画好|生成好|到哪|还要多久|完成了吗|怎么样了|咋样了|怎样了)"
    r"|(?:进度|好了吗|画好|生成好|还要多久|怎么样了|咋样了|怎样了).{0,8}"
    r"(?:图片|图画|图|画|成像)"
    r"|(?:在|开始)(?:画|绘制|生成|成像)(?:了|了吗|了吧|没有|没)[？?。！!]*$",
    re.IGNORECASE,
)
_ACTIVE_IMAGE_FOLLOWUP_RE = re.compile(
    r"^(?:图呢|图片呢|画完了吗|生成完了吗|怎么样了|咋样了|怎样了|还没好吗|"
    r"(?:把)?(?:图|图片|画)?发给我(?:看看)?(?:啊|呀|吧)?|核验一下|查一下状态)"
    r"[？?。！!]*$"
)
_IMAGE_CANCEL_RE = re.compile(
    r"^(?:取消|停止|别画了|不用画了|别生成了|不用生成了)(?:这张|画图|生成|任务)?"
    r"[？?。！!]*$"
)
_IMAGE_DELIVERY_FOLLOWUP_RE = re.compile(
    r"^(?:把)?(?:图|图片|画)?发给我(?:看看)?(?:啊|呀|吧)?[？?。！!]*$"
)
_QQ_FACE_TAG_RE = re.compile(
    r'<faceType=\d+,faceId="[^"]+",ext="(?P<ext>[^"]*)">'
)
_DEICTIC_IMAGE_QUESTION_RE = re.compile(
    r"^(?:这是谁|这(?:张)?图(?:里)?是谁|上图是谁|上面(?:这张)?图(?:里)?是谁|"
    r"这(?:张)?图(?:里)?是什么角色|上图是什么角色|你认识(?:这|上图)吗|"
    r"(?:这张|上面这张|上图|图中|图里).{0,24}(?:从左到右|依次是谁|怎么评价|"
    r"怎么样|美颜|有几个人|几个角色))[？?。！!]*$"
)
_ACTIVE_MESSAGE_PERMISSION_RECHECK_SECONDS = 6 * 60 * 60


@dataclass
class ImageJobState:
    started_at: float
    phase: str
    prompt: str
    job_id: int | None = None
    user_id: str = ""


def is_image_progress_question(content: str) -> bool:
    return bool(_IMAGE_PROGRESS_RE.search(content.strip()))


def is_image_delivery_followup(content: str) -> bool:
    return bool(_IMAGE_DELIVERY_FOLLOWUP_RE.match(content.strip()))


def normalize_qq_content(content: str) -> str:
    """Turn verbose QQ face payloads into compact, readable context."""

    def replace_face(match: re.Match[str]) -> str:
        encoded = match.group("ext")
        try:
            padded = encoded + "=" * (-len(encoded) % 4)
            payload = json.loads(base64.b64decode(padded).decode("utf-8"))
            label = str(payload.get("text") or "").strip()
            return f"[表情:{label}]" if label else "[表情]"
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
            return "[表情]"

    return _QQ_FACE_TAG_RE.sub(replace_face, content).strip()


def is_expired_reply_error(error: BaseException) -> bool:
    """Recognize QQ's permanent expired-reply error across localized messages."""
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = str(current).lower().replace(" ", "").replace("-", "_")
        has_message_id = any(
            marker in message for marker in ("msg_id", "msgid", "消息id")
        )
        if has_message_id and ("过期" in message or "expired" in message):
            return True
        current = current.__cause__ or current.__context__
    return False


def is_active_message_permission_error(error: BaseException) -> bool:
    """Recognize the explicit proactive-send denial, including SDK wrappers."""
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = str(current).lower().replace(" ", "")
        if "主动消息" in message and "无权限" in message:
            return True
        current = current.__cause__ or current.__context__
    return False


def image_progress_reply(
    job: ImageJobState | None,
    *,
    last_job: ImageJobRecord | None = None,
    now: float | None = None,
) -> str:
    if job is None:
        if last_job is not None and last_job.status == "sent":
            return "上一幅已经由 QQ 确认发送成功；当前没有正在生成或等待发送的图片。"
        if last_job is not None and last_job.status == "failed":
            return "上一幅在生成或发送阶段失败了；当前没有仍在运行的图片任务。"
        if last_job is not None and last_job.status == "interrupted":
            return "上一幅因服务重启而中断，没有伪装成完成；当前需要重新下达画图命令。"
        if last_job is not None and last_job.status == "cancelled":
            return "上一幅已经按要求取消；当前没有正在生成或等待发送的图片。"
        return "工坊当前没有正在生成或等待发送的图片；我不会拿一张并不存在的成图敷衍你。"
    current = time.time() if now is None else now
    elapsed = max(0, int(current - job.started_at))
    return (
        f"已经运行 {elapsed} 秒，现在处于“{job.phase}”阶段。"
        "上游不提供百分比——齿轮还在转，完成后我会直接把图送来。"
    )


class QQBotRunner:
    def __init__(
        self,
        *,
        app_id: str,
        app_secret: str,
        service: ChatService,
        image_generator: ImageGenerator,
        image_cooldown_seconds: int,
        owner_ids: frozenset[str],
        allowed_group_ids: frozenset[str],
        character_library: CharacterReferenceLibrary | None = None,
    ) -> None:
        self.http_client = httpx.AsyncClient(timeout=60.0)
        self.api = QQApiClient(app_id=app_id, client_secret=app_secret, log_tag="sandrone")
        self.api.setup(self.http_client)
        self.media_uploader = MediaUploader(self.api, self.http_client, log_tag="sandrone")
        self.service = service
        self.image_generator = image_generator
        self.character_library = character_library
        self.image_sources = ImageSourceCache(image_generator.output_dir / "sources")
        self.image_cooldown_seconds = image_cooldown_seconds
        self.owner_ids = owner_ids
        self.allowed_group_ids = allowed_group_ids
        self.ws: QQWebSocket | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        self._session_id: str | None = None
        self._last_seq: int | None = None
        self._bot_ids: set[str] = set()
        self._bot_names: set[str] = set()
        self._image_job: ImageJobState | None = None
        self._image_task: asyncio.Task[None] | None = None
        self._outbound_text_by_id: dict[str, str] = {}

    def _remember_outbound(self, result: object, content: str) -> None:
        if isinstance(result, dict):
            message_id = str(result.get("id") or result.get("message_id") or "")
        else:
            message_id = str(
                getattr(result, "id", "") or getattr(result, "message_id", "")
            )
        if not message_id:
            return
        outbound = getattr(self, "_outbound_text_by_id", None)
        if outbound is None:
            outbound = {}
            self._outbound_text_by_id = outbound
        outbound[message_id] = content[:500]
        while len(outbound) > 256:
            outbound.pop(next(iter(outbound)))

    async def _send_reply_text(self, event, content: str) -> dict:
        """Send one reply, falling back to a fresh message when msg_id expired."""
        try:
            result = await self.api.send_text(
                event.chat_scope,
                event.chat_id,
                content,
                reply_to=event.message_id,
                markdown=False,
                retries=1,
            )
        except Exception as exc:
            if not is_expired_reply_error(exc):
                raise
            logger.warning(
                "QQ 回复 msg_id 已过期，降级为普通消息 message_id=%s",
                event.message_id,
            )
            result = await self._send_fresh_text(event, content)
        self._remember_outbound(result, content)
        return result

    async def _send_fresh_text(self, event, content: str) -> dict:
        # The SDK retries every unrecognized 400. A permission denial cannot
        # improve on retry, and optional messages can wait for the next event.
        try:
            result = await self.api.send_text(
                event.chat_scope,
                event.chat_id,
                content,
                reply_to=None,
                markdown=False,
                retries=1,
            )
        except Exception as exc:
            if is_active_message_permission_error(exc):
                self._block_optional_messages(event)
            raise
        self._remember_outbound(result, content)
        return result

    @staticmethod
    def _optional_permission_key(event) -> str:
        return f"qq:active-message-denied:{event.chat_scope}:{event.chat_id}"

    def _optional_messages_available(self, event, *, passive: bool = False) -> bool:
        if passive and getattr(event, "message_id", ""):
            return True
        if getattr(event, "chat_scope", "") != "group":
            return True
        memory = getattr(getattr(self, "service", None), "memory", None)
        if memory is None:
            return True
        return not memory.rate_limit_remaining(
            self._optional_permission_key(event),
            _ACTIVE_MESSAGE_PERMISSION_RECHECK_SECONDS,
        )

    def _block_optional_messages(self, event) -> None:
        memory = getattr(getattr(self, "service", None), "memory", None)
        if memory is None or not getattr(event, "chat_id", ""):
            return
        claimed, _, _ = memory.claim_rate_limit(
            self._optional_permission_key(event),
            _ACTIVE_MESSAGE_PERMISSION_RECHECK_SECONDS,
        )
        if claimed:
            logger.warning(
                "QQ 主动消息权限不足，暂停该群主动问候/插话6小时；被动回复继续 group=%s",
                event.chat_id,
            )

    async def _send_media_with_reply_fallback(self, event, file_info: str) -> None:
        async def post(reply_to: str | None) -> None:
            message = MessageToCreate(
                msg_type=QQMessageType.RICH_MEDIA,
                msg_seq=self.api.next_msg_seq(),
                msg_id=reply_to or "",
                media=MediaInfo(file_info=file_info),
            )
            if event.chat_scope == "c2c":
                await self.api.post_c2c_message(event.chat_id, message)
            else:
                await self.api.post_group_message(event.chat_id, message)

        try:
            await post(event.message_id)
        except Exception as exc:
            if not is_expired_reply_error(exc):
                raise
            logger.warning(
                "QQ 图片回复 msg_id 已过期，降级为普通消息 message_id=%s",
                event.message_id,
            )
            await post(None)

    def _get_session(self) -> tuple[str | None, int | None]:
        return self._session_id, self._last_seq

    def _set_session(self, session_id: str | None, last_seq: int | None) -> None:
        self._session_id = session_id
        self._last_seq = last_seq

    async def _on_message(self, event_type: str, raw: dict) -> None:
        if event_type in REACTION_EVENT_TYPES:
            await self._on_reaction(event_type, raw)
            return
        if event_type == FULL_GROUP_EVENT:
            event = self._parse_full_group(raw)
            should_reply = self._is_bot_mentioned(raw)
        else:
            event = EventParser.parse(event_type, raw)
            should_reply = event_type != FULL_GROUP_EVENT
        if event is None:
            return
        if event.user_id and event.user_id in self._bot_ids:
            logger.debug("忽略机器人自身消息 message_id=%s", event.message_id)
            return
        if event_type == FULL_GROUP_EVENT and should_reply:
            event.content = self._strip_bot_mentions(event.content)
        if event.chat_scope != "group" or event.chat_id not in self.allowed_group_ids:
            logger.info(
                "收到非白名单群/消息: scope=%s chat_id=%s (若需接入该群，请将 chat_id 追加至 ALLOWED_GROUP_IDS)",
                event.chat_scope,
                event.chat_id,
            )
            return
        task = asyncio.create_task(self._handle_event(event, should_reply=should_reply))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    @staticmethod
    def _reaction_label(emoji: object) -> str:
        if not isinstance(emoji, dict):
            return "表情"
        emoji_id = str(emoji.get("id") or "")
        emoji_type = int(emoji.get("type") or 0)
        if emoji_type == 2 and emoji_id.isdigit():
            try:
                return chr(int(emoji_id))
            except (ValueError, OverflowError):
                pass
        return _QQ_REACTION_NAMES.get(emoji_id, f"表情{emoji_id}" if emoji_id else "表情")

    @staticmethod
    def _reaction_sentiment(label: str) -> str:
        if label in _POSITIVE_REACTIONS or label in {"👍", "❤️", "❤", "💕", "🥰", "😍"}:
            return "positive"
        if label in _PLAYFUL_REACTIONS or label in {"😏", "🤭", "😜", "😝"}:
            return "playful"
        if label in _SAD_REACTIONS:
            return "sad"
        if label in _NEGATIVE_REACTIONS:
            return "negative"
        return "neutral"

    async def _on_reaction(self, event_type: str, raw: dict) -> None:
        target = raw.get("target") or {}
        emoji = raw.get("emoji") or {}
        target_id = str(target.get("id") or raw.get("message_id") or "")
        group_id = str(raw.get("group_openid") or raw.get("group_id") or "")
        if (
            not group_id
            and target_id in self._outbound_text_by_id
            and len(self.allowed_group_ids) == 1
        ):
            group_id = next(iter(self.allowed_group_ids))
        if not group_id or group_id not in self.allowed_group_ids:
            logger.info(
                "收到无法归属目标群的贴表情事件 event=%s target=%s fields=%s",
                event_type,
                target_id,
                sorted(raw),
            )
            return
        label = self._reaction_label(emoji)
        user_id = str(
            raw.get("user_id")
            or raw.get("user_openid")
            or raw.get("member_openid")
            or "unknown-reaction-user"
        )
        conversation_key = f"group:{group_id}"
        target_text = self._outbound_text_by_id.get(target_id, "桑多涅此前的一条消息")
        profile = self.service.memory.member_profile(conversation_key, user_id)
        user_name = profile.user_name if profile and profile.user_name else user_id
        is_add = not event_type.endswith("REMOVE")
        changed = self.service.memory.set_message_reaction(
            conversation_key=conversation_key,
            target_id=target_id,
            user_id=user_id,
            user_name=user_name,
            label=label,
            target_text=target_text,
            sentiment=self._reaction_sentiment(label),
            active=is_add,
        )
        if not is_add:
            logger.info(
                "群聊贴表情已移除 group=%s target=%s emoji=%s changed=%s",
                group_id,
                target_id,
                label,
                changed,
            )
            return
        if not changed:
            logger.debug(
                "忽略重复贴表情事件 group=%s target=%s user=%s emoji=%s",
                group_id,
                target_id,
                user_id,
                label,
            )
            return
        content = f"[{user_name}] [贴表情:{label}] 对桑多涅的消息“{target_text[:120]}”"
        self.service.memory.append(conversation_key, user_id, "user", content)
        logger.info(
            "已记录群聊贴表情 group=%s user=%s target=%s emoji=%s",
            group_id,
            user_id,
            target_id,
            label,
        )

    def _parse_full_group(self, raw: dict) -> InboundEvent | None:
        parsed = parse_message(raw)
        author = raw.get("author") or {}
        member = raw.get("member") or {}
        group_id = str(raw.get("group_openid") or raw.get("group_id") or "")
        user_id = str(
            author.get("member_openid")
            or author.get("user_openid")
            or author.get("id")
            or ""
        )
        if not group_id or not user_id:
            logger.warning("全量群消息缺少群或用户标识")
            return None
        content = str(raw.get("content") or "").strip()
        return InboundEvent(
            event_type=FULL_GROUP_EVENT,
            chat_id=group_id,
            user_id=user_id,
            chat_scope="group",
            content=content,
            message_id=str(raw.get("id") or ""),
            timestamp=str(raw.get("timestamp") or ""),
            message_type=int(raw.get("message_type") or 0),
            user_name=(
                str(member.get("nick") or author.get("username") or "") or None
            ),
            attachments=parsed.attachments,
            msg_elements=parsed.msg_elements,
            raw=raw,
        )

    def _capture_bot_identity(self, ready) -> None:
        user = getattr(ready, "user", None)
        if user is None:
            return
        for field in ("id", "user_openid", "member_openid", "union_openid"):
            value = str(getattr(user, field, "") or "")
            if value:
                self._bot_ids.add(value)
        username = str(getattr(user, "username", "") or "").strip()
        if username:
            self._bot_names.add(username)
        logger.info("已记录机器人身份标识数量=%d", len(self._bot_ids))

    def _strip_bot_mentions(self, content: str) -> str:
        cleaned = content.strip()
        for bot_id in self._bot_ids:
            cleaned = re.sub(rf"<@!?{re.escape(bot_id)}>\s*", "", cleaned)
        for name in self._bot_names:
            cleaned = re.sub(rf"^@{re.escape(name)}\s*", "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    def _is_bot_mentioned(self, raw: dict) -> bool:
        mentions = raw.get("mentions") or []
        for mention in mentions:
            if not isinstance(mention, dict):
                continue
            values = {
                str(mention.get(key) or "")
                for key in ("id", "openid", "user_openid", "member_openid")
            }
            if mention.get("is_you") is True or mention.get("isYou") is True:
                self._bot_ids.update(value for value in values if value)
                return True
            if self._bot_ids.intersection(values):
                return True
        content = str(raw.get("content") or "")
        if any(
            f"<@{bot_id}>" in content or f"<@!{bot_id}>" in content
            for bot_id in self._bot_ids
        ):
            return True
        return any(
            re.match(rf"^@{re.escape(name)}(?:\s|$)", content.strip(), re.IGNORECASE)
            for name in self._bot_names
        )

    def _quoted_context(self, event) -> tuple[str, tuple[str, ...]]:
        if event.message_type != MSG_TYPE_QUOTE or not event.msg_elements:
            return "", ()
        element = event.msg_elements[0]
        content = normalize_qq_content(str(element.content or ""))
        images = tuple(
            self._attachment_url(attachment)
            for attachment in element.attachments
            if self._is_image_attachment(attachment)
        )
        return content, images

    async def _handle_event(self, event, *, should_reply: bool) -> None:
        incoming = None
        image_arrived_at = time.time()
        try:
            quoted_content, quoted_image_urls = self._quoted_context(event)
            quoted_user_name = ""
            if quoted_content or quoted_image_urls:
                quoted_user_name = self.service.memory.resolve_quoted_speaker(
                    f"{event.chat_scope}:{event.chat_id}", quoted_content
                ) or ""
                logger.info(
                    "检测到 QQ 引用消息 message_id=%s quoted_text=%s "
                    "quoted_images=%d quoted_speaker=%s",
                    event.message_id,
                    bool(quoted_content),
                    len(quoted_image_urls),
                    quoted_user_name or "unresolved",
                )
            conversation_key = f"{event.chat_scope}:{event.chat_id}"
            profile = self.service.memory.member_profile(
                conversation_key, event.user_id
            )
            canonical_name = (
                event.user_name
                or (profile.user_name if profile and profile.user_name else None)
            )
            incoming = IncomingMessage(
                event_id=event.message_id,
                scope=event.chat_scope,
                chat_id=event.chat_id,
                user_id=event.user_id,
                content=normalize_qq_content(event.content),
                user_name=canonical_name,
                image_urls=tuple(
                    self._attachment_url(attachment)
                    for attachment in event.attachments
                    if self._is_image_attachment(attachment)
                ),
                is_owner=self._is_owner(event),
                quoted_content=quoted_content,
                quoted_image_urls=quoted_image_urls,
                quoted_user_name=quoted_user_name,
            )
            if not should_reply:
                logger.info("消息路由 route=observe message_id=%s", event.message_id)
                observed = await self.service.observe(incoming)
                if not observed:
                    return
                proactive_due = self.service.note_group_message(incoming)
                if not self._optional_messages_available(event):
                    logger.debug(
                        "跳过无主动消息权限的群问候/插话 group=%s", event.chat_id
                    )
                    return
                greeting = await self.service.daily_greeting(incoming)
                if greeting:
                    logger.info(
                        "消息路由 route=daily_greeting message_id=%s", event.message_id
                    )
                    sent = await self._send_optional_text(event, greeting)
                    self.service.complete_daily_greeting(incoming, greeting, sent=sent)
                    if sent:
                        self.service.memory.reset_proactive_activity(
                            incoming.conversation_key
                        )
                elif proactive_due:
                    proactive = await self.service.proactive_reply(incoming)
                    if proactive:
                        logger.info(
                            "消息路由 route=proactive_chat message_id=%s",
                            event.message_id,
                        )
                        if await self._send_optional_text(event, proactive):
                            self.service.remember_assistant(incoming, proactive)
                logger.debug("已记录群聊消息，不主动回复 message_id=%s", event.message_id)
                return
            self.service.note_group_message(incoming)
            greeting = await self.service.daily_greeting(incoming)
            if greeting:
                logger.info(
                    "消息路由 route=daily_greeting message_id=%s", event.message_id
                )
                sent = await self._send_optional_text(event, greeting, passive=True)
                self.service.complete_daily_greeting(incoming, greeting, sent=sent)
                if sent:
                    self.service.memory.reset_proactive_activity(
                        incoming.conversation_key
                    )
            if event.chat_scope == "c2c":
                await self.api.send_typing(event.chat_id, event.message_id, input_seconds=60)
            if self._image_job is not None and _IMAGE_CANCEL_RE.fullmatch(
                incoming.content.strip()
            ):
                logger.info("消息路由 route=image_cancel message_id=%s", event.message_id)
                await self._handle_image_cancel(event, incoming)
                return
            if (
                not incoming.image_urls
                and not incoming.quoted_image_urls
                and _DEICTIC_IMAGE_QUESTION_RE.fullmatch(incoming.content.strip())
            ):
                await self.service.wait_for_recent_image_context(
                    incoming.conversation_key
                )
            if (
                is_image_progress_question(incoming.content)
                or is_image_delivery_followup(incoming.content)
                or (
                self._image_job is not None
                and _ACTIVE_IMAGE_FOLLOWUP_RE.match(incoming.content.strip())
                )
            ):
                logger.info("消息路由 route=image_status message_id=%s", event.message_id)
                await self._handle_image_progress(event, incoming)
                return
            if is_edit_request(incoming.content, has_image=bool(incoming.image_urls or incoming.quoted_image_urls)):
                logger.info("消息路由 route=image_edit message_id=%s", event.message_id)
                await self._handle_image_request(event, incoming, incoming.content)
                return
            try:
                web_search_query = await self.service.resolve_web_search_query(incoming)
            except Exception:
                logger.exception("网页检索意图分类失败 message_id=%s", event.message_id)
                web_search_query = self.service.extract_web_search_query(incoming.content)
            if web_search_query is not None:
                logger.info("消息路由 route=web_search message_id=%s", event.message_id)
                answer = await self.service.handle(
                    incoming, search_query_override=web_search_query
                )
                if answer is not None:
                    await self._send_reply_text(event, answer)
                return
            try:
                image_prompt = await self.service.resolve_image_prompt(incoming)
            except Exception:
                logger.exception("画图意图分类失败 message_id=%s", event.message_id)
                image_prompt = self.service.extract_image_prompt(incoming.content)
            if image_prompt is not None and event.chat_scope in {"c2c", "group"}:
                logger.info("消息路由 route=image_generate message_id=%s", event.message_id)
                await self._handle_image_request(event, incoming, image_prompt)
                return
            logger.info("消息路由 route=chat message_id=%s", event.message_id)
            answer = await self.service.handle(incoming)
            if answer is not None:
                await self._send_reply_text(event, answer)
        except Exception:
            logger.exception("处理 QQ 消息失败 message_id=%s", event.message_id)
            if not should_reply:
                return
            try:
                await self._send_reply_text(
                    event, "啧，传动结构出了点故障。稍后再试，别催。"
                )
            except Exception:
                logger.exception("发送错误提示失败 message_id=%s", event.message_id)
        finally:
            # Observe/handle must reserve the message position before any extra I/O.
            # Caching first would reorder an image behind the user's follow-up.
            if incoming and incoming.image_urls and not is_edit_request(incoming.content) and getattr(self, "image_sources", None) is not None:
                try:
                    await self.image_sources.fetch(
                        incoming, incoming.image_urls, self.service.llm._download_image_as_data_url,
                        created_at=image_arrived_at,
                    )
                except Exception:
                    logger.warning("原图缓存未成功 message_id=%s", event.message_id)

    async def _send_optional_text(self, event, text: str, *, passive: bool = False) -> bool:
        """Optional greetings/interjections must never abort a requested reply."""
        if not self._optional_messages_available(event, passive=passive):
            return False
        try:
            if passive and getattr(event, "message_id", ""):
                await self._send_reply_text(event, text)
            else:
                await self._send_fresh_text(event, text)
            return True
        except Exception as exc:
            if is_active_message_permission_error(exc):
                self._block_optional_messages(event)
            logger.warning("主动问候或插话发送失败，继续处理当前消息 message_id=%s", event.message_id)
            return False

    def _is_owner(self, event) -> bool:
        values = {str(event.user_id or "")}
        raw = event.raw if isinstance(event.raw, dict) else {}
        author = raw.get("author") or {}
        member = raw.get("member") or {}
        for source in (raw, author, member):
            if not isinstance(source, dict):
                continue
            values.update(
                str(source.get(key) or "")
                for key in (
                    "id",
                    "qq",
                    "uin",
                    "user_id",
                    "user_openid",
                    "member_openid",
                )
            )
        return bool(self.owner_ids.intersection(values))

    async def _handle_image_request(
        self, event, incoming: IncomingMessage, prompt: str
    ) -> None:
        if not await self.service.begin_direct_request(incoming):
            return
        if self._image_job is not None:
            refusal = "上一幅仍在工坊里运行。我不会并行启动第二幅，以免齿轮串线；等它结束再说。"
            self.service.remember_assistant(incoming, refusal)
            await self._send_reply_text(event, refusal)
            return
        allowed, remaining, claimed_at = self.service.memory.claim_rate_limit(
            "image:global", self.image_cooldown_seconds
        )
        if not allowed:
            if self._image_job is not None:
                refusal = (
                    f"上一幅还在工坊里成形，再给它一点时间。当前冷却还剩 {remaining} 秒，"
                    "等齿轮停稳后我再接你的新图。"
                )
            else:
                refusal = (
                    f"工坊刚处理完上一幅，机械还需要冷却 {remaining} 秒。"
                    "稍后再交给我吧，我会认真画。"
                )
            self.service.remember_assistant(incoming, refusal)
            await self._send_reply_text(event, refusal)
            return

        started_at = time.time()
        try:
            job_id = self.service.memory.create_image_job(
                event_id=incoming.event_id,
                conversation_key=incoming.conversation_key,
                user_id=incoming.user_id,
                prompt=prompt,
                phase="解析构图",
                started_at=started_at,
            )
        except Exception:
            if claimed_at is not None:
                self.service.memory.release_rate_limit("image:global", claimed_at)
            logger.exception("创建持久化生图任务失败 message_id=%s", event.message_id)
            failure = "工坊状态簿没有成功落锁，这次没有开工。稍后再试。"
            self.service.remember_assistant(incoming, failure)
            await self._send_reply_text(event, failure)
            return
        job = ImageJobState(started_at, "解析构图", prompt, job_id, incoming.user_id)
        self._image_job = job
        current_task = asyncio.current_task()
        self._image_task = current_task
        logger.info("生图任务开始 job_id=%s message_id=%s", job_id, event.message_id)
        path = None
        delivered = False
        source_workspace = None
        editing = is_edit_request(
            getattr(incoming, "content", prompt),
            has_image=bool(getattr(incoming, "image_urls", ()) or getattr(incoming, "quoted_image_urls", ())),
        )
        source_paths = ()
        character_references = None
        try:
            if editing:
                selected = await self.image_sources.resolve(
                    incoming, self.service.llm._download_image_as_data_url,
                )
                # Pin source files for the entire job, unaffected by cache eviction.
                source_workspace = tempfile.TemporaryDirectory(prefix="sandrone-edit-")
                source_paths = tuple(Path(source_workspace.name) / f"source-{i}.png" for i in range(len(selected)))
                for selected_path, pinned in zip(selected, source_paths):
                    shutil.copyfile(selected_path, pinned)
                resolved_prompt = prompt
                logger.info("原图编辑已绑定 job_id=%s source_count=%d", job_id, len(source_paths))
            else:
                library = getattr(self, "character_library", None)
                if library is not None:
                    job.phase = "核对角色参考"
                    self._update_image_job_safely(job, phase=job.phase)
                    names = await self.service.image_character_names(incoming, prompt)
                    character_references = await library.resolve(names)
                    resolved_prompt = await self.service.prepare_image_prompt(
                        incoming, prompt, character_references=character_references,
                    )
                else:
                    resolved_prompt = await self.service.prepare_image_prompt(incoming, prompt)
            job.phase = "模型渲染"
            self._update_image_job_safely(job, phase=job.phase)
            progress = "原图收到了。只动你指定的地方，别催坏我的精度。" if editing else "……知道了。别催，我的人偶正在构图。"
            await self._send_reply_text(event, progress)
            generated = None
            actual_description = ""
            safety_rewritten = False
            for attempt in range(2):
                if editing:
                    edit_prompt = resolved_prompt
                    if attempt:
                        edit_prompt += "\n上一版未通过校验，请仍从原图重新编辑，纠正这些问题：" + actual_description[:500]
                    generated = await self.image_generator.edit(edit_prompt, source_paths)
                else:
                    grounding = {} if character_references is None else {
                        "character_references": character_references,
                        "retry_feedback": actual_description if attempt else "",
                    }
                    generated = await self.image_generator.generate(
                        resolved_prompt, identity_retry=attempt > 0, **grounding,
                    )
                safety_rewritten = safety_rewritten or generated.safety_rewritten
                path = generated.path
                job.phase = "检查成图"
                self._update_image_job_safely(job, phase=job.phase)
                try:
                    if editing:
                        inspection = await self.service.inspect_edited_image(path, source_paths, resolved_prompt)
                    else:
                        review_kwargs = {} if not character_references else {
                            "character_references": character_references,
                        }
                        inspection = await self.service.inspect_generated_image(
                            path, resolved_prompt, generated.reference_paths, **review_kwargs,
                        )
                    actual_description = inspection.description
                except Exception as exc:
                    logger.exception("检查生成图片失败 message_id=%s", event.message_id)
                    if editing:
                        raise ImageEditMismatch("原图编辑校验不可用") from exc
                    if generated.identity_sensitive:
                        raise ImageIdentityVerificationUnavailable(
                            "Sandrone identity verification failed"
                        ) from exc
                    actual_description = "成图视觉复核失败；只保留构图解析，不能确认角色还原度。"
                    break

                if (not editing and not generated.identity_sensitive) or inspection.accepted:
                    break
                logger.warning(
                    "角色身份复核未通过 job_id=%s attempt=%d detail=%s",
                    job_id,
                    attempt + 1,
                    actual_description[:300],
                )
                if attempt == 1:
                    if editing:
                        raise ImageEditMismatch(actual_description)
                    raise ImageIdentityMismatch(actual_description)
                path.unlink(missing_ok=True)
                path = None
                job.phase = "原图编辑校准" if editing else "身份校准重绘"
                self._update_image_job_safely(job, phase=job.phase)

            if generated is None or path is None:
                raise RuntimeError("图像任务没有产生可发送文件")
            job.phase = "上传 QQ"
            self._update_image_job_safely(job, phase=job.phase)
            file_info = await self.media_uploader.upload(
                chat_type=event.chat_scope,
                chat_id=event.chat_id,
                source=str(path),
                file_type=MEDIA_TYPE_IMAGE,
            )
            await self._send_media_with_reply_fallback(event, file_info)
            delivered = True
            if getattr(self, "image_sources", None) is not None:
                try:
                    self.image_sources.store(
                        incoming.conversation_key, incoming.user_id, "result:" + incoming.event_id,
                        "result:" + incoming.event_id, path.read_bytes(),
                    )
                except Exception:
                    logger.warning("成图已发送，但后续编辑缓存失败 job_id=%s", job_id)
            final_phase = "安全改写后已发送" if safety_rewritten else "已发送"
            self._update_image_job_safely(job, phase=final_phase, status="sent")
            logger.info(
                "生图任务完成 job_id=%s elapsed_seconds=%d",
                job_id,
                int(time.time() - started_at),
            )
            try:
                self.service.remember_image_result(
                    incoming, prompt, resolved_prompt, actual_description
                )
            except Exception:
                logger.exception(
                    "图片已发送但工坊记忆写入失败 message_id=%s", event.message_id
                )
            if safety_rewritten:
                notice = (
                    "原构图被上游拦下了。我已把它校准成衣着完整、完全无暧昧的"
                    "卧室枕头嬉戏版本并成功送达。"
                )
                self.service.remember_assistant(incoming, notice)
                await self._send_reply_text(event, notice)
        except asyncio.CancelledError:
            if claimed_at is not None and not delivered:
                self.service.memory.release_rate_limit("image:global", claimed_at)
            if delivered:
                self._update_image_job_safely(job, phase="已发送", status="sent")
            else:
                self._update_image_job_safely(job, phase="已取消", status="cancelled")
            logger.info("生图任务已取消 job_id=%s delivered=%s", job_id, delivered)
            return
        except Exception as exc:
            if claimed_at is not None and not delivered:
                self.service.memory.release_rate_limit("image:global", claimed_at)
            logger.exception("生成或发送图片失败 message_id=%s", event.message_id)
            if delivered:
                self._update_image_job_safely(job, phase="已发送", status="sent")
                return
            policy_rejected = isinstance(exc, ImageContentPolicyError)
            self._update_image_job_safely(
                job,
                phase="画面描述被上游拒绝" if policy_rejected else "生成或发送失败",
                status="failed",
                error=(
                    "image content policy rejected safe alternative"
                    if policy_rejected
                    else "image generation or QQ delivery failed"
                ),
            )
            logger.warning(
                "生图任务失败 job_id=%s elapsed_seconds=%d",
                job_id,
                int(time.time() - started_at),
            )
            if isinstance(exc, ImageSourceUnavailable):
                failure = str(exc)
            elif isinstance(exc, ImageEditMismatch):
                failure = "这次修改没通过原图核对，我没有把不合格的成图发出来。稍后可以重试，或把要改的位置说得更具体些。"
            elif isinstance(exc, ImageIdentityMismatch):
                failure = (
                    "这次仍有角色偏离了对应参考档案。我不会把仿冒品递给你——"
                    "工坊没有发送，稍后再让我校准。"
                )
            elif isinstance(exc, ImageIdentityVerificationUnavailable):
                failure = (
                    "身份校验齿轮没有给出可靠回执。我不会在无法确认时冒充成图——"
                    "这次没有发送，稍后再试。"
                )
            elif policy_rejected:
                failure = "原图编辑被上游规则拦下了。这次没有改图，也没有换成别的画面。" if editing else (
                    "这次画面描述被上游规则拦下了，连无暧昧的安全改写也没有通过。"
                    "换成衣着完整、保持距离的日常互动，我再替你开工。"
                )
            else:
                failure = "这次生成或传送链路失败了。工坊已经释放任务锁，稍后可以重试。"
            self.service.remember_assistant(incoming, failure)
            await self._send_reply_text(event, failure)
        finally:
            if self._image_job is job:
                self._image_job = None
            if self._image_task is current_task:
                self._image_task = None
            if path is not None:
                path.unlink(missing_ok=True)
            if source_workspace is not None:
                source_workspace.cleanup()

    async def _handle_image_cancel(self, event, incoming: IncomingMessage) -> None:
        if not await self.service.begin_direct_request(incoming):
            return
        job = self._image_job
        if job is None:
            answer = "工坊当前没有正在生成的图片，不必取消。"
        elif not incoming.is_owner and incoming.user_id != job.user_id:
            answer = "这幅不是你下达的任务。除非最高指挥开口，否则我不会替别人停掉工坊。"
        else:
            job_id = job.job_id
            task = self._image_task
            if task is not None and task is not asyncio.current_task() and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            latest = self.service.memory.latest_image_job(incoming.conversation_key)
            if latest is not None and latest.id == job_id and latest.status == "sent":
                answer = "取消得太晚，图片已经由 QQ 确认发送完成。"
            else:
                answer = "已经停下了。这幅不会继续生成，也不会发送。"
        self.service.remember_assistant(incoming, answer)
        await self._send_reply_text(event, answer)

    async def _handle_image_progress(self, event, incoming: IncomingMessage) -> None:
        if not await self.service.begin_direct_request(incoming):
            return
        last_job = self.service.memory.latest_image_job(incoming.conversation_key)
        answer = image_progress_reply(self._image_job, last_job=last_job)
        self.service.remember_assistant(incoming, answer)
        await self._send_reply_text(event, answer)

    def _update_image_job_safely(
        self,
        job: ImageJobState,
        *,
        phase: str,
        status: str = "running",
        error: str = "",
    ) -> None:
        if job.job_id is None:
            return
        try:
            self.service.memory.update_image_job(
                job.job_id, phase=phase, status=status, error=error
            )
        except Exception:
            logger.exception("更新持久化生图状态失败 job_id=%s", job.job_id)

    @staticmethod
    def _attachment_url(attachment) -> str:
        url = str(getattr(attachment, "resolved_url", "") or getattr(attachment, "url", ""))
        return f"https:{url}" if url.startswith("//") else url

    @staticmethod
    def _is_image_attachment(attachment) -> bool:
        url = QQBotRunner._attachment_url(attachment).lower()
        content_type = str(getattr(attachment, "content_type", "")).lower()
        filename = str(getattr(attachment, "filename", "")).lower()
        return bool(url) and (
            content_type.startswith("image/")
            or filename.endswith((".jpg", ".jpeg", ".png", ".gif", ".webp"))
        )

    async def run(self) -> None:
        self.ws = QQWebSocket(
            callbacks=WSCallbacks(
                on_message_event=self._on_message,
                on_connected=lambda: logger.info("QQ WebSocket 已连接"),
                on_disconnected=lambda: logger.warning("QQ WebSocket 已断开，等待自动重连"),
                on_fatal_error=lambda code, message: logger.error(
                    "QQ WebSocket 致命错误 code=%s message=%s", code, message
                ),
                get_token=self.api.ensure_token_sync,
                get_session=self._get_session,
                set_session=self._set_session,
                set_heartbeat_interval=lambda interval: logger.debug(
                    "QQ 心跳间隔 %.1fs", interval
                ),
                clear_token=self.api.clear_token,
                fail_pending=lambda reason: logger.warning("QQ 挂起请求失败: %s", reason),
                get_gateway_url=self.api.get_gateway_url_sync,
                on_ready=self._capture_bot_identity,
            )
        )
        await self.api.ensure_token()
        gateway_url = await self.api.get_gateway_url()
        self.ws.start(gateway_url, asyncio.get_running_loop())
        logger.info("sandrone 已连接 QQ Gateway")
        try:
            await asyncio.Event().wait()
        finally:
            self.ws.stop()
            if self._tasks:
                for task in tuple(self._tasks):
                    task.cancel()
                await asyncio.gather(*self._tasks, return_exceptions=True)
            await self.http_client.aclose()
            self.image_sources.close()
