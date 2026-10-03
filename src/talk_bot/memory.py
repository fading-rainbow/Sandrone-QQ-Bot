from __future__ import annotations

import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from math import ceil
from pathlib import Path

_STORED_GROUP_SPEAKER_RE = re.compile(
    r"^(?:\[最高指挥\])?\["
    r"(?P<name>(?!(?:图片描述|图片处理中|引用消息|当前消息|网页检索|"
    r"桑多涅的工坊记录)\])[^\]\r\n]{1,100})\]\s*"
)
_QQ_MENTION_RE = re.compile(r"<@!?[^>]+>")
_NESTED_MESSAGE_CONTENT_RE = re.compile(
    r"\[消息内容]\s*(?P<content>.+?)(?=\s*\[消息类型])", re.DOTALL
)


def _normalize_quote_text(content: str) -> str:
    # QQ's nested quote summary may omit a mention that was present at either
    # edge of the original message, so mentions are identity context but not
    # part of the text fingerprint.
    without_mention = _QQ_MENTION_RE.sub("", content).strip()
    return " ".join(without_mention.split())


def _quote_text_candidates(content: str) -> tuple[str, ...]:
    """Return quote candidates in ownership order, outer message before ancestors."""

    nested = [
        match.group("content").strip()
        for match in _NESTED_MESSAGE_CONTENT_RE.finditer(content)
    ]
    raw_candidates: list[str] = []
    if nested:
        # QQ serializes a nested quote as message 1 followed by its associated
        # ancestors.  Message 1 is the message the user actually clicked.
        raw_candidates.extend(nested)
    elif "[当前消息]" in content:
        raw_candidates.append(content.rsplit("[当前消息]", 1)[1].strip())
    else:
        raw_candidates.append(content.strip())

    candidates: list[str] = []
    for raw in raw_candidates:
        value = _normalize_quote_text(raw)
        if value and value not in candidates:
            candidates.append(value)
    return tuple(candidates)


def _quote_text_variants(content: str) -> set[str]:
    """Return exact-comparison variants contained in one stored message."""

    raw_variants = [content.strip()]
    if "[当前消息]" in content:
        raw_variants.append(content.rsplit("[当前消息]", 1)[1].strip())
    raw_variants.extend(
        match.group("content").strip()
        for match in _NESTED_MESSAGE_CONTENT_RE.finditer(content)
    )
    normalized: set[str] = set()
    for raw in raw_variants:
        value = _normalize_quote_text(raw)
        if value:
            normalized.add(value)
    return normalized


@dataclass(frozen=True)
class StoredMessage:
    role: str
    content: str
    user_id: str = ""


@dataclass(frozen=True)
class StoredMessageRecord:
    id: int
    role: str
    content: str


@dataclass(frozen=True)
class SummaryState:
    content: str
    last_message_id: int


@dataclass(frozen=True)
class MemberProfile:
    user_id: str
    user_name: str
    long_term_content: str
    short_term_content: str
    last_message_id: int

    @property
    def content(self) -> str:
        parts = []
        if self.long_term_content:
            parts.append("长期印象：" + self.long_term_content)
        if self.short_term_content:
            parts.append("短期印象：" + self.short_term_content)
        return "\n".join(parts)


@dataclass(frozen=True)
class ConversationMember:
    user_id: str
    user_name: str
    message_count: int
    profile: str


@dataclass(frozen=True)
class ImageJobRecord:
    id: int
    event_id: str
    conversation_key: str
    user_id: str
    prompt: str
    status: str
    phase: str
    started_at: float
    updated_at: float
    error: str


@dataclass(frozen=True)
class WebSearchCacheEntry:
    answer: str
    created_at: float
    expires_at: float


@dataclass(frozen=True)
class ReactionFeedback:
    total_count: int
    positive_count: int
    playful_count: int
    sad_count: int
    negative_count: int
    last_label: str
    last_target_text: str
    last_sentiment: str
    last_at: float


class MemoryStore:
    """Small, thread-safe SQLite store for persistent chat history and facts."""

    def __init__(self, path: Path, *, max_messages_per_conversation: int = 4000) -> None:
        self.path = path
        self.max_messages = max_messages_per_conversation
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._create_schema()
            self._prune_housekeeping()

    def _create_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_key TEXT NOT NULL,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_messages_conversation
                ON messages(conversation_key, id);

            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_key TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_key, content)
            );

            CREATE TABLE IF NOT EXISTS processed_events (
                event_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS conversation_summaries (
                conversation_key TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                last_message_id INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS rate_limits (
                key TEXT PRIMARY KEY,
                last_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS member_profiles (
                conversation_key TEXT NOT NULL,
                user_id TEXT NOT NULL,
                user_name TEXT NOT NULL DEFAULT '',
                content TEXT NOT NULL,
                long_term_content TEXT NOT NULL DEFAULT '',
                short_term_content TEXT NOT NULL DEFAULT '',
                short_term_updated_at REAL NOT NULL DEFAULT 0,
                last_message_id INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(conversation_key, user_id)
            );

            CREATE TABLE IF NOT EXISTS member_addresses (
                conversation_key TEXT NOT NULL,
                user_id TEXT NOT NULL,
                address TEXT NOT NULL,
                set_by_user_id TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(conversation_key, user_id)
            );

            CREATE TABLE IF NOT EXISTS image_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE,
                conversation_key TEXT NOT NULL,
                user_id TEXT NOT NULL,
                prompt TEXT NOT NULL,
                status TEXT NOT NULL CHECK (
                    status IN ('running', 'sent', 'failed', 'interrupted')
                ),
                phase TEXT NOT NULL,
                started_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                error TEXT NOT NULL DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_image_jobs_conversation
                ON image_jobs(conversation_key, id DESC);

            CREATE TABLE IF NOT EXISTS web_search_cache (
                conversation_key TEXT NOT NULL,
                query_key TEXT NOT NULL,
                query TEXT NOT NULL,
                answer TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                PRIMARY KEY(conversation_key, query_key)
            );
            CREATE INDEX IF NOT EXISTS idx_web_search_cache_expiry
                ON web_search_cache(expires_at);

            CREATE TABLE IF NOT EXISTS daily_greetings (
                conversation_key TEXT NOT NULL,
                user_id TEXT NOT NULL,
                local_date TEXT NOT NULL,
                content TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(conversation_key, user_id, local_date)
            );

            CREATE TABLE IF NOT EXISTS proactive_activity (
                conversation_key TEXT PRIMARY KEY,
                message_count INTEGER NOT NULL DEFAULT 0,
                threshold INTEGER NOT NULL DEFAULT 20,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS message_reactions (
                conversation_key TEXT NOT NULL,
                target_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                label TEXT NOT NULL,
                user_name TEXT NOT NULL DEFAULT '',
                target_text TEXT NOT NULL DEFAULT '',
                sentiment TEXT NOT NULL CHECK (
                    sentiment IN ('positive', 'playful', 'sad', 'negative', 'neutral')
                ),
                active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY(conversation_key, target_id, user_id, label)
            );
            CREATE INDEX IF NOT EXISTS idx_message_reactions_member
                ON message_reactions(conversation_key, user_id, active, updated_at DESC);
            """
        )
        profile_columns = {
            str(row["name"])
            for row in self._conn.execute("PRAGMA table_info(member_profiles)").fetchall()
        }
        if "long_term_content" not in profile_columns:
            self._conn.execute(
                "ALTER TABLE member_profiles ADD COLUMN long_term_content TEXT NOT NULL DEFAULT ''"
            )
        if "short_term_content" not in profile_columns:
            self._conn.execute(
                "ALTER TABLE member_profiles ADD COLUMN short_term_content TEXT NOT NULL DEFAULT ''"
            )
        if "short_term_updated_at" not in profile_columns:
            self._conn.execute(
                "ALTER TABLE member_profiles ADD COLUMN short_term_updated_at REAL NOT NULL DEFAULT 0"
            )
        self._conn.execute(
            """
            UPDATE member_profiles
            SET long_term_content = substr(content, 1, 600)
            WHERE long_term_content = '' AND content <> ''
            """
        )
        self._conn.execute(
            """
            UPDATE member_profiles
            SET short_term_content = substr(content, 601, 400),
                short_term_updated_at = ?
            WHERE short_term_content = '' AND length(content) > 600
            """,
            (time.time(),),
        )
        self._conn.commit()

    def _prune_housekeeping(self) -> None:
        """Keep long-running one-group deployments from growing bookkeeping forever."""
        self._conn.execute(
            "DELETE FROM processed_events WHERE created_at < datetime('now', '-14 days')"
        )
        self._conn.execute(
            """
            DELETE FROM image_jobs WHERE id NOT IN (
                SELECT id FROM image_jobs ORDER BY id DESC LIMIT 100
            )
            """
        )
        self._conn.execute(
            "DELETE FROM web_search_cache WHERE expires_at <= ?", (time.time(),)
        )
        self._conn.execute(
            "DELETE FROM daily_greetings WHERE local_date < date('now', '-14 days')"
        )
        self._conn.execute(
            "UPDATE proactive_activity SET threshold = 20 WHERE threshold <> 20"
        )
        self._conn.execute(
            "DELETE FROM message_reactions WHERE active = 0 AND updated_at < ?",
            (time.time() - 14 * 86400,),
        )
        self._conn.commit()

    def set_message_reaction(
        self,
        *,
        conversation_key: str,
        target_id: str,
        user_id: str,
        user_name: str,
        label: str,
        target_text: str,
        sentiment: str,
        active: bool,
        now: float | None = None,
    ) -> bool:
        """Persist one reaction transition and reject duplicate gateway deliveries."""
        if sentiment not in {"positive", "playful", "sad", "negative", "neutral"}:
            raise ValueError("invalid reaction sentiment")
        current = time.time() if now is None else now
        key = (conversation_key, target_id, user_id, label)
        with self._lock:
            row = self._conn.execute(
                """
                SELECT active FROM message_reactions
                WHERE conversation_key = ? AND target_id = ? AND user_id = ? AND label = ?
                """,
                key,
            ).fetchone()
            wanted = 1 if active else 0
            if row is not None and int(row["active"]) == wanted:
                return False
            if row is None and not active:
                return False
            self._conn.execute(
                """
                INSERT INTO message_reactions(
                    conversation_key, target_id, user_id, label, user_name,
                    target_text, sentiment, active, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(conversation_key, target_id, user_id, label) DO UPDATE SET
                    user_name = excluded.user_name,
                    target_text = excluded.target_text,
                    sentiment = excluded.sentiment,
                    active = excluded.active,
                    updated_at = excluded.updated_at
                """,
                (*key, user_name[:100], target_text[:600], sentiment, wanted, current, current),
            )
            self._conn.commit()
        return True

    def reaction_feedback(
        self, conversation_key: str, user_id: str
    ) -> ReactionFeedback | None:
        """Return active long-term signals plus the newest reaction for short-term tone."""
        with self._lock:
            totals = self._conn.execute(
                """
                SELECT COUNT(*) AS total_count,
                       SUM(sentiment = 'positive') AS positive_count,
                       SUM(sentiment = 'playful') AS playful_count,
                       SUM(sentiment = 'sad') AS sad_count,
                       SUM(sentiment = 'negative') AS negative_count
                FROM message_reactions
                WHERE conversation_key = ? AND user_id = ? AND active = 1
                """,
                (conversation_key, user_id),
            ).fetchone()
            if totals is None or int(totals["total_count"] or 0) == 0:
                return None
            latest = self._conn.execute(
                """
                SELECT label, target_text, sentiment, updated_at
                FROM message_reactions
                WHERE conversation_key = ? AND user_id = ? AND active = 1
                ORDER BY updated_at DESC LIMIT 1
                """,
                (conversation_key, user_id),
            ).fetchone()
        assert latest is not None
        return ReactionFeedback(
            total_count=int(totals["total_count"] or 0),
            positive_count=int(totals["positive_count"] or 0),
            playful_count=int(totals["playful_count"] or 0),
            sad_count=int(totals["sad_count"] or 0),
            negative_count=int(totals["negative_count"] or 0),
            last_label=str(latest["label"]),
            last_target_text=str(latest["target_text"]),
            last_sentiment=str(latest["sentiment"]),
            last_at=float(latest["updated_at"]),
        )

    def claim_rate_limit(
        self, key: str, interval_seconds: int, *, now: float | None = None
    ) -> tuple[bool, int, float | None]:
        """Atomically claim a persistent cooldown slot."""
        current = time.time() if now is None else now
        with self._lock:
            row = self._conn.execute(
                "SELECT last_at FROM rate_limits WHERE key = ?", (key,)
            ).fetchone()
            if row is not None:
                remaining = float(row["last_at"]) + interval_seconds - current
                if remaining > 0:
                    return False, max(1, ceil(remaining)), None
            self._conn.execute(
                """
                INSERT INTO rate_limits(key, last_at) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET last_at = excluded.last_at
                """,
                (key, current),
            )
            self._conn.commit()
            return True, 0, current

    def rate_limit_remaining(
        self, key: str, interval_seconds: int, *, now: float | None = None
    ) -> int:
        """Read a persistent cooldown without claiming or extending it."""
        current = time.time() if now is None else now
        with self._lock:
            row = self._conn.execute(
                "SELECT last_at FROM rate_limits WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return 0
        remaining = float(row["last_at"]) + interval_seconds - current
        return max(0, ceil(remaining))

    def release_rate_limit(self, key: str, claim_timestamp: float) -> bool:
        """Release only the exact claim, so an old failure cannot erase a new slot."""
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM rate_limits WHERE key = ? AND last_at = ?",
                (key, claim_timestamp),
            )
            self._conn.commit()
            return cursor.rowcount > 0

    def web_search_cache(
        self,
        conversation_key: str,
        query_key: str,
        *,
        now: float | None = None,
    ) -> WebSearchCacheEntry | None:
        current = time.time() if now is None else now
        with self._lock:
            row = self._conn.execute(
                """
                SELECT answer, created_at, expires_at FROM web_search_cache
                WHERE conversation_key = ? AND query_key = ? AND expires_at > ?
                """,
                (conversation_key, query_key, current),
            ).fetchone()
            if row is None:
                self._conn.execute(
                    """
                    DELETE FROM web_search_cache
                    WHERE conversation_key = ? AND query_key = ? AND expires_at <= ?
                    """,
                    (conversation_key, query_key, current),
                )
                self._conn.commit()
                return None
        return WebSearchCacheEntry(
            answer=row["answer"],
            created_at=float(row["created_at"]),
            expires_at=float(row["expires_at"]),
        )

    def save_web_search_cache(
        self,
        conversation_key: str,
        query_key: str,
        query: str,
        answer: str,
        ttl_seconds: int,
        *,
        now: float | None = None,
    ) -> None:
        current = time.time() if now is None else now
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO web_search_cache(
                    conversation_key, query_key, query, answer, created_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(conversation_key, query_key) DO UPDATE SET
                    query = excluded.query,
                    answer = excluded.answer,
                    created_at = excluded.created_at,
                    expires_at = excluded.expires_at
                """,
                (
                    conversation_key,
                    query_key,
                    query[:500],
                    answer.strip(),
                    current,
                    current + ttl_seconds,
                ),
            )
            self._conn.execute(
                "DELETE FROM web_search_cache WHERE expires_at <= ?", (current,)
            )
            self._conn.commit()

    def claim_event(self, event_id: str) -> bool:
        if not event_id.strip():
            return False
        with self._lock:
            cursor = self._conn.execute(
                "INSERT OR IGNORE INTO processed_events(event_id) VALUES (?)", (event_id,)
            )
            self._conn.commit()
            return cursor.rowcount == 1

    def create_image_job(
        self,
        *,
        event_id: str,
        conversation_key: str,
        user_id: str,
        prompt: str,
        phase: str,
        started_at: float | None = None,
    ) -> int:
        current = time.time() if started_at is None else started_at
        with self._lock:
            cursor = self._conn.execute(
                """
                INSERT INTO image_jobs(
                    event_id, conversation_key, user_id, prompt,
                    status, phase, started_at, updated_at
                ) VALUES (?, ?, ?, ?, 'running', ?, ?, ?)
                """,
                (
                    event_id,
                    conversation_key,
                    user_id,
                    prompt.strip(),
                    phase,
                    current,
                    current,
                ),
            )
            self._conn.commit()
            return int(cursor.lastrowid)

    def update_image_job(
        self,
        job_id: int,
        *,
        phase: str,
        status: str = "running",
        error: str = "",
        now: float | None = None,
    ) -> None:
        if status not in {"running", "sent", "failed", "interrupted"}:
            raise ValueError("invalid image job status")
        current = time.time() if now is None else now
        with self._lock:
            self._conn.execute(
                """
                UPDATE image_jobs
                SET status = ?, phase = ?, error = ?, updated_at = ?
                WHERE id = ?
                """,
                (status, phase, error[:500], current, job_id),
            )
            self._conn.commit()

    @staticmethod
    def _image_job_from_row(row: sqlite3.Row | None) -> ImageJobRecord | None:
        if row is None:
            return None
        return ImageJobRecord(
            id=int(row["id"]),
            event_id=row["event_id"],
            conversation_key=row["conversation_key"],
            user_id=row["user_id"],
            prompt=row["prompt"],
            status=row["status"],
            phase=row["phase"],
            started_at=float(row["started_at"]),
            updated_at=float(row["updated_at"]),
            error=row["error"],
        )

    def latest_image_job(self, conversation_key: str) -> ImageJobRecord | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT * FROM image_jobs
                WHERE conversation_key = ? ORDER BY id DESC LIMIT 1
                """,
                (conversation_key,),
            ).fetchone()
        return self._image_job_from_row(row)

    def interrupt_running_image_jobs(self) -> int:
        """Mark jobs abandoned by a previous process; generation cannot be resumed safely."""
        current = time.time()
        with self._lock:
            cursor = self._conn.execute(
                """
                UPDATE image_jobs
                SET status = 'interrupted', phase = '服务重启中断',
                    error = 'worker restarted before completion', updated_at = ?
                WHERE status = 'running'
                """,
                (current,),
            )
            self._conn.commit()
            return cursor.rowcount

    def append(
        self, conversation_key: str, user_id: str, role: str, content: str
    ) -> int | None:
        if role not in {"user", "assistant"}:
            raise ValueError("invalid role")
        content = content.strip()
        if not content:
            return None
        with self._lock:
            cursor = self._conn.execute(
                "INSERT INTO messages(conversation_key, user_id, role, content) VALUES (?, ?, ?, ?)",
                (conversation_key, user_id, role, content),
            )
            self._conn.execute(
                """
                DELETE FROM messages
                WHERE conversation_key = ?
                AND id <= COALESCE((
                    SELECT last_message_id FROM conversation_summaries
                    WHERE conversation_key = ?
                ), 0)
                AND id NOT IN (
                    SELECT id FROM messages WHERE conversation_key = ?
                    ORDER BY id DESC LIMIT ?
                )
                """,
                (
                    conversation_key,
                    conversation_key,
                    conversation_key,
                    self.max_messages,
                ),
            )
            self._conn.commit()
            return int(cursor.lastrowid)

    def history(self, conversation_key: str, limit: int) -> list[StoredMessage]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT role, content, user_id FROM (
                    SELECT id, role, content, user_id FROM messages
                    WHERE conversation_key = ? ORDER BY id DESC LIMIT ?
                ) ORDER BY id ASC
                """,
                (conversation_key, limit),
            ).fetchall()
        return [
            StoredMessage(row["role"], row["content"], row["user_id"])
            for row in rows
        ]

    def resolve_quoted_speaker(
        self, conversation_key: str, quoted_content: str, *, limit: int = 200
    ) -> str | None:
        """Resolve a QQ quote to the recent human or bot message it copied."""
        quoted_candidates = _quote_text_candidates(quoted_content)
        if not quoted_candidates:
            return None
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT role, user_id, content
                FROM messages
                WHERE conversation_key = ?
                ORDER BY id DESC LIMIT ?
                """,
                (conversation_key, limit),
            ).fetchall()
        prepared_rows: list[tuple[str, set[str]]] = []
        for row in rows:
            body = str(row["content"] or "").strip()
            speaker = "桑多涅" if row["role"] == "assistant" else ""
            if row["role"] == "user":
                match = _STORED_GROUP_SPEAKER_RE.match(body)
                if match:
                    speaker = match.group("name").strip()
                    body = body[match.end() :].strip()
            elif body.startswith("[网页检索·") and "\n" in body:
                body = body.split("\n", 1)[1].strip()
            if speaker:
                prepared_rows.append((speaker, _quote_text_variants(body)))

        # Candidate priority is more important than recency.  In QQ's nested
        # serialization the first candidate is the message actually clicked;
        # later candidates are merely its quoted ancestors.
        for candidate in quoted_candidates:
            matches = {
                speaker for speaker, variants in prepared_rows if candidate in variants
            }
            if len(matches) == 1:
                return next(iter(matches))
            if len(matches) > 1:
                # The platform did not provide an author id and text alone is
                # ambiguous.  Returning unknown is safer than assigning the
                # quote to whichever duplicate happened to be most recent.
                return None
        return None

    def set_member_address(
        self,
        conversation_key: str,
        user_id: str,
        address: str,
        *,
        set_by_user_id: str,
    ) -> None:
        address = address.strip()[:40]
        if not address:
            raise ValueError("address must not be empty")
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO member_addresses(
                    conversation_key, user_id, address, set_by_user_id
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(conversation_key, user_id) DO UPDATE SET
                    address = excluded.address,
                    set_by_user_id = excluded.set_by_user_id,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (conversation_key, user_id, address, set_by_user_id),
            )
            self._conn.commit()

    def member_address(self, conversation_key: str, user_id: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT address FROM member_addresses
                WHERE conversation_key = ? AND user_id = ?
                """,
                (conversation_key, user_id),
            ).fetchone()
        return str(row["address"]) if row else None

    def member_addresses(self, conversation_key: str) -> dict[str, str]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT user_id, address FROM member_addresses
                WHERE conversation_key = ?
                """,
                (conversation_key,),
            ).fetchall()
        return {str(row["user_id"]): str(row["address"]) for row in rows}

    def message_count(self, conversation_key: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS count FROM messages WHERE conversation_key = ?",
                (conversation_key,),
            ).fetchone()
        return int(row["count"])

    def latest_message_id(self, conversation_key: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(id), 0) AS id FROM messages WHERE conversation_key = ?",
                (conversation_key,),
            ).fetchone()
        return int(row["id"])

    def latest_message_created_at(self, conversation_key: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT MAX(created_at) AS created_at FROM messages WHERE conversation_key = ?",
                (conversation_key,),
            ).fetchone()
        return str(row["created_at"]) if row["created_at"] else None

    def update_message_content(self, message_id: int, content: str) -> bool:
        content = content.strip()
        if not content:
            return False
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE messages SET content = ? WHERE id = ?", (content, message_id)
            )
            self._conn.commit()
        return cursor.rowcount > 0

    def active_member_names_since(
        self, conversation_key: str, since_utc: str
    ) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT m.user_id, COALESCE(p.user_name, '') AS user_name, m.content
                FROM messages AS m
                LEFT JOIN member_profiles AS p
                  ON p.conversation_key = m.conversation_key AND p.user_id = m.user_id
                WHERE m.conversation_key = ? AND m.role = 'user' AND m.created_at >= ?
                  AND m.id = (
                    SELECT MAX(m2.id) FROM messages AS m2
                    WHERE m2.conversation_key = m.conversation_key
                      AND m2.user_id = m.user_id AND m2.role = 'user'
                      AND m2.created_at >= ?
                  )
                ORDER BY m.id ASC
                """,
                (conversation_key, since_utc, since_utc),
            ).fetchall()
        names: list[str] = []
        for row in rows:
            name = str(row["user_name"] or "").strip()
            if not name:
                tags = []
                rest = str(row["content"] or "")
                while rest.startswith("[") and "]" in rest:
                    tag, rest = rest[1:].split("]", 1)
                    tags.append(tag)
                name = next((tag for tag in reversed(tags) if tag != "最高指挥"), "")
            names.append(name or f"群成员{str(row['user_id'])[-4:]}")
        return names

    def claim_daily_greeting(
        self, conversation_key: str, user_id: str, local_date: str
    ) -> bool:
        with self._lock:
            # A killed process must not leave today's member permanently marked
            # as greeted. Only incomplete reservations older than ten minutes.
            self._conn.execute(
                "DELETE FROM daily_greetings WHERE conversation_key=? AND user_id=? "
                "AND local_date=? AND content='' AND created_at < datetime('now', '-10 minutes')",
                (conversation_key, user_id, local_date),
            )
            cursor = self._conn.execute(
                """
                INSERT OR IGNORE INTO daily_greetings(
                    conversation_key, user_id, local_date, content
                ) VALUES (?, ?, ?, '')
                """,
                (conversation_key, user_id, local_date),
            )
            self._conn.commit()
        return cursor.rowcount > 0

    def release_daily_greeting(self, conversation_key: str, user_id: str, local_date: str) -> None:
        with self._lock:
            self._conn.execute(
                "DELETE FROM daily_greetings WHERE conversation_key=? AND user_id=? "
                "AND local_date=? AND content=''",
                (conversation_key, user_id, local_date),
            )
            self._conn.commit()

    def save_daily_greeting(
        self, conversation_key: str, user_id: str, local_date: str, content: str
    ) -> None:
        with self._lock:
            self._conn.execute(
                """
                UPDATE daily_greetings SET content = ?
                WHERE conversation_key = ? AND user_id = ? AND local_date = ?
                """,
                (content.strip(), conversation_key, user_id, local_date),
            )
            self._conn.commit()

    def previous_daily_greeting(
        self, conversation_key: str, user_id: str, before_date: str
    ) -> str:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT content FROM daily_greetings
                WHERE conversation_key = ? AND user_id = ? AND local_date < ?
                  AND content <> ''
                ORDER BY local_date DESC LIMIT 1
                """,
                (conversation_key, user_id, before_date),
            ).fetchone()
        return str(row["content"]) if row else ""

    def increment_proactive_activity(self, conversation_key: str) -> tuple[int, int]:
        with self._lock:
            row = self._conn.execute(
                "SELECT message_count, threshold FROM proactive_activity WHERE conversation_key = ?",
                (conversation_key,),
            ).fetchone()
            if row is None:
                count, threshold = 1, 20
                self._conn.execute(
                    "INSERT INTO proactive_activity VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
                    (conversation_key, count, threshold),
                )
            else:
                count, threshold = int(row["message_count"]) + 1, 20
                self._conn.execute(
                    """
                    UPDATE proactive_activity
                    SET message_count = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE conversation_key = ?
                    """,
                    (count, conversation_key),
                )
            self._conn.commit()
        return count, threshold

    def claim_proactive_activity(self, conversation_key: str) -> bool:
        """Atomically count one message and claim a due proactive turn once."""
        with self._lock:
            row = self._conn.execute(
                "SELECT message_count, threshold FROM proactive_activity WHERE conversation_key = ?",
                (conversation_key,),
            ).fetchone()
            count = int(row["message_count"]) + 1 if row else 1
            threshold = 20
            due = count >= threshold
            next_count = 0 if due else count
            next_threshold = 20
            self._conn.execute(
                """
                INSERT INTO proactive_activity(
                    conversation_key, message_count, threshold, updated_at
                ) VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(conversation_key) DO UPDATE SET
                    message_count = excluded.message_count,
                    threshold = excluded.threshold,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (conversation_key, next_count, next_threshold),
            )
            self._conn.commit()
        return due

    def reset_proactive_activity(self, conversation_key: str) -> int:
        threshold = 20
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO proactive_activity(
                    conversation_key, message_count, threshold, updated_at
                ) VALUES (?, 0, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(conversation_key) DO UPDATE SET
                    message_count = 0, threshold = excluded.threshold,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (conversation_key, threshold),
            )
            self._conn.commit()
        return threshold

    def unsummarized(
        self, conversation_key: str, after_message_id: int, limit: int = 400
    ) -> list[StoredMessageRecord]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, role, content FROM messages
                WHERE conversation_key = ? AND id > ?
                ORDER BY id ASC LIMIT ?
                """,
                (conversation_key, after_message_id, limit),
            ).fetchall()
        return [
            StoredMessageRecord(row["id"], row["role"], row["content"])
            for row in rows
        ]

    def summary(self, conversation_key: str) -> SummaryState:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT content, last_message_id FROM conversation_summaries
                WHERE conversation_key = ?
                """,
                (conversation_key,),
            ).fetchone()
        if row is None:
            return SummaryState("", 0)
        return SummaryState(row["content"], row["last_message_id"])

    def save_summary(
        self, conversation_key: str, content: str, last_message_id: int
    ) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO conversation_summaries(
                    conversation_key, content, last_message_id, updated_at
                ) VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(conversation_key) DO UPDATE SET
                    content = excluded.content,
                    last_message_id = excluded.last_message_id,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (conversation_key, content.strip(), last_message_id),
            )
            self._conn.commit()

    def member_profile(self, conversation_key: str, user_id: str) -> MemberProfile | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT user_id, user_name, long_term_content,
                       CASE WHEN short_term_updated_at >= ? THEN short_term_content ELSE '' END
                           AS short_term_content,
                       last_message_id
                FROM member_profiles WHERE conversation_key = ? AND user_id = ?
                """,
                (time.time() - 2 * 86400, conversation_key, user_id),
            ).fetchone()
        if row is None:
            return None
        return MemberProfile(
            row["user_id"],
            row["user_name"],
            row["long_term_content"],
            row["short_term_content"],
            row["last_message_id"],
        )

    def member_profiles(self, conversation_key: str, limit: int = 20) -> list[MemberProfile]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT user_id, user_name, long_term_content,
                       CASE WHEN short_term_updated_at >= ? THEN short_term_content ELSE '' END
                           AS short_term_content,
                       last_message_id
                FROM member_profiles WHERE conversation_key = ?
                ORDER BY updated_at DESC LIMIT ?
                """,
                (time.time() - 2 * 86400, conversation_key, limit),
            ).fetchall()
        return [
            MemberProfile(
                row["user_id"],
                row["user_name"],
                row["long_term_content"],
                row["short_term_content"],
                row["last_message_id"],
            )
            for row in rows
        ]

    def conversation_members(
        self, conversation_key: str, limit: int = 100
    ) -> list[ConversationMember]:
        """Return every retained/profiled speaker, not just recently active members."""
        with self._lock:
            rows = self._conn.execute(
                """
                WITH member_ids AS (
                    SELECT user_id FROM messages
                    WHERE conversation_key = ? AND role = 'user'
                    UNION
                    SELECT user_id FROM member_profiles
                    WHERE conversation_key = ?
                )
                SELECT ids.user_id,
                       COALESCE(profile.user_name, '') AS user_name,
                       COALESCE(message_stats.message_count, 0) AS message_count,
                       COALESCE(profile.long_term_content, '') AS long_term_content,
                       CASE WHEN profile.short_term_updated_at >= ?
                            THEN COALESCE(profile.short_term_content, '') ELSE '' END
                            AS short_term_content,
                       message_stats.latest_content
                FROM member_ids AS ids
                LEFT JOIN member_profiles AS profile
                  ON profile.conversation_key = ? AND profile.user_id = ids.user_id
                LEFT JOIN (
                    SELECT grouped.user_id, grouped.message_count, latest.content AS latest_content
                    FROM (
                        SELECT user_id, COUNT(*) AS message_count, MAX(id) AS latest_id
                        FROM messages
                        WHERE conversation_key = ? AND role = 'user'
                        GROUP BY user_id
                    ) AS grouped
                    LEFT JOIN messages AS latest ON latest.id = grouped.latest_id
                ) AS message_stats ON message_stats.user_id = ids.user_id
                ORDER BY message_count DESC, ids.user_id ASC
                LIMIT ?
                """,
                (
                    conversation_key,
                    conversation_key,
                    time.time() - 2 * 86400,
                    conversation_key,
                    conversation_key,
                    limit,
                ),
            ).fetchall()
        members: list[ConversationMember] = []
        for row in rows:
            name = str(row["user_name"] or "").strip()
            if not name:
                rest = str(row["latest_content"] or "")
                tags: list[str] = []
                while rest.startswith("[") and "]" in rest:
                    tag, rest = rest[1:].split("]", 1)
                    tags.append(tag)
                name = next((tag for tag in reversed(tags) if tag != "最高指挥"), "")
            members.append(
                ConversationMember(
                    user_id=str(row["user_id"]),
                    user_name=name or f"群成员{str(row['user_id'])[-4:]}",
                    message_count=int(row["message_count"]),
                    profile=(
                        ("长期印象：" + str(row["long_term_content"]))
                        if row["long_term_content"]
                        else ""
                    )
                    + (
                        ("\n短期印象：" + str(row["short_term_content"]))
                        if row["short_term_content"]
                        else ""
                    ),
                )
            )
        return members

    def member_messages_since(
        self, conversation_key: str, user_id: str, after_message_id: int, limit: int = 200
    ) -> list[StoredMessageRecord]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, role, content FROM messages
                WHERE conversation_key = ? AND user_id = ? AND role = 'user' AND id > ?
                ORDER BY id ASC LIMIT ?
                """,
                (conversation_key, user_id, after_message_id, limit),
            ).fetchall()
        return [
            StoredMessageRecord(row["id"], row["role"], row["content"])
            for row in rows
        ]

    def member_recent_messages(
        self, conversation_key: str, user_id: str, *, hours: int = 48, limit: int = 200
    ) -> list[StoredMessageRecord]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, role, content FROM messages
                WHERE conversation_key = ? AND user_id = ? AND role = 'user'
                  AND created_at >= datetime('now', ?)
                ORDER BY id ASC LIMIT ?
                """,
                (conversation_key, user_id, f"-{hours} hours", limit),
            ).fetchall()
        return [
            StoredMessageRecord(row["id"], row["role"], row["content"])
            for row in rows
        ]

    def save_member_profile(
        self,
        conversation_key: str,
        user_id: str,
        user_name: str,
        long_term_content: str,
        last_message_id: int,
        short_term_content: str = "",
    ) -> None:
        long_term_content = long_term_content.strip()[:600]
        short_term_content = short_term_content.strip()[:400]
        combined = "\n".join(
            part
            for part in (
                "长期印象：" + long_term_content if long_term_content else "",
                "短期印象：" + short_term_content if short_term_content else "",
            )
            if part
        )
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO member_profiles(
                    conversation_key, user_id, user_name, content, long_term_content,
                    short_term_content, short_term_updated_at, last_message_id, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(conversation_key, user_id) DO UPDATE SET
                    user_name = excluded.user_name,
                    content = excluded.content,
                    long_term_content = excluded.long_term_content,
                    short_term_content = excluded.short_term_content,
                    short_term_updated_at = excluded.short_term_updated_at,
                    last_message_id = excluded.last_message_id,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    conversation_key,
                    user_id,
                    user_name,
                    combined,
                    long_term_content,
                    short_term_content,
                    time.time(),
                    last_message_id,
                ),
            )
            self._conn.commit()

    def clear_history(self, conversation_key: str) -> int:
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM messages WHERE conversation_key = ?", (conversation_key,)
            )
            self._conn.commit()
            return cursor.rowcount

    def clear_conversation(self, conversation_key: str) -> int:
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM messages WHERE conversation_key = ?", (conversation_key,)
            )
            self._conn.execute(
                "DELETE FROM conversation_summaries WHERE conversation_key = ?",
                (conversation_key,),
            )
            self._conn.commit()
            return cursor.rowcount

    def add_fact(self, user_key: str, content: str) -> bool:
        with self._lock:
            cursor = self._conn.execute(
                "INSERT OR IGNORE INTO facts(user_key, content) VALUES (?, ?)",
                (user_key, content.strip()),
            )
            self._conn.commit()
            return cursor.rowcount == 1

    def replace_fact(self, user_key: str, prefix: str, content: str) -> bool:
        """Atomically replace one namespaced durable fact.

        Natural-language declarations such as a group relationship should not
        accumulate contradictory historical values forever.  The stable prefix
        acts as a tiny schema while keeping the existing lightweight facts table.
        """
        normalized_prefix = prefix.strip()
        normalized_content = content.strip()
        if not normalized_prefix or not normalized_content.startswith(normalized_prefix):
            raise ValueError("fact content must start with its non-empty prefix")
        with self._lock:
            existing = self._conn.execute(
                "SELECT content FROM facts WHERE user_key = ? AND content LIKE ?",
                (user_key, normalized_prefix + "%"),
            ).fetchall()
            if len(existing) == 1 and existing[0]["content"] == normalized_content:
                return False
            self._conn.execute(
                "DELETE FROM facts WHERE user_key = ? AND content LIKE ?",
                (user_key, normalized_prefix + "%"),
            )
            self._conn.execute(
                "INSERT INTO facts(user_key, content) VALUES (?, ?)",
                (user_key, normalized_content),
            )
            self._conn.commit()
            return True

    def facts(self, user_key: str, limit: int = 60) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT content FROM facts WHERE user_key = ? ORDER BY id DESC LIMIT ?",
                (user_key, limit),
            ).fetchall()
        return [row["content"] for row in rows]

    def remove_fact(self, user_key: str, content: str) -> bool:
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM facts WHERE user_key = ? AND content = ?",
                (user_key, content.strip()),
            )
            self._conn.commit()
            return cursor.rowcount > 0

    def clear_facts(self, user_key: str) -> int:
        with self._lock:
            cursor = self._conn.execute("DELETE FROM facts WHERE user_key = ?", (user_key,))
            self._conn.commit()
            return cursor.rowcount

    def close(self) -> None:
        with self._lock:
            self._conn.close()
