import sqlite3
import time
from pathlib import Path

from talk_bot.memory import MemoryStore


def test_history_is_persistent_and_bounded(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    store = MemoryStore(path, max_messages_per_conversation=3)
    for index in range(5):
        store.append("c2c:u1", "u1", "user", f"m{index}")
    last_id = store.unsummarized("c2c:u1", 0)[-1].id
    store.save_summary("c2c:u1", "摘要", last_id)
    store.append("c2c:u1", "u1", "user", "m5")
    store.close()

    reopened = MemoryStore(path, max_messages_per_conversation=3)
    assert [item.content for item in reopened.history("c2c:u1", 10)] == ["m3", "m4", "m5"]
    reopened.close()


def test_unsummarized_messages_are_never_pruned_by_capacity(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.db", max_messages_per_conversation=3)
    for index in range(5):
        store.append("group:g1", "u1", "user", f"m{index}")
    records = store.unsummarized("group:g1", 0)
    assert [item.content for item in records] == ["m0", "m1", "m2", "m3", "m4"]

    store.save_summary("group:g1", "前两条摘要", records[1].id)
    store.append("group:g1", "u1", "user", "m5")
    assert [item.content for item in store.history("group:g1", 10)] == [
        "m2",
        "m3",
        "m4",
        "m5",
    ]
    store.close()


def test_facts_are_scoped_and_deduplicated(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.db")
    assert store.add_fact("c2c:u1", "我喜欢咖啡") is True
    assert store.add_fact("c2c:u1", "我喜欢咖啡") is False
    assert store.facts("c2c:u1") == ["我喜欢咖啡"]
    assert store.facts("c2c:u2") == []
    assert store.remove_fact("c2c:u1", "我喜欢咖啡") is True
    store.close()


def test_rate_limit_remaining_does_not_claim_or_extend_slot(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.db")
    key = "proactive:speak:group:g1"
    assert store.rate_limit_remaining(key, 1200, now=1000) == 0
    assert store.claim_rate_limit(key, 1200, now=1000)[0] is True
    assert store.rate_limit_remaining(key, 1200, now=1100) == 1100
    assert store.rate_limit_remaining(key, 1200, now=2200) == 0
    assert store.claim_rate_limit(key, 1200, now=2200)[0] is True
    store.close()


def test_namespaced_fact_can_be_replaced_without_duplicates(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.db")
    prefix = "群聊固定关系设定（最高指挥的女朋友）："
    assert store.replace_fact("group:g1", prefix, prefix + "长江七号。") is True
    assert store.replace_fact("group:g1", prefix, prefix + "长江七号。") is False
    assert store.replace_fact("group:g1", prefix, prefix + "新名字。") is True
    assert store.facts("group:g1") == [prefix + "新名字。"]
    store.close()


def test_event_claim_prevents_duplicate_processing(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.db")
    assert store.claim_event("") is False
    assert store.claim_event("event-1") is True
    assert store.claim_event("event-1") is False
    store.close()


def test_message_count_is_scoped_to_conversation(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.db")
    store.append("group:g1", "u1", "user", "一")
    store.append("group:g1", "u1", "assistant", "二")
    store.append("group:g2", "u2", "user", "三")
    assert store.message_count("group:g1") == 2
    assert store.message_count("group:g2") == 1
    store.close()


def test_quote_speaker_is_resolved_from_exact_recent_message(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.db")
    store.append("group:g1", "u1", "user", "[最高指挥][MonHamed] 我去你世界")
    store.append("group:g1", "u2", "user", "[长江七号] 买券涨VIP等级")
    store.append("group:g1", "u2", "assistant", "这是桑多涅的回答。")

    assert store.resolve_quoted_speaker("group:g1", "我去你世界") == "MonHamed"
    assert store.resolve_quoted_speaker("group:g1", "买券涨VIP等级") == "长江七号"
    assert store.resolve_quoted_speaker("group:g1", "这是桑多涅的回答。") == "桑多涅"
    assert store.resolve_quoted_speaker("group:g1", "不存在的消息") is None
    store.close()


def test_quote_speaker_resolver_handles_mentions_and_nested_quotes(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.db")
    store.append(
        "group:g1",
        "owner",
        "user",
        "[最高指挥][MonHamed] <@u2> 晚上二刷欢迎来龙餐馆[表情:拜谢]",
    )
    store.append(
        "group:g1",
        "u2",
        "user",
        "[长江七号] [引用消息·原发送者：里]\n带我做桂车\n"
        "[当前消息]\n<@u3> 你也要弯下脊梁骨吗[表情:可怜]",
    )

    assert (
        store.resolve_quoted_speaker("group:g1", "晚上二刷欢迎来龙餐馆[表情:拜谢]")
        == "MonHamed"
    )
    nested = (
        "=== 消息 1 ===\n[消息内容] 你也要弯下脊梁骨吗[表情:可怜]\n"
        "[消息类型] 引用消息\n[关联消息]\n[消息内容] 带我做桂车\n"
        "[消息类型] 引用消息"
    )
    assert store.resolve_quoted_speaker("group:g1", nested) == "长江七号"
    store.close()


def test_nested_quote_prefers_clicked_outer_message_over_quoted_ancestor(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "memory.db")
    store.append(
        "group:g1",
        "owner",
        "user",
        "[最高指挥][MonHamed] 列举一下中国大学生竞赛白名单前十名",
    )
    store.append(
        "group:g1",
        "u2",
        "user",
        "[长江七号] [引用消息·原发送者：MonHamed] "
        "列举一下中国大学生竞赛白名单前十名 "
        "[当前消息] 数学竞赛竟然不在吗",
    )

    nested = (
        "=== 消息 1 ===\n[消息内容] 数学竞赛竟然不在吗\n"
        "[消息类型] 引用消息\n[关联消息]\n--- 第1条 ---\n"
        "[消息内容] 列举一下中国大学生竞赛白名单前十名\n"
        "[消息类型] 引用消息"
    )
    assert store.resolve_quoted_speaker("group:g1", nested) == "长江七号"
    store.close()


def test_member_address_is_keyed_by_stable_user_id(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.db")
    store.set_member_address("group:g1", "u2", "啥子", set_by_user_id="owner")
    assert store.member_address("group:g1", "u2") == "啥子"
    assert store.member_address("group:g1", "owner") is None
    assert store.member_addresses("group:g1") == {"u2": "啥子"}
    store.close()


def test_quote_speaker_does_not_guess_between_duplicate_texts(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.db")
    store.append("group:g1", "u1", "user", "[MonHamed] 你好")
    store.append("group:g1", "u2", "user", "[长江七号] 你好")
    assert store.resolve_quoted_speaker("group:g1", "你好") is None
    store.close()


def test_conversation_members_includes_every_retained_and_profiled_speaker(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "memory.db")
    store.append("group:g1", "u1", "user", "[最高指挥][MonHamed] 第一条")
    store.append("group:g1", "u1", "user", "[最高指挥][MonHamed] 第二条")
    store.append("group:g1", "u2", "user", "[长江七号] 只有一条")
    store.save_member_profile("group:g1", "u3", "里", "旧印象", 0)

    members = store.conversation_members("group:g1")
    assert [(item.user_name, item.message_count) for item in members] == [
        ("MonHamed", 2),
        ("长江七号", 1),
        ("里", 0),
    ]
    store.close()


def test_daily_greeting_is_claimed_once_and_remembers_previous_day(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.db")
    assert store.claim_daily_greeting("group:g1", "u1", "2026-08-27") is True
    assert store.claim_daily_greeting("group:g1", "u1", "2026-08-27") is False
    store.save_daily_greeting("group:g1", "u1", "2026-08-27", "第一天早安")
    assert (
        store.previous_daily_greeting("group:g1", "u1", "2026-08-28")
        == "第一天早安"
    )
    store.close()


def test_proactive_counter_uses_fixed_20_message_window_and_resets(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.db")
    reached_at = None
    for index in range(1, 21):
        count, threshold = store.increment_proactive_activity("group:g1")
        assert threshold == 20
        if count >= threshold:
            reached_at = index
            break
    assert reached_at == 20
    new_threshold = store.reset_proactive_activity("group:g1")
    count, threshold = store.increment_proactive_activity("group:g1")
    assert count == 1
    assert threshold == new_threshold == 20
    store.close()


def test_proactive_claim_is_atomic_and_immediately_starts_a_new_window(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "memory.db")
    claims = [store.claim_proactive_activity("group:g1") for _ in range(20)]
    assert sum(claims) == 1
    # A claimed turn resets its counter immediately, so concurrent handlers cannot
    # all observe the same overdue threshold.
    assert store.claim_proactive_activity("group:g1") is False
    store.close()


def test_legacy_random_proactive_threshold_is_migrated_to_20(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    store = MemoryStore(path)
    store.increment_proactive_activity("group:g1")
    store._conn.execute(
        "UPDATE proactive_activity SET message_count = 7, threshold = 29 "
        "WHERE conversation_key = 'group:g1'"
    )
    store._conn.commit()
    store.close()

    reopened = MemoryStore(path)
    count, threshold = reopened.increment_proactive_activity("group:g1")
    assert (count, threshold) == (8, 20)
    reopened.close()


def test_message_can_be_updated_in_place_without_changing_order(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.db")
    first = store.append("group:g1", "u1", "user", "[图片处理中]")
    store.append("group:g1", "u2", "user", "后来的文字")
    assert first is not None
    assert store.update_message_content(first, "[图片描述]完成")
    assert [item.content for item in store.history("group:g1", 10)] == [
        "[图片描述]完成",
        "后来的文字",
    ]
    assert store.latest_message_created_at("group:g1") is not None
    store.close()


def test_summary_is_persistent_and_cleared_with_conversation(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    store = MemoryStore(path)
    store.append("group:g1", "u1", "user", "[u1] 第一条")
    record = store.unsummarized("group:g1", 0)[0]
    store.save_summary("group:g1", "群里在讨论测试", record.id)
    store.close()

    reopened = MemoryStore(path)
    assert reopened.summary("group:g1").content == "群里在讨论测试"
    assert reopened.clear_conversation("group:g1") == 1
    assert reopened.summary("group:g1").content == ""
    reopened.close()


def test_image_job_state_is_persistent_and_running_jobs_are_interrupted(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    store = MemoryStore(path)
    job_id = store.create_image_job(
        event_id="draw-1",
        conversation_key="group:g1",
        user_id="u1",
        prompt="桑多涅立绘",
        phase="模型渲染",
        started_at=1000,
    )
    running = store.latest_image_job("group:g1")
    assert running is not None
    assert running.id == job_id
    assert running.status == "running"
    store.close()

    reopened = MemoryStore(path)
    assert reopened.interrupt_running_image_jobs() == 1
    interrupted = reopened.latest_image_job("group:g1")
    assert interrupted is not None
    assert interrupted.status == "interrupted"
    assert "重启" in interrupted.phase
    reopened.close()


def test_image_job_success_and_failure_are_queryable(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.db")
    sent_id = store.create_image_job(
        event_id="draw-sent",
        conversation_key="group:g1",
        user_id="u1",
        prompt="第一张",
        phase="上传 QQ",
        started_at=1000,
    )
    store.update_image_job(sent_id, phase="已发送", status="sent", now=1010)
    assert store.latest_image_job("group:g1").status == "sent"  # type: ignore[union-attr]

    failed_id = store.create_image_job(
        event_id="draw-failed",
        conversation_key="group:g1",
        user_id="u1",
        prompt="第二张",
        phase="模型渲染",
        started_at=1020,
    )
    store.update_image_job(
        failed_id, phase="生成失败", status="failed", error="timeout", now=1030
    )
    failed = store.latest_image_job("group:g1")
    assert failed is not None
    assert failed.status == "failed"
    assert failed.error == "timeout"
    store.close()


def test_rate_limit_is_persistent_and_failed_claim_can_be_released(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    store = MemoryStore(path)
    allowed, remaining, claim = store.claim_rate_limit("image:global", 120, now=1000)
    assert (allowed, remaining) == (True, 0)
    assert claim == 1000
    store.close()

    reopened = MemoryStore(path)
    allowed, remaining, _ = reopened.claim_rate_limit(
        "image:global", 120, now=1040
    )
    assert allowed is False
    assert remaining == 80
    assert reopened.release_rate_limit("image:global", claim) is True
    assert reopened.claim_rate_limit("image:global", 120, now=1041)[0] is True
    reopened.close()


def test_member_profiles_are_separate_and_persistent(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    store = MemoryStore(path)
    store.append("group:g1", "u1", "user", "[小王] 喜欢辣锅")
    store.append("group:g1", "u2", "user", "[小李] 喜欢清汤")
    u1_record = store.member_messages_since("group:g1", "u1", 0)[0]
    store.save_member_profile("group:g1", "u1", "小王", "直率，喜欢辣锅", u1_record.id)
    store.close()

    reopened = MemoryStore(path)
    profile = reopened.member_profile("group:g1", "u1")
    assert profile is not None
    assert profile.user_name == "小王"
    assert "辣锅" in profile.content
    assert reopened.member_profile("group:g1", "u2") is None
    assert [item.content for item in reopened.member_messages_since("group:g1", "u2", 0)] == [
        "[小李] 喜欢清汤"
    ]
    reopened.close()


def test_member_profile_has_bounded_long_and_short_layers(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.db")
    store.save_member_profile(
        "group:g1",
        "u1",
        "小王",
        "长" * 700,
        8,
        short_term_content="短" * 500,
    )
    profile = store.member_profile("group:g1", "u1")
    assert profile is not None
    assert len(profile.long_term_content) == 600
    assert len(profile.short_term_content) == 400
    assert "长期印象：" in profile.content and "短期印象：" in profile.content
    store.close()


def test_legacy_member_profile_is_migrated_without_losing_content(tmp_path: Path) -> None:
    path = tmp_path / "legacy.db"
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE member_profiles (
            conversation_key TEXT NOT NULL,
            user_id TEXT NOT NULL,
            user_name TEXT NOT NULL DEFAULT '',
            content TEXT NOT NULL,
            last_message_id INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(conversation_key, user_id)
        )
        """
    )
    connection.execute(
        "INSERT INTO member_profiles VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
        ("group:g1", "u1", "小王", "旧" * 700, 7),
    )
    connection.commit()
    connection.close()

    store = MemoryStore(path)
    profile = store.member_profile("group:g1", "u1")
    assert profile is not None
    assert profile.long_term_content == "旧" * 600
    assert profile.short_term_content == "旧" * 100
    assert profile.last_message_id == 7
    store.close()


def test_expired_short_term_profile_is_not_returned(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.db")
    store.save_member_profile(
        "group:g1", "u1", "小王", "稳定", 1, short_term_content="今日低落"
    )
    store._conn.execute(
        "UPDATE member_profiles SET short_term_updated_at = ?",
        (time.time() - 3 * 86400,),
    )
    store._conn.commit()
    profile = store.member_profile("group:g1", "u1")
    assert profile is not None
    assert profile.long_term_content == "稳定"
    assert profile.short_term_content == ""
    store.close()


def test_reaction_feedback_is_persistent_idempotent_and_removable(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.db")
    kwargs = {
        "conversation_key": "group:g1",
        "target_id": "bot-1",
        "user_id": "u1",
        "user_name": "小王",
        "label": "👍",
        "target_text": "校准完成。",
        "sentiment": "positive",
        "now": 1000,
    }
    assert store.set_message_reaction(**kwargs, active=True) is True
    assert store.set_message_reaction(**kwargs, active=True) is False
    feedback = store.reaction_feedback("group:g1", "u1")
    assert feedback is not None
    assert feedback.total_count == 1
    assert feedback.positive_count == 1
    assert feedback.last_target_text == "校准完成。"
    assert store.set_message_reaction(**kwargs, active=False) is True
    assert store.reaction_feedback("group:g1", "u1") is None
    store.close()


def test_web_search_cache_respects_conversation_and_expiry(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.db")
    store.save_web_search_cache(
        "group:g1", "latest news", "Latest News", "带来源的答案", 60, now=100
    )

    cached = store.web_search_cache("group:g1", "latest news", now=120)
    assert cached is not None
    assert cached.answer == "带来源的答案"
    assert cached.created_at == 100
    assert store.web_search_cache("group:g2", "latest news", now=120) is None
    assert store.web_search_cache("group:g1", "latest news", now=161) is None
    store.close()
