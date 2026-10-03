from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest
from talk_bot.qq_adapter import QQBotRunner

@pytest.mark.asyncio
async def test_optional_greeting_failure_is_nonfatal():
    runner = object.__new__(QQBotRunner)
    runner._send_fresh_text = AsyncMock(side_effect=RuntimeError('主动消息失败, 无权限'))
    assert await runner._send_optional_text(SimpleNamespace(message_id='test'), '早安') is False
    runner._send_fresh_text = AsyncMock()
    assert await runner._send_optional_text(SimpleNamespace(message_id='test'), '早安') is True
