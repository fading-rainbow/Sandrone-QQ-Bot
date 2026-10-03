"""Independent image credentials and Sunburst Images API regression coverage."""

import base64
import io
import json
from email import policy
from email.parser import BytesParser
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from openai import AsyncOpenAI
from PIL import Image

import talk_bot.config as config_module
import talk_bot.image_gen as image_gen_module
import talk_bot.main as main_module
from talk_bot.config import Settings
from talk_bot.image_gen import ImageGenerator

SUNBURST = "gpt-image-2.5-sunburst"
_ENV_NAMES = (
    "QQ_APP_ID", "QQ_APP_SECRET", "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL",
    "IMAGE_API_KEY", "IMAGE_BASE_URL", "IMAGE_MODEL", "IMAGE_COOLDOWN_SECONDS",
    "GENERATED_IMAGE_DIR", "SANDRONE_REFERENCE_IMAGES", "OPENAI_API_MODE",
    "REASONING_EFFORT", "MAX_OUTPUT_TOKENS", "HISTORY_MESSAGES",
    "SUMMARY_TRIGGER_MESSAGES", "SUMMARY_BATCH_MESSAGES", "PROFILE_TRIGGER_MESSAGES",
    "WEB_SEARCH_ENABLED", "WEB_SEARCH_CACHE_SECONDS", "WEB_SEARCH_USER_COOLDOWN_SECONDS",
    "WEB_SEARCH_GROUP_COOLDOWN_SECONDS", "DATABASE_PATH", "LOG_LEVEL", "BOT_SYSTEM_PROMPT",
    "BOT_OWNER_IDS", "ALLOWED_GROUP_IDS",
)


@pytest.fixture
def clean_settings_env(monkeypatch):
    # Never load the developer's real .env or inherit its credentials/settings.
    monkeypatch.setattr(config_module, "load_dotenv", lambda: None)
    for name in _ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_requested_chat_and_image_defaults_use_independent_backends(clean_settings_env):
    settings = Settings.from_env(require_secrets=False)

    assert settings.image_model == SUNBURST
    assert settings.openai_model == "gemini-3.8-flash"
    assert settings.image_api_key == settings.openai_api_key == ""
    assert settings.image_base_url == settings.openai_base_url == "https://api.openai.com/v1"


def test_legacy_chat_override_does_not_change_sunburst_default(clean_settings_env):
    clean_settings_env.setenv("OPENAI_MODEL", "gemini-3-flash")

    settings = Settings.from_env(require_secrets=False)

    assert settings.openai_model == "gemini-3-flash"
    assert settings.image_model == SUNBURST


def test_image_credentials_fall_back_to_chat_credentials(clean_settings_env):
    clean_settings_env.setenv("OPENAI_API_KEY", " chat-test-key ")
    clean_settings_env.setenv("OPENAI_BASE_URL", " https://chat.example/v1/ ")
    clean_settings_env.setenv("OPENAI_MODEL", "gemini-3.8-flash")

    settings = Settings.from_env(require_secrets=False)

    assert settings.image_api_key == settings.openai_api_key == "chat-test-key"
    assert settings.image_base_url == settings.openai_base_url == "https://chat.example/v1"
    assert settings.openai_model == "gemini-3.8-flash"
    assert settings.image_model == SUNBURST


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_optional_image_credentials_use_chat_fallback(clean_settings_env, blank):
    clean_settings_env.setenv("OPENAI_API_KEY", "chat-test-key")
    clean_settings_env.setenv("OPENAI_BASE_URL", "https://chat.example/v1")
    clean_settings_env.setenv("IMAGE_API_KEY", blank)
    clean_settings_env.setenv("IMAGE_BASE_URL", blank)

    settings = Settings.from_env(require_secrets=False)

    assert settings.image_api_key == "chat-test-key"
    assert settings.image_base_url == "https://chat.example/v1"


def test_image_credentials_can_be_independent_without_changing_chat(clean_settings_env):
    clean_settings_env.setenv("QQ_APP_ID", "test-qq-app")
    clean_settings_env.setenv("QQ_APP_SECRET", "test-qq-secret")
    clean_settings_env.setenv("OPENAI_API_KEY", "chat-test-key")
    clean_settings_env.setenv("OPENAI_BASE_URL", "https://chat.example/v1")
    clean_settings_env.setenv("OPENAI_MODEL", "gemini-3.8-flash")
    clean_settings_env.setenv("IMAGE_API_KEY", " image-test-key ")
    clean_settings_env.setenv("IMAGE_BASE_URL", " https://images.example/v1/// ")

    settings = Settings.from_env()

    assert settings.image_api_key == "image-test-key"
    assert settings.image_base_url == "https://images.example/v1"
    assert settings.image_model == SUNBURST
    assert settings.openai_api_key == "chat-test-key"
    assert settings.openai_base_url == "https://chat.example/v1"
    assert settings.openai_model == "gemini-3.8-flash"


def test_explicit_image_model_override_remains_supported(clean_settings_env):
    clean_settings_env.setenv("IMAGE_MODEL", " gemini-3.1-flash-image ")

    settings = Settings.from_env(require_secrets=False)

    assert settings.image_model == "gemini-3.1-flash-image"
    assert settings.openai_model == "gemini-3.8-flash"


@pytest.mark.parametrize("invalid_url", ["ftp://images.example/v1", "images.example/v1"])
def test_image_base_url_requires_http_or_https(clean_settings_env, invalid_url):
    clean_settings_env.setenv("IMAGE_BASE_URL", invalid_url)

    with pytest.raises(ValueError, match="IMAGE_BASE_URL"):
        Settings.from_env(require_secrets=False)


def test_image_credential_does_not_replace_required_chat_credential(clean_settings_env):
    clean_settings_env.setenv("QQ_APP_ID", "test-qq-app")
    clean_settings_env.setenv("QQ_APP_SECRET", "test-qq-secret")
    clean_settings_env.setenv("IMAGE_API_KEY", "image-test-key")

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        Settings.from_env()


@pytest.mark.asyncio
async def test_main_wires_independent_image_credentials_without_changing_llm(
    clean_settings_env, monkeypatch,
):
    clean_settings_env.setenv("OPENAI_API_KEY", "chat-test-key")
    clean_settings_env.setenv("OPENAI_BASE_URL", "https://chat.example/v1")
    clean_settings_env.setenv("OPENAI_MODEL", "gemini-3.8-flash")
    clean_settings_env.setenv("IMAGE_API_KEY", "image-test-key")
    clean_settings_env.setenv("IMAGE_BASE_URL", "https://images.example/v1")
    settings = Settings.from_env(require_secrets=False)
    monkeypatch.setattr(main_module, "Settings", SimpleNamespace(from_env=lambda: settings))
    created = {}
    memory = SimpleNamespace(interrupt_running_image_jobs=lambda: 0, close=lambda: None)
    llm = SimpleNamespace(close=AsyncMock())
    service = SimpleNamespace(close=AsyncMock())
    generator = SimpleNamespace(close=AsyncMock())
    runner = SimpleNamespace(run=AsyncMock())

    def make_llm(**kwargs):
        created["llm"] = kwargs
        return llm

    def make_generator(**kwargs):
        created["generator"] = kwargs
        return generator

    def make_runner(**kwargs):
        created["runner"] = kwargs
        return runner

    monkeypatch.setattr(main_module, "MemoryStore", lambda path: memory)
    monkeypatch.setattr(main_module, "LLMClient", make_llm)
    monkeypatch.setattr(main_module, "ChatService", lambda *args, **kwargs: service)
    monkeypatch.setattr(main_module, "ImageGenerator", make_generator)
    monkeypatch.setattr(main_module, "QQBotRunner", make_runner)

    await main_module.main()

    assert created["llm"]["api_key"] == "chat-test-key"
    assert created["llm"]["base_url"] == "https://chat.example/v1"
    assert created["llm"]["model"] == "gemini-3.8-flash"
    assert created["generator"]["api_key"] == "image-test-key"
    assert created["generator"]["base_url"] == "https://images.example/v1"
    assert created["generator"]["model"] == SUNBURST
    assert created["runner"]["service"] is service
    assert created["runner"]["image_generator"] is generator
    runner.run.assert_awaited_once()
    llm.close.assert_awaited_once()
    service.close.assert_awaited_once()
    generator.close.assert_awaited_once()


def _png_bytes(size: tuple[int, int]) -> bytes:
    raw = io.BytesIO()
    Image.new("RGB", size, "#887766").save(raw, format="PNG")
    return raw.getvalue()


def _image_response(size: tuple[int, int]) -> httpx.Response:
    return httpx.Response(200, json={
        "created": 1,
        "data": [{"b64_json": base64.b64encode(_png_bytes(size)).decode("ascii")}],
    })


def _generator_with_transport(monkeypatch, tmp_path, handler, references=()):
    # Use the actual SDK serialization/parsing, but route every request into MockTransport.
    def make_client(**kwargs):
        kwargs["http_client"] = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        return AsyncOpenAI(**kwargs)

    monkeypatch.setattr(image_gen_module, "AsyncOpenAI", make_client)
    return ImageGenerator(
        api_key="image-test-key",
        base_url="https://images.example/v1",
        model=SUNBURST,
        output_dir=tmp_path / "generated",
        sandrone_reference_paths=references,
    )


def _multipart_parts(request):
    content_type = request.headers["content-type"]
    assert content_type.startswith("multipart/form-data;")
    message = BytesParser(policy=policy.default).parsebytes(
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("ascii")
        + request.content
    )
    fields = {}
    files = []
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        payload = part.get_payload(decode=True)
        if part.get_filename() is not None:
            files.append((name, part.get_filename(), payload))
        else:
            fields[name] = payload.decode("utf-8")
    return fields, files


@pytest.mark.asyncio
async def test_sunburst_plain_generation_uses_images_generation_endpoint(monkeypatch, tmp_path):
    requests = []

    def handler(request):
        requests.append(request)
        assert request.method == "POST"
        assert request.url.path == "/v1/images/generations"
        body = json.loads(request.content)
        assert body["model"] == SUNBURST
        assert body["size"] == "1536x1024"
        assert body["quality"] == "medium"
        assert body["output_format"] == "jpeg"
        assert body["output_compression"] == 90
        assert body["n"] == 1
        assert "红色苹果" in body["prompt"]
        return _image_response((1536, 1024))

    generator = _generator_with_transport(monkeypatch, tmp_path, handler)
    try:
        result = await generator.generate("画一颗红色苹果")
    finally:
        await generator.close()

    assert len(requests) == 1
    assert result.reference_paths == ()
    with Image.open(result.path) as output:
        assert output.size == (1920, 1080)
        assert output.format == "JPEG"


@pytest.mark.asyncio
async def test_sunburst_sandrone_generation_uploads_references_to_images_edits(
    monkeypatch, tmp_path,
):
    references = (tmp_path / "face.png", tmp_path / "outfit.png")
    for reference in references:
        reference.write_bytes(_png_bytes((32, 32)))
    requests = []

    def handler(request):
        requests.append(request)
        assert request.method == "POST"
        assert request.url.path == "/v1/images/edits"
        fields, files = _multipart_parts(request)
        assert fields["model"] == SUNBURST
        assert fields["size"] == "1024x1536"
        assert fields["quality"] == "medium"
        assert fields["output_format"] == "jpeg"
        assert fields["output_compression"] == "90"
        assert fields["n"] == "1"
        assert "唯一的身份与造型基准" in fields["prompt"]
        assert [name for name, _, _ in files] == ["image[]", "image[]"]
        assert [filename for _, filename, _ in files] == [path.name for path in references]
        assert [payload for _, _, payload in files] == [path.read_bytes() for path in references]
        return _image_response((1024, 1536))

    generator = _generator_with_transport(monkeypatch, tmp_path, handler, references)
    try:
        result = await generator.generate("桑多涅单人立绘")
    finally:
        await generator.close()

    assert len(requests) == 1
    assert result.reference_paths == tuple(path.resolve() for path in references)
    with Image.open(result.path) as output:
        assert output.size == (1080, 1620)


@pytest.mark.asyncio
async def test_sunburst_original_edit_uploads_actual_pixels_and_preserves_size(
    monkeypatch, tmp_path,
):
    original = tmp_path / "source.png"
    original.write_bytes(_png_bytes((864, 1920)))
    requests = []

    def handler(request):
        requests.append(request)
        assert request.method == "POST"
        assert request.url.path == "/v1/images/edits"
        fields, files = _multipart_parts(request)
        assert fields["model"] == SUNBURST
        assert fields["size"] == "864x1920"
        assert fields["quality"] == "high"
        assert fields["output_format"] == "png"
        assert fields["n"] == "1"
        assert "output_compression" not in fields
        assert "65432" in fields["prompt"]
        assert "桑多涅唯一" not in fields["prompt"]
        assert files == [("image[]", original.name, original.read_bytes())]
        return _image_response((864, 1920))

    generator = _generator_with_transport(monkeypatch, tmp_path, handler)
    try:
        result = await generator.edit("把原石数量改成65432", (original,))
    finally:
        await generator.close()

    assert len(requests) == 1
    assert result.layout == "edit"
    assert not result.identity_sensitive
    with Image.open(result.path) as output:
        assert output.size == (864, 1920)
        assert output.format == "PNG"
