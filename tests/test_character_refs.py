import base64
import io
import json
import os
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from PIL import Image

from talk_bot.character_refs import (
    CharacterReference,
    CharacterReferenceLibrary,
    CharacterReferenceUnavailable,
)
from talk_bot.config import DEFAULT_PROMPT
from talk_bot.image_gen import GeneratedImage, ImageGenerator
from talk_bot.memory import MemoryStore
from talk_bot.service import ChatService, ImageInspection, IncomingMessage


def pixels(color="red"):
    data = io.BytesIO()
    Image.new("RGBA", (64, 128), color).save(data, format="PNG")
    return data.getvalue()


def official_transport(*, missing=False, ambiguous=False, bad_url=False, oversize=False):
    requests = []

    def handler(request):
        requests.append(request)
        if request.url.path.endswith("/search"):
            assert request.url.params["keyword"] == "沃雅妮莎"
            row = {"entry_page_id": "11702", "name": "沃雅妮莎",
                   "menu": {"id": "1", "sub_menus": [{"id": "2"}]}}
            rows = [] if missing else [row, row] if ambiguous else [row]
            return httpx.Response(200, json={"retcode": 0, "data": {"list": rows}})
        if request.url.path.endswith("/entry_page"):
            assert request.url.params["entry_page_id"] == "11702"
            return httpx.Response(200, json={"retcode": 0, "data": {"page": {
                "name": "沃雅妮莎", "menu_id": "2", "desc": "<p>离群的水妖</p>",
                "header_img_url": "https://127.0.0.1/private.png" if bad_url else
                    "https://act-webstatic.hoyoverse.com/splash.png",
                "icon_url": "https://act-webstatic.hoyoverse.com/icon.png",
            }}})
        assert request.url.host == "act-webstatic.hoyoverse.com"
        return httpx.Response(200, content=pixels(),
                              headers={"content-length": str(5 * 1024 * 1024)} if oversize else {})

    return httpx.MockTransport(handler), requests


@pytest.mark.asyncio
async def test_official_reference_download_cache_and_revalidation(tmp_path):
    transport, requests = official_transport()
    client = httpx.AsyncClient(transport=transport)
    library = CharacterReferenceLibrary(tmp_path / "cache", client=client)
    try:
        refs = await library.resolve(("沃雅妮莎",))
        assert len(requests) == 4
        assert refs[0].facts == "离群的水妖"
        assert "11702" in refs[0].source
        with Image.open(refs[0].path) as image:
            assert image.size == (1536, 1024)
            assert image.mode == "RGB"
        assert (await library.resolve(("沃雅妮莎",)))[0] == refs[0]
        assert len(requests) == 4
        meta = refs[0].path.with_suffix(".json")
        data = json.loads(meta.read_text())
        data["fetched_at"] = time.time() - library.TTL - 1
        meta.write_text(json.dumps(data))
        await library.resolve(("沃雅妮莎",))
        assert len(requests) == 8
    finally:
        await library.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["missing", "ambiguous", "bad_url", "oversize"])
async def test_unreliable_reference_fails_closed(tmp_path, failure):
    transport, requests = official_transport(**{failure: True})
    library = CharacterReferenceLibrary(tmp_path, client=httpx.AsyncClient(transport=transport))
    try:
        with pytest.raises(CharacterReferenceUnavailable, match="沃雅妮莎.*参考"):
            await library.resolve(("沃雅妮莎",))
        assert not list(tmp_path.glob("*.jpg"))
        assert all(request.url.host != "127.0.0.1" for request in requests)
    finally:
        await library.close()


@pytest.mark.asyncio
async def test_redirect_is_not_followed_and_no_guessed_entry_id(tmp_path):
    def handler(request):
        assert request.url.path.endswith("/search")
        return httpx.Response(302, headers={"location": "http://127.0.0.1/private"})
    library = CharacterReferenceLibrary(tmp_path, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    try:
        with pytest.raises(CharacterReferenceUnavailable):
            await library.resolve(("沃雅妮莎",))
    finally:
        await library.close()


@pytest.mark.asyncio
async def test_sandrone_local_archive_deduplicated_and_limit(tmp_path):
    path = tmp_path / "local.png"
    path.write_bytes(pixels())
    transport, requests = official_transport()
    library = CharacterReferenceLibrary(tmp_path / "cache", (path,),
                                        client=httpx.AsyncClient(transport=transport))
    try:
        refs = await library.resolve(("Sandrone", "桑多涅", "沃雅妮莎"))
        assert [ref.name for ref in refs] == ["桑多涅", "沃雅妮莎"]
        assert len(requests) == 4
        with pytest.raises(CharacterReferenceUnavailable, match="三名"):
            await library.resolve(("a", "b", "c", "d"))
    finally:
        await library.close()


def test_cache_bound_cleanup_protected_and_orphans(tmp_path):
    library = CharacterReferenceLibrary(tmp_path)
    library.MAX_ENTRIES = 2
    for i in range(4):
        key = library._key(str(i))
        (tmp_path / (key + ".json")).write_text("{}")
        (tmp_path / (key + ".jpg")).write_bytes(pixels())
        os.utime(tmp_path / (key + ".json"), (time.time() - i, time.time() - i))
    orphan = tmp_path / ("f" * 64 + ".jpg")
    orphan.write_bytes(pixels())
    unrelated = tmp_path / "do-not-delete.jpg"
    unrelated.write_bytes(pixels())
    library.cleanup(frozenset({library._key("3")}))
    assert len(list(tmp_path.glob("*.json"))) == 2
    assert (tmp_path / (library._key("3") + ".jpg")).exists()
    assert not orphan.exists()
    assert unrelated.exists()


@pytest.mark.asyncio
async def test_three_characters_fit_vision_limit_and_generic_images_stay_generic(tmp_path):
    refs = references(tmp_path, count=3)
    data = verdict()
    data["characters"].append({"name": "阿蕾奇诺", "match": True, "reason": "发型和服装一致"})
    service = service_with(tmp_path, reply='{"characters":[],"unresolved":false}', vision=json.dumps(data))
    candidate = tmp_path / "candidate.png"
    candidate.write_bytes(pixels())
    assert (await service.inspect_generated_image(candidate, "三人茶会", character_references=refs)).accepted
    assert len(service.llm.describe_images.await_args.kwargs["image_urls"]) == 4
    message = IncomingMessage("m", "group", "g", "u", "画桌上的苹果")
    assert await service.image_character_names(message, message.content) == ()
    service.llm.describe_images.reset_mock()
    service.llm.reply.return_value = "桌面苹果静物，阳光照明。"
    await service.prepare_image_prompt(message, message.content, character_references=())
    service.llm.describe_images.assert_not_awaited()
    assert "不增加既有作品角色" in service.llm.reply.await_args.kwargs["instructions"]
    service.memory.close()


def service_with(tmp_path, reply="", vision=""):
    llm = SimpleNamespace(reply=AsyncMock(return_value=reply),
                          describe_images=AsyncMock(return_value=vision))
    memory = MemoryStore(tmp_path / "memory.db")
    return ChatService(memory, llm, system_prompt=DEFAULT_PROMPT, history_messages=10)


def references(tmp_path, count=2):
    result = []
    for i, name in enumerate(("桑多涅", "沃雅妮莎", "阿蕾奇诺")[:count]):
        path = tmp_path / f"ref-{i}.png"
        path.write_bytes(pixels())
        result.append(CharacterReference(name, path, "https://wiki.hoyolab.com/official", "可靠资料"))
    return tuple(result)


@pytest.mark.asyncio
async def test_new_names_extracted_without_appearance_and_context_pronouns(tmp_path):
    service = service_with(tmp_path, reply='{"characters":["桑多涅","沃雅妮莎"],"unresolved":false}')
    service.memory.append("group:g", "u", "assistant", "她指沃雅妮莎。旧图错误地画了双角。")
    message = IncomingMessage("m", "group", "g", "u", "画你和她")
    assert await service.image_character_names(message, message.content) == ("桑多涅", "沃雅妮莎")
    kwargs = service.llm.reply.await_args.kwargs
    assert "不需要预先知道人物是谁" in kwargs["instructions"]
    assert "她指沃雅妮莎" in kwargs["messages"][0].content
    service.memory.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", ['{}', 'PASS', '{"characters":[],"unresolved":true}',
                                   '{"characters":[7],"unresolved":false}'])
async def test_entity_failure_never_silently_becomes_generic_image(tmp_path, reply):
    service = service_with(tmp_path, reply=reply)
    with pytest.raises(CharacterReferenceUnavailable):
        await service.image_character_names(IncomingMessage("m", "group", "g", "u", "画她"), "画她")
    service.memory.close()


@pytest.mark.asyncio
async def test_grounded_planning_uses_pixels_not_static_visual_index(tmp_path):
    refs = references(tmp_path)
    service = service_with(tmp_path, vision="桑多涅和沃雅妮莎在海边玩水，外观严格对照参考。")
    msg = IncomingMessage("m", "group", "g", "u", "画桑多涅和沃雅妮莎")
    await service.prepare_image_prompt(msg, msg.content, character_references=refs)
    service.llm.reply.assert_not_awaited()
    kwargs = service.llm.describe_images.await_args.kwargs
    assert len(kwargs["image_urls"]) == 2
    assert "图片1只对应角色【桑多涅】" in kwargs["prompt"]
    assert "图片2只对应角色【沃雅妮莎】" in kwargs["prompt"]
    assert "旧构图描述不是官方外观" in kwargs["prompt"]
    service.llm.describe_images.return_value = "桑多涅独自散步"
    with pytest.raises(CharacterReferenceUnavailable, match="遗漏"):
        await service.prepare_image_prompt(msg, msg.content, character_references=refs)
    service.memory.close()


def verdict(*, second=True):
    return {"characters": [
        {"name": "桑多涅", "match": True, "reason": "发型头饰一致"},
        {"name": "沃雅妮莎", "match": second, "reason": "头部有错误双角" if not second else "长发花饰一致"},
    ], "composition_ok": True, "description": "两人在海边玩水"}


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["pass", "second_wrong", "missing", "duplicate", "string_bool", "invalid"])
async def test_every_character_must_pass_pixel_review(tmp_path, case):
    data = verdict(second=case != "second_wrong")
    if case == "missing":
        data["characters"].pop()
    elif case == "duplicate":
        data["characters"][1]["name"] = "桑多涅"
    elif case == "string_bool":
        data["characters"][1]["match"] = "true"
    service = service_with(tmp_path, vision="PASS" if case == "invalid" else json.dumps(data))
    refs = references(tmp_path)
    candidate = tmp_path / "candidate.png"
    candidate.write_bytes(pixels())
    check = await service.inspect_generated_image(candidate, "桑多涅和沃雅妮莎", character_references=refs)
    assert check.accepted == (case == "pass")
    kwargs = service.llm.describe_images.await_args.kwargs
    assert len(kwargs["image_urls"]) == 3
    assert "图片3只对应角色【沃雅妮莎】" in kwargs["prompt"]
    service.memory.close()


@pytest.mark.asyncio
async def test_sunburst_receives_all_actual_reference_pixels_and_retry_details(tmp_path):
    refs = references(tmp_path)
    images = SimpleNamespace(edit=AsyncMock(return_value=SimpleNamespace(
        data=[SimpleNamespace(b64_json=base64.b64encode(pixels()).decode())])), generate=AsyncMock())
    generator = object.__new__(ImageGenerator)
    generator.model = "gpt-image-2.5-sunburst"
    generator.output_dir = tmp_path
    generator.client = SimpleNamespace(images=images)
    generator.sandrone_reference_paths = ()
    generated = await generator.generate("桑多涅和沃雅妮莎在海边", character_references=refs,
                                          identity_retry=True, retry_feedback="沃雅妮莎有错误双角")
    kwargs = images.edit.await_args.kwargs
    assert len(kwargs["image"]) == 2
    assert "图片2只对应角色【沃雅妮莎】" in kwargs["prompt"]
    assert "沃雅妮莎有错误双角" in kwargs["prompt"]
    assert generated.character_references == refs
    assert generated.identity_sensitive
    images.generate.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["pass", "retry", "reject", "missing", "unavailable"])
async def test_qq_grounding_pipeline_binding_retry_cap_and_failure_release(tmp_path, outcome):
    # Exercise the actual QQ job handler without contacting QQ or production memory.
    from test_qq_adapter import StubImageService, _runner_for_image_test

    memory = MemoryStore(tmp_path / "memory.db")
    service = StubImageService(memory)
    refs = references(tmp_path)
    service.image_character_names = AsyncMock(return_value=("桑多涅", "沃雅妮莎"))
    service.prepare_image_prompt = AsyncMock(return_value="桑多涅和沃雅妮莎在海边")
    checks = [ImageInspection(False, "沃雅妮莎错误双角"), ImageInspection(outcome == "retry", "再次核对")]
    service.inspect_generated_image = AsyncMock(
        side_effect=[ImageInspection(True, "已核对两人")] if outcome == "pass" else checks)
    if outcome == "unavailable":
        service.inspect_generated_image.side_effect = RuntimeError("vision unavailable")
    runner = _runner_for_image_test(service)
    runner.character_library = SimpleNamespace(resolve=AsyncMock(return_value=refs))
    if outcome == "missing":
        runner.character_library.resolve.side_effect = CharacterReferenceUnavailable("沃雅妮莎参考缺失，请发参考图")

    async def generate(prompt, **kwargs):
        assert kwargs["character_references"] == refs
        if kwargs["identity_retry"]:
            assert "沃雅妮莎错误双角" in kwargs["retry_feedback"]
        path = tmp_path / "output.png"
        path.write_bytes(pixels())
        return GeneratedImage(path, "landscape", tuple(r.path for r in refs), character_references=refs)

    runner.image_generator = SimpleNamespace(generate=AsyncMock(side_effect=generate))
    message = IncomingMessage("m", "group", "g", "u", "画桑多涅和沃雅妮莎在海边玩耍")
    event = SimpleNamespace(chat_scope="group", chat_id="g", message_id="m")
    await runner._handle_image_request(event, message, message.content)
    count = {"pass": 1, "retry": 2, "reject": 2, "missing": 0, "unavailable": 1}[outcome]
    assert runner.image_generator.generate.await_count == count
    assert runner.media_uploader.upload.await_count == (1 if outcome in {"pass", "retry"} else 0)
    assert runner._image_job is None
    if count:
        assert service.inspect_generated_image.await_args.kwargs["character_references"] == refs
    if outcome not in {"pass", "retry"}:
        assert memory.claim_rate_limit("image:global", 180)[0]
    memory.close()
