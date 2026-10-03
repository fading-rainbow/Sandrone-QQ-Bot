import io
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from PIL import Image

from talk_bot.character_refs import (
    CharacterReferenceLibrary,
    CharacterReferenceUnavailable,
    CharacterSubject,
)
from talk_bot.config import DEFAULT_PROMPT
from talk_bot.memory import MemoryStore
from talk_bot.service import ChatService, IncomingMessage


def image_bytes():
    out = io.BytesIO()
    Image.new("RGB", (200, 400), "white").save(out, format="PNG")
    return out.getvalue()


def general_library(tmp_path, *, person="Naruto Uzumaki", ambiguous=False, usable=True, disambiguation=False,
                    bad_image=False, no_image=False, kind="fictional"):
    calls = []
    def handler(request):
        calls.append(request)
        assert "x-rpc-language" not in request.headers
        assert "User-Agent" in request.headers
        if request.url.path.endswith("/w/api.php"):
            assert request.url.params["titles"].startswith(person)
            assert request.url.params["prop"] == "pageimages|extracts|pageprops"
            page = {"title": person, "pageprops": {"wikibase_item": "Q931"},
                    "extract": "A named fictional character of the manga Naruto." if kind == "fictional"
                        else "A German-born theoretical physicist.",
                    "thumbnail": {"source": "https://127.0.0.1/private" if bad_image else
                                   "https://upload.wikimedia.org/wikipedia/portrait.png", "width": 200, "height": 400}}
            if no_image:
                page.pop("thumbnail")
            if disambiguation:
                page["pageprops"]["disambiguation"] = ""
            return httpx.Response(200, json={"query": {"pages": [page]}})
        assert request.url.host == "upload.wikimedia.org"
        return httpx.Response(200, content=image_bytes())
    model = SimpleNamespace(reply=AsyncMock(return_value=json.dumps({"index":0,"exact_identity":True,"ambiguous":ambiguous})),
                            describe_images=AsyncMock(return_value=json.dumps({"usable":usable,"reason":"单人参考"})))
    library = CharacterReferenceLibrary(tmp_path / "cache", model=model,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    return library, model, calls


@pytest.mark.asyncio
@pytest.mark.parametrize("name,hint,kind", [("Naruto Uzumaki","Naruto","fictional"),
                                           ("Albert Einstein","theoretical physicist","real")])
async def test_non_genshin_and_real_person_lookup_grounded_in_actual_articles(tmp_path, name, hint, kind):
    library, model, calls = general_library(tmp_path, person=name, kind=kind)
    subject = CharacterSubject(name,hint,(),kind)
    try:
        refs = await library.resolve((subject,))
        assert refs[0].name == name
        assert "wikipedia.org/wiki/" in refs[0].source
        assert refs[0].path.is_file()
        assert len(calls) == 3
        assert "candidates" in model.reply.await_args.kwargs["messages"][0].content
        assert model.describe_images.await_args.kwargs["image_urls"][0].startswith("data:image/jpeg;base64,")
        assert await library.resolve((subject,)) == refs
        assert len(calls) == 3
    finally:
        await library.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("problem", ["ambiguous", "disambiguation", "bad_image", "no_image", "wrong_pixels"])
async def test_general_lookup_never_uses_wrong_ambiguous_or_unusable_references(tmp_path, problem):
    options = {problem: True} if problem != "wrong_pixels" else {"usable":False}
    library, _model, calls = general_library(tmp_path, **options)
    try:
        with pytest.raises(CharacterReferenceUnavailable):
            await library.resolve((CharacterSubject("Naruto Uzumaki","Naruto"),))
        assert not list((tmp_path / "cache").glob("*.jpg"))
        assert all(call.url.host != "127.0.0.1" for call in calls)
    finally:
        await library.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["private", "original"])
async def test_private_and_original_identity_requires_supplied_pixels_without_public_lookup(tmp_path, kind):
    library, model, calls = general_library(tmp_path)
    try:
        with pytest.raises(CharacterReferenceUnavailable):
            await library.resolve((CharacterSubject("小明","",(),kind),))
        assert not calls
        model.reply.assert_not_awaited()
    finally:
        await library.close()


@pytest.mark.asyncio
async def test_user_reference_is_job_local_and_never_shared_via_person_cache(tmp_path):
    library, model, calls = general_library(tmp_path)
    source = tmp_path / "source.png"
    source.write_bytes(image_bytes())
    workspace = tmp_path / "job"
    workspace.mkdir()
    try:
        ref = await library.from_supplied(CharacterSubject("我","本次用户本人",(),"private",1), (source,), workspace)
        assert ref.path.parent == workspace
        assert "不跨群共享" in ref.source
        assert not list((tmp_path / "cache").glob("*.jpg"))
        assert not calls
        model.reply.assert_not_awaited()
        model.describe_images.assert_awaited_once()
        assert "不用模型知识库去验证姓名" in model.describe_images.await_args.kwargs["prompt"]
        model.describe_images.return_value = '{"usable":false,"reason":"不明群像"}'
        with pytest.raises(CharacterReferenceUnavailable, match="单人"):
            await library.from_supplied(CharacterSubject("我"), (source,), workspace)
    finally:
        await library.close()


def test_names_in_different_works_or_versions_do_not_share_cached_identity():
    assert CharacterSubject("Alice","作品A").cache_name != CharacterSubject("Alice","作品B").cache_name
    assert CharacterSubject("Alice","作品A 少年版").cache_name != CharacterSubject("Alice","作品A 成年版").cache_name
    assert CharacterSubject("Alice",kind="fictional").cache_name != CharacterSubject("Alice",kind="real").cache_name


@pytest.mark.asyncio
async def test_rich_entities_keep_real_private_original_and_attachment_bindings(tmp_path):
    rows = [
        {"name":"我","kind":"private","reference_index":1},
        {"name":"小明","kind":"original","identity_hint":"原创机械师","reference_index":2},
    ]
    llm = SimpleNamespace(reply=AsyncMock(return_value=json.dumps({"characters":rows,"unresolved":False})))
    memory = MemoryStore(tmp_path / "memory.db")
    service = ChatService(memory,llm,system_prompt=DEFAULT_PROMPT,history_messages=10)
    msg = IncomingMessage("m","group","g","u","图1我，图2原创小明，画两人在海边",image_urls=("img1","img2"))
    subjects = await service.image_character_names(msg,msg.content,detailed=True)
    assert subjects[0].kind == "private" and subjects[0].reference_index == 1
    assert subjects[1].kind == "original" and subjects[1].identity_hint == "原创机械师"
    memory.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("row", [{"name":"小明","reference_index":True},
                                  {"name":"小明","reference_index":2},
                                  {"name":"小明","aliases":["https://127.0.0.1"]},
                                  {"name":"小明","kind":"guess"}])
async def test_invalid_entity_metadata_fails_before_download_or_generation(tmp_path, row):
    memory = MemoryStore(tmp_path / "memory.db")
    llm = SimpleNamespace(reply=AsyncMock(return_value=json.dumps({"characters":[row],"unresolved":False})))
    service = ChatService(memory,llm,system_prompt=DEFAULT_PROMPT,history_messages=10)
    msg = IncomingMessage("m","group","g","u","画小明",image_urls=("image",))
    with pytest.raises(CharacterReferenceUnavailable):
        await service.image_character_names(msg,msg.content,detailed=True)
    memory.close()


@pytest.mark.asyncio
async def test_multipeople_without_explicit_photo_binding_does_not_guess(tmp_path):
    memory = MemoryStore(tmp_path / "memory.db")
    llm = SimpleNamespace(reply=AsyncMock(return_value='{"characters":["Alice","Bob"],"unresolved":false}'))
    service = ChatService(memory,llm,system_prompt=DEFAULT_PROMPT,history_messages=10)
    msg = IncomingMessage("m","group","g","u","画Alice和Bob",image_urls=("image1","image2"))
    with pytest.raises(CharacterReferenceUnavailable,match="图1是谁"):
        await service.image_character_names(msg,msg.content,detailed=True)
    memory.close()
