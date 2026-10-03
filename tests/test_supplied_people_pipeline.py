import shutil
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from test_character_refs import pixels
from test_qq_adapter import StubImageService, _runner_for_image_test

from talk_bot.character_refs import CharacterReferenceLibrary, CharacterSubject
from talk_bot.image_gen import GeneratedImage
from talk_bot.memory import MemoryStore
from talk_bot.service import ImageInspection, IncomingMessage


@pytest.mark.asyncio
@pytest.mark.parametrize("count,duplicate", [(1, False), (2, False), (2, True)])
async def test_supplied_people_new_scene_uses_bound_refs_not_source_edit(tmp_path, count, duplicate):
    store = MemoryStore(tmp_path / "memory.db")
    service = StubImageService(store)
    service.llm = SimpleNamespace(_download_image_as_data_url=AsyncMock(),
        describe_images=AsyncMock(return_value='{"usable":true,"reason":"清晰单人图"}'))
    subjects = tuple(CharacterSubject(f"人物{i+1}","用户本人" if i==0 else "本次原创",(),
                                     "private" if i==0 else "original",i+1) for i in range(count))
    service.image_character_names = AsyncMock(return_value=subjects)
    service.prepare_image_prompt = AsyncMock(return_value="人物1和人物2在海边")
    service.inspect_generated_image = AsyncMock(return_value=ImageInspection(True,"逐人符合参考"))
    runner = _runner_for_image_test(service)
    source_paths = []
    for i in range(count):
        path = tmp_path / f"original-{i}.png"
        path.write_bytes(pixels())
        source_paths.append(path)
    runner.image_sources = SimpleNamespace(fetch=AsyncMock(return_value=tuple(source_paths[:1] if duplicate else source_paths)),store=lambda *args: None)
    runner.character_library = CharacterReferenceLibrary(tmp_path / "references",model=service.llm)
    reference_paths = []

    async def generate(prompt, **kwargs):
        refs = kwargs["character_references"]
        assert [ref.name for ref in refs] == [subject.name for subject in subjects]
        assert all(ref.path.is_file() and ref.path.parent != runner.character_library.directory for ref in refs)
        reference_paths.extend(ref.path for ref in refs)
        output = tmp_path / "output.png"
        shutil.copyfile(source_paths[0],output)
        return GeneratedImage(output,"landscape",tuple(ref.path for ref in refs),character_references=refs)

    runner.image_generator = SimpleNamespace(generate=AsyncMock(side_effect=generate),edit=AsyncMock())
    message = IncomingMessage("m","group","g","u","以图1为人物参考，画人物1在海边玩耍",
                              image_urls=tuple(f"image{i}" for i in range(count)))
    event = SimpleNamespace(chat_scope="group",chat_id="g",message_id="m")
    try:
        await runner._handle_image_request(event,message,message.content)
        runner.image_generator.edit.assert_not_awaited()
        if duplicate:
            runner.image_generator.generate.assert_not_awaited()
            runner.media_uploader.upload.assert_not_awaited()
            assert store.latest_image_job("group:g").status == "failed"
            assert store.claim_rate_limit("image:global", 180)[0]
            assert "参考图编号无法对齐" in service.remembered[-1]
            return
        runner.image_generator.generate.assert_awaited_once()
        runner.media_uploader.upload.assert_awaited_once()
        assert len(service.inspect_generated_image.await_args.kwargs["character_references"]) == count
        assert not any(path.exists() for path in reference_paths)
        assert not list(runner.character_library.directory.glob("*.jpg"))
        assert store.latest_image_job("group:g").status == "sent"
    finally:
        await runner.character_library.close()
        store.close()
