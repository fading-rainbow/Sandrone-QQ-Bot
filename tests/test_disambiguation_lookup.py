import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from talk_bot.character_refs import CharacterSubject
from talk_bot.person_lookup import WikipediaPersonLookup


@pytest.mark.asyncio
@pytest.mark.parametrize("hint", ["龙珠", ""])
async def test_disambiguation_uses_actual_article_links_only_with_identity_context(hint):
    calls = []
    async def download(url, limit, *, params):
        calls.append(params)
        if params["prop"] == "links":
            assert params["pageids"] == "42"
            assert params["pllimit"] == 12
            return json.dumps({"query":{"pages":[{"links":[{"ns":0,"title":"Goku"}]}]}}).encode()
        if params["titles"] == "Goku":
            page = {"title":"Goku","extract":"Dragon Ball main character",
                    "pageprops":{"wikibase_item":"Q2142"},
                    "thumbnail":{"source":"https://upload.wikimedia.org/goku.png","width":200,"height":300}}
        else:
            page = {"pageid":42,"title":"Son Goku","pageprops":{"disambiguation":""}}
        return json.dumps({"query":{"pages":[page]}}).encode()
    model = SimpleNamespace(reply=AsyncMock(return_value='{"index":0,"exact_identity":true,"ambiguous":false}'))
    lookup = WikipediaPersonLookup(download,model)
    subject = CharacterSubject("孙悟空",hint,("Son Goku",),"fictional")
    if not hint:
        with pytest.raises(ValueError,match="explicit work"):
            await lookup.find(subject)
        assert len(calls) == 1
        model.reply.assert_not_awaited()
    else:
        row = await lookup.find(subject)
        assert row["title"] == "Goku"
        assert len(calls) == 6
        payload = json.loads(model.reply.await_args.kwargs["messages"][0].content)
        assert payload["aliases"] == ["Son Goku"]
        assert payload["candidates"][0]["index"] == 0
