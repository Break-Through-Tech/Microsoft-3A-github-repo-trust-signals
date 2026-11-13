from collections.abc import AsyncIterator

from aiohttp import web
from aiohttp.test_utils import TestServer
import pytest

from gh_signals import github_client
from gh_signals.github_client import GitHubClient


@pytest.fixture
async def stats_server(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[list[int]]:
    """A fake GitHub whose contributor stats answer 202 (still computing) once, then 200. Yields the request count."""
    calls: list[int] = []

    async def handler(request: web.Request) -> web.Response:
        calls.append(1)
        if len(calls) == 1:
            return web.Response(status=202, text="", content_type="application/json")
        return web.json_response([{"total": 3, "weeks": [], "author": {"login": "a", "type": "User"}}])

    app = web.Application()
    app.router.add_get("/{tail:.*}", handler)
    server = TestServer(app)
    await server.start_server()
    monkeypatch.setattr(github_client, "API_URL", str(server.make_url("")).rstrip("/"))
    yield calls
    await server.close()


async def test_contributor_stats_polls_past_a_202_instead_of_failing(stats_server: list[int]) -> None:
    async with GitHubClient(token="test", cache_dir=None) as client:
        stats = await client.get_contributor_stats("owner/repo", retry_delay_seconds=0)

    assert [contributor.total for contributor in stats] == [3]
    assert len(stats_server) == 2
