from datetime import UTC, datetime, timedelta
from pathlib import Path

from aiohttp import web
from aiohttp.test_utils import TestServer
import pytest

from gh_signals import github_client
from gh_signals.github_client import GitHubClient, GitHubError
from gh_signals.schemas import ContributorStats, ContributorWeek, Owner, Repo, StarWeek, User
from gh_signals.score import (
    MIN_COVERAGE,
    SCORE_COLUMNS,
    RepoUnavailableError,
    TrustResult,
    assign_tier,
    load_model,
    parse_repo_name,
    score_repo_async,
)

AS_OF = datetime(2026, 11, 1, tzinfo=UTC)


class FakeClient(GitHubClient):
    """Answers from fixed data with no network. Each ``fail_*`` flag makes that lookup be rejected."""

    def __init__(
        self,
        *,
        archived: bool = False,
        fail_repo: bool = False,
        fail_contributors: bool = False,
        fail_users: bool = False,
    ) -> None:
        super().__init__(token="test", cache_dir=None)
        self.archived = archived
        self.fail_repo = fail_repo
        self.fail_contributors = fail_contributors
        self.fail_users = fail_users
        self.calls: list[str] = []

    async def get_repo(self, repo: str) -> Repo:
        self.calls.append("repo")
        if self.fail_repo:
            raise GitHubError(404, f"repos/{repo}", "Not Found")
        return Repo(
            full_name="Org/Name",
            owner=Owner(login="Org", type="Organization"),
            stargazers_count=500,
            forks_count=80,
            subscribers_count=20,
            open_issues_count=15,
            size=2000,
            created_at=AS_OF - timedelta(days=1500),
            pushed_at=AS_OF - timedelta(days=2),
            archived=self.archived,
            disabled=False,
            fork=False,
            has_issues=True,
        )

    async def get_contributor_stats(
        self, repo: str, max_attempts: int = 6, retry_delay_seconds: float = 2.0
    ) -> list[ContributorStats]:
        self.calls.append("contributors")
        if self.fail_contributors:
            raise GitHubError(403, f"repos/{repo}/stats/contributors", "Forbidden")
        week = ContributorWeek(w=AS_OF - timedelta(weeks=1), c=3)
        return [ContributorStats(total=10, weeks=[week], author=Owner(login=f"dev{i}", type="User")) for i in range(3)]

    async def get_user(self, username: str) -> User:
        self.calls.append("user")
        if self.fail_users:
            raise GitHubError(404, f"users/{username}", "Not Found")
        return User(login=username, type="User", created_at=AS_OF, public_repos=12, followers=40, following=3)

    async def get_star_history(self, repo: str) -> list[StarWeek]:
        self.calls.append("star_history")
        return []


async def test_complete_data_gives_a_full_result() -> None:
    result = await score_repo_async("org/name", client=FakeClient(), as_of=AS_OF)

    assert result.repo == "Org/Name"
    assert 0 <= result.score <= 100
    assert set(result.model_scores) == {"lr", "rf"}
    assert list(result.signals) == SCORE_COLUMNS
    assert result.missing_signals == []
    assert result.coverage == 1
    assert result.signals["owner_followers"] == 40
    assert result.signals["owner_public_repos"] == 12
    assert result.signals["is_archived"] is False
    assert result.as_of == AS_OF
    assert result.model_version == load_model().version


async def test_score_is_the_mean_of_the_model_scores() -> None:
    result = await score_repo_async("org/name", client=FakeClient(), as_of=AS_OF)

    assert result.score == pytest.approx(sum(result.model_scores.values()) / 2, abs=0.1)


async def test_scoring_skips_the_star_history_lookup() -> None:
    client = FakeClient()

    await score_repo_async("org/name", client=client, as_of=AS_OF)

    assert "star_history" not in client.calls


async def test_failed_contributor_stats_still_return_a_result() -> None:
    result = await score_repo_async("org/name", client=FakeClient(fail_contributors=True), as_of=AS_OF)

    assert {"contributor_activity_ratio", "ghost_contributor_ratio"} <= set(result.missing_signals)
    assert result.coverage < 1
    assert any("Imputed" in warning for warning in result.warnings)
    assert 0 <= result.score <= 100


async def test_failed_user_lookups_leave_owner_signals_missing() -> None:
    result = await score_repo_async("org/name", client=FakeClient(fail_users=True), as_of=AS_OF)

    assert {"owner_followers", "owner_public_repos", "ghost_contributor_ratio"} <= set(result.missing_signals)
    assert result.signals["owner_followers"] is None


async def test_too_few_signals_force_the_uncertain_tier() -> None:
    client = FakeClient(fail_contributors=True, fail_users=True)

    result = await score_repo_async("org/name", client=client, as_of=AS_OF)

    assert result.coverage < MIN_COVERAGE
    assert result.tier == "uncertain"
    assert any("tier is uncertain" in warning for warning in result.warnings)


async def test_archived_repo_is_flagged() -> None:
    result = await score_repo_async("org/name", client=FakeClient(archived=True), as_of=AS_OF)

    assert result.signals["is_archived"] is True
    assert "The repository is archived and read-only." in result.warnings


async def test_unavailable_repo_raises() -> None:
    with pytest.raises(RepoUnavailableError, match="404"):
        await score_repo_async("org/name", client=FakeClient(fail_repo=True), as_of=AS_OF)


async def test_result_round_trips_through_json() -> None:
    result = await score_repo_async("org/name", client=FakeClient(), as_of=AS_OF)

    assert TrustResult.model_validate_json(result.model_dump_json()) == result


def test_model_is_trained_once_per_process() -> None:
    assert load_model() is load_model()


@pytest.mark.parametrize(
    "text",
    [
        "microsoft/vscode",
        " microsoft/vscode ",
        "https://github.com/microsoft/vscode",
        "https://www.github.com/microsoft/vscode/",
        "github.com/microsoft/vscode.git",
    ],
)
def test_parse_repo_name_accepts_names_and_urls(text: str) -> None:
    assert parse_repo_name(text) == "microsoft/vscode"


@pytest.mark.parametrize("text", ["", "vscode", "microsoft/vscode/issues", "https://gitlab.com/a/b", "a b/c"])
def test_parse_repo_name_rejects_anything_else(text: str) -> None:
    with pytest.raises(ValueError, match="owner/name"):
        parse_repo_name(text)


@pytest.mark.parametrize(
    ("score", "coverage", "model_scores", "expected"),
    [
        (90, 1.0, {"lr": 90, "rf": 90}, "trusted"),
        (50, 1.0, {"lr": 50, "rf": 50}, "uncertain"),
        (10, 1.0, {"lr": 10, "rf": 10}, "low_confidence"),
        (90, MIN_COVERAGE - 0.01, {"lr": 90, "rf": 90}, "uncertain"),
        (10, MIN_COVERAGE - 0.01, {"lr": 10, "rf": 10}, "uncertain"),
        (75, 1.0, {"lr": 95, "rf": 55}, "uncertain"),
    ],
)
def test_assign_tier_prefers_uncertain_when_evidence_is_thin_or_conflicting(
    score: float, coverage: float, model_scores: dict[str, float], expected: str
) -> None:
    assert assign_tier(score, coverage, model_scores) == expected


async def test_client_disk_cache_prevents_repeat_requests(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    hits: list[str] = []

    async def handler(request: web.Request) -> web.Response:
        hits.append(request.path)
        return web.json_response({"login": "someone"})

    app = web.Application()
    app.router.add_get("/{tail:.*}", handler)
    server = TestServer(app)
    await server.start_server()
    monkeypatch.setattr(github_client, "API_URL", str(server.make_url("")).rstrip("/"))
    try:
        for _ in range(2):
            async with GitHubClient(token="test", cache_dir=tmp_path) as client:
                await client.get_repo_raw("owner/repo")
    finally:
        await server.close()

    assert hits == ["/repos/owner/repo"]
