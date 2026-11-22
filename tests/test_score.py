import asyncio
from datetime import UTC, datetime, timedelta
import json
import math
from pathlib import Path

from fastapi.testclient import TestClient
import pandas as pd
import pytest

from gh_signals.app import app
from gh_signals.build_dataset import FEATURES_PATH, MANIFEST_PATH
from gh_signals.feedback import record_feedback
from gh_signals.github_client import GitHubClient, GitHubError
from gh_signals.schemas import License, Owner, Repo, StarWeek, User
from gh_signals.score import (
    FEATURE_NAMES,
    MODEL_MISSING_MESSAGE,
    STAR_FEATURE_NAMES,
    TrainedModel,
    apply_star_count_floor,
    build_score,
    describe_usage,
    format_score,
    live_features,
    load_model,
    main,
    parse_repo_url,
    predict_probability,
    score_repository,
)

AS_OF = datetime(2026, 11, 8, tzinfo=UTC)


def test_parse_repo_url_accepts_common_forms() -> None:
    cases = {
        "https://github.com/microsoft/vscode": "microsoft/vscode",
        "https://github.com/microsoft/vscode.git": "microsoft/vscode",
        "https://github.com/microsoft/vscode/tree/main": "microsoft/vscode",
        "https://github.com/microsoft/vscode?tab=readme": "microsoft/vscode",
        "http://www.github.com/Microsoft/VSCode/": "Microsoft/VSCode",
        "github.com/microsoft/vscode": "microsoft/vscode",
        "microsoft/vscode": "microsoft/vscode",
    }
    for text, expected in cases.items():
        assert parse_repo_url(text) == expected


def test_parse_repo_url_rejects_other_hosts_and_incomplete_paths() -> None:
    with pytest.raises(ValueError, match=r"github\.com"):
        parse_repo_url("https://example.com/acme/widget")
    with pytest.raises(ValueError, match="owner and a repository"):
        parse_repo_url("https://github.com/microsoft")
    with pytest.raises(ValueError, match="Enter a GitHub"):
        parse_repo_url("  ")
    with pytest.raises(ValueError, match="characters GitHub does not allow"):
        parse_repo_url("https://github.com/owner/bad name")


def test_young_or_small_repositories_withhold_the_numeric_score() -> None:
    young = build_score(
        _flat_model(), full_name="example/new", features=_features(age_days=12), fetched_at=AS_OF, warnings=[]
    )
    small = build_score(
        _flat_model(), full_name="example/small", features=_features(stargazers_count=10), fetched_at=AS_OF, warnings=[]
    )

    assert young.score is None
    assert young.tier == "limited_evidence"
    assert "under 90 days" in young.summary
    assert small.score is None
    assert "fewer than 50 stars" in small.summary
    assert young.signals


def test_high_score_on_an_established_repository_keeps_the_number() -> None:
    names = list(FEATURE_NAMES)
    coefficients = [0.0] * len(names)
    coefficients[names.index("fork_star_ratio")] = 2.0
    result = build_score(
        _flat_model(coefficients=coefficients),
        full_name="example/project",
        features=_features(fork_star_ratio=3.0),
        fetched_at=AS_OF,
        warnings=[],
    )
    fork_row = next(row for row in result.signals if row.name == "fork_star_ratio")

    assert result.score == 100
    assert result.tier == "established"
    assert fork_row.direction == "up"
    assert fork_row.available is True


def test_low_score_on_an_old_repository_stays_limited_evidence() -> None:
    names = list(FEATURE_NAMES)
    coefficients = [0.0] * len(names)
    coefficients[names.index("stars_per_day")] = -3.0
    result = build_score(
        _flat_model(coefficients=coefficients),
        full_name="example/project",
        features=_features(stars_per_day=4.0),
        fetched_at=AS_OF,
        warnings=[],
    )

    assert result.score is not None
    assert result.score < 40
    assert result.tier == "limited_evidence"
    assert "not a verdict" in result.summary


def test_missing_star_timing_is_imputed_and_marked_unavailable() -> None:
    names = list(FEATURE_NAMES)
    medians = [0.0] * len(names)
    medians[names.index("burstiness")] = 0.2
    coefficients = [0.0] * len(names)
    coefficients[names.index("burstiness")] = 5.0
    features = _features(**dict.fromkeys(STAR_FEATURE_NAMES, None))
    probability, contributions = predict_probability(
        _flat_model(medians=medians, coefficients=coefficients),
        features,
    )
    result = build_score(
        _flat_model(medians=medians, coefficients=coefficients),
        full_name="example/project",
        features=features,
        fetched_at=AS_OF,
        warnings=[],
    )
    burst = next(row for row in result.signals if row.name == "burstiness")

    assert probability == pytest.approx(1 / (1 + math.exp(-1.0)))
    assert contributions[names.index("burstiness")] == pytest.approx(1.0)
    assert burst.available is False
    assert burst.direction == "unavailable"
    assert burst.display_value == "Not loaded"


def test_live_features_match_a_dataset_row_except_fields_that_need_extra_payloads() -> None:
    with MANIFEST_PATH.open(encoding="utf-8") as file:
        as_of = datetime.fromisoformat(str(json.load(file)["as_of"]))
    row = pd.read_csv(FEATURES_PATH).iloc[0]
    repo = Repo(
        full_name=str(row["full_name"]),
        owner=Owner(login="owner", type="Organization" if _flag(row["is_org_owned"]) else "User"),
        stargazers_count=int(row["stargazers_count"]),
        forks_count=int(row["forks_count"]),
        subscribers_count=int(row["subscribers_count"]),
        open_issues_count=int(row["open_issues_count"]),
        size=int(row["size"]),
        created_at=as_of - timedelta(days=int(row["age_days"])),
        pushed_at=as_of - timedelta(days=int(row["days_since_push"])),
        archived=False,
        disabled=False,
        fork=False,
        has_issues=True,
        license=License(key="mit") if _flag(row["has_license"]) else None,
    )
    features = live_features(repo, owner=None, star_weeks=None, as_of=as_of)
    skipped = STAR_FEATURE_NAMES | {"star_to_follower_ratio"}

    for name in FEATURE_NAMES:
        if name in skipped:
            continue
        assert features[name] == pytest.approx(float(row[name]))
    assert features["star_to_follower_ratio"] is None
    assert all(features[name] is None for name in STAR_FEATURE_NAMES)


async def test_score_repository_uses_metadata_and_reports_a_slow_star_history() -> None:
    result = await score_repository(
        "example/project",
        source=_FakeSource(star_delay_seconds=1),
        model=_flat_model(),
        as_of=AS_OF,
        star_timeout_seconds=0.01,
    )

    assert result.full_name == "example/project"
    assert result.github_requests is None
    assert result.elapsed_ms is not None
    assert result.usage_summary.startswith("Took")
    assert result.score == 50
    assert result.tier == "mixed"
    assert any("did not return in time" in warning for warning in result.warnings)
    assert all(not row.available for row in result.signals if row.name in STAR_FEATURE_NAMES)


async def test_score_repository_explains_a_missing_owner_and_star_history() -> None:
    result = await score_repository(
        "https://github.com/example/project",
        source=_FakeSource(
            owner_error=GitHubError(404, "users/example", "missing"), star_error=GitHubError(404, "stars", "no")
        ),
        model=_flat_model(),
        as_of=AS_OF,
    )

    assert any("owner profile" in warning for warning in result.warnings)
    assert any("Star timing is not available" in warning for warning in result.warnings)
    follower = next(row for row in result.signals if row.name == "star_to_follower_ratio")
    assert follower.available is False


def test_saved_model_covers_the_live_features_and_scores_in_range() -> None:
    model = load_model()
    features = dict(zip(model.feature_names, model.medians, strict=True))
    probability, _contributions = predict_probability(model, features)

    assert model.feature_names == list(FEATURE_NAMES)
    assert model.test_roc_auc >= 0.6
    assert 0.0 < probability < 1.0


def test_forty_three_thousand_stars_do_not_lower_the_score() -> None:
    model = load_model()
    features = {name: float(mean) for name, mean in zip(model.feature_names, model.means, strict=True)}
    features["age_days"] = 2000.0
    features["stargazers_count"] = 43000.0
    raw_probability, contributions = predict_probability(model, features)
    adjusted, floored = apply_star_count_floor(features, contributions, model.feature_names)
    star_index = model.feature_names.index("stargazers_count")
    result = build_score(model, full_name="example/popular", features=features, fetched_at=AS_OF, warnings=[])
    star = next(row for row in result.signals if row.name == "stargazers_count")

    assert contributions[star_index] < 0
    assert floored
    assert adjusted[star_index] == 0.0
    assert star.direction == "neutral"
    assert "1,000" in star.note
    assert any("1,000" in warning for warning in result.warnings)
    assert result.score is not None
    assert result.score >= round(raw_probability * 100)


def test_star_count_below_one_thousand_can_still_lower_the_score() -> None:
    model = load_model()
    features = {name: float(mean) for name, mean in zip(model.feature_names, model.means, strict=True)}
    features["stargazers_count"] = 500.0
    _probability, contributions = predict_probability(model, features)
    adjusted, floored = apply_star_count_floor(features, contributions, model.feature_names)

    assert not floored
    assert adjusted == contributions
    assert contributions[model.feature_names.index("stargazers_count")] < 0


def test_a_star_count_above_the_training_mean_still_raises_the_score() -> None:
    model = load_model()
    features = {name: float(mean) for name, mean in zip(model.feature_names, model.means, strict=True)}
    features["stargazers_count"] = 200000.0
    _probability, contributions = predict_probability(model, features)
    _adjusted, floored = apply_star_count_floor(features, contributions, model.feature_names)

    assert not floored
    assert contributions[model.feature_names.index("stargazers_count")] > 0


def test_index_and_bad_url_do_not_call_github() -> None:
    with TestClient(app) as client:
        page = client.get("/")
        slides = client.get("/slides")
        updates = client.get("/updates")
        rejected = client.post("/api/score", json={"url": "https://example.com/acme/widget"})

    assert page.status_code == 200
    assert "Score a public repository" in page.text
    assert "Skip the form" not in page.text
    assert slides.status_code == 200
    assert "A public URL, a trust score" in slides.text
    assert updates.status_code == 200
    assert "What changed after the coach review" in updates.text
    assert rejected.status_code == 400
    assert "github.com" in rejected.json()["detail"]


def test_format_score_lists_the_tier_signals_and_warnings() -> None:
    result = build_score(
        _flat_model(),
        full_name="example/project",
        features=_features(),
        fetched_at=AS_OF,
        warnings=["This repository is archived."],
    )
    text = format_score(result)

    assert "example/project" in text
    assert "https://github.com/example/project" in text
    assert "Score: 50 / 100" in text
    assert "Tier: Mixed" in text
    assert "Warning: This repository is archived." in text
    assert "Stars" in text
    assert "Little effect" in text
    assert result.disclaimer in text


def test_describe_usage_reports_cache_hits_and_the_hourly_limit() -> None:
    cached = describe_usage(180, github_requests=0, cache_hits=3, remaining=None, limit=None)
    spent = describe_usage(2400, github_requests=12, cache_hits=1, remaining=58, limit=60)
    unknown = describe_usage(40, github_requests=None, cache_hits=None, remaining=None, limit=None)

    assert cached == "Took 0.2 seconds. Served from cache, so this score did not use the hourly GitHub limit."
    assert spent == "Took 2.4 seconds. Used 12 GitHub requests and 1 cached response. 58 of 60 remain this hour."
    assert unknown == "Took under 0.1 seconds."


async def test_a_cached_read_does_not_spend_the_hourly_limit(tmp_path: Path) -> None:
    client = GitHubClient(cache_dir=tmp_path)
    (tmp_path / "repos_example_project.json").write_text("{}", encoding="utf-8")

    payload = await client.get_repo_raw("example/project")
    usage = client.usage()

    assert payload == {}
    assert usage.network_requests == 0
    assert usage.cache_hits == 1
    assert usage.remaining is None


def test_a_live_exchange_records_the_remaining_hourly_limit() -> None:
    client = GitHubClient(cache_dir=None)
    client._count_exchange(
        {"x-ratelimit-remaining": "58", "x-ratelimit-reset": "1700000000", "x-ratelimit-limit": "60"}
    )

    usage = client.usage()

    assert usage.network_requests == 1
    assert usage.cache_hits == 0
    assert usage.remaining == 58
    assert usage.limit == 60


def test_format_score_includes_the_usage_line() -> None:
    result = build_score(
        _flat_model(), full_name="example/project", features=_features(), fetched_at=AS_OF, warnings=[]
    ).model_copy(update={"usage_summary": "Took 0.4 seconds. Used 3 GitHub requests. 57 of 60 remain this hour."})

    assert "57 of 60 remain this hour." in format_score(result)


def test_format_score_says_when_the_number_is_withheld() -> None:
    result = build_score(
        _flat_model(), full_name="example/new", features=_features(age_days=12), fetched_at=AS_OF, warnings=[]
    )

    assert "Score: withheld" in format_score(result)
    assert "Tier: Limited evidence" in format_score(result)


def test_cli_prints_a_score_without_calling_github(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    expected = build_score(
        _flat_model(), full_name="example/project", features=_features(), fetched_at=AS_OF, warnings=[]
    )

    async def fake(text: str) -> object:
        assert text == "microsoft/vscode"
        return expected

    monkeypatch.setattr("gh_signals.score.score_repository", fake)
    main(["microsoft/vscode"])

    output = capsys.readouterr().out
    assert "Score: 50 / 100" in output
    assert "Tier: Mixed" in output
    assert "example/project" in output


def test_cli_rejects_a_url_that_is_not_github(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["https://example.com/acme/widget"])

    assert caught.value.code == 2
    assert "github.com" in capsys.readouterr().err


def test_cli_reports_a_missing_repository(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    async def missing(text: str) -> object:
        raise GitHubError(404, "repos/missing/repo", "not found")

    monkeypatch.setattr("gh_signals.score.score_repository", missing)
    with pytest.raises(SystemExit) as caught:
        main(["https://github.com/missing/repo"])

    assert caught.value.code == 1
    assert "could not find" in capsys.readouterr().err


def test_cli_reports_a_missing_model_file(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    async def missing_model(text: str) -> object:
        raise FileNotFoundError(MODEL_MISSING_MESSAGE)

    monkeypatch.setattr("gh_signals.score.score_repository", missing_model)
    with pytest.raises(SystemExit) as caught:
        main(["microsoft/vscode"])

    assert caught.value.code == 1
    assert MODEL_MISSING_MESSAGE in capsys.readouterr().err


def test_feedback_note_is_stored_apart_from_the_model(tmp_path: Path) -> None:
    path = tmp_path / "feedback.jsonl"
    record_feedback(
        full_name="example/project",
        score=64,
        tier="mixed",
        judgment="too-low",
        comment="43k stars should not hurt",
        path=path,
    )
    saved = json.loads(path.read_text(encoding="utf-8").strip())

    assert saved["judgment"] == "too-low"
    assert saved["score"] == 64
    assert saved["comment"] == "43k stars should not hurt"
    with pytest.raises(ValueError, match="Judgment"):
        record_feedback(full_name="example/project", score=1, tier="mixed", judgment="bad", path=path)


def test_cli_saves_an_annotation_without_calling_github(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    expected = build_score(
        _flat_model(), full_name="example/project", features=_features(), fetched_at=AS_OF, warnings=[]
    )

    async def fake(text: str) -> object:
        assert text == "example/project"
        return expected

    monkeypatch.setattr("gh_signals.score.score_repository", fake)
    monkeypatch.setattr("gh_signals.feedback.FEEDBACK_PATH", tmp_path / "notes.jsonl")
    main(["example/project", "--annotate", "too-low", "--comment", "stars"])

    assert "Saved a local note" in capsys.readouterr().out
    saved = json.loads((tmp_path / "notes.jsonl").read_text(encoding="utf-8").strip())
    assert saved["full_name"] == "example/project"
    assert saved["comment"] == "stars"
    assert saved["judgment"] == "too-low"


def test_feedback_endpoint_writes_a_local_note(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("gh_signals.feedback.FEEDBACK_PATH", tmp_path / "notes.jsonl")
    with TestClient(app) as client:
        saved = client.post(
            "/api/feedback",
            json={
                "full_name": "example/project",
                "score": 98,
                "tier": "established",
                "judgment": "about-right",
                "comment": "looks fair",
            },
        )
        rejected = client.post(
            "/api/feedback",
            json={"full_name": "example/project", "score": 98, "tier": "established", "judgment": "nope"},
        )

    assert saved.status_code == 200
    assert saved.json() == {"saved": True}
    assert "looks fair" in (tmp_path / "notes.jsonl").read_text(encoding="utf-8")
    assert rejected.status_code == 400


def test_api_maps_a_missing_repository_to_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    async def missing(text: str, **_kwargs: object) -> None:
        raise GitHubError(404, "repos/missing/repo", "not found")

    monkeypatch.setattr("gh_signals.app.score_repository", missing)
    with TestClient(app) as client:
        response = client.post("/api/score", json={"url": "https://github.com/missing/repo"})

    assert response.status_code == 404
    assert "could not find" in response.json()["detail"]


class _FakeSource:
    def __init__(
        self,
        *,
        star_delay_seconds: float = 0,
        owner_error: GitHubError | None = None,
        star_error: GitHubError | None = None,
        repo: Repo | None = None,
    ) -> None:
        self._star_delay_seconds = star_delay_seconds
        self._owner_error = owner_error
        self._star_error = star_error
        self._repo = repo or _repo()

    async def get_repo(self, repo: str) -> Repo:
        assert repo == "example/project"
        return self._repo

    async def get_user(self, username: str) -> User:
        if self._owner_error is not None:
            raise self._owner_error
        assert username == "example"
        return User(
            login="example",
            type="Organization",
            created_at=datetime(2018, 1, 1, tzinfo=UTC),
            public_repos=4,
            followers=100,
            following=1,
        )

    async def get_star_history(self, repo: str) -> list[StarWeek]:
        assert repo == "example/project"
        if self._star_delay_seconds:
            await asyncio.sleep(self._star_delay_seconds)
        if self._star_error is not None:
            raise self._star_error
        return []


def _repo() -> Repo:
    return Repo(
        full_name="example/project",
        owner=Owner(login="example", type="Organization"),
        stargazers_count=1000,
        forks_count=100,
        subscribers_count=40,
        open_issues_count=8,
        size=1500,
        created_at=datetime(2024, 1, 1, tzinfo=UTC),
        pushed_at=datetime(2026, 11, 1, tzinfo=UTC),
        archived=False,
        disabled=False,
        fork=False,
        has_issues=True,
        license=License(key="mit"),
    )


def _features(**overrides: float | None) -> dict[str, float | None]:
    values: dict[str, float | None] = dict.fromkeys(FEATURE_NAMES, 0.0)
    values["age_days"] = 400
    values["stargazers_count"] = 1000
    values.update(overrides)
    return values


def _flat_model(**overrides: object) -> TrainedModel:
    width = len(FEATURE_NAMES)
    payload: dict[str, object] = {
        "feature_names": list(FEATURE_NAMES),
        "medians": [0.0] * width,
        "means": [0.0] * width,
        "scales": [1.0] * width,
        "coefficients": [0.0] * width,
        "intercept": 0.0,
        "snapshot": "2026-10-14T01:20:43+00:00",
        "n_train": 84,
        "n_test": 22,
        "test_roc_auc": 0.9,
    }
    payload.update(overrides)
    return TrainedModel.model_validate(payload)


def _flag(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)
