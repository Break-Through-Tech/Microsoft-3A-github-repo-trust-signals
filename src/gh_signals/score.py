"""Score one public GitHub repository from metadata the API already exposes.

The numeric score is a logistic regression trained on the shared dataset. Fetching stays in the caller
so the same rules can be tested with a fake source and no network.

Print a score from the terminal with:

    uv run trust-score https://github.com/microsoft/vscode

The same command is ``uv run python -m gh_signals.score``. Add ``--annotate too-low`` to save a
local note. Notes stay out of training.
"""

import argparse
import asyncio
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Literal, Protocol
from urllib.parse import urlparse

from pydantic import BaseModel

from gh_signals.github_client import GitHubClient, GitHubError
from gh_signals.schemas import Repo, StarWeek, User
from gh_signals.signals import fork_to_stars_ratio, issues_to_stars_ratio, stars_to_followers_ratio
from gh_signals.star_burstiness import StarFeatures, compute_features, daily_counts

MODEL_PATH = Path(__file__).parents[2] / "models" / "trust_model.json"
MODEL_MISSING_MESSAGE = "The trained model file is missing. Run uv run python -m gh_signals.train."

MIN_AGE_DAYS = 90
MIN_STARS = 50
STAR_COUNT_FLOOR = 1000
ESTABLISHED_AT = 70
MIXED_AT = 40
STAR_HISTORY_TIMEOUT_SECONDS = 4.0
CONTRIBUTION_EPSILON = 0.05

Tier = Literal["established", "mixed", "limited_evidence"]
Direction = Literal["up", "down", "neutral", "unavailable"]
_SignalKind = Literal["count", "days", "size", "ratio", "share", "flag", "burst"]

_TIER_LABELS: dict[Tier, str] = {
    "established": "Established",
    "mixed": "Mixed",
    "limited_evidence": "Limited evidence",
}
_DIRECTION_LABELS: dict[Direction, str] = {
    "up": "Raises score",
    "down": "Lowers score",
    "neutral": "Little effect",
    "unavailable": "Typical value",
}
_SEGMENT = re.compile(r"^[A-Za-z0-9_.-]+$")
_STAR_TIMEOUT_WARNING = "Star timing did not return in time, so those signals use a typical training value."
_STAR_COUNT_FLOOR_WARNING = "This repository has at least 1,000 stars, so the star count does not lower the score."
_STAR_COUNT_FLOOR_NOTE = (
    "At least 1,000 stars. The training set is mostly more famous projects, "
    "so this count is not allowed to lower the score."
)
_STAR_MISSING_WARNING = (
    "Star timing is not available for this repository, so those signals use a typical training value."
)
_OWNER_WARNING = "The owner profile could not be loaded, so stars per owner follower uses a typical training value."


class SignalSpec:
    """How one model column is labeled and formatted."""

    def __init__(self, name: str, label: str, kind: _SignalKind) -> None:
        self.name = name
        self.label = label
        self.kind = kind


FEATURE_SPECS: tuple[SignalSpec, ...] = (
    SignalSpec("stargazers_count", "Stars", "count"),
    SignalSpec("forks_count", "Forks", "count"),
    SignalSpec("subscribers_count", "Watchers", "count"),
    SignalSpec("open_issues_count", "Open issues", "count"),
    SignalSpec("size", "Repository size", "size"),
    SignalSpec("age_days", "Age in days", "days"),
    SignalSpec("days_since_push", "Days since last push", "days"),
    SignalSpec("fork_star_ratio", "Forks per star", "ratio"),
    SignalSpec("subscriber_star_ratio", "Watchers per star", "ratio"),
    SignalSpec("issues_per_star", "Open issues per star", "ratio"),
    SignalSpec("stars_per_day", "Stars per day", "ratio"),
    SignalSpec("has_license", "License detected", "flag"),
    SignalSpec("is_org_owned", "Organization owner", "flag"),
    SignalSpec("star_to_follower_ratio", "Stars per owner follower", "ratio"),
    SignalSpec("peak_stars", "Stars on the busiest day", "count"),
    SignalSpec("max_day_share", "Busiest day's share of stars", "share"),
    SignalSpec("active_day_ratio", "Days with at least one star", "share"),
    SignalSpec("burstiness", "Star burstiness", "burst"),
)
FEATURE_NAMES: tuple[str, ...] = tuple(spec.name for spec in FEATURE_SPECS)
STAR_FEATURE_NAMES: frozenset[str] = frozenset({"peak_stars", "max_day_share", "active_day_ratio", "burstiness"})
_SPECS_BY_NAME = {spec.name: spec for spec in FEATURE_SPECS}


class RepoURLError(ValueError):
    """Raised when text is not a public GitHub repository URL or owner/name pair."""


class SignalView(BaseModel):
    """One row of the breakdown shown next to the score."""

    name: str
    label: str
    value: float | None
    display_value: str
    direction: Direction
    note: str
    available: bool


class ScoreResult(BaseModel):
    """The payload returned for one repository lookup."""

    full_name: str
    repo_url: str
    score: int | None
    tier: Tier
    tier_label: str
    summary: str
    signals: list[SignalView]
    warnings: list[str]
    disclaimer: str
    fetched_at: datetime
    trained_through: str
    elapsed_ms: int | None = None
    github_requests: int | None = None
    cache_hits: int | None = None
    rate_limit_remaining: int | None = None
    rate_limit_limit: int | None = None
    usage_summary: str = ""


class TrainedModel(BaseModel):
    """Logistic regression parameters, saved so scoring does not retrain on each lookup."""

    feature_names: list[str]
    medians: list[float]
    means: list[float]
    scales: list[float]
    coefficients: list[float]
    intercept: float
    snapshot: str
    n_train: int
    n_test: int
    test_roc_auc: float


class MetadataSource(Protocol):
    """The GitHub reads the live score needs. ``GitHubClient`` already implements this."""

    async def get_repo(self, repo: str) -> Repo:
        """Return repository metadata for ``owner/name``."""

    async def get_user(self, username: str) -> User:
        """Return the public profile for a login."""

    async def get_star_history(self, repo: str) -> list[StarWeek]:
        """Return daily star counts grouped by week."""


def parse_repo_url(text: str) -> str:
    """Return ``owner/name`` from a GitHub URL or an ``owner/name`` pair.

    Args:
        text: User input, such as ``https://github.com/microsoft/vscode`` or ``microsoft/vscode``.

    Returns:
        The repository in ``owner/name`` form, without a ``.git`` suffix.

    Raises:
        RepoURLError: If the text is empty, points at another host, or has no repository name.
    """
    raw = text.strip()
    if not raw:
        raise RepoURLError("Enter a GitHub repository URL.")

    if "://" not in raw and not raw.lower().startswith(("github.com", "www.github.com")):
        parts = [part for part in raw.split("/") if part]
        if len(parts) != 2:
            raise RepoURLError("Use a github.com URL or an owner/name pair, such as microsoft/vscode.")
        return _join_repo(parts[0], parts[1])

    if "://" not in raw:
        raw = f"https://{raw}"
    parsed = urlparse(raw)
    host = parsed.netloc.lower().split("@")[-1].split(":")[0]
    if host.startswith("www."):
        host = host.removeprefix("www.")
    if host != "github.com":
        raise RepoURLError("Use a public github.com repository URL.")
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        raise RepoURLError("That URL does not include an owner and a repository name.")
    return _join_repo(parts[0], parts[1])


def load_model(path: Path = MODEL_PATH) -> TrainedModel:
    """Load the saved logistic regression.

    Args:
        path: JSON artifact written by ``python -m gh_signals.train``.

    Returns:
        The fitted parameters.

    Raises:
        FileNotFoundError: If the artifact has not been trained yet.
        ValueError: If the coefficient arrays do not match the feature names.
    """
    with path.open(encoding="utf-8") as file:
        model = TrainedModel.model_validate(json.load(file))
    width = len(model.feature_names)
    arrays = (model.medians, model.means, model.scales, model.coefficients)
    if any(len(values) != width for values in arrays):
        raise ValueError(f"Model artifact at {path} has mismatched feature arrays.")
    return model


def predict_probability(model: TrainedModel, features: Mapping[str, float | None]) -> tuple[float, list[float]]:
    """Return P(known-good) and each feature's addition to the logit.

    Missing values are filled with the training median, matching the imputer used at fit time.
    A missing feature still contributes that typical value. Callers mark the row unavailable.

    Args:
        model: Parameters from :func:`load_model`.
        features: One entry per ``model.feature_names`` key. ``None`` means the value was not loaded.

    Returns:
        The probability in ``[0, 1]``, and one logit contribution per feature, in model order.
    """
    contributions: list[float] = []
    logit = model.intercept
    for index, name in enumerate(model.feature_names):
        raw = features[name]
        filled = model.medians[index] if raw is None else raw
        scale = model.scales[index] or 1.0
        contribution = model.coefficients[index] * ((filled - model.means[index]) / scale)
        contributions.append(contribution)
        logit += contribution
    return _sigmoid(logit), contributions


def apply_star_count_floor(
    features: Mapping[str, float | None],
    contributions: list[float],
    feature_names: list[str],
) -> tuple[list[float], bool]:
    """Stop an already popular star count from lowering the score.

    The training repositories average about 52,500 stars, so a count such as 43,000 is below that
    center and receives a negative contribution even though very few public repositories have more
    than 1,000 stars. At or above that floor, a negative star-count contribution is removed.
    Timing and ratio signals are left unchanged.

    Args:
        features: Live feature values. ``stargazers_count`` is read when present.
        contributions: One logit contribution per ``feature_names`` entry, from :func:`predict_probability`.
        feature_names: Column order matching ``contributions``.

    Returns:
        The adjusted contributions, and whether the star-count contribution was removed.
    """
    if "stargazers_count" not in feature_names:
        return list(contributions), False
    index = feature_names.index("stargazers_count")
    stars = features.get("stargazers_count")
    adjusted = list(contributions)
    if stars is None or stars < STAR_COUNT_FLOOR or adjusted[index] >= 0:
        return adjusted, False
    adjusted[index] = 0.0
    return adjusted, True


def live_features(
    repo: Repo,
    owner: User | None,
    star_weeks: list[StarWeek] | None,
    as_of: datetime,
) -> dict[str, float | None]:
    """Compute the live model columns from one repository payload.

    The formulas match ``build_row`` in ``gh_signals.build_dataset`` for every column this model uses.
    Star-timing columns are ``None`` when daily history is missing or empty.

    Args:
        repo: Repository metadata from ``GET /repos/{owner}/{repo}``.
        owner: Owner profile from ``GET /users/{login}``, or ``None`` if that call failed.
        star_weeks: Daily star history, or ``None`` if it was not loaded.
        as_of: Moment ages are measured from.

    Returns:
        A value for every name in :data:`FEATURE_NAMES`.
    """
    stars = repo.stargazers_count
    age_days = max(0, (as_of - repo.created_at).days)
    timing = _star_timing(repo, star_weeks, as_of)
    return {
        "stargazers_count": float(stars),
        "forks_count": float(repo.forks_count),
        "subscribers_count": float(repo.subscribers_count),
        "open_issues_count": float(repo.open_issues_count),
        "size": float(repo.size),
        "age_days": float(age_days),
        "days_since_push": float(max(0, (as_of - repo.pushed_at).days)),
        "fork_star_ratio": fork_to_stars_ratio(repo.forks_count, stars),
        "subscriber_star_ratio": repo.subscribers_count / stars if stars else None,
        "issues_per_star": issues_to_stars_ratio(repo.open_issues_count, stars),
        "stars_per_day": stars / max(age_days, 1),
        "has_license": float(repo.license is not None),
        "is_org_owned": float(repo.owner.type == "Organization"),
        "star_to_follower_ratio": stars_to_followers_ratio(stars, owner.followers) if owner else None,
        "peak_stars": _timing_value(timing.peak_stars if timing else None),
        "max_day_share": _timing_value(timing.max_day_share if timing else None),
        "active_day_ratio": _timing_value(timing.active_day_ratio if timing else None),
        "burstiness": _timing_value(timing.burstiness if timing else None),
    }


def build_score(
    model: TrainedModel,
    *,
    full_name: str,
    features: Mapping[str, float | None],
    fetched_at: datetime,
    warnings: list[str],
) -> ScoreResult:
    """Turn feature values into a score, a tier, and a signal breakdown.

    Repositories under 90 days old, or with fewer than 50 stars, keep their signal rows but the
    numeric score is left blank. The training set is mostly well-known projects, so a low number
    on a new or small repository would mostly reflect that gap.

    Args:
        model: Fitted parameters.
        full_name: Canonical ``owner/name`` from GitHub.
        features: Output of :func:`live_features`.
        fetched_at: When the metadata was read.
        warnings: Fetch problems already discovered, such as a missing owner profile.

    Returns:
        The score payload, including a standing disclaimer.
    """
    probability, contributions = predict_probability(model, features)
    contributions, star_count_floored = apply_star_count_floor(features, contributions, model.feature_names)
    if star_count_floored:
        probability = _sigmoid(model.intercept + sum(contributions))
    points = min(100, max(0, round(probability * 100)))
    age_days = int(features["age_days"] or 0)
    stars = int(features["stargazers_count"] or 0)
    score, tier, summary = _decide_tier(points, age_days, stars)
    shown_warnings = list(warnings)
    if star_count_floored:
        shown_warnings.append(_STAR_COUNT_FLOOR_WARNING)
    signals = [
        _signal_view(
            _SPECS_BY_NAME[name],
            features[name],
            contributions[index],
            note_override=_STAR_COUNT_FLOOR_NOTE if star_count_floored and name == "stargazers_count" else None,
        )
        for index, name in enumerate(model.feature_names)
    ]
    snapshot_day = model.snapshot[:10]
    return ScoreResult(
        full_name=full_name,
        repo_url=f"https://github.com/{full_name}",
        score=score,
        tier=tier,
        tier_label=_TIER_LABELS[tier],
        summary=summary,
        signals=signals,
        warnings=shown_warnings,
        disclaimer=(
            f"This score summarizes public GitHub metadata from the training snapshot on {snapshot_day}. "
            "A withheld or low score means limited evidence, not that the repository is harmful or malicious. "
            "Repository contents are not downloaded or run."
        ),
        fetched_at=fetched_at,
        trained_through=model.snapshot,
    )


async def score_repository(
    text: str,
    *,
    source: MetadataSource | None = None,
    model: TrainedModel | None = None,
    as_of: datetime | None = None,
    star_timeout_seconds: float = STAR_HISTORY_TIMEOUT_SECONDS,
) -> ScoreResult:
    """Fetch public metadata for one repository and score it.

    Star history is optional. If it does not return within ``star_timeout_seconds``, the score is
    still returned and those rows say they were not loaded.

    Args:
        text: A GitHub repository URL or ``owner/name``.
        source: Metadata reader. Defaults to a cached ``GitHubClient``.
        model: Fitted parameters. Defaults to the artifact at :data:`MODEL_PATH`.
        as_of: Clock used for ages. Defaults to the current time.
        star_timeout_seconds: How long to wait for daily star history.

    Returns:
        The score payload.

    Raises:
        RepoURLError: If ``text`` is not a repository URL.
        GitHubError: If GitHub rejects the repository lookup.
        FileNotFoundError: If the model artifact is missing.
    """
    requested = parse_repo_url(text)
    trained = model if model is not None else load_model()
    moment = as_of or datetime.now(UTC)
    started = time.perf_counter()
    if source is None:
        async with GitHubClient() as client:
            result = await _score_loaded(client, requested, trained, moment, star_timeout_seconds)
            return _with_usage(result, client, started)
    result = await _score_loaded(source, requested, trained, moment, star_timeout_seconds)
    return _with_usage(result, source, started)


def _join_repo(owner: str, name: str) -> str:
    if name.endswith(".git"):
        name = name[:-4]
    if owner in {".", ".."} or name in {".", ".."} or not _SEGMENT.fullmatch(owner) or not _SEGMENT.fullmatch(name):
        raise RepoURLError("That repository name contains characters GitHub does not allow.")
    return f"{owner}/{name}"


def _sigmoid(logit: float) -> float:
    if logit >= 30:
        return 1.0
    if logit <= -30:
        return 0.0
    return 1.0 / (1.0 + math.exp(-logit))


def _timing_value(value: float | int | None) -> float | None:
    if value is None:
        return None
    return float(value)


def _star_timing(repo: Repo, star_weeks: list[StarWeek] | None, as_of: datetime) -> StarFeatures | None:
    if not star_weeks:
        return None
    yesterday = as_of.date() - timedelta(days=1)
    created = repo.created_at.date()
    if yesterday < created:
        return None
    timing = compute_features(repo.full_name, "", daily_counts(star_weeks, created, yesterday))
    if timing.status != "ok":
        return None
    return timing


def _decide_tier(points: int, age_days: int, stars: int) -> tuple[int | None, Tier, str]:
    too_new = age_days < MIN_AGE_DAYS
    too_small = stars < MIN_STARS
    if too_new and too_small:
        summary = (
            "This repository is under 90 days old and has fewer than 50 stars. "
            "The numeric score is withheld so age and audience size are not read as a lack of trust."
        )
        return None, "limited_evidence", summary
    if too_new:
        summary = (
            "This repository is under 90 days old. The numeric score is withheld so age is not read as a lack of trust."
        )
        return None, "limited_evidence", summary
    if too_small:
        summary = (
            "This repository has fewer than 50 stars. "
            "The numeric score is withheld so a smaller audience is not read as a lack of trust."
        )
        return None, "limited_evidence", summary
    if points >= ESTABLISHED_AT:
        return points, "established", "Public metadata resembles the known-good repositories used to train this score."
    if points >= MIXED_AT:
        return (
            points,
            "mixed",
            "Public metadata is mixed. Some signals resemble known-good repositories and some do not.",
        )
    summary = (
        "Public metadata does not resemble the known-good repositories. "
        "This is limited evidence, not a verdict that the repository is harmful."
    )
    return points, "limited_evidence", summary


def _signal_view(
    spec: SignalSpec, value: float | None, contribution: float, note_override: str | None = None
) -> SignalView:
    available = value is not None
    if not available:
        direction: Direction = "unavailable"
        note = "Not loaded. The score uses the typical value from the training set."
    elif contribution > CONTRIBUTION_EPSILON:
        direction = "up"
        note = "Moves the score up compared with the repositories used to train this model."
    elif contribution < -CONTRIBUTION_EPSILON:
        direction = "down"
        note = "Moves the score down compared with the repositories used to train this model."
    else:
        direction = "neutral"
        note = "Close to the typical training value, so it barely moves the score."
    if note_override is not None and available:
        note = note_override
    return SignalView(
        name=spec.name,
        label=spec.label,
        value=value,
        display_value=_format_value(spec.kind, value, available),
        direction=direction,
        note=note,
        available=available,
    )


def _format_value(kind: _SignalKind, value: float | None, available: bool) -> str:
    if not available or value is None:
        return "Not loaded"
    if kind == "flag":
        return "Yes" if value >= 0.5 else "No"
    if kind == "share":
        return f"{value:.1%}"
    if kind == "burst":
        return f"{value:.2f}"
    if kind == "ratio":
        return f"{value:,.1f}" if abs(value) >= 100 else f"{value:.3f}"
    rendered = f"{round(value):,}"
    if kind == "size":
        return f"{rendered} KB"
    return rendered


async def _score_loaded(
    source: MetadataSource,
    requested: str,
    model: TrainedModel,
    as_of: datetime,
    star_timeout_seconds: float,
) -> ScoreResult:
    repo = await source.get_repo(requested)
    warnings: list[str] = []
    if repo.archived:
        warnings.append("This repository is archived.")
    if repo.fork:
        warnings.append("This repository is a fork.")

    owner: User | None
    try:
        owner = await source.get_user(repo.owner.login)
    except GitHubError:
        owner = None
        warnings.append(_OWNER_WARNING)

    star_weeks, star_warning = await _load_star_weeks(source, repo.full_name, star_timeout_seconds)
    if star_warning is not None:
        warnings.append(star_warning)

    features = live_features(repo, owner, star_weeks, as_of)
    if star_warning is None and all(features[name] is None for name in STAR_FEATURE_NAMES):
        warnings.append(_STAR_MISSING_WARNING)
    return build_score(model, full_name=repo.full_name, features=features, fetched_at=as_of, warnings=warnings)


async def _load_star_weeks(
    source: MetadataSource, full_name: str, timeout: float
) -> tuple[list[StarWeek] | None, str | None]:
    try:
        return await asyncio.wait_for(source.get_star_history(full_name), timeout=timeout), None
    except TimeoutError:
        return None, _STAR_TIMEOUT_WARNING
    except GitHubError:
        return None, _STAR_MISSING_WARNING


def describe_usage(
    elapsed_ms: int,
    *,
    github_requests: int | None,
    cache_hits: int | None,
    remaining: int | None,
    limit: int | None,
) -> str:
    """Describe how long a score took and what it cost against the hourly GitHub limit.

    Args:
        elapsed_ms: Wall time for the lookup, in milliseconds.
        github_requests: Live GitHub responses for this score, or ``None`` when the source does not report them.
        cache_hits: Responses read from the local cache.
        remaining: Requests GitHub says are left this hour, or ``None`` when no live response reported a limit.
        limit: The hourly allowance from GitHub, such as 60 or 5,000.

    Returns:
        One or two sentences safe to show on the page and in the terminal.
    """
    if elapsed_ms < 100:
        took = "Took under 0.1 seconds."
    elif elapsed_ms < 10_000:
        took = f"Took {elapsed_ms / 1000:.1f} seconds."
    else:
        took = f"Took {round(elapsed_ms / 1000)} seconds."
    if github_requests is None:
        return took
    if github_requests == 0:
        return f"{took} Served from cache, so this score did not use the hourly GitHub limit."
    request_word = "request" if github_requests == 1 else "requests"
    sentence = f"{took} Used {github_requests} GitHub {request_word}"
    if cache_hits:
        hit_word = "response" if cache_hits == 1 else "responses"
        sentence = f"{sentence} and {cache_hits} cached {hit_word}"
    sentence = f"{sentence}."
    if remaining is not None and limit is not None:
        return f"{sentence} {remaining:,} of {limit:,} remain this hour."
    if remaining is not None:
        return f"{sentence} {remaining:,} remain this hour."
    return sentence


def _with_usage(result: ScoreResult, source: MetadataSource, started: float) -> ScoreResult:
    """Attach elapsed time and, for a GitHub client, the quota this score spent."""
    elapsed_ms = max(0, round((time.perf_counter() - started) * 1000))
    requests: int | None = None
    hits: int | None = None
    remaining: int | None = None
    limit: int | None = None
    if isinstance(source, GitHubClient):
        usage = source.usage()
        requests = usage.network_requests
        hits = usage.cache_hits
        remaining = usage.remaining
        limit = usage.limit
    return result.model_copy(
        update={
            "elapsed_ms": elapsed_ms,
            "github_requests": requests,
            "cache_hits": hits,
            "rate_limit_remaining": remaining,
            "rate_limit_limit": limit,
            "usage_summary": describe_usage(
                elapsed_ms,
                github_requests=requests,
                cache_hits=hits,
                remaining=remaining,
                limit=limit,
            ),
        }
    )


def format_score(result: ScoreResult) -> str:
    """Render a score payload as plain text for the terminal.

    Args:
        result: The payload returned by :func:`score_repository`.

    Returns:
        A multi-line summary with the score, the tier, warnings, and one row per signal.
    """
    shown = "withheld" if result.score is None else f"{result.score} / 100"
    lines = [
        result.full_name,
        result.repo_url,
        f"Score: {shown}",
        f"Tier: {result.tier_label}",
    ]
    if result.usage_summary:
        lines.append(result.usage_summary)
    lines.extend(
        [
            "",
            result.summary,
        ]
    )
    if result.warnings:
        lines.append("")
        lines.extend(f"Warning: {warning}" for warning in result.warnings)
    lines.append("")
    label_width = max((len(signal.label) for signal in result.signals), default=0)
    for signal in result.signals:
        lines.append(
            f"{signal.label:<{label_width}}  {signal.display_value:<12}  {_DIRECTION_LABELS[signal.direction]}"
        )
    lines.extend(["", result.disclaimer])
    return "\n".join(lines)


def github_failure_message(status: int) -> str:
    """Return the sentence to show when GitHub rejects a repository lookup.

    Args:
        status: HTTP status from the GitHub API.

    Returns:
        A short explanation safe to show in the page and on the terminal.
    """
    if status == 404:
        return "GitHub could not find that repository. It may be private, renamed, or missing."
    if status in (403, 429):
        return "GitHub rate limit reached. Set GITHUB_TOKEN or run gh auth login, then try again."
    return "GitHub returned an error. Try again in a moment."


def main(argv: list[str] | None = None) -> None:
    """Score one public repository and print the same payload the web page shows.

    Args:
        argv: Arguments without the program name. Defaults to the process arguments.

    Raises:
        SystemExit: 2 when the text is not a repository URL, 1 when GitHub or the model file fails.
    """
    parser = argparse.ArgumentParser(
        prog="trust-score",
        description="Score one public GitHub repository and print the tier and signal breakdown.",
    )
    parser.add_argument("url", help="A github.com URL or owner/name, such as microsoft/vscode")
    parser.add_argument(
        "--annotate",
        choices=["about-right", "too-high", "too-low"],
        help="Save a local note about this score. The note is not used for training.",
    )
    parser.add_argument("--comment", default="", help="Optional note stored with --annotate")
    args = parser.parse_args(argv)
    if args.comment and args.annotate is None:
        print("Pass --annotate when adding a comment.", file=sys.stderr)
        raise SystemExit(2)
    try:
        result = asyncio.run(score_repository(args.url))
    except RepoURLError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    except GitHubError as exc:
        print(github_failure_message(exc.status), file=sys.stderr)
        raise SystemExit(1) from exc
    except FileNotFoundError as exc:
        print(MODEL_MISSING_MESSAGE, file=sys.stderr)
        raise SystemExit(1) from exc
    print(format_score(result))
    if args.annotate is not None:
        from gh_signals.feedback import record_feedback

        record_feedback(
            full_name=result.full_name,
            score=result.score,
            tier=result.tier,
            judgment=args.annotate,
            comment=args.comment,
        )
        print("Saved a local note. It is not used in the score.")


if __name__ == "__main__":
    main()
