"""Scores any public GitHub repository: the single entry point for the trust score.

Example:
    from gh_signals.score import score_repo

    result = score_repo("microsoft/vscode")
    print(result.score, result.tier, result.missing_signals)

Inside a running event loop, such as a Jupyter notebook, use ``await score_repo_async("microsoft/vscode")`` instead.

API responses are cached on disk in ``.cache/github`` for 24 hours, so scoring the same repo again within a day makes no
requests. The model is trained once per process from the committed ``data/features.csv`` and reused for every call.
"""

import asyncio
from datetime import UTC, datetime
from functools import cache
import json
import math
import re
from typing import Literal

import pandas as pd
from pydantic import BaseModel
from sklearn.pipeline import Pipeline

from gh_signals.build_dataset import FEATURES_PATH, MANIFEST_PATH, build_row, fetch_record
from gh_signals.external_eval import MODEL_COLUMNS, TRUSTED_FROM, UNCERTAIN_FROM, make_models
from gh_signals.github_client import DEFAULT_CACHE_DIR, GitHubClient, GitHubError

# The signals the score is computed from: the combined model's signals plus the owner reputation and archived signals.
SCORE_COLUMNS = [*MODEL_COLUMNS, "owner_followers", "owner_public_repos", "is_archived"]

# Below this share of available signals the score rests on too much imputation to place a repo in a confident tier.
MIN_COVERAGE = 0.8
# Model scores this far apart (on the 0-100 scale) mean the evidence is ambiguous.
MAX_MODEL_DISAGREEMENT = 30.0
CACHE_TTL_SECONDS = 24 * 60 * 60

Tier = Literal["trusted", "uncertain", "low_confidence"]
SignalValue = float | int | bool | None

_REPO_PATTERN = re.compile(r"^(?:https?://)?(?:www\.)?(?:github\.com/)?([\w.-]+)/([\w.-]+?)(?:\.git)?/?$")


class RepoUnavailableError(Exception):
    """Raised when a repository cannot be scored at all, such as when it does not exist or is private."""


class TrustResult(BaseModel):
    """The result contract of :func:`score_repo`.

    Attributes:
        repo: The repository as GitHub currently names it, in ``owner/name`` form.
        score: Trust from 0 to 100, the mean of ``model_scores``. Higher means more evidence of a high-quality repo. A
            low score means little evidence, never that the repo is bad.
        tier: ``trusted``, ``uncertain``, or ``low_confidence``. Forced to ``uncertain`` when too many signals are
            missing or the models disagree, so a thin result is never presented as confident either way.
        model_scores: Each model's score from 0 to 100, keyed by model name (``lr`` and ``rf``).
        signals: The value of every signal in :data:`SCORE_COLUMNS`. ``None`` means the signal could not be computed.
        missing_signals: The names of the signals that are ``None``. Their values were imputed with the training median.
        coverage: The share of signals that were available, from 0 to 1.
        warnings: Human-readable notes on anything that limits the result.
        as_of: When the repo was scored. Ages are measured from this moment.
        model_version: The first 12 characters of the training dataset's hash, from ``data/manifest.json``.
    """

    repo: str
    score: float
    tier: Tier
    model_scores: dict[str, float]
    signals: dict[str, SignalValue]
    missing_signals: list[str]
    coverage: float
    warnings: list[str]
    as_of: datetime
    model_version: str


class TrustModel:
    """The fitted models behind the score, trained on every labeled repo in the committed dataset."""

    def __init__(self, models: dict[str, Pipeline], version: str) -> None:
        self.models = models
        self.version = version

    def predict(self, signals: dict[str, SignalValue]) -> dict[str, float]:
        """Returns each model's score from 0 to 100. Missing signals are imputed inside each pipeline."""
        row = pd.DataFrame(
            [{column: math.nan if signals[column] is None else signals[column] for column in SCORE_COLUMNS}]
        )
        row = row.astype(float)
        return {name: float(model.predict_proba(row)[0, 1] * 100) for name, model in self.models.items()}


@cache
def load_model() -> TrustModel:
    """Trains the models once per process and returns the same fitted instance on every later call."""
    labeled = pd.read_csv(FEATURES_PATH)
    features, target = labeled[SCORE_COLUMNS].astype(float), labeled["target_label"]
    models = make_models()
    for model in models.values():
        model.fit(features, target)
    with MANIFEST_PATH.open(encoding="utf-8") as file:
        version = json.load(file)["features_sha256"][:12]
    return TrustModel(models, version)


def parse_repo_name(repo: str) -> str:
    """Normalizes ``owner/name`` or a GitHub URL to ``owner/name``.

    Raises:
        ValueError: If ``repo`` is not a repository name or URL.
    """
    match = _REPO_PATTERN.match(repo.strip())
    if match is None:
        raise ValueError(f"Expected a repository as owner/name or a GitHub URL, got {repo!r}")
    return f"{match[1]}/{match[2]}"


def assign_tier(score: float, coverage: float, model_scores: dict[str, float]) -> Tier:
    """Places a score in a tier, preferring ``uncertain`` whenever the evidence is thin or conflicting.

    Provisional: uses the bands from ``gh_signals.external_eval``. The official tiers (issue #37) replace this function.
    """
    if coverage < MIN_COVERAGE:
        return "uncertain"
    if max(model_scores.values()) - min(model_scores.values()) >= MAX_MODEL_DISAGREEMENT:
        return "uncertain"
    if score >= TRUSTED_FROM:
        return "trusted"
    if score >= UNCERTAIN_FROM:
        return "uncertain"
    return "low_confidence"


def build_warnings(
    signals: dict[str, SignalValue], missing: list[str], coverage: float, model_scores: dict[str, float]
) -> list[str]:
    warnings: list[str] = []
    if missing:
        warnings.append(f"Imputed {len(missing)} missing signal(s): {', '.join(missing)}.")
    if coverage < MIN_COVERAGE:
        warnings.append(f"Only {coverage:.0%} of signals were available, so the tier is uncertain.")
    if max(model_scores.values()) - min(model_scores.values()) >= MAX_MODEL_DISAGREEMENT:
        warnings.append("The models disagree by 30 or more points, so the tier is uncertain.")
    if signals["is_archived"]:
        warnings.append("The repository is archived and read-only.")
    return warnings


async def score_repo_async(
    repo: str, *, client: GitHubClient | None = None, as_of: datetime | None = None
) -> TrustResult:
    """Scores one repository. See :func:`score_repo`.

    Args:
        repo: The repository as ``owner/name`` or a GitHub URL.
        client: An open client to reuse. If omitted, one is opened with a 24 hour disk cache.
        as_of: The reference time for ages. Defaults to now.
    """
    name = parse_repo_name(repo)
    as_of = as_of or datetime.now(UTC).replace(microsecond=0)
    if client is None:
        async with GitHubClient(cache_dir=DEFAULT_CACHE_DIR, cache_ttl_seconds=CACHE_TTL_SECONDS) as own_client:
            return await score_repo_async(name, client=own_client, as_of=as_of)

    try:
        record = await fetch_record(client, asyncio.Semaphore(1), name, "unlabeled", strict=False, star_history=False)
    except GitHubError as error:
        raise RepoUnavailableError(f"Could not fetch {name}: GitHub returned {error.status}") from error

    row = build_row(record, as_of)
    signals: dict[str, SignalValue] = {column: _clean(row[column]) for column in SCORE_COLUMNS}
    missing = [column for column, value in signals.items() if value is None]
    coverage = 1 - len(missing) / len(SCORE_COLUMNS)

    model = load_model()
    model_scores = model.predict(signals)
    score = sum(model_scores.values()) / len(model_scores)
    return TrustResult(
        repo=record.repo.full_name,
        score=round(score, 1),
        tier=assign_tier(score, coverage, model_scores),
        model_scores={name: round(value, 1) for name, value in model_scores.items()},
        signals=signals,
        missing_signals=missing,
        coverage=round(coverage, 3),
        warnings=build_warnings(signals, missing, coverage, model_scores),
        as_of=as_of,
        model_version=model.version,
    )


def score_repo(repo: str) -> TrustResult:
    """Scores one public GitHub repository from 0 to 100.

    Signals that cannot be fetched are imputed and listed in ``missing_signals`` rather than failing the call, and the
    tier falls back to ``uncertain`` when too few signals are available.

    Args:
        repo: The repository as ``owner/name`` or a GitHub URL, such as ``https://github.com/microsoft/vscode``.

    Returns:
        The score, tier, signal values, and anything that limits the result. See :class:`TrustResult`.

    Raises:
        ValueError: If ``repo`` is not a repository name or URL.
        RepoUnavailableError: If the repository does not exist, is private, or GitHub rejects the lookup.
    """
    return asyncio.run(score_repo_async(repo))


def _clean(value: object) -> SignalValue:
    """Converts a row value to a plain Python signal value, mapping NaN to ``None``."""
    if value is None:
        return None
    if isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        return None if math.isnan(value) else value
    raise TypeError(f"Unexpected signal value {value!r}")
