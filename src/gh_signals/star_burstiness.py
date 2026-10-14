"""Star timing features computed from daily star history.

The features become columns of ``data/features.csv``; see ``gh_signals.build_dataset``.
"""

import csv
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
import statistics

from gh_signals.schemas import StarWeek

DATA_DIR = Path(__file__).parents[2] / "data"
LABEL_FILES = {"good": DATA_DIR / "good_repos.csv", "uncertain": DATA_DIR / "uncertain.csv"}


@dataclass
class StarFeatures:
    repo_name: str
    label: str
    status: str
    total_stars: int | None = None
    age_days: int | None = None
    peak_date: str | None = None
    peak_stars: int | None = None
    max_day_share: float | None = None
    active_day_ratio: float | None = None
    burstiness: float | None = None


def load_labeled_repos() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for label, path in LABEL_FILES.items():
        with path.open(encoding="utf-8") as file:
            pairs.extend((row["repo"].strip(), label) for row in csv.DictReader(file))
    return pairs


def daily_counts(weeks: list[StarWeek], created: date, end: date) -> dict[date, int]:
    """Returns stars per day from ``created`` to ``end`` inclusive, with zero for days without stars."""
    counts = {created + timedelta(days=i): 0 for i in range((end - created).days + 1)}
    for week in weeks:
        start = datetime.fromtimestamp(week.week, tz=UTC).date()
        for offset, stars in enumerate(week.days):
            day = start + timedelta(days=offset)
            if day in counts:
                counts[day] += stars
    return counts


def compute_features(repo_name: str, label: str, counts: dict[date, int]) -> StarFeatures:
    values = list(counts.values())
    total = sum(values)
    if total == 0:
        return StarFeatures(repo_name, label, status="no_stars", total_stars=0, age_days=len(values))
    peak_date = max(counts, key=lambda day: counts[day])
    mean = statistics.fmean(values)
    std = statistics.pstdev(values)
    return StarFeatures(
        repo_name=repo_name,
        label=label,
        status="ok",
        total_stars=total,
        age_days=len(values),
        peak_date=peak_date.isoformat(),
        peak_stars=counts[peak_date],
        max_day_share=counts[peak_date] / total,
        active_day_ratio=sum(value > 0 for value in values) / len(values),
        burstiness=(std - mean) / (std + mean),
    )
