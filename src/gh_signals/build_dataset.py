"""Builds the shared feature dataset and its train/test split from raw GitHub API data.

Pipeline:
    data/good_repos.csv + data/uncertain.csv  (the labels, and the source of truth for which repos exist)
    -> data/raw/                              (verbatim API responses plus snapshot.json, local and gitignored)
    -> data/features.csv                      (one row per repo: identifiers, label, split, every signal)
    -> data/train_split.csv, data/test_split.csv, data/manifest.json

Run ``uv run python -m gh_signals.build_dataset`` to build. Responses already in ``data/raw/`` are reused without
touching the network, so rebuilding from an existing snapshot is deterministic and an interrupted run resumes where
it stopped. Delete ``data/raw/`` to take a fresh snapshot. ``--verify`` checks the committed files offline.
"""

import argparse
import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import hashlib
import json
import math
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from gh_signals.github_client import GitHubClient, GitHubError
from gh_signals.schemas import ContributorStats, Repo, StarWeek, User
from gh_signals.signals import (
    contributor_activity_ratio,
    fork_to_stars_ratio,
    ghost_contributor_ratio,
    is_bot_contributor,
    issues_to_stars_ratio,
    stars_to_followers_ratio,
)
from gh_signals.star_burstiness import LABEL_FILES, compute_features, daily_counts, load_labeled_repos

DATA_DIR = Path(__file__).parents[2] / "data"
RAW_DIR = DATA_DIR / "raw"
SNAPSHOT_PATH = RAW_DIR / "snapshot.json"
FEATURES_PATH = DATA_DIR / "features.csv"
TRAIN_PATH = DATA_DIR / "train_split.csv"
TEST_PATH = DATA_DIR / "test_split.csv"
MANIFEST_PATH = DATA_DIR / "manifest.json"

TOP_N_CONTRIBUTORS = 10
TEST_SIZE = 0.2
RANDOM_SEED = 42
FETCH_CONCURRENCY = 8

# Never model inputs: they name the repo or reveal the label.
IDENTIFIER_COLUMNS = ["full_name", "listed_as", "label", "target_label", "split"]
SIGNAL_COLUMNS = [
    "stargazers_count",
    "forks_count",
    "subscribers_count",
    "open_issues_count",
    "size",
    "age_days",
    "days_since_push",
    "fork_star_ratio",
    "subscriber_star_ratio",
    "issues_per_star",
    "stars_per_day",
    "has_license",
    "is_org_owned",
    "contributor_activity_ratio",
    "ghost_contributor_ratio",
    "star_to_follower_ratio",
    "peak_stars",
    "max_day_share",
    "active_day_ratio",
    "burstiness",
]
ALL_COLUMNS = IDENTIFIER_COLUMNS + SIGNAL_COLUMNS


@dataclass
class RepoRecord:
    """Everything fetched for one repo, as typed API responses. ``None`` means the lookup was rejected."""

    listed_as: str
    label: str
    repo: Repo
    contributors: list[ContributorStats]
    top_users: list[User]
    owner: User | None
    star_weeks: list[StarWeek] | None


def read_snapshot_time() -> datetime:
    """Returns when the raw snapshot was started, creating the marker on the first fetch.

    Ages are measured from this moment rather than the current time, which is what makes a rebuild from the same raw
    data produce the same numbers on any day.
    """
    if SNAPSHOT_PATH.exists():
        with SNAPSHOT_PATH.open(encoding="utf-8") as file:
            return datetime.fromisoformat(json.load(file)["as_of"])
    as_of = datetime.now(UTC).replace(microsecond=0)
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SNAPSHOT_PATH.open("w", encoding="utf-8") as file:
        json.dump({"as_of": as_of.isoformat()}, file)
    return as_of


async def fetch_record(client: GitHubClient, semaphore: asyncio.Semaphore, listed_as: str, label: str) -> RepoRecord:
    async with semaphore:
        repo = await client.get_repo(listed_as)
        contributors = await client.get_contributor_stats(listed_as)

        humans = [c for c in contributors if c.author is not None and not is_bot_contributor(c)]
        top = sorted(humans, key=lambda c: c.total, reverse=True)[:TOP_N_CONTRIBUTORS]
        top_users: list[User] = []
        for contributor in top:
            if contributor.author is None:
                continue
            try:
                top_users.append(await client.get_user(contributor.author.login))
            except GitHubError:
                continue

        try:
            owner = await client.get_user(repo.owner.login)
        except GitHubError:
            owner = None
        try:
            star_weeks = await client.get_star_history(listed_as)
        except GitHubError:
            star_weeks = None
    return RepoRecord(listed_as, label, repo, contributors, top_users, owner, star_weeks)


async def collect() -> list[RepoRecord]:
    """Fetches every labeled repo, failing loudly rather than silently dropping any of them."""
    labeled = load_labeled_repos()
    semaphore = asyncio.Semaphore(FETCH_CONCURRENCY)
    async with GitHubClient(cache_dir=RAW_DIR, cache_ttl_seconds=math.inf) as client:
        results = await asyncio.gather(
            *(fetch_record(client, semaphore, repo, label) for repo, label in labeled), return_exceptions=True
        )

    records: list[RepoRecord] = []
    failures: list[str] = []
    for (repo, _), result in zip(labeled, results, strict=True):
        if isinstance(result, RepoRecord):
            records.append(result)
        else:
            failures.append(f"{repo}: {result!r}")
    if failures:
        raise RuntimeError(
            f"{len(failures)} of {len(labeled)} repos failed. Completed responses are saved in {RAW_DIR}, so "
            "rerunning resumes where this stopped.\n" + "\n".join(failures)
        )
    return records


def build_row(record: RepoRecord, as_of: datetime) -> dict[str, object]:
    """Computes one dataset row from fetched data. Pure: the same inputs always give the same row."""
    repo = record.repo
    stars = repo.stargazers_count
    age_days = max(0, (as_of - repo.created_at).days)

    star_timing = None
    if record.star_weeks is not None:
        yesterday = as_of.date() - timedelta(days=1)
        counts = daily_counts(record.star_weeks, repo.created_at.date(), yesterday)
        star_timing = compute_features(repo.full_name, record.label, counts)

    return {
        "full_name": repo.full_name,
        "listed_as": record.listed_as,
        "label": record.label,
        "target_label": int(record.label == "good"),
        "stargazers_count": stars,
        "forks_count": repo.forks_count,
        "subscribers_count": repo.subscribers_count,
        "open_issues_count": repo.open_issues_count,
        "size": repo.size,
        "age_days": age_days,
        "days_since_push": max(0, (as_of - repo.pushed_at).days),
        "fork_star_ratio": fork_to_stars_ratio(repo.forks_count, stars),
        "subscriber_star_ratio": repo.subscribers_count / stars if stars else None,
        "issues_per_star": issues_to_stars_ratio(repo.open_issues_count, stars),
        "stars_per_day": stars / max(age_days, 1),
        "has_license": repo.license is not None,
        "is_org_owned": repo.owner.type == "Organization",
        "contributor_activity_ratio": contributor_activity_ratio(record.contributors, now=as_of),
        "ghost_contributor_ratio": ghost_contributor_ratio(record.top_users),
        "star_to_follower_ratio": stars_to_followers_ratio(stars, record.owner.followers) if record.owner else None,
        "peak_stars": star_timing.peak_stars if star_timing else None,
        "max_day_share": star_timing.max_day_share if star_timing else None,
        "active_day_ratio": star_timing.active_day_ratio if star_timing else None,
        "burstiness": star_timing.burstiness if star_timing else None,
    }


def assign_splits(target_labels: list[int]) -> list[str]:
    """Labels each row ``train`` or ``test``: stratified 80/20 with a fixed seed, so it never changes between builds."""
    indices = list(range(len(target_labels)))
    _, test_indices = train_test_split(indices, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=target_labels)
    test = set(test_indices)
    return ["test" if index in test else "train" for index in indices]


def build_dataframe(records: list[RepoRecord], as_of: datetime) -> pd.DataFrame:
    df = pd.DataFrame([build_row(record, as_of) for record in records])
    df["split"] = assign_splits(df["target_label"].tolist())
    return df[ALL_COLUMNS]


def file_sha256(path: Path) -> str:
    """Hashes a text file with line endings normalized, so Windows and macOS checkouts agree."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def count_by_label(df: pd.DataFrame) -> dict[str, int]:
    return {label: int(count) for label, count in df["label"].value_counts().sort_index().items()}


def build_manifest(df: pd.DataFrame, as_of: datetime) -> dict[str, object]:
    missing = df[SIGNAL_COLUMNS].isna().sum()
    return {
        "as_of": as_of.isoformat(),
        "random_seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "top_n_contributors": TOP_N_CONTRIBUTORS,
        "n_repos": len(df),
        "label_counts": count_by_label(df),
        "split_counts": {name: count_by_label(df[df["split"] == name]) for name in ("train", "test")},
        "identifier_columns": IDENTIFIER_COLUMNS,
        "signal_columns": SIGNAL_COLUMNS,
        "missing_values": {column: int(count) for column, count in missing.items() if count},
        "label_files_sha256": {label: file_sha256(path) for label, path in LABEL_FILES.items()},
        "features_sha256": file_sha256(FEATURES_PATH),
    }


def write_outputs(df: pd.DataFrame, as_of: datetime) -> None:
    for path, frame in (
        (FEATURES_PATH, df),
        (TRAIN_PATH, df[df["split"] == "train"]),
        (TEST_PATH, df[df["split"] == "test"]),
    ):
        with path.open("w", encoding="utf-8", newline="") as file:
            file.write(frame.to_csv(index=False, lineterminator="\n"))
    with MANIFEST_PATH.open("w", encoding="utf-8", newline="") as file:
        json.dump(build_manifest(df, as_of), file, indent=2)
        file.write("\n")


def verify_dataset() -> list[str]:
    """Checks the committed dataset files against each other and the label files, with no network access.

    Returns:
        A description of every problem found; an empty list means everyone is looking at one consistent version.
    """
    problems: list[str] = []
    df = pd.read_csv(FEATURES_PATH)
    with MANIFEST_PATH.open(encoding="utf-8") as file:
        manifest = json.load(file)

    if list(df.columns) != ALL_COLUMNS:
        problems.append(f"features.csv columns differ from the schema in build_dataset.py: {list(df.columns)}")
        return problems

    labeled = load_labeled_repos()
    if list(zip(df["listed_as"], df["label"], strict=True)) != labeled:
        problems.append("Rows do not match good_repos.csv + uncertain.csv in order. Rebuild the dataset.")
    for label, path in LABEL_FILES.items():
        if file_sha256(path) != manifest["label_files_sha256"][label]:
            problems.append(f"{path.name} changed since the dataset was built. Rebuild the dataset.")

    if df[IDENTIFIER_COLUMNS].isna().any().any():
        problems.append("Identifier columns contain missing values.")
    if df["full_name"].str.lower().duplicated().any():
        problems.append("full_name contains duplicates (compared case-insensitively).")
    if (df["target_label"] != (df["label"] == "good").astype(int)).any():
        problems.append("target_label disagrees with label.")

    for column in SIGNAL_COLUMNS:
        if not (pd.api.types.is_numeric_dtype(df[column]) or pd.api.types.is_bool_dtype(df[column])):
            problems.append(f"Signal column {column} is not numeric.")
    missing = {column: int(count) for column, count in df[SIGNAL_COLUMNS].isna().sum().items() if count}
    if missing != manifest["missing_values"]:
        problems.append(f"Missing values {missing} differ from the manifest {manifest['missing_values']}.")

    if assign_splits(df["target_label"].tolist()) != df["split"].tolist():
        problems.append("split column is not what seed and stratification reproduce.")
    train, test = pd.read_csv(TRAIN_PATH), pd.read_csv(TEST_PATH)
    if not train.equals(df[df["split"] == "train"].reset_index(drop=True)):
        problems.append("train_split.csv is not the train rows of features.csv.")
    if not test.equals(df[df["split"] == "test"].reset_index(drop=True)):
        problems.append("test_split.csv is not the test rows of features.csv.")
    if set(train["full_name"]) & set(test["full_name"]):
        problems.append("A repository appears in both train and test.")

    if file_sha256(FEATURES_PATH) != manifest["features_sha256"]:
        problems.append(
            "features.csv does not match the manifest. Rebuild the dataset and commit all outputs together."
        )
    return problems


async def build() -> None:
    as_of = read_snapshot_time()
    records = await collect()
    df = build_dataframe(records, as_of)
    write_outputs(df, as_of)
    print(f"Built {len(df)} repos from the snapshot taken {as_of.isoformat()}.")
    print(f"Wrote {FEATURES_PATH.name}, {TRAIN_PATH.name}, {TEST_PATH.name}, {MANIFEST_PATH.name} in {DATA_DIR}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or verify the shared feature dataset.")
    parser.add_argument("--verify", action="store_true", help="check the committed dataset files without any network")
    args = parser.parse_args()

    if args.verify:
        problems = verify_dataset()
        for problem in problems:
            print(f"PROBLEM: {problem}")
        print("Dataset is consistent." if not problems else f"{len(problems)} problem(s) found.")
        raise SystemExit(1 if problems else 0)
    asyncio.run(build())


if __name__ == "__main__":
    main()
