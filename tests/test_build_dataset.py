from datetime import UTC, date, datetime, timedelta

from gh_signals.build_dataset import SIGNAL_COLUMNS, RepoRecord, assign_splits, build_row, verify_dataset
from gh_signals.schemas import Owner, Repo, StarWeek, User

AS_OF = datetime(2026, 10, 10, 12, tzinfo=UTC)


def _repo(*, stars: int = 100, created_days_ago: int = 10, pushed_days_ago: int = 1) -> Repo:
    return Repo(
        full_name="Org/Name",
        owner=Owner(login="Org", type="Organization"),
        stargazers_count=stars,
        forks_count=20,
        subscribers_count=5,
        open_issues_count=2,
        size=1,
        created_at=AS_OF - timedelta(days=created_days_ago),
        pushed_at=AS_OF - timedelta(days=pushed_days_ago),
        archived=False,
        disabled=False,
        fork=False,
        has_issues=True,
    )


def _owner(followers: int) -> User:
    return User(login="Org", type="Organization", created_at=AS_OF, public_repos=3, followers=followers, following=0)


def _record(repo: Repo, *, star_weeks: list[StarWeek] | None = None, owner: User | None = None) -> RepoRecord:
    return RepoRecord("org/name", "good", repo, [], [], owner, star_weeks)


def test_build_row_computes_ratios_with_the_shared_signal_functions() -> None:
    row = build_row(_record(_repo(stars=100), owner=_owner(followers=4)), AS_OF)

    assert row["full_name"] == "Org/Name"
    assert row["listed_as"] == "org/name"
    assert row["target_label"] == 1
    assert row["fork_star_ratio"] == 0.2
    assert row["issues_per_star"] == 0.02
    assert row["subscriber_star_ratio"] == 0.05
    assert row["star_to_follower_ratio"] == 25.0
    assert row["age_days"] == 10
    assert row["is_org_owned"] is True
    assert row["has_license"] is False


def test_build_row_leaves_ratios_missing_instead_of_dividing_by_zero() -> None:
    row = build_row(_record(_repo(stars=0), owner=_owner(followers=0)), AS_OF)

    assert row["fork_star_ratio"] is None
    assert row["issues_per_star"] is None
    assert row["subscriber_star_ratio"] is None
    assert row["star_to_follower_ratio"] is None


def test_build_row_without_star_history_or_owner_leaves_those_signals_missing() -> None:
    row = build_row(_record(_repo()), AS_OF)

    assert row["star_to_follower_ratio"] is None
    assert all(row[column] is None for column in ("peak_stars", "max_day_share", "active_day_ratio", "burstiness"))


def test_build_row_never_reports_negative_days_for_a_push_after_the_snapshot() -> None:
    row = build_row(_record(_repo(pushed_days_ago=-1)), AS_OF)

    assert row["days_since_push"] == 0


def test_build_row_star_timing_counts_only_days_from_creation_to_yesterday() -> None:
    created = AS_OF.date() - timedelta(days=10)
    start = created - timedelta(days=3)  # The week begins before the repo existed, so its first days are ignored.
    week = StarWeek(week=_epoch(start), total=0, days=[9, 9, 9, 5, 1, 1, 1])
    next_week = StarWeek(week=_epoch(start + timedelta(days=7)), total=0, days=[1, 1, 1, 1, 1, 1, 9])

    row = build_row(_record(_repo(created_days_ago=10), star_weeks=[week, next_week]), AS_OF)

    # Creation through yesterday is 5, 1, 1, 1, then six days of 1: 14 stars. The 9 on the last day is today, so excluded.
    assert row["peak_stars"] == 5
    assert row["max_day_share"] == 5 / 14


def _epoch(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp())


def test_assign_splits_is_stratified_eighty_twenty_and_deterministic() -> None:
    labels = [1] * 82 + [0] * 24

    splits = assign_splits(labels)

    assert splits == assign_splits(labels)
    assert len(splits) == 106
    assert splits.count("test") == 22
    test_labels = [label for label, split in zip(labels, splits, strict=True) if split == "test"]
    assert sum(test_labels) == 17


def test_signal_columns_are_unique() -> None:
    assert len(SIGNAL_COLUMNS) == len(set(SIGNAL_COLUMNS))


def test_committed_dataset_is_consistent() -> None:
    assert verify_dataset() == []
