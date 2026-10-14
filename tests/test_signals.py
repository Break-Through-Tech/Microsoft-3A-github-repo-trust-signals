from datetime import UTC, datetime, timedelta

from gh_signals.schemas import ContributorStats, ContributorWeek, Owner, User
from gh_signals.signals import (
    contributor_activity_ratio,
    ghost_contributor_ratio,
    is_bot_contributor,
    is_ghost_account,
    star_growth_burstiness,
)

EPOCH = datetime(2020, 1, 1, tzinfo=UTC)


def test_star_growth_burstiness_returns_none_below_three_points() -> None:
    assert star_growth_burstiness([]) is None
    assert star_growth_burstiness([EPOCH, EPOCH + timedelta(days=1)]) is None


def test_star_growth_burstiness_scores_regular_spacing_near_negative_one() -> None:
    timestamps = [EPOCH + timedelta(days=i) for i in range(30)]

    burstiness = star_growth_burstiness(timestamps)

    assert burstiness is not None
    assert burstiness < -0.9


def test_star_growth_burstiness_scores_a_single_spike_higher_than_regular_spacing() -> None:
    regular = [EPOCH + timedelta(days=i) for i in range(30)]
    burst = [EPOCH + timedelta(seconds=i) for i in range(20)] + [EPOCH + timedelta(days=365)]

    regular_score = star_growth_burstiness(regular)
    burst_score = star_growth_burstiness(burst)

    assert regular_score is not None
    assert burst_score is not None
    # A single long quiet gap after a tight cluster is far more lopsided than steady daily
    # spacing, so it scores noticeably higher, even though finite-sample bias keeps it well
    # short of the theoretical maximum of 1.0 for this few data points.
    assert burst_score > 0.5
    assert burst_score > regular_score


def test_star_growth_burstiness_ignores_input_order() -> None:
    timestamps = [EPOCH + timedelta(days=i) for i in range(10)]

    assert star_growth_burstiness(timestamps) == star_growth_burstiness(list(reversed(timestamps)))


def _contributor(*, weeks_ago: list[int], author_type: str = "User") -> ContributorStats:
    now = datetime.now(UTC)
    weeks = [ContributorWeek(w=now - timedelta(weeks=w), c=5) for w in weeks_ago]
    return ContributorStats(total=sum(week.c for week in weeks), weeks=weeks, author=Owner(login="a", type=author_type))


def test_contributor_activity_ratio_returns_none_when_empty() -> None:
    assert contributor_activity_ratio([]) is None


def test_contributor_activity_ratio_counts_only_recent_committers() -> None:
    active = _contributor(weeks_ago=[1, 2])
    inactive = _contributor(weeks_ago=[52, 60])

    ratio = contributor_activity_ratio([active, active, inactive], active_window_weeks=12)

    assert ratio == 2 / 3


def test_contributor_activity_ratio_respects_custom_window() -> None:
    contributor = _contributor(weeks_ago=[20])

    assert contributor_activity_ratio([contributor], active_window_weeks=12) == 0.0
    assert contributor_activity_ratio([contributor], active_window_weeks=30) == 1.0


def test_is_bot_contributor_checks_author_type() -> None:
    assert is_bot_contributor(_contributor(weeks_ago=[1], author_type="Bot")) is True
    assert is_bot_contributor(_contributor(weeks_ago=[1], author_type="User")) is False


def test_is_bot_contributor_handles_unresolved_author() -> None:
    unresolved = ContributorStats(total=1, weeks=[], author=None)

    assert is_bot_contributor(unresolved) is False


def test_contributor_activity_ratio_excludes_bots_by_default() -> None:
    active_bot = _contributor(weeks_ago=[1], author_type="Bot")
    inactive_human = _contributor(weeks_ago=[52])

    # The only human contributor is inactive; a busy bot should not paper over that.
    assert contributor_activity_ratio([active_bot, inactive_human], active_window_weeks=12) == 0.0
    assert contributor_activity_ratio([active_bot], active_window_weeks=12) is None


def test_contributor_activity_ratio_can_include_bots() -> None:
    active_bot = _contributor(weeks_ago=[1], author_type="Bot")

    assert contributor_activity_ratio([active_bot], active_window_weeks=12, include_bots=True) == 1.0


def _user(*, public_repos: int, followers: int, type_: str = "User") -> User:
    return User(
        login="u",
        type=type_,
        created_at=EPOCH,
        public_repos=public_repos,
        followers=followers,
        following=0,
    )


def test_is_ghost_account_flags_zero_presence() -> None:
    assert is_ghost_account(_user(public_repos=0, followers=0)) is True


def test_is_ghost_account_does_not_flag_established_accounts() -> None:
    assert is_ghost_account(_user(public_repos=12, followers=40)) is False
    assert is_ghost_account(_user(public_repos=1, followers=0)) is False
    assert is_ghost_account(_user(public_repos=0, followers=1)) is False


def test_is_ghost_account_never_flags_bots_or_orgs() -> None:
    # dependabot[bot] and similar automation naturally has zero followers and zero personal
    # repos; that is expected, not evidence of a disposable account.
    assert is_ghost_account(_user(public_repos=0, followers=0, type_="Bot")) is False
    assert is_ghost_account(_user(public_repos=0, followers=0, type_="Organization")) is False


def test_ghost_contributor_ratio_returns_none_when_empty() -> None:
    assert ghost_contributor_ratio([]) is None


def test_ghost_contributor_ratio_averages_over_all_contributors() -> None:
    ghosts = [_user(public_repos=0, followers=0)] * 3
    real = [_user(public_repos=10, followers=50)]

    assert ghost_contributor_ratio(ghosts + real) == 0.75
