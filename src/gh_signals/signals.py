"""Trust signals computed from GitHub metadata.

Every function here is a pure computation over already-fetched data (timestamps, contributor
stats, user profiles). Fetching that data is the caller's job, via ``gh_signals.github_client``;
keeping the two separate makes the signals themselves fast to test without hitting the network.
"""

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from itertools import pairwise
import statistics

from gh_signals.schemas import ContributorStats, User


def star_growth_burstiness(timestamps: Sequence[datetime]) -> float | None:
    """Scores how clustered a repo's stars are in time, using the Goh-Barabasi burstiness parameter.

    Computed from the gaps between consecutive stars (sorted oldest to newest): ``B = (stdev -
    mean) / (stdev + mean)`` of those gaps. A steady trickle of stars has gaps of similar size and
    scores near -1; a Poisson-like random arrival process scores near 0; a repo that got almost
    all of its stars in one or two tight bursts, separated by long quiet stretches, scores near 1.

    A sudden coordinated star campaign looks like the bursty end of this scale, but so does a
    single organic viral moment (a popular blog post, a conference talk), so this should be read
    alongside other signals such as :func:`gh_signals.signals.is_ghost_account`, not alone.

    Args:
        timestamps: When each star was given. Order does not matter, duplicates are fine.

    Returns:
        A value in ``[-1, 1]``, or ``None`` if fewer than three timestamps are given, too few to
        define a gap distribution.
    """
    if len(timestamps) < 3:
        return None
    ordered = sorted(timestamps)
    gaps = [(later - earlier).total_seconds() for earlier, later in pairwise(ordered)]
    mean_gap = statistics.mean(gaps)
    if mean_gap == 0:
        return 1.0
    stdev_gap = statistics.pstdev(gaps)
    return (stdev_gap - mean_gap) / (stdev_gap + mean_gap)


def is_bot_contributor(contributor: ContributorStats) -> bool:
    """True if the commits are attributed to a bot account (CI, dependency, or release automation).

    GitHub's own automation (Dependabot, release-please, and similar) shows up in a repo's
    contributor stats alongside human contributors. Bots naturally commit on a regular schedule
    and naturally have no followers or personal repos, so leaving them in would make
    :func:`contributor_activity_ratio` overstate human maintenance and :func:`is_ghost_account`
    misclassify routine automation as a suspicious account.
    """
    return contributor.author is not None and contributor.author.type == "Bot"


def contributor_activity_ratio(
    contributors: Sequence[ContributorStats],
    *,
    active_window_weeks: int = 12,
    now: datetime | None = None,
    include_bots: bool = False,
) -> float | None:
    """Fraction of a repo's all-time human contributors who have committed recently.

    A project can accumulate a long list of one-time or drive-by contributors over the years
    while its actual maintenance has stalled; this measures how much of that historical
    contributor list is still doing anything, independent of the repo's star or fork counts.

    Args:
        contributors: One entry per contributor, e.g. from ``GitHubClient.get_contributor_stats``.
        active_window_weeks: How recent a contributor's last commit must be to count as active.
        now: The reference time to measure recency from. Defaults to the current time.
        include_bots: If True, count bot-authored commits (see :func:`is_bot_contributor`) too.
            Off by default since a busy bot can make an unmaintained repo look active.

    Returns:
        The active fraction in ``[0, 1]``, or ``None`` if there are no (human, unless
        ``include_bots``) contributors to measure.
    """
    people = contributors if include_bots else [c for c in contributors if not is_bot_contributor(c)]
    if not people:
        return None
    now = now or datetime.now(UTC)
    cutoff = now - timedelta(weeks=active_window_weeks)
    active = sum(1 for contributor in people if any(week.c > 0 and week.w >= cutoff for week in contributor.weeks))
    return active / len(people)


def is_ghost_account(user: User, *, max_public_repos: int = 0, max_followers: int = 0) -> bool:
    """Flags a GitHub account with essentially no public presence beyond its commits.

    Mirrors the zero-follower, zero-repo stargazer profile from the fake-star literature (see
    ``docs/literature_review.md``), applied to contributors instead of stargazers: an account that
    has never followed or been followed by anyone, and has never published a repo of its own, is
    consistent with a disposable account created only to pad a contribution list.

    This checks visible presence only, not account age. The literature also describes a premium
    tier of purchased accounts that are deliberately aged with fabricated history specifically to
    defeat "new account" heuristics, so a ghost-like account by this definition can still be old;
    treat this as one signal to combine with others, not a standalone verdict.

    Bot and organization accounts are never flagged: a bot having no followers or personal repos
    is normal, not suspicious, since it was never meant to look like a person.

    Args:
        user: The account to check, e.g. from ``GitHubClient.get_user``.
        max_public_repos: The highest public repo count that still counts as "no presence".
        max_followers: The highest follower count that still counts as "no presence".

    Returns:
        True if the account looks ghost-like.
    """
    if user.type != "User":
        return False
    return user.public_repos <= max_public_repos and user.followers <= max_followers


def ghost_contributor_ratio(
    contributors: Sequence[User], *, max_public_repos: int = 0, max_followers: int = 0
) -> float | None:
    """Fraction of the given contributors that look like ghost accounts.

    A high ratio here, especially on a repo with otherwise few contributors, suggests the
    contribution history itself may have been padded, not just its stars.

    Args:
        contributors: The contributor profiles to check, e.g. from ``GitHubClient.get_user``.
        max_public_repos: Forwarded to :func:`is_ghost_account`.
        max_followers: Forwarded to :func:`is_ghost_account`.

    Returns:
        The ghost-like fraction in ``[0, 1]``, or ``None`` if ``contributors`` is empty.
    """
    if not contributors:
        return None
    ghost_count = sum(
        is_ghost_account(user, max_public_repos=max_public_repos, max_followers=max_followers) for user in contributors
    )
    return ghost_count / len(contributors)
