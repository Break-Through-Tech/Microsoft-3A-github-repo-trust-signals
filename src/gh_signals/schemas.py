"""Typed models for the parts of the GitHub API we consume."""

from datetime import datetime

from pydantic import BaseModel


class Owner(BaseModel):
    """The subset of a repository owner's fields we use, named exactly as the GitHub API names them."""

    login: str
    type: str


class License(BaseModel):
    """The subset of a repository license's fields we use, named exactly as the GitHub API names them."""

    key: str


class Repo(BaseModel):
    """The repository fields we use as trust signals, named exactly as the GitHub API names them.

    All fields below are returned by a single ``GET /repos/{owner}/{repo}`` call, so adding a field
    here never costs an extra request.
    """

    full_name: str
    owner: Owner
    stargazers_count: int
    forks_count: int
    subscribers_count: int
    open_issues_count: int
    size: int
    created_at: datetime
    pushed_at: datetime
    archived: bool
    disabled: bool
    fork: bool
    has_issues: bool
    language: str | None = None
    description: str | None = None
    license: License | None = None


class Stargazer(BaseModel):
    """A single star event, named exactly as the GitHub API names it.

    Only present in ``GET /repos/{owner}/{repo}/stargazers`` responses when the request sets an
    ``Accept: application/vnd.github.star+json`` header; without it GitHub omits ``starred_at``.
    """

    starred_at: datetime


class ContributorWeek(BaseModel):
    """One week of a contributor's commit activity, named exactly as the GitHub API names it."""

    w: datetime
    c: int


class ContributorStats(BaseModel):
    """A contributor's all-time commit total and weekly history, named exactly as the GitHub API names them.

    ``author`` is ``None`` when GitHub cannot resolve the commits to an account, e.g. a deleted
    user or commits authored with an email never linked to a GitHub login.
    """

    total: int
    weeks: list[ContributorWeek]
    author: Owner | None = None


class User(BaseModel):
    """The subset of a GitHub user's fields we use, named exactly as the GitHub API names them."""

    login: str
    type: str
    created_at: datetime
    public_repos: int
    followers: int
    following: int
