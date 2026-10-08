"""Async client for the GitHub REST API.

Example:
    async with GitHubClient() as client:
        repo = await client.get_repo("microsoft/vscode")
        print(repo.stargazers_count, repo.forks_count)
"""

import asyncio
from collections.abc import Mapping
from datetime import datetime
import json
import math
import os
from pathlib import Path
import time
from types import TracebackType
from typing import Any
from typing_extensions import Self

import aiohttp

from gh_signals.schemas import Repo, StarWeek
from gh_signals.schemas import ContributorStats, Repo, Stargazer, User

API_URL = "https://api.github.com"
DEFAULT_CACHE_DIR = Path(__file__).parents[2] / ".cache" / "github"
STAR_TIMESTAMP_ACCEPT = "application/vnd.github.star+json"


class GitHubError(Exception):
    """Raised when the GitHub API returns an unsuccessful response."""

    def __init__(self, status: int, path: str, body: str) -> None:
        super().__init__(f"GitHub returned {status} for {path}: {body}")
        self.status = status
        self.path = path


class GitHubClient:
    """Fetches repository metadata, caching responses and respecting rate limits."""

    def __init__(
        self,
        token: str | None = None,
        cache_dir: Path | None = DEFAULT_CACHE_DIR,
        cache_ttl_seconds: float = 24 * 60 * 60,
        min_remaining: int = 10,
    ) -> None:
        """Initializes the client.

        Args:
            token: A GitHub token
            cache_dir: Directory for cached responses. Pass ``None`` to disable caching.
            cache_ttl_seconds: How long a cached response stays fresh. Defaults to 24 hours.
            min_remaining: How many requests to leave unused before waiting for the hourly quota to reset. Useful to prevent totally exhausting your GitHub usage.
        """
        self._token = token
        self._cache_dir = cache_dir
        self._cache_ttl_seconds = cache_ttl_seconds
        self._min_remaining = min_remaining
        self._remaining = min_remaining + 1  # Unknown until the first response arrives.
        self._reset_at = 0.0
        self._session: aiohttp.ClientSession | None = None

    async def get_repo(self, repo: str) -> Repo:
        """Fetches the star and fork counts for a repository.

        Args:
            repo: A repository in ``owner/name`` form, such as ``microsoft/vscode``.

        Returns:
            The repository's current counts.

        Raises:
            GitHubError: If GitHub rejects the request, such as when the repository does not exist or is private.
        """
        response = await self._get(f"repos/{repo}")
        return Repo.model_validate(response)

    async def get_star_history(self, repo: str) -> list[StarWeek]:
        """Fetches a repository's daily star counts, grouped by week, most recent first."""
        weeks: list[StarWeek] = []
        for page in range(1, 101):
            response = await self._get(f"repos/{repo}/stargazers/history?per_page=30&page={page}")
            if not response:
                break
            weeks.extend(StarWeek.model_validate(week) for week in response)
        return weeks
    async def get_stargazer_timestamps(
        self, repo: str, stargazers_count: int, max_pages: int = 10, per_page: int = 100
    ) -> list[datetime]:
        """Fetches when a repo's most recent stargazers starred it.

        GitHub returns stargazers oldest-first with no way to reverse that order, so this jumps
        straight to the last page(s) using the already-known star count, rather than paging
        through the entire history to reach the end. That keeps the request count bounded
        regardless of how many stars a repo has, which matters for burstiness: a sudden recent
        spike is visible in the newest stars, not the oldest.

        As of GitHub's July 2026 starring-API changes, this endpoint is restricted to a repo's
        admins and collaborators (see
        `the docs <https://docs.github.com/rest/activity/starring>`_); calling it for a repo we do
        not have that access to raises :class:`GitHubError` with ``status=404``, even though the
        repo itself is public. GraphQL's equivalent field is subject to the same restriction. There
        is currently no way to fetch per-star timestamps for an arbitrary public repo through
        GitHub's API; callers should catch :class:`GitHubError` around this call.

        Args:
            repo: A repository in ``owner/name`` form.
            stargazers_count: The repo's current star count, e.g. from ``get_repo``.
            max_pages: How many of the most recent pages to fetch.
            per_page: Stars per page (GitHub's max is 100).

        Returns:
            Timestamps of the fetched stargazers, oldest first. Empty if the repo has no stars.

        Raises:
            GitHubError: If GitHub rejects the request, in particular ``status=404`` when we are
                not a collaborator on ``repo``.
        """
        last_page = math.ceil(stargazers_count / per_page) if stargazers_count > 0 else 0
        first_page = max(1, last_page - max_pages + 1)
        headers = {"Accept": STAR_TIMESTAMP_ACCEPT}

        async def fetch_page(page: int) -> list[Stargazer]:
            payload = await self._get(f"repos/{repo}/stargazers?per_page={per_page}&page={page}", headers=headers)
            return [Stargazer.model_validate(item) for item in payload]

        pages = await asyncio.gather(*(fetch_page(page) for page in range(first_page, last_page + 1)))
        return [stargazer.starred_at for page in pages for stargazer in page]

    async def get_contributor_stats(
        self, repo: str, max_attempts: int = 6, retry_delay_seconds: float = 2.0
    ) -> list[ContributorStats]:
        """Fetches each contributor's all-time commit total and weekly commit history.

        GitHub computes these stats asynchronously: a repo that has not been queried recently
        returns an empty list while it works, so this polls with a short delay until the stats
        are ready or ``max_attempts`` is reached. An empty (still-computing) response is never
        cached, so a later call retries from scratch instead of reusing a stale empty result.

        Args:
            repo: A repository in ``owner/name`` form.
            max_attempts: How many times to poll before giving up.
            retry_delay_seconds: How long to wait between polls.

        Returns:
            One entry per contributor, or an empty list if the stats never became ready.
        """
        path = f"repos/{repo}/stats/contributors"
        cached = self._read_cache(path)
        if cached is not None:
            return [ContributorStats.model_validate(item) for item in cached]

        for attempt in range(max_attempts):
            payload = await self._request(path)
            if payload:
                self._write_cache(path, payload)
                return [ContributorStats.model_validate(item) for item in payload]
            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay_seconds)
        return []

    async def get_user(self, username: str) -> User:
        """Fetches a GitHub user's public profile.

        Args:
            username: A GitHub login.

        Returns:
            The user's profile fields.

        Raises:
            GitHubError: If GitHub rejects the request, such as when the account does not exist.
        """
        response = await self._get(f"users/{username}")
        return User.model_validate(response)

    async def __aenter__(self) -> Self:
        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        token = await self._resolve_token()
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        self._session = aiohttp.ClientSession(headers=headers)
        return self

    async def _resolve_token(self) -> str | None:
        """Returns the first token available from the constructor, the environment, or the GitHub CLI."""
        return self._token or os.environ.get("GITHUB_TOKEN") or await _gh_cli_token()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def _get(self, path: str) -> dict[str, Any] | list[Any]:
    async def _get(self, path: str, headers: Mapping[str, str] | None = None) -> Any:
        """Returns the JSON payload for an API path, from cache when it is still fresh."""
        cached = self._read_cache(path)
        if cached is not None:
            return cached
        payload = await self._request(path, headers=headers)
        self._write_cache(path, payload)
        return payload

    async def _request(self, path: str, allow_retry: bool = True) -> dict[str, Any] | list[Any]:
        """Issues a single GET, retrying once if GitHub asks us to back off."""
    async def _request(self, path: str, allow_retry: bool = True, headers: Mapping[str, str] | None = None) -> Any:
        """Issues a single GET, retrying once if GitHub asks us to back off or the connection drops mid-response.

        A pooled connection occasionally gets closed partway through a large response (seen on
        ``stats/contributors`` for repos with a long contributor history) under concurrent load;
        that surfaces as ``aiohttp.ClientPayloadError`` rather than an HTTP error status, so it
        needs its own retry alongside the rate-limit one.
        """
        session = self._session
        if session is None:
            raise RuntimeError("Use GitHubClient as an async context manager: async with GitHubClient() as client")

        await self._wait_for_quota()
        try:
            async with session.get(f"{API_URL}/{path}", headers=headers) as response:
                self._record_quota(response.headers)
                retry_after = response.headers.get("Retry-After")
                if allow_retry and retry_after is not None and response.status in (403, 429):
                    await asyncio.sleep(float(retry_after))
                    return await self._request(path, allow_retry=False, headers=headers)
                if not response.ok:
                    raise GitHubError(response.status, path, await response.text())
                return await response.json()
        except aiohttp.ClientPayloadError:
            if not allow_retry:
                raise
            return await self._request(path, allow_retry=False, headers=headers)

    def _record_quota(self, headers: Mapping[str, str]) -> None:
        """Stores how much of the hourly quota is left, per GitHub's response headers."""
        remaining = headers.get("x-ratelimit-remaining")
        reset_at = headers.get("x-ratelimit-reset")
        if remaining is not None and reset_at is not None:
            self._remaining = int(remaining)
            self._reset_at = float(reset_at)

    async def _wait_for_quota(self) -> None:
        """Sleeps until the quota resets if we are close to exhausting it.

        Concurrent callers can read a stale count and slip past this check together, which is why ``min_remaining``
        holds back a buffer of requests rather than aiming to spend the quota down to zero.
        """
        if self._remaining > self._min_remaining:
            return
        delay = self._reset_at - time.time()
        if delay > 0:
            await asyncio.sleep(delay)
        self._remaining = self._min_remaining + 1  # Assume the quota refilled.

    def _cache_path(self, path: str) -> Path | None:
        """Maps an API path to a cache file, so ``repos/foo/bar`` becomes ``repos_foo_bar.json``."""
        if self._cache_dir is None:
            return None
        return self._cache_dir / f"{path.replace('/', '_')}.json"

    def _read_cache(self, path: str) -> dict[str, Any] | list[Any] | None:
    def _read_cache(self, path: str) -> Any | None:
        cache_path = self._cache_path(path)
        if cache_path is None or not cache_path.exists():
            return None
        if time.time() - cache_path.stat().st_mtime > self._cache_ttl_seconds:
            return None
        with cache_path.open(encoding="utf-8") as file:
            return json.load(file)

    def _write_cache(self, path: str, payload: dict[str, Any] | list[Any]) -> None:
    def _write_cache(self, path: str, payload: Any) -> None:
        cache_path = self._cache_path(path)
        if cache_path is None:
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file)


async def _gh_cli_token() -> str | None:
    """Returns the token from ``gh auth token``.

    Returns:
        The token, or ``None`` when the GitHub CLI is not installed or not logged in.
    """
    try:
        process = await asyncio.create_subprocess_exec(
            "gh",
            "auth",
            "token",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
    except (FileNotFoundError, NotImplementedError):
        return None
    stdout, _ = await process.communicate()
    if process.returncode != 0:
        return None
    return stdout.decode().strip() or None
