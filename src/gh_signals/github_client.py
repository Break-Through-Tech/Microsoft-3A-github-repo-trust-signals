"""Async client for the GitHub REST API.

Example:
    async with GitHubClient() as client:
        repo = await client.get_repo("microsoft/vscode")
        print(repo.stargazers_count, repo.forks_count)
"""

import asyncio
from collections.abc import Mapping
import json
import os
from pathlib import Path
import time
from types import TracebackType
from typing import Any
from typing_extensions import Self

import aiohttp

from gh_signals.schemas import Repo, StarWeek

API_URL = "https://api.github.com"
DEFAULT_CACHE_DIR = Path(__file__).parents[2] / ".cache" / "github"


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
        """Returns the JSON payload for an API path, from cache when it is still fresh."""
        cached = self._read_cache(path)
        if cached is not None:
            return cached
        payload = await self._request(path)
        self._write_cache(path, payload)
        return payload

    async def _request(self, path: str, allow_retry: bool = True) -> dict[str, Any] | list[Any]:
        """Issues a single GET, retrying once if GitHub asks us to back off."""
        session = self._session
        if session is None:
            raise RuntimeError("Use GitHubClient as an async context manager: async with GitHubClient() as client")

        await self._wait_for_quota()
        async with session.get(f"{API_URL}/{path}") as response:
            self._record_quota(response.headers)
            retry_after = response.headers.get("Retry-After")
            if allow_retry and retry_after is not None and response.status in (403, 429):
                await asyncio.sleep(float(retry_after))
                return await self._request(path, allow_retry=False)
            if not response.ok:
                raise GitHubError(response.status, path, await response.text())
            return await response.json()

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
        cache_path = self._cache_path(path)
        if cache_path is None or not cache_path.exists():
            return None
        if time.time() - cache_path.stat().st_mtime > self._cache_ttl_seconds:
            return None
        with cache_path.open(encoding="utf-8") as file:
            return json.load(file)

    def _write_cache(self, path: str, payload: dict[str, Any] | list[Any]) -> None:
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
