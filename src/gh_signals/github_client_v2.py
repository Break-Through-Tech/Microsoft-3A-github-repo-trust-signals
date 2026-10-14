"""Async client for the GitHub REST API.

Example:
    async with GitHubClient() as client:
        repo = await client.get_repo("microsoft/vscode")
        print(repo.stargazers_count, repo.forks_count)
"""

import asyncio
import json
import os
from pathlib import Path
import subprocess
import time
from types import TracebackType
from typing import Any, Self

import aiohttp

from gh_signals.schemas import Repo

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
        max_retries: int = 5,
        max_concurrent: int = 10,
    ) -> None:
        """Initializes the client.

        Args:
            token: A GitHub token
            cache_dir: Directory for cached responses. Pass ``None`` to disable caching.
            cache_ttl_seconds: How long a cached response stays fresh. Defaults to 24 hours.
            min_remaining: How many requests to leave unused before waiting for the hourly quota to reset.
            max_retries: How many times to retry a failed request before giving up.
            max_concurrent: How many requests can be in flight at once.
        """
        self._token = token
        self._cache_dir = cache_dir
        self._cache_ttl_seconds = cache_ttl_seconds
        self._min_remaining = min_remaining
        self._max_retries = max_retries
        self._max_concurrent = max_concurrent
        self._remaining = min_remaining + 1  # Unknown until the first response arrives.
        self._reset_at = 0.0  # Unix time when GitHub refills our hourly quota.
        self._session: aiohttp.ClientSession | None = None
        self._semaphore: asyncio.Semaphore | None = None

    async def get_repo(self, repo: str) -> Repo:
        """Fetches the star and fork counts for a repository.

        Args:
            repo: A repository in ``owner/name`` form, such as ``microsoft/vscode``.

        Raises:
            GitHubError: If GitHub rejects the request, such as when the repository does not exist or is private.
        """
        return Repo.model_validate(await self.get_repo_raw(repo))

    async def get_repo_raw(self, repo: str) -> dict[str, Any]:
        """Fetches the full JSON for a repository, from cache when it is still fresh."""
        path = f"repos/{repo}"

        # 1. Already saved on disk recently? Then we don't need to call GitHub at all.
        cached = self._read_cache(path)
        if cached is not None:
            return cached

        # 2. Otherwise ask GitHub, and 3. save the answer so we never fetch it twice.
        payload = await self._request(path)
        self._write_cache(path, payload)
        return payload

    async def __aenter__(self) -> Self:
        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}

        # Use the first token we can find. With a token we get 5,000 requests/hour instead of 60.
        token = self._token or os.environ.get("GITHUB_TOKEN") or await _gh_cli_token()
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        else:
            print("Warning: no GitHub token found, you are limited to 60 requests per hour.")

        # One session is reused for every request. The timeout stops a stuck request from hanging forever.
        self._session = aiohttp.ClientSession(headers=headers, timeout=aiohttp.ClientTimeout(total=30))
        # The semaphore caps how many requests run at once so we don't flood GitHub.
        self._semaphore = asyncio.Semaphore(self._max_concurrent)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def _request(self, path: str) -> dict[str, Any]:
        """Issues a GET, retrying with a growing delay if GitHub errors or asks us to slow down."""
        if self._session is None or self._semaphore is None:
            raise RuntimeError("Use GitHubClient as an async context manager: async with GitHubClient() as client")

        # Try once, then retry up to max_retries more times.
        for attempt in range(self._max_retries + 1):
            # Only max_concurrent requests can be inside this block at the same time.
            async with self._semaphore:
                await self._wait_for_quota()
                try:
                    async with self._session.get(f"{API_URL}/{path}") as response:
                        self._record_quota(response.headers)
                        status = response.status
                        headers = response.headers
                        body = await response.text()
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    # Network problem, wait a bit and try again (1s, 2s, 4s, ...).
                    if attempt == self._max_retries:
                        raise
                    await asyncio.sleep(2**attempt)
                    continue

            # 200 means it worked, so we're done.
            if status == 200:
                return json.loads(body)

            # Something went wrong. Work out how long to wait, or give up if retrying won't help.
            if status in (403, 429) and "Retry-After" in headers:
                # GitHub told us exactly how long to wait.
                wait = float(headers["Retry-After"])
            elif status in (403, 429) and headers.get("x-ratelimit-remaining") == "0":
                # We ran out of requests, wait until the quota resets (plus a second to be safe).
                wait = max(self._reset_at - time.time(), 0) + 1
            elif status >= 500:
                # GitHub had a server problem, back off a little longer each time.
                wait = 2**attempt
            else:
                # Things like 404 (repo doesn't exist). Retrying would give the same answer.
                raise GitHubError(status, path, body)

            if attempt == self._max_retries:
                raise GitHubError(status, path, body)
            print(f"Got {status} for {path}, retrying in {wait:.0f}s")
            await asyncio.sleep(wait)

        raise RuntimeError("unreachable")  # The loop always returns or raises before this.

    def _record_quota(self, headers: Any) -> None:
        """Stores how much of the hourly quota is left, per GitHub's response headers."""
        remaining = headers.get("x-ratelimit-remaining")
        reset_at = headers.get("x-ratelimit-reset")
        if remaining is not None and reset_at is not None:
            self._remaining = int(remaining)
            self._reset_at = float(reset_at)

    async def _wait_for_quota(self) -> None:
        """Sleeps until the quota resets if we are close to exhausting it."""
        if self._remaining > self._min_remaining:
            return  # Plenty left, carry on.
        delay = self._reset_at - time.time()
        if delay > 0:
            print(f"Rate limit almost used up, sleeping {delay:.0f}s until it resets")
            await asyncio.sleep(delay)
        self._remaining = self._min_remaining + 1  # Assume the quota refilled.

    def _cache_path(self, path: str) -> Path | None:
        """Maps an API path to a cache file, so ``repos/foo/bar`` becomes ``repos_foo_bar.json``."""
        if self._cache_dir is None:
            return None  # Caching is turned off.
        # Lowercase because GitHub treats Numpy/Numpy and numpy/numpy as the same repo.
        return self._cache_dir / f"{path.replace('/', '_').lower()}.json"

    def _read_cache(self, path: str) -> dict[str, Any] | None:
        cache_path = self._cache_path(path)
        if cache_path is None or not cache_path.exists():
            return None
        # Too old, so treat it as missing and fetch fresh data.
        if time.time() - cache_path.stat().st_mtime > self._cache_ttl_seconds:
            return None
        try:
            with cache_path.open(encoding="utf-8") as file:
                return json.load(file)
        except ValueError:
            return None  # Half-written or corrupt file, just fetch it again.

    def _write_cache(self, path: str, payload: dict[str, Any]) -> None:
        cache_path = self._cache_path(path)
        if cache_path is None:
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temp file first so a crash can't leave a broken cache file.
        temp_path = cache_path.with_suffix(".tmp")
        with temp_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file)
        temp_path.replace(cache_path)  # Renaming is instant, so the real file is never half-written.


async def _gh_cli_token() -> str | None:
    """Returns the token from ``gh auth token``, or ``None`` if the GitHub CLI is missing or not logged in."""
    try:
        # Run in a thread because asyncio's own subprocess support fails on Windows inside Jupyter.
        result = await asyncio.to_thread(subprocess.run, ["gh", "auth", "token"], capture_output=True, text=True)
    except FileNotFoundError:
        return None  # The gh command isn't installed (or isn't on the PATH).
    if result.returncode != 0:
        return None  # gh is installed but not logged in.
    return result.stdout.strip() or None