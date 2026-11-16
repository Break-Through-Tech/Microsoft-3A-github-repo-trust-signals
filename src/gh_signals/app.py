"""One-page app that scores a public GitHub repository URL.

Run it with:

    uv run python -m gh_signals.app

Then open http://127.0.0.1:8000
The walkthrough is at http://127.0.0.1:8000/slides

Score one repository in the terminal with:

    uv run trust-score https://github.com/microsoft/vscode
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from gh_signals.feedback import FeedbackError, record_feedback
from gh_signals.github_client import GitHubError
from gh_signals.score import MODEL_MISSING_MESSAGE, RepoURLError, ScoreResult, github_failure_message, score_repository

WEB_INDEX = Path(__file__).parents[2] / "web" / "index.html"
WEB_SLIDES = Path(__file__).parents[2] / "web" / "slides.html"

app = FastAPI(title="Repository trust score")


class ScoreRequest(BaseModel):
    """JSON body for ``POST /api/score``."""

    url: str


class FeedbackRequest(BaseModel):
    """JSON body for ``POST /api/feedback``.

    The note is stored locally and is not used to train or change a score.
    """

    full_name: str
    score: int | None
    tier: str
    judgment: str
    comment: str = ""


@app.get("/")
async def index() -> FileResponse:
    """Serve the single page."""
    return FileResponse(WEB_INDEX, headers={"Cache-Control": "no-cache"})


@app.get("/slides")
async def slides() -> FileResponse:
    """Serve the walkthrough of the score, the features, and the model."""
    return FileResponse(WEB_SLIDES, headers={"Cache-Control": "no-cache"})


@app.post("/api/score")
async def score(body: ScoreRequest) -> ScoreResult:
    """Score one public repository and return the tier plus the signal breakdown.

    Args:
        body: The pasted URL or ``owner/name``.

    Returns:
        The score payload.

    Raises:
        HTTPException: 400 for a bad URL, 404 when GitHub has no such repository, 429 on rate
            limit, 502 when GitHub fails, and 500 when the trained model file is missing.
    """
    try:
        return await score_repository(body.url)
    except RepoURLError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except GitHubError as exc:
        raise HTTPException(status_code=_github_status(exc.status), detail=github_failure_message(exc.status)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=MODEL_MISSING_MESSAGE) from exc


@app.post("/api/feedback")
async def feedback(body: FeedbackRequest) -> dict[str, bool]:
    """Save a reviewer's note about a score they already saw.

    Args:
        body: The repository, the score that was shown, and the reviewer's judgment.

    Returns:
        ``{"saved": True}`` after the note is appended.

    Raises:
        HTTPException: 400 when the note is missing a repository or uses an unknown judgment.
    """
    try:
        record_feedback(
            full_name=body.full_name,
            score=body.score,
            tier=body.tier,
            judgment=body.judgment,
            comment=body.comment,
        )
    except FeedbackError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"saved": True}


def _github_status(status: int) -> int:
    if status == 404:
        return 404
    if status in (403, 429):
        return 429
    return 502


def main() -> None:
    """Start the local server."""
    import uvicorn

    uvicorn.run("gh_signals.app:app", host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
