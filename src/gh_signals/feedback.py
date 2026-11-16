"""Local notes about scores.

Notes are for later review of the scorer. Training does not read this file.
"""

from datetime import UTC, datetime
import json
from pathlib import Path

FEEDBACK_PATH = Path(__file__).parents[2] / "data" / "feedback.jsonl"
JUDGMENTS = ("about-right", "too-high", "too-low")
_MAX_COMMENT_LENGTH = 500
_MAX_NAME_LENGTH = 200


class FeedbackError(ValueError):
    """Raised when a note is missing a repository or uses an unknown judgment."""


def record_feedback(
    *,
    full_name: str,
    score: int | None,
    tier: str,
    judgment: str,
    comment: str = "",
    path: Path | None = None,
) -> dict[str, object]:
    """Append one review note.

    Args:
        full_name: Repository in ``owner/name`` form.
        score: The score the reviewer saw, or ``None`` when it was withheld.
        tier: Tier label stored with the note, such as ``established``.
        judgment: One of ``about-right``, ``too-high``, or ``too-low``.
        comment: Optional free-text note, at most 500 characters.
        path: File to append. Defaults to :data:`FEEDBACK_PATH`.

    Returns:
        The record that was written.

    Raises:
        FeedbackError: If the name, judgment, or comment is not usable.
    """
    name = full_name.strip()
    if not name or len(name) > _MAX_NAME_LENGTH or any(character in name for character in "\n\r"):
        raise FeedbackError("Enter the repository as owner/name.")
    if judgment not in JUDGMENTS:
        raise FeedbackError("Judgment must be about-right, too-high, or too-low.")
    note = comment.strip()
    if len(note) > _MAX_COMMENT_LENGTH:
        raise FeedbackError("Keep the note to 500 characters.")
    record: dict[str, object] = {
        "full_name": name,
        "score": score,
        "tier": tier,
        "judgment": judgment,
        "comment": note,
        "recorded_at": datetime.now(UTC).isoformat(),
    }
    destination = FEEDBACK_PATH if path is None else path
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")
    return record
