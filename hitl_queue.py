"""
hitl_queue.py  —  Developer 2: Human-in-the-Loop Queue
=======================================================
Manages a persistent review queue stored in a local JSON file.
Reviewer actions: Approve, Edit, Reject, Need More Evidence.

PUBLIC API
----------
    from hitl_queue import (
        add_to_queue,
        get_queue,
        submit_review,
        get_pending_items,
        get_reviewed_items,
        clear_queue,
    )
"""

from __future__ import annotations

print("RUNNING HITL_QUEUE.PY")

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

# ── Config ────────────────────────────────────────────────────────────────────
QUEUE_FILE = Path("hitl_queue.json")

ReviewAction = Literal["Approve", "Edit", "Reject", "Need More Evidence"]
VALID_ACTIONS: set[str] = {"Approve", "Edit", "Reject", "Need More Evidence"}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_queue() -> list[dict]:
    if QUEUE_FILE.exists():
        try:
            return json.loads(QUEUE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
    return []


def _save_queue(queue: list[dict]) -> None:
    QUEUE_FILE.write_text(
        json.dumps(queue, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── Public API ────────────────────────────────────────────────────────────────

def add_to_queue(trust_card: dict) -> str:
    """
    Add a trust_card to the HITL queue.
    Automatically assigns a review_id and sets status to 'Pending'.

    Parameters
    ----------
    trust_card : dict
        The full trust_card produced by trust_engine.process_answer_package().

    Returns
    -------
    str  — The generated review_id (UUID4).
    """
    review_id = str(uuid.uuid4())
    entry = {
        "review_id":    review_id,
        "queued_at":    _now_iso(),
        "status":       "Pending",
        "action":       None,
        "reviewer_note": None,
        "reviewed_at":  None,
        "edited_answer": None,
        "trust_card":   trust_card,
    }
    queue = _load_queue()
    queue.append(entry)
    _save_queue(queue)
    return review_id


def get_queue() -> list[dict]:
    """Return the entire queue (all statuses)."""
    return _load_queue()


def get_pending_items() -> list[dict]:
    """Return only items with status='Pending'."""
    return [item for item in _load_queue() if item["status"] == "Pending"]


def get_reviewed_items() -> list[dict]:
    """Return only items that have been reviewed (status != 'Pending')."""
    return [item for item in _load_queue() if item["status"] != "Pending"]


def submit_review(
    review_id: str,
    action: ReviewAction,
    reviewer_note: str = "",
    edited_answer: str | None = None,
) -> dict:
    """
    Submit a reviewer decision for a queued item.

    Parameters
    ----------
    review_id     : str           — UUID of the item to review.
    action        : ReviewAction  — One of: Approve, Edit, Reject, Need More Evidence.
    reviewer_note : str           — Optional free-text comment from reviewer.
    edited_answer : str | None    — Required when action == "Edit".

    Returns
    -------
    dict — The updated queue entry.

    Raises
    ------
    ValueError — if review_id not found or action is invalid.
    """
    if action not in VALID_ACTIONS:
        raise ValueError(
            f"Invalid action '{action}'. Must be one of {sorted(VALID_ACTIONS)}."
        )

    queue = _load_queue()
    for item in queue:
        if item["review_id"] == review_id:
            if item["status"] != "Pending":
                raise ValueError(
                    f"Item '{review_id}' has already been reviewed "
                    f"(status: {item['status']})."
                )
            item["status"]       = "Reviewed"
            item["action"]       = action
            item["reviewer_note"] = reviewer_note
            item["reviewed_at"]  = _now_iso()
            if action == "Edit" and edited_answer:
                item["edited_answer"] = edited_answer
            _save_queue(queue)
            return item

    raise ValueError(f"review_id '{review_id}' not found in the queue.")


def get_item(review_id: str) -> dict | None:
    """Retrieve a single queue item by review_id. Returns None if not found."""
    for item in _load_queue():
        if item["review_id"] == review_id:
            return item
    return None


def clear_queue(confirmed: bool = False) -> int:
    """
    Delete all entries from the queue.
    Requires confirmed=True as a safety guard.
    Returns the number of items deleted.
    """
    if not confirmed:
        raise ValueError("Pass confirmed=True to clear the entire queue.")
    queue = _load_queue()
    count = len(queue)
    _save_queue([])
    return count


# ── Standalone demo ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Quick end-to-end test
    dummy_trust_card = {
        "confidence_score": 78,
        "confidence_band":  "Medium",
        "decision":         {"action": "Warn", "reason": "Medium confidence."},
        "issues":           ["Some citation issues."],
        "rag_output":       {"question": "Demo question", "answer": "Demo answer"},
    }

    rid = add_to_queue(dummy_trust_card)
    print(f"Added to queue: {rid}")

    pending = get_pending_items()
    print(f"Pending items: {len(pending)}")

    updated = submit_review(rid, "Approve", reviewer_note="Looks good!")
    print(f"Action taken: {updated['action']}")

    reviewed = get_reviewed_items()
    print(f"Reviewed items: {len(reviewed)}")
