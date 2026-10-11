from typing import Literal
from agents.state import ScribeState

# Confidence-gated routing (see docs.typesafe.ai/patterns/confidence-routing).
# The answer tells you what; confidence tells you whether to act.
CONFIDENCE_THRESHOLD = 0.6   # below this, do not trust the specialty choice
REVIEW_THRESHOLD = 0.5       # above this, the text is probably not a usable encounter


def route_after_triage(state: ScribeState) -> Literal["pediatrics", "general_medicine", "human_review"]:
    """Pick the next node based on the triage decision and its confidence."""

    if state["specialty"] == "human_review":
        return "human_review"

    if state["needs_review"] > REVIEW_THRESHOLD:
        return "human_review"

    if state["triage_confidence"] < CONFIDENCE_THRESHOLD:
        return "human_review"

    if state["specialty"] == "pediatrics":
        return "pediatrics"

    return "general_medicine"
