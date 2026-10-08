from typing import Literal
from agents.state import ScribeState


def route_after_triage(state: ScribeState) -> Literal["pediatrics", "general_medicine"]:
    """Pick the next node based on the triage decision."""

    if state["specialty"] == "pediatrics":
        return "pediatrics"

    return "general_medicine"
