from langchain_core.messages import AIMessage
from agents.state import ScribeState


def human_review(state: ScribeState) -> dict:
    """Terminal node for transcripts the router would not auto-route.

    No LLM call. In a real system this would open a task for a clinician.
    """

    message = (
        "[Human review] Not auto-routed. "
        f"specialty={state['specialty']} "
        f"confidence={state['triage_confidence']:.2f} "
        f"needs_review={state['needs_review']:.2f} "
        f"reason: {state['triage_reason']}"
    )

    return {
        "messages": [AIMessage(content=message)],
        "specialty": "human_review_complete",
        "note": "",
    }
