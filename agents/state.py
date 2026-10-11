from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ScribeState(TypedDict):
    """Shared state that flows through the medical scribe graph."""

    # Conversation log (triage + specialist messages). add_messages appends.
    messages: Annotated[list[BaseMessage], add_messages]

    # Input: raw text of the doctor-patient encounter.
    transcript: str

    # Set by triage: "pediatrics", "general_medicine", or "human_review".
    specialty: str
    triage_reason: str
    context_summary: str

    # Which decision backend made the triage call: "llm" or "jev".
    triage_backend: str

    # Decision signals used by the router's confidence gate.
    # The LLM backend fills these with safe defaults (1.0 / 0.0 / 0.0).
    # The Jev backend fills them from real model probabilities.
    triage_confidence: float   # 0..1, confidence in the chosen specialty
    needs_review: float        # 0..1, probability the transcript is not a usable encounter
    acuity: float              # 0 routine, 1 prompt follow-up, 2 urgent

    # Output: the generated clinical note.
    note: str
