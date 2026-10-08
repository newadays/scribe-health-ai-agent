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

    # Set by triage: "pediatrics" or "general_medicine".
    specialty: str
    triage_reason: str
    context_summary: str

    # Output: the generated clinical note.
    note: str
