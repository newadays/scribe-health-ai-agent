"""Triage backend 1: OpenAI LLM with structured output.

The LLM reads the transcript, picks a specialty, and also writes a
free-text reason and context summary. Compare with agents/triage_jev.py,
which makes the same decision with a dedicated decision model.
"""

from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from agents.state import ScribeState
from agents.llm import llm


class TriageDecision(BaseModel):
    specialty: Literal["pediatrics", "general_medicine"] = Field(
        description="Specialty agent that should write the note"
    )

    reason: str = Field(
        description="Why this specialty was selected"
    )

    context: str = Field(
        description="Key facts from the transcript the specialist should know "
                    "(patient name, age, chief complaint, vitals, weight)"
    )


def triage_llm(state: ScribeState) -> dict:
    """Read the transcript and decide which specialty agent should write the note."""

    system_prompt = """
    You are a medical scribe triage agent.

    You will receive the transcript of a clinical encounter.
    Decide which specialty agent should generate the clinical note.

    Routing rules:

    PEDIATRICS:
    - Patient is a child or adolescent (under 18 years old)
    - Mentions of parent/guardian speaking for the patient
    - Well-child visits, growth, immunizations, weight-based dosing

    GENERAL_MEDICINE:
    - Adult patients
    - Routine, chronic, or acute adult complaints
    - Anything not clearly pediatric

    Return the specialty, the reason, and a short context summary.
    """

    structured_llm = llm.with_structured_output(TriageDecision)

    decision = structured_llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=state["transcript"]),
        ]
    )

    return {
        "specialty": decision.specialty,
        "triage_reason": decision.reason,
        "context_summary": decision.context,
        "triage_backend": "llm",
        # The LLM gives no calibrated confidence, so the router gate is bypassed.
        "triage_confidence": 1.0,
        "needs_review": 0.0,
        "acuity": 0.0,
        "messages": [
            AIMessage(
                content=(
                    f"[Triage/LLM] Routing transcript to {decision.specialty.upper()} agent. "
                    f"Reason: {decision.reason}"
                )
            )
        ],
    }


# Backwards-compatible name used earlier in the tutorial.
triage_agent = triage_llm
