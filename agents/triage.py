"""Triage backend 1: OpenAI LLM as a judge, via structured output.

The LLM reads the transcript and answers the same three questions that the
Jev backend answers (specialty, needs_review, acuity), plus a self-reported
confidence. Compare with agents/triage_jev.py, which gets calibrated
probabilities from a dedicated decision model instead of asking a generative
model how sure it feels.
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

    confidence: float = Field(
        ge=0.0, le=1.0,
        description="How confident you are in the specialty choice, from 0 to 1",
    )

    needs_review: bool = Field(
        description="True if the text is incomplete, ambiguous, or not a clinical "
                    "encounter between a clinician and a patient"
    )

    acuity: Literal[0, 1, 2] = Field(
        description="0 = routine visit, 1 = needs prompt follow-up within days, "
                    "2 = urgent, same-day action or escalation discussed"
    )

    reason: str = Field(
        description="Why this specialty was selected"
    )

    context: str = Field(
        description="Key facts from the transcript the specialist should know "
                    "(patient name, age, chief complaint, vitals, weight)"
    )


SYSTEM_PROMPT = """
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

Also report:
- confidence: how sure you are of the specialty, 0 to 1
- needs_review: true if the text is not really a clinical encounter
  (for example a scheduling call) or is too incomplete to document
- acuity: 0 routine, 1 needs prompt follow-up, 2 urgent

Return the specialty, confidence, needs_review, acuity, reason, and context.
"""


def triage_llm(state: ScribeState) -> dict:
    """Read the transcript and decide which specialty agent should write the note."""

    structured_llm = llm.with_structured_output(TriageDecision)

    decision = structured_llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=state["transcript"]),
        ]
    )

    needs_review = 1.0 if decision.needs_review else 0.0

    return {
        "specialty": decision.specialty,
        "triage_reason": decision.reason,
        "context_summary": decision.context,
        "triage_backend": "llm",
        # Self-reported, not calibrated. The router still gates on it.
        "triage_confidence": decision.confidence,
        "needs_review": needs_review,
        "acuity": float(decision.acuity),
        "messages": [
            AIMessage(
                content=(
                    f"[Triage/LLM] {decision.specialty.upper()} "
                    f"(self-reported confidence {decision.confidence:.2f}). "
                    f"needs_review {needs_review:.0f}; acuity {decision.acuity}. "
                    f"Reason: {decision.reason}"
                )
            )
        ],
    }


# Backwards-compatible name used earlier in the tutorial.
triage_agent = triage_llm
