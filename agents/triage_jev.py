"""Triage backend 2: Jev, a decision model from TypeSafe.

Instead of asking a generative LLM to write JSON, we ask Jev three typed
questions in one request and get back probabilities plus a confidence score.
The confidence lets the router decide whether to trust the answer or send the
transcript to a human.
"""

from langchain_core.messages import AIMessage
from typesafe_sdk import Choice, Noul, Score, TypeSafeError
from agents.state import ScribeState
from agents.jev import get_jev

ACUITY_LEVELS = [
    "Routine visit, no time-sensitive concerns",
    "Needs prompt follow-up within days",
    "Urgent, same-day action or escalation discussed",
]

TRIAGE_QUESTIONS = {
    "specialty": Choice(
        instructions="Which specialty agent should write the clinical note for this transcript?",
        criteria={
            "pediatrics": (
                "Patient is a child or adolescent under 18; a parent or guardian speaks "
                "for the patient; well-child visit, growth, immunizations, weight-based dosing"
            ),
            "general_medicine": (
                "Adult patient; routine, chronic, or acute adult complaints; "
                "anything not clearly pediatric"
            ),
        },
    ),
    "needs_review": Noul(
        instructions=(
            "The text is incomplete, ambiguous, or is not a clinical encounter "
            "between a clinician and a patient"
        ),
    ),
    "acuity": Score(
        instructions="How time-sensitive is the care discussed in this encounter?",
        criteria=ACUITY_LEVELS,
    ),
}


def triage_jev(state: ScribeState) -> dict:
    """Ask Jev for the specialty, a needs-review flag, and an acuity score."""

    try:
        response = get_jev().system_one(
            state=state["transcript"],
            questions=TRIAGE_QUESTIONS,
        )
    except TypeSafeError as err:
        # Fail safe: send the transcript to a human instead of guessing.
        return {
            "specialty": "human_review",
            "triage_backend": "jev",
            "triage_confidence": 0.0,
            "needs_review": 1.0,
            "acuity": 0.0,
            "triage_reason": f"Jev call failed: {err}",
            "context_summary": "",
            "messages": [AIMessage(content=f"[Triage/Jev] Call failed, routing to human review: {err}")],
        }

    specialty = response.answers["specialty"]
    needs_review = response.answers["needs_review"].noul
    acuity = response.answers["acuity"]

    probabilities = ", ".join(
        f"{name} {prob:.2f}" for name, prob in specialty.probabilities.items()
    )
    acuity_label = ACUITY_LEVELS[round(acuity.score)]

    reason = (
        f"specialty probabilities: {probabilities}; "
        f"needs_review {needs_review:.2f}; acuity {acuity.score:.1f} ({acuity_label})"
    )

    return {
        "specialty": specialty.choice,
        "triage_backend": "jev",
        "triage_confidence": specialty.confidence,
        "needs_review": needs_review,
        "acuity": acuity.score,
        "triage_reason": reason,
        "context_summary": (
            f"Triage by Jev ({response.model}): {specialty.choice} with confidence "
            f"{specialty.confidence:.2f}. Acuity: {acuity_label}."
        ),
        "messages": [
            AIMessage(
                content=(
                    f"[Triage/Jev] {specialty.choice.upper()} "
                    f"(confidence {specialty.confidence:.2f}). {reason}"
                )
            )
        ],
    }
