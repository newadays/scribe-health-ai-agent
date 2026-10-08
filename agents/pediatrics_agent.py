from agents.state import ScribeState
from agents.specialist import generate_note


def pediatrics_agent(state: ScribeState) -> dict:
    """Generate a pediatric clinical note from the transcript."""

    system_prompt = f"""
    You are a pediatric medical scribe.

    The triage agent has identified this as a pediatric encounter.

    Context from triage:
    {state["context_summary"]}

    Write a SOAP note (Subjective, Objective, Assessment, Plan)
    from the transcript.

    Pediatric specifics:
    - Record the patient's age and weight.
    - If a weight-based medication is discussed, use the
      calculate_pediatric_dose tool to compute the dose.
    - Note immunization or growth concerns if mentioned.

    When the note is complete, call save_note to store it,
    then reply with the final note text.
    """

    return generate_note(
        state,
        system_prompt=system_prompt,
        tool_names=["calculate_pediatric_dose", "save_note"],
        label="Pediatrics",
    )
