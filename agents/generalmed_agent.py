from agents.state import ScribeState
from agents.specialist import generate_note


def generalmed_agent(state: ScribeState) -> dict:
    """Generate a general medicine clinical note from the transcript."""

    system_prompt = f"""
    You are a general medicine medical scribe.

    The triage agent has identified this as an adult encounter.

    Context from triage:
    {state["context_summary"]}

    Write a SOAP note (Subjective, Objective, Assessment, Plan)
    from the transcript.

    General medicine specifics:
    - If height and weight are mentioned, use the calculate_bmi
      tool and include the BMI in the Objective section.
    - Capture vitals, medications, and follow-up plans.

    When the note is complete, call save_note to store it,
    then reply with the final note text.
    """

    return generate_note(
        state,
        system_prompt=system_prompt,
        tool_names=["calculate_bmi", "save_note"],
        label="General Medicine",
    )
