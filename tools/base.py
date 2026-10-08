import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from langchain_core.tools import BaseTool, tool

NOTES_DIR = Path(__file__).resolve().parent.parent / "notes"


@tool
def calculate_bmi(weight_kg: float, height_cm: float) -> str:
    """Calculate Body Mass Index from weight in kilograms and height in centimetres."""
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)

    if bmi < 18.5:
        category = "underweight"
    elif bmi < 25:
        category = "normal"
    elif bmi < 30:
        category = "overweight"
    else:
        category = "obese"

    return f"BMI {bmi:.1f} ({category})"


@tool
def calculate_pediatric_dose(weight_kg: float, mg_per_kg: float, max_mg: Optional[float] = None) -> str:
    """Calculate a weight-based pediatric dose (mg) from weight in kg and mg per kg.
    Optionally cap the result at max_mg."""
    dose = weight_kg * mg_per_kg

    if max_mg is not None and dose > max_mg:
        return f"{max_mg:.0f} mg (calculated {dose:.0f} mg, capped at max {max_mg:.0f} mg)"

    return f"{dose:.0f} mg"


@tool
def save_note(patient_name: str, note: str) -> str:
    """Save the finished clinical note to the notes folder. Returns the file path."""
    NOTES_DIR.mkdir(exist_ok=True)

    safe_name = re.sub(r"[^a-z0-9]+", "_", patient_name.lower()).strip("_") or "patient"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = NOTES_DIR / f"{safe_name}_{timestamp}.md"

    path.write_text(note)

    return f"Note saved to {path}"


ALL_TOOLS: Dict[str, BaseTool] = {
    "calculate_bmi": calculate_bmi,
    "calculate_pediatric_dose": calculate_pediatric_dose,
    "save_note": save_note,
}


def get_tools(tool_names: Optional[List[str]] = None) -> List[BaseTool]:
    """Get the named tools, or all tools if tool_names is None."""
    if tool_names is None:
        return list(ALL_TOOLS.values())

    return [ALL_TOOLS[name] for name in tool_names if name in ALL_TOOLS]


def get_tools_by_name(tools: Optional[List[BaseTool]] = None) -> Dict[str, BaseTool]:
    """Get a dictionary of tools mapped by name."""
    if tools is None:
        tools = get_tools()

    return {t.name: t for t in tools}
