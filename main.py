import argparse
import os
import sys
from pathlib import Path

# Make `agents` and `tools` importable when this file is loaded by path,
# as LangGraph Studio (`langgraph dev`) does. Harmless when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage

from agents.state import ScribeState
from agents.triage import triage_llm
from agents.router import route_after_triage
from agents.pediatrics_agent import pediatrics_agent
from agents.generalmed_agent import generalmed_agent
from agents.human_review import human_review


def create_medical_scribe(triage: str = "llm"):
    """Build the graph shown in agentic_medical_scribe.png.

    transcript -> triage (LLM or Jev) -> router -> pediatrics | general_medicine | human_review

    Args:
        triage: "llm" uses OpenAI structured output (agents/triage.py),
                "jev" uses TypeSafe's Jev decision model (agents/triage_jev.py).
    """

    if triage == "jev":
        # Imported here so the LLM path runs without the typesafe-sdk installed.
        from agents.triage_jev import triage_jev
        triage_node = triage_jev
    else:
        triage_node = triage_llm

    graph = StateGraph(ScribeState)

    # Add nodes. The triage node is always called "triage" so the rest
    # of the graph is identical for both backends.
    graph.add_node("triage", triage_node)
    graph.add_node("pediatrics", pediatrics_agent)
    graph.add_node("general_medicine", generalmed_agent)
    graph.add_node("human_review", human_review)

    # Start with triage
    graph.add_edge(START, "triage")

    # Route based on triage decision and confidence
    graph.add_conditional_edges(
        "triage",
        route_after_triage,
        {
            "pediatrics": "pediatrics",
            "general_medicine": "general_medicine",
            "human_review": "human_review",
        },
    )

    # All branches finish the workflow
    graph.add_edge("pediatrics", END)
    graph.add_edge("general_medicine", END)
    graph.add_edge("human_review", END)

    return graph.compile()


# Zero-argument factories for LangGraph Studio (see langgraph.json).
# `langgraph dev` loads these and shows each graph in the Studio UI,
# where you can submit a transcript and watch it move through the nodes.
def scribe_llm():
    return create_medical_scribe(triage="llm")


def scribe_jev():
    return create_medical_scribe(triage="jev")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agentic medical scribe")
    parser.add_argument(
        "transcript",
        nargs="?",
        help="Path to a transcript text file (default: bundled general medicine sample)",
    )
    parser.add_argument(
        "--triage",
        choices=["llm", "jev"],
        default=os.environ.get("TRIAGE_BACKEND", "llm"),
        help="Decision backend for triage: llm (OpenAI) or jev (TypeSafe). "
             "Default: llm, or the TRIAGE_BACKEND env var.",
    )
    return parser.parse_args()


def load_transcript(path: str | None) -> str:
    """Use the given file, or a bundled sample."""

    if path:
        return Path(path).read_text()

    sample = Path(__file__).parent / "transcripts" / "general_medicine_visit.txt"
    print(f"No transcript given, using sample: {sample.name}\n")
    return sample.read_text()


def main():

    args = parse_args()

    agent = create_medical_scribe(triage=args.triage)

    transcript = load_transcript(args.transcript)

    print("Agentic Medical Scribe")
    print(f"Triage backend: {args.triage}")
    print("=" * 60)

    result = agent.invoke(
        {
            "transcript": transcript,
            "messages": [],
            "specialty": "",
            "triage_reason": "",
            "context_summary": "",
            "triage_backend": "",
            "triage_confidence": 0.0,
            "needs_review": 0.0,
            "acuity": 0.0,
            "note": "",
        }
    )

    print("\nAgent log:")
    for message in result["messages"]:
        if isinstance(message, AIMessage):
            print(message.content[:300] + ("..." if len(message.content) > 300 else ""))

    print("\n" + "=" * 60)
    if result["specialty"] == "human_review_complete":
        print("SENT TO HUMAN REVIEW (no note generated)")
    else:
        print("GENERATED NOTE")
    print("=" * 60)
    print(result["note"])


if __name__ == "__main__":
    main()
