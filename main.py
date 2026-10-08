import sys
from pathlib import Path

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage

from agents.state import ScribeState
from agents.triage import triage_agent
from agents.router import route_after_triage
from agents.pediatrics_agent import pediatrics_agent
from agents.generalmed_agent import generalmed_agent


def create_medical_scribe():
    """Build the graph shown in agentic_medical_scribe.png.

    transcript -> triage (LLM) -> router -> pediatrics | general_medicine -> note
    """

    graph = StateGraph(ScribeState)

    # Add nodes
    graph.add_node("triage", triage_agent)
    graph.add_node("pediatrics", pediatrics_agent)
    graph.add_node("general_medicine", generalmed_agent)

    # Start with triage
    graph.add_edge(START, "triage")

    # Route based on triage decision
    graph.add_conditional_edges(
        "triage",
        route_after_triage,
        {
            "pediatrics": "pediatrics",
            "general_medicine": "general_medicine",
        },
    )

    # Specialist agents finish the workflow
    graph.add_edge("pediatrics", END)
    graph.add_edge("general_medicine", END)

    return graph.compile()


def load_transcript() -> str:
    """Use the file path passed on the command line, or a bundled sample."""

    if len(sys.argv) > 1:
        return Path(sys.argv[1]).read_text()

    sample = Path(__file__).parent / "transcripts" / "general_medicine_visit.txt"
    print(f"No transcript given, using sample: {sample.name}\n")
    return sample.read_text()


def main():

    agent = create_medical_scribe()

    transcript = load_transcript()

    print("Agentic Medical Scribe")
    print("=" * 60)

    result = agent.invoke(
        {
            "transcript": transcript,
            "messages": [],
            "specialty": "",
            "triage_reason": "",
            "context_summary": "",
            "note": "",
        }
    )

    print("\nAgent log:")
    for message in result["messages"]:
        if isinstance(message, AIMessage):
            print(message.content[:200] + ("..." if len(message.content) > 200 else ""))

    print("\n" + "=" * 60)
    print("GENERATED NOTE")
    print("=" * 60)
    print(result["note"])


if __name__ == "__main__":
    main()
