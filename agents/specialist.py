from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from agents.state import ScribeState
from agents.llm import llm
from tools.base import get_tools, get_tools_by_name

MAX_TOOL_ROUNDS = 5


def generate_note(state: ScribeState, system_prompt: str, tool_names: list[str], label: str) -> dict:
    """Shared agent loop: call the LLM with tools, run any tool calls, return the note.

    Each specialist agent (pediatrics, general medicine) uses this with its own
    system prompt and its own set of tools.
    """

    tools = get_tools(tool_names)
    tools_by_name = get_tools_by_name(tools)
    llm_with_tools = llm.bind_tools(tools)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Transcript:\n\n{state['transcript']}"),
    ]

    response = llm_with_tools.invoke(messages)

    # Tool loop: keep running tools until the LLM answers with plain text.
    for _ in range(MAX_TOOL_ROUNDS):
        if not response.tool_calls:
            break

        messages.append(response)

        for call in response.tool_calls:
            tool = tools_by_name[call["name"]]
            result = tool.invoke(call["args"])
            print(f"  [tool] {call['name']}({call['args']}) -> {result}")
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

        response = llm_with_tools.invoke(messages)

    note = response.content

    return {
        "note": note,
        "messages": [AIMessage(content=f"[{label}] {note}")],
        "specialty": f"{state['specialty']}_complete",
    }
