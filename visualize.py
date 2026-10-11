"""Export the scribe graph as a diagram.

    python3 visualize.py            # LLM triage graph
    python3 visualize.py --triage jev

Writes graph_<backend>.mmd (Mermaid source, paste into https://mermaid.live)
and, when the network is available, graph_<backend>.png.
For an interactive view, run `langgraph dev` instead (see README).
"""

import argparse
from pathlib import Path

from main import create_medical_scribe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--triage", choices=["llm", "jev"], default="llm")
    args = parser.parse_args()

    graph = create_medical_scribe(triage=args.triage).get_graph()

    mermaid = graph.draw_mermaid()
    mmd_path = Path(f"graph_{args.triage}.mmd")
    mmd_path.write_text(mermaid)

    print(mermaid)
    print(f"\nMermaid source written to {mmd_path}")

    png_path = Path(f"graph_{args.triage}.png")
    try:
        # Renders through the mermaid.ink web service, so it needs network access.
        png_path.write_bytes(graph.draw_mermaid_png())
        print(f"PNG written to {png_path}")
    except Exception as err:
        print(f"PNG not written ({type(err).__name__}). Paste the .mmd into https://mermaid.live instead.")


if __name__ == "__main__":
    main()
