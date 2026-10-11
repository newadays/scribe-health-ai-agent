"""Benchmark the two triage backends: OpenAI LLM as a judge vs Jev.

Runs ONLY the triage node (no note generation) for every transcript, several
times per backend, and reports accuracy, consistency, confidence, and latency.

    python3 benchmark_triage.py            # 5 runs per transcript per backend
    python3 benchmark_triage.py --runs 10

Writes benchmark_results.md and benchmark_results.json.
"""

import argparse
import json
import statistics
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from langchain_core.callbacks import get_usage_metadata_callback  # noqa: E402

import agents.jev                                      # noqa: E402
from agents.router import route_after_triage          # noqa: E402
from agents.triage import triage_llm                   # noqa: E402
from agents.triage_jev import triage_jev               # noqa: E402

# List prices, USD per 1M tokens, checked 2026-10-11.
# OpenAI: https://platform.openai.com/docs/pricing  (gpt-4o-mini)
# TypeSafe: https://docs.typesafe.ai/models  (jev-1.13.0, output tokens free)
PRICES = {
    "llm": {"input": 0.15, "output": 0.60},
    "jev": {"input": 0.042, "output": 0.0},
}


def cost_usd(backend: str, input_tokens: int, output_tokens: int) -> float:
    p = PRICES[backend]
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000


# Capture Jev token usage by wrapping the client's system_one call.
_jev_usage = {}


def _wrap_jev_client(client):
    original = client.system_one

    def system_one(*args, **kwargs):
        response = original(*args, **kwargs)
        _jev_usage["input"] = response.usage.input_tokens or 0
        _jev_usage["output"] = response.usage.output_tokens or 0
        return response

    client.system_one = system_one
    return client

# transcript path -> the branch the router SHOULD take
CASES = {
    "transcripts/pediatrics_visit.txt": "pediatrics",
    "transcripts/general_medicine_visit.txt": "general_medicine",
    "transcripts/ambiguous_note.txt": "human_review",
    "sample_transcript/general_medicine.txt": "general_medicine",
    "transcripts/teen_visit.txt": "pediatrics",          # 17-year-old, no parent present
    "transcripts/truncated_visit.txt": "human_review",   # cut off after two lines
}

BACKENDS = {"llm": triage_llm, "jev": triage_jev}


def blank_state(transcript: str) -> dict:
    return {
        "transcript": transcript, "messages": [], "specialty": "", "triage_reason": "",
        "context_summary": "", "triage_backend": "", "triage_confidence": 0.0,
        "needs_review": 0.0, "acuity": 0.0, "note": "",
    }


def run_once(backend: str, transcript: str) -> dict:
    state = blank_state(transcript)
    start = time.perf_counter()
    if backend == "llm":
        with get_usage_metadata_callback() as usage_cb:
            update = BACKENDS[backend](state)
        usage = next(iter(usage_cb.usage_metadata.values()), {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
    else:
        _jev_usage.clear()
        update = BACKENDS[backend](state)
        input_tokens = _jev_usage.get("input", 0)
        output_tokens = _jev_usage.get("output", 0)
    latency = time.perf_counter() - start
    state.update(update)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost_usd(backend, input_tokens, output_tokens),
        "route": route_after_triage(state),
        "specialty": state["specialty"],
        "confidence": state["triage_confidence"],
        "needs_review": state["needs_review"],
        "acuity": state["acuity"],
        "latency_s": latency,
    }


def summarize(runs: list[dict], expected: str) -> dict:
    routes = Counter(r["route"] for r in runs)
    majority, majority_count = routes.most_common(1)[0]
    return {
        "expected": expected,
        "majority_route": majority,
        "accuracy": sum(r["route"] == expected for r in runs) / len(runs),
        "consistency": majority_count / len(runs),
        "mean_confidence": statistics.mean(r["confidence"] for r in runs),
        "mean_needs_review": statistics.mean(r["needs_review"] for r in runs),
        "mean_acuity": statistics.mean(r["acuity"] for r in runs),
        "median_latency_s": statistics.median(r["latency_s"] for r in runs),
        "mean_input_tokens": statistics.mean(r["input_tokens"] for r in runs),
        "mean_output_tokens": statistics.mean(r["output_tokens"] for r in runs),
        "mean_cost_usd": statistics.mean(r["cost_usd"] for r in runs),
        "routes": dict(routes),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    results = {b: {} for b in BACKENDS}
    _wrap_jev_client(agents.jev.get_jev())

    for path, expected in CASES.items():
        transcript = Path(path).read_text()
        for backend in BACKENDS:
            runs = []
            for i in range(args.runs):
                r = run_once(backend, transcript)
                runs.append(r)
                print(f"{backend:3s} {Path(path).name:30s} run {i+1}: {r['route']:17s} "
                      f"conf={r['confidence']:.2f} review={r['needs_review']:.2f} "
                      f"acuity={r['acuity']:.1f} {r['latency_s']:.2f}s "
                      f"tok={r['input_tokens']}+{r['output_tokens']} ${r['cost_usd']:.6f}")
            results[backend][path] = {"runs": runs, "summary": summarize(runs, expected)}

    # Overall numbers per backend
    overall = {}
    for backend in BACKENDS:
        all_runs = [r for c in results[backend].values() for r in c["runs"]]
        overall[backend] = {
            "accuracy": statistics.mean(
                r["route"] == CASES[p] for p, c in results[backend].items() for r in c["runs"]
            ),
            "consistency": statistics.mean(c["summary"]["consistency"] for c in results[backend].values()),
            "median_latency_s": statistics.median(r["latency_s"] for r in all_runs),
            "p95_latency_s": sorted(r["latency_s"] for r in all_runs)[int(0.95 * (len(all_runs) - 1))],
            "mean_input_tokens": statistics.mean(r["input_tokens"] for r in all_runs),
            "mean_output_tokens": statistics.mean(r["output_tokens"] for r in all_runs),
            "mean_cost_usd": statistics.mean(r["cost_usd"] for r in all_runs),
            "cost_per_1000_usd": 1000 * statistics.mean(r["cost_usd"] for r in all_runs),
        }

    # Markdown report
    lines = [f"Runs per transcript per backend: {args.runs}", ""]
    lines += ["| Backend | Routing accuracy | Consistency | Median latency | p95 latency | Mean tokens (in+out) | Mean cost / triage | Cost / 1,000 triages |",
              "|---|---|---|---|---|---|---|---|"]
    for b, o in overall.items():
        lines.append(f"| {b} | {o['accuracy']:.0%} | {o['consistency']:.0%} | "
                     f"{o['median_latency_s']:.2f} s | {o['p95_latency_s']:.2f} s | "
                     f"{o['mean_input_tokens']:.0f}+{o['mean_output_tokens']:.0f} | "
                     f"${o['mean_cost_usd']:.6f} | ${o['cost_per_1000_usd']:.3f} |")
    lines += ["", "| Transcript | Expected | Backend | Majority route | Accuracy | Mean confidence | Mean needs_review | Mean acuity | Median latency | Tokens (in+out) | Cost |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for path, expected in CASES.items():
        for b in BACKENDS:
            s = results[b][path]["summary"]
            lines.append(f"| {Path(path).name} | {expected} | {b} | {s['majority_route']} | "
                         f"{s['accuracy']:.0%} | {s['mean_confidence']:.2f} | {s['mean_needs_review']:.2f} | "
                         f"{s['mean_acuity']:.1f} | {s['median_latency_s']:.2f} s | "
                         f"{s['mean_input_tokens']:.0f}+{s['mean_output_tokens']:.0f} | ${s['mean_cost_usd']:.6f} |")
    report = "\n".join(lines)

    Path("benchmark_results.md").write_text(report + "\n")
    Path("benchmark_results.json").write_text(json.dumps({"overall": overall, "cases": results}, indent=2))
    print("\n" + report)
    print("\nWritten benchmark_results.md and benchmark_results.json")


if __name__ == "__main__":
    main()
