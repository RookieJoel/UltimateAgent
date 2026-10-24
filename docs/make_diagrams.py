"""Render the docs images.

    python docs/make_diagrams.py graph               # docs/images/graph.png from the compiled LangGraph
    python docs/make_diagrams.py trace TRACE.json    # docs/images/trace_<task>.png from exported LangSmith steps

TRACE.json holds the steps of one LangSmith trace (exported with the LangSmith client):
{"task": "...", "question": "...", "answer": "...", "total_tokens": N, "latency": S,
 "steps": [{"name": "assistant" | "formatter" | <tool>, "type": "chain" | "tool", "sec": S, "inp": {...}, "out": "..."}]}
Both images are rendered by mermaid.ink (only the diagram text is sent).
"""
import base64
import json
import sys
from pathlib import Path

import requests

IMAGES = Path(__file__).resolve().parent / "images"
INTERNAL_TOOLS = {"duckduckgo_results_json"}  # LangChain's inner run inside web_search_tool


def render(mermaid: str, target: Path) -> None:
    encoded = base64.urlsafe_b64encode(mermaid.encode()).decode()
    response = requests.get(f"https://mermaid.ink/img/{encoded}?type=png&bgColor=white", timeout=60)
    response.raise_for_status()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(response.content)
    print(f"wrote {target} ({len(response.content)} bytes)")


def graph_diagram() -> str:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from agent.agent import agent

    return agent.get_graph().draw_mermaid()


def short(text: str, limit: int = 60) -> str:
    text = str(text).split("', 'name':")[0].replace("\\n", " ")  # drop serialized ToolMessage tail
    for char in '"#;|*{}<>':
        text = text.replace(char, " ")
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def trace_diagram(trace: dict) -> str:
    lines = [
        "sequenceDiagram",
        "    autonumber",
        "    participant A as assistant (Gemini + tools)",
        "    participant T as ToolNode",
        "    participant F as formatter",
    ]
    assistant_turn = 0
    for step in trace["steps"]:
        name = step["name"]
        if name in INTERNAL_TOOLS:
            continue
        if name == "assistant":
            assistant_turn += 1
            lines.append(f"    Note over A: step {assistant_turn} · {step['sec']}s")
        elif name == "formatter":
            lines.append(f"    A->>F: final reply ({step['sec']}s)")
            lines.append(f"    Note over F: answer = {short(trace['answer'], 40)}")
        else:
            args = ", ".join(f"{k}={short(v, 45)}" for k, v in (step.get("inp") or {}).items())
            lines.append(f"    A->>T: {name}({args})")
            lines.append(f"    T-->>A: {short(step.get('out', ''), 55)} ({step['sec']}s)")
    lines.append(
        f"    Note over A,F: {trace['latency']}s total · {trace['total_tokens']} tokens · source: LangSmith trace"
    )
    return "\n".join(lines)


def main() -> None:
    if sys.argv[1] == "graph":
        render(graph_diagram(), IMAGES / "graph.png")
    else:
        trace = json.loads(Path(sys.argv[2]).read_text())
        render(trace_diagram(trace), IMAGES / f"trace_{trace['task']}.png")


if __name__ == "__main__":
    main()
