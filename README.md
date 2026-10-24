---
title: UltimateAgent
emoji: 🕵🏻‍♂️
colorFrom: indigo
colorTo: indigo
sdk: static
app_file: index.html
pinned: false
---

# UltimateAgent: HF Agents Course Unit 4 (GAIA)

LangGraph agent on Gemini with a Plan → Act → Verify prompt and 7 tools. Official score on the
course's 20 GAIA Level 1 questions: **16/20 (80%)**, up from 12/20 after fixes found in traces.

![LangGraph](docs/images/graph.png)

- **[Architecture](docs/ARCHITECTURE.md)**: graph, tools, guards, a real execution trace from LangSmith
- **[Design decisions](docs/DECISIONS.md)**: why each choice was made, what it cost, and what changed 60% → 80%

## Layout

- `agent/`: graph (`agent.py`), prompt, rate limiter, tools
- `run_eval.py`: run the 20 questions locally, cache answers, submit
- `gaia_api.py`: course API client and attachment download
- `app.py`: course Gradio runner (needs a Gradio Space; this Space is static and only hosts the code)
- `tests/`: offline tests (`pytest`)
- `docs/`: documentation and diagram generator

## Run locally

```bash
pip install -r requirements.txt
# .env: GOOGLE_API_KEY=..., HF_TOKEN=... (optional: LANGSMITH_TRACING=true, LANGSMITH_API_KEY, LANGSMITH_PROJECT)
python run_eval.py run
python run_eval.py submit --username USER --space-id USER/SPACE
```

LangGraph Studio: `langgraph dev` (uses `langgraph.json`).
