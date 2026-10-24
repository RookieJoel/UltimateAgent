# Design decisions

Each entry: the decision, why, and what it cost. Ordered roughly as they were made.

## Results so far

| Run | Change | Score (official API) |
|---|---|---|
| 1 | first full run, `MAX_STEP = 8` | **12/20 (60%)** |
| 2 | decisions 14–16 below, `MAX_STEP = 10`, only the 8 wrong questions rerun | **16/20 (80%)** |
| 3 | same code, fresh full run of all 20 questions | **17/20 (85%)** |

Remaining misses in run 3: bird species count in a video, chess move from a board image,
Japanese baseball roster by jersey number. The NASA award number (wrong in run 2) was correct.

Answers vary between runs on the same code: the bird count came back as 6 and 4, the chess
move as `Nc3+` and `Qe1#`. A second and third fresh run were cut short by the free-tier daily
quota, so run-to-run variance is observed but not yet measured.

---

## Architecture

### 1. LangGraph instead of smolagents
- **Why:** explicit State / Nodes / Edges make the control flow (step budget, forced final answer, formatter) visible and testable; smolagents was already covered earlier in the course.
- **Cost:** more wiring than a `CodeAgent` one-liner.

### 2. One assistant + one ToolNode (no manager / specialist / verifier agents)
- **Considered:** manager agent routing to web, media and code specialist nodes, plus a verifier loop.
- **Why dropped:** every extra agent adds LLM calls on a free tier capped by requests per minute and per day, and adds routing errors. GAIA Level 1 needs tool choice, not team coordination.
- **Cost:** one prompt has to cover every tool; the tool docstrings carry more weight.

### 3. Step budget with a forced final step, not a verifier
- **Why:** an unbounded loop hits LangGraph's recursion limit and raises `GraphRecursionError`, which the course runner records as `AGENT ERROR` (no answer at all). At `MAX_STEP - 1` the model is called **without tools**, so it has to answer in text.
- **Cost:** hard questions can run out of steps (7 of the 8 misses in run 1 used the whole budget).

### 4. Separate formatter node with structured output
- **Why:** grading is exact match. Mixing research and formatting in one reply produced extra words. A Pydantic schema whose field description carries the GAIA formatting rules (no units, no articles, comma lists) makes formatting a separate, testable step. The course forbids the literal "FINAL ANSWER", so the schema never asks for it.
- **Cost:** one extra LLM call per question.

### 5. Planner-style prompt adapted from customgpt-agent, without its critic
- **Why:** Plan → Act → Verify inside a single prompt gets most of the benefit of a planner/critic system for zero extra calls. A separate critic node was left out to stay inside free-tier quotas.
- **Cost:** verification depends on the model following instructions (see decision 16).

## Model and quotas

### 6. Gemini over Qwen
- **Why:** one API gives tool calling, image/audio/video/PDF input and server-side code execution. Qwen3-Omni was evaluated and dropped: no free quota, mandatory streaming, unclear tool-calling support on the international endpoint, no code execution.

### 7. `gemini-3.5-flash-lite` instead of `gemini-3.6-flash`
- **Why:** measured 429 errors came from the **daily** cap: 3.6-flash allows 5 RPM / 20 requests per day, which is less than one 20-question run. 3.5-flash-lite allows 15 RPM / 500 per day.
- **Cost:** a smaller model; it follows prompt rules less strictly.

### 8. One shared rate limiter
- **Why:** the agent model, `analyze_media` and `run_python` all spend the same per-model quota. Separate limiters would each think they had 15 RPM. Set to 13 RPM for headroom.

## Tools

### 9. Code runs in Gemini's sandbox, never locally
- **Why:** executing LLM-written code on the host is a security risk; Gemini code execution runs server-side.
- **Cost:** the sandbox cannot see local files (this caused the worst failure in run 1; fixed by decision 14).

### 10. Tools return error strings instead of raising
- **Why:** verified that LangGraph 1.x `ToolNode` does **not** catch generic tool exceptions; one failing HTTP call crashed the whole graph. An error string lets the model retry or pick another tool.

### 11. Read pages as cleaned Markdown, in chunks
- **Why:** search snippets rarely contain the exact fact. `fetch_url` converts HTML to Markdown, removes navigation, citations and edit links (a Wikipedia page went from 123k to 21k characters), and returns 12k-character chunks with a `start` offset so long pages fit the context window.

### 12. Attachments downloaded before the agent runs; file tools sandboxed to `assets/`
- **Why:** the agent receives only a file name; tools resolve it inside `assets/` and refuse anything outside (`../.env` is blocked). The course `/files` endpoint returned 404 ("No file path associated with task_id"), so the downloader falls back to the gated GAIA dataset on the Hub.

### 13. Answer cache and separate run / submit commands
- **Why:** a full run takes about 20 minutes and real quota. Caching per `task_id` means a crash or a single bad question costs one rerun, and submitting never re-runs the agent.

## Fixes after reading run 1's traces (60% → 80–85%)

### 14. `run_python` takes `file_path`
- **Symptom:** for the Excel question the model wrote `pd.read_excel("file.xlsx")` and then `os.walk("/")` inside the remote sandbox, used up every step, and returned an invented number (`12345.67`).
- **Fix:** with `file_path`, the tool reads the file locally (same loader as `read_file`) and embeds the data in the task with instructions to load it from text. The docstring already said "no local files"; changing the tool's interface worked where the warning had not.
- **Result:** correct answer (`89706.00`).

### 15. An exit strategy for the step budget
- **Symptom:** when steps ran out the model produced placeholder values.
- **Fix:** the final no-tools call gets `LAST_STEP_NOTE`: answer from the evidence collected, never a placeholder, choose what the evidence supports most.

### 16. "Search at most twice, then read"
- **Symptom:** 6 searches before opening the Wikipedia page that had the answer.
- **Fix:** prompt rule plus "fetch known URLs directly". `MAX_STEP` raised from 8 to 10.
- **Honest result:** the question became correct, but the trace (see [ARCHITECTURE.md](ARCHITECTURE.md#execution-flow-real-trace)) shows the model still ran 5 searches. Prompt rules shift behaviour; they do not enforce it. Enforcing it would need code, for example counting searches in state.

## Operations

### 17. LangSmith tracing with per-question metadata, and LangGraph Studio
- **Why:** the fixes above came from reading traces, not from guessing. Each trace carries `task_id`, model, `max_step` and a prompt-version hash so runs before and after a change can be filtered and compared. Tests disable tracing so fake runs do not pollute the project.

### 18. Static Space for the code
- **Why:** Hugging Face now requires PRO for Gradio / Docker Spaces on free hardware (`402 Payment Required`). The assignment needs a public code link, not a running app, so the Space is static and answers are produced and submitted locally with `run_eval.py`.
- **Cost:** the Gradio runner in `app.py` cannot run on this Space.
