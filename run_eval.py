"""Run the agent on the 20 GAIA questions locally, cache answers, and optionally submit.

Usage (from the project root):
    python run_eval.py run                 # answer every uncached question
    python run_eval.py run --task TASK_ID  # (re)answer one question
    python run_eval.py show                # print cached answers
    python run_eval.py submit --username USER --space-id USER/SPACE
"""
import argparse
import json
import time
from pathlib import Path

from langchain_core.messages import HumanMessage

import gaia_api

CACHE_PATH = Path(__file__).resolve().parent / "cache" / "answers.json"


def load_cache() -> dict:
    return json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}


def save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2, ensure_ascii=False))


def solve(agent, question: dict) -> dict:
    """Run the graph on one question and return the answer plus a short trace."""
    file_name = question.get("file_name") or None
    if file_name:
        gaia_api.download_attachment(question["task_id"], file_name)

    started = time.time()
    state = {"messages": [HumanMessage(content=question["question"])]}
    if file_name:
        state["file_path"] = file_name
    out = agent.invoke(state)

    tool_calls = [
        f"{call['name']}({json.dumps(call['args'], ensure_ascii=False)[:120]})"
        for message in out["messages"]
        for call in (getattr(message, "tool_calls", None) or [])
    ]
    return {
        "question": question["question"],
        "file_name": file_name,
        "answer": str(out.get("final_answer", "")).strip(),
        "steps": out.get("current_step"),
        "tool_calls": tool_calls,
        "seconds": round(time.time() - started, 1),
    }


def run(task_id: str | None) -> None:
    from agent.agent import agent  # imported here so `show`/`submit` don't need the API key

    questions = gaia_api.fetch_questions()
    cache = load_cache()
    todo = [q for q in questions if (q["task_id"] == task_id if task_id else q["task_id"] not in cache)]
    print(f"{len(questions)} questions, {len(cache)} cached, {len(todo)} to run")

    for index, question in enumerate(todo, 1):
        print(f"\n[{index}/{len(todo)}] {question['task_id']}: {question['question'][:100]}")
        try:
            result = solve(agent, question)
        except Exception as e:  # keep going; failed questions stay uncached and rerun next time
            print(f"  FAILED: {type(e).__name__}: {str(e)[:200]}")
            continue
        cache = {**cache, question["task_id"]: result}
        save_cache(cache)
        print(f"  answer={result['answer']!r} steps={result['steps']} time={result['seconds']}s")
        for call in result["tool_calls"]:
            print(f"    - {call}")


def show() -> None:
    cache = load_cache()
    for task_id, result in cache.items():
        print(f"{task_id}  {result['answer']!r}  | {result['question'][:80]}")
    print(f"\n{len(cache)} answers cached")


def submit(username: str, space_id: str) -> None:
    cache = load_cache()
    answers = [{"task_id": task_id, "submitted_answer": r["answer"]} for task_id, r in cache.items()]
    if not answers:
        print("No cached answers; run first.")
        return
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    print(f"Submitting {len(answers)} answers as {username} ({agent_code})")
    result = gaia_api.submit_answers(username, agent_code, answers)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--task", help="only (re)run this task_id")
    sub.add_parser("show")
    submit_parser = sub.add_parser("submit")
    submit_parser.add_argument("--username", required=True)
    submit_parser.add_argument("--space-id", required=True, help="e.g. your-name/Final_Assignment_Template")
    args = parser.parse_args()

    if args.command == "run":
        run(args.task)
    elif args.command == "show":
        show()
    else:
        submit(args.username, args.space_id)


if __name__ == "__main__":
    main()
