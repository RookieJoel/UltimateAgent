from langchain_core.tools import tool

from agent.config import GEMINI_MODEL
from agent.rate_limit import GEMINI_RATE_LIMITER
from agent.tools._common import get_genai_client, resolve_asset_path, truncate
from agent.tools.files import file_to_text

MAX_FILE_CHARS = 200_000  # attached data sent to the sandbox prompt


@tool
def run_python(task: str, file_path: str | None = None) -> str:
    """Solve a calculation or data task by running Python in Google's server-side sandbox.

Use this for arithmetic, counting, sorting, date math, unit conversion, or to
find the output of a given Python program. Do not compute these in your head.
The sandbox has NO internet and cannot open local paths itself. To compute over an
attached CSV, Excel or text file, pass its path as file_path: the full file content
is then sent along with the task. Never write code that opens the path yourself.

Args:
    task (str): Full description of what to compute, including all needed data or code.
    file_path (str, optional): Attached CSV, Excel or text file whose data the task needs.
"""
    if file_path:
        try:
            data = file_to_text(resolve_asset_path(file_path))
        except Exception as e:
            return f"run_python failed to load {file_path}: {e}"
        if len(data) > MAX_FILE_CHARS:
            return f"run_python: {file_path} is too large to send ({len(data)} chars)"
        task = (
            f"{task}\n\nThe full content of the attached file {file_path} is below "
            f"(Excel sheets appear as '## Sheet: name' followed by CSV). Load it from this text "
            f"with io.StringIO; do not try to open any file path.\n\n<file>\n{data}\n</file>"
        )
    try:
        from google.genai import types

        GEMINI_RATE_LIMITER.acquire()
        response = get_genai_client().models.generate_content(
            model=GEMINI_MODEL,
            contents=task,
            config=types.GenerateContentConfig(tools=[types.Tool(code_execution=types.ToolCodeExecution())]),
        )
    except Exception as e:
        return f"run_python failed: {e}"

    text = format_code_execution(response)
    return truncate(text) if text else "run_python: the model returned no code or output"


def format_code_execution(response) -> str:
    """Join the code, execution output and text parts of a Gemini response into one string."""
    candidates = getattr(response, "candidates", None) or []
    parts = (candidates[0].content.parts or []) if candidates and candidates[0].content else []
    sections = []
    for part in parts:
        if part.executable_code and part.executable_code.code:
            sections.append(f"Code:\n{part.executable_code.code}")
        if part.code_execution_result and part.code_execution_result.output is not None:
            sections.append(f"Output:\n{part.code_execution_result.output}")
        if part.text:
            sections.append(f"Answer:\n{part.text}")
    return "\n\n".join(sections)
