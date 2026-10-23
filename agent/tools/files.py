import pandas as pd
from langchain_core.tools import tool

from agent.tools._common import resolve_asset_path, truncate

TEXT_SUFFIXES = {".txt", ".py", ".md", ".json", ".jsonl", ".xml", ".html", ".tsv", ".log", ".yaml", ".yml"}
EXCEL_SUFFIXES = {".xlsx", ".xls"}


@tool
def read_file(file_path: str, start: int = 0) -> str:
    """Read an attached text, code, CSV or Excel file and return its contents as text.

Excel files return every sheet as CSV text. For images, audio, video or PDF
files use analyze_media instead.

Args:
    file_path (str): Path of the attached file exactly as given in the task.
    start (int): Character offset to start reading from, for long files (default 0).
"""
    try:
        path = resolve_asset_path(file_path)
    except (ValueError, FileNotFoundError) as e:
        return f"read_file failed: {e}"

    try:
        text = file_to_text(path)
    except UnsupportedFileType as e:
        return f"read_file: {e}. Use analyze_media for images, audio, video or PDF."
    except Exception as e:
        return f"read_file failed for {path.name}: {e}"

    return truncate(text, start=start) if text.strip() else f"read_file: {path.name} is empty"


class UnsupportedFileType(ValueError):
    pass


def file_to_text(path) -> str:
    """Full text of a CSV, Excel or text file (Excel: every sheet as CSV). Shared with run_python."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path).to_csv(index=False)
    if suffix in EXCEL_SUFFIXES:
        return _excel_to_text(path)
    if suffix in TEXT_SUFFIXES:
        return path.read_text(encoding="utf-8", errors="replace")
    raise UnsupportedFileType(f"unsupported file type {suffix!r}")


def _excel_to_text(path) -> str:
    sheets = pd.read_excel(path, sheet_name=None)
    return "\n\n".join(f"## Sheet: {name}\n{frame.to_csv(index=False)}" for name, frame in sheets.items())
