"""Helpers shared by the tools: output truncation, safe file paths, Gemini client."""
from functools import lru_cache
from pathlib import Path

from agent.config import MAX_TOOL_CHARS, assets_dir


def truncate(text: str, limit: int = MAX_TOOL_CHARS, start: int = 0) -> str:
    """Return text[start:start+limit] plus a note telling the model how to read the rest."""
    total = len(text)
    chunk = text[start:start + limit]
    end = start + len(chunk)
    if end < total:
        return (
            f"{chunk}\n\n[Truncated: showing characters {start}-{end} of {total}. "
            f"Call again with start={end} to read more.]"
        )
    return chunk


def resolve_asset_path(file_path: str) -> Path:
    """Resolve file_path and make sure it points to an existing file inside the assets directory.

    Blocks the model from reading anything else on disk (e.g. the .env file with the API key).
    """
    base = assets_dir()
    candidate = Path(file_path)
    path = (candidate if candidate.is_absolute() else base / candidate).resolve()
    if not path.is_relative_to(base):
        raise ValueError(f"access denied: {file_path} is outside the attachments folder")
    if not path.is_file():
        raise FileNotFoundError(f"file not found: {file_path}")
    return path


@lru_cache(maxsize=1)
def get_genai_client():
    """Create the google-genai client lazily; it reads GOOGLE_API_KEY from the environment."""
    from google import genai

    return genai.Client()
