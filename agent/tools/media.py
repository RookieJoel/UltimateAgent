from langchain_core.tools import tool

from agent.config import GEMINI_MODEL
from agent.rate_limit import GEMINI_RATE_LIMITER
from agent.tools._common import get_genai_client, resolve_asset_path, truncate

MEDIA_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".mp4": "video/mp4",
    ".pdf": "application/pdf",
}
MAX_INLINE_BYTES = 20 * 1024 * 1024  # Gemini inline-data request limit


@tool
def analyze_media(file_path: str, question: str) -> str:
    """Ask Gemini a question about an attached image, audio, video or PDF file.

Examples: describe a chess position, read numbers in a chart, transcribe an
audio recording, list ingredients mentioned in a voice memo, read a PDF.
Ask one specific question; if exact words matter, ask for a full verbatim transcript.

Args:
    file_path (str): Path of the attached file exactly as given in the task.
    question (str): What you want to know about the file.
"""
    try:
        path = resolve_asset_path(file_path)
    except (ValueError, FileNotFoundError) as e:
        return f"analyze_media failed: {e}"

    mime_type = MEDIA_MIME_TYPES.get(path.suffix.lower())
    if mime_type is None:
        return f"analyze_media: unsupported file type {path.suffix!r}. Use read_file for text, code, CSV or Excel."
    if path.stat().st_size > MAX_INLINE_BYTES:
        return f"analyze_media: {path.name} is larger than 20 MB and cannot be sent inline"

    try:
        from google.genai import types

        GEMINI_RATE_LIMITER.acquire()
        response = get_genai_client().models.generate_content(
            model=GEMINI_MODEL,
            contents=[types.Part.from_bytes(data=path.read_bytes(), mime_type=mime_type), question],
        )
    except Exception as e:
        return f"analyze_media failed for {path.name}: {e}"

    return truncate(response.text) if response.text else "analyze_media: the model returned no text"
