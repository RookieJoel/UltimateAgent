import os
from pathlib import Path

# Free tier: 3.5-flash-lite = 15 RPM / 500 RPD (3.6-flash is only 5 RPM / 20 RPD)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ASSETS_DIR = PROJECT_ROOT / "assets"

REQUEST_TIMEOUT = 20  # seconds
MAX_TOOL_CHARS = 12_000
HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; GAIA-course-agent/0.1; +https://huggingface.co/learn/agents-course)"
}


def assets_dir() -> Path:
    """Directory that file tools are allowed to read from (read at call time so tests can override)."""
    return Path(os.getenv("ASSETS_DIR", DEFAULT_ASSETS_DIR)).resolve()
