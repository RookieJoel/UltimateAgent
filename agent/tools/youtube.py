import re

from langchain_core.tools import tool
from youtube_transcript_api import YouTubeTranscriptApi

from agent.tools._common import truncate

VIDEO_ID_PATTERN = re.compile(r"(?:v=|youtu\.be/|shorts/|embed/)([A-Za-z0-9_-]{11})")
BARE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")
TRANSCRIPT_LANGUAGES = ["en", "en-US", "en-GB"]


def extract_video_id(url: str) -> str | None:
    """Return the 11-character YouTube video id from a URL (or a bare id), else None."""
    match = VIDEO_ID_PATTERN.search(url)
    if match:
        return match.group(1)
    return url if BARE_ID_PATTERN.match(url) else None


@tool
def youtube_transcript(url: str) -> str:
    """Get the English transcript (captions) of a YouTube video, with timestamps.

Use this for questions about what is said in a video. It cannot see the
video's images; it only returns spoken words.

Args:
    url (str): YouTube video URL, e.g. https://www.youtube.com/watch?v=VIDEO_ID
"""
    video_id = extract_video_id(url)
    if not video_id:
        return f"youtube_transcript failed: could not find a video id in {url!r}"

    try:
        transcript = YouTubeTranscriptApi().fetch(video_id, languages=TRANSCRIPT_LANGUAGES)
    except Exception as e:
        return f"youtube_transcript failed for {video_id}: {e}"

    text = "\n".join(f"[{int(snippet.start)}s] {snippet.text}" for snippet in transcript)
    return truncate(text) if text else f"youtube_transcript: video {video_id} has an empty transcript"
