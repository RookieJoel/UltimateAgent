from agent.tools.code import run_python
from agent.tools.files import read_file
from agent.tools.media import analyze_media
from agent.tools.search import web_search_tool, wiki_search
from agent.tools.web import fetch_url
from agent.tools.youtube import youtube_transcript

TOOLS = [
    web_search_tool,
    wiki_search,
    fetch_url,
    youtube_transcript,
    read_file,
    analyze_media,
    run_python,
]
