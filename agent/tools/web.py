import re
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from markdownify import markdownify

from agent.config import HTTP_HEADERS, REQUEST_TIMEOUT
from agent.tools._common import truncate

NOISE_TAGS = ["script", "style", "noscript", "nav", "footer", "header", "aside", "form", "svg"]
# Wikipedia clutter: [1] citation markers, [edit] links, reference lists, navigation boxes
NOISE_SELECTORS = "sup.reference, .mw-editsection, .reflist, .navbox, table.sidebar"


@tool
def fetch_url(url: str, start: int = 0) -> str:
    """Download a web page and return its main content as Markdown text (tables are kept).

Use this after web_search_tool or wiki_search to read a full page instead of
relying on short snippets. Long pages are returned in chunks: the output ends
with a note giving the start value for the next chunk.

Args:
    url (str): Full http(s) URL of the page.
    start (int): Character offset to start reading from (default 0).
"""
    if urlparse(url).scheme not in ("http", "https"):
        return f"fetch_url failed: only http(s) URLs are supported, got {url!r}"
    if start < 0:
        return "fetch_url failed: start must be >= 0"

    try:
        response = requests.get(url, headers=HTTP_HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as e:
        return f"fetch_url failed for {url}: {e}"

    content_type = response.headers.get("Content-Type", "").lower()
    if "html" in content_type:
        text = html_to_markdown(response.text)
    elif content_type.startswith("text/") or "json" in content_type:
        text = response.text
    else:
        return (
            f"fetch_url: {url} is not a text page (Content-Type: {content_type or 'unknown'}). "
            "This tool can only read HTML or plain text."
        )

    if not text.strip():
        return f"fetch_url: no readable text found at {url}"
    return truncate(text, start=start)


def html_to_markdown(html: str) -> str:
    """Strip scripts/menus from HTML and convert the main content to compact Markdown.

    Link URLs and images are dropped (link text is kept): they made pages ~5x longer.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(NOISE_TAGS):
        tag.decompose()
    for tag in soup.select(NOISE_SELECTORS):
        tag.decompose()
    main = soup.find("main") or soup.find(id="content") or soup.body or soup
    markdown = markdownify(str(main), heading_style="ATX", strip=["a", "img"])
    return re.sub(r"\n\s*\n+", "\n\n", markdown).strip()
