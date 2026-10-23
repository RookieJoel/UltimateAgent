from urllib.parse import quote

import requests
from bs4 import BeautifulSoup
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import tool

from agent.config import HTTP_HEADERS, REQUEST_TIMEOUT

WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_RESULTS = 5


@tool
def web_search_tool(query: str) -> str:
    """Search the web using DuckDuckGo and return the top five results.

Use this tool to find current information, verify facts, or locate external
sources relevant to a question. Snippets are short: to read a full page, pass
its link to fetch_url.

Args:
    query (str): A non-empty search query describing the information to find.
        Include specific keywords, names, dates, or locations when relevant.

Returns:
    str: Up to five relevant search results, each containing:
        - title: The title of the web page.
        - link: The URL of the web page.
        - snippet: A brief excerpt or summary relevant to the query.
        Fewer results may be returned if fewer than five are available.
"""
    try:
        search_tools = DuckDuckGoSearchResults(num_results=5)
        search_result = search_tools.invoke(query)
        return search_result if search_result else "No relevant web search results found."
    except Exception as e:
        return f"Web search failed: {e}"


@tool
def wiki_search(query: str) -> str:
    """Search English Wikipedia and return up to five matching articles.

Use this for encyclopedic facts (people, places, discographies, history).
Each result has a title, link and snippet. To read the full article
(including its tables), pass the link to fetch_url.

Args:
    query (str): Article title or keywords, e.g. "Mercedes Sosa discography".
"""
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": WIKI_RESULTS,
        "format": "json",
    }
    try:
        response = requests.get(WIKI_API, params=params, headers=HTTP_HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        hits = response.json().get("query", {}).get("search", [])
    except Exception as e:
        return f"Wikipedia search failed: {e}"

    if not hits:
        return f"No Wikipedia results for: {query}"
    return "\n".join(_format_wiki_hit(hit) for hit in hits)


def _format_wiki_hit(hit: dict) -> str:
    title = hit.get("title", "")
    link = f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
    snippet = BeautifulSoup(hit.get("snippet", ""), "html.parser").get_text()
    return f"title: {title}, link: {link}, snippet: {snippet}"
