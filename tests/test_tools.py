"""Offline tests for agent/tools: network and Gemini calls are replaced with fakes."""
from types import SimpleNamespace

import pandas as pd
import pytest
import requests

from agent.tools import _common, code, media, search, web, youtube
from agent.tools.files import read_file


@pytest.fixture
def assets(tmp_path, monkeypatch):
    monkeypatch.setenv("ASSETS_DIR", str(tmp_path))
    return tmp_path


class FakeResponse:
    def __init__(self, text="", content_type="text/html", json_data=None, status=200):
        self.text = text
        self.headers = {"Content-Type": content_type}
        self._json = json_data
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")

    def json(self):
        return self._json


# --- _common -----------------------------------------------------------------

def test_truncate_short_text_is_unchanged():
    assert _common.truncate("abc", limit=10) == "abc"


def test_truncate_long_text_tells_next_start():
    out = _common.truncate("a" * 25, limit=10)
    assert out.startswith("a" * 10)
    assert "start=10" in out and "of 25" in out


def test_truncate_with_start_reads_next_chunk():
    assert _common.truncate("0123456789", limit=4, start=8) == "89"


def test_resolve_asset_path_blocks_files_outside_assets(assets, tmp_path_factory):
    secret = tmp_path_factory.mktemp("outside") / ".env"
    secret.write_text("GOOGLE_API_KEY=secret")
    with pytest.raises(ValueError):
        _common.resolve_asset_path(str(secret))
    with pytest.raises(ValueError):
        _common.resolve_asset_path("../outside/.env")


def test_resolve_asset_path_accepts_relative_file_inside_assets(assets):
    (assets / "a.txt").write_text("hi")
    assert _common.resolve_asset_path("a.txt") == (assets / "a.txt").resolve()


# --- read_file -----------------------------------------------------------------

def test_read_file_csv(assets):
    pd.DataFrame({"item": ["burger", "fries"], "sales": [10, 5]}).to_csv(assets / "s.csv", index=False)
    out = read_file.invoke({"file_path": "s.csv"})
    assert "item,sales" in out and "burger,10" in out


def test_read_file_excel_shows_every_sheet(assets):
    with pd.ExcelWriter(assets / "b.xlsx") as writer:
        pd.DataFrame({"a": [1]}).to_excel(writer, sheet_name="Food", index=False)
        pd.DataFrame({"b": [2]}).to_excel(writer, sheet_name="Drinks", index=False)
    out = read_file.invoke({"file_path": "b.xlsx"})
    assert "## Sheet: Food" in out and "## Sheet: Drinks" in out


def test_read_file_python_source(assets):
    (assets / "p.py").write_text("print(1 + 1)\n")
    assert "print(1 + 1)" in read_file.invoke({"file_path": "p.py"})


def test_read_file_rejects_media_and_escapes(assets):
    (assets / "x.png").write_bytes(b"\x89PNG")
    assert "analyze_media" in read_file.invoke({"file_path": "x.png"})
    assert "access denied" in read_file.invoke({"file_path": "../../.env"})
    assert "not found" in read_file.invoke({"file_path": "missing.txt"})


# --- fetch_url -----------------------------------------------------------------

def test_fetch_url_converts_html_and_drops_scripts(monkeypatch):
    html = "<html><body><nav>menu</nav><main><h1>Title</h1><script>evil()</script>" \
           "<table><tr><th>Year</th></tr><tr><td>2005</td></tr></table></main></body></html>"
    monkeypatch.setattr(web.requests, "get", lambda *a, **k: FakeResponse(html))
    out = web.fetch_url.invoke({"url": "https://example.com"})
    assert "# Title" in out and "2005" in out
    assert "evil" not in out and "menu" not in out


def test_html_to_markdown_drops_link_urls_and_citations():
    html = '<main><p>Born in <a href="/wiki/Tucuman">Tucumán</a><sup class="reference">[1]</sup></p></main>'
    out = web.html_to_markdown(html)
    assert out == "Born in Tucumán"


def test_fetch_url_chunks_long_pages(monkeypatch):
    monkeypatch.setattr(web.requests, "get", lambda *a, **k: FakeResponse("x" * 30_000, "text/plain"))
    first = web.fetch_url.invoke({"url": "https://example.com"})
    assert "start=12000" in first
    second = web.fetch_url.invoke({"url": "https://example.com", "start": 24_000})
    assert second == "x" * 6_000


def test_fetch_url_errors_are_returned_as_text(monkeypatch):
    assert "only http(s)" in web.fetch_url.invoke({"url": "file:///etc/passwd"})

    def boom(*a, **k):
        raise requests.ConnectionError("timed out")

    monkeypatch.setattr(web.requests, "get", boom)
    assert "fetch_url failed" in web.fetch_url.invoke({"url": "https://example.com"})

    monkeypatch.setattr(web.requests, "get", lambda *a, **k: FakeResponse("", status=404))
    assert "404" in web.fetch_url.invoke({"url": "https://example.com"})

    monkeypatch.setattr(web.requests, "get", lambda *a, **k: FakeResponse("%PDF", "application/pdf"))
    assert "not a text page" in web.fetch_url.invoke({"url": "https://example.com/a.pdf"})


# --- search ----------------------------------------------------------------------

def test_web_search_failure_returns_message(monkeypatch):
    def boom(self, *a, **k):
        raise RuntimeError("rate limited")

    monkeypatch.setattr(search.DuckDuckGoSearchResults, "_run", boom)
    assert search.web_search_tool.invoke({"query": "x"}) == "Web search failed: rate limited"


def test_wiki_search_formats_hits(monkeypatch):
    data = {"query": {"search": [{"title": "Mercedes Sosa", "snippet": '<span class="searchmatch">Sosa</span> was'}]}}
    monkeypatch.setattr(search.requests, "get", lambda *a, **k: FakeResponse(json_data=data))
    out = search.wiki_search.invoke({"query": "Mercedes Sosa"})
    assert "https://en.wikipedia.org/wiki/Mercedes_Sosa" in out
    assert "Sosa was" in out and "<span" not in out


def test_wiki_search_no_results(monkeypatch):
    monkeypatch.setattr(search.requests, "get", lambda *a, **k: FakeResponse(json_data={"query": {"search": []}}))
    assert search.wiki_search.invoke({"query": "zzz"}) == "No Wikipedia results for: zzz"


# --- youtube ---------------------------------------------------------------------

@pytest.mark.parametrize("url", [
    "https://www.youtube.com/watch?v=L1vXCYZAYYM",
    "https://youtu.be/L1vXCYZAYYM",
    "https://www.youtube.com/shorts/L1vXCYZAYYM",
    "L1vXCYZAYYM",
])
def test_extract_video_id(url):
    assert youtube.extract_video_id(url) == "L1vXCYZAYYM"


def test_youtube_transcript_formats_and_handles_errors(monkeypatch):
    snippets = [SimpleNamespace(start=0.0, text="hello"), SimpleNamespace(start=3.5, text="world")]
    monkeypatch.setattr(youtube.YouTubeTranscriptApi, "fetch", lambda self, vid, languages: snippets)
    assert youtube.youtube_transcript.invoke({"url": "https://youtu.be/L1vXCYZAYYM"}) == "[0s] hello\n[3s] world"

    def boom(self, vid, languages):
        raise RuntimeError("no captions")

    monkeypatch.setattr(youtube.YouTubeTranscriptApi, "fetch", boom)
    assert "no captions" in youtube.youtube_transcript.invoke({"url": "https://youtu.be/L1vXCYZAYYM"})
    assert "could not find a video id" in youtube.youtube_transcript.invoke({"url": "https://example.com"})


# --- Gemini-backed tools -----------------------------------------------------------

class FakeModels:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def fake_client(monkeypatch, module, response):
    models = FakeModels(response)
    monkeypatch.setattr(module, "GEMINI_RATE_LIMITER", SimpleNamespace(acquire=lambda: True))
    monkeypatch.setattr(module, "get_genai_client", lambda: SimpleNamespace(models=models))
    return models


def test_analyze_media_sends_file_and_question(assets, monkeypatch):
    (assets / "board.png").write_bytes(b"\x89PNG fake")
    models = fake_client(monkeypatch, media, SimpleNamespace(text="Qd1#"))
    out = media.analyze_media.invoke({"file_path": "board.png", "question": "best move?"})
    assert out == "Qd1#"
    part, question = models.calls[0]["contents"]
    assert part.inline_data.mime_type == "image/png" and question == "best move?"


def test_analyze_media_rejects_unsupported_and_outside(assets, monkeypatch):
    (assets / "a.csv").write_text("x")
    fake_client(monkeypatch, media, SimpleNamespace(text="unused"))
    assert "read_file" in media.analyze_media.invoke({"file_path": "a.csv", "question": "?"})
    assert "access denied" in media.analyze_media.invoke({"file_path": "../.env", "question": "?"})


def test_run_python_formats_code_output_and_answer(monkeypatch):
    parts = [
        SimpleNamespace(executable_code=SimpleNamespace(code="print(6*7)"), code_execution_result=None, text=None),
        SimpleNamespace(executable_code=None, code_execution_result=SimpleNamespace(output="42\n"), text=None),
        SimpleNamespace(executable_code=None, code_execution_result=None, text="The result is 42."),
    ]
    response = SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=parts))])
    models = fake_client(monkeypatch, code, response)
    out = code.run_python.invoke({"task": "6*7"})
    assert "Code:\nprint(6*7)" in out and "Output:\n42" in out and "The result is 42." in out
    assert models.calls[0]["config"].tools[0].code_execution is not None


def test_run_python_api_error_returns_message(monkeypatch):
    def boom():
        raise RuntimeError("429 RESOURCE_EXHAUSTED")

    monkeypatch.setattr(code, "get_genai_client", boom)
    monkeypatch.setattr(code, "GEMINI_RATE_LIMITER", SimpleNamespace(acquire=lambda: True))
    assert "429" in code.run_python.invoke({"task": "1+1"})


def test_run_python_sends_attached_file_content(assets, monkeypatch):
    pd.DataFrame({"item": ["Burger", "Soda"], "sales": [10.5, 2]}).to_excel(assets / "s.xlsx", index=False)
    response = SimpleNamespace(candidates=[])
    models = fake_client(monkeypatch, code, response)
    code.run_python.invoke({"task": "total food sales", "file_path": "s.xlsx"})
    sent = models.calls[0]["contents"]
    assert "total food sales" in sent and "Burger,10.5" in sent and "io.StringIO" in sent


def test_run_python_file_errors_are_returned(assets, monkeypatch):
    fake_client(monkeypatch, code, SimpleNamespace(candidates=[]))
    assert "access denied" in code.run_python.invoke({"task": "x", "file_path": "../.env"})
