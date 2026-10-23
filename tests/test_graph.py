"""Graph wiring tests with fake models (no Gemini calls)."""
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import agent.agent as A
from agent.tools import TOOLS


class FakeModel:
    """Records the messages it receives and replies with plain text."""

    def __init__(self, reply="The answer is 3."):
        self.reply = reply
        self.seen = []

    def invoke(self, messages):
        self.seen.append(messages)
        return AIMessage(content=self.reply)

    def with_structured_output(self, schema):
        class Structured:
            def invoke(self, messages):
                return schema(final_answer="3")

        return Structured()


@pytest.fixture
def fake_models(monkeypatch):
    tools_model, plain_model = FakeModel(), FakeModel()
    monkeypatch.setattr(A, "model_with_tools", tools_model)
    monkeypatch.setattr(A, "model", plain_model)
    return tools_model, plain_model


def test_all_tools_are_bound_to_tool_node():
    names = {t.name for t in TOOLS}
    assert names == {
        "web_search_tool", "wiki_search", "fetch_url", "youtube_transcript",
        "read_file", "analyze_media", "run_python",
    }
    assert set(A.agent.get_graph().nodes) >= {"assistant", "tools", "formatter"}


def test_assistant_prepends_planner_system_prompt(fake_models):
    tools_model, _ = fake_models
    out = A.agent.invoke({"messages": [HumanMessage(content="How many albums?")]})
    sent = tools_model.seen[0]
    assert isinstance(sent[0], SystemMessage)
    assert "Plan -> Act -> Verify" in sent[0].content
    assert "attached file" not in sent[0].content
    assert out["final_answer"] == "3"
    # system prompt is not stored in state
    assert not any(isinstance(m, SystemMessage) for m in out["messages"])


def test_assistant_mentions_attached_file(fake_models):
    tools_model, _ = fake_models
    A.agent.invoke({"messages": [HumanMessage(content="What is in the sheet?")], "file_path": "sales.xlsx"})
    assert "An attached file is available at: sales.xlsx" in tools_model.seen[0][0].content


def test_last_step_tells_model_not_to_invent_placeholders(fake_models):
    _, plain_model = fake_models
    state = {"messages": [HumanMessage(content="Q")], "current_step": A.MAX_STEP - 1}
    A.assistant(state)
    assert "placeholder" in plain_model.seen[0][-1].content
