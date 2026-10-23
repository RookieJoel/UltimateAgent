from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import START, MessagesState, StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, trim_messages
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
from agent.config import GEMINI_MODEL
from agent.prompts import LAST_STEP_NOTE, build_system_prompt
from agent.rate_limit import GEMINI_RATE_LIMITER
from agent.tools import TOOLS

load_dotenv()  # Load environment variables from .env file

if not os.getenv("GOOGLE_API_KEY"):
    raise RuntimeError("GOOGLE_API_KEY is not set (use .env locally or Space Secrets)")

model = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    temperature=1.0,  # Gemini 3.0+ defaults to 1.0
    max_tokens=None,
    timeout=None,
    max_retries=2,
    rate_limiter=GEMINI_RATE_LIMITER,
)

MAX_STEP = 10

class FinalAnswerFormatter(BaseModel):
    final_answer : str = Field(description="finish your answer with the following template: YOUR FINAL ANSWER. YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.")

class AgentState(MessagesState):
    file_path : str
    current_step : int = 0
    final_answer : FinalAnswerFormatter

def formatter(state : AgentState) -> dict:
    """The formatter node that formats the final response for the user."""
    #get the question from the last human message
    last_human_message = next((msg for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)), None)
    #get the final answer from the model and format it according to the FinalAnswerFormatter
    agent_final_ans = state["messages"][-1]
    #form the final answer using the FinalAnswerFormatter
    to_formatter = [last_human_message] + [agent_final_ans] + [HumanMessage(content="Please format your answer according to the FinalAnswerFormatter.")]
    return model.with_structured_output(FinalAnswerFormatter).invoke(to_formatter)

# Tools (defined in agent/tools/)
tools = TOOLS
model_with_tools = model.bind_tools(tools)

def assistant(state: AgentState) -> dict:
    """The assistant node that routes the conversation to the appropriate tool or provides a direct response."""
    #retrieve current step and max step from the state
    current_step = state["current_step"] if "current_step" in state else 0
    #planner-style system prompt is prepended on every call (not stored in state)
    system_message = SystemMessage(content=build_system_prompt(state.get("file_path"), MAX_STEP))
    messages = [system_message] + state["messages"]
    if current_step == MAX_STEP-1:
        #last step: call the plain model (no tools) so the reply is text
        updated_step = current_step + 1
        response = model.invoke(messages + [HumanMessage(content=LAST_STEP_NOTE)])
        return {
            "messages": [response],
            "current_step": updated_step
        }

    elif current_step < MAX_STEP:
        #call the model with tools
        updated_step = current_step + 1
        response = model_with_tools.invoke(messages)
        return {
            "messages": [response],
            "current_step": updated_step
        }


# graph
builder = StateGraph(AgentState)

#nodes
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_node("formatter", formatter)

#route decision
def route_decision(state: AgentState) -> str:
    """Determine the next node based on the current state."""
    if state["current_step"] >= MAX_STEP:
        return "formatter"
    elif tools_condition(state) == "tools":
        return "tools"
    else:
        return "formatter"
#edges
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    route_decision,
    {
        "tools": "tools",
        "formatter": "formatter"
    }
)
builder.add_edge("tools", "assistant")
builder.add_edge("formatter", END)
agent = builder.compile()


