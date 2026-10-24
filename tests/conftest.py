import os

# Tests use fakes; keep them out of LangSmith (agent.agent's load_dotenv does not override this).
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
