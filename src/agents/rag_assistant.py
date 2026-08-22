import json
import re
from datetime import datetime
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import (
    RunnableConfig,
    RunnableLambda,
    RunnableSerializable,
)
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import ToolNode
from langgraph.types import StreamWriter

from agents.safeguard import Safeguard, SafeguardOutput, SafetyAssessment
from agents.tools import database_search
from agents.utils import CustomData
from core import get_model, settings


class AgentState(MessagesState, total=False):
    """`total=False` is PEP589 specs.

    documentation: https://typing.readthedocs.io/en/latest/spec/typeddict.html#totality
    """

    safety: SafeguardOutput
    remaining_steps: RemainingSteps
    citations: list[dict]


tools = [database_search]


current_date = datetime.now().strftime("%B %d, %Y")
instructions = f"""
    You are DocPilot, a grounded knowledge assistant that answers questions based on evidence from indexed PDF and DOCX documents.
    Today's date is {current_date}.

    RULES:
    1. For document-related questions, always call Database_Search first.
    2. Only use document content returned by the tool to support factual claims.
    3. Do NOT supplement gaps in retrieved evidence with external knowledge.
    4. If retrieved content is empty, irrelevant, or insufficient, clearly state:
       "I couldn't find sufficient evidence in the DocPilot knowledge base."
    5. Do NOT fabricate file names, sources, links, citations, policies, commands, or facts.
    6. End every answer that has supporting evidence with a "Sources:" section listing the sources used.
    7. Under "Sources:", list only the unique source values returned by the tool. Do not invent URLs.
    8. If no source supports the answer, write "Sources: None".
    9. The user cannot see the raw tool response. Summarize the relevant evidence in your own words.
    """


def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    bound_model = model.bind_tools(tools)
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=instructions)] + state["messages"],
        name="StateModifier",
    )
    return preprocessor | bound_model  # type: ignore[return-value]


def format_safety_message(safety: SafeguardOutput) -> AIMessage:
    content = (
        f"This conversation was flagged for unsafe content: {', '.join(safety.unsafe_categories)}"
    )
    return AIMessage(content=content)


async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m)
    response = await model_runnable.ainvoke(state, config)

    if state["remaining_steps"] < 2 and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, need more steps to process this request.",
                )
            ]
        }
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


async def safeguard_input(state: AgentState, config: RunnableConfig) -> AgentState:
    safeguard = Safeguard()
    safety_output = await safeguard.ainvoke(state["messages"])
    return {"safety": safety_output, "messages": []}


async def block_unsafe_content(state: AgentState, config: RunnableConfig) -> AgentState:
    safety: SafeguardOutput = state["safety"]
    return {"messages": [format_safety_message(safety)]}


_CITATIONS_RE = re.compile(r"CITATIONS_JSON: (\[.*\])\s*$", re.DOTALL)


async def collect_citations(state: AgentState, writer: StreamWriter) -> AgentState:
    """工具调用后：解析 ToolMessage 末尾的 CITATIONS_JSON，写入 state 并向前端发射。

    CustomData 只走 custom 流（不进 messages），LLM 不会看到它。
    """
    citations: list[dict] = []
    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage):
            match = _CITATIONS_RE.search(str(msg.content))
            if match:
                try:
                    citations = json.loads(match.group(1))
                except json.JSONDecodeError:
                    citations = []
            break

    if citations:
        CustomData(data={"docpilot_citations": citations}).dispatch(writer)
    return {"citations": citations, "messages": []}


# Define the graph
agent = StateGraph(AgentState)
agent.add_node("model", acall_model)
agent.add_node("tools", ToolNode(tools))
agent.add_node("collect_citations", collect_citations)
agent.add_node("guard_input", safeguard_input)
agent.add_node("block_unsafe_content", block_unsafe_content)
agent.set_entry_point("guard_input")


# Check for unsafe input and block further processing if found
def check_safety(state: AgentState) -> Literal["unsafe", "safe"]:
    safety: SafeguardOutput = state["safety"]
    match safety.safety_assessment:
        case SafetyAssessment.UNSAFE:
            return "unsafe"
        case _:
            return "safe"


agent.add_conditional_edges(
    "guard_input", check_safety, {"unsafe": "block_unsafe_content", "safe": "model"}
)

# Always END after blocking unsafe content
agent.add_edge("block_unsafe_content", END)

# Always run "collect_citations" then "model" after "tools"
agent.add_edge("tools", "collect_citations")
agent.add_edge("collect_citations", "model")


# After "model", if there are tool calls, run "tools". Otherwise END.
def pending_tool_calls(state: AgentState) -> Literal["tools", "done"]:
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        raise TypeError(f"Expected AIMessage, got {type(last_message)}")
    if last_message.tool_calls:
        return "tools"
    return "done"


agent.add_conditional_edges("model", pending_tool_calls, {"tools": "tools", "done": END})

rag_assistant = agent.compile()
