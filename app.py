"""
News Agent Application
A kid-friendly news assistant using LangGraph and Gradio
"""

from dotenv import load_dotenv
from typing import Annotated
from pydantic import BaseModel
from datetime import datetime

# LangGraph and LangChain
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
from langchain_core.messages import SystemMessage
from langchain_community.utilities import GoogleSerperAPIWrapper

# Gradio
import gradio as gr
import os

# Load environment variables
load_dotenv(override=True)


# Search Tool Setup
serper = GoogleSerperAPIWrapper()

tool_search = Tool(
    name="search",
    func=serper.run,
    description="Useful for searching today's news and information online. Use targeted, specific queries that combine related topics when possible to minimize search calls. Examples: 'today's top space and science news', 'latest sports achievements today', 'today's technology innovations','today's happenings around the world and India'"
)

tools = [tool_search]


# State Definition
class State(BaseModel):
    messages: Annotated[list, add_messages]


# Graph Setup
graph_builder = StateGraph(State)

llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)


def chatgpt_LLM_node(old_state: State) -> State:
    """LLM node that processes messages and can call tools."""
    response = llm_with_tools.invoke(old_state.messages)
    new_state = State(messages=[response])
    return new_state


# Add nodes to graph
graph_builder.add_node("chatgpt_LLM_node", chatgpt_LLM_node)
graph_builder.add_node("tools", ToolNode(tools=tools))

# Add edges to graph
graph_builder.add_edge(START, "chatgpt_LLM_node")
graph_builder.add_conditional_edges("chatgpt_LLM_node", tools_condition, "tools")
graph_builder.add_edge("tools", "chatgpt_LLM_node")
graph_builder.add_edge("chatgpt_LLM_node", END)

# Compile graph
graph = graph_builder.compile()


# Admin/System message for kid-friendly news agent
def get_admin_message() -> str:
    """Generate admin message with today's date dynamically."""
    today = datetime.now().strftime("%B %d, %Y")  # e.g., "January 15, 2024"
    return f"""You are a friendly news assistant helping a 9-year-old host create a daily news summary for {today}. Your role is to:

1. **Content Focus**: Prioritize news about:
   - Innovations and technology breakthroughs
   - Space exploration and discoveries
   - Sports achievements and events
   - Positive global happenings and cultural events
   - Science discoveries and nature stories
   - Educational and inspiring stories

2. **Content Avoidance**: 
   - Minimize or exclude political news, conflicts, and controversies
   - Avoid scary or distressing content
   - Skip complex political debates or policy discussions

3. **Tone & Style**:
   - Use simple, clear language appropriate for a 9-year-old
   - Make news exciting and engaging
   - Explain complex concepts in simple terms
   - Keep it positive and inspiring when possible

4. **Cost Efficiency**:
   - When searching for today's news ({today}), use targeted, specific search queries
   - Combine related topics in single searches when possible (e.g., "today's space news and innovations")
   - Prioritize the most important stories to minimize search calls
   - Use broad but specific queries like "today's top science and space news" instead of multiple separate searches

5. **Output Format**:
   - Present news in a clear, organized manner
   - Group related stories together
   - Make it easy for a young host to read and present
   - Always reference that this is news for {today}

Remember: Your goal is to help create an informative, engaging, and age-appropriate news summary for {today} that inspires curiosity and learning."""


def chat(user_input: str, history):
    """Chat function for Gradio interface."""
    # Add system message at the beginning with today's date
    system_msg = SystemMessage(content=get_admin_message())
    user_msg = {"role": "user", "content": user_input}
    messages = [system_msg, user_msg]
    state = State(messages=messages)
    result = graph.invoke(state)
    return result["messages"][-1].content


# Launch Gradio interface
if __name__ == "__main__":
    demo = gr.ChatInterface(chat)
    #local deployment - 
    #demo.launch() 
    #Render Deployment - 
    port = int(os.getenv("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)

