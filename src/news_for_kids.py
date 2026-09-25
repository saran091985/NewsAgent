"""
News for Kids — Kid-friendly news explainer.

Uses LangGraph + OpenAI + Google Serper to read topics from a file,
search the web for each topic, and generate fun, kid-friendly explanations.
Results are saved to output/Newsforkids_Explanations.txt.
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from pydantic import BaseModel

# LangGraph and LangChain
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_community.utilities import GoogleSerperAPIWrapper

# Load environment variables
load_dotenv(override=True)


# ---------------------------------------------------------------------------
# 1. Load topics
# ---------------------------------------------------------------------------

def load_topics(file_path: str = "output/Final1.txt") -> list[str]:
    """Load topics from the Final1.txt file."""
    topics: list[str] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.isspace():
                # Remove leading numbers and dots (e.g., "1.U.K. PM..." -> "U.K. PM...")
                topic = re.sub(r"^\d+\.", "", line).strip()
                if topic:
                    topics.append(topic)
    return topics


# ---------------------------------------------------------------------------
# 2. Setup LLM and Search Tool
# ---------------------------------------------------------------------------

# Initialize Google Serper API search tool
serper = GoogleSerperAPIWrapper()


def limited_search(query: str) -> str:
    """Perform a web search with limited results (max 1 search)."""
    results = serper.run(query)
    # Limit to 2000 chars to keep it focused
    return results[:2000] if len(results) > 2000 else results


# Wrap in LangChain Tool
search_tool = Tool(
    name="web_search",
    func=limited_search,
    description=(
        "Search the web for information about a topic. "
        "Use this to find current information about news topics. "
        "Search once per topic."
    ),
)

tools = [search_tool]

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
llm_with_tools = llm.bind_tools(tools)


# ---------------------------------------------------------------------------
# 3. Define State for LangGraph
# ---------------------------------------------------------------------------

class State(BaseModel):
    messages: Annotated[list, add_messages]


# ---------------------------------------------------------------------------
# 4. Create LangGraph workflow
# ---------------------------------------------------------------------------

def create_news_graph():
    """Create a LangGraph for processing news topics."""
    graph_builder = StateGraph(State)

    def llm_node(state: State) -> State:
        """LLM node that processes messages and can call tools."""
        response = llm_with_tools.invoke(state.messages)
        return State(messages=[response])

    graph_builder.add_node("llm_node", llm_node)
    graph_builder.add_node("tools", ToolNode(tools=tools))

    # Add edges - tools_condition routes to "tools" if tool calls exist, otherwise END
    graph_builder.add_edge(START, "llm_node")
    graph_builder.add_conditional_edges(
        "llm_node",
        tools_condition,
        {
            "tools": "tools",
            "__end__": END,
        },
    )
    # After tools execute, loop back to llm_node
    graph_builder.add_edge("tools", "llm_node")

    return graph_builder.compile()


# ---------------------------------------------------------------------------
# 5. Process topics and generate kid-friendly explanations
# ---------------------------------------------------------------------------

KID_FRIENDLY_SYSTEM_MESSAGE = (
    "You are a friendly and fun news explainer for kids! "
    "Your job is to explain news topics in a cool, exciting, and easy-to-understand "
    "way that kids will love.\n\n"
    "Format your response as a fun, friendly explanation that a kid would enjoy reading. "
    "IMPORTANT: Do not include or display any dates in your explanation."
)


def process_topics(topics: list[str], graph) -> list[str]:
    """Run each topic through the LangGraph and collect explanations."""
    all_explanations: list[str] = []

    print("Starting to process topics...\n")
    print("=" * 80)

    for i, topic in enumerate(topics, 1):
        print(f"\nProcessing topic {i}/{len(topics)}: {topic}")

        messages = [
            SystemMessage(content=KID_FRIENDLY_SYSTEM_MESSAGE),
            HumanMessage(
                content=(
                    "Explain the below topic in cool and kids friendly way - "
                    "search the internet(search once only) and get the correct "
                    "information before summarizing it.\n\n"
                    f"{topic}"
                )
            ),
        ]

        initial_state = State(messages=messages)
        try:
            final_state = graph.invoke(initial_state)

            # Extract the explanation from the final messages
            explanation = ""
            messages_list = final_state.get("messages", [])
            for msg in reversed(messages_list):
                if isinstance(msg, AIMessage):
                    if hasattr(msg, "content") and msg.content and msg.content.strip():
                        explanation = msg.content
                        break

            if explanation:
                explanation_text = f"**Topic {i}: {topic}**\n\n{explanation}\n\n"
                all_explanations.append(explanation_text)
                print(f"✓ Successfully processed topic {i}")
            else:
                print(f"⚠ No explanation generated for topic {i}")

        except Exception as e:
            print(f"✗ Error processing topic {i}: {e}")
            error_text = f"**Topic {i}: {topic}**\n\n[Error: Could not process this topic]\n\n"
            all_explanations.append(error_text)

    print("\n" + "=" * 80)
    print(f"\nCompleted processing {len(topics)} topics!")
    return all_explanations


# ---------------------------------------------------------------------------
# 6. Save explanations to file
# ---------------------------------------------------------------------------

def save_explanations(
    all_explanations: list[str],
    output_file: str = "output/Newsforkids_Explanations.txt",
) -> None:
    """Save all explanations to a text file."""
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("NEWS FOR KIDS - Kid-Friendly News Explanations\n")
        f.write(f"Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}\n")
        f.write("=" * 80 + "\n\n")

        for explanation in all_explanations:
            f.write(explanation)
            f.write("-" * 80 + "\n\n")

    print(f"✓ All explanations saved to: {output_file}")
    print(f"Total topics processed: {len(all_explanations)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    topics = load_topics()
    print(f"Loaded {len(topics)} topics:")
    for i, topic in enumerate(topics, 1):
        print(f"{i}. {topic}")

    print("\nLLM and Search Tool initialized successfully!")

    graph = create_news_graph()
    print("LangGraph created successfully!")

    explanations = process_topics(topics, graph)
    save_explanations(explanations)

    # Preview
    output_file = "output/Newsforkids_Explanations.txt"
    print("\nPreview of saved content:\n")
    with open(output_file, "r", encoding="utf-8") as f:
        content = f.read()
        print(content[:1000])
        if len(content) > 1000:
            print("\n... (content truncated for preview) ...")
