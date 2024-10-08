import os
os.environ['TAVILY_API_KEY'] = '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'

from typing import Annotated
# from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)
tool = TavilySearchResults(max_results=2)
tools = [tool]
# llm = ChatAnthropic(model="claude-3-haiku-20240307")
llm = ChatOpenAI(model='Qwen2-72B-Instruct', base_url='http://70.182.56.16:11000/v1/', api_key='sk-ChtJNYJD1sm5FqwA7bE8EfFa3eE847Fa9758E5626d64Cc9a',)
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)
tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")
graph = graph_builder.compile()


from IPython.display import Image, display
import matplotlib.pyplot as plt


try:
    display(Image(graph.get_graph().draw_mermaid_png()))

    from PIL import Image
    import io
    image_bytes = io.BytesIO(graph.get_graph().draw_mermaid_png())
    image = Image.open(image_bytes)
    image.show()


except Exception:
    # This requires some extra dependencies and is optional
    pass


while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break

    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            if isinstance(value["messages"][-1], BaseMessage):
                print("Assistant:", value["messages"][-1].content)
