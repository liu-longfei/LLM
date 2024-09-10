from typing import Annotated

from langchain_anthropic import ChatAnthropic
from typing_extensions import TypedDict

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

from langchain_openai import ChatOpenAI

import langchain
langchain.verbose = False
langchain.debug = False
langchain.llm_cache = False


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


# llm = ChatAnthropic(model="claude-3-haiku-20240307")
llm = ChatOpenAI(model='qwen2:7b', base_url='http://localhost:11434/v1/', api_key='ollama',)


def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)
graph_builder.set_entry_point("chatbot")
graph_builder.set_finish_point("chatbot")
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
    for event in graph.stream({"messages": ("user", user_input)}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)