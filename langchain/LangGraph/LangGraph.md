# 快速入门

在这个全面的快速入门中，我们将在 LangGraph 中构建一个支持聊天机器人，它可以：

- 通过搜索网络来回答常见问题
- 保持通话过程中的对话状态
- 将复杂的查询发送给人工进行审核
- 使用自定义状态控制其行为
- 回顾并探索替代对话路径

我们将从一个基本的聊天机器人开始，逐步添加更复杂的功能，并在此过程中介绍关键的 LangGraph 概念。

## 设置

首先，安装所需的软件包：

```bash
%%capture --no-stderr 
%pip install -U langgraph langsmith 
# 用于本教程；不是 LangGraph 的要求 
%pip install -U langchain_anthropic
```

接下来，设置您的 API 密钥：

```python
import getpass
import os


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("ANTHROPIC_API_KEY")
```

> 设置[LangSmith](https://smith.langchain.com/)进行 LangGraph 开发
>
> 注册 LangSmith 以快速发现问题并提高 LangGraph 项目的性能。LangSmith 可让您使用跟踪数据来调试、测试和监控使用 LangGraph 构建的 LLM 应用程序 —[在此处](https://docs.smith.langchain.com/)了解有关如何开始的更多信息。

## 第 1 部分：构建基本聊天机器人

我们首先使用 LangGraph 创建一个简单的聊天机器人。这个聊天机器人将直接响应用户消息。虽然很简单，但它将说明使用 LangGraph 构建的核心概念。在本节结束时，您将拥有一个构建好的初级聊天机器人。

```python
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)
```

> 笔记
>
> 定义图时要做的第一件事是定义`State`图的。`State`由图的模式以及指定如何将更新应用于状态的 Reducer 函数组成。在我们的示例中，`State`是`TypedDict`具有单个键的：`messages`。该`messages`键使用 Reducer 函数注释[`add_messages`](https://langchain-ai.github.io/langgraph/reference/graphs/?h=add+messages#add_messages)，它告诉 LangGraph 将新消息附加到现有列表，而不是覆盖它。没有注释的状态键将被每次更新覆盖，存储最新值。查看[此概念指南](https://langchain-ai.github.io/langgraph/reference/graphs/?h=add+messages#add_messages)以了解有关状态、Reducer 和其他低级概念的更多信息。

所以现在我们的图表知道两件事：

1. 我们定义的每个都`node`将接收当前`State`作为输入并返回更新该状态的值。
2. `messages`将被*附加*到当前列表，而不是直接覆盖。这是通过语法[`add_messages`](https://langchain-ai.github.io/langgraph/reference/graphs/?h=add+messages#add_messages)中的预建函数传达的`Annotated`。

接下来，添加一个“ `chatbot`”节点。节点代表工作单元。它们通常是常规的 Python 函数。

```python
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-3-haiku-20240307")


def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)
```

**注意**节点函数如何`chatbot`将当前`State`作为输入并返回一个包含`messages`键“messages”下更新列表的字典。这是所有 LangGraph 节点函数的基本模式。

`add_messages`我们的函数会将`State`llm 的响应消息附加到状态中已有的任何消息中。

接下来，添加一个`entry`点。这将告诉我们的图表每次运行时**从哪里开始工作。**

```python
graph_builder.add_edge(START, "chatbot")
```

同样，设置一个`finish`点。这将指示图表**“任何时候运行此节点，您都可以退出。”**

```python
graph_builder.add_edge("chatbot", END)
```

最后，我们希望能够运行我们的图表。为此，请`compile()`在图表构建器上调用“”。这将创建一个“ `CompiledGraph`”，我们可以在状态上使用“调用”。

```python
graph = graph_builder.compile()
```

`get_graph`您可以使用方法和其中一种“绘制”方法（例如`draw_ascii`或 ）来可视化图形`draw_png`。`draw`每个方法都需要额外的依赖项。

```python
from IPython.display import Image, display

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass
```

![image.png](assets/image.png?t=1725977029039)

现在让我们运行聊天机器人！

**提示：**您可以随时输入“quit”、“exit”或“q”退出聊天循环。

```python
while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    for event in graph.stream({"messages": ("user", user_input)}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)
```

```
User:  what's langgraph all about?
Assistant: LangGraph is a new open-source deep learning framework that focuses on enabling efficient training and deployment of large language models. Some key things to know about LangGraph:

1. Efficient Training: LangGraph is designed to accelerate the training of large language models by leveraging advanced optimization techniques and parallelization strategies.

2. Modular Architecture: LangGraph has a modular architecture that allows for easy customization and extension of language models, making it flexible for a variety of NLP tasks.

3. Hardware Acceleration: The framework is optimized for both CPU and GPU hardware, allowing for efficient model deployment on a wide range of devices.

4. Scalability: LangGraph is designed to handle large-scale language models with billions of parameters, enabling the development of state-of-the-art NLP applications.

5. Open-Source: LangGraph is an open-source project, allowing developers and researchers to collaborate, contribute, and build upon the framework.

6. Performance: The goal of LangGraph is to provide superior performance and efficiency compared to existing deep learning frameworks, particularly for training and deploying large language models.

Overall, LangGraph is a promising new deep learning framework that aims to address the challenges of building and deploying advanced natural language processing models at scale. It is an active area of research and development, with the potential to drive further advancements in the field of language AI.
User:  hm that doesn't seem right...
Assistant: I'm sorry, I don't have enough context to determine what doesn't seem right. Could you please provide more details about what you're referring to? That would help me better understand and respond appropriately.
User:  q
Goodbye!
```

**恭喜！**您已使用 LangGraph 构建了第一个聊天机器人。该机器人可以通过接收用户输入并使用 LLM 生成响应来进行基本对话。您可以在提供的链接处检查上述调用的[LangSmith Trace 。](https://smith.langchain.com/public/29ab0177-1177-4d25-9341-17ae7d94e0e0/r)

但是，您可能已经注意到，机器人的知识仅限于其训练数据中的内容。在下一部分中，我们将添加一个网络搜索工具来扩展机器人的知识并使其更强大。

以下是本节的完整代码，供您参考：

```python
from typing import Annotated

from langchain_anthropic import ChatAnthropic
from typing_extensions import TypedDict

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


llm = ChatAnthropic(model="claude-3-haiku-20240307")


def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)
graph_builder.set_entry_point("chatbot")
graph_builder.set_finish_point("chatbot")
graph = graph_builder.compile()
```



## 第 2 部分：使用工具增强聊天机器人

为了处理我们的聊天机器人无法“凭记忆”回答的问题，我们将集成一个网络搜索工具。我们的机器人可以使用此工具查找相关信息并提供更好的响应。

#### 要求

在开始之前，请确保您已安装必要的软件包并设置了 API 密钥：

[首先，安装使用Tavily 搜索引擎](https://python.langchain.com/v0.2/docs/integrations/tools/tavily_search/)的要求，并设置您的[TAVILY_API_KEY](https://tavily.com/)。

