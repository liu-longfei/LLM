"""Translate Langgraph"""

import operator
import os
import uuid
from typing import Annotated, Any, Dict, List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from munch import DefaultMunch
from typing_extensions import TypedDict

from bisheng.graph.langgraph_build import build_graph


# 定义模板默认节点
# 类型1 LLM
def create_chat_node(llm_config):
    """
    Model Config: LLM参数
    Generate Config: LLM输出
    """
    # LLM
    llm = ChatOpenAI(
        model=llm_config.llm.model,
        temperature=llm_config.llm.temperature,
        base_url=llm_config.llm.base_url,
        api_key=llm_config.llm.api_key,
    )
    # Chat Chain
    prompt = ChatPromptTemplate.from_messages(llm_config.prompt)
    chat_chain = prompt | llm | StrOutputParser()

    def generate(state: Dict[str, Any]) -> Dict[str, Any]:
        question = state["question"]
        documents = state.get("documents", None)
        generation_state_list = state.get("generation", [])

        if documents and "documents" in prompt:
            generation = chat_chain.invoke({"documents": documents, "question": question})
            return {"documents": documents, "question": question, "generation": generation_state_list + [generation]}
        else:
            generation = chat_chain.invoke({"question": question})
            return {"question": question, "generation": generation_state_list + [generation]}

    generate.__name__ = f"generate_{uuid.uuid4()}"
    return generate


llm_config = {
    "llm": {
        "model": "Qwen2-72B-Instruct",
        "temperature": 0,
        "base_url": os.environ["OPENAI_BASE_URL"],
        "api_key": os.environ["OPENAI_API_KEY"],
    },
    "prompt": [("system", "你是一个翻译人员，擅长中英文翻译"), ("human", "{question}")],
}
llm_judge_config = {
    "llm": {
        "model": "Qwen2.5-72B-Instruct",
        "temperature": 0,
        "base_url": os.environ["OPENAI_BASE_URL"],
        "api_key": os.environ["OPENAI_API_KEY"],
    },
    "prompt": [
        ("system", "你是一个翻译评委，你会收到两份翻译结果，请你参考两份翻译内容，对用户输入进行最后的翻译。"),
        ("human", "翻译结果\n{generation}"),
        ("human", "{question}"),
    ],
}


# 定义应用时需要定义State类
class State(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: Annotated[list, operator.add]  # 收集所有的node返回的generation
    documents: List[str]


###############################
translator1 = create_chat_node(DefaultMunch.fromDict(llm_config))
translator2 = create_chat_node(DefaultMunch.fromDict(llm_config))
translator_judge = create_chat_node(DefaultMunch.fromDict(llm_judge_config))
nodes = [("translator1", translator1), ("translator2", translator2), ("translator_judge", translator_judge)]
# nodes = [("translator1", translator1)]
edges = [
    ("START", "translator1"),
    ("START", "translator2"),
    ("translator1", "translator_judge"),
    ("translator2", "translator_judge"),
    ("translator_judge", "END"),
]
state = StateGraph(State)
app = build_graph(state, nodes, edges)
print(app.invoke({"question": "翻译以下：超融合基础设施"}))

