from fastapi import FastAPI, Request
import uvicorn, json, datetime

from langchain import SerpAPIWrapper
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain.utilities import ArxivAPIWrapper
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from Qwen_tools import SearchVector

from typing import Dict, Tuple
import os
import json
import requests

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# 为了使用谷歌搜索（SERPAPI）， WolframAlpha，您需要自行申请它们的 API KEY，然后填入此处
os.environ['SERPAPI_API_KEY'] = '9a860ad90aa50f9428933bb26c636e7cceb7155133e8a51dcb2d6d460df4d9ba'
os.environ['WOLFRAM_ALPHA_APPID'] = 'RH4ERE-VHKHLVUA8A'

search = SerpAPIWrapper()
WolframAlpha = WolframAlphaAPIWrapper()
arxiv = ArxivAPIWrapper()
python=PythonAstREPLTool()
vector = SearchVector()

def tool_wrapper_for_qwen(tool):
    def tool_(query):
        query = json.loads(query)["query"]
        return tool.run(query)
    return tool_

# 以下是给千问看的工具描述：
TOOLS = [
    {
        'name_for_human':
            'google search',
        'name_for_model':
            'Search',
        'description_for_model':
            'useful for when you need to answer questions about current events.',
        'parameters': [{
            "name": "query",
            "type": "string",
            "description": "search query of google",
            'required': True
        }], 
        'tool_api': tool_wrapper_for_qwen(search)
    },
    {
        'name_for_human':
            'Wolfram Alpha',
        'name_for_model':
            'Math',
        'description_for_model':
            'Useful for when you need to answer questions about Math, Science, Technology, Culture, Society and Everyday Life.',
        'parameters': [{
            "name": "query",
            "type": "string",
            "description": "the problem to solved by Wolfram Alpha",
            'required': True
        }], 
        'tool_api': tool_wrapper_for_qwen(WolframAlpha)
    },  
    {
        'name_for_human':
            'arxiv',
        'name_for_model':
            'Arxiv',
        'description_for_model':
            'A wrapper around Arxiv.org Useful for when you need to answer questions about Physics, Mathematics, Computer Science, Quantitative Biology, Quantitative Finance, Statistics, Electrical Engineering, and Economics from scientific articles on arxiv.org.',
        'parameters': [{
            "name": "query",
            "type": "string",
            "description": "the document id of arxiv to search",
            'required': True
        }], 
        'tool_api': tool_wrapper_for_qwen(arxiv)
    },
    {
        'name_for_human':
            'python',
        'name_for_model':
            'python',
        'description_for_model':
            "A Python shell. Use this to execute python commands. When using this tool, sometimes output is abbreviated - Make sure it does not look abbreviated before using it in your answer. "
            "Don't add comments to your python code.",
        'parameters': [{
            "name": "query",
            "type": "string",
            "description": "a valid python command.",
            'required': True
        }],
        'tool_api': tool_wrapper_for_qwen(python)
    },
    {
        'name_for_human':
            'vector_database',
        'name_for_model':
            'vector_database',
        'description_for_model':
            "It is very useful when you need to query the local database to answer questions.",
        'parameters': [{
            "name": "query",
            "type": "string",
            "description": "Vector database of local data.",
            'required': True
        }],
        'tool_api': tool_wrapper_for_qwen(vector)
    }
]

TOOL_DESC = """{name_for_model}: Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters} Format the arguments as a JSON object."""

REACT_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

{tool_descs}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {query}"""

def build_planning_prompt(TOOLS, query):
    tool_descs = []
    tool_names = []
    for info in TOOLS:
        tool_descs.append(
            TOOL_DESC.format(
                name_for_model=info['name_for_model'],
                name_for_human=info['name_for_human'],
                description_for_model=info['description_for_model'],
                parameters=json.dumps(
                    info['parameters'], ensure_ascii=False),
            )
        )
        tool_names.append(info['name_for_model'])
    tool_descs = '\n\n'.join(tool_descs)
    tool_names = ','.join(tool_names)

    prompt = REACT_PROMPT.format(tool_descs=tool_descs, tool_names=tool_names, query=query)
    return prompt


def parse_latest_plugin_call(text: str) -> Tuple[str, str]:
    i = text.rfind('\nAction:')
    j = text.rfind('\nAction Input:')
    k = text.rfind('\nObservation:')
    if 0 <= i < j:  # If the text has `Action` and `Action input`,
        if k < j:  # but does not contain `Observation`,
            # then it is likely that `Observation` is ommited by the LLM,
            # because the output text may have discarded the stop word.
            text = text.rstrip() + '\nObservation:'  # Add it back.
            k = text.rfind('\nObservation:')
    if 0 <= i < j < k:
        plugin_name = text[i + len('\nAction:'):j].strip()
        plugin_args = text[j + len('\nAction Input:'):k].strip()
        return plugin_name, plugin_args
    return '', ''

def use_api(tools, response):
    use_toolname, action_input = parse_latest_plugin_call(response)
    if use_toolname == "":
        return "no tool founds"

    used_tool_meta = list(filter(lambda x: x["name_for_model"] == use_toolname, tools))
    if len(used_tool_meta) == 0:
        return "no tool founds"
    
    api_output = used_tool_meta[0]["tool_api"](action_input)
    return api_output


def chat_qweb_72B(prompt, stop):
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer sk-ChtJNYJD1sm5FqwA7bE8EfFa3eE847Fa9758E5626d64Cc9a"
    }
    data = {
        "model": "Qwen-72B-Chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "top_p": 0.2,
        "stop": stop
    }

    response = requests.post("http://70.182.56.14:11000/v1/chat/completions", data=json.dumps(data),
                             headers=headers).content
    return str(response, encoding='utf-8')


def main(query, choose_tools):
    prompt = build_planning_prompt(choose_tools, query) # 组织prompt
    print(prompt)
    stop = ["Observation:", "Observation:\n"]
    response = chat_qweb_72B(prompt=prompt, stop=stop)
    print(response)
    response = json.loads(response).get('choices')[0].get('message').get('content')

    while "Final Answer:" not in response: # 出现final Answer时结束
        api_output = use_api(choose_tools, response) # 抽取入参并执行api
        api_output = str(api_output) # 部分api工具返回结果非字符串格式需进行转化后输出
        if "no tool founds" == api_output:
            break
        print("\033[32m" + response + "\033[0m" + "\033[34m" + ' ' + api_output + "\033[0m")
        prompt = prompt + response + ' ' + api_output # 合并api输出
        response = json.loads(chat_qweb_72B(prompt=prompt, stop=stop)).get('choices')[0].get('message').get('content')

    # print("\033[32m" + response + "\033[0m")
    return "\033[32m" + response + "\033[0m"


app = FastAPI()


@app.post("/")
async def create_item(request: Request):
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    query = json_post_list.get('query')
    ###################
    choose_tools = TOOLS # 选择备选工具
    result = main(query, choose_tools)
    ######################
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "query": query,
        "response": result,
        "status": 200,
        "time": time
    }
    print(answer)
    return answer


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=9700, workers=1)
