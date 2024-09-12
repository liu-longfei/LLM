# Hybrid Search

The standard search in LangChain is done by vector similarity. However, a number of vectorstores implementations (Astra DB, ElasticSearch, Neo4J, AzureSearch, ...) also support more advanced search combining vector similarity search and other search techniques (full-text, BM25, and so on). This is generally referred to as "Hybrid" search.

**Step 1: Make sure the vectorstore you are using supports hybrid search**

At the moment, there is no unified way to perform hybrid search in LangChain. Each vectorstore may have their own way to do it. This is generally exposed as a keyword argument that is passed in during `similarity_search`. By reading the documentation or source code, figure out whether the vectorstore you are using supports hybrid search, and, if so, how to use it.

**Step 2: Add that parameter as a configurable field for the chain**

This will let you easily call the chain and configure any relevant flags at runtime. See [this documentation](https://python.langchain.com/v0.1/docs/expression_language/primitives/configure/) for more information on configuration.

**Step 3: Call the chain with that configurable field**

Now, at runtime you can call this chain with configurable field.

## Code Example[](https://python.langchain.com/v0.1/docs/use_cases/question_answering/hybrid/#code-example)

Let's see a concrete example of what this looks like in code. We will use the Cassandra/CQL interface of Astra DB for this example.

Install the following Python package:

```python
!pip install "cassio>=0.1.7"
```

Get the [connection secrets](https://docs.datastax.com/en/astra/astra-db-vector/get-started/quickstart.html).

Initialize cassio:

```python
import cassio

cassio.init(
    database_id="Your database ID",
    token="Your application token",
    keyspace="Your key space",
)
```

Create the Cassandra VectorStore with a standard [index analyzer](https://docs.datastax.com/en/astra/astra-db-vector/cql/use-analyzers-with-cql.html). The index analyzer is needed to enable term matching.

```python
from cassio.table.cql import STANDARD_ANALYZER
from langchain_community.vectorstores import Cassandra
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
vectorstore = Cassandra(
    embedding=embeddings,
    table_name="test_hybrid",
    body_index_options=[STANDARD_ANALYZER],
    session=None,
    keyspace=None,
)

vectorstore.add_texts(
    [
        "In 2023, I visited Paris",
        "In 2022, I visited New York",
        "In 2021, I visited New Orleans",
    ]
)
```

#### API Reference:

- [Cassandra](https://api.python.langchain.com/en/latest/vectorstores/langchain_community.vectorstores.cassandra.Cassandra.html)
- [OpenAIEmbeddings](https://api.python.langchain.com/en/latest/embeddings/langchain_openai.embeddings.base.OpenAIEmbeddings.html)

If we do a standard similarity search, we get all the documents:

```python
vectorstore.as_retriever().invoke("What city did I visit last?")
```

```output
[Document(page_content='In 2022, I visited New York'),
Document(page_content='In 2023, I visited Paris'),
Document(page_content='In 2021, I visited New Orleans')]
```

The Astra DB vectorstore `body_search` argument can be used to filter the search on the term `new`.

```python
vectorstore.as_retriever(search_kwargs={"body_search": "new"}).invoke(
    "What city did I visit last?"
)
```

```output
[Document(page_content='In 2022, I visited New York'),
Document(page_content='In 2021, I visited New Orleans')]
```

We can now create the chain that we will use to do question-answering over

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    ConfigurableField,
    RunnablePassthrough,
)
from langchain_openai import ChatOpenAI
```

#### API Reference:

- [StrOutputParser](https://api.python.langchain.com/en/latest/output_parsers/langchain_core.output_parsers.string.StrOutputParser.html)
- [ChatPromptTemplate](https://api.python.langchain.com/en/latest/prompts/langchain_core.prompts.chat.ChatPromptTemplate.html)
- [ConfigurableField](https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.utils.ConfigurableField.html)
- [RunnablePassthrough](https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.passthrough.RunnablePassthrough.html)
- [ChatOpenAI](https://api.python.langchain.com/en/latest/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html)

This is basic question-answering chain set up.

```python
template = """Answer the question based only on the following context:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI()

retriever = vectorstore.as_retriever()
```

Here we mark the retriever as having a configurable field. All vectorstore retrievers have `search_kwargs` as a field. This is just a dictionary, with vectorstore specific fields

```python
configurable_retriever = retriever.configurable_fields(
    search_kwargs=ConfigurableField(
        id="search_kwargs",
        name="Search Kwargs",
        description="The search kwargs to use",
    )
)
```

We can now create the chain using our configurable retriever

```python
chain = (
    {"context": configurable_retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)
```

```python
chain.invoke("What city did I visit last?")
```

```output
Paris
```

We can now invoke the chain with configurable options. `search_kwargs` is the id of the configurable field. The value is the search kwargs to use for Astra DB.

```python
chain.invoke(
    "What city did I visit last?",
    config={"configurable": {"search_kwargs": {"body_search": "new"}}},
)
```

```output
New York
```

# Per-User Retrieval

When building a retrieval app, you often have to build it with multiple users in mind. This means that you may be storing data not just for one user, but for many different users, and they should not be able to see eachother's data. This means that you need to be able to configure your retrieval chain to only retrieve certain information. This generally involves two steps.

**Step 1: Make sure the retriever you are using supports multiple users**

At the moment, there is no unified flag or filter for this in LangChain. Rather, each vectorstore and retriever may have their own, and may be called different things (namespaces, multi-tenancy, etc). For vectorstores, this is generally exposed as a keyword argument that is passed in during `similarity_search`. By reading the documentation or source code, figure out whether the retriever you are using supports multiple users, and, if so, how to use it.

Note: adding documentation and/or support for multiple users for retrievers that do not support it (or document it) is a GREAT way to contribute to LangChain

**Step 2: Add that parameter as a configurable field for the chain**

This will let you easily call the chain and configure any relevant flags at runtime. See [this documentation](https://python.langchain.com/v0.1/docs/expression_language/primitives/configure/) for more information on configuration.

**Step 3: Call the chain with that configurable field**

Now, at runtime you can call this chain with configurable field.

## Code Example[](https://python.langchain.com/v0.1/docs/use_cases/question_answering/per_user/#code-example)

Let's see a concrete example of what this looks like in code. We will use Pinecone for this example.

To configure Pinecone, set the following environment variable:

- `PINECONE_API_KEY`: Your Pinecone API key

```python
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
```

#### API Reference:

- [OpenAIEmbeddings](https://api.python.langchain.com/en/latest/embeddings/langchain_openai.embeddings.base.OpenAIEmbeddings.html)
- [PineconeVectorStore](https://api.python.langchain.com/en/latest/vectorstores/langchain_pinecone.vectorstores.PineconeVectorStore.html)

```python
embeddings = OpenAIEmbeddings()
vectorstore = PineconeVectorStore(index_name="test-example", embedding=embeddings)

vectorstore.add_texts(["i worked at kensho"], namespace="harrison")
vectorstore.add_texts(["i worked at facebook"], namespace="ankush")
```

```output
['ce15571e-4e2f-44c9-98df-7e83f6f63095']
```

The pinecone kwarg for `namespace` can be used to separate documents

```python
# This will only get documents for Ankush
vectorstore.as_retriever(search_kwargs={"namespace": "ankush"}).invoke(
    "where did i work?"
)
```

```output
[Document(page_content='i worked at facebook')]
```

```python
# This will only get documents for Harrison
vectorstore.as_retriever(search_kwargs={"namespace": "harrison"}).invoke(
    "where did i work?"
)
```

```output
[Document(page_content='i worked at kensho')]
```

We can now create the chain that we will use to do question-answering over

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    ConfigurableField,
    RunnableBinding,
    RunnableLambda,
    RunnablePassthrough,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
```

#### API Reference:

- [StrOutputParser](https://api.python.langchain.com/en/latest/output_parsers/langchain_core.output_parsers.string.StrOutputParser.html)
- [ChatPromptTemplate](https://api.python.langchain.com/en/latest/prompts/langchain_core.prompts.chat.ChatPromptTemplate.html)
- [ConfigurableField](https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.utils.ConfigurableField.html)
- [RunnableBinding](https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.base.RunnableBinding.html)
- [RunnableLambda](https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.base.RunnableLambda.html)
- [RunnablePassthrough](https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.passthrough.RunnablePassthrough.html)
- [ChatOpenAI](https://api.python.langchain.com/en/latest/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html)
- [OpenAIEmbeddings](https://api.python.langchain.com/en/latest/embeddings/langchain_openai.embeddings.base.OpenAIEmbeddings.html)

This is basic question-answering chain set up.

```python
template = """Answer the question based only on the following context:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI()

retriever = vectorstore.as_retriever()
```

Here we mark the retriever as having a configurable field. All vectorstore retrievers have `search_kwargs` as a field. This is just a dictionary, with vectorstore specific fields

```python
configurable_retriever = retriever.configurable_fields(
    search_kwargs=ConfigurableField(
        id="search_kwargs",
        name="Search Kwargs",
        description="The search kwargs to use",
    )
)
```

We can now create the chain using our configurable retriever

```python
chain = (
    {"context": configurable_retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)
```

We can now invoke the chain with configurable options. `search_kwargs` is the id of the configurable field. The value is the search kwargs to use for Pinecone

```python
chain.invoke(
    "where did the user work?",
    config={"configurable": {"search_kwargs": {"namespace": "harrison"}}},
)
```

```output
'The user worked at Kensho.'
```

```python
chain.invoke(
    "where did the user work?",
    config={"configurable": {"search_kwargs": {"namespace": "ankush"}}},
)
```

```output
'The user worked at Facebook.'
```

For more vectorstore implementations for multi-user, please refer to specific pages, such as [Milvus](https://python.langchain.com/v0.1/docs/integrations/vectorstores/milvus/).

# Using agents

This is an agent specifically optimized for doing retrieval when necessary and also holding a conversation.

To start, we will set up the retriever we want to use, and then turn it into a retriever tool. Next, we will use the high level constructor for this type of agent. Finally, we will walk through how to construct a conversational retrieval agent from components.

```python
%pip install --upgrade --quiet  langchain langchain-community langchainhub langchain-openai faiss-cpu
```

## The Retriever[](https://python.langchain.com/v0.1/docs/use_cases/question_answering/conversational_retrieval_agents/#the-retriever)

To start, we need a retriever to use! The code here is mostly just example code. Feel free to use your own retriever and skip to the section on creating a retriever tool.

```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("../../modules/state_of_the_union.txt")
documents = loader.load()
```

#### API Reference:

- [TextLoader](https://api.python.langchain.com/en/latest/document_loaders/langchain_community.document_loaders.text.TextLoader.html)

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(texts, embeddings)
```

#### API Reference:

- [FAISS](https://api.python.langchain.com/en/latest/vectorstores/langchain_community.vectorstores.faiss.FAISS.html)
- [OpenAIEmbeddings](https://api.python.langchain.com/en/latest/embeddings/langchain_openai.embeddings.base.OpenAIEmbeddings.html)
- [CharacterTextSplitter](https://api.python.langchain.com/en/latest/character/langchain_text_splitters.character.CharacterTextSplitter.html)

```python
retriever = db.as_retriever()
```

## Retriever Tool[](https://python.langchain.com/v0.1/docs/use_cases/question_answering/conversational_retrieval_agents/#retriever-tool)

Now we need to create a tool for our retriever. The main things we need to pass in are a name for the retriever as well as a description. These will both be used by the language model, so they should be informative.

```python
from langchain.tools.retriever import create_retriever_tool

tool = create_retriever_tool(
    retriever,
    "search_state_of_union",
    "Searches and returns excerpts from the 2022 State of the Union.",
)
tools = [tool]
```

#### API Reference:

- [create_retriever_tool](https://api.python.langchain.com/en/latest/tools/langchain_core.tools.create_retriever_tool.html)

## Agent Constructor[](https://python.langchain.com/v0.1/docs/use_cases/question_answering/conversational_retrieval_agents/#agent-constructor)

Here, we will use the high level `create_openai_tools_agent` API to construct the agent.

Notice that beside the list of tools, the only thing we need to pass in is a language model to use. Under the hood, this agent is using the OpenAI tool-calling capabilities, so we need to use a ChatOpenAI model.

```python
from langchain import hub

prompt = hub.pull("hwchase17/openai-tools-agent")
prompt.messages
```

```output
[SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template='You are a helpful assistant')),
 MessagesPlaceholder(variable_name='chat_history', optional=True),
 HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], template='{input}')),
 MessagesPlaceholder(variable_name='agent_scratchpad')]
```

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0)
```

#### API Reference:

- [ChatOpenAI](https://api.python.langchain.com/en/latest/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html)

```python
from langchain.agents import AgentExecutor, create_openai_tools_agent

agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)
```

#### API Reference:

- [AgentExecutor](https://api.python.langchain.com/en/latest/agents/langchain.agents.agent.AgentExecutor.html)
- [create_openai_tools_agent](https://api.python.langchain.com/en/latest/agents/langchain.agents.openai_tools.base.create_openai_tools_agent.html)

We can now try it out!

```python
result = agent_executor.invoke({"input": "hi, im bob"})
```

```python
result["output"]
```

```output
'Hello Bob! How can I assist you today?'
```

Notice that it now does retrieval

```python
result = agent_executor.invoke(
    {
        "input": "what did the president say about ketanji brown jackson in the most recent state of the union?"
    }
)
```

```python
result["output"]
```

```output
"In the most recent state of the union, the President mentioned Kentaji Brown Jackson. The President nominated Circuit Court of Appeals Judge Ketanji Brown Jackson to serve on the United States Supreme Court. The President described Judge Ketanji Brown Jackson as one of our nation's top legal minds who will continue Justice Breyer's legacy of excellence."
```

Notice that the follow up question asks about information previously retrieved, so no need to do another retrieval

```python
result = agent_executor.invoke(
    {"input": "how long ago did the president nominate ketanji brown jackson?"}
)
```

```output
[1m> Entering new AgentExecutor chain...[0m
[32;1m[1;3mThe President nominated Judge Ketanji Brown Jackson four days ago.[0m

[1m> Finished chain.[0m
```

```python
result["output"]
```

```output
'The President nominated Judge Ketanji Brown Jackson four days ago.'
```

For more on how to use agents with retrievers and other tools, head to the [Agents](https://python.langchain.com/v0.1/docs/modules/agents/) section.

# Using local models

The popularity of projects like [PrivateGPT](https://github.com/imartinez/privateGPT), [llama.cpp](https://github.com/ggerganov/llama.cpp), [GPT4All](https://github.com/nomic-ai/gpt4all), and [llamafile](https://github.com/Mozilla-Ocho/llamafile) underscore the importance of running LLMs locally.

LangChain has [integrations](https://integrations.langchain.com/) with many open-source LLMs that can be run locally.

See [here](https://python.langchain.com/v0.1/docs/guides/development/local_llms/) for setup instructions for these LLMs.

For example, here we show how to run `GPT4All` or `LLaMA2` locally (e.g., on your laptop) using local embeddings and a local LLM.

## Document Loading[](https://python.langchain.com/v0.1/docs/use_cases/question_answering/local_retrieval_qa/#document-loading)

First, install packages needed for local embeddings and vector storage.

```python
%pip install --upgrade --quiet  langchain langchain-community langchainhub gpt4all langchain-chroma 
```

Load and split an example document.

We'll use a blog post on agents as an example.

```python
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)
```

#### API Reference:

- [WebBaseLoader](https://api.python.langchain.com/en/latest/document_loaders/langchain_community.document_loaders.web_base.WebBaseLoader.html)
- [RecursiveCharacterTextSplitter](https://api.python.langchain.com/en/latest/character/langchain_text_splitters.character.RecursiveCharacterTextSplitter.html)

Next, the below steps will download the `GPT4All` embeddings locally (if you don't already have them).

```python
from langchain_chroma import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings

vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings())
```

#### API Reference:

- [GPT4AllEmbeddings](https://api.python.langchain.com/en/latest/embeddings/langchain_community.embeddings.gpt4all.GPT4AllEmbeddings.html)

Test similarity search is working with our local embeddings.

```python
question = "What are the approaches to Task Decomposition?"
docs = vectorstore.similarity_search(question)
len(docs)
```

```output
4
```

```python
docs[0]
```

```output
Document(page_content='Task decomposition can be done (1) by LLM with simple prompting like "Steps for XYZ.\\n1.", "What are the subgoals for achieving XYZ?", (2) by using task-specific instructions; e.g. "Write a story outline." for writing a novel, or (3) with human inputs.', metadata={'description': 'Building agents with LLM (large language model) as its core controller is a cool concept. Several proof-of-concepts demos, such as AutoGPT, GPT-Engineer and BabyAGI, serve as inspiring examples. The potentiality of LLM extends beyond generating well-written copies, stories, essays and programs; it can be framed as a powerful general problem solver.\nAgent System Overview In a LLM-powered autonomous agent system, LLM functions as the agent’s brain, complemented by several key components:', 'language': 'en', 'source': 'https://lilianweng.github.io/posts/2023-06-23-agent/', 'title': "LLM Powered Autonomous Agents | Lil'Log"})
```

## Model[](https://python.langchain.com/v0.1/docs/use_cases/question_answering/local_retrieval_qa/#model)

### LLaMA2[](https://python.langchain.com/v0.1/docs/use_cases/question_answering/local_retrieval_qa/#llama2)

Note: new versions of `llama-cpp-python` use GGUF model files (see [here](https://github.com/abetlen/llama-cpp-python/pull/633)).

If you have an existing GGML model, see [here](https://python.langchain.com/v0.1/docs/integrations/llms/llamacpp/) for instructions for conversion for GGUF.

And / or, you can download a GGUF converted model (e.g., [here](https://huggingface.co/TheBloke)).

Finally, as noted in detail [here](https://python.langchain.com/v0.1/docs/guides/development/local_llms/) install `llama-cpp-python`

```python
%pip install --upgrade --quiet  llama-cpp-python
```

To enable use of GPU on Apple Silicon, follow the steps [here](https://github.com/abetlen/llama-cpp-python/blob/main/docs/install/macos.md) to use the Python binding `with Metal support`.

In particular, ensure that `conda` is using the correct virtual environment that you created (`miniforge3`).

E.g., for me:

```text
conda activate /Users/rlm/miniforge3/envs/llama
```

With this confirmed:

```python
! CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 /Users/rlm/miniforge3/envs/llama/bin/pip install -U llama-cpp-python --no-cache-dir
```

```python
from langchain_community.llms import LlamaCpp
```

#### API Reference:

- [LlamaCpp](https://api.python.langchain.com/en/latest/llms/langchain_community.llms.llamacpp.LlamaCpp.html)

Setting model parameters as noted in the [llama.cpp docs](https://python.langchain.com/v0.1/docs/integrations/llms/llamacpp/).

```python
n_gpu_layers = 1  # Metal set to 1 is enough.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="/Users/rlm/Desktop/Code/llama.cpp/models/llama-2-13b-chat.ggufv3.q4_0.bin",
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx=2048,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    verbose=True,
)
```

Note that these indicate that [Metal was enabled properly](https://python.langchain.com/v0.1/docs/integrations/llms/llamacpp/):

```text
ggml_metal_init: allocating
ggml_metal_init: using MPS
```

```python
llm.invoke("Simulate a rap battle between Stephen Colbert and John Oliver")
```

~~~output
Llama.generate: prefix-match hit
``````output
by jonathan 

Here's the hypothetical rap battle:

[Stephen Colbert]: Yo, this is Stephen Colbert, known for my comedy show. I'm here to put some sense in your mind, like an enema do-go. Your opponent? A man of laughter and witty quips, John Oliver! Now let's see who gets the most laughs while taking shots at each other

[John Oliver]: Yo, this is John Oliver, known for my own comedy show. I'm here to take your mind on an adventure through wit and humor. But first, allow me to you to our contestant: Stephen Colbert! His show has been around since the '90s, but it's time to see who can out-rap whom

[Stephen Colbert]: You claim to be a witty man, John Oliver, with your British charm and clever remarks. But my knows that I'm America's funnyman! Who's the one taking you? Nobody!

[John Oliver]: Hey Stephen Colbert, don't get too cocky. You may
``````output

llama_print_timings:        load time =  4481.74 ms
llama_print_timings:      sample time =   183.05 ms /   256 runs   (    0.72 ms per token,  1398.53 tokens per second)
llama_print_timings: prompt eval time =   456.05 ms /    13 tokens (   35.08 ms per token,    28.51 tokens per second)
llama_print_timings:        eval time =  7375.20 ms /   255 runs   (   28.92 ms per token,    34.58 tokens per second)
llama_print_timings:       total time =  8388.92 ms
~~~

```output
"by jonathan \n\nHere's the hypothetical rap battle:\n\n[Stephen Colbert]: Yo, this is Stephen Colbert, known for my comedy show. I'm here to put some sense in your mind, like an enema do-go. Your opponent? A man of laughter and witty quips, John Oliver! Now let's see who gets the most laughs while taking shots at each other\n\n[John Oliver]: Yo, this is John Oliver, known for my own comedy show. I'm here to take your mind on an adventure through wit and humor. But first, allow me to you to our contestant: Stephen Colbert! His show has been around since the '90s, but it's time to see who can out-rap whom\n\n[Stephen Colbert]: You claim to be a witty man, John Oliver, with your British charm and clever remarks. But my knows that I'm America's funnyman! Who's the one taking you? Nobody!\n\n[John Oliver]: Hey Stephen Colbert, don't get too cocky. You may"
```

### GPT4All[](https://python.langchain.com/v0.1/docs/use_cases/question_answering/local_retrieval_qa/#gpt4all)

Similarly, we can use `GPT4All`.

[Download the GPT4All model binary](https://python.langchain.com/v0.1/docs/integrations/llms/gpt4all/).

The Model Explorer on the [GPT4All](https://gpt4all.io/index.html) is a great way to choose and download a model.

Then, specify the path that you downloaded to to.

E.g., for me, the model lives here:

```
/Users/rlm/Desktop/Code/gpt4all/models/nous-hermes-13b.ggmlv3.q4_0.bin
from langchain_community.llms import GPT4All

gpt4all = GPT4All(
    model="/Users/rlm/Desktop/Code/gpt4all/models/nous-hermes-13b.ggmlv3.q4_0.bin",
    max_tokens=2048,
)
```

#### API Reference

- [GPT4All](https://api.python.langchain.com/en/latest/llms/langchain_community.llms.gpt4all.GPT4All.html)

### llamafile[](https://python.langchain.com/v0.1/docs/use_cases/question_answering/local_retrieval_qa/#llamafile)

One of the simplest ways to run an LLM locally is using a [llamafile](https://github.com/Mozilla-Ocho/llamafile). All you need to do is:

1) Download a llamafile from [HuggingFace](https://huggingface.co/models?other=llamafile) 2) Make the file executable 3) Run the file

llamafiles bundle model weights and a [specially-compiled](https://github.com/Mozilla-Ocho/llamafile?tab=readme-ov-file#technical-details) version of [`llama.cpp`](https://github.com/ggerganov/llama.cpp) into a single file that can run on most computers without any additional dependencies. They also come with an embedded inference server that provides an [API](https://github.com/Mozilla-Ocho/llamafile/blob/main/llama.cpp/server/README.md#api-endpoints) for interacting with your model.

Here's a simple bash script that shows all 3 setup steps:

```bash
# Download a llamafile from HuggingFace
wget https://huggingface.co/jartine/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/TinyLlama-1.1B-Chat-v1.0.Q5_K_M.llamafile

# Make the file executable. On Windows, instead just rename the file to end in ".exe".
chmod +x TinyLlama-1.1B-Chat-v1.0.Q5_K_M.llamafile

# Start the model server. Listens at http://localhost:8080 by default.
./TinyLlama-1.1B-Chat-v1.0.Q5_K_M.llamafile --server --nobrowser
```

After you run the above setup steps, you can interact with the model via LangChain:

```python
from langchain_community.llms.llamafile import Llamafile

llamafile = Llamafile()

llamafile.invoke("Here is my grandmother's beloved recipe for spaghetti and meatballs:")
```

#### API Reference:

- [Llamafile](https://api.python.langchain.com/en/latest/llms/langchain_community.llms.llamafile.Llamafile.html)

```output
'\n-1 1/2 (8 oz. Pounds) ground beef, browned and cooked until no longer pink\n-3 cups whole wheat spaghetti\n-4 (10 oz) cans diced tomatoes with garlic and basil\n-2 eggs, beaten\n-1 cup grated parmesan cheese\n-1/2 teaspoon salt\n-1/4 teaspoon black pepper\n-1 cup breadcrumbs (16 oz)\n-2 tablespoons olive oil\n\nInstructions:\n1. Cook spaghetti according to package directions. Drain and set aside.\n2. In a large skillet, brown ground beef over medium heat until no longer pink. Drain any excess grease.\n3. Stir in diced tomatoes with garlic and basil, and season with salt and pepper. Cook for 5 to 7 minutes or until sauce is heated through. Set aside.\n4. In a large bowl, beat eggs with a fork or whisk until fluffy. Add cheese, salt, and black pepper. Set aside.\n5. In another bowl, combine breadcrumbs and olive oil. Dip each spaghetti into the egg mixture and then coat in the breadcrumb mixture. Place on baking sheet lined with parchment paper to prevent sticking. Repeat until all spaghetti are coated.\n6. Heat oven to 375 degrees. Bake for 18 to 20 minutes, or until lightly golden brown.\n7. Serve hot with meatballs and sauce on the side. Enjoy!'
```

## Using in a chain[](https://python.langchain.com/v0.1/docs/use_cases/question_answering/local_retrieval_qa/#using-in-a-chain)

We can create a summarization chain with either model by passing in the retrieved docs and a simple prompt.

It formats the prompt template using the input key values provided and passes the formatted string to `GPT4All`, `LLama-V2`, or another specified LLM.

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

# Prompt
prompt = PromptTemplate.from_template(
    "Summarize the main themes in these retrieved docs: {docs}"
)


# Chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


chain = {"docs": format_docs} | prompt | llm | StrOutputParser()

# Run
question = "What are the approaches to Task Decomposition?"
docs = vectorstore.similarity_search(question)
chain.invoke(docs)
```

#### API Reference:

- [StrOutputParser](https://api.python.langchain.com/en/latest/output_parsers/langchain_core.output_parsers.string.StrOutputParser.html)
- [PromptTemplate](https://api.python.langchain.com/en/latest/prompts/langchain_core.prompts.prompt.PromptTemplate.html)

~~~output
Llama.generate: prefix-match hit
``````output

Based on the retrieved documents, the main themes are:
1. Task decomposition: The ability to break down complex tasks into smaller subtasks, which can be handled by an LLM or other components of the agent system.
2. LLM as the core controller: The use of a large language model (LLM) as the primary controller of an autonomous agent system, complemented by other key components such as a knowledge graph and a planner.
3. Potentiality of LLM: The idea that LLMs have the potential to be used as powerful general problem solvers, not just for generating well-written copies but also for solving complex tasks and achieving human-like intelligence.
4. Challenges in long-term planning: The challenges in planning over a lengthy history and effectively exploring the solution space, which are important limitations of current LLM-based autonomous agent systems.
``````output

llama_print_timings:        load time =  1191.88 ms
llama_print_timings:      sample time =   134.47 ms /   193 runs   (    0.70 ms per token,  1435.25 tokens per second)
llama_print_timings: prompt eval time = 39470.18 ms /  1055 tokens (   37.41 ms per token,    26.73 tokens per second)
llama_print_timings:        eval time =  8090.85 ms /   192 runs   (   42.14 ms per token,    23.73 tokens per second)
llama_print_timings:       total time = 47943.12 ms
~~~



```output
'\nBased on the retrieved documents, the main themes are:\n1. Task decomposition: The ability to break down complex tasks into smaller subtasks, which can be handled by an LLM or other components of the agent system.\n2. LLM as the core controller: The use of a large language model (LLM) as the primary controller of an autonomous agent system, complemented by other key components such as a knowledge graph and a planner.\n3. Potentiality of LLM: The idea that LLMs have the potential to be used as powerful general problem solvers, not just for generating well-written copies but also for solving complex tasks and achieving human-like intelligence.\n4. Challenges in long-term planning: The challenges in planning over a lengthy history and effectively exploring the solution space, which are important limitations of current LLM-based autonomous agent systems.'
```

## Q&A[](https://python.langchain.com/v0.1/docs/use_cases/question_answering/local_retrieval_qa/#qa)

We can also use the LangChain Prompt Hub to store and fetch prompts that are model-specific.

Let's try with a default RAG prompt, [here](https://smith.langchain.com/hub/rlm/rag-prompt).

```python
from langchain import hub

rag_prompt = hub.pull("rlm/rag-prompt")
rag_prompt.messages
```

```output
[HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context', 'question'], template="You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:"))]
```

```python
from langchain_core.runnables import RunnablePassthrough, RunnablePick

# Chain
chain = (
    RunnablePassthrough.assign(context=RunnablePick("context") | format_docs)
    | rag_prompt
    | llm
    | StrOutputParser()
)

# Run
chain.invoke({"context": docs, "question": question})
```

#### API Reference:

- [RunnablePassthrough](https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.passthrough.RunnablePassthrough.html)
- [RunnablePick](https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.passthrough.RunnablePick.html)

~~~output
Llama.generate: prefix-match hit
``````output

Task can be done by down a task into smaller subtasks, using simple prompting like "Steps for XYZ." or task-specific like "Write a story outline" for writing a novel.
``````output

llama_print_timings:        load time = 11326.20 ms
llama_print_timings:      sample time =    33.03 ms /    47 runs   (    0.70 ms per token,  1422.86 tokens per second)
llama_print_timings: prompt eval time =  1387.31 ms /   242 tokens (    5.73 ms per token,   174.44 tokens per second)
llama_print_timings:        eval time =  1321.62 ms /    46 runs   (   28.73 ms per token,    34.81 tokens per second)
llama_print_timings:       total time =  2801.08 ms
~~~

```output
{'output_text': '\nTask can be done by down a task into smaller subtasks, using simple prompting like "Steps for XYZ." or task-specific like "Write a story outline" for writing a novel.'}
```

Now, let's try with [a prompt specifically for LLaMA](https://smith.langchain.com/hub/rlm/rag-prompt-llama), which [includes special tokens](https://huggingface.co/blog/llama2#how-to-prompt-llama-2).

```python
# Prompt
rag_prompt_llama = hub.pull("rlm/rag-prompt-llama")
rag_prompt_llama.messages
```

```output
ChatPromptTemplate(input_variables=['question', 'context'], output_parser=None, partial_variables={}, messages=[HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['question', 'context'], output_parser=None, partial_variables={}, template="[INST]<<SYS>> You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.<</SYS>> \nQuestion: {question} \nContext: {context} \nAnswer: [/INST]", template_format='f-string', validate_template=True), additional_kwargs={})])
```

```python
# Chain
chain = (
    RunnablePassthrough.assign(context=RunnablePick("context") | format_docs)
    | rag_prompt_llama
    | llm
    | StrOutputParser()
)

# Run
chain.invoke({"context": docs, "question": question})
```

~~~output
Llama.generate: prefix-match hit
``````output
  Sure, I'd be happy to help! Based on the context, here are some to task:

1. LLM with simple prompting: This using a large model (LLM) with simple prompts like "Steps for XYZ" or "What are the subgoals for achieving XYZ?" to decompose tasks into smaller steps.
2. Task-specific: Another is to use task-specific, such as "Write a story outline" for writing a novel, to guide the of tasks.
3. Human inputs:, human inputs can be used to supplement the process, in cases where the task a high degree of creativity or expertise.

As fores in long-term and task, one major is that LLMs to adjust plans when faced with errors, making them less robust to humans who learn from trial and error.
``````output

llama_print_timings:        load time = 11326.20 ms
llama_print_timings:      sample time =   144.81 ms /   207 runs   (    0.70 ms per token,  1429.47 tokens per second)
llama_print_timings: prompt eval time =  1506.13 ms /   258 tokens (    5.84 ms per token,   171.30 tokens per second)
llama_print_timings:        eval time =  6231.92 ms /   206 runs   (   30.25 ms per token,    33.06 tokens per second)
llama_print_timings:       total time =  8158.41 ms
~~~

```output
{'output_text': '  Sure, I\'d be happy to help! Based on the context, here are some to task:\n\n1. LLM with simple prompting: This using a large model (LLM) with simple prompts like "Steps for XYZ" or "What are the subgoals for achieving XYZ?" to decompose tasks into smaller steps.\n2. Task-specific: Another is to use task-specific, such as "Write a story outline" for writing a novel, to guide the of tasks.\n3. Human inputs:, human inputs can be used to supplement the process, in cases where the task a high degree of creativity or expertise.\n\nAs fores in long-term and task, one major is that LLMs to adjust plans when faced with errors, making them less robust to humans who learn from trial and error.'}
```

## Q&A with retrieval[](https://python.langchain.com/v0.1/docs/use_cases/question_answering/local_retrieval_qa/#qa-with-retrieval)

Instead of manually passing in docs, we can automatically retrieve them from our vector store based on the user question.

This will use a QA default prompt (shown [here](https://github.com/langchain-ai/langchain/blob/275b926cf745b5668d3ea30236635e20e7866442/langchain/chains/retrieval_qa/prompt.py#L4)) and will retrieve from the vectorDB.

```python
retriever = vectorstore.as_retriever()
qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)
```

```python
qa_chain.invoke(question)
```

~~~output
Llama.generate: prefix-match hit
``````output
  Sure! Based on the context, here's my answer to your:

There are several to task,:

1. LLM-based with simple prompting, such as "Steps for XYZ" or "What are the subgoals for achieving XYZ?"
2. Task-specific, like "Write a story outline" for writing a novel.
3. Human inputs to guide the process.

These can be used to decompose complex tasks into smaller, more manageable subtasks, which can help improve the and effectiveness of task. However, long-term and task can being due to the need to plan over a lengthy history and explore the space., LLMs may to adjust plans when faced with errors, making them less robust to human learners who can learn from trial and error.
``````output

llama_print_timings:        load time = 11326.20 ms
llama_print_timings:      sample time =   139.20 ms /   200 runs   (    0.70 ms per token,  1436.76 tokens per second)
llama_print_timings: prompt eval time =  1532.26 ms /   258 tokens (    5.94 ms per token,   168.38 tokens per second)
llama_print_timings:        eval time =  5977.62 ms /   199 runs   (   30.04 ms per token,    33.29 tokens per second)
llama_print_timings:       total time =  7916.21 ms
~~~

```output
{'query': 'What are the approaches to Task Decomposition?',
 'result': '  Sure! Based on the context, here\'s my answer to your:\n\nThere are several to task,:\n\n1. LLM-based with simple prompting, such as "Steps for XYZ" or "What are the subgoals for achieving XYZ?"\n2. Task-specific, like "Write a story outline" for writing a novel.\n3. Human inputs to guide the process.\n\nThese can be used to decompose complex tasks into smaller, more manageable subtasks, which can help improve the and effectiveness of task. However, long-term and task can being due to the need to plan over a lengthy history and explore the space., LLMs may to adjust plans when faced with errors, making them less robust to human learners who can learn from trial and error.'}
```