# Quickstart

LangChain有许多组件，旨在帮助构建问答应用程序，以及更广泛的RAG应用程序。为了熟悉这些，我们将在文本数据源上构建一个简单的Q&A应用程序。沿着的方式，我们将讨论一个典型的Q&A架构，讨论相关的LangChain组件，并强调更先进的Q&A技术的其他资源。我们还将看到LangSmith如何帮助我们跟踪和理解我们的应用程序。随着我们的应用程序变得越来越复杂，LangSmith将变得越来越有用。

## Architecture[](https://python.langchain.com/v0.1/docs/use_cases/question_answering/quickstart/#architecture)

我们将创建一个典型的RAG应用程序，如Q&A介绍中所述，它有两个主要组件：

**Indexing**: 一个管道，用于从数据源获取数据并对其进行索引。这通常发生在离线状态。

**Retrieval and generation**: 实际的RAG链在运行时接受用户查询并从索引中检索相关数据，然后将其传递给模型。

从原始数据到答案的完整序列如下所示：

### Indexing[](https://python.langchain.com/v0.1/docs/use_cases/question_answering/quickstart/#indexing)

1. **Load**: 首先，我们需要加载数据。我们将为此使用DocumentLoaders。
2. **Split**:文本拆分器将大文档拆分为更小的块。这对于索引数据和将数据传递到模型都很有用，因为大块的数据更难搜索，并且不适合模型的有限上下文窗口。
3. **Store**: 我们需要一个地方来存储和索引我们的拆分，以便以后可以搜索它们。这通常使用VectorStore和Embeddings模型来完成。

### Retrieval and generation[](https://python.langchain.com/v0.1/docs/use_cases/question_answering/quickstart/#retrieval-and-generation)

1. **Retrieve**: 给定用户输入，使用检索器从存储中检索相关拆分。
2. **Generate**: ChatModel / LLM使用包含问题和检索到的数据的提示生成答案。

## Setup[](https://python.langchain.com/v0.1/docs/use_cases/question_answering/quickstart/#setup)

### Dependencies[](https://python.langchain.com/v0.1/docs/use_cases/question_answering/quickstart/#dependencies)

我们将在本演练中使用OpenAI聊天模型和嵌入以及Chroma向量存储，但这里显示的所有内容都适用于任何ChatModel或LLM，Embeddings和VectorStore或Retriever。

我们将使用以下包：

```python
%pip install --upgrade --quiet  langchain langchain-community langchainhub langchain-openai langchain-chroma bs4
```

我们需要为嵌入模型设置环境变量OPENAI_API_KEY，这可以直接完成，也可以从.env文件加载，如下所示：

```python
import getpass
import os

os.environ["OPENAI_API_KEY"] = getpass.getpass()

# import dotenv

# dotenv.load_dotenv()
```

### LangSmith[](https://python.langchain.com/v0.1/docs/use_cases/question_answering/quickstart/#langsmith)

你使用LangChain构建的许多应用程序都包含多个步骤，其中包含多个LLM调用。随着这些应用程序变得越来越复杂，能够检查链或代理内部到底发生了什么变得至关重要。最好的办法是和兰史密斯一起。

请注意，LangSmith不是必需的，但它是有帮助的。如果您确实想使用LangSmith，请在上面的链接处注册后，确保设置环境变量以开始记录跟踪：

```python
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()
```



## Preview[](https://python.langchain.com/v0.1/docs/use_cases/question_answering/quickstart/#preview)

在本指南中，我们将在Lilian Weng的LLM Powered Autonomous Agents博客文章上构建一个QA应用程序，该应用程序允许我们对帖子的内容提出问题。

我们可以创建一个简单的索引管道和RAG链来完成这一点，大约20行代码：

```python
import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
```



#### API Reference:

- [WebBaseLoader](https://api.python.langchain.com/en/latest/document_loaders/langchain_community.document_loaders.web_base.WebBaseLoader.html)
- [StrOutputParser](https://api.python.langchain.com/en/latest/output_parsers/langchain_core.output_parsers.string.StrOutputParser.html)
- [RunnablePassthrough](https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.passthrough.RunnablePassthrough.html)
- [OpenAIEmbeddings](https://api.python.langchain.com/en/latest/embeddings/langchain_openai.embeddings.base.OpenAIEmbeddings.html)
- [RecursiveCharacterTextSplitter](https://api.python.langchain.com/en/latest/character/langchain_text_splitters.character.RecursiveCharacterTextSplitter.html)

## OpenAI

##### Install dependencies

```bash
pip install -qU langchain-openai
```

##### Set environment variables

```python
import getpass
import os

os.environ["OPENAI_API_KEY"] = getpass.getpass()
```

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
```

```python
# Load, chunk and index the contents of the blog.
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

```python
rag_chain.invoke("What is Task Decomposition?")
```

```text
'Task decomposition is a technique used to break down complex tasks into smaller and simpler steps. It can be done through prompting techniques like Chain of Thought or Tree of Thoughts, or by using task-specific instructions or human inputs. Task decomposition helps agents plan ahead and manage complicated tasks more effectively.'
```

```python
# cleanup
vectorstore.delete_collection()
```

Check out the [LangSmith trace](https://smith.langchain.com/public/1c6ca97e-445b-4d00-84b4-c7befcbc59fe/r)

## Detailed walkthrough[](https://python.langchain.com/v0.1/docs/use_cases/question_answering/quickstart/#detailed-walkthrough)

让我们一步一步地看一下上面的代码，以真正理解发生了什么。

## 1. Indexing: Load[](https://python.langchain.com/v0.1/docs/use_cases/question_answering/quickstart/#indexing-load)

我们需要首先加载博客文章的内容。我们可以使用DocumentLoaders来实现这一点，DocumentLoaders是从源中加载数据并返回Documents列表的对象。文档是一个带有page_content（str）和元数据（dict）的对象。

在本例中，我们将使用WebBaseLoader，它使用urllib从Web URL加载HTML，并使用BeautifulSoup将其解析为文本。我们可以通过bs_kwargs向BeautifulSoup解析器传递参数来定制HTML ->文本解析（参见BeautifulSoup文档）。在这种情况下，只有类为“post-content”、“post-title”或“post-header”的HTML标签是相关的，所以我们将删除所有其他标签。

```python
import bs4
from langchain_community.document_loaders import WebBaseLoader

# Only keep post title, headers, and content from the full HTML.
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()
```



#### API Reference:

- [WebBaseLoader](https://api.python.langchain.com/en/latest/document_loaders/langchain_community.document_loaders.web_base.WebBaseLoader.html)

```python
len(docs[0].page_content)
```

```text
42824
```

```python
print(docs[0].page_content[:500])
```

```text
      LLM Powered Autonomous Agents

Date: June 23, 2023  |  Estimated Reading Time: 31 min  |  Author: Lilian Weng


Building agents with LLM (large language model) as its core controller is a cool concept. Several proof-of-concepts demos, such as AutoGPT, GPT-Engineer and BabyAGI, serve as inspiring examples. The potentiality of LLM extends beyond generating well-written copies, stories, essays and programs; it can be framed as a powerful general problem solver.
Agent System Overview#
In
```



### Go deeper[](https://python.langchain.com/v0.1/docs/use_cases/question_answering/quickstart/#go-deeper)

DocumentLoader：从源文件中加载数据的对象，作为文档列表。

- [Docs](https://python.langchain.com/v0.1/docs/modules/data_connection/document_loaders/): 关于如何使用DocumentLoaders的详细文档。
- [Integrations](https://python.langchain.com/v0.1/docs/integrations/document_loaders/): 160+集成选择。
- [Interface](https://api.python.langchain.com/en/latest/document_loaders/langchain_core.document_loaders.base.BaseLoader.html): 基接口的API参考。

## 2. Indexing: Split[](https://python.langchain.com/v0.1/docs/use_cases/question_answering/quickstart/#indexing-split)

我们加载的文档超过42k字符长。这太长了，不适合许多模型的上下文窗口。即使对于那些可以在上下文窗口中容纳完整帖子的模型，模型也很难在很长的输入中找到信息。

为了解决这个问题，我们将文档分割成块，用于嵌入和向量存储。这应该可以帮助我们在运行时只检索blog文章中最相关的部分。

在本例中，我们将文档分成1000个字符的块，块之间有200个字符的重叠。这种重叠有助于减少将语句与相关的重要上下文分隔开的可能性。我们使用RecursiveSegmenterTextSplitter，它将使用常见的分隔符（如新行）递归地拆分文档，直到每个块都是适当的大小。对于一般文本用例，这是推荐的文本拆分器。

我们设置add_start_index=True，以便每个拆分的Document在初始Document中开始的字符索引被保留为元数据属性“start_index”。

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)
```



#### API Reference:

- [RecursiveCharacterTextSplitter](https://api.python.langchain.com/en/latest/character/langchain_text_splitters.character.RecursiveCharacterTextSplitter.html)

```python
len(all_splits)
```

```text
66
```

```python
len(all_splits[0].page_content)
```

```text
969
```

```python
all_splits[10].metadata
```

```text
{'source': 'https://lilianweng.github.io/posts/2023-06-23-agent/',
 'start_index': 7056}
```



### Go deeper[](https://python.langchain.com/v0.1/docs/use_cases/question_answering/quickstart/#go-deeper-1)

TextSplitter：将文档列表拆分为较小块的对象。DocumentTransformer % s的子类。

- Explore `Context-aware splitters`, which keep the location (“context”) of each split in the original `Document`: - [Markdown files](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/markdown_header_metadata/)
- [Code (py or js)](https://python.langchain.com/v0.1/docs/integrations/document_loaders/source_code/)
- [Scientific papers](https://python.langchain.com/v0.1/docs/integrations/document_loaders/grobid/)
- [Interface](https://api.python.langchain.com/en/latest/base/langchain_text_splitters.base.TextSplitter.html): API reference for the base interface.

`DocumentTransformer`: Object that performs a transformation on a list of `Document`s.

- [Docs](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/): Detailed documentation on how to use `DocumentTransformers`
- [Integrations](https://python.langchain.com/v0.1/docs/integrations/document_transformers/)
- [Interface](https://api.python.langchain.com/en/latest/documents/langchain_core.documents.transformers.BaseDocumentTransformer.html): API reference for the base interface.

## 3. Indexing: Store[](https://python.langchain.com/v0.1/docs/use_cases/question_answering/quickstart/#indexing-store)

现在我们需要索引66个文本块，以便在运行时搜索它们。最常见的方法是嵌入每个文档拆分的内容，并将这些嵌入插入到向量数据库（或向量存储）中。当我们想要搜索我们的分割时，我们获取一个文本搜索查询，嵌入它，并执行某种“相似性”搜索，以识别与我们的查询嵌入最相似的嵌入的存储分割。最简单的相似性度量是余弦相似性-我们测量每对嵌入（高维向量）之间的夹角余弦。

我们可以使用Chroma vector store和OpenAIEmbeddings模型在一个命令中嵌入和存储所有的文档分割。

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
```

#### API Reference:

- [OpenAIEmbeddings](https://api.python.langchain.com/en/latest/embeddings/langchain_openai.embeddings.base.OpenAIEmbeddings.html)

### Go deeper[](https://python.langchain.com/v0.1/docs/use_cases/question_answering/quickstart/#go-deeper-2)

Embeddings：围绕文本嵌入模型的包装器，用于将文本转换为嵌入。

- [Docs](https://python.langchain.com/v0.1/docs/modules/data_connection/text_embedding/): 关于如何使用embeddings的详细文档。
- [Integrations](https://python.langchain.com/v0.1/docs/integrations/text_embedding/): 30+集成选择。
- [Interface](https://api.python.langchain.com/en/latest/embeddings/langchain_core.embeddings.Embeddings.html): API reference for the base interface.

`VectorStore`: Wrapper around a vector database, used for storing and querying embeddings.

- [Docs](https://python.langchain.com/v0.1/docs/modules/data_connection/vectorstores/): Detailed documentation on how to use vector stores.
- [Integrations](https://python.langchain.com/v0.1/docs/integrations/vectorstores/): 40+ integrations to choose from.
- [Interface](https://api.python.langchain.com/en/latest/vectorstores/langchain_core.vectorstores.VectorStore.html): API reference for the base interface.

This completes the **Indexing** portion of the pipeline. At this point we have a query-able vector store containing the chunked contents of our blog post. Given a user question, we should ideally be able to return the snippets of the blog post that answer the question.

## 4. Retrieval and Generation: Retrieve[](https://python.langchain.com/v0.1/docs/use_cases/question_answering/quickstart/#retrieval-and-generation-retrieve)

Now let’s write the actual application logic. We want to create a simple application that takes a user question, searches for documents relevant to that question, passes the retrieved documents and initial question to a model, and returns an answer.

First we need to define our logic for searching over documents. LangChain defines a [Retriever](https://python.langchain.com/v0.1/docs/modules/data_connection/retrievers/) interface which wraps an index that can return relevant `Documents` given a string query.

The most common type of `Retriever` is the [VectorStoreRetriever](https://python.langchain.com/v0.1/docs/modules/data_connection/retrievers/vectorstore/), which uses the similarity search capabilities of a vector store to facilitate retrieval. Any `VectorStore` can easily be turned into a `Retriever` with `VectorStore.as_retriever()`:

```python
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
```

```python
retrieved_docs = retriever.invoke("What are the approaches to Task Decomposition?")
```

```python
len(retrieved_docs)
```

```text
6
```

```python
print(retrieved_docs[0].page_content)
```

```text
Tree of Thoughts (Yao et al. 2023) extends CoT by exploring multiple reasoning possibilities at each step. It first decomposes the problem into multiple thought steps and generates multiple thoughts per step, creating a tree structure. The search process can be BFS (breadth-first search) or DFS (depth-first search) with each state evaluated by a classifier (via a prompt) or majority vote.
Task decomposition can be done (1) by LLM with simple prompting like "Steps for XYZ.\n1.", "What are the subgoals for achieving XYZ?", (2) by using task-specific instructions; e.g. "Write a story outline." for writing a novel, or (3) with human inputs.
```



### Go deeper[](https://python.langchain.com/v0.1/docs/use_cases/question_answering/quickstart/#go-deeper-3)

Vector stores are commonly used for retrieval, but there are other ways to do retrieval, too.

`Retriever`: An object that returns `Document`s given a text query

- Docs: Further documentation on the interface and built-in retrieval techniques. Some of which include:
  - `MultiQueryRetriever` [generates variants of the input question](https://python.langchain.com/v0.1/docs/modules/data_connection/retrievers/MultiQueryRetriever/) to improve retrieval hit rate.
  - `MultiVectorRetriever` (diagram below) instead generates [variants of the embeddings](https://python.langchain.com/v0.1/docs/modules/data_connection/retrievers/multi_vector/), also in order to improve retrieval hit rate.
  - `Max marginal relevance` selects for [relevance and diversity](https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf) among the retrieved documents to avoid passing in duplicate context.
  - Documents can be filtered during vector store retrieval using metadata filters, such as with a [Self Query Retriever](https://python.langchain.com/v0.1/docs/modules/data_connection/retrievers/self_query/).
- [Integrations](https://python.langchain.com/v0.1/docs/integrations/retrievers/): Integrations with retrieval services.
- [Interface](https://api.python.langchain.com/en/latest/retrievers/langchain_core.retrievers.BaseRetriever.html): API reference for the base interface.

## 5. Retrieval and Generation: Generate[](https://python.langchain.com/v0.1/docs/use_cases/question_answering/quickstart/#retrieval-and-generation-generate)

Let’s put it all together into a chain that takes a question, retrieves relevant documents, constructs a prompt, passes that to a model, and parses the output.

We’ll use the gpt-3.5-turbo OpenAI chat model, but any LangChain `LLM` or `ChatModel` could be substituted in.

### OpenAI

##### Install dependencies

```bash
pip install -qU langchain-openai
```



##### Set environment variables

```python
import getpass
import os

os.environ["OPENAI_API_KEY"] = getpass.getpass()
```

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
```

We’ll use a prompt for RAG that is checked into the LangChain prompt hub ([here](https://smith.langchain.com/hub/rlm/rag-prompt)).

```python
from langchain import hub

prompt = hub.pull("rlm/rag-prompt")
```

```python
example_messages = prompt.invoke(
    {"context": "filler context", "question": "filler question"}
).to_messages()
example_messages
```

```text
[HumanMessage(content="You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: filler question \nContext: filler context \nAnswer:")]
```

```python
print(example_messages[0].content)
```

```text
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: filler question
Context: filler context
Answer:
```

We’ll use the [LCEL Runnable](https://python.langchain.com/v0.1/docs/expression_language/) protocol to define the chain, allowing us to - pipe together components and functions in a transparent way - automatically trace our chain in LangSmith - get streaming, async, and batched calling out of the box

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

#### API Reference:

- [StrOutputParser](https://api.python.langchain.com/en/latest/output_parsers/langchain_core.output_parsers.string.StrOutputParser.html)
- [RunnablePassthrough](https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.passthrough.RunnablePassthrough.html)

```python
for chunk in rag_chain.stream("What is Task Decomposition?"):
    print(chunk, end="", flush=True)
```

```text
Task decomposition is a technique used to break down complex tasks into smaller and simpler steps. It involves transforming big tasks into multiple manageable tasks, allowing for easier interpretation and execution by autonomous agents or models. Task decomposition can be done through various methods, such as using prompting techniques, task-specific instructions, or human inputs.
```

Check out the [LangSmith trace](https://smith.langchain.com/public/1799e8db-8a6d-4eb2-84d5-46e8d7d5a99b/r)

### Go deeper[](https://python.langchain.com/v0.1/docs/use_cases/question_answering/quickstart/#go-deeper-4)

#### Choosing a model[](https://python.langchain.com/v0.1/docs/use_cases/question_answering/quickstart/#choosing-a-model)

`ChatModel`: An LLM-backed chat model. Takes in a sequence of messages and returns a message.

- [Docs](https://python.langchain.com/v0.1/docs/modules/model_io/chat/)
- [Integrations](https://python.langchain.com/v0.1/docs/integrations/chat/): 25+ integrations to choose from.
- [Interface](https://api.python.langchain.com/en/latest/language_models/langchain_core.language_models.chat_models.BaseChatModel.html): API reference for the base interface.

`LLM`: A text-in-text-out LLM. Takes in a string and returns a string.

- [Docs](https://python.langchain.com/v0.1/docs/modules/model_io/llms/)
- [Integrations](https://python.langchain.com/v0.1/docs/integrations/llms/): 75+ integrations to choose from.
- [Interface](https://api.python.langchain.com/en/latest/language_models/langchain_core.language_models.llms.BaseLLM.html): API reference for the base interface.

See a guide on RAG with locally-running models [here](https://python.langchain.com/v0.1/docs/use_cases/question_answering/local_retrieval_qa/).

#### Customizing the prompt[](https://python.langchain.com/v0.1/docs/use_cases/question_answering/quickstart/#customizing-the-prompt)

As shown above, we can load prompts (e.g., [this RAG prompt](https://smith.langchain.com/hub/rlm/rag-prompt)) from the prompt hub. The prompt can also be easily customized:

```python
from langchain_core.prompts import PromptTemplate

template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer.

{context}

Question: {question}

Helpful Answer:"""
custom_rag_prompt = PromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
)

rag_chain.invoke("What is Task Decomposition?")
```

#### API Reference:

- [PromptTemplate](https://api.python.langchain.com/en/latest/prompts/langchain_core.prompts.prompt.PromptTemplate.html)

```text
'Task decomposition is a technique used to break down complex tasks into smaller and simpler steps. It involves transforming big tasks into multiple manageable tasks, allowing for a more systematic and organized approach to problem-solving. Thanks for asking!'
```

Check out the [LangSmith trace](https://smith.langchain.com/public/da23c4d8-3b33-47fd-84df-a3a582eedf84/r)

## Next steps[](https://python.langchain.com/v0.1/docs/use_cases/question_answering/quickstart/#next-steps)

That’s a lot of content we’ve covered in a short amount of time. There’s plenty of features, integrations, and extensions to explore in each of the above sections. Along from the **Go deeper** sources mentioned above, good next steps include:

- [Return sources](https://python.langchain.com/v0.1/docs/use_cases/question_answering/sources/): Learn how to return source documents
- [Streaming](https://python.langchain.com/v0.1/docs/use_cases/question_answering/streaming/): Learn how to stream outputs and intermediate steps
- [Add chat history](https://python.langchain.com/v0.1/docs/use_cases/question_answering/chat_history/): Learn how to add chat history to your app