# Quickstart

LangChain has a number of components designed to help build question-answering applications, and RAG applications more generally. To familiarize ourselves with these, we’ll build a simple Q&A application over a text data source. Along the way we’ll go over a typical Q&A architecture, discuss the relevant LangChain components, and highlight additional resources for more advanced Q&A techniques. We’ll also see how LangSmith can help us trace and understand our application. LangSmith will become increasingly helpful as our application grows in complexity.

## Architecture[](https://python.langchain.com/v0.1/docs/use_cases/question_answering/quickstart/#architecture)

We’ll create a typical RAG application as outlined in the [Q&A introduction](https://python.langchain.com/v0.1/docs/use_cases/question_answering/), which has two main components:

**Indexing**: a pipeline for ingesting data from a source and indexing it. *This usually happens offline.*

**Retrieval and generation**: the actual RAG chain, which takes the user query at run time and retrieves the relevant data from the index, then passes that to the model.

The full sequence from raw data to answer will look like:

### Indexing[](https://python.langchain.com/v0.1/docs/use_cases/question_answering/quickstart/#indexing)

1. **Load**: First we need to load our data. We’ll use [DocumentLoaders](https://python.langchain.com/v0.1/docs/modules/data_connection/document_loaders/) for this.
2. **Split**: [Text splitters](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/) break large `Documents` into smaller chunks. This is useful both for indexing data and for passing it in to a model, since large chunks are harder to search over and won’t fit in a model’s finite context window.
3. **Store**: We need somewhere to store and index our splits, so that they can later be searched over. This is often done using a [VectorStore](https://python.langchain.com/v0.1/docs/modules/data_connection/vectorstores/) and [Embeddings](https://python.langchain.com/v0.1/docs/modules/data_connection/text_embedding/) model.

### Retrieval and generation[](https://python.langchain.com/v0.1/docs/use_cases/question_answering/quickstart/#retrieval-and-generation)

1. **Retrieve**: Given a user input, relevant splits are retrieved from storage using a [Retriever](https://python.langchain.com/v0.1/docs/modules/data_connection/retrievers/).
2. **Generate**: A [ChatModel](https://python.langchain.com/v0.1/docs/modules/model_io/chat/) / [LLM](https://python.langchain.com/v0.1/docs/modules/model_io/llms/) produces an answer using a prompt that includes the question and the retrieved data

## Setup[](https://python.langchain.com/v0.1/docs/use_cases/question_answering/quickstart/#setup)

### Dependencies[](https://python.langchain.com/v0.1/docs/use_cases/question_answering/quickstart/#dependencies)

We’ll use an OpenAI chat model and embeddings and a Chroma vector store in this walkthrough, but everything shown here works with any [ChatModel](https://python.langchain.com/v0.1/docs/modules/model_io/chat/) or [LLM](https://python.langchain.com/v0.1/docs/modules/model_io/llms/), [Embeddings](https://python.langchain.com/v0.1/docs/modules/data_connection/text_embedding/), and [VectorStore](https://python.langchain.com/v0.1/docs/modules/data_connection/vectorstores/) or [Retriever](https://python.langchain.com/v0.1/docs/modules/data_connection/retrievers/).

We’ll use the following packages:

```python
%pip install --upgrade --quiet  langchain langchain-community langchainhub langchain-openai langchain-chroma bs4
```

We need to set environment variable `OPENAI_API_KEY` for the embeddings model, which can be done directly or loaded from a `.env` file like so:

```python
import getpass
import os

os.environ["OPENAI_API_KEY"] = getpass.getpass()

# import dotenv

# dotenv.load_dotenv()
```

### LangSmith[](https://python.langchain.com/v0.1/docs/use_cases/question_answering/quickstart/#langsmith)

Many of the applications you build with LangChain will contain multiple steps with multiple invocations of LLM calls. As these applications get more and more complex, it becomes crucial to be able to inspect what exactly is going on inside your chain or agent. The best way to do this is with [LangSmith](https://smith.langchain.com/).

Note that LangSmith is not needed, but it is helpful. If you do want to use LangSmith, after you sign up at the link above, make sure to set your environment variables to start logging traces:

```python
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()
```



## Preview[](https://python.langchain.com/v0.1/docs/use_cases/question_answering/quickstart/#preview)

In this guide we’ll build a QA app over the [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) blog post by Lilian Weng, which allows us to ask questions about the contents of the post.

We can create a simple indexing pipeline and RAG chain to do this in ~20 lines of code:

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

Let’s go through the above code step-by-step to really understand what’s going on.

## 1. Indexing: Load[](https://python.langchain.com/v0.1/docs/use_cases/question_answering/quickstart/#indexing-load)

We need to first load the blog post contents. We can use [DocumentLoaders](https://python.langchain.com/v0.1/docs/modules/data_connection/document_loaders/) for this, which are objects that load in data from a source and return a list of [Documents](https://api.python.langchain.com/en/latest/documents/langchain_core.documents.base.Document.html). A `Document` is an object with some `page_content` (str) and `metadata` (dict).

In this case we’ll use the [WebBaseLoader](https://python.langchain.com/v0.1/docs/integrations/document_loaders/web_base/), which uses `urllib` to load HTML from web URLs and `BeautifulSoup` to parse it to text. We can customize the HTML -> text parsing by passing in parameters to the `BeautifulSoup` parser via `bs_kwargs` (see [BeautifulSoup docs](https://beautiful-soup-4.readthedocs.io/en/latest/#beautifulsoup)). In this case only HTML tags with class “post-content”, “post-title”, or “post-header” are relevant, so we’ll remove all others.

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

`DocumentLoader`: Object that loads data from a source as list of `Documents`.

- [Docs](https://python.langchain.com/v0.1/docs/modules/data_connection/document_loaders/): Detailed documentation on how to use `DocumentLoaders`.
- [Integrations](https://python.langchain.com/v0.1/docs/integrations/document_loaders/): 160+ integrations to choose from.
- [Interface](https://api.python.langchain.com/en/latest/document_loaders/langchain_core.document_loaders.base.BaseLoader.html): API reference  for the base interface.

## 2. Indexing: Split[](https://python.langchain.com/v0.1/docs/use_cases/question_answering/quickstart/#indexing-split)

Our loaded document is over 42k characters long. This is too long to fit in the context window of many models. Even for those models that could fit the full post in their context window, models can struggle to find information in very long inputs.

To handle this we’ll split the `Document` into chunks for embedding and vector storage. This should help us retrieve only the most relevant bits of the blog post at run time.

In this case we’ll split our documents into chunks of 1000 characters with 200 characters of overlap between chunks. The overlap helps mitigate the possibility of separating a statement from important context related to it. We use the [RecursiveCharacterTextSplitter](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/recursive_text_splitter/), which will recursively split the document using common separators like new lines until each chunk is the appropriate size. This is the recommended text splitter for generic text use cases.

We set `add_start_index=True` so that the character index at which each split Document starts within the initial Document is preserved as metadata attribute “start_index”.

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

`TextSplitter`: Object that splits a list of `Document`s into smaller chunks. Subclass of `DocumentTransformer`s.

- Explore `Context-aware splitters`, which keep the location (“context”) of each split in the original `Document`: - [Markdown files](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/markdown_header_metadata/)
- [Code (py or js)](https://python.langchain.com/v0.1/docs/integrations/document_loaders/source_code/)
- [Scientific papers](https://python.langchain.com/v0.1/docs/integrations/document_loaders/grobid/)
- [Interface](https://api.python.langchain.com/en/latest/base/langchain_text_splitters.base.TextSplitter.html): API reference for the base interface.

`DocumentTransformer`: Object that performs a transformation on a list of `Document`s.

- [Docs](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/): Detailed documentation on how to use `DocumentTransformers`
- [Integrations](https://python.langchain.com/v0.1/docs/integrations/document_transformers/)
- [Interface](https://api.python.langchain.com/en/latest/documents/langchain_core.documents.transformers.BaseDocumentTransformer.html): API reference for the base interface.

## 3. Indexing: Store[](https://python.langchain.com/v0.1/docs/use_cases/question_answering/quickstart/#indexing-store)

Now we need to index our 66 text chunks so that we can search over them at runtime. The most common way to do this is to embed the contents of each document split and insert these embeddings into a vector database (or vector store). When we want to search over our splits, we take a text search query, embed it, and perform some sort of “similarity” search to identify the stored splits with the most similar embeddings to our query embedding. The simplest similarity measure is cosine similarity — we measure the cosine of the angle between each pair of embeddings (which are high dimensional vectors).

We can embed and store all of our document splits in a single command using the [Chroma](https://python.langchain.com/v0.1/docs/integrations/vectorstores/chroma/) vector store and [OpenAIEmbeddings](https://python.langchain.com/v0.1/docs/integrations/text_embedding/openai/) model.

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
```

#### API Reference:

- [OpenAIEmbeddings](https://api.python.langchain.com/en/latest/embeddings/langchain_openai.embeddings.base.OpenAIEmbeddings.html)

### Go deeper[](https://python.langchain.com/v0.1/docs/use_cases/question_answering/quickstart/#go-deeper-2)

`Embeddings`: Wrapper around a text embedding model, used for converting text to embeddings.

- [Docs](https://python.langchain.com/v0.1/docs/modules/data_connection/text_embedding/): Detailed documentation on how to use embeddings.
- [Integrations](https://python.langchain.com/v0.1/docs/integrations/text_embedding/): 30+ integrations to choose from.
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