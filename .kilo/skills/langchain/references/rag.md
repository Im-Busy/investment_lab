# LangChain RAG Guide

> **Load when**: building retrieval-augmented generation pipelines, document Q&A, or conversational RAG.

## Complete RAG Pipeline (5 Steps)

```python
# 1. Load documents
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://docs.python.org/3/tutorial/")
docs = loader.load()

# 2. Split into chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)
splits = splitter.split_documents(docs)

# 3. Create embeddings and vector store
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=OpenAIEmbeddings(),
    persist_directory="./chroma_db"
)

# 4. Create retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",  # "similarity", "mmr", "similarity_score_threshold"
    search_kwargs={"k": 4}
)

# 5. Create QA chain
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4o"),
    chain_type="stuff",           # "stuff", "map_reduce", "refine", "map_rerank"
    retriever=retriever,
    return_source_documents=True,
)

result = qa_chain.invoke({"query": "What are Python decorators?"})
print(result["result"])
print(f"Sources: {result['source_documents']}")
```

## Document Loaders

### Web
```python
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://example.com")
docs = loader.load()
```

### PDFs
```python
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("paper.pdf")
docs = loader.load()
```

### GitHub
```python
from langchain_community.document_loaders import GithubFileLoader
loader = GithubFileLoader(
    repo="owner/repo",
    access_token="ghp_...",
    github_api_url="https://api.github.com",
    file_filter=lambda x: x.endswith(".py")
)
docs = loader.load()
```

### CSV
```python
from langchain_community.document_loaders import CSVLoader
loader = CSVLoader("data.csv")
docs = loader.load()
```

### Directory
```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader
loader = DirectoryLoader("./docs/", glob="**/*.md", loader_cls=TextLoader)
docs = loader.load()
```

## Text Splitters

### Recursive (General Purpose)
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,       # Target chunk size in characters
    chunk_overlap=200,     # Overlap between chunks
    length_function=len,
    separators=["\n\n", "\n", " ", ""],
)
```

### Token-Based (LLM-Aware)
```python
from langchain.text_splitter import TokenTextSplitter

splitter = TokenTextSplitter(
    chunk_size=500,        # Tokens
    chunk_overlap=50,
)
```

### Code-Aware
```python
from langchain.text_splitter import PythonCodeTextSplitter
splitter = PythonCodeTextSplitter(chunk_size=500, chunk_overlap=50)
```

### Semantic (Embeds then splits)
```python
from langchain_experimental.text_splitter import SemanticChunker
splitter = SemanticChunker(OpenAIEmbeddings())
```

## Retriever Types

```python
# Similarity (default)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# MMR (Max Marginal Relevance — diverse results)
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 20})

# Similarity with score threshold
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.5, "k": 4}
)

# Self-query (metadata-aware)
from langchain.retrievers.self_query.base import SelfQueryRetriever
retriever = SelfQueryRetriever.from_llm(
    llm=ChatOpenAI(model="gpt-4o"),
    vectorstore=vectorstore,
    document_description="Python documentation pages",
    metadata_field_info=[...],
)
```

## Chain Types

| Chain | When | Trade-off |
|-------|------|-----------|
| `stuff` | Short docs, < 4 chunks | Fast, one LLM call |
| `map_reduce` | Many docs, parallel | More LLM calls, parallelizable |
| `refine` | Sequential refinement | Better for long docs, sequential |
| `map_rerank` | Rerank by relevance | Good for QA with many docs |

## Conversational RAG

```python
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
)

qa = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-4o"),
    retriever=retriever,
    memory=memory,
    verbose=True,
)

response1 = qa.invoke({"question": "What is Python used for?"})
response2 = qa.invoke({"question": "Can you elaborate?"})  # Remembers context
```

## Hybrid Search (Vector + Keyword)

```python
from langchain.retrievers import BM25Retriever, EnsembleRetriever

bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 4

vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

ensemble = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.3, 0.7],  # Weighted combination
)
```

## RAG Evaluation

```python
# Check if answer is grounded in retrieved docs
from langchain.evaluation import load_evaluator

evaluator = load_evaluator("labeled_criteria", criteria="correctness")
eval_result = evaluator.evaluate_strings(
    prediction="The answer is X",
    reference="The correct answer is Y",
    input="What is X?",
)
```

## Common Pitfalls

1. **Chunk size too small** → Loss of context. Use 500-1000 tokens
2. **Chunk overlap too small** → Missing connections. Use 10-20% overlap
3. **Wrong chain type** → `stuff` fails on many docs. Use `map_reduce` or `refine`
4. **No source citation** → Always `return_source_documents=True`
5. **Stale vector store** → Re-index when documents change
6. **Embedding cost** → Embed once, persist, reload. Don't re-embed.
