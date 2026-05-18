# LangChain Integration Guide

> **Load when**: integrating vector stores (Chroma, Pinecone, FAISS), setting up LangSmith observability, or deploying to production.

## Vector Stores

### Chroma (Local)
```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=OpenAIEmbeddings(),
    persist_directory="./chroma_db"
)

# Reload later
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=OpenAIEmbeddings(),
)
```

### FAISS (Fast, No Metadata)
```python
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())
vectorstore.save_local("faiss_index")

# Reload
vectorstore = FAISS.load_local(
    "faiss_index",
    OpenAIEmbeddings(),
    allow_dangerous_deserialization=True  # Only for trusted sources
)
```

### Pinecone (Cloud)
```python
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

pc = Pinecone(api_key="your-key")
index = pc.Index("my-index")

vectorstore = PineconeVectorStore(
    index=index,
    embedding=OpenAIEmbeddings(),
)
```

### Weaviate (GraphQL-Native)
```python
from langchain_weaviate import WeaviateVectorStore
import weaviate

client = weaviate.connect_to_local()
vectorstore = WeaviateVectorStore(client=client, index_name="Docs", embedding=OpenAIEmbeddings())
```

## LangSmith Observability

### Setup
```python
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "ls__..."
os.environ["LANGCHAIN_PROJECT"] = "my-project"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
```

### View Traces
```python
# All chains and agents are automatically traced after setup
# View at: https://smith.langchain.com

# Tag runs for filtering
from langchain.callbacks.tracers.langchain import LangChainTracer

tracer = LangChainTracer(project_name="my-project")
agent.invoke({"input": "test"}, config={"callbacks": [tracer]})
```

### Feedback and Evaluation
```python
from langsmith import Client

client = Client()

# Log feedback on a run
client.create_feedback(
    run_id="run-uuid",
    key="user-satisfaction",
    score=0.9,
    comment="Accurate and helpful",
)
```

## Model Providers

### OpenAI
```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
```

### Anthropic
```python
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-sonnet-4-5-20250929")
```

### Google (Gemini)
```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")
```

### Local Models (Ollama)
```python
from langchain_ollama import ChatOllama, OllamaEmbeddings

llm = ChatOllama(model="llama3.1")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
```

## Caching

### In-Memory Cache
```python
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache

set_llm_cache(InMemoryCache())
```

### SQLite Cache
```python
from langchain.cache import SQLiteCache

set_llm_cache(SQLiteCache(database_path=".langchain.db"))
```

## Callbacks

### Custom Callback
```python
from langchain.callbacks.base import BaseCallbackHandler

class LoggingHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"LLM called with {len(prompts)} prompt(s)")

    def on_llm_end(self, response, **kwargs):
        print(f"LLM response: {response.generations[0][0].text[:100]}...")

    def on_tool_start(self, serialized, input_str, **kwargs):
        print(f"Tool called: {serialized['name']}")

    def on_tool_end(self, output, **kwargs):
        print(f"Tool output: {str(output)[:100]}...")

# Use in agent
agent.invoke(
    {"input": "test"},
    config={"callbacks": [LoggingHandler()]}
)
```

### Cost Tracking
```python
from langchain.callbacks import get_openai_callback

with get_openai_callback() as cb:
    result = agent.invoke({"input": "Research AI trends"})

print(f"Total tokens: {cb.total_tokens}")
print(f"Total cost: ${cb.total_cost:.4f}")
```

## Production Deployment

### Serialize and Load
```python
# Save chain
import pickle
with open("chain.pkl", "wb") as f:
    pickle.dump(qa_chain, f)

# Load chain
with open("chain.pkl", "rb") as f:
    qa_chain = pickle.load(f)
```

### Serve with FastAPI
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/qa")
def qa_endpoint(query: Query):
    result = qa_chain.invoke({"query": query.question})
    return {"answer": result["result"], "sources": result["source_documents"]}
```

## Best Practices

1. **Cache LLM calls** — Use InMemoryCache or SQLiteCache
2. **Persist vector stores** — Don't re-embed on every run
3. **Use LangSmith** — Essential for debugging agent behavior
4. **Track costs** — Use `get_openai_callback()` context manager
5. **Set timeouts** — Add `request_timeout` to LLM config
6. **Version prompts** — Store in config files, not hardcoded
7. **Use environment variables** — Never hardcode API keys
