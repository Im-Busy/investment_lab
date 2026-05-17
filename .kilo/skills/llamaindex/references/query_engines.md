# LlamaIndex Query Engines Guide

> **Load when**: configuring query modes, customizing retrieval, streaming responses, or optimizing RAG performance.

## Query Engine Creation

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)

# Default query engine
query_engine = index.as_query_engine()
response = query_engine.query("What is this document about?")
```

## Response Modes

| Mode | Description | When to Use |
|------|-------------|-------------|
| `compact` | Concatenate chunks, single LLM call (default) | Most cases, efficient |
| `tree_summarize` | Summarize chunks bottom-up | Long docs, many chunks |
| `refine` | Sequential refinement, chunk by chunk | Preserves detail across chunks |
| `simple_summarize` | Truncate to fit context | Quick, lossy |
| `accumulate` | Apply query to each chunk, accumulate | Joining answers from chunks |
| `compact_accumulate` | Compact then accumulate | Best of both |
| `generation` | Ignore retrieved chunks | Testing LLM standalone |
| `no_text` | Return nodes without LLM | Debugging retrieval quality |

```python
query_engine = index.as_query_engine(
    response_mode="tree_summarize",
    similarity_top_k=10,
    verbose=True,
)
```

## Streaming

```python
query_engine = index.as_query_engine(streaming=True)
response = query_engine.query("Explain the main concepts")

# Stream token by token
for token in response.response_gen:
    print(token, end="", flush=True)
```

## Retrieval Configuration

### Top-K and Similarity
```python
query_engine = index.as_query_engine(similarity_top_k=5)
```

### Metadata Filtering
```python
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter

filters = MetadataFilters(
    filters=[
        ExactMatchFilter(key="category", value="tutorial"),
        ExactMatchFilter(key="difficulty", value="beginner"),
    ]
)

query_engine = index.as_query_engine(similarity_top_k=3, filters=filters)
```

### Custom Retriever
```python
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore

class CustomRetriever(BaseRetriever):
    def _retrieve(self, query_bundle):
        # Your retrieval logic
        nodes = [NodeWithScore(node=node, score=0.9) for node in self._index.docstore.docs.values()]
        return nodes[:5]
```

## Customization

### Custom Prompt
```python
from llama_index.core import PromptTemplate

qa_prompt = PromptTemplate(
    "Context information:\n{context_str}\n\n"
    "Question: {query_str}\n\n"
    "Answer concisely using only the context. If unsure, say 'I don't know'.\n"
    "Answer: "
)

query_engine = index.as_query_engine(text_qa_template=qa_prompt)
```

### Custom LLM
```python
from llama_index.llms.anthropic import Anthropic

query_engine = index.as_query_engine(llm=Anthropic(model="claude-sonnet-4-5-20250929"))
```

### Hybrid Query Engine (Custom + Default)
```python
from llama_index.core.query_engine import CustomQueryEngine

class MyQueryEngine(CustomQueryEngine):
    llm: Any
    retriever: Any

    def custom_query(self, query_str: str) -> Response:
        nodes = self.retriever.retrieve(query_str)
        context = "\n".join([n.text for n in nodes])
        response = self.llm.complete(f"Context: {context}\nQuery: {query_str}")
        return Response(response=str(response))

engine = MyQueryEngine(llm=llm, retriever=retriever)
```

## Structured Output

```python
from pydantic import BaseModel
from llama_index.core.output_parsers import PydanticOutputParser

class Summary(BaseModel):
    title: str
    main_points: list[str]
    conclusion: str

parser = PydanticOutputParser(output_cls=Summary)
query_engine = index.as_query_engine(output_parser=parser)

response = query_engine.query("Summarize the document")
print(response.title, response.main_points)
```

## Sub-Question Query Engine

Breaks complex queries into sub-questions, answers each, combines:

```python
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine

tool = QueryEngineTool(
    query_engine=index.as_query_engine(),
    metadata=ToolMetadata(name="docs", description="Documentation about the product"),
)

sub_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=[tool],
    llm=llm,
)

response = sub_engine.query("Compare features A and B, and recommend which to use")
```

## Evaluation

```python
from llama_index.core.evaluation import RelevancyEvaluator, FaithfulnessEvaluator

relevancy = RelevancyEvaluator(llm=llm)
relevancy_result = relevancy.evaluate_response(query="What is X?", response=response)
print(f"Relevant: {relevancy_result.passing}")

faithfulness = FaithfulnessEvaluator(llm=llm)
faith_result = faithfulness.evaluate_response(query="What is X?", response=response)
print(f"Faithful: {faith_result.passing}")
```

## Query Engine vs Chat Engine

| Feature | Query Engine | Chat Engine |
|---------|-------------|-------------|
| Memory | No | Yes |
| Multi-turn | No | Yes |
| Use for | Single questions | Conversations |
| API | `query("text")` | `chat("text")` |

```python
# Chat engine (conversational)
chat_engine = index.as_chat_engine(chat_mode="condense_plus_context")
response1 = chat_engine.chat("What is Python?")
response2 = chat_engine.chat("Give examples")  # Remembers Python context
```

## Common Pitfalls

1. **Response mode mismatch** → `compact` fails with very long docs; use `tree_summarize`
2. **`similarity_top_k` too low** → Missing relevant chunks; start with 5-10
3. **Metadata filters not working** → Ensure metadata exists on nodes
4. **Streaming with non-streaming LLM** → Check LLM supports streaming
5. **Forgetting to persist** → Index lost on restart; use `storage_context.persist()`
