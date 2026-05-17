# LlamaIndex Agents Guide

> **Load when**: creating RAG agents, building tool-calling agents with document search, or implementing multi-step reasoning.

## Basic Agent

```python
from llama_index.core.agent import FunctionAgent
from llama_index.llms.openai import OpenAI

def multiply(a: int, b: int) -> int:
    """Multiply two numbers. Use for multiplication."""
    return a * b

def add(a: int, b: int) -> int:
    """Add two numbers. Use for addition."""
    return a + b

llm = OpenAI(model="gpt-4o")
agent = FunctionAgent.from_tools(
    tools=[multiply, add],
    llm=llm,
    verbose=True,
)
response = agent.chat("What is 25 * 17 + 142?")
print(response)
```

## RAG Agent (Document + Tools)

```python
from llama_index.core.tools import QueryEngineTool, ToolMetadata

# Create index
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=3)

# Wrap as tool
doc_tool = QueryEngineTool(
    query_engine=query_engine,
    metadata=ToolMetadata(
        name="python_docs",
        description="Search Python documentation. Use for Python-related questions."
    ),
)

# Agent with doc search + calculator
agent = FunctionAgent.from_tools(
    tools=[doc_tool, multiply, add],
    llm=llm,
    verbose=True,
)

# Agent decides when to search docs vs use calculator
response = agent.chat("According to the docs, what is the recommended way to handle errors?")
```

## React Agent

```python
from llama_index.core.agent import ReActAgent

agent = ReActAgent.from_tools(
    tools=[doc_tool, multiply, add],
    llm=llm,
    verbose=True,
    max_iterations=10,
)
```

## Tool Specifications

### Function as Tool
```python
from llama_index.core.tools import FunctionTool

def get_stock_price(ticker: str) -> str:
    """Get current stock price.
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')
    """
    return f"{ticker}: $150.25"

stock_tool = FunctionTool.from_defaults(
    fn=get_stock_price,
    name="stock_price",
    description="Get current stock price by ticker symbol",
)
```

### Class-Based Tool
```python
from llama_index.core.tools import BaseTool
from pydantic import BaseModel, Field

class WeatherInput(BaseModel):
    city: str = Field(description="City name")

class WeatherTool(BaseTool):
    def _run(self, city: str) -> str:
        return f"Weather in {city}: 72°F, sunny"

    async def _arun(self, city: str) -> str:
        return self._run(city)
```

### Multiple Query Engine Tools
```python
tools = [
    QueryEngineTool(
        query_engine=python_index.as_query_engine(),
        metadata=ToolMetadata(name="python_docs", description="Python documentation"),
    ),
    QueryEngineTool(
        query_engine=js_index.as_query_engine(),
        metadata=ToolMetadata(name="js_docs", description="JavaScript documentation"),
    ),
]

agent = FunctionAgent.from_tools(tools=tools, llm=llm)
response = agent.chat("Compare Python's async with JavaScript promises")
```

## Agent Configuration

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `max_iterations` | 10 | Max reasoning steps |
| `verbose` | False | Print reasoning |
| `system_prompt` | None | Custom system prompt |
| `max_function_calls` | None | Max tool calls per step |
| `tool_choice` | "auto" | "auto", "required", or specific tool |

## Custom System Prompt

```python
agent = FunctionAgent.from_tools(
    tools=[doc_tool, calculator_tool],
    llm=llm,
    system_prompt=(
        "You are a research assistant specializing in Python.\n"
        "Always cite specific documentation when answering.\n"
        "If you don't know something, say so."
    ),
)
```

## Multi-Step Reasoning

```python
agent = FunctionAgent.from_tools(
    tools=[search_tool, analysis_tool, write_tool],
    llm=llm,
    verbose=True,
)

# Agent will automatically chain tool calls:
# 1. Search for data
# 2. Analyze results
# 3. Synthesize answer
response = agent.chat(
    "Research the top 3 Python web frameworks and recommend the best for a startup"
)
```

## OpenAIAgent (Advanced)

```python
from llama_index.agent.openai import OpenAIAgent

agent = OpenAIAgent.from_tools(
    tools=[doc_tool, calculator_tool],
    llm=OpenAI(model="gpt-4o"),
    verbose=True,
)
```

## Common Patterns

### Conditional Tool Selection
```python
# Agent automatically chooses tools based on query
# No special code needed — it's built-in
response = agent.chat("Calculate 100 * 50")        # Uses calculator
response = agent.chat("What is a decorator?")      # Uses docs search
```

### Error Recovery
```python
def safe_api_call(query: str) -> str:
    """Call external API with error handling."""
    try:
        result = api.search(query)
        return str(result)
    except Exception as e:
        return f"Error calling API: {str(e)}. Please try a different approach."

agent = FunctionAgent.from_tools(
    tools=[FunctionTool.from_defaults(fn=safe_api_call)],
    llm=llm,
)
```

## Common Pitfalls

1. **Tool descriptions too vague** → Agent won't know when to use them
2. **No error handling in tools** → Tool failure crashes the agent
3. **Too many tools** → Agent gets confused; 3-7 tools is optimal
4. **Missing metadata** → QueryEngineTool needs descriptive name + description
5. **Agent loops** → Set `max_iterations` to prevent infinite loops
