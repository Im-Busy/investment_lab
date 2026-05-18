# LangChain Agents Guide

> **Load when**: creating agents with tool calling, implementing ReAct pattern, streaming agent execution, or debugging agent behavior.

## Agent Creation (Simplest)

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"It's sunny in {city}, 72°F"

agent = create_agent(
    model=ChatOpenAI(model="gpt-4o"),
    tools=[get_weather],
    system_prompt="You are a helpful assistant. Use tools when needed."
)

result = agent.invoke({"messages": [{"role": "user", "content": "Weather in Paris?"}]})
print(result["messages"][-1].content)
```

## Tool Calling Agent

```python
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool

@tool
def calculator(expression: str) -> str:
    """Evaluate math expressions. Input: '2 + 2'."""
    return str(eval(expression))

@tool
def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant with access to tools."),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

agent = create_tool_calling_agent(
    llm=ChatOpenAI(model="gpt-4o"),
    tools=[calculator, search],
    prompt=prompt,
)

executor = AgentExecutor(
    agent=agent,
    tools=[calculator, search],
    verbose=True,
    max_iterations=10,
    handle_parsing_errors=True,
)

result = executor.invoke({"input": "What is 25 * 17 + 142?"})
```

## ReAct Agent

```python
from langchain.agents import create_react_agent
from langchain_core.prompts import PromptTemplate

react_template = """Answer the following questions as best you can.
You have access to the following tools:

{tools}

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

prompt = PromptTemplate.from_template(react_template)

agent = create_react_agent(
    llm=ChatOpenAI(model="gpt-4o"),
    tools=[calculator, search],
    prompt=prompt,
)

executor = AgentExecutor(
    agent=agent,
    tools=[calculator, search],
    verbose=True,
    max_iterations=15,
    handle_parsing_errors=True,
)
```

## Streaming Agents

```python
# Stream agent execution step by step
for event in executor.stream({"input": "Research latest AI trends"}):
    if "actions" in event:
        for action in event["actions"]:
            print(f"Tool: {action.tool}, Input: {action.tool_input}")
    if "output" in event:
        print(f"Output: {event['output']}")
```

## Structured Output Agent

```python
from langchain_core.pydantic_v1 import BaseModel, Field

class AnalysisResult(BaseModel):
    sentiment: str = Field(description="positive, negative, or neutral")
    confidence: float = Field(description="0.0 to 1.0")
    key_points: list[str] = Field(description="3-5 key findings")

structured_llm = ChatOpenAI(model="gpt-4o").with_structured_output(AnalysisResult)

agent = create_agent(
    model=structured_llm,
    tools=[],
    system_prompt="Analyze the text and return structured analysis."
)
```

## Tool Error Handling

```python
@tool
def risky_api(query: str) -> str:
    """Call external API that might fail."""
    try:
        result = external_api_call(query)
        return str(result)
    except TimeoutError:
        return "Error: API timed out. Try a simpler query."
    except ConnectionError:
        return "Error: API unavailable. Try again later."
    except Exception as e:
        return f"Error: {str(e)}"
```

## Agent Configuration Options

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `max_iterations` | 15 | Max tool-calling cycles |
| `max_execution_time` | None | Max wall-clock time (seconds) |
| `handle_parsing_errors` | False | Auto-retry on malformed output |
| `early_stopping_method` | "force" | "force" or "generate" |
| `return_intermediate_steps` | False | Include tool call history |
| `verbose` | False | Print reasoning steps |

## Multi-Agent Patterns

### Sequential Agents
```python
# Agent 1: Gather data
result1 = data_agent.invoke({"messages": [{"role": "user", "content": "Find stock data for AAPL"}]})

# Agent 2: Analyze data (uses Agent 1's output as context)
result2 = analysis_agent.invoke({
    "messages": [{"role": "user", "content": f"Analyze this data: {result1['messages'][-1].content}"}]
})
```

### Agent Delegation
```python
# Supervisor agent that delegates to specialist agents
tools = [
    Tool(name="research_agent", func=research_agent.invoke, description="Research topics"),
    Tool(name="coder_agent", func=coder_agent.invoke, description="Write code"),
]

supervisor = create_agent(
    model=ChatOpenAI(model="gpt-4o"),
    tools=tools,
    system_prompt="Delegate tasks to specialist agents based on the request type."
)
```

## Common Pitfalls

1. **Agent loops infinitely** → Set `max_iterations` (default 15 is usually fine)
2. **Parsing errors** → Enable `handle_parsing_errors=True`
3. **Tool not being called** → Make tool descriptions clearer and more specific
4. **Token limit exceeded** → Reduce system prompt length, limit intermediate steps
5. **Slow response** → Use streaming, reduce max_iterations, use faster model
