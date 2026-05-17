# Troubleshooting — CrewAI Common Issues

> **Load when**: agents get stuck, tasks fail, memory errors occur, or debugging agent behavior.

## Agent Issues

### Agent Stuck in Loop
**Symptom**: Agent repeats the same action endlessly.

**Fix**:
```python
agent = Agent(
    role="Researcher",
    max_iter=10,           # Limit reasoning iterations (default 15)
    max_rpm=5              # Rate limit requests per minute
)
```

### Agent Not Using Context
**Symptom**: Task 2 doesn't use output from Task 1.

**Fix**:
```python
task2 = Task(
    description="Analyze the research findings...",
    context=[task1],       # Explicitly pass previous task as context
    agent=analyst
)
```

### Agent Ignores Tools
**Symptom**: Agent provides generic answer instead of using available tools.

**Fix**:
- Make tool descriptions more specific
- Include "When to use this tool" in the description
- Reduce the number of tools (3-5 max)

### Agent Hallucinates Tool Output
**Symptom**: Agent claims it used a tool but didn't.

**Fix**:
```python
agent = Agent(
    role="Researcher",
    verbose=True,          # See actual tool calls in output
    allow_delegation=False # Prevent agent from delegating to imaginary agents
)
```

## Task Issues

### Task Never Completes
**Symptom**: Task runs indefinitely.

**Fix**:
```python
task = Task(
    description="Research topic",
    expected_output="A 500-word summary",  # Be specific about expected output
    agent=researcher,
)
```

### Wrong Output Format
**Symptom**: Task produces output in unexpected format.

**Fix**: Be explicit in `expected_output`:
```python
expected_output="A JSON object with keys: 'summary', 'key_points' (list), 'sources' (list)"
```

### Task Context is Stale
**Symptom**: Task uses outdated information.

**Fix**:
```python
crew = Crew(
    agents=[agent],
    tasks=[task],
    cache=False,           # Disable cache if data changes frequently
    memory=True,           # But keep memory for conversation context
)
```

## Memory Issues

### Memory Storage Error
**Symptom**: `sqlite3.OperationalError` or `PermissionError`.

**Fix**:
```python
import os
os.environ["CREWAI_STORAGE_DIR"] = "./my_storage"  # Custom storage path
```

Or set explicitly:
```python
crew = Crew(
    agents=[...],
    tasks=[...],
    memory=True,
    embedder={
        "provider": "openai",
        "config": {"model": "text-embedding-3-small"}
    }
)
```

### Memory Bloat
**Symptom**: Memory grows too large, slowing down crew.

**Fix**:
```python
# Use ephemeral storage for development
os.environ["CREWAI_STORAGE_DIR"] = tempfile.mkdtemp()

# Or manually clear cache
import shutil
shutil.rmtree("./my_storage/chroma_db", ignore_errors=True)
```

## API Issues

### Rate Limiting
**Symptom**: HTTP 429 errors from LLM provider.

**Fix**:
```python
agent = Agent(
    role="Researcher",
    max_rpm=10,            # Limit requests per minute
)

crew = Crew(
    agents=[agent],
    tasks=[task],
    max_rpm=10,            # Crew-level rate limit
)
```

### OpenAI Authentication
**Symptom**: `AuthenticationError`.

**Fix**: Set `OPENAI_API_KEY` environment variable before running crew.

### Token Limit Exceeded
**Symptom**: Task fails with context length error.

**Fix**:
```python
task = Task(
    description="Summarize the following: {long_text}",
    expected_output="A brief summary",
    agent=summarizer,
)

# Split long context into chunks before passing to crew
```

## Performance Issues

### Slow Crew Execution
**Fix**:
```python
# 1. Use simpler LLM for simple tasks
simple_agent = Agent(
    role="Classifier",
    llm="gpt-3.5-turbo",  # Faster than gpt-4
)

# 2. Limit tool usage
agent = Agent(
    role="Researcher",
    tools=[search_tool],   # One tool instead of many
    max_iter=5,            # Fewer iterations
)

# 3. Disable verbose in production
crew = Crew(
    agents=[agent],
    tasks=[task],
    verbose=False,          # No debug output
)
```

### High Token Usage
**Fix**:
```python
# Monitor token usage
result = crew.kickoff(inputs={"topic": "AI"})
print(f"Tokens used: {result.token_usage}")
```

## Debugging Checklist

- [ ] Set `verbose=True` on agents, tasks, and crew
- [ ] Check agent `max_iter` isn't too low (< 5)
- [ ] Verify tool descriptions are clear and specific
- [ ] Confirm `context=[task1]` is set where needed
- [ ] Check API keys are set in environment
- [ ] Verify `expected_output` is specific (not vague)
- [ ] Test tools individually before adding to agent
- [ ] Check `CREWAI_STORAGE_DIR` is writable
- [ ] Verify LLM model is available and configured
- [ ] Try with `cache=False` to force fresh execution

## Common Error Messages

| Error | Cause | Fix |
|-------|-------|-----|
| `AuthenticationError` | Missing/wrong API key | Check `OPENAI_API_KEY` |
| `RateLimitError` | Too many requests | Set `max_rpm` |
| `ContextLengthExceededError` | Input too long | Truncate context |
| `ToolError` | Tool failed | Add try/except in tool |
| `ValidationError` | Wrong input format | Check tool args_schema |
| `MemoryError` | Storage issue | Set `CREWAI_STORAGE_DIR` |
