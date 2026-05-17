# Flows — Event-Driven Orchestration

> **Load when**: building complex stateful workflows with conditional branching, routing, or multi-step agent pipelines.

## When to Use Flows vs Crews

| Feature | Crews | Flows |
|---------|-------|-------|
| Execution | Sequential/Hierarchical | Event-driven |
| State | Blackboard (shared context) | Explicit (Pydantic model) |
| Branching | No | Yes (router decorator) |
| Cycles | No | Yes |
| Human-in-loop | Basic | Advanced |
| Best for | Simple agent pipelines | Complex multi-step workflows |

## Quick Start

```python
from crewai.flow.flow import Flow, listen, start, router
from pydantic import BaseModel

class MyState(BaseModel):
    query: str = ""
    confidence: float = 0.0
    final_answer: str = ""

class ResearchFlow(Flow[MyState]):
    @start()
    def gather_data(self):
        """Entry point — always runs first."""
        print(f"Gathering data for: {self.state.query}")
        return {"raw_data": "collected"}

    @listen(gather_data)
    def analyze(self, raw_data):
        """Runs after gather_data completes."""
        self.state.confidence = 0.85
        return analysis_result

    @router(analyze)
    def decide_next(self):
        """Routes based on state. Returns a method name string."""
        if self.state.confidence > 0.8:
            return "high_confidence"
        return "low_confidence"

    @listen("high_confidence")
    def generate_report(self):
        """Only runs if router returns 'high_confidence'."""
        self.state.final_answer = "Report generated"

    @listen("low_confidence")
    def request_more_data(self):
        """Only runs if router returns 'low_confidence'."""
        print("Need more data, running additional queries")

# Execute
flow = ResearchFlow()
result = flow.kickoff(inputs={"query": "What is quantum computing?"})
```

## Decorator Reference

### `@start()`
Marks the entry point. Always runs first. Only one per flow.

### `@listen(method_name)` or `@listen("condition")`
Triggers this method when the named method/condition completes.

### `@router(method_name)`
Must return a string matching a `@listen` condition. Determines which path to take.

### `@or_listen(method_a, method_b)`
Triggers when ANY of the listed methods complete (logical OR).

### `@and_listen([method_a, method_b])`
Triggers when ALL listed methods complete (logical AND).

## State Management

State is a Pydantic model. Access via `self.state`:

```python
class State(BaseModel):
    # Simple fields
    query: str = ""
    count: int = 0

    # Nested
    config: dict = {}

    # With defaults
    results: list = []

class MyFlow(Flow[State]):
    @start()
    def step_one(self):
        self.state.count += 1
        self.state.results.append("step_one_done")
```

**State is preserved across method calls.** Access `self.state` in any flow method.

## Flows + Crews Hybrid

Flows can invoke Crews for agent-based steps:

```python
class HybridFlow(Flow[MyState]):
    @start()
    def research_phase(self):
        research_crew = Crew(
            agents=[researcher],
            tasks=[research_task],
        )
        result = research_crew.kickoff(inputs={"topic": self.state.query})
        return result

    @listen(research_phase)
    def write_phase(self):
        write_crew = Crew(
            agents=[writer, editor],
            tasks=[write_task, edit_task],
        )
        result = write_crew.kickoff()
        self.state.final_answer = result.raw
```

## Unstructured Flows

For looser orchestration without explicit state:

```python
from crewai.flow import Flow

class SimpleFlow(Flow):
    @start()
    def step1(self):
        return "done"

    @listen(step1)
    def step2(self):
        print("Step 2 executing")
```

## Common Patterns

### Multi-Path Routing
```python
@router(analyze)
def route(self):
    score = self.state.confidence
    if score > 0.9:
        return "publish"
    elif score > 0.7:
        return "review"
    return "retry"
```

### Parallel Execution
```python
@start()
def start_task(self):
    return "started"

@listen(start_task)
def branch_a(self):
    return "a_done"

@listen(start_task)
def branch_b(self):
    return "b_done"

@and_listen([branch_a, branch_b])
def merge(self):
    print("Both branches complete")
```

### Conditional Crew Execution
```python
@listen(gather_data)
def analyze_or_skip(self, data):
    if data.get("skip"):
        return
    return analysis_crew.kickoff(inputs=data)
```

## Common Pitfalls

1. **Router not returning a string** → Must return a method name (str) not a boolean
2. **Missing @listen for router output** → If router returns "foo", you need `@listen("foo")`
3. **State mutation outside flow methods** → Only mutate `self.state` inside flow methods
4. **Large state objects** → Keep state serializable (Pydantic-compatible types only)
5. **Infinite loops** → Avoid cycles that route back to the same method repeatedly
