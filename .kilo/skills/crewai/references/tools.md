# Tools Guide — Built-in, Custom, and MCP

> **Load when**: adding tools to agents, creating custom tools, or integrating MCP tools.

## Built-in Tools (50+)

Install with: `pip install 'crewai[tools]'`

### Search & Web

```python
from crewai_tools import (
    SerperDevTool,        # Web search (requires SERPER_API_KEY)
    ScrapeWebsiteTool,    # Extract web page content
    WebsiteSearchTool,    # Semantic search on a website
)
```

### Files & Documents

```python
from crewai_tools import (
    FileReadTool,         # Read file contents
    PDFSearchTool,        # Search inside PDFs
    DirectoryReadTool,    # List directory contents
    DOCXSearchTool,       # Search in .docx files
    TXTSearchTool,        # Search in .txt files
    JSONSearchTool,       # Search in JSON files
    CSVSearchTool,        # Search in CSV files
    XMLSearchTool,        # Search in XML files
    MDXSearchTool,        # Search in .mdx files
)
```

### Code & Documentation

```python
from crewai_tools import (
    CodeDocsSearchTool,         # Search code documentation
    CodeInterpreterTool,        # Execute Python code
    GithubSearchTool,           # Search GitHub repos
    YoutubeVideoSearchTool,     # Search YouTube video content
    YoutubeChannelSearchTool,   # Search YouTube channels
)
```

### Databases

```python
from crewai_tools import (
    MySQLSearchTool,      # Query MySQL
    PGSearchTool,         # Query PostgreSQL
    SnowflakeSearchTool,  # Query Snowflake
    SQLiteSearchTool,     # Query SQLite
)
```

### AI Services

```python
from crewai_tools import (
    DallETool,            # Generate images (DALL-E)
    VisionTool,           # Analyze images (GPT-4V)
    NL2SQLTool,           # Natural language to SQL
    EXASearchTool,        # Exa AI search
)
```

### Tool Configuration

Most built-in tools accept API keys:

```python
search_tool = SerperDevTool(api_key=os.getenv("SERPER_API_KEY"))
scrape_tool = ScrapeWebsiteTool()
file_tool = FileReadTool(file_path="data/report.txt")
pdf_tool = PDFSearchTool(pdf="docs/manual.pdf")
```

## Custom Tools

### Simple Function Tool

```python
from crewai.tools import tool

@tool("Calculator")
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression.
    Args:
        expression: A valid Python arithmetic expression (e.g., '2 + 2')
    """
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

# Use in agent
agent = Agent(
    role="Analyst",
    tools=[calculator],
)
```

### Class-Based Tool

```python
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

class WeatherInput(BaseModel):
    city: str = Field(description="City name")
    country: str = Field(default="US", description="Country code")

class WeatherTool(BaseTool):
    name: str = "Weather Lookup"
    description: str = "Get current weather for a city"
    args_schema: type[BaseModel] = WeatherInput

    def _run(self, city: str, country: str = "US") -> str:
        # Your API call here
        return f"Weather in {city}, {country}: Sunny, 72°F"
```

### Tool with API Calls

```python
import requests
from crewai.tools import BaseTool

class StockPriceTool(BaseTool):
    name: str = "Stock Price"
    description: str = "Get current stock price for a ticker symbol"
    api_key: str = ""

    def _run(self, ticker: str) -> str:
        url = f"https://api.example.com/stock/{ticker}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.get(url, headers=headers)
        data = response.json()
        return f"{ticker}: ${data['price']}"
```

## Tool Best Practices

### Write Good Descriptions
```python
# ✅ Good: Clear, specific description
"Get current weather for a city. Input: city name (e.g., 'San Francisco')"

# ❌ Bad: Vague
"Get weather"
```

### Handle Errors Gracefully
```python
def _run(self, query: str) -> str:
    try:
        result = self.api.search(query)
        return str(result)
    except ConnectionError:
        return "Error: API unavailable. Try again later."
    except Exception as e:
        return f"Error: {str(e)}"
```

### Limit Tools Per Agent
```python
# ✅ Good: 3-5 targeted tools
agent = Agent(
    role="Researcher",
    tools=[SerperDevTool(), ScrapeWebsiteTool(), FileReadTool()]
)

# ❌ Bad: Too many tools confuse the agent
agent = Agent(tools=[tool1, tool2, tool3, tool4, tool5, tool6, tool7, tool8, ...])
```

### Tool Caching
```python
crew = Crew(
    agents=[agent],
    tasks=[task],
    cache=True,  # Cache tool results across runs
)
```

## MCP (Model Context Protocol) Integration

CrewAI supports MCP tools:

```python
# MCP tools are auto-discovered if MCP server is running
# No special code needed — just ensure MCP server is configured
```

## Common Pitfalls

1. **Tool description too vague** → Agent won't know when to use it
2. **No error handling** → Tool failure crashes the entire crew
3. **Too many tools** → Agent gets confused, picks wrong tool
4. **Missing API keys** → Tool fails silently; check env vars
5. **Slow tools** → Web scraping/research tools can add latency; set timeouts
