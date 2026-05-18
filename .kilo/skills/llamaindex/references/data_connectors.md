# LlamaIndex Data Connectors Guide

> **Load when**: loading documents from files, web, databases, APIs, or creating custom loaders.

## SimpleDirectoryReader (Most Common)

```python
from llama_index.core import SimpleDirectoryReader

# Load all files in directory
documents = SimpleDirectoryReader("data/").load_data()

# Load specific file types
documents = SimpleDirectoryReader(
    "data/",
    recursive=True,                          # Recursive search
    required_exts=[".pdf", ".txt", ".md"],   # Only these types
    exclude=["data/excluded/"],              # Skip directory
).load_data()

# Load individual files
documents = SimpleDirectoryReader(
    input_files=["data/report.pdf", "data/notes.txt"]
).load_data()
```

## File-Specific Readers

### PDF
```python
from llama_index.readers.file import PDFReader

reader = PDFReader()
documents = reader.load_data(file="paper.pdf")
```

### Markdown
```python
from llama_index.readers.file import MarkdownReader

reader = MarkdownReader()
documents = reader.load_data(file="README.md")
```

### Docx
```python
from llama_index.readers.file import DocxReader

reader = DocxReader()
documents = reader.load_data(file="report.docx")
```

### CSV
```python
from llama_index.readers.file import CSVReader

reader = CSVReader(concat_rows=False)  # One document per row
documents = reader.load_data(file="data.csv")
```

### HTML
```python
from llama_index.readers.file import HTMLTagReader

reader = HTMLTagReader(tag="article")
documents = reader.load_data(file="page.html")
```

### Unstructured (Catch-all)
```python
from llama_index.readers.file import UnstructuredReader

reader = UnstructuredReader()
documents = reader.load_data(file="document.xyz")  # Supports many formats
```

## Web Readers

### Simple Web Page
```python
from llama_index.readers.web import SimpleWebPageReader

reader = SimpleWebPageReader(html_to_text=True)
documents = reader.load_data(urls=["https://example.com"])
```

### BeautifulSoup (More Control)
```python
from llama_index.readers.web import BeautifulSoupWebReader

reader = BeautifulSoupWebReader()
documents = reader.load_data(urls=["https://docs.python.org/3/"])
```

### Sitemap
```python
from llama_index.readers.web import SitemapReader

reader = SitemapReader()
documents = reader.load_data("https://example.com/sitemap.xml")
```

### RSS Feed
```python
from llama_index.readers.web import RSSNewsReader

reader = RSSNewsReader()
documents = reader.load_data(urls=["https://example.com/feed.xml"])
```

## Database Readers

### SQL
```python
from llama_index.readers.database import DatabaseReader

reader = DatabaseReader(
    sql_database_uri="postgresql://user:pass@localhost/db",
)
documents = reader.load_data("SELECT title, content FROM articles WHERE published = true")
```

### MongoDB
```python
from llama_index.readers.mongodb import SimpleMongoReader

reader = SimpleMongoReader(uri="mongodb://localhost:27017")
documents = reader.load_data(
    db_name="my_db",
    collection_name="documents",
    field_names=["title", "content"],
)
```

## API Readers

### JSON
```python
from llama_index.readers.json import JSONReader

reader = JSONReader(levels_back=0)
documents = reader.load_data("data.json")
# Or from URL
documents = reader.load_data("https://api.example.com/data.json")
```

### Notion
```python
from llama_index.readers.notion import NotionPageReader

reader = NotionPageReader(integration_token="secret_...")
documents = reader.load_data(page_ids=["page-id-1", "page-id-2"])
```

### Google Docs
```python
from llama_index.readers.google import GoogleDocsReader

reader = GoogleDocsReader()
documents = reader.load_data(document_ids=["doc-id"])
```

## GitHub Reader

```python
from llama_index.readers.github import GithubRepositoryReader, GithubClient

client = GithubClient(github_token="ghp_...")
reader = GithubRepositoryReader(
    github_client=client,
    owner="owner",
    repo="repo",
    filter_file_extensions=[".py", ".md"],
)
documents = reader.load_data(branch="main")
```

## Discord Reader

```python
from llama_index.readers.discord import DiscordReader

reader = DiscordReader(discord_token="...")
documents = reader.load_data(
    channel_ids=["channel-id"],
    limit=100,
)
```

## YouTube Transcript Reader

```python
from llama_index.readers.youtube_transcript import YoutubeTranscriptReader

reader = YoutubeTranscriptReader()
documents = reader.load_data(ytlinks=["https://youtube.com/watch?v=..."])
```

## Custom Document Loader

```python
from llama_index.core.readers.base import BaseReader
from llama_index.core import Document

class CustomReader(BaseReader):
    def load_data(self, file: str) -> list[Document]:
        with open(file) as f:
            content = f.read()

        # Custom parsing logic
        sections = content.split("\n---\n")

        return [
            Document(
                text=section,
                metadata={"source": file, "section": i}
            )
            for i, section in enumerate(sections)
        ]

reader = CustomReader()
documents = reader.load_data("custom_format.data")
```

## Manual Document Creation

```python
from llama_index.core import Document

doc = Document(
    text="This is the document content.",
    metadata={
        "source": "manual",
        "date": "2025-01-01",
        "author": "John Doe",
    },
    doc_id="unique-id-001",
)

# Create from text
documents = [Document(text=t) for t in ["Doc 1", "Doc 2", "Doc 3"]]
```

## LlamaHub (300+ Connectors)

Available at https://llamahub.ai:

```python
# Install specific reader
# pip install llama-index-readers-{name}

from llama_index.readers.arxiv import ArxivReader
from llama_index.readers.wikipedia import WikipediaReader
from llama_index.readers.slack import SlackReader
from llama_index.readers.twitter import TwitterTweetReader
```

## Metadata Best Practices

```python
documents = SimpleDirectoryReader("data/").load_data()

# Enrich metadata
for doc in documents:
    doc.metadata.update({
        "source_type": Path(doc.metadata["file_path"]).suffix,
        "loaded_at": datetime.now().isoformat(),
        "word_count": len(doc.text.split()),
    })
```

## Common Patterns

### Load + Preprocess
```python
def load_and_clean(directory: str) -> list[Document]:
    documents = SimpleDirectoryReader(directory).load_data()

    clean_docs = []
    for doc in documents:
        # Remove empty docs
        if not doc.text.strip():
            continue
        # Normalize whitespace
        doc.text = " ".join(doc.text.split())
        clean_docs.append(doc)

    return clean_docs
```

### Multi-Source Load
```python
# Load from multiple sources in parallel
web_docs = SimpleWebPageReader().load_data(["https://docs.python.org/3/"])
pdf_docs = PDFReader().load_data("spec.pdf")
code_docs = GithubRepositoryReader(...).load_data(branch="main")

all_docs = web_docs + pdf_docs + code_docs
```

## Common Pitfalls

1. **Forgetting metadata** → Always add source/file metadata for tracing
2. **Encoding issues** → Specify encoding when reading text files
3. **Large files** → Use streaming readers for files > 100MB
4. **API rate limits** → Implement retry logic for web/API readers
5. **Missing dependencies** → Each reader may need specific pip installs
