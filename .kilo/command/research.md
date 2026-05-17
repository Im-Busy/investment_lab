---
description: Search for academic papers, reference implementations, and benchmarks
argument-hint: <query or topic>
---

Activate the Researcher agent. Search for: $ARGUMENTS

Follow the search hierarchy:
1. Academic/theoretical → Google Scholar + ArXiv (exa_web_search_exa, tavily_tavily-search)
2. Implementation/code → GitHub search + clone to `useful_resources/useful_repos/`
3. Finance data → database-lookup skill (FRED, World Bank, US Treasury)
4. After finding papers → knowledge graph pipeline: `uv run useful_resources/_knowledge_analysis.py`

Document all findings in:
- `progress_docs/current.md`
- Relevant phase plan in `progress_docs/plans/`
- `useful_resources/useful_repos/README.md` (if repo cloned)

Load `trading-papers` skill if the topic relates to existing project papers.
Load `brainstorming-research-ideas` skill if exploring novel directions.
