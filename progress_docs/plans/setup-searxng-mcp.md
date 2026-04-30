---
type: setup
name: "SearXNG MCP Setup"
status: active
started: 2026-04-26
completed: null
tools_installed:
  - SearXNG MCP server (partial)
configs_changed:
  - .kilocode/mcp.json
---

# Setup: SearXNG MCP Setup

## Overview

Set up SearXNG as a privacy-respecting meta-search engine with MCP server for agent-driven web research. Partially complete.

## Status

- SearXNG MCP server configured in `.kilocode/mcp.json`
- Docker dependency being evaluated (SearXNG runs in Docker)
- Alternative: hosted SearXNG instance or direct API usage

## Tasks

| # | Task | Status |
|---|------|--------|
| 1 | Evaluate Docker vs hosted SearXNG | ⏳ |
| 2 | Configure MCP server connection | 🔄 Partial |
| 3 | Test search queries from agent | ⏳ |
| 4 | Document in COMMAND_CHEATSHEET.md | ⏳ |

## Notes
- SearXNG provides meta-search across multiple engines without tracking
- Alternative search tools already available: Tavily, Exa
- May reconsider priority given existing search tools
