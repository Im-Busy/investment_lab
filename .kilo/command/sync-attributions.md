---
description: Remove all sources marked for deletion in SOURCE_MANIFEST.md from code and registries.
---

# Sync Attributions

Usage: `/sync-attributions`

## What it does
Reads `SOURCE_MANIFEST.md`, finds all entries marked with `~~strikethrough~~`, and purges them from:
- Code files (removes `Source:` / `Reference:` / `Paper:` / `Based on:` lines)
- `docs/research_logic_map/insight_registry.md` (paper entries)
- `docs/research_logic_map/web_source_registry.md` (web source entries)
- `SOURCE_MANIFEST.md` (strips strikethrough entries)

## When to use
After manually marking sources for deletion in `SOURCE_MANIFEST.md` (wrapping the ID line in double tildes: `~~P1: Some Paper~~`).

## Related
- Agent: `.kilo/agent/sync-attributions.md`
- Manifest: `SOURCE_MANIFEST.md` (private)
- Registries: `docs/research_logic_map/insight_registry.md`, `docs/research_logic_map/web_source_registry.md`
