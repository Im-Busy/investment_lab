"""Re-extract PDFs to markdown using markitdown, then merge with existing .md files.

Merge strategy:
  - markitdown produces raw text extraction (more complete text)
  - Old .md files may contain graph/chart descriptions and better formatting
  - Keep raw text as base, inject old file content that's missing from new extraction:
      * Sections mentioning figures, charts, graphs, patterns, diagrams
      * Any paragraph/section present in old but absent from new
"""

from __future__ import annotations

import difflib
import re
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDFS_FOUND_DIR = PROJECT_ROOT / "useful_resources" / "PDFs_Found_Online"
PAPERS_DIR = PROJECT_ROOT / "useful_resources" / "papers"
PAPERS_MD_DIR = PROJECT_ROOT / "useful_resources" / "papers_md"
TEMP_DIR = Path("C:/Users/Joey Chiu/AppData/Local/Temp/kilo/pdf_re_extract")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Keywords indicating graph/chart/visual content that markitdown can't extract
VISUAL_KEYWORDS = re.compile(
    r"(figure|chart|graph|diagram|illustration|pattern.*(?:formation|detection|description)|"
    r"price.*action|visual.*(?:representation|description|depiction)|"
    r"candlestick|bar.*chart|line.*chart|plot|drawing|screenshot|image)",
    re.IGNORECASE,
)

# Keywords for structured content worth preserving from old files
STRUCTURAL_PATTERNS = [
    r"^#{1,6}\s",  # headings
    r"^\|.*\|.*\|",  # tables
    r"^\s*[-*+]\s",  # bullet lists
    r"^\d+\.\s",  # numbered lists
    r"^```",  # code blocks
    r"^>\s",  # blockquotes
]


def run_markitdown(pdf_path: Path, output_path: Path) -> bool:
    """Extract text from a PDF using markitdown."""
    result = subprocess.run(
        ["uv", "run", "markitdown", str(pdf_path), "-o", str(output_path)],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode != 0:
        print(f"  ERROR: markitdown failed for {pdf_path.name}: {result.stderr[:200]}")
        return False
    if not output_path.exists() or output_path.stat().st_size == 0:
        print(f"  ERROR: markitdown produced empty output for {pdf_path.name}")
        return False
    return True


def normalize_text(text: str) -> str:
    """Normalize text for comparison: collapse whitespace, lowercase."""
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def extract_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs (blocks separated by blank lines)."""
    blocks = re.split(r"\n\s*\n", text)
    return [b.strip() for b in blocks if b.strip() and len(b.strip()) > 20]


def has_visual_content(paragraph: str) -> bool:
    """Check if a paragraph describes visual/graph content."""
    return bool(VISUAL_KEYWORDS.search(paragraph))


def paragraphs_not_in_target(old_paragraphs: list[str], new_text: str) -> list[str]:
    """Find paragraphs from old file that have no near-match in new text."""
    new_normalized = normalize_text(new_text)
    missing = []
    for para in old_paragraphs:
        para_norm = normalize_text(para)
        # Check if a meaningful chunk of this paragraph exists in new text
        # Use a 100-char sliding window
        if len(para_norm) < 30:
            continue
        chunk = para_norm[: min(120, len(para_norm))]
        if chunk not in new_normalized:
            missing.append(para)
    return missing


RAW_EXTRACTION_MIN_CHARS = 500  # below this, treat as image-only PDF


def merge_files(
    old_md_path: Path,
    new_raw_path: Path,
    output_path: Path,
    pdf_path: Path | None = None,
) -> str:
    """Merge old .md content with new markitdown raw extraction.

    Three quality tiers:
      TIER 1 - Image-only PDF (raw < 500 chars):
        Keep old file entirely as base. Append raw extraction if any.
      TIER 2 - Raw extraction has less text than old file:
        Old file has more structured content. Use old as base, append raw.
      TIER 3 - Raw extraction has more text:
        Text-heavy PDF. Use raw as base, inject old file's unique content.
    """
    old_text = old_md_path.read_text(encoding="utf-8", errors="replace")
    new_text = new_raw_path.read_text(encoding="utf-8", errors="replace")

    new_chars = len(new_text.strip())
    old_chars = len(old_text.strip())

    source_name = pdf_path.stem if pdf_path else old_md_path.stem

    if new_chars < RAW_EXTRACTION_MIN_CHARS:
        # TIER 1: Image-only PDF — keep old file as primary content
        lines = [
            f"# {source_name}",
            "",
            f"> *Source PDF: {source_name}.pdf*",
            "> *Extraction method: Manual/AI extraction from image-heavy PDF (markitdown could not extract text)*",
            "",
            "---",
            "",
            old_text,
        ]
        if new_chars > 0:
            lines.extend(
                [
                    "",
                    "---",
                    "",
                    "## Raw OCR Text (markitdown — limited extraction)",
                    "",
                    new_text,
                ]
            )
        merged = "\n".join(lines)
        output_path.write_text(merged, encoding="utf-8")
        return merged

    if old_chars > new_chars * 1.2:
        # TIER 2: Old file has significantly more content — use old as base
        old_paragraphs = extract_paragraphs(old_text)
        missing_in_new = paragraphs_not_in_target(old_paragraphs, new_text)

        lines = [
            f"# {source_name}",
            "",
            f"> *Source PDF: {source_name}.pdf*",
            "> *Extraction: Combined — previous structured extraction (base) + markitdown raw text*",
            "",
            "---",
            "",
            old_text,
        ]
        if missing_in_new:
            lines.extend(
                [
                    "",
                    "---",
                    "",
                    "## Content Unique to Old Extraction (not found in markitdown output)",
                    "",
                ]
            )
            for i, para in enumerate(missing_in_new[:30], 1):
                lines.append(f"### {i}. {para[:80]}...")
                lines.append("")
                lines.append(para)
                lines.append("")

        lines.extend(
            [
                "",
                "---",
                "",
                "## Raw Markitdown Extraction (full text)",
                "",
                new_text,
            ]
        )
        merged = "\n".join(lines)
        output_path.write_text(merged, encoding="utf-8")
        return merged

    # TIER 3: Raw extraction has more text — use raw as base
    old_paragraphs = extract_paragraphs(old_text)
    missing_in_new = paragraphs_not_in_target(old_paragraphs, new_text)

    visual_missing = [p for p in missing_in_new if has_visual_content(p)]
    other_missing = [p for p in missing_in_new if not has_visual_content(p)]

    lines = [
        f"# {source_name}",
        "",
        f"> *Source PDF: {source_name}.pdf*",
        "> *Extraction: Combined — markitdown raw text (base) + previous extraction supplements*",
        "",
        "---",
        "",
        new_text,
    ]

    if visual_missing or other_missing:
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("## Content from Previous Extraction (not in markitdown output)")
        lines.append("")

        if visual_missing:
            lines.append("### Visual/Chart/Graph Descriptions")
            lines.append("")
            for i, para in enumerate(visual_missing[:30], 1):
                lines.append(f"#### {i}")
                lines.append("")
                lines.append(para)
                lines.append("")

        if other_missing:
            lines.append("### Additional Content")
            lines.append("")
            for i, para in enumerate(other_missing[:20], 1):
                lines.append(f"#### {i}")
                lines.append("")
                lines.append(para)
                lines.append("")

    merged = "\n".join(lines)
    output_path.write_text(merged, encoding="utf-8")
    return merged


def extract_straight(pdf_path: Path, final_output: Path, raw_text: str) -> str:
    """Create markdown from PDF with no old file to merge (straight markitdown)."""
    lines = [
        f"# {pdf_path.stem}",
        "",
        f"> *Source PDF: {pdf_path.name}*",
        "> *Extracted with: markitdown*",
        "",
        "---",
        "",
        raw_text,
    ]
    merged = "\n".join(lines)
    final_output.write_text(merged, encoding="utf-8")
    return merged


# Known filename mismatches: PDF stem -> old .md stem
KNOWN_MISMATCHES: dict[str, str] = {
    "idenitfying-chart-patterns_fidelity": "identifying chart patterns with technical analysis",
    "technical-analysis-price-patterns_ncfe": "technical analysis price patterns",
    "the-ultimate-harmonic-pattern-trading-guides-full-version": "the ultimate harmonic pattern trading guiddes",
}

# PDF stems that should NOT fuzzy-match (typically full books that share prefix with page-range splits)
NO_FUZZY_MATCH: set[str] = {
    "trade-chart-patterns-guide",  # full 293-page book; splits are -1-85, -86-169, etc.
}


def map_pdfs_to_old_md(pdf_dir: Path, md_dir: Path) -> dict[Path, Path | None]:
    """Map each PDF to its existing .md counterpart.

    Returns dict: pdf_path -> old_md_path (or None if no match)
    """
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    md_files = {f.stem.lower(): f for f in md_dir.glob("*.md")}

    mapping: dict[Path, Path | None] = {}
    for pdf in pdfs:
        pdf_stem_lower = pdf.stem.lower()
        # Check known mismatches first
        if pdf_stem_lower in KNOWN_MISMATCHES:
            old_key = KNOWN_MISMATCHES[pdf_stem_lower]
            if old_key in md_files:
                mapping[pdf] = md_files[old_key]
                continue

        # Try exact match
        if pdf_stem_lower in md_files:
            mapping[pdf] = md_files[pdf_stem_lower]
        elif pdf_stem_lower not in NO_FUZZY_MATCH:
            # Try fuzzy match: check if any md file contains the PDF stem
            found = None
            for md_stem, md_path in md_files.items():
                if pdf_stem_lower in md_stem or md_stem in pdf_stem_lower:
                    found = md_path
                    break
            mapping[pdf] = found
        else:
            mapping[pdf] = None

    return mapping


def process_directory(
    pdf_dir: Path,
    md_dir: Path,
    label: str,
    dry_run: bool = False,
) -> list[str]:
    """Process all PDFs in a directory: extract, merge, replace."""
    results: list[str] = []
    mapping = map_pdfs_to_old_md(pdf_dir, md_dir)

    print(f"\n{'=' * 60}")
    print(f"Processing: {label}")
    print(f"  PDF dir: {pdf_dir}")
    print(f"  MD dir:  {md_dir}")
    print(f"  PDFs found: {len(mapping)}")
    print(f"{'=' * 60}")

    for pdf_path, old_md_path in sorted(mapping.items()):
        pdf_basename = pdf_path.stem
        raw_output = TEMP_DIR / f"{pdf_basename}_raw.md"
        final_output = md_dir / f"{pdf_basename}.md"

        print(f"\n--- {pdf_path.name} ---")

        # Step 1: Extract with markitdown
        if not raw_output.exists():
            print("  Extracting with markitdown...")
            if not run_markitdown(pdf_path, raw_output):
                results.append(f"FAILED_EXTRACT: {pdf_path.name}")
                continue
        else:
            print(f"  Using cached extraction: {raw_output.name}")

        new_size = raw_output.stat().st_size
        raw_text = raw_output.read_text(encoding="utf-8", errors="replace")
        new_lines = len(raw_text.splitlines())
        print(f"  Raw extraction: {new_lines} lines, {new_size:,} bytes")

        # Step 2: Determine if there's an old file to merge
        raw_size = len(raw_text.strip())

        if old_md_path and old_md_path.exists():
            old_text = old_md_path.read_text(encoding="utf-8", errors="replace")
            old_size = old_md_path.stat().st_size
            old_lines = len(old_text.splitlines())
            print(f"  Old .md: {old_lines} lines, {old_size:,} bytes")

            if raw_size == 0:
                print("  Image-only PDF (markitdown empty) — keeping old .md as base")
            elif raw_size < old_size:
                print("  Old .md has more content — using old as base, appending raw")
            else:
                print(
                    "  Raw extraction has more content — using raw as base, adding old supplements"
                )

            print("  Merging...")

            if not dry_run:
                merge_files(old_md_path, raw_output, final_output, pdf_path)
                # Delete old .md, but only if it is different from the output file
                if old_md_path.resolve() != final_output.resolve():
                    old_md_path.unlink()
                    print(f"  Deleted old: {old_md_path.name}")
                else:
                    print(f"  Overwrote old: {old_md_path.name}")
        else:
            print("  No old .md found — straight extraction")
            if not dry_run:
                extract_straight(pdf_path, final_output, raw_text)

        final_size = final_output.stat().st_size if final_output.exists() else 0
        final_lines = (
            len(final_output.read_text(encoding="utf-8", errors="replace").splitlines())
            if final_output.exists()
            else 0
        )
        print(f"  Final output: {final_lines} lines, {final_size:,} bytes")
        results.append(f"OK: {pdf_path.name}")

    # Summary
    ok = sum(1 for r in results if r.startswith("OK"))
    failed = sum(1 for r in results if r.startswith("FAILED"))
    print(f"\n  Results: {ok} OK, {failed} FAILED out of {len(results)}")
    return results


def main() -> None:
    dry_run = "--dry-run" in sys.argv

    all_results: list[str] = []

    # Phase 1: PDFs_Found_Online
    results_1 = process_directory(PDFS_FOUND_DIR, PDFS_FOUND_DIR, "PDFs_Found_Online", dry_run)
    all_results.extend(results_1)

    # Phase 2: papers -> papers_md
    results_2 = process_directory(PAPERS_DIR, PAPERS_MD_DIR, "papers -> papers_md", dry_run)
    all_results.extend(results_2)

    # Final summary
    ok = sum(1 for r in all_results if r.startswith("OK"))
    failed = sum(1 for r in all_results if r.startswith("FAILED"))
    print(f"\n{'=' * 60}")
    print(f"FINAL: {ok} OK, {failed} FAILED out of {len(all_results)} total")
    print(f"{'=' * 60}")

    if dry_run:
        print("\n[Dry run — no files modified]")


if __name__ == "__main__":
    main()
