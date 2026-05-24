"""Batch convert PDF papers to markdown using pdfplumber + pymupdf fallback."""

from __future__ import annotations

import sys
from pathlib import Path

import pdfplumber
import pymupdf


def pdf_to_text_pdfplumber(pdf_path: Path) -> str | None:
    """Extract text using pdfplumber. Returns None if extraction fails."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
            if pages:
                return "\n\n".join(pages)
    except Exception:
        pass
    return None


def pdf_to_text_pymupdf(pdf_path: Path) -> str | None:
    """Extract text using PyMuPDF. Returns None if extraction fails."""
    try:
        doc = pymupdf.open(pdf_path)
        pages = []
        for page in doc:
            text = page.get_text()
            if text.strip():
                pages.append(text)
        doc.close()
        if pages:
            return "\n\n".join(pages)
    except Exception:
        pass
    return None


def convert_pdf(pdf_path: Path, out_dir: Path) -> str:
    """Convert a single PDF to markdown. Returns status string."""
    md_name = pdf_path.stem + ".md"
    out_path = out_dir / md_name

    if out_path.exists():
        return "skipped (exists)"

    text = pdf_to_text_pdfplumber(pdf_path)
    if text is None or not text.strip():
        text = pdf_to_text_pymupdf(pdf_path)

    if text is None or not text.strip():
        return "FAILED (no text extracted)"

    # Simple markdown formatting
    lines = []
    lines.append(f"# {pdf_path.stem.replace('_', ' ')}\n")
    lines.append(text)

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return f"ok ({len(text)} chars)"


def main() -> None:
    papers_dir = Path("useful_resources/papers")
    out_dir = Path("useful_resources/papers_md")
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(papers_dir.glob("*.pdf"))
    total = len(pdf_files)
    ok = 0
    skipped = 0
    failed = 0

    for i, pdf_path in enumerate(pdf_files, 1):
        status = convert_pdf(pdf_path, out_dir)
        print(f"[{i:03d}/{total:03d}] {pdf_path.name}: {status}")
        if status.startswith("ok"):
            ok += 1
        elif status.startswith("skipped"):
            skipped += 1
        else:
            failed += 1

    print(f"\nDone: {ok} ok, {skipped} skipped, {failed} failed (out of {total})")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
