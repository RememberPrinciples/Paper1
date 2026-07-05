#!/usr/bin/env python
"""Compile Markdown reports in this directory to PDF.

The converter supports common Markdown plus inline/display math delimited by
`$...$` and `$$...$$`. Math is converted to MathML before WeasyPrint renders
the PDF.
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

import markdown
from latex2mathml.converter import convert as latex_to_mathml
from weasyprint import HTML


STYLE = """
@page {
  size: A4;
  margin: 22mm 18mm;
}
body {
  font-family: "DejaVu Sans", "Noto Sans CJK SC", sans-serif;
  font-size: 11pt;
  line-height: 1.55;
  color: #111;
}
h1, h2, h3 {
  page-break-after: avoid;
}
h1 {
  font-size: 20pt;
  border-bottom: 1px solid #ddd;
  padding-bottom: 0.25em;
}
h2 {
  font-size: 15pt;
  margin-top: 1.5em;
}
h3 {
  font-size: 12.5pt;
}
code, pre {
  font-family: "DejaVu Sans Mono", monospace;
}
pre {
  background: #f6f8fa;
  border: 1px solid #e5e7eb;
  border-radius: 4px;
  padding: 0.75em;
  overflow-wrap: break-word;
  white-space: pre-wrap;
}
table {
  border-collapse: collapse;
  width: 100%;
  font-size: 9.5pt;
}
th, td {
  border: 1px solid #d0d7de;
  padding: 0.35em 0.5em;
  vertical-align: top;
}
blockquote {
  border-left: 4px solid #d0d7de;
  color: #555;
  margin-left: 0;
  padding-left: 1em;
}
math[display="block"] {
  display: block;
  margin: 0.8em auto;
  text-align: center;
}
"""


FENCE_RE = re.compile(r"(```.*?```)", re.DOTALL)
DISPLAY_MATH_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
INLINE_MATH_RE = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", re.DOTALL)


def _protect_code_fences(text: str) -> tuple[str, list[str]]:
    fences: list[str] = []

    def replace(match: re.Match[str]) -> str:
        fences.append(match.group(1))
        return f"@@CODE_FENCE_{len(fences) - 1}@@"

    return FENCE_RE.sub(replace, text), fences


def _restore_code_fences(text: str, fences: list[str]) -> str:
    for i, fence in enumerate(fences):
        text = text.replace(f"@@CODE_FENCE_{i}@@", fence)
    return text


def _mathml(tex: str, display: str) -> str:
    try:
        return latex_to_mathml(tex.strip(), display=display)
    except Exception as exc:  # Keep the report buildable and show the issue.
        escaped = html.escape(tex)
        return f'<span class="math-error">[math conversion failed: {escaped}]</span>'


def convert_math(text: str) -> str:
    protected, fences = _protect_code_fences(text)

    protected = DISPLAY_MATH_RE.sub(
        lambda m: "\n\n" + _mathml(m.group(1), "block") + "\n\n",
        protected,
    )
    protected = INLINE_MATH_RE.sub(
        lambda m: _mathml(m.group(1), "inline"),
        protected,
    )

    return _restore_code_fences(protected, fences)


def compile_one(md_path: Path) -> Path:
    source = md_path.read_text(encoding="utf-8")
    source = convert_math(source)
    body = markdown.markdown(
        source,
        extensions=["extra", "tables", "fenced_code", "sane_lists"],
        output_format="html5",
    )
    html_doc = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>{STYLE}</style>
</head>
<body>
{body}
</body>
</html>
"""
    pdf_path = md_path.with_suffix(".pdf")
    HTML(string=html_doc, base_url=str(md_path.parent)).write_pdf(str(pdf_path))
    return pdf_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", help="Markdown files to compile")
    args = parser.parse_args()

    report_dir = Path(__file__).resolve().parent
    paths = [Path(p) for p in args.paths] if args.paths else sorted(report_dir.glob("*.md"))

    for path in paths:
        if path.name.startswith("."):
            continue
        pdf = compile_one(path)
        print(f"compiled {path} -> {pdf}")


if __name__ == "__main__":
    main()
