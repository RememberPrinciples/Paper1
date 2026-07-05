# Report PDF Compilation Rule Update

Date: 2026-07-05

## Request

The user requested that every answer written into `report/` should be automatically compiled to PDF and that this workflow should be recorded in `.ai_rules`.

## Rule Added

`.ai_rules` now requires:

- every Markdown report in `report/` to be compiled into a same-name PDF;
- the local compiler `python report/compile_report_pdf.py <report.md>` to be used by default;
- `pandoc` with `xelatex` to be acceptable if available;
- the task not to be considered complete until PDF generation has been attempted and reported;
- missing PDF tooling to be treated as a blocker unless it can be installed or configured.

## Current Toolchain Check

Checked tools in the current environment:

- `pandoc`: not available
- `xelatex`: not available
- `wkhtmltopdf`: not available
- `weasyprint`: installed successfully through `python -m pip install weasyprint`
- `latex2mathml`: installed successfully through `python -m pip install latex2mathml`
- Python `markdown`: available
- Python `reportlab`: not available

## Status

The rule has been written and a local compiler has been added:

```bash
python report/compile_report_pdf.py
```

All current Markdown reports in `report/` have been compiled to same-name PDFs:

- `README.pdf`
- `mab_model_algorithm_review.pdf`
- `pdf_compile_rule_update.pdf`
- `tpot_cost_reward_update.pdf`
