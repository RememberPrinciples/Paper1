# Report Directory

This folder replaces the old root-level `current_answer.md` workflow.

Use this folder for:

- modeling and algorithm review reports;
- decision records after major updates;
- issue lists and follow-up plans;
- standalone reports for each substantial AI response or review.

Current files:

- `mab_model_algorithm_review.md`: multi-agent review of the MAB model, reward/regret definition, and modeling fixes.
- `tpot_cost_reward_update.md`: summary of the update that adopts $T/G$ as the cost-type reward while retaining $\alpha$ learning.
- `pdf_compile_rule_update.md`: summary of the report-to-PDF workflow update.
- `compile_report_pdf.py`: local Markdown-to-PDF compiler for report files.

PDF files are generated next to their Markdown sources.
