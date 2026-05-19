# Balanced natural-prefix draft entropy experiment

- Design: balanced natural-prefix; no random middle-window truncation
- Context lengths: [64, 256]
- Samples per type: 5
- Source types: natural_language, chat, code, math, json_config

## Main figures

![overall](overall_entropy_acceptance_by_context.png)

![controls](detailed_controls_ctx64.png)

![per-source ctx 64](per_source_exact_accept_ctx64.png)

![per-source ctx 256](per_source_exact_accept_ctx256.png)
