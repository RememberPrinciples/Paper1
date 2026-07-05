# Task Routing Guide

Use these defaults when splitting work.

## Codebase Understanding

Use explorers in parallel:

- Explorer A: repository structure, entry points, and dependencies.
- Explorer B: experiment scripts and data/output layout.
- Explorer C: paper/docs alignment with code artifacts.

## Implementation

Use workers only with disjoint write scopes:

- Worker A: one Python module or one experiment family.
- Worker B: documentation or LaTeX files.
- Worker C: tests or verification scripts.

## Verification

Use a verifier when implementation can continue independently or when final validation is broad.

Suggested commands:

```bash
conda run -n SD_Blackwell python -m compileall .
conda run -n SD_Blackwell python -m pytest
git status --short --branch
```

## Git Policy

- Manager owns staging, committing, branch changes, pulling, and pushing.
- Subagents should not push to remote repositories.
- Subagents should not delete branches.
- Subagents should not run destructive cleanup commands unless explicitly assigned and approved.

