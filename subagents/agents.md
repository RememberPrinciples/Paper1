# Agent Roles

## Manager

Purpose:

- Owns planning, task decomposition, integration, git state, final review, and final user response.
- Decides which tasks stay local and which tasks can run in parallel.
- Keeps the critical path local when waiting would slow progress.
- Reviews worker output before treating it as complete.

Write scope:

- Any project file, but only after accounting for concurrent worker ownership.

Required skills:

- Git branch and working tree management.
- Task decomposition.
- Conflict prevention.
- Final integration and verification.

## Explorer

Purpose:

- Read-only codebase investigation.
- Answer specific questions about structure, dependencies, risks, or implementation options.
- Produce concise findings with file references.

Write scope:

- None.

Required skills:

- Fast code search with `rg`.
- Reading Python, shell scripts, LaTeX, markdown, and experiment outputs.
- Summarizing actionable findings.

Default command style:

```bash
rg ...
sed -n 'START,ENDp' FILE
git status --short --branch
```

## Worker

Purpose:

- Implement a bounded change in assigned files or modules.
- Keep edits scoped to the assigned ownership area.
- Report changed files and verification results.

Write scope:

- Only files explicitly assigned by the manager.

Required skills:

- Python project editing.
- Experiment script maintenance.
- Documentation and paper-support file updates.
- Focused verification.

Default command style:

```bash
conda run -n SD_Blackwell python ...
```

## Verifier

Purpose:

- Run tests, scripts, import checks, formatting checks, or reproduction commands.
- Inspect outputs and report pass/fail status.
- Avoid modifying files unless explicitly assigned a worker role.

Write scope:

- None by default.

Required skills:

- Environment-aware command execution.
- Failure triage.
- Clear reporting of command, result, and likely cause.

Default command style:

```bash
conda run -n SD_Blackwell python -m pytest ...
conda run -n SD_Blackwell python SCRIPT.py
```

