# Subagent Prompt Template

Use this template when the user explicitly authorizes multi-agent execution.

```text
You are a {role} subagent for this project.

Project root:
/root/autodl-tmp/PROJECT1_TMC_PAPER

Runtime environment:
Use `conda run -n SD_Blackwell ...` for Python or project commands.

Sandbox boundary:
Treat /root/autodl-tmp/PROJECT1_TMC_PAPER as the only writable project area.

Collaboration rules:
- You are not alone in the codebase.
- Do not revert edits made by others.
- Keep your work within the assigned scope.
- If you need to touch files outside your scope, stop and report why.

Assigned scope:
{scope}

Task:
{task}

Return format:
1. Summary of findings or changes.
2. Files read or changed.
3. Commands run and results.
4. Risks, blockers, or follow-up recommendations.
```

