# Subagent Registry

This directory records the default manager-worker setup for this project.

Project root:

```text
/root/autodl-tmp/PROJECT1_TMC_PAPER
```

Runtime environment:

```text
conda run -n SD_Blackwell ...
```

Sandbox expectation:

```text
workspace-write limited to /root/autodl-tmp/PROJECT1_TMC_PAPER
```

Important behavior:

- The manager must not spawn subagents unless the user explicitly asks for multi-agent, subagent, manager-worker, worker, explorer, verifier, or parallel-agent execution.
- When subagents are authorized, the manager should read this directory first and use the role definitions in `agents.md`.
- Workers must receive disjoint write scopes.
- Workers must not revert edits made by others.
- Explorers are read-only unless explicitly reassigned as workers.
- Verification commands should use `conda run -n SD_Blackwell ...`.

