# Docs

This directory contains the maintained LaTeX source for the system model and
algorithm note.

## Files

- `System_Model_and_Algorithm.tex`: main LaTeX source.
- `System_Model_and_Algorithm.pdf`: compiled PDF artifact.
- `derivation.md`: chronological modeling and maintenance notes.
- `Makefile`: reproducible local build entry.

## Build

From the repository root:

```bash
make -C docs
```

The build prefers `latexmk -xelatex` and falls back to two `xelatex` passes.
The source uses `ctexart`, so the TeX installation must include Chinese CTeX
support plus the standard math, algorithm, hyperref, and cleveref packages.

To remove temporary LaTeX artifacts:

```bash
make -C docs clean
```
