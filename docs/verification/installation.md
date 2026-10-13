# Installation documentation audit (2026-10-13)

- Audited direct third-party imports in all distributed Python sources, plus external-tool invocations.
- Confirmed the core wheel builds successfully using pip isolated build (`pip wheel . --no-deps`).
- Confirmed wheel METADATA declares exactly the five pinned runtime dependencies in requirements.txt.
- Checked local installation/documentation links.
- Corrected the ESM feature-generation requirement from fair-esm to transformers, matching the supplied script.
- Added optional figure and baseline dependency groups; these ranges are not recovered paper lockfiles.

This audit does not claim a fresh full dependency installation, fresh GPU feature extraction,
or successful reruns of every optional baseline/figure workflow. The existing recorded
runtime passed eight source/numerical checks and real CPU inference on the four paper
examples; see examples/expected and docs/FOUR_PROTEIN_DEMO.md. No PyPI or Bioconda
publication was performed. See INSTALL.md and DEPENDENCIES.md at the repository root.
