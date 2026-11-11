# Contributing to FLIP

Documentation improvements, reproducibility fixes, and dataset questions are welcome.
For a new dataset or a change to an established split, please open an issue explaining
the proposed task and evaluation protocol before implementing a large change.

## Reporting a problem

Include the dataset, exact split filename, download source or release version, and a
minimal example. For code problems, include the command, Python and dependency versions,
and traceback. For data problems, include the relevant row identifiers or a small excerpt
that can be redistributed, plus the expected and observed behavior.

## Preparing a pull request

- Explain the problem, the resulting behavior, and how you checked the change.
- Preserve existing train, validation, and test assignments unless the PR explicitly
  proposes a benchmark correction. Describe the effect on published comparisons.
- For dataset changes, document the source publication, processing steps, attribution,
  license, and split rationale. Link large raw data instead of committing it.
- Keep generated results, model weights, private paths, and notebook execution outputs
  out of documentation-only changes.
- For documentation, check relative links and anchors, and run any new executable examples.
  For code changes, include a focused check that demonstrates the fix.

The [data guide](docs/data-guide.md) explains the current research-code limitations.
Please distinguish checks you ran from experiments requiring unavailable data or GPUs.
