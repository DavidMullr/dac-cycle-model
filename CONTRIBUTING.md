# Contributing

Use the locked Python environment and run checks from the repository root. Start with the [project map](docs/PROJECT_MAP.md), [equations](model/report/project/model_reference.html) and [review checks](docs/REVIEW_CHECKS.md).

Keep scientific changes small enough to assess. State the changed assumption or equation, its units and evidence, and which results need to be rerun. Use a named case for changed geometry or operating inputs. Preserve saved results and their original source identities.

Changes to executable model files invalidate the readiness record. Review the change and its numerical checks before recording a new source identity; do not replace hashes just to remove a warning. The supporting dependency record is `model/config/mof_screening/dependencies.json`; the current execution record is `PREFLIGHT.json` beside it.

Run the Python, JavaScript, result and report checks in the review guide. The live Windows launcher needs a separate start/stop/restart check. For report edits, rebuild the HTML and inspect the output. If the main report changes, also rebuild and inspect its PDF.

In a review, explain the resulting behavior, relevant evidence and remaining limits. A passing software check does not establish measured material or prototype performance.
