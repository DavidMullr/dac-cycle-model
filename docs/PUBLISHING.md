# Sharing the model

Use this project directory as the repository root. The surrounding workspace contains private research, team documents and archives.

Run the checks in [Review checks](REVIEW_CHECKS.md), then prepare the release from this root:

```text
python -B tools/release.py manifest
python -B tools/release.py verify
python -B tools/release.py build --output ../carbon-team-dac.zip
```

Use a new output filename. The release tool checks the selected files and source identities, excludes environments and local outputs, and verifies every ZIP member against the manifest. Any later edit requires a new manifest and review.

Extract the ZIP into an empty directory for colleague review. The package includes the app, model, reviewed exports, reports, tests and exact dependency lock. It excludes local simulations, saved quotes, caches and private working records.

Use the same archive for colleague review and for a public repository. All 29 selector entries, their evidence and the native result exports travel together. A GitHub repository can carry original software and source data under different terms. Retain the [source credits](../THIRD_PARTY_NOTICES.md), including the noncommercial terms on specified deposits. The remaining source-specific question concerns the COF-1000 and COF-999 series if they are treated as protected compilations. The source links identify the original workbooks and articles.

CI on a remote host and installation on a second computer still require a run on those systems.
