# Review checks

The standalone review used 64-bit CPython 3.12.3 with the 17 pinned packages. Run from the repository root:

```text
python -B -m pytest -c pytest.ini -q
python -B tools/check_javascript.py
python -B tools/check_submission.py
python -B tools/check_reports.py
python -B tools/release.py verify
```

| Evidence | What was checked |
|---|---|
| Python and JavaScript | Material laws, phase derivatives, mass and energy accounting, input validation, worker state, result export, worksheets and replay |
| Saved comparison | All 16 complete exports, independent half-bed product and electrical meter sums, source identities, summaries, and the two incomplete sensitivity entries |
| Bounded execution | One hour of simulated time on each engine through preview, worker, monitoring, JSON and CSV export, and the engineering endpoint |
| Reports | Four report pages, four supporting guides, local links, anchors, equations and the four-page PDF |

The [verification record](data/submission/review_checks.json) records the observed counts. The [technical reference](../model/report/project/model_reference.html#reading-saved-diagnostics) explains the two-site loading correction and the scope of historical humidity diagnostics. Reviewed exports keep their original bytes. The bounded runs did not replace any published case.

The 16-export audit covers the archived comparison. For later runs, use the job and source hashes in the [matched-resin table](data/resin-comparison-65rh.csv) and the export hashes in the [aging reference data](data/bed-aging-lewatit.json). The [timing study](data/adsorption-duration.csv) records how many cycles were integrated or reused. To audit a native Lewatit export with steam, run `python tools/audit_lewatit_regeneration.py RESULT.json` and check net uptake, collected CO₂, both drains and generator input. A110-specific tests exercise the humidity curves, low-pressure extension and rates. The aging page interpolates the four completed cases.

The independent raw-energy helper handles constant-heat laws. Material-specific historical energy audits accompany the corresponding exports. Hosted CI, a second computer, mesh convergence, cyclic steady state, hardware limits and physical performance require separate evidence.
