# Quick start

Use a 64-bit CPython 3.12 virtual environment from the project root:

```text
python -m pip install -r model/requirements-lock.txt
python -B model/run.py serve
```

On Windows, `INSTALL.cmd` and `START.cmd` provide the same setup and launch
workflow; `STOP.cmd` stops the app. Open <http://127.0.0.1:8772/>.

Start with **Results & comparison** to inspect the reviewed results. For a new
calculation, use **Materials & simulations**, then **Preview exact case** and
**Start screening**. Opening the app or previewing inputs starts no simulation.

For **Purolite A110**, select **Purolite A110 (55% RH, 1 h)**. For **direct steam**, select **Direct steam · Lewatit (1 h)** and use the spatial engine. Both presets load a material and a case. Keep the case fixed when comparing resins.

Open **Bed aging** to see how measured capacity loss affects the one-pump electrical Lewatit cycle. Enter retained CO₂ capacity or try a count up to the published 180-cycle endpoint. The page shows the first 48 hours of collection alongside the last complete cycle’s rate.

Read [Using the DAC model](../model/report/project/user_guide.html) for the
workflow, input choices, calculation steps and interpretation of results.
The [project map](PROJECT_MAP.md) explains where those steps live in the code.

The [four-page report](../model/report/project/model_and_results.pdf) and saved
exports can be read without installing Python. [Review checks](REVIEW_CHECKS.md)
lists the verification commands. Node.js is needed for JavaScript tests; Pandoc (also supplied with Quarto)
and the optional PDF dependencies are needed only to rebuild documentation.
