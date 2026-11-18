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

Read [Using the DAC model](../model/report/project/user_guide.html) for the
workflow, input choices, calculation steps and interpretation of results.
The [project map](PROJECT_MAP.md) explains where those steps live in the code.

The [four-page report](../model/report/project/model_and_results.pdf) and saved
exports can be read without installing Python. [Review checks](REVIEW_CHECKS.md)
lists the verification commands. Node.js is needed for JavaScript tests; Pandoc (also supplied with Quarto)
and the optional PDF dependencies are needed only to rebuild documentation.
