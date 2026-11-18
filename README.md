# Carbon Team DAC Cycle Model

A Python model for screening solid adsorbents and testing operating conditions for direct air capture. Vary feed humidity, phase durations, regeneration temperature and equipment settings. The local app offers spatial and lumped cycle simulations, energy accounting, engineering checks and interactive 3D replay.

Start with [Using the DAC model](model/report/project/user_guide.md) for setup, running a case and interpreting the output. The [four-page report](model/report/project/model_and_results.pdf) summarizes the comparison. The [physics reference](model/report/project/model_reference.qmd) derives the material laws, balances and cycle accounting. The [project map](docs/PROJECT_MAP.md#follow-one-calculation) traces the function calls from Preview to either solver, then to exports and Engineering.

## Main result

Lewatit VP OC 1065 is the provisional material for the small prototype. In the common 48-hour humid case with 10 kg of dry shaped material, the model gives **2.570 kg CO₂/day** and **5.47 kWh/kg CO₂**. The matched dry case gives **1.186 kg/day**. Upstream drying and downstream compression are outside that electricity figure.

The review set contains 16 complete exports and two incomplete sensitivities. These are conditional screening calculations. Material ranking, prototype performance, cyclic steady state and mesh independence have not been established for this comparison.

## Open the project

On Windows with 64-bit CPython 3.12 installed:

```powershell
.\INSTALL.cmd
.\START.cmd
```

Open <http://127.0.0.1:8772/>. Use `STOP.cmd` to stop the app. If Windows blocks the background launcher’s process check, use `.\model\.venv\Scripts\python.exe -B model/run.py serve` and stop it with Ctrl+C. Opening pages never starts a simulation: a run requires **Preview**, then **Start**. Reports and saved exports can be read without Python.

On another platform, install `model/requirements-lock.txt` in a Python 3.12 virtual environment, then run `python -B model/run.py serve`. See the [quick start](docs/QUICKSTART.md).

The engineering page uses a selected complete result for geometry, pressure loss, fan power, heating, cooling, vacuum, water and electricity. Enter quotes or assumptions to build its cost worksheet. Prices are blank by default, and each worksheet belongs to its exact result.

## Review and continue

Run `python -B -m pytest -c pytest.ini -q` from this directory in the locked
environment. The [review record](docs/REVIEW_CHECKS.md) lists all checks
and their limits. [Contributing](CONTRIBUTING.md) covers changes to the model.

The full catalogue, regression fixtures and 16 original exports ship together for labmates and public readers. Use the contents of the verified ZIP as the repository root. The [publishing guide](docs/PUBLISHING.md) explains the checked ZIP. Local jobs, environments and private research are excluded. Original software uses [MIT](LICENSE); third-party sources retain their [own terms](THIRD_PARTY_NOTICES.md).

The small prototype will be a new case in this model, using its actual geometry, material properties, wire heating and pump duty. The present full-size case remains the comparison reference.
