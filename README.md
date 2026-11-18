# Carbon Team DAC Cycle Model

A Python model for screening solid adsorbents and testing operating conditions for direct air capture (DAC). Use the spatial model to follow temperature and loading through the bed thickness, or the lumped model for a well-mixed bed. Both let you vary weather, cycle timing, regeneration conditions and equipment settings.

Start with the [operating guide](https://davidmullr.github.io/dac-cycle-model/user_guide.html) to run a case, or read the [four-page results report (PDF)](https://davidmullr.github.io/dac-cycle-model/model_and_results.pdf) for the findings. You can read all chapters online. For offline reading, open `model/report/project/index.html` in your downloaded copy.

## Run a case

On Windows with 64-bit CPython 3.12 installed, run these commands from the repository root. Run the installer once.

```powershell
.\INSTALL.cmd
.\START.cmd
```

Open <http://127.0.0.1:8772/>. Use `STOP.cmd` to stop the background server. If Windows blocks the launcher’s process check, run `.\model\.venv\Scripts\python.exe -B model/run.py serve` in the foreground and stop it with Ctrl+C. The [quick start](https://davidmullr.github.io/dac-cycle-model/quickstart.html) also covers installation and launch on macOS and Linux.

1. **Inspect a saved case.** In **Results & comparison**, open the humid Lewatit result. **JSON** gives you the complete export. **Engineering** opens a worksheet for that specific result, including its equipment assumptions and any prices you enter. Missing prices stay blank.
2. **Test an operating condition.** In **Materials & simulations**, select **Reviewed Lewatit (65% RH, 48 h)** to load the report’s material and case together. For a shorter first run, use **Load 1-hour coupled-humidity diagnostic**. To compare operating conditions, keep the material fixed and change one input, such as feed humidity or regeneration temperature.
3. **Preview, run and export.** Select **Preview exact case** to check the inputs and warnings, then **Start screening**. Follow the job in **Monitor runs**. Export the completed result as JSON for the case, material, states and meters, or CSV for the sampled traces.
4. **Replay the cycle.** In **3D DAC setup**, **Add run** opens a completed local job or an exported result JSON. Select a cycle and scrub the timeline to see grams of CO₂ adsorbed in the whole bed. The schematic geometry is illustrative. The [replay notes](docs/assets/geometry/PROCESS_REPLAY.md) explain the sampling times and the separate collected-product meter.

## Where to find the explanations and results

| What you want to understand | Where to look |
|---|---|
| Which materials and conditions were compared, and what did the simulations predict? | The [model and results report](https://davidmullr.github.io/dac-cycle-model/) gives the common case, baseline results and humidity and regeneration sensitivities. [Quarto source](model/report/project/index.qmd). |
| What physics and numerics does the model use? | The [physics reference](https://davidmullr.github.io/dac-cycle-model/model_reference.html) derives the material laws, species and energy balances, phase switches and reporting units. It also explains the solver controls and walks through a saved export. [Quarto source](model/report/project/model_reference.qmd). |
| What calls what when I press Preview or Start? | [Follow one calculation](https://davidmullr.github.io/dac-cycle-model/project_map.html#follow-one-calculation) traces the form through validation, material selection, the spatial or lumped solver, monitoring, export and Engineering. It names the functions and files at each step. |
| Which claims have evidence, and what still needs measuring? | The [evidence chapter](https://davidmullr.github.io/dac-cycle-model/evidence.html) distinguishes material measurements, assumptions and numerical checks. [Prototype data needed](https://davidmullr.github.io/dac-cycle-model/DATA_NEEDED.html) lists the geometry, material and equipment measurements needed to model the small device. |
| Where are the inputs and their sources? | The [material packages](model/materials/screening/) contain equilibrium and kinetic parameters, units and provenance. [Case defaults and presets](model/config/mof_screening/defaults.json) define the operating inputs. See the [bibliography](model/report/references.bib) for the publications and [source notices](THIRD_PARTY_NOTICES.md) for reuse terms. |
| Which files support a reported number? | Start with the [comparison table (CSV)](docs/data/submission/results.csv) or [result index (JSON)](docs/data/submission/results.json). Follow a completed row to its [native export](docs/data/submission/results/) for the exact inputs, phase records and integrated meters. The index also records why two sensitivities stopped early. [Per-case audits](docs/data/submission/audits/) accompany the exports. |
| How do I check a result or change the project? | [Review checks](https://davidmullr.github.io/dac-cycle-model/REVIEW_CHECKS.html) gives the commands and what each verifies. [Contributing](CONTRIBUTING.md) covers scientific changes, [maintenance](docs/MAINTENANCE.md) covers rebuilding the documentation, and [packaging](docs/PUBLISHING.md) explains the release manifest and ZIP. |

## What the reference case shows

Lewatit VP OC 1065 is the provisional material for the small prototype. In the common 48-hour humid case with 10 kg of dry shaped material, the model gives **2.570 kg CO₂/day** and **5.47 kWh/kg CO₂**. The matched dry case gives **1.186 kg CO₂/day**. The electricity figure excludes upstream drying and downstream compression.

The review set contains 16 complete exports and two incomplete sensitivities. These screening results do not establish a material ranking, cyclic steady state, mesh independence or prototype performance. To study the small prototype, enter its measured geometry, material properties, wire heating and pump duty as a separate operating case.

Local runs and saved engineering worksheets are stored under `model/outputs/` and excluded from Git. The original software uses the [MIT licence](LICENSE). Third-party data and assets retain the terms in [Sources and reuse](THIRD_PARTY_NOTICES.md).
