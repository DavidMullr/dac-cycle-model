# Using the DAC model

## Start with a reviewed result

From the repository root on Windows, run `INSTALL.cmd` once and launch with `START.cmd`. Open <http://127.0.0.1:8772/>. If the background launcher reports that it cannot check process identity, start the foreground server in PowerShell:

```powershell
.\model\.venv\Scripts\python.exe -B model/run.py serve
```

On macOS or Linux, install `model/requirements-lock.txt` in a Python 3.12 virtual environment and run `python -B model/run.py serve`. Stop a foreground server with Ctrl+C. [Installation and launch details](../../../docs/QUICKSTART.md)

Open **Results & comparison** and find the Lewatit VP OC 1065 humid row. **Engineering** opens its worksheet. **JSON** downloads the complete export, including `case`, `material_package`, `traces`, `cycles` and `metrics`. At 10 kg dry shaped mass this case averages 2.570 kg CO₂/day and 5.47 kWh/kg CO₂ over 48 hours. All 16 completed exports include the material and case that produced them. Two further sensitivities stopped before the requested horizon.

## Set up a calculation

Go to **Materials & simulations**. The material package defines adsorption and
water laws, rate coefficients, thermal properties and their evidence. The case
defines the device and how it operates. Keep that distinction when comparing runs.

1. **Choose the material.** The selector has 29 records, including dry and humid scenarios for the same adsorbent and two examples. Check the selected record’s evidence, humidity model and temperature range. Use Import
   JSON or Create material for a custom package. The default hypothetical package
   is an interface example, not the reported Lewatit case.
2. **Choose the representation.** Spatial resolves temperature and loading
   through the bed thickness. Lumped companion treats the bed as well mixed.
   Use Spatial for comparison with the reported results.
3. **Set the operating case.** Give it a useful label, choose weather and horizon,
   then check the equipment, cycle, initial state and collection settings.
4. **Preview exact case.** Review the interpreted inputs, limits and warnings.
   Preview validates the setup but does not integrate the equations. Any input
   edit invalidates it.
5. **Start screening.** Follow progress through **Monitor runs**. **Cancel job** retains
   partial diagnostics; it does not turn an interrupted horizon into a complete
   result. Simultaneous-run limits affect computer use, not model physics.

| Input group | What to check |
|---|---|
| Weather and horizon | Temperature, pressure, humidity, dry CO₂ concentration and elapsed hours; fixed conditions or the supplied Pully series |
| Equipment and geometry | Complete dry shaped mass, thickness, face area, particles, void fraction, airflow, heating and heat transfer |
| Cycle and initial state | Phase durations, regeneration temperature and pressure requests, clean/dry or equilibrated start |
| Collection | Which portions of the cycle count as collected product |
| Numerical resolution | Spatial cells, tolerances and time-step controls; changing these requires a numerical comparison |

To reproduce the inputs shown in the report, select **Reviewed Lewatit (65% RH, 48 h)** or **Reviewed Lewatit (dry, 48 h)**. Either button fills both the material and the case. Use **Load 1-hour coupled-humidity diagnostic** for a short first run. The exact
`material_package` and `case` are embedded in each result JSON, together with its
source and numerical-policy identities.

## What the model calculates

The cycle advances through **adsorption, evacuation, heated desorption, cooling
and repressurization**. At each step, the material law gives equilibrium loading
at the local temperature and gas composition. Rate laws determine how quickly
loading approaches that state. Gas and adsorbed inventories change together,
while heat transfer and adsorption energies change temperatures and pressures.

The spatial model divides a symmetric half-bed into cells. It resolves heat
conduction through the thickness with parallel gas paths, plus heater and wall
states. It does not resolve an axial breakthrough front or manifold flow
maldistribution. The lumped model uses the same material interface and case
inputs with a well-mixed transport description.

Pump withdrawal, pressure control, heating and the shared cooling budget act
through the cycle. An ideal inlet cooler and knockout account for water removal
before the pump. Product, water and electricity meters accumulate over the
requested horizon. Requested temperature and pressure are control targets;
inspect the achieved values. The [technical reference](model_reference.qmd)
gives the equations and boundaries.

## Read the result

Check horizon completion first, then the material-domain, conservation and
numerical diagnostics. A solver failure is not evidence that a material performs
poorly. A completed run can still miss operating targets or use uncertain inputs.

Read collected CO₂/day together with kWh/kg CO₂, the electricity components,
achieved temperatures and pressures, and water behavior. Wet and dry product
purities use different denominators. The reported daily rate is a finite-horizon
average; the 48-hour comparison includes the clean-start transient and collection
from the final partial cycle. It does not establish cyclic steady state or mesh
independence.

For comparisons, keep mass, weather, horizon, initial state and operating policy
consistent. Differences in material evidence still limit the conclusions. The
modeled electricity excludes upstream drying and downstream compression.
Under **Monitor runs**, open the completed local job. **Export JSON** includes the exact inputs and full accounting. **Export CSV traces** gives sampled observations. For the reviewed comparison, use its [summary CSV](../../../docs/data/submission/results.csv) and the individual JSON exports on the Results page.

Before plotting saved COF-1000 or CALF-20 loadings, or interpreting the saved Lewatit local humidity traces, see [Reading saved diagnostics](model_reference.qmd#reading-saved-diagnostics).

## Engineering and the next prototype

**Engineering & costs** reads one complete result and derives geometry, pressure
loss, heating, cooling, vacuum, water and electricity duties. It does not run
another cycle or select hardware. Enter prices with a quote or assumption;
missing prices remain unknown. **Update calculation** previews the worksheet, **Save worksheet** keeps it
for that exact result, and JSON/CSV exports include its assumptions.

**3D DAC setup** opens cycle 10 of the reviewed Lewatit case at 65% RH. Select **Play cycle** or drag the time slider. **CO₂ adsorbed in the bed** shows the total mass on the 10 kg of dry sorbent at the displayed sample time. Click its graph or use the arrow keys on the graph to inspect individual samples. **Collected in this cycle** is the cumulative product at the last saved phase boundary. Three dry replays are also available. Equipment geometry is illustrative.

**To replay your own run**, open **3D DAC setup → Add run**. Choose a completed local run and select **Open run**, or use the file chooser to open a colleague’s native result JSON from **Materials & simulations**. Added runs appear under **Saved case** in this browser. Select **Cycle in run** to inspect any cycle, including a partial final cycle. **Remove from replay list** removes the browser copy. Share the original result JSON to give another colleague the same replay.

For added runs, bed colour shows mean solid CO₂ loading. Spatial runs show inventory at the saved trace times. Lumped runs show the initial state, completed phase endpoints and the final state. The readout holds its last recorded value between samples. Collected CO₂ and electricity are totals for the entire exported run.

The small prototype should be a named case in this model, using its measured
geometry and material/component properties. Keep the full-size reference for
comparison. The saved peak near 120°C does not establish a safe operating limit.
Use [Data needed](../../../docs/DATA_NEEDED.md) to prepare the case,
[Project map](../../../docs/PROJECT_MAP.md) to follow the code, and
[Review checks](../../../docs/REVIEW_CHECKS.md) when changing it.
