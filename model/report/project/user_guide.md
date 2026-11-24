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

1. **Choose the material.** The selector has 30 records, including dry and humid scenarios for the same adsorbent and two examples. Check the selected record’s evidence, humidity model and temperature range. Use Import
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

### Compare Purolite A110 and Lewatit

Choose **Purolite A110 (55% RH, 1 h)** to inspect the new resin at 25 °C. For a material comparison, load **Reviewed Lewatit (65% RH, 48 h)** first, then change just the material in the selector and preview again. Loading a different case preset can also change the weather and timing. The completed equal-mass pair used identical case and solver hashes. Purolite collected 2.58 kg in 48 hours and the Lewatit control collected 5.14 kg. If cartridge volume is the design constraint, set the dry charge from each resin’s packed density. [Equal-mass comparison](../../../docs/data/resin-comparison-65rh.csv) · [Equal-volume comparison](../../../docs/data/resin-comparison-fixed-volume.csv)

## What the model calculates

The cycle advances through **adsorption, evacuation, heated desorption, cooling
and repressurization**. At each step, the material law gives equilibrium loading
at the local temperature and gas composition. Rate laws determine how quickly
loading approaches that state. Gas and adsorbed inventories change together,
while heat transfer and adsorption energies change temperatures and pressures.

For electrical heating and the indirect steam jacket, the spatial model resolves heat conduction across the bed with parallel gas paths. Direct steam flows through axial cells in series and can condense in the bed. Neither geometry resolves manifold maldistribution. The lumped companion uses electrical heating and one well-mixed bed state.

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

## Compare heat sources and cycle timing

Under **Materials & simulations**, select **Direct steam · Lewatit (1 h)**, then **Preview exact case** and **Start screening**. Direct injection requires the Young advisory Lewatit law and the spatial engine. This preset selects that material, four axial cells and a 25 °C pump-inlet knockout. The 10 kg/h setting is the maximum steam flow during injection. Actual steam consumption comes from the inlet meter. Open **Regeneration heat source** to change supply pressure, flow, generator energy carrier, transfer/drainage times or the wet sorption plateau. **Indirect steam jacket** heats through the wall and has its own heat-transfer conductance. **Electrical heater** uses the reviewed wall-heated spatial representation. The well-mixed companion supports electrical heating.

For a heat-source comparison on the same axial grid, duplicate the direct-steam case, set the steam flow to zero and enable electric trim heat. Keep the same material, initial state, weather, geometry, timing and resolution. Compare collected CO₂/day, kg of steam per kg of product CO₂, wet purity, generator energy and peak local relative humidity. The first hour starts with CO₂ already on the sorbent. For sustained production, compare complete cycles and check whether inventories repeat. The collected/inlet CO₂ ratio includes changes in the initial inventory. Read the signed change in stored CO₂ beside it. Engineering can price electricity, external heat and makeup water. Set the fraction of steam that needs fresh feedwater from the intended condensate-return arrangement. The numerical repeatability check uses three consecutive fixed-weather comparisons of cycle duration, collected CO₂ and all physical endpoint states, including retained liquid.

Use pressure and cooling setpoints to operate the equipment. Rank physically usable, conserved cases on **collected kg CO₂ per elapsed day**. A missed setpoint alone carries no ranking penalty. A longer cooling phase can improve the next adsorption step while reducing how many cycles fit in a day. Vary one timing or pressure setting per run. Check dry and wet product purity, achieved pressures, temperatures and pump operation against any actual delivery or equipment limits. For hardware still being selected, enter measurements when available. The flow, cooler capacity, leak and dead volume can be varied independently in the spatial setup.

The [direct steam screening results](model_reference.qmd#direct-steam-screening-results) compare the matched electrical control, two steam rates and electric trim. The per-cycle plot shows how the initial CO₂ inventory affects the product meter.

The later 48-hour timing and Pully studies used a separate study runner that admits air once both the wall and hottest resin cell reach 55 °C. It could reuse repeating cycles for fixed-weather steam cases. The one-hour steam preset and the archived two-hour-phase Lewatit preset have their own cooling rules. Loading either preset alone will therefore give different inputs from the timing study. [Timing results](../../../docs/data/adsorption-duration.csv) · [One-pump weather results](../../../docs/data/pully-seasonal-one-pump-summary.csv)

## Estimate the effect of bed aging

Open **Bed aging** in the sidebar. Choose **Number of completed cycles** to compare linear and exponential loss over the first 180 cycles, or **Measured CO₂ capacity** if you have a resin sample. For a sample, divide its CO₂ uptake by fresh uptake under the same test conditions and enter the percentage. A sample taken after cycle 180 can still be used. Estimates cover 68.8–100% retained capacity.

The page reads four completed Lewatit simulations for a 10 kg bed with one vacuum pump, electrical heating and a 25 °C, 65% RH feed. It gives CO₂ collected in 48 hours, the last complete cycle’s daily rate and electricity per kg over the full run. Changing the input updates the estimate on this page. The simulation form runs its selected material law at fixed properties. A steam cycle or a different material needs its own capacity runs. [Aging assumptions](model_reference.qmd#lewatit-bed-aging)

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
