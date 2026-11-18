# Reading the saved process replay

The complete setup shows the bed, adsorption inlet and exhaust, regeneration branch, heater, shared cooling, inlet cooler, knockout, vacuum pump and collection interface. Dimensions, valves and piping are illustrative. It is not a manufacturing design or a qualified pump installation.

The default replay is cycle 10 of the reviewed 48-hour Lewatit run at 65% RH. The step plot and the inventory readout follow 125 saved trace rows. Each mass in grams equals the mean CO₂ loading in mol/kg times 10 kg of dry shaped adsorbent times 44.0095 g/mol. The inventory is already on the full-bed basis. At intermediate clock positions the readout holds the last sample and shows its timestamp. **Bed CO₂ inventory CSV** includes the original cycle-relative times and unrounded masses. Collected CO₂ uses a separate outlet meter.

Use **Add run** to open a completed local run or a native screening result JSON. The **Cycle in run** selector includes the final partial cycle if the simulation reached its horizon partway through a cycle. Spatial inventory uses the full dry sorbent mass and the sum of CO₂ sites per cell, averaged across the bed. The reader checks saved phase-end cell states to identify the basis of older multi-site traces. Lumped inventory uses the initial and final solid states and the completed phase endpoints. Bed colour represents the whole-bed mean for added runs. The product and electricity meters show full-run totals. The inventory CSV contains the selected cycle.

The three dry replay scenarios use saved samples from `docs/data/process_saved_cycles_v1.json`. For these dry cases the solid CO₂ inventory is available only at the six saved boundaries. Spatial temperature, loading and gas composition hold the last saved phase endpoint, whose timestamp is displayed. The transverse half-bed is mirrored for visualization, with thickness enlarged for inspection; unresolved directions do not acquire a simulated field.

N16 means 16 independent cells across the 8 mm half-bed. Gas inventory is mol per cell; solid storage is mol/kg. Half-device cumulative meters are doubled once. The 48-hour cold-start averages and latest-cycle daily equivalents have different denominators and remain separate.

In the bundled replays, only desorption and cooling feed collection. Added spatial runs follow the exported purge duration and cooling route. Evacuation vents separately, and repressurization uses passive ambient fill. Moving markers indicate direction, not measured flow speed. Pumped phases can have zero throttled withdrawal despite motor input. Cooling is one shared 2 kW budget with conditioner priority and COP 3.

These saved calculations do not establish cyclic steady state, mesh independence or hardware accuracy. The 65% RH feed and hot regeneration extend beyond the coupled Lewatit measurements. HOB retains its failed hot representation. Use the exact reviewed exports for the current comparison.

`tests/test_process_contract.js` checks routes, passive filling, held samples, unit conversions and immutability. `tests/test_geometry_spec.js` checks geometry bounds and the local renderer assets. Original result exports and earlier source comparisons remain in the separate workspace archive.
