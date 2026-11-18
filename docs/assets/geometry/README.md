# 3D DAC setup

Open **3D DAC setup** in the app. Orbit by dragging, zoom with the wheel and pan with the right mouse button. Camera and keyboard controls provide alternatives. The complete setup combines saved cycle playback, equipment inspection and synchronized trends.

The default replay uses the reviewed Lewatit case at 65% RH. Play or scrub its last complete cycle to follow total adsorbed CO₂ in grams. Three dry cases remain available for comparison. The viewer reads `docs/data/lewatit_humid_cycle.json`, `docs/data/process_saved_cycles_v1.json` and `docs/data/architecture_cycle_pilot_scenario.json`, using local Three.js assets. The separate cassette and channel presets illustrate possible geometry; they are not the selected prototype.

Use **Add run** for a completed local job or a JSON export. Choose **Cycle in run** to replay its saved bed inventory. Added cases stay in this browser and can be shared by sending the original JSON.

See [Replay interpretation](PROCESS_REPLAY.md) for units, sampling and limits. `spec.js` validates geometry; `viewer.js` draws it; `process-contract.js`, `process-playback.js` and `process-scene.js` handle saved states and playback. The Node tests in `tests/` check geometry bounds, routes, units and immutable replay behavior.
