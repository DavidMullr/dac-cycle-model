"""Extract the last complete cycle of the reviewed humid Lewatit export for playback."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "docs/data/submission/results/lewatit_humid.json"
TARGET = ROOT / "docs/data/lewatit_humid_cycle.json"
CO2_KG_PER_MOL = 0.0440095


def build(raw):
    result = json.loads(raw)
    layout = result["state_layout"]
    case = result["case"]
    mass = case["equipment"]["dry_shaped_mass_kg"]
    cells = layout["n_cells_half"]
    adapter = result["material"]["adapter"]
    if (not result["horizon"]["complete"] or result["engine"] != "spatial"
            or adapter["adapter"] != "young2021_advisory_wadst_gab"
            or (adapter["co2_states"], adapter["water_states"], layout["n_storage"]) != (1, 1, 2)):
        raise ValueError("Expected the reviewed spatial Lewatit result with one CO2 and one water site")
    cycle = next(c for c in reversed(result["cycles"]) if c["complete"])
    duration = cycle["cycle_time_s"]
    phases = []
    snapshots = []
    for phase in cycle["phase_evidence"]:
        start, end = phase["state_start"], phase["state_end"]
        # Device energy and outlet meters are twice the stored half-bed meters.
        def energy(key):
            return 2 * (end[layout[key]] - start[layout[key]]) / 3.6e6

        total = energy("E_electric_total")
        heater, pump, blower = energy("E_heat"), energy("E_vac"), energy("E_fan")
        product = layout["product"][0]
        phases.append({
            "phase": phase["phase"], "start_s": phase["start_s"], "end_s": phase["end_s"],
            "collected_CO2_kg": 2 * (end[product] - start[product]) * CO2_KG_PER_MOL,
            "electricity_kWh": {"heater": heater, "pump": pump, "blower": blower, "total": total,
                                "cooling_and_auxiliaries": total - heater - pump - blower},
        })
        states = [(phase["end_s"], phase["phase"], end)]
        if not snapshots:
            states.insert(0, (0, "cycle start", start))
        for time, name, state in states:
            q = state[slice(*layout["q"])]
            gas = state[slice(*layout["n_gas"])]
            mole_fractions = []
            for i in range(0, len(gas), 4):
                amounts = gas[i:i + 4]
                mole_fractions.append([n / sum(amounts) for n in amounts[:2]])
            snapshots.append({
                "time_s": time, "phase": name, "bed_temperature_K": state[slice(*layout["T_bed"])],
                "gas_temperature_K": state[slice(*layout["T_gas"])],
                "solid_CO2_mol_kg": q[::2], "solid_H2O_mol_kg": q[1::2],
                "gas_CO2_mole_fraction": [x[0] for x in mole_fractions],
                "gas_H2O_mole_fraction": [x[1] for x in mole_fractions],
                "solid_CO2_kg": sum(q[::2]) / cells * mass * CO2_KG_PER_MOL,
                "collected_CO2_kg": 2 * state[product] * CO2_KG_PER_MOL,
                "electricity_kWh": 2 * state[layout["E_electric_total"]] / 3.6e6,
            })

    trace = [r for r in result["traces"] if r["cycle"] == cycle["cycle"]]
    samples = [{
        "time_s": r["elapsed_s"] - trace[0]["elapsed_s"], "phase": r["phase"],
        "temperature_K": r["bed_temperature_mean_K"], "temperature_min_K": r["bed_temperature_min_K"],
        "temperature_max_K": r["bed_temperature_max_K"], "pressure_Pa": r["pressure_Pa"],
        "solid_CO2_kg": r["co2_loading_mean_mol_kg"] * mass * CO2_KG_PER_MOL,
    } for r in trace]
    half_mm = case["equipment"]["bed_thickness_m"] * 500
    return {
        "id": "lewatit_humid", "label": "Lewatit VP OC 1065 · 65% RH", "cycle": cycle["cycle"],
        "inventory_sampling": "trace", "duration_s": duration, "complete": cycle["complete"],
        "CSS_qualified": result["horizon"]["css_established"], "geometry": result["geometry"],
        "coordinates": {"axis": "transverse half-thickness", "cells_half": cells,
                        "cell_centers_mm": [(i + 0.5) * half_mm / cells for i in range(cells)],
                        "symmetry": "centre plane to wall; reflected cells show the other half of the bed"},
        "identity": {"original_result": SOURCE, "result_sha256": hashlib.sha256(raw).hexdigest(),
                     "source_sha256": result["source_identity"]["release_sha256"],
                     "case_sha256": result["input_identity"]["case_sha256"],
                     "material_sha256": result["input_identity"]["material_sha256"],
                     "job_id": result["job_id"], "material_id": result["material"]["material_id"],
                     "engine": result["engine"]},
        "case": case, "evidence": result["material"]["evidence"], "metrics": result["metrics"],
        "phase_policy": result["operating_policy"], "numerical_qualification": result["numerical_qualification"],
        "latest_cycle_equivalent_kg_day": snapshots[-1]["collected_CO2_kg"] * 86400 / duration,
        "phases": phases, "snapshots": snapshots, "samples": samples,
    }


if __name__ == "__main__":
    replay = build((ROOT / SOURCE).read_bytes())
    TARGET.write_bytes((json.dumps(replay, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))
