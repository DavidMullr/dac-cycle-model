"""Recompute steam-comparison results from exported Lewatit states and meters.

Pass one or more complete spatial result JSON files. Standard output is CSV.
"""
import argparse
import csv
import hashlib
import json
from math import isclose
from pathlib import Path
import sys


CO2_KG_MOL = 0.0440095
WATER_KG_MOL = 0.01801528


def audit(path):
    raw = path.read_bytes()
    result = json.loads(raw)
    case, layout = result["case"], result["state_layout"]
    if (not result["horizon"]["complete"] or result["engine"] != "spatial"
            or result["material_package"]["equilibrium_family"] != "young2021_advisory_wadst_gab"
            or layout["n_storage"] != 2):
        raise ValueError(f"{path}: requires a complete spatial Young advisory Lewatit export")
    cycles = result["cycles"]
    full_mass = case["equipment"]["dry_shaped_mass_kg"]
    cells = layout["n_cells_half"]
    wet = "liquid" in layout and layout["liquid"] is not None

    def inventory(state):
        # Two equal-area lanes, equal dry mass per cell, q ordered as CO2 then water.
        gas = state[slice(*layout["n_gas"])]
        q = state[slice(*layout["q"])]
        totals = [2 * sum(gas[i::4]) for i in range(4)]
        for i in (0, 1):
            totals[i] += full_mass * sum(q[i::2]) / cells
        if wet:
            totals[1] += 2 * sum(state[slice(*layout["liquid"])])
        return totals

    def meter(cycle, key, species=None):
        if not layout.get("extended") and key in ("condensate", "e_source", "e_source_input"):
            return 0.0
        index = layout[key] if species is None else layout[key][0] + species
        return 2 * cycle["state_after"][index]

    def total(key, species=None):
        return sum(meter(cycle, key, species) for cycle in cycles)

    def check(name, actual, expected, atol=1e-8):
        if not isclose(actual, expected, rel_tol=1e-6, abs_tol=atol):
            raise ValueError(f"{path}: {name} differs ({actual:g}, expected {expected:g})")

    species_residuals = [0.0] * 4
    max_energy_relative = 0.0
    for cycle in cycles:
        start, end = inventory(cycle["state_before"]), inventory(cycle["state_after"])
        for i in range(4):
            incoming = meter(cycle, "inlet", i)
            outgoing = meter(cycle, "vent", i) + meter(cycle, "product", i)
            if i == 1:
                outgoing += meter(cycle, "condensate")
                if wet:
                    outgoing += meter(cycle, "bed_drain")
            residual = end[i] - start[i] - incoming + outgoing
            relative = abs(residual) / max(incoming + outgoing, 1e-9)
            species_residuals[i] = max(species_residuals[i], relative)
        balance = cycle["energy_balance"]
        boundary = [meter(cycle, "E_heat"), meter(cycle, "e_source"), -meter(cycle, "E_cool"),
                    -meter(cycle, "E_loss"), meter(cycle, "E_netflow")]
        delta_energy = 2 * balance["delta_inventory_J"]
        scale = max(abs(delta_energy), sum(map(abs, boundary)), 2 * balance["absolute_floor_J"])
        energy_relative = abs(delta_energy - sum(boundary)) / scale
        if energy_relative > balance["tolerance"]:
            raise ValueError(f"{path}: cycle {cycle['cycle']} fails energy balance")
        max_energy_relative = max(max_energy_relative, energy_relative)
    max_mass_relative = max(species_residuals)
    if max_mass_relative > 1e-5 or not result["checks"]["energy_closure_pass"]:
        raise ValueError(f"{path}: failed species or energy balance")

    product = [total("product", i) for i in range(4)]
    collected = product[0] * CO2_KG_MOL
    uptake = (total("inlet", 0) - total("vent", 0)) * CO2_KG_MOL
    initial, final = inventory(cycles[0]["state_before"]), inventory(cycles[-1]["state_after"])
    inventory_change = (final[0] - initial[0]) * CO2_KG_MOL
    check("CO2 uptake minus collection", uptake - collected, inventory_change)
    kwh = total("E_electric_total") / 3.6e6
    generator = total("e_source_input") / 3.6e6
    steam = total("steam_in") * WATER_KG_MOL if wet else None
    metrics = result["metrics"]
    check("product", collected, metrics["collected_CO2_kg"])
    check("electricity", kwh, metrics["electricity_kWh"])
    if "steam_generator_input_kWh" in metrics:
        check("generator input", generator, metrics["steam_generator_input_kWh"])
    if wet:
        check("steam supplied", steam, metrics["steam_in_kg"])
        retained = 2 * sum(cycles[-1]["state_after"][slice(*layout["liquid"])])
        film_change = retained - 2 * sum(cycles[0]["state_before"][slice(*layout["liquid"])])
        check("condensation minus evaporation and bed drainage",
              total("bed_condensed") - total("bed_evaporated") - total("bed_drain"), film_change)
    complete = [cycle for cycle in cycles if cycle["complete"]]
    last = complete[-1] if complete else None
    last_product = [meter(last, "product", i) for i in range(4)] if last else []
    last_kg = last_product[0] * CO2_KG_MOL if last else 0.0
    last_uptake = (meter(last, "inlet", 0) - meter(last, "vent", 0)) * CO2_KG_MOL if last else None
    endpoints = {trace["phase"]: trace for trace in result["traces"]
                 if last and trace["cycle"] == last["cycle"]}
    desorption = endpoints.get("desorption", {})
    cooling = endpoints.get("cooling", {})
    outlet_pressure = desorption.get("axial_pressure_Pa")
    # All water sources contribute to this outlet-composition diagnostic.
    previous_time = crossing_low = crossing_high = None
    for trace in result["traces"]:
        if trace["cycle"] == 1 and trace["phase"] == "desorption" and wet:
            if trace["outlet_water_mole_fraction"] >= 0.5:
                crossing_low = previous_time
                crossing_high = trace["phase_elapsed_s"]
                break
            previous_time = trace["phase_elapsed_s"]
    regen = case.get("regeneration", {"kind": "electrical"})
    # Energy added above the feedwater reference for direct steam, or transferred
    # across the jacket for indirect steam. Include separately metered trim heat.
    steam_heat = generator * regen["generator_efficiency"] if wet else total("e_source") / 3.6e6
    supplied_heat = steam_heat + total("E_heat") / 3.6e6
    inlet_co2 = total("inlet", 0)
    seconds = result["horizon"]["simulated_s"]
    return {
        "case": case["label"], "method": regen["kind"], "hours": seconds / 3600,
        "cells": cells, "rtol": case["solver"]["rtol"], "max_step_s": case["solver"]["max_step_s"],
        "max_steam_kg_h": regen.get("flow_kg_h"), "phase_transfer_s": regen.get("phase_transfer_s"),
        "drain_time_s": regen.get("drain_time_s"), "electric_trim": regen.get("trim_heater"),
        "complete_cycles": len(complete), "cycles_repeat": result["horizon"]["css_established"],
        "collected_CO2_kg": collected, "net_air_uptake_CO2_kg": uptake,
        "apparent_collected_over_inlet_CO2": product[0] / inlet_co2 if inlet_co2 else None,
        "inventory_change_CO2_kg": inventory_change, "collected_CO2_kg_day": collected * 86400 / seconds,
        "last_complete_cycle_s": last["cycle_time_s"] if last else None,
        "last_complete_cycle_CO2_kg_day": last_kg * 86400 / last["cycle_time_s"] if last else None,
        "last_complete_cycle_net_air_uptake_CO2_kg_day": last_uptake * 86400 / last["cycle_time_s"] if last else None,
        "last_complete_cycle_wet_CO2_mole_fraction": last_product[0] / sum(last_product) if sum(last_product) > 0 else None,
        "last_complete_cycle_electricity_kWh_per_kg_CO2": meter(last, "E_electric_total") / 3.6e6 / last_kg if last_kg > 0 else None,
        "last_complete_cycle_steam_kg_per_kg_CO2": meter(last, "steam_in") * WATER_KG_MOL / last_kg if wet and last_kg > 0 else None,
        "last_complete_desorption_outlet_kPa": outlet_pressure[-1] / 1000 if outlet_pressure else None,
        "last_complete_cooling_max_bed_C": cooling["bed_temperature_max_K"] - 273.15 if cooling else None,
        "wet_CO2_mole_fraction": product[0] / sum(product) if sum(product) > 0 else None,
        "dry_CO2_mole_fraction": product[0] / (product[0] + sum(product[2:])) if product[0] + sum(product[2:]) > 0 else None,
        "injected_steam_kg": steam, "steam_kg_per_kg_CO2": steam / collected if wet and collected > 0 else None,
        "bed_condensed_kg": total("bed_condensed") * WATER_KG_MOL if wet else None,
        "bed_evaporated_kg": total("bed_evaporated") * WATER_KG_MOL if wet else None,
        "bed_drain_kg": total("bed_drain") * WATER_KG_MOL if wet else None,
        "retained_bed_liquid_final_kg": retained * WATER_KG_MOL if wet else None,
        "knockout_drain_kg": total("condensate") * WATER_KG_MOL,
        "generator_carrier": regen.get("generator_carrier"), "generator_input_kWh": generator,
        "supplied_heat_kWh": supplied_heat,
        "supplied_heat_kWh_per_kg_CO2": supplied_heat / collected if collected > 0 else None,
        "electricity_kWh": kwh, "electricity_kWh_per_kg_CO2": kwh / collected if collected > 0 else None,
        "sampled_max_water_saturation_ratio": metrics["local_relative_humidity_maximum"],
        "sampled_peak_bed_C": metrics["peak_bed_temperature_K"] - 273.15,
        "first_desorption_outlet_50pct_H2O_lower_s": crossing_low,
        "first_desorption_outlet_50pct_H2O_upper_s": crossing_high,
        "max_species_relative_residual": max_mass_relative,
        **{f"max_{name}_relative_residual": value
           for name, value in zip(("CO2", "H2O", "N2", "O2"), species_residuals)},
        "max_energy_relative_residual": max_energy_relative,
        "job_id": result["job_id"], "source_sha256": result["source_identity"]["release_sha256"],
        "result_sha256": hashlib.sha256(raw).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    rows = [audit(path) for path in parser.parse_args().results]
    writer = csv.DictWriter(sys.stdout, fieldnames=rows[0], lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()
