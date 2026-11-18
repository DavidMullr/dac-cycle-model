"""Independent raw-state audit for constant-heat laws; no engine helpers imported.

Humidity extensions and Young's temperature-dependent potential require their
own energy reconstruction. Their reviewed audits are supplied with the results.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re

import numpy as np


R = 8.31446261815324
MW = np.array([0.0440095, 0.01801528, 0.0280134, 0.0319988])
CP_GAS = np.array([37.5, 34.0, 29.2, 29.4])
TOLERANCE = 1e-6
FLOOR_J = 1.0
ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "model/config/mof_screening/spatial_profile.json"
SOURCE_PROFILES = ROOT / "docs/data/submission/source_profiles"


def _delta(state_before, state_after, layout, key):
    bounds = layout[key]
    start, stop = (
        (int(bounds[0]), int(bounds[1]))
        if isinstance(bounds, (list, tuple))
        else (int(bounds), int(bounds) + 1)
    )
    return sum(
        float(state_after[index]) - float(state_before[index])
        for index in range(start, stop)
    )


def _energy_model(result: dict):
    package = result["material_package"]
    layout = result["state_layout"]
    family = package["equilibrium_family"]
    supported = {
        "independent_langmuir_sips", "activity_shift_reference_isotherm",
        "split_rate_activity_shift_reference_isotherm", "site_resolved_concentration_langmuir",
    }
    if package.get("humidity_candidate") or family not in supported:
        raise ValueError(
            "This audit supports constant-heat material laws without humidity extensions. "
            "Use the material-specific audit supplied with the reviewed result."
        )
    reference = float(package["reference_temperature_K"])
    if family in {
        "independent_langmuir_sips",
        "activity_shift_reference_isotherm",
    }:
        species = np.arange(4)
        cp = np.asarray(package["adsorbed_phase_cp_J_kg_K"]) * MW
        dry = package["water"].get("model") == "dry_only"
        heat = np.array(
            [
                0.0 if key == "water" and dry else package[key]["heat_J_mol"]
                for key in ("co2", "water", "nitrogen", "oxygen")
            ]
        )

        def energies(temperature):
            value = (
                cp * (temperature[:, None] - reference)
                + R * reference
                - heat
            )
            if dry:
                value[:, 1] = 0.0
            return value

    elif family == "split_rate_activity_shift_reference_isotherm":
        species = np.array([0, 0, 1, 2, 3])
        cp = np.asarray(package["adsorbed_phase_cp_J_kg_K"]) * MW[species]
        heat = np.array(
            [
                package["co2"]["heat_J_mol"],
                package["co2"]["heat_J_mol"],
                0.0,
                package["nitrogen"]["heat_J_mol"],
                package["oxygen"]["heat_J_mol"],
            ]
        )

        def energies(temperature):
            value = (
                cp * (temperature[:, None] - reference)
                + R * reference
                - heat
            )
            value[:, 2] = 0.0
            return value

    else:
        species = np.array([0, 0, 1])
        site_u = np.asarray(package["co2"]["site_internal_energy_J_mol"])
        heat_water = package["water"]["sorption_heat_J_mol"]

        def energies(temperature):
            sites = site_u + (CP_GAS[0] - R) * (
                temperature[:, None] - reference
            )
            water = (
                (CP_GAS[1] - R) * (temperature - reference)
                + R * temperature
                - heat_water
            )
            return np.column_stack((sites, water))

    return species, energies


def independent_cycle(cycle: dict, result: dict, heater_half_cp: float) -> dict:
    layout = result["state_layout"]
    package = result["material_package"]
    equipment = result["case"]["equipment"]
    start = np.asarray(cycle["phase_evidence"][0]["state_start"], dtype=float)
    end = np.asarray(cycle["state_after"], dtype=float)
    n = int(layout["n_cells_half"])
    mass = float(equipment["dry_shaped_mass_kg"]) / (2 * n)
    reference = float(package["reference_temperature_K"])
    species, energies = _energy_model(result)

    def inventory(state):
        gas = state[slice(*layout["n_gas"])].reshape(n, 4).sum(axis=0)
        loadings = (
            state[slice(*layout["q"])].reshape(n, len(species)).sum(axis=0)
            * mass
        )
        for index, amount in zip(species, loadings):
            gas[index] += amount
        return gas

    def energy(state):
        bed_temperature = state[slice(*layout["T_bed"])]
        gas_temperature = state[slice(*layout["T_gas"])]
        q = state[slice(*layout["q"])].reshape(n, len(species))
        gas = state[slice(*layout["n_gas"])].reshape(n, 4)
        return float(
            mass * package["solid_cp_J_kg_K"]
            * np.sum(bed_temperature - reference)
            + mass * np.sum(q * energies(bed_temperature))
            + np.sum(
                gas * (CP_GAS - R) * (gas_temperature[:, None] - reference)
            )
            + heater_half_cp * (state[layout["T_h"]] - reference)
            + 0.5 * equipment["wall_heat_capacity_J_K"]
            * (state[layout["T_w"]] - reference)
        )

    delta_inventory = energy(end) - energy(start)
    delta_heat = _delta(start, end, layout, "E_heat")
    delta_cool = _delta(start, end, layout, "E_cool")
    delta_loss = _delta(start, end, layout, "E_loss")
    delta_netflow = _delta(start, end, layout, "E_netflow")
    delivered_source = (
        _delta(start, end, layout, "e_source")
        if "e_source" in layout
        else 0.0
    )
    net_boundary = (
        delta_heat
        + delivered_source
        - delta_cool
        - delta_loss
        + delta_netflow
    )
    gross = (
        abs(delta_heat)
        + abs(delivered_source)
        + abs(delta_cool)
        + abs(delta_loss)
        + abs(delta_netflow)
    )
    residual = delta_inventory - net_boundary
    gross_scale = max(abs(delta_inventory), gross, FLOOR_J)
    net_scale = max(abs(delta_inventory), abs(net_boundary), FLOOR_J)
    return {
        "cycle": cycle["cycle"],
        "delta_inventory_J": delta_inventory,
        "net_boundary_J": net_boundary,
        "residual_J": residual,
        "gross_boundary_exchange_J": gross,
        "gross_scale_J": gross_scale,
        "gross_relative_residual": abs(residual) / gross_scale,
        "gross_pass": abs(residual) / gross_scale <= TOLERANCE,
        "legacy_net_scale_J": net_scale,
        "legacy_net_relative_residual": abs(residual) / net_scale,
        "legacy_net_pass": abs(residual) / net_scale <= TOLERANCE,
        "raw_species_inventory_start_mol": inventory(start).tolist(),
        "raw_species_inventory_end_mol": inventory(end).tolist(),
        "engine_residual_J_reported_only": cycle.get("energy_residual_half_J"),
        "engine_boundary_J_reported_only": cycle.get("energy_boundary_half_J"),
    }


def _source_profile(result: dict, profile_path: Path | None) -> tuple[bytes, bool]:
    """Select recorded profile bytes; an explicit override is always checked as supplied."""
    source_identity = result.get("source_identity") or {}
    hashes = source_identity.get("source_hashes") or {}
    candidates = [digest for name, digest in hashes.items()
                  if name.startswith("config/mof_screening/")
                  and Path(name).name.endswith("spatial_profile.json")]
    expected = candidates[0] if len(candidates) == 1 else None
    valid_digest = isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected)
    profile_bytes = (PROFILE if profile_path is None else profile_path).read_bytes()
    if profile_path is None and valid_digest and hashlib.sha256(profile_bytes).hexdigest() != expected:
        evidence = SOURCE_PROFILES / (expected + ".json")
        if evidence.is_file():
            profile_bytes = evidence.read_bytes()
    verified = bool(valid_digest and hashlib.sha256(profile_bytes).hexdigest() == expected)
    return profile_bytes, verified


def audit_result(result: dict, *, profile_path: Path | None = None) -> dict:
    profile_bytes, profile_verified = _source_profile(result, profile_path)
    heater_half_cp = (
        float(json.loads(profile_bytes)["thermal"]["heater_heat_capacity_J_K"])
        / 2.0
    )
    cycles = [
        independent_cycle(cycle, result, heater_half_cp)
        for cycle in result.get("cycles", [])
        if cycle.get("phase_evidence")
    ]
    return {
        "schema": "tvsa-independent-raw-energy-audit/1",
        "job_id": result.get("job_id"),
        "profile_source_verified": profile_verified,
        "profile_sha256": hashlib.sha256(profile_bytes).hexdigest(),
        "criterion_relative": TOLERANCE,
        "criterion_floor_J": FLOOR_J,
        "cycles": cycles,
        "all_gross_pass": bool(cycles) and all(
            row["gross_pass"] for row in cycles
        ),
        "all_legacy_net_pass": bool(cycles) and all(
            row["legacy_net_pass"] for row in cycles
        ),
        "method": (
            "Raw solid/gas/site/heater/wall energy inventory from phase-start "
            "and cycle-end states; raw ledger meters include delivered source "
            "once. Engine residual and boundary fields are reported but not "
            "used."
        ),
    }


def verify_csv(result: dict, csv_path: Path) -> dict:
    rows = list(csv.DictReader(io.StringIO(csv_path.read_text(encoding="utf-8-sig"))))
    traces = result.get("traces", [])
    mismatches = []
    for row_index, (row, trace) in enumerate(zip(rows, traces), 1):
        for key, value in row.items():
            expected = trace.get("elapsed_s" if key == "time_s" else key)
            matches = (value == "") if expected is None else (
                math.isfinite(float(value))
                and float(value) == expected
                if isinstance(expected, (int, float))
                else value == str(expected)
            )
            if not matches:
                mismatches.append((row_index, key))
    return {
        "row_count_match": len(rows) == len(traces),
        "field_count": sum(len(row) for row in rows),
        "mismatches": mismatches,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    root = args.path.resolve()
    result_paths = (
        sorted(root.glob("attempt_*/RESULT.json"))
        if root.is_dir()
        else [root]
    )
    audits = []
    for result_path in result_paths:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        audit = audit_result(result)
        csv_path = result_path.with_name("RESULT.csv")
        if csv_path.is_file():
            audit["csv_export"] = verify_csv(result, csv_path)
        audits.append(audit)
    payload = {
        "schema": "tvsa-independent-campaign-energy-audit/1",
        "campaign_or_result": str(root),
        "audits": audits,
        "pass": bool(audits) and all(
            audit["all_gross_pass"]
            and audit["profile_source_verified"]
            and audit.get("csv_export", {}).get("row_count_match", True)
            and not audit.get("csv_export", {}).get("mismatches", [])
            for audit in audits
        ),
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
