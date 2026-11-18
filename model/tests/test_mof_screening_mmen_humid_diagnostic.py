from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest

from screening.mof_screening.contract import (
    build_case,
    canonical_sha256,
    load_builtin_material,
    load_defaults,
    preview_payload,
)
from screening.mof_screening.manager import MofScreeningManager
from screening.mof_screening.materials import build_material_adapter
from screening.mof_screening.runtime import build_spatial_config
from screening.mof_screening.weather import load_weather_window


MODEL = Path(__file__).resolve().parents[1]
MATERIAL_ID = "mmen_mg2_dobpdc_mcdonald2012_conditional"
MATERIAL_PATH = (
    MODEL
    / "materials/screening/mmen_mg2_dobpdc_mcdonald2012_conditional.json"
)
DRY_ID = "mmen_mcdonald2012_independent_humid_48h_rh0"
HUMID_ID = "mmen_mcdonald2012_independent_humid_48h_rh030"
DRY_CASE_SHA256 = (
    "c246e39357249046e9308bd53ef1478bb85aadda055ba0b806c820180e6bbe90"
)
HUMID_CASE_SHA256 = (
    "e51b8491f932655a09adf9512494765bd46fa6c21cf9246e7c6c80a909292015"
)


def _preset(preset_id: str) -> dict:
    return next(
        row
        for row in load_defaults()["case_presets"]
        if row["id"] == preset_id
    )


def test_existing_mmen_package_is_unchanged_and_physically_humid():
    assert sha256(MATERIAL_PATH.read_bytes()).hexdigest() == (
        "f4a7387ddd13da66fc3be7121a2924b2c785aa7fda415289df4d75c057eca7ac"
    )
    material = load_builtin_material(MATERIAL_ID)
    assert material["equilibrium_family"] == (
        "activity_shift_reference_isotherm"
    )
    assert material["water"]["saturation_mol_kg"] == pytest.approx(
        47.19637392
    )
    assert material["co2_water_penalty_per_mol_kg"] == 0.0
    adapter = build_material_adapter(material)
    assert adapter.storage_labels == ("CO2", "H2O", "N2", "O2")
    report = adapter.source_report()
    assert report["water_mode"] == "unary_sips"
    assert report["water_states"] == 1
    assert report["independent_component_storage"] is True
    assert report["co2_water_competition_enabled"] is False
    assert report["moisture_sorption_heat_included"] is True
    assert report["water_sorption_rate_validated"] is False
    assert report["hot_desorption_rate_validated"] is False


def test_mmen_presets_use_exact_original_48h_cases_and_differ_only_by_rh_label():
    dry = _preset(DRY_ID)
    humid = _preset(HUMID_ID)
    assert dry["material_id"] == MATERIAL_ID
    assert humid["material_id"] == MATERIAL_ID
    assert dry["case_sha256"] == DRY_CASE_SHA256
    assert humid["case_sha256"] == HUMID_CASE_SHA256
    comparable = deepcopy(dry["case"])
    comparable["weather"]["relative_humidity"] = 0.30
    comparable["label"] = humid["case"]["label"]
    assert comparable == humid["case"]


def test_mmen_humidity_changes_water_heat_and_rates_but_not_co2_equilibrium():
    material, case = build_case(
        load_builtin_material(MATERIAL_ID), _preset(HUMID_ID)["case"]
    )
    adapter = build_material_adapter(material)
    weather = load_weather_window(case["weather"]).rows[0]
    config = build_spatial_config(material, case, weather)
    assert config["ambient"]["relative_humidity"] == 0.30
    assert config["ambient"]["water_mole_fraction"] == pytest.approx(
        weather.water_mole_fraction
    )

    temperature_K = 298.15
    p_water = weather.prepared_buck_water_vapour_partial_Pa
    dry = adapter.equilibrium(
        temperature_K, np.array([42.0, 0.0, 79_000.0, 21_000.0])
    )
    wet = adapter.equilibrium(
        temperature_K,
        np.array([42.0, p_water, 79_000.0, 21_000.0]),
    )
    assert dry[0] == pytest.approx(wet[0], rel=0, abs=1e-15)
    assert dry[1] == 0.0
    assert wet[1] > 0.0

    dry_rates, dry_species_rates = adapter.rates(
        temperature_K,
        np.array([42.0, 0.0, 79_000.0, 21_000.0]),
        dry,
    )
    wet_rates, wet_species_rates = adapter.rates(
        temperature_K,
        np.array([42.0, p_water, 79_000.0, 21_000.0]),
        dry,
    )
    assert wet_rates[1] > dry_rates[1]
    assert wet_species_rates[1] > dry_species_rates[1]

    unit_energy = adapter.absolute_energy_J_mol(temperature_K)
    dry_energy = unit_energy @ dry
    wet_energy = unit_energy @ wet
    assert wet_energy != pytest.approx(dry_energy)


def test_mmen_preview_warning_and_export_preserve_independent_component_scope(
    tmp_path,
):
    material, case = build_case(
        load_builtin_material(MATERIAL_ID), _preset(HUMID_ID)["case"]
    )
    preview = preview_payload(material, case, source="mmen_test")
    warning = " ".join(preview["warnings"])
    assert "MMEN INDEPENDENT-COMPONENT HUMID DIAGNOSTIC" in warning
    assert "Direct humidity enhancement or competition of CO2 is absent" in warning
    assert "separate Bose specimen" in warning
    assert "hot desorption rates remain assumed" in warning
    assert preview["material_id"] == MATERIAL_ID

    manager = MofScreeningManager(
        tmp_path / "jobs",
        execution_lock_path=tmp_path / "LOCK.json",
    )
    job_id = "2" * 32
    root = manager.job_root / job_id
    root.mkdir(parents=True, exist_ok=True)
    result = {
        "schema": "mof-screening-result/1",
        "job_id": job_id,
        "temperature_policy": {"mode": "record_scenario_excursions"},
        "temperature_guard_assessment": {"status": "within_scenario_guard"},
        "accepted_state_temperature_observations": {
            "maximum_bed_temperature_K": 350.0
        },
        "material_query_temperature_observations": {
            "minimum_finite_temperature_K": 298.0,
            "maximum_finite_temperature_K": 355.0,
        },
        "water_burden": {
            "initial_loading_mol_kg": 0.0,
            "final_loading_mol_kg": 4.0,
        },
        "metrics": {
            "water_condensed_mol": 1.0,
            "water_vented_mol": 2.0,
            "water_in_product_mol": 3.0,
        },
        "checks": {
            "mass_closure_pass": True,
            "energy_closure_pass": True,
            "energy_relative_residual": 1e-7,
        },
        "traces": [
            {
                "elapsed_s": 1.0,
                "cycle": 1,
                "phase": "adsorption",
                "water_loading_mean_mol_kg": 4.0,
                "bed_temperature_max_K": 300.0,
            }
        ],
    }
    (root / "STATE.json").write_text(
        json.dumps({"state": "completed", "job_id": job_id}),
        encoding="utf-8",
    )
    (root / "RESULT.json").write_text(
        json.dumps(result), encoding="utf-8"
    )
    csv_text, _filename, _content_type = manager.export(job_id, "csv")
    assert "water_loading_mean_mol_kg" in csv_text
    assert "water_vented_mol" in csv_text
    assert "energy_relative_residual" in csv_text
    manager.close()
