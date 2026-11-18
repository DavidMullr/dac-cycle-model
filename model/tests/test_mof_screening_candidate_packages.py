import csv
import math
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import PureWindowsPath
from unittest.mock import patch

import numpy as np
import pytest

from screening.mof_screening.contract import (
    COMPATIBLE_EXTRAPOLATION_POLICY,
    MATERIAL_ROOT,
    NAMED_EXTRAPOLATION_POLICY,
    ROOT,
    STRICT_POLICY,
    MofScreeningError,
    build_case,
    file_sha256,
    load_builtin_material,
    load_defaults,
    validate_material,
)
from screening.mof_screening.materials import build_material_adapter


NEW_CONDITIONAL_DRY_MATERIAL_IDS = (
    "mof808_lys_2024_25c_dry_conditional",
    "mof808_tapa_2024_25c_dry_conditional",
    "tepa50_mg2_dobpdc_2025_25c_apparent_dry_conditional",
)

ALL_DRY_ONLY_MATERIAL_IDS = (
    "mmen_mg2_hob_powder_2024_dry_conditional",
    "mmen_mg2_hob_pvdf_2024_dry_conditional",
) + NEW_CONDITIONAL_DRY_MATERIAL_IDS


def test_mmen_package_preserves_full_source_curve_and_marks_mixture_gap():
    material = load_builtin_material(
        "mmen_mg2_dobpdc_mcdonald2012_conditional"
    )
    curve = material["co2"]["reference_curve"]

    assert validate_material(material) == material
    assert len(curve["pressure_Pa"]) == 41
    assert curve["pressure_Pa"][0] == 5.742
    assert curve["pressure_Pa"][-1] == 110060.392
    assert curve["loading_mol_kg"][-1] == 3.26559
    assert material["co2"]["heat_J_mol"] == 71000.0
    assert material["missing_evidence"]["direct_humid_co2_equilibrium_closure"] is None
    assert material["missing_evidence"]["hot_desorption_kinetics"] is None
    assert "not independent withheld-temperature validation" in (
        material["provenance"]["co2_25C_anchor"]["note"]
    )


def test_mmen_adapter_matches_published_dac_anchor_and_has_finite_rates():
    material = load_builtin_material(
        "mmen_mg2_dobpdc_mcdonald2012_conditional"
    )
    adapter = build_material_adapter(material)
    partials = np.array([39.0, 0.0, 79050.0, 20950.0])

    equilibrium = adapter.equilibrium(298.15, partials)
    rates, species_rates = adapter.rates(
        298.15, partials, np.zeros(adapter.n_storage)
    )

    assert 1.95 < equilibrium[0] < 2.05
    assert np.all(np.isfinite(equilibrium))
    assert np.all(np.isfinite(rates))
    assert np.all(np.isfinite(species_rates))
    assert adapter.dry_only_water is False
    assert adapter.source_report()["water_mode"] == "unary_sips"

    legacy_water = adapter.equilibrium(
        298.15, np.array([39.0, 1000.0, 79000.0, 21000.0])
    )
    assert legacy_water[1] > 0.0


def test_lewatit_package_is_conditional_and_retains_hot_fit_failure():
    material = load_builtin_material(
        "lewatit_vp_oc_1065_low2023_conditional"
    )
    fit = material["provenance"]["co2_temperature_shift"]

    assert validate_material(material) == material
    assert material["evidence"]["status"].startswith("CONDITIONAL")
    assert len(material["co2"]["reference_curve"]["pressure_Pa"]) == 49
    deployed = fit["deployed_fit"]
    failed_fold = fit["separate_fold_withholding_393_15K"]
    assert deployed["Q_J_mol"] == 75883.673
    assert deployed["withheld_temperature_K"] == 353.15
    assert math.isclose(
        deployed["withheld_RMSE_mol_kg"], 0.1752, rel_tol=1.0e-4
    )
    assert 393.15 in deployed["training_temperatures_K"]
    assert math.isclose(
        deployed["in_sample_393_15K_RMSE_mol_kg"],
        0.255664,
        rel_tol=1.0e-4,
    )
    assert failed_fold["Q_J_mol"] == 68024.53
    assert failed_fold["withheld_temperature_K"] == 393.15
    assert failed_fold["withheld_RMSE_mol_kg"] > 0.30
    assert "not a predeclared gate" in fit["posthoc_diagnostic_note"]
    assert material["missing_evidence"]["direct_humid_co2_coadsorption_adapter"] is None


def test_lewatit_adapter_uses_measured_reference_shape_without_humid_claim():
    material = load_builtin_material(
        "lewatit_vp_oc_1065_low2023_conditional"
    )
    adapter = build_material_adapter(material)
    partials = np.array([40.0, 0.0, 79050.0, 20950.0])

    equilibrium = adapter.equilibrium(298.15, partials)
    rates, species_rates = adapter.rates(
        298.15, partials, np.zeros(adapter.n_storage)
    )

    assert 0.90 < equilibrium[0] < 1.00
    assert np.all(np.isfinite(equilibrium))
    assert np.all(np.isfinite(rates))
    assert np.all(np.isfinite(species_rates))
    assert "not direct CO2-water coadsorption" in material["water"]["basis"]


@pytest.mark.parametrize(
    (
        "material_id",
        "package_sha256",
        "point_count",
        "table_loading",
        "digitized_loading",
    ),
    [
        (
            "mof808_lys_2024_25c_dry_conditional",
            "b2044afbbbad486b4ff818e37285d725cfde73b866f2167c77e81ff87c0008ab",
            34,
            0.612,
            0.614289940806174,
        ),
        (
            "mof808_tapa_2024_25c_dry_conditional",
            "9247f07b8dad4fc4df898d24b917f057030ea4b5fd490acbf6b6ca436f67093e",
            35,
            0.498,
            0.47498845810672735,
        ),
    ],
)
def test_mof808_packages_preserve_exact_digitized_source_and_table_discrepancy(
    material_id,
    package_sha256,
    point_count,
    table_loading,
    digitized_loading,
):
    material = load_builtin_material(material_id)
    curve = material["co2"]["reference_curve"]
    provenance = material["provenance"]["co2_reference_curve"]
    package_path = MATERIAL_ROOT / f"{material_id}.json"
    source_path = ROOT / "tests/fixtures/mof_screening/source_tables" / PureWindowsPath(provenance["source_asset"]).name
    with source_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert file_sha256(package_path) == package_sha256
    assert len(curve["pressure_Pa"]) == point_count == len(rows)
    assert curve["pressure_Pa"] == [
        float(row["pressure_Pa"]) for row in rows
    ]
    assert curve["loading_mol_kg"] == [
        float(row["loading_mol_kg"]) for row in rows
    ]
    assert provenance["source_table_at_400_ppm_mol_kg"] == table_loading
    assert provenance["digitized_at_40_Pa_mol_kg"] == digitized_loading
    assert provenance["reported_experimental_uncertainty"] is None
    assert provenance["difference_table_minus_digitized_mol_kg"] == pytest.approx(
        table_loading - digitized_loading
    )
    assert file_sha256(source_path) == curve["source_csv_sha256"]
    assert (
        file_sha256(
            ROOT / "tests/fixtures/mof_screening/source_tables" / PureWindowsPath(provenance["source_audit"]).name
        )
        == curve["digitization_audit_sha256"]
    )
    assert material["co2"]["support"]["policy"] == "named_extrapolation_scenario"
    assert material["water"]["model"] == "dry_only"
    assert material["provenance"]["co2_temperature_shift"]["class"] == "assumed"
    mass_mapping = material["provenance"]["mass_mapping"]
    assert mass_mapping["class"] == "assumed"
    assert mass_mapping["unity_loading_retention_and_mass_transfer"] is True
    assert mass_mapping["final_binder_mass_fraction"] is None
    assert mass_mapping["shaped_density"] is None
    assert mass_mapping["shaped_effective_transport"] is None
    for value in material["missing_evidence"].values():
        assert value is None


def test_tepa_package_preserves_raw_nonmonotonicity_and_fitted_projection():
    material_id = "tepa50_mg2_dobpdc_2025_25c_apparent_dry_conditional"
    material = load_builtin_material(material_id)
    curve = material["co2"]["reference_curve"]
    provenance = material["provenance"]["co2_reference_curve"]
    package_path = MATERIAL_ROOT / f"{material_id}.json"
    source_path = ROOT / "tests/fixtures/mof_screening/source_tables" / PureWindowsPath(provenance["source_asset"]).name
    with source_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    pressures = np.asarray(
        [float(row["pressure_Pa"]) for row in rows], dtype=float
    )
    raw_loadings = np.asarray(
        [float(row["loading_mol_kg"]) for row in rows], dtype=float
    )
    expected_projection = np.maximum.accumulate(raw_loadings)

    assert file_sha256(package_path) == (
        "8a524de458802f9cb66398fd366022561f0a03852f02fb923f3d73ae596f7737"
    )
    assert len(rows) == 21
    np.testing.assert_array_equal(curve["pressure_Pa"], pressures)
    np.testing.assert_array_equal(curve["loading_mol_kg"], expected_projection)
    assert curve["class"].startswith("FITTED nonnegative monotone")
    decrease = provenance["raw_nonmonotone_observations"][0]
    assert decrease["to_loading_mol_kg"] < decrease["from_loading_mol_kg"]
    assert decrease["decrease_mol_kg"] == pytest.approx(
        decrease["to_loading_mol_kg"] - decrease["from_loading_mol_kg"]
    )
    assert file_sha256(source_path) == curve["source_csv_sha256"]
    assert material["provenance"]["sample_identity"][
        "description"
    ].startswith("TEPA(50)-Mg2(dobpdc)")
    assert "not a TEPA-to-framework molar ratio" in material["provenance"][
        "sample_identity"
    ]["description"]
    assert material["co2"]["support"]["policy"] == "named_extrapolation_scenario"
    assert material["water"]["model"] == "dry_only"
    mass_mapping = material["provenance"]["mass_mapping"]
    assert mass_mapping["unity_loading_retention_and_mass_transfer"] is True
    assert mass_mapping["additional_amine_dilution_factor"] == 1.0
    assert mass_mapping["additional_amine_dilution_applied"] is False
    assert "full dry TEPA(50)-impregnated sorbent" in mass_mapping[
        "source_loading_basis"
    ]
    assert mass_mapping["final_binder_mass_fraction"] is None
    assert mass_mapping["shaped_density"] is None
    assert mass_mapping["shaped_effective_transport"] is None
    assert material["missing_evidence"][
        "true_equilibrium_isotherm_at_multiple_temperatures"
    ] is None
    assert all(
        value is None for value in material["missing_evidence"].values()
    )




@pytest.mark.parametrize("material_id", NEW_CONDITIONAL_DRY_MATERIAL_IDS)
def test_new_dry_only_packages_have_finite_terms_and_zero_water(material_id):
    material = load_builtin_material(material_id)
    adapter = build_material_adapter(material)
    partials = np.array([40.0, 1.0, 79000.0, 21000.0])
    storage = np.array([0.0, 0.25, 0.0, 0.0])

    equilibrium = adapter.equilibrium(298.15, partials)
    rates, species_rates = adapter.rates(298.15, partials, storage)
    cp_molar = adapter.component_cp_molar_J_mol_K()
    energies = adapter.absolute_energy_J_mol(310.0)

    assert np.all(np.isfinite(equilibrium))
    assert np.all(np.isfinite(rates))
    assert np.all(np.isfinite(species_rates))
    assert np.all(np.isfinite(cp_molar))
    assert np.all(np.isfinite(energies))
    assert equilibrium[0] > 0.0
    assert equilibrium[1] == rates[1] == species_rates[1] == 0.0
    assert cp_molar[0] == pytest.approx(37.5)
    assert np.all(cp_molar[1:] == 0.0)
    assert energies[1] == 0.0
    assert adapter.source_report()["water_mode"] == "dry_only"
    assert adapter.source_report()["physical_water_storage_mol_kg"] == 0.0
    assert (
        adapter.source_report()["co2_support_policy"]
        == NAMED_EXTRAPOLATION_POLICY
    )


@pytest.mark.parametrize("material_id", NEW_CONDITIONAL_DRY_MATERIAL_IDS)
@pytest.mark.parametrize("temperature_K", (298.655, 353.15, 373.15))
def test_named_extrapolation_direct_pressure_and_hot_temperature_support(
    material_id, temperature_K
):
    material = deepcopy(load_builtin_material(material_id))
    material["nitrogen"] = {
        "henry_mol_kg_Pa_at_reference": 1.0e-7,
        "heat_J_mol": 20000.0,
    }
    material["oxygen"] = {
        "henry_mol_kg_Pa_at_reference": 2.0e-7,
        "heat_J_mol": 18000.0,
    }
    material["adsorbed_phase_cp_J_kg_K"] = [
        37.5 / 0.0440095,
        0.0,
        1000.0,
        800.0,
    ]
    material = validate_material(material)
    adapter = build_material_adapter(deepcopy(material))
    model = adapter.model
    factor = math.exp(
        float(model.heats[0])
        / 8.31446261815324
        * (1.0 / temperature_K - 1.0 / model.curve_temperature_K)
    )

    for mapped_pressure_Pa in (
        float(model.curve_pressure[0]) * 0.5,
        float(model.curve_pressure[-1]) * 1.01,
    ):
        raw_pressure_Pa = mapped_pressure_Pa / factor
        partials = np.array(
            [raw_pressure_Pa, 1.0, 79000.0, 21000.0]
        )
        equilibrium = adapter.equilibrium(temperature_K, partials)
        rates, species_rates = adapter.rates(
            temperature_K,
            partials,
            np.array([0.0, 0.25, 0.0, 0.0]),
        )
        energy = adapter.absolute_energy_J_mol(temperature_K)

        assert model.support_of(
            temperature_K, raw_pressure_Pa
        )["policy"] == NAMED_EXTRAPOLATION_POLICY
        assert np.all(np.isfinite(equilibrium))
        assert np.all(np.isfinite(rates))
        assert np.all(np.isfinite(species_rates))
        assert np.all(np.isfinite(energy))
        assert equilibrium[0] >= 0.0
        assert equilibrium[1] == rates[1] == species_rates[1] == 0.0
        assert equilibrium[2] > 0.0
        assert equilibrium[3] > 0.0
        assert adapter.component_cp_molar_J_mol_K()[1] == 0.0
        assert adapter.component_cp_molar_J_mol_K()[2] > 0.0
        assert adapter.component_cp_molar_J_mol_K()[3] > 0.0


def test_dry_support_policy_strict_alias_and_unknown_contract_semantics():
    source = load_builtin_material(
        "mof808_lys_2024_25c_dry_conditional"
    )

    legacy = deepcopy(source)
    legacy["co2"]["support"]["policy"] = (
        COMPATIBLE_EXTRAPOLATION_POLICY
    )
    legacy = validate_material(legacy)
    legacy_adapter = build_material_adapter(legacy)
    curve = legacy["co2"]["reference_curve"]
    raw_pressure = float(curve["pressure_Pa"][0]) * 0.5
    legacy_equilibrium = legacy_adapter.equilibrium(
        373.15, np.array([raw_pressure, 1.0, 79000.0, 21000.0])
    )
    assert np.all(np.isfinite(legacy_equilibrium))
    assert legacy_adapter.source_report()["co2_support_policy"] == (
        COMPATIBLE_EXTRAPOLATION_POLICY
    )

    strict = deepcopy(source)
    strict["co2"]["support"]["policy"] = STRICT_POLICY
    strict_adapter = build_material_adapter(validate_material(strict))
    with pytest.raises(ValueError, match="outside the measured support"):
        strict_adapter.equilibrium(
            373.15,
            np.array([raw_pressure, 1.0, 79000.0, 21000.0]),
        )

    unknown = deepcopy(source)
    unknown["co2"]["support"]["policy"] = "silent_extrapolation"
    with pytest.raises(MofScreeningError) as captured:
        validate_material(unknown)
    assert captured.value.field == "material.co2.support.policy"


def test_candidate_packages_have_finite_spatial_and_lumped_rhs_without_integration():
    from screening.geometry import build_bare_geometry
    from screening.mof_screening.contract import load_defaults
    from screening.mof_screening.lumped_port.engine import run_cycle
    from screening.mof_screening.runtime import (
        build_lumped_config,
        build_spatial_config,
    )
    from screening.mof_screening.spatial import (
        build_model,
        cycle_rhs,
        initial_state,
    )
    from screening.mof_screening.weather import load_weather_window

    defaults = load_defaults()
    weather = load_weather_window(defaults["case"]["weather"])
    for material_id in (
        "mmen_mg2_dobpdc_mcdonald2012_conditional",
        "lewatit_vp_oc_1065_low2023_conditional",
    ):
        material = load_builtin_material(material_id)
        adapter = build_material_adapter(material)

        spatial_config = build_spatial_config(
            material, defaults["case"], weather.rows[0]
        )
        geometry = build_bare_geometry(spatial_config)
        model = build_model(spatial_config, adapter, geometry)
        state = initial_state(spatial_config, adapter, geometry, model)
        derivative, _ = cycle_rhs(
            spatial_config,
            adapter,
            geometry,
            "adsorption",
            model=model,
        )(0.0, state)
        assert derivative.shape == state.shape
        assert np.all(np.isfinite(derivative))

        lumped_config = build_lumped_config(
            material, defaults["case"], weather.rows[0]
        )
        bundle = run_cycle(
            adapter,
            lumped_config,
            relative_humidity=weather.rows[0].relative_humidity,
            rhs_only=True,
        )
        lumped_state = bundle["initial_state"]()
        for phase, rhs in bundle["rhs_by_phase"].items():
            derivative = np.asarray(rhs(0.0, lumped_state), dtype=float)
            assert derivative.shape == lumped_state.shape, (
                material_id,
                phase,
            )
            assert np.all(np.isfinite(derivative)), (material_id, phase)


@pytest.mark.parametrize(
    ("material_id", "point_count", "heat_J_mol"),
    [
        ("mmen_mg2_hob_powder_2024_dry_conditional", 42, 71853.135434),
        ("mmen_mg2_hob_pvdf_2024_dry_conditional", 50, 49117.725234),
    ],
)
def test_hob_packages_preserve_failed_fit_and_dry_only_terms(
    material_id, point_count, heat_J_mol
):
    material = load_builtin_material(material_id)
    roundtripped = validate_material(json.loads(json.dumps(material)))
    adapter = build_material_adapter(roundtripped)
    partials = np.array([40.0, 1.0, 79000.0, 21000.0])
    equilibrium = adapter.equilibrium(298.15, partials)
    trial_storage = np.array([0.0, 0.25, 0.0, 0.0])
    rates, species_rates = adapter.rates(298.15, partials, trial_storage)

    assert roundtripped == material
    assert len(material["co2"]["reference_curve"]["pressure_Pa"]) == point_count
    assert math.isclose(material["co2"]["heat_J_mol"], heat_J_mol, rel_tol=1e-10)
    assert material["water"]["model"] == "dry_only"
    assert material["water"]["physical_storage_mol_kg"] == 0.0
    assert material["water"]["solver_normalizer_mol_kg"] > 0.0
    assert material["provenance"]["co2_temperature_shift"]["gate_pass"] is False
    assert material["missing_evidence"]["direct_humid_co2_equilibrium_closure"] is None
    assert material["missing_evidence"]["hot_desorption_kinetics"] is None
    amine = material["provenance"]["amine_loading"]
    assert "diamine_per_Mg2" not in amine
    assert "not per Mg2 formula unit" in amine["locator"]
    if material_id.endswith("powder_2024_dry_conditional"):
        assert amine["diamine_per_magnesium_ion"] == 0.93
    else:
        assert amine["diamine_per_magnesium_ion_before_humid_tpd"] == 0.92
        assert amine["diamine_per_magnesium_ion_after_humid_tpd"] == 0.93

    assert equilibrium[1] == 0.0
    assert rates[1] == 0.0
    assert species_rates[1] == 0.0
    assert np.all(adapter.component_cp_molar_J_mol_K()[1:] == 0.0)
    assert np.all(adapter.absolute_energies_J_mol(298.15)[1] == 0.0)
    assert adapter.storage_scale()[1] == material["water"][
        "solver_normalizer_mol_kg"
    ]
    assert not hasattr(adapter.model, "b_water_ref")
    assert adapter.source_report()["water_mode"] == "dry_only"
    assert adapter.source_report()["physical_water_storage_mol_kg"] == 0.0


@pytest.mark.parametrize("material_id", ALL_DRY_ONLY_MATERIAL_IDS)
def test_dry_only_packages_reject_water_at_case_boundary(material_id):
    defaults = load_defaults()
    material = load_builtin_material(material_id)

    with pytest.raises(MofScreeningError) as captured:
        build_case(material, defaults["case"])
    assert captured.value.field == "case.weather.relative_humidity"

    tiny_humidity_case = deepcopy(defaults["case"])
    tiny_humidity_case["weather"]["relative_humidity"] = 1.0e-15
    with pytest.raises(MofScreeningError) as tiny_capture:
        build_case(material, tiny_humidity_case)
    assert tiny_capture.value.field == "case.weather.relative_humidity"

    dry_case = deepcopy(defaults["case"])
    dry_case["weather"]["relative_humidity"] = 0.0
    normalized, validated_case = build_case(material, dry_case)
    assert normalized["water"]["physical_storage_mol_kg"] == 0.0
    assert validated_case["weather"]["relative_humidity"] == 0.0


@pytest.mark.parametrize("material_id", ALL_DRY_ONLY_MATERIAL_IDS)
def test_dry_only_packages_reject_later_humid_weather_interval(material_id):
    from types import SimpleNamespace

    from screening.mof_screening.runtime import _validate_weather_for_material

    material = load_builtin_material(material_id)
    _validate_weather_for_material(
        material, [SimpleNamespace(relative_humidity=0.0)]
    )
    with pytest.raises(MofScreeningError) as captured:
        _validate_weather_for_material(
            material,
            [
                SimpleNamespace(relative_humidity=0.0),
                SimpleNamespace(relative_humidity=0.1),
            ],
        )
    assert captured.value.field == "case.weather.relative_humidity"


@pytest.mark.parametrize("material_id", ALL_DRY_ONLY_MATERIAL_IDS)
def test_dry_only_official_preview_validates_every_window_row(
    monkeypatch, material_id
):
    from types import SimpleNamespace

    from screening.mof_screening import weather as weather_module

    defaults = load_defaults()
    material = load_builtin_material(material_id)
    case = deepcopy(defaults["case"])
    case["weather"].update(
        {"mode": "official", "hours": 2, "start_utc": "2023-01-15T00:00:00Z"}
    )

    monkeypatch.setattr(
        weather_module,
        "load_weather_window",
        lambda spec: SimpleNamespace(
            rows=[
                SimpleNamespace(relative_humidity=0.0),
                SimpleNamespace(relative_humidity=0.1),
            ]
        ),
    )
    with pytest.raises(MofScreeningError) as captured:
        build_case(material, case)
    assert captured.value.field == "case.weather.relative_humidity"

    monkeypatch.setattr(
        weather_module,
        "load_weather_window",
        lambda spec: SimpleNamespace(
            rows=[
                SimpleNamespace(relative_humidity=0.0),
                SimpleNamespace(relative_humidity=0.0),
            ]
        ),
    )
    normalized, normalized_case = build_case(material, case)
    assert normalized["material_id"] == material["material_id"]
    assert normalized_case["weather"]["mode"] == "official"


@pytest.mark.parametrize(
    ("mutation", "field"),
    [
        (
            lambda material: material["adsorbed_phase_cp_J_kg_K"].__setitem__(
                1, 1000.0
            ),
            "material.adsorbed_phase_cp_J_kg_K[1]",
        ),
        (
            lambda material: material["kinetics"]["adsorption_rate_s_inv"].__setitem__(
                1, 0.1
            ),
            "material.kinetics.adsorption_rate_s_inv[1]",
        ),
        (
            lambda material: material["kinetics"]["desorption_rate_s_inv"].__setitem__(
                1, 0.1
            ),
            "material.kinetics.desorption_rate_s_inv[1]",
        ),
        (
            lambda material: material.__setitem__(
                "co2_water_penalty_per_mol_kg", 10.0
            ),
            "material.co2_water_penalty_per_mol_kg",
        ),
    ],
)
def test_hob_dry_only_contract_rejects_conflicting_water_semantics(
    mutation, field
):
    material = deepcopy(
        load_builtin_material("mmen_mg2_hob_powder_2024_dry_conditional")
    )
    mutation(material)

    with pytest.raises(MofScreeningError) as captured:
        validate_material(material)
    assert captured.value.field == field


def test_hob_dry_only_preserves_inert_terms_and_zeroes_water_trials():
    material = deepcopy(
        load_builtin_material("mmen_mg2_hob_powder_2024_dry_conditional")
    )
    material["adsorbed_phase_cp_J_kg_K"] = [
        37.5 / 0.0440095,
        0.0,
        1000.0,
        800.0,
    ]
    material["nitrogen"] = {
        "henry_mol_kg_Pa_at_reference": 1.0e-6,
        "heat_J_mol": 20000.0,
    }
    material["oxygen"] = {
        "henry_mol_kg_Pa_at_reference": 2.0e-6,
        "heat_J_mol": 18000.0,
    }
    material = validate_material(material)
    adapter = build_material_adapter(material)
    dry_partials = np.array([40.0, 0.0, 79000.0, 21000.0])
    solver_trial_partials = np.array([40.0, 1.0, 79000.0, 21000.0])

    dry_equilibrium = adapter.equilibrium(298.15, dry_partials)
    trial_equilibrium = adapter.equilibrium(298.15, solver_trial_partials)
    rates, species_rates = adapter.rates(
        298.15,
        solver_trial_partials,
        np.array([0.0, 0.25, 0.0, 0.0]),
    )

    assert dry_equilibrium[0] == trial_equilibrium[0]
    assert dry_equilibrium[1] == trial_equilibrium[1] == 0.0
    assert rates[1] == species_rates[1] == 0.0
    cp_molar = adapter.component_cp_molar_J_mol_K()
    energies = adapter.absolute_energies_J_mol(310.0)
    assert cp_molar[1] == 0.0
    assert cp_molar[2] > 0.0
    assert cp_molar[3] > 0.0
    assert energies[1] == 0.0
    assert energies[2] != 0.0
    assert energies[3] != 0.0
    gas_constant = 8.31446261815324
    assert energies[2] == pytest.approx(
        cp_molar[2] * (310.0 - material["reference_temperature_K"])
        + gas_constant * material["reference_temperature_K"]
        - 20000.0
    )
    assert energies[3] == pytest.approx(
        cp_molar[3] * (310.0 - material["reference_temperature_K"])
        + gas_constant * material["reference_temperature_K"]
        - 18000.0
    )
    assert np.all(np.isfinite(rates))
    assert np.all(np.isfinite(species_rates))
    assert np.all(np.isfinite(energies))


def test_hob_dry_only_rhs_is_finite_without_integration():
    from screening.geometry import build_bare_geometry
    from screening.mof_screening.lumped_port.engine import run_cycle
    from screening.mof_screening.runtime import (
        build_lumped_config,
        build_spatial_config,
    )
    from screening.mof_screening.spatial import (
        build_model,
        cycle_rhs,
        initial_state,
    )
    from screening.mof_screening.weather import load_weather_window

    defaults = load_defaults()
    dry_case = deepcopy(defaults["case"])
    dry_case["weather"]["relative_humidity"] = 0.0
    weather = load_weather_window(dry_case["weather"])
    for material_id in (
        "mmen_mg2_hob_powder_2024_dry_conditional",
        "mmen_mg2_hob_pvdf_2024_dry_conditional",
    ):
        material = load_builtin_material(material_id)
        adapter = build_material_adapter(material)

        spatial_config = build_spatial_config(
            material, dry_case, weather.rows[0]
        )
        geometry = build_bare_geometry(spatial_config)
        model = build_model(spatial_config, adapter, geometry)
        state = initial_state(spatial_config, adapter, geometry, model)
        derivative, _ = cycle_rhs(
            spatial_config,
            adapter,
            geometry,
            "adsorption",
            model=model,
        )(0.0, state)
        assert derivative.shape == state.shape
        assert np.all(np.isfinite(derivative))

        lumped_config = build_lumped_config(
            material, dry_case, weather.rows[0]
        )
        bundle = run_cycle(
            adapter,
            lumped_config,
            relative_humidity=0.0,
            rhs_only=True,
        )
        lumped_state = bundle["initial_state"]()
        for phase, rhs in bundle["rhs_by_phase"].items():
            derivative = np.asarray(rhs(0.0, lumped_state), dtype=float)
            assert derivative.shape == lumped_state.shape, phase
            assert np.all(np.isfinite(derivative)), phase


@pytest.mark.parametrize("material_id", NEW_CONDITIONAL_DRY_MATERIAL_IDS)
def test_new_dry_only_packages_have_finite_both_engine_all_phase_rhs(
    material_id,
):
    from screening.geometry import build_bare_geometry
    from screening.mof_screening.lumped_port.engine import run_cycle
    from screening.mof_screening.runtime import (
        build_lumped_config,
        build_spatial_config,
    )
    from screening.mof_screening.spatial import (
        STEPS,
        build_model,
        cycle_rhs,
        initial_state,
    )
    from screening.mof_screening.weather import load_weather_window

    defaults = load_defaults()
    dry_case = deepcopy(defaults["case"])
    dry_case["weather"]["relative_humidity"] = 0.0
    material, dry_case = build_case(
        load_builtin_material(material_id), dry_case
    )
    adapter = build_material_adapter(material)
    weather = load_weather_window(dry_case["weather"])

    spatial_config = build_spatial_config(
        material, dry_case, weather.rows[0]
    )
    geometry = build_bare_geometry(spatial_config)
    model = build_model(spatial_config, adapter, geometry)
    state = initial_state(spatial_config, adapter, geometry, model)
    for phase in STEPS:
        derivative, _ = cycle_rhs(
            spatial_config,
            adapter,
            geometry,
            phase,
            model=model,
        )(0.0, state)
        assert derivative.shape == state.shape, phase
        assert np.all(np.isfinite(derivative)), phase

    lumped_config = build_lumped_config(
        material, dry_case, weather.rows[0]
    )
    bundle = run_cycle(
        adapter,
        lumped_config,
        relative_humidity=0.0,
        rhs_only=True,
    )
    lumped_state = bundle["initial_state"]()
    assert set(bundle["rhs_by_phase"]) == set(STEPS)
    for phase, rhs in bundle["rhs_by_phase"].items():
        derivative = np.asarray(rhs(0.0, lumped_state), dtype=float)
        assert derivative.shape == lumped_state.shape, phase
        assert np.all(np.isfinite(derivative)), phase


@pytest.mark.parametrize("material_id", NEW_CONDITIONAL_DRY_MATERIAL_IDS)
@pytest.mark.parametrize("temperature_K", (298.655, 353.15, 373.15))
def test_hot_named_extrapolation_all_phase_rhs_is_finite(
    material_id, temperature_K
):
    from screening.geometry import build_bare_geometry
    from screening.mof_screening.lumped_port.engine import run_cycle
    from screening.mof_screening.runtime import (
        build_lumped_config,
        build_spatial_config,
    )
    from screening.mof_screening.spatial import (
        STEPS,
        build_model,
        cycle_rhs,
        initial_state,
    )
    from screening.mof_screening.weather import WeatherHour

    defaults = load_defaults()
    case = deepcopy(defaults["case"])
    case["weather"].update(
        {
            "temperature_C": min(temperature_K - 273.15, 99.85),
            "relative_humidity": 0.0,
            "co2_ppm_dry": 40.0,
        }
    )
    material, case = build_case(
        load_builtin_material(material_id), case
    )
    adapter = build_material_adapter(material)
    weather = WeatherHour(
        interval_start_utc="2000-01-01T00:00:00+00:00",
        interval_end_utc="2000-01-01T01:00:00+00:00",
        temperature_C=temperature_K - 273.15,
        relative_humidity=0.0,
        pressure_Pa=100000.0,
        co2_ppm_dry=40.0,
    )

    with ExitStack() as correlations:
        if temperature_K == 373.15:
            correlations.enter_context(
                patch(
                    "screening.common.water_saturation_pressure_Pa",
                    return_value=0.0,
                )
            )
            correlations.enter_context(
                patch(
                    "screening.mof_screening.lumped_port.engine."
                    "water_saturation_pressure_Pa",
                    return_value=0.0,
                )
            )
        spatial_config = build_spatial_config(material, case, weather)
        geometry = build_bare_geometry(spatial_config)
        model = build_model(spatial_config, adapter, geometry)
        state = initial_state(spatial_config, adapter, geometry, model)
        for phase in STEPS:
            derivative, _ = cycle_rhs(
                spatial_config, adapter, geometry, phase, model=model
            )(0.0, state)
            assert derivative.shape == state.shape, phase
            assert np.all(np.isfinite(derivative)), (phase, temperature_K)

        lumped_config = build_lumped_config(
            material, case, weather
        )
        bundle = run_cycle(
            adapter,
            lumped_config,
            relative_humidity=0.0,
            rhs_only=True,
        )
        lumped_state = bundle["initial_state"]()
        for phase, rhs in bundle["rhs_by_phase"].items():
            derivative = np.asarray(rhs(0.0, lumped_state), dtype=float)
            assert derivative.shape == lumped_state.shape, phase
            assert np.all(np.isfinite(derivative)), (phase, temperature_K)
