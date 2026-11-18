from copy import deepcopy
import json
import math

import numpy as np
import pytest

from screening.geometry import build_bare_geometry
from screening.mof_screening.contract import (
    MATERIAL_ROOT,
    TEMPERATURE_REGISTRY_PATH,
    MofScreeningError,
    build_case,
    canonical_sha256,
    effective_temperature_policy,
    file_sha256,
    load_builtin_material,
    load_defaults,
    preview_payload,
    read_json_strict,
    validate_material,
)
from screening.mof_screening.evaluation import annotate_result
from screening.mof_screening.lumped_port import weather_driver
from screening.mof_screening.lumped_port.engine import (
    diagnostic_water_saturation_ratios,
)
from screening.mof_screening.lumped_port.engine import run_cycle
from screening.mof_screening.manager import MofScreeningManager
from screening.mof_screening.materials import build_material_adapter
from screening.mof_screening.preflight import (
    execution_source_identity,
    dependency_status,
)
from screening.mof_screening.runtime import (
    _trace_rows,
    build_lumped_config,
    build_spatial_config,
)
from screening.mof_screening.spatial import (
    STEPS,
    build_model,
    cycle_rhs,
    initial_state,
)
from screening.mof_screening.stored_water_shift import saturation_pressure_Pa
from screening.mof_screening.weather import load_weather_window


CANDIDATES = {
    "cof1000_humid_water_shift_scenario_v1": (
        "cof1000_2025_25c_split_rate_dry_conditional",
        ("CO2_fast", "CO2_slow", "H2O", "N2", "O2"),
    ),
    "cof999_humid_water_shift_scenario_v1": (
        "cof999_2024_25c_dry_conditional",
        ("CO2", "H2O", "N2", "O2"),
    ),
    "mmen_independent_unary_humid_diagnostic_v1": (
        "mmen_mg2_dobpdc_mcdonald2012_conditional",
        ("CO2", "H2O", "N2", "O2"),
    ),
    "hob_powder_measured_water_transport_independent_v1": (
        "mmen_mg2_hob_powder_2024_dry_conditional",
        ("CO2", "H2O", "N2", "O2"),
    ),
    "hob_pvdf_same_framework_water_transfer_scenario_v1": (
        "mmen_mg2_hob_pvdf_2024_dry_conditional",
        ("CO2", "H2O", "N2", "O2"),
    ),
    "tepa50_finite_window_water_proxy_scenario_v1": (
        "tepa50_mg2_dobpdc_2025_25c_apparent_dry_conditional",
        ("CO2", "H2O", "N2", "O2"),
    ),
    "mof808_lys_humid_one_node_scenario_v1": (
        "mof808_lys_2024_25c_dry_conditional",
        ("CO2", "H2O", "N2", "O2"),
    ),
    "mof808_tapa_water_shift_scenario_v1": (
        "mof808_tapa_2024_25c_dry_conditional",
        ("CO2", "H2O", "N2", "O2"),
    ),
    "mil101_pei800_control_measured_water_diagnostic_v1": (
        "mil101_pei800_30_2026_25c_dry_conditional",
        ("CO2", "H2O", "N2", "O2"),
    ),
    "mil101_pei800_glycerin_measured_water_diagnostic_v1": (
        "mil101_pei800_30_glycerin_2026_25c_dry_conditional",
        ("CO2", "H2O", "N2", "O2"),
    ),
    "calf20_independent_water_diagnostic_v1": (
        "calf20_nguyen2024_80_20_shaped",
        ("CO2_site_1", "CO2_site_2", "H2O"),
    ),
}


def _load_dry(dry_id):
    return load_builtin_material(
        "calf20_shaped" if dry_id == "calf20_nguyen2024_80_20_shaped" else dry_id
    )


@pytest.mark.parametrize("candidate_id", sorted(CANDIDATES))
def test_candidate_preserves_exact_dry_origin_and_storage_layout(candidate_id):
    dry_id, expected_labels = CANDIDATES[candidate_id]
    candidate = load_builtin_material(candidate_id)
    dry = _load_dry(dry_id)
    extension = candidate["humidity_candidate"]

    assert validation_idempotent(candidate)
    assert candidate["material_id"] == candidate_id
    assert extension["dry_origin_material_id"] == dry_id
    assert extension["dry_origin_package_sha256"] == file_sha256(
        MATERIAL_ROOT
        / (
            "calf20_shaped.json"
            if dry_id == "calf20_nguyen2024_80_20_shaped"
            else f"{dry_id}.json"
        )
    )
    assert extension["dry_origin_package_canonical_sha256"] == (
        canonical_sha256(
            read_json_strict(
                MATERIAL_ROOT
                / (
                    "calf20_shaped.json"
                    if dry_id == "calf20_nguyen2024_80_20_shaped"
                    else f"{dry_id}.json"
                )
            )
        )
    )
    assert extension["reciprocal_thermodynamics"] is False
    assert extension["validated_humid_cycle"] is False
    assert tuple(extension["storage"]["labels"]) == expected_labels

    candidate_adapter = build_material_adapter(candidate)
    dry_adapter = build_material_adapter(dry)
    assert candidate_adapter.storage_labels == dry_adapter.storage_labels
    np.testing.assert_array_equal(
        candidate_adapter.species_for_storage,
        dry_adapter.species_for_storage,
    )
    partials = np.array([40.0, 0.0, 79000.0, 21000.0])
    candidate_target = candidate_adapter.equilibrium(298.15, partials)
    dry_target = dry_adapter.equilibrium(298.15, partials)
    wire = candidate_adapter.water_index
    nonwater = np.arange(candidate_adapter.n_storage) != wire
    np.testing.assert_array_equal(candidate_target[nonwater], dry_target[nonwater])
    storage = 0.25 * candidate_adapter.storage_scale()
    storage[wire] = 0.0
    candidate_rates, candidate_species = candidate_adapter.rates(
        298.15, partials, storage
    )
    dry_rates, dry_species = dry_adapter.rates(298.15, partials, storage)
    np.testing.assert_array_equal(candidate_rates[nonwater], dry_rates[nonwater])
    np.testing.assert_array_equal(candidate_species, dry_species)
    candidate_energy = candidate_adapter.storage_energy_terms(313.15, storage)
    dry_energy = dry_adapter.storage_energy_terms(313.15, storage)
    assert candidate_energy[0] == dry_energy[0]
    assert candidate_energy[1] == dry_energy[1]
    np.testing.assert_array_equal(
        candidate_energy[2][nonwater], dry_energy[2][nonwater]
    )


def validation_idempotent(material):
    validated = validate_material(material)
    return validate_material(validated) == validated


def test_stored_water_controls_coupling_while_raw_state_is_preserved():
    material = load_builtin_material("cof999_humid_water_shift_scenario_v1")
    adapter = build_material_adapter(material)
    water = adapter.water_index
    storage = 0.2 * adapter.storage_scale()
    dry_gas = np.array([40.0, 0.0, 79000.0, 21000.0])
    low = storage.copy()
    high = storage.copy()
    high[water] = 0.75 * adapter.storage_scale()[water]
    low_rate, _ = adapter.rates(298.15, dry_gas, low)
    high_rate, _ = adapter.rates(298.15, dry_gas, high)
    water_target = 0.2 * adapter.storage_scale()[water]
    no_stored_coupling = adapter._target_from_coupling_loading(
        298.15,
        dry_gas,
        water_target,
        0.0,
    )
    stored_coupling = adapter._target_from_coupling_loading(
        298.15,
        dry_gas,
        water_target,
        high[water],
    )

    assert stored_coupling[0] > no_stored_coupling[0]
    assert high_rate[0] != low_rate[0]
    assert high_rate[water] < low_rate[water]
    assert (
        "actual stored-water inventory"
        in adapter.source_report()["rate_coupling_basis"]
    )


@pytest.mark.parametrize(
    "candidate_id",
    (
        "mmen_independent_unary_humid_diagnostic_v1",
        "hob_powder_measured_water_transport_independent_v1",
        "hob_pvdf_same_framework_water_transfer_scenario_v1",
        "calf20_independent_water_diagnostic_v1",
    ),
)
def test_uncoupled_candidates_do_not_claim_water_to_co2_response(candidate_id):
    report = build_material_adapter(load_builtin_material(candidate_id)).source_report()
    assert report["co2_coupling"]["effective_coupling_active"] is False
    assert "No water-to-CO2 response is modeled" in report["rate_coupling_basis"]
    assert "controls the one-way CO2 coupling" not in report["rate_coupling_basis"]


def test_tiny_negative_water_is_constitutive_only_and_larger_negative_fails():
    adapter = build_material_adapter(
        load_builtin_material("cof999_humid_water_shift_scenario_v1")
    )
    water = adapter.water_index
    storage = 0.2 * adapter.storage_scale()
    storage[water] = -5e-10
    before = storage.copy()
    rate, species_rate = adapter.rates(
        298.15,
        np.array([40.0, 0.0, 79000.0, 21000.0]),
        storage,
    )
    np.testing.assert_array_equal(storage, before)
    assert np.all(np.isfinite(rate))
    assert np.all(np.isfinite(species_rate))
    report = adapter.source_report()
    assert report["negative_water_constitutive_calls"] == 1
    assert report["raw_state_inventory_rate_delta_energy_mutated"] is False

    storage[water] = -2e-9
    with pytest.raises(ValueError, match="constitutive numerical"):
        adapter.rates(
            298.15,
            np.array([40.0, 0.0, 79000.0, 21000.0]),
            storage,
        )


def test_calf_domain_is_checked_before_native_clipping():
    materialized = build_material_adapter(
        load_builtin_material("calf20_independent_water_diagnostic_v1")
    )
    materialized_calls = materialized.base_adapter._water_outside_calls
    materialized.equilibrium(
        298.15,
        np.array([40.0, 0.5 * saturation_pressure_Pa(298.15), 79000.0, 21000.0]),
    )
    assert materialized.base_adapter._water_outside_calls == materialized_calls

    adapter = build_material_adapter(
        load_builtin_material("calf20_independent_water_diagnostic_v1")
    )
    # Exercise the retained native PCHIP delegation path explicitly. The
    # materialized candidate normally uses its accepted temperature table.
    adapter.water_law = None
    before_calls = adapter.base_adapter._water_outside_calls
    partials = np.array(
        [
            40.0,
            1.001 * saturation_pressure_Pa(298.15),
            79000.0,
            21000.0,
        ]
    )
    with pytest.raises(ValueError, match="water activity"):
        adapter.equilibrium(298.15, partials)
    assert adapter.base_adapter._water_outside_calls == before_calls

    transformed = np.array(
        [40.0, 0.95 * saturation_pressure_Pa(273.15), 79000.0, 21000.0]
    )
    with pytest.raises(ValueError, match="reference activity"):
        adapter.equilibrium(273.15, transformed)
    assert adapter.base_adapter._water_outside_calls == before_calls


@pytest.mark.parametrize("candidate_id", sorted(CANDIDATES))
def test_hot_116c_and_cold_rich_states_follow_declared_domains(candidate_id):
    adapter = build_material_adapter(load_builtin_material(candidate_id))
    for state_name, temperature_K, water_partial_Pa in (
        ("hot_116C", 389.15, 5000.0),
        ("cold_0C", 273.15, 500.0),
    ):
        partials = np.array([16000.0, water_partial_Pa, 2000.0, 1000.0])
        try:
            target = adapter.equilibrium(temperature_K, partials)
            storage = 0.25 * adapter.storage_scale()
            rate, species_rate = adapter.rates(temperature_K, partials, storage)
            energy, cp, gradient = adapter.storage_energy_terms(temperature_K, storage)
        except ValueError as error:
            assert state_name == "cold_0C"
            assert "support" in str(error).lower() or "domain" in str(error).lower()
            continue
        assert np.all(np.isfinite(target)), (candidate_id, state_name)
        assert np.all(np.isfinite(rate)), (candidate_id, state_name)
        assert np.all(np.isfinite(species_rate)), (candidate_id, state_name)
        assert math.isfinite(energy) and math.isfinite(cp)
        assert np.all(np.isfinite(gradient))


@pytest.mark.parametrize("candidate_id", sorted(CANDIDATES))
def test_both_engines_all_phases_at_rh_half_and_spatial_n16_n32(candidate_id):
    defaults = load_defaults()
    material = load_builtin_material(candidate_id)
    case = deepcopy(defaults["case"])
    case["weather"]["relative_humidity"] = 0.5
    material, case = build_case(material, case)
    adapter = build_material_adapter(material)
    weather = load_weather_window(case["weather"])
    row = weather.rows[0]

    for cells in (16, 32):
        spatial_case = deepcopy(case)
        spatial_case["solver"]["cells"] = cells
        config = build_spatial_config(material, spatial_case, row)
        geometry = build_bare_geometry(config)
        model = build_model(config, adapter, geometry)
        state = initial_state(config, adapter, geometry, model)
        assert model.n_cells == cells
        for phase in STEPS:
            derivative, _ = cycle_rhs(
                config,
                adapter,
                geometry,
                phase,
                model=model,
            )(0.0, state)
            assert derivative.shape == state.shape, (candidate_id, cells, phase)
            assert np.all(np.isfinite(derivative)), (
                candidate_id,
                cells,
                phase,
            )

    lumped_case = deepcopy(case)
    lumped_case["engine"] = "lumped"
    lumped_case["solver"]["cells"] = 1
    lumped_case = build_case(material, lumped_case)[1]
    config = build_lumped_config(material, lumped_case, row)
    bundle = run_cycle(
        adapter,
        config,
        relative_humidity=0.5,
        rhs_only=True,
    )
    state = bundle["initial_state"]()
    assert set(bundle["rhs_by_phase"]) == set(STEPS)
    for phase, rhs in bundle["rhs_by_phase"].items():
        derivative = np.asarray(rhs(0.0, state), dtype=float)
        assert derivative.shape == state.shape, (candidate_id, phase)
        assert np.all(np.isfinite(derivative)), (candidate_id, phase)


@pytest.mark.parametrize(
    ("mutation", "field_part"),
    [
        (
            lambda extension: extension.__setitem__(
                "schema", "mof-screening-humidity-candidate/0"
            ),
            "schema",
        ),
        (
            lambda extension: extension.__setitem__("reciprocal_thermodynamics", True),
            "reciprocal_thermodynamics",
        ),
        (
            lambda extension: extension.__setitem__("validated_humid_cycle", True),
            "validated_humid_cycle",
        ),
        (
            lambda extension: extension.__setitem__(
                "dry_origin_package_sha256", "0" * 64
            ),
            "dry_origin_package_sha256",
        ),
        (
            lambda extension: extension["co2_coupling"].pop("calibration_status"),
            "co2_coupling.calibration_status",
        ),
        (
            lambda extension: extension["dry_limit"].__setitem__(
                "stored_water_mol_kg", 1e-6
            ),
            "stored_water_mol_kg",
        ),
    ],
)
def test_candidate_schema_refuses_inconsistent_extensions(mutation, field_part):
    material = deepcopy(load_builtin_material("mof808_lys_humid_one_node_scenario_v1"))
    mutation(material["humidity_candidate"])
    with pytest.raises(MofScreeningError) as captured:
        validate_material(material)
    assert field_part in captured.value.field


def test_native_candidates_retain_exact_reviewed_evidence_boundaries():
    cof = load_builtin_material("cof999_humid_water_shift_scenario_v1")
    assert cof["humidity_candidate"]["co2_coupling"]["holdout_status"] == (
        "LEAVE_ONE_OUT_FAILED_RETAINED"
    )

    lys = load_builtin_material("mof808_lys_humid_one_node_scenario_v1")
    assert (
        lys["humidity_candidate"]["provenance"]["protocol_anchors"][
            "co2_loading_mol_kg"
        ][1]
        == 0.613
    )
    assert any(
        "0.17 g/g" in assumption
        for assumption in lys["humidity_candidate"]["water_law"]["assumptions"]
    )

    tepa = load_builtin_material("tepa50_finite_window_water_proxy_scenario_v1")
    water = tepa["humidity_candidate"]["water_law"]
    coupling = tepa["humidity_candidate"]["co2_coupling"]
    assert water["evidence_class"] == (
        "ASSUMED_CONDITIONAL_FINITE_WINDOW_WET_MASS_PROXY_"
        "DRY_STEADY_TARGET_NOT_MEASURED_EQUILIBRIUM"
    )
    assert water["anchor_loading_mol_kg"] == pytest.approx(19.69049911427419)
    assert water["provenance"]["water_proxy_scale_sensitivity"] == [
        0.5,
        1.0,
        2.0,
    ]
    assert water["provenance"]["common_window_cutoff_class"] == (
        "EXTRACTION_SUPPORT_NOT_THERMODYNAMIC_ENDPOINT"
    )
    assert water["provenance"]["common_window_end_relative_values"] == {
        "CO2_C_over_C0": 0.838,
        "H2O_C_over_C0": 0.892,
    }
    assert water["provenance"]["extended_trace_sensitivity"][
        "conditional_increment_mol_kg_reported_wet_charge"
    ] == pytest.approx(20.7023)
    assert water["provenance"]["measured_dry_equilibrium_water_anchor"] is None
    independent_proxy = (
        (22.0 / 22414.0) * (0.7 * 611.21 / 100000.0) * 281.33008070685105 / 0.00006
    )
    assert independent_proxy == pytest.approx(19.69049911427419)
    assert water["provenance"]["derived_water_increment_decision"] == (
        "REFUSED_ENDPOINT_MISMATCH"
    )
    assert coupling["kind"] == ("water_activated_temperature_langmuir_added_target")
    assert coupling["added_capacity_mol_kg"] == pytest.approx(11.01086841855594)
    assert coupling["reference_added_target_mol_kg"] == pytest.approx(
        3.4800981489626492
    )
    assert coupling["water_normalizer_mol_kg"] == pytest.approx(19.69049911427419)
    assert coupling["co2_heat_J_mol"] == 70000.0
    assert coupling["co2_heat_sensitivity_J_mol"] == [
        35000.0,
        70000.0,
        105000.0,
    ]
    assert coupling["parameter_ranges"]["water_proxy_scale"] == [
        0.5,
        1.0,
        2.0,
    ]
    assert any("wet sample mass" in item for item in coupling["assumptions"])
    adapter = build_material_adapter(tepa)
    partials = np.array([40.0, 0.7 * saturation_pressure_Pa(298.15), 79000.0, 21000.0])
    assert adapter.equilibrium(298.15, partials)[0] == pytest.approx(3.9)
    hot = adapter._target_from_coupling_loading(
        389.15,
        partials,
        adapter.water_law.loading(389.15, partials[1]),
        adapter.water_law.loading(389.15, partials[1]),
    )
    assert np.all(np.isfinite(hot))
    assert 0.0 <= hot[0] <= 3.9
    coupling_adapter = adapter.coupling
    for temperature_K in (250.0, 389.15, 650.0):
        for co2_partial_Pa in (0.0, 40.0, 1.0e12):
            value = coupling_adapter.added_loading(
                temperature_K,
                co2_partial_Pa,
                water["anchor_loading_mol_kg"],
            )
            assert np.isfinite(value)
            assert 0.0 <= value <= coupling_adapter.added_capacity_mol_kg

    assert "tepa50_parent_water_transfer_hypothesis_v1" not in {
        row["id"] for row in load_defaults()["built_in_materials"]
    }
    registry = read_json_strict(TEMPERATURE_REGISTRY_PATH)
    assert "tepa50_parent_water_transfer_hypothesis_v1" not in {
        row["material_id"] for row in registry["records"].values()
    }
    with pytest.raises(MofScreeningError, match="unknown built-in material"):
        load_builtin_material("tepa50_parent_water_transfer_hypothesis_v1")

    for material_id in (
        "mil101_pei800_control_measured_water_diagnostic_v1",
        "mil101_pei800_glycerin_measured_water_diagnostic_v1",
    ):
        mil = load_builtin_material(material_id)
        extension = mil["humidity_candidate"]
        assert extension["evidence_tier"] == (
            "MEASURED_UNARY_WATER_WITH_ONE_WAY_CO2_SHIFT"
        )
        assert extension["co2_coupling"]["kind"] == ("water_fraction_quadratic")


@pytest.mark.parametrize(
    ("mutation", "field_part"),
    [
        (
            lambda material: material["co2"].__setitem__(
                "heat_J_mol", material["co2"]["heat_J_mol"] + 1000.0
            ),
            "dry_origin_material_id",
        ),
        (
            lambda material: material["water"].__setitem__(
                "basis", material["water"]["basis"] + " mutated"
            ),
            "dry_origin_material_id",
        ),
        (
            lambda material: material["kinetics"]["adsorption_rate_s_inv"].__setitem__(
                0, 9.0
            ),
            "dry_origin_material_id",
        ),
        (
            lambda material: material["adsorbed_phase_cp_J_kg_K"].__setitem__(0, 999.0),
            "dry_origin_material_id",
        ),
        (
            lambda material: material["humidity_candidate"]["storage"][
                "labels"
            ].__setitem__(0, "CO2_tampered"),
            "storage.labels",
        ),
    ],
)
def test_candidate_cannot_mutate_bound_dry_science(mutation, field_part):
    material = deepcopy(load_builtin_material("cof1000_humid_water_shift_scenario_v1"))
    mutation(material)
    with pytest.raises(MofScreeningError) as captured:
        validate_material(material)
    assert field_part in captured.value.field


def test_extension_hypothesis_edits_remain_explicit_and_registry_gated():
    material = deepcopy(load_builtin_material("mof808_lys_humid_one_node_scenario_v1"))
    material["humidity_candidate"]["co2_coupling"]["coefficients"][0] += 0.01
    validated = validate_material(material)
    case = deepcopy(load_defaults()["case"])
    case["temperature_policy"] = {
        "schema": "mof-screening-temperature-policy/1",
        "mode": "record_scenario_excursions",
    }
    with pytest.raises(MofScreeningError, match="exact reviewed package record"):
        effective_temperature_policy(case, validated)


def test_factory_uses_bound_dry_origin_even_if_direct_input_is_tampered():
    clean = load_builtin_material("cof999_humid_water_shift_scenario_v1")
    tampered = deepcopy(clean)
    tampered["co2"]["reference_curve"]["loading_mol_kg"] = [
        value * 2.0 for value in tampered["co2"]["reference_curve"]["loading_mol_kg"]
    ]
    partials = np.array([40.0, 0.0, 79000.0, 21000.0])
    clean_target = build_material_adapter(clean).equilibrium(298.15, partials)
    tampered_target = build_material_adapter(tampered).equilibrium(298.15, partials)
    np.testing.assert_array_equal(tampered_target, clean_target)


def test_hot_accepted_state_diagnostics_use_candidate_saturation_law():
    adapter = build_material_adapter(
        load_builtin_material("tepa50_finite_window_water_proxy_scenario_v1")
    )
    temperatures = np.array([389.15, 389.15])
    water_partial = np.array([500.0, 750.0])
    engine_diagnostic = diagnostic_water_saturation_ratios(
        adapter,
        temperatures,
        water_partial,
    )
    assert np.all(np.isfinite(engine_diagnostic["saturation_pressures_Pa"]))
    assert np.all(np.isfinite(engine_diagnostic["saturation_ratios"]))
    assert engine_diagnostic["saturation_pressures_Pa"][0] > 100000.0
    assert "IAPWS" in engine_diagnostic["saturation_diagnostic_law"]

    accepted = np.zeros((6, 2), dtype=float)
    accepted[0, :] = 389.15
    accepted[1:5, :] = np.array([[10.0], [0.1], [7.9], [2.0]])
    closure = {
        "indices": {
            "TEMP": 0,
            "GAS": slice(1, 5),
            "Q": [5],
            "QW": 5,
        },
        "pressure": lambda _state: 100000.0,
        "pump_evaluate": lambda _state: {
            "inlet_temperature_K": 328.15,
            "pressure_Pa": 20000.0,
            "media_temperature_overrun_K": 0.0,
            "upper_domain_exceeded": False,
            "pressure_regime": "test",
            "below_or_at_ultimate": False,
        },
        "pump_assumptions": {
            "profile_name": "test",
            "ultimate_pressure_Pa": 1000.0,
        },
        "pump_conditioner_identity": None,
    }
    weather_ready = weather_driver._diagnostics(
        type("Solution", (), {"y": accepted})(),
        closure,
        adapter,
        "desorption",
    )
    assert weather_ready["maximum_temperature_K"] == pytest.approx(389.15)
    assert weather_ready["flags"]["correlation_domain_violation"] is False
    assert "IAPWS" in weather_ready["saturation_diagnostic_law"]
    assert "candidate p_sat" in weather_ready["relative_humidity_basis"]


def test_diagnostic_saturation_selection_preserves_legacy_material_behavior():
    legacy = build_material_adapter(
        load_builtin_material("mof808_lys_2024_25c_dry_conditional")
    )
    assert not hasattr(legacy, "diagnostic_saturation_pressure_Pa")
    engine_diagnostic = diagnostic_water_saturation_ratios(
        legacy,
        np.array([298.15]),
        np.array([1000.0]),
    )
    expected_pressure = saturation_pressure_Pa(298.15)
    assert engine_diagnostic["saturation_pressures_Pa"][0] == pytest.approx(
        expected_pressure
    )
    assert engine_diagnostic["saturation_diagnostic_law"] == (
        "screening.common NIST/Stull Antoine"
    )
    weather_relative_humidity = weather_driver._diagnostic_relative_humidity(
        legacy,
        1000.0,
        298.15,
    )
    assert weather_relative_humidity == pytest.approx(
        1000.0 / weather_driver.buck_saturation_pressure_Pa(298.15)
    )
    assert weather_driver._diagnostic_relative_humidity_basis(legacy) == (
        "Buck 1996 saturation at T_bed"
    )


def test_spatial_trace_uses_gas_temperature_and_candidate_saturation():
    defaults = load_defaults()
    material, case = build_case(
        load_builtin_material("tepa50_finite_window_water_proxy_scenario_v1"),
        defaults["case"],
    )
    adapter = build_material_adapter(material)
    weather = load_weather_window(case["weather"]).rows[0]
    config = build_spatial_config(material, case, weather)
    geometry = build_bare_geometry(config)
    model = build_model(config, adapter, geometry)
    state = initial_state(config, adapter, geometry, model)
    state[model.layout.T_gas] = 389.15
    state[model.layout.T_bed] = 370.0
    rows = _trace_rows(
        model,
        [0.0],
        [state],
        ["adsorption"],
        material=adapter,
    )
    assert len(rows) == 1
    n_water = state[model.layout.n_gas].reshape(model.n_cells, 4)[:, 1]
    expected_partial = 8.314462618 * 389.15 * n_water / model.cell_gas_volume_m3
    expected_rh = expected_partial / np.array(
        [adapter.diagnostic_saturation_pressure_Pa(370.0) for _ in range(model.n_cells)]
    )
    assert rows[0]["local_rh_mean"] == pytest.approx(float(np.mean(expected_rh)))
    bed_temperature_rh = expected_partial / np.array(
        [
            adapter.diagnostic_saturation_pressure_Pa(389.15)
            for _ in range(model.n_cells)
        ]
    )
    assert not np.isclose(
        float(np.mean(expected_rh)),
        float(np.mean(bed_temperature_rh)),
    )
    assert "T_gas" in rows[0]["local_relative_humidity_basis"]
    assert "IAPWS" in rows[0]["saturation_diagnostic_law"]


def test_temperature_registry_is_exact_for_every_candidate():
    defaults = load_defaults()
    for candidate_id in CANDIDATES:
        material = load_builtin_material(candidate_id)
        preview_case = next(
            preset["case"]
            for preset in defaults["case_presets"]
            if preset["id"] == "humid_candidate_common_48h_rh050_exploratory"
        )
        policy = effective_temperature_policy(preview_case, material)
        assert policy["registry_matched"] is True
        assert policy["registry_classification"] == ("unqualified_scenario_guard")
        assert policy["package_canonical_sha256"] == canonical_sha256(material)


def test_common_preset_and_preview_survive_result_annotation():
    defaults = load_defaults()
    preset = next(
        row
        for row in defaults["case_presets"]
        if row["id"] == "humid_candidate_common_48h_rh050_exploratory"
    )
    assert preset["case_sha256"] == canonical_sha256(preset["case"])
    assert preset["case"]["weather"]["hours"] == 48
    assert preset["case"]["weather"]["relative_humidity"] == 0.5
    assert preset["case"]["cycle"]["regeneration_temperature_C"] == 116
    assert preset["case"]["cycle"]["regeneration_pressure_Pa"] == 16000

    material = load_builtin_material("cof1000_humid_water_shift_scenario_v1")
    material, case = build_case(material, preset["case"])
    preview = preview_payload(material, case, source="native-test")
    assert preview["material_sha256"] == canonical_sha256(material)
    assert preview["case_sha256"] == canonical_sha256(case)
    assert preview["material"]["humidity_candidate"] == (material["humidity_candidate"])
    assert any("EXPLORATORY HUMIDITY CANDIDATE" in item for item in preview["warnings"])
    adapter_report = build_material_adapter(material).source_report()
    result = annotate_result(
        {
            "schema": "mof-screening-result/1",
            "input_identity": {
                "material_sha256": preview["material_sha256"],
                "case_sha256": preview["case_sha256"],
            },
            "material": {
                "material_id": material["material_id"],
                "adapter": adapter_report,
            },
            "material_package": material,
            "case": case,
            "checks": {},
        }
    )
    assert result["material_package"]["humidity_candidate"] == (
        material["humidity_candidate"]
    )
    assert result["material"]["adapter"]["validated_humid_cycle"] is False
    assert result["material"]["adapter"]["runtime_research_imports"] is False
    assert result["input_identity"]["material_sha256"] == canonical_sha256(material)


def test_production_preview_and_json_export_preserve_humid_identity(tmp_path):
    defaults = load_defaults()
    preset = next(
        row
        for row in defaults["case_presets"]
        if row["id"] == "humid_candidate_common_48h_rh050_exploratory"
    )
    material = load_builtin_material("mof808_lys_humid_one_node_scenario_v1")
    manager = MofScreeningManager(
        tmp_path / "jobs", execution_lock_path=tmp_path / "LOCK.json"
    )
    try:
        preview = manager.preview(
            material,
            preset["case"],
            source="built_in_or_imported",
        )
        preview_path = manager.preview_root / f"{preview['preview_id']}.json"
        persisted_preview = json.loads(preview_path.read_text(encoding="utf-8"))
        assert persisted_preview["material_sha256"] == canonical_sha256(
            persisted_preview["material"]
        )
        assert persisted_preview["case_sha256"] == canonical_sha256(
            persisted_preview["case"]
        )
        assert (
            persisted_preview["material"]["humidity_candidate"]["validated_humid_cycle"]
            is False
        )
        reloaded = manager._load_preview(preview["preview_id"])
        assert reloaded["material"]["material_id"] == (
            "mof808_lys_humid_one_node_scenario_v1"
        )
        assert canonical_sha256(reloaded["material"]) == reloaded["material_sha256"]

        job_id = "a" * 32
        job_root = manager.job_root / job_id
        job_root.mkdir()
        adapter = build_material_adapter(persisted_preview["material"])
        hot_diagnostic = diagnostic_water_saturation_ratios(
            adapter,
            np.array([389.15]),
            np.array([750.0]),
        )
        result = {
            "schema": "mof-screening-result/1",
            "job_id": job_id,
            "preview_id": persisted_preview["preview_id"],
            "case_label": persisted_preview["case"]["label"],
            "engine": "spatial",
            "engine_identity": "offline-contract-fixture",
            "source_identity": execution_source_identity(),
            "input_identity": {
                "material_sha256": persisted_preview["material_sha256"],
                "case_sha256": persisted_preview["case_sha256"],
                "combined_sha256": canonical_sha256(
                    {
                        "material": persisted_preview["material"],
                        "case": persisted_preview["case"],
                    }
                ),
            },
            "preview_material_sha256": persisted_preview["material_sha256"],
            "preview_case_sha256": persisted_preview["case_sha256"],
            "material": {
                "material_id": persisted_preview["material_id"],
                "name": persisted_preview["material_name"],
                "adapter": adapter.source_report(),
            },
            "material_package": persisted_preview["material"],
            "case": persisted_preview["case"],
            "checks": {},
            "horizon": {},
            "endpoint": {},
            "cycles": [],
            "traces": [
                {
                    "bed_temperature_min_K": 389.15,
                    "bed_temperature_mean_K": 389.15,
                    "bed_temperature_max_K": 389.15,
                    "local_rh_mean": float(hot_diagnostic["saturation_ratios"][0]),
                    "local_relative_humidity_basis": (
                        adapter.diagnostic_relative_humidity_basis
                    ),
                    "saturation_diagnostic_law": hot_diagnostic[
                        "saturation_diagnostic_law"
                    ],
                }
            ],
            "water_burden": {
                "local_relative_humidity_basis": (
                    adapter.diagnostic_relative_humidity_basis
                ),
                "saturation_diagnostic_law": hot_diagnostic[
                    "saturation_diagnostic_law"
                ],
                "maximum_local_relative_humidity": float(
                    hot_diagnostic["saturation_ratios"][0]
                ),
            },
        }
        (job_root / "STATE.json").write_text(
            json.dumps(
                {
                    "schema": "mof-screening-job-state/1",
                    "job_id": job_id,
                    "state": "completed",
                    "progress": 1.0,
                    "message": "Offline contract fixture",
                    "updated_utc": "2026-11-08T00:00:00+00:00",
                }
            ),
            encoding="utf-8",
        )
        (job_root / "RESULT.json").write_text(json.dumps(result), encoding="utf-8")
        text, filename, content_type = manager.export(job_id, "json")
        exported = json.loads(text)
        assert filename.endswith(".json")
        assert content_type.startswith("application/json")
        assert exported["preview_material_sha256"] == (
            persisted_preview["material_sha256"]
        )
        assert exported["material_package"]["material_id"] == (
            "mof808_lys_humid_one_node_scenario_v1"
        )
        assert exported["material"]["adapter"]["dry_origin_material_id"] == (
            "mof808_lys_2024_25c_dry_conditional"
        )
        assert exported["material"]["adapter"]["runtime_research_imports"] is False
        assert exported["traces"][0]["bed_temperature_mean_K"] == 389.15
        assert "IAPWS" in exported["traces"][0]["saturation_diagnostic_law"]
        assert "T_gas" in exported["water_burden"]["local_relative_humidity_basis"]
    finally:
        manager.close()


def test_preflight_source_identity_includes_native_candidates_and_keeps_closure():
    identity = execution_source_identity()
    expected_paths = {
        "screening/mof_screening/stored_water_shift.py",
        "screening/mof_screening/contract.py",
        "screening/mof_screening/materials.py",
        "screening/mof_screening/preflight.py",
        "screening/mof_screening/runtime.py",
        "screening/mof_screening/lumped_port/engine.py",
        "screening/mof_screening/lumped_port/weather_driver.py",
        "app/mof_screening/static/app.js",
        "app/mof_screening/static/index.html",
        "app/mof_screening/static/case_presets.js",
        "config/mof_screening/defaults.json",
        "config/mof_screening/temperature_registry.json",
        *(f"materials/screening/{material_id}.json" for material_id in CANDIDATES),
    }
    for relative_path in expected_paths:
        assert relative_path in identity["source_hashes"]
    closure = dependency_status()
    assert closure["valid"] is True
    assert identity["frozen_closure"] == closure
