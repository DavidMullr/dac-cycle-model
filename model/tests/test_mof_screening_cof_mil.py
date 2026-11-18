from copy import deepcopy
import csv
from pathlib import Path

import numpy as np
import pytest

from screening.common import CP_GAS, MW
from screening.geometry import build_bare_geometry
from screening.mof_screening.contract import (
    MofScreeningError,
    ROOT,
    build_case,
    load_builtin_material,
    load_defaults,
    validate_material,
)
from screening.mof_screening.guided_material import build_guided_material
from screening.mof_screening.lumped_port.engine import run_cycle
from screening.mof_screening.materials import build_material_adapter
from screening.mof_screening.runtime import (
    build_lumped_config,
    build_spatial_config,
)
from screening.mof_screening.spatial import (
    STEPS,
    build_model,
    cycle_rhs,
    energy_inventory,
    initial_state,
)
from screening.mof_screening.weather import WeatherHour


COF_MATERIAL_IDS = (
    "cof1000_2025_25c_split_rate_dry_conditional",
    "cof999_2024_25c_dry_conditional",
)
MIL_MATERIAL_IDS = (
    "mil101_pei800_30_glycerin_2026_25c_dry_conditional",
    "mil101_pei800_30_2026_25c_dry_conditional",
)
NEW_MATERIAL_IDS = COF_MATERIAL_IDS + MIL_MATERIAL_IDS


def _dry_weather(temperature_C=25.0, co2_ppm=40.0):
    return WeatherHour(
        interval_start_utc="2000-01-01T00:00:00+00:00",
        interval_end_utc="2000-01-01T01:00:00+00:00",
        temperature_C=temperature_C,
        relative_humidity=0.0,
        pressure_Pa=100000.0,
        co2_ppm_dry=co2_ppm,
    )


def _configs(material, *, cells=4, temperature_C=25.0, co2_ppm=40.0):
    case = deepcopy(load_defaults()["case"])
    case["weather"].update(
        {
            "temperature_C": temperature_C,
            "relative_humidity": 0.0,
            "co2_ppm_dry": co2_ppm,
        }
    )
    case["solver"]["cells"] = cells
    material, case = build_case(material, case)
    weather = _dry_weather(temperature_C, co2_ppm)
    spatial = build_spatial_config(material, case, weather)
    lumped_case = deepcopy(case)
    lumped_case["engine"] = "lumped"
    lumped_case["solver"]["cells"] = 1
    lumped = build_lumped_config(material, lumped_case, weather)
    return material, case, spatial, lumped


def test_cof_and_mil_packages_preserve_sources_classes_and_gaps():
    assert {
        load_builtin_material(material_id)["material_class"]
        for material_id in COF_MATERIAL_IDS
    } == {"COF"}
    assert {
        load_builtin_material(material_id)["material_class"]
        for material_id in MIL_MATERIAL_IDS
    } == {"MOF composite"}

    cof1000 = load_builtin_material(COF_MATERIAL_IDS[0])
    curve = cof1000["co2"]["reference_curve"]
    source = ROOT / "tests/fixtures/mof_screening/source_tables" / Path(cof1000["provenance"][
        "co2_reference_curve"
    ]["source_asset"]).name
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(curve["pressure_Pa"]) == 36
    assert curve["pressure_Pa"] == [
        float(row["pressure_Pa"]) for row in rows
    ]
    assert curve["loading_mol_kg"] == [
        float(row["loading_mol_kg"]) for row in rows
    ]
    assert cof1000["co2"]["reservoir_order"] == ["fast", "slow"]
    assert sum(cof1000["co2"]["reservoir_fractions"]) == pytest.approx(1.0)
    assert cof1000["provenance"]["kinetics"][
        "experimental_lag_applied_to_device"
    ] is False
    assert cof1000["material_class"] == "COF"

    cof999 = load_builtin_material(COF_MATERIAL_IDS[1])
    assert len(cof999["co2"]["reference_curve"]["pressure_Pa"]) == 26
    assert cof999["provenance"]["co2_temperature_shift"][
        "source_estimate_loading_limit_mol_kg"
    ] == 1.0
    assert cof999["provenance"]["co2_temperature_shift"][
        "global_use_is_assumption"
    ] is True
    assert cof999["provenance"]["solid_cp"][
        "candidate_specific_measurement"
    ] is False

    for material_id, withheld, point_count in (
        (MIL_MATERIAL_IDS[0], 0.4355820244, 14),
        (MIL_MATERIAL_IDS[1], 1.5070558884, 15),
    ):
        material = load_builtin_material(material_id)
        shift = material["provenance"]["co2_temperature_shift"]
        assert (
            len(material["co2"]["reference_curve"]["pressure_Pa"])
            == point_count
        )
        assert shift["withheld_temperature_K"] == 333.15
        assert shift["withheld_relative_RMS"] == pytest.approx(withheld)
        assert shift["required_withheld_relative_RMS_max"] == 0.30
        assert shift["gate_pass"] is False
        assert material["provenance"]["inert_gases"]["class"] == (
            "assumed executable zero"
        )
        assert material["missing_evidence"][
            "measured_true_inert_gas_henry_coefficients"
        ] is None

    for material_id in NEW_MATERIAL_IDS:
        material = load_builtin_material(material_id)
        assert material["water"]["model"] == "dry_only"
        assert material["evidence"]["status"].startswith("CONDITIONAL")
        assert material["provenance"]["inert_gases"]["class"] == (
            "assumed executable zero"
        )
        assert material["nitrogen"] == {
            "henry_mol_kg_Pa_at_reference": 0.0,
            "heat_J_mol": 0.0,
        }
        assert material["oxygen"] == {
            "henry_mol_kg_Pa_at_reference": 0.0,
            "heat_J_mol": 0.0,
        }
        for key in (
            "measured_true_inert_gas_henry_coefficients",
            "measured_true_inert_gas_isosteric_heats",
            "measured_true_inert_gas_sorption_rates",
        ):
            assert material["missing_evidence"][key] is None


def test_split_contract_rejects_order_fraction_and_water_semantics():
    source = deepcopy(load_builtin_material(COF_MATERIAL_IDS[0]))

    wrong_order = deepcopy(source)
    wrong_order["co2"]["reservoir_order"] = ["slow", "fast"]
    with pytest.raises(MofScreeningError) as order_error:
        validate_material(wrong_order)
    assert order_error.value.field == "material.co2.reservoir_order"

    wrong_sum = deepcopy(source)
    wrong_sum["co2"]["reservoir_fractions"] = [0.2, 0.2]
    with pytest.raises(MofScreeningError) as sum_error:
        validate_material(wrong_sum)
    assert sum_error.value.field == "material.co2.reservoir_fractions"

    wet_rate = deepcopy(source)
    wet_rate["kinetics"]["adsorption_rate_s_inv"][2] = 1.0e-3
    with pytest.raises(MofScreeningError) as rate_error:
        validate_material(wet_rate)
    assert rate_error.value.field == (
        "material.kinetics.adsorption_rate_s_inv[2]"
    )

    defaults = load_defaults()
    with pytest.raises(MofScreeningError) as humidity_error:
        build_case(source, defaults["case"])
    assert humidity_error.value.field == "case.weather.relative_humidity"

    non_dry = deepcopy(source)
    non_dry["water"] = {
        "model": "unary_sips",
        "saturation_mol_kg": 18.0,
        "half_loading_relative_humidity_at_reference": 0.5,
        "sips_exponent": 2.0,
        "heat_J_mol": 42000.0,
        "basis": "fully populated clone retained to prove it is rejected",
    }
    with pytest.raises(MofScreeningError) as non_dry_error:
        validate_material(non_dry)
    assert non_dry_error.value.field == "material.water.model"

    missing_mode = deepcopy(source)
    missing_mode["water"].pop("model")
    with pytest.raises(MofScreeningError) as missing_error:
        validate_material(missing_mode)
    assert missing_error.value.field == "material.water.model"


def test_split_water_energy_and_site_rates_are_zero_at_positive_trial():
    material = load_builtin_material(COF_MATERIAL_IDS[0])
    adapter = build_material_adapter(material)
    partials = np.array([40.0, 1000.0, 79000.0, 21000.0])
    storage = adapter.equilibrium(298.15, partials) * 0.25
    storage[adapter.water_storage_index] = 0.25

    _, species_rates = adapter.rates(330.0, partials, storage)
    helper_rates = adapter.site_rates(
        330.0,
        partials,
        storage[: adapter.n_sites],
        float(storage[adapter.water_storage_index]),
        storage[[3, 4]],
    )

    assert adapter.water_storage_index == 2
    assert storage[adapter.water_storage_index] > 0.0
    assert helper_rates[adapter.n_sites] == 0.0
    assert species_rates[1] == 0.0
    for temperature_K in (255.9, 298.15, 330.0, 393.15):
        absolute_energy = adapter.absolute_energy_J_mol(temperature_K)
        assert absolute_energy[adapter.water_storage_index] == 0.0
        assert adapter.component_cp_molar_J_mol_K()[
            adapter.water_storage_index
        ] == 0.0


def test_split_positive_water_solver_trials_are_invariant_in_both_engines():
    material = load_builtin_material(COF_MATERIAL_IDS[0])
    adapter = build_material_adapter(material)
    _, _, spatial_config, lumped_config = _configs(material, cells=4)
    geometry = build_bare_geometry(spatial_config)
    model = build_model(spatial_config, adapter, geometry)
    base_state = initial_state(spatial_config, adapter, geometry, model)
    trial_state = base_state.copy()
    q_base = base_state[model.layout.q].reshape(
        model.n_cells, adapter.n_storage
    ).copy()
    q_trial = q_base.copy()
    q_trial[:, adapter.water_storage_index] += 0.25
    trial_state[model.layout.q] = q_trial.ravel()

    base_energy = energy_inventory(
        spatial_config, adapter, geometry, base_state, model
    )["total_J"]
    trial_energy = energy_inventory(
        spatial_config, adapter, geometry, trial_state, model
    )["total_J"]
    assert trial_energy == pytest.approx(base_energy, rel=0.0, abs=1.0e-9)
    for phase in STEPS:
        derivative, _ = cycle_rhs(
            spatial_config, adapter, geometry, phase, model=model
        )(0.0, trial_state)
        water_derivative = derivative[model.layout.q].reshape(
            model.n_cells, adapter.n_storage
        )[:, adapter.water_storage_index]
        assert np.all(water_derivative == 0.0), phase
        assert np.all(np.isfinite(derivative)), phase

    bundle = run_cycle(
        adapter,
        lumped_config,
        relative_humidity=0.0,
        rhs_only=True,
    )
    lumped_base = bundle["initial_state"]()
    lumped_trial = lumped_base.copy()
    indices = bundle["indices"]
    base_inventory = bundle["inventory"](lumped_base)
    lumped_trial[indices["QW"]] = 0.25
    assert bundle["inventory"](lumped_trial) == pytest.approx(
        base_inventory, rel=0.0, abs=1.0e-9
    )
    for phase, rhs in bundle["rhs_by_phase"].items():
        derivative = np.asarray(rhs(0.0, lumped_trial), dtype=float)
        assert derivative[indices["QW"]] == 0.0, phase
        assert np.all(np.isfinite(derivative)), phase


def test_split_equal_rate_matches_aggregate_one_pool_adapter():
    split = deepcopy(load_builtin_material(COF_MATERIAL_IDS[0]))
    rate = 0.004
    split["kinetics"]["adsorption_rate_s_inv"] = [rate, rate, 0.0, 0.0, 0.0]
    split["kinetics"]["desorption_rate_s_inv"] = [rate, rate, 0.0, 0.0, 0.0]
    aggregate = deepcopy(split)
    aggregate["equilibrium_family"] = "activity_shift_reference_isotherm"
    aggregate["co2"].pop("reservoir_order")
    aggregate["co2"].pop("reservoir_fractions")
    aggregate["kinetics"]["adsorption_rate_s_inv"] = [rate, 0.0, 0.0, 0.0]
    aggregate["kinetics"]["desorption_rate_s_inv"] = [rate, 0.0, 0.0, 0.0]
    aggregate["adsorbed_phase_cp_J_kg_K"] = [
        CP_GAS[0] / MW[0],
        0.0,
        0.0,
        0.0,
    ]

    split_adapter = build_material_adapter(validate_material(split))
    aggregate_adapter = build_material_adapter(validate_material(aggregate))
    partials = np.array([40.0, 1.0, 79000.0, 21000.0])

    split_equilibrium = split_adapter.equilibrium(350.0, partials)
    aggregate_equilibrium = aggregate_adapter.equilibrium(350.0, partials)
    assert split_adapter.storage_labels == (
        "CO2_fast",
        "CO2_slow",
        "H2O",
        "N2",
        "O2",
    )
    assert split_adapter.species_for_storage.tolist() == [0, 0, 1, 2, 3]
    assert split_adapter.water_storage_index == 2
    np.testing.assert_allclose(
        [
            split_equilibrium[0] + split_equilibrium[1],
            *split_equilibrium[2:],
        ],
        aggregate_equilibrium,
        rtol=0.0,
        atol=1e-14,
    )

    split_storage = split_equilibrium * 0.37
    aggregate_storage = aggregate_equilibrium * 0.37
    split_storage[split_adapter.water_storage_index] = 0.25
    aggregate_storage[aggregate_adapter.water_storage_index] = 0.25
    split_rates, split_species = split_adapter.rates(
        350.0, partials, split_storage
    )
    aggregate_rates, aggregate_species = aggregate_adapter.rates(
        350.0, partials, aggregate_storage
    )
    np.testing.assert_allclose(
        split_rates[0] + split_rates[1],
        aggregate_rates[0],
        rtol=0.0,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        [split_species[0], *split_species[1:]],
        aggregate_species,
        rtol=0.0,
        atol=1e-14,
    )
    assert split_rates[split_adapter.water_storage_index] == 0.0
    assert aggregate_rates[aggregate_adapter.water_storage_index] == 0.0
    split_u = split_adapter.absolute_energy_J_mol(365.0)
    aggregate_u = aggregate_adapter.absolute_energy_J_mol(365.0)
    np.testing.assert_allclose(
        [
            split_storage[0] * split_u[0]
            + split_storage[1] * split_u[1],
            split_storage[2] * split_u[2],
            split_storage[3] * split_u[3],
            split_storage[4] * split_u[4],
        ],
        aggregate_storage * aggregate_u,
        rtol=0.0,
        atol=1e-10,
    )
    split_cp = split_adapter.component_cp_molar_J_mol_K()
    aggregate_cp = aggregate_adapter.component_cp_molar_J_mol_K()
    np.testing.assert_allclose(
        [
            split_storage[0] * split_cp[0] + split_storage[1] * split_cp[1],
            *(split_storage[2:] * split_cp[2:]),
        ],
        aggregate_storage * aggregate_cp,
        rtol=0.0,
        atol=1e-10,
    )


@pytest.mark.parametrize("cells", (4, 16))
@pytest.mark.parametrize(
    ("temperature_C", "co2_ppm"),
    ((0.0, 40.0), (25.0, 40.0), (25.0, 1000.0), (74.0, 1000.0)),
)
@pytest.mark.parametrize("material_id", NEW_MATERIAL_IDS)
def test_new_packages_have_finite_both_engine_all_phase_rhs(
    material_id, cells, temperature_C, co2_ppm
):
    material = load_builtin_material(material_id)
    adapter = build_material_adapter(material)
    material, case, spatial_config, lumped_config = _configs(
        material,
        cells=cells,
        temperature_C=temperature_C,
        co2_ppm=co2_ppm,
    )
    geometry = build_bare_geometry(spatial_config)
    model = build_model(spatial_config, adapter, geometry)
    state = initial_state(spatial_config, adapter, geometry, model)
    for phase in STEPS:
        derivative, _ = cycle_rhs(
            spatial_config, adapter, geometry, phase, model=model
        )(0.0, state)
        assert derivative.shape == state.shape, (material_id, cells, phase)
        assert np.all(np.isfinite(derivative)), (material_id, cells, phase)

    bundle = run_cycle(
        adapter,
        lumped_config,
        relative_humidity=0.0,
        rhs_only=True,
    )
    lumped_state = bundle["initial_state"]()
    for phase, rhs in bundle["rhs_by_phase"].items():
        derivative = np.asarray(rhs(0.0, lumped_state), dtype=float)
        assert derivative.shape == lumped_state.shape, (material_id, phase)
        assert np.all(np.isfinite(derivative)), (material_id, phase)


def _split_equal_rate_pair():
    split = deepcopy(load_builtin_material(COF_MATERIAL_IDS[0]))
    rate = 0.004
    split["kinetics"]["adsorption_rate_s_inv"] = [rate, rate, 0.0, 0.0, 0.0]
    split["kinetics"]["desorption_rate_s_inv"] = [rate, rate, 0.0, 0.0, 0.0]
    aggregate = deepcopy(split)
    aggregate["equilibrium_family"] = "activity_shift_reference_isotherm"
    aggregate["co2"].pop("reservoir_order")
    aggregate["co2"].pop("reservoir_fractions")
    aggregate["kinetics"]["adsorption_rate_s_inv"] = [rate, 0.0, 0.0, 0.0]
    aggregate["kinetics"]["desorption_rate_s_inv"] = [rate, 0.0, 0.0, 0.0]
    aggregate["adsorbed_phase_cp_J_kg_K"] = [
        CP_GAS[0] / MW[0],
        0.0,
        0.0,
        0.0,
    ]
    return validate_material(split), validate_material(aggregate)


@pytest.mark.parametrize("cells", (4, 16))
def test_equal_rate_split_rhs_maps_to_legacy_spatial_rhs(cells):
    split, aggregate = _split_equal_rate_pair()
    split_adapter = build_material_adapter(split)
    aggregate_adapter = build_material_adapter(aggregate)
    _, _, split_config, _ = _configs(split, cells=cells)
    _, _, aggregate_config, _ = _configs(aggregate, cells=cells)
    split_geometry = build_bare_geometry(split_config)
    aggregate_geometry = build_bare_geometry(aggregate_config)
    split_model = build_model(split_config, split_adapter, split_geometry)
    aggregate_model = build_model(
        aggregate_config, aggregate_adapter, aggregate_geometry
    )
    split_state = initial_state(
        split_config, split_adapter, split_geometry, split_model
    )
    aggregate_state = initial_state(
        aggregate_config,
        aggregate_adapter,
        aggregate_geometry,
        aggregate_model,
    )
    for phase in STEPS:
        split_rhs, _ = cycle_rhs(
            split_config,
            split_adapter,
            split_geometry,
            phase,
            model=split_model,
        )(0.0, split_state)
        aggregate_rhs, _ = cycle_rhs(
            aggregate_config,
            aggregate_adapter,
            aggregate_geometry,
            phase,
            model=aggregate_model,
        )(0.0, aggregate_state)
        split_q = split_rhs[split_model.layout.q].reshape(cells, 5)
        aggregate_q = aggregate_rhs[aggregate_model.layout.q].reshape(cells, 4)
        np.testing.assert_allclose(
            split_q[:, :2].sum(axis=1),
            aggregate_q[:, 0],
            rtol=1e-13,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            split_q[:, 2:],
            aggregate_q[:, 1:],
            rtol=1e-13,
            atol=1e-14,
        )
        for field in (
            "T_bed",
            "T_gas",
            "n_gas",
            "inlet",
            "vent",
            "product",
        ):
            np.testing.assert_allclose(
                split_rhs[getattr(split_model.layout, field)],
                aggregate_rhs[getattr(aggregate_model.layout, field)],
                rtol=1e-12,
                atol=1e-12,
                err_msg=f"{cells} {phase} {field}",
            )


def test_equal_rate_split_maps_to_legacy_lumped_rhs():
    split, aggregate = _split_equal_rate_pair()
    split_adapter = build_material_adapter(split)
    aggregate_adapter = build_material_adapter(aggregate)
    _, _, _, split_config = _configs(split, cells=4)
    _, _, _, aggregate_config = _configs(aggregate, cells=4)
    split_bundle = run_cycle(
        split_adapter,
        split_config,
        relative_humidity=0.0,
        rhs_only=True,
    )
    aggregate_bundle = run_cycle(
        aggregate_adapter,
        aggregate_config,
        relative_humidity=0.0,
        rhs_only=True,
    )
    split_state = split_bundle["initial_state"]()
    aggregate_state = aggregate_bundle["initial_state"]()
    split_indices = split_bundle["indices"]
    aggregate_indices = aggregate_bundle["indices"]
    for phase in STEPS:
        split_rhs = np.asarray(
            split_bundle["rhs_by_phase"][phase](0.0, split_state),
            dtype=float,
        )
        aggregate_rhs = np.asarray(
            aggregate_bundle["rhs_by_phase"][phase](0.0, aggregate_state),
            dtype=float,
        )
        assert np.sum(split_rhs[split_indices["Q"]]) == pytest.approx(
            np.sum(aggregate_rhs[aggregate_indices["Q"]]),
            rel=1e-12,
            abs=1e-14,
        )
        for field in (
            "QW",
            "QN",
            "QO",
            "TEMP",
            "GAS",
            "IN",
            "VENT",
            "PRODUCT",
        ):
            np.testing.assert_allclose(
                np.atleast_1d(split_rhs[split_indices[field]]),
                np.atleast_1d(aggregate_rhs[aggregate_indices[field]]),
                rtol=1e-12,
                atol=1e-12,
                err_msg=f"{phase} {field}",
            )


def test_split_caloric_closure_passes_independent_spatial_and_lumped_finite_differences():
    material = load_builtin_material(COF_MATERIAL_IDS[0])
    adapter = build_material_adapter(material)
    _, _, spatial_config, lumped_config = _configs(material, cells=4)
    geometry = build_bare_geometry(spatial_config)
    model = build_model(spatial_config, adapter, geometry)
    state = initial_state(spatial_config, adapter, geometry, model)
    partials = np.array([5000.0, 0.0, 70000.0, 25000.0])
    loading = adapter.equilibrium(370.0, partials)
    state[model.layout.T_bed] = 370.0
    state[model.layout.T_gas] = 360.0
    state[model.layout.T_w] = 365.0
    state[model.layout.T_h] = 380.0
    state[model.layout.q] = np.tile(loading, model.n_cells)
    derivative, observation = cycle_rhs(
        spatial_config,
        adapter,
        geometry,
        "desorption",
        model=model,
    )(0.0, state)
    epsilon = 1.0e-7
    before = energy_inventory(
        spatial_config, adapter, geometry, state, model
    )["total_J"]
    after = energy_inventory(
        spatial_config,
        adapter,
        geometry,
        state + epsilon * derivative,
        model,
    )["total_J"]
    finite_derivative = (after - before) / epsilon
    boundary = (
        observation["heater_W"]
        + observation["source_delivered_W"]
        - observation["cooling_W"]
        - observation["ambient_loss_W"]
        + derivative[model.layout.E_netflow]
    )
    assert abs(observation["first_law_residual_W"]) <= 1.0e-6
    assert abs(finite_derivative - boundary) / max(
        abs(finite_derivative), abs(boundary), 1.0
    ) <= 1.0e-4

    bundle = run_cycle(
        adapter,
        lumped_config,
        relative_humidity=0.0,
        rhs_only=True,
    )
    lumped_state = bundle["initial_state"]()
    indices = bundle["indices"]
    lumped_state[indices["TEMP"]] = 370.0
    lumped_state[indices["Q"]] = loading[indices["Q"]]
    lumped_state[indices["QW"]] = 0.0
    lumped_state[indices["QN"]] = loading[3]
    lumped_state[indices["QO"]] = loading[4]
    lumped_derivative = bundle["rhs_by_phase"]["desorption"](
        0.0, lumped_state
    )
    lumped_before = bundle["inventory"](lumped_state)
    lumped_after = bundle["inventory"](
        lumped_state + epsilon * lumped_derivative
    )
    lumped_finite = (lumped_after - lumped_before) / epsilon
    lumped_boundary = (
        lumped_derivative[indices["HEAT"]]
        - lumped_derivative[indices["COOL"]]
        - lumped_derivative[indices["LOSS"]]
        + lumped_derivative[indices["INLET_H"]]
        - lumped_derivative[indices["OUTLET_H"]]
    )
    assert abs(lumped_finite - lumped_boundary) / max(
        abs(lumped_finite), abs(lumped_boundary), 1.0
    ) <= 1.0e-4


def test_split_guided_roundtrip_preserves_ordered_fractions():
    source = load_builtin_material(COF_MATERIAL_IDS[0])
    curve = source["co2"]["reference_curve"]
    provenance = source["provenance"]
    declaration = lambda key: (
        provenance[key].get("statement") or provenance[key].get("class")
    )
    draft = {
        "base_material": source,
        "material_id": source["material_id"],
        "name": source["name"],
        "equilibrium_family": source["equilibrium_family"],
        "evidence_basis": source["evidence"]["basis"],
        "evidence_not_for": source["evidence"]["not_for"],
        "shaping_basis": source["shaped_material"]["basis"],
        "binder_support": source["shaped_material"]["binder_support"],
        "particle_geometry": source["shaped_material"]["particle_geometry"],
        "reference_temperature_K": source["reference_temperature_K"],
        "maximum_bed_temperature_C": source["maximum_bed_temperature_C"],
        "solid_cp_J_kg_K": source["solid_cp_J_kg_K"],
        "co2_evidence_type": "source",
        "water_evidence_type": "assumption",
        "thermal_evidence_type": "assumption",
        "kinetics_evidence_type": "assumption",
        "co2_declaration": declaration("co2"),
        "water_declaration": declaration("water"),
        "thermal_declaration": declaration("thermal"),
        "kinetics_declaration": declaration("kinetics"),
        "assumptions_text": "\n".join(source["assumptions"]),
        "co2_validity": source["validity"]["co2"],
        "water_validity": source["validity"]["water"],
        "thermal_validity": source["validity"]["thermal"],
        "kinetics_validity": source["validity"]["kinetics"],
        "uncertainty_classification": source["uncertainty"]["classification"],
        "uncertainty_statement": source["uncertainty"]["statement"],
        "adsorbed_phase_cp_J_kg_K": ", ".join(
            str(value) for value in source["adsorbed_phase_cp_J_kg_K"]
        ),
        "adsorption_rate_s_inv": ", ".join(
            str(value)
            for value in source["kinetics"]["adsorption_rate_s_inv"]
        ),
        "desorption_rate_s_inv": ", ".join(
            str(value)
            for value in source["kinetics"]["desorption_rate_s_inv"]
        ),
        "co2_heat_J_mol": str(source["co2"]["heat_J_mol"]),
        "co2_water_penalty_per_mol_kg": "0",
        "co2_curve_temperature_K": str(curve["temperature_K"]),
        "co2_curve_csv": "pressure_Pa,loading_mol_kg\n"
        + "\n".join(
            f"{pressure},{loading}"
            for pressure, loading in zip(
                curve["pressure_Pa"], curve["loading_mol_kg"]
            )
        ),
        "split_fast_fraction": str(
            source["co2"]["reservoir_fractions"][0]
        ),
        "split_slow_fraction": str(
            source["co2"]["reservoir_fractions"][1]
        ),
        "water_solver_normalizer_mol_kg": str(
            source["water"]["solver_normalizer_mol_kg"]
        ),
        "nitrogen_henry_mol_kg_Pa_at_reference": "0",
        "nitrogen_heat_J_mol": "0",
        "oxygen_henry_mol_kg_Pa_at_reference": "0",
        "oxygen_heat_J_mol": "0",
    }
    result = build_guided_material({"draft": draft})
    assert result["valid"] is True, result["issues"]
    material = result["material"]
    assert material["equilibrium_family"] == (
        "split_rate_activity_shift_reference_isotherm"
    )
    assert material["co2"]["reservoir_order"] == ["fast", "slow"]
    assert material["co2"]["reservoir_fractions"] == source["co2"][
        "reservoir_fractions"
    ]
    assert build_material_adapter(material).water_storage_index == 2
