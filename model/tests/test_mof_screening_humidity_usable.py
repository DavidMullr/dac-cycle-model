from copy import deepcopy
from types import MethodType

import numpy as np
import pytest

from screening.geometry import build_bare_geometry
from screening.mof_screening.contract import (
    build_case,
    load_builtin_material,
    load_defaults,
)
from screening.mof_screening.lumped_port.engine import run_cycle
from screening.mof_screening.materials import build_material_adapter
from screening.mof_screening.runtime import (
    build_lumped_config,
    build_spatial_config,
    run_lumped,
    run_spatial,
)
from screening.mof_screening.spatial import (
    build_model,
    cycle_rhs,
    energy_inventory,
    initial_state,
)
from screening.mof_screening.weather import load_weather_window
from screening.mof_screening import young2021
from screening.mof_screening import runtime as runtime_module
from screening.mof_screening.lumped_port import weather_driver as weather_driver_module
from screening.mof_screening.lumped_port.pressure_control import POLICY_VARIANT


YOUNG_ID = "lewatit_vp_oc_1065_young2021_wadst_conditional"


def _humidity_case():
    defaults = load_defaults()
    raw_material = load_builtin_material(YOUNG_ID)
    preset = next(
        value
        for value in defaults["case_presets"]
        if value.get("material_id") == YOUNG_ID
    )
    material, case = build_case(raw_material, deepcopy(preset["case"]))
    weather = load_weather_window(case["weather"])
    return defaults, preset, material, case, weather


def _cross_loading_adapter(base):
    def cross_terms(self, temperature_K, storage):
        q = np.asarray(storage, dtype=float)
        if q.shape != (self.n_storage,):
            raise ValueError("synthetic potential received wrong loading shape")
        unit_u = np.asarray(self.absolute_energy_J_mol(temperature_K))
        unit_cp = np.asarray(self.component_cp_molar_J_mol_K())
        coupling = 12_500.0
        energy = float(q @ unit_u) + coupling * float(q[0] * q[1])
        capacity = float(q @ unit_cp)
        gradient = unit_u.copy()
        gradient[0] += coupling * q[1]
        gradient[1] += coupling * q[0]
        return energy, capacity, gradient

    base.storage_energy_terms = MethodType(cross_terms, base)
    return base


def test_young_equilibrium_matches_reviewed_source_reproduction():
    expected = {
        (298.15, 0.30): (1.2162731778751266, 1.9620246764512157),
        (298.15, 0.55): (1.40594575653534, 4.131251972115188),
        (323.15, 0.30): (0.8147326947730584, 1.9233744718202945),
        (343.15, 0.55): (0.794081445477366, 3.6791626032380997),
    }
    for (temperature_K, rh), expected_values in expected.items():
        p_water = (
            rh * young2021.buck_saturation_pressure_Pa(temperature_K)
        )
        co2, water, details = young2021.equilibrium(
            temperature_K, 40.0, p_water
        )
        np.testing.assert_allclose(
            [co2, water], expected_values, rtol=0.0, atol=1e-14
        )
        assert details["local_relative_humidity"] == pytest.approx(rh)


def test_young_local_activity_domain_is_strict_and_uses_buck():
    temperature_K = 323.15
    p_sat = young2021.buck_saturation_pressure_Pa(temperature_K)
    _co2, water, _details = young2021.equilibrium(
        temperature_K, 40.0, 0.55 * p_sat
    )
    assert water > 0.0
    with pytest.raises(ValueError, match="relative_humidity"):
        young2021.equilibrium(
            temperature_K, 40.0, (0.55 + 2e-12) * p_sat
        )
    with pytest.raises(ValueError, match="temperature_K"):
        young2021.equilibrium(343.16, 40.0, 100.0)
    with pytest.raises(ValueError, match="co2_Pa"):
        young2021.equilibrium(323.15, 100_000.01, 100.0)
    with pytest.raises(ValueError, match="co2_Pa"):
        young2021.equilibrium(323.15, -1e-12, 100.0)
    with pytest.raises(ValueError, match="finite"):
        young2021.equilibrium(float("nan"), 40.0, 100.0)

    # A 25 C value produced by binary conversion may sit a few ulps below
    # 298.15 K. Admit only representation-sized roundoff.
    lower_state = 298.15 - 5e-10
    lower_p_sat = young2021.buck_saturation_pressure_Pa(lower_state)
    lower = young2021.equilibrium(lower_state, 40.0, 0.3 * lower_p_sat)
    assert lower[0] > 0.0 and lower[1] > 0.0
    with pytest.raises(ValueError, match="temperature_K"):
        young2021.equilibrium(
            298.15 - 2e-9,
            40.0,
            0.3 * young2021.buck_saturation_pressure_Pa(298.15 - 2e-9),
        )
    upper_p_sat = young2021.buck_saturation_pressure_Pa(343.15)
    with pytest.raises(ValueError, match="relative_humidity"):
        young2021.equilibrium(
            343.15, 40.0, (0.55 + 2e-12) * upper_p_sat
        )


def test_actual_water_inventory_drives_transient_co2_target():
    adapter = build_material_adapter(load_builtin_material(YOUNG_ID))
    temperature_K = 298.15
    partials = np.array(
        [
            40.0,
            0.30 * young2021.buck_saturation_pressure_Pa(temperature_K),
            0.0,
            0.0,
        ]
    )
    dry_state = np.array([0.5, 0.01])
    water_inventory_state = np.array([0.5, 1.5])
    dry_rates, _ = adapter.rates(temperature_K, partials, dry_state)
    water_rates, _ = adapter.rates(
        temperature_K, partials, water_inventory_state
    )
    dry_target = dry_state[0] + dry_rates[0] / adapter._ad_rate[0]
    water_target = (
        water_inventory_state[0]
        + water_rates[0] / adapter._ad_rate[0]
    )
    assert water_target > dry_target + 0.1

    low_co2_state = np.array([0.01, 1.5])
    high_co2_state = np.array([2.0, 1.5])
    low_rates, _ = adapter.rates(
        temperature_K, partials, low_co2_state
    )
    high_rates, _ = adapter.rates(
        temperature_K, partials, high_co2_state
    )
    assert low_rates[1] == pytest.approx(high_rates[1])


def test_young_caloric_reference_and_equal_ldf_flags_are_derived():
    material = load_builtin_material(YOUNG_ID)
    adapter = build_material_adapter(material)
    reference = 298.15
    q = np.array([1.25, 2.5])
    _energy, capacity, gradient = adapter.storage_energy_terms(reference, q)
    gas_enthalpy = np.array([37.5, 34.0]) * (
        reference - reference
    ) + 8.31446261815324 * reference
    np.testing.assert_allclose(
        gas_enthalpy - gradient,
        [70_000.0, 46_000.0],
        rtol=0.0,
        atol=1e-12,
    )
    assert capacity == pytest.approx(float(q @ np.array([37.5, 34.0])))
    assert adapter.source_report()["kinetics"][
        "equal_adsorption_desorption"
    ] is True

    edited = deepcopy(material)
    edited["kinetics"]["desorption_rate_s_inv"][1] *= 0.5
    edited_adapter = build_material_adapter(edited)
    assert edited_adapter.source_report()["kinetics"][
        "equal_adsorption_desorption"
    ] is False


def test_humid_preset_is_visible_and_initializes_coupled_water_and_co2():
    defaults, preset, material, case, weather = _humidity_case()
    assert any(
        item["id"] == "young2021_humid_short_cycle"
        for item in defaults["case_presets"]
    )
    assert preset["case"]["weather"]["temperature_C"] == 30.0
    assert preset["case"]["weather"]["relative_humidity"] == 0.3
    assert preset["case"]["weather"]["co2_ppm_dry"] * 100000.0 / 1e6 == 40.0
    assert preset["case"]["cycle"]["regeneration_temperature_C"] == 55.0
    assert case["initial_state"]["basis"] == "equilibrated_with_first_weather"

    adapter = build_material_adapter(material)
    spatial_config = build_spatial_config(
        material, case, weather.rows[0]
    )
    assert spatial_config["material_file"] == (
        f"materials/screening/{YOUNG_ID}.json"
    )
    assert spatial_config["initial_state"]["basis"] == "equilibrated"
    geometry = build_bare_geometry(spatial_config)
    model = build_model(spatial_config, adapter, geometry)
    spatial_state = initial_state(
        spatial_config, adapter, geometry, model
    )
    spatial_q = spatial_state[model.layout.q].reshape(
        model.n_cells, model.n_storage
    )
    assert np.all(spatial_q[:, 0] > 0.0)
    assert np.all(spatial_q[:, 1] > 0.0)

    lumped_config = build_lumped_config(
        material, case, weather.rows[0]
    )
    assert lumped_config["material_file"] == (
        f"materials/screening/{YOUNG_ID}.json"
    )
    bundle = run_cycle(
        adapter,
        lumped_config,
        relative_humidity=weather.rows[0].relative_humidity,
        water_vapour_partial_pressure_Pa=(
            weather.rows[0].prepared_buck_water_vapour_partial_Pa
        ),
        rhs_only=True,
    )
    lumped_state = bundle["initial_state"]()
    indices = bundle["indices"]
    assert float(np.sum(lumped_state[indices["Q"]])) > 0.0
    assert float(lumped_state[indices["QW"]]) > 0.0
    spatial_q_co2 = float(spatial_q[:, 0].mean())
    spatial_q_water = float(spatial_q[:, 1].mean())
    lumped_q_co2 = float(np.sum(lumped_state[indices["Q"]]))
    lumped_q_water = float(lumped_state[indices["QW"]])
    assert lumped_q_co2 == pytest.approx(spatial_q_co2, rel=0.0, abs=1e-12)
    assert lumped_q_water == pytest.approx(
        spatial_q_water, rel=0.0, abs=1e-12
    )

    spatial_p_water = (
        spatial_state[model.layout.n_gas].reshape(model.n_cells, 4)[:, 1]
        * 8.31446261815324
        * spatial_state[model.layout.T_gas]
        / model.cell_gas_volume_m3
    )
    spatial_rh = spatial_p_water / np.array([
        young2021.buck_saturation_pressure_Pa(float(value))
        for value in spatial_state[model.layout.T_bed]
    ])
    assert np.allclose(spatial_rh, 0.3, rtol=0.0, atol=1e-10)

    lumped_gas = np.maximum(lumped_state[indices["GAS"]], 0.0)
    lumped_rh = (
        lumped_gas[1]
        / lumped_gas.sum()
        * bundle["pressure"](lumped_state)
        / young2021.buck_saturation_pressure_Pa(
            float(lumped_state[indices["TEMP"]])
        )
    )
    assert float(lumped_rh) == pytest.approx(0.3, rel=0.0, abs=1e-10)


def test_young_lumped_is_rejected_at_shared_case_preview():
    defaults = load_defaults()
    raw_material = load_builtin_material(YOUNG_ID)
    case_document = deepcopy(
        next(
            item["case"]
            for item in defaults["case_presets"]
            if item.get("material_id") == YOUNG_ID
        )
    )
    case_document["engine"] = "lumped"
    case_document["solver"]["cells"] = 1
    with pytest.raises(
        runtime_module.MofScreeningError,
        match="spatial engine only",
    ) as captured:
        build_case(raw_material, case_document)
    assert captured.value.field == "case.engine"


def test_cross_loading_potential_is_used_by_both_engines_without_integration():
    defaults = load_defaults()
    raw_material = load_builtin_material("hypothetical_mmen_like")
    material, case = build_case(raw_material, defaults["case"])
    adapter = _cross_loading_adapter(build_material_adapter(material))
    weather = load_weather_window(case["weather"])

    spatial_config = build_spatial_config(
        material, case, weather.rows[0]
    )
    geometry = build_bare_geometry(spatial_config)
    model = build_model(spatial_config, adapter, geometry)
    spatial_state = initial_state(
        spatial_config, adapter, geometry, model
    )
    derivative, observation = cycle_rhs(
        spatial_config,
        adapter,
        geometry,
        "desorption",
        model=model,
    )(0.0, spatial_state)
    epsilon = 1e-7
    before = energy_inventory(
        spatial_config, adapter, geometry, spatial_state, model
    )["total_J"]
    after = energy_inventory(
        spatial_config,
        adapter,
        geometry,
        spatial_state + epsilon * derivative,
        model,
    )["total_J"]
    finite = (after - before) / epsilon
    boundary = (
        observation["heater_W"]
        + observation["source_delivered_W"]
        - observation["cooling_W"]
        - observation["ambient_loss_W"]
        + derivative[model.layout.E_netflow]
    )
    assert abs(finite - boundary) / max(abs(finite), abs(boundary), 1.0) <= 1e-4
    assert abs(observation["first_law_residual_W"]) <= 1e-6

    lumped_config = build_lumped_config(
        material, case, weather.rows[0]
    )
    bundle = run_cycle(
        adapter,
        lumped_config,
        relative_humidity=weather.rows[0].relative_humidity,
        rhs_only=True,
    )
    state = bundle["initial_state"]()
    lumped_derivative = bundle["rhs_by_phase"]["desorption"](0.0, state)
    before = bundle["inventory"](state)
    after = bundle["inventory"](state + epsilon * lumped_derivative)
    finite = (after - before) / epsilon
    indices = bundle["indices"]
    boundary = (
        lumped_derivative[indices["HEAT"]]
        - lumped_derivative[indices["COOL"]]
        - lumped_derivative[indices["LOSS"]]
        + lumped_derivative[indices["INLET_H"]]
        - lumped_derivative[indices["OUTLET_H"]]
    )
    assert abs(finite - boundary) / max(abs(finite), abs(boundary), 1.0) <= 1e-4


def test_legacy_default_energy_wrapper_preserves_finite_tiny_negative_loading():
    material = load_builtin_material("hypothetical_mmen_like")
    adapter = build_material_adapter(material)
    loading = np.zeros(adapter.n_storage)
    loading[-1] = -1.334e-9
    energy, capacity, gradient = adapter.storage_energy_terms(298.15, loading)
    assert np.isfinite(energy)
    assert np.isfinite(capacity)
    assert np.all(np.isfinite(gradient))


def test_young_source_report_is_bounded_and_keeps_limitations_explicit():
    _defaults, _preset, material, _case, _weather = _humidity_case()
    adapter = build_material_adapter(material)
    partials = np.array(
        [
            40.0,
            0.3 * young2021.buck_saturation_pressure_Pa(298.15),
            0.0,
            0.0,
        ]
    )
    for _ in range(1000):
        adapter.equilibrium(298.15, partials)
    report = adapter.source_report()
    assert report["local_relative_humidity_evaluation_count"] == 1000
    assert report["minimum_observed_local_relative_humidity"] == pytest.approx(0.3)
    assert report["maximum_observed_local_relative_humidity"] == pytest.approx(0.3)
    assert report["water_to_co2_coupling"] is True
    assert report["co2_to_water_coupling"] is False
    assert report["kinetics"]["hot_vacuum_transfer_validated"] is False
    assert report["kinetics"]["water_dependent_modulation"] is False
    assert not hasattr(adapter, "_local_rh_calls")


def test_public_spatial_result_enrichment_accepts_material_dict_and_adapter(
    monkeypatch,
):
    _defaults, _preset, material, case, weather = _humidity_case()
    case = deepcopy(case)
    case["solver"]["cells"] = 1
    case["solver"]["max_step_s"] = 10.0
    adapter = build_material_adapter(material)
    config = build_spatial_config(material, case, weather.rows[0])
    geometry = build_bare_geometry(config)
    model = build_model(config, adapter, geometry)
    initial = initial_state(config, adapter, geometry, model)
    phase_names = (
        "adsorption",
        "evacuation",
        "desorption",
        "cooling",
        "repressurization",
    )
    windows = [
        ("adsorption", 0.0, 600.0),
        ("evacuation", 600.0, 660.0),
        ("desorption", 660.0, 1260.0),
        ("cooling", 1260.0, 1560.0),
        ("repressurization", 1560.0, 1680.0),
    ]

    def mocked_one_cycle(*_args, **_kwargs):
        times = np.array([0.0, 600.0, 660.0, 1260.0, 1560.0])
        states = np.tile(initial, (times.size, 1))
        phases_result = np.asarray(phase_names)
        completion = {
            phase: {"horizon_truncated": False} for phase in phase_names
        }
        return (
            np.array([0.0, 600.0, 660.0, 1260.0, 1560.0, 3600.0]),
            np.tile(initial, (6, 1)),
            np.asarray([*phase_names, "repressurization"]),
            windows,
            completion,
            times,
            states,
            phases_result,
        )

    monkeypatch.setattr(runtime_module, "one_cycle", mocked_one_cycle)
    result = run_spatial(
        material,
        case,
        source_identity={"release_sha256": "test"},
    )
    assert result["material"]["material_id"] == YOUNG_ID
    assert result["traces"]
    assert all("water_loading_mean_mol_kg" in row for row in result["traces"])
    assert all("local_rh_max" in row for row in result["traces"])


def test_public_lumped_result_enrichment_accepts_material_dict(monkeypatch):
    _defaults, _preset, material, case, _weather = _humidity_case()
    adapter = build_material_adapter(material)
    q_co2 = float(adapter.equilibrium(
        303.15,
        np.array([40.0, 0.3 * young2021.buck_saturation_pressure_Pa(303.15), 0.0, 0.0]),
    )[0])
    q_water = float(adapter.equilibrium(
        303.15,
        np.array([40.0, 0.3 * young2021.buck_saturation_pressure_Pa(303.15), 0.0, 0.0]),
    )[1])
    phase_rows = []
    for index, phase in enumerate(
        ("adsorption", "evacuation", "desorption", "cooling", "repressurization"),
        start=1,
    ):
        phase_rows.append(
            {
                "cycle": 1,
                "phase": phase,
                "start_elapsed_s": float((index - 1) * 600),
                "end_elapsed_s": float(index * 600),
                "duration_s": 600.0,
                "pressure_end_Pa": (
                    20000.0 if phase == "evacuation" else 100000.0
                ),
                "temperature_end_K": (
                    328.15 if phase == "desorption" else 303.15
                ),
                "co2_loading_end_mol_kg": q_co2,
                "water_loading_end_mol_kg": q_water,
                "local_relative_humidity_end": 0.3,
                "endpoint": {
                    "temperature_ok": True,
                    "pressure_ok": True,
                    "cooling_event_ok": True,
                },
                "energy_closure_relative_error": 0.0,
            }
        )
    summary = {
        "elapsed_s": 3600.0,
        "product_mol": [0.0, 0.0, 0.0, 0.0],
        "vent_mol": [0.0, 0.0, 0.0, 0.0],
        "phase_history": phase_rows,
        "completed_cycles": 1,
        "endpoint_violations": [],
        "temperature_endpoint_violation_count": 0,
        "pressure_endpoint_violation_count": 0,
        "gross_collected_co2_kg_day": 0.0,
        "electricity_J": 0.0,
        "wet_basis_co2_fraction": None,
        "dry_purity": None,
        "liquid_drain_mol": [0.0, 0.0, 0.0, 0.0],
        "conditioned_vent_mol": [0.0, 0.0, 0.0, 0.0],
        "delivered_heat_J": 0.0,
        "pump_electric_J": 0.0,
        "fan_electric_J": 0.0,
        "cooling_removed_J": 0.0,
        "conditioner_cooling_removed_J": 0.0,
        "state_size": 20,
        "mass_closure_relative_error": 0.0,
        "energy_residual_J": 0.0,
        "energy_closure_relative_error": 0.0,
        "energy_legacy_net_relative_residual": 0.0,
        "energy_legacy_net_closure_pass": True,
        "energy_normalization": "test basis",
        "flags": {"temperature_support_excursion": False},
        "conservation_pass": True,
    }

    def mocked_weather_session(*_args, **kwargs):
        assert kwargs["use_bounded_jacobian"] is True
        return {
            "summary": summary,
            "checkpoint_path": "mocked-checkpoint.json",
            "checkpoint_sha256": "0" * 64,
        }

    monkeypatch.setattr(
        runtime_module,
        "run_weather_session",
        mocked_weather_session,
    )
    result = run_lumped(
        material,
        case,
        source_identity={"release_sha256": "test"},
    )
    assert result["material"]["material_id"] == YOUNG_ID
    assert len(result["traces"]) == 5
    assert all("water_loading_mean_mol_kg" in row for row in result["traces"])
    assert all("local_rh_max" in row for row in result["traces"])


def test_young_lumped_factory_passes_bounded_jacobian_to_solve_ivp(
    monkeypatch, tmp_path
):
    defaults = load_defaults()
    raw_material = load_builtin_material(YOUNG_ID)
    case_document = deepcopy(
        next(
            item["case"]
            for item in defaults["case_presets"]
            if item.get("material_id") == YOUNG_ID
        )
    )
    material, case = build_case(raw_material, case_document)
    adapter = build_material_adapter(material)
    weather = load_weather_window(case["weather"])
    config = build_lumped_config(material, case, weather.rows[0])

    class SolveReached(RuntimeError):
        pass

    def mocked_solve_ivp(_fun, t_span, state, **kwargs):
        assert kwargs["method"] == "BDF"
        assert callable(kwargs["jac"])
        jacobian = kwargs["jac"](t_span[0], state)
        assert jacobian.shape == (len(state), len(state))
        assert np.all(np.isfinite(jacobian))
        raise SolveReached("mocked before integration")

    monkeypatch.setattr(weather_driver_module, "solve_ivp", mocked_solve_ivp)
    with pytest.raises(SolveReached, match="mocked before integration"):
        weather_driver_module.run_weather_session(
            adapter,
            weather_hours=list(weather.rows),
            base_config=config,
            job_id="j" * 32,
            attempt_id="a" * 32,
            restart_of_attempt_id=None,
            checkpoint_path=tmp_path / "checkpoint.json",
            context={
                "source_release_sha256": "source",
                "source_hashes": {},
            },
            p_water_mode="buck_explicit",
            rtol=float(case["solver"]["rtol"]),
            max_step_s=float(case["solver"]["max_step_s"]),
            maximum_cycles=None,
            control_variant=POLICY_VARIANT,
            pump_conditioner=False,
            use_bounded_jacobian=True,
        )


def test_legacy_lumped_path_keeps_previous_jacobian_policy(monkeypatch):
    defaults = load_defaults()
    material = load_builtin_material("hypothetical_mmen_like")
    captured = {}

    def mocked_weather_session(*_args, **kwargs):
        captured["use_bounded_jacobian"] = kwargs["use_bounded_jacobian"]
        raise runtime_module.MofScreeningError("stop after policy check")

    monkeypatch.setattr(
        runtime_module,
        "run_weather_session",
        mocked_weather_session,
    )
    with pytest.raises(
        runtime_module.MofScreeningError, match="stop after policy check"
    ):
        run_lumped(
            material,
            defaults["case"],
            source_identity={"release_sha256": "test"},
        )
    assert captured["use_bounded_jacobian"] is False
