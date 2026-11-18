from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

import screening.mof_screening.spatial as spatial_module
from screening.common import ExpectedScenarioError
from screening.geometry import build_bare_geometry, build_geometry
from screening.mof_screening.contract import (
    MofScreeningError,
    build_case,
    load_builtin_material,
    load_defaults,
)
from screening.mof_screening.lumped_port.engine import run_cycle
from screening.mof_screening.materials import build_material_adapter
from screening.mof_screening.runtime import (
    _trace_rows,
    build_lumped_config,
    build_spatial_config,
    derived_geometry,
    phase_sequence_complete,
)
from screening.mof_screening.spatial import (
    STEPS,
    _event_kind_fired,
    _events,
    _raise_for_temperature_guard,
    _weather_feed_composition,
    build_model,
    cycle_rhs,
    energy_inventory,
    initial_state,
    one_cycle,
    phase_actuator_request,
)
from screening.mof_screening.weather import load_weather_window


def _model(material_id="hypothetical_mmen_like"):
    defaults = load_defaults()
    material = load_builtin_material(material_id)
    adapter = build_material_adapter(material)
    weather = load_weather_window(defaults["case"]["weather"])
    config = build_spatial_config(material, defaults["case"], weather.rows[0])
    geometry = build_bare_geometry(config)
    model = build_model(config, adapter, geometry)
    state = initial_state(config, adapter, geometry, model)
    return defaults, material, adapter, config, geometry, model, state


@pytest.mark.parametrize("cells", (16, 32))
def test_spatial_n16_and_n32_construct_finite_all_phase_rhs(cells):
    defaults = load_defaults()
    material = load_builtin_material("hypothetical_mmen_like")
    case = deepcopy(defaults["case"])
    case["solver"]["cells"] = cells
    material, case = build_case(material, case)
    adapter = build_material_adapter(material)
    weather = load_weather_window(case["weather"])
    config = build_spatial_config(material, case, weather.rows[0])
    geometry = build_bare_geometry(config)
    model = build_model(config, adapter, geometry)
    state = initial_state(config, adapter, geometry, model)

    assert model.n_cells == cells
    for phase in STEPS:
        derivative, _ = cycle_rhs(
            config, adapter, geometry, phase, model=model
        )(0.0, state)
        assert derivative.shape == state.shape
        assert np.all(np.isfinite(derivative)), phase


@pytest.mark.parametrize("state_controlled", (False, True))
def test_both_spatial_solver_paths_invoke_time_matched_bounded_jacobian(
    monkeypatch, state_controlled
):
    defaults, _material, adapter, config, geometry, model, state = _model()
    weather = load_weather_window(defaults["case"]["weather"])

    class Control:
        mode = "state_control"
        state_controlled = True
        checkpoint_s = 1.0
        minimum_duration_s = 1000.0

    if state_controlled:
        monkeypatch.setattr(
            spatial_module, "STEPS", ("desorption",)
        )
        monkeypatch.setattr(
            spatial_module,
            "load_desorption_control",
            lambda _config: Control(),
        )
        config["cycle"]["adsorption_s"] = 2.0
    else:
        monkeypatch.setattr(spatial_module, "STEPS", ("adsorption",))

    provider_times = []

    def weather_provider(absolute_time):
        provider_times.append(float(absolute_time))
        return weather.at(float(absolute_time))

    config["_runtime_control"] = {
        "weather_provider": weather_provider,
        "cycle_start_s": 0.0,
        "phase_offset_s": 0.0,
        "phase_duration_s": 0.0,
        "cycle_number": 1,
        "temperature_guard_limit_K": None,
    }

    solver_calls = []
    absolute_probe_times = []

    def fake_solve(function, interval, initial, **kwargs):
        start, stop = map(float, interval)
        solver_calls.append((start, stop))
        local_time = start + 0.25 * (stop - start)

        rhs_offset = len(provider_times)
        derivative = function(local_time, initial)
        assert derivative.shape == initial.shape
        assert np.all(np.isfinite(derivative))
        rhs_time = provider_times[rhs_offset]

        jac_offset = len(provider_times)
        jacobian = kwargs["jac"](local_time, initial)
        assert jacobian.shape == (model.layout.size, model.layout.size)
        assert np.all(np.isfinite(jacobian))
        jac_time = provider_times[jac_offset]
        assert jac_time == pytest.approx(rhs_time)
        absolute_probe_times.append(rhs_time)

        def dense(values):
            values = np.atleast_1d(values)
            return np.tile(initial[:, None], (1, values.size))

        return SimpleNamespace(
            success=True,
            y=initial[:, None],
            t=np.asarray([start, stop]),
            t_events=[np.asarray([]) for _ in kwargs["events"]],
            sol=dense,
        )

    monkeypatch.setattr(spatial_module, "solve_ivp", fake_solve)
    one_cycle(config, adapter, geometry, state, model=model)

    expected_probe_times = (
        [0.25, 1.25]
        if state_controlled
        else [0.25 * config["cycle"]["adsorption_s"]]
    )
    assert absolute_probe_times == pytest.approx(expected_probe_times)
    audit = config["_runtime_control"]["spatial_jacobian_audit"]
    assert audit["calls"] == len(solver_calls)
    assert audit["physical_columns"] == model.layout.inlet.start
    assert audit["ledger_columns"] == (
        model.layout.size - model.layout.inlet.start
    )
    assert audit["ledger_column_nonzero_entries"] == 0
    assert len(audit["matrix_sha256_by_call"]) == len(solver_calls)


def test_spatial_mesh_rejects_more_than_32_cells():
    defaults = load_defaults()
    case = deepcopy(defaults["case"])
    case["solver"]["cells"] = 33
    with pytest.raises(MofScreeningError) as captured:
        build_case(load_builtin_material("hypothetical_mmen_like"), case)
    assert captured.value.field == "case.solver.cells"


def test_geometry_is_derived_consistently_from_mass_packing_and_volume():
    defaults = load_defaults()
    geometry = derived_geometry(defaults["case"])
    equipment = defaults["case"]["equipment"]
    assert np.isclose(
        geometry["volume_m3"],
        equipment["dry_shaped_mass_kg"]
        / geometry["bulk_density_kg_m3"],
    )
    assert np.isclose(
        geometry["pellet_density_kg_m3"]
        * (1 - equipment["bed_void_fraction"]),
        geometry["bulk_density_kg_m3"],
    )
    _, _, _, config, _, _, _ = _model()
    assert config["cycle"]["purge_s"] == defaults["case"]["cycle"]["purge_s"]
    assert config["collection"] == defaults["case"]["collection"]


@pytest.mark.parametrize(
    "material_id", ["hypothetical_mmen_like", "calf20_shaped"]
)
def test_species_rates_enter_gas_balance_with_cell_dry_mass(material_id):
    _, _, adapter, config, geometry, model, state = _model(material_id)
    # At ambient pressure, repressurization contributes no feed or withdrawal.
    state[model.layout.q] *= 0.5
    derivative, _ = cycle_rhs(
        config, adapter, geometry, "repressurization", model=model
    )(0.0, state)
    q = state[model.layout.q].reshape(model.n_cells, adapter.n_storage)
    ng = state[model.layout.n_gas].reshape(model.n_cells, 4)
    pressure = (
        state[model.layout.T_gas]
        * ng.sum(axis=1)
        * 8.314462618
        / model.cell_gas_volume_m3
    )
    partials = ng / ng.sum(axis=1)[:, None] * pressure[:, None]
    expected_species = np.zeros((model.n_cells, 4))
    for cell in range(model.n_cells):
        _, species_rates = adapter.rates(
            float(state[model.layout.T_bed][cell]),
            partials[cell],
            q[cell],
        )
        expected_species[cell] = model.cell_mass_kg * species_rates
    dng = derivative[model.layout.n_gas].reshape(model.n_cells, 4)
    np.testing.assert_allclose(
        dng, -expected_species, rtol=1e-10, atol=1e-12
    )


@pytest.mark.parametrize(
    "material_id", ["hypothetical_mmen_like", "calf20_shaped"]
)
def test_differentiated_energy_identity_at_hot_site_state(material_id):
    _, _, adapter, config, geometry, model, state = _model(material_id)
    state[model.layout.T_bed] = 370.0
    state[model.layout.T_gas] = 360.0
    state[model.layout.T_w] = 365.0
    state[model.layout.T_h] = 380.0
    state[model.layout.q] = np.tile(
        adapter.equilibrium(
            370.0, np.array([5000.0, 1000.0, 70000.0, 24000.0])
        ),
        model.n_cells,
    )
    derivative, observation = cycle_rhs(
        config, adapter, geometry, "desorption", model=model
    )(0.0, state)
    epsilon = 1e-7
    before = energy_inventory(
        config, adapter, geometry, state, model
    )["total_J"]
    after = energy_inventory(
        config,
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
    assert abs(observation["first_law_residual_W"]) <= 1e-6
    assert abs(finite_derivative - boundary) / max(
        abs(finite_derivative), abs(boundary), 1.0
    ) <= 1e-4


def test_runtime_guard_runs_inside_rhs():
    _, _, adapter, config, geometry, model, state = _model()
    config["_runtime_control"] = {
        "cancel_check": lambda: (_ for _ in ()).throw(
            ExpectedScenarioError("cancelled in RHS")
        )
    }
    with pytest.raises(ExpectedScenarioError, match="cancelled"):
        cycle_rhs(
            config, adapter, geometry, "adsorption", model=model
        )(0.0, state)


def test_runtime_deadline_runs_inside_rhs_and_none_disables_it():
    _, material, adapter, config, geometry, model, state = _model()
    config["_runtime_control"] = {
        "cancel_check": lambda: None,
        "deadline_monotonic": None,
    }
    derivative = cycle_rhs(
        config, adapter, geometry, "adsorption", model=model
    )(0.0, state)[0]
    assert np.all(np.isfinite(derivative))

    config["_runtime_control"]["deadline_monotonic"] = 0.0
    with pytest.raises(ExpectedScenarioError, match="wall-time cap"):
        cycle_rhs(
            config, adapter, geometry, "adsorption", model=model
        )(0.0, state)

    config["_runtime_control"] = {"cancel_check": lambda: None}
    hot_state = state.copy()
    hot_state[model.layout.T_bed] = (
        float(material["maximum_bed_temperature_C"]) + 274.0
    )
    with pytest.raises(ExpectedScenarioError, match="temperature guard"):
        cycle_rhs(
            config, adapter, geometry, "adsorption", model=model
        )(0.0, hot_state)

    config["_runtime_control"]["temperature_guard_limit_K"] = None
    derivative = cycle_rhs(
        config, adapter, geometry, "adsorption", model=model
    )(0.0, hot_state)[0]
    assert np.all(np.isfinite(derivative))


def test_spatial_temperature_guard_event_is_policy_controlled():
    _, _material, adapter, config, geometry, model, state = _model(
        "lewatit_vp_oc_1065_low2023_conditional"
    )
    state[model.layout.T_bed] = 394.0
    config["_runtime_control"] = {
        "temperature_guard_limit_K": 393.15
    }
    events = _events(config, adapter, geometry, model, "adsorption")
    assert max(event(0.0, state) for event in events) > 0.0


def test_normal_cooling_and_repressurization_events_are_not_thermal_failures():
    def normal_event(kind):
        def event(_time, _state):
            return 0.0
        event.event_kind = kind
        return event

    def guard_event():
        def event(_time, _state):
            return 0.0
        event.event_kind = "temperature_guard"
        return event

    cooling = normal_event("cooling_completion")
    repress = normal_event("repressurization_completion")
    normal_solution = SimpleNamespace(
        t_events=[np.zeros(0), np.array([12.0])]
    )
    assert not _event_kind_fired(
        [cooling, repress], normal_solution, "temperature_guard"
    )
    _raise_for_temperature_guard([cooling, repress], normal_solution)

    guard = guard_event()
    strict_solution = SimpleNamespace(
        t_events=[np.zeros(0), np.array([4.0])]
    )
    with pytest.raises(ExpectedScenarioError, match="temperature limit"):
        _raise_for_temperature_guard([repress, guard], strict_solution)


def test_record_policy_has_only_named_phase_end_events():
    _, _material, adapter, config, geometry, model, state = _model(
        "lewatit_vp_oc_1065_low2023_conditional"
    )
    state[model.layout.T_bed] = 394.0
    config["_runtime_control"] = {"temperature_guard_limit_K": None}
    cooling = _events(config, adapter, geometry, model, "cooling")
    repress = _events(config, adapter, geometry, model, "repressurization")
    assert [event.event_kind for event in cooling] == [
        "cooling_completion"
    ]
    assert [event.event_kind for event in repress] == [
        "repressurization_completion"
    ]

    config["_runtime_control"]["temperature_guard_limit_K"] = None
    events = _events(config, adapter, geometry, model, "adsorption")
    assert events == []

    del config["_runtime_control"]["temperature_guard_limit_K"]
    events = _events(config, adapter, geometry, model, "adsorption")
    assert max(event(0.0, state) for event in events) > 0.0


def test_spatial_rhs_uses_hourly_forcing_inside_a_phase():
    defaults = load_defaults()
    spec = dict(defaults["case"]["weather"])
    spec.update(
        {
            "mode": "official",
            "hours": 2,
            "start_utc": "2023-01-15T00:00:00Z",
        }
    )
    material = load_builtin_material("hypothetical_mmen_like")
    adapter = build_material_adapter(material)
    weather = load_weather_window(spec)
    config = build_spatial_config(
        material, defaults["case"], weather.rows[0]
    )
    geometry = build_bare_geometry(config)
    model = build_model(config, adapter, geometry)
    state = initial_state(config, adapter, geometry, model)
    config["_runtime_control"] = {
        "weather_provider": weather.at,
        "cycle_start_s": 0.0,
        "phase_offset_s": 0.0,
    }
    rhs = cycle_rhs(
        config, adapter, geometry, "adsorption", model=model
    )
    first, _ = rhs(0.0, state)
    second, _ = rhs(3600.0, state)
    assert not np.allclose(first, second, rtol=0.0, atol=0.0)


def test_reviewed_conditioner_and_shared_policy_drive_spatial_rhs():
    _, _, adapter, config, geometry, model, state = _model()
    state[model.layout.T_bed] = 360.0
    state[model.layout.T_gas] = 350.0
    state[model.layout.T_w] = 350.0
    state[model.layout.T_h] = 380.0
    state[model.layout.q] = np.tile(
        adapter.equilibrium(
            360.0,
            np.array([5000.0, 1000.0, 70000.0, 24000.0]),
        ),
        model.n_cells,
    )

    derivative, observation = cycle_rhs(
        config, adapter, geometry, "desorption", model=model
    )(0.0, state)

    assert observation["conditioner_policy"] == (
        "pump_inlet_cooler_knockout_v1"
    )
    assert observation["conditioner_pressure_drop_Pa"] == 0.0
    assert observation["conditioner_bed_flow_mol_s"] >= 0.0
    assert observation["prepump_cooling_full_W"] >= 0.0
    assert observation["combined_cooling_full_W"] <= 2000.0 + 1e-9
    assert np.isclose(observation["vacuum_W"] * 2.0, 135.0)
    assert abs(observation["first_law_residual_W"]) <= 1e-6
    assert np.all(np.isfinite(derivative))


def test_shared_wall_controls_have_correct_full_and_half_units():
    defaults, material, adapter, config, geometry, model, state = _model()
    equipment = defaults["case"]["equipment"]
    assert model.wall_capacity_J_K == pytest.approx(
        0.5 * equipment["wall_heat_capacity_J_K"]
    )
    assert (
        config["thermal"]["cooling_h_W_m2_K"]
        * model.half_face_area_m2
    ) == pytest.approx(equipment["wall_cooling_UA_W_K"])

    lumped = build_lumped_config(
        material,
        defaults["case"],
        load_weather_window(defaults["case"]["weather"]).rows[0],
    )
    lumped_geometry = build_geometry(lumped)
    assert (
        lumped_geometry.metal_mass_kg
        * lumped["geometry"]["metal_cp_J_kg_K"]
    ) == pytest.approx(equipment["wall_heat_capacity_J_K"])
    assert lumped["shared_equipment"] == config["shared_equipment"]
    assert lumped["shared_equipment"]["lumped_applicability"]

    state[model.layout.T_bed] = 338.15
    state[model.layout.T_gas] = 338.15
    state[model.layout.T_w] = 318.15
    _, observation = cycle_rhs(
        config, adapter, geometry, "adsorption", model=model
    )(0.0, state)
    assert observation["cooler_request_full_W"] == pytest.approx(2000.0)
    assert 2.0 * observation["cooling_W"] <= equipment[
        "wall_cooling_UA_W_K"
    ] * 20.0 + 1e-9
    assert observation["combined_cooling_full_W"] <= 2000.0 + 1e-9


def test_phase_actuator_availability_matches_reviewed_policy():
    _, _, _, config, _, _, _ = _model()
    ambient_K = 298.15
    assert phase_actuator_request(
        config,
        phase="adsorption",
        bed_temperature_K=308.15,
        ambient_temperature_K=ambient_K,
        fixed_bed_balance_W=0.0,
    )[:2] == pytest.approx((0.0, 2000.0))
    assert phase_actuator_request(
        config,
        phase="desorption",
        bed_temperature_K=353.15,
        ambient_temperature_K=ambient_K,
        fixed_bed_balance_W=0.0,
    )[:2] == pytest.approx((2000.0, 0.0))
    assert phase_actuator_request(
        config,
        phase="cooling",
        bed_temperature_K=308.15,
        ambient_temperature_K=ambient_K,
        fixed_bed_balance_W=0.0,
    )[:2] == pytest.approx((0.0, 2000.0))
    assert phase_actuator_request(
        config,
        phase="evacuation",
        bed_temperature_K=308.15,
        ambient_temperature_K=ambient_K,
        fixed_bed_balance_W=125.0,
    )[:2] == pytest.approx((0.0, 125.0))


def test_final_repressurization_truncated_by_horizon_is_incomplete():
    windows = [(phase, index, index + 1) for index, phase in enumerate(STEPS)]
    completion = {
        phase: {"event_fired": False, "at_entry": False}
        for phase in STEPS
    }
    assert phase_sequence_complete(windows, completion)
    completion["repressurization"]["horizon_truncated"] = True
    assert not phase_sequence_complete(windows, completion)


def test_spatial_trace_phase_elapsed_uses_matching_window_start():
    _, _, adapter, _, _, model, state = _model()
    times = np.array([100.0, 110.0, 200.0])
    states = np.tile(state, (3, 1))
    phases = np.array(["adsorption", "adsorption", "evacuation"])
    windows = [
        ("adsorption", 100.0, 120.0),
        ("evacuation", 120.0, 220.0),
    ]

    rows = _trace_rows(model, times, states, phases, windows, material=adapter)

    assert [row["elapsed_s"] for row in rows] == [100.0, 110.0, 200.0]
    assert [row["phase_elapsed_s"] for row in rows] == [0.0, 10.0, 80.0]


@pytest.mark.parametrize("weather_mode", ["fixed", "official"])
def test_fixed_and_official_weather_share_buck_composition_and_initial_inventory(
    weather_mode,
):
    defaults = load_defaults()
    case = deepcopy(defaults["case"])
    if weather_mode == "official":
        case["weather"].update(
            {
                "mode": "official",
                "hours": 1,
                "start_utc": "2023-01-15T00:00:00Z",
            }
        )
    material = load_builtin_material("hypothetical_mmen_like")
    adapter = build_material_adapter(material)
    weather = load_weather_window(case["weather"])
    row = weather.rows[0]
    canonical = _weather_feed_composition(row)
    assert canonical[1] == pytest.approx(row.water_mole_fraction)
    assert row.prepared_buck_water_vapour_partial_Pa == pytest.approx(
        canonical[1] * row.pressure_Pa
    )

    spatial_config = build_spatial_config(material, case, row)
    assert spatial_config["ambient"]["water_mole_fraction"] == pytest.approx(
        row.water_mole_fraction
    )
    assert spatial_config["ambient"]["water_pressure_law"] == "Buck 1996"
    geometry = build_bare_geometry(spatial_config)
    spatial_model = build_model(spatial_config, adapter, geometry)
    spatial_state = initial_state(
        spatial_config, adapter, geometry, spatial_model
    )
    spatial_gas = spatial_state[spatial_model.layout.n_gas].reshape(
        spatial_model.n_cells, 4
    )
    np.testing.assert_allclose(
        spatial_gas[0] / spatial_gas[0].sum(),
        canonical,
        rtol=0.0,
        atol=1e-15,
    )
    expected_loading = adapter.equilibrium(
        row.temperature_C + 273.15,
        canonical * row.pressure_Pa,
    )
    np.testing.assert_allclose(
        spatial_state[spatial_model.layout.q].reshape(
            spatial_model.n_cells, adapter.n_storage
        )[0],
        expected_loading,
        rtol=1e-14,
        atol=1e-15,
    )

    lumped_config = build_lumped_config(material, case, row)
    bundle = run_cycle(
        adapter,
        lumped_config,
        relative_humidity=row.relative_humidity,
        water_vapour_partial_pressure_Pa=(
            row.prepared_buck_water_vapour_partial_Pa
        ),
        rhs_only=True,
    )
    lumped_state = bundle["initial_state"]()
    indices = bundle["indices"]
    lumped_gas = lumped_state[indices["GAS"]]
    np.testing.assert_allclose(
        lumped_gas / lumped_gas.sum(),
        canonical,
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        lumped_state[indices["Q"]],
        expected_loading[: adapter.n_sites],
        rtol=1e-14,
        atol=1e-15,
    )


def test_trace_sums_co2_reservoirs_and_uses_gas_eos_for_water():
    _, _, adapter, _, _, model, state = _model("calf20_shaped")
    q = state[model.layout.q].reshape(model.n_cells, -1)
    q[:, adapter.species_for_storage == 0] = [0.15, 0.35]
    state[model.layout.T_gas] = 310.0
    state[model.layout.T_bed] = 300.0
    trace = _trace_rows(model, [0.0], [state], ["adsorption"], material=adapter)[0]
    assert trace["co2_loading_mean_mol_kg"] == pytest.approx(0.5)
    from screening.mof_screening.young2021 import buck_saturation_pressure_Pa
    n_water = state[model.layout.n_gas].reshape(model.n_cells, 4)[:, 1]
    expected_rh = n_water * 8.314462618 * 310.0 / model.cell_gas_volume_m3 / buck_saturation_pressure_Pa(300.0)
    assert trace["local_rh_mean"] == pytest.approx(expected_rh.mean())


@pytest.mark.parametrize("phase, t, collecting", [
    ("desorption", 4.0, False), ("desorption", 5.0, True), ("cooling", 5.0, False)
])
def test_condensate_meter_follows_selected_product_valve(phase, t, collecting):
    _, _, adapter, config, geometry, model, state = _model()
    config["cycle"]["purge_s"] = 5
    config["collection"]["product_route"] = "desorption_only"
    gas = state[model.layout.n_gas].reshape(model.n_cells, 4)
    gas[:] = gas.sum(axis=1)[:, None] * np.array([0.1, 0.7, 0.15, 0.05])
    state[model.layout.T_gas] = state[model.layout.T_bed] = 360.0
    derivative, _ = cycle_rhs(config, adapter, geometry, phase, model=model)(t, state)
    cond = derivative[model.layout.condensate]
    assert cond > 0
    assert derivative[model.layout.condensate_product] == (cond if collecting else 0.0)
    assert (derivative[model.layout.product].sum() > 0) == collecting
