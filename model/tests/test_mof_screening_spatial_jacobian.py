"""Zero-integration tests for the bounded spatial Jacobian provider."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from screening.common import ExpectedScenarioError
from screening.geometry import build_bare_geometry
from screening.mof_screening import runtime
from screening.mof_screening.contract import (
    MofScreeningError,
    build_case,
    load_builtin_material,
    load_defaults,
    read_json_strict,
)
from screening.mof_screening.materials import build_material_adapter
from screening.mof_screening.spatial_jacobian import (
    SpatialJacobianAudit,
    SpatialJacobianError,
    bounded_spatial_jacobian,
)
from screening.mof_screening.spatial import (
    build_model,
    cycle_rhs,
    initial_state,
    reset_current_cycle_state_meters,
    state_layout,
)
from screening.systems import build_systems
from screening.utilities import build_utilities


MODEL = Path(__file__).resolve().parents[1]
FIXTURES = MODEL / "tests/fixtures/mof_screening"
JOB_ROOT = FIXTURES / "jobs"


class _ScaleMaterial:
    def __init__(self, count: int):
        self.count = count

    def storage_scale(self):
        return np.arange(1.0, self.count + 1.0)


def _layout_rhs(layout):
    def rhs(_time, values):
        values = np.asarray(values, dtype=float)
        result = np.zeros_like(values)
        result[: layout.inlet.start] = -0.25 * values[: layout.inlet.start]
        result[layout.E_heat] = 3.0 * values[layout.T_bed.start]
        result[layout.E_cool] = 2.0 * values[layout.T_gas.start]
        return result
    return rhs


@pytest.mark.parametrize("storage_count", (4, 5))
def test_layout_aware_jacobian_retains_all_physical_and_ledger_rows(
    storage_count,
):
    layout = state_layout(2, extended=True, n_storage=storage_count)
    state = np.ones(layout.size)
    state[layout.T_h] = 300.0
    state[layout.T_w] = 301.0
    state[layout.T_bed] = 302.0
    state[layout.T_gas] = 303.0
    audit = SpatialJacobianAudit(
        physical_columns=layout.inlet.start,
        ledger_columns=layout.size - layout.inlet.start,
    )
    jacobian = bounded_spatial_jacobian(
        _layout_rhs(layout),
        0.0,
        state,
        layout=layout,
        material=_ScaleMaterial(storage_count),
        audit=audit,
    )
    assert jacobian.shape == (layout.size, layout.size)
    assert np.all(np.isfinite(jacobian))
    assert np.all(jacobian[:, layout.inlet.start :] == 0.0)
    assert audit.ledger_column_nonzero_entries == 0
    assert audit.physical_to_ledger_row_nonzero is True
    assert audit.one_sided_steps + audit.central_steps == layout.inlet.start


def test_jacobian_is_invariant_to_ledger_gauge_and_retains_energy_coupling():
    layout = state_layout(3, extended=True, n_storage=5)
    state = np.ones(layout.size)
    state[layout.T_h] = 300.0
    state[layout.T_w] = 300.0
    state[layout.T_bed] = 305.0
    state[layout.T_gas] = 304.0
    changed_gauge = state.copy()
    changed_gauge[layout.inlet.start :] += 1.0e6
    rhs = _layout_rhs(layout)
    first = bounded_spatial_jacobian(
        rhs, 0.0, state, layout=layout, material=_ScaleMaterial(5)
    )
    second = bounded_spatial_jacobian(
        rhs,
        0.0,
        changed_gauge,
        layout=layout,
        material=_ScaleMaterial(5),
    )
    np.testing.assert_array_equal(first, second)
    assert first[layout.E_heat, layout.T_bed.start] == pytest.approx(3.0)
    assert first[layout.E_cool, layout.T_gas.start] == pytest.approx(2.0)
    assert np.all(first[:, layout.inlet.start :] == 0.0)


def test_finite_difference_step_converges_for_analytic_temperature_rhs():
    layout = state_layout(1, extended=False, n_storage=4)
    state = np.ones(layout.size)
    state[layout.T_h] = 300.0
    state[layout.T_w] = 300.0
    state[layout.T_bed] = 300.0
    state[layout.T_gas] = 300.0

    def rhs(_time, values):
        result = np.zeros_like(values)
        result[layout.T_h] = np.exp(
            (values[layout.T_h] - 300.0) / 100.0
        )
        return result

    exact = 1.0 / 100.0
    errors = []
    for step in (8e-5, 4e-5, 2e-5):
        jacobian = bounded_spatial_jacobian(
            rhs,
            0.0,
            state,
            layout=layout,
            material=_ScaleMaterial(4),
            base_relative_step=step,
            minimum_relative_step=1e-10,
        )
        errors.append(abs(jacobian[layout.T_h, layout.T_h] - exact))
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] <= 2e-8


def test_runtime_and_support_exceptions_propagate_without_probe_fallback():
    layout = state_layout(1, extended=False, n_storage=4)
    state = np.ones(layout.size)
    calls = 0

    def cancelled(_time, values):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise ExpectedScenarioError("manual cancellation reached RHS")
        return np.zeros_like(values)

    with pytest.raises(ExpectedScenarioError, match="manual cancellation"):
        bounded_spatial_jacobian(
            cancelled,
            0.0,
            state,
            layout=layout,
            material=_ScaleMaterial(4),
        )
    assert calls == 2


def test_jacobian_failure_details_survive_runtime_wrapper(monkeypatch):
    defaults = load_defaults()
    material, case = build_case(
        load_builtin_material("hypothetical_mmen_like"),
        deepcopy(defaults["case"]),
    )
    evidence = {
        "time_s": 12.5,
        "column": 3,
        "step": -np.inf,
        "state_sha256": "a" * 64,
        "positive_finite": False,
        "rhs_finite_count": np.int64(4),
        "rhs_size": 5,
    }

    def fail_cycle(*_args, **_kwargs):
        raise SpatialJacobianError(
            "no finite spatial Jacobian probe", evidence=evidence
        )

    monkeypatch.setattr(runtime, "one_cycle", fail_cycle)
    with pytest.raises(MofScreeningError) as captured:
        runtime.run_spatial(
            material,
            case,
            source_identity={"release_sha256": "b" * 64},
        )

    message = str(captured.value)
    assert message.startswith(
        "spatial cycle 1 failed: no finite spatial Jacobian probe"
    )
    assert "time_s=12.5" in message
    assert "column=3" in message
    assert "step=-Infinity" in message
    assert f"state_sha256={'a' * 64}" in message
    assert "positive_finite=False" in message
    assert "rhs_finite_count=4" in message
    assert "rhs_size=5" in message
    assert isinstance(captured.value.__cause__, SpatialJacobianError)
    assert captured.value.__cause__.evidence == evidence


def _job_context(job_id: str, *, failed_state: bool = False):
    job = JOB_ROOT / job_id
    material, case = build_case(
        read_json_strict(job / "input_material.json"),
        read_json_strict(job / "input_case.json"),
    )
    weather = runtime.load_weather_window(case["weather"])
    config = runtime.build_spatial_config(material, case, weather.rows[0])
    config["_runtime_control"] = {
        "weather_provider": weather.at,
        "cycle_start_s": 0.0,
        "phase_offset_s": 0.0,
        "cycle_number": 1,
        "temperature_guard_limit_K": None,
    }
    adapter = build_material_adapter(material)
    geometry = build_bare_geometry(config)
    systems = build_systems(config)
    utilities = build_utilities(config)
    model = build_model(
        config, adapter, geometry, systems=systems, utilities=utilities
    )
    if failed_state:
        data = read_json_strict(FIXTURES / "replay_failure_state.json")
        time_value = float(data["time_s"])
        state = np.asarray(data["base_state"], dtype=float)
    else:
        checkpoint = read_json_strict(job / "SPATIAL_CHECKPOINT.json")
        state = np.asarray(checkpoint["state"], dtype=float)
        time_value = 0.0
    return config, adapter, geometry, model, state, time_value


def test_bounded_jacobian_is_finite_on_exact_replay3_failure_state():
    config, adapter, geometry, model, state, time_value = _job_context(
        "6c785953b9ab418a9f5150a5086a5783", failed_state=True
    )
    rhs = cycle_rhs(
        config, adapter, geometry, "adsorption", model=model
    )
    jacobian = bounded_spatial_jacobian(
        lambda inner_time, inner_state: rhs(inner_time, inner_state)[0],
        time_value,
        state,
        layout=model.layout,
        material=adapter,
    )
    assert np.all(np.isfinite(jacobian))
    assert np.all(jacobian[:, model.layout.inlet.start :] == 0.0)
    assert np.any(
        jacobian[
            model.layout.inlet.start :, : model.layout.inlet.start
        ] != 0.0
    )


@pytest.mark.parametrize(
    "job_id,phase",
    (
        ("6c785953b9ab418a9f5150a5086a5783", "adsorption"),
        ("c9e57eb3c22e47d5a593d3a0efdbe823", "desorption"),
        ("54d9f469c90e4c85bd52d80191e963ca", "desorption"),
        ("73f1ef3a26f24a2e89884ddfd6c88701", "desorption"),
    ),
)
def test_all_four_failure_checkpoints_have_finite_physical_jacobian(
    job_id, phase,
):
    config, adapter, geometry, model, state, time_value = _job_context(job_id)
    rhs = cycle_rhs(config, adapter, geometry, phase, model=model)
    jacobian = bounded_spatial_jacobian(
        lambda inner_time, inner_state: rhs(inner_time, inner_state)[0],
        time_value,
        state,
        layout=model.layout,
        material=adapter,
    )
    assert np.all(np.isfinite(jacobian))
    assert jacobian.shape == (model.layout.size, model.layout.size)
    assert np.all(jacobian[:, model.layout.inlet.start :] == 0.0)


@pytest.mark.parametrize(
    "material_id,storage_count",
    (
        ("calf20_shaped", 3),
        ("cof999_2024_25c_dry_conditional", 4),
        ("cof1000_2025_25c_split_rate_dry_conditional", 5),
    ),
)
def test_supported_storage_layouts_are_retained(
    material_id, storage_count,
):
    defaults = load_defaults()
    case = deepcopy(defaults["case"])
    case["solver"]["cells"] = 2
    case["weather"]["relative_humidity"] = 0.0
    material, case = build_case(load_builtin_material(material_id), case)
    adapter = build_material_adapter(material)
    assert adapter.n_storage == storage_count
    weather = runtime.load_weather_window(case["weather"])
    config = runtime.build_spatial_config(material, case, weather.rows[0])
    geometry = build_bare_geometry(config)
    model = build_model(config, adapter, geometry)
    state = np.ones(model.layout.size)
    state[model.layout.T_h] = 300.0
    state[model.layout.T_w] = 300.0
    state[model.layout.T_bed] = 300.0
    state[model.layout.T_gas] = 300.0
    rhs = cycle_rhs(
        config, adapter, geometry, "adsorption", model=model
    )
    jacobian = bounded_spatial_jacobian(
        lambda inner_time, inner_state: rhs(inner_time, inner_state)[0],
        0.0,
        state,
        layout=model.layout,
        material=adapter,
    )
    assert jacobian.shape == (model.layout.size, model.layout.size)
    assert np.all(np.isfinite(jacobian))
    assert np.all(jacobian[:, model.layout.inlet.start :] == 0.0)


def _hot_path(rh: float):
    defaults = load_defaults()
    material = load_builtin_material(
        "lewatit_vp_oc_1065_young2021_hot116_conditional"
    )
    case = deepcopy(
        next(
            preset["case"]
            for preset in defaults["case_presets"]
            if preset.get("material_id") == material["material_id"]
            and preset["case"]["weather"]["relative_humidity"] == rh
        )
    )
    case["solver"]["cells"] = 2
    material, case = build_case(material, case)
    adapter = build_material_adapter(material)
    weather = runtime.load_weather_window(case["weather"])
    config = runtime.build_spatial_config(material, case, weather.rows[0])
    config["_runtime_control"] = {
        "weather_provider": weather.at,
        "cycle_start_s": 0.0,
        "phase_offset_s": 0.0,
        "cycle_number": 1,
        "temperature_guard_limit_K": None,
    }
    geometry = build_bare_geometry(config)
    model = build_model(config, adapter, geometry)
    state = initial_state(config, adapter, geometry, model)
    reset_current_cycle_state_meters(state, model.layout)
    return config, adapter, geometry, model, state


@pytest.mark.parametrize("relative_humidity", (0.0, 0.30))
@pytest.mark.parametrize(
    ("boundary_K", "expected_direction"),
    ((298.15, "forward"), (393.15, "backward")),
)
def test_hot_adapter_rhs_and_jacobian_are_finite_at_material_probe_bounds(
    relative_humidity, boundary_K, expected_direction
):
    config, adapter, geometry, model, state = _hot_path(
        relative_humidity
    )
    state[model.layout.T_h] = boundary_K
    state[model.layout.T_w] = boundary_K
    state[model.layout.T_bed] = boundary_K
    state[model.layout.T_gas] = boundary_K
    assert adapter.temperature_probe_bounds_K() == (298.15, 393.15)
    rhs = cycle_rhs(
        config, adapter, geometry, "adsorption", model=model
    )
    derivative, _observation = rhs(0.0, state)
    derivative = np.asarray(derivative, dtype=float)
    assert np.all(np.isfinite(derivative))
    audit = SpatialJacobianAudit(
        physical_columns=model.layout.inlet.start,
        ledger_columns=model.layout.size - model.layout.inlet.start,
    )
    probe_temperatures = []

    def bounded_rhs(inner_time, inner_state):
        probe_temperatures.append(
            float(np.min(inner_state[model.layout.T_bed]))
        )
        probe_temperatures.append(
            float(np.max(inner_state[model.layout.T_bed]))
        )
        return rhs(inner_time, inner_state)[0]

    jacobian = bounded_spatial_jacobian(
        bounded_rhs,
        0.0,
        state,
        layout=model.layout,
        material=adapter,
        audit=audit,
    )
    assert np.all(np.isfinite(jacobian))
    assert audit.one_sided_steps >= model.n_cells
    assert audit.central_steps + audit.one_sided_steps == (
        model.layout.inlet.start
    )
    if expected_direction == "forward":
        assert min(probe_temperatures) >= 298.15
    else:
        assert max(probe_temperatures) <= 393.15


@pytest.mark.parametrize("relative_humidity", (0.0, 0.30))
def test_hot_adapter_still_rejects_physical_state_outside_bounds(
    relative_humidity,
):
    config, adapter, geometry, model, state = _hot_path(
        relative_humidity
    )
    rhs = cycle_rhs(
        config, adapter, geometry, "adsorption", model=model
    )
    below = state.copy()
    below[model.layout.T_h] = 298.15 - 1e-6
    below[model.layout.T_w] = 298.15 - 1e-6
    below[model.layout.T_bed] = 298.15 - 1e-6
    below[model.layout.T_gas] = 298.15 - 1e-6
    with pytest.raises(ValueError, match="evaluation envelope"):
        rhs(0.0, below)

    above = state.copy()
    above[model.layout.T_h] = 393.15 + 1e-6
    above[model.layout.T_w] = 393.15 + 1e-6
    above[model.layout.T_bed] = 393.15 + 1e-6
    above[model.layout.T_gas] = 393.15 + 1e-6
    with pytest.raises((ValueError, ExpectedScenarioError)):
        rhs(0.0, above)


@pytest.mark.parametrize(
    "job_id",
    (
        "7ebc1386f2694ec48ad2dd91ed7bf65b",
        "bed9435c6b694f6dbcc4fb86939ff984",
        "3b569f312a014d03ac311fb41321859d",
    ),
)
def test_clean_start_rhs_and_jacobian_remain_finite(job_id):
    config, adapter, geometry, model, state, time_value = _job_context(
        job_id
    )
    rhs = cycle_rhs(
        config, adapter, geometry, "adsorption", model=model
    )
    derivative, _observation = rhs(time_value, state)
    assert np.all(np.isfinite(np.asarray(derivative, dtype=float)))
    jacobian = bounded_spatial_jacobian(
        lambda inner_time, inner_state: rhs(
            inner_time, inner_state
        )[0],
        time_value,
        state,
        layout=model.layout,
        material=adapter,
    )
    assert np.all(np.isfinite(jacobian))
    assert min(state[model.layout.T_bed]) == pytest.approx(298.15)
