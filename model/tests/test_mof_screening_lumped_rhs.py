import numpy as np
import pytest

from screening.mof_screening.contract import (
    load_builtin_material,
    load_defaults,
)
from screening.mof_screening.lumped_port.engine import CycleFailure, run_cycle
from screening.mof_screening.lumped_port.pressure_control import (
    POLICY_VARIANT,
    PRESSURE_CONTROL,
)
from screening.mof_screening.lumped_port.weather_driver import weather_config
from screening.mof_screening.lumped_port import weather_driver as weather_driver_module
from screening.mof_screening.materials import build_material_adapter
from screening.mof_screening.runtime import build_lumped_config
from screening.mof_screening.weather import load_weather_window


@pytest.mark.parametrize(
    "material_id", ["hypothetical_mmen_like", "calf20_shaped"]
)
def test_lumped_port_all_phase_rhs_are_finite_without_integration(material_id):
    defaults = load_defaults()
    material = load_builtin_material(material_id)
    adapter = build_material_adapter(material)
    weather = load_weather_window(defaults["case"]["weather"])
    config = build_lumped_config(material, defaults["case"], weather.rows[0])
    bundle = run_cycle(
        adapter,
        config,
        relative_humidity=weather.rows[0].relative_humidity,
        rhs_only=True,
    )
    state = bundle["initial_state"]()
    assert np.all(np.isfinite(state))
    assert bundle["pressure"](state) > 0.0
    assert np.any(state[: adapter.n_sites + 3] > 1e-9)
    for phase, rhs in bundle["rhs_by_phase"].items():
        derivative = np.asarray(rhs(0.0, state), dtype=float)
        assert derivative.shape == state.shape, phase
        assert np.all(np.isfinite(derivative)), phase


def test_lumped_rhs_honors_no_deadline_and_explicit_deadline():
    defaults = load_defaults()
    material = load_builtin_material(
        "lewatit_vp_oc_1065_low2023_conditional"
    )
    adapter = build_material_adapter(material)
    weather = load_weather_window(defaults["case"]["weather"])
    config = build_lumped_config(material, defaults["case"], weather.rows[0])
    config["_runtime_control"] = {
        "cancel_check": lambda: None,
        "deadline_monotonic": None,
    }
    bundle = run_cycle(
        adapter,
        config,
        relative_humidity=weather.rows[0].relative_humidity,
        rhs_only=True,
    )
    state = bundle["initial_state"]()
    derivative = bundle["rhs_by_phase"]["adsorption"](0.0, state)
    assert np.all(np.isfinite(derivative))

    config["_runtime_control"]["cancel_check"] = (
        lambda: (_ for _ in ()).throw(
            CycleFailure(
                "cancelled in lumped RHS",
                phase="adsorption",
                cycle=1,
                records=[],
            )
        )
    )
    with pytest.raises(CycleFailure, match="cancelled"):
        bundle["rhs_by_phase"]["adsorption"](0.0, state)

    config["_runtime_control"]["cancel_check"] = lambda: None
    config["_runtime_control"]["deadline_monotonic"] = 0.0
    with pytest.raises(CycleFailure, match="wall-time cap"):
        bundle["rhs_by_phase"]["adsorption"](0.0, state)

    config["_runtime_control"]["deadline_monotonic"] = None
    config["_runtime_control"]["temperature_guard_limit_K"] = 393.15
    hot_state = state.copy()
    hot_state[adapter.n_sites + 3] = 394.0
    with pytest.raises(CycleFailure, match="temperature guard"):
        bundle["rhs_by_phase"]["adsorption"](0.0, hot_state)

    config["_runtime_control"]["temperature_guard_limit_K"] = None
    derivative = bundle["rhs_by_phase"]["adsorption"](0.0, hot_state)
    assert np.all(np.isfinite(derivative))

    del config["_runtime_control"]["temperature_guard_limit_K"]
    with pytest.raises(CycleFailure, match="temperature guard"):
        bundle["rhs_by_phase"]["adsorption"](0.0, hot_state)


def test_hourly_config_preserves_case_timing_and_shared_pressure_policy():
    defaults = load_defaults()
    assert defaults["case"]["solver"]["rtol"] == 1e-7
    assert defaults["case"]["solver"]["max_step_s"] == 10.0
    material = load_builtin_material("hypothetical_mmen_like")
    weather = load_weather_window(defaults["case"]["weather"])
    base = build_lumped_config(material, defaults["case"], weather.rows[0])
    configured = weather_config(
        base,
        weather.rows[0],
        "buck_explicit",
        control_variant=POLICY_VARIANT,
    )
    assert configured["cycle"]["adsorption_s"] == defaults["case"]["cycle"][
        "adsorption_s"
    ]
    assert configured["cycle"]["desorption_s"] == defaults["case"]["cycle"][
        "desorption_s"
    ]
    assert configured["cycle"]["cooling_max_s"] == defaults["case"]["cycle"][
        "cooling_max_s"
    ]
    assert configured["cycle"]["adsorption_control"] == PRESSURE_CONTROL


def test_ui_lumped_entrypoint_reaches_first_step_with_null_maximum_cycles(
    monkeypatch, tmp_path
):
    class EntrypointReached(RuntimeError):
        pass

    def stop_before_integration(*_args, **kwargs):
        assert kwargs["first_step"] is not None
        assert kwargs["first_step"] > 0.0
        raise EntrypointReached("mocked before solve integration")

    defaults = load_defaults()
    material = load_builtin_material("hypothetical_mmen_like")
    adapter = build_material_adapter(material)
    weather = load_weather_window(defaults["case"]["weather"])
    config = build_lumped_config(
        material, defaults["case"], weather.rows[0]
    )
    assert config["solver"]["initial_step_depletion_fraction"] == 0.1
    monkeypatch.setattr(
        weather_driver_module,
        "solve_ivp",
        stop_before_integration,
    )

    with pytest.raises(EntrypointReached, match="mocked"):
        weather_driver_module.run_weather_session(
            adapter,
            weather_hours=list(weather.rows),
            base_config=config,
            job_id="test-lumped-ui",
            attempt_id="test-attempt",
            restart_of_attempt_id=None,
            checkpoint_path=tmp_path / "checkpoint.json",
            context={
                "source_release_sha256": "a" * 64,
                "source_hashes": {},
            },
            p_water_mode="buck_explicit",
            rtol=1e-7,
            max_step_s=10.0,
            maximum_cycles=None,
            control_variant=POLICY_VARIANT,
            pump_conditioner=False,
        )
