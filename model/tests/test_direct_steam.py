"""Balances for direct steam and the boundary between a request and achieved duty."""
from copy import deepcopy

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from screening.common import MW, R, gas_enthalpy_mol
from screening.geometry import build_bare_geometry
from screening.mof_screening import direct_steam
from screening.mof_screening.bdf_trial_recovery import make_recovering_bdf
from screening.mof_screening.contract import build_case, load_builtin_material, load_defaults, MofScreeningError
from screening.mof_screening.evaluation import _trace_phase_endpoint
from screening.mof_screening.materials import build_material_adapter
from screening.mof_screening.runtime import build_spatial_config, cycle_repeatability
from screening.mof_screening.spatial import build_model, cycle_rhs, initial_state
from screening.mof_screening.weather import load_weather_window


def steam_case():
    preset = next(p for p in load_defaults()["case_presets"] if p["id"] == "lewatit_direct_steam_1h")
    return load_builtin_material(preset["material_id"]), deepcopy(preset["case"])


def physical_model(material, inputs):
    material, case = build_case(material, inputs)
    adapter = build_material_adapter(material)
    config = build_spatial_config(material, case, load_weather_window(case["weather"]).rows[0])
    geometry = build_bare_geometry(config)
    model = build_model(config, adapter, geometry)
    return config, adapter, geometry, model, initial_state(config, adapter, geometry, model)


@pytest.mark.parametrize("liquid,ratio,sign", [(0, 0.5, 0), (0.02, 0.5, -1), (0, 1.5, 1)])
def test_phase_transfer_conserves_species_and_energy(liquid, ratio, sign):
    config, adapter, geometry, model, state = physical_model(*steam_case())
    layout = model.layout
    state[layout.liquid] = liquid
    gas = state[layout.n_gas].reshape(model.n_cells, 4)
    gas *= 0.4
    temperature = state[layout.T_bed][0]
    volume = direct_steam.gas_volume(model, state)
    gas[:, 1] = ratio * direct_steam.saturation_pressure_Pa(temperature) * volume / (R * temperature)
    dy, obs = cycle_rhs(config, adapter, geometry, "desorption", model=model)(0, state)
    storage = np.zeros(4)
    dq = dy[layout.q].reshape(model.n_cells, model.n_storage)
    np.add.at(storage, adapter.species_for_storage, model.cell_mass_kg @ dq)
    inventory = storage + dy[layout.n_gas].reshape(model.n_cells, 4).sum(axis=0)
    inventory[1] += dy[layout.liquid].sum()
    boundary = dy[layout.inlet] - dy[layout.vent] - dy[layout.product]
    boundary[1] -= dy[layout.bed_drain] + dy[layout.condensate]
    np.testing.assert_allclose(inventory, boundary, atol=1e-12)
    assert abs(obs["first_law_residual_W"]) < 1e-7
    assert np.sign(dy[layout.bed_condensed] - dy[layout.bed_evaporated]) == sign
    assert dy[layout.steam_in] > 0 and dy[layout.E_heat] == 0
    supply = direct_steam.SteamSupply.from_case(config["direct_steam"], adapter.reference_temperature_K)
    assert dy[layout.e_source_input] == pytest.approx(dy[layout.steam_in] * supply.generator_J_mol)


def test_regulator_back_pressure_and_wet_fan():
    _, case = steam_case()
    supply = direct_steam.SteamSupply.from_case(case["regeneration"], 298.15)
    assert supply.flow(100000) == 0
    assert supply.flow(20000) * MW[1] * 3600 == pytest.approx(10)
    assert supply.flow(97500) == pytest.approx(supply.flow(20000) / 2)
    dry = {"viscous_drop_Pa": 12, "inertial_drop_Pa": 2, "minor_drop_Pa": 1}
    assert direct_steam.wet_bed_drop(dry, 0.4, np.array([0.4, 0.4])) == pytest.approx(15)
    assert direct_steam.wet_bed_drop(dry, 0.4, np.array([0.4, 0.3])) > 15


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("void", [0.4, 0.34])
def test_axial_face_recovers_imposed_ergun_flow_and_upstream_enthalpy(reverse, void):
    temperature = np.array([330.0, 310.0])
    fractions = np.array([[0.1, 0.9, 0, 0], [0.02, 0.03, 0.75, 0.2]])
    donor = int(reverse)
    pressure = np.full(2, 80000.0)
    molar_density = pressure[donor] / (R * temperature[donor])
    density = molar_density * (fractions[donor] @ MW)
    velocity, length, area, diameter, viscosity = 0.08, 0.04, 0.02, 0.0015, 1.85e-5
    drop = length * (150 * viscosity * (1 - void)**2 * velocity / (void**3 * diameter**2)
                     + 1.75 * density * (1 - void) * velocity**2 / (void**3 * diameter))
    pressure[1 - donor] -= drop
    dn, heat = direct_steam.axial_exchange(
        pressure, temperature, fractions, np.full(2, void), area, length, diameter, viscosity, 298.15)
    transported = area * velocity * molar_density * fractions[donor]
    np.testing.assert_allclose(dn[1 - donor], transported, rtol=1e-10)
    assert heat[1 - donor] == pytest.approx(transported @ gas_enthalpy_mol(temperature[donor], 298.15))
    np.testing.assert_array_equal(dn.sum(axis=0), np.zeros(4))
    assert heat.sum() == 0


def test_liquid_displaces_gas_and_flooding_is_local():
    _, _, _, model, state = physical_model(*steam_case())
    capacity = model.cell_pore_volume_m3 / direct_steam.LIQUID_VOLUME_M3_MOL
    state[model.layout.liquid][0] = 0.5 * capacity[0]
    expected = model.cell_gas_volume_m3.copy()
    expected[0] -= 0.5 * model.cell_pore_volume_m3[0]
    np.testing.assert_allclose(direct_steam.gas_volume(model, state), expected)
    state[model.layout.liquid][0] = 1.001 * capacity[0]
    with pytest.raises(ValueError, match="flooding"):
        direct_steam.gas_volume(model, state)


def test_desorption_pressure_target_uses_axial_outlet_on_saved_runs():
    trace = dict(cycle=1, elapsed_s=1, phase="desorption", min_pressure_Pa=18000,
                 axial_pressure_Pa=[22000, 18000, 24000])
    result = {"traces": [trace], "endpoint": {"desorption_pressure_target_Pa": 20000}}
    check = _trace_phase_endpoint(result, 1, "desorption")
    assert check["pressure_end_Pa"] == 24000 and not check["pressure_ok"]
    trace["phase"] = "repressurization"
    assert _trace_phase_endpoint(result, 1, "repressurization")["pressure_end_Pa"] == 18000
    trace.pop("axial_pressure_Pa")
    trace["phase"] = "desorption"
    assert _trace_phase_endpoint(result, 1, "desorption")["pressure_ok"]


def test_weak_jacket_runs_and_keeps_steam_out_of_bed():
    material, case = steam_case()
    case["regeneration"].update(kind="steam_jacket", jacket_UA_W_K=60, supply_pressure_Pa=10000)
    config, adapter, geometry, model, state = physical_model(material, case)
    assert model.layout.liquid is None
    for phase in ("adsorption", "desorption"):
        _, obs = cycle_rhs(config, adapter, geometry, phase, model=model)(0, state)
        assert (obs["source_delivered_W"] > 0) == (phase == "desorption")
        assert abs(obs["first_law_residual_W"]) < 1e-7


def test_dry_only_sorbent_cannot_acquire_an_implicit_wet_isotherm():
    _, case = steam_case()
    with pytest.raises(MofScreeningError, match="Direct steam"):
        build_case(load_builtin_material("cof1000_2025_25c_split_rate_dry_conditional"), case)


def test_pinned_bdf_advances_and_cycle_repeatability_needs_three_comparisons():
    material, case = steam_case()
    config, adapter, geometry, model, state = physical_model(material, case)
    reviewed = build_material_adapter(load_builtin_material("lewatit_vp_oc_1065_young2021_dry_bdf_recovery_v1"))
    method, tracker = make_recovering_bdf(reviewed, model.layout, model)
    assert tracker is not None
    sol = solve_ivp(lambda t, y: -0.1*y, (0, 0.5), state, method=method)
    assert sol.success
    row = dict(complete=True, state_before=state, state_after=state.copy(), cycle_time_s=60, product_CO2_mol_half=1)
    assert cycle_repeatability([row]*4, model.layout, fixed_weather=True)["established"]
    assert not cycle_repeatability([row]*3, model.layout, fixed_weather=True)["established"]
    assert not cycle_repeatability([row]*4, model.layout, fixed_weather=False)["established"]
    changed = deepcopy(row)
    changed["state_after"][model.layout.liquid] += 0.01
    assert not cycle_repeatability([row]*4 + [changed], model.layout, fixed_weather=True)["established"]



@pytest.mark.parametrize("vapour_ratio,initial_liquid", [(1.8, 0.0), (0.15, 0.001)])
def test_insulated_inert_cell_reaches_independent_water_flash(monkeypatch, vapour_ratio, initial_liquid):
    from scipy.optimize import brentq
    from screening.common import CP_GAS
    from screening.water_properties import liquid_enthalpy_J_mol, saturation_pressure_Pa

    material, case = steam_case()
    case["equipment"]["leak_Pa_m3_s"] = 0
    case["regeneration"].update(flow_kg_h=0, drain_time_s=1e25)
    config, adapter, geometry, _, _ = physical_model(material, case)
    config["coupled_through_thickness"]["n_cells_half"] = 1
    config["thermal"].update(heat_loss_U_W_m2_K=0, heater_wall_conductance_W_K=2000)
    model = build_model(config, adapter, geometry)
    state = initial_state(config, adapter, geometry, model)
    lay = model.layout
    state[lay.q] = 0
    monkeypatch.setattr(adapter, "rates", lambda *args: (np.zeros(model.n_storage), np.zeros(4)))
    state[lay.liquid] = initial_liquid
    state[lay.n_gas] = 0
    ti = float(state[lay.T_bed][0])
    vi = model.cell_gas_volume_m3[0]
    vliq = direct_steam.LIQUID_VOLUME_M3_MOL
    vapour = vapour_ratio * saturation_pressure_Pa(ti) * (vi - initial_liquid*vliq) / (R*ti)
    state[lay.n_gas.start + 1] = vapour
    water = vapour + initial_liquid
    capacity = model.solid_capacity_J_K.sum() + model.wall_capacity_J_K + model.heater_capacity_J_K
    ref = adapter.reference_temperature_K

    def energy(t, film):
        return (capacity * (t-ref) + (water-film)*(CP_GAS[1]-R)*(t-ref)
                + film*liquid_enthalpy_J_mol(t, ref))

    def equilibrium_film(t):
        sat = saturation_pressure_Pa(t) / (R*t)
        return max(0.0, (water-sat*vi) / (1-sat*vliq))

    initial_energy = energy(ti, initial_liquid)
    final_T = brentq(lambda t: energy(t, equilibrium_film(t)) - initial_energy, ti-2, ti+2)
    rhs = cycle_rhs(config, adapter, geometry, "desorption", model=model)
    sol = solve_ivp(lambda t, y: rhs(t, y)[0], (0, 2500), state, method="BDF", rtol=1e-8, atol=1e-10)
    assert sol.success
    end = sol.y[:, -1]
    assert end[lay.T_bed][0] == pytest.approx(final_T, abs=3e-5)
    assert end[lay.liquid][0] == pytest.approx(equilibrium_film(final_T), abs=5e-8)
    assert end[lay.n_gas].sum() + end[lay.liquid][0] == pytest.approx(water, abs=1e-9)



@pytest.mark.parametrize("carrier,expected", [("external_heat", 5.5), ("electricity", 3.5)])
def test_utilities_do_not_charge_electric_steam_twice(carrier, expected):
    from app.engineering import costs_for, default_costs
    metrics = dict(steam_generator_carrier=carrier, steam_generator_input_kWh=10,
                   electricity_kWh_day=12, electricity_kWh_kg_CO2=6, collected_CO2_kg_day=2, steam_in_kg=100)
    costs = default_costs(steam=True)
    costs.update(electricity_CHF_kWh=0.25, external_heat_CHF_kWh=0.2, feedwater_CHF_m3=10, makeup_fraction=0.5)
    assert costs_for(costs, metrics, days=1)["utilities_CHF_day"] == pytest.approx(expected)
    costs["makeup_fraction"] = None
    assert costs_for(costs, metrics, days=1)["utilities_CHF_day"] is None
