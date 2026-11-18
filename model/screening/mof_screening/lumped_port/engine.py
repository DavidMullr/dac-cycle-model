"""Common finite-cycle engine for material adapters."""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy.integrate import solve_ivp

from ...airflow import airflow
from ...common import CP_GAS, R, pump_profile_for, water_saturation_pressure_Pa
from ...geometry import build_geometry
from ...pump_rates import rate_at_pressure

from .pressure_control import (
    FIXED_CONTROL,
    PRESSURE_CONTROL,
    control_identity,
)
from .pump_conditioner import (
    evaluate_pump_conditioner,
    pump_conditioner_config,
    pump_conditioner_identity,
)
from .solver_start import (
    DEFAULT_DEPLETION_FRACTION,
    DEFAULT_INITIAL_STEP_MAX_S,
    INITIAL_STEP_POLICY,
    initial_step_identity,
    positive_gas_first_step,
)

PHASES = ("adsorption", "evacuation", "desorption", "cooling", "repressurization")
COLLECTION_TRACE_OFFSETS_S = (0.0, 300.0, 600.0, 1200.0)
ADSORPTION_PRESSURE_TOLERANCE_FRACTION = 0.01


def diagnostic_saturation_pressure_Pa(adapter, temperature_K):
    """Select a candidate diagnostic law while retaining reference materials."""
    selector = getattr(adapter, "diagnostic_saturation_pressure_Pa", None)
    if callable(selector):
        return float(selector(float(temperature_K)))
    return float(water_saturation_pressure_Pa(float(temperature_K)))


def diagnostic_saturation_law(adapter):
    if callable(getattr(adapter, "diagnostic_saturation_pressure_Pa", None)):
        return str(adapter.diagnostic_saturation_law)
    return "screening.common NIST/Stull Antoine"


def diagnostic_water_saturation_ratios(
    adapter, temperatures_K, water_partial_pressure_Pa
):
    """Evaluate accepted-state saturation ratios without solver access."""
    temperatures = np.asarray(temperatures_K, dtype=float)
    water_partial_pressure = np.asarray(
        water_partial_pressure_Pa, dtype=float
    )
    saturation_pressures = np.array(
        [
            diagnostic_saturation_pressure_Pa(adapter, float(value))
            for value in temperatures
        ]
    )
    saturation_ratios = np.divide(
        water_partial_pressure,
        saturation_pressures,
        out=np.zeros_like(water_partial_pressure),
        where=saturation_pressures > 0.0,
    )
    return {
        "saturation_pressures_Pa": saturation_pressures,
        "saturation_ratios": saturation_ratios,
        "saturation_diagnostic_law": diagnostic_saturation_law(adapter),
    }


class CycleFailure(RuntimeError):
    def __init__(self, message: str, *, phase: str, cycle: int, records: list[dict]):
        super().__init__(message)
        self.phase = phase
        self.cycle = cycle
        self.records = records


def setpoint_requested_duty(
    *,
    fixed_energy_derivative_W: float,
    loss_W: float,
    inlet_enthalpy_W: float,
    outlet_enthalpy_W: float,
    bed_temperature_K: float,
    target_temperature_K: float,
    gain_W_K: float,
) -> float:
    """Signed duty: zero-dT base duty plus proportional setpoint feedback."""
    zero_rate_duty = (
        fixed_energy_derivative_W
        + loss_W
        - inlet_enthalpy_W
        + outlet_enthalpy_W
    )
    return float(
        zero_rate_duty
        + gain_W_K * (target_temperature_K - bed_temperature_K)
    )


def split_actuator_duty(
    requested_duty_W: float,
    *,
    heater_limit_W: float,
    cooling_limit_W: float,
) -> tuple[float, float]:
    heat = min(max(requested_duty_W, 0.0), heater_limit_W)
    cooling = min(max(-requested_duty_W, 0.0), cooling_limit_W)
    return float(heat), float(cooling)


def phase_endpoints_ok(
    phase_records,
    *,
    ambient_temperature_K: float,
    ambient_pressure_Pa: float,
    regeneration_temperature_K: float,
    regeneration_pressure_Pa: float,
    cooling_endpoint_K: float | None = None,
) -> bool:
    """Apply the unchanged finite phase endpoint gates."""
    return bool(
        abs(phase_records[0]["temperature_end_K"] - ambient_temperature_K) <= 1.0
        and abs(
            phase_records[0]["pressure_end_Pa"] - ambient_pressure_Pa
        )
        <= ADSORPTION_PRESSURE_TOLERANCE_FRACTION * ambient_pressure_Pa
        and abs(
            phase_records[1]["pressure_end_Pa"] - regeneration_pressure_Pa
        )
        <= 0.02 * regeneration_pressure_Pa
        and abs(
            phase_records[2]["temperature_end_K"] - regeneration_temperature_K
        )
        <= 2.0
        and phase_records[3]["temperature_end_K"]
        <= (
            cooling_endpoint_K
            if cooling_endpoint_K is not None
            else ambient_temperature_K + 1.0
        )
        and phase_records[4]["pressure_end_Pa"] >= 0.99 * ambient_pressure_Pa
    )


def limit_adsorption_demand(
    component_rates,
    *,
    site_species,
    available_mol_s,
    dry_mass_kg: float,
) -> np.ndarray:
    """Limit positive adsorption demand per species; retain desorption terms."""
    rates = np.asarray(component_rates, dtype=float).copy()
    available = np.asarray(available_mol_s, dtype=float)
    ns = len(site_species)
    for species in range(4):
        indices = [i for i, value in enumerate(site_species) if value == species]
        positive_site_demand = sum(max(rates[i], 0.0) for i in indices)
        extra_index = {1: ns, 2: ns + 1, 3: ns + 2}.get(species)
        positive_extra = (
            max(rates[extra_index], 0.0)
            if extra_index is not None
            else 0.0
        )
        demand = positive_site_demand + positive_extra
        available_rate = available[species] / dry_mass_kg
        if demand <= available_rate or demand <= 0.0:
            continue
        scale = available_rate / demand
        for index in indices:
            if rates[index] > 0.0:
                rates[index] *= scale
        if extra_index is not None and rates[extra_index] > 0.0:
            rates[extra_index] *= scale
    return rates


def species_mass_ledger(
    *,
    initial_sites,
    final_sites,
    initial_water: float,
    final_water: float,
    initial_n2: float,
    final_n2: float,
    initial_o2: float,
    final_o2: float,
    initial_gas,
    final_gas,
    dry_mass_kg: float,
    site_species,
    inlet,
    vent,
    product,
    liquid_drain=None,
) -> dict:
    """Map one/two site inventories to four species and return exact closure."""
    sorbed_change = np.zeros(4)
    for index, species in enumerate(site_species):
        sorbed_change[species] += dry_mass_kg * (
            final_sites[index] - initial_sites[index]
        )
    sorbed_change[1] += dry_mass_kg * (final_water - initial_water)
    sorbed_change[2] += dry_mass_kg * (final_n2 - initial_n2)
    sorbed_change[3] += dry_mass_kg * (final_o2 - initial_o2)
    inventory_change = (
        sorbed_change + np.asarray(final_gas) - np.asarray(initial_gas)
    )
    inlet = np.asarray(inlet)
    vent = np.asarray(vent)
    product = np.asarray(product)
    liquid_drain = (
        np.zeros(4)
        if liquid_drain is None
        else np.asarray(liquid_drain, dtype=float)
    )
    residual = inventory_change - inlet + vent + product + liquid_drain
    throughput = inlet + vent + product + liquid_drain
    relative = np.abs(residual) / np.maximum(throughput, 1e-9)
    return {
        "sorbed_change_mol": sorbed_change,
        "inventory_change_mol": inventory_change,
        "residual_mol": residual,
        "throughput_mol": throughput,
        "relative_error": relative,
    }


@dataclass
class CycleResult:
    summary: dict
    raw: dict


def _feed(
    config: dict,
    relative_humidity: float,
    water_vapour_partial_pressure_Pa: float | None = None,
):
    ambient = config["ambient"]
    ambient_K = float(ambient["temperature_C"]) + 273.15
    total_flow = (
        float(config["airflow"]["volume_flow_m3_s"])
        * float(ambient["pressure_Pa"])
        / (R * ambient_K)
    )
    p_water = (
        float(water_vapour_partial_pressure_Pa)
        if water_vapour_partial_pressure_Pa is not None
        else relative_humidity * water_saturation_pressure_Pa(ambient_K)
    )
    if not np.isfinite(p_water) or not 0.0 <= p_water < float(ambient["pressure_Pa"]):
        raise ValueError("water vapour partial pressure must be finite and below total pressure")
    y_water = p_water / float(ambient["pressure_Pa"])
    y_co2 = float(ambient["co2_ppm_dry"]) * 1e-6 * (1.0 - y_water)
    remainder = 1.0 - y_water - y_co2
    y = np.array([y_co2, y_water, 0.7905 * remainder, 0.2095 * remainder])
    return total_flow, p_water, y


def _build_collection_trace(
    *,
    cycle_number,
    dense_solution,
    product_indices,
    expected_state_size,
    phase_clocks,
    endpoint_product,
    cycle_start_state,
    desorption_start_state,
    cooling_end_state,
    cycle_end_state,
    phase_endpoint_checks,
    pump_audit,
):
    hot_desorption = next(
        row for row in phase_clocks if row["phase"] == "desorption"
    )
    samples = []
    for offset_s in COLLECTION_TRACE_OFFSETS_S:
        if offset_s > hot_desorption["duration_s"] + 1e-10:
            raise ValueError("collection trace offset exceeds hot desorption")
        if offset_s == 0.0:
            traced_state = np.asarray(desorption_start_state, dtype=float)
            sample_source = "exact_desorption_start_state"
        else:
            traced_state = np.asarray(dense_solution(offset_s), dtype=float)
            sample_source = "retained_dense_interpolant"
        traced_product = np.asarray(traced_state[product_indices], dtype=float)
        if (
            traced_state.shape != (expected_state_size,)
            or traced_product.shape != (4,)
            or not np.all(np.isfinite(traced_product))
        ):
            raise ValueError(
                "collection trace evidence is missing or nonfinite"
            )
        if np.any(traced_product < 0.0):
            raise ValueError("collection trace product is negative")
        samples.append(
            {
                "offset_s": offset_s,
                "cycle_elapsed_s": hot_desorption["start_s"] + offset_s,
                "source": sample_source,
                "conditioned_product_pump_suction_mol": {
                    "co2": float(traced_product[0]),
                    "water": float(traced_product[1]),
                    "n2": float(traced_product[2]),
                    "o2": float(traced_product[3]),
                },
            }
        )
    endpoint = np.asarray(endpoint_product, dtype=float)
    if (
        endpoint.shape != (4,)
        or not np.all(np.isfinite(endpoint))
        or np.any(endpoint < 0.0)
    ):
        raise ValueError("collection trace endpoint product is invalid")
    start_state = np.asarray(cycle_start_state, dtype=float)
    desorption_state = np.asarray(desorption_start_state, dtype=float)
    cooling_state = np.asarray(cooling_end_state, dtype=float)
    end_state = np.asarray(cycle_end_state, dtype=float)
    if (
        start_state.shape != (expected_state_size,)
        or desorption_state.shape != (expected_state_size,)
        or cooling_state.shape != (expected_state_size,)
        or end_state.shape != (expected_state_size,)
        or not np.all(np.isfinite(start_state))
        or not np.all(np.isfinite(desorption_state))
        or not np.all(np.isfinite(cooling_state))
        or not np.all(np.isfinite(end_state))
    ):
        raise ValueError("collection trace cycle states are invalid")
    return {
        "schema": "collection-trace/1",
        "cycle_number": cycle_number,
        "ledger_origin": {
            "rule": "after_cycle_ledger_reset",
            "cycle_start_s": 0.0,
            "reset_before_adsorption": True,
        },
        "cycle_start_state": start_state.tolist(),
        "desorption_start_state": desorption_state.tolist(),
        "cooling_end_state": cooling_state.tolist(),
        "cycle_end_state": end_state.tolist(),
        "phase_endpoint_checks": phase_endpoint_checks,
        "pump_audit": pump_audit,
        "offsets_s_after_hot_desorption_start": list(
            COLLECTION_TRACE_OFFSETS_S
        ),
        "hot_desorption_start_s": hot_desorption["start_s"],
        "hot_desorption_end_s": hot_desorption["end_s"],
        "phase_clocks": phase_clocks,
        "samples": samples,
        "collection_end": {
            "cycle_elapsed_s": next(
                row["end_s"]
                for row in phase_clocks
                if row["phase"] == "cooling"
            ),
            "includes_cooling_tail": True,
            "conditioned_product_pump_suction_mol": {
                "co2": float(cooling_state[product_indices][0]),
                "water": float(cooling_state[product_indices][1]),
                "n2": float(cooling_state[product_indices][2]),
                "o2": float(cooling_state[product_indices][3]),
            },
        },
        "cycle_end": {
            "cycle_elapsed_s": phase_clocks[-1]["end_s"],
            "phase": "repressurization",
            "includes_all_phases": True,
            "conditioned_product_pump_suction_mol": {
                "co2": float(endpoint[0]),
                "water": float(endpoint[1]),
                "n2": float(endpoint[2]),
                "o2": float(endpoint[3]),
            },
        },
        "interpolation": (
            "retained SciPy BDF solver interpolant; numerical interpolant "
            "error, not an extra IVP or physical solution"
        ),
    }


def _collection_phase_endpoint_checks(
    phase_records,
    *,
    ambient_temperature_K,
    ambient_pressure_Pa,
    regeneration_temperature_K,
    regeneration_pressure_Pa,
    cooling_endpoint_K,
):
    cooling_target = (
        float(cooling_endpoint_K)
        if cooling_endpoint_K is not None
        else float(ambient_temperature_K) + 1.0
    )
    adsorption = phase_records[0]
    evacuation = phase_records[1]
    desorption = phase_records[2]
    cooling = phase_records[3]
    repressurization = phase_records[4]
    checks = {
        "adsorption": {
            "pass": bool(
                abs(adsorption["temperature_end_K"] - ambient_temperature_K)
                <= 1.0
                and abs(
                    adsorption["pressure_end_Pa"] - ambient_pressure_Pa
                )
                <= ADSORPTION_PRESSURE_TOLERANCE_FRACTION
                * ambient_pressure_Pa
            ),
            "temperature_end_K": float(adsorption["temperature_end_K"]),
            "pressure_end_Pa": float(adsorption["pressure_end_Pa"]),
        },
        "evacuation": {
            "pass": bool(
                abs(
                    evacuation["pressure_end_Pa"]
                    - regeneration_pressure_Pa
                )
                <= 0.02 * regeneration_pressure_Pa
            ),
            "pressure_end_Pa": float(evacuation["pressure_end_Pa"]),
        },
        "desorption": {
            "pass": bool(
                abs(
                    desorption["temperature_end_K"]
                    - regeneration_temperature_K
                )
                <= 2.0
            ),
            "temperature_end_K": float(desorption["temperature_end_K"]),
        },
        "cooling": {
            "pass": bool(cooling["temperature_end_K"] <= cooling_target),
            "temperature_end_K": float(cooling["temperature_end_K"]),
            "target_K": cooling_target,
        },
        "repressurization": {
            "pass": bool(
                repressurization["pressure_end_Pa"]
                >= 0.99 * ambient_pressure_Pa
            ),
            "pressure_end_Pa": float(repressurization["pressure_end_Pa"]),
            "target_min_Pa": 0.99 * ambient_pressure_Pa,
        },
    }
    checks["all_pass"] = all(row["pass"] for row in checks.values())
    return checks


def _collection_pump_audit(solution, pump_evaluate, phase):
    values = [
        pump_evaluate(state)
        for state in np.asarray(solution.y, dtype=float).T
    ]
    flows = np.array(
        [float(row["molar_flow_mol_s"]) for row in values],
        dtype=float,
    )
    overrun = np.array(
        [float(row["media_temperature_overrun_K"]) for row in values],
        dtype=float,
    )
    return {
        "phase": phase,
        "accepted_state_count": len(values),
        "minimum_molar_flow_mol_s": float(np.min(flows)),
        "maximum_molar_flow_mol_s": float(np.max(flows)),
        "positive_flow_pass": bool(np.all(flows >= 0.0)),
        "upper_domain_exceeded": bool(
            any(row["upper_domain_exceeded"] for row in values)
        ),
        "below_or_at_ultimate_observed": bool(
            any(row["below_or_at_ultimate"] for row in values)
        ),
        "maximum_media_temperature_overrun_K": float(np.max(overrun)),
        "media_temperature_pass": bool(np.max(overrun) <= 1e-9),
    }


def run_cycle(
    adapter,
    config: dict,
    *,
    relative_humidity: float,
    rtol: float = 2.5e-8,
    max_step_s: float = 1.0,
    maximum_cycles: int = 64,
    rhs_only: bool = False,
    progress_callback=None,
    start_state=None,
    initial_cycle_index: int = 0,
    minimum_fresh_cycles: int | None = None,
    water_vapour_partial_pressure_Pa: float | None = None,
    collection_trace: bool = False,
):
    started = time.monotonic()
    ambient_K = float(config["ambient"]["temperature_C"]) + 273.15
    reference_K = 298.15
    ambient_p = float(config["ambient"]["pressure_Pa"])
    regen_K = float(config["cycle"]["regeneration_temperature_C"]) + 273.15
    regen_p = float(config["cycle"]["regeneration_pressure_Pa"])
    cooling_stop_offset_K = config["cycle"].get("cooling_stop_offset_K")
    cooling_endpoint_K = (
        ambient_K + float(cooling_stop_offset_K)
        if cooling_stop_offset_K is not None
        else None
    )
    adsorption_control = config["cycle"].get("adsorption_control", FIXED_CONTROL)
    if adsorption_control not in (FIXED_CONTROL, PRESSURE_CONTROL):
        raise ValueError(f"unknown adsorption control: {adsorption_control}")
    pressure_response_tau_s = (
        float(config["cycle"].get("pressure_response_tau_s", 0.5))
        if adsorption_control == PRESSURE_CONTROL
        else None
    )
    if pressure_response_tau_s is not None and pressure_response_tau_s <= 0.0:
        raise ValueError("pressure response time must be positive")
    initial_step_policy = config.get("solver", {}).get(
        "initial_step_policy"
    )
    if initial_step_policy not in (None, INITIAL_STEP_POLICY):
        raise ValueError(f"unknown initial step policy: {initial_step_policy}")
    initial_step_max_s = float(
        config.get("solver", {}).get(
            "initial_step_max_s", DEFAULT_INITIAL_STEP_MAX_S
        )
    )
    initial_step_depletion_fraction = float(
        config.get("solver", {}).get(
            "initial_step_depletion_fraction",
            DEFAULT_DEPLETION_FRACTION,
        )
    )
    mass = float(config["bed"]["dry_sorbent_mass_kg"])
    geometry = build_geometry(config)
    volume = float(geometry.gas_volume_m3)
    metal = float(geometry.metal_mass_kg)
    metal_cp = float(config["geometry"]["metal_cp_J_kg_K"])
    heater_capacity = float(config["thermal"]["heater_heat_capacity_J_K"])
    heater_limit = float(config["thermal"]["heater_power_W"])
    cooling_limit = float(config["utilities"]["wall_cooling"]["max_cooling_capacity_W"])
    cooling_cop = float(config["utilities"]["wall_cooling"]["cop"])
    control_gain = float(config["thermal"]["setpoint_gain_W_K"])
    loss_ua = float(config["thermal"]["heat_loss_U_W_m2_K"]) * float(geometry.heated_area_m2)
    feed_flow, _p_water, y_feed = _feed(
        config,
        relative_humidity,
        water_vapour_partial_pressure_Pa=water_vapour_partial_pressure_Pa,
    )
    fan = airflow(config, geometry)
    pump_profile = pump_profile_for(config)
    bank_count = int(config.get("pump", {}).get("bank_count", 1))
    pump_mode = config["coupled_through_thickness"]["pump_flow_basis"]
    conditioner = pump_conditioner_config(config)
    conditioner_enabled = conditioner is not None
    if not isinstance(collection_trace, bool):
        raise TypeError("collection_trace must be a boolean")
    if collection_trace and not conditioner_enabled:
        raise ValueError("collection trace requires the conditioner")

    ns = adapter.n_sites
    Q = slice(0, ns)
    QW, QN, QO, TEMP = ns, ns + 1, ns + 2, ns + 3
    GAS = slice(ns + 4, ns + 8)
    M = ns + 8
    IN = slice(M, M + 4)
    PRODUCT = slice(M + 4, M + 8)
    VENT = slice(M + 8, M + 12)
    HEAT, PUMP, FAN, COOL, INLET_H, OUTLET_H, LOSS = range(M + 12, M + 19)
    BASE_LEDGER_END = M + 19
    if conditioner_enabled:
        CONDITIONED_PRODUCT = slice(BASE_LEDGER_END, BASE_LEDGER_END + 4)
        CONDITIONED_VENT = slice(
            BASE_LEDGER_END + 4, BASE_LEDGER_END + 8
        )
        DIRECT_VENT = slice(BASE_LEDGER_END + 8, BASE_LEDGER_END + 12)
        LIQUID_DRAIN = slice(BASE_LEDGER_END + 12, BASE_LEDGER_END + 16)
        CONDITIONER_COOL = BASE_LEDGER_END + 16
        EXTERNAL_OUTLET_H = BASE_LEDGER_END + 17
        SIZE = BASE_LEDGER_END + 18
    else:
        CONDITIONED_PRODUCT = None
        CONDITIONED_VENT = None
        DIRECT_VENT = None
        LIQUID_DRAIN = None
        CONDITIONER_COOL = None
        EXTERNAL_OUTLET_H = None
        SIZE = BASE_LEDGER_END
    Cv = CP_GAS - R

    def pressure(state):
        return float(np.sum(np.maximum(state[GAS], 0.0)) * R * max(state[TEMP], 0.0) / volume)

    def composition(state):
        n = np.maximum(state[GAS], 0.0)
        return n / n.sum() if n.sum() > 1e-30 else np.zeros(4)

    def pump_inlet_temperature(state):
        bed_temperature = float(state[TEMP])
        if conditioner_enabled:
            return min(bed_temperature, conditioner.target_temperature_K)
        return bed_temperature

    def pump_value_at_state(state):
        return rate_at_pressure(
            pump_profile,
            max(pressure(state), 1.0),
            max(pump_inlet_temperature(state), 250.0),
            bank_count=bank_count,
            mode=pump_mode,
        )

    def conditioner_details_at_state(state):
        if not conditioner_enabled:
            return None
        pump_value = pump_value_at_state(state)
        alpha = float(
            np.clip(
                (pressure(state) - regen_p) / 2000.0 + 0.02,
                0.0,
                1.0,
            )
        )
        return evaluate_pump_conditioner(
            bed_mole_fractions=composition(state),
            pressure_Pa=pressure(state),
            bed_temperature_K=float(state[TEMP]),
            pump_capacity_mol_s=float(
                pump_value["molar_flow_mol_s"]
            ),
            alpha=alpha,
            target_temperature_K=conditioner.target_temperature_K,
            total_cooling_power_W=conditioner.total_cooling_power_W,
            cooling_cop=conditioner.cooling_cop,
            pump_electric_power_W=float(
                pump_value["electric_power_W"]
            ),
        )

    def storage_loading_vector(state):
        values = np.r_[state[Q], state[QW]]
        if adapter.n_storage >= ns + 3:
            values = np.r_[values, state[QN], state[QO]]
        if values.shape != (adapter.n_storage,):
            raise ValueError(
                f"storage layout mismatch: {values.shape} vs "
                f"{(adapter.n_storage,)}"
            )
        return values

    def energy_gradient(loading_gradient):
        gradient = np.zeros(ns + 3, dtype=float)
        gradient[: adapter.n_storage] = loading_gradient
        return gradient

    def sorption_details(state):
        gas = composition(state)
        p = np.maximum(gas * pressure(state), 0.0)
        return adapter.site_rates(float(state[TEMP]), p, state[Q], float(state[QW]), state[[QN, QO]])

    def gas_rates(component_rates):
        result = np.zeros(4)
        for index, species in enumerate(adapter.site_species):
            result[species] += component_rates[index]
        result[1] += component_rates[ns]
        result[2] += component_rates[ns + 1]
        result[3] += component_rates[ns + 2]
        return result

    def inventory(state):
        loads = storage_loading_vector(state)
        e_ads, _cp_ads, _h_ads = adapter.storage_energy_terms(
            float(state[TEMP]), loads
        )
        ug = Cv * (state[TEMP] - reference_K)
        return float(
            (mass * adapter.solid_cp_J_kg_K + metal * metal_cp + heater_capacity)
            * (state[TEMP] - reference_K)
            + mass * e_ads
            + float(state[GAS] @ ug)
        )

    def controller(state, component_rates, dn_gas, *, requested_heat, requested_cooling, loss, inlet_h, outlet_h):
        T = float(state[TEMP])
        loads = storage_loading_vector(state)
        _e_ads, cp_ads, h_ads = adapter.storage_energy_terms(T, loads)
        rate_energy = energy_gradient(h_ads)
        ug = Cv * (T - reference_K)
        d_fixed = mass * float(component_rates @ rate_energy) + float(dn_gas @ ug)
        capacity = (
            mass * (adapter.solid_cp_J_kg_K + cp_ads)
            + metal * metal_cp + heater_capacity + float(state[GAS] @ Cv)
        )
        heat = min(max(requested_heat, 0.0), heater_limit)
        cooling = min(max(requested_cooling, 0.0), cooling_limit)
        tdot = (heat - cooling - loss + inlet_h - outlet_h - d_fixed) / max(capacity, 1e-12)
        residual = capacity * tdot + d_fixed - heat + cooling + loss - inlet_h + outlet_h
        if abs(residual) > 1e-7:
            raise RuntimeError(f"RHS first-law residual {residual}")
        return tdot, heat, cooling

    def pressure_coupled_terms(state, component_rates):
        """Return the pressure-regulated adsorption coupling at one state."""
        raw_gas = np.asarray(state[GAS], dtype=float)
        if np.any(raw_gas < -1e-12):
            raise FloatingPointError(
                "pressure-regulated adsorption received negative gas inventory"
            )
        T = float(state[TEMP])
        total_gas = float(np.sum(raw_gas))
        if total_gas <= 1e-30 or T <= 0.0:
            raise FloatingPointError(
                "pressure-regulated adsorption requires positive gas and temperature"
            )
        gas = composition(state)
        gas_removed = gas_rates(component_rates)
        inlet = feed_flow * y_feed
        beta = volume / (R * T)
        pressure_derivative_required = (
            ambient_p - pressure(state)
        ) / pressure_response_tau_s
        outlet_flow_zero_rate = (
            feed_flow
            - mass * float(np.sum(gas_removed))
            - beta * pressure_derivative_required
        )
        dn_zero_rate = (
            inlet - outlet_flow_zero_rate * gas - mass * gas_removed
        )
        ug = Cv * (T - reference_K)
        h_in = CP_GAS * (ambient_K - reference_K) + R * reference_K
        h_out = CP_GAS * (T - reference_K) + R * reference_K
        inlet_h = float(inlet @ h_in)
        outlet_h_per_mol = float(gas @ h_out)
        loads = storage_loading_vector(state)
        _e_ads, cp_ads, h_ads = adapter.storage_energy_terms(T, loads)
        rate_energy = energy_gradient(h_ads)
        zero_rate_duty = (
            mass * float(component_rates @ rate_energy)
            + float(dn_zero_rate @ ug)
            - inlet_h
            + outlet_flow_zero_rate * outlet_h_per_mol
        )
        requested_duty = zero_rate_duty + control_gain * (ambient_K - T)
        heat, cooling = split_actuator_duty(
            requested_duty,
            heater_limit_W=heater_limit,
            cooling_limit_W=cooling_limit,
        )
        fixed_capacity = (
            mass * (adapter.solid_cp_J_kg_K + cp_ads)
            + metal * metal_cp
            + heater_capacity
            + float(raw_gas @ Cv)
        )
        coupled_capacity = fixed_capacity + total_gas * R
        if not np.isfinite(coupled_capacity) or coupled_capacity <= 0.0:
            raise FloatingPointError(
                "pressure-regulated adsorption heat capacity invalid"
            )
        tdot = (heat - cooling - zero_rate_duty) / coupled_capacity
        outlet_flow = outlet_flow_zero_rate + (total_gas / T) * tdot
        if outlet_flow < 0.0:
            raise FloatingPointError(
                "pressure-regulated adsorption required negative outlet flow"
            )
        outlet = outlet_flow * gas
        dn = inlet - outlet - mass * gas_removed
        pressure_derivative = R / volume * (
            T * float(np.sum(dn)) + total_gas * tdot
        )
        eos_scale = max(
            abs(pressure_derivative_required),
            abs(pressure_derivative),
            1.0,
        )
        if abs(
            pressure_derivative - pressure_derivative_required
        ) > 1e-9 * eos_scale:
            raise RuntimeError(
                "pressure-regulated adsorption EOS coupling mismatch"
            )
        return {
            "gas_removed": gas_removed,
            "inlet": inlet,
            "outlet": outlet,
            "dn": dn,
            "total_gas": total_gas,
            "pressure_derivative_required": pressure_derivative_required,
            "pressure_derivative": pressure_derivative,
            "outlet_flow_zero_rate": outlet_flow_zero_rate,
            "outlet_flow": outlet_flow,
            "zero_rate_duty": zero_rate_duty,
            "requested_duty": requested_duty,
            "heat": heat,
            "cooling": cooling,
            "tdot": tdot,
            "inlet_h": inlet_h,
            "outlet_h": outlet_flow * outlet_h_per_mol,
            "coupled_capacity": coupled_capacity,
        }

    def rhs(phase):
        def derivative(_t, state):
            runtime_control = config.get("_runtime_control") or {}
            cancel_check = runtime_control.get("cancel_check")
            if cancel_check is not None:
                cancel_check()
            deadline = runtime_control.get("deadline_monotonic")
            if deadline is not None and time.monotonic() > float(deadline):
                raise CycleFailure(
                    "integration exceeded its declared wall-time cap",
                    phase=phase,
                    cycle=1,
                    records=[],
                )
            temperature_limit = (
                runtime_control["temperature_guard_limit_K"]
                if "temperature_guard_limit_K" in runtime_control
                else float(adapter.maximum_bed_temperature_K)
            )
            if (
                temperature_limit is not None
                and float(state[TEMP]) > float(temperature_limit) + 1e-9
            ):
                raise CycleFailure(
                    "scenario temperature guard reached",
                    phase=phase,
                    cycle=1,
                    records=[],
                )
            inlet_h = 0.0
            outlet_h = 0.0
            if phase == "adsorption":
                component_rates = sorption_details(state)
                component_rates = limit_adsorption_demand(
                    component_rates,
                    site_species=adapter.site_species,
                    available_mol_s=feed_flow * y_feed,
                    dry_mass_kg=mass,
                )
            else:
                component_rates = sorption_details(state)
            gas_removed = gas_rates(component_rates)
            out = np.zeros(SIZE)
            out[Q] = component_rates[Q]
            out[QW:QO + 1] = component_rates[ns:ns + 3]
            gas = composition(state)
            if phase == "adsorption":
                inlet = feed_flow * y_feed
                T = float(state[TEMP])
                h_in = CP_GAS * (ambient_K - reference_K) + R * reference_K
                h_out = CP_GAS * (T - reference_K) + R * reference_K
                inlet_h = float(inlet @ h_in)
                outlet_h_per_mol = float(gas @ h_out)
                if adsorption_control == PRESSURE_CONTROL:
                    coupling = pressure_coupled_terms(
                        state, component_rates
                    )
                    outlet = coupling["outlet"]
                    dn = coupling["dn"]
                    tdot = coupling["tdot"]
                    heat = coupling["heat"]
                    cooling = coupling["cooling"]
                    outlet_h = coupling["outlet_h"]
                else:
                    outlet_flow = feed_flow - mass * float(
                        np.sum(gas_removed)
                    )
                    if outlet_flow < -1e-12:
                        raise RuntimeError(
                            "adsorption sorption demand exceeds feed flow: "
                            f"{outlet_flow}"
                        )
                    outlet_flow = max(outlet_flow, 0.0)
                    outlet = outlet_flow * gas
                    dn = inlet - outlet - mass * gas_removed
                    outlet_h = float(outlet @ h_out)
                    _e_ads, cp_ads, h_ads = adapter.storage_energy_terms(
                        T, storage_loading_vector(state)
                    )
                    rate_energy = energy_gradient(h_ads)
                    imbalance = (
                        mass * float(component_rates @ rate_energy)
                        - inlet_h
                        + outlet_h
                    )
                    fixed_derivative = (
                        mass * float(component_rates @ rate_energy)
                        + float(dn @ Cv) * (T - reference_K)
                    )
                    requested_duty = setpoint_requested_duty(
                        fixed_energy_derivative_W=fixed_derivative,
                        loss_W=0.0,
                        inlet_enthalpy_W=inlet_h,
                        outlet_enthalpy_W=outlet_h,
                        bed_temperature_K=T,
                        target_temperature_K=ambient_K,
                        gain_W_K=control_gain,
                    )
                    requested_heat, requested_cooling = split_actuator_duty(
                        requested_duty,
                        heater_limit_W=heater_limit,
                        cooling_limit_W=cooling_limit,
                    )
                    tdot, heat, cooling = controller(
                        state,
                        component_rates,
                        dn,
                        requested_heat=requested_heat,
                        requested_cooling=requested_cooling,
                        loss=0.0,
                        inlet_h=inlet_h,
                        outlet_h=outlet_h,
                    )
                out[IN] = inlet
                out[GAS] = dn
                out[VENT] = outlet
                if conditioner_enabled:
                    out[DIRECT_VENT] = outlet
                    out[EXTERNAL_OUTLET_H] = outlet_h
                out[FAN] = float(fan["fan_power_W"])
            elif phase == "repressurization":
                inlet = feed_flow * y_feed
                dn = inlet - mass * gas_removed
                out[GAS] = dn
                out[IN] = inlet
                T = float(state[TEMP])
                inlet_h = float(
                    inlet
                    @ (
                        CP_GAS * (ambient_K - reference_K)
                        + R * reference_K
                    )
                )
                loss = loss_ua * max(T - ambient_K, 0.0)
                _e_ads, cp_ads, h_ads = adapter.storage_energy_terms(
                    T, storage_loading_vector(state)
                )
                rate_energy = energy_gradient(h_ads)
                fixed_derivative = (
                    mass * float(component_rates @ rate_energy)
                    + float(dn @ Cv) * (T - reference_K)
                )
                requested_duty = setpoint_requested_duty(
                    fixed_energy_derivative_W=fixed_derivative,
                    loss_W=loss,
                    inlet_enthalpy_W=inlet_h,
                    outlet_enthalpy_W=0.0,
                    bed_temperature_K=T,
                    target_temperature_K=ambient_K,
                    gain_W_K=control_gain,
                )
                requested_heat, requested_cooling = split_actuator_duty(
                    requested_duty,
                    heater_limit_W=heater_limit,
                    cooling_limit_W=cooling_limit,
                )
                tdot, heat, cooling = controller(state, component_rates, dn, requested_heat=requested_heat, requested_cooling=requested_cooling, loss=loss, inlet_h=inlet_h, outlet_h=0.0)
                out[LOSS] = loss
            else:
                pump_value = pump_value_at_state(state)
                max_pump = float(pump_value["molar_flow_mol_s"])
                alpha = float(
                    np.clip(
                        (pressure(state) - regen_p) / 2000.0 + 0.02,
                        0.0,
                        1.0,
                    )
                )
                T = float(state[TEMP])
                _e_ads, cp_ads, h_ads = adapter.storage_energy_terms(
                    T, storage_loading_vector(state)
                )
                rate_energy = energy_gradient(h_ads)
                if conditioner_enabled:
                    conditioner_details = evaluate_pump_conditioner(
                        bed_mole_fractions=gas,
                        pressure_Pa=pressure(state),
                        bed_temperature_K=T,
                        pump_capacity_mol_s=max_pump,
                        alpha=alpha,
                        target_temperature_K=conditioner.target_temperature_K,
                        total_cooling_power_W=conditioner.total_cooling_power_W,
                        cooling_cop=conditioner.cooling_cop,
                        pump_electric_power_W=float(
                            pump_value["electric_power_W"]
                        ),
                    )
                    outlet = np.asarray(
                        conditioner_details["bed_outlet_mol_s"],
                        dtype=float,
                    )
                    conditioned_gas = np.asarray(
                        conditioner_details["conditioned_gas_mol_s"],
                        dtype=float,
                    )
                    liquid_drain = np.asarray(
                        conditioner_details["liquid_drain_mol_s"],
                        dtype=float,
                    )
                    if phase == "evacuation":
                        out[CONDITIONED_VENT] = conditioned_gas
                    else:
                        out[CONDITIONED_PRODUCT] = conditioned_gas
                    out[LIQUID_DRAIN] = liquid_drain
                    out[CONDITIONER_COOL] = conditioner_details[
                        "conditioner_cooling_W"
                    ]
                    external_outlet_h = (
                        conditioner_details["conditioned_gas_h_W"]
                        + conditioner_details["liquid_drain_h_W"]
                    )
                    out[EXTERNAL_OUTLET_H] = external_outlet_h
                    outlet_h = conditioner_details["bed_outlet_h_W"]
                    available_bed_cooling = conditioner_details[
                        "available_bed_cooling_W"
                    ]
                else:
                    pump_rate = max_pump * alpha
                    outlet = pump_rate * gas
                    outlet_h = float(
                        outlet
                        @ (
                            CP_GAS * (T - reference_K)
                            + R * reference_K
                        )
                    )
                    available_bed_cooling = cooling_limit
                dn = -outlet - mass * gas_removed
                out[GAS] = dn
                out[VENT if phase == "evacuation" else PRODUCT] = outlet
                out[PUMP] = float(pump_value["electric_power_W"])
                loss = loss_ua * max(T - ambient_K, 0.0)
                if phase == "desorption":
                    requested_heat = min(heater_limit, 5000.0 * max(regen_K - T, 0.0))
                    requested_cooling = 0.0
                elif phase == "cooling":
                    requested_heat = 0.0
                    requested_cooling = min(cooling_limit, 1000.0 * max(T - ambient_K, 0.0))
                else:
                    imbalance = (
                        mass * float(component_rates @ rate_energy)
                        + float(dn @ Cv) * (T - reference_K)
                        + outlet_h
                        + loss
                    )
                    requested_heat = max(imbalance, 0.0)
                    requested_cooling = max(-imbalance, 0.0)
                requested_cooling = min(
                    requested_cooling, available_bed_cooling
                )
                tdot, heat, cooling = controller(state, component_rates, dn, requested_heat=requested_heat, requested_cooling=requested_cooling, loss=loss, inlet_h=0.0, outlet_h=outlet_h)
                out[LOSS] = loss
            out[TEMP] = tdot
            out[HEAT] = heat
            out[COOL] = cooling
            out[INLET_H] = inlet_h
            out[OUTLET_H] = outlet_h
            return out
        return derivative

    def accepted_phase_diagnostics(solution, phase):
        """Summarize accepted solver states without mutating the integration."""
        accepted = np.asarray(solution.y, dtype=float)
        temperatures = accepted[TEMP]
        gas_inventory = np.maximum(accepted[GAS], 0.0)
        total_gas = gas_inventory.sum(axis=0)
        pressures = total_gas * R * np.maximum(temperatures, 0.0) / volume
        water_fraction = np.divide(
            gas_inventory[1],
            total_gas,
            out=np.zeros_like(total_gas),
            where=total_gas > 1e-30,
        )
        water_partial_pressure = water_fraction * pressures
        saturation = diagnostic_water_saturation_ratios(
            adapter,
            temperatures,
            water_partial_pressure,
        )
        saturation_ratios = saturation["saturation_ratios"]
        loading_values = np.vstack(
            [accepted[Q], accepted[[QW, QN, QO]]]
        )
        water_support_fraction = accepted[QW] / max(
            float(adapter.water_scale_mol_kg), 1e-30
        )
        nonfinite = bool(
            not np.all(np.isfinite(accepted))
            or not np.all(np.isfinite(pressures))
            or not np.all(np.isfinite(saturation_ratios))
        )
        feed_limited = False
        if phase == "adsorption":
            for accepted_state in accepted.T:
                raw_rates = sorption_details(accepted_state)
                limited_rates = limit_adsorption_demand(
                    raw_rates,
                    site_species=adapter.site_species,
                    available_mol_s=feed_flow * y_feed,
                    dry_mass_kg=mass,
                )
                if np.any(
                    (raw_rates > 0.0)
                    & (limited_rates < raw_rates - 1e-14)
                ):
                    feed_limited = True
                    break
        maximum_saturation_ratio = float(np.max(saturation_ratios))
        maximum_water_support_fraction = float(np.max(water_support_fraction))
        maximum_temperature = float(np.max(temperatures))
        supersaturated = maximum_saturation_ratio > 1.0 + 1e-10
        temperature_excursion = maximum_temperature > (
            float(adapter.maximum_bed_temperature_K) + 1e-9
        )
        water_support_excursion = maximum_water_support_fraction > 1.0 + 1e-10
        source_conditional = adapter.material_id.startswith("calf")
        return {
            "phase": phase,
            "sampling_scope": (
                "solve_ivp accepted states in fresh near-CSS continuation cycles"
            ),
            "accepted_state_count": int(accepted.shape[1]),
            "temperature_K": {
                "minimum": float(np.min(temperatures)),
                "maximum": maximum_temperature,
            },
            "pressure_Pa": {
                "minimum": float(np.min(pressures)),
                "maximum": float(np.max(pressures)),
            },
            "gas_inventory_mol": {
                "minimum_total": float(np.min(total_gas)),
                "maximum_total": float(np.max(total_gas)),
                "minimum_by_species": gas_inventory.min(axis=1).tolist(),
                "maximum_by_species": gas_inventory.max(axis=1).tolist(),
            },
            "loading_mol_kg": {
                "minimum_by_component": loading_values.min(axis=1).tolist(),
                "maximum_by_component": loading_values.max(axis=1).tolist(),
            },
            "maximum_water_saturation_ratio": maximum_saturation_ratio,
            "maximum_water_support_fraction": maximum_water_support_fraction,
            "saturation_diagnostic_law": saturation[
                "saturation_diagnostic_law"
            ],
            "flags": {
                "actual_supersaturation_excursion": supersaturated,
                "outside_no_condensation_model": supersaturated,
                "temperature_support_excursion": temperature_excursion,
                "water_support_excursion": water_support_excursion,
                "adsorption_feed_limited": feed_limited,
                "source_transfer_conditional": source_conditional,
                "nonfinite_accepted_state": nonfinite,
            },
        }

    def aggregate_envelope(phase_records):
        diagnostics = [record["envelope"] for record in phase_records]
        flags = {
            key: any(item["flags"][key] for item in diagnostics)
            for key in diagnostics[0]["flags"]
        }
        supersaturated = flags["actual_supersaturation_excursion"]
        if supersaturated:
            envelope_classification = "outside_no_condensation_model"
        elif flags["temperature_support_excursion"] or flags["water_support_excursion"]:
            envelope_classification = "outside_material_support"
        else:
            envelope_classification = "conditional_model_envelope"
        return {
            "sampling_scope": (
                "fresh near-CSS cycles; accepted solve_ivp states only"
            ),
            "accepted_state_count": sum(
                item["accepted_state_count"] for item in diagnostics
            ),
            "maximum_water_saturation_ratio": max(
                item["maximum_water_saturation_ratio"] for item in diagnostics
            ),
            "maximum_water_support_fraction": max(
                item["maximum_water_support_fraction"] for item in diagnostics
            ),
            "saturation_diagnostic_law": diagnostics[0][
                "saturation_diagnostic_law"
            ],
            "minimum_temperature_K": min(
                item["temperature_K"]["minimum"] for item in diagnostics
            ),
            "maximum_temperature_K": max(
                item["temperature_K"]["maximum"] for item in diagnostics
            ),
            "minimum_pressure_Pa": min(
                item["pressure_Pa"]["minimum"] for item in diagnostics
            ),
            "maximum_pressure_Pa": max(
                item["pressure_Pa"]["maximum"] for item in diagnostics
            ),
            "minimum_total_gas_inventory_mol": min(
                item["gas_inventory_mol"]["minimum_total"]
                for item in diagnostics
            ),
            "maximum_total_gas_inventory_mol": max(
                item["gas_inventory_mol"]["maximum_total"]
                for item in diagnostics
            ),
            "flags": flags,
            "classification": envelope_classification,
            "full_physical_qualification": False,
        }

    def initial_state():
        if start_state is not None:
            value = np.asarray(start_state, dtype=float).copy()
            if value.shape != (SIZE,):
                raise ValueError(
                    f"start state has shape {value.shape}, expected {(SIZE,)}"
                )
            if not np.all(np.isfinite(value)):
                raise ValueError("start state contains nonfinite values")
            return value
        state = np.zeros(SIZE)
        state[TEMP] = ambient_K
        state[GAS] = ambient_p * volume / (R * ambient_K) * y_feed
        if config.get("initial_state", {}).get("basis") == "equilibrated":
            partials = ambient_p * y_feed
            equilibrium = adapter.equilibrium(ambient_K, partials)
            co2_indices = [
                index
                for index, species in enumerate(adapter.species_for_storage)
                if species == 0
            ]
            state[Q] = equilibrium[co2_indices]
            state[QW] = equilibrium[adapter.water_storage_index]
            inert = {
                int(species): equilibrium[index]
                for index, species in enumerate(adapter.species_for_storage)
                if int(species) in (2, 3)
            }
            state[QN] = inert.get(2, 0.0)
            state[QO] = inert.get(3, 0.0)
        return state

    if rhs_only:
        return {
            "rhs_by_phase": {phase: rhs(phase) for phase in PHASES},
            "initial_state": initial_state,
            "inventory": inventory,
            "pressure": pressure,
            "pump_evaluate": pump_value_at_state,
            "adsorption_control": adsorption_control,
            "pressure_response_tau_s": pressure_response_tau_s,
            "control_identity": control_identity(config),
            "pump_conditioner_identity": pump_conditioner_identity(config),
            "conditioner_evaluate_state": conditioner_details_at_state,
            "physical_columns": ns + 8,
            "pump_assumptions": {
                "profile_name": pump_profile.name,
                "ultimate_pressure_Pa": float(
                    pump_profile.ultimate_pressure_Pa
                ),
                "media_temperature_limit_C": pump_profile.media_temperature_limit_C,
                "bank_count": bank_count,
                "flow_basis": pump_mode,
                "inlet_temperature_basis": (
                    (
                        "conditioned gas entering the pump at "
                        "Tout=min(Tbed,328.15K); negligible pressure drop "
                        "and line holdup"
                    )
                    if conditioner_enabled
                    else "gas entering the pump from the bed, no inlet cooler"
                ),
            },
            "pressure_coupled_terms": lambda value: pressure_coupled_terms(
                value, np.zeros(ns + 3)
            ),
            "indices": {
                "Q": Q,
                "QW": QW,
                "QN": QN,
                "QO": QO,
                "TEMP": TEMP,
                "GAS": GAS,
                "IN": IN,
                "PRODUCT": PRODUCT,
                "VENT": VENT,
                "HEAT": HEAT,
                "PUMP": PUMP,
                "FAN": FAN,
                "COOL": COOL,
                "INLET_H": INLET_H,
                "OUTLET_H": OUTLET_H,
                "LOSS": LOSS,
                "CONDITIONED_PRODUCT": CONDITIONED_PRODUCT,
                "CONDITIONED_VENT": CONDITIONED_VENT,
                "DIRECT_VENT": DIRECT_VENT,
                "LIQUID_DRAIN": LIQUID_DRAIN,
                "CONDITIONER_COOL": CONDITIONER_COOL,
                "EXTERNAL_OUTLET_H": EXTERNAL_OUTLET_H,
            },
            "ledger_columns": SIZE - (ns + 8),
            "state_size": SIZE,
        }

    state = initial_state()
    cycles = []
    initial_step_records = []
    converged = False
    all_closure = True
    final_cycle_trace = None
    minimum_cycles = (
        int(config["cycle"]["minimum_cycles"])
        if minimum_fresh_cycles is None
        else int(minimum_fresh_cycles)
    )
    for fresh_cycle_number in range(1, maximum_cycles + 1):
        cycle_number = int(initial_cycle_index) + fresh_cycle_number
        initial = state.copy()
        state[M:] = 0.0
        cycle_start_state = state.copy()
        phase_records = []
        phase_clocks = []
        phase_end_states = {}
        phase_start_states = {}
        phase_pump_audits = []
        dense_solutions = {}
        cycle_clock_s = 0.0
        for phase, duration in zip(
            PHASES,
            (
                float(config["cycle"]["adsorption_s"]),
                float(config["cycle"]["evacuation_s"]),
                float(config["cycle"]["desorption_s"]),
                float(config["cycle"]["cooling_max_s"]),
                float(config["cycle"]["repressurization_max_s"]),
            ),
        ):
            before = state.copy()
            phase_start_states[phase] = before.copy()
            events = []
            if phase == "evacuation":
                ev = lambda _t, s: pressure(s) - regen_p
                ev.terminal = True
                ev.direction = -1
                events.append(ev)
            if phase == "repressurization":
                ev = lambda _t, s: pressure(s) - 0.995 * ambient_p
                ev.terminal = True
                ev.direction = 1
                events.append(ev)
            if phase == "cooling" and cooling_endpoint_K is not None:
                ev = lambda _t, s: s[TEMP] - cooling_endpoint_K
                ev.terminal = True
                ev.direction = -1
                events.append(ev)
            atol = np.r_[
                np.full(ns + 3, 1e-9),
                np.full(4, 1e-10),
                np.full(SIZE - ns - 7, 1e-7),
            ]
            first_step_value = None
            first_step_record = None
            if initial_step_policy == INITIAL_STEP_POLICY:
                first_step_record = positive_gas_first_step(
                    rhs(phase),
                    0.0,
                    state,
                    duration,
                    max_step_s=max_step_s,
                    rtol=rtol,
                    atol=atol,
                    gas_indices=GAS,
                    maximum_first_step_s=initial_step_max_s,
                    depletion_fraction=initial_step_depletion_fraction,
                )
                first_step_value = first_step_record["first_step_s"]
                first_step_record = {
                    **first_step_record,
                    "cycle": cycle_number,
                    "phase": phase,
                }
                initial_step_records.append(first_step_record)
            solve_options = {}
            if collection_trace:
                solve_options["dense_output"] = True
            solution = solve_ivp(
                rhs(phase),
                (0.0, duration),
                state,
                method="BDF",
                rtol=rtol,
                max_step=max_step_s,
                events=events,
                atol=atol,
                first_step=first_step_value,
                **solve_options,
            )
            if not solution.success:
                raise CycleFailure(solution.message, phase=phase, cycle=cycle_number, records=cycles + phase_records)
            if collection_trace:
                if solution.sol is None:
                    raise CycleFailure(
                        "collection trace dense output unavailable",
                        phase=phase,
                        cycle=cycle_number,
                        records=cycles + phase_records,
                    )
                dense_solutions[phase] = solution.sol
                phase_pump_audits.append(
                    _collection_pump_audit(
                        solution, pump_value_at_state, phase
                    )
                )
            phase_end_s = cycle_clock_s + float(solution.t[-1])
            phase_clocks.append(
                {
                    "phase": phase,
                    "start_s": cycle_clock_s,
                    "end_s": phase_end_s,
                    "duration_s": float(solution.t[-1]),
                }
            )
            cycle_clock_s = phase_end_s
            state = solution.y[:, -1]
            if np.any(state[:ns + 3] < -1e-8) or np.any(state[GAS] < -1e-8):
                raise CycleFailure("negative inventory", phase=phase, cycle=cycle_number, records=cycles + phase_records)
            state[:ns + 3] = np.maximum(state[:ns + 3], 0.0)
            state[GAS] = np.maximum(state[GAS], 0.0)
            phase_end_states[phase] = state.copy()
            boundary = (state[HEAT] - before[HEAT] - state[COOL] + before[COOL] - state[LOSS] + before[LOSS] + state[INLET_H] - before[INLET_H] - state[OUTLET_H] + before[OUTLET_H])
            phase_records.append(
                {
                    "phase": phase,
                    "duration_s": float(solution.t[-1]),
                    "temperature_end_K": float(state[TEMP]),
                    "pressure_end_Pa": pressure(state),
                    "co2_loading_end_mol_kg": float(state[Q].sum()),
                    "water_loading_end_mol_kg": float(state[QW]),
                    "energy_residual_J": inventory(state) - inventory(before) - boundary,
                    "event_terminated": bool(
                        events
                        and solution.t_events
                        and any(len(event_times) for event_times in solution.t_events)
                    ),
                    "initial_step": first_step_record,
                    "envelope": accepted_phase_diagnostics(solution, phase),
                }
            )
        phase_records[0]["adsorption_pressure_endpoint_ok"] = bool(
            abs(phase_records[0]["pressure_end_Pa"] - ambient_p)
            <= ADSORPTION_PRESSURE_TOLERANCE_FRACTION * ambient_p
        )
        endpoint_ok = phase_endpoints_ok(
            phase_records,
            ambient_temperature_K=ambient_K,
            ambient_pressure_Pa=ambient_p,
            regeneration_temperature_K=regen_K,
            regeneration_pressure_Pa=regen_p,
            cooling_endpoint_K=cooling_endpoint_K,
        )
        phase_records[-1]["endpoint_ok"] = endpoint_ok
        raw_product = state[PRODUCT]
        raw_vent = state[VENT]
        if conditioner_enabled:
            product = state[CONDITIONED_PRODUCT]
            physical_vent = state[DIRECT_VENT] + state[CONDITIONED_VENT]
            liquid_drain = state[LIQUID_DRAIN]
        else:
            product = raw_product
            physical_vent = raw_vent
            liquid_drain = np.zeros(4)
        duration = sum(row["duration_s"] for row in phase_records)
        mass_ledger = species_mass_ledger(
            initial_sites=initial[Q],
            final_sites=state[Q],
            initial_water=initial[QW],
            final_water=state[QW],
            initial_n2=initial[QN],
            final_n2=state[QN],
            initial_o2=initial[QO],
            final_o2=state[QO],
            initial_gas=initial[GAS],
            final_gas=state[GAS],
            dry_mass_kg=mass,
            site_species=adapter.site_species,
            inlet=state[IN],
            vent=physical_vent,
            product=product,
            liquid_drain=liquid_drain,
        )
        species_residual = mass_ledger["residual_mol"]
        species_throughput = mass_ledger["throughput_mol"]
        relative_mass_error = float(np.max(mass_ledger["relative_error"]))
        inventory_change = inventory(state) - inventory(initial)
        energy_boundary = state[HEAT] - state[COOL] - state[LOSS] + state[INLET_H] - state[OUTLET_H]
        energy_residual = inventory_change - energy_boundary
        energy_transfer_scale = max(
            state[HEAT]
            + state[COOL]
            + state[LOSS]
            + abs(state[INLET_H] - state[OUTLET_H]),
            abs(inventory_change),
            1.0,
        )
        relative_energy_error = abs(energy_residual) / energy_transfer_scale
        physical_change = np.r_[state[Q] - initial[Q], state[QW] - initial[QW], state[QN] - initial[QN], state[QO] - initial[QO], state[TEMP] - initial[TEMP], state[GAS] - initial[GAS]]
        physical_scale = np.r_[np.full(ns, 0.1), 0.1, 0.01, 0.01, 1.0, np.full(4, 0.01)]
        max_physical_change = float(np.max(np.abs(physical_change) / physical_scale))
        envelope_audit = aggregate_envelope(phase_records)
        product_total_mol = float(product.sum())
        product_dry_total_mol = float(product[0] + product[2] + product[3])
        product_co2_fraction_wet = (
            float(product[0] / product_total_mol)
            if product_total_mol > 0.0
            else None
        )
        product_co2_fraction_dry = (
            float(product[0] / product_dry_total_mol)
            if product_dry_total_mol > 0.0
            else None
        )
        water_sorbed_change_mol = mass * (
            float(state[QW]) - float(initial[QW])
        )
        water_gas_change_mol = float(state[GAS][1] - initial[GAS][1])
        water_inlet_mol = float(state[IN][1])
        water_vent_mol = float(physical_vent[1])
        water_product_mol = float(product[1])
        water_liquid_drain_mol = float(liquid_drain[1])
        water_residual_mol = (
            water_sorbed_change_mol
            + water_gas_change_mol
            - water_inlet_mol
            + water_vent_mol
            + water_product_mol
            + water_liquid_drain_mol
        )

        def water_kg_day(moles):
            return float(moles * 0.01801528 * 86400.0 / max(duration, 1.0))

        water_handling = {
            "inlet_mol": water_inlet_mol,
            "vent_mol": water_vent_mol,
            "product_mol": water_product_mol,
            "liquid_drain_mol": water_liquid_drain_mol,
            "sorbed_inventory_change_mol": water_sorbed_change_mol,
            "gas_inventory_change_mol": water_gas_change_mol,
            "residual_mol": water_residual_mol,
            "inlet_kg_day": water_kg_day(water_inlet_mol),
            "vent_kg_day": water_kg_day(water_vent_mol),
            "product_kg_day": water_kg_day(water_product_mol),
            "sorbed_inventory_change_kg_day": water_kg_day(
                water_sorbed_change_mol
            ),
        }
        conditioner_cooling_removed_J = (
            float(state[CONDITIONER_COOL]) if conditioner_enabled else 0.0
        )
        conditioner_energy_residual_J = (
            float(
                state[CONDITIONER_COOL]
                - (state[OUTLET_H] - state[EXTERNAL_OUTLET_H])
            )
            if conditioner_enabled
            else 0.0
        )
        cycle = {
            "cycle": cycle_number,
            "co2_product_mol": float(product[0]),
            "co2_kg_day": float(product[0] * 0.0440095 * 86400 / max(duration, 1.0)),
            "co2_swing_mol_kg": float(phase_records[0]["co2_loading_end_mol_kg"] - min(row["co2_loading_end_mol_kg"] for row in phase_records[1:4])),
            "water_swing_mol_kg": float(phase_records[0]["water_loading_end_mol_kg"] - min(row["water_loading_end_mol_kg"] for row in phase_records[1:4])),
            "delivered_heat_J": float(state[HEAT]),
            "pump_electric_J": float(state[PUMP]),
            "fan_electric_J": float(state[FAN]),
            "cooling_removed_J": float(state[COOL]),
            "conditioner_cooling_removed_J": conditioner_cooling_removed_J,
            "total_cooling_removed_J": (
                float(state[COOL]) + conditioner_cooling_removed_J
            ),
            "cooling_electric_J": (
                float(state[COOL]) + conditioner_cooling_removed_J
            )
            / cooling_cop,
            "conditioner_energy_boundary_residual_J": (
                conditioner_energy_residual_J
            ),
            "product_mol": {
                "co2": float(product[0]),
                "water": float(product[1]),
                "n2": float(product[2]),
                "o2": float(product[3]),
            },
            "product_co2_fraction_wet_basis": product_co2_fraction_wet,
            "product_co2_fraction_dry_basis": product_co2_fraction_dry,
            "product_fraction_dry": product_co2_fraction_dry,
            "product_water_kg_day": water_handling["product_kg_day"],
            "water_handling": water_handling,
            "raw_preconditioning_product_mol": {
                "co2": float(raw_product[0]),
                "water": float(raw_product[1]),
                "n2": float(raw_product[2]),
                "o2": float(raw_product[3]),
            },
            "raw_preconditioning_vent_mol": {
                "co2": float(raw_vent[0]),
                "water": float(raw_vent[1]),
                "n2": float(raw_vent[2]),
                "o2": float(raw_vent[3]),
            },
            "conditioned_vent_mol": (
                {
                    "co2": float(state[CONDITIONED_VENT][0]),
                    "water": float(state[CONDITIONED_VENT][1]),
                    "n2": float(state[CONDITIONED_VENT][2]),
                    "o2": float(state[CONDITIONED_VENT][3]),
                }
                if conditioner_enabled
                else None
            ),
            "direct_adsorption_vent_mol": (
                {
                    "co2": float(state[DIRECT_VENT][0]),
                    "water": float(state[DIRECT_VENT][1]),
                    "n2": float(state[DIRECT_VENT][2]),
                    "o2": float(state[DIRECT_VENT][3]),
                }
                if conditioner_enabled
                else None
            ),
            "liquid_drain_mol": (
                {
                    "co2": float(liquid_drain[0]),
                    "water": float(liquid_drain[1]),
                    "n2": float(liquid_drain[2]),
                    "o2": float(liquid_drain[3]),
                }
                if conditioner_enabled
                else None
            ),
            "duration_s": duration,
            "mass_residual_mol": species_residual.tolist(),
            "mass_closure_relative_error": relative_mass_error,
            "energy_residual_J": float(energy_residual),
            "energy_transfer_scale_J": float(energy_transfer_scale),
            "energy_closure_relative_error": float(relative_energy_error),
            "max_physical_state_relative_change": max_physical_change,
            "closure_pass": bool(relative_mass_error <= 1e-5 and relative_energy_error <= 1e-6),
            "envelope_audit": envelope_audit,
            "phase_records": phase_records,
        }
        cycles.append(cycle)
        if collection_trace:
            try:
                phase_endpoint_checks = _collection_phase_endpoint_checks(
                    phase_records,
                    ambient_temperature_K=ambient_K,
                    ambient_pressure_Pa=ambient_p,
                    regeneration_temperature_K=regen_K,
                    regeneration_pressure_Pa=regen_p,
                    cooling_endpoint_K=cooling_endpoint_K,
                )
                pump_audit = {
                    "sampling_scope": (
                        "accepted solve_ivp states in final accepted CSS cycle"
                    ),
                    "phases": phase_pump_audits,
                    "positive_flow_pass": all(
                        row["positive_flow_pass"]
                        for row in phase_pump_audits
                    ),
                    "upper_domain_exceeded": any(
                        row["upper_domain_exceeded"]
                        for row in phase_pump_audits
                    ),
                    "below_or_at_ultimate_observed": any(
                        row["below_or_at_ultimate_observed"]
                        for row in phase_pump_audits
                    ),
                    "maximum_media_temperature_overrun_K": max(
                        row["maximum_media_temperature_overrun_K"]
                        for row in phase_pump_audits
                    ),
                    "media_temperature_pass": all(
                        row["media_temperature_pass"]
                        for row in phase_pump_audits
                    ),
                }
                final_cycle_trace = _build_collection_trace(
                    cycle_number=cycle_number,
                    dense_solution=dense_solutions["desorption"],
                    product_indices=CONDITIONED_PRODUCT,
                    expected_state_size=SIZE,
                    phase_clocks=phase_clocks,
                    endpoint_product=state[CONDITIONED_PRODUCT],
                    cycle_start_state=cycle_start_state,
                    desorption_start_state=phase_start_states["desorption"],
                    cooling_end_state=phase_end_states["cooling"],
                    cycle_end_state=state,
                    phase_endpoint_checks=phase_endpoint_checks,
                    pump_audit=pump_audit,
                )
            except ValueError as error:
                raise CycleFailure(
                    str(error),
                    phase="desorption",
                    cycle=cycle_number,
                    records=cycles,
                ) from error
        if progress_callback is not None:
            progress_callback(
                {
                    "cycle": cycle_number,
                    "fresh_cycle": fresh_cycle_number,
                    "state": state.tolist(),
                    "phase_temperatures_K": [
                        row["temperature_end_K"] for row in phase_records
                    ],
                    "co2_product_mol": cycle["co2_product_mol"],
                    "mass_closure_relative_error": cycle["mass_closure_relative_error"],
                    "energy_residual_J": cycle["energy_residual_J"],
                    "energy_transfer_scale_J": cycle["energy_transfer_scale_J"],
                    "energy_closure_relative_error": cycle["energy_closure_relative_error"],
                    "max_physical_state_relative_change": cycle["max_physical_state_relative_change"],
                    "closure_pass": cycle["closure_pass"],
                    "envelope_audit": cycle["envelope_audit"],
                    "product_mol": cycle["product_mol"],
                    "product_co2_fraction_wet_basis": cycle[
                        "product_co2_fraction_wet_basis"
                    ],
                    "product_co2_fraction_dry_basis": cycle[
                        "product_co2_fraction_dry_basis"
                    ],
                    "product_water_kg_day": cycle["product_water_kg_day"],
                    "water_handling": cycle["water_handling"],
                    "conditioner_cooling_removed_J": cycle[
                        "conditioner_cooling_removed_J"
                    ],
                    "total_cooling_removed_J": cycle[
                        "total_cooling_removed_J"
                    ],
                    "conditioned_vent_mol": cycle[
                        "conditioned_vent_mol"
                    ],
                    "direct_adsorption_vent_mol": cycle[
                        "direct_adsorption_vent_mol"
                    ],
                    "liquid_drain_mol": cycle["liquid_drain_mol"],
                    "raw_preconditioning_product_mol": cycle[
                        "raw_preconditioning_product_mol"
                    ],
                    "raw_preconditioning_vent_mol": cycle[
                        "raw_preconditioning_vent_mol"
                    ],
                }
            )
        all_closure = all_closure and cycle["closure_pass"]
        if not cycle["closure_pass"]:
            raise CycleFailure("integrated mass/energy closure failed", phase="cycle", cycle=cycle_number, records=cycles)
        if len(cycles) >= 2:
            metric_keys = ("co2_product_mol", "delivered_heat_J", "pump_electric_J", "fan_electric_J", "cooling_removed_J", "duration_s")
            current = np.array([cycle[key] for key in metric_keys])
            previous = np.array([cycles[-2][key] for key in metric_keys])
            metric_change = float(np.max(np.abs(current - previous) / np.maximum(np.abs(previous), 1e-9)))
            converged = bool(metric_change < 0.002 and max_physical_change < 0.002)
        if converged and fresh_cycle_number >= minimum_cycles:
            break
    last = cycles[-1]
    duration = last["duration_s"]
    total_electric = last["delivered_heat_J"] + last["pump_electric_J"] + last["fan_electric_J"] + last["cooling_electric_J"]
    status = (
        "conditional_pass"
        if converged and all_closure and last["phase_records"][-1]["endpoint_ok"]
        else "numerically_failed"
    )
    summary = {
        "material_id": adapter.material_id,
        "relative_humidity": relative_humidity,
        "water_vapour_partial_pressure_Pa": _p_water,
        "water_vapour_pressure_source": (
            "explicit_inlet_override"
            if water_vapour_partial_pressure_Pa is not None
            else "frozen_core_rh_helper"
        ),
        "status": status,
        "cyclic_steady_state": converged,
        "cycles_executed": len(cycles),
        "fresh_cycles_executed": len(cycles),
        "absolute_cycles_executed": int(initial_cycle_index) + len(cycles),
        "co2_kg_day": last["co2_product_mol"] * 0.0440095 * 86400 / duration,
        "co2_kg_day_per_kg_dry": last["co2_product_mol"] * 0.0440095 * 86400 / duration / mass,
        "co2_swing_mol_kg": last["co2_swing_mol_kg"],
        "water_swing_mol_kg": last["water_swing_mol_kg"],
        "delivered_heat_J_day": last["delivered_heat_J"] * 86400 / duration,
        "heater_electric_J_day": last["delivered_heat_J"] * 86400 / duration,
        "pump_electric_J_day": last["pump_electric_J"] * 86400 / duration,
        "fan_electric_J_day": last["fan_electric_J"] * 86400 / duration,
        "cooling_electric_J_day": last["cooling_electric_J"] * 86400 / duration,
        "conditioner_cooling_removed_J_day": last[
            "conditioner_cooling_removed_J"
        ]
        * 86400
        / duration,
        "total_cooling_removed_J_day": last["total_cooling_removed_J"]
        * 86400
        / duration,
        "conditioning_boundary": (
            "pump-inlet cooler/knockout modeled to pump suction; "
            "compression and downstream drying excluded"
            if conditioner_enabled
            else "UNKNOWN; feed dryer excluded"
        ),
        "energy_per_kg_co2_J_kg": total_electric / (last["co2_product_mol"] * 0.0440095) if last["co2_product_mol"] > 0 else None,
        "product_mol": last["product_mol"],
        "product_co2_fraction_wet_basis": last[
            "product_co2_fraction_wet_basis"
        ],
        "product_co2_fraction_dry_basis": last[
            "product_co2_fraction_dry_basis"
        ],
        "product_dry_fraction": last["product_co2_fraction_dry_basis"],
        "product_water_kg_day": last["product_water_kg_day"],
        "water_handling": last["water_handling"],
        "conditioned_pump_inlet_mol": (
            last["product_mol"] if conditioner_enabled else None
        ),
        "conditioned_vent_mol": last["conditioned_vent_mol"],
        "direct_adsorption_vent_mol": last[
            "direct_adsorption_vent_mol"
        ],
        "liquid_drain_mol": last["liquid_drain_mol"],
        "raw_preconditioning_product_mol": last[
            "raw_preconditioning_product_mol"
        ],
        "raw_preconditioning_vent_mol": last[
            "raw_preconditioning_vent_mol"
        ],
        "conditioner_energy_boundary_residual_J": last[
            "conditioner_energy_boundary_residual_J"
        ],
        "conditioner_cooling_removed_J": last[
            "conditioner_cooling_removed_J"
        ],
        "total_cooling_removed_J": last["total_cooling_removed_J"],
        "pump_conditioner_identity": pump_conditioner_identity(config),
        "state_size": SIZE,
        "physical_columns": ns + 8,
        "ledger_columns": SIZE - (ns + 8),
        "dry_purity_interpretation": (
            (
                "CO2 fraction of conditioned pump-inlet gas after excluding "
                "retained water vapor; this is not a delivered dry product"
            )
            if conditioner_enabled
            else (
                "CO2 fraction after excluding water from the raw product "
                "composition; no condenser or dried product stream is modeled"
            )
        ),
        "recovery_of_inlet": None,
        "cycle_duration_s": duration,
        "adsorption_gas_residence_s": volume
        / float(config["airflow"]["volume_flow_m3_s"]),
        "adsorption_end_pressure_tolerance_fraction": (
            ADSORPTION_PRESSURE_TOLERANCE_FRACTION
        ),
        "adsorption_end_pressure_ok": last["phase_records"][0][
            "adsorption_pressure_endpoint_ok"
        ],
        "adsorption_control": adsorption_control,
        "control_identity": control_identity(config),
        "pressure_response_tau_s": pressure_response_tau_s,
        "initial_step_identity": initial_step_identity(config),
        "initial_step_records": initial_step_records,
        "adsorption_boundary": (
            "pressure-regulated finite-gas adsorption with current-inlet "
            "pressure target and coupled outlet flow"
            if adsorption_control == PRESSURE_CONTROL
            else "finite mixed-gas adsorption balance with feed inlet and actual bed composition/pressure"
        ),
        "adsorption_total_molar_holdup_constant": (
            adsorption_control == FIXED_CONTROL
        ),
        "adsorption_component_molar_holdups_constant": False,
        "adsorption_inlet_molar_flow_constant_at_fixed_weather": True,
        "adsorption_outlet_molar_flow_constant": False,
        "molar_holdup_constant_in_other_phases": False,
        "adsorption_total_molar_identity": (
            "Fout=F0+(n/T)*Tdot; dn=Fin*y_feed-Fout*y_bed-mass*sorption; "
            "Pdot=R/V*(T*sum(dn)+n*Tdot)=Pdot_required"
            if adsorption_control == PRESSURE_CONTROL
            else (
                "Fout=Fin-mass*sum(sorption); dn=Fin*y_feed-Fout*y_bed-"
                "mass*sorption; sum(dn)=0 exactly"
            )
        ),
        "cooling_stop_offset_K": (
            float(cooling_stop_offset_K)
            if cooling_stop_offset_K is not None
            else None
        ),
        "cooling_stop_target_K": cooling_endpoint_K,
        "cooling_max_s": float(config["cycle"]["cooling_max_s"]),
        "flow_limitations": [
            "well-mixed lumped gas volume",
            "component inventories and outlet molar flow vary during adsorption",
            (
                "total gas molar holdup is controlled through the declared "
                "finite pressure response and conditional ideal outlet valve"
                if adsorption_control == PRESSURE_CONTROL
                else "total gas molar holdup changes during phases other than adsorption"
            ),
            "fixed-weather inlet molar flow is constant; fan pressure drop remains lumped",
        ],
        "envelope_audit": aggregate_envelope(last["phase_records"]),
        "mass_closure_relative_error": last["mass_closure_relative_error"],
        "energy_residual_J": last["energy_residual_J"],
        "energy_transfer_scale_J": last["energy_transfer_scale_J"],
        "energy_closure_relative_error": last["energy_closure_relative_error"],
        "max_physical_state_relative_change": last["max_physical_state_relative_change"],
        "elapsed_s": time.monotonic() - started,
        "adapter": adapter.source_report(),
        "limitations": [
            "well-mixed lumped material screen",
            "dryer energy unknown/excluded",
            "cooler/COP hypothetical",
            (
                "pump-inlet water condensation modeled; bed condensation absent"
                if conditioner_enabled
                else "condensation absent"
            ),
            "no product-purity equivalence selection",
            (
                "conditioned pump-inlet gas is not a delivered dry product"
                if conditioner_enabled
                else "no condenser or dried product stream"
            ),
            "fresh near-CSS envelope audit does not cover cold-start trajectories",
        ]
        + (
            [
                "ideal zero-pressure-drop, zero-holdup pump-inlet knockout",
                "conditioned wet-gas pump curve transfer is unresolved",
                "liquid pressure dependence and CO2 dissolution are neglected",
                "bed supersaturation remains independent of downstream conditioning",
                "hardware qualification remains false",
            ]
            if conditioner_enabled
            else []
        ),
    }
    if collection_trace:
        summary["collection_trace"] = (
            final_cycle_trace if status == "conditional_pass" else None
        )
    return CycleResult(summary, {"cycles": cycles, "final_state": state.tolist()})
