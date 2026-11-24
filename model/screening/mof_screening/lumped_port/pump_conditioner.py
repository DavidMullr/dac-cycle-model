"""Opt-in pump-inlet cooler and water knockout algebra.

This module models only the gas/liquid split and shared cooling budget at
pump suction. Pump compression and downstream drying remain outside the
control volume.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import math

import numpy as np

from ...common import CP_GAS, R

from .water_properties import physical_saturation_pressure_Pa


POLICY = "pump_inlet_cooler_knockout_v1"
REFERENCE_TEMPERATURE_K = 298.15
DEFAULT_TARGET_TEMPERATURE_C = 55.0
LIQUID_WATER_CP_J_MOL_K = 75.351
STANDARD_VAPOR_MINUS_LIQUID_ENTHALPY_J_MOL = 44004.0
PUMP_RATED_INPUT_W = 135.0


@dataclass(frozen=True)
class PumpConditionerConfig:
    target_temperature_K: float
    total_cooling_power_W: float
    cooling_cop: float
    pump_bank_count: int
    pump_rated_input_W: float


def gas_enthalpy_J_mol(temperature_K: float) -> np.ndarray:
    """Existing engine gas reference: Cp*(T-Tref)+R*Tref."""
    temperature = float(temperature_K)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("gas temperature must be finite and positive")
    return np.asarray(
        CP_GAS * (temperature - REFERENCE_TEMPERATURE_K)
        + R * REFERENCE_TEMPERATURE_K,
        dtype=float,
    )


def liquid_water_enthalpy_J_mol(temperature_K: float) -> float:
    """Approximate liquid-water enthalpy anchored to the existing gas reference."""
    temperature = float(temperature_K)
    if not math.isfinite(temperature):
        raise ValueError("liquid temperature must be finite")
    if temperature < 273.15:
        raise ValueError("unsupported subfreezing liquid-water state")
    h_vapour_reference = R * REFERENCE_TEMPERATURE_K
    h_liquid_reference = (
        h_vapour_reference - STANDARD_VAPOR_MINUS_LIQUID_ENTHALPY_J_MOL
    )
    return float(
        h_liquid_reference
        + LIQUID_WATER_CP_J_MOL_K
        * (temperature - REFERENCE_TEMPERATURE_K)
    )


def water_saturation_properties_at_Kelvin(
    temperature_K: float,
) -> tuple[float, str]:
    """Evaluate the accepted physical diagnostic helper with explicit Kelvin."""
    temperature = float(temperature_K)
    if not math.isfinite(temperature) or temperature < 273.15:
        raise ValueError("water saturation requires a supported liquid state")
    pressure, law = physical_saturation_pressure_Pa(temperature)
    pressure = float(pressure)
    if not math.isfinite(pressure) or pressure <= 0.0:
        raise ValueError("water saturation pressure must be finite and positive")
    return pressure, law


def water_saturation_pressure_at_Kelvin(temperature_K: float) -> float:
    """Compatibility accessor for the physical saturation pressure."""
    return water_saturation_properties_at_Kelvin(temperature_K)[0]


def enable_pump_conditioner(
    config: dict,
    *,
    target_temperature_C: float = DEFAULT_TARGET_TEMPERATURE_C,
) -> dict:
    """Return a deep-copied config with the common conditioner option enabled."""
    if not 0.0 <= float(target_temperature_C) <= 100.0:
        raise ValueError(
            "conditioner temperature must be within 0–100 C"
        )
    result = deepcopy(config)
    cooling = result["utilities"]["wall_cooling"]
    result["pump_conditioner"] = {
        "schema": "pump-conditioner-config/1",
        "enabled": True,
        "policy": POLICY,
        "target_temperature_C": float(target_temperature_C),
        "total_cooling_power_W": float(cooling["max_cooling_capacity_W"]),
        "cooling_cop": float(cooling["cop"]),
        "pump_bank_count": int(result.get("pump", {}).get("bank_count", 1)),
        "pump_rated_input_W": PUMP_RATED_INPUT_W,
        "pressure_drop_Pa": 0.0,
        "line_holdup_mol": 0.0,
        "liquid_pressure_dependence": False,
        "dissolved_co2_modeled": False,
    }
    return result


def pump_conditioner_config(config: dict) -> PumpConditionerConfig | None:
    section = config.get("pump_conditioner")
    if not section or not section.get("enabled", False):
        return None
    if section.get("schema") != "pump-conditioner-config/1":
        raise ValueError("unsupported pump-conditioner schema")
    if section.get("policy") != POLICY:
        raise ValueError("unsupported pump-conditioner policy")
    cooling = config["utilities"]["wall_cooling"]
    total = float(cooling["max_cooling_capacity_W"])
    cop = float(cooling["cop"])
    bank = int(config.get("pump", {}).get("bank_count", 1))
    values = {
        "target_temperature_K": (
            float(section["target_temperature_C"]) + 273.15
        ),
        "total_cooling_power_W": total,
        "cooling_cop": cop,
        "pump_bank_count": bank,
        "pump_rated_input_W": float(section["pump_rated_input_W"]),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("pump-conditioner configuration must be finite")
    if values["target_temperature_K"] > 373.15:
        raise ValueError("pump-conditioner target exceeds 100 C")
    if values["target_temperature_K"] < 273.15:
        raise ValueError("pump-conditioner target must remain above freezing")
    if total <= 0.0 or cop <= 0.0:
        raise ValueError("pump-conditioner cooling power and COP must be positive")
    if float(section["total_cooling_power_W"]) != total:
        raise ValueError("conditioner must share the configured total cooler")
    if float(section["cooling_cop"]) != cop:
        raise ValueError("conditioner must share the configured cooler COP")
    if bank != 1 or int(section["pump_bank_count"]) != 1:
        raise ValueError("pump conditioner requires exactly one pump")
    if values["pump_rated_input_W"] != PUMP_RATED_INPUT_W:
        raise ValueError("pump conditioner requires the 135 W rated pump input")
    if float(section.get("pressure_drop_Pa")) != 0.0:
        raise ValueError("nonzero conditioner pressure drop is unsupported")
    if float(section.get("line_holdup_mol")) != 0.0:
        raise ValueError("nonzero conditioner line holdup is unsupported")
    if section.get("liquid_pressure_dependence") is not False:
        raise ValueError("liquid pressure dependence is unsupported")
    if section.get("dissolved_co2_modeled") is not False:
        raise ValueError("CO2 dissolution is unsupported")
    return PumpConditionerConfig(**values)


def pump_conditioner_identity(config: dict) -> dict | None:
    normalized = pump_conditioner_config(config)
    if normalized is None:
        return None
    return {
        "schema": "pump-conditioner-identity/1",
        "policy": POLICY,
        "target_temperature_C": (
            normalized.target_temperature_K - 273.15
        ),
        "target_rule": "Tout=min(Tbed,target); never heat",
        "total_cooling_power_W": normalized.total_cooling_power_W,
        "cooling_cop": normalized.cooling_cop,
        "cooling_allocation": "conditioner_first_then_bed",
        "pump_bank_count": normalized.pump_bank_count,
        "pump_rated_input_W": normalized.pump_rated_input_W,
        "pump_capacity_temperature": "conditioned_inlet_Tout",
        "gas_enthalpy_reference_J_mol": (
            "Cp_gas*(T-298.15)+R*298.15"
        ),
        "liquid_water_cp_J_mol_K": LIQUID_WATER_CP_J_MOL_K,
        "liquid_enthalpy_anchor": (
            "h_liq(298.15)=h_gas_existing(298.15)-44004 J/mol"
        ),
        "pressure_drop_Pa": 0.0,
        "line_holdup_mol": 0.0,
        "dissolved_co2_modeled": False,
        "hardware_qualification": False,
    }


def _normalized_mole_fractions(values, *, label: str) -> np.ndarray:
    fractions = np.asarray(values, dtype=float)
    if fractions.shape != (4,):
        raise ValueError(f"{label} must contain four species")
    if not np.all(np.isfinite(fractions)) or np.any(fractions < 0.0):
        raise ValueError(f"{label} must be finite and nonnegative")
    total = float(fractions.sum())
    if total <= 0.0 or abs(total - 1.0) > 1e-10:
        raise ValueError(f"{label} must sum to one")
    return fractions


def split_pump_inlet_streams(
    *,
    bed_mole_fractions,
    pressure_Pa: float,
    bed_temperature_K: float,
    target_temperature_K: float,
) -> dict:
    """Split one bed-outlet mole into conditioned gas and liquid drain."""
    y = _normalized_mole_fractions(
        bed_mole_fractions, label="bed mole fractions"
    )
    pressure = float(pressure_Pa)
    bed_temperature = float(bed_temperature_K)
    target_temperature = float(target_temperature_K)
    if not math.isfinite(pressure) or pressure <= 0.0:
        raise ValueError("conditioner pressure must be finite and positive")
    if not math.isfinite(bed_temperature) or bed_temperature <= 0.0:
        raise ValueError("bed temperature must be finite and positive")
    if not math.isfinite(target_temperature) or target_temperature < 273.15:
        raise ValueError("conditioner target must be above freezing")
    outlet_temperature = min(bed_temperature, target_temperature)
    p_sat, saturation_law = water_saturation_properties_at_Kelvin(
        outlet_temperature
    )
    z = min(float(y[1]), p_sat / pressure)
    dry_fraction = 1.0 - float(y[1])
    if dry_fraction <= 0.0:
        if z >= 1.0 - 1e-15:
            gamma, liquid_fraction = 1.0, 0.0
        else:
            gamma, liquid_fraction = 0.0, 1.0
    elif float(y[1]) <= z:
        gamma, liquid_fraction = 1.0, 0.0
    else:
        gamma = dry_fraction / (1.0 - z)
        liquid_fraction = float(y[1]) - gamma * z
    conditioned = np.array([y[0], gamma * z, y[2], y[3]], dtype=float)
    liquid = np.array([0.0, liquid_fraction, 0.0, 0.0], dtype=float)
    total_balance = gamma + liquid_fraction
    species_balance = conditioned + liquid - y
    if abs(total_balance - 1.0) > 1e-12 or np.max(np.abs(species_balance)) > 1e-12:
        raise FloatingPointError("conditioner stream split violates mass balance")
    if gamma < -1e-15 or liquid_fraction < -1e-15:
        raise FloatingPointError("conditioner stream fractions must be nonnegative")
    if gamma > 1.0 + 1e-15 or liquid_fraction > 1.0 + 1e-15:
        raise FloatingPointError("conditioner stream fractions exceed one")
    return {
        "outlet_temperature_K": outlet_temperature,
        "saturation_pressure_Pa": p_sat,
        "saturation_law": saturation_law,
        "equilibrium_water_fraction": z,
        "retained_gas_fraction": gamma,
        "liquid_fraction": liquid_fraction,
        "conditioned_gas_fractions": conditioned,
        "liquid_drain_fractions": liquid,
        "total_balance": total_balance,
        "species_balance": species_balance,
        "condensation_occurred": bool(liquid_fraction > 0.0),
    }


def cooling_per_bed_mol_J_mol(
    *,
    bed_mole_fractions,
    bed_temperature_K: float,
    split: dict,
) -> float:
    y = _normalized_mole_fractions(
        bed_mole_fractions, label="bed mole fractions"
    )
    bed_h = float(y @ gas_enthalpy_J_mol(float(bed_temperature_K)))
    outlet_temperature = float(split["outlet_temperature_K"])
    conditioned = np.asarray(split["conditioned_gas_fractions"], dtype=float)
    liquid_fraction = float(split["liquid_fraction"])
    conditioned_h = float(
        conditioned @ gas_enthalpy_J_mol(outlet_temperature)
    )
    liquid_h = liquid_fraction * liquid_water_enthalpy_J_mol(
        outlet_temperature
    )
    q = bed_h - conditioned_h - liquid_h
    if not math.isfinite(q) or q < -1e-10:
        raise FloatingPointError("conditioner cooling per mole must be nonnegative")
    return float(max(q, 0.0)) if abs(q) <= 1e-10 else float(q)


def evaluate_pump_conditioner(
    *,
    bed_mole_fractions,
    pressure_Pa: float,
    bed_temperature_K: float,
    pump_capacity_mol_s: float,
    alpha: float,
    target_temperature_K: float,
    total_cooling_power_W: float,
    cooling_cop: float,
    pump_electric_power_W: float,
) -> dict:
    """Evaluate the conditioner-first, shared-cooling pump-inlet path."""
    y = _normalized_mole_fractions(
        bed_mole_fractions, label="bed mole fractions"
    )
    alpha_value = float(alpha)
    capacity = float(pump_capacity_mol_s)
    total_cooling = float(total_cooling_power_W)
    cop = float(cooling_cop)
    pump_power = float(pump_electric_power_W)
    for value, label in (
        (alpha_value, "alpha"),
        (capacity, "pump capacity"),
        (total_cooling, "total cooling power"),
        (cop, "cooling COP"),
        (pump_power, "pump electric power"),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{label} must be finite and nonnegative")
    if alpha_value > 1.0:
        raise ValueError("alpha may not exceed one")
    if total_cooling <= 0.0 or cop <= 0.0:
        raise ValueError("total cooling power and COP must be positive")
    split = split_pump_inlet_streams(
        bed_mole_fractions=y,
        pressure_Pa=pressure_Pa,
        bed_temperature_K=bed_temperature_K,
        target_temperature_K=target_temperature_K,
    )
    q = cooling_per_bed_mol_J_mol(
        bed_mole_fractions=y,
        bed_temperature_K=bed_temperature_K,
        split=split,
    )
    gamma = float(split["retained_gas_fraction"])
    liquid_fraction = float(split["liquid_fraction"])
    gas_limit = None if gamma == 0.0 else capacity / gamma
    energy_limit = None if q <= 0.0 else total_cooling / q
    active_limits = [
        value
        for value in (gas_limit, energy_limit)
        if value is not None
    ]
    if capacity == 0.0 or alpha_value == 0.0:
        maximum_bed_flow = None if not active_limits else min(active_limits)
        bed_flow = pump_flow = liquid_flow = duty = 0.0
        withdrawal_reason = (
            "zero_pump_capacity" if capacity == 0.0 else "alpha_zero"
        )
    else:
        if not active_limits:
            raise FloatingPointError(
                "conditioner withdrawal has no finite gas or energy constraint"
            )
        maximum_bed_flow = min(active_limits)
        bed_flow = alpha_value * maximum_bed_flow
        pump_flow = gamma * bed_flow
        liquid_flow = liquid_fraction * bed_flow
        duty = q * bed_flow
        withdrawal_reason = "active"
    gas_constraint_binding = bool(
        bed_flow > 0.0
        and gas_limit is not None
        and maximum_bed_flow is not None
        and abs(maximum_bed_flow - gas_limit)
        <= 1e-12 * max(abs(gas_limit), 1.0)
    )
    energy_constraint_binding = bool(
        bed_flow > 0.0
        and energy_limit is not None
        and maximum_bed_flow is not None
        and abs(maximum_bed_flow - energy_limit)
        <= 1e-12 * max(abs(energy_limit), 1.0)
    )
    available_bed_cooling = total_cooling - duty
    bed_outlet = bed_flow * y
    conditioned_gas = bed_flow * np.asarray(
        split["conditioned_gas_fractions"], dtype=float
    )
    liquid_drain = bed_flow * np.asarray(
        split["liquid_drain_fractions"], dtype=float
    )
    bed_outlet_h = float(
        bed_outlet @ gas_enthalpy_J_mol(float(bed_temperature_K))
    )
    conditioned_h = float(
        conditioned_gas @ gas_enthalpy_J_mol(split["outlet_temperature_K"])
    )
    liquid_h = float(
        liquid_flow
        * liquid_water_enthalpy_J_mol(split["outlet_temperature_K"])
    )
    boundary_duty = bed_outlet_h - conditioned_h - liquid_h
    if abs(boundary_duty - duty) > 1e-9 * max(
        abs(boundary_duty), abs(duty), 1.0
    ):
        raise RuntimeError("conditioner energy boundary mismatch")
    conditioned_total = float(conditioned_gas.sum())
    liquid_total = float(liquid_drain.sum())
    if abs(conditioned_total - pump_flow) > 1e-12 * max(pump_flow, 1.0):
        raise FloatingPointError("conditioned gas flow mismatch")
    if abs(liquid_total - liquid_flow) > 1e-12 * max(liquid_flow, 1.0):
        raise FloatingPointError("liquid drain flow mismatch")
    if abs(float(bed_outlet.sum()) - bed_flow) > 1e-12 * max(bed_flow, 1.0):
        raise FloatingPointError("raw bed outlet flow mismatch")
    if available_bed_cooling < -1e-10:
        raise FloatingPointError("conditioner exceeded shared cooling budget")
    values = {
        "bed_flow_mol_s": bed_flow,
        "pump_flow_mol_s": pump_flow,
        "liquid_flow_mol_s": liquid_flow,
        "maximum_bed_flow_mol_s": maximum_bed_flow,
        "gas_capacity_limit_mol_s": gas_limit,
        "energy_capacity_limit_mol_s": energy_limit,
        "cooling_per_bed_mol_J_mol": q,
        "conditioner_cooling_W": duty,
        "available_bed_cooling_W": max(available_bed_cooling, 0.0),
        "conditioner_electric_W": duty / cop,
        "pump_electric_W": pump_power,
        "bed_outlet_h_W": bed_outlet_h,
        "conditioned_gas_h_W": conditioned_h,
        "liquid_drain_h_W": liquid_h,
        "conditioner_boundary_residual_W": boundary_duty - duty,
        "bed_outlet_mol_s": bed_outlet.tolist(),
        "conditioned_gas_mol_s": conditioned_gas.tolist(),
        "liquid_drain_mol_s": liquid_drain.tolist(),
        "conditioned_total_flow_mol_s": conditioned_total,
        "liquid_total_flow_mol_s": liquid_total,
        "retained_gas_fraction": gamma,
        "liquid_fraction": liquid_fraction,
        "condensation_occurred": split["condensation_occurred"],
        "outlet_temperature_K": split["outlet_temperature_K"],
        "saturation_pressure_Pa": split["saturation_pressure_Pa"],
        "saturation_law": split["saturation_law"],
        "equilibrium_water_fraction": split["equilibrium_water_fraction"],
        "alpha": alpha_value,
        "gas_capacity_constraint_present": gas_limit is not None,
        "energy_capacity_constraint_present": energy_limit is not None,
        "gas_capacity_constraint_binding": gas_constraint_binding,
        "energy_capacity_constraint_binding": energy_constraint_binding,
        "withdrawal_reason": withdrawal_reason,
    }
    for value in values.values():
        if isinstance(value, float) and not math.isfinite(value):
            raise FloatingPointError("conditioner output must be finite")
    return values


def strict_json_diagnostics(diagnostics: dict) -> dict:
    """Validate and return a strict-JSON-safe conditioner diagnostic."""
    json.dumps(diagnostics, allow_nan=False)
    return deepcopy(diagnostics)
