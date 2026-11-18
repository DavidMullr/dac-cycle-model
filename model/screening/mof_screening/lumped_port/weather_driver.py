"""Restartable chronological weather driver with explicit forcing provenance."""
from __future__ import annotations

import csv
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
from scipy.integrate import solve_ivp

from .engine import PHASES, run_cycle, species_mass_ledger
from .bounded_jacobian import (
    JacobianAudit,
    bounded_physical_jacobian,
)
from .pressure_control import (
    FIXED_CONTROL,
    apply_control_variant,
    control_identity,
)
from .pump_conditioner import (
    enable_pump_conditioner,
    pump_conditioner_identity,
)
from .solver_start import (
    INITIAL_STEP_POLICY,
    initial_step_identity,
    positive_gas_first_step,
)
from .water_properties import (
    buck_saturation_pressure_Pa,
    physical_saturation_pressure_Pa,
)

from ...common import R, water_saturation_pressure_Pa
from ..contract import CONFIG_ROOT, WEATHER_PATH
from ..energy_balance import gross_energy_scale


ROOT = Path(__file__).resolve().parents[4]
WEATHER_CSV = WEATHER_PATH
COMMON_SCREEN = CONFIG_ROOT / "spatial_profile.json"
CHECKPOINT_SCHEMA = "matched-material-weather-checkpoint/3"
SOURCE_MIGRATION_SCHEMA = "numerical-source-migration/1"
SUPPORTED_CHECKPOINT_SCHEMAS = {
    "matched-material-weather-checkpoint/2",
    CHECKPOINT_SCHEMA,
}


def canonical_sha(payload) -> str:
    data = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_write_json(path: Path, payload) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(
        payload, indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8")
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return hashlib.sha256(data).hexdigest()


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
        timezone.utc
    )


@dataclass(frozen=True)
class WeatherHour:
    interval_start_utc: str
    interval_end_utc: str
    source_timestamp: str
    temperature_C: float
    relative_humidity: float
    station_pressure_Pa: float
    prepared_buck_water_vapour_partial_Pa: float
    prepared_buck_water_mole_fraction: float
    source_row_sha256: str

    def as_dict(self):
        return {
            "interval_start_utc": self.interval_start_utc,
            "interval_end_utc": self.interval_end_utc,
            "source_timestamp": self.source_timestamp,
            "temperature_C": self.temperature_C,
            "relative_humidity": self.relative_humidity,
            "station_pressure_Pa": self.station_pressure_Pa,
            "prepared_buck_water_vapour_partial_Pa": (
                self.prepared_buck_water_vapour_partial_Pa
            ),
            "prepared_buck_water_mole_fraction": (
                self.prepared_buck_water_mole_fraction
            ),
            "source_row_sha256": self.source_row_sha256,
            "prepared_law": "Buck 1996",
            "simulation_feed_law": "Buck 1996 explicit override",
            "fan_property_helper_law": (
                "screening.common feed_composition / NIST-Stull Antoine"
            ),
        }


def load_weather_window(start_utc: str, *, hours: int = 48) -> list[WeatherHour]:
    start = _parse_utc(start_utc)
    first = start + timedelta(hours=1)
    last = start + timedelta(hours=hours)
    selected = []
    with WEATHER_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            timestamp = _parse_utc(row["utc_timestamp"])
            if timestamp < first or timestamp > last:
                continue
            if row["complete"].lower() != "true":
                raise ValueError(f"incomplete weather row at {row['utc_timestamp']}")
            p_water = float(row["water_vapour_partial_Pa"])
            water_fraction = float(row["water_mole_fraction"])
            selected.append(
                WeatherHour(
                    interval_start_utc=(
                        timestamp - timedelta(hours=1)
                    ).isoformat().replace("+00:00", "Z"),
                    interval_end_utc=timestamp.isoformat().replace("+00:00", "Z"),
                    source_timestamp=row["utc_timestamp"],
                    temperature_C=float(row["temperature_C"]),
                    relative_humidity=float(row["outdoor_rh_fraction"]),
                    station_pressure_Pa=float(row["station_pressure_Pa"]),
                    prepared_buck_water_vapour_partial_Pa=p_water,
                    prepared_buck_water_mole_fraction=water_fraction,
                    source_row_sha256=canonical_sha(row),
                )
            )
    if len(selected) != hours:
        raise ValueError(
            f"expected {hours} complete rows for {start_utc}, found {len(selected)}"
        )
    for previous, current in zip(selected, selected[1:]):
        if _parse_utc(current.interval_end_utc) - _parse_utc(
            previous.interval_end_utc
        ) != timedelta(hours=1):
            raise ValueError("weather rows are not contiguous hourly interval ends")
    return selected


def constant_weather_hours(
    *,
    temperature_C: float = 25.0,
    relative_humidity: float = 0.5,
    pressure_Pa: float = 100000.0,
    hours: int = 12,
) -> list[WeatherHour]:
    p_water = relative_humidity * buck_saturation_pressure_Pa(temperature_C)
    rows = []
    for index in range(hours):
        row = {
            "source_timestamp": f"constant-hour-{index + 1}",
            "temperature_C": temperature_C,
            "relative_humidity": relative_humidity,
            "station_pressure_Pa": pressure_Pa,
            "prepared_buck_water_vapour_partial_Pa": p_water,
        }
        rows.append(
            WeatherHour(
                interval_start_utc=f"2000-01-01T{index:02d}:00:00Z",
                interval_end_utc=f"2000-01-01T{index + 1:02d}:00:00Z",
                source_timestamp=row["source_timestamp"],
                temperature_C=temperature_C,
                relative_humidity=relative_humidity,
                station_pressure_Pa=pressure_Pa,
                prepared_buck_water_vapour_partial_Pa=p_water,
                prepared_buck_water_mole_fraction=p_water / pressure_Pa,
                source_row_sha256=canonical_sha(row),
            )
        )
    return rows


def weather_rows_sha256(rows: list[WeatherHour]) -> str:
    return canonical_sha([row.as_dict() for row in rows])


def load_common_config() -> dict:
    return json.loads(COMMON_SCREEN.read_text(encoding="utf-8"))


def weather_config(
    base_config: dict,
    hour: WeatherHour,
    p_water_mode: str,
    *,
    control_variant: str | None = FIXED_CONTROL,
) -> dict:
    config = deepcopy(base_config)
    config.setdefault("ambient", {}).update(
        {
            "temperature_C": hour.temperature_C,
            "pressure_Pa": getattr(
                hour, "station_pressure_Pa", hour.pressure_Pa
            ),
            "co2_ppm_dry": getattr(hour, "co2_ppm_dry", 420.0),
        }
    )
    if p_water_mode == "buck_explicit":
        config["ambient"]["relative_humidity"] = hour.relative_humidity
    elif p_water_mode != "core_rh":
        raise ValueError(p_water_mode)
    return apply_control_variant(config, control_variant)


def core_closure(
    adapter,
    config,
    hour,
    *,
    p_water_mode,
    rtol,
    max_step_s,
    pump_conditioner=False,
):
    override = (
        hour.prepared_buck_water_vapour_partial_Pa
        if p_water_mode == "buck_explicit"
        else None
    )
    if pump_conditioner:
        config = enable_pump_conditioner(config)
    return run_cycle(
        adapter,
        config,
        relative_humidity=hour.relative_humidity,
        rtol=rtol,
        max_step_s=max_step_s,
        rhs_only=True,
        water_vapour_partial_pressure_Pa=override,
    )


def _phase_duration(config, phase):
    values = {
        "adsorption": config["cycle"]["adsorption_s"],
        "evacuation": config["cycle"]["evacuation_s"],
        "desorption": config["cycle"]["desorption_s"],
        "cooling": config["cycle"]["cooling_max_s"],
        "repressurization": config["cycle"]["repressurization_max_s"],
    }
    return float(values[phase])


def _endpoint(phase, state, closure, config):
    indices = closure["indices"]
    ambient_K = float(config["ambient"]["temperature_C"]) + 273.15
    ambient_p = float(config["ambient"]["pressure_Pa"])
    temp = float(state[indices["TEMP"]])
    pressure = closure["pressure"](state)
    if phase == "adsorption":
        temp_error = abs(temp - ambient_K)
        pressure_error = abs(pressure - ambient_p)
        return {
            "temperature_end_K": temp,
            "temperature_target_K": ambient_K,
            "temperature_error_K": temp_error,
            "temperature_ok": temp_error <= 1.0,
            "pressure_end_Pa": pressure,
            "pressure_target_Pa": ambient_p,
            "pressure_error_Pa": pressure_error,
            "pressure_ok": pressure_error <= 0.01 * ambient_p,
            "endpoint_ok": temp_error <= 1.0
            and pressure_error <= 0.01 * ambient_p,
        }
    if phase == "evacuation":
        target = float(config["cycle"]["regeneration_pressure_Pa"])
        error = abs(pressure - target)
        return {
            "pressure_end_Pa": pressure,
            "pressure_target_Pa": target,
            "pressure_error_Pa": error,
            "pressure_ok": error <= 0.02 * target,
            "endpoint_ok": error <= 0.02 * target,
        }
    if phase == "desorption":
        target = float(config["cycle"]["regeneration_temperature_C"]) + 273.15
        error = abs(temp - target)
        return {
            "temperature_end_K": temp,
            "temperature_target_K": target,
            "temperature_error_K": error,
            "temperature_ok": error <= 2.0,
            "endpoint_ok": error <= 2.0,
        }
    if phase == "cooling":
        error = temp - ambient_K
        cooling_stop_offset_K = config["cycle"].get("cooling_stop_offset_K")
        cooling_event_target_K = (
            ambient_K + float(cooling_stop_offset_K)
            if cooling_stop_offset_K is not None
            else None
        )
        return {
            "temperature_end_K": temp,
            "temperature_target_K": ambient_K + 1.0,
            "temperature_error_K": error,
            "temperature_ok": temp <= ambient_K + 1.0,
            "cooling_event_target_K": cooling_event_target_K,
            "cooling_event_ok": (
                None
                if cooling_event_target_K is None
                else temp <= cooling_event_target_K
            ),
            "endpoint_ok": temp <= ambient_K + 1.0,
        }
    if phase == "repressurization":
        target = 0.99 * ambient_p
        return {
            "pressure_end_Pa": pressure,
            "pressure_target_Pa": target,
            "pressure_error_Pa": pressure - target,
            "pressure_ok": pressure >= target,
            "endpoint_ok": pressure >= target,
        }
    raise ValueError(phase)


def _event(phase, state, closure, config):
    if phase == "evacuation":
        return (
            closure["pressure"](state)
            - float(config["cycle"]["regeneration_pressure_Pa"])
        )
    if phase == "repressurization":
        return closure["pressure"](state) - 0.995 * float(
            config["ambient"]["pressure_Pa"]
        )
    if phase == "cooling":
        cooling_stop_offset_K = config["cycle"].get("cooling_stop_offset_K")
        if cooling_stop_offset_K is None:
            return None
        return float(state[closure["indices"]["TEMP"]]) - (
            float(config["ambient"]["temperature_C"])
            + 273.15
            + float(cooling_stop_offset_K)
        )
    return None


def _segment_scale(*, delta_heat, delta_cool, delta_loss, delta_h_in, delta_h_out, delta_u):
    return gross_energy_scale(
        delta_u,
        abs(delta_heat)
        + abs(delta_cool)
        + abs(delta_loss)
        + abs(delta_h_in - delta_h_out),
    )


def _diagnostic_saturation_pressure_Pa(adapter, temperature_K):
    selector = getattr(adapter, "diagnostic_saturation_pressure_Pa", None)
    if callable(selector):
        return float(selector(float(temperature_K))), str(
            adapter.diagnostic_saturation_law
        )
    return physical_saturation_pressure_Pa(float(temperature_K))


def _diagnostic_relative_humidity(adapter, water_partial_Pa, temperature_K):
    selector = getattr(adapter, "diagnostic_saturation_pressure_Pa", None)
    if callable(selector):
        return float(water_partial_Pa) / float(
            selector(float(temperature_K))
        )
    return float(water_partial_Pa) / buck_saturation_pressure_Pa(
        float(temperature_K)
    )


def _diagnostic_relative_humidity_basis(adapter):
    if callable(getattr(adapter, "diagnostic_saturation_pressure_Pa", None)):
        return str(adapter.diagnostic_relative_humidity_basis)
    return "Buck 1996 saturation at T_bed"


def _diagnostics(solution, closure, adapter, phase):
    accepted = np.asarray(solution.y, dtype=float)
    indices = closure["indices"]
    temperatures = accepted[indices["TEMP"]]
    gas = np.maximum(accepted[indices["GAS"]], 0.0)
    total = gas.sum(axis=0)
    pressures = total * R * np.maximum(temperatures, 0.0) / max(
        total[0] * R * max(temperatures[0], 0.0) / max(closure["pressure"](accepted[:, 0]), 1.0),
        1e-30,
    )
    saturation_rows = [
        _diagnostic_saturation_pressure_Pa(adapter, float(value))
        for value in temperatures
    ]
    sat_pressure = np.array([row[0] for row in saturation_rows])
    saturation_laws = sorted({row[1] for row in saturation_rows})
    water_pressure = np.divide(gas[1], total, out=np.zeros_like(total), where=total > 1e-30) * pressures
    saturation_ratio = np.divide(
        water_pressure,
        sat_pressure,
        out=np.zeros_like(water_pressure),
        where=sat_pressure > 0.0,
    )
    pump_values = [
        closure["pump_evaluate"](accepted[:, index])
        for index in range(accepted.shape[1])
    ]
    pump_temperatures = np.array(
        [row["inlet_temperature_K"] for row in pump_values]
    )
    pump_pressures = np.array(
        [row["pressure_Pa"] for row in pump_values]
    )
    pump_overrun = np.array(
        [row["media_temperature_overrun_K"] for row in pump_values]
    )
    conditioner_identity = closure.get("pump_conditioner_identity")
    conditioner_connected = bool(
        conditioner_identity
        and phase in ("evacuation", "desorption", "cooling")
    )
    if conditioner_connected:
        conditioner_values = [
            closure["conditioner_evaluate_state"](accepted[:, index])
            for index in range(accepted.shape[1])
        ]
    else:
        conditioner_values = []
    water_support = accepted[indices["QW"]] / max(
        float(adapter.water_scale_mol_kg), 1e-30
    )
    min_temp = float(np.min(temperatures))
    max_temp = float(np.max(temperatures))
    return {
        "accepted_state_count": int(accepted.shape[1]),
        "minimum_temperature_K": min_temp,
        "maximum_temperature_K": max_temp,
        "minimum_pressure_Pa": float(np.min(pressures)),
        "maximum_pressure_Pa": float(np.max(pressures)),
        "minimum_pump_inlet_temperature_K": float(
            np.min(pump_temperatures)
        ),
        "maximum_pump_inlet_temperature_K": float(
            np.max(pump_temperatures)
        ),
        "minimum_pump_pressure_Pa": float(np.min(pump_pressures)),
        "maximum_pump_pressure_Pa": float(np.max(pump_pressures)),
        "maximum_pump_media_temperature_overrun_K": float(
            np.max(pump_overrun)
        ),
        "conditioner_connected": conditioner_connected,
        "conditioner_identity": conditioner_identity,
        "minimum_conditioned_pump_inlet_temperature_K": (
            float(
                np.min(
                    [
                        row["outlet_temperature_K"]
                        for row in conditioner_values
                    ]
                )
            )
            if conditioner_values
            else None
        ),
        "maximum_conditioned_pump_inlet_temperature_K": (
            float(
                np.max(
                    [
                        row["outlet_temperature_K"]
                        for row in conditioner_values
                    ]
                )
            )
            if conditioner_values
            else None
        ),
        "maximum_conditioning_cooling_W": (
            float(
                np.max(
                    [
                        row["conditioner_cooling_W"]
                        for row in conditioner_values
                    ]
                )
            )
            if conditioner_values
            else 0.0
        ),
        "maximum_liquid_flow_mol_s": (
            float(
                np.max(
                    [
                        row["liquid_flow_mol_s"]
                        for row in conditioner_values
                    ]
                )
            )
            if conditioner_values
            else 0.0
        ),
        "maximum_liquid_fraction": (
            float(
                np.max(
                    [
                        row["liquid_fraction"]
                        for row in conditioner_values
                    ]
                )
            )
            if conditioner_values
            else 0.0
        ),
        "minimum_water_saturation_ratio": float(np.min(saturation_ratio)),
        "maximum_water_saturation_ratio": float(np.max(saturation_ratio)),
        "maximum_water_support_fraction": float(np.max(water_support)),
        "minimum_water_loading_mol_kg": float(np.min(accepted[indices["QW"]])),
        "maximum_water_loading_mol_kg": float(np.max(accepted[indices["QW"]])),
        "flags": {
            "actual_supersaturation_excursion": bool(
                np.max(saturation_ratio) > 1.0 + 1e-10
            ),
            "temperature_support_excursion": bool(
                max_temp > float(adapter.maximum_bed_temperature_K) + 1e-9
            ),
            "water_support_excursion": bool(np.max(water_support) > 1.0 + 1e-10),
            "subfreezing_bed_state_observed": bool(min_temp < 273.15),
            "frost_model_absent": True,
            "correlation_domain_violation": bool(
                min_temp < 255.9
                or max_temp
                > (
                    647.096
                    if callable(
                        getattr(
                            adapter,
                            "diagnostic_saturation_pressure_Pa",
                            None,
                        )
                    )
                    else 373.0
                )
            ),
            "nonfinite_accepted_state": bool(
                not np.all(np.isfinite(accepted))
                or not np.all(np.isfinite(saturation_ratio))
            ),
            "pump_media_temperature_excursion": bool(
                np.max(pump_overrun) > 0.0
            ),
            "pump_curve_upper_domain_exceeded": bool(
                any(row["upper_domain_exceeded"] for row in pump_values)
            ),
            "conditioner_liquid_drain_present": bool(
                conditioner_connected
                and any(
                    row["liquid_flow_mol_s"] > 0.0
                    for row in conditioner_values
                )
            ),
            "conditioned_pump_media_temperature_excursion": bool(
                conditioner_connected and np.max(pump_overrun) > 0.0
            ),
            "conditioner_hardware_qualified": False,
        },
        "pump_assumptions": {
            **closure["pump_assumptions"],
            "pressure_regime_seen": sorted(
                {row["pressure_regime"] for row in pump_values}
            ),
            "below_or_at_ultimate_observed": bool(
                any(row["below_or_at_ultimate"] for row in pump_values)
            ),
        },
        "saturation_diagnostic_law": "; ".join(saturation_laws),
        "relative_humidity_basis": _diagnostic_relative_humidity_basis(
            adapter
        ),
    }


def _merge_diagnostics(left, right):
    keys = (
        "actual_supersaturation_excursion",
        "temperature_support_excursion",
        "water_support_excursion",
        "subfreezing_bed_state_observed",
        "frost_model_absent",
        "correlation_domain_violation",
        "nonfinite_accepted_state",
        "pump_media_temperature_excursion",
        "pump_curve_upper_domain_exceeded",
        "conditioner_liquid_drain_present",
        "conditioned_pump_media_temperature_excursion",
        "conditioner_hardware_qualified",
    )
    return {
        "accepted_state_count": left["accepted_state_count"]
        + right["accepted_state_count"],
        "minimum_temperature_K": min(
            left["minimum_temperature_K"], right["minimum_temperature_K"]
        ),
        "maximum_temperature_K": max(
            left["maximum_temperature_K"], right["maximum_temperature_K"]
        ),
        "minimum_pressure_Pa": min(
            left["minimum_pressure_Pa"], right["minimum_pressure_Pa"]
        ),
        "maximum_pressure_Pa": max(
            left["maximum_pressure_Pa"], right["maximum_pressure_Pa"]
        ),
        "minimum_pump_inlet_temperature_K": min(
            left["minimum_pump_inlet_temperature_K"],
            right["minimum_pump_inlet_temperature_K"],
        ),
        "maximum_pump_inlet_temperature_K": max(
            left["maximum_pump_inlet_temperature_K"],
            right["maximum_pump_inlet_temperature_K"],
        ),
        "minimum_pump_pressure_Pa": min(
            left["minimum_pump_pressure_Pa"],
            right["minimum_pump_pressure_Pa"],
        ),
        "maximum_pump_pressure_Pa": max(
            left["maximum_pump_pressure_Pa"],
            right["maximum_pump_pressure_Pa"],
        ),
        "maximum_pump_media_temperature_overrun_K": max(
            left["maximum_pump_media_temperature_overrun_K"],
            right["maximum_pump_media_temperature_overrun_K"],
        ),
        "conditioner_connected": bool(
            left["conditioner_connected"] or right["conditioner_connected"]
        ),
        "conditioner_identity": (
            right["conditioner_identity"]
            if right["conditioner_identity"] is not None
            else left["conditioner_identity"]
        ),
        "minimum_conditioned_pump_inlet_temperature_K": min(
            (
                value
                for value in (
                    left["minimum_conditioned_pump_inlet_temperature_K"],
                    right["minimum_conditioned_pump_inlet_temperature_K"],
                )
                if value is not None
            ),
            default=None,
        ),
        "maximum_conditioned_pump_inlet_temperature_K": max(
            (
                value
                for value in (
                    left["maximum_conditioned_pump_inlet_temperature_K"],
                    right["maximum_conditioned_pump_inlet_temperature_K"],
                )
                if value is not None
            ),
            default=None,
        ),
        "maximum_conditioning_cooling_W": max(
            left["maximum_conditioning_cooling_W"],
            right["maximum_conditioning_cooling_W"],
        ),
        "maximum_liquid_flow_mol_s": max(
            left["maximum_liquid_flow_mol_s"],
            right["maximum_liquid_flow_mol_s"],
        ),
        "maximum_liquid_fraction": max(
            left["maximum_liquid_fraction"],
            right["maximum_liquid_fraction"],
        ),
        "minimum_water_saturation_ratio": min(
            left["minimum_water_saturation_ratio"],
            right["minimum_water_saturation_ratio"],
        ),
        "maximum_water_saturation_ratio": max(
            left["maximum_water_saturation_ratio"],
            right["maximum_water_saturation_ratio"],
        ),
        "maximum_water_support_fraction": max(
            left["maximum_water_support_fraction"],
            right["maximum_water_support_fraction"],
        ),
        "flags": {key: left["flags"][key] or right["flags"][key] for key in keys},
        "pump_assumptions": {
            **right["pump_assumptions"],
            "pressure_regime_seen": sorted(
                set(left["pump_assumptions"]["pressure_regime_seen"])
                | set(right["pump_assumptions"]["pressure_regime_seen"])
            ),
            "below_or_at_ultimate_observed": bool(
                left["pump_assumptions"]["below_or_at_ultimate_observed"]
                or right["pump_assumptions"]["below_or_at_ultimate_observed"]
            ),
        },
        "saturation_diagnostic_law": left.get(
            "saturation_diagnostic_law",
            "Buck 1996 at T<=50C; NIST/Stull above 50C",
        ),
        "relative_humidity_basis": left.get(
            "relative_humidity_basis",
            "Buck 1996 saturation at T_bed",
        ),
    }


def _fluxes(state, closure):
    indices = closure["indices"]
    names = {
        "in": indices["IN"],
        "product": indices["PRODUCT"],
        "vent": indices["VENT"],
        "heat": indices["HEAT"],
        "pump": indices["PUMP"],
        "fan": indices["FAN"],
        "cool": indices["COOL"],
        "inlet_h": indices["INLET_H"],
        "outlet_h": indices["OUTLET_H"],
        "loss": indices["LOSS"],
    }
    for name in (
        "conditioned_product",
        "conditioned_vent",
        "direct_vent",
        "liquid_drain",
    ):
        index = indices.get(name.upper())
        if index is not None:
            names[name] = index
    for name in ("conditioner_cool", "external_outlet_h"):
        index = indices.get(name.upper())
        if index is not None:
            names[name] = index
    result = {}
    for name, value in names.items():
        if isinstance(value, slice):
            result[name] = {
                key: float(state[index])
                for key, index in zip(("co2", "water", "n2", "o2"), range(value.start, value.stop))
            }
        else:
            result[name] = float(state[value])
    return result


def _checkpoint_payload(
    *,
    job_id,
    attempt_id,
    restart_of_attempt_id,
    context,
    state,
    initial_state,
    active_phase_start_state,
    cursor,
    phase_history,
    chunk_records,
    active_diagnostics,
    weather_rows,
    checkpoint_status,
    sequence,
    closure,
    initial_step_records=None,
):
    return {
        "schema": CHECKPOINT_SCHEMA,
        "status": checkpoint_status,
        "sequence": sequence,
        "job_id": job_id,
        "attempt_id": attempt_id,
        "restart_of_attempt_id": restart_of_attempt_id,
        "context": deepcopy(context),
        "state": np.asarray(state, dtype=float).tolist(),
        "state_sha256": canonical_sha(np.asarray(state, dtype=float).tolist()),
        "initial_state": np.asarray(initial_state, dtype=float).tolist(),
        "initial_state_sha256": canonical_sha(
            np.asarray(initial_state, dtype=float).tolist()
        ),
        "active_phase_start_state": (
            np.asarray(active_phase_start_state, dtype=float).tolist()
            if active_phase_start_state is not None
            else None
        ),
        "active_phase_start_state_sha256": (
            canonical_sha(np.asarray(active_phase_start_state, dtype=float).tolist())
            if active_phase_start_state is not None
            else None
        ),
        "cursor": deepcopy(cursor),
        "phase_history": deepcopy(phase_history),
        "chunk_records": deepcopy(chunk_records),
        "active_diagnostics": deepcopy(active_diagnostics),
        "weather_rows": [row.as_dict() for row in weather_rows],
        "weather_rows_sha256": weather_rows_sha256(weather_rows),
        "initial_step_records": deepcopy(initial_step_records or []),
        "initial_step_records_sha256": canonical_sha(
            initial_step_records or []
        ),
        "cumulative_fluxes": _fluxes(np.asarray(state, dtype=float), closure),
    }


def save_checkpoint(path: Path, payload: dict) -> str:
    payload = deepcopy(payload)
    payload.pop("checkpoint_payload_sha256", None)
    payload["checkpoint_payload_sha256"] = canonical_sha(payload)
    latest_hash = atomic_write_json(path, payload)
    history = path.parent / "checkpoints"
    history.mkdir(parents=True, exist_ok=True)
    atomic_write_json(history / f"checkpoint_{payload['sequence']:06d}.json", payload)
    return latest_hash


def load_checkpoint(path: Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    claimed = payload.get("checkpoint_payload_sha256")
    copy = dict(payload)
    copy.pop("checkpoint_payload_sha256", None)
    if claimed != canonical_sha(copy):
        raise ValueError("checkpoint payload hash mismatch")
    if payload.get("schema") not in SUPPORTED_CHECKPOINT_SCHEMAS:
        raise ValueError("unsupported checkpoint schema")
    return payload


def validate_checkpoint(
    checkpoint,
    *,
    job_id,
    material_id,
    config_sha256,
    weather_rows_sha256_expected,
    source_release_sha256,
    source_hashes,
    control_identity_expected=None,
    initial_step_identity_expected=None,
    pump_conditioner_identity_expected=None,
):
    context = checkpoint["context"]
    expected = {
        "job_id": job_id,
        "material_id": material_id,
        "config_sha256": config_sha256,
        "weather_rows_sha256": weather_rows_sha256_expected,
        "source_release_sha256": source_release_sha256,
    }
    for key, value in expected.items():
        actual = checkpoint["job_id"] if key == "job_id" else context.get(key)
        if actual != value:
            raise ValueError(f"checkpoint {key} mismatch")
    if context.get("source_hashes") != source_hashes:
        raise ValueError("checkpoint source hash mismatch")
    if (
        control_identity_expected is not None
        and context.get("control_identity") != control_identity_expected
    ):
        raise ValueError("checkpoint control-policy identity mismatch")
    if (
        initial_step_identity_expected is not None
        and context.get("initial_step_identity")
        != initial_step_identity_expected
    ):
        raise ValueError("checkpoint initial-step identity mismatch")
    if (
        pump_conditioner_identity_expected is not None
        and context.get("pump_conditioner_identity")
        != pump_conditioner_identity_expected
    ):
        raise ValueError("checkpoint pump-conditioner identity mismatch")
    if checkpoint["state_sha256"] != canonical_sha(checkpoint["state"]):
        raise ValueError("checkpoint state hash mismatch")
    if checkpoint["initial_state_sha256"] != canonical_sha(
        checkpoint["initial_state"]
    ):
        raise ValueError("checkpoint initial-state hash mismatch")
    if checkpoint["active_phase_start_state_sha256"] != canonical_sha(
        checkpoint["active_phase_start_state"]
    ):
        raise ValueError("checkpoint active-phase-start hash mismatch")
    if checkpoint["weather_rows_sha256"] != canonical_sha(
        checkpoint["weather_rows"]
    ):
        raise ValueError("checkpoint weather-row hash mismatch")
    if "initial_step_records" in checkpoint:
        if checkpoint.get("initial_step_records_sha256") != canonical_sha(
            checkpoint["initial_step_records"]
        ):
            raise ValueError("checkpoint initial-step record hash mismatch")
    cursor = checkpoint["cursor"]
    phase_index = int(cursor["phase_index"])
    if not 0 <= phase_index < len(PHASES):
        raise ValueError("checkpoint phase index outside schedule")
    if cursor["phase"] != PHASES[phase_index]:
        raise ValueError("checkpoint phase/index mismatch")
    if abs(
        float(cursor["phase_elapsed_s"])
        - (float(cursor["elapsed_s"]) - float(cursor["phase_start_elapsed_s"]))
    ) > 1e-8:
        raise ValueError("checkpoint phase cursor mismatch")
    if not 0.0 <= float(cursor["phase_elapsed_s"]) <= float(
        cursor["active_phase_duration_s"]
    ) + 1e-8:
        raise ValueError("checkpoint phase elapsed outside phase duration")
    weather_index = int(cursor["weather_index"])
    if not 0 <= weather_index < len(checkpoint["weather_rows"]):
        raise ValueError("checkpoint weather index outside weather rows")
    expected_weather_index = min(
        int(float(cursor["elapsed_s"]) // 3600),
        len(checkpoint["weather_rows"]) - 1,
    )
    if weather_index != expected_weather_index:
        raise ValueError("checkpoint weather cursor mismatch")
    completed_repressurizations = sum(
        row["phase"] == "repressurization" for row in checkpoint["phase_history"]
    )
    if int(cursor["completed_cycles"]) != completed_repressurizations:
        raise ValueError("checkpoint completed-cycle cursor mismatch")
    if len(checkpoint["phase_history"]) % len(PHASES) != phase_index:
        raise ValueError("checkpoint phase-history/index mismatch")
    for history_index, row in enumerate(checkpoint["phase_history"]):
        if row["phase"] != PHASES[history_index % len(PHASES)]:
            raise ValueError("checkpoint phase-history chronology mismatch")
    state = np.asarray(checkpoint["state"], dtype=float)
    conditioner_ledger_columns = (
        18 if context.get("pump_conditioner_identity") is not None else 0
    )
    n_sites = len(state) - 27 - conditioner_ledger_columns
    if n_sites <= 0:
        raise ValueError("checkpoint state shape invalid")
    indices = {
        "Q": slice(0, n_sites),
        "IN": slice(n_sites + 8, n_sites + 12),
        "PRODUCT": slice(n_sites + 12, n_sites + 16),
        "VENT": slice(n_sites + 16, n_sites + 20),
        "HEAT": n_sites + 20,
        "PUMP": n_sites + 21,
        "FAN": n_sites + 22,
        "COOL": n_sites + 23,
        "INLET_H": n_sites + 24,
        "OUTLET_H": n_sites + 25,
        "LOSS": n_sites + 26,
    }
    if context.get("pump_conditioner_identity") is not None:
        base = n_sites + 27
        indices.update(
            {
                "CONDITIONED_PRODUCT": slice(base, base + 4),
                "CONDITIONED_VENT": slice(base + 4, base + 8),
                "DIRECT_VENT": slice(base + 8, base + 12),
                "LIQUID_DRAIN": slice(base + 12, base + 16),
                "CONDITIONER_COOL": base + 16,
                "EXTERNAL_OUTLET_H": base + 17,
            }
        )
    expected_fluxes = _fluxes(state, {"indices": indices})
    if checkpoint["cumulative_fluxes"] != expected_fluxes:
        raise ValueError("checkpoint cumulative-flux mismatch")
    return checkpoint


def validate_numerical_source_migration(
    checkpoint,
    *,
    checkpoint_file_sha256,
    source_migration,
    current_source_release_sha256,
    current_source_hashes,
    job_id,
    material_id,
    config_sha256,
    weather_rows_sha256_expected,
    control_identity_expected,
    initial_step_identity_expected,
    pump_conditioner_identity_expected,
):
    """Validate a pinned checkpoint before an explicit numerical-source resume."""
    required = {
        "schema",
        "original_attempt_id",
        "job_id",
        "checkpoint_file_sha256",
        "checkpoint_payload_sha256",
        "state_sha256",
        "initial_state_sha256",
        "active_phase_start_state_sha256",
        "cursor_sha256",
        "phase_history_sha256",
        "chunk_records_sha256",
        "cumulative_fluxes_sha256",
        "original_source_release_sha256",
        "original_source_hashes",
        "new_source_release_sha256",
        "new_source_hashes",
        "allowed_changed_paths",
        "allowed_added_paths",
    }
    missing = sorted(required - set(source_migration))
    if missing:
        raise ValueError(f"source migration missing fields: {missing}")
    if source_migration["schema"] != SOURCE_MIGRATION_SCHEMA:
        raise ValueError("unsupported source migration schema")
    if source_migration["job_id"] != job_id:
        raise ValueError("source migration job identity mismatch")
    if checkpoint["job_id"] != job_id:
        raise ValueError("source migration checkpoint job mismatch")
    if checkpoint["attempt_id"] != source_migration["original_attempt_id"]:
        raise ValueError("source migration original attempt mismatch")
    if checkpoint_file_sha256 != source_migration["checkpoint_file_sha256"]:
        raise ValueError("source migration checkpoint file hash mismatch")
    expected_hashes = {
        "checkpoint_payload_sha256": checkpoint[
            "checkpoint_payload_sha256"
        ],
        "state_sha256": checkpoint["state_sha256"],
        "initial_state_sha256": checkpoint["initial_state_sha256"],
        "active_phase_start_state_sha256": checkpoint[
            "active_phase_start_state_sha256"
        ],
        "cursor_sha256": canonical_sha(checkpoint["cursor"]),
        "phase_history_sha256": canonical_sha(checkpoint["phase_history"]),
        "chunk_records_sha256": canonical_sha(checkpoint["chunk_records"]),
        "cumulative_fluxes_sha256": canonical_sha(
            checkpoint["cumulative_fluxes"]
        ),
    }
    for key, actual in expected_hashes.items():
        if source_migration[key] != actual:
            raise ValueError(f"source migration {key} mismatch")
    original_release = source_migration["original_source_release_sha256"]
    original_sources = source_migration["original_source_hashes"]
    validate_checkpoint(
        checkpoint,
        job_id=job_id,
        material_id=material_id,
        config_sha256=config_sha256,
        weather_rows_sha256_expected=weather_rows_sha256_expected,
        source_release_sha256=original_release,
        source_hashes=original_sources,
        control_identity_expected=control_identity_expected,
        initial_step_identity_expected=initial_step_identity_expected,
        pump_conditioner_identity_expected=pump_conditioner_identity_expected,
    )
    if current_source_release_sha256 != source_migration[
        "new_source_release_sha256"
    ]:
        raise ValueError("source migration new release mismatch")
    if current_source_hashes != source_migration["new_source_hashes"]:
        raise ValueError("source migration new source hashes mismatch")
    old_paths = set(original_sources)
    new_paths = set(current_source_hashes)
    removed = sorted(old_paths - new_paths)
    added = sorted(new_paths - old_paths)
    changed = sorted(
        path
        for path in old_paths & new_paths
        if original_sources[path] != current_source_hashes[path]
    )
    if removed:
        raise ValueError(f"source migration removed paths: {removed}")
    if changed != sorted(source_migration["allowed_changed_paths"]):
        raise ValueError(
            f"source migration undeclared changed paths: {changed}"
        )
    if added != sorted(source_migration["allowed_added_paths"]):
        raise ValueError(
            f"source migration undeclared added paths: {added}"
        )
    return {
        "schema": SOURCE_MIGRATION_SCHEMA,
        "status": "validated",
        "original_attempt_id": checkpoint["attempt_id"],
        "original_checkpoint_file_sha256": checkpoint_file_sha256,
        "original_checkpoint_payload_sha256": checkpoint[
            "checkpoint_payload_sha256"
        ],
        "original_state_sha256": checkpoint["state_sha256"],
        "original_source_release_sha256": original_release,
        "new_source_release_sha256": current_source_release_sha256,
        "changed_paths": changed,
        "added_paths": added,
        "physical_state_modified": False,
        "history_modified": False,
        "cursor_modified": False,
        "ledgers_modified": False,
    }


def _phase_endpoint_violations(phase_history):
    return [
        {
            "cycle": row["cycle"],
            "phase": row["phase"],
            "endpoint": row["endpoint"],
        }
        for row in phase_history
        if not row["endpoint"]["endpoint_ok"]
    ]


def run_weather_session(
    adapter,
    *,
    weather_hours,
    base_config,
    job_id,
    attempt_id,
    restart_of_attempt_id,
    checkpoint_path,
    context,
    p_water_mode="buck_explicit",
    initial_state=None,
    resume_checkpoint=None,
    source_migration=None,
    rtol=1e-9,
    max_step_s=0.5,
    maximum_cycles=None,
    planned_stop_s=None,
    progress_callback=None,
    solver_method="BDF",
    use_bounded_jacobian=False,
    milestone_at_s=None,
    control_variant=FIXED_CONTROL,
    pump_conditioner=False,
    runtime_control=None,
):
    if p_water_mode not in ("buck_explicit", "core_rh"):
        raise ValueError(p_water_mode)
    if not weather_hours:
        raise ValueError("weather window empty")
    horizon_s = float(len(weather_hours) * 3600)
    if pump_conditioner:
        base_config = enable_pump_conditioner(base_config)
    rows_hash = weather_rows_sha256(weather_hours)
    config_sha = canonical_sha(base_config)
    policy_config = weather_config(
        base_config,
        weather_hours[0],
        p_water_mode,
        control_variant=control_variant,
    )
    checkpoint_control_identity = control_identity(policy_config)
    checkpoint_initial_step_identity = initial_step_identity(policy_config)
    context = dict(context)
    context.update(
        {
            "material_id": adapter.material_id,
            "config_sha256": config_sha,
            "weather_rows_sha256": rows_hash,
            "p_water_mode": p_water_mode,
            "rtol": rtol,
            "max_step_s": max_step_s,
            "solver_method": solver_method,
            "use_bounded_jacobian": bool(use_bounded_jacobian),
            "control_identity": checkpoint_control_identity,
            "initial_step_identity": checkpoint_initial_step_identity,
            "pump_conditioner_identity": pump_conditioner_identity(
                policy_config
            ),
            "environment": {
                "python": __import__("sys").version,
                "numpy": np.__version__,
                "scipy": __import__("scipy").__version__,
            },
        }
    )
    if resume_checkpoint is not None and source_migration is not None:
        context["source_migration"] = validate_numerical_source_migration(
            resume_checkpoint,
            checkpoint_file_sha256=source_migration[
                "checkpoint_file_sha256"
            ],
            source_migration=source_migration,
            current_source_release_sha256=context[
                "source_release_sha256"
            ],
            current_source_hashes=context["source_hashes"],
            job_id=job_id,
            material_id=adapter.material_id,
            config_sha256=config_sha,
            weather_rows_sha256_expected=rows_hash,
            control_identity_expected=checkpoint_control_identity,
            initial_step_identity_expected=checkpoint_initial_step_identity,
            pump_conditioner_identity_expected=context.get(
                "pump_conditioner_identity"
            ),
        )
    elif resume_checkpoint is not None:
        validate_checkpoint(
            resume_checkpoint,
            job_id=job_id,
            material_id=adapter.material_id,
            config_sha256=config_sha,
            weather_rows_sha256_expected=rows_hash,
            source_release_sha256=context["source_release_sha256"],
            source_hashes=context["source_hashes"],
            control_identity_expected=checkpoint_control_identity,
            initial_step_identity_expected=checkpoint_initial_step_identity,
            pump_conditioner_identity_expected=context.get(
                "pump_conditioner_identity"
            ),
        )
    elif source_migration is not None:
        raise ValueError("source migration requires a resume checkpoint")
    if resume_checkpoint is not None:
        cursor = deepcopy(resume_checkpoint["cursor"])
        state = np.asarray(resume_checkpoint["state"], dtype=float)
        initial_state_value = np.asarray(
            resume_checkpoint["initial_state"], dtype=float
        )
        active_phase_start_state = (
            np.asarray(checkpoint_value, dtype=float)
            if (checkpoint_value := resume_checkpoint["active_phase_start_state"])
            is not None
            else state.copy()
        )
        phase_history = deepcopy(resume_checkpoint["phase_history"])
        chunk_records = deepcopy(resume_checkpoint["chunk_records"])
        active_diagnostics = deepcopy(resume_checkpoint["active_diagnostics"])
        initial_step_records = deepcopy(
            resume_checkpoint.get("initial_step_records", [])
        )
        sequence = int(resume_checkpoint["sequence"]) + 1
    else:
        hour = weather_hours[0]
        config = weather_config(
            base_config,
            hour,
            p_water_mode,
            control_variant=control_variant,
        )
        if runtime_control is not None:
            config["_runtime_control"] = runtime_control
        closure = core_closure(
            adapter,
            config,
            hour,
            p_water_mode=p_water_mode,
            rtol=rtol,
            max_step_s=max_step_s,
        )
        state = np.asarray(
            closure["initial_state"]() if initial_state is None else initial_state,
            dtype=float,
        ).copy()
        state[adapter.n_sites + 8 :] = 0.0
        if not np.all(np.isfinite(state)):
            raise ValueError("initial state nonfinite")
        initial_state_value = state.copy()
        active_phase_start_state = state.copy()
        phase_history = []
        chunk_records = []
        active_diagnostics = None
        initial_step_records = []
        sequence = 0
        cursor = {
            "elapsed_s": 0.0,
            "weather_index": 0,
            "phase_index": 0,
            "phase": PHASES[0],
            "phase_start_elapsed_s": 0.0,
            "phase_elapsed_s": 0.0,
            "active_phase_duration_s": _phase_duration(config, PHASES[0]),
            "completed_cycles": 0,
        }

    def current_cursor():
        return {
            **cursor,
            "elapsed_s": float(cursor["elapsed_s"]),
            "weather_index": int(cursor["weather_index"]),
            "phase_index": int(cursor["phase_index"]),
            "phase": PHASES[int(cursor["phase_index"])],
            "phase_start_elapsed_s": float(cursor["phase_start_elapsed_s"]),
            "phase_elapsed_s": float(cursor["phase_elapsed_s"]),
            "completed_cycles": int(cursor["completed_cycles"]),
        }

    def current_config_and_closure():
        hour = weather_hours[int(cursor["weather_index"])]
        config = weather_config(
            base_config,
            hour,
            p_water_mode,
            control_variant=control_variant,
        )
        if runtime_control is not None:
            config["_runtime_control"] = runtime_control
        closure = core_closure(
            adapter,
            config,
            hour,
            p_water_mode=p_water_mode,
            rtol=rtol,
            max_step_s=max_step_s,
        )
        return hour, config, closure

    hour, config, closure = current_config_and_closure()
    jacobian_audit = (
        JacobianAudit(
            physical_columns=adapter.n_sites + 8,
            temperature_index=adapter.n_sites + 3,
        )
        if use_bounded_jacobian
        else None
    )
    milestone_saved = False
    milestone_hash = None
    milestone_path = checkpoint_path.parent / "milestone_3h.json"

    def checkpoint(status):
        nonlocal sequence
        payload = _checkpoint_payload(
            job_id=job_id,
            attempt_id=attempt_id,
            restart_of_attempt_id=restart_of_attempt_id,
            context=context,
            state=state,
            initial_state=initial_state_value,
            active_phase_start_state=active_phase_start_state,
            cursor=current_cursor(),
            phase_history=phase_history,
            chunk_records=chunk_records,
            active_diagnostics=active_diagnostics,
            weather_rows=weather_hours,
            initial_step_records=initial_step_records,
            checkpoint_status=status,
            sequence=sequence,
            closure=closure,
        )
        digest = save_checkpoint(checkpoint_path, payload)
        sequence += 1
        return digest

    checkpoint_hash = checkpoint("active")

    def finish_phase(reason):
        nonlocal active_phase_start_state, active_diagnostics, phase_history
        phase = PHASES[int(cursor["phase_index"])]
        endpoint = _endpoint(phase, state, closure, config)
        boundary = (
            state[closure["indices"]["HEAT"]]
            - active_phase_start_state[closure["indices"]["HEAT"]]
            - state[closure["indices"]["COOL"]]
            + active_phase_start_state[closure["indices"]["COOL"]]
            - state[closure["indices"]["LOSS"]]
            + active_phase_start_state[closure["indices"]["LOSS"]]
            + state[closure["indices"]["INLET_H"]]
            - active_phase_start_state[closure["indices"]["INLET_H"]]
            - state[closure["indices"]["OUTLET_H"]]
            + active_phase_start_state[closure["indices"]["OUTLET_H"]]
        )
        delta_u = closure["inventory"](state) - closure["inventory"](
            active_phase_start_state
        )
        residual = delta_u - boundary
        scale = _segment_scale(
            delta_heat=(
                state[closure["indices"]["HEAT"]]
                - active_phase_start_state[closure["indices"]["HEAT"]]
            ),
            delta_cool=(
                state[closure["indices"]["COOL"]]
                - active_phase_start_state[closure["indices"]["COOL"]]
            ),
            delta_loss=(
                state[closure["indices"]["LOSS"]]
                - active_phase_start_state[closure["indices"]["LOSS"]]
            ),
            delta_h_in=(
                state[closure["indices"]["INLET_H"]]
                - active_phase_start_state[closure["indices"]["INLET_H"]]
            ),
            delta_h_out=(
                state[closure["indices"]["OUTLET_H"]]
                - active_phase_start_state[closure["indices"]["OUTLET_H"]]
            ),
            delta_u=delta_u,
        )
        water_partial_Pa = (
            (
                np.maximum(state[closure["indices"]["GAS"]], 0.0)
                / max(
                    np.sum(
                        np.maximum(
                            state[closure["indices"]["GAS"]], 0.0
                        )
                    ),
                    1e-30,
                )
            )[1]
            * closure["pressure"](state)
        )
        diagnostic_law = _diagnostic_saturation_pressure_Pa(
            adapter,
            float(state[closure["indices"]["TEMP"]]),
        )[1]
        record = {
            "cycle": int(cursor["completed_cycles"]) + 1,
            "phase": phase,
            "weather_hour_index_start": int(
                cursor.get(
                    "phase_weather_index_start",
                    cursor["weather_index"],
                )
            ),
            "weather_hour_index_end": int(cursor["weather_index"]),
            "weather_interval_end_utc": hour.interval_end_utc,
            "start_elapsed_s": float(cursor["phase_start_elapsed_s"]),
            "end_elapsed_s": float(cursor["elapsed_s"]),
            "duration_s": float(cursor["phase_elapsed_s"]),
            "completion_reason": reason,
            "endpoint": endpoint,
            "energy_residual_J": float(residual),
            "energy_transfer_scale_J": float(scale),
            "energy_closure_relative_error": float(abs(residual) / scale),
            "diagnostics": active_diagnostics,
            "temperature_end_K": float(state[closure["indices"]["TEMP"]]),
            "pressure_end_Pa": closure["pressure"](state),
            "co2_loading_end_mol_kg": float(
                np.sum(state[closure["indices"]["Q"]])
            ),
            "water_loading_end_mol_kg": float(
                state[closure["indices"]["QW"]]
            ),
            "local_relative_humidity_end": float(
                _diagnostic_relative_humidity(
                    adapter,
                    water_partial_Pa,
                    float(state[closure["indices"]["TEMP"]]),
                )
            ),
            "local_relative_humidity_basis": (
                _diagnostic_relative_humidity_basis(adapter)
            ),
            "saturation_diagnostic_law": diagnostic_law,
        }
        phase_history.append(record)
        completed = int(cursor["completed_cycles"])
        if phase == "repressurization":
            completed += 1
        cursor["phase_index"] = (int(cursor["phase_index"]) + 1) % len(PHASES)
        cursor["phase"] = PHASES[cursor["phase_index"]]
        cursor["phase_start_elapsed_s"] = float(cursor["elapsed_s"])
        cursor["phase_elapsed_s"] = 0.0
        cursor["completed_cycles"] = completed
        cursor["phase_weather_index_start"] = int(cursor["weather_index"])
        cursor["active_phase_duration_s"] = _phase_duration(
            weather_config(
                base_config,
                weather_hours[cursor["weather_index"]],
                p_water_mode,
                control_variant=control_variant,
            ),
            cursor["phase"],
        )
        active_phase_start_state = state.copy()
        active_diagnostics = None

    def report_phase_completed(completed_phase, cycle_number):
        if progress_callback is None:
            return
        progress_callback(
            {
                "event": "phase_completed",
                "attempt_id": attempt_id,
                "elapsed_s": float(cursor["elapsed_s"]),
                "phase_elapsed_s": 0.0,
                "phase_duration_s": _phase_duration(
                    config, cursor["phase"]
                ),
                "completed_cycles": int(cursor["completed_cycles"]),
                "cycle": int(cycle_number),
                "weather_index": int(cursor["weather_index"]),
                "phase": completed_phase,
                "state_sha256": canonical_sha(state.tolist()),
            }
        )

    while float(cursor["elapsed_s"]) < horizon_s - 1e-12:
        if maximum_cycles is not None and int(cursor["completed_cycles"]) >= int(
            maximum_cycles
        ):
            break
        if (
            milestone_at_s is not None
            and not milestone_saved
            and float(cursor["elapsed_s"])
            >= float(milestone_at_s) - 1e-10
        ):
            checkpoint("milestone")
            shutil.copy2(checkpoint_path, milestone_path)
            milestone_hash = file_sha(milestone_path)
            milestone_saved = True
        if planned_stop_s is not None and float(cursor["elapsed_s"]) >= float(
            planned_stop_s
        ) - 1e-10:
            checkpoint("planned_pause")
            return {
                "status": "planned_pause",
                "job_id": job_id,
                "attempt_id": attempt_id,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": file_sha(checkpoint_path),
                "cursor": current_cursor(),
                "initial_state": initial_state_value.tolist(),
                "state": state.tolist(),
                "weather_rows": [row.as_dict() for row in weather_hours],
                "phase_history": phase_history,
                "chunk_records": chunk_records,
                "active_diagnostics": active_diagnostics,
            }
        phase = PHASES[int(cursor["phase_index"])]
        duration = _phase_duration(config, phase)
        cursor["active_phase_duration_s"] = duration
        remaining_phase = duration - float(cursor["phase_elapsed_s"])
        event_value = _event(phase, state, closure, config)
        cooling_event_ready = bool(
            phase == "cooling"
            and event_value is not None
            and event_value <= 0.0
        )
        event_ready = bool(
            event_value is not None
            and (
                (phase == "evacuation" and event_value <= 0.0)
                or (phase == "repressurization" and event_value >= 0.0)
                or (
                    cooling_event_ready
                    and float(cursor["phase_elapsed_s"]) > 1e-12
                )
            )
        )
        if remaining_phase <= 1e-12 or event_ready:
            completed_cycle = int(cursor["completed_cycles"]) + 1
            finish_phase(
                "event_already_satisfied" if event_ready else "fixed_duration"
            )
            report_phase_completed(phase, completed_cycle)
            checkpoint_hash = checkpoint("active")
            hour, config, closure = current_config_and_closure()
            continue
        next_weather_end = (int(cursor["weather_index"]) + 1) * 3600.0
        step_limits = [remaining_phase, next_weather_end - float(cursor["elapsed_s"]), horizon_s - float(cursor["elapsed_s"])]
        if planned_stop_s is not None:
            step_limits.append(float(planned_stop_s) - float(cursor["elapsed_s"]))
        step_s = min(step_limits)
        if step_s <= 1e-12:
            cursor["weather_index"] = min(
                int(cursor["weather_index"]) + 1, len(weather_hours) - 1
            )
            hour, config, closure = current_config_and_closure()
            checkpoint_hash = checkpoint("active")
            continue

        before = state.copy()
        events = []
        if phase == "evacuation":
            def event_fn(_time, value):
                return _event(phase, value, closure, config)
            event_fn.terminal = True
            event_fn.direction = -1
            events.append(event_fn)
        elif phase == "repressurization":
            def event_fn(_time, value):
                return _event(phase, value, closure, config)
            event_fn.terminal = True
            event_fn.direction = 1
            events.append(event_fn)
        elif (
            phase == "cooling"
            and event_value is not None
            and event_value > 0.0
        ):
            def event_fn(_time, value):
                return _event(phase, value, closure, config)
            event_fn.terminal = True
            event_fn.direction = -1
            events.append(event_fn)
        atol = np.r_[
            np.full(adapter.n_sites + 3, 1e-9),
            np.full(4, 1e-10),
            np.full(len(state) - adapter.n_sites - 7, 1e-7),
        ]
        rhs_for_phase = closure["rhs_by_phase"][phase]
        jacobian = (
            (
                lambda time_value, state_value: bounded_physical_jacobian(
                    rhs_for_phase,
                    time_value,
                    state_value,
                    physical_columns=adapter.n_sites + 8,
                    temperature_index=adapter.n_sites + 3,
                    audit=jacobian_audit,
                )
            )
            if use_bounded_jacobian
            else None
        )
        first_step_value = None
        if (
            checkpoint_initial_step_identity["policy"]
            == INITIAL_STEP_POLICY
        ):
            first_step_value = positive_gas_first_step(
                rhs_for_phase,
                0.0,
                state,
                step_s,
                max_step_s=max_step_s,
                rtol=rtol,
                atol=atol,
                gas_indices=closure["indices"]["GAS"],
                maximum_first_step_s=checkpoint_initial_step_identity[
                    "maximum_first_step_s"
                ],
                depletion_fraction=checkpoint_initial_step_identity[
                    "gas_depletion_fraction"
                ],
            )
            initial_step_records.append(
                {
                    **first_step_value,
                    "weather_index": int(cursor["weather_index"]),
                    "phase": phase,
                    "elapsed_s": float(cursor["elapsed_s"]),
                    "cycle": int(cursor["completed_cycles"]) + 1,
                }
            )
            first_step_value = first_step_value["first_step_s"]
            checkpoint_hash = checkpoint("active")
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "integration_started",
                    "attempt_id": attempt_id,
                    "elapsed_s": float(cursor["elapsed_s"]),
                    "phase_elapsed_s": float(cursor["phase_elapsed_s"]),
                    "phase_duration_s": float(duration),
                    "completed_cycles": int(cursor["completed_cycles"]),
                    "cycle": int(cursor["completed_cycles"]) + 1,
                    "weather_index": int(cursor["weather_index"]),
                    "phase": phase,
                    "state_sha256": canonical_sha(state.tolist()),
                }
            )
        solution = solve_ivp(
            rhs_for_phase,
            (0.0, step_s),
            state,
            method=solver_method,
            rtol=rtol,
            max_step=max_step_s,
            events=events,
            atol=atol,
            jac=jacobian,
            first_step=first_step_value,
        )
        if not solution.success:
            raise RuntimeError(solution.message)
        state = np.asarray(solution.y[:, -1], dtype=float).copy()
        if np.any(state[: adapter.n_sites + 3] < -1e-8) or np.any(
            state[closure["indices"]["GAS"]] < -1e-8
        ):
            raise RuntimeError("negative inventory")
        state[: adapter.n_sites + 3] = np.maximum(
            state[: adapter.n_sites + 3], 0.0
        )
        state[closure["indices"]["GAS"]] = np.maximum(
            state[closure["indices"]["GAS"]], 0.0
        )
        integrated_s = float(solution.t[-1])
        cursor["elapsed_s"] = float(cursor["elapsed_s"]) + integrated_s
        cursor["phase_elapsed_s"] = float(cursor["phase_elapsed_s"]) + integrated_s
        diag = _diagnostics(solution, closure, adapter, phase)
        active_diagnostics = (
            diag
            if active_diagnostics is None
            else _merge_diagnostics(active_diagnostics, diag)
        )
        delta_u = closure["inventory"](state) - closure["inventory"](before)
        boundary = (
            state[closure["indices"]["HEAT"]]
            - before[closure["indices"]["HEAT"]]
            - state[closure["indices"]["COOL"]]
            + before[closure["indices"]["COOL"]]
            - state[closure["indices"]["LOSS"]]
            + before[closure["indices"]["LOSS"]]
            + state[closure["indices"]["INLET_H"]]
            - before[closure["indices"]["INLET_H"]]
            - state[closure["indices"]["OUTLET_H"]]
            + before[closure["indices"]["OUTLET_H"]]
        )
        residual = delta_u - boundary
        scale = _segment_scale(
            delta_heat=(
                state[closure["indices"]["HEAT"]] - before[closure["indices"]["HEAT"]]
            ),
            delta_cool=(
                state[closure["indices"]["COOL"]] - before[closure["indices"]["COOL"]]
            ),
            delta_loss=(
                state[closure["indices"]["LOSS"]] - before[closure["indices"]["LOSS"]]
            ),
            delta_h_in=(
                state[closure["indices"]["INLET_H"]]
                - before[closure["indices"]["INLET_H"]]
            ),
            delta_h_out=(
                state[closure["indices"]["OUTLET_H"]]
                - before[closure["indices"]["OUTLET_H"]]
            ),
            delta_u=delta_u,
        )
        relative = abs(residual) / scale
        if relative > 1e-6:
            raise RuntimeError(f"segment energy closure failed: {relative}")
        chunk_records.append(
            {
                "weather_index": int(cursor["weather_index"]),
                "weather_interval_end_utc": hour.interval_end_utc,
                "phase": phase,
                "elapsed_start_s": float(cursor["elapsed_s"]) - integrated_s,
                "elapsed_end_s": float(cursor["elapsed_s"]),
                "duration_s": integrated_s,
                "energy_residual_J": float(residual),
                "energy_transfer_scale_J": float(scale),
                "energy_closure_relative_error": float(relative),
                "event_terminated": bool(
                    solution.t_events and any(len(v) for v in solution.t_events)
                ),
                "diagnostics": diag,
            }
        )
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "integration_step",
                    "attempt_id": attempt_id,
                    "elapsed_s": cursor["elapsed_s"],
                    "phase_elapsed_s": cursor["phase_elapsed_s"],
                    "phase_duration_s": duration,
                    "completed_cycles": int(cursor["completed_cycles"]),
                    "cycle": int(cursor["completed_cycles"]) + 1,
                    "weather_index": cursor["weather_index"],
                    "phase": phase,
                    "state_sha256": canonical_sha(state.tolist()),
                }
            )
        event_terminated = bool(
            solution.t_events and any(len(v) for v in solution.t_events)
        )
        phase_done = event_terminated or cursor["phase_elapsed_s"] >= duration - 1e-12
        at_weather_boundary = abs(
            cursor["elapsed_s"] - next_weather_end
        ) <= 1e-10
        if phase_done:
            completed_cycle = int(cursor["completed_cycles"]) + 1
            finish_phase("physical_event" if event_terminated else "fixed_duration")
            report_phase_completed(phase, completed_cycle)
        if at_weather_boundary:
            cursor["weather_index"] = min(
                int(cursor["weather_index"]) + 1, len(weather_hours) - 1
            )
            hour, config, closure = current_config_and_closure()
            cursor["active_phase_duration_s"] = _phase_duration(config, PHASES[cursor["phase_index"]])
        elif phase_done:
            hour, config, closure = current_config_and_closure()
        checkpoint_hash = checkpoint("active")

    final_inventory_change = closure["inventory"](state) - closure["inventory"](
        initial_state_value
    )
    total_boundary = (
        state[closure["indices"]["HEAT"]]
        - state[closure["indices"]["COOL"]]
        - state[closure["indices"]["LOSS"]]
        + state[closure["indices"]["INLET_H"]]
        - state[closure["indices"]["OUTLET_H"]]
    )
    total_residual = final_inventory_change - total_boundary
    total_scale = _segment_scale(
        delta_heat=state[closure["indices"]["HEAT"]],
        delta_cool=state[closure["indices"]["COOL"]],
        delta_loss=state[closure["indices"]["LOSS"]],
        delta_h_in=state[closure["indices"]["INLET_H"]],
        delta_h_out=state[closure["indices"]["OUTLET_H"]],
        delta_u=final_inventory_change,
    )
    total_energy_relative = abs(total_residual) / total_scale
    net_scale = max(abs(final_inventory_change), abs(total_boundary), 1.0)
    net_relative = abs(total_residual) / net_scale
    conditioner_enabled = (
        closure.get("pump_conditioner_identity") is not None
    )
    if conditioner_enabled:
        physical_vent = (
            state[closure["indices"]["DIRECT_VENT"]]
            + state[closure["indices"]["CONDITIONED_VENT"]]
        )
        physical_product = state[
            closure["indices"]["CONDITIONED_PRODUCT"]
        ]
        liquid_drain = state[closure["indices"]["LIQUID_DRAIN"]]
    else:
        physical_vent = state[closure["indices"]["VENT"]]
        physical_product = state[closure["indices"]["PRODUCT"]]
        liquid_drain = np.zeros(4)
    mass = species_mass_ledger(
        initial_sites=initial_state_value[closure["indices"]["Q"]],
        final_sites=state[closure["indices"]["Q"]],
        initial_water=float(initial_state_value[closure["indices"]["QW"]]),
        final_water=float(state[closure["indices"]["QW"]]),
        initial_n2=float(initial_state_value[closure["indices"]["QN"]]),
        final_n2=float(state[closure["indices"]["QN"]]),
        initial_o2=float(initial_state_value[closure["indices"]["QO"]]),
        final_o2=float(state[closure["indices"]["QO"]]),
        initial_gas=initial_state_value[closure["indices"]["GAS"]],
        final_gas=state[closure["indices"]["GAS"]],
        dry_mass_kg=float(base_config["bed"]["dry_sorbent_mass_kg"]),
        site_species=adapter.site_species,
        inlet=state[closure["indices"]["IN"]],
        vent=physical_vent,
        product=physical_product,
        liquid_drain=liquid_drain,
    )
    mass_relative = float(np.max(mass["relative_error"]))
    checkpoint_hash = checkpoint("complete")
    endpoint_violations = _phase_endpoint_violations(phase_history)
    product = physical_product
    product_total = float(product.sum())
    product_dry = float(product[0] + product[2] + product[3])
    all_diag = [row["diagnostics"] for row in phase_history if row["diagnostics"]]
    if active_diagnostics is not None:
        all_diag.append(active_diagnostics)
    flags = {
        key: any(item["flags"][key] for item in all_diag)
        for key in all_diag[0]["flags"]
    }
    pump_envelope = {
        "assumptions": all_diag[0]["pump_assumptions"],
        "minimum_inlet_temperature_K": min(
            item["minimum_pump_inlet_temperature_K"] for item in all_diag
        ),
        "maximum_inlet_temperature_K": max(
            item["maximum_pump_inlet_temperature_K"] for item in all_diag
        ),
        "minimum_pressure_Pa": min(
            item["minimum_pump_pressure_Pa"] for item in all_diag
        ),
        "maximum_pressure_Pa": max(
            item["maximum_pump_pressure_Pa"] for item in all_diag
        ),
        "maximum_media_temperature_overrun_K": max(
            item["maximum_pump_media_temperature_overrun_K"]
            for item in all_diag
        ),
        "conditioner": all_diag[0].get("conditioner_identity"),
        "minimum_conditioned_pump_inlet_temperature_K": min(
            (
                item["minimum_conditioned_pump_inlet_temperature_K"]
                for item in all_diag
                if item["minimum_conditioned_pump_inlet_temperature_K"]
                is not None
            ),
            default=None,
        ),
        "maximum_conditioned_pump_inlet_temperature_K": max(
            (
                item["maximum_conditioned_pump_inlet_temperature_K"]
                for item in all_diag
                if item["maximum_conditioned_pump_inlet_temperature_K"]
                is not None
            ),
            default=None,
        ),
        "maximum_conditioning_cooling_W": max(
            item["maximum_conditioning_cooling_W"]
            for item in all_diag
        ),
        "maximum_liquid_flow_mol_s": max(
            item["maximum_liquid_flow_mol_s"] for item in all_diag
        ),
        "maximum_liquid_fraction": max(
            item["maximum_liquid_fraction"] for item in all_diag
        ),
        "curve_pressure_regimes": sorted(
            {
                regime
                for item in all_diag
                for regime in item["pump_assumptions"][
                    "pressure_regime_seen"
                ]
            }
        ),
        "media_temperature_excursion": flags[
            "pump_media_temperature_excursion"
        ],
        "curve_upper_domain_exceeded": flags[
            "pump_curve_upper_domain_exceeded"
        ],
        "below_or_at_ultimate_observed": any(
            item["pump_assumptions"]["below_or_at_ultimate_observed"]
            for item in all_diag
        ),
        "conditioned_media_temperature_excursion": flags[
            "conditioned_pump_media_temperature_excursion"
        ],
        "conditioner_liquid_drain_present": flags[
            "conditioner_liquid_drain_present"
        ],
        "hardware_qualified": False,
    }
    cooling_phase_records = [
        row for row in phase_history if row["phase"] == "cooling"
    ]
    summary = {
        "schema": "matched-material-weather-result/2",
        "status": "integration_complete",
        "job_id": job_id,
        "attempt_id": attempt_id,
        "restart_of_attempt_id": restart_of_attempt_id,
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "source_release_sha256": context["source_release_sha256"],
        "source_hashes": context["source_hashes"],
        "source_migration": context.get("source_migration"),
        "material_id": adapter.material_id,
        "control_variant": control_variant,
        "control_identity": checkpoint_control_identity,
        "initial_step_identity": checkpoint_initial_step_identity,
        "initial_step_records": initial_step_records,
        "pump_conditioner_identity": closure.get(
            "pump_conditioner_identity"
        ),
        "adsorption_control": closure["adsorption_control"],
        "pressure_response_tau_s": closure["pressure_response_tau_s"],
        "cooling_policy": {
            "stop_offset_K": policy_config["cycle"].get(
                "cooling_stop_offset_K"
            ),
            "max_duration_s": _phase_duration(
                policy_config, "cooling"
            ),
            "completed_cooling_durations_s": [
                row["duration_s"] for row in cooling_phase_records
            ],
        },
        "p_water_mode": p_water_mode,
        "rtol": rtol,
        "max_step_s": max_step_s,
        "solver_method": solver_method,
        "bounded_jacobian": use_bounded_jacobian,
        "jacobian_audit": jacobian_audit.as_dict() if jacobian_audit else None,
        "milestone_3h": {
            "target_s": milestone_at_s,
            "saved": milestone_saved,
            "path": str(milestone_path) if milestone_saved else None,
            "sha256": milestone_hash,
        },
        "elapsed_s": float(cursor["elapsed_s"]),
        "declared_horizon_s": horizon_s,
        "completed_cycles": int(cursor["completed_cycles"]),
        "cursor": current_cursor(),
        "phase_history": phase_history,
        "chunk_records": chunk_records,
        "initial_water_loading_mol_kg": float(initial_state_value[closure["indices"]["QW"]]),
        "final_water_loading_mol_kg": float(state[closure["indices"]["QW"]]),
        "minimum_water_loading_mol_kg": (min(row["diagnostics"]["minimum_water_loading_mol_kg"] for row in chunk_records)
            if all("minimum_water_loading_mol_kg" in row["diagnostics"] for row in chunk_records) else None),
        "maximum_water_loading_mol_kg": (max(row["diagnostics"]["maximum_water_loading_mol_kg"] for row in chunk_records)
            if all("maximum_water_loading_mol_kg" in row["diagnostics"] for row in chunk_records) else None),
        "minimum_local_relative_humidity": min(row["diagnostics"]["minimum_water_saturation_ratio"] for row in chunk_records),
        "maximum_local_relative_humidity": max(row["diagnostics"]["maximum_water_saturation_ratio"] for row in chunk_records),
        "endpoint_violations": endpoint_violations,
        "temperature_endpoint_violation_count": sum(
            not (
                row["endpoint"].get("temperature_ok", True)
            )
            for row in phase_history
        ),
        "pressure_endpoint_violation_count": sum(
            not (row["endpoint"].get("pressure_ok", True))
            for row in phase_history
        ),
        "physical_endpoint_violation_count": len(endpoint_violations),
        "flags": flags,
        "pump_envelope": pump_envelope,
        "minimum_pressure_Pa": min(
            item["minimum_pressure_Pa"] for item in all_diag
        ),
        "maximum_pressure_Pa": max(
            item["maximum_pressure_Pa"] for item in all_diag
        ),
        "source_assumptions": {
            "CALF_source_transfer_conditional": adapter.material_id.startswith(
                "calf"
            ),
            "fan_helper_uses_core_RH_composition_not_Buck_override": True,
            "measured_T_RH_QFE_passed_directly": True,
            "prepared_Buck_p_water_used_for_bed_feed": (
                p_water_mode == "buck_explicit"
            ),
            "pressure_regulated_adsorption": (
                closure["adsorption_control"] != FIXED_CONTROL
            ),
            "adaptive_cooling_offset_present": (
                base_config["cycle"].get("cooling_stop_offset_K") is not None
            ),
            "no_inlet_cooler_or_condenser_modeled": not conditioner_enabled,
            "pump_inlet_cooler_knockout_modeled": conditioner_enabled,
            "conditioner_hardware_qualified": False,
        },
        "weather_rows": [row.as_dict() for row in weather_hours],
        "weather_rows_sha256": rows_hash,
        "mass_closure_relative_error": mass_relative,
        "mass_residual_mol": mass["residual_mol"].tolist(),
        "energy_residual_J": float(total_residual),
        "energy_transfer_scale_J": float(total_scale),
        "energy_closure_relative_error": float(total_energy_relative),
        "energy_net_boundary_J": float(total_boundary),
        "energy_legacy_net_scale_J": float(net_scale),
        "energy_legacy_net_relative_residual": float(net_relative),
        "energy_legacy_net_closure_pass": bool(net_relative <= 1e-6),
        "energy_normalization": (
            "max(abs(delta_inventory_J), gross_boundary_exchange_J, 1 J)"
        ),
        "conservation_pass": bool(
            mass_relative <= 1e-5 and total_energy_relative <= 1e-6
        ),
        "full_physical_qualification": False,
        "material_winner": None,
        "annual_estimate": None,
        "initial_state": initial_state_value.tolist(),
        "final_state": state.tolist(),
        "state_size": len(state),
        "physical_columns": closure["physical_columns"],
        "ledger_columns": closure["ledger_columns"],
        "product_mol": product.tolist(),
        "inlet_mol": state[closure["indices"]["IN"]].tolist(),
        "vent_mol": state[closure["indices"]["VENT"]].tolist(),
        "delivered_heat_J": float(state[closure["indices"]["HEAT"]]),
        "pump_electric_J": float(state[closure["indices"]["PUMP"]]),
        "fan_electric_J": float(state[closure["indices"]["FAN"]]),
        "cooling_removed_J": float(state[closure["indices"]["COOL"]]),
        "conditioner_cooling_removed_J": (
            float(state[closure["indices"]["CONDITIONER_COOL"]])
            if conditioner_enabled
            else 0.0
        ),
        "total_cooling_removed_J": (
            float(state[closure["indices"]["COOL"]])
            + (
                float(state[closure["indices"]["CONDITIONER_COOL"]])
                if conditioner_enabled
                else 0.0
            )
        ),
        "bed_outlet_h_J": float(
            state[closure["indices"]["OUTLET_H"]]
        ),
        "external_outlet_h_J": float(
            state[closure["indices"]["EXTERNAL_OUTLET_H"]]
            if conditioner_enabled
            else state[closure["indices"]["OUTLET_H"]]
        ),
        "conditioner_energy_boundary_residual_J": (
            float(
                state[closure["indices"]["CONDITIONER_COOL"]]
                - (
                    state[closure["indices"]["OUTLET_H"]]
                    - state[closure["indices"]["EXTERNAL_OUTLET_H"]]
                )
            )
            if conditioner_enabled
            else 0.0
        ),
        "conditioned_pump_inlet_mol": (
            state[closure["indices"]["CONDITIONED_PRODUCT"]].tolist()
            if conditioner_enabled
            else None
        ),
        "conditioned_vent_mol": (
            state[closure["indices"]["CONDITIONED_VENT"]].tolist()
            if conditioner_enabled
            else None
        ),
        "direct_adsorption_vent_mol": (
            state[closure["indices"]["DIRECT_VENT"]].tolist()
            if conditioner_enabled
            else None
        ),
        "liquid_drain_mol": (
            state[closure["indices"]["LIQUID_DRAIN"]].tolist()
            if conditioner_enabled
            else None
        ),
        "raw_preconditioning_product_mol": state[
            closure["indices"]["PRODUCT"]
        ].tolist(),
        "raw_preconditioning_vent_mol": state[
            closure["indices"]["VENT"]
        ].tolist(),
        "electricity_J": float(
            state[closure["indices"]["HEAT"]]
            + state[closure["indices"]["PUMP"]]
            + state[closure["indices"]["FAN"]]
            + (
                state[closure["indices"]["COOL"]]
                + (
                    state[closure["indices"]["CONDITIONER_COOL"]]
                    if conditioner_enabled
                    else 0.0
                )
            )
            / float(base_config["utilities"]["wall_cooling"]["cop"])
        ),
        "gross_collected_co2_mol": float(product[0]),
        "gross_collected_co2_kg": float(product[0] * 0.0440095),
        "gross_collected_co2_kg_day": float(
            product[0] * 0.0440095 * 86400.0 / max(float(cursor["elapsed_s"]), 1.0)
        ),
        "dry_purity": (
            float(product[0] / product_dry) if product_dry > 0.0 else None
        ),
        "wet_basis_co2_fraction": (
            float(product[0] / product_total) if product_total > 0.0 else None
        ),
    }
    return {
        "status": "complete",
        "summary": summary,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
    }
