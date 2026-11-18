"""Tabulated pump profiles: validated, preloaded vendor curves (no per-evaluation file I/O).

Two flow bases are possible when a datasheet plots volumetric throughput against inlet
pressure, and they differ by `P_ref / P`:

* **reference-condition basis (used here).** The datasheet footnote states that the flow
  is determined at 20 C and 1013 mbar abs, so the plotted value is the throughput already
  referred to those conditions. The molar throughput is then
  `n_dot = Q_ref(P) * P_ref / (R * T)`; the bench volumetric speed at the pump inlet is
  `S(P) = Q_ref(P) * P_ref / P`.
* **inlet-volume basis (rejected).** Reading the plotted volume as the volume at the
  current inlet pressure gives `n_dot = Q_ref(P) * P / (R * T)`, which collapses towards
  zero as the inlet pressure falls.

Cross-check on the same datasheet page: the 20 L vessel pump-down curve reaches 200 mbar in
about 2.28 min. The reference basis predicts 2.18 min (-4%, within the graphical resolution of
the chart and consistent with the datasheet's +-10% flow rating at atmospheric pressure, which
is not an error band on the whole curve); the inlet-volume basis predicts 5.98 min (2.6x too
slow). This is graphical consistency against one published curve, not a validation set. The
reference basis is therefore used, and the transfer of the curve to other gas temperatures or
compositions is labelled as an assumption rather than a measurement.

Profiles are vendor datasheet readings, not measured confidence intervals. Nothing here
validates wet, hot or degraded performance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from .common import R, ExpectedScenarioError

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "equipment/pump_profiles.json"
DOMAIN_TOLERANCE = 1e-6  # relative slack on the top of the tabulated pressure map
IDEAL_PARALLEL_BANK_CONTRACT = "ideal_parallel_unit_curve"

_CACHE: dict[tuple[str, str], "PumpProfile"] = {}


@dataclass(frozen=True)
class PumpProfile:
    """A validated tabulated pump: reference flow map, limits and electrical basis."""

    name: str
    source: dict
    pressure_Pa: np.ndarray
    reference_flow_m3_s: np.ndarray
    reference_pressure_Pa: float
    reference_temperature_K: float
    ultimate_pressure_Pa: float
    flow_basis: str
    temperature_transfer_basis: str
    electrical_basis: str
    electrical_input_W: float | None = None
    electrical_pressure_Pa: np.ndarray | None = None
    electrical_power_W: np.ndarray | None = None
    media_temperature_limit_C: float | None = None
    rated_input_W: float | None = None
    annotations: dict = field(default_factory=dict)
    pumpdown_check: dict = field(default_factory=dict)
    registry_path: str = ""

    @property
    def pressure_range_Pa(self) -> tuple[float, float]:
        return float(self.pressure_Pa[0]), float(self.pressure_Pa[-1])

    def as_export(self) -> dict:
        """JSON-safe description for exports and the input digest."""
        return {
            "name": self.name, "source": self.source, "registry_path": self.registry_path,
            "pressure_domain_Pa": [float(v) for v in self.pressure_range_Pa],
            "pressure_map_Pa": [float(v) for v in self.pressure_Pa],
            "reference_flow_m3_s": [float(v) for v in self.reference_flow_m3_s],
            "reference_pressure_Pa": self.reference_pressure_Pa,
            "reference_temperature_K": self.reference_temperature_K,
            "ultimate_pressure_Pa": self.ultimate_pressure_Pa,
            "flow_basis": self.flow_basis, "temperature_transfer_basis": self.temperature_transfer_basis,
            "electrical_basis": self.electrical_basis, "electrical_input_W": self.electrical_input_W,
            "electrical_map": (None if self.electrical_pressure_Pa is None else
                               {"pressure_Pa": [float(v) for v in self.electrical_pressure_Pa],
                                "power_W": [float(v) for v in self.electrical_power_W]}),
            "media_temperature_limit_C": self.media_temperature_limit_C,
            "rated_input_W": self.rated_input_W, "annotations": self.annotations,
            "pumpdown_check": self.pumpdown_check,
        }

    def sha256(self) -> str:
        payload = json.dumps(self.as_export(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def curve_domain_exceeded(self, pressure_Pa: float) -> bool:
        """True when the requested inlet pressure is above the tabulated operating map."""
        return float(pressure_Pa) > self.pressure_range_Pa[1] * (1.0 + DOMAIN_TOLERANCE)

    def reference_flow_m3_s_at(self, pressure_Pa: float) -> float:
        """Reference-condition volumetric throughput; zero at or below the ultimate pressure.

        Above the tabulated map the top-of-map value is held (no invented extrapolation); the
        caller is responsible for flagging that clamp through `curve_domain_exceeded`.
        """
        pressure = float(pressure_Pa)
        if pressure <= self.ultimate_pressure_Pa:
            return 0.0
        if self.curve_domain_exceeded(pressure):
            return float(self.reference_flow_m3_s[-1])
        return float(np.interp(pressure, self.pressure_Pa, self.reference_flow_m3_s))

    def volumetric_speed_m3_s(self, pressure_Pa: float, temperature_K: float) -> float:
        """Bench volumetric speed at the pump inlet, `S(P) = Q_ref(P) * P_ref / P`."""
        flow = self.reference_flow_m3_s_at(pressure_Pa)
        return flow * self.reference_pressure_Pa / max(float(pressure_Pa), 1e-20) if flow > 0.0 else 0.0

    def molar_flow_mol_s(self, pressure_Pa: float, temperature_K: float) -> float:
        """Molar throughput `Q_ref(P) * P_ref / (R * T)` using the local gas temperature."""
        return self.reference_flow_m3_s_at(pressure_Pa) * self.reference_pressure_Pa / (R * float(temperature_K))

    def electric_power_W(self, pressure_Pa: float) -> float:
        """Running electrical input. Tabulated map if supplied, else the constant-input assumption."""
        if self.electrical_pressure_Pa is not None:
            pressure = min(max(float(pressure_Pa), float(self.electrical_pressure_Pa[0])),
                           float(self.electrical_pressure_Pa[-1]))
            return float(np.interp(pressure, self.electrical_pressure_Pa, self.electrical_power_W))
        return float(self.electrical_input_W or 0.0)

    def media_temperature_overrun_K(self, temperature_K: float) -> float:
        """How far the pumped gas exceeds the datasheet media limit, in K (0 when inside)."""
        if self.media_temperature_limit_C is None:
            return 0.0
        return max(0.0, float(temperature_K) - (self.media_temperature_limit_C + 273.15))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _build_profile(name: str, data: dict, registry_path: str) -> PumpProfile:
    """Validate one registry entry and return an immutable profile."""
    _require(isinstance(data, dict), f"Pump profile {name} must be an object.")
    for key in ("source", "reference", "curve", "flow_model", "electrical"):
        _require(key in data, f"Pump profile {name} is missing the required section {key}.")
    reference = data["reference"]
    for key in ("pressure_Pa", "temperature_K"):
        value = reference.get(key)
        _require(isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value) and value > 0.0,
                 f"Pump profile {name} reference.{key} must be a finite positive number.")
    pressure_hPa = np.asarray(data["curve"]["pressure_hPa"], dtype=float)
    flow_l_min = np.asarray(data["curve"]["reference_flow_l_min"], dtype=float)
    _require(pressure_hPa.ndim == 1 and flow_l_min.ndim == 1, f"Pump profile {name} curve arrays must be 1-D.")
    _require(pressure_hPa.size == flow_l_min.size and pressure_hPa.size >= 3,
             f"Pump profile {name} curve arrays must be equal-length and hold at least three anchors.")
    _require(bool(np.all(np.isfinite(pressure_hPa))) and bool(np.all(np.isfinite(flow_l_min))),
             f"Pump profile {name} curve contains non-finite values.")
    _require(bool(np.all(np.diff(pressure_hPa) > 0.0)), f"Pump profile {name} pressures must be strictly increasing.")
    _require(bool(np.all(flow_l_min >= 0.0)), f"Pump profile {name} flows must be nonnegative.")
    pressure_Pa = pressure_hPa * 100.0  # chart axis is hPa / mbar
    reference_pressure_Pa = float(reference["pressure_Pa"])
    reference_temperature_K = float(reference["temperature_K"])
    ultimate_Pa = float(data["ultimate_pressure_Pa"])
    _require(np.isfinite(ultimate_Pa) and ultimate_Pa > 0.0, f"Pump profile {name} ultimate pressure must be positive.")
    _require(ultimate_Pa <= pressure_Pa[0] * (1.0 + DOMAIN_TOLERANCE),
             f"Pump profile {name} ultimate pressure {ultimate_Pa:g} Pa is above the start of its own curve "
             f"({pressure_Pa[0]:g} Pa).")
    _require(pressure_Pa[-1] >= reference_pressure_Pa * (1.0 - 1e-3),
             f"Pump profile {name} curve must reach the reference pressure {reference_pressure_Pa:g} Pa.")
    for key in ("media_temperature_limit_C", "rated_input_W"):
        value = data.get(key)
        if value is not None:
            _require(isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value) and value > 0.0,
                     f"Pump profile {name} {key} must be a finite positive number when present.")
    electrical = data["electrical"]
    electrical_map = None
    if electrical.get("pressure_Pa") is not None:
        map_pressure = np.asarray(electrical["pressure_Pa"], dtype=float)
        map_power = np.asarray(electrical["power_W"], dtype=float)
        _require(map_pressure.ndim == 1 and map_power.ndim == 1 and map_pressure.size == map_power.size
                 and map_pressure.size >= 2,
                 f"Pump profile {name} electrical map arrays must be equal-length 1-D arrays.")
        _require(bool(np.all(np.isfinite(map_pressure))) and bool(np.all(map_pressure > 0.0)),
                 f"Pump profile {name} electrical map pressures must be finite and positive.")
        _require(bool(np.all(np.diff(map_pressure) > 0.0)), f"Pump profile {name} electrical map pressures must increase.")
        _require(bool(np.all(np.isfinite(map_power))) and bool(np.all(map_power >= 0.0)),
                 f"Pump profile {name} electrical map must be finite and nonnegative.")
        electrical_map = (map_pressure, map_power)
    else:
        # Without a tabulated power map the constant running input is the whole electrical model,
        # so a missing or non-physical value must fail instead of silently becoming zero watts.
        input_W = electrical.get("input_W")
        _require(isinstance(input_W, (int, float)) and not isinstance(input_W, bool)
                 and np.isfinite(input_W) and input_W > 0.0,
                 f"Pump profile {name} electrical.input_W must be a finite positive number when no "
                 "electrical.pressure_Pa/power_W map is supplied.")
    return PumpProfile(
        name=name, source=data["source"], pressure_Pa=pressure_Pa,
        reference_flow_m3_s=flow_l_min / 60000.0,  # l/min -> m3/s at reference conditions
        reference_pressure_Pa=reference_pressure_Pa, reference_temperature_K=reference_temperature_K,
        ultimate_pressure_Pa=ultimate_Pa, flow_basis=data["flow_model"]["basis"],
        temperature_transfer_basis=data["flow_model"]["temperature_transfer"],
        electrical_basis=electrical["basis"], electrical_input_W=electrical.get("input_W"),
        electrical_pressure_Pa=None if electrical_map is None else electrical_map[0],
        electrical_power_W=None if electrical_map is None else electrical_map[1],
        media_temperature_limit_C=data.get("media_temperature_limit_C"),
        rated_input_W=data.get("rated_input_W"), annotations=data.get("annotations", {}),
        pumpdown_check=data.get("pumpdown_check", {}), registry_path=registry_path)


def load_pump_profile(name: str, path: str | Path | None = None) -> PumpProfile:
    """Load and cache one profile; the registry file is read once per process."""
    registry = Path(path) if path is not None else DEFAULT_REGISTRY
    if not registry.is_absolute():
        registry = ROOT / registry
    key = (str(registry.resolve()), name)
    if key in _CACHE:
        return _CACHE[key]
    if not registry.is_file():
        raise ValueError(f"Pump profile registry not found: {registry}")
    try:
        payload = json.loads(registry.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Pump profile registry {registry.name} is not valid JSON: {error}") from error
    profiles = payload.get("profiles", {})
    if name not in profiles:
        raise ValueError(f"Unknown pump profile {name!r}; registry defines {sorted(profiles)}.")
    try:
        registry_label = str(registry.relative_to(ROOT))
    except ValueError:  # a registry outside the project root, e.g. a test copy
        registry_label = str(registry)
    profile = _build_profile(name, profiles[name], registry_label)
    _CACHE[key] = profile
    return profile


def profile_for(config: dict) -> PumpProfile | None:
    """Resolve the profile named by a config, or None for the generic illustrative pump."""
    name = config.get("pump", {}).get("profile")
    if name in (None, "", "generic"):
        return None
    return load_pump_profile(name, config["pump"].get("profile_file") or None)


def bank_count(config: Mapping) -> int:
    """Validate the opt-in identical-pump bank count.

    Absent or 1 is exact single-unit parity. Counts above one require the
    explicit ideal-parallel contract; speed maps are not implemented.
    """
    pump = config.get("pump", {})
    value = pump.get("bank_count", 1)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("pump.bank_count must be a positive integer; bool/float/NaN are refused.")
    for key in ("speed_fraction", "speed_scale", "speed_ratio", "speed_rpm"):
        if key in pump:
            raise ValueError(
                f"pump.{key} is unsupported: no vendor speed-map evidence exists; "
                "refusing affinity-law extrapolation or silent speed scaling."
            )
    if value > 1 and pump.get("bank_contract") != IDEAL_PARALLEL_BANK_CONTRACT:
        raise ValueError(
            "pump.bank_count > 1 requires pump.bank_contract = "
            f"{IDEAL_PARALLEL_BANK_CONTRACT!r}; shared/spatial consumers are not certified."
        )
    return value


def preload(config: dict) -> dict | None:
    """Resolve and describe the configured profile once, for validation and export."""
    profile = profile_for(config)
    return None if profile is None else profile.as_export()
