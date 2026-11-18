"""Optional upstream drying with explicit flow and resource accounting.

The dryer conserves the dry carrier and oxygen, removes water to a declared
target and records any declared CO2 loss. upstream_raw and downstream_conditioned
distinguish the configured volume-flow basis. Vessel leaks bypass the dryer.

Water removed times each declared specific duty gives regeneration heat. Standby
heat, auxiliary electricity and dryer pressure loss are separate terms. Missing
coefficients remain unknown; zero water removal makes proportional duties zero.
The outlet humidity and duties are scenario inputs, not hardware predictions.

Raw composition is computed without calling the dryer-aware feed helper, avoiding
recursive configuration resolution. Explicit drying and the specific upstream
conditioning-energy option are mutually exclusive."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .common import MW, R, water_saturation_pressure_Pa

UPSTREAM_RAW = "upstream_raw"
DOWNSTREAM_CONDITIONED = "downstream_conditioned"
FLOW_BASES = (UPSTREAM_RAW, DOWNSTREAM_CONDITIONED)
WATER = 1
DRY_SPECIES = (0, 2, 3)          # CO2, N2, O2 on a dry basis
LEAK_BYPASS_BASIS = ("the vessel leak is unconditioned ambient air: it bypasses the dryer completely and "
                     "enters at the raw ambient composition, so it is never processed or counted in the "
                     "conditioned feed")
DRYER_SECTION_KEYS = frozenset({"target_relative_humidity", "flow_basis", "co2_loss_fraction",
                                "specific_water_desorption_heat_J_per_kg",
                                "specific_drybed_sensible_heat_J_per_kg_water",
                                "regeneration_efficiency", "auxiliary_electric_W",
                                "auxiliary_on_fraction_of_cycle", "added_pressure_drop_Pa",
                                "standby_regeneration_heat_W", "water_working_capacity_kg"})
#: Optional keys, in report order, with the analyst convention used when they are omitted.
OPTIONAL_KEYS = ("co2_loss_fraction", "specific_water_desorption_heat_J_per_kg",
                 "specific_drybed_sensible_heat_J_per_kg_water", "regeneration_efficiency",
                 "auxiliary_electric_W", "auxiliary_on_fraction_of_cycle", "added_pressure_drop_Pa",
                 "standby_regeneration_heat_W", "water_working_capacity_kg")
DEFAULT_CONVENTION = {"co2_loss_fraction": 0.0, "standby_regeneration_heat_W": 0.0}
DEFAULT_CONVENTION_NOTE = {
    "co2_loss_fraction": "zero CO2 loss is the labelled analyst default; the raw and column capture "
                         "denominators are still reported separately",
    "standby_regeneration_heat_W": "no standby regeneration heat: the dryer is not assumed to stay hot while idle",
}
DEFAULT_BASIS = ("only co2_loss_fraction and standby_regeneration_heat_W carry an analyst default convention; every "
                 "other omitted coefficient is reported as unknown, never as zero, and a bounded scenario can "
                 "declare an explicit 0.0 instead")
CONSERVATION_TOLERANCE = 1e-12
CAPACITY_CONSEQUENCE = ("the declared dryer water working capacity cannot cover the water removed from the "
                        "integrated feed in one cycle, so the target conditioned humidity is conditional: the "
                        "conditioned composition is a requested boundary, not a guaranteed hardware output")
_BUILDING = False


def _finite(value, name: str, *, minimum: float | None = None, maximum: float | None = None,
            allow_none: bool = False) -> float | None:
    """Validated finite number; booleans are rejected because they are not measurements."""
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}; got {value}.")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}; got {value}.")
    return float(value)


def _reject_unknown(section: dict, allowed: frozenset, name: str) -> None:
    """Unknown keys are refused so a declared resource can never be silently dropped."""
    unknown = sorted(set(section) - set(allowed))
    if unknown:
        raise ValueError(f"{name} has unsupported key(s) {unknown}; supported keys are {sorted(allowed)}.")


def raw_ambient_composition(config: dict) -> np.ndarray:
    """Unconditioned ambient composition, identical to ``common.feed_composition(config, conditioned=False)``.

    Kept local so the dryer never depends on the shared helper at runtime: the only allowed edge is
    ``feed_composition(conditioned=True) -> build_dryer``. A test pins this function to the shared
    implementation so the two cannot silently diverge.
    """
    ambient = config["ambient"]
    temperature_K = ambient["temperature_C"] + 273.15
    water = ambient["relative_humidity"] * water_saturation_pressure_Pa(temperature_K) / ambient["pressure_Pa"]
    co2_dry = ambient["co2_ppm_dry"] * 1e-6
    dry = np.array([co2_dry, 0.0, (1 - co2_dry) * 0.7905, (1 - co2_dry) * 0.2095])
    return dry * (1 - water) + np.array([0, water, 0, 0])


@dataclass(frozen=True)
class DryerSpec:
    """Validated declaration of the upstream dryer.

    ``None`` means "not declared, therefore unknown" for a coefficient that scales a real load.
    ``provided_keys`` records exactly which optional keys the user wrote, so the resolved spec can
    label every default instead of claiming a silent zero.
    """

    target_relative_humidity: float
    flow_basis: str
    co2_loss_fraction: float
    specific_water_desorption_heat_J_per_kg: float | None
    specific_drybed_sensible_heat_J_per_kg_water: float | None
    regeneration_efficiency: float | None
    auxiliary_electric_W: float | None
    auxiliary_on_fraction_of_cycle: float
    added_pressure_drop_Pa: float | None
    standby_regeneration_heat_W: float
    water_working_capacity_kg: float | None
    provided_keys: tuple[str, ...] = ()

    def declaration_status(self) -> dict:
        """Per optional key: declared by the user, defaulted by convention, or left unknown."""
        status = {}
        for key in OPTIONAL_KEYS:
            if key in self.provided_keys:
                status[key] = {"status": "declared"}
            elif key in DEFAULT_CONVENTION:
                status[key] = {"status": "defaulted_analyst_convention", "value": DEFAULT_CONVENTION[key],
                               "note": DEFAULT_CONVENTION_NOTE[key]}
            else:
                status[key] = {"status": "absent_unknown",
                               "note": "omitted coefficients stay unknown: the resource they scale is reported as "
                                       "None and the ledger is incomplete"}
        status["_basis"] = DEFAULT_BASIS
        return status

    def as_dict(self) -> dict:
        return {"target_relative_humidity": self.target_relative_humidity, "flow_basis": self.flow_basis,
                "co2_loss_fraction": self.co2_loss_fraction,
                "specific_water_desorption_heat_J_per_kg": self.specific_water_desorption_heat_J_per_kg,
                "specific_drybed_sensible_heat_J_per_kg_water": self.specific_drybed_sensible_heat_J_per_kg_water,
                "regeneration_efficiency": self.regeneration_efficiency,
                "auxiliary_electric_W": self.auxiliary_electric_W,
                "auxiliary_on_fraction_of_cycle": self.auxiliary_on_fraction_of_cycle,
                "added_pressure_drop_Pa": self.added_pressure_drop_Pa,
                "standby_regeneration_heat_W": self.standby_regeneration_heat_W,
                "water_working_capacity_kg": self.water_working_capacity_kg,
                "provided_keys": list(self.provided_keys),
                "declaration_status": self.declaration_status(),
                "resource_basis": ("declared specific regeneration duties per kilogram of water removed, divided "
                                   "by a declared efficiency; not a measured dryer and not a desiccant performance "
                                   "curve"),
                "co2_loss_basis": ("declared fraction of the inlet CO2 that the conditioning step loses; the raw "
                                   "feed and the column feed are reported as separate capture denominators"),
                "standby_basis": ("standby regeneration heat is a separate declared power term integrated over the "
                                  "cycle duration; it is never assumed and appears as its own ledger entry"),
                "water_capacity_basis": None if self.water_working_capacity_kg is None else CAPACITY_CONSEQUENCE}


@dataclass(frozen=True)
class Dryer:
    """Resolved dryer: conditioned composition, flow ratio, per-mol ledger and cycle resources."""

    spec: DryerSpec
    ambient_temperature_K: float
    ambient_pressure_Pa: float
    ambient_relative_humidity: float
    y_raw: np.ndarray
    y_conditioned: np.ndarray
    raw_per_conditioned: float
    water_removed_per_mol_conditioned: float
    co2_raw_per_mol_conditioned: float
    co2_lost_per_mol_conditioned: float
    dry_carrier_per_mol_conditioned: float
    nominal_raw_mol_s: float
    nominal_conditioned_mol_s: float
    fan_efficiency: float
    dry_air_fractions: tuple[float, float, float]
    conservation_residual: dict = field(default_factory=dict)

    # -- composition and flow ---------------------------------------------------------------
    @property
    def water_removal_active(self) -> bool:
        return self.water_removed_per_mol_conditioned > 0.0

    def raw_molar_flow(self, conditioned_flow_mol_s: float) -> float:
        """Raw moles through the dryer for any conditioned molar flow the cycle uses."""
        return float(conditioned_flow_mol_s) * self.raw_per_conditioned

    def conditioned_molar_flow(self, raw_flow_mol_s: float) -> float:
        """Conditioned moles delivered to the column for a raw molar flow through the dryer."""
        return float(raw_flow_mol_s) / self.raw_per_conditioned

    def species_ledger_mol(self, conditioned_flow_mol_s: float) -> dict:
        """Species in/out per second (or integrated amount) for a given conditioned flow."""
        flow = float(conditioned_flow_mol_s)
        raw = self.raw_molar_flow(flow)
        raw_in = raw * self.y_raw
        lost = np.zeros(4)
        lost[0] = raw_in[0] * self.spec.co2_loss_fraction
        removed = np.zeros(4)
        removed[WATER] = raw_in[WATER] - flow * self.y_conditioned[WATER]
        out = flow * self.y_conditioned
        return {"conditioned_flow_mol_s": flow, "raw_flow_mol_s": raw,
                "raw_in_mol_s": raw_in.tolist(), "conditioned_out_mol_s": out.tolist(),
                "water_removed_mol_s": float(removed[WATER]), "co2_lost_mol_s": float(lost[0]),
                "dry_carrier_in_mol_s": float(raw_in[2] + raw_in[3]),
                "dry_carrier_out_mol_s": float(out[2] + out[3]),
                "co2_in_raw_mol_s": float(raw_in[0])}

    def added_fan_power_W(self, conditioned_flow_mol_s: float) -> float | None:
        """Dryer fan electricity at ambient T and P, or None when the added loss was not declared."""
        if self.spec.added_pressure_drop_Pa is None:
            return None
        raw = self.raw_molar_flow(conditioned_flow_mol_s)
        volume_flow_m3_s = raw * R * self.ambient_temperature_K / self.ambient_pressure_Pa
        return volume_flow_m3_s * self.spec.added_pressure_drop_Pa / self.fan_efficiency

    def co2_balance(self, *, conditioned_feed_mol: float, co2_collected_mol: float = 0.0,
                    raw_leak_mol: float = 0.0) -> dict:
        """CO2 capture denominators on the upstream raw basis and the terminal column-feed basis.

        The upstream denominator contains the CO2 the dryer vented, so a conditioning loss can never
        silently inflate the reported recovery. The leak is raw ambient air and is included in both
        denominators because it was withdrawn from the atmosphere and it does enter the bed.
        """
        feed = _finite(conditioned_feed_mol, "conditioned_feed_mol", minimum=0.0)
        collected = _finite(co2_collected_mol, "co2_collected_mol", minimum=0.0)
        leak = _finite(raw_leak_mol, "raw_leak_mol", minimum=0.0)
        raw_feed_mol = self.raw_molar_flow(feed)
        co2_raw_feed = self.co2_raw_per_mol_conditioned * feed
        co2_column_feed = float(self.y_conditioned[0]) * feed
        co2_lost = self.co2_lost_per_mol_conditioned * feed
        co2_leak = float(self.y_raw[0]) * leak
        upstream = co2_raw_feed + co2_leak
        terminal = co2_column_feed + co2_leak
        residual = upstream - (co2_column_feed + co2_lost + co2_leak)
        scale = max(abs(upstream), 1e-30)
        return {"raw_feed_mol": raw_feed_mol, "conditioned_feed_mol": feed, "raw_leak_mol": leak,
                "co2_upstream_raw_mol": float(upstream), "co2_terminal_column_feed_mol": float(terminal),
                "co2_column_feed_from_dryer_mol": float(co2_column_feed),
                "co2_lost_in_conditioning_mol": float(co2_lost), "co2_leak_mol": float(co2_leak),
                "co2_collected_mol": collected,
                "recovery_upstream_raw_basis": None if upstream <= 0 else collected / upstream,
                "recovery_terminal_basis": None if terminal <= 0 else collected / terminal,
                "conditioning_co2_loss_fraction": self.spec.co2_loss_fraction,
                "closure_residual_mol": float(residual),
                "closure_residual_relative": float(abs(residual) / scale),
                "denominator_basis": ("recovery_upstream_raw_basis counts every CO2 withdrawn from the "
                                      "atmosphere that reaches the train (dryer inlet plus raw vessel leak); "
                                      "recovery_terminal_basis counts only what reaches the bed after "
                                      "conditioning")}

    # -- cycle resources ---------------------------------------------------------------------
    def resources(self, *, conditioned_feed_mol: float, feed_duration_s: float,
                  cycle_duration_s: float) -> dict:
        """Dryer resource ledger for one cycle from the feed and durations that were actually integrated.

        ``None`` means unknown. A zero appears only where the quantity really is zero: an inactive dryer
        has zero water-proportional regeneration duty by multiplication, and a declared zero coefficient is
        a deliberate bounded-scenario choice.
        """
        feed = _finite(conditioned_feed_mol, "conditioned_feed_mol", minimum=0.0)
        feed_duration = _finite(feed_duration_s, "feed_duration_s", minimum=0.0)
        cycle_duration = _finite(cycle_duration_s, "cycle_duration_s", minimum=0.0)
        raw_feed_mol = self.raw_molar_flow(feed)
        water_mol = self.water_removed_per_mol_conditioned * feed
        water_kg = water_mol * float(MW[WATER])
        co2_lost_mol = self.co2_lost_per_mol_conditioned * feed
        co2_lost_kg = co2_lost_mol * float(MW[0])
        raw_volume_m3 = raw_feed_mol * R * self.ambient_temperature_K / self.ambient_pressure_Pa
        active = self.water_removal_active and water_mol > 0.0
        mode = "active" if active else "inactive"
        unknown: list[dict] = []

        # Electricity side: its own declaration, never inferred from the thermal side.
        drop_Pa = self.spec.added_pressure_drop_Pa
        if drop_Pa is None:
            fan_electric_J: float | None = None
            unknown.append({"resource": "fan_electric_J", "missing": "added_pressure_drop_Pa",
                            "note": "the added dryer pressure loss is not declared, so the fan work and the "
                                    "complete-plant electric boundary are unknown; declare 0.0 to state that a "
                                    "bounded comparison deliberately excludes it"})
        else:
            fan_electric_J = raw_volume_m3 * drop_Pa / self.fan_efficiency
        auxiliary_W = self.spec.auxiliary_electric_W
        if auxiliary_W is None:
            auxiliary_electric_J: float | None = None
            unknown.append({"resource": "auxiliary_electric_J", "missing": "auxiliary_electric_W",
                            "note": "no dryer auxiliary electricity is declared; declare 0.0 to state that the "
                                    "bounded comparison has no auxiliary load"})
        else:
            auxiliary_electric_J = auxiliary_W * self.spec.auxiliary_on_fraction_of_cycle * cycle_duration
        dryer_electric_J = (None if fan_electric_J is None or auxiliary_electric_J is None
                            else float(fan_electric_J + auxiliary_electric_J))

        # Thermal side: water-proportional terms are zero by multiplication when the dryer is inactive.
        desorption: float | None
        sensible: float | None
        if active:
            if self.spec.specific_water_desorption_heat_J_per_kg is None:
                desorption = None
                unknown.append({"resource": "desorption_heat_J",
                                "missing": "specific_water_desorption_heat_J_per_kg"})
            else:
                desorption = water_kg * self.spec.specific_water_desorption_heat_J_per_kg
            if self.spec.specific_drybed_sensible_heat_J_per_kg_water is None:
                sensible = None
                unknown.append({"resource": "sensible_regeneration_heat_J",
                                "missing": "specific_drybed_sensible_heat_J_per_kg_water"})
            else:
                sensible = water_kg * self.spec.specific_drybed_sensible_heat_J_per_kg_water
        else:
            desorption = 0.0
            sensible = 0.0
        standby = self.spec.standby_regeneration_heat_W * cycle_duration
        parts = [value for value in (desorption, sensible, standby) if value is not None]
        delivered = float(sum(parts)) if len(parts) == 3 else None
        efficiency = self.spec.regeneration_efficiency
        regeneration_thermal: float | None
        if delivered is None:
            regeneration_thermal = None
            missing = "-and-".join(name for name, value in
                                   (("specific_water_desorption_heat_J_per_kg", desorption),
                                    ("specific_drybed_sensible_heat_J_per_kg_water", sensible))
                                   if value is None)
            unknown.append({"resource": "regeneration_thermal_input_J", "missing": missing,
                            "note": "the regeneration input cannot be formed while a delivered-heat component is "
                                    "unknown"})
        elif delivered == 0.0:
            regeneration_thermal = 0.0        # zero load needs no efficiency to stay zero
        elif efficiency is None:
            regeneration_thermal = None
            unknown.append({"resource": "regeneration_thermal_input_J", "missing": "regeneration_efficiency"})
        else:
            regeneration_thermal = delivered / efficiency

        capacity_kg = self.spec.water_working_capacity_kg
        if capacity_kg is None:
            capacity = {"declared_kg": None, "removed_kg": water_kg, "shortfall_kg": None,
                        "sufficient": None, "consequence": None,
                        "basis": "no dryer water working capacity was declared, so the target humidity is a "
                                 "requested boundary with no hardware limit attached to it"}
        else:
            shortfall = max(water_kg - capacity_kg, 0.0)
            capacity = {"declared_kg": capacity_kg, "removed_kg": water_kg, "shortfall_kg": shortfall,
                        "sufficient": bool(shortfall <= 0.0), "consequence": CAPACITY_CONSEQUENCE,
                        "basis": "declared finite water working capacity compared with the water removed by the "
                                 "integrated feed; a shortfall is a flag, not a silently guaranteed outlet humidity"}
        regeneration_basis = (
            "the integrated feed carried no water, so the water-proportional desorption and drybed sensible "
            "duties are zero by multiplication and no specific duty or efficiency needs to be declared"
            if not active else
            "desorption and drybed sensible duties follow from the water actually removed and the declared "
            "specific duties; a missing specific duty is unknown, not zero")
        return {"conditioned_feed_mol": feed, "raw_feed_mol": raw_feed_mol,
                "mean_conditioned_flow_mol_s": (None if feed_duration <= 0.0 else feed / feed_duration),
                "feed_duration_s": feed_duration, "cycle_duration_s": cycle_duration,
                "mode": mode, "water_removal_active": self.water_removal_active,
                "water_removed_mol": water_mol, "water_removed_kg": water_kg,
                "co2_lost_mol": co2_lost_mol, "co2_lost_kg": co2_lost_kg,
                "desorption_heat_J": None if desorption is None else float(desorption),
                "sensible_regeneration_heat_J": None if sensible is None else float(sensible),
                "standby_regeneration_heat_J": float(standby),
                "standby_regeneration_heat_W": self.spec.standby_regeneration_heat_W,
                "standby_declared_by_user": "standby_regeneration_heat_W" in self.spec.provided_keys,
                "delivered_regeneration_heat_J": delivered,
                "regeneration_thermal_input_J": None if regeneration_thermal is None else float(regeneration_thermal),
                "regeneration_efficiency": efficiency,
                "fan_electric_J": None if fan_electric_J is None else float(fan_electric_J),
                "auxiliary_electric_J": None if auxiliary_electric_J is None else float(auxiliary_electric_J),
                "dryer_electric_J": dryer_electric_J,
                "added_pressure_drop_Pa": drop_Pa,
                "water_capacity": capacity,
                "complete": len(unknown) == 0,
                "unknown_resources": unknown,
                "regeneration_basis": regeneration_basis,
                "standby_basis": ("standby regeneration heat is quoted as power and integrated over the cycle "
                                  "duration; it is a separate declared term reported on its own line and is never "
                                  "folded into the water-proportional duty"),
                "carrier_separation": ("regeneration thermal input and dryer electricity are separate carriers and "
                                       "are never added together or against each other"),
                "basis": ("resources scale with the water the integrated feed actually carried, not with a nominal "
                          "adsorption flow; the fan work is the raw volume through the dryer times the declared "
                          "added pressure drop divided by the airflow fan efficiency")}

    def as_dict(self, *, include_composition: bool = True) -> dict:
        out = {"spec": self.spec.as_dict(),
               "ambient_temperature_K": self.ambient_temperature_K,
               "ambient_pressure_Pa": self.ambient_pressure_Pa,
               "ambient_relative_humidity": self.ambient_relative_humidity,
               "raw_per_conditioned": self.raw_per_conditioned,
               "water_removed_per_mol_conditioned": self.water_removed_per_mol_conditioned,
               "co2_raw_per_mol_conditioned": self.co2_raw_per_mol_conditioned,
               "co2_lost_per_mol_conditioned": self.co2_lost_per_mol_conditioned,
               "dry_carrier_per_mol_conditioned": self.dry_carrier_per_mol_conditioned,
               "nominal_raw_mol_s": self.nominal_raw_mol_s,
               "nominal_conditioned_mol_s": self.nominal_conditioned_mol_s,
               "water_removal_active": self.water_removal_active,
               "conservation_residual": self.conservation_residual,
               "leak_basis": LEAK_BYPASS_BASIS,
               "default_convention": DEFAULT_BASIS,
               "integration_direction": ("feed_composition(conditioned=True) -> build_dryer is the only allowed "
                                         "edge; build_dryer builds the raw ambient composition itself and guards "
                                         "against re-entrant calls"),
               "species_order": ["CO2", "H2O", "N2", "O2"]}
        if include_composition:
            out["y_raw"] = self.y_raw.tolist()
            out["y_conditioned"] = self.y_conditioned.tolist()
            out["molar_mass_raw_kg_mol"] = float(self.y_raw @ MW)
            out["molar_mass_conditioned_kg_mol"] = float(self.y_conditioned @ MW)
        return out


def _dry_air_fractions(config: dict) -> tuple[float, float, float]:
    """Dry-air CO2/N2/O2 fractions with argon lumped into N2, matching `common.feed_composition`."""
    co2_dry = config["ambient"]["co2_ppm_dry"] * 1e-6
    return co2_dry, (1 - co2_dry) * 0.7905, (1 - co2_dry) * 0.2095


def build_dryer(config: dict) -> Dryer | None:
    """Resolve and validate the opt-in dryer, or return None when the section is absent."""
    global _BUILDING
    if _BUILDING:
        raise RuntimeError(
            "build_dryer was re-entered. The only supported direction is "
            "feed_composition(conditioned=True) -> build_dryer; a dryer that is reached from inside its own "
            "construction would recurse. Build the dry feed once at the call site and pass the resolved Dryer "
            "around instead.")
    section = config.get("dryer")
    if section is None:
        return None
    _BUILDING = True
    try:
        return _build_dryer(config, section)
    finally:
        _BUILDING = False


def _build_dryer(config: dict, section) -> Dryer:
    if not isinstance(section, dict) or not section:
        raise ValueError("The dryer section must be a nonempty object when it is present.")
    conditioning = config.get("conditioning") or {}
    if conditioning.get("outlet_relative_humidity") is not None:
        raise ValueError(
            "The configuration declares both an explicit dryer section and the specific-energy humidity reduction "
            "(conditioning.outlet_relative_humidity). They describe the same water removal with and without a "
            "resource model, so they are mutually exclusive; removing both would meter the same dehydration "
            "twice. Keep the explicit dryer and leave conditioning.outlet_relative_humidity null.")
    _reject_unknown(section, DRYER_SECTION_KEYS, "dryer")
    target_rh = _finite(section.get("target_relative_humidity"), "dryer.target_relative_humidity",
                        minimum=0.0, maximum=1.0)
    if "flow_basis" not in section:
        raise ValueError(f"dryer.flow_basis must be declared explicitly as one of {list(FLOW_BASES)}; a guessed "
                         "basis would silently change the raw and conditioned capture denominators.")
    flow_basis = section["flow_basis"]
    if flow_basis not in FLOW_BASES:
        raise ValueError(f"dryer.flow_basis must be one of {list(FLOW_BASES)}; got {flow_basis!r}.")
    loss = _finite(section.get("co2_loss_fraction", DEFAULT_CONVENTION["co2_loss_fraction"]),
                   "dryer.co2_loss_fraction", minimum=0.0, maximum=1.0)
    if loss >= 1.0:
        raise ValueError("dryer.co2_loss_fraction must be below 1: a conditioning step that loses all CO2 leaves "
                         "nothing to screen.")
    desorption_heat = _finite(section.get("specific_water_desorption_heat_J_per_kg"),
                              "dryer.specific_water_desorption_heat_J_per_kg", minimum=0.0, allow_none=True)
    sensible_heat = _finite(section.get("specific_drybed_sensible_heat_J_per_kg_water"),
                            "dryer.specific_drybed_sensible_heat_J_per_kg_water", minimum=0.0, allow_none=True)
    efficiency = _finite(section.get("regeneration_efficiency"), "dryer.regeneration_efficiency",
                         minimum=1e-9, maximum=1.0, allow_none=True)
    auxiliary = _finite(section.get("auxiliary_electric_W"), "dryer.auxiliary_electric_W",
                        minimum=0.0, allow_none=True)
    if auxiliary is not None and auxiliary > 0.0 and "auxiliary_on_fraction_of_cycle" not in section:
        raise ValueError("dryer.auxiliary_electric_W is declared without dryer.auxiliary_on_fraction_of_cycle. "
                         "Assuming the auxiliary runs the whole cycle would be an undeclared schedule; state the "
                         "on-fraction (1.0 for continuous).")
    on_fraction = _finite(section.get("auxiliary_on_fraction_of_cycle", 1.0),
                          "dryer.auxiliary_on_fraction_of_cycle", minimum=0.0, maximum=1.0)
    pressure_drop = _finite(section.get("added_pressure_drop_Pa"), "dryer.added_pressure_drop_Pa",
                            minimum=0.0, allow_none=True)
    standby = _finite(section.get("standby_regeneration_heat_W", DEFAULT_CONVENTION["standby_regeneration_heat_W"]),
                      "dryer.standby_regeneration_heat_W", minimum=0.0)
    capacity = _finite(section.get("water_working_capacity_kg"), "dryer.water_working_capacity_kg",
                       minimum=0.0, allow_none=True)
    spec = DryerSpec(target_relative_humidity=target_rh, flow_basis=flow_basis, co2_loss_fraction=loss,
                     specific_water_desorption_heat_J_per_kg=desorption_heat,
                     specific_drybed_sensible_heat_J_per_kg_water=sensible_heat,
                     regeneration_efficiency=efficiency, auxiliary_electric_W=auxiliary,
                     auxiliary_on_fraction_of_cycle=on_fraction, added_pressure_drop_Pa=pressure_drop,
                     standby_regeneration_heat_W=standby, water_working_capacity_kg=capacity,
                     provided_keys=tuple(sorted(section)))
    outflow = _finite((config.get("airflow") or {}).get("volume_flow_m3_s"), "airflow.volume_flow_m3_s", minimum=0.0)
    fan_efficiency = _finite((config.get("airflow") or {}).get("fan_efficiency"), "airflow.fan_efficiency",
                             minimum=1e-9, maximum=1.0)

    ambient = config["ambient"]
    temperature_K = float(ambient["temperature_C"]) + 273.15
    pressure_Pa = float(ambient["pressure_Pa"])
    ambient_rh = float(ambient["relative_humidity"])
    y_raw = raw_ambient_composition(config)
    w_raw = float(y_raw[WATER])
    p_sat = water_saturation_pressure_Pa(temperature_K)
    w_target = target_rh * p_sat / pressure_Pa
    if w_target > w_raw + 1e-15:
        raise ValueError(
            f"dryer.target_relative_humidity = {target_rh:g} is above the ambient humidity "
            f"({ambient_rh:g}); a dryer removes water and this model does not humidify the feed.")
    d_co2, d_n2, d_o2 = _dry_air_fractions(config)
    normaliser = (1.0 - loss) * d_co2 + d_n2 + d_o2
    dry_conditioned = np.array([(1.0 - loss) * d_co2, d_n2, d_o2]) / normaliser
    y_conditioned = np.zeros(4)
    y_conditioned[list(DRY_SPECIES)] = (1.0 - w_target) * dry_conditioned
    y_conditioned[WATER] = w_target
    dry_raw = 1.0 - w_raw
    if dry_raw <= 0.0:
        raise ValueError("The ambient composition has no dry carrier, so no dryer balance exists.")
    raw_per_conditioned = (1.0 - w_target) / (normaliser * dry_raw)
    water_removed = w_raw * raw_per_conditioned - w_target
    co2_raw = raw_per_conditioned * dry_raw * d_co2
    co2_lost = loss * co2_raw
    dry_carrier = raw_per_conditioned * dry_raw * (d_n2 + d_o2)
    raw_in = raw_per_conditioned * y_raw
    residuals = {"n2_mol": raw_in[2] - y_conditioned[2], "o2_mol": raw_in[3] - y_conditioned[3],
                 "dry_carrier_mol": (raw_in[2] + raw_in[3]) - (y_conditioned[2] + y_conditioned[3]),
                 "co2_mol": raw_in[0] - co2_lost - y_conditioned[0],
                 "water_mol": (raw_in[WATER] - water_removed) - y_conditioned[WATER]}
    worst = max(abs(v) for v in residuals.values())
    if worst > CONSERVATION_TOLERANCE:
        raise RuntimeError(f"Dryer species balance did not close: worst per-mol residual {worst:g} mol.")
    nominal_volume_mol_s = outflow * pressure_Pa / (R * temperature_K)
    if flow_basis == UPSTREAM_RAW:
        nominal_raw_mol_s = nominal_volume_mol_s
        nominal_conditioned_mol_s = nominal_volume_mol_s / raw_per_conditioned
    else:
        nominal_conditioned_mol_s = nominal_volume_mol_s
        nominal_raw_mol_s = nominal_volume_mol_s * raw_per_conditioned
    return Dryer(spec=spec, ambient_temperature_K=temperature_K, ambient_pressure_Pa=pressure_Pa,
                 ambient_relative_humidity=ambient_rh, y_raw=y_raw, y_conditioned=y_conditioned,
                 raw_per_conditioned=raw_per_conditioned, water_removed_per_mol_conditioned=water_removed,
                 co2_raw_per_mol_conditioned=co2_raw, co2_lost_per_mol_conditioned=co2_lost,
                 dry_carrier_per_mol_conditioned=dry_carrier,
                 nominal_raw_mol_s=nominal_raw_mol_s, nominal_conditioned_mol_s=nominal_conditioned_mol_s,
                 fan_efficiency=fan_efficiency, dry_air_fractions=(d_co2, d_n2, d_o2),
                 conservation_residual=residuals)


def conditioning_resource_guard(config: dict) -> dict:
    """Check the reference and explicit conditioning modes are not both active.

    Returns a small status record the report can quote; raises instead of accepting a configuration
    where two different resource models would describe the same water removal.
    """
    conditioning = config.get("conditioning") or {}
    specific_conditioning_rh = conditioning.get("outlet_relative_humidity")
    specific_conditioning_energy = conditioning.get("electric_J_per_kg_water_removed")
    explicit = config.get("dryer") is not None
    if explicit and specific_conditioning_rh is not None:
        raise ValueError(
            "Both an explicit `dryer` section and the specific-energy humidity reduction "
            "(conditioning.outlet_relative_humidity) are declared. They are mutually exclusive: the explicit "
            "dryer carries its own species balance and resource ledger, while the other option is a specific "
            "conditioning energy with no dryer model. Declare one of them.")
    return {"explicit_dryer": explicit,
            "legacy_mode_active": specific_conditioning_rh is not None,
            "legacy_specific_energy_J_per_kg": None if specific_conditioning_rh is None else specific_conditioning_energy,
            "legacy_energy_meter": ("conditioning.electric_J_per_kg_water_removed integrates the cycle's E_COND "
                                    "meter, which is the upstream feed-drying electricity; it is not the "
                                    "pre-pump cooler, whose duty is the extended `e_cooler` meter converted by "
                                    "screening.utilities"),
            "basis": ("explicit dryer: conditioned composition, flow-basis mapping and dryer resource ledger; "
                      "specific-energy mode: humidity reduction with the supplied specific conditioning energy and no "
                      "dryer resource model; unconditioned: neither is active"),
            "unconditioned_feed": specific_conditioning_rh is None and not explicit}
