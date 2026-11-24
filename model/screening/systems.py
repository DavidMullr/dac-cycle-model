"""Optional heating-source and pre-pump conditioning systems.

Both feature groups are opt-in. When the configuration has neither a non-electrical
`heating_source` nor a `conditioner`, `build_systems` returns a base layout and the coupled
cycle uses exactly the original 30-state equations. Extended runs append their cumulative
meters after index 29, so every existing state and meter index stays stable.

Heating sources
---------------
`electrical` (or an absent section) keeps the original controller, heater node and its stored
energy. `hot_liquid` and `condensing_steam` instead deliver heat straight to the vessel wall
through a user-specified source-to-wall conductance, bounded by three things taken together:
the same bed-temperature controller demand, a finite source power limit, and a positive
source-to-wall temperature difference. For steam the source temperature is derived from the
absolute steam pressure through the IAPWS saturation relation rather than being assumed, and
the delivered heat is converted into steam mass with the same latent-heat closure used by the
conditioner. When the source is not electrical the heater node is inert: its derivative is
zero and its stored energy is excluded from the bed first-law inventory.

Energy carriers are kept apart: the electrical element is electricity; the hot-liquid utility
declares an external thermal input with a supply efficiency; steam generation optionally
accounts for feedwater sensible heat plus latent heat divided by a generator efficiency. None
of these numbers is a confirmed utility or equipment rating.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .water_properties import (CRITICAL_PRESSURE_PA, WATER_CP_LIQUID_J_MOL_K, WATER_MOLAR_MASS_KG_MOL,
                               CLOSURE_MAX_TEMPERATURE_K, CLOSURE_MIN_TEMPERATURE_K, latent_heat_J_mol,
                               saturation_pressure_Pa, saturation_temperature_K)

BASE_STATE_SIZE = 30
#: Appended blocks of the opt-in layout, in order, after the base 0..29 block.
EXTENDED_BLOCKS = (("raw", 4), ("condensate", 1), ("condensate_product", 1), ("e_source", 1),
                   ("e_source_input", 1), ("e_source_aux", 1), ("e_cooler", 1), ("e_cooler_aux", 1))
EXTENDED_STATE_SIZE = BASE_STATE_SIZE + sum(size for _, size in EXTENDED_BLOCKS)   # 41


@dataclass(frozen=True)
class Layout:
    """State-vector layout: base indices plus the appended opt-in meters."""

    size: int
    extended: bool
    raw: slice | None = None
    condensate: int | None = None
    condensate_product: int | None = None
    e_source: int | None = None
    e_source_input: int | None = None
    e_source_aux: int | None = None
    e_cooler: int | None = None
    e_cooler_aux: int | None = None

    def as_dict(self) -> dict:
        return {"size": self.size, "extended": self.extended,
                "raw_slice": None if self.raw is None else [self.raw.start, self.raw.stop],
                "condensate_index": self.condensate, "condensate_product_index": self.condensate_product,
                "e_source_index": self.e_source, "e_source_input_index": self.e_source_input,
                "e_source_aux_index": self.e_source_aux, "e_cooler_index": self.e_cooler,
                "e_cooler_aux_index": self.e_cooler_aux}


def _extended_layout() -> Layout:
    index, indices = BASE_STATE_SIZE, {}
    for name, size in EXTENDED_BLOCKS:
        indices[name] = slice(index, index + size) if size > 1 else index
        index += size
    assert index == EXTENDED_STATE_SIZE
    return Layout(size=index, extended=True, raw=indices["raw"], condensate=indices["condensate"],
                  condensate_product=indices["condensate_product"], e_source=indices["e_source"],
                  e_source_input=indices["e_source_input"], e_source_aux=indices["e_source_aux"],
                  e_cooler=indices["e_cooler"], e_cooler_aux=indices["e_cooler_aux"])


BASE_LAYOUT = Layout(size=BASE_STATE_SIZE, extended=False)
EXTENDED_LAYOUT = _extended_layout()


@dataclass(frozen=True)
class SteamSpec:
    pressure_Pa_abs: float
    feedwater_temperature_C: float | None
    generator_efficiency: float | None

    @property
    def generation_declared(self) -> bool:
        """True when the cold-water steam-generation input is fully declared."""
        return self.feedwater_temperature_C is not None and self.generator_efficiency is not None

    @property
    def saturation_temperature_K(self) -> float:
        return saturation_temperature_K(self.pressure_Pa_abs)

    def latent_heat_J_mol(self) -> float:
        return latent_heat_J_mol(self.saturation_temperature_K)

    def latent_heat_J_kg(self) -> float:
        return self.latent_heat_J_mol() / WATER_MOLAR_MASS_KG_MOL

    def steam_mass_kg_s(self, delivered_W: float) -> float:
        return 0.0 if delivered_W <= 0.0 else delivered_W / self.latent_heat_J_kg()

    def generation_input_W(self, delivered_W: float) -> float | None:
        """External thermal input to the steam generator, or None when not requested."""
        if not self.generation_declared:
            return None
        mass = self.steam_mass_kg_s(delivered_W)
        liquid_cp_J_kg_K = WATER_CP_LIQUID_J_MOL_K / WATER_MOLAR_MASS_KG_MOL
        sensible_J_kg = liquid_cp_J_kg_K * (self.saturation_temperature_K - (self.feedwater_temperature_C + 273.15))
        return mass * (self.latent_heat_J_kg() + sensible_J_kg) / self.generator_efficiency

    def as_dict(self) -> dict:
        saturation_C = self.saturation_temperature_K - 273.15
        return {"pressure_Pa_abs": self.pressure_Pa_abs, "pressure_bar_abs": self.pressure_Pa_abs / 1e5,
                "saturation_temperature_C": saturation_C,
                "latent_heat_kJ_kg": self.latent_heat_J_kg() / 1000.0,
                "feedwater_temperature_C": self.feedwater_temperature_C,
                "generator_efficiency": self.generator_efficiency,
                "generation_basis": ("cold-water steam generation = feedwater sensible heat to saturation plus "
                                     "latent heat, divided by the generator efficiency; energy carrier is external "
                                     "thermal input, not electricity") if self.generation_declared else None}


@dataclass(frozen=True)
class SourceSpec:
    kind: str
    source_temperature_K: float
    source_to_wall_UA_W_K: float
    power_limit_W: float
    jacket_heat_capacity_J_K: float
    auxiliary_electric_W: float
    supply_efficiency: float | None
    steam: SteamSpec | None
    source_temperature_basis: str

    @property
    def carrier(self) -> str:
        if self.kind == "hot_liquid":
            return "external thermal input to the rated hot-liquid loop of the hypothetical heat-transfer fluid"
        return "external thermal input to the steam generator (or the declared steam supply)"

    def delivered_W(self, wall_K: float, bed_K: float, command_W: float, *, heating: bool) -> tuple[float, str]:
        """Heat delivered to the wall with the limiting mechanism named."""
        if not heating:
            return 0.0, "source off outside heating"
        driving = self.source_to_wall_UA_W_K * max(self.source_temperature_K - wall_K, 0.0)
        delivered = min(command_W, self.power_limit_W, driving)
        if delivered <= 0.0:
            return 0.0, "no positive source-to-wall driving difference"
        if driving <= command_W and driving <= self.power_limit_W:
            limit = "source-to-wall driving difference"
        elif self.power_limit_W <= command_W:
            limit = "source power limit"
        else:
            limit = "bed-temperature controller demand"
        return float(delivered), limit

    def steam_mass_kg_s(self, delivered_W: float) -> float:
        return 0.0 if self.steam is None else self.steam.steam_mass_kg_s(delivered_W)

    @property
    def thermal_input_declared(self) -> bool:
        """True when the utility-side input of this carrier is a declared number, not an assumption we lack."""
        if self.steam is not None:
            return self.steam.generation_declared
        return self.supply_efficiency is not None

    def thermal_input_W(self, delivered_W: float) -> float | None:
        """Source-side input in the declared carrier, or None when not requested."""
        if self.steam is not None:
            return self.steam.generation_input_W(delivered_W)
        if self.supply_efficiency is None:
            return None
        return delivered_W / self.supply_efficiency

    def as_dict(self) -> dict:
        return {"kind": self.kind, "source_temperature_C": self.source_temperature_K - 273.15,
                "source_temperature_basis": self.source_temperature_basis,
                "source_to_wall_UA_W_K": self.source_to_wall_UA_W_K, "power_limit_W": self.power_limit_W,
                "jacket_heat_capacity_J_K": self.jacket_heat_capacity_J_K,
                "auxiliary_electric_W": self.auxiliary_electric_W, "carrier": self.carrier,
                "supply_efficiency": self.supply_efficiency,
                "thermal_input_declared": self.thermal_input_declared,
                "steam": None if self.steam is None else self.steam.as_dict()}


@dataclass(frozen=True)
class ConditionerSpec:
    cooling_target_K: float
    max_cooling_duty_W: float
    auxiliary_electric_W: float
    valve_max_conductance_mol_s_Pa: float | None
    line_molar_conductance_mol_s_Pa: float | None

    def as_dict(self) -> dict:
        return {"cooling_target_C": self.cooling_target_K - 273.15,
                "max_cooling_duty_W": self.max_cooling_duty_W,
                "cooling_auxiliary_electric_W": self.auxiliary_electric_W,
                "valve_max_conductance_mol_s_Pa": self.valve_max_conductance_mol_s_Pa,
                "line_molar_conductance_mol_s_Pa": self.line_molar_conductance_mol_s_Pa,
                "topology": "bed -> effective line + suction valve -> cooler/separator at P_pump -> pump",
                "hydraulic_basis": ("suction-valve conductance C_valve_eff = opening * C_valve_max, combined in "
                                    "series with the line conductance: C_total = 1/(1/C_line + 1/C_valve_eff), and "
                                    "n_bed = C_total * (P_bed - P_pump). The pump runs on its full unthrottled curve "
                                    "at the actual inlet pressure and cooler outlet temperature. Linear "
                                    "opening-to-conductance relation; an explicitly assumed effective restriction, "
                                    "not a valve, pipe or manifold model. C_line = None means no extra line loss; "
                                    "both conductances None give the loss-free helper limit P_pump = P_bed."),
                "separator_basis": ("equilibrium flash separator at the pump-inlet pressure and the cooler outlet "
                                    "temperature: ideal gas, no holdup, perfect liquid separation, zero dissolved CO2"),
                "cooling_basis": ("capacity-limited gas cooler: if the sensible-plus-latent requirement exceeds the "
                                  "maximum duty the warmer achievable outlet is solved for; gas entering below the "
                                  "target is never heated; a zero-capacity cooler is the inactive "
                                  "no-cooling/no-condensation limit"),
                "property_range_basis": (f"cooling target bounded to {CLOSURE_MIN_TEMPERATURE_K - 273.15:.2f}-"
                                         f"{CLOSURE_MAX_TEMPERATURE_K - 273.15:.0f} C, the screening range of the "
                                         "IAPWS saturation line and the constant-Cp water enthalpy closure; a "
                                         "subfreezing target is rejected rather than extrapolated into an ice model"),
                "cooling_auxiliary_basis": ("assumed on/off schedule: the auxiliary load is counted whenever the pump "
                                            "motor runs - including a bypassed (zero-capacity) cooler - and is zero "
                                            "while the pump is off; it is not duty-proportional")}


@dataclass(frozen=True)
class Systems:
    source: SourceSpec | None
    conditioner: ConditionerSpec | None
    layout: Layout

    @property
    def uses_extended_state(self) -> bool:
        return self.layout.extended

    def as_dict(self) -> dict:
        return {"source": None if self.source is None else self.source.as_dict(),
                "conditioner": None if self.conditioner is None else self.conditioner.as_dict(),
                "state_layout": self.layout.as_dict()}


BASE_SYSTEMS = Systems(source=None, conditioner=None, layout=BASE_LAYOUT)


def _number(value, name: str, *, minimum: float | None = None, maximum: float | None = None,
            allow_none: bool = False) -> float | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}; got {value}.")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}; got {value}.")
    return float(value)


def build_heating_source(config: dict) -> SourceSpec | None:
    """Validated source specification, or None for the base electrical element."""
    section = config.get("heating_source")
    if not section:
        return None
    kind = section.get("kind", "electrical")
    if kind == "electrical":
        return None
    if kind not in ("hot_liquid", "condensing_steam"):
        raise ValueError("heating_source.kind must be 'electrical', 'hot_liquid' or 'condensing_steam'.")
    thermal = config["thermal"]
    t_target = config["cycle"]["regeneration_temperature_C"] + 273.15
    UA = _number(section.get("source_to_wall_UA_W_K", 20.0), "heating_source.source_to_wall_UA_W_K", minimum=1e-9)
    limit = _number(section.get("source_power_limit_W", thermal["heater_power_W"]),
                    "heating_source.source_power_limit_W", minimum=1e-9)
    jacket = _number(section.get("jacket_heat_capacity_J_K", 0.0), "heating_source.jacket_heat_capacity_J_K",
                     minimum=0.0)
    auxiliary = _number(section.get("auxiliary_electric_W", 0.0), "heating_source.auxiliary_electric_W", minimum=0.0)
    steam = None
    if kind == "condensing_steam":
        steam_section = section.get("steam") or {}
        pressure = _number(steam_section.get("pressure_Pa_abs", 700000.0), "heating_source.steam.pressure_Pa_abs",
                           minimum=1.0, maximum=CRITICAL_PRESSURE_PA * 0.99)
        feed = _number(steam_section.get("feedwater_temperature_C"), "heating_source.steam.feedwater_temperature_C",
                       minimum=0.0, maximum=300.0, allow_none=True)
        efficiency = _number(steam_section.get("generator_efficiency"), "heating_source.steam.generator_efficiency",
                             minimum=1e-3, maximum=1.0, allow_none=True)
        if (feed is None) != (efficiency is None):
            raise ValueError("heating_source.steam needs both feedwater_temperature_C and generator_efficiency to "
                             "declare a cold-water steam-generation input, or neither of them. A partial pair is not "
                             "a declared utility input and would report an undefined external thermal input.")
        steam = SteamSpec(pressure_Pa_abs=pressure, feedwater_temperature_C=feed, generator_efficiency=efficiency)
        source_K = steam.saturation_temperature_K
        if source_K > CLOSURE_MAX_TEMPERATURE_K:
            raise ValueError("heating_source.steam.pressure_Pa_abs gives a saturation temperature of "
                             f"{source_K - 273.15:.1f} C, above the {CLOSURE_MAX_TEMPERATURE_K - 273.15:.0f} C upper "
                             "bound of the documented constant-Cp water enthalpy closure. The highest supported "
                             f"absolute pressure is {saturation_pressure_Pa(CLOSURE_MAX_TEMPERATURE_K) / 1e5:.2f} bar.")
        if feed is not None and feed > source_K - 273.15:
            raise ValueError("heating_source.steam.feedwater_temperature_C cannot exceed the steam saturation "
                             f"temperature ({source_K - 273.15:.2f} C) in this cold-water generation closure.")
        if "source_temperature_C" in section:
            declared = _number(section["source_temperature_C"], "heating_source.source_temperature_C", minimum=1.0)
            if abs(declared - (source_K - 273.15)) > 1.0:
                raise ValueError("heating_source.source_temperature_C contradicts the saturation temperature "
                                 f"({source_K - 273.15:.2f} C) derived from the absolute steam pressure.")
        basis = ("derived from the absolute steam pressure through the IAPWS saturation relation; the source is an "
                 "indirect condensing jacket, so steam never enters the process gas")
        supply_efficiency = None
    else:
        source_K = _number(section.get("source_temperature_C", 165.0), "heating_source.source_temperature_C",
                           minimum=1.0, maximum=CLOSURE_MAX_TEMPERATURE_K - 273.15) + 273.15
        supply_efficiency = _number(section.get("supply_efficiency"), "heating_source.supply_efficiency",
                                    minimum=1e-3, maximum=1.0, allow_none=True)
        basis = ("prescribed supply temperature of a hypothetical rated-loop heat-transfer fluid (analyst "
                 "assumption; a 165 C loop is a pressurised or high-boiling rated circuit, not atmospheric hot "
                 "water, and no fluid, loop, boiler or equipment is selected or confirmed)")
    if source_K <= t_target and not section.get("advisory_temperature_target", False):
        raise ValueError("The heating source temperature must exceed the bed regeneration target to provide a "
                         f"driving difference; source {source_K - 273.15:.2f} C vs target {t_target - 273.15:.2f} C.")
    return SourceSpec(kind=kind, source_temperature_K=source_K, source_to_wall_UA_W_K=UA, power_limit_W=limit,
                      jacket_heat_capacity_J_K=jacket, auxiliary_electric_W=auxiliary,
                      supply_efficiency=supply_efficiency, steam=steam, source_temperature_basis=basis)


def build_conditioner(config: dict) -> ConditionerSpec | None:
    """Validated conditioner specification, or None when no pre-pump conditioning is configured."""
    section = config.get("conditioner")
    if not section:
        return None
    # The target is a bulk-water equilibrium temperature, so it must stay inside the documented
    # 0.01-200 C screening range of the saturation and enthalpy relations. A subfreezing target is
    # rejected here instead of silently reaching the liquid-water property guard during integration.
    target = _number(section.get("cooling_target_C", 40.0), "conditioner.cooling_target_C",
                     minimum=CLOSURE_MIN_TEMPERATURE_K - 273.15, maximum=CLOSURE_MAX_TEMPERATURE_K - 273.15)
    duty = _number(section.get("max_cooling_duty_W", 500.0), "conditioner.max_cooling_duty_W", minimum=0.0)
    auxiliary = _number(section.get("cooling_auxiliary_electric_W", 0.0),
                        "conditioner.cooling_auxiliary_electric_W", minimum=0.0)
    valve = _number(section.get("valve_max_conductance_mol_s_Pa"),
                    "conditioner.valve_max_conductance_mol_s_Pa", minimum=1e-12, allow_none=True)
    line = _number(section.get("line_molar_conductance_mol_s_Pa"),
                   "conditioner.line_molar_conductance_mol_s_Pa", minimum=1e-12, allow_none=True)
    return ConditionerSpec(cooling_target_K=target + 273.15, max_cooling_duty_W=duty,
                           auxiliary_electric_W=auxiliary, valve_max_conductance_mol_s_Pa=valve,
                           line_molar_conductance_mol_s_Pa=line)


def build_systems(config: dict) -> Systems:
    """Assemble the validated opt-in systems; absent sections give the base path."""
    source = build_heating_source(config)
    conditioner = build_conditioner(config)
    if source is None and conditioner is None:
        return BASE_SYSTEMS
    return Systems(source=source, conditioner=conditioner, layout=EXTENDED_LAYOUT)
