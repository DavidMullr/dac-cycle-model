"""Convert cooling duties into declared electricity or thermal input.

Wall cooling and the pump-inlet cooler are independent service meters. Constant
COP or thermal efficiency converts their useful duty; auxiliary input is added
over the operating duration. Missing conversion coefficients remain unknown.
Passive services cannot also declare an active conversion coefficient.

The upstream feed-conditioning electricity meter is separate from these cooling
services. Physical capacity limits and purchased-energy accounting have distinct
roles; neither implies a selected or measured device."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Declared service kinds. `passive` buys nothing; `constant_cop` converts with constant factors.
PASSIVE = "passive"
CONSTANT_COP = "constant_cop"
KINDS = (PASSIVE, CONSTANT_COP)
#: The two cooling services, in reporting order.
SERVICES = ("wall_cooling", "prepump_cooling")
#: Unit-sanity ceiling on a declared constant coefficient (a COP of 50 is already implausible).
MAX_CONVERSION_COEFFICIENT = 50.0
SERVICE_KEYS = frozenset({"kind", "cop", "thermal_efficiency", "auxiliary_electric_W",
                          "max_cooling_capacity_W", "capacity_basis"})
DEFAULT_CAPACITY_BASIS = ("declared physical cooling capacity used as a cap by the physical model; "
                          "no equipment has been selected, rated or measured")
ENERGY_UNIT_RULE = ("energies are summed and reported in J; the capacity audit compares a peak duty in W with a "
                    "declared capacity in W; an integrated energy is never compared with a power capacity and an "
                    "auxiliary power is never added to an energy")
UNDECLARED_BASIS = ("this service is not declared in the `utilities` section, so its purchased carrier is "
                    "unknown. A positive duty therefore makes the plant boundary incomplete instead of being "
                    "dropped; declare the service (or a `passive` kind) to close the boundary.")


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


def _validate_removed_heat(value, name: str) -> float:
    """A cooling duty or energy is a removed heat: finite and nonnegative."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")
    if value < 0.0:
        raise ValueError(f"{name} is a removed heat and cannot be negative; got {value}. Pass the cooling duty or "
                         "energy (>= 0) this service actually metered.")
    return float(value)


@dataclass(frozen=True)
class CoolingService:
    """One declared cooling service: constant-carrier conversion plus optional physical capacity."""

    service: str
    kind: str
    cop: float | None
    thermal_efficiency: float | None
    auxiliary_electric_W: float | None
    max_cooling_capacity_W: float | None
    capacity_basis: str | None

    @property
    def passive(self) -> bool:
        return self.kind == PASSIVE

    @property
    def conversion_declared(self) -> bool:
        """True when the purchased carrier(s) of this service are declared numbers."""
        return self.passive or self.cop is not None or self.thermal_efficiency is not None

    def electricity_W(self, duty_W: float) -> float | None:
        """Purchased electricity power for a useful cooling power, or None when unknown."""
        if self.passive:
            return 0.0
        if self.cop is not None:
            return duty_W / self.cop
        if self.thermal_efficiency is not None:
            return 0.0                 # declared thermally driven machine: no external electricity
        return None

    def thermal_input_W(self, duty_W: float) -> float | None:
        """External thermal input power for a useful cooling power, or None when unknown."""
        if self.passive:
            return 0.0
        if self.thermal_efficiency is not None:
            return duty_W / self.thermal_efficiency
        if self.cop is not None:
            return 0.0                 # declared electricity-driven machine: no external heat
        return None

    @property
    def conversion_basis(self) -> str:
        if self.passive:
            return ("passive service: no purchased carrier is declared, so both electricity and external "
                    "thermal input are exactly zero by choice")
        if self.cop is not None and self.thermal_efficiency is not None:
            return (f"constant COP {self.cop:g} on electricity plus an independent external thermal input "
                    f"with efficiency {self.thermal_efficiency:g} (for example an absorption machine with an "
                    "electric solution pump modelled as the declared auxiliary)")
        if self.cop is not None:
            return f"electricity-driven machine with constant COP {self.cop:g}; external thermal input zero by declaration"
        if self.thermal_efficiency is not None:
            return (f"thermally driven machine with constant efficiency {self.thermal_efficiency:g}; external "
                    "electricity zero except for the declared auxiliary")
        return ("active service with no declared conversion coefficient: its electricity and external thermal "
                "input are unknown, so the whole-train input stays incomplete")

    def as_dict(self) -> dict:
        return {"service": self.service, "kind": self.kind, "cop": self.cop,
                "thermal_efficiency": self.thermal_efficiency,
                "auxiliary_electric_W": self.auxiliary_electric_W,
                "auxiliary_declared": self.auxiliary_electric_W is not None,
                "max_cooling_capacity_W": self.max_cooling_capacity_W,
                "capacity_basis": self.capacity_basis,
                "conversion_declared": self.conversion_declared,
                "conversion_basis": self.conversion_basis}


def _capacity_audit(service: CoolingService, peak_duty_W: float | None) -> dict:
    """Compare a peak **power** with a declared **power** capacity; never an energy with a power."""
    capacity = service.max_cooling_capacity_W
    if capacity is None:
        return {"peak_duty_W": peak_duty_W, "max_cooling_capacity_W": None, "capacity_shortfall_W": None,
                "capacity_exceeded": None, "capacity_audit": "no capacity declared for this service",
                "capacity_audit_incomplete": False}
    if peak_duty_W is None:
        return {"peak_duty_W": None, "max_cooling_capacity_W": capacity, "capacity_shortfall_W": None,
                "capacity_exceeded": None,
                "capacity_audit": ("a declared capacity exists but no peak duty was supplied, so the physical "
                                   "limit could not be audited"),
                "capacity_audit_incomplete": True}
    peak = _validate_removed_heat(peak_duty_W, f"{service.service} peak_duty_W")
    shortfall = max(peak - capacity, 0.0)
    return {"peak_duty_W": peak, "max_cooling_capacity_W": capacity, "capacity_shortfall_W": shortfall,
            "capacity_exceeded": bool(shortfall > 0.0),
            "capacity_audit": "peak duty (W) compared with the declared capacity (W)",
            "capacity_audit_incomplete": False}


def _undeclared_service_report(service: str, duty) -> dict:
    """A metered duty for a service that the section does not declare: unknown, never dropped."""
    return {"service": service, "undeclared_service": True, "useful_duty_W": duty,
            "electricity_W": None, "thermal_input_W": None, "auxiliary_electric_W": None,
            "total_electric_W": None, "max_cooling_capacity_W": None, "capacity_shortfall_W": None,
            "capacity_exceeded": None, "capacity_audit": "the service is not declared, so no capacity is declared",
            "capacity_audit_incomplete": False, "conversion_declared": False,
            "unknown_carriers": ["electricity", "thermal_input", "auxiliary_electric"], "basis": UNDECLARED_BASIS}


def service_report(service: CoolingService, duty_W: float) -> dict:
    """Conversion of one already-metered cooling power; the duty itself is passed through unchanged."""
    duty = _validate_removed_heat(duty_W, f"{service.service} duty_W")
    electricity = service.electricity_W(duty)
    thermal = service.thermal_input_W(duty)
    auxiliary = service.auxiliary_electric_W
    unknown = [carrier for carrier, value in (("electricity", electricity), ("thermal_input", thermal),
                                              ("auxiliary_electric", auxiliary)) if value is None]
    report = {"service": service.service, "undeclared_service": False, "useful_duty_W": duty,
              "electricity_W": electricity, "thermal_input_W": thermal,
              "auxiliary_electric_W": auxiliary,
              "total_electric_W": (None if electricity is None or auxiliary is None else electricity + auxiliary),
              "conversion_declared": service.conversion_declared,
              "unknown_carriers": unknown, "basis": service.conversion_basis}
    # The instantaneous API takes a power, so this metered duty *is* the peak power for the audit.
    report.update(_capacity_audit(service, duty))
    return report


def service_energy_report(service: CoolingService | None, *, duty_J: float, on_duration_s: float,
                          peak_duty_W: float | None = None, name: str) -> dict | None:
    """Cycle-energy conversion of one service: E/COP, auxiliary W * duration s, peak W audited separately.

    Returns None when the service is not declared *and* no cooling was metered, because then nothing was
    purchased and there is no boundary gap to report.
    """
    duty = _validate_removed_heat(duty_J, f"{name} duty_J")
    duration = _finite(on_duration_s, f"{name} on_duration_s", minimum=0.0)
    if service is None:
        if duty <= 0.0:
            return None
        report = _undeclared_service_report(name, duty)
        report.update({"useful_duty_J": duty, "on_duration_s": duration, "mean_duty_W": None,
                       "electricity_J": None, "thermal_input_J": None, "auxiliary_electric_J": None,
                       "total_electric_J": None})
        if peak_duty_W is not None:
            report["peak_duty_W"] = _validate_removed_heat(peak_duty_W, f"{name} peak_duty_W")
        report["mean_duty_W"] = None if duration <= 0.0 else duty / duration
        return report
    if duty <= 0.0 and duration <= 0.0:
        return None                                  # nothing was metered and no service time was spent
    electricity_W = service.electricity_W(1.0)       # constant coefficient: 1 W probe gives the factor
    thermal_W = service.thermal_input_W(1.0)
    electricity_J = None if electricity_W is None else duty * electricity_W
    thermal_J = None if thermal_W is None else duty * thermal_W
    auxiliary_J = None if service.auxiliary_electric_W is None else service.auxiliary_electric_W * duration
    unknown = [carrier for carrier, value in (("electricity", electricity_J), ("thermal_input", thermal_J),
                                              ("auxiliary_electric", auxiliary_J)) if value is None]
    report = {"service": name, "undeclared_service": False, "useful_duty_J": duty, "on_duration_s": duration,
              "mean_duty_W": None if duration <= 0.0 else duty / duration,
              "electricity_J": electricity_J, "thermal_input_J": thermal_J,
              "auxiliary_electric_W": service.auxiliary_electric_W, "auxiliary_electric_J": auxiliary_J,
              "total_electric_J": (None if electricity_J is None or auxiliary_J is None
                                   else electricity_J + auxiliary_J),
              "conversion_declared": service.conversion_declared, "unknown_carriers": unknown,
              "basis": service.conversion_basis}
    report.update(_capacity_audit(service, peak_duty_W))
    return report


@dataclass(frozen=True)
class Utilities:
    """Validated opt-in utility section: the cooling services and their declared conversions."""

    wall_cooling: CoolingService | None
    prepump_cooling: CoolingService | None
    declared_empty: bool = False

    @property
    def has_services(self) -> bool:
        return self.wall_cooling is not None or self.prepump_cooling is not None

    def services(self) -> tuple[CoolingService, ...]:
        return tuple(s for s in (self.wall_cooling, self.prepump_cooling) if s is not None)

    def wall_cooling_capacity_W(self) -> float | None:
        """Physical wall-cooling cap for the thermal right-hand side, or None when not declared."""
        return None if self.wall_cooling is None else self.wall_cooling.max_cooling_capacity_W

    def wall_cooling_capacity_basis(self) -> str | None:
        return None if self.wall_cooling is None else self.wall_cooling.capacity_basis

    def _service_pairs(self, wall_duty, prepump_duty):
        return (("wall_cooling", self.wall_cooling, wall_duty),
                ("prepump_cooling", self.prepump_cooling, prepump_duty))

    def convert(self, *, wall_cooling_duty_W: float = 0.0,
                prepump_cooling_duty_W: float = 0.0) -> dict:
        """Instantaneous conversion in W of both metered powers, keeping unknown carriers as None.

        The caller passes the duties its own integrators produced. This function adds nothing else: the
        existing conditioner auxiliary and the heating-source loop auxiliary are metered by
        `screening/systems.py` and must not be passed here.
        """
        reports: dict[str, dict] = {}
        for name, service, duty in self._service_pairs(wall_cooling_duty_W, prepump_cooling_duty_W):
            metered = _validate_removed_heat(duty, f"{name} duty_W")
            if service is None:
                if metered > 0.0:
                    reports[name] = _undeclared_service_report(name, metered)
                continue
            reports[name] = service_report(service, metered)
        electricity_terms = [r["electricity_W"] for r in reports.values()]
        thermal_terms = [r["thermal_input_W"] for r in reports.values()]
        unknown = [{"service": name, "carrier": carrier} for name, r in reports.items()
                   for carrier in r["unknown_carriers"]]
        electricity = None if any(v is None for v in electricity_terms) else float(sum(electricity_terms))
        thermal = None if any(v is None for v in thermal_terms) else float(sum(thermal_terms))
        auxiliary_terms = [r["auxiliary_electric_W"] for r in reports.values()]
        auxiliary = None if any(v is None for v in auxiliary_terms) else float(sum(auxiliary_terms))
        return {"services": reports,
                "useful_cooling_duty_W": float(sum(r["useful_duty_W"] for r in reports.values())),
                "electricity_W": electricity,
                "thermal_input_W": thermal,
                "auxiliary_electric_W": auxiliary,
                "total_electric_W": None if electricity is None or auxiliary is None else electricity + auxiliary,
                "complete": len(unknown) == 0,
                "unknown_carriers": unknown,
                "declared_services": sorted(name for name in SERVICES
                                            if getattr(self, name) is not None),
                "omitted_services": sorted(name for name in SERVICES
                                           if getattr(self, name) is None),
                "declared_empty_boundary": self.declared_empty,
                "capacities": {f"{name}_max_cooling_capacity_W": r["max_cooling_capacity_W"]
                               for name, r in reports.items()},
                "excluded_inputs": ("the pre-pump conditioner auxiliary and the heating-source loop auxiliary are "
                                    "metered by their own cycle states and are deliberately not added here"),
                "energy_unit_rule": ENERGY_UNIT_RULE}

    def cycle_energy(self, *, wall_cooling_duty_J: float = 0.0, wall_cooling_on_duration_s: float = 0.0,
                     wall_cooling_peak_duty_W: float | None = None,
                     prepump_cooling_duty_J: float = 0.0, prepump_cooling_on_duration_s: float = 0.0,
                     prepump_cooling_peak_duty_W: float | None = None) -> dict:
        """Cumulative cycle conversion in J, with the peak power supplied separately for the capacity audit.

        ``E_electric = E_useful / COP`` holds because the coefficient is constant, so the integrated duty
        may be converted directly. The auxiliary is ``auxiliary_W * on_duration_s``. The capacity audit
        compares ``peak_duty_W`` (W) with the declared capacity (W); an energy is never compared with a
        power. A missing peak with a declared capacity is reported as an incomplete audit rather than as a
        pass.
        """
        reports: dict[str, dict] = {}
        pairs = (("wall_cooling", self.wall_cooling, wall_cooling_duty_J, wall_cooling_on_duration_s,
                  wall_cooling_peak_duty_W),
                 ("prepump_cooling", self.prepump_cooling, prepump_cooling_duty_J, prepump_cooling_on_duration_s,
                  prepump_cooling_peak_duty_W))
        for name, service, duty, duration, peak in pairs:
            report = service_energy_report(service, duty_J=duty, on_duration_s=duration, peak_duty_W=peak, name=name)
            if report is not None:
                reports[name] = report
        electricity_terms = [r["electricity_J"] for r in reports.values()]
        thermal_terms = [r["thermal_input_J"] for r in reports.values()]
        auxiliary_terms = [r["auxiliary_electric_J"] for r in reports.values()]
        unknown = [{"service": name, "carrier": carrier} for name, r in reports.items()
                   for carrier in r["unknown_carriers"]]
        incomplete_audits = sorted(name for name, r in reports.items() if r["capacity_audit_incomplete"])
        exceeded = sorted(name for name, r in reports.items() if r["capacity_exceeded"])
        electricity = None if any(v is None for v in electricity_terms) else float(sum(electricity_terms))
        thermal = None if any(v is None for v in thermal_terms) else float(sum(thermal_terms))
        auxiliary = None if any(v is None for v in auxiliary_terms) else float(sum(auxiliary_terms))
        return {"services": reports,
                "useful_cooling_duty_J": float(sum(r["useful_duty_J"] for r in reports.values())),
                "electricity_J": electricity,
                "thermal_input_J": thermal,
                "auxiliary_electric_J": auxiliary,
                "total_electric_J": None if electricity is None or auxiliary is None else electricity + auxiliary,
                "complete": len(unknown) == 0,
                "unknown_carriers": unknown,
                "capacity_exceeded_services": exceeded,
                "capacity_audit_incomplete_services": incomplete_audits,
                "declared_services": sorted(name for name in SERVICES if getattr(self, name) is not None),
                "omitted_services": sorted(name for name in SERVICES if getattr(self, name) is None),
                "declared_empty_boundary": self.declared_empty,
                "excluded_inputs": ("the pre-pump conditioner auxiliary and the heating-source loop auxiliary are "
                                    "metered by their own cycle states and are deliberately not added here"),
                "energy_unit_rule": ENERGY_UNIT_RULE,
                "energy_basis": ("constant coefficients, so an integrated duty converts as E/COP or E/eta; the "
                                 "auxiliary integrates as auxiliary_W * on_duration_s")}

    def as_dict(self) -> dict:
        return {"wall_cooling": None if self.wall_cooling is None else self.wall_cooling.as_dict(),
                "prepump_cooling": None if self.prepump_cooling is None else self.prepump_cooling.as_dict(),
                "declared_empty_boundary": self.declared_empty,
                "conversion_basis": ("constant coefficient applied to the independently integrated useful duty; "
                                     "the duty is never clamped by a postprocessor, a declared capacity is exported "
                                     "for the physical model and an over-capacity duty is flagged"),
                "meter_ownership": ("the pre-pump cooler duty is the conditioner's `e_cooler` meter; E_COND is the "
                                    "upstream feed-drying electricity, not the pre-pump cooler"),
                "unit_rule": ENERGY_UNIT_RULE,
                "carrier_separation": ("useful heat, purchased electricity and external thermal input are reported "
                                       "as separate quantities and are never added or ranked against each other"),
                "undeclared_service_rule": UNDECLARED_BASIS}


def _build_service(name: str, section) -> CoolingService:
    if not isinstance(section, dict):
        raise ValueError(f"utilities.{name} must be an object.")
    _reject_unknown(section, SERVICE_KEYS, f"utilities.{name}")
    if "kind" not in section:
        raise ValueError(f"utilities.{name}.kind must be declared explicitly as one of {list(KINDS)}.")
    kind = section["kind"]
    if kind not in KINDS:
        raise ValueError(f"utilities.{name}.kind must be one of {list(KINDS)}; got {kind!r}.")
    cop = _finite(section.get("cop"), f"utilities.{name}.cop", minimum=1e-9,
                  maximum=MAX_CONVERSION_COEFFICIENT, allow_none=True)
    thermal = _finite(section.get("thermal_efficiency"), f"utilities.{name}.thermal_efficiency",
                      minimum=1e-9, maximum=MAX_CONVERSION_COEFFICIENT, allow_none=True)
    # A passive service buys nothing, so an omitted auxiliary is a declared zero. For an active service an
    # omitted auxiliary would be a silent zero on a real purchased load, so it stays unknown instead.
    auxiliary = _finite(section.get("auxiliary_electric_W"), f"utilities.{name}.auxiliary_electric_W",
                        minimum=0.0, allow_none=True)
    if kind == PASSIVE and auxiliary is None:
        auxiliary = 0.0
    capacity = _finite(section.get("max_cooling_capacity_W"), f"utilities.{name}.max_cooling_capacity_W",
                       minimum=0.0, allow_none=True)
    capacity_basis = section.get("capacity_basis")
    if capacity is not None and not isinstance(capacity_basis, str):
        capacity_basis = DEFAULT_CAPACITY_BASIS
    if capacity is None and capacity_basis is not None:
        raise ValueError(f"utilities.{name}.capacity_basis is declared without a max_cooling_capacity_W to describe.")
    if kind == PASSIVE and (cop is not None or thermal is not None):
        raise ValueError(f"utilities.{name} is declared passive but also declares a conversion coefficient. A "
                         "passive service buys nothing; remove cop/thermal_efficiency or declare kind "
                         f"'{CONSTANT_COP}'.")
    return CoolingService(service=name, kind=kind, cop=cop, thermal_efficiency=thermal,
                          auxiliary_electric_W=auxiliary, max_cooling_capacity_W=capacity,
                          capacity_basis=capacity_basis)


def build_utilities(config: dict) -> Utilities | None:
    """Validated utility section, or None when the opt-in section is absent (reference configuration).

    An empty object is accepted as an explicitly declared empty utility boundary; any metered duty then
    reports unknown carriers and an incomplete train rather than silently disappearing.
    """
    section = config.get("utilities")
    if section is None:
        return None
    if not isinstance(section, dict):
        raise ValueError("The utilities section must be an object when it is present.")
    _reject_unknown(section, frozenset(SERVICES), "utilities")
    if not section:
        return Utilities(wall_cooling=None, prepump_cooling=None, declared_empty=True)
    wall = _build_service("wall_cooling", section["wall_cooling"]) if "wall_cooling" in section else None
    prepump = (_build_service("prepump_cooling", section["prepump_cooling"])
               if "prepump_cooling" in section else None)
    if prepump is not None and not config.get("conditioner"):
        raise ValueError("utilities.prepump_cooling describes the duty of the pre-pump cooler, but no `conditioner` "
                         "section is configured, so that duty does not exist in this train. Configure the conditioner "
                         "or remove the service.")
    conditioner = config.get("conditioner") or {}
    conditioner_aux = _finite(conditioner.get("cooling_auxiliary_electric_W", 0.0),
                              "conditioner.cooling_auxiliary_electric_W", minimum=0.0)
    if prepump is not None and (prepump.auxiliary_electric_W or 0.0) > 0.0 and conditioner_aux > 0.0:
        raise ValueError("Both utilities.prepump_cooling.auxiliary_electric_W and "
                         "conditioner.cooling_auxiliary_electric_W are nonzero; the pre-pump cooler auxiliary would "
                         "be metered twice. Keep the heat-source loop/conditioner auxiliaries in the conditioner "
                         "section and use the utility service for the conversion of its useful duty only.")
    return Utilities(wall_cooling=wall, prepump_cooling=prepump)


def utility_report(config: dict, *, wall_cooling_duty_W: float = 0.0,
                   prepump_cooling_duty_W: float = 0.0) -> dict | None:
    """Convenience entry point: build the section (if any) and convert both instantaneous duties."""
    utilities = build_utilities(config)
    return None if utilities is None else utilities.convert(wall_cooling_duty_W=wall_cooling_duty_W,
                                                            prepump_cooling_duty_W=prepump_cooling_duty_W)


def utility_energy_report(config: dict, **energy_kwargs) -> dict | None:
    """Convenience entry point for the cycle-energy API; see `Utilities.cycle_energy`."""
    utilities = build_utilities(config)
    return None if utilities is None else utilities.cycle_energy(**energy_kwargs)
