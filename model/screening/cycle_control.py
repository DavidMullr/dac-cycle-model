"""Checkpointed desorption control outside the ODE right-hand side.

The controller observes accepted integration endpoints. It never updates its
history, peak or timer during solver trial evaluations. With no control section,
desorption uses one integration interval. Fixed-timer and state-controlled cases
can share a checkpoint grid for matched numerical comparisons."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

SCHEMA = "desorption-control/1"
MODES = ("fixed_timer", "state_control")
CONTROL_KEYS = frozenset({
    "schema",
    "mode",
    "minimum_duration_s",
    "checkpoint_s",
    "bed_temperature_margin_K",
    "tail_window_s",
    "tail_rate_limit_mol_s",
    "tail_mass_fraction_of_product",
    "require_all_tail_samples_below_rate",
})


class DesorptionControlError(ValueError):
    """A desorption controller declaration is incomplete or unsafe."""


def _number(value: Any, name: str, *, minimum: float | None = None,
            maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DesorptionControlError(f"{name} must be a finite number.")
    number = float(value)
    if not math.isfinite(number):
        raise DesorptionControlError(f"{name} must be finite.")
    if minimum is not None and number < minimum:
        raise DesorptionControlError(f"{name} must be >= {minimum}; got {number}.")
    if maximum is not None and number > maximum:
        raise DesorptionControlError(f"{name} must be <= {maximum}; got {number}.")
    return number


@dataclass(frozen=True)
class DesorptionControl:
    mode: str
    minimum_duration_s: float
    hard_max_duration_s: float
    checkpoint_s: float
    bed_temperature_margin_K: float
    tail_window_s: float
    tail_rate_limit_mol_s: float
    tail_mass_fraction_of_product: float
    require_all_tail_samples_below_rate: bool

    @property
    def state_controlled(self) -> bool:
        return self.mode == "state_control"


def validate_desorption_control(
    section: Mapping | None, *, hard_max_duration_s: float
) -> DesorptionControl | None:
    if section is None:
        return None
    if not isinstance(section, Mapping):
        raise DesorptionControlError("cycle.desorption_control must be an object.")
    unknown = sorted(set(section) - CONTROL_KEYS)
    if unknown:
        raise DesorptionControlError(
            f"cycle.desorption_control has unsupported key(s) {unknown}."
        )
    if section.get("schema") != SCHEMA:
        raise DesorptionControlError(f"desorption control schema must be {SCHEMA!r}.")
    mode = section.get("mode")
    if mode not in MODES:
        raise DesorptionControlError(f"desorption control mode must be one of {MODES}.")
    hard_max = _number(hard_max_duration_s, "cycle.desorption_s", minimum=1e-9)
    minimum = _number(
        section.get("minimum_duration_s"),
        "cycle.desorption_control.minimum_duration_s",
        minimum=1e-9,
        maximum=hard_max,
    )
    if minimum >= hard_max:
        raise DesorptionControlError(
            "The declared minimum desorption duration must be below the hard maximum."
        )
    checkpoint = _number(
        section.get("checkpoint_s"),
        "cycle.desorption_control.checkpoint_s",
        minimum=1e-3,
        maximum=min(60.0, minimum),
    )
    margin = _number(
        section.get("bed_temperature_margin_K"),
        "cycle.desorption_control.bed_temperature_margin_K",
        minimum=1e-9,
        maximum=2.0,
    )
    tail_window = _number(
        section.get("tail_window_s"),
        "cycle.desorption_control.tail_window_s",
        minimum=checkpoint,
        maximum=hard_max,
    )
    ratio = tail_window / checkpoint
    if abs(ratio - round(ratio)) > 1e-9:
        raise DesorptionControlError(
            "tail_window_s must be an integer number of checkpoint_s intervals."
        )
    tail_rate = _number(
        section.get("tail_rate_limit_mol_s"),
        "cycle.desorption_control.tail_rate_limit_mol_s",
        minimum=1e-12,
    )
    tail_mass_fraction = _number(
        section.get("tail_mass_fraction_of_product"),
        "cycle.desorption_control.tail_mass_fraction_of_product",
        minimum=0.0,
        maximum=1.0,
    )
    all_samples = section.get("require_all_tail_samples_below_rate")
    if not isinstance(all_samples, bool) or all_samples is not True:
        raise DesorptionControlError(
            "require_all_tail_samples_below_rate must be true; a single "
            "instantaneous value is not a bounded tail."
        )
    return DesorptionControl(
        mode=mode,
        minimum_duration_s=minimum,
        hard_max_duration_s=hard_max,
        checkpoint_s=checkpoint,
        bed_temperature_margin_K=margin,
        tail_window_s=tail_window,
        tail_rate_limit_mol_s=tail_rate,
        tail_mass_fraction_of_product=tail_mass_fraction,
        require_all_tail_samples_below_rate=all_samples,
    )


def load_desorption_control(config: Mapping) -> DesorptionControl | None:
    cycle = config.get("cycle") or {}
    section = cycle.get("desorption_control")
    if section is None:
        return None
    hard_max = cycle.get("desorption_s")
    if isinstance(hard_max, bool) or not isinstance(hard_max, (int, float)):
        raise DesorptionControlError(
            "cycle.desorption_s must be declared before enabling desorption control."
        )
    return validate_desorption_control(section, hard_max_duration_s=float(hard_max))


@dataclass
class DesorptionControlRuntime:
    """Pure observation of accepted chunk endpoints for one desorption phase."""

    control: DesorptionControl
    target_bed_K: float
    start_product_mol: float
    _samples: list[dict] = field(default_factory=list)
    _trace_peak_rate: float = 0.0
    _trigger: dict | None = None

    def _new_sample(self, time_s: float, bed_K: float, product_flux: float,
                    product_mol: float) -> dict:
        if self._samples and time_s <= self._samples[-1]["time_s"] + 1e-12:
            raise DesorptionControlError("control checkpoints must advance in time.")
        if not all(math.isfinite(float(value)) for value in (time_s, bed_K, product_flux, product_mol)):
            raise DesorptionControlError("control checkpoint metrics must be finite.")
        self._trace_peak_rate = max(self._trace_peak_rate, float(product_flux))
        return {
            "time_s": float(time_s),
            "bed_temperature_C": float(bed_K) - 273.15,
            "product_CO2_mol_s": float(product_flux),
            "product_CO2_mol": float(product_mol),
        }

    def observe(self, time_s: float, bed_K: float, product_flux: float,
                product_mol: float) -> None:
        """Record one accepted endpoint. This is never called from solve_ivp."""
        self._samples.append(
            self._new_sample(time_s, bed_K, product_flux, product_mol)
        )

    def _window_metrics(self, time_s: float) -> dict:
        cutoff = time_s - self.control.tail_window_s
        window = [row for row in self._samples if row["time_s"] >= cutoff - 1e-12]
        if not window:
            return {
                "available": False,
                "reason": "no checkpoint samples in the declared tail window",
            }
        first, last = window[0], window[-1]
        covered = first["time_s"] <= cutoff + self.control.checkpoint_s + 1e-9
        required_samples = int(round(self.control.tail_window_s / self.control.checkpoint_s)) + 1
        threshold_K = self.target_bed_K - self.control.bed_temperature_margin_K
        tail_mass = last["product_CO2_mol"] - first["product_CO2_mol"]
        tail_mass_limit = self.control.tail_mass_fraction_of_product * last["product_CO2_mol"]
        checks = {
            "window_covered": covered,
            "sample_count": len(window) >= required_samples,
            "bed_temperature_adequate": all(
                row["bed_temperature_C"] >= threshold_K - 273.15 for row in window
            ),
            "rate_bounded": all(
                row["product_CO2_mol_s"] <= self.control.tail_rate_limit_mol_s
                for row in window
            ),
            "mass_bounded": tail_mass <= tail_mass_limit + 1e-15,
        }
        return {
            "available": True,
            "start_s": first["time_s"],
            "end_s": last["time_s"],
            "declared_window_s": self.control.tail_window_s,
            "samples": len(window),
            "required_samples": required_samples,
            "max_tail_product_CO2_mol_s": max(
                row["product_CO2_mol_s"] for row in window
            ),
            "max_tail_bed_temperature_C": max(
                row["bed_temperature_C"] for row in window
            ),
            "min_tail_bed_temperature_C": min(
                row["bed_temperature_C"] for row in window
            ),
            "tail_mass_mol": tail_mass,
            "tail_mass_limit_mol": tail_mass_limit,
            "checks": checks,
            "all_pass": all(checks.values()),
        }

    def should_stop(self, time_s: float) -> bool:
        if not self.control.state_controlled:
            return False
        if time_s < self.control.minimum_duration_s - 1e-12:
            return False
        metrics = self._window_metrics(time_s)
        if metrics.get("all_pass"):
            self._trigger = {"time_s": time_s, **metrics}
            return True
        return False

    def record(self, *, actual_duration_s: float, trigger: str) -> dict:
        final = self._window_metrics(actual_duration_s)
        return {
            "schema": "desorption-control-record/1",
            "mode": self.control.mode,
            "trigger": trigger,
            "actual_duration_s": float(actual_duration_s),
            "minimum_duration_s": self.control.minimum_duration_s,
            "hard_max_duration_s": self.control.hard_max_duration_s,
            "checkpoint_s": self.control.checkpoint_s,
            "numerical_path": (
                "checkpointed" if self.control.mode in MODES else "legacy_single_integration"
            ),
            "decision_basis": (
                "accepted integration-chunk endpoints only; no RHS or event-callback "
                "mutation of controller history"
            ),
            "bed_temperature_target_C": self.target_bed_K - 273.15,
            "bed_temperature_margin_K": self.control.bed_temperature_margin_K,
            "bed_temperature_adequacy_C": (
                self.target_bed_K
                - self.control.bed_temperature_margin_K
                - 273.15
            ),
            "tail_window_s": self.control.tail_window_s,
            "tail_rate_limit_mol_s": self.control.tail_rate_limit_mol_s,
            "tail_mass_fraction_of_product": self.control.tail_mass_fraction_of_product,
            "trace_peak_product_CO2_mol_s": self._trace_peak_rate,
            "sample_count": len(self._samples),
            "trigger_metrics": self._trigger,
            "final_window_metrics": final,
            "hard_max_reached": trigger == "hard_max",
            "samples": self._samples,
        }
