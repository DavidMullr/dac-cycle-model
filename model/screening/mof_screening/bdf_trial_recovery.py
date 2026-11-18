"""Opt-in BDF recovery for infeasible near-zero water trial states.

This module wraps one solver's accepted history. It does not patch SciPy or
project states. A recognized material-rate guard failure restores the exact
pre-step physical state and history, retains the accepted-history Jacobian,
discards the cached LU factorization, reduces from the actual failed trial
step, and retries the internal BDF step a bounded number of times.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import math
from typing import Any

import numpy as np
import scipy
from scipy.integrate._ivp import bdf as scipy_bdf


POLICY_SCHEMA = "bdf-water-trial-recovery/1"
SUPPORTED_SCIPY = (1, 17)


class BDFRecoveryConfigurationError(RuntimeError):
    """The wrapped SciPy BDF implementation is not the reviewed version."""


class BDFRecoveryExhausted(RuntimeError):
    """A recoverable dry-water trial exceeded its bounded recovery budget."""

    def __init__(self, evidence: dict):
        self.evidence = evidence
        super().__init__(
            "BDF near-zero water trial recovery exhausted "
            f"after {evidence['attempts']} attempts at t="
            f"{evidence['local_time_s']!r}"
        )


def _array_hash(values) -> str:
    return sha256(
        np.ascontiguousarray(np.asarray(values, dtype="<f8")).tobytes(
            order="C"
        )
    ).hexdigest()


def _supported_scipy() -> dict:
    version = tuple(
        int(part)
        for part in scipy.__version__.split(".")[:2]
        if part.isdigit()
    )
    required_methods = ("step", "_step_impl")
    missing = [
        name for name in required_methods if not hasattr(scipy_bdf.BDF, name)
    ]
    if version != SUPPORTED_SCIPY or missing:
        raise BDFRecoveryConfigurationError(
            "unsupported SciPy BDF implementation: "
            f"version={scipy.__version__} missing={missing}"
        )
    return {
        "scipy_version": scipy.__version__,
        "supported_major_minor": list(SUPPORTED_SCIPY),
        "required_methods_present": True,
        "history_attributes_checked_on_instantiation": True,
    }


@dataclass
class BDFRecoveryTracker:
    policy: dict
    scipy_support: dict = field(default_factory=dict)
    recovery_attempt_count: int = 0
    recovered_trial_count: int = 0
    exhaustion_count: int = 0
    rejected_trial_events: list[dict] = field(default_factory=list)
    accepted_step_samples: list[dict] = field(default_factory=list)
    recovery_events: list[dict] = field(default_factory=list)
    fatal_events: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "schema": POLICY_SCHEMA,
            "policy": self.policy,
            "scipy_support": self.scipy_support,
            "recovery_attempt_count": self.recovery_attempt_count,
            "recovered_trial_count": self.recovered_trial_count,
            "exhaustion_count": self.exhaustion_count,
            "rejected_trial_count": len(self.rejected_trial_events),
            "accepted_step_sample_count": len(self.accepted_step_samples),
            "rejected_trial_events": self.rejected_trial_events,
            "accepted_step_samples": self.accepted_step_samples,
            "recovery_events": self.recovery_events,
            "fatal_events": self.fatal_events,
            "terminal_failure_retained": bool(self.fatal_events),
            "accepted_states_projected": False,
            "accepted_states_mutated": False,
        }


def _policy_for(material) -> dict | None:
    raw = material.data.get("water_evaluation_policy", {}).get(
        "bdf_trial_recovery"
    )
    if not raw:
        return None
    if raw.get("schema") != POLICY_SCHEMA:
        raise BDFRecoveryConfigurationError(
            "unsupported BDF trial recovery policy schema"
        )
    max_attempts = int(raw.get("max_recoveries_per_accepted_step", 0))
    factor = float(raw.get("step_reduction_factor", 0.0))
    numerical = material.data.get("water_evaluation_policy", {}).get(
        "numerical_tolerance_policy"
    )
    if (
        max_attempts < 1
        or max_attempts > 8
        or math.isclose(factor, 0.5) is False
        or not isinstance(numerical, dict)
        or numerical.get("name") != "scale_aware_5atol_v1"
        or numerical.get("component_atol_mol_kg") != [1e-9, 1e-9]
        or numerical.get("continuation_boundary_mol_kg") != -5e-9
        or float(raw.get("water_continuation_boundary_mol_kg", 0.0))
        != -5e-9
    ):
        raise BDFRecoveryConfigurationError(
            "BDF trial recovery bounds or policy metadata are invalid"
        )
    if material.n_storage != 2 or material.water_storage_index != 1:
        raise BDFRecoveryConfigurationError(
            "BDF trial recovery requires two storage components with water "
            "at index 1"
        )
    return dict(raw)


def _recoverable(
    error: Exception,
    *,
    accepted_state,
    adapter,
    layout,
    policy: dict,
    model=None,
) -> tuple[bool, dict]:
    diagnostics = getattr(error, "diagnostics", None)
    if not isinstance(diagnostics, dict):
        return False, {"reason": "missing_diagnostics"}
    if (
        diagnostics.get("adapter") != "young2021_advisory_wadst_gab"
        or diagnostics.get("call") != "rates"
        or "storage components must be nonnegative" not in str(error)
    ):
        return False, {"reason": "not_water_loading_guard"}
    try:
        cell = int(diagnostics["cell"])
        query_state = np.asarray(
            diagnostics["query_state_vector"], dtype=float
        )
        temperature_K = float(diagnostics["temperature_K"])
        partials = np.asarray(diagnostics["partials_Pa"], dtype=float)
    except (KeyError, TypeError, ValueError):
        return False, {"reason": "missing_query_identity"}
    if (
        accepted_state.shape != query_state.shape
        or query_state.shape != (layout.size,)
        or not np.all(np.isfinite(accepted_state))
        or not np.all(np.isfinite(query_state))
        or not math.isfinite(temperature_K)
        or partials.shape != (4,)
        or not np.all(np.isfinite(partials))
        or np.any(partials < 0.0)
    ):
        return False, {"reason": "invalid_query_or_state"}
    raw = np.asarray(diagnostics.get("raw_storage_mol_kg"), dtype=float)
    if raw.shape != (2,) or not np.all(np.isfinite(raw)):
        return False, {"reason": "invalid_failed_cell_storage"}
    boundary = -abs(
        float(policy["water_continuation_boundary_mol_kg"])
    )
    if raw[0] < boundary or raw[1] >= boundary:
        return False, {"reason": "outside_water_trial_boundary"}
    storage_count = material_storage_count(adapter)
    n_cells = (layout.q.stop - layout.q.start) // storage_count
    if not 0 <= cell < n_cells:
        return False, {"reason": "cell_out_of_range"}
    accepted_q = np.asarray(accepted_state[layout.q], dtype=float).reshape(
        -1, storage_count
    )
    query_q = query_state[layout.q].reshape(-1, storage_count)
    if accepted_q.shape[0] != query_q.shape[0]:
        return False, {"reason": "storage_layout_mismatch"}
    if not np.array_equal(raw, query_q[cell]):
        return False, {
            "reason": "failed_cell_storage_metadata_mismatch",
            "raw_storage_mol_kg": raw.tolist(),
            "query_cell_storage_mol_kg": query_q[cell].tolist(),
        }
    accepted_cell_water = float(
        accepted_q[cell, adapter.water_storage_index]
    )
    if not (
        math.isfinite(accepted_cell_water)
        and boundary <= accepted_cell_water <= -boundary
    ):
        return False, {
            "reason": "accepted_cell_not_near_zero",
            "accepted_cell_water_mol_kg": accepted_cell_water,
        }
    if (
        np.min(query_q[:, 0]) < boundary
    ):
        return False, {"reason": "another_component_outside_boundary"}
    if model is not None:
        from .spatial import check_inventory

        try:
            check_inventory(query_state, model, where="rejected BDF trial")
            gas = np.maximum(
                query_state[layout.n_gas].reshape(model.n_cells, 4), 0.0
            )
            total = gas.sum(axis=1)
            derived_partials = (
                gas
                / np.maximum(total[:, None], 1e-30)
                * (total * 8.31446261815324)[:, None]
                * query_state[layout.T_gas][:, None]
                / model.cell_gas_volume_m3[:, None]
            )
            if (
                not math.isclose(
                    temperature_K,
                    float(query_state[layout.T_bed][cell]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or not np.allclose(
                    partials,
                    derived_partials[cell],
                    rtol=1e-10,
                    atol=1e-15,
                )
            ):
                return False, {
                    "reason": "query_metadata_state_mismatch"
                }
        except Exception as error_value:
            return False, {
                "reason": "rejected_query_inventory_invalid",
                "error": str(error_value),
            }
    try:
        from . import young2021
        from . import young2021_advisory_temperature

        local_rh = young2021_advisory_temperature.local_relative_humidity(
            temperature_K, float(partials[1])
        )
        if not math.isfinite(local_rh) or not 0.0 <= local_rh < 1.0:
            return False, {"reason": "invalid_relative_humidity"}
        gab = young2021.gab_water_loading(local_rh, temperature_K)
        if not math.isfinite(gab) or gab < 0.0:
            return False, {"reason": "invalid_gab_branch"}
        young2021_advisory_temperature.equilibrium(
            temperature_K,
            float(partials[0]),
            float(partials[1]),
            water_loading_mol_kg=0.0,
            reference_temperature_K=adapter.reference_temperature_K,
        )
    except Exception as error_value:
        return False, {
            "reason": "invalid_constitutive_inputs",
            "error": str(error_value),
        }
    return True, {
        "reason": "valid_finite_dry_water_trial",
        "cell": cell,
        "accepted_cell_water_mol_kg": accepted_cell_water,
        "trial_water_mol_kg": float(raw[1]),
        "temperature_K": temperature_K,
        "partials_Pa": partials.tolist(),
        "local_relative_humidity": float(local_rh),
        "gab_water_target_mol_kg": float(gab),
    }


def material_storage_count(adapter) -> int:
    return int(getattr(adapter, "water_storage_index", 1)) + 1


def make_recovering_bdf(
    material,
    layout,
    model=None,
):
    """Return a reviewed BDF class and tracker, or the stock BDF class."""
    policy = _policy_for(material)
    if policy is None:
        return scipy_bdf.BDF, None
    support = _supported_scipy()
    tracker = BDFRecoveryTracker(policy=policy, scipy_support=support)
    adapter = material

    class RecoveringBDF(scipy_bdf.BDF):
        def __init__(self, fun, *args, **kwargs):
            self._last_bdf_query_time = None

            def observed_fun(time_value, state_value):
                self._last_bdf_query_time = float(time_value)
                return fun(time_value, state_value)

            super().__init__(observed_fun, *args, **kwargs)
            required = (
                "t",
                "t_old",
                "y",
                "D",
                "order",
                "h_abs",
                "h_abs_old",
                "error_norm_old",
                "n_equal_steps",
                "J",
                "LU",
                "status",
                "n",
            )
            missing = [
                name for name in required if not hasattr(self, name)
            ]
            if missing:
                raise BDFRecoveryConfigurationError(
                    "SciPy BDF history attributes changed: "
                    + ", ".join(missing)
                )

        def _snapshot(self) -> dict:
            return {
                "t": self.t,
                "t_old": self.t_old,
                "y": self.y.copy(),
                "D": self.D.copy(),
                "order": self.order,
                "h_abs": self.h_abs,
                "h_abs_old": self.h_abs_old,
                "error_norm_old": self.error_norm_old,
                "n_equal_steps": self.n_equal_steps,
                "J": None if self.J is None else self.J.copy(),
                "LU": self.LU,
                "status": self.status,
                "n": self.n,
            }

        def _restore(self, snapshot: dict, *, reduced_h_abs: float) -> None:
            self.t = snapshot["t"]
            self.t_old = snapshot["t_old"]
            self.y = snapshot["y"].copy()
            self.D = snapshot["D"].copy()
            if snapshot["h_abs"] > 0.0 and reduced_h_abs != snapshot["h_abs"]:
                scipy_bdf.change_D(
                    self.D,
                    snapshot["order"],
                    reduced_h_abs / snapshot["h_abs"],
                )
            self.order = snapshot["order"]
            self.h_abs = reduced_h_abs
            self.h_abs_old = None
            self.error_norm_old = None
            self.n_equal_steps = 0
            self.J = snapshot["J"].copy()
            self.LU = None
            self.status = snapshot["status"]
            self.n = snapshot["n"]

        def step(self):
            snapshot = self._snapshot()
            history_hash = _array_hash(snapshot["D"])
            accepted_hash = _array_hash(snapshot["y"])
            accepted_water_min = float(
                np.min(
                    snapshot["y"][layout.q]
                    .reshape(-1, material_storage_count(adapter))[
                        :, adapter.water_storage_index
                    ]
                )
            )
            attempts_at_time = 0
            attempted_h_abs = float(snapshot["h_abs"])
            while True:
                try:
                    message = super().step()
                except Exception as error:
                    eligible, eligibility = _recoverable(
                        error,
                        accepted_state=snapshot["y"],
                        adapter=adapter,
                        layout=layout,
                        policy=policy,
                        model=model,
                    )
                    if not eligible:
                        tracker.fatal_events.append(
                            {
                                "local_time_s": float(snapshot["t"]),
                                "stage": "unrecoverable_exception",
                                "error_type": type(error).__name__,
                                "error": str(error),
                                "diagnostics": getattr(
                                    error, "diagnostics", None
                                ),
                                "eligibility": eligibility,
                            }
                        )
                        raise
                    attempts_at_time += 1
                    tracker.recovery_attempt_count += 1
                    diagnostics = getattr(error, "diagnostics", {})
                    raw = diagnostics.get("raw_storage_mol_kg")
                    actual_attempt_h = (
                        None
                        if self._last_bdf_query_time is None
                        else abs(
                            float(self._last_bdf_query_time)
                            - float(snapshot["t"])
                        )
                    )
                    if (
                        actual_attempt_h is None
                        or not math.isfinite(actual_attempt_h)
                        or actual_attempt_h <= 0.0
                    ):
                        tracker.fatal_events.append(
                            {
                                "local_time_s": float(snapshot["t"]),
                                "stage": "missing_actual_failed_step",
                                "attempted_h_abs_s": attempted_h_abs,
                            }
                        )
                        raise BDFRecoveryConfigurationError(
                            "recoverable BDF failure lacks a valid actual "
                            "failed query time"
                        ) from error
                    rejection = {
                        "local_time_s": float(snapshot["t"]),
                        "attempt": attempts_at_time,
                        "accepted_state_sha256": accepted_hash,
                        "accepted_history_sha256": history_hash,
                        "accepted_water_min_mol_kg": accepted_water_min,
                        "trial_raw_storage_mol_kg": raw,
                        "trial_water_mol_kg": (
                            None
                            if not isinstance(raw, list) or len(raw) < 2
                            else float(raw[1])
                        ),
                        "attempted_h_abs_s": attempted_h_abs,
                        "actual_failed_h_abs_s": actual_attempt_h,
                        "bdf_local_query_time_s": float(
                            self._last_bdf_query_time
                        ),
                        "order": int(snapshot["order"]),
                        "classification": (
                            "recoverable_near_zero_water_trial"
                        ),
                        "error": str(error),
                        "diagnostics": diagnostics,
                        "eligibility": eligibility,
                    }
                    tracker.rejected_trial_events.append(rejection)
                    if attempts_at_time > int(
                        policy["max_recoveries_per_accepted_step"]
                    ):
                        evidence = {
                            **rejection,
                            "attempts": attempts_at_time,
                        }
                        tracker.exhaustion_count += 1
                        tracker.fatal_events.append(
                            {
                                **evidence,
                                "stage": "recovery_exhausted",
                            }
                        )
                        self._restore(
                            snapshot,
                            reduced_h_abs=attempted_h_abs,
                        )
                        raise BDFRecoveryExhausted(evidence) from error
                    reduced = min(attempted_h_abs, actual_attempt_h) * float(
                        policy["step_reduction_factor"]
                    )
                    self._restore(snapshot, reduced_h_abs=reduced)
                    tracker.recovered_trial_count += 1
                    recovery = {
                        **rejection,
                        "restored_accepted_state": True,
                        "restored_accepted_history": True,
                        "new_h_abs_s": reduced,
                        "cached_jacobian_retained_from_accepted_history": True,
                        "cached_lu_discarded": True,
                        "state_projection": False,
                    }
                    tracker.recovery_events.append(recovery)
                    attempted_h_abs = reduced
                    acknowledge = getattr(
                        adapter, "acknowledge_recovered_trial", None
                    )
                    if callable(acknowledge):
                        acknowledge(recovery)
                    continue
                accepted_min = float(
                    np.min(
                        self.y[layout.q]
                        .reshape(-1, material_storage_count(adapter))[
                            :, adapter.water_storage_index
                        ]
                    )
                )
                accepted_q = self.y[layout.q].reshape(
                    -1, material_storage_count(adapter)
                )
                if self.status == "failed":
                    failed_message = str(message)
                    candidate_state = self.y.copy()
                    candidate_time = float(self.t)
                    candidate_hash = _array_hash(candidate_state)
                    self._restore(
                        snapshot,
                        reduced_h_abs=snapshot["h_abs"],
                    )
                    tracker.fatal_events.append(
                        {
                            "local_time_s": float(snapshot["t"]),
                            "candidate_local_time_s": candidate_time,
                            "stage": "stock_solver_failed",
                            "message": failed_message,
                            "accepted_state_sha256": accepted_hash,
                            "candidate_state_sha256": candidate_hash,
                            "candidate_state_vector": candidate_state.tolist(),
                            "restored_accepted_state": True,
                        }
                    )
                    raise RuntimeError(
                        "BDF stock solver failed: " + failed_message
                    )
                if not np.all(np.isfinite(self.y)):
                    candidate_state = self.y.copy()
                    candidate_time = float(self.t)
                    candidate_hash = _array_hash(candidate_state)
                    nonfinite_indices = np.flatnonzero(
                        ~np.isfinite(candidate_state)
                    ).tolist()
                    self._restore(
                        snapshot,
                        reduced_h_abs=snapshot["h_abs"],
                    )
                    tracker.fatal_events.append(
                        {
                            "local_time_s": float(snapshot["t"]),
                            "candidate_local_time_s": candidate_time,
                            "stage": "accepted_nonfinite_storage",
                            "candidate_state_sha256": candidate_hash,
                            "candidate_nonfinite_indices": nonfinite_indices,
                            "accepted_state_sha256": accepted_hash,
                            "restored_accepted_state": True,
                        }
                    )
                    raise RuntimeError(
                        "BDF produced a nonfinite candidate state"
                    )
                continuation = float(policy["water_continuation_boundary_mol_kg"])
                candidate_co2_invalid = bool(
                    np.min(accepted_q[:, 0]) < continuation
                )
                candidate_water_invalid = accepted_min < continuation
                if model is not None:
                    from .spatial import check_inventory

                    try:
                        check_inventory(
                            self.y, model, where="candidate accepted step"
                        )
                    except Exception as error:
                        candidate_co2_invalid = True
                        inventory_error = str(error)
                    else:
                        inventory_error = None
                else:
                    inventory_error = None
                if candidate_co2_invalid:
                    candidate_state = self.y.copy()
                    candidate_time = float(self.t)
                    candidate_hash = _array_hash(candidate_state)
                    self._restore(
                        snapshot,
                        reduced_h_abs=snapshot["h_abs"],
                    )
                    tracker.fatal_events.append(
                        {
                            "local_time_s": float(snapshot["t"]),
                            "candidate_local_time_s": candidate_time,
                            "stage": "invalid_candidate_inventory_or_co2",
                            "error": inventory_error
                            or "candidate CO2 crossed boundary",
                            "candidate_state_sha256": candidate_hash,
                            "candidate_state_vector": candidate_state.tolist(),
                            "accepted_state_sha256": accepted_hash,
                            "restored_accepted_state": True,
                        }
                    )
                    raise RuntimeError(
                        inventory_error
                        or "BDF candidate CO2 crossed the component boundary"
                    )
                if candidate_water_invalid:
                    bad_cells = np.flatnonzero(
                        accepted_q[:, adapter.water_storage_index]
                        < continuation
                    )
                    if len(bad_cells) != 1:
                        candidate_state = self.y.copy()
                        candidate_time = float(self.t)
                        candidate_hash = _array_hash(candidate_state)
                        self._restore(
                            snapshot,
                            reduced_h_abs=snapshot["h_abs"],
                        )
                        tracker.fatal_events.append(
                            {
                                "local_time_s": float(snapshot["t"]),
                                "candidate_local_time_s": candidate_time,
                                "stage": "multiple_candidate_water_cells",
                                "bad_cells": bad_cells.tolist(),
                                "candidate_state_sha256": candidate_hash,
                                "candidate_state_vector": candidate_state.tolist(),
                                "accepted_state_sha256": accepted_hash,
                                "restored_accepted_state": True,
                            }
                        )
                        raise RuntimeError(
                            "candidate water boundary failed in multiple cells"
                        )
                    cell = int(bad_cells[0])
                    accepted_water = float(
                        np.asarray(
                            snapshot["y"][layout.q], dtype=float
                        ).reshape(-1, material_storage_count(adapter))[
                            cell, adapter.water_storage_index
                        ]
                    )
                    if not (
                        continuation
                        <= accepted_water
                        <= -continuation
                    ):
                        candidate_state = self.y.copy()
                        candidate_time = float(self.t)
                        candidate_hash = _array_hash(candidate_state)
                        self._restore(
                            snapshot,
                            reduced_h_abs=snapshot["h_abs"],
                        )
                        tracker.fatal_events.append(
                            {
                                "local_time_s": float(snapshot["t"]),
                                "candidate_local_time_s": candidate_time,
                                "stage": "invalid_candidate_dry_boundary",
                                "cell": cell,
                                "accepted_cell_water_mol_kg": accepted_water,
                                "candidate_state_sha256": candidate_hash,
                                "candidate_state_vector": candidate_state.tolist(),
                                "accepted_state_sha256": accepted_hash,
                                "restored_accepted_state": True,
                            }
                        )
                        raise RuntimeError(
                            "candidate cell is not at a valid dry accepted "
                            "boundary"
                        )
                    attempts_at_time += 1
                    tracker.recovery_attempt_count += 1
                    rejection = {
                        "local_time_s": float(self.t),
                        "attempt": attempts_at_time,
                        "accepted_state_sha256": accepted_hash,
                        "accepted_history_sha256": history_hash,
                        "attempted_h_abs_s": attempted_h_abs,
                        "order": int(snapshot["order"]),
                        "cell": cell,
                        "accepted_cell_water_mol_kg": accepted_water,
                        "candidate_accepted_state_sha256": _array_hash(self.y),
                        "candidate_accepted_qwater_min_mol_kg": accepted_min,
                        "classification": (
                            "recoverable_infeasible_candidate_accepted_step"
                        ),
                    }
                    tracker.rejected_trial_events.append(rejection)
                    if attempts_at_time > int(
                        policy["max_recoveries_per_accepted_step"]
                    ):
                        evidence = {
                            **rejection,
                            "attempts": attempts_at_time,
                        }
                        tracker.exhaustion_count += 1
                        tracker.fatal_events.append(
                            {
                                **evidence,
                                "stage": "recovery_exhausted",
                            }
                        )
                        self._restore(
                            snapshot,
                            reduced_h_abs=attempted_h_abs,
                        )
                        raise BDFRecoveryExhausted(evidence)
                    actual_attempt_h = abs(float(self.t) - float(snapshot["t"]))
                    if not math.isfinite(actual_attempt_h) or actual_attempt_h <= 0.0:
                        self._restore(
                            snapshot,
                            reduced_h_abs=snapshot["h_abs"],
                        )
                        raise BDFRecoveryConfigurationError(
                            "infeasible candidate lacks a valid actual step size"
                        )
                    rejection["actual_failed_h_abs_s"] = actual_attempt_h
                    reduced = min(attempted_h_abs, actual_attempt_h) * float(
                        policy["step_reduction_factor"]
                    )
                    self._restore(snapshot, reduced_h_abs=reduced)
                    tracker.recovered_trial_count += 1
                    recovery = {
                        **rejection,
                        "restored_accepted_state": True,
                        "restored_accepted_history": True,
                        "new_h_abs_s": reduced,
                        "cached_jacobian_retained_from_accepted_history": True,
                        "cached_lu_discarded": True,
                        "state_projection": False,
                    }
                    tracker.recovery_events.append(recovery)
                    attempted_h_abs = reduced
                    continue
                tracker.accepted_step_samples.append(
                    {
                        "local_time_s": float(self.t),
                        "accepted_qwater_min_mol_kg": accepted_min,
                        "accepted_negative": bool(accepted_min < 0.0),
                        "state_sha256": _array_hash(self.y),
                        "status": self.status,
                    }
                )
                return message

    return RecoveringBDF, tracker


def merge_trackers(trackers) -> dict:
    rows = [tracker.as_dict() for tracker in trackers if tracker is not None]
    return {
        "schema": "bdf-water-trial-recovery-history/1",
        "tracker_count": len(rows),
        "recovery_attempt_count": sum(
            row["recovery_attempt_count"] for row in rows
        ),
        "recovered_trial_count": sum(
            row["recovered_trial_count"] for row in rows
        ),
        "exhaustion_count": sum(row["exhaustion_count"] for row in rows),
        "rejected_trial_count": sum(
            row["rejected_trial_count"] for row in rows
        ),
        "accepted_step_sample_count": sum(
            row["accepted_step_sample_count"] for row in rows
        ),
        "terminal_failure_retained": any(
            row["terminal_failure_retained"] for row in rows
        ),
        "trackers": rows,
    }
