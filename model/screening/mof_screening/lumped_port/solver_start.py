"""Opt-in positivity-aware first step for stiff finite-gas integrations."""
from __future__ import annotations

import numpy as np
from scipy.integrate._ivp.common import norm


INITIAL_STEP_POLICY = "positive_gas_depletion_bound_v1"
DEFAULT_INITIAL_STEP_MAX_S = 1e-6
DEFAULT_DEPLETION_FRACTION = 0.1


def initial_step_identity(config: dict) -> dict:
    solver = config.get("solver", {})
    return {
        "policy": solver.get("initial_step_policy"),
        "maximum_first_step_s": solver.get("initial_step_max_s"),
        "gas_depletion_fraction": solver.get(
            "initial_step_depletion_fraction"
        ),
        "first_step_default": None,
    }


def apply_initial_step_policy(
    config: dict,
    *,
    maximum_first_step_s: float = DEFAULT_INITIAL_STEP_MAX_S,
    depletion_fraction: float = DEFAULT_DEPLETION_FRACTION,
) -> dict:
    from copy import deepcopy

    result = deepcopy(config)
    result.setdefault("solver", {}).update(
        {
            "initial_step_policy": INITIAL_STEP_POLICY,
            "initial_step_max_s": float(maximum_first_step_s),
            "initial_step_depletion_fraction": float(depletion_fraction),
        }
    )
    return result


def positive_gas_first_step(
    rhs,
    time: float,
    state,
    t_bound: float,
    *,
    max_step_s: float,
    rtol: float,
    atol,
    gas_indices,
    maximum_first_step_s: float = DEFAULT_INITIAL_STEP_MAX_S,
    depletion_fraction: float = DEFAULT_DEPLETION_FRACTION,
):
    """Choose a positive finite first step before any trial state is accepted."""
    y0 = np.asarray(state, dtype=float)
    atol = np.asarray(atol, dtype=float)
    interval = abs(float(t_bound) - float(time))
    if y0.size != atol.size or not np.all(np.isfinite(y0)):
        raise FloatingPointError("invalid state/layout for first-step policy")
    if interval <= 0.0 or max_step_s <= 0.0:
        raise ValueError("positive integration interval and max step required")
    if maximum_first_step_s <= 0.0 or not 0.0 < depletion_fraction <= 1.0:
        raise ValueError("invalid first-step policy constants")
    f0 = np.asarray(rhs(time, y0), dtype=float)
    if not np.all(np.isfinite(f0)):
        raise FloatingPointError("nonfinite starting RHS")

    scale = atol + np.abs(y0) * rtol
    d0 = float(norm(y0 / scale))
    d1 = float(norm(f0 / scale))
    default_h0 = 1e-6 if d0 < 1e-5 or d1 < 1e-5 else 0.01 * d0 / d1
    # SciPy bounds h0 by the integration interval before its Euler probe;
    # max_step enters only the final returned-step bound.
    default_h0 = min(default_h0, interval)
    default_trial = y0 + default_h0 * f0
    gas0 = y0[gas_indices]
    gas_rate = f0[gas_indices]
    trial_gas = default_trial[gas_indices]

    depletion_times = []
    for inventory, rate in zip(gas0, gas_rate):
        if inventory < 0.0:
            raise FloatingPointError(
                "negative starting gas inventory"
            )
        if inventory == 0.0 and rate < 0.0:
            raise FloatingPointError(
                "zero starting gas inventory with negative derivative"
            )
        depletion_times.append(
            float(inventory / (-rate)) if rate < 0.0 else float("inf")
        )
    minimum_depletion = min(depletion_times)
    depletion_bound = (
        depletion_fraction * minimum_depletion
        if np.isfinite(minimum_depletion)
        else float("inf")
    )
    first_step = min(
        maximum_first_step_s,
        interval,
        max_step_s,
        depletion_bound,
    )
    if not np.isfinite(first_step) or first_step <= 0.0:
        raise FloatingPointError("nonpositive or nonfinite selected first step")
    trial = y0 + first_step * f0
    trial_gas_selected = trial[gas_indices]
    stationary_zero = (
        (gas0 == 0.0)
        & (gas_rate == 0.0)
        & (trial_gas_selected == 0.0)
    )
    strictly_positive = (
        (gas0 > 0.0) & (trial_gas_selected > 0.0)
    )
    refilling_zero = (
        (gas0 == 0.0)
        & (gas_rate > 0.0)
        & (trial_gas_selected > 0.0)
    )
    if (
        not np.all(np.isfinite(trial))
        or not np.all(np.isfinite(trial_gas_selected))
        or not np.all(
            stationary_zero | strictly_positive | refilling_zero
        )
    ):
        raise FloatingPointError("selected first-step trial leaves gas positivity")
    trial_rhs = np.asarray(
        rhs(time + first_step, trial), dtype=float
    )
    if not np.all(np.isfinite(trial_rhs)):
        raise FloatingPointError("selected first-step trial RHS is nonfinite")

    limiting = int(
        np.argmin(depletion_times)
        if np.isfinite(minimum_depletion)
        else -1
    )
    return {
        "policy": INITIAL_STEP_POLICY,
        "first_step_s": float(first_step),
        "interval_s": float(interval),
        "max_step_s": float(max_step_s),
        "maximum_first_step_s": float(maximum_first_step_s),
        "depletion_fraction": float(depletion_fraction),
        "minimum_gas_depletion_s": (
            float(minimum_depletion)
            if np.isfinite(minimum_depletion)
            else None
        ),
        "depletion_bound_s": (
            float(depletion_bound) if np.isfinite(depletion_bound) else None
        ),
        "limiting_gas_index": (
            int(gas_indices.start + limiting) if limiting >= 0 else None
        ),
        "default_h0_s": float(default_h0),
        "default_trial_gas_positive": bool(np.all(trial_gas > 0.0)),
        "selected_trial_gas_positive": bool(
            np.all(trial_gas_selected > 0.0)
        ),
        "selected_trial_gas_admissible": bool(
            np.all(
                stationary_zero | strictly_positive | refilling_zero
            )
        ),
        "accepted_gas_mol": gas0.tolist(),
        "gas_derivative_mol_s": gas_rate.tolist(),
        "trial_gas_mol": trial_gas_selected.tolist(),
        "stationary_zero_mask": stationary_zero.tolist(),
        "selected_trial_rhs_finite": True,
        "starting_rhs_sha256": _array_sha(f0),
        "selected_trial_state_sha256": _array_sha(trial),
    }


def _array_sha(values) -> str:
    import hashlib

    return hashlib.sha256(
        np.asarray(values, dtype="<f8").tobytes(order="C")
    ).hexdigest()
