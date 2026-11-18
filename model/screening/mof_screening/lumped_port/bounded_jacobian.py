"""Deterministic physical-column finite-difference Jacobian support."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np


def _canonical_array_sha(values: np.ndarray) -> str:
    data = np.asarray(values, dtype="<f8").tobytes(order="C")
    return hashlib.sha256(data).hexdigest()


@dataclass
class JacobianAudit:
    physical_columns: int
    temperature_index: int
    calls: int = 0
    RHS_evaluations: int = 0
    one_sided_steps: int = 0
    central_steps: int = 0
    fallback_steps: int = 0
    physical_to_ledger_row_nonzero: bool = False
    ledger_column_nonzero_entries: int = 0
    minimum_step_by_column: list | None = None
    maximum_step_by_column: list | None = None
    matrix_sha256: list | None = None

    def as_dict(self):
        return {
            "method": "deterministic bounded finite difference",
            "physical_columns": self.physical_columns,
            "temperature_index": self.temperature_index,
            "ledger_columns": "structurally-zero, never probed",
            "calls": self.calls,
            "RHS_evaluations": self.RHS_evaluations,
            "one_sided_steps": self.one_sided_steps,
            "central_steps": self.central_steps,
            "fallback_steps": self.fallback_steps,
            "physical_to_ledger_row_nonzero": self.physical_to_ledger_row_nonzero,
            "ledger_column_nonzero_entries": self.ledger_column_nonzero_entries,
            "minimum_step_by_column": self.minimum_step_by_column,
            "maximum_step_by_column": self.maximum_step_by_column,
            "matrix_sha256_by_call": self.matrix_sha256,
        }


def bounded_physical_jacobian(
    rhs,
    time,
    state,
    *,
    physical_columns: int,
    temperature_index: int | None = None,
    audit: JacobianAudit | None = None,
    base_relative_step: float = 2e-6,
    minimum_relative_step: float = 2e-9,
):
    """Differentiate physical columns; cumulative ledger columns are zero."""
    y = np.asarray(state, dtype=float)
    f0 = np.asarray(rhs(time, y), dtype=float)
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(f0)):
        raise FloatingPointError("base RHS/state must be finite for Jacobian")
    n = len(y)
    p = int(physical_columns)
    if not 0 < p <= n:
        raise ValueError("physical column count outside state size")
    if temperature_index is None:
        temperature_index = p - 5
    temperature_index = int(temperature_index)
    if not 0 <= temperature_index < p:
        raise ValueError("temperature index outside physical state columns")
    jacobian = np.zeros((n, n), dtype=float)
    minimum_steps = np.full(p, np.inf)
    maximum_steps = np.zeros(p, dtype=float)
    if audit is not None:
        audit.calls += 1
    for column in range(p):
        value = float(y[column])
        if column < temperature_index:
            scale_floor = 0.1
        elif column == temperature_index:
            scale_floor = 250.0
        else:
            scale_floor = 1e-3
        relative_scale = max(abs(value), scale_floor)
        step = base_relative_step * relative_scale
        temperature_column = column == temperature_index
        used = None
        for fallback in range(7):
            trial_step = step * (0.5 ** fallback)
            if trial_step < minimum_relative_step * relative_scale:
                break
            positive = y.copy()
            positive[column] += trial_step
            can_central = (
                temperature_column
                and value - trial_step >= 250.0
            ) or (
                not temperature_column
                and value - trial_step >= 0.0
            )
            f_positive = np.asarray(rhs(time, positive), dtype=float)
            if audit is not None:
                audit.RHS_evaluations += 1
            if not np.all(np.isfinite(f_positive)):
                continue
            if can_central:
                negative = y.copy()
                negative[column] -= trial_step
                f_negative = np.asarray(rhs(time, negative), dtype=float)
                if audit is not None:
                    audit.RHS_evaluations += 1
                if not np.all(np.isfinite(f_negative)):
                    continue
                jacobian[:, column] = (f_positive - f_negative) / (
                    2.0 * trial_step
                )
                if audit is not None:
                    audit.central_steps += 1
            else:
                jacobian[:, column] = (f_positive - f0) / trial_step
                if audit is not None:
                    audit.one_sided_steps += 1
            used = trial_step
            if fallback and audit is not None:
                audit.fallback_steps += 1
            break
        if used is None:
            raise FloatingPointError(
                f"no finite bounded finite-difference probe for physical column {column}"
            )
        minimum_steps[column] = min(minimum_steps[column], used)
        maximum_steps[column] = max(maximum_steps[column], used)
    # Physical rows causing ledger derivatives are retained. Only the
    # independent cumulative-ledger columns have zero RHS influence.
    jacobian[:, p:] = 0.0
    if not np.all(np.isfinite(jacobian)):
        raise FloatingPointError("bounded Jacobian nonfinite")
    if audit is not None:
        audit.physical_to_ledger_row_nonzero = bool(
            audit.physical_to_ledger_row_nonzero
            or np.any(jacobian[p:, :p] != 0.0)
        )
        audit.ledger_column_nonzero_entries += int(
            np.count_nonzero(jacobian[:, p:])
        )
        if audit.minimum_step_by_column is None:
            audit.minimum_step_by_column = minimum_steps.tolist()
            audit.maximum_step_by_column = maximum_steps.tolist()
            audit.matrix_sha256 = []
        else:
            audit.minimum_step_by_column = np.minimum(
                audit.minimum_step_by_column, minimum_steps
            ).tolist()
            audit.maximum_step_by_column = np.maximum(
                audit.maximum_step_by_column, maximum_steps
            ).tolist()
        audit.matrix_sha256.append(_canonical_array_sha(jacobian))
    return jacobian


def jacobian_convergence(
    rhs,
    time,
    state,
    *,
    physical_columns,
    steps,
    temperature_index=None,
):
    """Check Jacobian convergence over caller-supplied relative steps."""
    matrices = []
    for step in steps:
        effective_temperature_index = (
            physical_columns - 5
            if temperature_index is None
            else int(temperature_index)
        )
        audit = JacobianAudit(
            physical_columns=physical_columns,
            temperature_index=effective_temperature_index,
        )
        matrices.append(
            bounded_physical_jacobian(
                rhs,
                time,
                state,
                physical_columns=physical_columns,
                temperature_index=effective_temperature_index,
                audit=audit,
                base_relative_step=float(step),
            )
        )
    reference = matrices[-1]
    return {
        "steps": [float(value) for value in steps],
        "finite": all(np.all(np.isfinite(matrix)) for matrix in matrices),
        "ledger_columns_zero": all(
            np.all(matrix[:, physical_columns:] == 0.0) for matrix in matrices
        ),
        "relative_change_from_finest": [
            float(
                np.linalg.norm(matrix - reference)
                / max(np.linalg.norm(reference), 1e-30)
            )
            for matrix in matrices[:-1]
        ],
        "matrix_sha256": [_canonical_array_sha(matrix) for matrix in matrices],
    }
