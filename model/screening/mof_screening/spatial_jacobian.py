"""Deterministic physical-column Jacobian for the spatial screening engine."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np


_ERROR_DETAIL_KEYS = (
    "time_s",
    "column",
    "step",
    "state_sha256",
    "rhs_sha256",
    "positive_finite",
    "negative_finite",
    "finite_count",
    "nonfinite_count",
    "rhs_finite_count",
    "rhs_size",
    "physical_columns",
    "state_shape",
    "layout_size",
)


def _safe_error_value(value) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0.0 else "-Infinity"
        return repr(value)
    if isinstance(value, (bool, int)):
        return str(value)
    if value is None:
        return "None"
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_safe_error_value(item) for item in value) + "]"
    return repr(value)


class SpatialJacobianError(ValueError):
    """A spatial Jacobian could not be formed from finite probes."""

    def __init__(self, message: str, *, evidence: dict | None = None):
        self.evidence = evidence or {}
        details = []
        for key in _ERROR_DETAIL_KEYS:
            if key in self.evidence:
                details.append(
                    f"{key}={_safe_error_value(self.evidence[key])}"
                )
        rendered = message
        if details:
            rendered = f"{message} [{', '.join(details)}]"
        super().__init__(rendered)


def _array_sha256(values) -> str:
    data = np.ascontiguousarray(np.asarray(values, dtype="<f8")).tobytes(
        order="C"
    )
    return hashlib.sha256(data).hexdigest()


@dataclass
class SpatialJacobianAudit:
    physical_columns: int
    ledger_columns: int
    calls: int = 0
    rhs_evaluations: int = 0
    one_sided_steps: int = 0
    central_steps: int = 0
    fallback_steps: int = 0
    physical_to_ledger_row_nonzero: bool = False
    ledger_column_nonzero_entries: int = 0
    minimum_step_by_column: list | None = None
    maximum_step_by_column: list | None = None
    matrix_sha256: list | None = None

    def as_dict(self) -> dict:
        return {
            "method": "layout-aware deterministic bounded physical finite difference",
            "physical_columns": self.physical_columns,
            "ledger_columns": self.ledger_columns,
            "ledger_policy": (
                "independent ledger columns exactly zero; all physical columns "
                "and physical-to-ledger rows retained"
            ),
            "calls": self.calls,
            "rhs_evaluations": self.rhs_evaluations,
            "one_sided_steps": self.one_sided_steps,
            "central_steps": self.central_steps,
            "fallback_steps": self.fallback_steps,
            "physical_to_ledger_row_nonzero": (
                self.physical_to_ledger_row_nonzero
            ),
            "ledger_column_nonzero_entries": (
                self.ledger_column_nonzero_entries
            ),
            "minimum_step_by_column": self.minimum_step_by_column,
            "maximum_step_by_column": self.maximum_step_by_column,
            "matrix_sha256_by_call": self.matrix_sha256,
        }


def _physical_layout(layout) -> tuple[int, set[int], set[int]]:
    physical_columns = int(layout.inlet.start)
    if physical_columns <= 0 or physical_columns >= int(layout.size):
        raise SpatialJacobianError(
            "state layout has no separable physical prefix",
            evidence={
                "physical_columns": physical_columns,
                "state_size": int(layout.size),
            },
        )
    temperature_columns = {int(layout.T_h), int(layout.T_w)}
    temperature_columns.update(range(layout.T_bed.start, layout.T_bed.stop))
    temperature_columns.update(range(layout.T_gas.start, layout.T_gas.stop))
    nonnegative_columns = set(range(layout.q.start, layout.q.stop))
    nonnegative_columns.update(range(layout.n_gas.start, layout.n_gas.stop))
    classified = (
        {0, 1}
        | set(range(layout.T_bed.start, layout.T_bed.stop))
        | set(range(layout.T_gas.start, layout.T_gas.stop))
        | set(range(layout.q.start, layout.q.stop))
        | set(range(layout.n_gas.start, layout.n_gas.stop))
    )
    if classified != set(range(physical_columns)):
        raise SpatialJacobianError(
            "physical state columns do not exactly follow the reviewed layout",
            evidence={
                "physical_columns": physical_columns,
                "unclassified": sorted(
                    set(range(physical_columns)) - classified
                ),
                "outside_prefix": sorted(
                    classified - set(range(physical_columns))
                ),
            },
        )
    if layout.n_gas.stop != physical_columns:
        raise SpatialJacobianError(
            "physical prefix does not end at gas inventory"
        )
    return physical_columns, temperature_columns, nonnegative_columns


def _step_floors(layout, material) -> np.ndarray:
    physical_columns, temperature_columns, _ = _physical_layout(layout)
    floors = np.full(physical_columns, 1e-6, dtype=float)
    for index in temperature_columns:
        floors[index] = 250.0
    storage_scale = np.asarray(material.storage_scale(), dtype=float)
    q_size = layout.q.stop - layout.q.start
    if (
        storage_scale.size == 0
        or not np.all(np.isfinite(storage_scale))
        or np.any(storage_scale <= 0.0)
        or q_size % storage_scale.size != 0
    ):
        raise SpatialJacobianError(
            "material storage scale is invalid for the state layout",
            evidence={
                "storage_scale": storage_scale.tolist(),
                "loading_columns": q_size,
            },
        )
    floors[layout.q] = np.tile(storage_scale, q_size // storage_scale.size)
    floors[layout.n_gas] = 1e-6
    return floors


def _storage_probe_bounds(layout, material) -> tuple[np.ndarray, np.ndarray] | None:
    getter = getattr(material, "storage_probe_bounds_mol_kg", None)
    if not callable(getter):
        return None
    raw = getter()
    if not isinstance(raw, dict):
        raise SpatialJacobianError(
            "material storage probe bounds must be an object"
        )
    lower = np.asarray(raw.get("lower"), dtype=float)
    upper = np.asarray(raw.get("upper"), dtype=float)
    storage_scale = np.asarray(material.storage_scale(), dtype=float)
    if (
        lower.shape != storage_scale.shape
        or upper.shape != storage_scale.shape
        or not np.all(np.isfinite(lower))
        or not np.all(np.isfinite(upper))
        or np.any(lower > upper)
    ):
        raise SpatialJacobianError(
            "material storage probe bounds are invalid",
            evidence={
                "lower": lower.tolist(),
                "upper": upper.tolist(),
                "storage_scale": storage_scale.tolist(),
            },
        )
    q_size = layout.q.stop - layout.q.start
    if q_size % storage_scale.size:
        raise SpatialJacobianError(
            "material storage probe bounds do not divide the loading state"
        )
    repeats = q_size // storage_scale.size
    return np.tile(lower, repeats), np.tile(upper, repeats)


def bounded_spatial_jacobian(
    rhs,
    time,
    state,
    *,
    layout,
    material,
    audit: SpatialJacobianAudit | None = None,
    base_relative_step: float = 2e-6,
    minimum_relative_step: float = 2e-9,
):
    """Differentiate every physical column and structurally zero ledgers.

    Runtime/control and material-support exceptions from ``rhs`` propagate
    unchanged. Only nonfinite trial results cause a smaller deterministic step.
    """
    values = np.asarray(state, dtype=float)
    if values.shape != (layout.size,):
        raise SpatialJacobianError(
            "spatial state shape differs from its layout",
            evidence={
                "state_shape": list(values.shape),
                "layout_size": int(layout.size),
            },
        )
    if not np.all(np.isfinite(values)):
        raise SpatialJacobianError(
            "spatial Jacobian base state is nonfinite",
            evidence={"state_sha256": _array_sha256(values)},
        )
    base_rhs = np.asarray(rhs(time, values), dtype=float)
    if base_rhs.shape != values.shape or not np.all(np.isfinite(base_rhs)):
        raise SpatialJacobianError(
            "spatial Jacobian base RHS is nonfinite or dimensionally invalid",
            evidence={
                "time_s": float(time),
                "state_sha256": _array_sha256(values),
                "rhs_sha256": _array_sha256(base_rhs),
                "rhs_finite_count": int(
                    np.count_nonzero(np.isfinite(base_rhs))
                ),
                "rhs_size": int(base_rhs.size),
            },
        )
    physical_columns, temperature_columns, _ = _physical_layout(layout)
    material_temperature_bounds = None
    bounds_getter = getattr(material, "temperature_probe_bounds_K", None)
    if callable(bounds_getter):
        raw_bounds = bounds_getter()
        if (
            not isinstance(raw_bounds, (tuple, list))
            or len(raw_bounds) != 2
            or not all(math.isfinite(float(value)) for value in raw_bounds)
            or float(raw_bounds[0]) >= float(raw_bounds[1])
        ):
            raise SpatialJacobianError(
                "material temperature probe bounds are invalid",
                evidence={"bounds": repr(raw_bounds)},
            )
        material_temperature_bounds = (
            float(raw_bounds[0]),
            float(raw_bounds[1]),
        )
    material_temperature_columns = set(
        range(layout.T_bed.start, layout.T_bed.stop)
    )
    material_storage_bounds = _storage_probe_bounds(layout, material)
    floors = _step_floors(layout, material)
    jacobian = np.zeros((layout.size, layout.size), dtype=float)
    min_steps = np.full(physical_columns, np.inf)
    max_steps = np.zeros(physical_columns, dtype=float)
    if audit is not None:
        audit.calls += 1
        audit.rhs_evaluations += 1
    for column in range(physical_columns):
        relative_scale = max(
            abs(float(values[column])), float(floors[column])
        )
        trial_step = float(base_relative_step) * relative_scale
        accepted = False
        last_evidence = None
        for fallback in range(12):
            step = trial_step * (0.5 ** fallback)
            if step < float(minimum_relative_step) * relative_scale:
                break
            bounded_material_temperature = bool(
                material_temperature_bounds is not None
                and column in material_temperature_columns
            )
            bounded_material_storage = bool(
                material_storage_bounds is not None
                and layout.q.start <= column < layout.q.stop
            )
            positive_allowed = True
            negative_allowed = True
            if bounded_material_temperature:
                lower, upper = material_temperature_bounds
                if not (
                    lower - 1e-12
                    <= float(values[column])
                    <= upper + 1e-12
                ):
                    raise ValueError(
                        "material temperature state is outside its probe bounds"
                    )
                positive_allowed = (
                    float(values[column]) + step <= upper + 1e-12
                )
                negative_allowed = (
                    float(values[column]) - step >= lower - 1e-12
                )
                central = positive_allowed and negative_allowed
            elif bounded_material_storage:
                storage_index = column - layout.q.start
                lower = float(material_storage_bounds[0][storage_index])
                upper = float(material_storage_bounds[1][storage_index])
                if not (
                    lower - 1e-12
                    <= float(values[column])
                    <= upper + 1e-12
                ):
                    raise ValueError(
                        "material storage state is outside its probe bounds"
                    )
                positive_allowed = (
                    float(values[column]) + step <= upper + 1e-12
                )
                negative_allowed = (
                    float(values[column]) - step >= lower - 1e-12
                )
                central = positive_allowed and negative_allowed
            elif column in temperature_columns:
                central = float(values[column] - step) >= 250.0
                negative_allowed = central
            else:
                central = False
                negative_allowed = False
            if not positive_allowed and not negative_allowed:
                continue
            positive_rhs = None
            if positive_allowed:
                positive = values.copy()
                positive[column] += step
                positive_rhs = np.asarray(rhs(time, positive), dtype=float)
                if audit is not None:
                    audit.rhs_evaluations += 1
            if central:
                negative = values.copy()
                negative[column] -= step
                negative_rhs = np.asarray(rhs(time, negative), dtype=float)
                if audit is not None:
                    audit.rhs_evaluations += 1
                if (
                    positive_rhs.shape != values.shape
                    or negative_rhs.shape != values.shape
                    or not np.all(np.isfinite(positive_rhs))
                    or not np.all(np.isfinite(negative_rhs))
                ):
                    last_evidence = {
                        "time_s": float(time),
                        "column": int(column),
                        "step": float(step),
                        "positive_finite": bool(
                            np.all(np.isfinite(positive_rhs))
                        ),
                        "negative_finite": bool(
                            np.all(np.isfinite(negative_rhs))
                        ),
                    }
                    continue
                jacobian[:, column] = (
                    positive_rhs - negative_rhs
                ) / (2.0 * step)
                if audit is not None:
                    audit.central_steps += 1
            elif negative_allowed:
                negative = values.copy()
                negative[column] -= step
                negative_rhs = np.asarray(rhs(time, negative), dtype=float)
                if audit is not None:
                    audit.rhs_evaluations += 1
                if (
                    negative_rhs.shape != values.shape
                    or not np.all(np.isfinite(negative_rhs))
                ):
                    last_evidence = {
                        "time_s": float(time),
                        "column": int(column),
                        "step": float(step),
                        "positive_finite": False,
                    }
                    continue
                jacobian[:, column] = (
                    base_rhs - negative_rhs
                ) / step
                if audit is not None:
                    audit.one_sided_steps += 1
            else:
                if (
                    positive_rhs.shape != values.shape
                    or not np.all(np.isfinite(positive_rhs))
                ):
                    last_evidence = {
                        "time_s": float(time),
                        "column": int(column),
                        "step": float(step),
                        "positive_finite": False,
                    }
                    continue
                jacobian[:, column] = (positive_rhs - base_rhs) / step
                if audit is not None:
                    audit.one_sided_steps += 1
            if audit is not None and fallback:
                audit.fallback_steps += 1
            min_steps[column] = step
            max_steps[column] = step
            accepted = True
            break
        if not accepted:
            raise SpatialJacobianError(
                f"no finite spatial Jacobian probe for physical column {column}",
                evidence={
                    **(last_evidence or {}),
                    "state_sha256": _array_sha256(values),
                    "physical_columns": physical_columns,
                },
            )
    jacobian[:, physical_columns:] = 0.0
    if not np.all(np.isfinite(jacobian)):
        raise SpatialJacobianError(
            "spatial Jacobian contains nonfinite entries",
            evidence={
                "time_s": float(time),
                "state_sha256": _array_sha256(values),
                "physical_columns": physical_columns,
            },
        )
    if audit is not None:
        audit.physical_to_ledger_row_nonzero = bool(
            audit.physical_to_ledger_row_nonzero
            or np.any(
                jacobian[physical_columns:, :physical_columns] != 0.0
            )
        )
        audit.ledger_column_nonzero_entries += int(
            np.count_nonzero(jacobian[:, physical_columns:])
        )
        if audit.minimum_step_by_column is None:
            audit.minimum_step_by_column = min_steps.tolist()
            audit.maximum_step_by_column = max_steps.tolist()
            audit.matrix_sha256 = []
        else:
            audit.minimum_step_by_column = np.minimum(
                audit.minimum_step_by_column, min_steps
            ).tolist()
            audit.maximum_step_by_column = np.maximum(
                audit.maximum_step_by_column, max_steps
            ).tolist()
        audit.matrix_sha256.append(_array_sha256(jacobian))
    return jacobian
