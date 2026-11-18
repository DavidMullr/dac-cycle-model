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
BRANCH_POLICY_V1 = "branch-aware-admissible-fd/v1"
BRANCH_POLICY_ENV = "MOF_BRANCH_JACOBIAN_POLICY"
JACOBIAN_CACHE_POLICY_EXACT = "exact-cell-evaluation-cache/v1"
JACOBIAN_CACHE_POLICY_ENV = "MOF_JACOBIAN_CACHE_POLICY"


def _exact_array_key(values) -> tuple:
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    return array.shape, array.tobytes(order="C")


class ExactMaterialEvaluationCache:
    """Delegate to material while memoizing pure cell evaluations.

    Caching is active only between ``begin`` and ``end``. Inputs are keyed by
    their exact float64 bit patterns, and cached results are copied before
    return so callers cannot mutate an entry.
    """

    policy = JACOBIAN_CACHE_POLICY_EXACT

    def __init__(self, material):
        self.material = material
        self._active = False
        self._rates = {}
        self._energy = {}
        self.hits = 0
        self.misses = 0

    def __getattr__(self, name):
        return getattr(self.material, name)

    @property
    def active(self) -> bool:
        return self._active

    @property
    def entry_count(self) -> int:
        return len(self._rates) + len(self._energy)

    def begin(self) -> None:
        if self._active:
            raise RuntimeError("exact material cache is already active")
        self._rates.clear()
        self._energy.clear()
        self.hits = 0
        self.misses = 0
        self._active = True

    def end(self) -> dict:
        if not self._active:
            raise RuntimeError("exact material cache is not active")
        stats = {
            "policy": self.policy,
            "hits": int(self.hits),
            "misses": int(self.misses),
            "entries": int(self.entry_count),
        }
        self._rates.clear()
        self._energy.clear()
        self._active = False
        return stats

    def _water_is_negative(self, storage) -> bool:
        water_index = getattr(self.material, "water_index", None)
        return bool(
            water_index is not None
            and float(np.asarray(storage)[int(water_index)]) < 0.0
        )

    def _cached_rates(self, temperature_K, partials_Pa, storage):
        delegate = self.material.rates_with_directional_delta
        if not self._active or self._water_is_negative(storage):
            return delegate(temperature_K, partials_Pa, storage), False
        key = (
            float(temperature_K).hex(),
            _exact_array_key(partials_Pa),
            _exact_array_key(storage),
        )
        result = self._rates.get(key)
        if result is None:
            result = delegate(temperature_K, partials_Pa, storage)
            self._rates[key] = result
            self.misses += 1
        else:
            self.hits += 1
        return result, True

    def rates(self, temperature_K, partials_Pa, storage):
        if not self._active:
            return self.material.rates(
                temperature_K, partials_Pa, storage
            )
        result, _cached = self._cached_rates(
            temperature_K, partials_Pa, storage
        )
        return result[0].copy(), result[1].copy()

    def rates_with_directional_delta(
        self, temperature_K, partials_Pa, storage
    ):
        result, cached = self._cached_rates(
            temperature_K, partials_Pa, storage
        )
        if not cached:
            return result
        return result[0].copy(), result[1].copy(), result[2]

    def storage_energy_terms(self, temperature_K, storage):
        delegate = self.material.storage_energy_terms
        if not self._active or self._water_is_negative(storage):
            return delegate(temperature_K, storage)
        key = (float(temperature_K).hex(), _exact_array_key(storage))
        result = self._energy.get(key)
        if result is None:
            result = delegate(temperature_K, storage)
            self._energy[key] = result
            self.misses += 1
        else:
            self.hits += 1
        energy, capacity, gradient = result
        return energy, capacity, np.asarray(gradient, dtype=float).copy()


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
    branch_policy: str | None = None
    branch_probe_strategy_by_call: list | None = None
    branch_switch_rejections: int = 0
    branch_exact_switch_probes: int = 0
    branch_unresolved_columns: list | None = None
    jacobian_cache_policy: str | None = None
    jacobian_cache_calls: int = 0
    jacobian_cache_hits: int = 0
    jacobian_cache_misses: int = 0
    jacobian_cache_stats_by_call: list | None = None
    jacobian_cache_active: bool = False

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
            "branch_policy": self.branch_policy,
            "branch_probe_strategy_by_call": (
                self.branch_probe_strategy_by_call
            ),
            "branch_switch_rejections": self.branch_switch_rejections,
            "branch_exact_switch_probes": self.branch_exact_switch_probes,
            "branch_unresolved_columns": self.branch_unresolved_columns,
            "jacobian_cache_policy": self.jacobian_cache_policy,
            "jacobian_cache_calls": self.jacobian_cache_calls,
            "jacobian_cache_hits": self.jacobian_cache_hits,
            "jacobian_cache_misses": self.jacobian_cache_misses,
            "jacobian_cache_stats_by_call": self.jacobian_cache_stats_by_call,
            "jacobian_cache_active": self.jacobian_cache_active,
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
    physical_columns, temperature_columns, nonnegative_columns = _physical_layout(
        layout
    )
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


def _branch_signature(value):
    if value is None:
        return None
    signature = []
    for item in value:
        try:
            cell, storage_index, delta = item
        except (TypeError, ValueError) as error:
            raise SpatialJacobianError(
                "directional-rate branch signature has an invalid entry"
            ) from error
        if (
            type(cell) is not int
            or type(storage_index) is not int
            or type(delta) not in {int, float}
            or not math.isfinite(float(delta))
        ):
            raise SpatialJacobianError(
                "directional-rate branch signature has an invalid value"
            )
        numeric = float(delta)
        sign = 0 if numeric == 0.0 else (1 if numeric > 0.0 else -1)
        signature.append((cell, storage_index, sign))
    return tuple(signature)


def _affected_material_cell(layout, column: int, n_storage: int) -> int | None:
    """Map a physical state column to the only material cell it can switch."""

    if layout.T_bed.start <= column < layout.T_bed.stop:
        return int(column - layout.T_bed.start)
    if layout.T_gas.start <= column < layout.T_gas.stop:
        return int(column - layout.T_gas.start)
    if layout.q.start <= column < layout.q.stop:
        return int((column - layout.q.start) // n_storage)
    if layout.n_gas.start <= column < layout.n_gas.stop:
        return int((column - layout.n_gas.start) // 4)
    return None


def _signature_for_cell(signature, cell: int | None):
    if signature is None or cell is None:
        return signature
    selected = tuple(item for item in signature if item[0] == cell)
    return selected


def _branch_change(base_signature, trial_signature):
    """Return whether a nonzero branch changes and how many switches leave zero."""

    if base_signature is None or trial_signature is None:
        raise SpatialJacobianError(
            "branch-aware probe did not expose directional-rate state"
        )
    if len(base_signature) != len(trial_signature):
        raise SpatialJacobianError(
            "directional-rate branch signature length changed between probes"
        )
    nonzero_change = False
    exact_switch = 0
    for base, trial in zip(base_signature, trial_signature, strict=True):
        if base[:2] != trial[:2]:
            raise SpatialJacobianError(
                "directional-rate branch signature coordinates changed"
            )
        base_sign, trial_sign = base[2], trial[2]
        if base_sign == 0 and trial_sign != 0:
            exact_switch += 1
        elif base_sign != 0 and trial_sign != base_sign:
            nonzero_change = True
    return nonzero_change, exact_switch


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
    branch_policy: str | None = None,
):
    """Differentiate every physical column and structurally zero ledgers.

    Runtime/control and material-support exceptions from ``rhs`` propagate
    unchanged. The default path preserves the established nonfinite-only
    fallback. ``BRANCH_POLICY_V1`` additionally rejects probes that change any
    nonzero directional rate branch, preferring an admissible backward or
    smaller forward probe without changing the RHS.
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
    if branch_policy not in {None, BRANCH_POLICY_V1}:
        raise SpatialJacobianError("unknown branch-aware Jacobian policy")
    branch_aware = branch_policy == BRANCH_POLICY_V1
    base_rhs = np.asarray(rhs(time, values), dtype=float)
    base_branch_signature = (
        _branch_signature(getattr(rhs, "last_directional_rate_deltas", None))
        if branch_aware
        else None
    )
    if branch_aware and base_branch_signature is None:
        raise SpatialJacobianError(
            "branch-aware policy requires directional-rate branch metadata"
        )
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
    physical_columns, temperature_columns, nonnegative_columns = (
        _physical_layout(layout)
    )
    n_storage = int(
        (layout.q.stop - layout.q.start) // int(layout.n_cells)
    )
    if n_storage <= 0:
        raise SpatialJacobianError(
            "state layout has no material storage cells",
            evidence={"loading_columns": layout.q.stop - layout.q.start},
        )
    affected_cell_by_column = {
        column: _affected_material_cell(layout, column, n_storage)
        for column in range(physical_columns)
    }
    base_branch_by_cell = {}
    if branch_aware:
        for cell in range(int(layout.n_cells)):
            base_branch_by_cell[cell] = _signature_for_cell(
                base_branch_signature, cell
            )
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
    branch_strategy = [None] * physical_columns
    branch_unresolved = []
    if audit is not None:
        if branch_aware:
            audit.branch_policy = BRANCH_POLICY_V1
        audit.calls += 1
        audit.rhs_evaluations += 1
    for column in range(physical_columns):
        relative_scale = max(
            abs(float(values[column])), float(floors[column])
        )
        trial_step = float(base_relative_step) * relative_scale
        accepted = False
        last_evidence = None
        affected_cell = affected_cell_by_column[column]
        base_cell_signature = (
            base_branch_by_cell.get(affected_cell)
            if branch_aware and affected_cell is not None
            else base_branch_signature
        )
        for fallback in range(24 if branch_aware else 12):
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
                negative_allowed = bool(
                    branch_aware
                    and column in nonnegative_columns
                    and float(values[column]) - step >= 0.0
                )
            if not positive_allowed and not negative_allowed:
                continue
            positive_rhs = None
            positive_branch = None
            if positive_allowed:
                positive = values.copy()
                positive[column] += step
                positive_rhs = np.asarray(rhs(time, positive), dtype=float)
                if audit is not None:
                    audit.rhs_evaluations += 1
                if branch_aware:
                    positive_branch = _signature_for_cell(
                        _branch_signature(
                            getattr(rhs, "last_directional_rate_deltas", None)
                        ),
                        affected_cell,
                    )
            if branch_aware:
                positive_valid = bool(
                    positive_rhs is not None
                    and positive_rhs.shape == values.shape
                    and np.all(np.isfinite(positive_rhs))
                )
                if not positive_valid:
                    last_evidence = {
                        "time_s": float(time),
                        "column": int(column),
                        "step": float(step),
                        "positive_finite": False,
                    }
                positive_change = False
                positive_exact = 0
                if not (
                    positive_rhs is None
                    or positive_rhs.shape != values.shape
                    or not np.all(np.isfinite(positive_rhs))
                ):
                    positive_change, positive_exact = _branch_change(
                        base_cell_signature, positive_branch
                    )
                if central:
                    if not negative_allowed:
                        continue
                    negative = values.copy()
                    negative[column] -= step
                    negative_rhs = np.asarray(
                        rhs(time, negative), dtype=float
                    )
                    if audit is not None:
                        audit.rhs_evaluations += 1
                    negative_branch = _signature_for_cell(
                        _branch_signature(
                            getattr(rhs, "last_directional_rate_deltas", None)
                        ),
                        affected_cell,
                    )
                    if (
                        negative_rhs.shape != values.shape
                        or not np.all(np.isfinite(negative_rhs))
                    ):
                        last_evidence = {
                            "time_s": float(time),
                            "column": int(column),
                            "step": float(step),
                            "positive_finite": True,
                            "negative_finite": False,
                        }
                        continue
                    negative_change, negative_exact = _branch_change(
                        base_cell_signature, negative_branch
                    )
                    if not positive_valid:
                        if negative_change:
                            if audit is not None:
                                audit.branch_switch_rejections += 1
                            last_evidence = {
                                "time_s": float(time),
                                "column": int(column),
                                "step": float(step),
                                "branch_policy": BRANCH_POLICY_V1,
                                "positive_finite": False,
                                "negative_branch_change": True,
                            }
                            continue
                        jacobian[:, column] = (
                            base_rhs - negative_rhs
                        ) / step
                        if audit is not None:
                            audit.one_sided_steps += 1
                            audit.branch_exact_switch_probes += negative_exact
                        branch_strategy[column] = "backward"
                    elif positive_change and negative_change:
                        if audit is not None:
                            audit.branch_switch_rejections += 1
                        last_evidence = {
                            "time_s": float(time),
                            "column": int(column),
                            "step": float(step),
                            "branch_policy": BRANCH_POLICY_V1,
                            "positive_branch_change": positive_change,
                            "negative_branch_change": negative_change,
                        }
                        continue
                    elif negative_change:
                        jacobian[:, column] = (
                            positive_rhs - base_rhs
                        ) / step
                        if audit is not None:
                            audit.one_sided_steps += 1
                            audit.branch_exact_switch_probes += positive_exact
                        branch_strategy[column] = (
                            "exact_switch_forward"
                            if positive_exact
                            else "forward"
                        )
                    elif positive_change:
                        jacobian[:, column] = (
                            base_rhs - negative_rhs
                        ) / step
                        if audit is not None:
                            audit.one_sided_steps += 1
                            audit.branch_exact_switch_probes += negative_exact
                        branch_strategy[column] = (
                            "exact_switch_backward"
                            if negative_exact
                            else "backward"
                        )
                    else:
                        jacobian[:, column] = (
                            positive_rhs - negative_rhs
                        ) / (2.0 * step)
                        if audit is not None:
                            audit.central_steps += 1
                            audit.branch_exact_switch_probes += (
                                positive_exact + negative_exact
                            )
                        branch_strategy[column] = (
                            "exact_switch_central"
                            if positive_exact or negative_exact
                            else "central"
                        )
                elif positive_valid and not positive_change:
                    jacobian[:, column] = (
                        positive_rhs - base_rhs
                    ) / step
                    if audit is not None:
                        audit.one_sided_steps += 1
                        audit.branch_exact_switch_probes += positive_exact
                    branch_strategy[column] = (
                        "exact_switch_forward"
                        if positive_exact
                        else "small_forward" if fallback else "forward"
                    )
                elif negative_allowed:
                    negative = values.copy()
                    negative[column] -= step
                    negative_rhs = np.asarray(
                        rhs(time, negative), dtype=float
                    )
                    if audit is not None:
                        audit.rhs_evaluations += 1
                    negative_branch = _signature_for_cell(
                        _branch_signature(
                            getattr(rhs, "last_directional_rate_deltas", None)
                        ),
                        affected_cell,
                    )
                    if (
                        negative_rhs.shape != values.shape
                        or not np.all(np.isfinite(negative_rhs))
                    ):
                        last_evidence = {
                            "time_s": float(time),
                            "column": int(column),
                            "step": float(step),
                            "positive_finite": True,
                            "negative_finite": False,
                        }
                        continue
                    negative_change, negative_exact = _branch_change(
                        base_cell_signature, negative_branch
                    )
                    if negative_change:
                        if audit is not None:
                            audit.branch_switch_rejections += 1
                        last_evidence = {
                            "time_s": float(time),
                            "column": int(column),
                            "step": float(step),
                            "branch_policy": BRANCH_POLICY_V1,
                            "negative_branch_change": True,
                        }
                        continue
                    jacobian[:, column] = (base_rhs - negative_rhs) / step
                    if audit is not None:
                        audit.one_sided_steps += 1
                        audit.branch_exact_switch_probes += negative_exact
                    branch_strategy[column] = (
                        "exact_switch_backward"
                        if negative_exact
                        else "backward"
                    )
                else:
                    if audit is not None:
                        audit.branch_switch_rejections += 1
                    last_evidence = {
                        "time_s": float(time),
                        "column": int(column),
                        "step": float(step),
                        "branch_policy": BRANCH_POLICY_V1,
                        "positive_branch_change": True,
                        "negative_allowed": False,
                    }
                    continue
            elif central:
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
            if branch_aware:
                branch_unresolved.append(int(column))
                if audit is not None:
                    audit.branch_unresolved_columns = list(branch_unresolved)
            raise SpatialJacobianError(
                f"no finite spatial Jacobian probe for physical column {column}",
                evidence={
                    **(last_evidence or {}),
                    "state_sha256": _array_sha256(values),
                    "physical_columns": physical_columns,
                },
            )
    if audit is not None and branch_aware:
        if audit.branch_probe_strategy_by_call is None:
            audit.branch_probe_strategy_by_call = []
        audit.branch_probe_strategy_by_call.append(list(branch_strategy))
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


def bounded_spatial_jacobian_with_material_cache(
    rhs,
    time,
    state,
    *,
    layout,
    material,
    material_cache: ExactMaterialEvaluationCache,
    audit: SpatialJacobianAudit | None = None,
    base_relative_step: float = 2e-6,
    minimum_relative_step: float = 2e-9,
    branch_policy: str | None = None,
):
    """Build one Jacobian with cache lifetime confined to that call."""

    if not isinstance(material_cache, ExactMaterialEvaluationCache):
        raise TypeError(
            "material_cache must be ExactMaterialEvaluationCache"
        )
    material_cache.begin()
    if audit is not None:
        audit.jacobian_cache_policy = material_cache.policy
        audit.jacobian_cache_active = True
        if audit.jacobian_cache_stats_by_call is None:
            audit.jacobian_cache_stats_by_call = []
    try:
        return bounded_spatial_jacobian(
            rhs,
            time,
            state,
            layout=layout,
            material=material,
            audit=audit,
            base_relative_step=base_relative_step,
            minimum_relative_step=minimum_relative_step,
            branch_policy=branch_policy,
        )
    finally:
        stats = material_cache.end()
        if audit is not None:
            audit.jacobian_cache_active = False
            audit.jacobian_cache_calls += 1
            audit.jacobian_cache_hits += stats["hits"]
            audit.jacobian_cache_misses += stats["misses"]
            audit.jacobian_cache_stats_by_call.append(stats)
