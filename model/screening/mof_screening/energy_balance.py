"""Shared gross-exchange energy closure scale and net-exchange net diagnostics."""
from __future__ import annotations

import math


ENERGY_RELATIVE_TOLERANCE = 1e-6
ENERGY_ABSOLUTE_FLOOR_J = 1.0


def _finite_value(name: str, value: float, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite numeric value")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite numeric value") from error
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive " if positive else ""
        raise ValueError(f"{name} must be a finite {qualifier}value")
    return result


def gross_energy_scale(
    delta_inventory_J: float,
    gross_boundary_exchange_J: float,
    *,
    floor_J: float = ENERGY_ABSOLUTE_FLOOR_J,
) -> float:
    delta = _finite_value("delta_inventory_J", delta_inventory_J)
    gross = _finite_value(
        "gross_boundary_exchange_J", gross_boundary_exchange_J
    )
    floor = _finite_value("floor_J", floor_J, positive=True)
    return max(
        abs(delta),
        abs(gross),
        floor,
    )


def evaluate_energy_balance(
    residual_J: float,
    delta_inventory_J: float,
    net_boundary_J: float,
    gross_boundary_exchange_J: float,
    *,
    tolerance: float = ENERGY_RELATIVE_TOLERANCE,
    floor_J: float = ENERGY_ABSOLUTE_FLOOR_J,
) -> dict:
    """Evaluate gross relative closure while retaining the net-exchange net basis."""
    residual = _finite_value("residual_J", residual_J)
    delta = _finite_value("delta_inventory_J", delta_inventory_J)
    net_boundary = _finite_value("net_boundary_J", net_boundary_J)
    gross_boundary = _finite_value(
        "gross_boundary_exchange_J", gross_boundary_exchange_J
    )
    tolerance_value = _finite_value(
        "tolerance", tolerance, positive=True
    )
    floor = _finite_value("floor_J", floor_J, positive=True)
    gross_scale = gross_energy_scale(
        delta,
        gross_boundary,
        floor_J=floor,
    )
    net_scale = max(
        abs(delta),
        abs(net_boundary),
        floor,
    )
    gross_relative = abs(residual) / gross_scale
    net_relative = abs(residual) / net_scale
    return {
        "residual_J": residual,
        "delta_inventory_J": delta,
        "net_boundary_J": net_boundary,
        "gross_boundary_exchange_J": gross_boundary,
        "gross_scale_J": gross_scale,
        "gross_relative_residual": gross_relative,
        "gross_pass": bool(
            math.isfinite(gross_relative)
            and gross_relative <= tolerance_value
        ),
        "legacy_net_scale_J": net_scale,
        "legacy_net_relative_residual": net_relative,
        "legacy_net_pass": bool(
            math.isfinite(net_relative)
            and net_relative <= tolerance_value
        ),
        "tolerance": tolerance_value,
        "absolute_floor_J": floor,
        "normalization": (
            "max(abs(delta_inventory_J), gross_boundary_exchange_J, "
            "absolute_floor_J)"
        ),
        "legacy_normalization": (
            "max(abs(delta_inventory_J), abs(net_boundary_J), "
            "absolute_floor_J)"
        ),
    }
