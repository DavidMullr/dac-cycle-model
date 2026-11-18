import json
import importlib.util
from pathlib import Path

import pytest

from screening.mof_screening.energy_balance import evaluate_energy_balance


def test_periodic_finite_exchange_uses_gross_scale_and_retains_legacy_net():
    result = evaluate_energy_balance(
        residual_J=-0.0011578885605558753,
        delta_inventory_J=-18364.95643896761,
        net_boundary_J=-18364.95528107905,
        gross_boundary_exchange_J=3962143.0891637807,
    )
    assert result["gross_relative_residual"] == pytest.approx(2.922379466e-10)
    assert result["gross_pass"] is True
    assert result["legacy_net_relative_residual"] == pytest.approx(
        6.304880517e-8
    )
    assert result["legacy_net_pass"] is True
    assert result["normalization"].startswith("max(abs(delta_inventory_J)")


def test_gross_scale_distinguishes_near_zero_net_from_finite_exchange():
    result = evaluate_energy_balance(
        residual_J=0.01,
        delta_inventory_J=-647.74,
        net_boundary_J=-647.75,
        gross_boundary_exchange_J=3_000_000.0,
    )
    assert result["legacy_net_relative_residual"] > 1e-6
    assert result["gross_relative_residual"] < 1e-6
    assert result["gross_pass"] is True
    assert result["legacy_net_pass"] is False


def test_zero_exchange_uses_one_joule_floor():
    passing = evaluate_energy_balance(1e-7, 0.0, 0.0, 0.0)
    failing = evaluate_energy_balance(2.0, 0.0, 0.0, 0.0)
    assert passing["gross_scale_J"] == 1.0
    assert passing["gross_pass"] is True
    assert failing["gross_pass"] is False


def test_meaningful_imbalance_and_symmetry_scaling():
    imbalance = evaluate_energy_balance(2.0, 1000.0, 0.0, 1_000_000.0)
    symmetric = evaluate_energy_balance(1e-5, 0.0, 0.0, -4_000_000.0)
    assert imbalance["gross_pass"] is False
    assert symmetric["gross_scale_J"] == 4_000_000.0
    assert symmetric["gross_pass"] is True


@pytest.mark.parametrize(
    "arguments",
    (
        dict(residual_J=float("nan"), delta_inventory_J=0.0, net_boundary_J=0.0, gross_boundary_exchange_J=0.0),
        dict(residual_J=0.0, delta_inventory_J=float("inf"), net_boundary_J=0.0, gross_boundary_exchange_J=0.0),
        dict(residual_J=0.0, delta_inventory_J=0.0, net_boundary_J=float("nan"), gross_boundary_exchange_J=0.0),
        dict(residual_J=0.0, delta_inventory_J=0.0, net_boundary_J=0.0, gross_boundary_exchange_J=float("inf")),
        dict(residual_J=0.0, delta_inventory_J=0.0, net_boundary_J=0.0, gross_boundary_exchange_J=0.0, tolerance=0.0),
        dict(residual_J=0.0, delta_inventory_J=0.0, net_boundary_J=0.0, gross_boundary_exchange_J=0.0, floor_J=float("nan")),
    ),
)
def test_energy_balance_rejects_nonfinite_or_meaningless_inputs(arguments):
    with pytest.raises(ValueError, match="finite|positive"):
        evaluate_energy_balance(**arguments)


def test_saved_twelve_hour_cycle_two_raw_state_fixture():
    root = Path(__file__).resolve().parents[2]
    result_path = (
        root
        / "model/tests/fixtures/mof_screening/energy_result.json"
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    cycle = result["cycles"][1]
    layout = result["state_layout"]
    after = cycle["state_after"]
    gross = (
        abs(after[layout["E_heat"]])
        + abs(after[layout["e_source"]])
        + abs(after[layout["E_cool"]])
        + abs(after[layout["E_loss"]])
        + abs(after[layout["E_netflow"]])
    )
    audit = evaluate_energy_balance(
        cycle["energy_residual_half_J"],
        cycle["energy_residual_half_J"]
        + cycle["energy_boundary_half_J"],
        cycle["energy_boundary_half_J"],
        gross,
    )
    assert gross == pytest.approx(3_962_143.089, rel=1e-10)
    assert audit["gross_relative_residual"] == pytest.approx(
        2.922379466e-10, rel=1e-8
    )
    assert audit["legacy_net_relative_residual"] == pytest.approx(
        cycle["energy_relative_residual"], rel=0, abs=1e-16
    )


def test_independent_audit_ignores_engine_fields_but_detects_raw_change():
    root = Path(__file__).resolve().parents[2]
    result_path = (
        root
        / "model/tests/fixtures/mof_screening/energy_result.json"
    )
    script_path = (
        root
        / "tools/audit_cycle_energy.py"
    )
    spec = importlib.util.spec_from_file_location(
        "independent_energy_audit", script_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = json.loads(result_path.read_text(encoding="utf-8"))
    baseline = module.audit_result(original)["cycles"][1]

    perturbed_fields = json.loads(json.dumps(original))
    perturbed_fields["cycles"][1]["energy_residual_half_J"] = 1e99
    perturbed_fields["cycles"][1]["energy_boundary_half_J"] = -1e99
    from_fields = module.audit_result(perturbed_fields)["cycles"][1]
    assert from_fields["residual_J"] == pytest.approx(baseline["residual_J"])
    assert from_fields["gross_pass"] == baseline["gross_pass"]

    perturbed_raw = json.loads(json.dumps(original))
    layout = perturbed_raw["state_layout"]
    perturbed_raw["cycles"][1]["state_after"][layout["E_heat"]] += 5_000_000.0
    raw_changed = module.audit_result(perturbed_raw)["cycles"][1]
    assert raw_changed["residual_J"] != pytest.approx(baseline["residual_J"])
    assert raw_changed["gross_pass"] is False
