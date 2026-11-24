"""A110 source points, interpolation, kinetic response and normal screening entry points."""
from copy import deepcopy

import numpy as np
import pytest

from screening.mof_screening.contract import (
    MofScreeningError, build_case, canonical_sha256, load_builtin_material, load_defaults,
    preview_payload, validate_material,
)
from screening.mof_screening.materials import build_material_adapter, buck_saturation_pressure_Pa


MATERIAL_ID = "purolite_a110_humid"


def _partials(co2, rh):
    return np.array([co2, rh * buck_saturation_pressure_Pa(25.), 79000., 21000.])


def test_source_points_and_humidity_interpolation():
    adapter = build_material_adapter(load_builtin_material(MATERIAL_ID))
    # Piscina et al. Fig. 9b near 0.1 bar, including the decline at high RH.
    for rh, p, q in [(0., 10000., 2.94), (0.3, 10000., 3.29),
                     (0.55, 10964.78, 3.33), (0.8, 10000., 2.69)]:
        assert adapter.equilibrium(298.15, _partials(p, rh))[0] == pytest.approx(q)
    eq = lambda rh: adapter.equilibrium(298.15, _partials(40., rh))
    np.testing.assert_allclose(eq(.425)[0], (eq(.3)[0]+eq(.55)[0])/2)
    assert eq(.55)[1] == pytest.approx(5.82, abs=.01)  # Low et al. Table 9 GAB
    assert eq(1.)[0] == eq(.8)[0] and eq(1.)[1] > eq(.8)[1]
    assert adapter.equilibrium(298.15, _partials(0., 1.))[0] == 0.
    assert 1.8 < eq(.55)[0] < 1.9
    assert eq(0.)[0] < eq(.55)[0]
    assert adapter.equilibrium(333.15, _partials(40., .55))[0] < eq(.55)[0]


def test_wet_bead_target_and_equilibrium_rate_closure():
    adapter = build_material_adapter(load_builtin_material(MATERIAL_ID))
    for rh in (0., .11, .30, .425, .55, .8, .90, 1.):
        p = _partials(40., rh)
        target = adapter.equilibrium(298.15, p)
        rates, species = adapter.rates(298.15, p, target)
        np.testing.assert_allclose(species, 0., atol=1.e-14)
        np.testing.assert_allclose(rates, 0., atol=1.e-14)
    p = _partials(40., .55)
    dry = adapter.equilibrium(298.15, _partials(40., 0.))
    rates, _ = adapter.rates(298.15, p, dry)
    assert rates[0] == pytest.approx(0., abs=1.e-14) and rates[1] > 0
    dry[1] = adapter.equilibrium(298.15, p)[1]
    rates, _ = adapter.rates(298.15, p, dry)
    assert rates[0] > 0


@pytest.mark.parametrize("engine", ["spatial", "lumped"])
def test_preset_preview(engine):
    defaults = load_defaults()
    assert any(row["id"] == MATERIAL_ID for row in defaults["built_in_materials"])
    preset = next(p for p in defaults["case_presets"] if p.get("material_id") == MATERIAL_ID)
    case = deepcopy(preset["case"])
    m, c = build_case(load_builtin_material(MATERIAL_ID), case)
    assert canonical_sha256(c) == preset["case_sha256"]
    case["engine"] = engine
    case["solver"]["cells"] = 1 if engine == "lumped" else 4
    case["weather"]["relative_humidity"] = .97
    m, c = build_case(m, case)
    preview = preview_payload(m, c, source="test")
    assert preview["material_id"] == MATERIAL_ID
    assert preview["temperature_policy"]["registry_classification"] == "unqualified_scenario_guard"
    assert any("Above 80%" in text for text in preview["warnings"])


def test_invalid_curve_rejected_before_simulation():
    m = load_builtin_material(MATERIAL_ID)
    m["co2"]["curves"][0]["pressure_Pa"][1] = 1
    with pytest.raises(MofScreeningError, match="increasing pressures"):
        validate_material(m)

