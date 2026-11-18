import json
from pathlib import Path

import numpy as np
import pytest

from screening.mof_screening.contract import load_builtin_material
from screening.mof_screening.materials import build_material_adapter


FIXTURES = Path(__file__).resolve().parent / "fixtures/mof_screening/material_source_values.json"


def _adapter(name):
    material_id = (
        "hypothetical_mmen_like"
        if name == "hypothetical"
        else "calf20_shaped"
    )
    return build_material_adapter(load_builtin_material(material_id))


@pytest.mark.parametrize(
    "fixture",
    json.loads(FIXTURES.read_text(encoding="utf-8"))["fixtures"],
    ids=lambda row: f"{row['material']}-{row['label']}",
)
def test_adapter_matches_accepted_source_fixture(fixture):
    adapter = _adapter(fixture["material"])
    temperature = fixture["temperature_K"]
    partials = np.asarray(fixture["partial_pressures_Pa"], dtype=float)
    expected_equilibrium = np.asarray(fixture["equilibrium_mol_kg"], dtype=float)
    if fixture["material"] == "calf":
        sites = np.asarray(fixture["site_loadings_mol_kg"], dtype=float)
        expected_adapter = np.r_[sites, expected_equilibrium[1]]
    else:
        expected_adapter = expected_equilibrium
    equilibrium = adapter.equilibrium(temperature, partials)
    np.testing.assert_allclose(
        equilibrium, expected_adapter, rtol=2e-5, atol=5e-9
    )

    if fixture["material"] == "hypothetical":
        state = expected_equilibrium * np.array(
            [
                fixture["site_state_fraction"],
                fixture["water_state_fraction"],
                fixture["inert_state_fraction"],
                fixture["inert_state_fraction"],
            ]
        )
    else:
        state = np.r_[
            np.asarray(fixture["site_loadings_mol_kg"])
            * fixture["site_state_fraction"],
            expected_equilibrium[1] * fixture["water_state_fraction"],
        ]
    rates, _ = adapter.rates(temperature, partials, state)
    expected_rates = np.asarray(
        fixture["rates_at_declared_loadings_mol_kg_s"][
            : adapter.n_storage
        ],
        dtype=float,
    )
    np.testing.assert_allclose(rates, expected_rates, rtol=2e-5, atol=5e-9)

    expected_energy = np.asarray(
        fixture["absolute_energies_J_mol"], dtype=float
    )[: adapter.n_storage]
    np.testing.assert_allclose(
        adapter.absolute_energy_J_mol(temperature),
        expected_energy,
        rtol=0.0,
        atol=1e-6,
    )
    expected_cp = np.asarray(
        fixture["component_cp_J_mol_K"], dtype=float
    )[: adapter.n_storage]
    np.testing.assert_allclose(
        adapter.component_cp_molar_J_mol_K(),
        expected_cp,
        rtol=0.0,
        atol=1e-9,
    )


def test_calf_keeps_distinct_site_states_and_energies():
    adapter = _adapter("calf")
    loadings = adapter.equilibrium(
        298.15, np.array([100.0, 1000.0, 79000.0, 21000.0])
    )
    assert adapter.n_storage == 3
    assert loadings.shape == (3,)
    assert adapter.storage_labels[:2] == ("CO2_site_1", "CO2_site_2")
    assert abs(loadings[0] - loadings[1]) > 1e-12
    energies = adapter.absolute_energy_J_mol(350.0)
    assert abs(energies[0] - energies[1]) > 1e-12
