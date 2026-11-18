"""Saved sensitivities retain exact dry-package dependencies outside the active library."""
import json
from pathlib import Path

import pytest

from screening.mof_screening.contract import (
    MATERIAL_ROOT, MofScreeningError, _humidity_origin_path,
    load_humidity_candidate_origin, validate_material,
)


def test_saved_humidity_candidates_resolve_exact_dry_inputs():
    root = Path(__file__).resolve().parents[1]
    checked = 0
    for path in (root / 'docs/data/submission/results').glob('*.json'):
        package = json.loads(path.read_text(encoding='utf-8'))['material_package']
        extension = package.get('humidity_candidate')
        if not extension:
            continue
        validate_material(package)
        origin = load_humidity_candidate_origin(extension)
        assert origin['material_id'] == extension['dry_origin_material_id']
        checked += 1
    assert checked > 0


@pytest.mark.parametrize('value', ['../outside', '..\\outside', 'C:outside', '.', '..'])
def test_dry_origin_rejects_path_like_identifiers(value):
    with pytest.raises(MofScreeningError, match='invalid dry origin material id'):
        _humidity_origin_path(value)


def test_active_material_remains_the_primary_origin():
    assert _humidity_origin_path('hypothetical_mmen_like') == MATERIAL_ROOT / 'hypothetical_mmen_like.json'
