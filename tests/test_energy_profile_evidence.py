"""An independent audit must use the exact configuration bound to its result."""
from copy import deepcopy
import hashlib
import json

import pytest

from tools import audit_cycle_energy as audit


@pytest.fixture
def saved_result():
    return json.loads((audit.ROOT / 'model/tests/fixtures/mof_screening/energy_result.json').read_text(encoding='utf-8'))


def test_saved_result_uses_its_hash_bound_profile(saved_result):
    result = audit.audit_result(saved_result)
    assert result['profile_source_verified'] is True
    assert result['all_gross_pass'] is True
    assert result['profile_sha256'] != hashlib.sha256(audit.PROFILE.read_bytes()).hexdigest()


def test_current_identity_uses_current_profile(saved_result):
    digest = hashlib.sha256(audit.PROFILE.read_bytes()).hexdigest()
    saved_result['source_identity']['source_hashes'] = {'config/mof_screening/spatial_profile.json': digest}
    result = audit.audit_result(saved_result)
    assert result['profile_source_verified'] is True
    assert result['profile_sha256'] == digest
    assert result['all_gross_pass'] is True


def test_explicit_override_does_not_fall_back_to_saved_evidence(saved_result):
    result = audit.audit_result(saved_result, profile_path=audit.PROFILE)
    assert result['profile_source_verified'] is False
    assert result['profile_sha256'] == hashlib.sha256(audit.PROFILE.read_bytes()).hexdigest()


def test_changed_evidence_is_not_verified(saved_result, tmp_path, monkeypatch):
    original, verified = audit._source_profile(saved_result, None)
    assert verified
    digest = hashlib.sha256(original).hexdigest()
    (tmp_path / (digest + '.json')).write_bytes(original + b'\n')
    monkeypatch.setattr(audit, 'SOURCE_PROFILES', tmp_path)
    assert audit.audit_result(saved_result)['profile_source_verified'] is False


@pytest.mark.parametrize('hashes', [
    {},
    {'config/mof_screening/spatial_profile.json': '../../untrusted'},
    {'config/mof_screening/spatial_profile.json': '0' * 64},
    {'config/mof_screening/spatial_profile.json': '0' * 64,
     'config/mof_screening/other_spatial_profile.json': '1' * 64},
])
def test_missing_invalid_or_ambiguous_identity_is_unverified(saved_result, hashes):
    saved_result = deepcopy(saved_result)
    saved_result['source_identity']['source_hashes'] = hashes
    assert audit.audit_result(saved_result)['profile_source_verified'] is False


@pytest.mark.parametrize('name', ['lewatit_humid', 'cof1000_humid'])
def test_material_specific_energy_laws_require_their_own_audit(name):
    result = json.loads((audit.ROOT / 'docs/data/submission/results' / (name + '.json')).read_text())
    with pytest.raises(ValueError, match='material-specific audit'):
        audit.audit_result(result)
