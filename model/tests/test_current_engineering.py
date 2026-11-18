"""Current-run duties, pricing and HTTP workflow, without new cycle integration."""
from copy import deepcopy
import json
from pathlib import Path
import threading
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from app import engineering as eng
from app.server import build_server
from screening.mof_screening.contract import MofScreeningError


@pytest.fixture
def manager(tmp_path):
    return SimpleNamespace(output_root=tmp_path / 'runs', history=lambda: [])


def test_all_reviewed_exports_have_distinct_selectors_and_correct_duties(manager):
    catalogue = eng.catalogue(manager)
    selectors = [row['selector'] for row in catalogue['results']]
    assert len(selectors) == len(set(selectors)) == 16
    assert len(catalogue['incomplete_sensitivities']) == 2
    identities = set()
    for selector in selectors:
        result, binding = eng.load_result(selector, manager)
        report = eng.assess(result, binding)
        identities.add(binding['result_sha256'])
        seconds = result['horizon']['simulated_s']
        metrics = result['metrics']
        assert report['electricity']['average_input_W'] == pytest.approx(
            sum(metrics['electricity_components_kWh'].values()) * 3_600_000 / seconds)
        assert report['production']['collected_CO2_kg_day'] == metrics['collected_CO2_kg_day']
        assert report['costs']['listed_components_total_CHF'] is None
    assert len(identities) == 16


def test_lewatit_independent_arithmetic_and_unknown_water(manager):
    result, binding = eng.load_result('reviewed:lewatit_humid', manager)
    report = eng.assess(result, binding)
    assert report['bed']['volume_m3'] == pytest.approx(1 / 36)
    assert report['electricity']['kWh_day'] == pytest.approx(14.056521178017412)
    assert report['water']['condensate_kg_day'] == pytest.approx(428.40912524270897 * .01801528 / 2)
    assert report['thermal']['heater_average_electric_W'] == pytest.approx(19.635473264800268 * 1000 / 48)
    assert report['fan']['bed_pressure_drop_max_Pa'] == pytest.approx(13.693551425491298)
    assert report['fan']['model_fan_input_max_W'] == pytest.approx(.18 * 13.693551425491298 / .55)
    # A missing saved quantity must remain unknown.
    result['metrics'].pop('water_condensed_mol', None)
    assert eng.assess(result, binding)['water']['condensate_kg_day'] is None


def test_explicit_prices_revision_and_cross_result_isolation(manager):
    selector = 'reviewed:lewatit_humid'
    costs = eng.default_costs()
    costs['electricity_CHF_kWh'] = .25
    costs['components'][0].update(quantity=10, unit_cost_CHF=20, basis='Test quote per kg')
    result = eng.save(selector, manager, costs, 0)
    assert result['worksheet_revision'] == 1
    assert result['costs']['known_subtotal_CHF'] == 200
    assert result['costs']['listed_components_total_CHF'] is None
    assert result['costs']['electricity_cost_CHF_day'] == pytest.approx(14.056521178017412 * .25)
    assert eng.calculate(selector, manager)['costs'] == result['costs']
    assert eng.calculate('reviewed:lewatit_dry', manager)['worksheet_revision'] == 0
    with pytest.raises(MofScreeningError, match='changed'):
        eng.save(selector, manager, costs, 0)
    costs['components'][1]['unit_cost_CHF'] = 0
    with pytest.raises(MofScreeningError, match='basis'):
        eng.save(selector, manager, costs, 1)
    assert eng.calculate(selector, manager)['worksheet_revision'] == 1


def test_tampered_and_incomplete_exports_rejected(manager, monkeypatch, tmp_path):
    result, _ = eng.load_result('reviewed:lewatit_humid', manager)
    folder = tmp_path / 'docs/data/submission/results'
    folder.mkdir(parents=True)
    export = folder / 'test.json'
    export.write_text(json.dumps(result))
    index = tmp_path / 'index.json'
    document = {'rows':[{'result':'docs/data/submission/results/test.json', 'material':'Test', 'group':'humid'}],
                'bindings':[{'path':'docs/data/submission/results/test.json','sha256':'wrong'}]}
    index.write_text(json.dumps(document))
    monkeypatch.setattr(eng, 'ROOT', tmp_path)
    monkeypatch.setattr(eng, 'INDEX', index)
    with pytest.raises(MofScreeningError, match='identity'):
        eng.load_result('reviewed:test', manager)
    result['horizon']['complete'] = False
    export.write_text(json.dumps(result))
    document['bindings'][0]['sha256'] = eng.sha(export.read_bytes())
    index.write_text(json.dumps(document))
    with pytest.raises(MofScreeningError, match='complete simulation'):
        eng.load_result('reviewed:test', manager)


def test_current_http_workflow_and_boundaries(tmp_path):
    server = build_server(0, output_root=tmp_path / 'runs')
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    def request(path, body=None, headers=None):
        request_headers = {'Content-Type':'application/json', **(headers or {})}
        req = Request(base + path, data=None if body is None else json.dumps(body).encode(), headers=request_headers)
        try:
            with urlopen(req, timeout=10) as response:
                return response.status, response.read()
        except HTTPError as error:
            return error.code, error.read()
    try:
        for path in ('/', '/results/', '/engineering/', '/screening/', '/docs/geometry.html', '/health',
                     '/project-report/', '/project-report/model_reference.html', '/project-report/evidence.html',
                     '/project-report/user_guide.html', '/project-report/model_and_results.pdf'):
            assert request(path)[0] == 200
        assert request('/health', headers={'Host':'evil.example'})[0] == 403
        assert request('/docs/%2e%2e/model/requirements-lock.txt')[0] == 404
        assert request('/Archive/private.json')[0] == 404
        body = {'result':'reviewed:lewatit_humid', 'costs':eng.default_costs(), 'expected_revision':0}
        assert request('/api/engineering/save', body, {'Origin':'https://evil.example'})[0] == 403
        assert request('/api/engineering/save', body, {'Content-Type':'text/plain'})[0] == 415
        status, raw = request('/api/engineering/save', body)
        assert status == 200 and json.loads(raw)['worksheet_revision'] == 1
        assert request('/api/engineering/save', body)[0] == 409
        status, raw = request('/api/engineering?result=reviewed:lewatit_humid')
        assert status == 200 and json.loads(raw)['worksheet_revision'] == 1
        body['costs']['components'][0].update(name='=BAD()', unit_cost_CHF=0, basis='Already owned')
        status, raw = request('/api/engineering/export?format=csv', body)
        assert status == 200 and "'=BAD()" in raw.decode()
        status, raw = request('/api/engineering/export?format=json', body)
        assert status == 200 and json.loads(raw)['binding']['selector'] == body['result']
        assert server.manager.history() == []
        assert not server.manager.active_event.is_set()
    finally:
        server.shutdown()
        server.manager.close()
        server.server_close()


def test_undefined_specific_cost_and_unbound_fan_remain_unknown(manager):
    result, binding = eng.load_result("reviewed:lewatit_humid", manager)
    result["source_identity"]["source_hashes"] = {}
    result["metrics"].update(collected_CO2_kg=0.0, collected_CO2_kg_day=0.0, electricity_kWh_kg_CO2=None)
    costs = eng.default_costs()
    costs["electricity_CHF_kWh"] = 0.25
    report = eng.assess(result, binding, costs)
    assert report["costs"]["electricity_cost_CHF_day"] == pytest.approx(0.25 * result["metrics"]["electricity_kWh_day"])
    assert report["costs"]["electricity_cost_CHF_kg_CO2"] is None
    assert report["fan"]["model_fan_input_max_W"] is None
    assert report["costs"]["listed_components_total_CHF"] is None


def test_bound_fan_uses_exported_efficiency(manager):
    result, binding = eng.load_result("reviewed:lewatit_humid", manager)
    baseline = eng.assess(result, binding)["fan"]
    result["shared_equipment"]["fan_model"] = {**baseline["assumptions"], "fan_efficiency": 0.275}
    assert eng.assess(result, binding)["fan"]["model_fan_input_max_W"] == pytest.approx(2 * baseline["model_fan_input_max_W"])
