"""Verify published result identities and independently recompute headline metrics."""
from __future__ import annotations

import hashlib
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def check(root=ROOT):
    summary = read(root / 'docs/data/submission/results.json')
    errors = []
    bindings = {row['path']: row for row in summary['bindings']}
    expected_paths = {row['result'] for row in summary['rows']}
    actual_paths = {p.relative_to(root).as_posix() for p in (root / 'docs/data/submission/results').glob('*.json')}
    if expected_paths != actual_paths or set(bindings) != actual_paths or len(expected_paths) != len(summary['rows']):
        errors.append('Result index, bindings and export directory differ')
    for row in summary['rows']:
        path = root / row['result']
        if hashlib.sha256(path.read_bytes()).hexdigest() != bindings[row['result']]['sha256']:
            errors.append(row['result'] + ': published bytes changed')
        result = read(path)
        metrics = result['metrics']
        hours = result['case']['weather']['hours']
        case = dict(result['case'])
        case.pop('label', None)
        case_hash = hashlib.sha256(json.dumps(case, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        if case_hash != row['case_without_label_sha256']:
            errors.append(row['result'] + ': operating-input identity differs')
        if result['source_identity']['release_sha256'] != row['source_release_sha256']:
            errors.append(row['result'] + ': executable-source identity differs')
        horizon = result['horizon']
        if not horizon.get('complete') or horizon.get('simulated_s') != 172800 or horizon.get('requested_s') != 172800:
            errors.append(row['result'] + ': incomplete 48-hour horizon')
        if hours != 48 or result['case']['equipment']['dry_shaped_mass_kg'] != 10:
            errors.append(row['result'] + ': comparison basis changed')
        # Read the raw half-bed meters including the final partial cycle, independently of the summary.
        cycles, layout = result['cycles'], result['state_layout']
        states = [cycle['state_after'] for cycle in cycles]
        product = [2 * sum(state[layout['product'][0] + i] for state in states) for i in range(4)]
        def full_device_kWh(*meters):
            return 2 * sum(state[layout[key]] for state in states for key in meters) / 3.6e6

        product_kg = product[0] * 0.0440095
        total_kWh = full_device_kWh('E_electric_total')
        # These reviewed cases share one COP for wall and pre-pump cooling and use no dryer.
        components = {
            'heater': full_device_kWh('E_heat'),
            'fan': full_device_kWh('E_fan'),
            'pump': full_device_kWh('E_vac'),
            'cooling': full_device_kWh('E_cool', 'e_cooler') / result['shared_equipment']['cooling_cop'],
            'conditioning': full_device_kWh('E_cond', 'e_source_aux', 'e_cooler_aux'),
        }
        saved_components = metrics['electricity_components_kWh']
        if components.keys() != saved_components.keys() or any(
            not math.isclose(value, saved_components[key], rel_tol=1e-10, abs_tol=1e-10)
            for key, value in components.items()
        ):
            errors.append(row['result'] + ': electrical components differ from raw meters')
        if not math.isclose(sum(components.values()), total_kWh, rel_tol=1e-10, abs_tol=1e-10):
            errors.append(row['result'] + ': raw electrical meters do not reconcile')
        expected = {
            'collected_CO2_kg': product_kg,
            'collected_CO2_kg_day': product_kg * 24 / hours,
            'electricity_kWh': total_kWh,
            'electricity_kWh_day': total_kWh * 24 / hours,
            'electricity_kWh_kg_CO2': total_kWh / product_kg,
            'wet_product_CO2_fraction': product[0] / sum(product),
            'dry_product_CO2_fraction': product[0] / (product[0] + product[2] + product[3]),
            'water_in_product_mol': product[1],
            'water_condensed_mol': 2 * sum(state[layout['condensate']] for state in states),
        }
        if not math.isclose(sum(c['cycle_time_s'] for c in cycles), horizon['simulated_s'], abs_tol=1e-6):
            errors.append(row['result'] + ': recorded cycle times do not cover the horizon')
        if row['metrics'] != metrics or row['horizon'] != horizon:
            errors.append(row['result'] + ': indexed metrics or horizon differ from the export')
        source_hash = hashlib.sha256(json.dumps(result['source_identity']['source_hashes'], sort_keys=True,
                                                separators=(',', ':')).encode()).hexdigest()
        if source_hash != row['source_release_sha256']:
            errors.append(row['result'] + ': source hash list differs from its identity')
        for name, value in expected.items():
            if not math.isclose(value, metrics[name], rel_tol=1e-10, abs_tol=1e-10):
                errors.append(row['result'] + ': inconsistent ' + name)
        for name in ('collected_CO2_kg_day', 'electricity_kWh_kg_CO2', 'dry_product_CO2_fraction', 'wet_product_CO2_fraction'):
            if row[name] != metrics[name]:
                errors.append(row['result'] + ': summary differs from result ' + name)
        audit = root / row['audit']
        if hashlib.sha256(audit.read_bytes()).hexdigest() != row['audit_sha256']:
            errors.append(row['audit'] + ': audit bytes changed')
        if row['group'] == 'humid' and row['case_without_label_sha256'] != summary['primary_case_without_label_sha256']:
            errors.append(row['result'] + ': primary cases differ')
        if result.get('failure') is not None:
            errors.append(row['result'] + ': failed result presented as complete')
    with (root / 'docs/data/submission/results.csv').open(encoding='utf-8', newline='') as stream:
        csv_rows = list(csv.DictReader(stream))
    expected_rows = {(row['material'], row['group']): row for row in summary['rows'] + summary['incomplete_sensitivities']}
    if len(csv_rows) != len(expected_rows) or {(row['material'], row['group']) for row in csv_rows} != set(expected_rows):
        errors.append('results.csv: result selection differs')
    else:
        for csv_row in csv_rows:
            row = expected_rows[(csv_row['material'], csv_row['group'])]
            for key in csv_row:
                expected_value = row.get(key, '')
                value = float(csv_row[key]) if isinstance(expected_value, (int, float)) else (None if expected_value is None and csv_row[key] == '' else csv_row[key])
                if value != expected_value:
                    errors.append('results.csv: value differs for ' + row['slug'] + '/' + key)
    for row in summary['incomplete_sensitivities']:
        if any(row.get(key) is not None for key in ('result', 'metrics', 'electricity_kWh_kg_CO2', 'collected_CO2_kg_day')) or row['status'] != 'incomplete':
            errors.append(row['slug'] + ': incomplete attempt has a performance claim')
    for row in read(root / 'docs/data/submission/solver_source/manifest.json')['files']:
        if hashlib.sha256((root / row['path']).read_bytes()).hexdigest() != row['sha256']:
            errors.append(row['path'] + ': numerical-source snapshot changed')
    return {'status': 'PASS' if not errors else 'FAIL', 'completed_results': len(summary['rows']),
            'incomplete_attempts': len(summary['incomplete_sensitivities']), 'errors': errors,
            'new_simulations': 0, 'physical_validation_claimed': False}


if __name__ == '__main__':
    result = check()
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['status'] == 'PASS' else 1)
