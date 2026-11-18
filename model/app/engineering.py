"""Engineering duties and user-entered costs bound to a selected complete result."""
from __future__ import annotations
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import threading

from screening.airflow import airflow
from screening.geometry import build_bare_geometry
from screening.mof_screening.contract import MofScreeningError
from screening.mof_screening.runtime import PROFILE_PATH
from screening.mof_screening.weather import load_weather_window

ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / 'docs/data/submission/results.json'
COMPONENTS = ('Adsorbent', 'Vessel and supports', 'Heater and controls', 'Insulation',
              'Fan', 'Vacuum pump', 'Cooling and condensate handling', 'Valves and tubing',
              'Sensors and instrumentation', 'Assembly')
LOCK = threading.Lock()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def error(message, status=400):
    raise MofScreeningError(message, status=status)


def load_result(selector, manager):
    if not isinstance(selector, str):
        error('Choose a completed result')
    if selector.startswith('reviewed:'):
        index = read(INDEX)
        row = next((r for r in index['rows'] if Path(r['result']).stem == selector[9:]), None)
        if row is None:
            error('Reviewed result not found', 404)
        path = (ROOT / row['result']).resolve()
        if not path.is_relative_to(ROOT / 'docs/data/submission/results'):
            error('Result path is outside the reviewed result directory')
        raw = path.read_bytes()
        binding = next(r for r in index['bindings'] if r['path'] == row['result'])
        if sha(raw) != binding['sha256']:
            error('Reviewed result identity has changed', 409)
        label = row['material'] + ' · ' + row['group'].replace('_', ' ')
    elif re.fullmatch(r'job:[a-f0-9]{32}', selector):
        job_id = selector[4:]
        if manager._state(job_id).get('state') != 'completed':
            error('Engineering checks require a completed job')
        raw = (manager.job_root / job_id / 'RESULT.json').read_bytes()
        label = 'Local completed run'
    else:
        error('Result identifier is invalid')
    result = json.loads(raw)
    if result.get('failure') is not None or not result.get('horizon', {}).get('complete'):
        error('Engineering checks require a complete simulation horizon')
    return result, {'selector':selector, 'label':label, 'result_sha256':sha(raw),
                    'case_label':result.get('case_label'),
                    'material':result.get('material', {}).get('name'),
                    'input_identity':result.get('input_identity'),
                    'source_identity':result.get('source_identity')}


def catalogue(manager):
    index = read(INDEX)
    rows = [{'selector':'reviewed:' + Path(row['result']).stem, 'label':row['material'] + ' · ' + row['group'].replace('_', ' '),
             'origin':'Reviewed export', 'production_kg_day':row['collected_CO2_kg_day'],
             'electricity_kWh_kg_CO2':row['electricity_kWh_kg_CO2']} for row in index['rows']]
    for job in manager.history():
        if job.get('state') != 'completed':
            continue
        try:
            result, binding = load_result('job:' + job['job_id'], manager)
            rows.insert(0, {'selector':binding['selector'], 'label':result.get('case_label') or job['job_id'],
                           'origin':'Local completed run',
                           'production_kg_day':result['metrics'].get('collected_CO2_kg_day'),
                           'electricity_kWh_kg_CO2':result['metrics'].get('electricity_kWh_kg_CO2')})
        except (ValueError, OSError):
            continue
    return {'results':rows, 'incomplete_sensitivities':index['incomplete_sensitivities']}


def finite(value, name, *, minimum=0, optional=False):
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum:
        error(name + ' must be a finite number at least ' + str(minimum))
    return float(value)


def default_costs():
    return {'electricity_CHF_kWh':None,
            'components':[{'name':name, 'quantity':1, 'unit_cost_CHF':None, 'basis':''} for name in COMPONENTS]}


def costs_for(value, metrics):
    if not isinstance(value, dict) or set(value) != {'electricity_CHF_kWh', 'components'}:
        error('Cost worksheet must contain electricity_CHF_kWh and components')
    price = finite(value['electricity_CHF_kWh'], 'Electricity price', optional=True)
    if not isinstance(value['components'], list) or not 1 <= len(value['components']) <= 30:
        error('Provide between 1 and 30 cost components')
    items = []
    names = set()
    for row in value['components']:
        if not isinstance(row, dict) or set(row) != {'name', 'quantity', 'unit_cost_CHF', 'basis'}:
            error('Cost components require name, quantity, unit_cost_CHF and basis')
        name, basis = row['name'], row['basis']
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 100 or name.strip() in names:
            error('Cost component names must be unique and nonempty')
        if not isinstance(basis, str) or len(basis) > 500:
            error('Cost basis must be text of at most 500 characters')
        quantity = finite(row['quantity'], 'Quantity', minimum=0.000001)
        unit = finite(row['unit_cost_CHF'], 'Unit cost', optional=True)
        if unit is not None and not basis.strip():
            error('Each entered price, including zero, needs a quote or assumption basis')
        names.add(name.strip())
        items.append({'name':name.strip(), 'quantity':quantity, 'unit_cost_CHF':unit, 'basis':basis.strip(),
                      'subtotal_CHF':None if unit is None else quantity * unit})
    missing = [row['name'] for row in items if row['subtotal_CHF'] is None]
    subtotal = sum(row['subtotal_CHF'] or 0 for row in items)
    intensity = metrics['electricity_kWh_kg_CO2']
    daily = None if price is None else price * metrics['electricity_kWh_day']
    return {'electricity_CHF_kWh':price, 'components':items, 'known_subtotal_CHF':subtotal,
            'listed_components_total_CHF':None if missing else subtotal, 'unpriced_components':missing,
            'electricity_cost_CHF_day':daily,
            'electricity_cost_CHF_kg_CO2':None if price is None or intensity is None else price * intensity,
            'scope':'Listed components and modeled electricity only. Quotes and completeness require review; no installed-cost or life-cycle claim.'}


def fan_duties(result):
    case = result['case']
    equipment = case['equipment']
    flow = equipment['airflow_m3_s']
    duties = {'airflow_m3_s': flow, 'bed_pressure_drop_max_Pa': None, 'model_fan_input_max_W': None}
    settings = result['shared_equipment'].get('fan_model')
    if settings is None:
        # Older exports identify their hydraulic assumptions by the profile hash.
        hashes = result.get('source_identity', {}).get('source_hashes', {})
        candidates = [digest for path, digest in hashes.items()
                      if path.startswith('config/mof_screening/') and path.endswith('spatial_profile.json')]
        if len(candidates) == 1 and re.fullmatch(r'[0-9a-f]{64}', candidates[0]):
            saved = ROOT / 'docs/data/submission/source_profiles' / (candidates[0] + '.json')
            profile = saved if saved.is_file() else PROFILE_PATH
            raw = profile.read_bytes()
            if sha(raw) == candidates[0]:
                settings = json.loads(raw)['airflow']
    if settings is None:
        return {**duties, 'scope': 'Fan assumptions unavailable for this result.'}
    weather = load_weather_window(case['weather'])
    if weather.mode == 'official' and weather.source_sha256 != result['weather'].get('source_sha256'):
        return {**duties, 'scope': 'Weather source differs from the selected result.'}
    # Reconstruct only the saved cassette hydraulics, independent of today's material catalogue.
    config = {
        'bed': {'dry_sorbent_mass_kg': equipment['dry_shaped_mass_kg'],
                'bed_void_fraction': equipment['bed_void_fraction'],
                'pellet_density_kg_m3': result['geometry']['pellet_density_kg_m3'],
                'particle_diameter_m': equipment['particle_diameter_m']},
        'geometry': {'kind': 'cassette', 'bed_thickness_m': equipment['bed_thickness_m'],
                     'metal_thickness_m': 0, 'metal_density_kg_m3': 0, 'frame_mass_kg': 0,
                     'extra_dead_volume_m3': 0},
        'airflow': {**settings, 'volume_flow_m3_s': flow},
        'conditioning': {'outlet_relative_humidity': None},
    }
    geometry = build_bare_geometry(config)
    rows = [airflow({**config, 'ambient': interval.as_dict()}, geometry) for interval in weather.rows]
    return {**duties, 'bed_pressure_drop_max_Pa': max(row['pressure_drop_Pa'] for row in rows),
            'model_fan_input_max_W': max(row['fan_power_W'] for row in rows),
            'assumptions': config['airflow'],
            'scope': 'Packed bed and configured minor loss at imposed flow. Maximum across weather inputs; external duct and filter losses require measurements.'}


def assess(result, binding, costs=None):
    case = result['case']
    metrics = result['metrics']
    seconds = finite(result['horizon']['simulated_s'], 'Simulated duration', minimum=1)
    days = seconds / 86400
    components = metrics['electricity_components_kWh']
    total = sum(finite(v, 'Electrical component') for v in components.values())
    if not math.isclose(total, metrics['electricity_kWh'], rel_tol=1e-9, abs_tol=1e-8):
        error('Electrical component accounting does not reconcile')
    if not math.isclose(total / days, metrics['electricity_kWh_day'], rel_tol=1e-9, abs_tol=1e-8):
        error('Daily electricity does not match the complete horizon')
    shared = result['shared_equipment']
    equipment = case['equipment']
    traces = result.get('traces', [])
    pressures = [row['min_pressure_Pa'] for row in traces if isinstance(row.get('min_pressure_Pa'), (int, float))]
    def daily_water(key):
        value = metrics.get(key)
        return None if value is None else finite(value, key) * 0.01801528 / days
    return {
        'schema':'dac-current-run-engineering/1', 'binding':binding,
        'scope':'Duties of the selected simulation and explicitly entered costs; no new cycle integration.',
        'simulated_hours':seconds / 3600,
        'production':{'collected_CO2_kg_day':metrics['collected_CO2_kg_day'],
                      'dry_product_CO2_fraction':metrics.get('dry_product_CO2_fraction'),
                      'wet_product_CO2_fraction':metrics.get('wet_product_CO2_fraction')},
        'bed':{'dry_shaped_mass_kg':equipment['dry_shaped_mass_kg'],
               'volume_m3':equipment['bed_face_area_m2'] * equipment['bed_thickness_m'],
               'face_area_m2':equipment['bed_face_area_m2'], 'thickness_m':equipment['bed_thickness_m'],
               'particle_diameter_m':equipment['particle_diameter_m'], 'void_fraction':equipment['bed_void_fraction']},
        'fan':fan_duties(result),
        'thermal':{'heater_request_W':equipment['heater_power_W'],
                   'combined_cooling_limit_W':shared['combined_cooling_power_full_W'],
                   'cooling_COP':shared['cooling_cop'],
                   'regeneration_request_C':case['cycle']['regeneration_temperature_C'],
                   'observed_peak_bed_C':None if metrics.get('peak_bed_temperature_K') is None else metrics['peak_bed_temperature_K'] - 273.15,
                   'heater_average_electric_W':None if components.get('heater') is None else components['heater'] * 3.6e6 / seconds,
                   'cooling_average_electric_W':None if components.get('cooling') is None else components['cooling'] * 3.6e6 / seconds},
        'vacuum':{'regeneration_request_Pa':case['cycle']['regeneration_pressure_Pa'],
                  'lowest_saved_cell_pressure_Pa':min(pressures) if pressures else None,
                  'running_input_W':shared['pump_running_input_W'],
                  'simultaneous_vacuum_and_pressurized_discharge_verified':False},
        'water':{'condensate_kg_day':daily_water('water_condensed_mol'),
                 'water_in_product_kg_day':daily_water('water_in_product_mol')},
        'electricity':{'kWh_day':total / days, 'kWh_kg_CO2':metrics['electricity_kWh_kg_CO2'],
                       'average_input_W':total * 3.6e6 / seconds,
                       'components_kWh_day':{key:value / days for key, value in components.items()}},
        'costs':costs_for(default_costs() if costs is None else costs, metrics),
        'open_inputs':['Actual prototype dimensions and wire layout', 'Material temperature limit and local hot spots',
                       'Measured complete-system pressure loss', 'Pump flow at simultaneous inlet vacuum and discharge pressure',
                       'Supplier quotes, component life and installation scope'],
    }


def worksheet_path(binding, manager):
    return manager.output_root.parent / 'engineering' / (binding['result_sha256'] + '.json')


def read_worksheet(binding, manager):
    path = worksheet_path(binding, manager)
    if not path.is_file():
        return {'revision':0, 'costs':default_costs()}
    data = read(path)
    if data.get('result_sha256') != binding['result_sha256']:
        error('Cost worksheet belongs to a different result', 409)
    return data


def calculate(selector, manager, costs=None):
    result, binding = load_result(selector, manager)
    stored = read_worksheet(binding, manager)
    assessment = assess(result, binding, stored['costs'] if costs is None else costs)
    assessment['worksheet_revision'] = stored['revision']
    return assessment


def save(selector, manager, costs, expected_revision):
    result, binding = load_result(selector, manager)
    costs_for(costs, result['metrics'])
    if isinstance(expected_revision, bool) or not isinstance(expected_revision, int) or expected_revision < 0:
        error('A nonnegative integer worksheet revision is required')
    with LOCK:
        stored = read_worksheet(binding, manager)
        if stored['revision'] != expected_revision:
            error('Worksheet changed; reload before saving', 409)
        path = worksheet_path(binding, manager)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {'schema':'dac-run-costs/1', 'result_sha256':binding['result_sha256'],
                'selector':selector, 'revision':expected_revision + 1, 'costs':costs}
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
        temporary.replace(path)
    return calculate(selector, manager)


def csv_text(report):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['field', 'value'])
    def visit(prefix, value):
        if isinstance(value, dict):
            for key, item in value.items():
                visit(prefix + '.' + key if prefix else key, item)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(prefix + '.' + str(index), item)
        else:
            text = '' if value is None else str(value)
            if text.startswith(('=', '+', '-', '@', '\t', '\r')):
                text = "'" + text
            writer.writerow([prefix, text])
    visit('', report)
    return output.getvalue()
