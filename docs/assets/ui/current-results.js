(() => {
  'use strict';
  const number = value => typeof value === 'number' && Number.isFinite(value) ? value.toFixed(3) : 'Not available';
  const cell = value => { const el = document.createElement('td'); el.textContent = value; return el; };
  fetch('/api/results').then(async response => {
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Results could not be loaded');
    for (const result of data.results) {
      const row = document.createElement('tr');
      row.append(cell(result.label), cell(result.origin), cell(number(result.production_kg_day)), cell(number(result.electricity_kWh_kg_CO2)));
      const links = cell('');
      const engineering = document.createElement('a');
      engineering.href = '/engineering/?result=' + encodeURIComponent(result.selector);
      engineering.textContent = 'Engineering';
      const download = document.createElement('a');
      download.href = '/api/result?result=' + encodeURIComponent(result.selector);
      download.textContent = 'JSON'; download.download = result.selector.replace(':', '_') + '.json';
      links.append(engineering, ' · ', download); row.append(links);
      document.getElementById('results-body').append(row);
    }
    document.getElementById('result-status').textContent = `${data.results.length} completed results available.`;
    const incomplete = data.incomplete_sensitivities.map(row => row.material).join(' and ');
    document.getElementById('incomplete-results').textContent = `${incomplete} faster-desorption sensitivities are incomplete and have no final performance value.`;
  }).catch(error => { document.getElementById('result-status').textContent = error.message; });
})();
