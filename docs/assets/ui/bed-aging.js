/* Capacity sensitivity of the one-pump Lewatit electrical cycle. */
(async () => {
  'use strict';

  const byId = id => document.getElementById(id);
  const basis = byId('aging-basis');
  const cycles = byId('aging-cycles');
  const measured = byId('aging-measured');
  const status = byId('aging-status');
  const results = byId('estimate-results');
  const estimates = byId('estimate-rows');
  const metrics = ['steady_kg_day', 'kg48', 'kwh_kg'];

  function appendRow(body, values) {
    const row = body.insertRow();
    const label = document.createElement('th');
    label.scope = 'row';
    label.textContent = values[0];
    row.append(label);
    for (const value of values.slice(1)) row.insertCell().textContent = value;
    return row;
  }

  function format(run) {
    return [
      run.retained_pct.toLocaleString('en', {maximumFractionDigits: 2}) + '%',
      run.steady_kg_day.toFixed(3), run.kg48.toFixed(3), run.kwh_kg.toFixed(2)
    ];
  }

  try {
    const response = await fetch('/docs/data/bed-aging-lewatit.json');
    if (!response.ok) throw new Error('Reference runs unavailable');
    const {benchmark, runs} = await response.json();
    runs.sort((a, b) => a.retained_pct - b.retained_pct);
    const minimum = runs[0].retained_pct;
    const fresh = runs[runs.length - 1];
    const referenceRows = byId('reference-rows');
    for (const run of runs.slice().reverse()) {
      const row = appendRow(referenceRows, format(run));
      const link = document.createElement('a');
      link.href = '/docs/data/' + encodeURIComponent(run.source_export);
      link.download = run.source_export;
      link.textContent = 'JSON.gz';
      link.setAttribute('aria-label', 'Download simulation at ' + run.retained_pct + '% capacity');
      row.insertCell().append(link);
    }
    cycles.max = benchmark.cycles;

    function performance(capacity) {
      const upperIndex = runs.findIndex(run => run.retained_pct >= capacity);
      const upper = runs[upperIndex];
      if (upper.retained_pct === capacity) return upper;
      const lower = runs[upperIndex - 1];
      const fraction = (capacity - lower.retained_pct) / (upper.retained_pct - lower.retained_pct);
      const result = {retained_pct: capacity};
      for (const key of metrics) result[key] = lower[key] + fraction * (upper[key] - lower[key]);
      return result;
    }

    function render() {
      const useCycles = basis.value === 'cycles';
      const input = useCycles ? cycles : measured;
      byId('cycles-field').hidden = !useCycles;
      byId('measured-field').hidden = useCycles;
      cycles.disabled = !useCycles;
      measured.disabled = useCycles;
      cycles.removeAttribute('aria-invalid');
      measured.removeAttribute('aria-invalid');
      estimates.replaceChildren();
      results.hidden = true;

      const value = input.valueAsNumber;
      if (!Number.isFinite(value) || value < 0 || (useCycles && (!Number.isInteger(value) || value > benchmark.cycles))) {
        input.setAttribute('aria-invalid', 'true');
        status.textContent = useCycles
          ? `Enter a whole number from 0 to ${benchmark.cycles}. For a bed with more cycles, use a measured capacity.`
          : 'Enter the measured capacity as a percentage of fresh capacity.';
        return;
      }
      if (!useCycles && (value < minimum || value > fresh.retained_pct)) {
        status.textContent = `The reference runs cover ${minimum}% to ${fresh.retained_pct}%. A simulation at this resin condition is needed to estimate performance.`;
        return;
      }

      appendRow(estimates, ['Fresh reference', ...format(fresh)]);
      if (useCycles) {
        const elapsed = value / benchmark.cycles;
        const endpoint = benchmark.retained_pct;
        const linear = fresh.retained_pct + (endpoint - fresh.retained_pct) * elapsed;
        const exponential = fresh.retained_pct * (endpoint / fresh.retained_pct) ** elapsed;
        appendRow(estimates, ['Linear loss', ...format(performance(linear))]);
        appendRow(estimates, ['Exponential loss', ...format(performance(exponential))]);
        status.textContent = `Estimated performance after ${value} completed cycles under the two benchmark scenarios.`;
      } else {
        const exact = runs.some(run => run.retained_pct === value);
        appendRow(estimates, ['Measured resin condition', ...format(performance(value))]);
        status.textContent = exact
          ? 'Performance comes from the completed simulation at this resin capacity.'
          : 'Performance is interpolated between the two nearest simulated capacities.';
      }
      results.hidden = false;
    }

    basis.disabled = false;
    basis.addEventListener('change', render);
    cycles.addEventListener('input', render);
    measured.addEventListener('input', render);
    render();
  } catch {
    status.textContent = 'The aging reference runs could not be loaded. Reload the page to try again.';
  }
})();
