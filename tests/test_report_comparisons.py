"""Check the published report and app tables against the completed simulations."""
import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/data"


def read_csv(filename):
    with (DATA / filename).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def published_rows(table_id):
    report = (ROOT / "model/report/project/index.html").read_text(encoding="utf-8")
    workspace = (ROOT / "docs/workspace/results.html").read_text(encoding="utf-8")
    fragment = report.split(f'<div id="{table_id}">', 1)[1].split("</div>", 1)[0].strip()
    app_table = workspace.split(f"<!-- {table_id} -->", 1)[1].split(f"<!-- /{table_id} -->", 1)[0]
    assert app_table.strip() == fragment
    return [[cell.text for cell in row] for row in ET.fromstring(fragment).findall("tbody/tr")]


def test_common_baseline_includes_matched_purolite_once():
    original = [r for r in read_csv("submission/results.csv") if r["group"] == "humid"]
    paired = {r["material"]: r for r in read_csv("resin-comparison-65rh.csv")}
    purolite = paired["Purolite A110"]
    repeat = paired["Lewatit, same solver"]
    saved = paired["Lewatit, saved reference"]
    assert purolite["case_sha256"] == repeat["case_sha256"] == saved["case_sha256"]
    assert purolite["source_sha256"] == repeat["source_sha256"]
    expected = sorted(original + [purolite], key=lambda r: -float(r["collected_CO2_kg_day"]))
    assert published_rows("purolite-a110-and-lewatit") == [
        [r["material"], f'{float(r["collected_CO2_kg_day"]):.3f}', f'{float(r["electricity_kWh_kg_CO2"]):.2f}']
        for r in expected
    ]


def test_weather_rows_and_fixed_inlet_use_full_horizon_product():
    weather = read_csv("pully-seasonal-one-pump-summary.csv")
    rows = published_rows("weather-comparison-table")
    assert len(rows) == len(weather) + 1
    for reported, run in zip(rows, weather):
        assert reported[0].startswith(run["season"].capitalize())
        total = float(run["kg_in_48h"])
        assert reported[1:] == [
            f'{float(run["mean_C"]):.1f}', f'{float(run["mean_RH_pct"]):.1f}',
            f"{total:.3f}", f"{total / 2:.3f}", f'{float(run["kWh_per_kg"]):.2f}',
        ]
    reference = json.loads((DATA / "bed-aging-lewatit.json").read_bytes())
    fresh = next(r for r in reference["runs"] if r["retained_pct"] == 100)
    assert rows[-1] == ["Fixed inlet, same cycle", "25.0", "65.0", f'{fresh["kg48"]:.3f}',
                        f'{fresh["kg48"] / 2:.3f}', f'{fresh["kwh_kg"]:.2f}']
