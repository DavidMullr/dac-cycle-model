from copy import deepcopy
import io
import json
from pathlib import Path

import numpy as np
import pytest

from app.mof_screening import routes
from screening.mof_screening import young2021, young2021_hot
from screening.mof_screening.contract import (
    MofScreeningError,
    build_case,
    canonical_sha256,
    load_builtin_material,
    load_defaults,
    material_temperature_guard,
    preview_payload,
    validate_case,
    validate_material,
)
from screening.mof_screening.materials import build_material_adapter
from screening.mof_screening.manager import MofScreeningManager
from screening.mof_screening.preflight import execution_source_identity
from screening.mof_screening.runtime import (
    build_spatial_config,
)
from screening.mof_screening.weather import load_weather_window


STRICT_ID = "lewatit_vp_oc_1065_young2021_wadst_conditional"
HOT_ID = "lewatit_vp_oc_1065_young2021_hot116_conditional"
BROWSER_CASE_PATH = Path(__file__).resolve().parent / "fixtures/mof_screening/advisory_case.json"


class _Headers(dict):
    def get_all(self, key, default=None):
        value = self.get(key)
        return [value] if value is not None else ([] if default is None else default)


class _Handler:
    def __init__(self, payload, manager):
        body = json.dumps(payload).encode("utf-8")
        self.headers = _Headers(
            {
                "Content-Length": str(len(body)),
                "Content-Type": "application/json",
            }
        )
        self.rfile = io.BytesIO(body)
        self.mof_screening_manager = manager
        self.response = None

    def _allowed_host(self):
        return True

    def _allowed_origin(self):
        return True

    def json(self, status, payload):
        self.response = (status, payload)


def _preset(preset_id):
    return next(
        item
        for item in load_defaults()["case_presets"]
        if item["id"] == preset_id
    )


def test_hot_provider_uses_unchanged_young_coefficients_rates_and_heats():
    strict = load_builtin_material(STRICT_ID)
    hot = load_builtin_material(HOT_ID)
    assert hot["reference_temperature_K"] == strict["reference_temperature_K"]
    assert hot["solid_cp_J_kg_K"] == strict["solid_cp_J_kg_K"]
    assert hot["co2"]["heat_J_mol"] == strict["co2"]["heat_J_mol"]
    assert hot["water"]["heat_J_mol"] == strict["water"]["heat_J_mol"]
    assert hot["kinetics"]["adsorption_rate_s_inv"] == strict[
        "kinetics"
    ]["adsorption_rate_s_inv"]
    assert hot["kinetics"]["desorption_rate_s_inv"] == strict[
        "kinetics"
    ]["desorption_rate_s_inv"]
    assert hot["adsorbed_phase_cp_J_kg_K"] == strict[
        "adsorbed_phase_cp_J_kg_K"
    ]
    assert young2021_hot.young2021.YOUNG2021_DRY == young2021.YOUNG2021_DRY
    assert young2021_hot.young2021.YOUNG2021_WET == young2021.YOUNG2021_WET
    assert young2021_hot.young2021.YOUNG2021_GAB == young2021.YOUNG2021_GAB


def test_strict_young_rejects_116_but_hot_provider_accepts_explicit_envelope():
    defaults = load_defaults()
    strict = load_builtin_material(STRICT_ID)
    strict_case = deepcopy(defaults["case"])
    strict_case["cycle"]["regeneration_temperature_C"] = 116.0
    with pytest.raises(MofScreeningError, match="exceeds"):
        build_case(strict, strict_case)

    hot = load_builtin_material(HOT_ID)
    hot_case = _preset("young2021_hot116_48h_rh0")["case"]
    normalized, accepted = build_case(hot, hot_case)
    assert normalized["equilibrium_family"] == (
        "young2021_hot_wadst_gab_extrapolated"
    )
    assert accepted["cycle"]["regeneration_temperature_C"] == 116.0

    adapter = build_material_adapter(hot)
    with pytest.raises(ValueError, match="evaluation envelope"):
        adapter.equilibrium(
            393.16,
            np.array(
                [
                    400.0,
                    15_000.0,
                    0.0,
                    0.0,
                ]
            ),
        )

    excessive_ambient_rh = deepcopy(defaults["case"])
    excessive_ambient_rh["weather"]["relative_humidity"] = 0.56
    with pytest.raises(MofScreeningError, match="relative humidity"):
        build_case(hot, excessive_ambient_rh)


def test_hot_provider_is_spatial_only_and_lumped_is_withheld():
    hot = load_builtin_material(HOT_ID)
    case = deepcopy(_preset("young2021_hot116_48h_rh0")["case"])
    case["engine"] = "lumped"
    case["solver"]["cells"] = 1
    with pytest.raises(MofScreeningError, match="spatial engine only") as captured:
        build_case(hot, case)
    assert captured.value.field == "case.engine"


def test_hot_equilibrium_uses_iapws_local_rh_and_evolving_water():
    temperature_K = 389.15
    p_h2o = 0.09 * young2021_hot.iapws_saturation_pressure_Pa(temperature_K)
    co2, first_water, details = young2021_hot.equilibrium(
        temperature_K, 400.0, p_h2o
    )
    assert details["local_relative_humidity"] == pytest.approx(0.09)
    assert details["water_saturation_pressure_Pa"] == pytest.approx(
        174_771.44677280128
    )
    assert details["hot_binary_validated"] is False
    assert np.isfinite([co2, first_water]).all()

    adapter = build_material_adapter(load_builtin_material(HOT_ID))
    partials = np.array([400.0, p_h2o, 0.0, 0.0])
    dry_storage = np.array([0.1, 0.01])
    wet_storage = np.array([0.1, 1.0])
    dry_rates, _ = adapter.rates(temperature_K, partials, dry_storage)
    wet_rates, _ = adapter.rates(temperature_K, partials, wet_storage)
    dry_target = dry_storage[0] + dry_rates[0] / adapter._ad_rate[0]
    wet_target = wet_storage[0] + wet_rates[0] / adapter._ad_rate[0]
    assert wet_target > dry_target
    low_co2, _ = adapter.rates(
        temperature_K, partials, np.array([0.01, 1.0])
    )
    high_co2, _ = adapter.rates(
        temperature_K, partials, np.array([0.5, 1.0])
    )
    assert low_co2[1] == pytest.approx(high_co2[1])


def test_hot_presets_match_original_48h_baseline_and_keep_feed_local_rh_separate():
    dry = _preset("young2021_hot116_48h_rh0")
    humid = _preset("young2021_hot116_48h_rh030")
    browser_case = json.loads(BROWSER_CASE_PATH.read_text(encoding="utf-8"))
    assert dry["case"] == browser_case
    assert dry["case_sha256"] == (
        "c246e39357249046e9308bd53ef1478bb85aadda055ba0b806c820180e6bbe90"
    )
    assert humid["case"]["weather"]["temperature_C"] == 25.0
    assert humid["case"]["weather"]["relative_humidity"] == 0.3
    assert humid["case"]["cycle"]["regeneration_temperature_C"] == 116.0
    assert humid["case"]["cycle"]["regeneration_pressure_Pa"] == 16_000.0
    assert canonical_sha256(validate_case(humid["case"])) == humid[
        "case_sha256"
    ]


def test_hot_preview_and_source_report_expose_separate_evidence_ranges():
    material = load_builtin_material(HOT_ID)
    case = _preset("young2021_hot116_48h_rh030")["case"]
    normalized, accepted = build_case(material, case)
    preview = preview_payload(normalized, accepted, source="test")
    warning = " ".join(preview["warnings"])
    assert "measured Young coupled support ends at 70 C" in warning
    assert "no measured 116 C accuracy is claimed" in warning
    assert preview["material_id"] == HOT_ID
    assert material_temperature_guard(normalized)["matched"] is True

    adapter = build_material_adapter(normalized)
    report = adapter.source_report()
    assert report["measured_coupled_support_temperature_K"] == [
        298.15,
        343.15,
    ]
    assert report["evaluation_envelope_temperature_K"] == [
        298.15,
        393.15,
    ]
    assert report["saturation_pressure_law"] == "IAPWS SR1-86(1992)"
    assert report["hot_binary_validated"] is False
    assert report["hot_rate_validated"] is False
    assert report["hot_energy_validated"] is False
    assert report["lumped_engine_withheld"] is True

    weather = load_weather_window(accepted["weather"]).rows[0]
    config = build_spatial_config(normalized, accepted, weather)
    assert config["ambient"]["water_pressure_law"] == "Buck 1996"
    assert config["ambient"]["local_bed_water_pressure_law"] == (
        "IAPWS SR1-86(1992)"
    )


@pytest.mark.parametrize(
    "preset_id",
    ("young2021_hot116_48h_rh0", "young2021_hot116_48h_rh030"),
)
def test_api_preview_returns_hot_warning_without_starting_a_job(
    tmp_path, preset_id
):
    manager = MofScreeningManager(
        tmp_path / "jobs", execution_lock_path=tmp_path / "LOCK.json"
    )
    preset = _preset(preset_id)
    handler = _Handler(
        {
            "material": load_builtin_material(HOT_ID),
            "case": preset["case"],
        },
        manager,
    )
    try:
        assert routes.post(handler, "/api/mof-screening/preview") is True
        assert handler.response[0] == 200
        preview = handler.response[1]
        assert preview["material_id"] == HOT_ID
        assert preview["case_sha256"] == preset["case_sha256"]
        assert any(
            "HOT CONDITIONAL EXTRAPOLATION" in warning
            for warning in preview["warnings"]
        )
        assert manager.status()["state"] is None
    finally:
        manager.close()


def test_source_identity_binds_hot_module_material_and_configs():
    identity = execution_source_identity()
    hashes = identity["source_hashes"]
    for relative in (
        "screening/mof_screening/young2021.py",
        "screening/mof_screening/young2021_hot.py",
        "screening/mof_screening/young2021_cold_hot.py",
        "materials/screening/lewatit_vp_oc_1065_young2021_wadst_conditional.json",
        "materials/screening/lewatit_vp_oc_1065_young2021_hot116_conditional.json",
        "materials/screening/lewatit_vp_oc_1065_young2021_cold_hot116_conditional.json",
        "config/mof_screening/defaults.json",
        "config/mof_screening/temperature_registry.json",
    ):
        assert len(hashes[relative]) == 64
    assert identity["frozen_closure"]["sha256"] == hashes["config/mof_screening/dependencies.json"]


@pytest.mark.parametrize(
    ("mutator", "flag"),
    (
        (
            lambda material: material["kinetics"][
                "adsorption_rate_s_inv"
            ].__setitem__(0, 0.0031),
            "young_ldf_rates_unchanged",
        ),
        (
            lambda material: material["water"].__setitem__(
                "heat_J_mol", 46_500.0
            ),
            "young_process_heats_unchanged",
        ),
    ),
)
def test_hot_edits_cannot_retain_unchanged_source_flags(mutator, flag):
    material = deepcopy(load_builtin_material(HOT_ID))
    mutator(material)
    normalized = validate_material(material)
    assert normalized["evidence"]["status"] == "USER_EDITED_ASSUMPTION"
    assert normalized["evidence"]["edit_detected"] is True

    adapter_report = build_material_adapter(normalized).source_report()
    assert adapter_report[flag] is False
    assert adapter_report["source_reproduction_complete"] is False

    case = deepcopy(_preset("young2021_hot116_48h_rh0")["case"])
    case["temperature_policy"]["mode"] = "stop_at_scenario_guard"
    normalized_case = build_case(normalized, case)[1]
    warnings = " ".join(
        preview_payload(normalized, normalized_case, source="test")["warnings"]
    )
    assert "source-reproduction claims are withdrawn" in warnings
