"""Declared energy-carrier requirements for an external heat source."""
from __future__ import annotations

from copy import deepcopy

from .common import ExpectedScenarioError
from .systems import SourceSpec, build_heating_source


class HeatSourceContractError(ExpectedScenarioError):
    """An external source cannot execute under the declared heat-source contract."""


def source_execution_diagnostic(
    config: dict,
    source: SourceSpec | None = None,
) -> dict:
    """Return a serializable source availability/execution diagnostic."""
    source = build_heating_source(config) if source is None else source
    if source is None:
        return {
            "schema": "heat-source-contract/1",
            "kind": "electrical",
            "carrier_applicable": False,
            "thermal_input_declared": True,
            "external_carrier_complete": True,
            "status": "not_applicable_by_electrical_profile",
            "thermal_input_J": None,
            "thermal_input_W": None,
            "unavailable_reason": None,
            "hardware_or_supply_readiness": "UNKNOWN",
            "jacket_pressure_boundary_readiness": "NOT_APPLICABLE_ELECTRICAL",
            "fluid_loop_or_steam_generator_readiness": "NOT_APPLICABLE_ELECTRICAL",
            "seals_fasteners_manifolds_readiness": "NOT_APPLICABLE_ELECTRICAL",
            "legacy_100kPa_construction_screen_establishes_700kPa_fitness": False,
            "source_active_phase": None,
            "source_active_outside_desorption": False,
        }
    declared = source.thermal_input_declared
    return {
        "schema": "heat-source-contract/1",
        "kind": source.kind,
        "carrier": source.carrier,
        "carrier_applicable": True,
        "thermal_input_declared": declared,
        "external_carrier_complete": declared,
        "status": "declared_for_model_execution" if declared
        else "refused_undeclared_thermal_conversion",
        "thermal_input_J": None,
        "thermal_input_W": None,
        "unavailable_reason": None if declared else (
            "The external source lacks a declared supply or generation "
            "conversion, so upstream thermal input cannot be quantified."
        ),
        "source": deepcopy(source.as_dict()),
        "hardware_or_supply_readiness": "UNKNOWN",
        "jacket_pressure_boundary_readiness": "UNKNOWN",
        "fluid_loop_or_steam_generator_readiness": "UNKNOWN",
        "seals_fasteners_manifolds_readiness": "UNKNOWN",
        "legacy_100kPa_construction_screen_establishes_700kPa_fitness": False,
        "source_active_phase": "desorption",
        "source_active_outside_desorption": False,
    }


def require_declared_thermal_conversion(
    config: dict,
    source: SourceSpec | None = None,
) -> dict:
    """Fail closed before integration when an external input is unquantified."""
    diagnostic = source_execution_diagnostic(config, source)
    if diagnostic["carrier_applicable"] and not diagnostic["external_carrier_complete"]:
        raise HeatSourceContractError(diagnostic["unavailable_reason"])
    return diagnostic
