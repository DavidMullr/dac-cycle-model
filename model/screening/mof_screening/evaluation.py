"""Derived screening interpretation without changing process equations.

reference result fields remain intact so saved jobs are still readable. This
module adds an explicit interpretation that keeps advisory operating targets,
known hard limits, conservation, and numerical reliability separate.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable

from .weather import load_weather_window


EVALUATION_SCHEMA = "mof-screening-evaluation/1"


def _known_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _target_check(
    check_id: str,
    label: str,
    *,
    achieved: Any,
    target: Any,
    passed: Any,
    unit: str,
    direction: str,
) -> dict:
    known = (
        _known_number(achieved)
        and _known_number(target)
        and passed is not None
    )
    return {
        "id": check_id,
        "label": label,
        "known": known,
        "passed": None if not known or passed is None else bool(passed),
        "achieved": achieved if _known_number(achieved) else None,
        "target": target if _known_number(target) else None,
        "unit": unit,
        "direction": direction,
    }


def phase_endpoint_pressure(row: dict):
    """Use pump-end pressure during axial withdrawal and minimum pressure for refill."""
    axial = row.get("axial_pressure_Pa")
    if axial and row.get("phase") in ("evacuation", "desorption", "cooling"):
        return axial[-1]
    return row.get("min_pressure_Pa")


def _trace_phase_endpoint(
    result: dict, cycle: int, phase: str
) -> dict:
    rows = [
        row
        for row in result.get("traces") or []
        if int(row.get("cycle", -1)) == int(cycle)
        and row.get("phase") == phase
    ]
    if not rows:
        return {}
    row = max(rows, key=lambda item: float(item.get("elapsed_s", 0.0)))
    endpoint = {
        "temperature_end_K": row.get(
            "bed_temperature_min_K"
            if phase == "desorption"
            else "bed_temperature_max_K"
        ),
        "pressure_end_Pa": phase_endpoint_pressure(row),
    }
    top_endpoint = result.get("endpoint") or {}
    if phase == "desorption":
        target = top_endpoint.get("regeneration_temperature_target_K")
        endpoint["temperature_target_K"] = target
        if _known_number(target) and _known_number(
            endpoint.get("temperature_end_K")
        ):
            endpoint["temperature_ok"] = (
                endpoint["temperature_end_K"] >= target - 2.0
            )
        pressure_target = top_endpoint.get("desorption_pressure_target_Pa")
        endpoint["pressure_target_Pa"] = pressure_target
        if _known_number(pressure_target) and _known_number(
            endpoint.get("pressure_end_Pa")
        ):
            endpoint["pressure_ok"] = (
                endpoint["pressure_end_Pa"] <= 1.05 * pressure_target
            )
    elif phase == "cooling":
        case_weather = (
            (result.get("case") or {}).get("weather")
            if isinstance(result.get("case"), dict)
            else None
        )
        if case_weather is not None:
            weather = load_weather_window(case_weather)
            phase_weather = weather.at(float(row.get("elapsed_s", 0.0)))
            target = phase_weather.temperature_C + 273.15 + 0.5
        else:
            target = top_endpoint.get("cooling_temperature_target_K")
        endpoint["cooling_event_target_K"] = target
        if _known_number(target) and _known_number(
            endpoint.get("temperature_end_K")
        ):
            endpoint["cooling_event_ok"] = (
                endpoint["temperature_end_K"] <= target
            )
    elif phase == "repressurization":
        case_weather = (
            (result.get("case") or {}).get("weather")
            if isinstance(result.get("case"), dict)
            else None
        )
        if case_weather is not None:
            weather = load_weather_window(case_weather)
            phase_weather = weather.at(float(row.get("elapsed_s", 0.0)))
            endpoint["pressure_target_Pa"] = (
                0.98 * phase_weather.pressure_Pa
            )
        else:
            endpoint["pressure_target_Pa"] = top_endpoint.get(
                "repressurization_pressure_target_Pa"
            )
    return endpoint


def _cycle_evidence(result: dict) -> list[dict]:
    raw_cycles = result.get("cycles") or []
    horizon = result.get("horizon") or {}
    completed_cycles_value = horizon.get("cycles_completed")
    completed_cycles_known = completed_cycles_value is not None
    completed_cycles = (
        int(completed_cycles_value) if completed_cycles_known else 0
    )
    groups: dict[int, dict] = {}
    for row in raw_cycles:
        number = int(row.get("cycle", 1))
        group = groups.setdefault(
            number,
            {
                "cycle": number,
                "complete": False,
                "explicit_complete": None,
                "phases": {},
            },
        )
        if "phase" in row:
            phase = row["phase"]
            group["phases"][phase] = {
                "endpoint": deepcopy(row.get("endpoint") or {}),
                "start_s": row.get("start_elapsed_s"),
                "end_s": row.get("end_elapsed_s"),
                "duration_s": row.get("duration_s"),
            }
            continue
        group["complete"] = bool(row.get("complete"))
        group["explicit_complete"] = bool(row.get("complete"))
        completion = row.get("phase_completion") or {}
        evidence = {
            item.get("phase"): item
            for item in row.get("phase_evidence") or []
        }
        for window in row.get("windows") or []:
            phase = window.get("phase")
            phase_completion = completion.get(phase) or {}
            group["phases"][phase] = {
                "endpoint": _trace_phase_endpoint(result, number, phase),
                "start_s": window.get("start_s"),
                "end_s": window.get("end_s"),
                "duration_s": float(window.get("end_s", 0.0))
                - float(window.get("start_s", 0.0)),
                "complete": not bool(
                    phase_completion.get("horizon_truncated", False)
                ),
                "phase_evidence": evidence.get(phase),
            }

    for number, group in groups.items():
        if group["explicit_complete"] is None:
            group["complete"] = (
                number <= completed_cycles
                if completed_cycles_known
                else False
            )
        else:
            group["complete"] = group["explicit_complete"]
        if (
            group["explicit_complete"] is None
            and not completed_cycles_known
            and {
                "adsorption",
                "evacuation",
                "desorption",
                "cooling",
                "repressurization",
            }.issubset(group["phases"])
        ):
            group["complete"] = True
        for stage in group["phases"].values():
            if group["complete"]:
                stage["complete"] = True
            elif "complete" not in stage:
                stage["complete"] = False
        if not group["complete"] and group["phases"]:
            last_phase = next(reversed(group["phases"]))
            for phase in group["phases"]:
                if phase != last_phase:
                    group["phases"][phase]["complete"] = True
    return [groups[number] for number in sorted(groups)]


def _stage_check(
    stages: dict[str, dict],
    *,
    primary_phase: str,
    check_id: str,
    label: str,
    unit: str,
    direction: str,
    measurement: str,
    fallbacks: tuple[str, ...] = (),
) -> dict | None:
    unknown = None
    for phase in (primary_phase, *fallbacks):
        stage = stages.get(phase)
        if not stage:
            continue
        endpoint = stage.get("endpoint") or {}
        if measurement == "temperature":
            achieved = endpoint.get("temperature_end_K")
            target = endpoint.get("temperature_target_K")
            passed = endpoint.get("temperature_ok")
        elif measurement == "pressure":
            achieved = endpoint.get("pressure_end_Pa")
            target = endpoint.get("pressure_target_Pa")
            passed = endpoint.get("pressure_ok")
        elif measurement == "cooling":
            achieved = endpoint.get("temperature_end_K")
            target = endpoint.get(
                "cooling_event_target_K",
                endpoint.get("temperature_target_K"),
            )
            passed = endpoint.get(
                "cooling_event_ok", endpoint.get("temperature_ok")
            )
        else:
            achieved = target = passed = None
        check = _target_check(
            check_id,
            label,
            achieved=achieved,
            target=target,
            passed=passed,
            unit=unit,
            direction=direction,
        )
        check.update(
            {
                "cycle": stage.get("_cycle"),
                "phase": phase,
                "duration_s": stage.get("duration_s"),
                "phase_complete": bool(stage.get("complete", True)),
                "basis": "completed_cycle",
            }
        )
        if check["known"]:
            return check
        if unknown is None:
            unknown = check
    return unknown


def target_attainment(result: dict) -> dict:
    """Describe requested operating targets as advisory screening outcomes."""
    cycles = _cycle_evidence(result)
    completed = [cycle for cycle in cycles if cycle["complete"]]
    latest_completed = completed[-1] if completed else None
    endpoint = result.get("endpoint") or {}
    if cycles:
        stages = {}
        if latest_completed is not None:
            for phase, stage in latest_completed["phases"].items():
                stages[phase] = {**stage, "_cycle": latest_completed["cycle"]}
        checks = [
            _stage_check(
                stages,
                primary_phase="desorption",
                check_id="regeneration_temperature",
                label="Regeneration temperature",
                unit="K",
                direction="at_or_above",
                measurement="temperature",
            ),
            _stage_check(
                stages,
                primary_phase="desorption",
                check_id="desorption_pressure",
                label="Desorption pressure",
                unit="Pa",
                direction="at_or_below",
                measurement="pressure",
                fallbacks=("evacuation",),
            ),
            _stage_check(
                stages,
                primary_phase="cooling",
                check_id="cooling_temperature",
                label="Cooling temperature",
                unit="K",
                direction="at_or_below",
                measurement="cooling",
            ),
        ]
        checks = [
            check
            if check is not None
            else _target_check(
                check_id,
                label,
                achieved=None,
                target=None,
                passed=None,
                unit=unit,
                direction=direction,
            )
            for check, (check_id, label, unit, direction) in zip(
                checks,
                (
                    (
                        "regeneration_temperature",
                        "Regeneration temperature",
                        "K",
                        "at_or_above",
                    ),
                    (
                        "desorption_pressure",
                        "Desorption pressure",
                        "Pa",
                        "at_or_below",
                    ),
                    (
                        "cooling_temperature",
                        "Cooling temperature",
                        "K",
                        "at_or_below",
                    ),
                ),
            )
        ]
    else:
        checks = [
            _target_check(
                "regeneration_temperature",
                "Regeneration temperature",
                achieved=endpoint.get("regeneration_temperature_K"),
                target=endpoint.get("regeneration_temperature_target_K"),
                passed=endpoint.get("regeneration_temperature_pass"),
                unit="K",
                direction="at_or_above",
            ),
            _target_check(
                "desorption_pressure",
                "Desorption pressure",
                achieved=endpoint.get("desorption_pressure_Pa"),
                target=endpoint.get("desorption_pressure_target_Pa"),
                passed=endpoint.get("desorption_pressure_pass"),
                unit="Pa",
                direction="at_or_below",
            ),
            _target_check(
                "cooling_temperature",
                "Cooling temperature",
                achieved=endpoint.get("cooling_temperature_K"),
                target=endpoint.get("cooling_temperature_target_K"),
                passed=endpoint.get("cooling_pass"),
                unit="K",
                direction="at_or_below",
            ),
        ]
        for check in checks:
            check.update(
                {
                    "cycle": None,
                    "phase": None,
                    "duration_s": None,
                    "phase_complete": None,
                    "basis": "legacy_endpoint_fallback",
                }
            )
    incomplete = [cycle for cycle in cycles if not cycle["complete"]]
    partial_stage_observations = []
    for cycle in incomplete:
        for phase, stage in cycle["phases"].items():
            if stage.get("complete") is not False:
                continue
            endpoint_value = stage.get("endpoint") or {}
            if phase == "desorption":
                achieved = endpoint_value.get("temperature_end_K")
                target = endpoint_value.get("temperature_target_K")
            elif phase == "cooling":
                achieved = endpoint_value.get("temperature_end_K")
                target = endpoint_value.get(
                    "cooling_event_target_K",
                    endpoint_value.get("temperature_target_K"),
                )
            else:
                achieved = endpoint_value.get("pressure_end_Pa")
                target = endpoint_value.get("pressure_target_Pa")
            partial_stage_observations.append(
                {
                    "cycle": cycle["cycle"],
                    "phase": phase,
                    "duration_s": stage.get("duration_s"),
                    "achieved": achieved if _known_number(achieved) else None,
                    "target": target if _known_number(target) else None,
                    "unit": "K" if phase in {"desorption", "cooling"} else "Pa",
                    "counts_toward_completed_targets": False,
                    "label": "partial_stage_observation",
                }
            )
    known = [check for check in checks if check["known"]]
    unknown = [check for check in checks if not check["known"]]
    if not known:
        status = "not_available"
    elif any(check["passed"] is False for check in known):
        status = "advisory_miss_with_unknown" if unknown else "advisory_miss"
    elif unknown:
        status = "incomplete"
    else:
        status = "satisfied"
    return {
        "status": status,
        "advisory": True,
        "miss_alone_excludes_case": False,
        "miss_alone_ranks_case_down": False,
        "latest_completed_cycle": (
            None if latest_completed is None else latest_completed["cycle"]
        ),
        "completed_cycles_evaluated": [
            cycle["cycle"] for cycle in completed
        ],
        "partial_stage_observations": partial_stage_observations,
        "partial_stages_counted_as_completed_misses": False,
        "checks": checks,
        "statement": (
            "Requested regeneration, vacuum and cooling endpoints are "
            "advisory operating outcomes. A miss does not by itself reject "
            "or rank down a screening case. Completed-stage attainment uses "
            "the latest completed cycle; partial tails are reported "
            "separately and are not counted as completed-target misses."
        ),
    }


def _limit_check(
    check_id: str,
    label: str,
    value: Any,
    passed: Any,
    basis: str,
) -> dict:
    return {
        "id": check_id,
        "label": label,
        "known": passed is not None,
        "pass": None if passed is None else bool(passed),
        "value": value,
        "basis": basis,
    }


def hard_limits(result: dict) -> dict:
    """Report known physical/model-domain constraints without hiding gaps."""
    checks = result.get("checks") or {}
    assessment = result.get("temperature_guard_assessment")
    known_checks = []
    unassessed_temperature = None
    if assessment is not None:
        policy = assessment.get("policy") or {}
        if policy.get("physical_limit_known") is False:
            unassessed_temperature = (
                "The declared temperature scalar is an unqualified scenario "
                "guard; no evidenced physical degradation limit is recorded."
            )
        else:
            known_checks.append(
                _limit_check(
                    "material_temperature_limit",
                    "Physical degradation temperature limit",
                    assessment.get("peak_bed_temperature_C"),
                    assessment.get(
                        "physical_degradation_limit_pass"
                    ),
                    "Explicit evidenced physical degradation limit.",
                )
            )
    else:
        known_checks.append(
            _limit_check(
                "material_temperature_limit",
                "Material temperature limit",
                checks.get("temperature_limit_pass"),
                checks.get("temperature_limit_pass"),
                "Configured material temperature guard.",
            )
        )

    summary = (result.get("raw_model_evidence") or {}).get("summary") or {}
    flags = summary.get("flags") or {}
    flag_specs = (
        (
            "pump_curve_domain",
            "Pump curve domain",
            "pump_curve_upper_domain_exceeded",
        ),
        (
            "temperature_support",
            "Material temperature support",
            "temperature_support_excursion",
        ),
        (
            "water_support",
            "Water-law support",
            "water_support_excursion",
        ),
        (
            "saturation_domain",
            "Water saturation domain",
            "actual_supersaturation_excursion",
        ),
        (
            "correlation_domain",
            "Correlation domain",
            "correlation_domain_violation",
        ),
    )
    for check_id, label, flag in flag_specs:
        if flag in flags:
            observed = bool(flags[flag])
            known_checks.append(
                _limit_check(
                    check_id,
                    label,
                    observed,
                    not observed,
                    f"Engine diagnostic flag: {flag}",
                )
            )

    assessed = [check for check in known_checks if check["known"]]
    unassessed = [
        "Pump, conditioner, material-law and hardware qualification not "
        "recorded by this result remain unestablished."
    ]
    if unassessed_temperature is not None:
        unassessed.append(unassessed_temperature)
    if flags.get("frost_model_absent"):
        unassessed.append("A frost/ice model is absent.")
    if flags.get("conditioner_hardware_qualified") is False:
        unassessed.append(
            "The ideal pump-inlet conditioner is not qualified hardware."
        )
    if any(check["pass"] is False for check in assessed):
        status = "fail"
    elif not assessed:
        status = "not_assessed"
    elif unassessed:
        status = "partial"
    else:
        status = "pass"
    return {
        "status": status,
        "checks": known_checks,
        "unassessed_limits": unassessed,
    }


def conservation(result: dict) -> dict:
    checks = result.get("checks") or {}
    mass_pass = checks.get("mass_closure_pass")
    energy_pass = checks.get("energy_closure_pass")
    passed = bool(mass_pass is True and energy_pass is True)
    return {
        "status": "pass" if passed else "fail",
        "mass_pass": mass_pass,
        "energy_pass": energy_pass,
        "mass_relative_residual_by_species": deepcopy(
            checks.get("mass_relative_residual_by_species")
        ),
        "energy_relative_residual": checks.get("energy_relative_residual"),
        "energy_legacy_net_relative_residual": checks.get(
            "energy_legacy_net_relative_residual"
        ),
        "energy_legacy_net_closure_pass": checks.get(
            "energy_legacy_net_closure_pass"
        ),
        "energy_normalization": deepcopy(
            checks.get("energy_normalization")
        ),
    }


def numerical_reliability(result: dict) -> dict:
    document = deepcopy(result.get("numerical_qualification") or {})
    document["finite_observations"] = (
        result.get("checks") or {}
    ).get("finite_observations")
    document["separate_from_targets"] = True
    document["single_run_qualifies_family"] = False
    return document


def temperature_guard(result: dict) -> dict:
    assessment = deepcopy(result.get("temperature_guard_assessment"))
    if assessment is not None:
        return assessment
    return {
        "status": "legacy_unclassified",
        "policy": None,
        "statement": (
            "Saved result has no recorded temperature-policy "
            "classification."
        ),
    }


def annotate_result(result: dict) -> dict:
    """Attach the explicit derived evaluation to a new result document."""
    result["evaluation_schema"] = EVALUATION_SCHEMA
    result["target_attainment"] = target_attainment(result)
    result["temperature_guard"] = temperature_guard(result)
    result["hard_limits"] = hard_limits(result)
    result["conservation"] = conservation(result)
    result["numerical_reliability"] = numerical_reliability(result)
    return result


def evaluate_existing_result(result: dict) -> dict:
    """Interpret reference and current records without rewriting saved files."""
    if result.get("evaluation_schema") != EVALUATION_SCHEMA:
        return {
            "schema": EVALUATION_SCHEMA,
            "target_attainment": target_attainment(result),
            "temperature_guard": temperature_guard(result),
            "hard_limits": hard_limits(result),
            "conservation": conservation(result),
            "numerical_reliability": numerical_reliability(result),
        }
    return {
        "schema": EVALUATION_SCHEMA,
        "target_attainment": (
            target_attainment(result)
            if (result.get("engine_identity") == "axial-direct-steam/1"
                and (result.get("endpoint") or {}).get("desorption_pressure_basis") != "axial_outlet")
            else deepcopy(result["target_attainment"])
        ),
        "temperature_guard": temperature_guard(result),
        "hard_limits": deepcopy(result["hard_limits"]),
        "conservation": deepcopy(result["conservation"]),
        "numerical_reliability": deepcopy(result["numerical_reliability"]),
    }
