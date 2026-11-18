"""Dimension-based cassette metal inventory.

Face sheets, sidewalls, a frame and an external support grid contribute mass.
Sheet mass is area times thickness times density. The grid volume subtracts rib
crossings once. The assembly replaces the bare geometry's metal term; it is not
added on top of that term. A single declared heat capacity supplies wall inertia.

Internal obstructions, heated fins and added contact-area claims are refused
because their flow, volume and heat-transfer effects are not represented.
This module does not check stresses, buckling, welds or pressure-vessel fitness.
The two-face heat path is an idealized contact boundary, not a verified design."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

REQUIRED_SECTIONS = ("face_sheets", "sidewalls", "frame")
OPTIONAL_SECTIONS = ("support_grid",)

SECTION_KEYS = {
    "face_sheets": ("thickness_m", "density_kg_m3", "area_multiplier"),
    "sidewalls": ("thickness_m", "density_kg_m3", "perimeter_m", "height_m"),
    "frame": ("mass_kg",),
    "support_grid": ("span_x_m", "span_y_m", "ribs_x", "ribs_y", "thickness_m", "density_kg_m3",
                     "rib_depth_m", "inside_envelope", "fins_are_heated"),
}

REFUSED_SECTIONS = {
    "internal_plates": ("impermeable internal plates normal to the declared flow cannot leave the gas path "
                        "unchanged, and no consistent perforated/channel representation is implemented, so "
                        "the option is refused instead of adding metal with no flow consequence"),
    "heated_fins": ("a heated-fin area would add thermal area without a matching gas-flow path; heated area "
                    "and heat path stay at the geometry builder's declared values"),
    "fins": ("a finned insert is not implemented; heated area and heat path stay at the geometry builder's "
             "declared values"),
}

REFUSAL_REASONS = {
    "inside_envelope": ("an inside-envelope grid displaces bed volume and blocks gas flow, so the bed void, the "
                        "flow area and the heat path would all have to change together; this bounded module "
                        "accounts for external support mass only"),
    "fins_are_heated": ("treating the grid as heated would add thermal area with no declared gas-flow path; "
                        "the grid adds mass only and heated area stays exactly at the geometry builder's value"),
}

CONVENTION_BASIS = ("Optional construction keys default to labelled analyst conventions: both faces are "
                    "exactly the bed plan area, the plan is square, the sidewall band is as high as the bed is "
                    "thick, undeclared rib counts are zero and a rib is as deep as it is wide. Every default "
                    "used is reported in `declaration_status`.")

UNITS = {
    "plan_area_m2": "m2", "plan_perimeter_m": "m", "sidewall_height_m": "m", "face_area_m2": "m2",
    "face_mass_kg": "kg", "sidewall_mass_kg": "kg", "frame_mass_kg": "kg", "support_mass_kg": "kg",
    "assembly_metal_mass_kg": "kg", "grid_volume_m3": "m3", "grid_projected_footprint_m2": "m2",
    "displaced_volume_m3": "m3", "blocked_flow_area_m2": "m2", "heated_area_credited_m2": "m2",
    "metal_cp_J_kg_K": "J/kg/K",
}

SCOPE = ("Screening construction mass and plan geometry only: no structural, buckling, weld, flange or "
         "pressure-boundary calculation is performed, the load path is not verified and no safety factor or "
         "vessel certification is claimed.")

LIMITATIONS = (
    "The support grid is external: it adds mass and leaves the bed void, flow area and heat path unchanged.",
    "An inside-envelope grid, heated fins and internal plates are refused rather than approximated.",
    "No thermal-area credit is taken from the grid; the two-face contact area and heat path remain the "
    "declared values, and how two faces admit the gas flow is still an unresolved boundary assumption.",
    "One common declared metal heat capacity is applied to every part; the metal is lumped into the single "
    "wall node used by the model, so a part that would really sit at a different temperature "
    "still shares the wall-node temperature here.",
    "Sidewall dimensions come from the declared (or square-plan) perimeter and height, so a non-square plan "
    "must declare its perimeter or the sidewall mass is a square-plan estimate.",
)

_BUILDING = False


@dataclass
class ConstructionSpec:
    """Resolved construction inventory. Masses are in kg, areas in m2 and volumes in m3."""

    sections: dict = field(default_factory=dict)
    plan_area_m2: float | None = None
    plan_perimeter_m: float | None = None
    perimeter_basis: str = ""
    sidewall_height_m: float = 0.0
    face_area_m2: float | None = None
    face_mass_kg: float = 0.0
    sidewall_mass_kg: float = 0.0
    frame_mass_kg: float = 0.0
    support_mass_kg: float = 0.0
    grid_volume_m3: float = 0.0
    grid_projected_footprint_m2: float = 0.0
    displaced_volume_m3: float = 0.0
    blocked_flow_area_m2: float = 0.0
    supports_inside_envelope: bool = False
    rib_intersections_counted_once: int = 0
    heated_area_credited_m2: float = 0.0
    metal_cp_J_kg_K: float | None = None
    base_metal_mass_kg: float | None = None
    status: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def assembly_metal_mass_kg(self) -> float:
        """Metal mass of the declared assembly; this replaces the bare-geometry geometry metal term."""
        return self.face_mass_kg + self.sidewall_mass_kg + self.frame_mass_kg + self.support_mass_kg

    @property
    def total_mass_kg(self) -> float:
        return self.assembly_metal_mass_kg

    @property
    def delta_mass_kg(self) -> float | None:
        """Assembly mass minus the bare-geometry metal term it replaces; None when no bare-geometry base was resolved."""
        if self.base_metal_mass_kg is None:
            return None
        return self.assembly_metal_mass_kg - self.base_metal_mass_kg

    def declaration_status(self) -> dict:
        status = dict(self.status)
        status["_basis"] = CONVENTION_BASIS
        return status

    def as_dict(self) -> dict:
        return {
            "sections": self.sections,
            "plan_area_m2": self.plan_area_m2,
            "plan_perimeter_m": self.plan_perimeter_m,
            "perimeter_basis": self.perimeter_basis,
            "sidewall_height_m": self.sidewall_height_m,
            "face_area_m2": self.face_area_m2,
            "mass_kg": {"face_sheets": self.face_mass_kg, "sidewalls": self.sidewall_mass_kg,
                        "frame": self.frame_mass_kg, "support_grid": self.support_mass_kg,
                        "total": self.assembly_metal_mass_kg},
            "grid_volume_m3": self.grid_volume_m3,
            "grid_projected_footprint_m2": self.grid_projected_footprint_m2,
            "displaced_volume_m3": self.displaced_volume_m3,
            "blocked_flow_area_m2": self.blocked_flow_area_m2,
            "supports_inside_envelope": self.supports_inside_envelope,
            "rib_intersections_counted_once": self.rib_intersections_counted_once,
            "heated_area_credited_m2": self.heated_area_credited_m2,
            "metal_cp_J_kg_K": self.metal_cp_J_kg_K,
            "replacement": {
                "legacy_geometry_metal_mass_kg": self.base_metal_mass_kg,
                "construction_metal_mass_kg": self.assembly_metal_mass_kg,
                "delta_kg": self.delta_mass_kg,
                "applies_to": "Geometry.metal_mass_kg",
                "basis": ("the construction assembly replaces the bare face/frame term; it is never added to "
                          "it, and with no construction section the base geometry is returned unchanged")},
            "flow_basis": ("supports are external, so bed void, gas volume, flow area, blocked flow area, "
                           "displaced volume, heated area and heat path are all unchanged"),
            "units": UNITS,
            "declaration_status": self.declaration_status(),
            "notes": self.notes,
            "scope": SCOPE,
            "limitations": list(LIMITATIONS),
        }


def _number(value, name: str) -> float:
    """Finite real number. Booleans, strings and NaN/infinity are refused, not coerced."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"construction.{name} must be a finite number, got {value!r}.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"construction.{name} must be finite, got {value!r}.")
    return number


def _positive(value, name: str) -> float:
    number = _number(value, name)
    if number <= 0:
        raise ValueError(f"construction.{name} must be positive, got {value!r}.")
    return number


def _count(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"construction.{name} must be a finite nonnegative integer count, got {value!r}.")
    if not math.isfinite(float(value)) or float(value) != int(value) or int(value) < 0:
        raise ValueError(f"construction.{name} must be a finite nonnegative integer count, got {value!r}.")
    return int(value)


def _section(section: dict, name: str) -> dict:
    """One declared sub-section, with unknown keys refused rather than silently ignored."""
    if not isinstance(section, dict):
        raise ValueError(f"construction.{name} must be an object of declared dimensions, got {section!r}.")
    unknown = sorted(set(section) - set(SECTION_KEYS[name]))
    if unknown:
        raise ValueError(f"construction.{name} has unknown key(s) {unknown}; "
                         f"allowed keys are {sorted(SECTION_KEYS[name])}.")
    return section


def _sheet(section: dict, name: str) -> tuple[float, float]:
    thickness = _positive(section.get("thickness_m"), f"{name}.thickness_m")
    density = _positive(section.get("density_kg_m3"), f"{name}.density_kg_m3")
    return thickness, density


def _refuse_flags(section: dict) -> None:
    """Boolean option gates: a present key must be a real boolean, `false` is accepted and `true` refused.

    A numeric or missing-looking value is not treated as `false`: silently reading `0` or `None` as "the
    option was not requested" would make a mistyped flag look like a deliberate declaration.
    """
    for key, reason in REFUSAL_REASONS.items():
        if key not in section:
            continue
        value = section[key]
        if not isinstance(value, bool):
            raise ValueError(f"construction.support_grid.{key} must be a boolean flag, got {value!r}; "
                             "declare `false` to select the implemented external-grid behaviour.")
        if value:
            raise ValueError(f"construction.support_grid.{key} is refused: {reason}.")


def _base_geometry(config: dict, geometry):
    """The construction-free geometry this inventory is dimensioned against.

    The geometry builder calls this module with its own bare result, so a resolved construction never
    feeds back into geometry construction. A direct caller may omit it, and then the bare builder is
    used: `build_bare_geometry` applies the bare-geometry lumped metal term, which is exactly the value the
    assembly replaces, and it cannot resolve the construction section itself. The re-entrancy guard in
    `build_construction` turns the unsupported forward-reference direction into a loud error rather
    than recursion.
    """
    if geometry is not None:
        missing = [name for name in ("kind", "flow_area_m2", "flow_path_m", "metal_mass_kg")
                   if not hasattr(geometry, name)]
        if missing:
            raise ValueError(f"construction was given a geometry object without {missing}; pass the resolved "
                             "geometry from the geometry builder.")
        return geometry
    from . import geometry as geometry_module
    return geometry_module.build_bare_geometry(config)


def _plan_sides(plan_area_m2: float, perimeter_m: float) -> tuple[float, float]:
    """Short and long side of the rectangle with this area and perimeter."""
    root = math.sqrt(max(perimeter_m ** 2 - 16.0 * plan_area_m2, 0.0))
    return (perimeter_m - root) / 4.0, (perimeter_m + root) / 4.0


def build_construction(config: dict, *, geometry=None) -> ConstructionSpec | None:
    """Resolve and validate the opt-in ``construction`` section, or return None when it is absent.

    ``geometry`` is the construction-free base geometry. The geometry builder passes its own result
    here; when it is omitted the bare geometry is resolved internally.
    """
    global _BUILDING
    section = config.get("construction")
    if section is None:
        return None
    if _BUILDING:
        raise RuntimeError(
            "build_construction was re-entered. The supported direction is "
            "build_geometry(config) -> build_construction(config, geometry=bare_geometry). A construction "
            "section resolved from inside its own construction would recurse, so pass the construction-free "
            "geometry explicitly instead.")
    _BUILDING = True
    try:
        return _build_construction(config, section, geometry)
    finally:
        _BUILDING = False


def _build_construction(config: dict, section, geometry) -> ConstructionSpec:
    if not isinstance(section, dict) or not section:
        raise ValueError("construction must be a nonempty object when it is present.")
    refused = sorted(set(section) & set(REFUSED_SECTIONS))
    if refused:
        raise ValueError(f"construction.{refused[0]} is refused: {REFUSED_SECTIONS[refused[0]]}.")
    allowed = set(REQUIRED_SECTIONS) | set(OPTIONAL_SECTIONS) | set(REFUSED_SECTIONS)
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise ValueError(f"construction has unknown key(s) {unknown}; allowed keys are "
                         f"{sorted(set(REQUIRED_SECTIONS) | set(OPTIONAL_SECTIONS))} and refused keys are "
                         f"{sorted(REFUSED_SECTIONS)}.")
    missing = [name for name in REQUIRED_SECTIONS if name not in section]
    if missing:
        raise ValueError(f"construction is missing required section(s): {missing}.")
    for name, declared in section.items():
        if not isinstance(declared, dict):
            raise ValueError(f"construction.{name} must be an object of declared dimensions, "
                             f"got {declared!r}.")

    base = _base_geometry(config, geometry)
    kind = getattr(base, "kind", None)
    if kind != "cassette":
        raise ValueError(
            f"construction is defined for `geometry.kind = \"cassette\"` only, got {kind!r}. An annulus or "
            "cylinder plan is not a rectangle, so its face and sidewall dimensions need their own declared "
            "plan geometry; this bounded module refuses the section instead of reusing cassette arithmetic "
            "on a curved surface.")

    spec = ConstructionSpec(sections={name: dict(section[name]) for name in section},
                            metal_cp_J_kg_K=_positive(config["geometry"]["metal_cp_J_kg_K"],
                                                      "geometry.metal_cp_J_kg_K"),
                            base_metal_mass_kg=float(base.metal_mass_kg))
    plan_area = float(base.flow_area_m2)              # cassette: plan area = bed volume / bed thickness
    square_perimeter = 4.0 * math.sqrt(plan_area)
    spec.plan_area_m2 = plan_area

    # ---- face sheets -------------------------------------------------------------------------
    face = _section(section["face_sheets"], "face_sheets")
    thickness, density = _sheet(face, "face_sheets")
    multiplier = _positive(face.get("area_multiplier", 1.0), "face_sheets.area_multiplier")
    spec.face_area_m2 = 2.0 * plan_area * multiplier
    spec.face_mass_kg = spec.face_area_m2 * thickness * density
    spec.status["face_sheets.area_multiplier"] = (
        {"status": "declared", "value": multiplier} if "area_multiplier" in face
        else {"status": "defaulted_analyst_convention", "value": 1.0,
              "note": "both faces are exactly the cassette plan area"})
    spec.notes.append(f"face sheets: 2 x plan area {plan_area:.6g} m2 x area_multiplier {multiplier:g} = "
                      f"{spec.face_area_m2:.6g} m2, {thickness * 1e3:g} mm, {density:g} kg/m3 -> "
                      f"{spec.face_mass_kg:.6g} kg")

    # ---- sidewalls ---------------------------------------------------------------------------
    wall = _section(section["sidewalls"], "sidewalls")
    thickness, density = _sheet(wall, "sidewalls")
    if "perimeter_m" in wall:
        perimeter = _positive(wall["perimeter_m"], "sidewalls.perimeter_m")
        if perimeter < square_perimeter * (1.0 - 1e-9):
            raise ValueError(
                f"construction.sidewalls.perimeter_m = {perimeter:.6g} m is impossible for a plan of "
                f"{plan_area:.6g} m2: a rectangle of that area has a perimeter of at least "
                f"{square_perimeter:.6g} m.")
        spec.perimeter_basis = "declared"
        spec.status["sidewalls.perimeter_m"] = {"status": "declared", "value": perimeter}
    else:
        perimeter = square_perimeter
        spec.perimeter_basis = "square_plan_assumption"
        spec.status["sidewalls.perimeter_m"] = {
            "status": "defaulted_analyst_convention", "value": square_perimeter,
            "note": "square plan assumption 4*sqrt(plan_area); a non-square plan must declare its perimeter"}
    spec.plan_perimeter_m = perimeter
    height = (_positive(wall["height_m"], "sidewalls.height_m") if "height_m" in wall
              else float(base.flow_path_m))
    spec.status["sidewalls.height_m"] = (
        {"status": "declared", "value": height} if "height_m" in wall
        else {"status": "defaulted_analyst_convention", "value": float(base.flow_path_m),
              "note": "the sidewall band is as high as the bed is thick (geometry.flow_path_m)"})
    spec.sidewall_height_m = height
    spec.sidewall_mass_kg = perimeter * height * thickness * density
    spec.notes.append(f"sidewalls: perimeter {perimeter:.6g} m ({spec.perimeter_basis}) x height {height:.6g} m "
                      f"x {thickness * 1e3:g} mm, {density:g} kg/m3 -> {spec.sidewall_mass_kg:.6g} kg")

    # ---- frame -------------------------------------------------------------------------------
    frame = _section(section["frame"], "frame")
    spec.frame_mass_kg = _positive(frame.get("mass_kg"), "frame.mass_kg")
    spec.notes.append(f"frame: declared part mass {spec.frame_mass_kg:g} kg replaces the geometry "
                      f"frame_mass_kg of {float(config['geometry'].get('frame_mass_kg', 0.0)):g} kg")

    # ---- external support grid ---------------------------------------------------------------
    grid = section.get("support_grid")
    if grid is not None:
        grid = _section(grid, "support_grid")
        if not grid:
            raise ValueError("construction.support_grid is present but empty: declare the grid dimensions or "
                             "remove the section.")
        _refuse_flags(grid)
        thickness, density = _sheet(grid, "support_grid")
        span_x = _positive(grid.get("span_x_m"), "support_grid.span_x_m")
        span_y = _positive(grid.get("span_y_m"), "support_grid.span_y_m")
        ribs_x = _count(grid.get("ribs_x", 0), "support_grid.ribs_x")
        ribs_y = _count(grid.get("ribs_y", 0), "support_grid.ribs_y")
        depth = _positive(grid.get("rib_depth_m", thickness), "support_grid.rib_depth_m")
        if ribs_x + ribs_y < 1:
            raise ValueError("construction.support_grid must declare at least one rib: ribs_x and ribs_y are "
                             "both zero, so the section carries no useful dimension.")
        short_side, long_side = _plan_sides(plan_area, perimeter)
        span_min, span_max = sorted((span_x, span_y))
        if span_min > short_side * (1.0 + 1e-9) or span_max > long_side * (1.0 + 1e-9):
            raise ValueError(
                f"construction.support_grid spans {span_x:.6g} x {span_y:.6g} m do not fit the cassette plan: "
                f"a plan with area {plan_area:.6g} m2 and perimeter {perimeter:.6g} m measures "
                f"{short_side:.6g} x {long_side:.6g} m, and the grid must lie inside it.")
        if ribs_x and ribs_x * thickness > span_y * (1.0 + 1e-9):
            raise ValueError(
                f"construction.support_grid.ribs_x = {ribs_x} ribs of {thickness:g} m width do not fit across "
                f"span_y = {span_y:g} m without overlapping; declare fewer or thinner ribs.")
        if ribs_y and ribs_y * thickness > span_x * (1.0 + 1e-9):
            raise ValueError(
                f"construction.support_grid.ribs_y = {ribs_y} ribs of {thickness:g} m width do not fit across "
                f"span_x = {span_x:g} m without overlapping; declare fewer or thinner ribs.")
        crossings = ribs_x * ribs_y
        footprint = ribs_x * span_x * thickness + ribs_y * span_y * thickness - crossings * thickness ** 2
        spec.rib_intersections_counted_once = crossings
        spec.grid_projected_footprint_m2 = footprint
        spec.grid_volume_m3 = footprint * depth
        spec.support_mass_kg = spec.grid_volume_m3 * density
        spec.status["support_grid.ribs_x"] = (
            {"status": "declared", "value": ribs_x} if "ribs_x" in grid
            else {"status": "defaulted_analyst_convention", "value": 0, "note": "no ribs in this direction"})
        spec.status["support_grid.ribs_y"] = (
            {"status": "declared", "value": ribs_y} if "ribs_y" in grid
            else {"status": "defaulted_analyst_convention", "value": 0, "note": "no ribs in this direction"})
        spec.status["support_grid.rib_depth_m"] = (
            {"status": "declared", "value": depth} if "rib_depth_m" in grid
            else {"status": "defaulted_analyst_convention", "value": thickness,
                  "note": "a rib is as deep through the bed as it is wide"})
        spec.notes.append(
            f"support grid: {ribs_x} x-ribs of {span_x:.6g} m and {ribs_y} y-ribs of {span_y:.6g} m, in-plane "
            f"width {thickness * 1e3:g} mm, depth {depth * 1e3:g} mm, {density:g} kg/m3")
        spec.notes.append(
            f"support grid volume: {ribs_x}*{span_x:g}*{thickness:g} + {ribs_y}*{span_y:g}*{thickness:g} - "
            f"{crossings} crossings of {thickness:g}^2 = {footprint:.6g} m2 of rib footprint, times "
            f"{depth:g} m depth = {spec.grid_volume_m3:.6g} m3 -> {spec.support_mass_kg:.6g} kg; the "
            f"{crossings} rib crossing region(s) are counted once")
        spec.status["support_grid.inside_envelope"] = {
            "status": "refused_option_not_taken", "value": False,
            "note": REFUSAL_REASONS["inside_envelope"]}
        spec.status["support_grid.fins_are_heated"] = {
            "status": "refused_option_not_taken", "value": False,
            "note": REFUSAL_REASONS["fins_are_heated"]}
        spec.notes.append("support grid: outside the gas/adsorbent envelope, so it adds mass only. Displaced "
                          "bed volume and blocked flow area stay exactly zero, and the projected footprint "
                          f"{footprint:.6g} m2 is reported as the flow area it would block if it sat inside.")

    total = spec.assembly_metal_mass_kg
    if not math.isfinite(total) or total <= 0:
        raise ValueError(f"construction must carry a finite positive metal mass, resolved {total!r} kg.")
    if spec.heated_area_credited_m2 or spec.displaced_volume_m3 or spec.blocked_flow_area_m2:
        raise ValueError("construction must not credit heated area or displaced/blocked bed volume.")
    spec.notes.append(f"assembly metal mass {total:.6g} kg replaces the base geometry metal term "
                      f"{spec.base_metal_mass_kg:.6g} kg (delta {spec.delta_mass_kg:+.6g} kg); the two are "
                      "never added together")
    spec.notes.append(f"one common declared metal heat capacity {spec.metal_cp_J_kg_K:g} J/kg/K is applied to "
                      "every part: the assembly adds wall inertia, not heat-transfer area")
    return spec


def construction_report(config: dict, *, geometry=None) -> dict | None:
    """The exportable construction report, or None when no construction section is configured."""
    spec = build_construction(config, geometry=geometry)
    return spec.as_dict() if spec else None
