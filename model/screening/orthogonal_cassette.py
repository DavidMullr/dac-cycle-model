"""Physically represented rectangular cassette with orthogonal heat and gas paths.

The gas crosses the packed bed along the declared flow height ``H`` while heat crosses the thin
direction ``t``.  This is deliberately different from the reference cassette, whose heat path and flow
path are both the bed thickness.  The geometry is opt-in under ``geometry.kind =
"orthogonal_cassette"`` and keeps the coupled-cycle interface unchanged:

``flow_area_m2 = W*t``, ``flow_path_m = H``, ``heated_area_m2 = 2*W*H`` and
``heat_path_m = t/2``.

Equal dry sorbent mass and bulk density fix the bed envelope volume ``V = W*H*t``.  ``t`` and ``H``
are declared; ``W`` is derived from that equation unless it is also declared, in which case the
declared value must close the same equation to numerical precision.  An over-determined mismatch is
refused instead of being overwritten.

The metal inventory is a union of non-overlapping parts outside the bed envelope:

* two solid face sheets on the heated faces, each ``W*H``;
* two side rails along the flow direction, each ``H*t`` in cross-section;
* two end-wall frames around the ``W*t`` gas openings, each ring area
  ``(W + 2*r)*(t + 2*r) - W*t`` for rim width ``r``.

The frames lie upstream and downstream of the bed rather than inside it.  Their openings are exactly
the gas area, and no metal or dead volume displaces sorbent or closes the gas passage.  The assembly
mass therefore has no free heated area, blocked flow area, or doubled component volume.  Every part
is reported once with its extents, contact joints and mass; no separate lumped frame term is added.

This module certifies geometry and mass bookkeeping only.  It does not certify a pressure boundary,
manifold/distributor, support structure, external frame, or thermal contact hardware.  The Ergun
pressure model, where the coupled cycle calls it, receives the derived gas area and flow path; the
opening and manifold pressure fields are outside this model's scope.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from .geometry import Geometry

KIND = "orthogonal_cassette"
SECTION = "orthogonal_cassette"
ALLOWED_GEOMETRY_KEYS = frozenset({"kind", SECTION, "extra_dead_volume_m3", "metal_cp_J_kg_K"})
REQUIRED_PART_KEYS = {
    "face_sheets": frozenset({"thickness_m", "density_kg_m3"}),
    "side_rails": frozenset({"thickness_m", "density_kg_m3"}),
    "end_frames": frozenset({"rim_width_m", "thickness_m", "density_kg_m3"}),
}
CERTIFICATION_SCOPE = (
    "Geometry and assembly-mass bookkeeping only. The pressure boundary, gas manifolds and "
    "distributors, external support structure, joining details, thermal contact hardware, and "
    "structural integrity are not modelled or certified. The Ergun calculation is a packed-bed "
    "screening estimate with uniform inlet/outlet pressure and no manifold losses."
)


@dataclass(frozen=True)
class OrthogonalCassetteGeometry(Geometry):
    """Standard ``Geometry`` plus the rectangular dimensions needed to audit the mapping."""

    width_m: float = 0.0
    thickness_m: float = 0.0
    gas_opening_area_m2: float = 0.0
    face_area_m2: float = 0.0
    flow_direction: str = "height"
    heat_direction: str = "thickness"

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class OrthogonalCassetteSpec:
    """Resolved, auditable geometry and metal inventory."""

    width_m: float
    height_m: float
    thickness_m: float
    width_basis: str
    bed_volume_m3: float
    bulk_density_kg_m3: float
    bed_void_fraction: float
    bed_void_volume_m3: float
    extra_dead_volume_m3: float
    gas_volume_m3: float
    gas_area_m2: float
    flow_path_m: float
    gas_opening_area_each_m2: float
    heated_face_area_each_m2: float
    heated_area_m2: float
    heat_path_m: float
    face_sheet_volume_each_m3: float
    side_rail_volume_each_m3: float
    end_frame_inner_area_m2: float
    end_frame_outer_area_m2: float
    end_frame_area_each_m2: float
    end_frame_volume_each_m3: float
    face_sheet_mass_kg: float
    side_rail_mass_kg: float
    end_frame_mass_kg: float
    metal_volume_m3: float
    metal_mass_kg: float
    bounding_length_m: float
    bounding_width_m: float
    bounding_thickness_m: float
    bounding_footprint_m2: float
    bom: tuple[dict, ...]
    geometry: OrthogonalCassetteGeometry
    limitations: tuple[str, ...]
    qualification: dict

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["limitations"] = list(self.limitations)
        return payload


def _number(value, path: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be a finite "
                         f"{'nonnegative' if allow_zero else 'positive'} number, got {value!r}.")
    result = float(value)
    invalid = (not math.isfinite(result) or result < 0.0) if allow_zero else (
        not math.isfinite(result) or result <= 0.0)
    if invalid:
        raise ValueError(f"{path} must be finite and "
                         f"{'nonnegative' if allow_zero else 'positive'}, got {value!r}.")
    return result


def _part(section: dict, name: str) -> dict:
    if name not in section:
        raise ValueError(f"geometry.{SECTION} is missing required part '{name}'.")
    value = section[name]
    if not isinstance(value, dict) or not value:
        raise ValueError(f"geometry.{SECTION}.{name} must be a nonempty object.")
    required = REQUIRED_PART_KEYS[name]
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required)
    if missing:
        raise ValueError(f"geometry.{SECTION}.{name} is missing required key(s) {missing}.")
    if unknown:
        raise ValueError(f"geometry.{SECTION}.{name} has unknown key(s) {unknown}; "
                         f"allowed keys are {sorted(required)}.")
    return value


def build_orthogonal_cassette(config: dict) -> OrthogonalCassetteSpec:
    """Resolve the opt-in orthogonal cassette and audit all derived geometry and masses."""
    geometry = config.get("geometry")
    if not isinstance(geometry, dict):
        raise ValueError("geometry must be an object for an orthogonal cassette.")
    if geometry.get("kind") != KIND:
        raise ValueError(f"build_orthogonal_cassette needs geometry.kind = {KIND!r}, "
                         f"got {geometry.get('kind')!r}.")
    unknown = sorted(set(geometry) - ALLOWED_GEOMETRY_KEYS)
    if unknown:
        raise ValueError(
            f"geometry has key(s) {unknown}, which do not belong to an orthogonal cassette. "
            f"Cassette/annulus/cylinder dimensions are refused rather than ignored; "
            f"allowed keys are {sorted(ALLOWED_GEOMETRY_KEYS)}.")
    section = geometry.get(SECTION)
    if not isinstance(section, dict) or not section:
        raise ValueError(f"geometry.{SECTION} is required and must be a nonempty object.")
    required = {"thickness_m", "height_m", "face_sheets", "side_rails", "end_frames"}
    missing = sorted(required - set(section))
    unknown = sorted(set(section) - required - {"width_m"})
    if missing:
        raise ValueError(f"geometry.{SECTION} is missing required key(s) {missing}.")
    if unknown:
        raise ValueError(f"geometry.{SECTION} has unknown key(s) {unknown}.")

    bed = config.get("bed")
    if not isinstance(bed, dict):
        raise ValueError("bed must be an object.")
    mass = _number(bed.get("dry_sorbent_mass_kg"), "bed.dry_sorbent_mass_kg")
    pellet_density = _number(bed.get("pellet_density_kg_m3"), "bed.pellet_density_kg_m3")
    void = bed.get("bed_void_fraction")
    if isinstance(void, bool) or not isinstance(void, (int, float)) or not math.isfinite(float(void)) \
            or not 0.0 < float(void) < 1.0:
        raise ValueError("bed.bed_void_fraction must be finite and strictly between 0 and 1.")
    void = float(void)
    bulk_density = (1.0 - void) * pellet_density
    bed_volume = mass / bulk_density

    thickness = _number(section.get("thickness_m"), f"geometry.{SECTION}.thickness_m")
    height = _number(section.get("height_m"), f"geometry.{SECTION}.height_m")
    envelope_volume_per_metre = height * thickness
    derived_width = bed_volume / envelope_volume_per_metre
    if not math.isfinite(derived_width) or derived_width <= 0.0:
        raise ValueError("Declared thickness_m and height_m do not produce a finite positive width.")
    if "width_m" in section:
        width = _number(section["width_m"], f"geometry.{SECTION}.width_m")
        relative_error = abs(width * height * thickness - bed_volume) / bed_volume
        if relative_error > 1e-9:
            raise ValueError(
                f"geometry.{SECTION} is over-determined: W*H*t = {width * height * thickness:.12g} m3 "
                f"but equal dry mass and bulk density require {bed_volume:.12g} m3 "
                f"(relative error {relative_error:.3g}). Omit width_m to derive it, or correct the "
                "declared dimensions.")
        width_basis = "declared_and_equal_mass_checked"
    else:
        width = derived_width
        width_basis = "derived_from_equal_dry_mass"

    extra_dead = geometry.get("extra_dead_volume_m3", 0.0)
    if isinstance(extra_dead, bool) or not isinstance(extra_dead, (int, float)) \
            or not math.isfinite(float(extra_dead)) or float(extra_dead) < 0.0:
        raise ValueError("geometry.extra_dead_volume_m3 must be finite and nonnegative.")
    extra_dead = float(extra_dead)
    bed_void_volume = bed_volume * void
    gas_volume = bed_void_volume + extra_dead

    face = _part(section, "face_sheets")
    side = _part(section, "side_rails")
    frame = _part(section, "end_frames")
    face_thickness = _number(face["thickness_m"], f"geometry.{SECTION}.face_sheets.thickness_m")
    face_density = _number(face["density_kg_m3"], f"geometry.{SECTION}.face_sheets.density_kg_m3")
    side_thickness = _number(side["thickness_m"], f"geometry.{SECTION}.side_rails.thickness_m")
    side_density = _number(side["density_kg_m3"], f"geometry.{SECTION}.side_rails.density_kg_m3")
    rim_width = _number(frame["rim_width_m"], f"geometry.{SECTION}.end_frames.rim_width_m")
    frame_thickness = _number(frame["thickness_m"], f"geometry.{SECTION}.end_frames.thickness_m")
    frame_density = _number(frame["density_kg_m3"], f"geometry.{SECTION}.end_frames.density_kg_m3")

    gas_area = width * thickness
    heated_face_area = width * height
    face_volume_each = heated_face_area * face_thickness
    side_volume_each = height * thickness * side_thickness
    end_inner_area = width * thickness
    if rim_width < max(side_thickness, face_thickness) - 1e-15:
        raise ValueError(
            "geometry.orthogonal_cassette.end_frames.rim_width_m must cover both the side-rail "
            "thickness and face-sheet thickness at the end joint "
            f"(minimum {max(side_thickness, face_thickness):.12g} m).")
    end_outer_area = (width + 2.0 * rim_width) * (thickness + 2.0 * rim_width)
    end_ring_area = end_outer_area - end_inner_area
    if end_ring_area <= 0.0:
        raise ValueError("End-frame dimensions do not leave a positive ring area around the gas opening.")
    # The axial end-frame thickness is independent of the in-plane rim width.  Using the declared
    # axial thickness keeps this a plate-like frame instead of a solid block around the opening.
    end_volume_each = end_ring_area * frame_thickness

    face_mass = 2.0 * face_volume_each * face_density
    side_mass = 2.0 * side_volume_each * side_density
    end_mass = 2.0 * end_volume_each * frame_density
    metal_volume = 2.0 * face_volume_each + 2.0 * side_volume_each + 2.0 * end_volume_each
    metal_mass = face_mass + side_mass + end_mass
    if not all(math.isfinite(value) and value > 0.0
               for value in (width, height, thickness, gas_area, heated_face_area, metal_volume, metal_mass)):
        raise ValueError("Orthogonal cassette derived geometry is not finite and positive.")

    bounding_length = height + 2.0 * frame_thickness
    bounding_width = width + 2.0 * rim_width
    bounding_thickness = thickness + 2.0 * rim_width
    bom = (
        {
            "part": "heated_face_sheets",
            "count": 2,
            "material_density_kg_m3": face_density,
            "unit_extents_m": {
                "x_flow": [0.0, height],
                "y_width": [0.0, width],
                "z_thickness": [[-thickness / 2.0 - face_thickness, -thickness / 2.0],
                                [thickness / 2.0, thickness / 2.0 + face_thickness]],
            },
            "unit_volume_m3": face_volume_each,
            "unit_mass_kg": face_volume_each * face_density,
            "total_mass_kg": face_mass,
            "heated_area_credited_m2": 2.0 * heated_face_area,
        },
        {
            "part": "external_side_rails",
            "count": 2,
            "material_density_kg_m3": side_density,
            "unit_extents_m": {
                "x_flow": [0.0, height],
                "y_width": [[-side_thickness, 0.0], [width, width + side_thickness]],
                "z_thickness": [-thickness / 2.0, thickness / 2.0],
            },
            "unit_volume_m3": side_volume_each,
            "unit_mass_kg": side_volume_each * side_density,
            "total_mass_kg": side_mass,
            "heated_area_credited_m2": 0.0,
        },
        {
            "part": "open_end_rings",
            "count": 2,
            "material_density_kg_m3": frame_density,
            "unit_extents_m": {
                "x_flow": [[-frame_thickness, 0.0], [height, height + frame_thickness]],
                "y_width": [-rim_width, width + rim_width],
                "z_thickness": [-thickness / 2.0 - rim_width, thickness / 2.0 + rim_width],
                "gas_opening_y_width": [0.0, width],
                "gas_opening_z_thickness": [-thickness / 2.0, thickness / 2.0],
            },
            "unit_volume_m3": end_volume_each,
            "unit_mass_kg": end_volume_each * frame_density,
            "total_mass_kg": end_mass,
            "heated_area_credited_m2": 0.0,
        },
    )
    resolved = OrthogonalCassetteGeometry(
        kind=KIND,
        bed_volume_m3=bed_volume,
        bulk_density_kg_m3=bulk_density,
        flow_area_m2=gas_area,
        flow_path_m=height,
        heated_area_m2=2.0 * heated_face_area,
        heat_path_m=thickness / 2.0,
        metal_mass_kg=metal_mass,
        gas_volume_m3=gas_volume,
        height_m=height,
        width_m=width,
        thickness_m=thickness,
        gas_opening_area_m2=gas_area,
        face_area_m2=heated_face_area,
    )
    qualification = {
        "scope": CERTIFICATION_SCOPE,
        "gas_flow_axis": "height H",
        "heat_flow_axis": "thin thickness t",
        "gas_openings": {
            "count": 2,
            "area_each_m2": gas_area,
            "solid_face_area_uncredited_m2": heated_face_area,
            "note": "The two W*t end openings carry gas; the two W*H faces are solid and heated.",
        },
        "pressure_boundary": (
            "Not certified: uniform inlet and outlet pressure on the two W*t openings is assumed. "
            "Manifold, distributor, entrance/exit, frame-rim and lateral-leak losses are outside the "
            "Ergun bed calculation."
        ),
        "supports_and_manifolds": (
            "Not certified: the side rails and end frames are counted as metal mass, not checked as a "
            "pressure-retaining or load-bearing structure. Supports, welds, fasteners and removable "
            "manifolds are not modelled."
        ),
        "thermal_boundary": (
            "The two W*H sheets are the only heated faces. The configured wall-contact coefficient and "
            "effective conductivity are applied across t/2; contact hardware and temperature "
            "nonuniformity are not certified."
        ),
        "displacement": {
            "bed_envelope_volume_m3": bed_volume,
            "bed_metal_overlap_m3": 0.0,
            "blocked_gas_opening_area_m2": 0.0,
            "component_volumes_pairwise_disjoint": True,
        },
        "geometry_extents": {
            "bed": {"x_flow_m": [0.0, height], "y_width_m": [0.0, width],
                    "z_thickness_m": [-thickness / 2.0, thickness / 2.0]},
            "gas_inlet_m": {"plane_x_m": 0.0, "opening_y_width_m": [0.0, width],
                            "opening_z_thickness_m": [-thickness / 2.0, thickness / 2.0],
                            "open_area_m2": gas_area},
            "gas_outlet_m": {"plane_x_m": height, "opening_y_width_m": [0.0, width],
                             "opening_z_thickness_m": [-thickness / 2.0, thickness / 2.0],
                             "open_area_m2": gas_area},
            "bounding": {"length_along_flow_m": bounding_length, "width_m": bounding_width,
                         "thickness_m": bounding_thickness,
                         "plan_footprint_flow_by_width_m2": bounding_length * bounding_width},
            "joints": (
                "Face sheets contact the bed only on z = +/-t/2; side rails contact it only on y = 0 "
                "and y = W. End rings are wholly outside x in [0,H], joined face-to-face at x = 0 and "
                "x = H. The ring opening is the complete W*t gas opening; no solid plate spans it."
            ),
        },
        "bom": {
            "groups": list(bom),
            "total_metal_volume_m3": metal_volume,
            "total_metal_mass_kg": metal_mass,
            "legacy_lumped_frame_mass_applied": False,
            "count_rule": "Each of the three part groups has count=2; group unit mass is counted exactly twice.",
        },
        "gas_inventory": {
            "bed_void_fraction": void,
            "bed_void_volume_m3": bed_void_volume,
            "extra_dead_volume_m3": extra_dead,
            "total_gas_volume_m3": gas_volume,
            "note": "Extra dead volume is external to W*H*t and is not credited as bed volume or heated area.",
        },
    }
    limitations = (
        "Geometry and metal-mass concept only; no pressure boundary, support, manifold or structural certification.",
        "Uniform packed-bed flow and assumed uniform opening pressure; no inlet/outlet or manifold pressure field.",
        "Separate gas and heat directions are represented only by distinct flow and heat path lengths in the lumped cycle.",
        "The existing 1D spatial solver remains reserved for coincident-path cassette geometry.",
    )
    return OrthogonalCassetteSpec(
        width_m=width,
        height_m=height,
        thickness_m=thickness,
        width_basis=width_basis,
        bed_volume_m3=bed_volume,
        bulk_density_kg_m3=bulk_density,
        bed_void_fraction=void,
        bed_void_volume_m3=bed_void_volume,
        extra_dead_volume_m3=extra_dead,
        gas_volume_m3=gas_volume,
        gas_area_m2=gas_area,
        flow_path_m=height,
        gas_opening_area_each_m2=gas_area,
        heated_face_area_each_m2=heated_face_area,
        heated_area_m2=2.0 * heated_face_area,
        heat_path_m=thickness / 2.0,
        face_sheet_volume_each_m3=face_volume_each,
        side_rail_volume_each_m3=side_volume_each,
        end_frame_inner_area_m2=end_inner_area,
        end_frame_outer_area_m2=end_outer_area,
        end_frame_area_each_m2=end_ring_area,
        end_frame_volume_each_m3=end_volume_each,
        face_sheet_mass_kg=face_mass,
        side_rail_mass_kg=side_mass,
        end_frame_mass_kg=end_mass,
        metal_volume_m3=metal_volume,
        metal_mass_kg=metal_mass,
        bounding_length_m=bounding_length,
        bounding_width_m=bounding_width,
        bounding_thickness_m=bounding_thickness,
        bounding_footprint_m2=bounding_length * bounding_width,
        bom=bom,
        geometry=resolved,
        limitations=limitations,
        qualification=qualification,
    )


def build_orthogonal_geometry(config: dict) -> OrthogonalCassetteGeometry:
    """Return only the standard coupled-cycle geometry for this opt-in cassette."""
    return build_orthogonal_cassette(config).geometry
