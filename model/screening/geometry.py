"""Bed geometry, gas flow paths and thermal metal inventory.

build_bare_geometry derives dimensions from the bed and geometry inputs.
Optional construction replaces its lumped metal mass with a dimension-derived
assembly mass. It leaves bed volume, flow area, gas volume and heat paths intact.

Orthogonal cassettes carry gas along their height and heat across their thickness.
Their dimensions cannot use the coincident-flow cassette's construction model."""
import math
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from math import pi


@dataclass(frozen=True)
class Geometry:
    kind: str
    bed_volume_m3: float
    bulk_density_kg_m3: float
    flow_area_m2: float
    flow_path_m: float
    heated_area_m2: float
    heat_path_m: float
    metal_mass_kg: float
    gas_volume_m3: float
    inner_radius_m: float = 0.0
    outer_radius_m: float = 0.0
    height_m: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


def build_bare_geometry(c: dict) -> Geometry:
    """The construction-free geometry: bed/plan dimensions and the base lumped metal mass."""
    b, g = c["bed"], c["geometry"]
    density = (1 - b["bed_void_fraction"]) * b["pellet_density_kg_m3"]
    volume = b["dry_sorbent_mass_kg"] / density
    kind = g["kind"]
    if kind == "parallel_cassette_bank":
        return build_parallel_cassette_bank(c)
    if kind == "orthogonal_cassette":
        from .orthogonal_cassette import build_orthogonal_geometry
        return build_orthogonal_geometry(c)
    gap = g["bed_thickness_m"]
    ri = ro = h = 0.0
    if kind == "cassette":
        area = volume / gap
        heated, heat_path, flow_path = 2 * area, gap / 2, gap
    elif kind == "annulus":
        ri, ro = g["annulus_mean_radius_m"] - gap / 2, g["annulus_mean_radius_m"] + gap / 2
        h = volume / (pi * (ro**2 - ri**2))
        area = 2 * pi * ri * h  # highest radial velocity at inner face
        heated, heat_path, flow_path = 2 * pi * (ri + ro) * h, gap / 2, gap
    elif kind == "cylinder":
        ro = g["cylinder_radius_m"]
        area = pi * ro**2
        h = volume / area
        heated, heat_path, flow_path = 2 * pi * ro * h, ro / 2, h
    else:
        raise ValueError(f"Unknown geometry: {kind}")
    metal_mass = heated * g["metal_thickness_m"] * g["metal_density_kg_m3"] + g["frame_mass_kg"]
    # Interparticle space and explicit dead volume. Intraparticle gas is omitted.
    gas_volume = volume * b["bed_void_fraction"] + g["extra_dead_volume_m3"]
    return Geometry(kind, volume, density, area, flow_path, heated, heat_path,
                    metal_mass, gas_volume, ri, ro, h)


def build_parallel_cassette_bank(c: dict) -> Geometry:
    """Synchronized equivalent for identical parallel orthogonal cassettes.

    The configuration declares the total bed dry mass and a module mass/count.
    The returned lumped geometry has summed bed/gas volumes, gas and heated
    areas, and repeated non-overlapping metal inventory. Module flow and heat
    paths are retained. This is geometry bookkeeping, not a manifold or
    pressure-boundary model.
    """
    from .orthogonal_cassette import build_orthogonal_geometry

    section = c["geometry"].get("parallel_cassette_bank")
    if not isinstance(section, dict):
        raise ValueError("geometry.parallel_cassette_bank must be an object.")
    required = ("module_count", "module_mass_kg", "gap_m")
    missing = [key for key in required if key not in section]
    if missing:
        raise ValueError(f"geometry.parallel_cassette_bank is missing {missing}.")
    count = section["module_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("parallel_cassette_bank.module_count must be a positive integer.")
    module_mass = section["module_mass_kg"]
    if isinstance(module_mass, bool) or not isinstance(module_mass, (int, float)) \
            or not math.isfinite(float(module_mass)) or float(module_mass) <= 0.0:
        raise ValueError("parallel_cassette_bank.module_mass_kg must be positive and finite.")
    if not math.isclose(
        float(c["bed"]["dry_sorbent_mass_kg"]),
        count * float(module_mass),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "parallel_cassette_bank module mass x count must equal bed.dry_sorbent_mass_kg; "
            "the total charge is not multiplied after simulation."
        )
    if "module_geometry" in section:
        module_geometry = deepcopy(section["module_geometry"])
        rows = section.get("rack_rows", section.get("rows"))
        columns = section.get("rack_columns", section.get("columns"))
    else:
        for key in ("height_m", "thickness_m", "face_sheets", "side_rails", "end_frames"):
            if key not in section:
                raise ValueError(f"geometry.parallel_cassette_bank is missing {key!r}.")
        module_geometry = {
            "kind": "orthogonal_cassette",
            "metal_cp_J_kg_K": c["geometry"].get("metal_cp_J_kg_K", 900.0),
            "extra_dead_volume_m3": section.get(
                "extra_dead_volume_m3", c["geometry"].get("extra_dead_volume_m3", 0.0005)
            ),
            "orthogonal_cassette": {
                "height_m": section["height_m"],
                "thickness_m": section["thickness_m"],
                "face_sheets": deepcopy(section["face_sheets"]),
                "side_rails": deepcopy(section["side_rails"]),
                "end_frames": deepcopy(section["end_frames"]),
            },
        }
        rows = section.get("rows", section.get("rack_rows"))
        columns = section.get("columns", section.get("rack_columns"))
    if isinstance(rows, bool) or not isinstance(rows, int) or rows < 1:
        raise ValueError("parallel_cassette_bank.rows must be a positive integer (bool/float refused).")
    if isinstance(columns, bool) or not isinstance(columns, int) or columns < 1:
        raise ValueError("parallel_cassette_bank.columns must be a positive integer (bool/float refused).")
    if rows * columns != count:
        raise ValueError("parallel cassette rack rows x columns must equal module_count.")
    gap = section.get("gap_m")
    if isinstance(gap, bool) or not isinstance(gap, (int, float)) \
            or not math.isfinite(float(gap)) or float(gap) < 0.0:
        raise ValueError("parallel_cassette_bank.gap_m must be finite and nonnegative.")
    total_cp = c["geometry"].get("metal_cp_J_kg_K")
    if isinstance(total_cp, bool) or not isinstance(total_cp, (int, float)) \
            or not math.isfinite(float(total_cp)) or float(total_cp) <= 0.0:
        raise ValueError("parallel_cassette_bank metal_cp_J_kg_K must be finite and positive.")
    module_cp = module_geometry.get("metal_cp_J_kg_K")
    if module_cp is not None:
        if isinstance(module_cp, bool) or not isinstance(module_cp, (int, float)) \
                or not math.isfinite(float(module_cp)) \
                or not math.isclose(float(module_cp), float(total_cp), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                "parallel_cassette_bank module_geometry.metal_cp_J_kg_K must agree with "
                "geometry.metal_cp_J_kg_K; thermal RHS and module inventory cannot diverge."
            )
    else:
        module_geometry["metal_cp_J_kg_K"] = float(total_cp)
    if module_geometry.get("kind") != "orthogonal_cassette":
        raise ValueError("parallel_cassette_bank modules must use orthogonal_cassette geometry.")
    module_config = deepcopy(c)
    module_config["bed"]["dry_sorbent_mass_kg"] = float(module_mass)
    module_config["geometry"] = module_geometry
    module = build_orthogonal_geometry(module_config)
    penalty = section.get("construction_penalty")
    if penalty is None:
        penalty = c.get("geometry", {}).get("construction_penalty")
    added_gas = 0.0
    coupled_metal = 0.0
    if penalty is not None:
        if not isinstance(penalty, dict):
            raise ValueError("construction_penalty must be an object.")
        allowed = {
            "schema",
            "construction",
            "scenario_id",
            "added_vacuum_volume_m3",
            "added_vacuum_volume_decomposition",
            "added_coupled_metal_kg",
            "thermally_coupled_metal_mass_kg",
            "unheated_metal_kg",
            "unheated_isolated_metal_mass_kg",
            "unheated_metal_assumption",
            "structural_support_coupled_fraction",
            "included_surfaces",
            "rack_gaps_inside_boundary",
            "physical_envelope_status",
            "derived_baseline",
            "owner_group",
            "construction_scenario_id",
            "source_ref",
        }
        unknown = sorted(set(penalty) - allowed)
        if unknown:
            raise ValueError(
                f"construction_penalty has unsupported key(s) {unknown}."
            )
        if penalty.get("owner_group") not in (None, "B"):
            raise ValueError(
                "construction_penalty is a construction-owned B recipe and "
                "must declare owner_group 'B'."
            )
        for key in (
            "added_vacuum_volume_m3",
            "added_coupled_metal_kg",
            "thermally_coupled_metal_mass_kg",
            "unheated_metal_kg",
            "unheated_isolated_metal_mass_kg",
        ):
            value = penalty.get(key, 0.0)
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(
                    f"construction_penalty.{key} must be finite and nonnegative."
                )
        added_gas = float(penalty.get("added_vacuum_volume_m3", 0.0))
        coupled_metal = float(
            penalty.get(
                "added_coupled_metal_kg",
                penalty.get("thermally_coupled_metal_mass_kg", 0.0),
            )
        )
        source = penalty.get("source_ref")
        if source is not None and (
            not isinstance(source, dict)
            or not isinstance(source.get("path"), str)
            or not isinstance(source.get("sha256"), str)
            or len(source["sha256"]) != 64
        ):
            raise ValueError(
                "construction_penalty.source_ref must be {path, sha256}."
            )
    return Geometry(
        kind="parallel_cassette_bank",
        bed_volume_m3=count * module.bed_volume_m3,
        bulk_density_kg_m3=module.bulk_density_kg_m3,
        flow_area_m2=count * module.flow_area_m2,
        flow_path_m=module.flow_path_m,
        heated_area_m2=count * module.heated_area_m2,
        heat_path_m=module.heat_path_m,
        metal_mass_kg=count * module.metal_mass_kg + coupled_metal,
        gas_volume_m3=count * module.gas_volume_m3 + added_gas,
        height_m=module.height_m,
    )


def build_geometry(c: dict, construction=None) -> Geometry:
    """The bare geometry, with the declared construction mass replacing the base metal term.

    `construction` is the resolved `ConstructionSpec` when the caller already has one; otherwise the
    optional `construction` section of `c` is resolved here against the bare geometry, so the
    construction module never has to call this builder back. The assembly mass replaces the base
    face/frame term and is never added to it, and a configuration without a construction section
    returns the bare geometry unchanged.
    """
    bare = build_bare_geometry(c)
    if bare.kind in ("orthogonal_cassette", "parallel_cassette_bank") and (
            construction is not None or c.get("construction") is not None):
        raise ValueError(
            f"The separate `construction` section is refused for geometry.kind = "
            f"`{bare.kind}`. Its plan/face arithmetic does not represent this geometry's "
            "dimension-derived metal inventory.")
    if bare.kind == "parallel_cassette_bank":
        return bare
    if construction is None:
        # Function-level import: screening.construction reads Geometry from this module, so a
        # module-level import in either direction would be a cycle.
        from .construction import build_construction
        construction = build_construction(c, geometry=bare)
    if construction is None:
        return bare
    mass = getattr(construction, "assembly_metal_mass_kg", None)
    if not isinstance(mass, (int, float)) or isinstance(mass, bool) or not math.isfinite(float(mass)) \
            or not mass > 0:
        raise ValueError("build_geometry needs a resolved construction with a finite positive "
                         f"`assembly_metal_mass_kg`, got {mass!r}; pass the ConstructionSpec from "
                         "`build_construction`.")
    return replace(bare, metal_mass_kg=float(mass))
