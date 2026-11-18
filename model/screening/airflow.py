"""Packed-bed pressure loss and fan electrical input.

The Ergun equation uses the composition and volume flow entering the bed.
An explicit upstream dryer changes that stream according to its declared flow
basis. The dryer meters its own pressure loss separately. Configured feed flow
and bed-side flow are retained as distinct quantities."""
from math import log, pi

import numpy as np

from .common import MW, R, feed_composition
from .feed_dryer import UPSTREAM_RAW, build_dryer
from .geometry import Geometry


def airflow(c: dict, g: Geometry, volume_flow_m3_s: float | None = None) -> dict:
    a, b, f = c["ambient"], c["bed"], c["airflow"]
    dryer = build_dryer(c)
    flow = f["volume_flow_m3_s"] if volume_flow_m3_s is None else volume_flow_m3_s
    if flow < 0:
        raise ValueError("Volume flow cannot be negative.")
    T = a["temperature_C"] + 273.15
    # Molar flow of the configured stream, and of the stream the bed actually processes.
    configured_mol_s = flow * a["pressure_Pa"] / (R * T)
    if dryer is None:
        feed_basis, y, bed_mol_s = "legacy_configured", feed_composition(c), configured_mol_s
        # Keep the reference arithmetic bit-identical: the bed-side volume flow is the configured one.
        bed_volume_flow_m3_s = flow
    else:
        # `feed_composition` resolves the declared dryer itself, so the bed-side mixture is the
        # conditioned stream in both bases; only the flow mapping depends on `flow_basis`.
        feed_basis, y = dryer.spec.flow_basis, feed_composition(c)
        bed_mol_s = (dryer.conditioned_molar_flow(configured_mol_s)
                     if dryer.spec.flow_basis == UPSTREAM_RAW else configured_mol_s)
        bed_volume_flow_m3_s = bed_mol_s * R * T / a["pressure_Pa"]
    raw_mol_s = (configured_mol_s if dryer is None or dryer.spec.flow_basis == UPSTREAM_RAW
                 else dryer.raw_molar_flow(configured_mol_s))
    density = a["pressure_Pa"] * float(y @ MW) / (R * T)
    eps, dp, mu = b["bed_void_fraction"], b["particle_diameter_m"], f["air_viscosity_Pa_s"]
    viscous = 150 * mu * (1 - eps)**2 / (eps**3 * dp**2)
    inertial = 1.75 * density * (1 - eps) / (eps**3 * dp)
    velocity = bed_volume_flow_m3_s / g.flow_area_m2
    if g.kind == "annulus":
        viscous_drop = (viscous * bed_volume_flow_m3_s / (2 * pi * g.height_m)
                        * log(g.outer_radius_m / g.inner_radius_m))
        inertial_drop = (inertial * bed_volume_flow_m3_s**2 / (4 * pi**2 * g.height_m**2)
                         * (1 / g.inner_radius_m - 1 / g.outer_radius_m))
    else:
        viscous_drop = viscous * velocity * g.flow_path_m
        inertial_drop = inertial * velocity**2 * g.flow_path_m
    minor_drop = f["minor_loss_coefficient"] * density * velocity**2 / 2
    drop = viscous_drop + inertial_drop + minor_drop
    return {"geometry": g.kind, "volume_flow_m3_s": flow, "face_velocity_m_s": velocity,
            "pressure_drop_Pa": drop, "viscous_drop_Pa": viscous_drop,
            "inertial_drop_Pa": inertial_drop, "minor_drop_Pa": minor_drop,
            # The fan moves the bed-side volume against the bed-side drop. The dryer's own added
            # pressure loss is a separate dryer-fan meter and is deliberately not added here.
            "fan_power_W": bed_volume_flow_m3_s * drop / f["fan_efficiency"],
            "feed_mol_s": configured_mol_s,
            "particle_Reynolds": density * velocity * dp / mu,
            "pressure_drop_fraction_of_ambient": drop / a["pressure_Pa"],
            # Feed-basis provenance for the coupled cycle and the report.
            "feed_basis": feed_basis,
            "configured_basis_mol_s": configured_mol_s,
            "raw_feed_mol_s": raw_mol_s,
            "conditioned_feed_mol_s": bed_mol_s,
            "bed_volume_flow_m3_s": bed_volume_flow_m3_s,
            "bed_molar_mass_kg_mol": float(y @ MW),
            "dryer_added_pressure_drop_Pa": None if dryer is None else dryer.spec.added_pressure_drop_Pa,
            "assumption": "Spherical-equivalent packed particles; no distributors, filters or maldistribution; not for monoliths"}
