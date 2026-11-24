"""Direct contact with saturated steam in a one-dimensional packed bed.

Gas is ideal; liquid has constant density and heat capacity. Phase transfer
relaxes the local vapour concentration towards saturation at the bead surface.
Drainage is a first-order distributed sink. Both times are case inputs.
"""
from dataclasses import dataclass
import numpy as np

from ..common import MW, R, gas_enthalpy_mol
from ..water_properties import (
    WATER_CP_LIQUID_J_MOL_K, liquid_enthalpy_J_mol, saturation_pressure_Pa,
    saturation_temperature_K, vapour_enthalpy_J_mol,
)

LIQUID_VOLUME_M3_MOL = MW[1] / 1000.0
LIQUID_CP = WATER_CP_LIQUID_J_MOL_K


@dataclass(frozen=True)
class SteamSupply:
    pressure_Pa: float
    max_mol_s: float
    vapour_h: float
    generator_J_mol: float

    @classmethod
    def from_case(cls, section, ref):
        pressure = section["supply_pressure_Pa"]
        temperature = saturation_temperature_K(pressure)
        h = vapour_enthalpy_J_mol(temperature, ref)
        feed_h = liquid_enthalpy_J_mol(section["feedwater_C"] + 273.15, ref)
        return cls(pressure, section["flow_kg_h"] / (3600 * MW[1]), h,
                   (h - feed_h) / section["generator_efficiency"])

    def flow(self, inlet_pressure):
        # Flow-limited regulator; opening falls continuously near backpressure.
        opening = np.clip((self.pressure_Pa - inlet_pressure) / (0.05 * self.pressure_Pa), 0, 1)
        return float(self.max_mol_s * opening)


def gas_volume(model, state):
    if model.layout.liquid is None:
        return model.cell_gas_volume_m3
    liquid = np.maximum(state[model.layout.liquid], 0)
    occupied = liquid * LIQUID_VOLUME_M3_MOL
    if np.any(occupied >= model.cell_pore_volume_m3):
        raise ValueError("Liquid fills the packed void in a steam cell; the gas-continuous model ends at flooding.")
    return model.cell_gas_volume_m3 - occupied


def film_rates(vapour_mol, gas_T, solid_T, volume, liquid, section):
    saturation = np.array([saturation_pressure_Pa(t) for t in solid_T])
    excess = vapour_mol - saturation * volume / (R * gas_T)
    # Reverse transfer can only draw on the retained liquid.
    rate = np.maximum(excess, -liquid) / section["phase_transfer_s"]
    return saturation, rate, liquid / section["drain_time_s"]


def wet_bed_drop(flow, dry_void, gas_void):
    """Imposed adsorption airflow through equal-length cells, with reduced free void."""
    ratio = dry_void / gas_void
    solid_ratio = (1 - gas_void) / (1 - dry_void)
    return float(flow["viscous_drop_Pa"] * np.mean(ratio**3 * solid_ratio**2)
                 + flow["inertial_drop_Pa"] * np.mean(ratio**3 * solid_ratio)
                 + flow["minor_drop_Pa"])


def axial_exchange(pressure, temperature, fractions, gas_void, face_area, dx, diameter, viscosity, ref):
    """Invert Ergun between cell centres; upwind both composition and enthalpy."""
    dp = pressure[:-1] - pressure[1:]
    donor = np.where(dp >= 0, np.arange(dp.size), np.arange(dp.size) + 1)
    eps = 0.5 * (gas_void[1:] + gas_void[:-1])
    density_mol = pressure[donor] / (R * temperature[donor])
    density = density_mol * (fractions[donor] @ MW)
    a = 150 * viscosity * (1-eps)**2 / (eps**3 * diameter**2)
    b = 1.75 * density * (1-eps) / (eps**3 * diameter)
    gradient = dp / dx
    velocity = 2 * gradient / (a + np.sqrt(a*a + 4*b*np.abs(gradient)))
    molar_flux = face_area * velocity * density_mol
    species = fractions[donor] * molar_flux[:, None]
    enthalpy = (species * gas_enthalpy_mol(temperature[donor, None], ref)).sum(axis=1)
    dn = np.zeros_like(fractions)
    heat = np.zeros_like(temperature)
    dn[:-1] -= species
    dn[1:] += species
    heat[:-1] -= enthalpy
    heat[1:] += enthalpy
    return dn, heat
