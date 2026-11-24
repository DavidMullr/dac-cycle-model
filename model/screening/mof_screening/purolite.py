"""Purolite A110 water-conditioned reference isotherms, on a dry resin mass basis."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np

from .materials import MW, R, MaterialAdapter, buck_saturation_pressure_Pa


# Piscina et al. Tables 3 and 4. Only diffusivity ratios enter the LDF scaling.
_WATER_RH = (0.075, 0.1, 0.15, 0.2, 0.4, 0.5, 0.6, 0.7)
_WATER_D = np.array([4.48, 4.41, 2.63, 2.11, 0.115, 0.0917, 0.0650, 0.0646])
_CO2_RH = (0., 0.30, 0.55, 0.80)
_CO2_D = np.array([0.311, 0.161, 0.316, 0.59])


def _low_pressure_shape(p):
    # Low et al. Table 7, td-Toth at 298.15 K. Normalization cancels q_inf.
    bp = 3.4e-14 * math.exp(70100 / (R * 298.15)) * p
    return bp / (1 + bp**0.401)**(1 / 0.401)


def _gab_coefficients(temperature):
    # Low et al. Table 9 and Eqs. 17-22, with energies converted to J/mol.
    t = min(308.15, max(288.15, temperature))
    e10 = 57220 - 44.38*t
    c = math.exp((45790 - math.exp(0.02441*t) - e10) / (R*t))
    k = math.exp((52520 - 31.08*t - e10) / (R*t))
    return c, k


def _water_loading(rh, c, k):
    z = k*rh
    return 7.75*c*z / ((1-z)*(1+(c-1)*z))


@dataclass
class PuroliteAdapter(MaterialAdapter):
    data: Mapping
    storage_labels = ("CO2", "H2O")
    species_for_storage = np.array([0, 1])

    def __post_init__(self):
        for key in ("material_id", "name", "equilibrium_family", "reference_temperature_K",
                    "maximum_bed_temperature_C", "solid_cp_J_kg_K"):
            setattr(self, key, self.data[key])
        self.heats = np.array([self.data[key]["heat_J_mol"] for key in ("co2", "water")])
        self.cp_molar = MW[:2] * self.data["adsorbed_phase_cp_J_kg_K"]
        self.ad = np.asarray(self.data["kinetics"]["adsorption_rate_s_inv"])
        self.de = np.asarray(self.data["kinetics"]["desorption_rate_s_inv"])
        self.curves = []
        for row in self.data["co2"]["curves"]:
            p = np.array(row["pressure_Pa"])
            q = np.array(row["loading_mol_kg"])
            self.curves.append((np.log(p), q, q[0]/_low_pressure_shape(p[0])))
        self.max_co2 = max(q[-1] for _, q, _ in self.curves)

    def _co2(self, temperature, partial, rh):
        if partial <= 0:
            return 0.
        mapped = partial * math.exp(self.heats[0] / R * (1/temperature - 1/self.reference_temperature_K))
        logp = math.log(mapped)
        i = min(2, max(0, int(np.searchsorted(_CO2_RH, rh)) - 1))
        fraction = min(1., max(0., (rh-_CO2_RH[i])/(_CO2_RH[i+1]-_CO2_RH[i])))
        targets = []
        for pressure, loading, scale in self.curves[i:i+2]:
            targets.append(scale*_low_pressure_shape(mapped) if logp < pressure[0]
                           else np.interp(logp, pressure, loading))
        return targets[0] + fraction*(targets[1]-targets[0])

    def _water(self, temperature, partial):
        rh = min(1., max(0., partial/buck_saturation_pressure_Pa(temperature-273.15)))
        c, k = _gab_coefficients(temperature)
        return rh, _water_loading(rh, c, k), c, k

    def equilibrium(self, temperature_K, partials_Pa):
        rh, water, _, _ = self._water(temperature_K, partials_Pa[1])
        return np.array([self._co2(temperature_K, partials_Pa[0], rh), water])

    def rates(self, temperature_K, partials_Pa, storage):
        gas_rh, water, c, k = self._water(temperature_K, partials_Pa[1])
        # Invert GAB on stored water to recover the corresponding bead humidity.
        # Use the conjugate root for stable evaluation down to a dry bead.
        y = max(0., storage[1]) / 7.75
        b = c - y*(c-2)
        bead_rh = min(1., 2*y / (k*(b + math.sqrt(b*b + 4*y*y*(c-1)))))
        co2 = self._co2(temperature_K, partials_Pa[0], bead_rh)
        t = min(343.15, max(293.15, temperature_K))
        co2_scale = math.exp(47890/R*(1/self.reference_temperature_K-1/t))
        scale = np.array([
            co2_scale * np.interp(bead_rh, _CO2_RH, _CO2_D) / _CO2_D[0],
            np.interp(gas_rh, _WATER_RH, _WATER_D) / _WATER_D[0],
        ])
        delta = np.array([co2, water]) - storage
        rates = np.where(delta >= 0, self.ad, self.de) * scale * delta
        return rates, np.array([rates[0], rates[1], 0., 0.])

    def component_cp_molar_J_mol_K(self):
        return self.cp_molar.copy()

    def absolute_energy_J_mol(self, temperature_K):
        return self.cp_molar*(temperature_K-self.reference_temperature_K) + R*self.reference_temperature_K - self.heats

    def storage_scale(self):
        return np.array([self.max_co2, _water_loading(1., *_gab_coefficients(self.reference_temperature_K))])

    def source_report(self):
        return {"adapter": self.equilibrium_family, "co2_states": 1,
                "species_states": list(self.storage_labels),
                "co2_equilibrium_RH_nodes": list(_CO2_RH),
                "co2_equilibrium_temperature_K": self.reference_temperature_K,
                "high_RH_policy": "Hold 80% CO2 curve above 80% RH. Water follows GAB up to saturation.",
                "pressure_extension": "Matched dry Toth below first point on each curve; constant above last.",
                "caloric_closure": "Constant unary reference heats; humidity interaction heat omitted.",
                "water_mixture": "Presaturated CO2 curves interpolated at the humidity corresponding to stored water."}

