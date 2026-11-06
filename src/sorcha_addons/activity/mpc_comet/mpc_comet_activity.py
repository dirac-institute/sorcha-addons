"""MPC total-magnitude law, evaluated before Sorcha's detection filters."""

import astropy.units as u
import numpy as np
from sorcha.activity.base_activity import AbstractCometaryActivity

MODEL_VERSION = "mpc-hm-v1"


def mpc_comet_magnitude(h, g, rh, delta, phase_deg):
    """Return H + 5 log10(delta) + 2.5 G log10(rh) - 2.5 log10(Phi).

    ``h`` is the comet total absolute magnitude in the observation's band;
    ``g`` is the MPC heliocentric brightness exponent, NOT the asteroid HG
    slope. Distances are in au and phase angles in degrees. Inputs broadcast
    like NumPy arrays. Unknown (NaN) inputs yield unknown magnitudes. Finite
    distances must be positive and phase angles must lie in [0, 180]. Infinite
    inputs are rejected. No missing H or G is imputed by this reusable model.

    Phi is sbpy's zero-degree-normalized Halley–Marcus composite dust phase
    function. Applying it to catalog total magnitudes assumes dust-dominated
    emission and treats H as a zero-phase normalization. Catalog H may absorb
    phase/aperture effects; this is a first-order model, not a recalibration.
    """
    from sbpy.activity import phase_HalleyMarcus

    h, g, rh, delta, phase = np.broadcast_arrays(
        *[np.asarray(value, dtype=float) for value in (h, g, rh, delta, phase_deg)]
    )
    if any(np.isinf(value).any() for value in (h, g, rh, delta, phase)):
        raise ValueError("Comet photometry inputs must be finite or NaN")
    if np.any(rh <= 0) or np.any(delta <= 0):
        raise ValueError("Comet distances must be positive (au)")
    if np.any((phase < 0) | (phase > 180)):
        raise ValueError("Comet phase angles must lie in [0, 180] degrees")

    result = np.full(h.shape, np.nan)
    valid = np.isfinite(h) & np.isfinite(g) & np.isfinite(rh) & np.isfinite(delta) & np.isfinite(phase)
    if np.any(valid):
        phi = phase_HalleyMarcus(phase[valid] * u.deg)
        result[valid] = (
            h[valid]
            + 5.0 * np.log10(delta[valid])
            + 2.5 * g[valid] * np.log10(rh[valid])
            - 2.5 * np.log10(phi)
        )
    return result.item() if result.ndim == 0 else result


class MPCCometActivity(AbstractCometaryActivity):
    """Replace Sorcha's nucleus estimate with MPC total comet photometry.

    Supply H in the ordinary physical-parameter file and ``comet_G`` in the
    complex-parameter file. Sorcha supplies the color-adjusted ``H_filter``.
    Select ``phase_function = none`` and ``lc_model = none``: this model
    computes the total magnitude, rather than adding a coma to a nucleus.
    """

    def __init__(self):
        super().__init__(["H_filter", "comet_G"])

    @staticmethod
    def name_id():
        return "mpc_hm"

    def compute(self, df, observing_filters, rho, delta, alpha):
        """Evaluate total magnitude using Sorcha's light-time-corrected geometry."""
        self._validate_column_names(df)
        df["trailedSourceMagTrue"] = mpc_comet_magnitude(df["H_filter"], df["comet_G"], rho, delta, alpha)
        return df

    def maxBrightness(self, df, observing_filters, q, delta, alpha):
        """Conservatively retain all objects in Sorcha's preliminary culling.

        That filter's zero-phase perihelion estimate is not a general bound
        for this model (forward scattering and arbitrary comet G). Returning
        -inf disables only the preliminary cull; actual visit magnitudes and
        subsequent detection cuts are still evaluated normally.
        """
        return np.full(len(df), -np.inf)
