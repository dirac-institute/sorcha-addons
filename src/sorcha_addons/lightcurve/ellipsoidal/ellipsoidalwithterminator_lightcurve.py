from sorcha.lightcurves.base_lightcurve import AbstractLightCurve

from typing import List
import pandas as pd
import numpy as np

# Constants
au = 1.495978707e8  # AU in km
c = 299792.458  # speed of light in km/s
c_kmday = c * 86400  # speed of light in km/day


def cos_aspect_angle(ra, dec, ra0, dec0):
    """Compute the cosine of the aspect angle

    This angle is computed from the coordinates of the target and
    the coordinates of its pole.
    See Eq 12.4 "Introduction to Ephemerides and Astronomical Phenomena", IMCCE

    Parameters
    ----------
    ra: float
        Right ascension of the target in radians.
    dec: float
        Declination of the target in radians.
    ra0: float
        Right ascension of the pole in radians.
    dec0: float
        Declination of the pole in radians.

    Returns
    -------
    float: The cosine of the aspect angle
    """
    return np.sin(dec) * np.sin(dec0) + np.cos(dec) * np.cos(dec0) * np.cos(ra - ra0)


def rotation_phase(t, W0, W1, t0):
    """Compute the rotational phase

    This angle is computed from the location of the prime meridian at
    at reference epoch (W0, t0), and an angular velocity (W1)
    See Eq 12.1 "Introduction to Ephemerides and Astronomical Phenomena", IMCCE

    Parameters
    ----------
    t: float
        Time (JD)
    W0: float
        Location of the prime meridian at reference epoch (radian)
    W1: float
        Angular velocity of the target in radians/day.
    t0: float
        Reference epoch (JD)

    Returns
    -------
    float: The rotational phase W (radian)
    """
    return W0 + W1 * (t - t0)


def subobserver_longitude(ra, dec, ra0, dec0, W):
    """Compute the subobserver longitude (radian)

    This angle is computed from the coordinates of the target,
    the coordinates of its pole, and its rotation phase
    See Eq 12.4 "Introduction to Ephemerides and Astronomical Phenomena", IMCCE

    Parameters
    ----------
    ra: float
        Right ascension of the target in radians.
    dec: float
        Declination of the target in radians.
    ra0: float
        Right ascension of the pole in radians.
    dec0: float
        Declination of the pole in radians.
    W: float
        Rotation phase of the target in radians.

    Returns
    -------
    float: The subobserver longitude in radians.
    """
    x = -np.cos(dec0) * np.sin(dec) + np.sin(dec0) * np.cos(dec) * np.cos(ra - ra0)
    y = -(np.cos(dec) * np.sin(ra - ra0))
    return W - np.arctan2(x, y)


class EllipsoidalWithTerminatorLightCurve(AbstractLightCurve):
    """
    Produces a lightcurve from a spinning ellipsoid, partly illuminated by
    the Sun. That is, the lightcurve accounts for limb+terminator, compared
    with the simplier EllipsoidLightcurve class, for which only limbs are considered.
    It is based on the work on Ostro et al. (1988).

    The observation dataframe provided to the ``compute``
    method should have the following columns:

    * ``FieldMJD_TAI`` - time of observation [MJD].
    * ``Range_LTC_km`` - Distance to target at time of observation [km].
    * ``RA`` - SSO right ascension [deg].
    * ``Dec`` - SSO declination [deg].
    * ``phase_deg`` - Phase angle [deg].
    * ``RA_s`` - SSO right ascension as seen from the Sun [deg].
    * ``Dec_s`` - SSO declination as seen from the Sun [deg].
    * ``Period`` - Sidereal rotation period [days].
    * ``Time0`` - Reference time for the light curve [days].
    * ``phi0`` - Reference rotational phase for the light curve [radians].
    * ``RA0`` - SSO spin-axis right ascension [radians].
    * ``Dec0`` - SSO spin-axis declination [radians].
    * ``a/b`` - SSO ratio of equatorial diameters [unitless].
    * ``a/c`` - SSO ratio of longest equatorial to polar diameters [unitless].
    """

    def __init__(
        self,
        required_column_names: List[str] = [
            "fieldMJD_TAI",
            "Range_LTC_km",
            "RA_deg",
            "Dec_deg",
            "RA_s_deg",
            "Dec_s_deg",
            "Period",
            "Time0",
            "phi0",
            "RA0",
            "Dec0",
            "a/b",
            "a/c",
        ],
    ) -> None:
        super().__init__(required_column_names)

    def compute(self, df: pd.DataFrame) -> np.array:
        # Verify that the input data frame contains each of the required columns.
        self._validate_column_names(df)

        # Extract the relevant columns from the input DataFrame.
        ep = df["fieldMJD_TAI"].values - df["Range_LTC_km"] / c_kmday
        ra = np.radians(df["RA_deg"].values)
        dec = np.radians(df["Dec_deg"].values)
        ra_s = np.radians(df["RA_s_deg"].values)
        dec_s = np.radians(df["Dec_s_deg"].values)
        period = df["Period"].values
        t0 = df["Time0"].values
        phi0 = df["phi0"].values
        alpha0 = df["RA0"].values
        delta0 = df["Dec0"].values
        a_b = df["a/b"].values
        a_c = df["a/c"].values

        # Rotation (in an inertial reference frame)
        W = rotation_phase(ep, phi0, 2 * np.pi / period, t0)

        # Sub-observer angles
        cos_aspect = cos_aspect_angle(ra, dec, alpha0, delta0)
        cos_aspect_2 = cos_aspect**2
        sin_aspect_2 = 1 - cos_aspect_2
        sin_aspect = np.sqrt(sin_aspect_2)
        rot_phase = subobserver_longitude(ra, dec, alpha0, delta0, W)

        # Subsolar angles
        cos_aspect_s = cos_aspect_angle(ra_s, dec_s, alpha0, delta0)
        cos_aspect_s_2 = cos_aspect_s**2
        sin_aspect_s_2 = 1 - cos_aspect_s_2
        sin_aspect_s = np.sqrt(sin_aspect_s_2)
        rot_phase_s = subobserver_longitude(ra_s, dec_s, alpha0, delta0, W)

        # Sub-observer (e.TQe):
        eQe = (
            sin_aspect_2 * (np.cos(rot_phase) ** 2 + (a_b**2) * np.sin(rot_phase) ** 2)
            + cos_aspect_2 * a_c**2
        )

        # Sub-Solar (s.TQs):
        sQs = (
            sin_aspect_s_2 * (np.cos(rot_phase_s) ** 2 + (a_b**2) * np.sin(rot_phase_s) ** 2)
            + cos_aspect_s_2 * a_c**2
        )

        # Cross-term (e.TQs):
        eQs = (
            sin_aspect * np.cos(rot_phase) * sin_aspect_s * np.cos(rot_phase_s)
            + sin_aspect * np.sin(rot_phase) * sin_aspect_s * np.sin(rot_phase_s) * (a_b**2)
            + cos_aspect * cos_aspect_s * a_c**2
        )

        # Integrated flux is the sum of half-limb + half-terminator ellipses
        I_tot = (np.sqrt(eQe) + eQs / np.sqrt(sQs)) / 2.0
        return -2.5 * np.log10(I_tot)

    # this method defines the same of the class inside LC_METHODS
    @staticmethod
    def name_id() -> str:
        return "ellipsoidalwithterminator"

    def maxBrightness(self, df: pd.DataFrame) -> float:
        return -2.5 * np.log10(df["a/c"])  # Brightest seen from the pole
