from sorcha_addons.lightcurve.ellipsoidal.ellipsoidalwithterminator_lightcurve import (
    EllipsoidalWithTerminatorLightCurve,
)

import pandas as pd
import numpy as np


def test_ellipsoidalwithterminator_lightcurve_name():
    assert "ellipsoidalwithterminator" == EllipsoidalWithTerminatorLightCurve.name_id()

    # Constants to correct for light travel time
    c = 299792.458  # speed of light in km/s
    c_kmday = c * 86400  # speed of light in km/day

    # def test_compute_simple():
    df = pd.DataFrame(
        {
            "fieldMJD_TAI": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0],
            "Range_LTC_km": [
                c_kmday,
                c_kmday,
                c_kmday,
                c_kmday,
                c_kmday,
                c_kmday,
                c_kmday,
            ],
            "RA_deg": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "Dec_deg": [0.0, 0.0, -90.0, 0.0, 0.0, -90.0, 0.0],
            "RA_s_deg": [0.0, 0.0, 0.0, 90.0, 90.0, 90.0, 179.9],
            "Dec_s_deg": [0.0, 0.0, -90.0, 0.0, 0.0, 0.0, 0.0],
            "Period": [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
            "Time0": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "phi0": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "RA0": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "Dec0": [
                np.pi / 2,
                np.pi / 2.0,
                np.pi / 2,
                np.pi / 2,
                np.pi / 2,
                np.pi / 2,
                np.pi / 2,
            ],
            "a/b": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2],
            "a/c": [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4],
        }
    )

    model = EllipsoidalWithTerminatorLightCurve()
    output = model.compute(df)

    # Expected values from the three axes
    assert np.isclose(output.values[0], -2.5 * np.log10(2))
    assert np.isclose(output.values[1], -2.5 * np.log10(1))
    assert np.isclose(output.values[2], -2.5 * np.log10(4))
    assert np.isclose(output.values[3], -2.5 * np.log10(1))
    assert np.isclose(output.values[4], -2.5 * np.log10(0.5))
    assert np.isclose(output.values[5], -2.5 * np.log10(2))
    assert np.isclose(output.values[6], -2.5 * np.log10(3.8077e-7))


test_ellipsoidalwithterminator_lightcurve_name()
