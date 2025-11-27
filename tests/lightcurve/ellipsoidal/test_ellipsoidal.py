from sorcha_addons.lightcurve.ellipsoidal.ellipsoidal_lightcurve import (
    EllipsoidalLightCurve,
    cos_aspect_angle,
    rotation_phase,
    subobserver_longitude,
)
import pandas as pd
import numpy as np


def test_ellipsoidal_lightcurve_name():
    assert "ellipsoidal" == EllipsoidalLightCurve.name_id()

    # Constants to correct for light travel time
    c = 299792.458  # speed of light in km/s
    c_kmday = c * 86400  # speed of light in km/day

    # def test_compute_simple():
    data_dict = {
        "fieldMJD_TAI": [1.0, 2.0, 3.0],
        "Range_LTC_km": [c_kmday, c_kmday, c_kmday],
        "RA_deg": [0.0, 0.0, 0.0],
        "Dec_deg": [0.0, 0.0, -90.0],
        "Period": [4.0, 4.0, 4.0],
        "Time0": [0.0, 0.0, 0.0],
        "phi0": [0.0, 0.0, 0.0],
        "RA0": [0.0, 0.0, 0.0],
        "Dec0": [np.pi / 2, np.pi / 2.0, np.pi / 2],
        "a/b": [2.0, 2.0, 2.0],
        "a/c": [4.0, 4.0, 4.0],
    }

    df = pd.DataFrame.from_dict(data_dict)
    model = EllipsoidalLightCurve()
    output = model.compute(df)

    # Expected values from the three axes
    assert np.isclose(output.values[0], -2.5 * np.log10(2))
    assert np.isclose(output.values[1], -2.5 * np.log10(1))
    assert np.isclose(output.values[2], -2.5 * np.log10(4))

test_ellipsoidal_lightcurve_name()
