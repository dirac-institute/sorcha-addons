import numpy as np
import pandas as pd
import pytest

from sorcha.activity.activity_registration import CA_METHODS, update_activity_subclasses
from sorcha.modules.PPCalculateApparentMagnitude import PPCalculateApparentMagnitude
from sorcha.modules.PPFaintObjectCullingFilter import PPFaintObjectCullingFilter
from sorcha.modules.PPMagnitudeLimit import PPMagnitudeLimit
from sorcha_addons.activity.mpc_comet import MPCCometActivity, mpc_comet_magnitude


@pytest.mark.parametrize(
    "angle,phi",
    [
        (0, 1),
        (15, 0.5872),
        (30, 0.4142),
        (50, 0.3397),
        (90, 0.5375),
        (150, 9.7981),
        (180, 1231.1735),
    ],
)
def test_published_phase_values(angle, phi):
    # Lowell's zero-degree normalized table; sbpy tabulates rounded flux ratios.
    assert mpc_comet_magnitude(0, 4, 1, 1, angle) == pytest.approx(-2.5 * np.log10(phi), abs=3e-5)


def test_distance_exponent_colors_and_unknowns():
    result = mpc_comet_magnitude([10, 10.5, 10, np.nan], [4, 4, 2, 4], 2, 3, 0)
    np.testing.assert_allclose(
        result[:3],
        [10, 10.5, 10] + 5 * np.log10(3) + np.array([10, 10, 5]) * np.log10(2),
    )
    assert np.isnan(result[3])
    assert np.isnan(mpc_comet_magnitude(10, np.nan, 1, 1, 0))
    assert np.isnan(mpc_comet_magnitude(10, 4, np.nan, 1, 0))
    assert np.isnan(mpc_comet_magnitude(10, 4, 1, 1, np.nan))
    assert mpc_comet_magnitude([], [], [], [], []).shape == (0,)


@pytest.mark.parametrize(
    "args",
    [
        (10, 4, 0, 1, 0),
        (10, 4, 1, -1, 0),
        (10, 4, 1, 1, -1),
        (10, 4, 1, 1, 181),
        (np.inf, 4, 1, 1, 0),
    ],
)
def test_invalid_inputs(args):
    with pytest.raises(ValueError):
        mpc_comet_magnitude(*args)


def test_interpolation_is_finite_and_continuous():
    angles = np.linspace(0, 180, 18001)
    mag = mpc_comet_magnitude(0, 4, 1, 1, angles)
    assert np.isfinite(mag).all()
    assert np.abs(np.diff(mag)).max() < 0.01
    assert mag[3000] > 0
    assert mag[15000] < 0


def test_registered_model_runs_before_selection():
    update_activity_subclasses()
    assert CA_METHODS["mpc_hm"] is MPCCometActivity
    au = 149597870.7
    frame = pd.DataFrame(
        {
            "ObjID": ["A", "B", "C"],
            "H_r": [20.0, 20.0, np.nan],
            "g-r": [0.5] * 3,
            "comet_G": [4.0] * 3,
            "optFilter": ["r", "g", "r"],
            "phase_deg": [30.0] * 3,
            "Obj_Sun_LTC_km": [2 * au] * 3,
            "Range_LTC_km": [np.sqrt(3) * au] * 3,
        }
    )
    legacy = PPCalculateApparentMagnitude(frame.copy(), "none", "r", ["g-r"], ["r", "g"])
    active = PPCalculateApparentMagnitude(
        frame.copy(),
        "none",
        "r",
        ["g-r"],
        ["r", "g"],
        cometary_activity_choice="mpc_hm",
    )
    expected = np.array([20.0, 20.5]) + 10 * np.log10(2) + 5 * np.log10(np.sqrt(3)) - 2.5 * np.log10(0.4142)
    np.testing.assert_allclose(active.trailedSourceMagTrue[:2], expected)
    assert np.isnan(active.trailedSourceMagTrue[2])
    # Apply the actual magnitude filter to true magnitudes, without randomization.
    assert len(PPMagnitudeLimit(legacy, 24, colname="trailedSourceMagTrue")) == 2
    assert len(PPMagnitudeLimit(active, 24, colname="trailedSourceMagTrue")) == 0


def test_early_cull_retains_faint_and_unknown_comets():
    update_activity_subclasses()
    frame = pd.DataFrame(
        {
            "ObjID": ["faint", "unknown"],
            "q": [5.0, 5.0],
            "H_r": [40.0, np.nan],
            "comet_G": [4.0, 4.0],
        }
    )
    pointings = pd.DataFrame({"optFilter": ["r"], "fieldFiveSigmaDepth_mag": [24.0]})
    result = PPFaintObjectCullingFilter(frame, pointings, "r", ["r"], None, "mpc_hm")
    assert result.ObjID.tolist() == ["faint", "unknown"]


def test_compute_requires_catalog_slope():
    with pytest.raises(ValueError, match="comet_G"):
        MPCCometActivity().compute(pd.DataFrame({"H_filter": [10]}), ["r"], [1], [1], [0])
