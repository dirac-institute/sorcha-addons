MPC Comet Total Magnitudes
============================

``MPCCometActivity`` (``comet_activity = mpc_hm``) computes total comet
magnitudes from catalog absolute magnitude H and heliocentric exponent G,
including the zero-degree-normalized Halley–Marcus (Schleicher–Marcus)
composite dust phase function:

.. math::

   m_f = H_f + 5\log_{10}\Delta + 2.5G\log_{10}r_h
         - 2.5\log_{10}\Phi_{HM}(\alpha).

Distances are in au and phase angles in degrees. ``comet_G`` is the MPC
comet brightness exponent, not the asteroid HG slope ``GS``. The result
replaces the nucleus estimate; it does not add another coma flux to an
already total magnitude. Sorcha applies the normal color offsets to H first.

Configuration and inputs
------------------------

Use the following Sorcha configuration::

    [PHASECURVES]
    phase_function = none
    [LIGHTCURVE]
    lc_model = none
    [ACTIVITY]
    comet_activity = mpc_hm
    [EXPERT]
    brute_force = True

Supply H in the regular physical-parameter file, for example::

    ObjID,H_r,g-r
    comet_a,12.0,0.5
    comet_b,14.0,0.5

Supply the MPC exponent through a separate complex-parameter file::

    ObjID,comet_G
    comet_a,4.0
    comet_b,6.0

Pass the second file with ``sorcha run --cp complex.csv`` alongside the usual
orbit, physical-parameter, pointing, and configuration arguments. All three
object files must contain matching IDs. If H is not already in the selected
reference band, the caller must choose and document a band conversion.

This add-on does not impute missing H or G: NaNs produce unknown magnitudes.
Callers may supply an explicitly recorded fallback exponent. Angles outside
[0, 180], nonpositive distances, and infinite inputs raise ``ValueError``.
The phase curve uses ``sbpy.activity.phase_HalleyMarcus`` (spline interpolation
when SciPy is available). No network access is needed for the calculation.

The ordinary phase and rotational lightcurve settings should remain ``none``;
this model replaces the total magnitude and does not combine those effects.
Sorcha's early faint-object cull is bypassed with ``brute_force = True``.
For direct callers with culling enabled, ``maxBrightness`` conservatively
returns negative infinity, retaining every object for propagation. Actual
visit magnitudes and subsequent detection filtering remain enabled.

Assumptions and citations
-------------------------

The model treats H as a zero-phase total-magnitude normalization and assumes
dust-dominated emission. Catalog fits may already absorb phase, aperture,
and activity variations. This is a first-order approximation; it does not
recalibrate H/G, separate gas and dust, model outbursts, or convert integrated
coma magnitudes to point-source detection magnitudes.

Please cite the phase-function work:

* Schleicher, Millis & Birch (1998), Icarus 132, 397–417.
* Marcus (2007), International Comet Quarterly 29, 39–66.
* Schleicher & Bair (2011), AJ 141, 177.
* `Schleicher's composite phase function and table <https://asteroid.lowell.edu/comet/dustphase/>`_.

The MPC H/G total-magnitude convention is also described in the
`BAA handbook <https://britastro.org/wp-content/uploads/2018/01/Handbook_2026.pdf>`_.
Please also acknowledge Sorcha, sorcha-addons, and sbpy when using this implementation.

.. autofunction:: sorcha_addons.activity.mpc_comet.mpc_comet_magnitude
    :no-index:

.. autoclass:: sorcha_addons.activity.mpc_comet.MPCCometActivity
    :members:
    :no-index:
