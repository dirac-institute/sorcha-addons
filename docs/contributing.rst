.. _contributing: 

Contribution to Sorcha Add-ons
============================================

``Sorcha`` has the ability user provided functions though python classes that augment/change the apparent brightness calculations for the synthetic Solar System objects. Rather than forcing the user directly modify  the ``Sorcha`` codebase every time they want to apply a different model for representing the effects of rotational light curves or cometary activity, we provide the ability to develop separate activity and light curve/brightness enhancement functions as  plugins using our template classes  and add them to the ``Sorcha add-ons`` package. In both cases, any derived class must inherit from the corresponding base class (`cometary activity template class <https://sorcha.readthedocs.io/en/latest/postprocessing.html#cometary-activity-template-class>`_ and `light curve template class <https://sorcha.readthedocs.io/en/latest/postprocessing.html#lightcurve-template-class>`_) and follow its API, to ensure that ``Sorcha`` knows how to find and use your class. Please use the instructions below to contribute your new activity/lightcurve perscriptions to ``Sorcha add-ons``.

Create Your Environment and Install the Package in Development Mode
-----------------------------------------------------------------------

**Step 1** Create a directory to contain the ``Sorcha add-ons`` repo::

   mkdir sorcha-addons

**Step 2** Navigate to the directory you want to store the ``Sorcha add-ons`` source code in::

   cd sorcha-addons

**Step 3* Clonek``Sorcha Add-ons`` source code via::

   git clone https://github.com/dirac-institute/sorcha-addons.git

**Step 4** Navigate to the  ``Sorcha Add-ons`` repository directory::

   cd sorcha-addons

**Step 5** Install the full development version::

   pip install -e '.[dev]'


Create your Contribution
---------------------------

Add src code in new module under ``.../src/socha``

Add tests in new folder under ``.../tests/``

Add example notebook in ``.../docs/notebooks/``. Update ``.../docs/notebooks.rst``.

Add documentation page in ``.../docs/community_modules``. Update ``.../docs/community_modules.rst``.
Be sure to include information about how to cite your work.

Create a Pull Request For Review
----------------------------------

Commit your changes to a fork of the github repository  and create a pull request for review.

Add **mschwamb** as the reviewer.
