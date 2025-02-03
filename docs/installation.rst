
Installing Sorcha Add-ons
==============================

.. note::
   ``Sorcha add-ons`` is both conda/mamba and pip installable. We recommend installing via conda/mamba.

Step 1: First Install Sorcha
------------------------------

``Sorcha add-ons`` is a companion package to ``Sorcha``, a Solar System survey simulator designed for the `Vera C. Rubin Observatory Legacy Survey of Space and Time (LSST) <https://rubinobservatory.org>`_. 

The first step is to install ``Sorcha``. Folow the instructions `here <https://sorcha.readthedocs.io/en/latest/installation.html>`_ to set up a python environment and get ``Sorcha`` and its associated packages installed. 


Step 2: Installing Sorcha Add-ons
--------------------------------------

Once you installed ``Sorcha`` and its associated packages, it's straightforward to install ``Sorcha Add-ons``. Unless you're editing the source code,you can use the version of  ``Sorcha add-ons`` published on conda-forge.

If using conda::

   conda install -c conda-forge sorcha-addons

If using mamba::

   mamba install -c conda-forge sorcha-addons

You can install ``Sorcha add-ons`` via from PyPi using pip, but installation via  conda/mamba is recommended.

If using pip::

   pip install sorcha-addons


Installing Sorcha Add-ons in Development Mode
---------------------------------------------------------------------

**This is the installation method for adding/editing ``Sorcha add-ons``'s codebase or for working on/updating ``Sorcha add-ons`'s  documentation.**

**Step 1** Create a directory to contain the ``Sorcha add-ons`` repo::

   mkdir sorcha-addons

**Step 2** Navigate to the directory you want to store the ``Sorcha add-ons`` source code in::

   cd sorcha-addons

**Step 3** Download the ``Sorcha Add-ons`` source code via::

   git clone https://github.com/dirac-institute/sorcha-addons.git

**Step 4** Navigate to the  ``Sorcha Add-ons`` repository directory::

   cd sorcha-addons

**Step 5** Install an editable (in-place) development version of ``Sorcha Add-ons``. This will allow you to run the code from the source directory.

If you just want the source code installed so edits in the source code are automatically installed::

   pip install -e .

If you are going to be editing documentation or significantly modifying unit tests, it is best to install the full development version::

   pip install -e '.[dev]'

**Step 6 (Optional unless working on documentation):** You will need to install the pandoc package (either via conda/pip or `direct download <https://pandoc.org/installing.html>`_ 


