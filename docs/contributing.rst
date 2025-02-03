.. _contributing: 

Contribution guide
==================

Create Your Environment and Install the Package in Development Mode
-------------------------------------

Clone the source code with ``git clone ...``.

Install the in developer mode ``pip install -e .[dev]``.

Initialize pre-commit with ``pre-commit install``.


Create your Contribution
------------------------

Add src code in new module under ``.../src/socha_community_utils``

Add tests in new folder under ``.../tests/``

Add example notebook in ``.../docs/notebooks/``. Update ``.../docs/notebooks.rst``.

Add documentation page in ``.../docs/community_modules``. Update ``.../docs/community_modules.rst``.
Be sure to include information about how to cite your work.

Create a PR for review
----------------------

Commit your changes to a new git branch and create a PR for review.

Add **mschwamb** as the reviewer.
