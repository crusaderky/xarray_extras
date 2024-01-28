Development Guidelines
======================

Install
-------

1. Clone this repository with git:

.. code-block:: bash

     git clone git@github.com:crusaderky/xarray_extras.git
     cd xarray_extras

2. Install anaconda or miniconda (OS-dependent)
3. .. code-block:: bash

     conda env create -n xarray_extras --file ci/requirements.yml
     conda activate xarray_extras

4. Install C compilation stack:

   Linux
       .. code-block:: bash

          conda install gcc_linux-64

   MacOSX
        .. code-block:: bash

           conda install clang_osx-64

   Windows
        You need to manually install the Microsoft C compiler tools. Refer to CPython
        documentation.


To keep a fork in sync with the upstream source:

.. code-block:: bash

   cd xarray_extras
   git remote add upstream git@github.com:crusaderky/xarray_extras.git
   git remote -v
   git fetch -a upstream
   git checkout main
   git pull upstream main
   git push origin main

Test
----

Test using ``py.test``:

.. code-block:: bash

   python setup.py build_ext --inplace
   py.test xarray_extras

Code Formatting
---------------

xarray_extras uses several code linters (black, ruff, mypy), which are enforced by CI.
Developers should run them locally before they submit a PR, through the single command

.. code-block:: bash

    pre-commit run --all-files

This makes sure that linter versions and options are aligned for all developers.

Optionally, you may wish to setup the `pre-commit hooks <https://pre-commit.com/>`_ to
run automatically when you make a git commit. This can be done by running:

.. code-block:: bash

   pre-commit install

from the root of the xarray_extras repository. Now the code linters will be run each time
you commit changes. You can skip these checks with ``git commit --no-verify`` or with
the short version ``git commit -n``.
