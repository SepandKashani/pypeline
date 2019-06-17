.. ############################################################################
.. install.rst
.. ===========
.. Author : Sepand KASHANI [kashani.sepand@gmail.com]
.. ############################################################################


Installation
============

Pypeline modules are written in Python3/C++14 and tested on x86_64 systems running Linux.

The C++ library ``libpypeline.so`` requires the following tools to be available:

+-------------+------------+
| Library     |    Version |
+=============+============+
| Eigen       |      3.3.7 |
+-------------+------------+
| PyBind11    |      2.3.0 |
+-------------+------------+
| FFTW        |      3.3.8 |
+-------------+------------+
| Intel MKL   |     2019.4 |
+-------------+------------+

Aside from Intel MKL which ships with `conda <https://conda.io/docs/>`_, other dependencies are
downloaded and compiled during the installation process.

After installing `Miniconda <https://conda.io/miniconda.html>`_:

* Install (most) C++ performance libraries::

    $ cd <pypeline_dir>/
    $ conda create --name=pypeline        \
                    --channel=defaults    \
                    --channel=conda-forge \
                    --file=conda_requirements.txt
    $ source pypeline.sh --no_shell

* Install `pyFFS <https://github.com/imagingofthings/pyFFS>`_::

    $ cd <pypeline_dir>/
    $ git clone git@github.com:imagingofthings/pyFFS.git
    $ cd <pyFFS_dir>/
    $ git checkout v1.0
    $ python3 setup.py develop
    $ python3 setup.py build_sphinx

* Install `ImoT_tools <https://github.com/imagingofthings/ImoT_tools>`_::

    $ cd <pypeline_dir>/
    $ git clone git@github.com:imagingofthings/ImoT_tools.git
    $ cd <ImoT_tools_dir>/
    $ git checkout v1.0
    $ python3 setup.py develop
    $ python3 setup.py build_sphinx

* Download/Compile dependencies (~20 minutes)::

    $ cd <pypeline_dir>/
    $ PYPELINE_C_COMPILER=<path_to_executable>
    $ PYPELINE_CXX_COMPILER=<path_to_executable>
    $ python3 build.py --download_dependencies
    $ python3 build.py --install_dependencies                    \
                       --C_compiler="${PYPELINE_C_COMPILER}"     \
                       --CXX_compiler="${PYPELINE_CXX_COMPILER}" \
                      [--OpenMP]

* Install `pypeline`::

    $ cd <pypeline_dir>/
    $ python3 build.py --lib={Debug, Release}                    \
                       --C_compiler="${PYPELINE_C_COMPILER}"     \
                       --CXX_compiler="${PYPELINE_CXX_COMPILER}" \
                      [--OpenMP]
    $ python3 test.py         # Run test suite (optional, recommended)
    $ python3 build.py --doc  # Generate documentation (optional)


To launch a Python3 shell containing Pypeline, run ``pypeline.sh``.


Remarks
-------

* Pypeline is tested with GCC 8.3.0 and Clang 8.0.0 on x86_64 systems running Linux.
  It should also run correctly on macOS, but we provide no support for this.

* If building with ``--OpenMP``, Cmake may incorrectly link ``libpypeline.so`` with a version of
  OpenMP shipped with `conda` instead of the system's OpenMP shared library.
  In case the compilation stage above fails, inspect Cmake's log files for OpenMP ambiguities.

* The ``--install_dependencies`` command above will automatically download and install all C++
  dependencies listed in the table.
  If the libraries are already available on the system and you wish to use them instead of the ones
  we provide, then you will have to modify ``CMakeLists.txt`` and configuration files under ``cmake/``
  accordingly.

* ``pypeline.sh`` sets up the required environment variables to access built libraries and setup
  components such as OpenMP thread-count and MKL precision preferences.
  It is highly recommended to check the ``load_pypeline_env()`` function in this file and tailor
  some environment variables to your system.
