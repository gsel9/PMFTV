=======
matcomp
=======

*Low-rank matrix completion for longitudinal data*

.. image:: https://github.com/gsel9/matcomp/actions/workflows/Tests.yml/badge.svg
    :target: https://github.com/gsel9/matcomp/actions/workflows/Tests.yml
    :alt: Tests

.. image:: https://codecov.io/gh/gsel9/matcomp/branch/main/graph/badge.svg?token=GDCXEF2MGE
    :target: https://codecov.io/gh/gsel9/matcomp
    :alt: Coverage

.. image:: https://readthedocs.org/projects/matcomp/badge/?version=latest
        :target: https://matcomp.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://zenodo.org/badge/402865945.svg
   :target: https://zenodo.org/badge/latestdoi/402865945

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

matcomp is a Python library for learning coupled matrix factorizations with flexible constraints and regularization.
For a quick introduction to coupled matrix factorization see the `online documentation <https://matcomp.readthedocs.io/en/latest/index.html>`_.

Installation
------------

To install matcomp and all MIT-compatible dependencies from PyPI, you can run

.. code::

        pip install matcomp

If you also want to enable total variation regularization, you need to install all components, which comes with a MIT lisence

.. code::

        pip install matcomp[gpl]

About
-----

.. image:: docs/figures/matcomp.svg
    :alt: Illustration of a coupled matrix factorization

matcomp is a Python library that adds support for ... 


Example
-------

TODO


References
----------

* [1]: Langberg, Geir Severin RE, et al. "Matrix factorization for the reconstruction of cervical cancer screening histories and prediction of future screening results." BMC bioinformatics 23.12 (2022): 1-15.
* [2]: Langberg, Geir Severin RE, et al. "Towards a data-driven system for personalized cervical cancer risk stratification." Scientific Reports 12.1 (2022): 12083.
* [2]: TODO: cross-pop paper 