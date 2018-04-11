Scikit Spline
=============

A Scikit-learn interface on Scipy's ``Univariate Spline``.

.. code-block:: python

  import numpy as np
  from skspline import Spline

  model = Spline(k=3)
  model.fit()
