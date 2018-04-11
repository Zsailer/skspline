from functools import wraps

from scipy.interpolate import UnivariateSpline
from sklearn.metrics import r2_score
from sklearn.base import (BaseEstimator,
                          RegressorMixin,
                          TransformerMixin)


class Spline(BaseEstimator, RegressorMixin, TransformerMixin):
    """One-dimensional smoothing spline fit to a given set of data points.

    Fits a spline y = spl(x) of degree k to the provided x, y data. s
    specifies the number of knots by specifying a smoothing condition.

    Parameters
    ----------
    k : int, optional
        Degree of the smoothing spline. Must be <= 5. Default is k=3,
        a cubic spline.

    s : float or None, optional
        Positive smoothing factor used to choose the number of knots.
        Number of knots will be increased until the smoothing condition is
        satisfied:

        .. code-block::

            sum((w[i] * (y[i]-spl(x[i])))**2, axis=0) <= s

        f None (default), ``s = len(w)`` which should be a good value if
        ``1/w[i]`` is an estimate of the standard deviation of ``y[i]``.
        If 0, spline will interpolate through all data points.

    Attributes
    ----------
    coef_ : array
        Spline coefficients.

    knots_ : array
        position of knots.
    """
    def __init__(self, k=3, s=None):
        self.k = k
        self.s = s
        self._spline = None

    @property
    def coefs_(self):
        """Spline coefficients"""
        return self._spline.get_coeffs()

    @property
    def knots_(self):
        """Position of knots."""
        return self._spline.get_knots()

    def fit(self, X, y, sample_weight=None):
        """Fits a spline y = spl(x) of degree k to the provided x, y data. s
        specifies the number of knots by specifying a smoothing condition.

        Parameters
        ----------
        X : array
            Training data.

        y : array
            Target values.

        sample_weight : array, optional

        Returns
        -------
        self :
            returns an instance of self.
        """
        # Fit the spline
        self._spline = UnivariateSpline(
            x=X,
            y=y,
            w=sample_weight,
            k=self.k,
            s=self.s)

        # Return model
        return self

    def predict(self, X):
        """Predict using a spline.

        Parameters
        ----------
        X : array
            Samples.

        Returns
        -------
        C : array
            Returns predicted values.
        """
        return self._spline(X)

    @wraps(RegressorMixin.score)
    def score(self, X, y):
        # Compute y-true and y-predicted
        y_true = y
        y_pred = self.predict(X)

        # Compute coefficient of determination.
        return r2_score(y_true, y_pred)
