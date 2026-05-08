# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements SPCIForecaster — Sequential Predictive Conformal Inference.

Reference: Xu & Xie, "Sequential Predictive Conformal Inference for Time Series",
ICML 2023. https://arxiv.org/abs/2212.03463
"""

__all__ = ["SPCIForecaster"]
__author__ = ["marrov"]

from math import floor

import numpy as np
import pandas as pd
from sklearn.base import clone

from sktime.forecasting.base._base_proba import BaseProbaForecaster
from sktime.forecasting.naive import NaiveForecaster


class SPCIForecaster(BaseProbaForecaster):
    r"""Sequential Predictive Conformal Inference (SPCI) Forecaster.

    Implements the SPCI method from Xu & Xie (2023) [1]_, which uses a
    quantile random forest (QRF) to model the conditional distribution of
    one-step-ahead forecast residuals from an arbitrary base forecaster.
    Prediction intervals and probabilistic forecasts are derived by adding
    SPCI-conditioned residual offsets to the base point forecast.

    Parameters
    ----------
    forecaster : sktime BaseForecaster, optional (default=None)
        The base point forecaster. If None, defaults to
        ``NaiveForecaster(strategy="mean")``.
    past_window : int, optional (default=100)
        Lag width ``w`` for the autoregressive feature vector fed to the QRF.
        The QRF is trained on sliding windows of ``past_window`` residuals to
        predict the next residual.
    n_estimators : int, optional (default=10)
        Number of trees in the quantile random forest.
    max_depth : int or None, optional (default=2)
        Maximum depth of each tree in the QRF. ``None`` means unlimited depth.
    n_bins : int, optional (default=5)
        Number of beta grid points for the asymmetric interval search.
        The beta grid has ``n_bins + 1`` values in ``[0, alpha]``.
    initial_window : int, float, or None, optional (default=None)
        Defines the minimum history size before LOO residuals are computed.
        - If ``None``: resolved to ``max(10, floor(0.1 * n))``.
        - If float in ``(0, 1)``: fraction of training length.
        - If int: absolute number of observations.
    n_jobs : int, optional (default=1)
        Number of parallel jobs for LOO residual computation.
        ``-1`` uses all available processors.
    random_state : int, RandomState instance, or None, optional (default=None)
        Seed for the QRF random number generator.

    Attributes
    ----------
    forecaster_ : sktime BaseForecaster
        Fitted clone of the base forecaster.
    loo_residuals_ : np.ndarray of shape (n - n_initial,)
        Leave-one-out one-step-ahead residuals computed during ``fit``.
    qrf_ : RandomForestQuantileRegressor or None
        Fitted quantile random forest. ``None`` if insufficient residuals.
    qrf_quantile_levels_ : np.ndarray of shape (99,)
        Quantile levels used when fitting/predicting with the QRF.

    Notes
    -----
    Multi-step SPCI (Algorithm 3 in [1]_) is **not yet implemented**.
    All forecast horizons use the same single-step QRF, which is a simplification
    that works reasonably well in practice but is not strictly correct for h > 1.

    Fitting cost is O(n²) in the number of training observations because of the
    expanding-window LOO loop.

    ``alpha`` (miscoverage) is NOT an ``__init__`` parameter. It is derived from
    the requested ``coverage`` at prediction time as ``alpha = 1 - coverage``.

    References
    ----------
    .. [1] Chen Xu and Yao Xie. "Sequential Predictive Conformal Inference for
       Time Series". International Conference on Machine Learning (ICML), 2023.
       https://arxiv.org/abs/2212.03463

    Examples
    --------
    >>> from sktime.datasets import load_airline  # doctest: +SKIP
    >>> from sktime.forecasting.spci import SPCIForecaster  # doctest: +SKIP
    >>> y = load_airline()  # doctest: +SKIP
    >>> forecaster = SPCIForecaster()  # doctest: +SKIP
    >>> forecaster.fit(y, fh=[1, 2, 3])  # doctest: +SKIP
    SPCIForecaster(...)
    >>> pred_int = forecaster.predict_interval(coverage=0.9)  # doctest: +SKIP
    >>> pred_proba = forecaster.predict_proba()  # doctest: +SKIP
    """

    _tags = {
        "authors": ["marrov"],
        "python_dependencies": ["sklearn_quantile"],
        "capability:pred_int": True,
        "capability:pred_int:insample": False,
        "capability:multivariate": False,
        "capability:exogenous": False,
        "capability:missing_values": False,
        "requires-fh-in-fit": False,
        "tests:core": True,
    }

    def __init__(
        self,
        forecaster=None,
        past_window=100,
        n_estimators=10,
        max_depth=2,
        n_bins=5,
        initial_window=None,
        n_jobs=1,
        random_state=None,
    ):
        self.forecaster = forecaster
        self.past_window = past_window
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_bins = n_bins
        self.initial_window = initial_window
        self.n_jobs = n_jobs
        self.random_state = random_state

        super().__init__()

        _fc = (
            NaiveForecaster(strategy="mean") if forecaster is None else forecaster
        )
        # Note: capability:exogenous is intentionally NOT cloned here because
        # _compute_loo_residuals does not pass X to inner forecaster clones.
        # SPCI always operates on the univariate residual sequence.
        tags_to_clone = [
            "requires-fh-in-fit",
            "X-y-must-have-same-index",
            "enforce_index_type",
        ]
        self.clone_tags(_fc, tags_to_clone)

    # ------------------------------------------------------------------ #
    #  Core sktime interface                                               #
    # ------------------------------------------------------------------ #

    def _fit(self, y, X, fh):
        """Fit the forecaster.

        Parameters
        ----------
        y : pd.Series
            Target time series.
        X : pd.DataFrame or None
            Exogenous features (not used; kept for API compatibility).
        fh : ForecastingHorizon or None
            Forecasting horizon.

        Returns
        -------
        self : reference to self
        """
        _fc = NaiveForecaster(strategy="mean") if self.forecaster is None else self.forecaster
        self.forecaster_ = clone(_fc)
        self.forecaster_.fit(y=y, X=X, fh=fh)

        # Compute LOO residuals
        self.loo_residuals_ = self._compute_loo_residuals(y, X)

        # Fit QRF if sufficient data
        if len(self.loo_residuals_) > self.past_window + 1:
            self.qrf_ = self._fit_qrf(self.loo_residuals_)
        else:
            self.qrf_ = None
            self.qrf_quantile_levels_ = np.linspace(0.01, 0.99, 99)

        return self

    def _predict(self, fh, X):
        """Compute point predictions.

        Parameters
        ----------
        fh : ForecastingHorizon
            Forecasting horizon.
        X : pd.DataFrame or None
            Exogenous features.

        Returns
        -------
        y_pred : pd.Series
            Point predictions.
        """
        return self.forecaster_.predict(fh=fh, X=X)

    def _predict_interval(self, fh, X, coverage):
        """Compute prediction intervals.

        Parameters
        ----------
        fh : ForecastingHorizon
            Forecasting horizon.
        X : pd.DataFrame or None
            Exogenous features.
        coverage : list of float
            Nominal coverage levels for the intervals.

        Returns
        -------
        pred_int : pd.DataFrame
            DataFrame with MultiIndex columns ``(var_name, coverage, "lower"/"upper")``.
        """
        fh_absolute = fh.to_absolute(self.cutoff)
        fh_absolute_idx = fh_absolute.to_pandas()

        var_name = self._get_varnames()[0]
        y_pred = self.forecaster_.predict(fh=fh, X=X)

        # Build MultiIndex columns
        col_idx = pd.MultiIndex.from_product(
            [[var_name], coverage, ["lower", "upper"]]
        )
        pred_int = pd.DataFrame(index=fh_absolute_idx, columns=col_idx, dtype=float)

        for cov in coverage:
            alpha = 1.0 - cov
            lower_off, upper_off = self._spci_offsets(alpha)

            for fh_ind in fh_absolute_idx:
                yhat = float(y_pred.loc[fh_ind])
                pred_int.loc[fh_ind, (var_name, cov, "lower")] = yhat + lower_off
                pred_int.loc[fh_ind, (var_name, cov, "upper")] = yhat + upper_off

        return pred_int.convert_dtypes()

    def _predict_proba(self, fh, X=None, marginal=True):
        """Compute fully probabilistic forecasts.

        Parameters
        ----------
        fh : ForecastingHorizon
            Forecasting horizon.
        X : pd.DataFrame or None, optional (default=None)
            Exogenous features.
        marginal : bool, optional (default=True)
            Whether the returned distribution is marginal by time point.

        Returns
        -------
        pred_dist : skpro Empirical
            Empirical predictive distribution.
        """
        from skpro.distributions.empirical import Empirical

        n_samples = 500
        rng = np.random.default_rng(self.random_state)

        fh_absolute = fh.to_absolute(self.cutoff)
        fh_absolute_idx = fh_absolute.to_pandas()

        var_name = self._get_varnames()[0]
        y_pred_s = self.forecaster_.predict(fh=fh, X=X)

        resid_samples = self._spci_samples(n_samples=n_samples, rng=rng)

        # Build spl DataFrame with MultiIndex (sample, time)
        _y_time_name = (
            self._y.index.names[-1]
            if hasattr(self._y.index, "names")
            else self._y.index.name
        )
        time_name = _y_time_name or fh_absolute_idx.name or "time"
        n_fh = len(fh_absolute_idx)

        sample_arr = np.repeat(np.arange(n_samples), n_fh)
        time_arr = np.tile(fh_absolute_idx.to_numpy(), n_samples)

        multi_idx = pd.MultiIndex.from_arrays(
            [sample_arr, time_arr],
            names=["sample", time_name],
        )

        # For each fh index, add the point forecast to each residual sample
        yhat_arr = np.array([float(y_pred_s.loc[t]) for t in fh_absolute_idx])
        # resid_samples: (n_samples,), yhat_arr: (n_fh,)
        # values: (n_samples * n_fh,) — tile yhat, repeat resid
        values = np.tile(yhat_arr, n_samples) + np.repeat(resid_samples, n_fh)

        spl = pd.DataFrame(values, index=multi_idx, columns=[var_name])

        return Empirical(
            spl=spl,
            index=fh_absolute_idx,
            columns=pd.Index([var_name]),
        )

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _parse_initial_window(self, n):
        """Resolve ``initial_window`` to an integer.

        Parameters
        ----------
        n : int
            Length of the training series.

        Returns
        -------
        n_initial : int
            Resolved initial window size.
        """
        iw = self.initial_window
        if iw is None:
            return max(10, floor(0.1 * n))
        if isinstance(iw, float):
            return max(1, floor(iw * n))
        return int(iw)

    def _compute_loo_residuals(self, y, X):
        """Compute expanding-window LOO one-step-ahead residuals.

        Parameters
        ----------
        y : pd.Series
            Training target series.
        X : pd.DataFrame or None
            Exogenous features (currently not passed to inner clones).

        Returns
        -------
        residuals : np.ndarray of shape (n - n_initial,)
            Array of signed residuals ``y[i] - yhat[i]``.
        """
        from joblib import Parallel, delayed

        n = len(y)
        n_initial = self._parse_initial_window(n)

        _fc = (
            NaiveForecaster(strategy="mean") if self.forecaster is None else self.forecaster
        )

        def _fit_predict_one(i):
            fc_clone = clone(_fc)
            fc_clone.fit(y=y.iloc[:i], fh=[1])
            y_pred = fc_clone.predict()
            return float(y.iloc[i]) - float(y_pred.iloc[0])

        residuals = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_predict_one)(i) for i in range(n_initial, n)
        )
        return np.array(residuals, dtype=float)

    def _fit_qrf(self, residuals):
        """Fit a quantile random forest on the LOO residual sequence.

        Parameters
        ----------
        residuals : np.ndarray
            Array of LOO residuals from ``_compute_loo_residuals``.

        Returns
        -------
        qrf : RandomForestQuantileRegressor
            Fitted QRF estimator.
        """
        from numpy.lib.stride_tricks import sliding_window_view

        from sklearn_quantile import RandomForestQuantileRegressor

        w = self.past_window
        n = len(residuals)

        X_resid = sliding_window_view(residuals[:-1], window_shape=w)  # (n-w, w)
        y_resid = residuals[w:]  # (n-w,)

        # Align lengths
        min_len = min(len(X_resid), len(y_resid))
        X_resid = X_resid[:min_len]
        y_resid = y_resid[:min_len]

        q_levels = np.linspace(0.01, 0.99, 99)
        self.qrf_quantile_levels_ = q_levels

        qrf = RandomForestQuantileRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            q=q_levels,
        )
        qrf.fit(X_resid, y_resid)
        return qrf

    def _qrf_predict_quantiles(self):
        """Predict quantiles of the next residual using the fitted QRF.

        Returns
        -------
        q_vals : np.ndarray of shape (99,) or None
            Predicted quantile values, or ``None`` if QRF is unavailable.
        """
        if self.qrf_ is None or len(self.loo_residuals_) < self.past_window:
            return None

        X_pred = self.loo_residuals_[-self.past_window :].reshape(1, -1)
        q_vals = self.qrf_.predict(X_pred)

        if q_vals.ndim == 2:
            q_vals = q_vals.flatten()

        return q_vals

    def _spci_offsets(self, alpha):
        """Return SPCI lower/upper residual offsets for a given miscoverage level.

        Parameters
        ----------
        alpha : float
            Miscoverage level (``1 - coverage``).

        Returns
        -------
        lower_off : float
            Residual offset for the lower interval end.
        upper_off : float
            Residual offset for the upper interval end.
        """
        predicted_q = self._qrf_predict_quantiles()

        if predicted_q is not None:
            return self._qrf_asymmetric_interval(predicted_q, alpha)
        return self._empirical_asymmetric_interval(self.loo_residuals_, alpha)

    def _qrf_asymmetric_interval(self, predicted_quantiles, alpha):
        """Find the shortest asymmetric interval from QRF-predicted quantiles.

        Parameters
        ----------
        predicted_quantiles : np.ndarray of shape (99,)
            Quantile predictions from the QRF at levels ``qrf_quantile_levels_``.
        alpha : float
            Miscoverage level.

        Returns
        -------
        best_lower : float
            Lower offset achieving the shortest interval.
        best_upper : float
            Upper offset achieving the shortest interval.
        """
        q_levels = self.qrf_quantile_levels_
        beta_grid = np.linspace(0, alpha, self.n_bins + 1)

        best_lower = np.interp(0.0, q_levels, predicted_quantiles)
        best_upper = np.interp(1.0 - alpha, q_levels, predicted_quantiles)
        best_width = best_upper - best_lower

        for beta in beta_grid:
            lower = np.interp(beta, q_levels, predicted_quantiles)
            upper = np.interp(1.0 - alpha + beta, q_levels, predicted_quantiles)
            width = upper - lower
            if width < best_width:
                best_width = width
                best_lower = lower
                best_upper = upper

        return float(best_lower), float(best_upper)

    def _empirical_asymmetric_interval(self, residuals, alpha):
        """Find the shortest asymmetric interval from empirical residual quantiles.

        Parameters
        ----------
        residuals : np.ndarray
            LOO residual array.
        alpha : float
            Miscoverage level.

        Returns
        -------
        best_lower : float
            Lower offset achieving the shortest interval.
        best_upper : float
            Upper offset achieving the shortest interval.
        """
        if len(residuals) == 0:
            return 0.0, 0.0

        resid_window = residuals[-self.past_window :]
        beta_grid = np.linspace(0, alpha, self.n_bins + 1)

        best_lower = float(np.percentile(resid_window, 0.0))
        best_upper = float(np.percentile(resid_window, 100.0 * (1.0 - alpha)))
        best_width = best_upper - best_lower

        for beta in beta_grid:
            lower = float(np.percentile(resid_window, 100.0 * beta))
            upper = float(np.percentile(resid_window, 100.0 * (1.0 - alpha + beta)))
            width = upper - lower
            if width < best_width:
                best_width = width
                best_lower = lower
                best_upper = upper

        return best_lower, best_upper

    def _spci_samples(self, n_samples, rng):
        """Draw residual samples from the SPCI-conditioned distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples to draw.
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        samples : np.ndarray of shape (n_samples,)
            Drawn residual values.
        """
        predicted_q = self._qrf_predict_quantiles()

        if predicted_q is not None:
            q_levels = self.qrf_quantile_levels_
            u = rng.uniform(0.0, 1.0, n_samples)
            u_clipped = np.clip(u, q_levels[0], q_levels[-1])
            return np.interp(u_clipped, q_levels, predicted_q)

        # Fallback: bootstrap from most recent residuals
        resid_window = self.loo_residuals_[-self.past_window :]
        if len(resid_window) == 0:
            return np.zeros(n_samples)
        return rng.choice(resid_window, size=n_samples, replace=True)

    # ------------------------------------------------------------------ #
    #  Test parameters                                                     #
    # ------------------------------------------------------------------ #

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : list of dict
            Parameters to create testing instances of the class.
        """
        params1 = {
            "forecaster": NaiveForecaster(),
            "past_window": 10,
            "n_estimators": 5,
        }
        params2 = {
            "forecaster": NaiveForecaster(strategy="mean"),
            "past_window": 5,
            "n_estimators": 3,
            "n_bins": 3,
            "initial_window": 5,
        }
        return [params1, params2]
