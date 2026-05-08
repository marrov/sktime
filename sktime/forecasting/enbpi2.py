# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements EnbPIForecaster v2 — no tsbootstrap dependency."""

__all__ = ["EnbPIForecaster"]
__author__ = ["benheid", "marrov"]

import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.utils import check_random_state

from sktime.forecasting.base._base_proba import BaseProbaForecaster
from sktime.forecasting.naive import NaiveForecaster
from sktime.libs._aws_fortuna_enbpi.enbpi import EnbPI
from sktime.transformations.bootstrap import MovingBlockBootstrapTransformer
from sktime.utils.parallel import parallelize


def _fit_one_bootstrap(bs_key, meta):
    """Fit one bootstrap forecaster and return ``(forecaster, in_sample_preds)``.

    Module-level function so it is picklable by joblib/loky when ``n_jobs != 1``.

    Parameters
    ----------
    bs_key : str
        Bootstrap sample key (e.g. ``"synthetic_0"``).
    meta : dict
        Context dict with keys: ``forecaster_``, ``bootstrapped_ts``,
        ``y_index``, ``fh``, ``X``.

    Returns
    -------
    fc_test : fitted forecaster
        Forecaster fitted with the real ``fh`` for test-time predictions.
    in_sample_pred : pd.DataFrame
        In-sample predictions on the bootstrap series (aligned to ``y_index``).
    """
    from sklearn.base import clone

    forecaster_ = meta["forecaster_"]
    bootstrapped_ts = meta["bootstrapped_ts"]
    y_index = meta["y_index"]
    fh = meta["fh"]
    X = meta["X"]

    bs_ts = bootstrapped_ts.loc[bs_key]
    if len(bs_ts) != len(y_index):
        raise ValueError(
            f"Bootstrap sample length {len(bs_ts)} != training length {len(y_index)}. "
            "Ensure the bootstrap transformer produces series of the same length."
        )
    bs_df = pd.DataFrame(bs_ts.values, index=y_index, columns=bs_ts.columns)

    # Fit the test forecaster with the actual fh
    fc_test = (
        forecaster_.clone() if hasattr(forecaster_, "clone") else clone(forecaster_)
    )
    fc_test.fit(y=bs_df, fh=fh, X=X)

    # Separately get in-sample predictions using a fresh clone
    # This avoids fh-in-fit incompatibility issues (PR #8221 fix)
    fc_train = (
        forecaster_.clone() if hasattr(forecaster_, "clone") else clone(forecaster_)
    )
    try:
        in_sample_pred = fc_train.fit_predict(y=bs_df, fh=y_index, X=X)
    except Exception:
        # Fallback: fit with y_index as fh, then predict
        fc_train.fit(y=bs_df, fh=y_index, X=X)
        in_sample_pred = fc_train.predict(fh=y_index, X=X)

    return fc_test, in_sample_pred


class EnbPIForecaster(BaseProbaForecaster):
    """Ensemble Bootstrap Prediction Interval Forecaster (v2, no tsbootstrap).

    The forecaster combines sktime forecasters with sktime bootstrap transformers
    and the EnbPI algorithm [1] to produce prediction intervals and probabilistic
    forecasts.

    Unlike ``enbpi.EnbPIForecaster``, this implementation:

    * Does **not** depend on ``tsbootstrap`` — uses native sktime bootstrap
      transformers such as ``MovingBlockBootstrapTransformer``.
    * Inherits from ``BaseProbaForecaster`` and implements ``_predict_proba``,
      returning a full ``skpro.Empirical`` distribution.
    * Parallelises the bootstrap fit loop via :func:`sktime.utils.parallel.parallelize`.
    * Optionally checks stationarity of the OOB train residuals after fitting and
      warns if they are non-stationary.

    Parameters
    ----------
    forecaster : sktime BaseForecaster, optional (default=None)
        The base forecaster.  Defaults to ``NaiveForecaster()``.
    bootstrap_transformer : sktime transformer, optional (default=None)
        Bootstrap transformer with ``capability:bootstrap_index`` tag and
        ``return_indices=True``.  Defaults to
        ``MovingBlockBootstrapTransformer(return_indices=True)``.
    random_state : int, RandomState or None, optional (default=None)
        Random state for reproducibility.
    aggregation_function : str, optional (default="mean")
        How to aggregate bootstrap predictions.  One of ``"mean"`` or ``"median"``.
    n_jobs : int, optional (default=1)
        Number of parallel jobs for the bootstrap fit loop.  ``-1`` uses all
        available processors.
    stationarity_estimator : sktime param estimator or None, optional (default=None)
        Estimator used to check stationarity of OOB residuals after fit.
        Defaults to ``StationarityKPSS()``.  Set to ``False`` to disable the check.

    Attributes
    ----------
    forecasters : list of fitted forecasters
        One fitted forecaster per bootstrap sample.
    indexes : np.ndarray of shape (n_bootstraps, n_train_times)
        Bootstrap indices for each sample.

    References
    ----------
    .. [1] Chen Xu & Yao Xie (2021). Conformal Prediction Interval for Dynamic
       Time-Series. ICML.

    Examples
    --------
    >>> from sktime.forecasting.enbpi2 import EnbPIForecaster
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.base import ForecastingHorizon
    >>> import numpy as np
    >>> y = load_airline()
    >>> forecaster = EnbPIForecaster(forecaster=NaiveForecaster(sp=12))
    >>> fh = ForecastingHorizon(np.arange(1, 13))
    >>> forecaster.fit(y, fh=fh)  # doctest: +SKIP
    EnbPIForecaster(...)
    >>> pred_int = forecaster.predict_interval(coverage=[0.9])  # doctest: +SKIP
    >>> pred_proba = forecaster.predict_proba()  # doctest: +SKIP
    """

    _tags = {
        "authors": ["benheid", "marrov"],
        "capability:multivariate": False,
        "capability:exogenous": True,
        "capability:missing_values": False,
        "y_inner_mtype": "pd.DataFrame",
        "X_inner_mtype": "pd.DataFrame",
        "X-y-must-have-same-index": True,
        "requires-fh-in-fit": True,
        "enforce_index_type": None,
        "capability:insample": False,
        "capability:pred_int": True,
        "capability:pred_int:insample": False,
    }

    def __init__(
        self,
        forecaster=None,
        bootstrap_transformer=None,
        random_state=None,
        aggregation_function="mean",
        n_jobs=1,
        stationarity_estimator=None,
        n_samples=500,
    ):
        self.forecaster = forecaster
        self.bootstrap_transformer = bootstrap_transformer
        self.random_state = random_state
        self.aggregation_function = aggregation_function
        self.n_jobs = n_jobs
        self.stationarity_estimator = stationarity_estimator
        self.n_samples = n_samples

        # Validate before super().__init__()
        if aggregation_function not in ("mean", "median"):
            raise ValueError(
                f"aggregation_function '{aggregation_function}' is not supported. "
                "Please choose either 'mean' or 'median'."
            )

        super().__init__()

        # Assign _aggregation_function after super().__init__()
        if aggregation_function == "mean":
            self._aggregation_function = np.mean
        else:
            self._aggregation_function = np.median

        # Internal clones — set up after super().__init__()
        _fc = forecaster if forecaster is not None else NaiveForecaster()
        self.forecaster_ = (
            _fc.clone() if hasattr(_fc, "clone") else clone(_fc)
        )

        if bootstrap_transformer is not None:
            self.bootstrap_transformer_ = (
                bootstrap_transformer.clone()
                if hasattr(bootstrap_transformer, "clone")
                else clone(bootstrap_transformer)
            )
        else:
            self.bootstrap_transformer_ = MovingBlockBootstrapTransformer(
                return_indices=True, return_actual=False
            )

        # Ensure return_indices is set
        if not getattr(self.bootstrap_transformer_, "return_indices", False):
            self.bootstrap_transformer_.return_indices = True

        bs_capable = self.bootstrap_transformer_.get_tag(
            "capability:bootstrap_index", False, raise_error=False
        )
        if not bs_capable:
            raise ValueError(
                "The bootstrap_transformer must have the tag "
                "'capability:bootstrap_index'. "
                f"Got: {type(self.bootstrap_transformer_).__name__}"
            )

    # ------------------------------------------------------------------ #
    #  Core sktime interface                                               #
    # ------------------------------------------------------------------ #

    def _fit(self, y, X, fh):
        """Fit bootstrapped forecasters.

        Parameters
        ----------
        y : pd.DataFrame
            Target time series.
        X : pd.DataFrame or None
            Exogenous features.
        fh : ForecastingHorizon
            Forecasting horizon.

        Returns
        -------
        self : reference to self
        """
        self._fh = fh
        self._y_index = y.index
        self.random_state_ = check_random_state(self.random_state)

        # Bootstrap samples
        bs_ts_index = self.bootstrap_transformer_.fit_transform(y)
        bootstrapped_ts = bs_ts_index[y.columns]

        # Filter out "actual" series — only keep synthetic bootstrap samples
        all_keys = bootstrapped_ts.index.get_level_values(0).unique()
        bs_keys = [k for k in all_keys if k != "actual"]

        # Extract resampled indices only for the synthetic bootstrap keys
        synthetic_mask = bs_ts_index.index.get_level_values(0).isin(bs_keys)
        self.indexes = (
            bs_ts_index.loc[synthetic_mask, "resampled_index"]
            .values
            .reshape((len(bs_keys), len(y)))
        )
        bootstrapped_ts = bootstrapped_ts[synthetic_mask]

        n_bootstraps = len(bs_keys)

        # Build per-bootstrap meta for parallelize
        meta = {
            "forecaster_": self.forecaster_,
            "bootstrapped_ts": bootstrapped_ts,
            "y_index": y.index,
            "fh": fh,
            "X": X,
        }

        results = parallelize(
            fun=_fit_one_bootstrap,
            iter=bs_keys,
            meta=meta,
            backend="loky" if self.n_jobs != 1 else None,
            backend_params={"n_jobs": self.n_jobs} if self.n_jobs != 1 else None,
        )

        self.forecasters = [r[0] for r in results]
        self._preds = [r[1].values.squeeze() for r in results]

        # Compute and store OOB signed residuals for _predict_proba
        self._train_residuals_, self._oob_signed_residuals_ = (
            self._compute_oob_residuals(y)
        )

        # Optional stationarity check
        self._check_stationarity(self._train_residuals_)

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
        y_pred : pd.DataFrame
            Point predictions.
        """
        preds = [fc.predict(fh=fh, X=X) for fc in self.forecasters]
        return pd.DataFrame(
            self._aggregation_function(np.stack([p.values.squeeze() for p in preds], axis=0), axis=0).reshape(-1, 1),
            index=list(fh.to_absolute(self.cutoff)),
            columns=self._get_varnames(),
        )

    def _predict_interval(self, fh, X, coverage):
        """Compute prediction intervals via EnbPI.

        Parameters
        ----------
        fh : ForecastingHorizon
            Forecasting horizon.
        X : pd.DataFrame or None
            Exogenous features.
        coverage : list of float
            Nominal coverage levels.

        Returns
        -------
        pred_int : pd.DataFrame
            Prediction intervals with MultiIndex columns.
        """
        preds = [fc.predict(fh=fh, X=X).values.squeeze() for fc in self.forecasters]

        train_targets = self._y.copy()
        train_targets.index = pd.RangeIndex(len(train_targets))

        intervals = []
        for cov in coverage:
            conformal_intervals, train_residuals = EnbPI(
                self.aggregation_function
            ).conformal_interval(
                bootstrap_indices=self.indexes,
                bootstrap_train_preds=np.stack(self._preds),
                bootstrap_test_preds=np.stack(preds),
                train_targets=train_targets.values,
                error=1 - cov,
                return_residuals=True,
            )
            intervals.append(conformal_intervals.reshape(-1, 2))

        # Check stationarity of residuals (warn only; already done in fit but
        # residuals here are coverage-specific — keep it lightweight, skip re-check)

        cols = self._get_columns(method="predict_interval", coverage=coverage)
        fh_absolute_idx = fh.to_absolute_index(self.cutoff)
        pred_int = pd.DataFrame(
            np.concatenate(intervals, axis=1), index=fh_absolute_idx, columns=cols
        )
        return pred_int

    def _predict_proba(self, fh, X=None, marginal=True):
        """Compute fully probabilistic forecasts as an skpro Empirical distribution.

        Parameters
        ----------
        fh : ForecastingHorizon
            Forecasting horizon.
        X : pd.DataFrame or None, optional (default=None)
            Exogenous features.
        marginal : bool, optional (default=True)
            Whether to return marginal distributions.

        Returns
        -------
        pred_dist : skpro Empirical
            Empirical predictive distribution.
        """
        from skpro.distributions.empirical import Empirical

        n_samples = self.n_samples
        rng = np.random.default_rng(self.random_state)

        fh_absolute = fh.to_absolute(self.cutoff)
        fh_absolute_idx = fh_absolute.to_pandas()
        var_name = self._get_varnames()[0]

        # Point forecast: aggregate of bootstrap test predictions
        preds_test = [fc.predict(fh=fh, X=X) for fc in self.forecasters]
        y_pred_arr = self._aggregation_function(
            np.stack([p.values.flatten() for p in preds_test], axis=0), axis=0
        )  # shape (n_fh,)

        # Signed OOB residuals stored from _fit
        signed_residuals = self._oob_signed_residuals_  # shape (n_train,)

        if len(signed_residuals) == 0:
            signed_residuals = np.zeros(1)

        # Build spl DataFrame: MultiIndex (sample, time)
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

        # Sample independently for each (sample, horizon) combination
        sampled_residuals = rng.choice(signed_residuals, size=(n_samples, n_fh), replace=True)
        # y_pred_arr shape: (n_fh,); sampled_residuals shape: (n_samples, n_fh)
        values_matrix = y_pred_arr[np.newaxis, :] + sampled_residuals  # (n_samples, n_fh)
        values = values_matrix.flatten()  # row-major: [s0h1, s0h2, ..., s1h1, ...]

        spl = pd.DataFrame(values, index=multi_idx, columns=[var_name])

        return Empirical(
            spl=spl,
            index=fh_absolute_idx,
            columns=pd.Index([var_name]),
        )

    def _update(self, y, X=None, update_params=True):
        """Update by refitting on all available data.

        Parameters
        ----------
        y : pd.Series or pd.DataFrame
            Target time series.
        X : pd.DataFrame, optional (default=None)
            Exogenous features.
        update_params : bool, optional (default=True)
            Whether to update model parameters.

        Returns
        -------
        self : reference to self
        """
        self.fit(y=self._y, X=self._X, fh=self._fh)
        return self

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _compute_oob_residuals(self, y):
        """Compute OOB residuals from bootstrap indices and in-sample predictions.

        For each training time step t, the OOB bootstraps are those where t was
        NOT included in the bootstrap.  The signed residual is
        ``y_t - aggregate(oob_bootstrap_preds_t)``.

        Parameters
        ----------
        y : pd.DataFrame
            Training target series (inner mtype).

        Returns
        -------
        abs_residuals : np.ndarray of shape (n_train,)
            Absolute OOB residuals (used by EnbPI).
        signed_residuals : np.ndarray of shape (n_train_valid,)
            Signed OOB residuals (used by _predict_proba).
        """
        n_bootstraps, n_train_times = self.indexes.shape

        in_bootstrap_indices = np.zeros((n_bootstraps, n_train_times), dtype=bool)
        np.put_along_axis(in_bootstrap_indices, self.indexes, values=1, axis=1)

        # Stack in-sample predictions: (n_bootstraps, n_train) — already squeezed in _fit
        preds_stack = np.stack(self._preds, axis=0)

        train_targets = y.values.squeeze()  # shape: (n_train,)

        abs_residuals = np.zeros(n_train_times)
        signed_residuals_list = []

        for t in range(n_train_times):
            oob_mask = ~in_bootstrap_indices[:, t]
            which_oob = np.where(oob_mask)[0]
            if len(which_oob) > 0:
                oob_pred_t = self._aggregation_function(preds_stack[which_oob, t])  # scalar
                residual = float(train_targets[t]) - float(oob_pred_t)
                abs_residuals[t] = abs(residual)
                signed_residuals_list.append(residual)
            else:
                abs_residuals[t] = abs(float(train_targets[t]))

        signed_residuals = np.array(signed_residuals_list, dtype=float)
        return abs_residuals, signed_residuals

    def _check_stationarity(self, residuals):
        """Check stationarity of train residuals and warn if non-stationary.

        Parameters
        ----------
        residuals : np.ndarray
            Absolute OOB train residuals.
        """
        if self.stationarity_estimator is False:
            return

        try:
            from sktime.param_est.stationarity import StationarityKPSS

            est_cls = StationarityKPSS
            if self.stationarity_estimator is not None:
                stat_est = (
                    self.stationarity_estimator.clone()
                    if hasattr(self.stationarity_estimator, "clone")
                    else clone(self.stationarity_estimator)
                )
            else:
                stat_est = est_cls()

            # StationarityKPSS requires univariate Series
            resid_series = pd.Series(residuals, name="residuals")
            stat_est.fit(resid_series)
            is_stationary = stat_est.stationary_

            if not is_stationary:
                warnings.warn(
                    "EnbPIForecaster: OOB train residuals appear to be "
                    "non-stationary according to the KPSS test. "
                    "Prediction intervals may not be well-calibrated.",
                    UserWarning,
                    stacklevel=2,
                )
        except Exception:
            # Stationarity check is best-effort; never fail fit because of it
            pass

    # ------------------------------------------------------------------ #
    #  Test parameters                                                     #
    # ------------------------------------------------------------------ #

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return.

        Returns
        -------
        params : list of dict
            Parameters to create testing instances of the class.
        """
        params = [
            {
                "bootstrap_transformer": MovingBlockBootstrapTransformer(
                    n_series=5, return_indices=True
                ),
            },
            {
                "forecaster": NaiveForecaster(),
                "bootstrap_transformer": MovingBlockBootstrapTransformer(
                    n_series=3, return_indices=True
                ),
                "aggregation_function": "median",
            },
        ]
        return params
