# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements pipelines for probabilistic forecasting."""

import pandas as pd

from sktime.forecasting.compose import TransformedTargetForecaster


class TransformedTargetProbaForecaster(TransformedTargetForecaster):
    """Meta-estimator for probabilistic forecasting with transformed time series.

    This is a specialized version of TransformedTargetForecaster that properly
    handles probabilistic forecasts (predict_proba) when pre-transformers are present.
    It applies inverse transformation to each sample trajectory in an Empirical
    distribution, maintaining the probabilistic nature of the forecast.

    Parameters
    ----------
    steps : list of sktime transformers and forecasters, or
        list of tuples (str, estimator) of sktime transformers or forecasters.
        The list must contain exactly one forecaster.
        These are "blueprint" transformers resp forecasters,
        forecaster/transformer states do not change when fit is called.

    Attributes
    ----------
    steps_ : list of tuples (str, estimator) of sktime transformers or forecasters
        clones of estimators in steps which are fitted in the pipeline
        is always in (str, estimator) format, even if steps is just a list
        strings not passed in steps are replaced by unique generated strings
        i-th transformer in steps_ is clone of i-th in steps
    forecaster_ : estimator, reference to the unique forecaster in steps_
    transformers_pre_ : list of tuples (str, transformer) of sktime transformers
        reference to pairs in steps_ that precede forecaster_
    transformers_post_ : list of tuples (str, transformer) of sktime transformers
        reference to pairs in steps_ that succeed forecaster_

    Examples
    --------
    >>> from sklearn.linear_model import LinearRegression
    >>> from skpro.regression.residual import ResidualDouble
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.compose._reduce_proba import (
    ...     MCRecursiveProbaReductionForecaster,
    ... )
    >>> from sktime.transformations.series.difference import Differencer
    >>> from server.forecasting import TransformedTargetProbaForecaster
    >>>
    >>> y = load_airline()
    >>>
    >>> forecaster = TransformedTargetProbaForecaster(
    ...     [
    ...         Differencer(),
    ...         MCRecursiveProbaReductionForecaster(
    ...             estimator=ResidualDouble(LinearRegression()),
    ...             window_length=3,
    ...         ),
    ...     ]
    ... )
    >>> forecaster.fit(y)  # doctest: +SKIP
    >>> y_pred_dist = forecaster.predict_proba(fh=[1, 2, 3])  # doctest: +SKIP

    Notes
    -----
    This class is designed to work with forecasters that return Empirical
    distributions (like MCRecursiveProbaReductionForecaster). When transformers
    are present, it applies the inverse transformation to each sample trajectory,
    preserving the probabilistic structure.

    The implementation follows the fail-early principle: it will raise a clear
    NotImplementedError if the inner forecaster returns a non-Empirical
    distribution when transformers are present.

    Limitations
    -----------
    Only Empirical distributions are supported when pre-transformers or
    post-transformers are present. Parametric distributions (e.g., Normal,
    Poisson) would require distribution-specific transformation logic for
    their parameters, which is not implemented.
    """

    _tags = {
        "authors": ["marrov"],
        "python_dependencies": ["skpro>=2.11.0"],
    }

    def _predict_proba(self, fh, X, marginal=True):
        """Compute/return fully probabilistic forecasts.

        Private _predict_proba containing the core logic, called from predict_proba.

        Delegates to the inner forecaster's predict_proba and applies inverse
        transformation to the resulting distribution. For Empirical distributions,
        the inverse transform is applied to each sample trajectory.

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon
            The forecasting horizon with the steps ahead to to predict.
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series to predict from.
        marginal : bool, optional (default=True)
            whether returned distribution is marginal by time index

        Returns
        -------
        pred_dist : sktime BaseDistribution
            predictive distribution
            if marginal=True, will be marginal distribution by time point
            if marginal=False and implemented by method, will be joint

        Raises
        ------
        NotImplementedError
            If the inner forecaster returns a non-Empirical distribution
            when pre-transformers are present.
        """
        from skpro.distributions import Empirical

        # Get probabilistic prediction from inner forecaster
        pred_dist = self.forecaster_.predict_proba(fh=fh, X=X, marginal=marginal)

        has_pre_transformers = len(self.transformers_pre_) > 0
        has_post_transformers = len(self.transformers_post_) > 0

        # If no transformers at all, return as-is
        if not has_pre_transformers and not has_post_transformers:
            return pred_dist

        # For Empirical distributions, apply transforms to each sample
        if not isinstance(pred_dist, Empirical):
            raise NotImplementedError(
                "TransformedTargetProbaForecaster._predict_proba only supports "
                "Empirical distributions from the inner forecaster when transformers "
                f"are present. Received distribution of type "
                f"{type(pred_dist).__name__}. Parametric distributions would require "
                "distribution-specific transformation logic for their parameters."
            )

        spl = pred_dist.spl  # DataFrame with samples
        sample_level_name = spl.index.names[0]

        # Group by sample and apply transforms to each sample trajectory
        transformed_samples = []
        for sample_idx in spl.index.get_level_values(0).unique():
            sample_data = spl.xs(sample_idx, level=0)

            # Apply inverse transform for pre-transformers
            if has_pre_transformers:
                sample_transformed = self._get_inverse_transform(
                    self.transformers_pre_, sample_data, X
                )
            else:
                sample_transformed = sample_data

            # Apply post-transformers (forward transform)
            for _, t in self.transformers_post_:
                sample_transformed = t.transform(X=sample_transformed, y=X)

            # Restore sample index level
            new_index_tuples = [
                (sample_idx,) + (idx if isinstance(idx, tuple) else (idx,))
                for idx in sample_transformed.index
            ]
            new_index = pd.MultiIndex.from_tuples(
                new_index_tuples,
                names=[sample_level_name] + list(sample_transformed.index.names),
            )
            sample_transformed.index = new_index
            transformed_samples.append(sample_transformed)

        # Concatenate all transformed samples
        transformed_spl = pd.concat(transformed_samples, axis=0)

        return Empirical(
            spl=transformed_spl,
            index=pred_dist.index,
            columns=pred_dist.columns,
        )
