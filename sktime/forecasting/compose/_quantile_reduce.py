import pandas as pd

from ._reduce import DirectReductionForecaster, RecursiveReductionForecaster


class _QuantilePredictionMixin:
    """Mixin providing quantile prediction capability for reduction forecasters."""

    _quantile_tags = {
        "capability:pred_int": True,
        "capability:missing_values": True,
    }

    def _predict_quantiles(self, fh, X, alpha):
        """Compute quantile forecasts using underlying quantile estimator.

        Parameters
        ----------
        fh : ForecastingHorizon
            Forecasting horizon with steps ahead to predict.
        X : pd.DataFrame, optional
            Exogeneous time series.
        alpha : list of float
            Quantile probabilities.

        Returns
        -------
        pd.DataFrame
            Quantile predictions with MultiIndex columns (variable_name, alpha).
        """
        y_pred = self._predict(fh=fh, X=X)
        var_names = self._y.columns.tolist()
        col_index = pd.MultiIndex.from_product([var_names, alpha])
        pred_values = pd.concat([y_pred] * len(alpha), axis=1)
        pred_values.columns = col_index
        return pred_values


class QuantileRecursiveReductionForecaster(
    _QuantilePredictionMixin, RecursiveReductionForecaster
):
    """Recursive reduction forecaster with quantile prediction.

    Parameters
    ----------
    estimator : sklearn regressor
        Regressor with quantile objective (e.g., LGBMRegressor with objective='quantile').
    quantile : float
        Quantile value (alpha) for predictions.
    window_length : int, default=10
        Window length for reduction algorithm.
    impute_method : str, None, or sktime transformation, optional
        Imputation method for missing values.
    pooling : str, default="local"
        Pooling level: "local", "global", or "panel".
    """

    def __init__(
        self,
        estimator,
        quantile: float,
        window_length: int = 10,
        impute_method: str | None = "bfill",
        pooling: str = "local",
    ):
        self.quantile = quantile
        super().__init__(
            estimator=estimator,
            window_length=window_length,
            impute_method=impute_method,
            pooling=pooling,
        )
        self.set_tags(**self._quantile_tags)


class QuantileDirectReductionForecaster(
    _QuantilePredictionMixin, DirectReductionForecaster
):
    """Direct reduction forecaster with quantile prediction.

    Parameters
    ----------
    estimator : sklearn regressor
        Regressor with quantile objective (e.g., LGBMRegressor with objective='quantile').
    quantile : float
        Quantile value (alpha) for predictions.
    window_length : int, default=10
        Window length for reduction algorithm.
    transformers : optional
        Currently not used.
    X_treatment : str, default="concurrent"
        Determines X timestamps: "concurrent" or "shifted".
    impute_method : str, None, or sktime transformation, optional
        Imputation method for missing values.
    pooling : str, default="local"
        Pooling level: "local", "global", or "panel".
    windows_identical : bool, default=False
        Whether all direct models use same number of observations.
    """

    def __init__(
        self,
        estimator,
        quantile: float,
        window_length: int = 10,
        transformers=None,
        X_treatment: str = "concurrent",
        impute_method: str | None = "bfill",
        pooling: str = "local",
        windows_identical: bool = False,
    ):
        self.quantile = quantile
        super().__init__(
            estimator=estimator,
            window_length=window_length,
            transformers=transformers,
            X_treatment=X_treatment,
            impute_method=impute_method,
            pooling=pooling,
            windows_identical=windows_identical,
        )
        self.set_tags(**self._quantile_tags)