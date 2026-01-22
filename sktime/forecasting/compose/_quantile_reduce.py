import pandas as pd

from ._reduce import RecursiveReductionForecaster


class QuantileRecursiveReductionForecaster(RecursiveReductionForecaster):
    """Recursive reduction forecaster with quantile prediction capability.

    This class extends RecursiveReductionForecaster to support quantile predictions
    by using an underlying estimator (e.g., LGBMRegressor) with a quantile objective.
    The _predict_quantiles method calls _predict and reformats the output to match
    sktime's expected quantile output format with a MultiIndex column structure.

    Parameters
    ----------
    estimator : sklearn regressor with quantile objective
        Tabular regression algorithm with quantile objective (e.g., LGBMRegressor
        with objective='quantile').
    quantile : float
        The quantile value (alpha) that the estimator is trained to predict.
    window_length : int, optional, default=10
        Window length used in the reduction algorithm.
    impute_method : str, None, or sktime transformation, optional
        Imputation method to use for missing values in the lagged data.
    pooling : str, one of ["local", "global", "panel"], optional, default="local"
        Level on which data are pooled to fit the supervised regression model.
    """
    _tags = {
        "capability:pred_int": True,
        "capability:missing_values": True,
    }
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
        self.set_tags(**self._tags)

    def _predict_quantiles(self, fh, X, alpha):
        """Compute/return prediction quantiles for a forecast.

        This method uses the underlying estimator's point predictions (which are
        quantile predictions when the estimator has a quantile objective) and
        reformats them into sktime's expected quantile DataFrame format.

        Parameters
        ----------
        fh : ForecastingHorizon
            The forecasting horizon with the steps ahead to predict.
        X : pd.DataFrame, optional (default=None)
            Exogeneous time series for the forecast.
        alpha : list of float
            A list of probabilities at which quantile forecasts are computed.
            Note: This forecaster only supports the quantile it was trained with.

        Returns
        -------
        quantiles : pd.DataFrame
            Column has multi-index: first level is variable name from y in fit,
            second level being the values of alpha passed to the function.
            Row index is fh, with additional levels for hierarchical data.
        """
        # Get point predictions from the underlying quantile estimator
        y_pred = self._predict(fh=fh, X=X)

        # Get the variable names from the fitted y
        var_names = self._y.columns.tolist()

        # Create multi-index columns: (variable_name, alpha_value)
        # For each alpha requested, we return the same prediction since
        # the underlying model is trained for a specific quantile
        col_index = pd.MultiIndex.from_product([var_names, alpha])

        # Replicate predictions for each alpha value requested
        pred_values = pd.concat([y_pred] * len(alpha), axis=1)
        pred_values.columns = col_index

        return pred_values