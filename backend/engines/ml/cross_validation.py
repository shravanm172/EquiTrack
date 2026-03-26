from __future__ import annotations

from typing import Callable
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures

from engines.ml.split_dataset import rolling_year_splits
from engines.ml.feature_scaling import scale_fold_features

def run_rolling_cv(X: pd.DataFrame, 
                   y: pd.Series, 
                   model_factory: Callable[[], object], 
                   start_date:str, end_date:str, 
                   train_years: int = 4, 
                   validation_years: int = 1,  
                   baseline_column: str | None = None,
                   return_predictions: bool = False) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run rolling window cross-validation
    Returns:
    pd.DataFrame
        One row per fold with train/validation metrics
    """
    results = []
    prediction_frames = []
    for fold, X_train, y_train, X_val, y_val in rolling_year_splits(X,y,start_date=start_date, end_date=end_date, train_years=train_years,validation_years=validation_years):
        # Save raw baseline values before scaling
        baseline_val_pred = None
        if baseline_column is not None:
            if baseline_column not in X.columns:
                raise ValueError(
                    f"Baseline column '{baseline_column}' not found in X columns."
                )
            baseline_val_pred = X_val[baseline_column]
        #Scale this fold
        X_train_scaled, X_val_scaled, scaler = scale_fold_features(X_train, X_val)
        #Create fresh model for this fold
        model = model_factory()
        #Fit model on training set
        model.fit(X_train_scaled, y_train)
        # DEBUG: feature importance for this fold
        coeffs = pd.Series(model.coef_, index=X.columns)
        print(f"\nFold {fold} coefficients:")
        print(coeffs.sort_values(key=abs, ascending=False))
        #Make predictions for training set and validation set
        y_train_pred = model.predict(X_train_scaled)
        y_val_pred = model.predict(X_val_scaled)
        
        #Compute training set and validation set errors (mean-squared)
        train_mse = mean_squared_error(y_train, y_train_pred)
        val_mse = mean_squared_error(y_val, y_val_pred)
        #Convert to root mean squared error
        train_rmse = np.sqrt(train_mse)
        val_rmse = np.sqrt(val_mse)
        #Compute training set and validation set errors (mean absolute error)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        val_mae = mean_absolute_error(y_val, y_val_pred)


        train_dates = pd.to_datetime(X_train.index.get_level_values(0))
        val_dates = pd.to_datetime(X_val.index.get_level_values(0))
        #Record results for fold
        row = {
            "fold": fold,
            "train_start": train_dates.min(),
            "train_end": train_dates.max(),
            "val_start": val_dates.min(),
            "val_end": val_dates.max(),
            "n_train": len(X_train),
            "n_val": len(X_val),
            "train_mse": train_mse,
            "val_mse": val_mse,
            "train_rmse": train_rmse,
            "val_rmse": val_rmse,
            "train_mae": train_mae,
            "val_mae": val_mae,
        }

        # Optional baseline metrics for benchmark comparison
        if baseline_val_pred is not None:
            baseline_val_mse = mean_squared_error(y_val, baseline_val_pred)
            baseline_val_rmse = np.sqrt(baseline_val_mse)
            baseline_val_mae = mean_absolute_error(y_val, baseline_val_pred)

            row["baseline_column"] = baseline_column
            row["baseline_val_mse"] = baseline_val_mse
            row["baseline_val_rmse"] = baseline_val_rmse
            row["baseline_val_mae"] = baseline_val_mae
            row["rmse_improvement_vs_baseline"] = baseline_val_rmse - val_rmse
            row["mae_improvement_vs_baseline"] = baseline_val_mae - val_mae

        results.append(row)

        # Optional validation predictions for plotting
        if return_predictions:
            pred_df = pd.DataFrame(
                {
                    "fold": fold,
                    "date": pd.to_datetime(X_val.index.get_level_values(0)),
                    "ticker": X_val.index.get_level_values(1),
                    "y_true": y_val.to_numpy(),
                    "y_pred": y_val_pred,
                }
            )

            if baseline_val_pred is not None:
                pred_df["baseline_pred"] = baseline_val_pred.to_numpy()

            prediction_frames.append(pred_df)
    

    results_df = pd.DataFrame(results)

    if return_predictions:
        predictions_df = pd.concat(prediction_frames, ignore_index=True)
        return results_df, predictions_df

    return results_df
