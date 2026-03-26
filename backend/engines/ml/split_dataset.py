from __future__ import annotations

from collections.abc import Iterator
import pandas as pd


def _validate_dataset_index(
    X: pd.DataFrame,
    y: pd.Series,
) -> None:
    """
    Validate that X and y use a MultiIndex of (date, ticker) and are aligned.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")
    if not isinstance(y, pd.Series):
        raise TypeError("y must be a pandas Series.")

    if X.empty:
        raise ValueError("X must not be empty.")
    if y.empty:
        raise ValueError("y must not be empty.")

    if not isinstance(X.index, pd.MultiIndex):
        raise TypeError("X must have a MultiIndex of (date, ticker).")
    if not isinstance(y.index, pd.MultiIndex):
        raise TypeError("y must have a MultiIndex of (date, ticker).")

    if X.index.nlevels != 2:
        raise ValueError("X index must have exactly 2 levels: (date, ticker).")
    if y.index.nlevels != 2:
        raise ValueError("y index must have exactly 2 levels: (date, ticker).")

    if not X.index.equals(y.index):
        raise ValueError("X and y must have the same aligned MultiIndex.")
    
def rolling_year_splits(
    X: pd.DataFrame,
    y: pd.Series,
    start_date: str,
    end_date: str,
    train_years: int = 4,
    validation_years: int = 1,
) -> Iterator[tuple[int, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]]:
    """
    Generate rolling-window yearly train/validation splits.
    """

    #Validation
    _validate_dataset_index(X, y)
    if not isinstance(train_years, int) or train_years <= 0:
        raise ValueError("train_years must be a positive integer.")
    if not isinstance(validation_years, int) or validation_years <= 0:
        raise ValueError("validation_years must be a positive integer.")

    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    if(start_date >= end_date):
        raise ValueError("start date must be greater than end date")
    
    dates = pd.to_datetime(X.index.get_level_values(0))
    
    fold_number = 0
    current_start = start_date
    
    while True:
        train_start = current_start
        train_end = train_start + pd.DateOffset(years=train_years)

        validation_start = train_end
        validation_end = validation_start + pd.DateOffset(years=validation_years)

        #No more data available
        if validation_end > end_date:
            break

        train_mask = (dates >= train_start) & (dates < train_end)
        validation_mask = (dates >= validation_start) & (dates < validation_end)

        X_train = X.loc[train_mask]
        y_train = y.loc[train_mask]
        X_validation = X.loc[validation_mask]
        y_validation = y.loc[validation_mask]

        if X_train.empty or y_train.empty:
            raise ValueError(
                f"Training split is empty for window [{train_start.date()}, {train_end.date()})."
            )
        if X_validation.empty or y_validation.empty:
            raise ValueError(
                f"Validation split is empty for window [{validation_start.date()}, {validation_end.date()})."
            )
        
        fold_number += 1
        yield fold_number, X_train, y_train, X_validation, y_validation

        current_start = current_start + pd.DateOffset(years=validation_years)