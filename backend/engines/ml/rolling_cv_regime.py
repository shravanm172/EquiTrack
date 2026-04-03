from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import pandas as pd
import numpy as np


from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler

from providers.market_data import fetch_price_history
from engines.portfolio_engine import prices_to_returns
from engines.ml.split_dataset import rolling_time_splits
from engines.ml.models.regime_predictor import build_feature_frames
from engines.targets.regime_labels import horizon_regime_score, fit_regime_cutoffs, apply_regime_cutoffs
from engines.ml.dataset_builder import assemble_dataset

@dataclass
class RegimeFoldResult:
    fold_number: int
    horizon: int
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    n_validation_rows: int
    log_loss: float | None
    accuracy: float | None

def run_rolling_validation(
    horizon: int,
    tickers: list[str],
    fetch_start: str,
    fetch_end: str,
    split_start: str,
    split_end: str,
    train_years: int = 4,
    validation_months: int = 3,
    step_months: int = 3,
)->list[RegimeFoldResult]:
    """
        Performs a rolling validation of the regime predictor model by iteratively training the model on an increasing training set size
        and validating it against a forecast horizon
        Ensure to keep most recent data out of this step completely (it will be the test set)
    """ 

    #Build full dataset once for fold slicing 
    price_history = fetch_price_history(tickers=tickers, start=fetch_start, end=fetch_end)
    prices = price_history.prices
    returns = prices_to_returns(prices)

    features = build_feature_frames(prices=prices, returns=returns)

    score_df = horizon_regime_score(
        returns=returns,
        horizon=horizon,
        annualize=True,
    )

    results: list[RegimeFoldResult] = []

    # temporary assembled dataset just to drive the split dates
    # we use score_df here only as an aligned placeholder target
    X_full, y_full_placeholder = assemble_dataset(features, score_df)

    for (
        fold_number,
        X_train_placeholder,
        y_train_score,
        X_validation_placeholder,
        y_validation_score,
    ) in rolling_time_splits(
        X_full,
        y_full_placeholder,
        start_date=split_start,
        end_date=split_end,
        train_years=train_years,
        validation_months=validation_months,
        step_months=step_months,
    ):
        #Define split (fold boundaries)
        train_dates = pd.to_datetime(X_train_placeholder.index.get_level_values(0))
        validation_dates = pd.to_datetime(X_validation_placeholder.index.get_level_values(0))

        train_start_fold = str(train_dates.min().date())
        train_end_fold = str(train_dates.max().date())
        validation_start_fold = str(validation_dates.min().date())
        validation_end_fold = str(validation_dates.max().date())

        # 1. Slice panel features for this fold
        train_features = {
            name: df.loc[train_start_fold:train_end_fold]
            for name, df in features.items()
        }
        validation_features = {
            name: df.loc[validation_start_fold:validation_end_fold]
            for name, df in features.items()
        }

        # 2. Slice score context for this fold
        train_score_df = score_df.loc[train_start_fold:train_end_fold]
        validation_score_df = score_df.loc[validation_start_fold:validation_end_fold]

        # 3. Fit cutoffs on training context only
        cutoffs = fit_regime_cutoffs(train_score_df)

        # 4. Build fold-specific regime targets
        y_train_df = apply_regime_cutoffs(train_score_df, cutoffs)
        y_validation_df = apply_regime_cutoffs(validation_score_df, cutoffs)

        # 5. Assemble fold datasets
        X_train, y_train = assemble_dataset(train_features, y_train_df)
        X_validation, y_validation = assemble_dataset(validation_features, y_validation_df)

        print("\n=== Fold", fold_number, "===")

        print("Train class distribution:")
        print(y_train.value_counts(normalize=True).sort_index())

        print("Validation class distribution:")
        print(y_validation.value_counts(normalize=True).sort_index())

        # 6. Clean and align labels
        y_train = y_train.astype(int)
        y_validation = y_validation.astype(int)

        # 7. Scale features
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            index=X_train.index,
            columns=X_train.columns,
        )
        X_validation_scaled = pd.DataFrame(
            scaler.transform(X_validation),
            index=X_validation.index,
            columns=X_validation.columns,
        )

        # 8. Fit fresh model on this fold
        model = HistGradientBoostingClassifier(
            loss="log_loss",
            max_depth=4,
            learning_rate=0.05,
            max_iter=300,
            min_samples_leaf=40,
            random_state=42,
        )
        model.fit(X_train_scaled, y_train)

        # 9. Predict on validation fold
        y_validation_proba = model.predict_proba(X_validation_scaled)
        y_validation_pred = model.predict(X_validation_scaled)

        # 10. Compute metrics
        fold_log_loss = log_loss(
            y_validation,
            y_validation_proba,
            labels=[0, 1, 2],
        )
        fold_accuracy = accuracy_score(y_validation, y_validation_pred)

        results.append(
            RegimeFoldResult(
                fold_number=fold_number,
                horizon=horizon,
                train_start=train_start_fold,
                train_end=train_end_fold,
                validation_start=validation_start_fold,
                validation_end=validation_end_fold,
                n_validation_rows=len(X_validation),
                log_loss=float(fold_log_loss),
                accuracy=float(fold_accuracy),
            )
        )

    return results


def main():
    results = run_rolling_validation(
        horizon=21,
        tickers=["AAPL", "MSFT", "AMZN", "GOOGL", "NVDA"],
        fetch_start="2017-01-01",
        fetch_end="2026-03-20",
        split_start="2017-01-01",
        split_end="2025-01-01",
        train_years=4,
        validation_months=1,
        step_months=1,
    )

    if not results:
        print("No rolling-validation folds were produced.")
        return

    print("=== Rolling Regime Validation Results ===")
    print(f"Folds: {len(results)}")

    total_log_loss = 0.0
    total_accuracy = 0.0

    for result in results:
        print()
        print(
            f"Fold {result.fold_number}: "
            f"Train [{result.train_start} -> {result.train_end}] | "
            f"Validation [{result.validation_start} -> {result.validation_end}]"
        )
        print(f"Validation rows: {result.n_validation_rows}")
        print(f"Log loss: {result.log_loss:.6f}")
        print(f"Accuracy: {result.accuracy:.6f}")

        total_log_loss += result.log_loss
        total_accuracy += result.accuracy

    mean_log_loss = total_log_loss / len(results)
    mean_accuracy = total_accuracy / len(results)

    print()
    print("=== Aggregate Metrics ===")
    print(f"Mean log loss: {mean_log_loss:.6f}")
    print(f"Mean accuracy: {mean_accuracy:.6f}")


if __name__ in "__main__":
    main()