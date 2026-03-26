from sklearn.preprocessing import StandardScaler
import pandas as pd


def scale_fold_features(
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Perform z-score normalization using training data only.

    Returns:
        X_train_scaled, X_validation_scaled, fitted scaler
    """

    scaler = StandardScaler()

    # Fit ONLY on training data
    X_train_scaled = scaler.fit_transform(X_train)

    # Transform validation using same scaler
    X_validation_scaled = scaler.transform(X_validation)

    # Convert back to DataFrames 
    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        index=X_train.index,
        columns=X_train.columns,
    )

    X_validation_scaled = pd.DataFrame(
        X_validation_scaled,
        index=X_validation.index,
        columns=X_validation.columns,
    )

    return X_train_scaled, X_validation_scaled, scaler