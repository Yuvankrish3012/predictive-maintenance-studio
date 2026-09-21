from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


# ============================================================
# PROJECT PATHS
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent

# Your actual NASA C-MAPSS folder
DATA_DIR = PROJECT_DIR / "CMAPSSData"

# Where trained models/results will be stored
RUNTIME_DIR = BACKEND_DIR / "runtime"
MODEL_DIR = RUNTIME_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATASET CONFIGURATION
# ============================================================

DATASETS = ["FD001", "FD002", "FD003", "FD004"]

COLUMN_NAMES = [
    "unit",
    "cycle",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
] + [f"sensor_{i}" for i in range(1, 22)]

SENSORS = [f"sensor_{i}" for i in range(1, 22)]

ROLLING_WINDOW = 5

# Standard capped-RUL convention used in many C-MAPSS studies.
RUL_CAP = 125


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Predictive Maintenance Studio",
    version="2.0.0",
    description="NASA C-MAPSS Remaining Useful Life research platform",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# DATA READING
# ============================================================

def get_dataset_paths(dataset: str):
    dataset = dataset.upper()

    if dataset not in DATASETS:
        raise ValueError(
            f"Invalid dataset: {dataset}. "
            f"Choose one of {DATASETS}."
        )

    train_path = DATA_DIR / f"train_{dataset}.txt"
    test_path = DATA_DIR / f"test_{dataset}.txt"
    rul_path = DATA_DIR / f"RUL_{dataset}.txt"

    return train_path, test_path, rul_path


def read_cmapps_file(path: Path) -> pd.DataFrame:
    """
    Read a NASA C-MAPSS whitespace-separated file.

    Expected structure:
    26 columns =
        engine ID
        cycle
        3 operating settings
        21 sensors
    """

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        engine="python",
    )

    # NASA files sometimes contain trailing whitespace.
    # Remove completely empty columns if present.
    df = df.dropna(axis=1, how="all")

    if df.shape[1] != 26:
        raise ValueError(
            f"{path.name} contains {df.shape[1]} columns. "
            f"Expected 26."
        )

    df.columns = COLUMN_NAMES

    df["unit"] = df["unit"].astype(int)
    df["cycle"] = df["cycle"].astype(int)

    return df


def read_rul_file(path: Path) -> np.ndarray:
    """
    Read the RUL_FD00X.txt file.
    """

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        engine="python",
    )

    values = (
        df.iloc[:, 0]
        .dropna()
        .astype(float)
        .to_numpy()
    )

    return values


# ============================================================
# RUL GENERATION
# ============================================================

def create_training_rul(df: pd.DataFrame) -> pd.DataFrame:
    """
    For training engines that run until failure:

        RUL = final_cycle - current_cycle

    Then cap RUL at 125 cycles.

    IMPORTANT:
    The final cycle is used ONLY to create the target.
    It is NOT used as an input feature.
    """

    result = df.copy()

    final_cycle = (
        result.groupby("unit")["cycle"]
        .transform("max")
    )

    result["rul"] = (
        final_cycle - result["cycle"]
    ).clip(
        upper=RUL_CAP
    )

    return result


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_features(
    df: pd.DataFrame,
    constant_columns: Optional[list[str]] = None,
) -> pd.DataFrame:

    result = (
        df.copy()
        .sort_values(["unit", "cycle"])
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Basic temporal feature
    # --------------------------------------------------------

    # Current engine age.
    # This is valid because the current cycle is known.
    result["engine_age"] = result["cycle"].astype(float)

    # --------------------------------------------------------
    # Causal rolling sensor features
    # --------------------------------------------------------
    #
    # We ONLY use current and previous observations.
    # No future observations are included.
    #

    grouped = result.groupby(
        "unit",
        sort=False
    )

    for sensor in SENSORS:

        result[f"{sensor}_roll_mean"] = (
            grouped[sensor]
            .transform(
                lambda x:
                    x.rolling(
                        ROLLING_WINDOW,
                        min_periods=1
                    ).mean()
            )
        )

        result[f"{sensor}_roll_std"] = (
            grouped[sensor]
            .transform(
                lambda x:
                    x.rolling(
                        ROLLING_WINDOW,
                        min_periods=2
                    ).std()
            )
            .fillna(0.0)
        )

    # --------------------------------------------------------
    # Remove training-derived constant features
    # --------------------------------------------------------

    if constant_columns:
        result = result.drop(
            columns=[
                c for c in constant_columns
                if c in result.columns
            ],
            errors="ignore",
        )

    return result


def find_constant_columns(df: pd.DataFrame) -> list[str]:

    candidates = [
        c for c in df.columns
        if c.startswith("sensor_")
        or c.startswith("op_setting_")
    ]

    return [
        c for c in candidates
        if df[c].nunique(dropna=False) <= 1
    ]


def get_feature_columns(df: pd.DataFrame) -> list[str]:

    excluded = {
        "unit",
        "rul",
    }

    feature_columns = []

    for column in df.columns:

        if column in excluded:
            continue

        if df[column].nunique(dropna=False) <= 1:
            continue

        feature_columns.append(column)

    return feature_columns


# ============================================================
# NASA ASYMMETRIC SCORE
# ============================================================

def nasa_score(
    y_true,
    y_pred,
) -> float:

    """
    NASA C-MAPSS asymmetric scoring function.

    d = prediction - actual

    Early prediction:
        d < 0
        penalty scale = 13

    Late prediction:
        d >= 0
        penalty scale = 10
    """

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    error = y_pred - y_true

    score = np.where(
        error < 0,
        np.exp(-error / 13.0) - 1.0,
        np.exp(error / 10.0) - 1.0,
    )

    return float(np.sum(score))


def calculate_metrics(
    y_true,
    y_pred,
) -> dict:

    y_pred = np.maximum(
        np.asarray(y_pred, dtype=float),
        0.0,
    )

    return {
        "mae": float(
            mean_absolute_error(
                y_true,
                y_pred,
            )
        ),
        "rmse": float(
            math.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred,
                )
            )
        ),
        "nasa_score": nasa_score(
            y_true,
            y_pred,
        ),
    }


# ============================================================
# ENGINE-SEPARATED VALIDATION
# ============================================================

def train_validation_models(
    train_features: pd.DataFrame,
    feature_columns: list[str],
):

    all_units = np.sort(
        train_features["unit"].unique()
    )

    train_units, validation_units = train_test_split(
        all_units,
        test_size=0.20,
        random_state=42,
    )

    training_data = train_features[
        train_features["unit"].isin(train_units)
    ].copy()

    validation_data = train_features[
        train_features["unit"].isin(validation_units)
    ].copy()

    # --------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------

    baseline_value = float(
        training_data["rul"].mean()
    )

    baseline_predictions = np.full(
        len(validation_data),
        baseline_value,
    )

    baseline_metrics = calculate_metrics(
        validation_data["rul"],
        baseline_predictions,
    )

    # --------------------------------------------------------
    # XGBOOST
    # --------------------------------------------------------

    model = XGBRegressor(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=42,
        n_jobs=4,
        importance_type="gain",
    )

    model.fit(
        training_data[feature_columns],
        training_data["rul"],
    )

    predictions = model.predict(
        validation_data[feature_columns]
    )

    predictions = np.maximum(
        predictions,
        0.0,
    )

    xgb_metrics = calculate_metrics(
        validation_data["rul"],
        predictions,
    )

    return {
        "train_units": int(len(train_units)),
        "validation_units": int(len(validation_units)),
        "baseline": baseline_metrics,
        "xgboost": xgb_metrics,
    }


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def get_feature_importance(
    model: XGBRegressor,
    feature_columns: list[str],
    top_n: int = 20,
) -> list[dict]:
    """
    Return frontend-friendly XGBoost feature importance.

    Importance is based on XGBoost gain and is normalized to a
    percentage for display. The raw importance is retained too.
    """
    raw_values = np.asarray(
        model.feature_importances_,
        dtype=float,
    )

    if len(raw_values) != len(feature_columns):
        raise ValueError(
            "XGBoost feature-importance length does not match "
            "the feature-column list."
        )

    total = float(np.sum(raw_values))
    rows = []

    for feature, value in zip(feature_columns, raw_values):
        gain = float(value)
        rows.append(
            {
                "feature": str(feature),
                "importance": gain,
                "gain": gain,
                "percentage": (
                    float((gain / total) * 100.0)
                    if total > 0
                    else 0.0
                ),
            }
        )

    rows.sort(
        key=lambda item: item["importance"],
        reverse=True,
    )

    return rows[:top_n]


# ============================================================
# FULL DATASET TRAINING
# ============================================================

def train_dataset(dataset: str):

    dataset = dataset.upper()

    train_path, test_path, rul_path = (
        get_dataset_paths(dataset)
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    train_df = read_cmapps_file(
        train_path
    )

    test_df = read_cmapps_file(
        test_path
    )

    true_test_rul = read_rul_file(
        rul_path
    )

    train_engine_count = int(
        train_df["unit"].nunique()
    )

    test_engine_count = int(
        test_df["unit"].nunique()
    )

    if test_engine_count != len(true_test_rul):

        raise ValueError(
            f"{dataset}: "
            f"{test_engine_count} test engines "
            f"but {len(true_test_rul)} RUL labels."
        )

    # --------------------------------------------------------
    # Generate training RUL
    # --------------------------------------------------------

    labeled_train = create_training_rul(
        train_df
    )

    # --------------------------------------------------------
    # Determine constant columns ONLY from training data
    # --------------------------------------------------------

    raw_constant_columns = (
        find_constant_columns(
            labeled_train
        )
    )

    # --------------------------------------------------------
    # Build training features
    # --------------------------------------------------------

    training_features = build_features(
        labeled_train,
        constant_columns=raw_constant_columns,
    )

    feature_columns = get_feature_columns(
        training_features
    )

    if not feature_columns:
        raise ValueError(
            f"{dataset}: no usable features found."
        )

    # --------------------------------------------------------
    # Engine-separated validation
    # --------------------------------------------------------

    validation_metrics = (
        train_validation_models(
            training_features,
            feature_columns,
        )
    )

    # --------------------------------------------------------
    # Final XGBoost model
    #
    # Train on ALL training engines after validation.
    # --------------------------------------------------------

    final_model = XGBRegressor(
        n_estimators=700,
        max_depth=7,
        learning_rate=0.035,
        subsample=0.88,
        colsample_bytree=0.88,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=42,
        n_jobs=4,
        importance_type="gain",
    )

    final_model.fit(
        training_features[feature_columns],
        training_features["rul"],
    )

    # --------------------------------------------------------
    # XGBoost feature importance
    # --------------------------------------------------------

    feature_importance = get_feature_importance(
        final_model,
        feature_columns,
        top_n=20,
    )

    # --------------------------------------------------------
    # Test feature generation
    # --------------------------------------------------------

    test_features = build_features(
        test_df,
        constant_columns=raw_constant_columns,
    )

    # --------------------------------------------------------
    # Only the LAST observation of each test engine is used
    # for final RUL prediction.
    # --------------------------------------------------------

    last_rows = (
        test_features
        .sort_values(["unit", "cycle"])
        .groupby(
            "unit",
            as_index=False,
        )
        .tail(1)
        .sort_values("unit")
    )

    predictions = final_model.predict(
        last_rows[feature_columns]
    )

    predictions = np.maximum(
        predictions,
        0.0,
    )

    # --------------------------------------------------------
    # Test evaluation
    # --------------------------------------------------------

    test_metrics = calculate_metrics(
        true_test_rul,
        predictions,
    )

    # --------------------------------------------------------
    # Prediction table
    # --------------------------------------------------------

    prediction_rows = []

    for unit, predicted, actual in zip(
        last_rows["unit"].to_numpy(),
        predictions,
        true_test_rul,
    ):

        error = float(
            predicted - actual
        )

        if predicted <= 15:
            alert = "CRITICAL"
        elif predicted <= 30:
            alert = "WARNING"
        else:
            alert = "NORMAL"

        prediction_rows.append(
            {
                "unit": int(unit),
                "last_cycle": int(
                    last_rows.loc[
                        last_rows["unit"] == unit,
                        "cycle"
                    ].iloc[0]
                ),
                "predicted_rul": round(
                    float(predicted),
                    3,
                ),
                "actual_rul": round(
                    float(actual),
                    3,
                ),
                "error": round(
                    error,
                    3,
                ),
                "alert": alert,
            }
        )

    predictions_df = pd.DataFrame(
        prediction_rows
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    prediction_path = (
        MODEL_DIR /
        f"{dataset}_predictions.csv"
    )

    predictions_df.to_csv(
        prediction_path,
        index=False,
    )

    # --------------------------------------------------------
    # Save model artifact
    # --------------------------------------------------------

    artifact = {
        "dataset": dataset,
        "model": final_model,
        "feature_columns": feature_columns,
        "constant_columns": raw_constant_columns,
        "rolling_window": ROLLING_WINDOW,
        "rul_cap": RUL_CAP,
        "feature_importance": feature_importance,
        "feature_importance_type": "XGBoost gain importance",
    }

    model_path = (
        MODEL_DIR /
        f"{dataset}_model.joblib"
    )

    joblib.dump(
        artifact,
        model_path,
    )

    # --------------------------------------------------------
    # Complete metrics
    # --------------------------------------------------------

    metrics = {
        "dataset": dataset,
        "status": "completed",
        "training_status": "completed",
        "training_completed": True,
        "message": (
            f"{dataset} training completed successfully. "
            "XGBoost model trained and final test evaluation generated."
        ),
        "train_engines": train_engine_count,
        "test_engines": test_engine_count,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "feature_count": int(
            len(feature_columns)
        ),
        "feature_importance": feature_importance,
        "feature_importance_type": "XGBoost gain importance",
        "validation": validation_metrics,
        "test_evaluation": test_metrics,
    }

    metrics_path = (
        MODEL_DIR /
        f"{dataset}_metrics.json"
    )

    metrics_path.write_text(
        json.dumps(
            metrics,
            indent=2,
        ),
        encoding="utf-8",
    )

    return metrics


# ============================================================
# DATASET STATUS
# ============================================================

def get_dataset_status():

    results = []

    for dataset in DATASETS:

        train_path, test_path, rul_path = (
            get_dataset_paths(dataset)
        )

        loaded = (
            train_path.exists()
            and test_path.exists()
            and rul_path.exists()
        )

        item = {
            "dataset": dataset,
            "loaded": loaded,
            "trained": False,
            "status": "ready" if loaded else "missing",
            "training_status": "not_trained",
            "training_completed": False,
        }

        if loaded:

            try:

                train_df = read_cmapps_file(
                    train_path
                )

                test_df = read_cmapps_file(
                    test_path
                )

                rul = read_rul_file(
                    rul_path
                )

                item.update(
                    {
                        "train_engines": int(
                            train_df["unit"].nunique()
                        ),
                        "test_engines": int(
                            test_df["unit"].nunique()
                        ),
                        "train_rows": int(
                            len(train_df)
                        ),
                        "test_rows": int(
                            len(test_df)
                        ),
                        "rul_labels": int(
                            len(rul)
                        ),
                        "valid": (
                            test_df["unit"].nunique()
                            == len(rul)
                        ),
                    }
                )

            except Exception as exc:

                item["valid"] = False
                item["error"] = str(exc)

        metrics_path = (
            MODEL_DIR /
            f"{dataset}_metrics.json"
        )

        if metrics_path.exists():

            item["trained"] = True
            item["status"] = "trained"
            item["training_status"] = "completed"
            item["training_completed"] = True

            item["metrics"] = json.loads(
                metrics_path.read_text(
                    encoding="utf-8"
                )
            )

        results.append(item)

    return results


# ============================================================
# API ROUTES
# ============================================================

@app.get("/api/health")
def health():

    return {
        "status": "ok",
        "application": "Predictive Maintenance Studio",
        "data_directory": str(DATA_DIR),
        "datasets": DATASETS,
    }


@app.get("/api/datasets")
def datasets():

    return get_dataset_status()


@app.post("/api/train/{dataset}")
def train(dataset: str):

    dataset = dataset.upper()

    if dataset not in DATASETS:

        raise HTTPException(
            status_code=404,
            detail=(
                "Dataset must be "
                "FD001, FD002, FD003 or FD004."
            ),
        )

    try:

        return train_dataset(
            dataset
        )

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )


@app.get("/api/metrics/{dataset}")
def metrics(dataset: str):

    dataset = dataset.upper()

    path = (
        MODEL_DIR /
        f"{dataset}_metrics.json"
    )

    if not path.exists():

        raise HTTPException(
            status_code=404,
            detail=(
                f"{dataset} has not been trained yet."
            ),
        )

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


@app.get("/api/predictions/{dataset}")
def predictions(dataset: str):

    dataset = dataset.upper()

    path = (
        MODEL_DIR /
        f"{dataset}_predictions.csv"
    )

    if not path.exists():

        raise HTTPException(
            status_code=404,
            detail=(
                f"{dataset} has not been trained yet."
            ),
        )

    return pd.read_csv(
        path
    ).to_dict(
        orient="records"
    )


@app.get("/api/download/{dataset}")
def download_predictions(dataset: str):

    dataset = dataset.upper()

    path = (
        MODEL_DIR /
        f"{dataset}_predictions.csv"
    )

    if not path.exists():

        raise HTTPException(
            status_code=404,
            detail=(
                f"{dataset} has not been trained yet."
            ),
        )

    return FileResponse(
        path,
        filename=f"{dataset}_predictions.csv",
        media_type="text/csv",
    )


@app.get("/api/sensors/{dataset}/{unit}")
def sensor_history(
    dataset: str,
    unit: int,
    sensor: str = "sensor_2",
):

    dataset = dataset.upper()

    if sensor not in SENSORS:

        raise HTTPException(
            status_code=400,
            detail="Invalid sensor.",
        )

    try:

        train_path, _, _ = (
            get_dataset_paths(dataset)
        )

        df = read_cmapps_file(
            train_path
        )

        engine = (
            df[
                df["unit"] == unit
            ][
                ["cycle", sensor]
            ]
            .sort_values("cycle")
        )

        if engine.empty:

            raise ValueError(
                f"Engine {unit} not found."
            )

        return {
            "dataset": dataset,
            "unit": unit,
            "sensor": sensor,
            "history": [
                {
                    "cycle": int(row["cycle"]),
                    "value": float(row[sensor]),
                }
                for _, row in engine.iterrows()
            ],
        }

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )


# ============================================================
# STARTUP CHECK
# ============================================================

@app.on_event("startup")
def startup_check():

    print("=" * 60)
    print("PREDICTIVE MAINTENANCE STUDIO")
    print("=" * 60)
    print(f"Project directory : {PROJECT_DIR}")
    print(f"C-MAPSS directory : {DATA_DIR}")
    print()

    for dataset in DATASETS:

        train_path, test_path, rul_path = (
            get_dataset_paths(dataset)
        )

        status = (
            train_path.exists()
            and test_path.exists()
            and rul_path.exists()
        )

        print(
            f"{dataset}: "
            f"{'READY' if status else 'MISSING'}"
        )

    print("=" * 60)