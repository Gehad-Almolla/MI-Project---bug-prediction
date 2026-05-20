"""
export_models.py
────────────────
Recreates training from CLEAN_dataset.csv and saves all SIX models:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - SVM (wrapped in sklearn Pipeline with StandardScaler)
  - ANN / Neural Network (Keras/TensorFlow, saved as .keras)
  - LightGBM

Run ONCE before launching the Streamlit app:
    python export_models.py
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATASET_PATH = "CLEAN_dataset.csv"
MODEL_DIR    = "models"
RANDOM_STATE = 42

FEATURE_COLUMNS = [
    "parent_count",
    "files_changed",
    "additions",
    "deletions",
    "total_changes",
    "message_length",
    "commit_hour",
    "commit_day",
    "developer_experience",
    "directories_changed",
]


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def load_data():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found at '{DATASET_PATH}'.\n"
            "Place CLEAN_dataset.csv in the same folder as this script."
        )
    df = pd.read_csv(DATASET_PATH)
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    X = df[FEATURE_COLUMNS]
    y = df["buggy_label"]
    return X, y


def evaluate(name, model, X_test, y_test, is_keras=False):
    if is_keras:
        y_pred = (model.predict(X_test, verbose=0) >= 0.5).astype(int).flatten()
    else:
        y_pred = model.predict(X_test)
    print(f"\n{'─'*42}")
    print(f"  {name}")
    print(f"{'─'*42}")
    print(f"  Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  Recall   : {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  F1       : {f1_score(y_test, y_pred, zero_division=0):.4f}")
    print(classification_report(y_test, y_pred, target_names=["Not Buggy", "Buggy"], zero_division=0))


def save_model(model, filename):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, filename)
    joblib.dump(model, path)
    print(f"  ✓ Saved → {path}")


# ─────────────────────────────────────────────
# SKLEARN MODELS
# ─────────────────────────────────────────────

def train_logistic_regression(X_train, y_train):
    print("\n[1/6] Training Logistic Regression …")
    params = {
        "C":        [0.01, 0.1, 1, 10, 100],
        "solver":   ["liblinear", "lbfgs"],
        "max_iter": [500, 1000],
    }
    grid = GridSearchCV(
        LogisticRegression(random_state=RANDOM_STATE),
        param_grid=params, cv=5, scoring="accuracy", n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    print(f"  Best params: {grid.best_params_}")
    return grid.best_estimator_


def train_decision_tree(X_train, y_train):
    print("\n[2/6] Training Decision Tree …")
    params = {
        "criterion":         ["gini", "entropy"],
        "max_depth":         [3, 5, 7, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf":  [1, 3, 5],
    }
    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        param_grid=params, cv=5, scoring="accuracy", n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    print(f"  Best params: {grid.best_params_}")
    return grid.best_estimator_


def train_random_forest(X_train, y_train):
    print("\n[3/6] Training Random Forest …")
    params = {
        "n_estimators":      [100, 200, 300],
        "max_depth":         [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf":  [1, 2],
        "max_features":      ["sqrt", "log2"],
    }
    grid = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        param_grid=params, cv=5, scoring="accuracy", n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    print(f"  Best params: {grid.best_params_}")
    return grid.best_estimator_


def train_svm(X_train, y_train):
    """SVM wrapped in a Pipeline with StandardScaler so predict() works without pre-scaling."""
    print("\n[4/6] Training SVM …")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svc",    SVC(kernel="linear", probability=True, random_state=RANDOM_STATE)),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


# ─────────────────────────────────────────────
# ANN (Keras)
# ─────────────────────────────────────────────

def train_ann(X_train, y_train):
    """
    Trains the ANN defined in the notebook (64-32-16-1 architecture).
    Saves:
      - models/ann_scaler.pkl  (StandardScaler fitted on X_train)
      - models/ann_model.keras (the Keras model)
    Returns (scaler, keras_model).
    """
    print("\n[5/6] Training ANN (Keras) …")

    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
    except ImportError:
        raise ImportError(
            "TensorFlow is not installed.\n"
            "Install it with:  pip install tensorflow"
        )

    # Scale
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_train)

    # Architecture (final notebook version: 64-32-16-1 with Dropout 0.3)
    model = Sequential([
        Dense(64, activation="relu", input_shape=(X_tr.shape[1],)),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(16, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    model.summary()

    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(
        X_tr, y_train,
        epochs=30,
        batch_size=32,
        validation_split=0.2,
        callbacks=[es],
        verbose=1,
    )
    return scaler, model


# ─────────────────────────────────────────────
# LightGBM
# ─────────────────────────────────────────────

def train_lightgbm(X_train, y_train):
    """
    LightGBM with class-weight balancing + GridSearchCV (matches notebook).
    The model is saved with joblib like the sklearn models.
    X_train here is the RAW (unscaled) data — LightGBM doesn't need scaling.
    """
    print("\n[6/6] Training LightGBM …")

    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        raise ImportError(
            "LightGBM is not installed.\n"
            "Install it with:  pip install lightgbm"
        )

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=y_train,
    )
    class_weight_dict = {0: weights[0], 1: weights[1]}

    param_grid = {
        "n_estimators":      [100, 200, 300],
        "num_leaves":        [31, 63, 127],
        "learning_rate":     [0.05, 0.1, 0.2],
        "min_child_samples": [20, 50],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid_search = GridSearchCV(
        LGBMClassifier(
            random_state=RANDOM_STATE,
            verbosity=-1,
            class_weight=class_weight_dict,
        ),
        param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=1,
    )
    grid_search.fit(X_train, y_train)
    print(f"  Best params  : {grid_search.best_params_}")
    print(f"  Best AUC-ROC : {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 48)
    print("  Buggy Commit Predictor — Model Export (All 6)")
    print("=" * 48)

    print(f"\nLoading dataset from '{DATASET_PATH}' …")
    X, y = load_data()
    print(f"  Rows    : {len(X)}")
    print(f"  Features: {list(X.columns)}")
    print(f"  Buggy   : {y.sum()} ({y.mean():.1%})")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y,
    )
    print(f"\n  Train: {len(X_train)} samples  |  Test: {len(X_test)} samples")

    # ── Train sklearn models on raw data ──────
    lr  = train_logistic_regression(X_train, y_train)
    dt  = train_decision_tree(X_train, y_train)
    rf  = train_random_forest(X_train, y_train)
    svm = train_svm(X_train, y_train)          # Pipeline (scales internally)
    lgbm = train_lightgbm(X_train, y_train)    # raw data, no scaling needed

    # ── Train ANN (Keras) ─────────────────────
    ann_scaler, ann_model = train_ann(X_train, y_train)

    # ── Evaluate ──────────────────────────────
    print("\n" + "=" * 48)
    print("  Evaluation on Test Set")
    print("=" * 48)
    evaluate("Logistic Regression", lr,  X_test, y_test)
    evaluate("Decision Tree",       dt,  X_test, y_test)
    evaluate("Random Forest",       rf,  X_test, y_test)
    evaluate("SVM (Pipeline)",      svm, X_test, y_test)
    evaluate("LightGBM",            lgbm, X_test, y_test)

    # ANN needs scaled test data
    X_test_scaled = ann_scaler.transform(X_test)
    evaluate("ANN (Keras)", ann_model, X_test_scaled, y_test, is_keras=True)

    # ── Save ──────────────────────────────────
    print("\n" + "=" * 48)
    print("  Saving Models")
    print("=" * 48)
    os.makedirs(MODEL_DIR, exist_ok=True)

    save_model(lr,         "logistic_regression.pkl")
    save_model(dt,         "decision_tree.pkl")
    save_model(rf,         "random_forest.pkl")
    save_model(svm,        "svm.pkl")
    save_model(lgbm,       "lightgbm.pkl")
    save_model(ann_scaler, "ann_scaler.pkl")

    # Keras model saved in native format
    ann_path = os.path.join(MODEL_DIR, "ann_model.keras")
    ann_model.save(ann_path)
    print(f"  ✓ Saved → {ann_path}")

    print("\n✅ All 6 models exported to the models/ directory.")
    print("   You can now run:  streamlit run app.py\n")


if __name__ == "__main__":
    main()
