#!/usr/bin/env python

"""
Nested CV for 7 models on 0–48months healthy transition study.

Data files required in current directory:
- feature_table_5000.tsv
- Healthy_0-4_agebins.txt    (must contain column 'Age_bin4')

Steps:
1) Load feature table and metadata
2) Align samples
3) 75/25 train/test split (test held out, not used)
4) Nested CV on 75% train:
   - Outer: 5-fold StratifiedKFold
   - Inner: 4-fold StratifiedKFold + GridSearchCV
5) For each model, compute:
   - Inner: best ROC AUC (macro, ovr) from inner CV per outer fold
   - Outer: ROC AUC (macro, ovr) on validation fold
6) Save summary and per-fold results to TSV files.
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

import xgboost as xgb
from lightgbm import LGBMClassifier

# -----------------------------
# Global settings
# -----------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.25        # 75/25 split (train/test)
N_OUTER = 5             # outer folds
N_INNER = 4             # inner folds
N_JOBS = 8              # limit parallelism to 8 threads


# -----------------------------
# Helper functions
# -----------------------------
#creat a function to read the TSV file and return a dataframe
def load_feature_table(path):
    """Load feature table from TSV file.
    Robust loader from the biom converted files 
    handles the comment lines as #Construct from biom"""
     # Read lines first to find where the real header starts
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    header_idx = None
    header_candidates = (
        "#OTU ID", "OTU ID",
        "#OTU_ID", "OTU_ID",
        "#Feature ID", "Feature ID",
        "#feature-id", "feature-id",
    )

    # Find the first non-empty line that looks like the header
    for i, line in enumerate(lines):
        s = line.strip()
        if not s:
            continue  # skip empty lines

        # If the line contains tabs and starts with one of the known header patterns
        if "\t" in s and any(s.startswith(h) for h in header_candidates):
            header_idx = i
            break

        # Fallback: if it's the first tabbed line that is NOT a comment-only line
        # (useful if header doesn't match candidates)
        if "\t" in s and not s.startswith("##"):
            # Still could be a comment, but likely header
            header_idx = i
            break

    if header_idx is None:
        raise ValueError("Could not find a valid header line in the feature table TSV.")

    # Now load from that header line onward
    df = pd.read_csv(path, sep="\t", header=0, skiprows=header_idx)

    # Rename first column to 'feature_id'
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "feature_id"})

    # Set index and transpose -> samples as rows
    df = df.set_index("feature_id").T
    df.index.name = "sample-id"

    # Convert to numeric
    df = df.astype(float)

    return df


def load_metadata(path):
    """
    Load metadata with Age_bin4 column.
    Index: sample-id
    """
    meta = pd.read_csv(path, sep="\t")
    meta = meta.set_index("sample-id")
    return meta


def multiclass_roc_auc_ovr_scorer(estimator, X, y):
    """
    Custom scorer for GridSearchCV:
    - Uses predict_proba
    - Returns ROC AUC (macro, ovr) for multi-class
    - Returns np.nan gracefully if something goes wrong
      (e.g., only one class in y or wrong proba shape).
    """
    # Need at least 2 classes to compute ROC AUC
    if len(np.unique(y)) < 2:
        return np.nan

    # Get probabilities or decision scores
    if hasattr(estimator, "predict_proba"):
        y_score = estimator.predict_proba(X)
    elif hasattr(estimator, "decision_function"):
        y_score = estimator.decision_function(X)
    else:
        return np.nan

    # Ensure 2D array
    y_score = np.asarray(y_score)
    if y_score.ndim == 1:
        # Shape (n_samples,) is not acceptable for multi-class ROC AUC
        return np.nan

    try:
        score = roc_auc_score(
            y,
            y_score,
            multi_class="ovr",
            average="macro",
        )
    except Exception:
        score = np.nan

    return score


def safe_roc_auc_macro_ovr(y_true, y_score):
    """
    Safely compute ROC AUC macro ovr for outer validation fold.
    Returns np.nan if it fails.
    """
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        return np.nan
    if y_score.ndim == 1:
        return np.nan
    try:
        return roc_auc_score(
            y_true,
            y_score,
            multi_class="ovr",
            average="macro",
        )
    except Exception:
        return np.nan


def build_models_and_grids(n_classes):
    """
    Define the 7 models and small hyperparameter grids for inner CV.
    We keep grids modest so runtime doesn't explode.
    All probabilistic classifiers (predict_proba available).
    """

    models = {}
    param_grids = {}

    # ----------------------
    # 1) Random Forest
    # ----------------------
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=1,
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
    )
    models["RandomForest"] = rf
    param_grids["RandomForest"] = {
        "n_estimators": [300, 500],
        "max_depth": [None, 15],
        "min_samples_leaf": [1, 2],
    }

    # ----------------------
    # 2) Extra Trees
    # ----------------------
    et = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=1,
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
    )
    models["ExtraTrees"] = et
    param_grids["ExtraTrees"] = {
        "n_estimators": [300, 500],
        "max_depth": [None, 15],
        "min_samples_leaf": [1, 2],
    }

    # ----------------------
    # 3) LightGBM
    # ----------------------
    lgbm = LGBMClassifier(
        objective="multiclass",
        num_class=n_classes,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )
    models["LightGBM"] = lgbm
    param_grids["LightGBM"] = {
        "num_leaves": [31, 63],
        "learning_rate": [0.05, 0.1],
        "n_estimators": [200, 400],
    }

    # ----------------------
    # 4) XGBoost
    # ----------------------
    xgb_clf = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        use_label_encoder=False,
    )
    models["XGBoost"] = xgb_clf
    param_grids["XGBoost"] = {
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
        "n_estimators": [200, 400],
        "subsample": [0.8, 1.0],
    }

    # ----------------------
    # Scaled models via Pipeline
    # ----------------------
    # 5) RBF-SVM
    svm_pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
        ]
    )
    models["RBF-SVM"] = svm_pipe
    param_grids["RBF-SVM"] = {
        "clf__C": [1.0, 10.0],
        "clf__gamma": ["scale", 0.1],
    }

    # 6) Multinomial Logistic Regression
    lr_pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "clf",
                LogisticRegression(
                    multi_class="multinomial",
                    solver="lbfgs",
                    penalty="l2",
                    max_iter=1000,
                    n_jobs=N_JOBS,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    models["MultinomialLR"] = lr_pipe
    param_grids["MultinomialLR"] = {
        "clf__C": [0.5, 1.0, 2.0],
    }

    # 7) MLP
    mlp_pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                MLPClassifier(
                    hidden_layer_sizes=(100,),
                    activation="relu",
                    solver="adam",
                    max_iter=300,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    models["MLP"] = mlp_pipe
    param_grids["MLP"] = {
        "clf__hidden_layer_sizes": [(50,), (100,)],
        "clf__alpha": [0.0001, 0.001],
    }

    return models, param_grids


def main():
    # -----------------------------
    # 1) Load data
    # -----------------------------
    print("Loading feature table and metadata...")
    X_all = load_feature_table("feature_table_5000.tsv")
    meta = load_metadata("Healthy_0-4_agebins.txt")

    if "Age_bin4" not in meta.columns:
        raise ValueError(
            "Metadata file does not contain 'Age_bin4'. "
            "Please add age bins before running this script."
        )

    # Align samples between feature table and metadata
    common_samples = X_all.index.intersection(meta.index)
    print(f"Total samples in feature table: {X_all.shape[0]}")
    print(f"Total samples in metadata    : {meta.shape[0]}")
    print(f"Common samples               : {len(common_samples)}")

    X_all = X_all.loc[common_samples]
    y_labels = meta.loc[common_samples, "Age_bin4"]

    # Encode labels
    le = LabelEncoder()
    y_all = le.fit_transform(y_labels)
    class_names = list(le.classes_)
    n_classes = len(class_names)

    print("Classes (age bins):", class_names)
    print("Number of classes :", n_classes)

    # -----------------------------
    # 2) 75/25 train/test split
    # -----------------------------
    print("\nSplitting into 75% train and 25% test (test is held out)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_all,
        y_all,
        test_size=TEST_SIZE,
        stratify=y_all,
        random_state=RANDOM_STATE,
    )

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print("Test set will NOT be used in nested CV\n")

    # -----------------------------
    # 3) Build models & param grids
    # -----------------------------
    models, param_grids = build_models_and_grids(n_classes)

    # -----------------------------
    # 4) Nested CV for each model
    # -----------------------------
    summary_rows = []
    outer_details = []

    outer_cv = StratifiedKFold(
        n_splits=N_OUTER, shuffle=True, random_state=RANDOM_STATE
    )

    for model_name, base_estimator in models.items():
        print("=" * 80)
        print(f"Model: {model_name}")
        print("=" * 80)

        param_grid = param_grids[model_name]

        inner_best_scores = []
        outer_scores = []

        # Outer loop
        for outer_fold, (train_idx, val_idx) in enumerate(
            outer_cv.split(X_train, y_train), start=1
        ):
            print(f"\n  Outer fold {outer_fold}/{N_OUTER} for {model_name}")

            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            # Inner CV
            inner_cv = StratifiedKFold(
                n_splits=N_INNER, shuffle=True, random_state=RANDOM_STATE
            )

            gs = GridSearchCV(
                estimator=base_estimator,
                param_grid=param_grid,
                scoring=multiclass_roc_auc_ovr_scorer,
                cv=inner_cv,
                n_jobs=N_JOBS,
                refit=True,
            )

            print("    Running inner 4-fold CV with GridSearchCV...")
            gs.fit(X_tr, y_tr)

            # GridSearchCV stores cv_results_; best_score_ may be nan if all folds failed
            best_inner_score = float(gs.best_score_) if gs.best_score_ is not None else np.nan
            inner_best_scores.append(best_inner_score)

            print(f"    Best inner ROC AUC (macro ovr): {best_inner_score:.4f}")
            print(f"    Best params: {gs.best_params_}")

            # Evaluate on outer validation fold
            if hasattr(gs.best_estimator_, "predict_proba"):
                y_val_proba = gs.best_estimator_.predict_proba(X_val)
            elif hasattr(gs.best_estimator_, "decision_function"):
                y_val_proba = gs.best_estimator_.decision_function(X_val)
            else:
                y_val_proba = None

            if y_val_proba is not None:
                outer_roc = safe_roc_auc_macro_ovr(y_val, y_val_proba)
            else:
                outer_roc = np.nan

            outer_scores.append(outer_roc)

            print(f"    Outer fold ROC AUC (macro ovr): {outer_roc:.4f}")

            outer_details.append(
                {
                    "model": model_name,
                    "outer_fold": outer_fold,
                    "roc_auc_macro_outer": outer_roc,
                    "inner_best_roc_auc_macro": best_inner_score,
                }
            )

        # After all outer folds for this model
        inner_mean = float(np.nanmean(inner_best_scores))
        inner_std = float(np.nanstd(inner_best_scores, ddof=1))
        outer_mean = float(np.nanmean(outer_scores))
        outer_std = float(np.nanstd(outer_scores, ddof=1))

        print("\nSummary for model:", model_name)
        print(f"  Inner CV best ROC AUC (mean ± sd): {inner_mean:.4f} ± {inner_std:.4f}")
        print(f"  Outer CV ROC AUC (mean ± sd)     : {outer_mean:.4f} ± {outer_std:.4f}")

        summary_rows.append(
            {
                "model": model_name,
                "inner_mean_roc_auc_macro": inner_mean,
                "inner_std_roc_auc_macro": inner_std,
                "outer_mean_roc_auc_macro": outer_mean,
                "outer_std_roc_auc_macro": outer_std,
            }
        )

    # -----------------------------
    # 5) Save results
    # -----------------------------
    summary_df = pd.DataFrame(summary_rows)
    outer_df = pd.DataFrame(outer_details)

    summary_out = "nested_cv_summary_7models.tsv"
    outer_out = "nested_cv_outer_folds_7models.tsv"

    summary_df.to_csv(summary_out, sep="\t", index=False)
    outer_df.to_csv(outer_out, sep="\t", index=False)

    print("\nSaved nested CV summary to:", summary_out)
    print("Saved per-outer-fold details to:", outer_out)
    print("\nDone.")


if __name__ == "__main__":
    main()

