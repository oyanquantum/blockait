"""
train.py — Step 3: train and evaluate all Blockait models, end-to-end.

Pipeline:
  1. Load the stratified train/test split.
  2. Fit the TextFeaturizer (sentence embeddings + engineered features) on train.
  3. CATEGORY classifier  — model selection over {LogReg, RandomForest, MLP}
     by 5-fold macro-F1 CV on train; final report on the held-out test set.
  4. URGENCY classifier   — same selection procedure (4 classes).
  5. COST regressor        — selection over {GradientBoosting, RandomForest}
     (log-target) by CV MAE; report MAE & R² on test.
  6. RESOLUTION-TIME regressor — same procedure.
  7. Save every artifact to models/ (joblib) + metrics.json for the demo's
     "model performance" tab.

Department routing is a deterministic lookup (keywords.DEPARTMENT_MAP), and the
priority score is a transparent formula (see pipeline.py) — neither is an ML
model, by design.

Run:  python train.py
"""

import os
import json
import time
import warnings

import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
    mean_absolute_error, r2_score,
)

from features import TextFeaturizer, HANDCRAFTED_FEATURES

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
MODELS_DIR = os.path.join(HERE, "models")
SEED = 42
os.makedirs(MODELS_DIR, exist_ok=True)


def banner(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ---------------------------------------------------------------------------
# Classification: select best of 3 candidates by CV macro-F1, report on test.
# ---------------------------------------------------------------------------
def train_classifier(name, Xtr, ytr, Xte, yte, class_names):
    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=2000, C=2.0,
                                                  class_weight="balanced", random_state=SEED),
        # n_jobs=1 on the estimator: parallelism is handled by cross_val_score below,
        # so we avoid nested -1/-1 oversubscription (CPU²/memory blow-up -> OOM).
        "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=None,
                                               class_weight="balanced", random_state=SEED, n_jobs=1),
        "MLP": MLPClassifier(hidden_layer_sizes=(128,), max_iter=500,
                             early_stopping=True, random_state=SEED),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = {}
    for cname, clf in candidates.items():
        s = cross_val_score(clf, Xtr, ytr, cv=cv, scoring="f1_macro", n_jobs=-1)
        scores[cname] = float(np.mean(s))
        print(f"  [{name}] {cname:<18} CV macro-F1 = {scores[cname]:.3f}")
    best_name = max(scores, key=scores.get)
    best = candidates[best_name].fit(Xtr, ytr)
    print(f"  [{name}] -> selected: {best_name}")

    pred = best.predict(Xte)
    acc = accuracy_score(yte, pred)
    f1m = f1_score(yte, pred, average="macro")
    print(f"  [{name}] TEST accuracy = {acc:.3f}   macro-F1 = {f1m:.3f}")
    report = classification_report(yte, pred, target_names=class_names,
                                   output_dict=True, zero_division=0)
    print(classification_report(yte, pred, target_names=class_names, zero_division=0))
    cm = confusion_matrix(yte, pred).tolist()

    metrics = {
        "selected_model": best_name,
        "cv_macro_f1": scores,
        "test_accuracy": float(acc),
        "test_macro_f1": float(f1m),
        "classification_report": report,
        "confusion_matrix": cm,
        "class_names": list(class_names),
    }
    return best, metrics


# ---------------------------------------------------------------------------
# Regression: select best of 2 candidates by CV MAE (log target), report on test.
# ---------------------------------------------------------------------------
def train_regressor(name, Xtr, ytr, Xte, yte, unit):
    def wrap(est):
        # log1p target stabilises the heavy right-skew of cost/days.
        return TransformedTargetRegressor(regressor=est, func=np.log1p, inverse_func=np.expm1)

    candidates = {
        "GradientBoosting": wrap(GradientBoostingRegressor(random_state=SEED)),
        # max_depth caps tree size (regression trees grow very deep on continuous
        # targets) -> much smaller artifacts & bounded memory; n_jobs=1 avoids nesting.
        "RandomForest": wrap(RandomForestRegressor(n_estimators=300, max_depth=20,
                                                   random_state=SEED, n_jobs=1)),
    }
    scores = {}
    for cname, reg in candidates.items():
        s = cross_val_score(reg, Xtr, ytr, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1)
        scores[cname] = float(-np.mean(s))
        print(f"  [{name}] {cname:<18} CV MAE = {scores[cname]:,.0f} {unit}")
    best_name = min(scores, key=scores.get)
    best = candidates[best_name].fit(Xtr, ytr)
    print(f"  [{name}] -> selected: {best_name}")

    pred = np.clip(best.predict(Xte), 0, None)
    mae = mean_absolute_error(yte, pred)
    r2 = r2_score(yte, pred)
    print(f"  [{name}] TEST MAE = {mae:,.0f} {unit}   R² = {r2:.3f}")

    metrics = {
        "selected_model": best_name,
        "cv_mae": scores,
        "test_mae": float(mae),
        "test_r2": float(r2),
        "unit": unit,
    }
    return best, metrics


def main():
    t0 = time.time()
    banner("Blockait — training pipeline")

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    print(f"Loaded train={len(train_df)}  test={len(test_df)}")

    # --- Featurization -----------------------------------------------------
    # Backend is configurable: BLOCKAIT_BACKEND = auto | sbert | tfidf
    #   auto  -> sbert if sentence-transformers is installed, else tfidf
    #   tfidf -> lightweight, no downloads (fully self-contained demo)
    backend = os.environ.get("BLOCKAIT_BACKEND", "auto")
    print(f"\nFitting featurizer (BLOCKAIT_BACKEND={backend})...")
    feat = TextFeaturizer(backend=backend)
    Xtr = feat.fit_transform(train_df["text"].tolist())
    Xte = feat.transform(test_df["text"].tolist())
    print(f"Backend = {feat.resolved_backend}   feature dim = {Xtr.shape[1]} "
          f"(embedding={feat.embedding_dim_} + handcrafted={len(HANDCRAFTED_FEATURES)})")

    # --- Category classifier ----------------------------------------------
    banner("CATEGORY classifier")
    cat_enc = LabelEncoder().fit(train_df["category"])
    ytr_cat = cat_enc.transform(train_df["category"])
    yte_cat = cat_enc.transform(test_df["category"])
    cat_clf, cat_metrics = train_classifier("category", Xtr, ytr_cat, Xte, yte_cat,
                                            list(cat_enc.classes_))

    # --- Urgency classifier -----------------------------------------------
    banner("URGENCY classifier")
    urg_enc = LabelEncoder().fit(train_df["urgency"])
    ytr_urg = urg_enc.transform(train_df["urgency"])
    yte_urg = urg_enc.transform(test_df["urgency"])
    urg_clf, urg_metrics = train_classifier("urgency", Xtr, ytr_urg, Xte, yte_urg,
                                            list(urg_enc.classes_))

    # --- Cost regressor ----------------------------------------------------
    banner("COST regressor (KZT)")
    cost_reg, cost_metrics = train_regressor("cost", Xtr, train_df["cost_kzt"].values,
                                             Xte, test_df["cost_kzt"].values, "KZT")

    # --- Resolution-time regressor ----------------------------------------
    banner("RESOLUTION-TIME regressor (days)")
    days_reg, days_metrics = train_regressor("days", Xtr, train_df["resolution_days"].values,
                                             Xte, test_df["resolution_days"].values, "days")

    # --- Persist artifacts (compress=3 keeps tree models small enough to commit) -
    banner("Saving artifacts -> models/")
    joblib.dump(feat, os.path.join(MODELS_DIR, "featurizer.joblib"), compress=3)
    joblib.dump({"model": cat_clf, "encoder": cat_enc}, os.path.join(MODELS_DIR, "category_clf.joblib"), compress=3)
    joblib.dump({"model": urg_clf, "encoder": urg_enc}, os.path.join(MODELS_DIR, "urgency_clf.joblib"), compress=3)
    joblib.dump(cost_reg, os.path.join(MODELS_DIR, "cost_reg.joblib"), compress=3)
    joblib.dump(days_reg, os.path.join(MODELS_DIR, "days_reg.joblib"), compress=3)

    metrics = {
        "embedding_backend": feat.resolved_backend,
        "feature_dim": int(Xtr.shape[1]),
        "embedding_dim": int(feat.embedding_dim_),
        "n_handcrafted": len(HANDCRAFTED_FEATURES),
        "handcrafted_features": HANDCRAFTED_FEATURES,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "category": cat_metrics,
        "urgency": urg_metrics,
        "cost": cost_metrics,
        "resolution_days": days_metrics,
        "trained_seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # Per-category p2–p98 cost/days ranges — inference-time guardrails that clamp
    # the regressors to realistic values (see pipeline.py). Computed from the full
    # labelled set so they're stable.
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    category_ranges = {}
    for cat, g in full_df.groupby("category"):
        category_ranges[cat] = {
            "cost": [round(float(g["cost_kzt"].quantile(0.02)), -3),
                     round(float(g["cost_kzt"].quantile(0.98)), -3)],
            "days": [int(g["resolution_days"].quantile(0.02)),
                     int(round(g["resolution_days"].quantile(0.98)))],
        }
    with open(os.path.join(MODELS_DIR, "category_ranges.json"), "w", encoding="utf-8") as f:
        json.dump(category_ranges, f, ensure_ascii=False, indent=2)

    for fn in ["featurizer.joblib", "category_clf.joblib", "urgency_clf.joblib",
               "cost_reg.joblib", "days_reg.joblib", "metrics.json"]:
        size = os.path.getsize(os.path.join(MODELS_DIR, fn)) / 1e6
        print(f"  {fn:<22} {size:6.2f} MB")

    banner("SUMMARY")
    print(f"Embedding backend     : {feat.resolved_backend}")
    print(f"Category   accuracy/F1: {cat_metrics['test_accuracy']:.3f} / {cat_metrics['test_macro_f1']:.3f}"
          f"  ({cat_metrics['selected_model']})")
    print(f"Urgency    accuracy/F1: {urg_metrics['test_accuracy']:.3f} / {urg_metrics['test_macro_f1']:.3f}"
          f"  ({urg_metrics['selected_model']})")
    print(f"Cost       MAE / R²   : {cost_metrics['test_mae']:,.0f} KZT / {cost_metrics['test_r2']:.3f}"
          f"  ({cost_metrics['selected_model']})")
    print(f"Resolution MAE / R²   : {days_metrics['test_mae']:.1f} days / {days_metrics['test_r2']:.3f}"
          f"  ({days_metrics['selected_model']})")
    print(f"\nDone in {metrics['trained_seconds']}s. Artifacts in models/.")


if __name__ == "__main__":
    main()
