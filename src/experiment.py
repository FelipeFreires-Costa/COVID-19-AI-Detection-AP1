"""Nucleo experimental: modelos, metricas e validacao cruzada aninhada agrupada por paciente.

Todo passo ajustado a dados (imputacao, padronizacao, PCA, SMOTE) esta dentro do Pipeline,
logo e ajustado apenas na particao de treino de cada dobra.
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, log_loss, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC

import config as C

MAIN_METRICS = ["auc_macro", "ap_macro", "bal_acc", "f1_macro", "bin_auc", "logloss"]


def load_dataset(tag="default"):
    feats = pd.read_pickle(C.WORK_DIR / f"features_study_{tag}.pkl")
    splits = C.read_splits()
    return splits.merge(feats, on="study_id", how="inner").reset_index(drop=True)


def feature_columns(df, featureset):
    fams = C.FAMILIES if featureset == "all" else featureset.split("+")
    return [c for c in df.columns if c.split("_")[0] in fams]


def make_model(name, n_features, imbalance="balanced"):
    cw = "balanced" if imbalance == "balanced" else None
    steps = [
        ("imp", SimpleImputer(strategy="median")),
        ("var", VarianceThreshold(0.0)),
        ("sc", StandardScaler()),
    ]
    if n_features > 200:
        steps.append(("pca", PCA(n_components=100, random_state=C.SEED)))

    if name == "dummy":
        clf, grid = DummyClassifier(strategy="prior"), {}
    elif name == "logreg":
        clf = LogisticRegression(max_iter=5000, class_weight=cw)
        grid = {"clf__C": [0.01, 0.1, 1.0]}
    elif name == "svm_rbf":
        clf = SVC(kernel="rbf", probability=True, class_weight=cw, random_state=C.SEED)
        grid = {"clf__C": [0.3, 1.0, 3.0]}
    elif name == "rf":
        clf = RandomForestClassifier(n_estimators=300, n_jobs=1, random_state=C.SEED,
                                     class_weight="balanced_subsample" if cw else None)
        grid = {"clf__max_depth": [None, 12], "clf__min_samples_leaf": [1, 5]}
    elif name == "lgbm":
        clf = LGBMClassifier(n_estimators=300, learning_rate=0.05, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.5, class_weight=cw, random_state=C.SEED,
                             n_jobs=1, verbose=-1)
        grid = {"clf__num_leaves": [15, 31], "clf__min_child_samples": [20, 50]}
    else:
        raise ValueError(name)

    if imbalance == "smote":
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline
        steps.append(("smote", SMOTE(random_state=C.SEED)))
        steps.append(("clf", clf))
        return ImbPipeline(steps), grid
    steps.append(("clf", clf))
    return Pipeline(steps), grid


def fit_with_search(pipe, grid, X, y, groups):
    if not grid:
        return pipe.fit(X, y)
    inner = StratifiedGroupKFold(n_splits=C.N_INNER, shuffle=True, random_state=C.SEED)
    gs = GridSearchCV(pipe, grid, scoring="roc_auc_ovr", cv=inner, n_jobs=C.N_JOBS, refit=True)
    gs.fit(X, y, groups=groups)
    return gs


def compute_metrics(y, proba):
    y = np.asarray(y)
    proba = np.clip(proba, 1e-7, 1)
    proba = proba / proba.sum(axis=1, keepdims=True)
    pred = proba.argmax(axis=1)
    Y = label_binarize(y, classes=list(range(C.N_CLASSES)))
    m = {
        "auc_macro": roc_auc_score(Y, proba, average="macro"),
        "ap_macro": average_precision_score(Y, proba, average="macro"),
        "bal_acc": balanced_accuracy_score(y, pred),
        "f1_macro": f1_score(y, pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y, pred, average="weighted", zero_division=0),
        "logloss": log_loss(y, proba, labels=list(range(C.N_CLASSES))),
    }
    cm = confusion_matrix(y, pred, labels=list(range(C.N_CLASSES)))
    for k in range(C.N_CLASSES):
        tp = cm[k, k]
        fn = cm[k].sum() - tp
        fp = cm[:, k].sum() - tp
        tn = cm.sum() - tp - fn - fp
        m[f"sens_{k}"] = tp / (tp + fn) if tp + fn else np.nan
        m[f"spec_{k}"] = tn / (tn + fp) if tn + fp else np.nan
        m[f"auc_{k}"] = roc_auc_score(Y[:, k], proba[:, k])
        m[f"ap_{k}"] = average_precision_score(Y[:, k], proba[:, k])
    # Tarefa binaria derivada: pneumonia (qualquer aparencia) vs. negativo
    yb = (y != 0).astype(int)
    pb = 1 - proba[:, 0]
    m["bin_auc"] = roc_auc_score(yb, pb)
    m["bin_ap"] = average_precision_score(yb, pb)
    pbin = (pb >= 0.5).astype(int)
    m["bin_sens"] = ((pbin == 1) & (yb == 1)).sum() / max((yb == 1).sum(), 1)
    m["bin_spec"] = ((pbin == 0) & (yb == 0)).sum() / max((yb == 0).sum(), 1)
    return m


def _xyg(df, cols):
    return df[cols].to_numpy(np.float32), df["label"].to_numpy(), df["patient_id"].astype(str).to_numpy()


def nested_cv(df, cols, model_name, imbalance="balanced"):
    """CV externa = 5 dobras congeladas (splits.csv) do conjunto dev; CV interna = 3 dobras
    agrupadas por paciente para busca de hiperparametros. Retorna (metricas por dobra, OOF)."""
    dev = df[df["split"] == "dev"].reset_index(drop=True)
    X, y, g = _xyg(dev, cols)
    folds = dev["cv_fold"].to_numpy()
    oof = np.zeros((len(dev), C.N_CLASSES))
    rows = []
    for k in range(C.N_OUTER):
        tr, va = folds != k, folds == k
        pipe, grid = make_model(model_name, len(cols), imbalance)
        est = fit_with_search(pipe, grid, X[tr], y[tr], g[tr])
        p = est.predict_proba(X[va])
        oof[va] = p
        m = compute_metrics(y[va], p)
        m["fold"] = k
        m["best_params"] = str(getattr(est, "best_params_", {}))
        rows.append(m)
    oof_df = pd.DataFrame(oof, columns=[f"p{k}" for k in range(C.N_CLASSES)])
    oof_df.insert(0, "study_id", dev["study_id"])
    oof_df["label"] = y
    oof_df["cv_fold"] = folds
    return pd.DataFrame(rows), oof_df


def fit_dev_eval_test(df, cols, model_name, imbalance="balanced"):
    dev = df[df["split"] == "dev"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    Xd, yd, gd = _xyg(dev, cols)
    Xt, yt, _ = _xyg(test, cols)
    pipe, grid = make_model(model_name, len(cols), imbalance)
    est = fit_with_search(pipe, grid, Xd, yd, gd)
    p = est.predict_proba(Xt)
    preds = pd.DataFrame(p, columns=[f"p{k}" for k in range(C.N_CLASSES)])
    preds.insert(0, "study_id", test["study_id"])
    preds["label"] = yt
    return est, preds, compute_metrics(yt, p)


def summarize(folds, keys, metrics=MAIN_METRICS):
    agg = folds.groupby(keys)[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return agg.reset_index()


def fmt(mean, std, bold=False):
    s = f"{mean:.3f} $\\pm$ {std:.3f}"
    return f"\\textbf{{{s}}}" if bold else s
