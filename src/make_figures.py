"""Etapa 5 - figuras do artigo e analise de erro.

  fig_masks.png        controle de qualidade da mascara pulmonar
  fig_roc_cm.png       curvas ROC por classe + matriz de confusao (hold-out de teste)
  fig_importance.png   importancia por familia (permutacao em grupo) e top-20 atributos (LightGBM)
  fig_errors.png       acertos e erros mais confiantes por classe
  error_by_group.csv   desempenho por fabricante/modalidade (variabilidade entre instituicoes)

Uso:  python src/make_figures.py [--only masks]
"""
import argparse

import cv2
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize

import config as C
from experiment import feature_columns, load_dataset
from preprocessing import lung_mask

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]


def _img(image_id):
    return cv2.imread(str(C.IMG_DIR / f"{image_id}.png"), cv2.IMREAD_GRAYSCALE)


def fig_masks(meta, n=12):
    ids = meta.sample(n=min(n, len(meta)), random_state=C.SEED)["image_id"]
    fig, axes = plt.subplots(2, n // 2, figsize=(13, 5))
    for ax, iid in zip(axes.ravel(), ids):
        img = _img(iid)
        mask, fb = lung_mask(img)
        ax.imshow(img, cmap="gray")
        ax.contour(mask, levels=[0.5], colors="red" if fb else "lime", linewidths=1)
        ax.set_title("fallback" if fb else "ok", fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(C.FIG_DIR / "fig_masks.png", dpi=120)
    plt.close()


def fig_roc_cm(preds):
    y = preds["label"].to_numpy()
    P = preds[[f"p{k}" for k in range(C.N_CLASSES)]].to_numpy()
    Y = label_binarize(y, classes=list(range(C.N_CLASSES)))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for k, name in enumerate(C.CLASS_NAMES):
        fpr, tpr, _ = roc_curve(Y[:, k], P[:, k])
        axes[0].plot(fpr, tpr, color=COLORS[k], label=f"{name} (AUC={roc_auc_score(Y[:, k], P[:, k]):.2f})")
    yb = (y != 0).astype(int)
    fpr, tpr, _ = roc_curve(yb, 1 - P[:, 0])
    axes[0].plot(fpr, tpr, "k--", label=f"Pneumonia vs. neg. (AUC={roc_auc_score(yb, 1 - P[:, 0]):.2f})")
    axes[0].plot([0, 1], [0, 1], color="gray", lw=0.8)
    axes[0].set_xlabel("1 - Especificidade")
    axes[0].set_ylabel("Sensibilidade")
    axes[0].legend(fontsize=8, loc="lower right")
    axes[0].set_title("ROC no hold-out de teste")
    cm = confusion_matrix(y, P.argmax(1), labels=list(range(C.N_CLASSES)), normalize="true")
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", cbar=False, ax=axes[1],
                xticklabels=C.CLASS_NAMES, yticklabels=C.CLASS_NAMES)
    axes[1].set_xlabel("Predito")
    axes[1].set_ylabel("Verdadeiro")
    axes[1].set_title("Matriz de confus\u00e3o (normalizada por linha)")
    plt.tight_layout()
    plt.savefig(C.FIG_DIR / "fig_roc_cm.png", dpi=200)
    plt.close()


def fig_importance(df, bundle, n_repeats=10):
    test = df[df["split"] == "test"].reset_index(drop=True)
    dev = df[df["split"] == "dev"].reset_index(drop=True)
    cols, model = bundle["cols"], bundle["model"]
    Xt = test[cols].to_numpy(np.float32)
    Yt = label_binarize(test["label"], classes=list(range(C.N_CLASSES)))
    base = roc_auc_score(Yt, model.predict_proba(Xt), average="macro")
    rng = np.random.default_rng(C.SEED)
    fams = [f for f in C.FAMILIES if any(c.startswith(f + "_") for c in cols)]
    imp = []
    for fam in fams:
        idx = [i for i, c in enumerate(cols) if c.startswith(fam + "_")]
        for _ in range(n_repeats):
            Xp = Xt.copy()
            Xp[:, idx] = Xt[rng.permutation(len(Xt))][:, idx]
            imp.append({"familia": C.FAMILY_NAMES[fam],
                        "queda_auc": base - roc_auc_score(Yt, model.predict_proba(Xp), average="macro")})
    imp = pd.DataFrame(imp)
    imp.to_csv(C.TAB_DIR / "importance_family.csv", index=False)

    all_cols = feature_columns(df, "all")
    lgb = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, colsample_bytree=0.5,
                         class_weight="balanced", importance_type="gain", random_state=C.SEED, verbose=-1)
    lgb.fit(dev[all_cols].to_numpy(np.float32), dev["label"])
    gain = pd.Series(lgb.feature_importances_, index=all_cols).sort_values(ascending=False)
    gain.to_csv(C.TAB_DIR / "importance_lgbm_gain.csv")
    top = gain.head(20)[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [1, 1.4]})
    if len(fams) > 1:
        sns.barplot(data=imp, x="queda_auc", y="familia", ax=axes[0], color="#4C72B0", errorbar="sd")
        axes[0].set_title("Permuta\u00e7\u00e3o por fam\u00edlia (teste)")
        axes[0].set_xlabel("Queda na AUC macro")
        axes[0].set_ylabel("")
    else:
        axes[0].axis("off")
    fam_color = {f: COLORS[i] for i, f in enumerate(C.FAMILIES)}
    axes[1].barh(top.index, top.values, color=[fam_color[c.split("_")[0]] for c in top.index])
    axes[1].set_title("Top-20 atributos (ganho LightGBM, dev)")
    axes[1].tick_params(axis="y", labelsize=7)
    plt.tight_layout()
    plt.savefig(C.FIG_DIR / "fig_importance.png", dpi=200)
    plt.close()


def fig_errors(preds, meta):
    first_img = meta.sort_values("image_id").groupby("study_id")["image_id"].first()
    P = preds[[f"p{k}" for k in range(C.N_CLASSES)]].to_numpy()
    preds = preds.assign(pred=P.argmax(1), conf=P.max(1), image_id=preds["study_id"].map(first_img))
    fig, axes = plt.subplots(C.N_CLASSES, 4, figsize=(10, 10.5))
    for k in range(C.N_CLASSES):
        sub = preds[preds["label"] == k]
        right = sub[sub["pred"] == k].sort_values("conf", ascending=False).head(2)
        wrong = sub[sub["pred"] != k].sort_values("conf", ascending=False).head(2)
        for j, (_, r) in enumerate(pd.concat([right, wrong]).iterrows()):
            ax = axes[k, j]
            ax.imshow(_img(r["image_id"]), cmap="gray")
            pred = int(r["pred"])
            ax.set_title(f"V:{C.CLASS_NAMES[k]} P:{C.CLASS_NAMES[pred]} ({r['conf']:.2f})",
                         fontsize=7, color="green" if pred == k else "red")
        for ax in axes[k]:
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(C.FIG_DIR / "fig_errors.png", dpi=150)
    plt.close()
    preds.to_csv(C.TAB_DIR / "test_errors_detail.csv", index=False)


def error_by_group(preds, splits, feats_img):
    df = preds.merge(splits[["study_id", "manufacturer", "modality", "n_images"]], on="study_id")
    fb = feats_img.groupby("study_id")["qc_mask_fallback"].max()
    df["mask_fallback"] = df["study_id"].map(fb)
    P = df[[f"p{k}" for k in range(C.N_CLASSES)]].to_numpy()
    df["correct"] = (P.argmax(1) == df["label"]).astype(int)
    df["p_pneumonia"] = 1 - df["p0"]
    out = []
    for col in ("manufacturer", "modality", "mask_fallback"):
        for val, g in df.groupby(col):
            yb = (g["label"] != 0).astype(int)
            out.append({"variavel": col, "valor": val, "n": len(g), "acuracia": g["correct"].mean(),
                        "auc_binaria": roc_auc_score(yb, g["p_pneumonia"]) if yb.nunique() == 2 else np.nan,
                        "prev_pneumonia": yb.mean()})
    pd.DataFrame(out).to_csv(C.TAB_DIR / "error_by_group.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="'masks' gera so o QC da mascara")
    args = ap.parse_args()
    C.ensure_dirs()
    meta = C.read_meta()
    fig_masks(meta)
    if args.only == "masks":
        return
    preds = pd.read_csv(C.PRED_DIR / "test_preds_best.csv", dtype=C.ID_DTYPES)
    bundle = joblib.load(C.WORK_DIR / "best_model.joblib")
    df = load_dataset("default")
    fig_roc_cm(preds)
    fig_importance(df, bundle)
    fig_errors(preds, meta)
    error_by_group(preds, C.read_splits(), pd.read_pickle(C.WORK_DIR / "features_img_default.pkl"))
    print("Figuras em", C.FIG_DIR)


if __name__ == "__main__":
    main()
