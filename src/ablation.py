"""Etapa 4 - ablacoes.

(a) Pre-processamento: CLAHE (sim/nao) x mascara pulmonar (sim/nao), com o melhor modelo
    e todas as familias disponiveis. Requer as extracoes com as tags raw_full, clahe_full,
    raw_mask e default.
(b) Desbalanceamento: nenhum tratamento vs. pesos de classe vs. SMOTE, com Regressao
    Logistica sobre o melhor conjunto de descritores.

Uso:  python src/ablation.py [--part prep|imb|all]
"""
import argparse
import json

import pandas as pd

import config as C
from experiment import feature_columns, fmt, load_dataset, nested_cv, summarize

PREP_TAGS = [("raw_full", 0, 0), ("clahe_full", 1, 0), ("raw_mask", 0, 1), ("default", 1, 1)]
ABL_METRICS = ["auc_macro", "ap_macro", "bal_acc", "f1_macro", "sens_3"]


def _best():
    p = C.TAB_DIR / "best_config.json"
    if p.exists():
        with open(p) as fh:
            return json.load(fh)
    return {"featureset": "all", "model": "lgbm"}


def _latex(summary, label_col, labels, caption, label):
    head = ["AUC-ROC", "AUC-PR", "Acur. bal.", "F1 macro", "Sens. At\u00edpico"]
    lines = ["\\begin{table}[ht]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\small", "\\begin{tabular}{l" + "c" * len(head) + "}", "\\hline",
             "Configura\u00e7\u00e3o & " + " & ".join(head) + " \\\\", "\\hline"]
    for _, r in summary.iterrows():
        cells = [fmt(r[f"{m}_mean"], r[f"{m}_std"]) for m in ABL_METRICS]
        lines.append(f"{labels.get(r[label_col], r[label_col])} & " + " & ".join(cells) + " \\\\")
    lines += ["\\hline", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def ablation_prep(model):
    rows = []
    for tag, clahe, mask in PREP_TAGS:
        if not (C.WORK_DIR / f"features_study_{tag}.pkl").exists():
            print(f"[pulando] {tag}: rode extract_features.py --tag {tag} --clahe {clahe} --mask {mask}")
            continue
        df = load_dataset(tag)
        cols = feature_columns(df, "all")
        folds, _ = nested_cv(df, cols, model)
        folds["tag"] = tag
        rows.append(folds)
        print(f"{tag:>10} | AUC {folds.auc_macro.mean():.3f} +- {folds.auc_macro.std():.3f}")
    if not rows:
        return
    folds = pd.concat(rows, ignore_index=True)
    folds.to_csv(C.TAB_DIR / "ablation_prep_folds.csv", index=False)
    s = summarize(folds, ["tag"], ABL_METRICS)
    s["order"] = s["tag"].map({t[0]: i for i, t in enumerate(PREP_TAGS)})
    s = s.sort_values("order").drop(columns="order")
    s.to_csv(C.TAB_DIR / "ablation_prep.csv", index=False)
    labels = {"raw_full": "Sem CLAHE, imagem inteira", "clahe_full": "CLAHE, imagem inteira",
              "raw_mask": "Sem CLAHE, m\u00e1scara pulmonar", "default": "CLAHE + m\u00e1scara (padr\u00e3o)"}
    (C.TAB_DIR / "table_ablation_prep.tex").write_text(_latex(
        s, "tag", labels,
        f"Abla\u00e7\u00e3o do pr\u00e9-processamento (todas as fam\u00edlias, {C.MODEL_NAMES[model]}).",
        "tab:abl_prep"), encoding="utf-8")


def ablation_imb(featureset):
    df = load_dataset("default")
    cols = feature_columns(df, featureset)
    rows = []
    for imb in ("none", "balanced", "smote"):
        folds, _ = nested_cv(df, cols, "logreg", imbalance=imb)
        folds["imbalance"] = imb
        rows.append(folds)
        print(f"{imb:>9} | BAcc {folds.bal_acc.mean():.3f} | Sens Atipico {folds.sens_3.mean():.3f}")
    folds = pd.concat(rows, ignore_index=True)
    folds.to_csv(C.TAB_DIR / "ablation_imb_folds.csv", index=False)
    s = summarize(folds, ["imbalance"], ABL_METRICS)
    s["order"] = s["imbalance"].map({"none": 0, "balanced": 1, "smote": 2})
    s = s.sort_values("order").drop(columns="order")
    s.to_csv(C.TAB_DIR / "ablation_imb.csv", index=False)
    labels = {"none": "Sem tratamento", "balanced": "Pesos de classe", "smote": "SMOTE"}
    (C.TAB_DIR / "table_ablation_imb.tex").write_text(_latex(
        s, "imbalance", labels,
        f"Tratamento do desbalanceamento (Reg. Log\u00edstica, {C.FAMILY_NAMES.get(featureset, featureset)}).",
        "tab:abl_imb"), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["prep", "imb", "all"])
    args = ap.parse_args()
    C.ensure_dirs()
    best = _best()
    if args.part in ("prep", "all"):
        ablation_prep(best["model"])
    if args.part in ("imb", "all"):
        ablation_imb(best["featureset"])


if __name__ == "__main__":
    main()
