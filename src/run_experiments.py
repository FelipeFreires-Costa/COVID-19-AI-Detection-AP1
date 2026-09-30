"""Etapa 3 - grade descritor x modelo com CV aninhada no conjunto dev, selecao do melhor
modelo pela AUC macro media da CV (nunca pelo teste) e avaliacao unica no hold-out de teste.

Uso:  python src/run_experiments.py [--tag default] [--featuresets int,glcm,...] [--models ...]
Retomavel: combinacoes ja presentes em cv_folds.csv sao puladas.
"""
import argparse
import json
import time

import joblib
import pandas as pd

import config as C
from experiment import (feature_columns, fit_dev_eval_test, fmt, load_dataset, nested_cv,
                        summarize)

DEFAULT_FS = "int,glcm,lbp,gabor,hog,all"
DEFAULT_MODELS = "dummy,logreg,svm_rbf,rf,lgbm"


def write_main_table(summary, featuresets, models):
    models = [m for m in models if m != "dummy"]
    best = summary[summary["model"] != "dummy"]["auc_macro_mean"].max()
    lines = [
        "\\begin{table}[ht]", "\\centering",
        "\\caption{AUC-ROC macro (um-contra-todos) na valida\u00e7\u00e3o cruzada aninhada "
        "(5 dobras externas agrupadas por paciente), m\u00e9dia $\\pm$ desvio-padr\u00e3o.}",
        "\\label{tab:main}", "\\footnotesize", "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{l" + "c" * len(models) + "}", "\\hline",
        "Descritor & " + " & ".join(C.MODEL_NAMES[m] for m in models) + " \\\\", "\\hline",
    ]
    for fs in featuresets:
        cells = []
        for m in models:
            r = summary[(summary["featureset"] == fs) & (summary["model"] == m)]
            if r.empty:
                cells.append("--")
                continue
            r = r.iloc[0]
            cells.append(fmt(r["auc_macro_mean"], r["auc_macro_std"], bold=r["auc_macro_mean"] == best))
        lines.append(f"{C.FAMILY_NAMES.get(fs, fs)} & " + " & ".join(cells) + " \\\\")
    d = summary[summary["model"] == "dummy"]
    if not d.empty:
        d = d.iloc[0]
        lines.append("\\hline")
        lines.append(f"Baseline trivial (majorit\u00e1rio) & \\multicolumn{{{len(models)}}}{{c}}"
                     f"{{{fmt(d['auc_macro_mean'], d['auc_macro_std'])}}} \\\\")
    lines += ["\\hline", "\\end{tabular}", "\\end{table}"]
    (C.TAB_DIR / "table_main.tex").write_text("\n".join(lines), encoding="utf-8")


def write_test_table(dev_row, test_best, test_dummy, best_name):
    rows = [("auc_macro", "AUC-ROC macro"), ("ap_macro", "AUC-PR macro"),
            ("bal_acc", "Acur\u00e1cia balanceada"), ("f1_macro", "F1 macro"),
            ("bin_auc", "AUC-ROC bin\u00e1ria (pneumonia)"),
            ("sens_0", "Sensibilidade Negativo"), ("sens_1", "Sensibilidade T\u00edpico"),
            ("sens_2", "Sensibilidade Indeterminado"), ("sens_3", "Sensibilidade At\u00edpico")]
    lines = [
        "\\begin{table}[ht]", "\\centering",
        f"\\caption{{Melhor configura\u00e7\u00e3o ({best_name}) na CV do dev e no hold-out de teste, "
        "contra o baseline trivial.}", "\\label{tab:test}", "\\small",
        "\\begin{tabular}{lccc}", "\\hline",
        "M\u00e9trica & CV dev (m\u00e9dia $\\pm$ dp) & Teste & Trivial (teste) \\\\", "\\hline",
    ]
    for key, name in rows:
        dev_cell = fmt(dev_row[f"{key}_mean"], dev_row[f"{key}_std"]) if f"{key}_mean" in dev_row else "--"
        lines.append(f"{name} & {dev_cell} & {test_best[key]:.3f} & {test_dummy[key]:.3f} \\\\")
    lines += ["\\hline", "\\end{tabular}", "\\end{table}"]
    (C.TAB_DIR / "table_test.tex").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="default")
    ap.add_argument("--featuresets", default=DEFAULT_FS)
    ap.add_argument("--models", default=DEFAULT_MODELS)
    args = ap.parse_args()
    C.ensure_dirs()
    featuresets = args.featuresets.split(",")
    models = args.models.split(",")

    df = load_dataset(args.tag)
    print(f"Estudos: {len(df)} (dev={int((df.split == 'dev').sum())}, teste={int((df.split == 'test').sum())})")

    folds_path = C.TAB_DIR / "cv_folds.csv"
    all_folds = pd.read_csv(folds_path) if folds_path.exists() else pd.DataFrame()
    done = set(zip(all_folds.get("featureset", []), all_folds.get("model", [])))

    for fs in featuresets:
        cols = feature_columns(df, fs)
        if not cols:
            print(f"[pulando] {fs}: sem colunas")
            continue
        for m in models:
            fs_label = "-" if m == "dummy" else fs
            if (fs_label, m) in done:
                continue
            t0 = time.time()
            folds, oof = nested_cv(df, cols, m)
            folds["featureset"], folds["model"], folds["n_features"] = fs_label, m, len(cols)
            oof.to_csv(C.PRED_DIR / f"oof_{fs_label}_{m}.csv", index=False)
            all_folds = pd.concat([all_folds, folds], ignore_index=True)
            all_folds.to_csv(folds_path, index=False)
            done.add((fs_label, m))
            print(f"{fs:>6} | {m:>8} | AUC {folds.auc_macro.mean():.3f} +- {folds.auc_macro.std():.3f} "
                  f"| BAcc {folds.bal_acc.mean():.3f} | {time.time() - t0:.0f}s")

    metric_cols = [c for c in all_folds.columns
                   if c.startswith(("auc", "ap", "bal", "f1", "bin", "sens", "spec", "logloss"))]
    summary = summarize(all_folds, ["featureset", "model"], metrics=metric_cols)
    summary.to_csv(C.TAB_DIR / "cv_summary.csv", index=False)
    write_main_table(summary, featuresets, models)

    cand = summary[summary["model"] != "dummy"].sort_values("auc_macro_mean", ascending=False)
    best = cand.iloc[0]
    fs, m = best["featureset"], best["model"]
    cols = feature_columns(df, fs)
    print(f"\nMelhor na CV do dev: {fs} + {m} (AUC {best.auc_macro_mean:.3f})")

    est, preds, test_best = fit_dev_eval_test(df, cols, m)
    preds.to_csv(C.PRED_DIR / "test_preds_best.csv", index=False)
    _, _, test_dummy = fit_dev_eval_test(df, cols, "dummy")
    joblib.dump({"model": est, "cols": cols, "featureset": fs, "model_name": m}, C.WORK_DIR / "best_model.joblib")

    pd.DataFrame([{"config": f"{fs}+{m}", **test_best}, {"config": "dummy", **test_dummy}]).to_csv(
        C.TAB_DIR / "test_metrics.csv", index=False)
    write_test_table(best, test_best, test_dummy, f"{C.FAMILY_NAMES.get(fs, fs)} + {C.MODEL_NAMES[m]}")
    with open(C.TAB_DIR / "best_config.json", "w") as fh:
        json.dump({"featureset": fs, "model": m, "n_features": len(cols),
                   "best_params": str(getattr(est, "best_params_", {})),
                   "cv_auc_macro": float(best.auc_macro_mean),
                   "test_auc_macro": float(test_best["auc_macro"])}, fh, indent=2)
    print("Teste (hold-out):", {k: round(v, 3) for k, v in test_best.items()
                                if k in ("auc_macro", "ap_macro", "bal_acc", "f1_macro", "bin_auc")})


if __name__ == "__main__":
    main()
