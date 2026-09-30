"""Etapa 6 - preenche o artigo com os resultados.

Gera paper/numbers.tex (macros usadas no texto), copia as tabelas e a figura principal
para paper/ e empacota o projeto LaTeX em TP1_G4_paper_overleaf.zip (upload direto no Overleaf).

Uso:  python src/fill_paper.py [--paper-dir paper]
"""
import argparse
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

import config as C

TEX_ACCENTS = {"\u00e1": "\\'a", "\u00e9": "\\'e", "\u00ed": "\\'{\\i}", "\u00f3": "\\'o", "\u00fa": "\\'u",
               "\u00e2": "\\^a", "\u00ea": "\\^e", "\u00f4": "\\^o", "\u00e3": "\\~a", "\u00f5": "\\~o",
               "\u00e7": "\\c{c}", "\u00c1": "\\'A", "\u00c9": "\\'E"}


def tex(s):
    s = str(s).replace("_", "\\_").replace("%", "\\%")
    for k, v in TEX_ACCENTS.items():
        s = s.replace(k, v)
    return s


def num(x, nd=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:.{nd}f}".replace(".", "{,}")


def pct(x, nd=1):
    return "--" if x is None or np.isnan(x) else f"{100 * x:.{nd}f}".replace(".", "{,}") + "\\%"


def signed(x):
    return "--" if x is None or np.isnan(x) else ("+" if x >= 0 else "$-$") + num(abs(x))


def _read(name):
    p = C.TAB_DIR / name
    return pd.read_csv(p) if p.exists() else None


def collect():
    m = {}
    st = C.read_splits()
    if (C.WORK_DIR / "image_meta.csv").exists():
        n_images = len(C.read_meta())
    else:
        eda = (C.TAB_DIR / "eda_summary.txt").read_text(encoding="utf-8", errors="replace")
        n_images = int(next(l for l in eda.splitlines() if l.startswith("Imagens:")).split(":")[1])
    m["NStudies"] = f"{len(st):,}".replace(",", ".")
    m["NImages"] = f"{n_images:,}".replace(",", ".")
    m["NPatients"] = f"{st['patient_id'].nunique():,}".replace(",", ".")
    m["NDev"] = f"{int((st['split'] == 'dev').sum()):,}".replace(",", ".")
    m["NTest"] = f"{int((st['split'] == 'test').sum()):,}".replace(",", ".")
    freq = st["label"].value_counts(normalize=True)
    for k, key in enumerate(["PctNeg", "PctTyp", "PctInd", "PctAty"]):
        m[key] = pct(freq.get(k, 0.0))

    ext = _read("extraction_default.csv")
    m["MaskFallback"] = pct(ext["mask_fallback_rate"].iloc[0]) if ext is not None else "--"

    summ = _read("cv_summary.csv")
    models = summ[summ["model"] != "dummy"]
    single = models[models["featureset"] != "all"].sort_values("auc_macro_mean", ascending=False).iloc[0]
    m["BestFamily"] = tex(C.FAMILY_NAMES.get(single["featureset"], single["featureset"]))
    m["BestFamilyAUC"] = num(single["auc_macro_mean"])
    allrows = models[models["featureset"] == "all"]
    m["AllBestAUC"] = num(allrows["auc_macro_mean"].max()) if len(allrows) else "--"
    by_model = models.groupby("model")["auc_macro_mean"].mean().sort_values(ascending=False)
    m["BestModelAvg"] = tex(C.MODEL_NAMES[by_model.index[0]])
    m["WorstModelAvg"] = tex(C.MODEL_NAMES[by_model.index[-1]])

    with open(C.TAB_DIR / "best_config.json") as fh:
        best = json.load(fh)
    m["BestConfig"] = tex(f"{C.FAMILY_NAMES.get(best['featureset'], best['featureset'])} + {C.MODEL_NAMES[best['model']]}")
    brow = models[(models["featureset"] == best["featureset"]) & (models["model"] == best["model"])].iloc[0]
    m["DevAUC"] = f"{num(brow['auc_macro_mean'])} $\\pm$ {num(brow['auc_macro_std'])}"
    m["DevBalAcc"] = num(brow["bal_acc_mean"])

    test = _read("test_metrics.csv").set_index("config")
    tb = test.iloc[0]
    m["TestAUC"] = num(tb["auc_macro"])
    m["TestAUCEn"] = f"{tb['auc_macro']:.3f}"
    m["TestAP"] = num(tb["ap_macro"])
    m["TestBalAcc"] = num(tb["bal_acc"])
    m["TestFone"] = num(tb["f1_macro"])
    m["TestBinAUC"] = num(tb["bin_auc"])
    m["TestAPDummy"] = num(test.loc["dummy", "ap_macro"])

    preds = pd.read_csv(C.PRED_DIR / "test_preds_best.csv", dtype=C.ID_DTYPES)
    P = preds[[f"p{k}" for k in range(C.N_CLASSES)]].to_numpy()
    cm = confusion_matrix(preds["label"], P.argmax(1), labels=list(range(C.N_CLASSES)), normalize="true")
    m["IndToTyp"] = pct(cm[2, 1])
    m["AtyToTyp"] = pct(cm[3, 1])
    m["SensNeg"], m["SensTyp"], m["SensInd"], m["SensAty"] = (pct(cm[k, k]) for k in range(4))

    abl = _read("ablation_prep.csv")
    if abl is not None:
        a = abl.set_index("tag")["auc_macro_mean"]
        m["AblMaskDelta"] = signed(a.get("default", np.nan) - a.get("clahe_full", np.nan))
        m["AblClaheDelta"] = signed(a.get("default", np.nan) - a.get("raw_mask", np.nan))
        m["AblRawAUC"] = num(a.get("raw_full", np.nan))
        m["AblDefaultAUC"] = num(a.get("default", np.nan))
    imb = _read("ablation_imb.csv")
    if imb is not None:
        i = imb.set_index("imbalance")
        for key, name in (("none", "None"), ("balanced", "Bal"), ("smote", "Smote")):
            m[f"SensAty{name}"] = num(i.loc[key, "sens_3_mean"]) if key in i.index else "--"
            m[f"BalAcc{name}"] = num(i.loc[key, "bal_acc_mean"]) if key in i.index else "--"
            m[f"AUCImb{name}"] = num(i.loc[key, "auc_macro_mean"]) if key in i.index else "--"

    grp = _read("error_by_group.csv")
    if grp is not None:
        fb = grp[grp["variavel"] == "mask_fallback"].copy()
        fb.index = fb["valor"].map(lambda v: str(int(float(v))))
        m["AccMaskOk"] = num(fb.loc["0", "acuracia"]) if "0" in fb.index else "--"
        m["AccMaskFb"] = num(fb.loc["1", "acuracia"]) if "1" in fb.index else "--"
        m["PrevPneuMaskOk"] = pct(fb.loc["0", "prev_pneumonia"]) if "0" in fb.index else "--"
        m["PrevPneuMaskFb"] = pct(fb.loc["1", "prev_pneumonia"]) if "1" in fb.index else "--"
        mod = grp[grp["variavel"] == "modality"].set_index("valor")
        for key in ("CR", "DX"):
            m[f"AccMod{key}"] = num(mod.loc[key, "acuracia"]) if key in mod.index else "--"
            m[f"BinAUCMod{key}"] = num(mod.loc[key, "auc_binaria"]) if key in mod.index else "--"

    fam = _read("importance_family.csv")
    if fam is not None and fam["familia"].nunique() > 1:
        f = fam.groupby("familia")["queda_auc"].mean().sort_values(ascending=False)
        m["TopFamily"] = tex(f.index[0])
        m["TopFamilyDrop"] = num(f.iloc[0])
    else:
        m["TopFamily"], m["TopFamilyDrop"] = m["BestFamily"], "--"
    gain = pd.read_csv(C.TAB_DIR / "importance_lgbm_gain.csv", index_col=0)
    m["TopFeature"] = tex(gain.index[0])
    return m


def write_numbers(m, path):
    lines = ["% Gerado automaticamente por src/fill_paper.py - nao editar a mao."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(m.items())]
    path.write_text("\n".join(lines) + "\n", encoding="ascii", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-dir", default=str(C.ROOT / "paper"))
    args = ap.parse_args()
    paper = Path(args.paper_dir)
    (paper / "tables").mkdir(parents=True, exist_ok=True)
    (paper / "figures").mkdir(parents=True, exist_ok=True)

    write_numbers(collect(), paper / "numbers.tex")
    for t in ("table_main.tex", "table_test.tex", "table_ablation_prep.tex", "table_ablation_imb.tex"):
        if (C.TAB_DIR / t).exists():
            shutil.copy(C.TAB_DIR / t, paper / "tables" / t)
    for f in ("fig_roc_cm.png", "fig_importance.png", "fig_errors.png", "fig_masks.png",
              "eda_overview.png", "eda_examples.png"):
        if (C.FIG_DIR / f).exists():
            shutil.copy(C.FIG_DIR / f, paper / "figures" / f)

    out = C.RESULTS_DIR / "TP1_G4_paper_overleaf.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in paper.rglob("*"):
            if p.is_file() and p.suffix not in (".aux", ".log", ".bbl", ".blg", ".out", ".pdf"):
                z.write(p, p.relative_to(paper))
    print(f"numbers.tex atualizado; projeto LaTeX empacotado em {out}")


if __name__ == "__main__":
    main()
