"""Etapa 2 - extracao de caracteristicas por imagem e agregacao por estudo (media).

Uso:
  python src/extract_features.py --tag default --clahe 1 --mask 1      # pipeline principal
  python src/extract_features.py --tag raw_full --clahe 0 --mask 0     # ablacao
"""
import argparse
import time

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import config as C
from features import FEATURE_FUNCS
from preprocessing import apply_clahe, lung_mask


def process(image_id, use_clahe, use_mask, families):
    img = cv2.imread(str(C.IMG_DIR / f"{image_id}.png"), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    if use_mask:
        mask, fallback = lung_mask(img)
    else:
        mask, fallback = np.ones_like(img, dtype=bool), False
    work = apply_clahe(img) if use_clahe else img
    feats = {"image_id": image_id, "qc_mask_frac": float(mask.mean()), "qc_mask_fallback": int(fallback)}
    for fam in families:
        feats.update(FEATURE_FUNCS[fam](work, mask))
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="default")
    ap.add_argument("--clahe", type=int, default=1)
    ap.add_argument("--mask", type=int, default=1)
    ap.add_argument("--families", default=",".join(C.FAMILIES))
    args = ap.parse_args()
    families = args.families.split(",")
    C.ensure_dirs()

    meta = C.read_meta()
    t0 = time.time()
    rows = Parallel(n_jobs=C.N_JOBS, verbose=5, batch_size=16)(
        delayed(process)(i, bool(args.clahe), bool(args.mask), families) for i in meta["image_id"]
    )
    rows = [r for r in rows if r is not None]
    fi = pd.DataFrame(rows).merge(meta[["image_id", "study_id"]], on="image_id")
    print(f"{len(fi)} imagens, {fi.shape[1] - 2} colunas, {time.time() - t0:.0f}s")
    fi.to_pickle(C.WORK_DIR / f"features_img_{args.tag}.pkl")

    num_cols = [c for c in fi.columns if c not in ("image_id", "study_id")]
    fs = fi.groupby("study_id")[num_cols].mean().reset_index()
    fs.to_pickle(C.WORK_DIR / f"features_study_{args.tag}.pkl")
    print(f"Estudos: {len(fs)} | fallback de mascara: {fi['qc_mask_fallback'].mean():.1%}")
    pd.DataFrame([{
        "tag": args.tag, "clahe": args.clahe, "mask": args.mask, "n_images": len(fi),
        "n_studies": len(fs), "mask_fallback_rate": fi["qc_mask_fallback"].mean(),
        "mask_frac_mean": fi["qc_mask_frac"].mean(), "seconds": time.time() - t0,
    }]).to_csv(C.TAB_DIR / f"extraction_{args.tag}.csv", index=False)


if __name__ == "__main__":
    main()
