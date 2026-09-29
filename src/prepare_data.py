"""Etapa 1 - dados.

1. Le os rotulos de estudo (4 classes mutuamente exclusivas) e de imagem (caixas de opacidade).
2. Amostragem estratificada por classe (opcional, TP1_SAMPLE_FRAC), com semente fixa.
3. Le cada DICOM: aplica Modality LUT (RescaleSlope/Intercept), VOI LUT (janela do
   equipamento), inverte MONOCHROME1, normaliza por percentis e salva PNG 512x512.
4. Extrai metadados DICOM relevantes (fabricante, modalidade, espacamento de pixel...).
5. Cria a particao congelada POR PACIENTE: 20% teste (hold-out) + 5 dobras no restante.

Uso:  python src/prepare_data.py
"""
import argparse
import os

import cv2
import numpy as np
import pandas as pd
import pydicom
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedGroupKFold

import config as C

try:
    from pydicom.pixels import apply_modality_lut, apply_voi_lut
except ImportError:  # pydicom < 3
    from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut

META_TAGS = [
    "PatientID", "PatientSex", "Modality", "Manufacturer", "BodyPartExamined",
    "PhotometricInterpretation", "BitsStored", "Rows", "Columns",
    "RescaleSlope", "RescaleIntercept", "WindowCenter", "WindowWidth",
]


def _tag(ds, name):
    v = ds.get(name, None)
    if v is None or v == "":
        return np.nan
    if isinstance(v, (pydicom.multival.MultiValue, list, tuple)):
        v = v[0] if len(v) else np.nan
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    return str(v)


def _pixel_spacing(ds):
    for name in ("PixelSpacing", "ImagerPixelSpacing"):
        v = ds.get(name, None)
        if v is not None and len(v):
            return float(v[0])
    return np.nan


def dicom_to_uint8(ds):
    arr = ds.pixel_array
    try:
        arr = apply_modality_lut(arr, ds)
        arr = apply_voi_lut(arr, ds)
    except Exception:
        pass
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr.mean(axis=-1)
    if ds.get("PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() - arr
    lo, hi = np.percentile(arr, [0.5, 99.5])
    arr = np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1)
    return (arr * 255).astype(np.uint8)


def process_image(image_id, path):
    out = C.IMG_DIR / f"{image_id}.png"
    rec = {"image_id": image_id, "ok": True, "error": ""}
    try:
        if out.exists():
            ds = pydicom.dcmread(path, stop_before_pixels=True)
        else:
            ds = pydicom.dcmread(path)
            img = dicom_to_uint8(ds)
            img = cv2.resize(img, (C.IMG_SIZE, C.IMG_SIZE), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out), img)
        for t in META_TAGS:
            rec[t] = _tag(ds, t)
        rec["PixelSpacing"] = _pixel_spacing(ds)
        rec["TransferSyntax"] = str(getattr(ds.file_meta, "TransferSyntaxUID", ""))
    except Exception as e:  # registra e segue; a contagem de falhas vai para o relatorio
        rec["ok"] = False
        rec["error"] = repr(e)[:200]
    return rec


def index_dicoms(train_dir):
    paths = {}
    for root, _, files in os.walk(train_dir):
        for f in files:
            if f.endswith(".dcm"):
                paths[f[:-4]] = os.path.join(root, f)
    return paths


def load_labels():
    study = pd.read_csv(C.DATA_DIR / "train_study_level.csv")
    study["study_id"] = study["id"].str.replace("_study", "", regex=False)
    study["label"] = study[C.CLASS_COLS].to_numpy().argmax(axis=1)
    study = study[["study_id", "label"]]

    img = pd.read_csv(C.DATA_DIR / "train_image_level.csv")
    img["image_id"] = img["id"].str.replace("_image", "", regex=False)
    img["n_boxes"] = img["label"].fillna("").str.count("opacity")
    img = img.rename(columns={"StudyInstanceUID": "study_id"})[["image_id", "study_id", "n_boxes"]]
    return study, img


def stratified_sample(study):
    if C.SAMPLE_FRAC >= 1.0:
        return study
    return (
        study.groupby("label", group_keys=False)
        .apply(lambda d: d.sample(frac=C.SAMPLE_FRAC, random_state=C.SEED))
        .reset_index(drop=True)
    )


def make_splits(st):
    st = st.reset_index(drop=True).copy()
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=C.SEED)
    _, te_idx = next(outer.split(st, st["label"], st["patient_id"]))
    st["split"] = "dev"
    st.loc[te_idx, "split"] = "test"
    st["cv_fold"] = -1
    dev = st[st["split"] == "dev"]
    inner = StratifiedGroupKFold(n_splits=C.N_OUTER, shuffle=True, random_state=C.SEED)
    for k, (_, va) in enumerate(inner.split(dev, dev["label"], dev["patient_id"])):
        st.loc[dev.index[va], "cv_fold"] = k
    te_pat = set(st.loc[st["split"] == "test", "patient_id"])
    dev_pat = set(st.loc[st["split"] == "dev", "patient_id"])
    assert not te_pat & dev_pat, "Vazamento: paciente presente em dev e teste"
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resplit", action="store_true", help="refaz a particao mesmo se ja existir")
    args = ap.parse_args()
    C.ensure_dirs()

    study, img = load_labels()
    print(f"Estudos no treino oficial: {len(study)} | imagens: {len(img)}")
    study = stratified_sample(study)
    img = img.merge(study, on="study_id", how="inner")
    print(f"Amostra (frac={C.SAMPLE_FRAC}): {len(study)} estudos, {len(img)} imagens")

    paths = index_dicoms(C.DATA_DIR / "train")
    img = img[img["image_id"].isin(paths)].copy()
    recs = Parallel(n_jobs=C.N_JOBS, verbose=5)(
        delayed(process_image)(i, paths[i]) for i in img["image_id"]
    )
    meta = img.merge(pd.DataFrame(recs), on="image_id", how="left")
    n_fail = int((~meta["ok"]).sum())
    print(f"Falhas de leitura DICOM: {n_fail}")
    if n_fail:
        meta.loc[~meta["ok"], ["image_id", "error"]].to_csv(C.TAB_DIR / "dicom_failures.csv", index=False)
    meta = meta[meta["ok"]].drop(columns=["ok", "error"])
    meta.to_csv(C.WORK_DIR / "image_meta.csv", index=False)

    st = (
        meta.sort_values("image_id")
        .groupby("study_id")
        .agg(
            label=("label", "first"),
            patient_id=("PatientID", "first"),
            n_images=("image_id", "count"),
            manufacturer=("Manufacturer", "first"),
            modality=("Modality", "first"),
        )
        .reset_index()
    )
    st["patient_id"] = st["patient_id"].fillna(st["study_id"])
    st["manufacturer"] = st["manufacturer"].fillna("Desconhecido")

    if C.SPLITS_FILE.exists() and not args.resplit:
        frozen = C.read_splits()
        print(f"Particao congelada reutilizada: {C.SPLITS_FILE.name} ({len(frozen)} estudos)")
        st = frozen[frozen["study_id"].isin(st["study_id"])].reset_index(drop=True)
    else:
        st = make_splits(st)
    st.to_csv(C.SPLITS_FILE, index=False)

    counts = pd.crosstab(st["label"].map(dict(enumerate(C.CLASS_NAMES))), st["split"], margins=True)
    counts.to_csv(C.TAB_DIR / "sample_counts.csv")
    print(counts)
    print(f"Pacientes unicos: {st['patient_id'].nunique()} | estudos: {len(st)}")


if __name__ == "__main__":
    main()
