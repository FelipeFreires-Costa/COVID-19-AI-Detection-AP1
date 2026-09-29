"""Gera um conjunto SINTETICO com a mesma estrutura da competicao SIIM-FISABIO-RSNA
(train/<study>/<series>/<image>.dcm + train_study_level.csv + train_image_level.csv),
usado apenas para testar o pipeline de ponta a ponta. Nao tem valor cientifico.

Uso:  python tests/make_synthetic_data.py <pasta_saida> [n_estudos]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

CLASS_COLS = ["Negative for Pneumonia", "Typical Appearance",
              "Indeterminate Appearance", "Atypical Appearance"]
PRIORS = [0.28, 0.47, 0.17, 0.08]


def fake_cxr(rng, label, size=640):
    yy, xx = np.mgrid[0:size, 0:size] / size
    body = (((xx - 0.5) / 0.44) ** 2 + ((yy - 0.6) / 0.55) ** 2) < 1
    img = np.where(body, 0.75 - 0.1 * yy, 0.05)
    trachea = (np.abs(xx - 0.5) < 0.02) & (yy < 0.3)
    img[trachea] -= 0.3
    for cx in (0.33, 0.67):
        lung = (((xx - cx) / 0.13) ** 2 + ((yy - 0.5) / 0.28) ** 2) < 1
        img[lung] -= 0.35
        if label == 1:  # bilateral, inferior, periferica
            op = (((xx - cx) / 0.12) ** 2 + ((yy - 0.65) / 0.12) ** 2) < 1
            img[op & lung] += 0.18
        elif label == 2 and cx < 0.5:
            op = (((xx - cx) / 0.10) ** 2 + ((yy - 0.5) / 0.15) ** 2) < 1
            img[op & lung] += 0.15
        elif label == 3 and cx > 0.5:
            op = (((xx - cx) / 0.07) ** 2 + ((yy - 0.35) / 0.08) ** 2) < 1
            img[op & lung] += 0.3
    img += rng.normal(0, 0.03, img.shape)
    return np.clip(img * 4000, 0, 4095).astype(np.uint16)


def write_dcm(path, arr, patient_id, study_uid, series_uid, sop_uid, mono1, manufacturer):
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1.1"
    meta.MediaStorageSOPInstanceUID = sop_uid
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.PatientID = patient_id
    ds.PatientSex = "M"
    ds.Modality = "DX"
    ds.Manufacturer = manufacturer
    ds.BodyPartExamined = "CHEST"
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.SOPInstanceUID = sop_uid
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    if mono1:
        arr = 4095 - arr
    ds.PhotometricInterpretation = "MONOCHROME1" if mono1 else "MONOCHROME2"
    ds.SamplesPerPixel = 1
    ds.Rows, ds.Columns = arr.shape
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    ds.PixelRepresentation = 0
    ds.RescaleSlope = 1
    ds.RescaleIntercept = 0
    ds.ImagerPixelSpacing = [0.143, 0.143]
    ds.PixelData = arr.tobytes()
    ds.save_as(str(path), enforce_file_format=True)


def main():
    out = Path(sys.argv[1])
    n_studies = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    rng = np.random.default_rng(0)
    (out / "train").mkdir(parents=True, exist_ok=True)
    study_rows, img_rows = [], []
    n_patients = int(n_studies * 0.7)
    for s in range(n_studies):
        label = rng.choice(4, p=PRIORS)
        study_uid = generate_uid()
        series_uid = generate_uid()
        patient = f"P{rng.integers(n_patients):04d}"
        manufacturer = ["GE", "Philips", "Siemens"][s % 3]
        mono1 = s % 5 == 0
        n_img = 2 if s % 12 == 0 else 1
        for _ in range(n_img):
            sop = generate_uid()
            image_id = sop.replace(".", "")[-12:]
            d = out / "train" / study_uid / series_uid
            d.mkdir(parents=True, exist_ok=True)
            write_dcm(d / f"{image_id}.dcm", fake_cxr(rng, label), patient, study_uid, series_uid,
                      sop, mono1, manufacturer)
            box = "none 1 0 0 1 1" if label == 0 else "opacity 0.9 100 200 300 400 opacity 0.8 350 200 500 420"
            img_rows.append({"id": f"{image_id}_image", "boxes": "", "label": box, "StudyInstanceUID": study_uid})
        onehot = [int(k == label) for k in range(4)]
        study_rows.append({"id": f"{study_uid}_study", **dict(zip(CLASS_COLS, onehot))})
    pd.DataFrame(study_rows).to_csv(out / "train_study_level.csv", index=False)
    pd.DataFrame(img_rows).to_csv(out / "train_image_level.csv", index=False)
    print(f"{n_studies} estudos, {len(img_rows)} imagens em {out}")


if __name__ == "__main__":
    main()
