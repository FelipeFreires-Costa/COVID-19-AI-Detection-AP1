"""Configuracao central do projeto: caminhos, semente, classes e parametros globais.

Todos os caminhos sao relativos a raiz do repositorio ou definidos por variaveis de
ambiente, para que o codigo rode igual no Kaggle, no Colab ou localmente.
"""
import os
from pathlib import Path

import pandas as pd

SEED = 42

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("SIIM_DATA_DIR", "/kaggle/input/siim-covid19-detection"))
WORK_DIR = Path(os.environ.get("TP1_WORK_DIR", str(ROOT / "work")))

# Fracao de estudos usada (amostragem estratificada por classe). 1.0 = conjunto de treino inteiro.
SAMPLE_FRAC = float(os.environ.get("TP1_SAMPLE_FRAC", "1.0"))
_SUFFIX = "" if SAMPLE_FRAC >= 1.0 else f"_frac{SAMPLE_FRAC:g}"

IMG_DIR = WORK_DIR / "png512"
SPLITS_DIR = Path(os.environ.get("TP1_SPLITS_DIR", str(ROOT / "splits")))
SPLITS_FILE = SPLITS_DIR / f"splits{_SUFFIX}.csv"
RESULTS_DIR = Path(os.environ.get("TP1_RESULTS_DIR", str(ROOT / f"results{_SUFFIX}")))
FIG_DIR = RESULTS_DIR / "figures"
TAB_DIR = RESULTS_DIR / "tables"
PRED_DIR = RESULTS_DIR / "preds"

IMG_SIZE = 512
N_JOBS = int(os.environ.get("TP1_N_JOBS", str(os.cpu_count() or 1)))
N_OUTER = 5
N_INNER = 3

CLASS_COLS = [
    "Negative for Pneumonia",
    "Typical Appearance",
    "Indeterminate Appearance",
    "Atypical Appearance",
]
CLASS_NAMES = ["Negativo", "T\u00edpico", "Indeterminado", "At\u00edpico"]
N_CLASSES = len(CLASS_COLS)

FAMILIES = ["int", "glcm", "lbp", "gabor", "hog"]
FAMILY_NAMES = {
    "int": "Intensidade (zonal)",
    "glcm": "GLCM/Haralick",
    "lbp": "LBP",
    "gabor": "Gabor",
    "hog": "HOG",
    "all": "Todas",
}
MODEL_NAMES = {
    "dummy": "Majorit\u00e1rio",
    "logreg": "Reg. Log\u00edstica",
    "svm_rbf": "SVM-RBF",
    "rf": "Random Forest",
    "lgbm": "LightGBM",
}


ID_DTYPES = {"image_id": str, "study_id": str, "patient_id": str, "PatientID": str}


def read_meta():
    return pd.read_csv(WORK_DIR / "image_meta.csv", dtype=ID_DTYPES)


def read_splits():
    return pd.read_csv(SPLITS_FILE, dtype=ID_DTYPES)


def ensure_dirs():
    for d in (WORK_DIR, IMG_DIR, SPLITS_DIR, FIG_DIR, TAB_DIR, PRED_DIR):
        d.mkdir(parents=True, exist_ok=True)
