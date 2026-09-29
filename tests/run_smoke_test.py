"""Teste de ponta a ponta com dados SINTETICOS: gera um mini conjunto no formato da
competicao e roda todas as etapas do pipeline, verificando se as saidas esperadas existem.
Nao toca em results/ nem em splits/ do repositorio.

Uso:  python tests/run_smoke_test.py [n_estudos]      (~5 min com 300 estudos)
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
N = sys.argv[1] if len(sys.argv) > 1 else "300"

STEPS = [
    ["src/prepare_data.py"],
    ["src/eda.py"],
    ["src/extract_features.py", "--tag", "default", "--clahe", "1", "--mask", "1"],
    ["src/run_experiments.py", "--tag", "default"],
    ["src/extract_features.py", "--tag", "raw_full", "--clahe", "0", "--mask", "0"],
    ["src/extract_features.py", "--tag", "clahe_full", "--clahe", "1", "--mask", "0"],
    ["src/extract_features.py", "--tag", "raw_mask", "--clahe", "0", "--mask", "1"],
    ["src/ablation.py", "--part", "all"],
    ["src/make_figures.py"],
]
EXPECTED = ["tables/table_main.tex", "tables/table_test.tex", "tables/cv_summary.csv",
            "tables/ablation_prep.csv", "tables/ablation_imb.csv", "tables/error_by_group.csv",
            "figures/fig_roc_cm.png", "figures/fig_importance.png", "figures/fig_errors.png",
            "figures/fig_masks.png", "figures/eda_overview.png"]


def main():
    tmp = Path(tempfile.mkdtemp(prefix="tp1_smoke_"))
    env = dict(os.environ, SIIM_DATA_DIR=str(tmp / "data"), TP1_WORK_DIR=str(tmp / "work"),
               TP1_RESULTS_DIR=str(tmp / "results"), TP1_SPLITS_DIR=str(tmp / "splits"),
               PYTHONIOENCODING="utf-8")
    subprocess.run([sys.executable, "tests/make_synthetic_data.py", str(tmp / "data"), N],
                   cwd=ROOT, check=True)
    for step in STEPS:
        print(">>", " ".join(step), flush=True)
        r = subprocess.run([sys.executable, *step], cwd=ROOT, env=env, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        if r.returncode != 0:
            print(r.stdout[-3000:], r.stderr[-3000:])
            sys.exit(f"FALHOU em {step[0]}")
    subprocess.run([sys.executable, "src/fill_paper.py", "--paper-dir", str(tmp / "paper")],
                   cwd=ROOT, env=env, check=True)
    missing = [e for e in EXPECTED if not (tmp / "results" / e).exists()]
    if missing:
        sys.exit(f"Saidas ausentes: {missing}")
    print(f"OK - todas as etapas rodaram. Saidas em {tmp}")


if __name__ == "__main__":
    main()
