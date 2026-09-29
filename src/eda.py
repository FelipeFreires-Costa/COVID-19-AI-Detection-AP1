"""Analise exploratoria: distribuicao de classes, metadados DICOM (variabilidade entre
instituicoes/equipamentos), estudos por paciente, caixas de opacidade e exemplos por classe.

Uso:  python src/eda.py    (requer prepare_data.py ja executado)
"""
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config as C


def main():
    C.ensure_dirs()
    meta = C.read_meta()
    st = C.read_splits()
    names = dict(enumerate(C.CLASS_NAMES))
    st["classe"] = st["label"].map(names)
    meta["classe"] = meta["label"].map(names)
    per_patient = st.groupby("patient_id").size()

    lines = [
        f"Estudos: {len(st)}", f"Imagens: {len(meta)}", f"Pacientes: {st['patient_id'].nunique()}",
        f"Estudos com >1 imagem: {(st['n_images'] > 1).sum()}",
        f"Estudos por paciente (max): {per_patient.max()}",
        f"Pacientes com >1 estudo: {(per_patient > 1).sum()}",
        "", "Classes (estudos):", st["classe"].value_counts().to_string(),
        "", "Classes (%):", (st["classe"].value_counts(normalize=True) * 100).round(1).to_string(),
        "", "Particao x classe:", pd.crosstab(st["classe"], st["split"]).to_string(),
        "", "Modalidade:", meta["Modality"].value_counts(dropna=False).to_string(),
        "", "PhotometricInterpretation:", meta["PhotometricInterpretation"].value_counts(dropna=False).to_string(),
        "", "BitsStored:", meta["BitsStored"].value_counts(dropna=False).to_string(),
        "", "Fabricantes (top 10):", meta["Manufacturer"].value_counts(dropna=False).head(10).to_string(),
        "", f"PixelSpacing (mm) mediana: {meta['PixelSpacing'].median():.3f} | faltante: {meta['PixelSpacing'].isna().mean():.1%}",
        f"Resolucao original mediana: {meta['Rows'].median():.0f} x {meta['Columns'].median():.0f}",
        "", "Caixas de opacidade por imagem (media por classe):",
        meta.groupby("classe")["n_boxes"].mean().round(2).to_string(),
    ]
    (C.TAB_DIR / "eda_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    order = C.CLASS_NAMES
    sns.countplot(data=st, x="classe", order=order, ax=axes[0], color="#4C72B0")
    axes[0].set_title("Estudos por classe")
    axes[0].set_xlabel("")
    manuf = meta["Manufacturer"].fillna("Desconhecido")
    top = manuf.value_counts().head(8).index
    ct = pd.crosstab(manuf.where(manuf.isin(top), "Outros"), meta["classe"], normalize="index")
    ct = ct[[c for c in order if c in ct.columns]]
    sns.heatmap(ct, annot=True, fmt=".2f", cmap="Blues", ax=axes[1], cbar=False)
    axes[1].set_title("Distribui\u00e7\u00e3o de classes por fabricante")
    axes[1].set_ylabel("")
    axes[1].set_xlabel("")
    sns.boxplot(data=meta, x="classe", y="n_boxes", order=order, ax=axes[2], color="#DD8452")
    axes[2].set_title("Caixas de opacidade por imagem")
    axes[2].set_xlabel("")
    plt.tight_layout()
    plt.savefig(C.FIG_DIR / "eda_overview.png", dpi=150)
    plt.close()

    rng = np.random.default_rng(C.SEED)
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for k, name in enumerate(order):
        ids = meta.loc[meta["label"] == k, "image_id"].to_numpy()
        for j, iid in enumerate(rng.choice(ids, size=min(4, len(ids)), replace=False)):
            img = cv2.imread(str(C.IMG_DIR / f"{iid}.png"), cv2.IMREAD_GRAYSCALE)
            axes[k, j].imshow(img, cmap="gray")
            if j == 0:
                axes[k, j].set_title(name, loc="left", fontsize=11)
        for ax in axes[k]:
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(C.FIG_DIR / "eda_examples.png", dpi=120)
    plt.close()


if __name__ == "__main__":
    main()
