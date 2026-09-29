# TP1 &mdash; G4 &mdash; Baseline cl&aacute;ssico para detec&ccedil;&atilde;o de COVID-19 em radiografias de t&oacute;rax

T&oacute;picos Especiais em Sistemas de Informa&ccedil;&atilde;o, Prof. Me. D&eacute;cio Gon&ccedil;alves de Aguiar Neto.
Grupo G4: Felipe Freires da Costa, Eduardo Nobre Nogueira de Oliveira e Pedro Paulo Alves dos Santos Nunes.

Este reposit&oacute;rio tem o baseline **sem aprendizado profundo** (descritores hand-crafted e classificadores cl&aacute;ssicos) para o desafio **SIIM-FISABIO-RSNA COVID-19 Detection (RSNA AI Challenge 2021)**.

## Tarefa

- **Entrada:** radiografia(s) de t&oacute;rax (DICOM) de um estudo.
- **Sa&iacute;da:** uma de 4 classes mutuamente exclusivas por estudo: *Negative for Pneumonia*, *Typical*, *Indeterminate* ou *Atypical Appearance*. Tamb&eacute;m reportamos a tarefa bin&aacute;ria derivada (pneumonia vs. negativo).
- **Unidade de amostra:** o estudo. Quando o estudo tem mais de uma imagem, usamos a m&eacute;dia das caracter&iacute;sticas das imagens.
- **Unidade de parti&ccedil;&atilde;o:** o paciente (`PatientID` do DICOM), para evitar vazamento.

## Dados

1. Crie uma conta no Kaggle e aceite as regras da competi&ccedil;&atilde;o: <https://www.kaggle.com/competitions/siim-covid19-detection/rules>
2. Os dados ficam em <https://www.kaggle.com/competitions/siim-covid19-detection/data>. Usamos apenas `train/`, `train_study_level.csv` e `train_image_level.csv`. Os r&oacute;tulos do teste oficial n&atilde;o s&atilde;o p&uacute;blicos, ent&atilde;o todo o protocolo usa s&oacute; o conjunto de treino.
3. Os dados brutos **n&atilde;o** est&atilde;o neste reposit&oacute;rio. A amostra usada est&aacute; definida por `splits/splits.csv`: lista de `study_id`, paciente, classe, parti&ccedil;&atilde;o (`dev`/`test`) e dobra da CV, gerada com semente 42.

## Como reproduzir (recomendado: Kaggle, sem baixar ~120 GB)

1. Kaggle &rarr; *Code* &rarr; *New Notebook* &rarr; *File* &rarr; *Import Notebook* &rarr; envie `notebooks/TP1_G4_kaggle.ipynb`.
2. *Add Input* &rarr; *Competitions* &rarr; `siim-covid19-detection`. Em *Settings*, ligue *Internet*. O acelerador pode ficar como *None*.
3. Na primeira c&eacute;lula, ajuste `REPO_URL` para a URL deste reposit&oacute;rio.
4. *Save Version* &rarr; *Save & Run All*. O pipeline completo leva algumas horas em CPU. As tabelas saem em `results/tables/` e as figuras em `results/figures/`.

Para um teste r&aacute;pido, use `SAMPLE_FRAC = 0.1`. Os resultados v&atilde;o para `results_frac0.1/` e a parti&ccedil;&atilde;o para `splits/splits_frac0.1.csv`, sem sobrescrever a execu&ccedil;&atilde;o final.

### Execu&ccedil;&atilde;o local (alternativa)

```bash
pip install -r requirements.txt
kaggle competitions download -c siim-covid19-detection -p data && unzip -q data/siim-covid19-detection.zip -d data/siim
export SIIM_DATA_DIR=data/siim          # Windows PowerShell: $env:SIIM_DATA_DIR="data/siim"
python src/prepare_data.py
python src/eda.py
python src/extract_features.py --tag default --clahe 1 --mask 1
python src/run_experiments.py --tag default
python src/extract_features.py --tag raw_full   --clahe 0 --mask 0
python src/extract_features.py --tag clahe_full --clahe 1 --mask 0
python src/extract_features.py --tag raw_mask   --clahe 0 --mask 1
python src/ablation.py
python src/make_figures.py
python src/fill_paper.py     # preenche paper/numbers.tex e gera o zip para o Overleaf
```

### Teste r&aacute;pido sem os dados reais

`python tests/run_smoke_test.py` gera um mini conjunto DICOM **sint&eacute;tico** no formato da competi&ccedil;&atilde;o e roda todas as etapas acima numa pasta tempor&aacute;ria (~5 min). Serve s&oacute; para verificar que o c&oacute;digo roda; os n&uacute;meros n&atilde;o t&ecirc;m valor cient&iacute;fico.

## Artigo

O artigo fica em `paper/main.tex`, com o template SBC (`sbc-template.sty` e `sbc.bst`) j&aacute; inclu&iacute;do. Todos os n&uacute;meros do texto s&atilde;o macros definidas em `paper/numbers.tex`, gerado por `src/fill_paper.py` a partir de `results/`. Para compilar no Overleaf, use *New Project* &rarr; *Upload Project* com o arquivo `results/TP1_G4_paper_overleaf.zip`.

## Estrutura

| Arquivo | Fun&ccedil;&atilde;o |
|---|---|
| `src/config.py` | Semente (42), caminhos relativos e vari&aacute;veis de ambiente |
| `src/prepare_data.py` | Leitura DICOM (Modality LUT, VOI LUT, MONOCHROME1), metadados, PNG 512&sup2;, amostragem e parti&ccedil;&atilde;o por paciente |
| `src/preprocessing.py` | CLAHE, m&aacute;scara pulmonar cl&aacute;ssica (Otsu, morfologia e fecho convexo) e 6 zonas pulmonares |
| `src/features.py` | 5 fam&iacute;lias de descritores: intensidade zonal, GLCM/Haralick, LBP, Gabor e HOG |
| `src/extract_features.py` | Extra&ccedil;&atilde;o por imagem e agrega&ccedil;&atilde;o por estudo |
| `src/experiment.py` | Pipelines, modelos, m&eacute;tricas e CV aninhada agrupada |
| `src/run_experiments.py` | Grade descritor &times; modelo, sele&ccedil;&atilde;o do melhor pelo dev e avalia&ccedil;&atilde;o &uacute;nica no teste |
| `src/ablation.py` | Abla&ccedil;&otilde;es de pr&eacute;-processamento e de desbalanceamento |
| `src/eda.py`, `src/make_figures.py` | Figuras, import&acirc;ncia de caracter&iacute;sticas e an&aacute;lise de erro |
| `src/fill_paper.py` | Preenche o artigo com os resultados e empacota o projeto LaTeX |
| `tests/` | Gerador de dados sint&eacute;ticos e teste de ponta a ponta |
| `paper/` | Artigo em LaTeX (template SBC) |

## Protocolo experimental

- Os 20% de pacientes do hold-out de teste s&atilde;o separados uma &uacute;nica vez com `StratifiedGroupKFold` e ficam congelados em `splits/splits.csv`.
- Nos 80% restantes (dev) rodamos CV aninhada: 5 dobras externas agrupadas por paciente e, dentro de cada uma, busca de hiperpar&acirc;metros com 3 dobras internas, tamb&eacute;m agrupadas.
- Imputa&ccedil;&atilde;o, padroniza&ccedil;&atilde;o, PCA e SMOTE ficam dentro do `Pipeline`, ent&atilde;o s&atilde;o ajustados s&oacute; no treino de cada dobra.
- O melhor modelo &eacute; escolhido pela AUC macro m&eacute;dia da CV do dev. Ele &eacute; reajustado no dev inteiro e avaliado **uma &uacute;nica vez** no teste.
- M&eacute;tricas: AUC-ROC e AUC-PR macro (um-contra-todos), acur&aacute;cia balanceada, F1 macro/ponderado, sensibilidade e especificidade por classe, log-loss e as m&eacute;tricas da tarefa bin&aacute;ria. O baseline trivial &eacute; um classificador que prev&ecirc; as probabilidades a priori (classe majorit&aacute;ria).
- Nenhuma rede neural &eacute; usada, nem como extratora de caracter&iacute;sticas.
