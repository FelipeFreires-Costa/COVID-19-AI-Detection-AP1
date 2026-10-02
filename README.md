# Baseline clássico para radiografias de tórax — SIIM-FISABIO-RSNA COVID-19

Trabalho Prático 1 da disciplina Tópicos Especiais em Sistemas de Informação (Prof. Me. Décio Gonçalves de Aguiar Neto).

Grupo G4: Felipe Freires da Costa, Eduardo Nobre Nogueira de Oliveira e Pedro Paulo Alves dos Santos Nunes.

## Sobre o projeto

Construímos um baseline sem aprendizado profundo para classificar estudos de radiografia de tórax em quatro classes do desafio SIIM-FISABIO-RSNA COVID-19 (2021): negativo para pneumonia, típico, indeterminado e atípico.

O pipeline lê os DICOMs, aplica CLAHE, segmenta os pulmões com uma máscara clássica (Otsu e morfologia) e extrai cinco famílias de descritores: intensidade zonal, GLCM/Haralick, LBP, Gabor e HOG. Em seguida, comparamos regressão logística, SVM-RBF, Random Forest e LightGBM em validação cruzada aninhada, agrupada por paciente, e avaliamos o modelo escolhido uma única vez num conjunto de teste congelado.

No teste, o melhor modelo chegou a AUC-ROC macro de 0,729 e acurácia balanceada de 0,412, contra 0,500 e 0,250 do baseline trivial. Os detalhes estão no artigo, em `paper/`.

## Dados

Usamos apenas o conjunto de treino da competição no Kaggle ([siim-covid19-detection](https://www.kaggle.com/competitions/siim-covid19-detection)), já que os rótulos do teste oficial não são públicos. Os dados não estão no repositório. A partição usada (dev/teste e dobras, semente 42) está em `splits/splits.csv`.

## Reprodução

O caminho mais simples é importar `notebooks/TP1_G4_kaggle.ipynb` no Kaggle, adicionar a competição como input, ligar a internet e rodar tudo. A execução completa leva algumas horas em CPU. O notebook já executado está em `notebooks/TP1_G4_kaggle_executado.ipynb`.

Para rodar localmente, instale as dependências de `requirements.txt`, defina `SIIM_DATA_DIR` apontando para os dados e execute os scripts de `src/` na ordem: `prepare_data`, `eda`, `extract_features`, `run_experiments`, `ablation`, `make_figures` e `fill_paper`. O `tests/run_smoke_test.py` roda o pipeline inteiro com dados sintéticos, apenas para conferir se o código funciona.

## Organização

- `src/`: código do pipeline (leitura, pré-processamento, descritores, experimentos e figuras)
- `notebooks/`: notebook do Kaggle
- `splits/`: partição congelada por paciente
- `paper/`: artigo em LaTeX no template SBC
- `docs/`: fichamento, plano do grupo e tabela de contribuição
- `tests/`: teste com dados sintéticos

## Uso de IA generativa

O grupo usou IA para dúvidas de código, revisão, tradução e ajuda na formulação de alguns trechos. O conteúdo e as referências foram revisados pelo grupo.
