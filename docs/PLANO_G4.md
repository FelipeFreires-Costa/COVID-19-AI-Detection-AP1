# Plano do G4 &mdash; divis&atilde;o de tarefas, cronograma e links

Entrega: **04/10/2026 (domingo), 23h59**, pelo AVA, em um &uacute;nico `TP1_G4_COVID19.zip`.

## Situa&ccedil;&atilde;o atual (29/09)

- C&oacute;digo pronto e **testado de ponta a ponta** com dados sint&eacute;ticos (`tests/run_smoke_test.py`).
- O artigo compila em **exatamente 4 p&aacute;ginas** no template SBC, com 14 refer&ecirc;ncias. Todos os n&uacute;meros s&atilde;o preenchidos automaticamente por `src/fill_paper.py`.
- Falta: rodar no Kaggle com os dados reais, subir para o GitHub, preencher nomes, institui&ccedil;&atilde;o e link, e montar o zip.

## Passo a passo m&iacute;nimo (quem executa: Felipe, ~1 h de trabalho ativo)

1. **Kaggle:** aceitar as regras da competi&ccedil;&atilde;o e verificar o telefone.
2. **GitHub:** criar um reposit&oacute;rio **p&uacute;blico** `tp1-g4-covid19-baseline` e fazer push desta pasta (comandos em `docs/COMANDOS_GITHUB.md`).
3. **Kaggle:** *New Notebook* &rarr; *File* &rarr; *Import Notebook* &rarr; `notebooks/TP1_G4_kaggle.ipynb`. Em *Add Input* &rarr; *Competitions*, adicionar `siim-covid19-detection`. Em *Settings*, ligar *Internet*. Trocar o `REPO_URL` na 1&ordf; c&eacute;lula.
4. Rodar s&oacute; as duas primeiras c&eacute;lulas com `SAMPLE_FRAC = 0.1` para ver se o clone e os dados funcionam. Depois colocar `SAMPLE_FRAC = 1.0` e usar **Save Version &rarr; Save & Run All**. Ele roda sozinho por algumas horas, e voc&ecirc; pode fechar o navegador.
5. Quando terminar, baixar na aba *Output* os arquivos `tp1_results.zip` e `TP1_G4_paper_overleaf.zip`.
6. Descompactar `tp1_results.zip` por cima da pasta do reposit&oacute;rio e fazer commit e push.
7. **Overleaf:** *New Project* &rarr; *Upload Project* &rarr; `TP1_G4_paper_overleaf.zip`. Trocar `SEU_USUARIO`, `NOME DA INSTITUI&Ccedil;&Atilde;O`, `CIDADE -- UF` e `DOMINIO`, compilar, conferir se ficou em 4 p&aacute;ginas e baixar o PDF.
8. Imprimir `docs/TP1_G4_contribuicao.html` em PDF, colher as assinaturas e montar o zip final.

## Divis&atilde;o (para a tabela de contribui&ccedil;&atilde;o e o hist&oacute;rico de commits)

O professor pode comparar a tabela de contribui&ccedil;&atilde;o com o hist&oacute;rico de commits. **Pe&ccedil;a para Eduardo e Pedro fazerem ao menos um commit cada** da parte deles, por exemplo revisando coment&aacute;rios ou rodando o teste e subindo os resultados.

| Pessoa | Responsabilidade |
|---|---|
| **Felipe** | Dados e EDA (`prepare_data.py`, `eda.py`), reposit&oacute;rio, notebook do Kaggle, escrita do artigo, refer&ecirc;ncias e zip final |
| **Eduardo** | Pr&eacute;-processamento e descritores (`preprocessing.py`, `features.py`, `extract_features.py`): conferir `fig_masks.png` e a taxa de fallback e revisar o par&aacute;grafo de pr&eacute;-processamento e descritores |
| **Pedro** | Modelos, valida&ccedil;&atilde;o e erros (`experiment.py`, `run_experiments.py`, `ablation.py`, `make_figures.py`): conferir as tabelas e `fig_errors.png` e revisar Resultados e An&aacute;lise de erro |

## Cronograma (restam 6 dias)

| Dia | O qu&ecirc; |
|---|---|
| Ter 29/09 | Passos 1&ndash;4 (disparar a execu&ccedil;&atilde;o completa no Kaggle) |
| Qua 30/09 | Passos 5&ndash;7 (resultados no GitHub e artigo no Overleaf) |
| Qui 01/10 | Os tr&ecirc;s leem o PDF; Eduardo e Pedro fazem os commits deles |
| Sex 02/10 | Conferir as 14 refer&ecirc;ncias pelos DOIs (lista abaixo) |
| S&aacute;b 03/10 | Assinaturas e zip final |
| Dom 04/10 | Submeter (folga) |

## Links

- Regras da competi&ccedil;&atilde;o (aceitar): <https://www.kaggle.com/competitions/siim-covid19-detection/rules>
- Dados: <https://www.kaggle.com/competitions/siim-covid19-detection/data>
- Verifica&ccedil;&atilde;o de telefone no Kaggle: <https://www.kaggle.com/settings>
- Criar notebook: <https://www.kaggle.com/code>
- Criar reposit&oacute;rio: <https://github.com/new>
- Overleaf: <https://www.overleaf.com/project>

## Refer&ecirc;ncias citadas no artigo (conferir cada DOI)

1. Lakhani et al. 2023 &mdash; <https://doi.org/10.1007/s10278-022-00706-8>
2. Litmanovich et al. 2020 &mdash; <https://doi.org/10.1097/RTI.0000000000000541>
3. Wong et al. 2020 &mdash; <https://doi.org/10.1148/radiol.2020201160>
4. DeGrave et al. 2021 &mdash; <https://doi.org/10.1038/s42256-021-00338-7>
5. Roberts et al. 2021 &mdash; <https://doi.org/10.1038/s42256-021-00307-0>
6. Jaeger et al. 2014 &mdash; <https://doi.org/10.1109/TMI.2013.2284099>
7. Hussain et al. 2020 &mdash; <https://doi.org/10.1186/s12938-020-00831-x>
8. Varma & Simon 2006 &mdash; <https://doi.org/10.1186/1471-2105-7-91>
9. Haralick et al. 1973 &mdash; <https://doi.org/10.1109/TSMC.1973.4309314>
10. Ojala et al. 2002 &mdash; <https://doi.org/10.1109/TPAMI.2002.1017623>
11. Jain & Farrokhnia 1991 &mdash; <https://doi.org/10.1016/0031-3203(91)90143-S>
12. Dalal & Triggs 2005 &mdash; <https://doi.org/10.1109/CVPR.2005.177>
13. Chawla et al. 2002 &mdash; <https://doi.org/10.1613/jair.953>
14. Zuiderveld 1994 &mdash; cap&iacute;tulo do livro *Graphics Gems IV*, sem DOI

## Checklist final (do enunciado)

- [ ] PDF com no m&aacute;ximo 4 p&aacute;ginas, template SBC, resumo e abstract
- [ ] Link do reposit&oacute;rio no texto (Metodologia) e reposit&oacute;rio p&uacute;blico
- [ ] Nenhum `XX`, `SEU_USUARIO` ou `DOMINIO` sobrando no PDF
- [ ] Baseline trivial na tabela; tabela com m&eacute;dia &plusmn; desvio; figura
- [ ] 14 refer&ecirc;ncias conferidas (m&iacute;nimo de 10, sendo 6 revisadas por pares)
- [ ] Nota de uso de IA no final do artigo
- [ ] `results/` e `splits/` commitados no GitHub
- [ ] `TP1_G4_contribuicao.pdf` assinado
- [ ] Zip `TP1_G4_COVID19.zip` com o PDF do artigo, o zip do projeto do Overleaf (*Menu &rarr; Download &rarr; Source*) e o PDF da contribui&ccedil;&atilde;o
