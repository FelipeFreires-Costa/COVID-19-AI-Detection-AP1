# Fichamento de refer&ecirc;ncias &mdash; G4 (Marco 1)

> Rascunho para servir de roteiro de leitura. **Cada item precisa ser conferido lendo o artigo original** antes de ser citado. Onde est&aacute; escrito *[conferir]*, o detalhe deve ser confirmado no texto.

## 1. Lakhani et al. (2023) &mdash; descri&ccedil;&atilde;o do desafio SIIM-FISABIO-RSNA
- **Refer&ecirc;ncia:** J. Digit. Imaging 36:365&ndash;372. DOI 10.1007/s10278-022-00706-8.
- **Objetivo:** descrever como o conjunto do desafio foi montado e anotado.
- **Dados:** radiografias de t&oacute;rax de m&uacute;ltiplas institui&ccedil;&otilde;es (BIMCV-COVID19+ e MIDRC-RICORD), anotadas por radiologistas em 4 categorias de estudo e com caixas de opacidade por imagem *[conferir o n&uacute;mero de leitores e o processo de consenso]*.
- **Relev&acirc;ncia para n&oacute;s:** fonte prim&aacute;ria sobre a formula&ccedil;&atilde;o da tarefa, a rotulagem em m&uacute;ltiplos n&iacute;veis (estudo e imagem) e a m&eacute;trica (mAP). Justifica o uso do estudo como unidade de amostra.

## 2. Litmanovich et al. (2020) &mdash; linguagem de laudo para RX na COVID-19
- **Refer&ecirc;ncia:** J. Thorac. Imaging 35(6):354&ndash;360.
- **Conte&uacute;do:** revisa os achados de RX na COVID-19 e prop&otilde;e as categorias t&iacute;pica, indeterminada, at&iacute;pica e negativa. A apar&ecirc;ncia t&iacute;pica envolve opacidades multifocais, bilaterais, perif&eacute;ricas e basais; a at&iacute;pica envolve consolida&ccedil;&atilde;o lobar, n&oacute;dulos, cavita&ccedil;&atilde;o ou derrame *[conferir as defini&ccedil;&otilde;es exatas]*.
- **Relev&acirc;ncia:** explica por que as classes se diferenciam sobretudo pela **distribui&ccedil;&atilde;o espacial**, o que motiva nossos descritores zonais e a an&aacute;lise de erro.

## 3. Roberts et al. (2021) &mdash; armadilhas de ML para COVID-19
- **Refer&ecirc;ncia:** Nat. Mach. Intell. 3:199&ndash;217.
- **M&eacute;todo:** revis&atilde;o sistem&aacute;tica de modelos de ML para diagn&oacute;stico e progn&oacute;stico de COVID-19 em RX e TC.
- **Achados:** a grande maioria dos modelos tinha vieses metodol&oacute;gicos graves, como conjuntos "Frankenstein", fontes diferentes por classe, falta de valida&ccedil;&atilde;o externa e parti&ccedil;&atilde;o inadequada, e nenhum foi considerado pronto para uso cl&iacute;nico *[conferir os n&uacute;meros]*.
- **Relev&acirc;ncia:** justifica a parti&ccedil;&atilde;o por paciente, a CV aninhada e o cuidado com vazamento.

## 4. DeGrave, Janizek & Lee (2021) &mdash; atalhos em RX de COVID-19
- **Refer&ecirc;ncia:** Nat. Mach. Intell. 3:610&ndash;619.
- **M&eacute;todo:** avaliam redes treinadas em conjuntos de COVID-19 compostos por m&uacute;ltiplas fontes, com teste externo e t&eacute;cnicas de explicabilidade.
- **Achados:** os modelos exploram atalhos, como marcadores laterais, texto e bordas da imagem, e perdem desempenho em dados externos.
- **Relev&acirc;ncia:** motiva restringir os descritores &agrave; m&aacute;scara pulmonar e testar isso na abla&ccedil;&atilde;o (imagem inteira vs. m&aacute;scara).

## 5. Jaeger et al. (2014) &mdash; triagem autom&aacute;tica de TB em RX
- **Refer&ecirc;ncia:** IEEE TMI 33(2):233&ndash;245.
- **M&eacute;todo:** segmenta&ccedil;&atilde;o pulmonar, seguida de descritores de intensidade, bordas, forma e textura (incluindo histogramas, LBP e HOG) e SVM *[conferir o conjunto exato de descritores]*.
- **Resultados:** AUC em torno de 0,87&ndash;0,90 em duas bases p&uacute;blicas *[conferir]*.
- **Relev&acirc;ncia:** precedente direto de pipeline cl&aacute;ssico para achados pulmonares difusos em RX, bem pr&oacute;ximo do nosso desenho.

## 6. Hussain et al. (2020) &mdash; textura para COVID-19 em RX port&aacute;til
- **Refer&ecirc;ncia:** BioMed. Eng. OnLine 19:88.
- **M&eacute;todo:** descritores de textura e morfologia com classificadores cl&aacute;ssicos para separar COVID-19, pneumonia e normal *[conferir as classes e os classificadores]*.
- **Limita&ccedil;&atilde;o:** base pequena e composta por fontes diferentes por classe, exatamente o tipo de desenho criticado por Roberts et al.
- **Relev&acirc;ncia:** mostra que descritores cl&aacute;ssicos foram aplicados &agrave; COVID-19; o nosso trabalho se posiciona com um protocolo mais rigoroso e uma tarefa mais dif&iacute;cil.

## Descritores e protocolo (trabalhos seminais j&aacute; citados no artigo)
- Haralick et al. (1973): GLCM. Ojala et al. (2002): LBP. Dalal & Triggs (2005): HOG. Jain & Farrokhnia (1991): Gabor.
- Zuiderveld (1994): CLAHE. Otsu (1979): limiar para a m&aacute;scara.
- Varma & Simon (2006): vi&eacute;s ao selecionar modelos com a mesma CV que estima o erro, o que justifica a CV aninhada.
- Saito & Rehmsmeier (2015): AUC-PR sob desbalanceamento. Chawla et al. (2002): SMOTE.
