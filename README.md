# Detecção Automatizada de EPIs por Visão Computacional
# YOLOv8n vs. SSD MobileNetV3 vs. Ensemble WBF

Projeto de Trabalho de Conclusão de Curso voltado à detecção automatizada de Equipamentos de Proteção Individual (EPIs) e à identificação de violações de segurança em ambientes de construção civil utilizando Visão Computacional e Deep Learning.

O projeto compara três abordagens:

* **YOLOv8**
* **SSD MobileNetV3 (SSDLite320)**
* **Ensemble YOLOv8 + SSD**, utilizando *Weighted Boxes Fusion* (WBF)

O experimento foi desenvolvido em **Google Colab**, utilizando GPU para treinamento e avaliação dos modelos.

---

## Integrantes

* `Patrícia Leão`
* `Erinaldo Ferreira`
* `Reginaldo Paiva`
* `Carlos André`
* `Elayne Lima`
* `Rodrigo Couto`

---

## Problema

A fiscalização do uso adequado de Equipamentos de Proteção Individual em canteiros de obras é uma atividade importante para a segurança dos trabalhadores. Entretanto, a inspeção exclusivamente manual é limitada pela necessidade de acompanhamento contínuo, extensão dos ambientes e possibilidade de falhas humanas.

Este projeto investiga a utilização de Visão Computacional para auxiliar na identificação automática de EPIs e, principalmente, de situações de ausência de equipamentos.

A pergunta central do experimento é:

*Em que medida YOLOv8n, SSD MobileNetV3 e um Ensemble dos dois modelos por Weighted Boxes Fusion diferem em desempenho na detecção automatizada de EPIs e infrações de segurança?

O objetivo não é substituir a fiscalização humana, mas avaliar experimentalmente a capacidade das diferentes abordagens de atuar como ferramenta de apoio.

---
## Objetivo

*Objetivo geral

Projetar, implementar e validar experimentalmente um sistema de detecção automatizada de EPIs e identificação de não conformidades utilizando diferentes arquiteturas de detecção de objetos.

*Objetivos específicos
  -Comparar YOLOv8n e SSD MobileNetV3;
  -Avaliar um Ensemble utilizando Weighted Boxes Fusion;
  -Utilizar Validação Cruzada Estratificada com K = 10 folds;
  -Avaliar os modelos em um conjunto Hold-Out independente;
  -Comparar mAP@0.5, Precisão, Revocação e F1-Score;
  -Avaliar velocidade de inferência e latência;
  -Avaliar especificamente as classes relacionadas às violações de segurança;
  -Aplicar testes estatísticos às diferenças observadas;
  -Investigar limitações e possíveis ameaças à validade experimental.

## Dataset

O experimento utiliza o dataset público **Construction-PPE**, composto por:

* **1.416 imagens**
* **11 classes**
* 294 imagens possuem pelo menos uma violação de EPI.
* Média de aproximadamente 8,1 caixas delimitadoras por imagem.
* Mediana de 7 caixas por imagem.

As classes utilizadas são:

```text
helmet
gloves
vest
boots
goggles
none
Person
no_helmet
no_goggle
no_gloves
no_boots
```

### Classes de violação

```text
no_helmet
no_goggle
no_gloves
no_boots
```

### Observação sobre `no_vest`

O dataset não possui uma classe `no_vest`.
Por esse motivo, não foi realizada uma comparação entre uso e ausência de colete. Essa limitação pertence ao próprio dataset utilizado.

A classe `no_boots`  apresenta baixa quantidade de exemplos, o que reduz a confiabilidade das métricas calculadas exclusivamente para essa classe.

---

## Metodologia

Foram comparadas três abordagens.

### YOLOv8

Foi utilizado o modelo YOLOv8 para detecção de objetos.

O treinamento utiliza entrada de 640×640 pixels e o modelo pré-treinado possui pesos provenientes do COCO.

### SSD MobileNetV3

Foi utilizado o **SSD MobileNetV3 com SSDLite320**.

A entrada nativa utilizada pelo modelo é de 320×320 pixels.

### Ensemble

A terceira abordagem combina as predições do YOLOv8 e do SSD MobileNetV3 utilizando **Weighted Boxes Fusion (WBF)**.

O Ensemble não possui um treinamento próprio. Ele reutiliza os pesos dos dois modelos e realiza a fusão das predições durante a inferência.

---

## Validação experimental

O experimento utiliza validação cruzada estratificada com **10 folds**.

A estratificação considera a presença de pelo menos uma violação de EPI na imagem.

Além da validação cruzada, foi separado um **Hold-Out de 20%**, que permanece isolado até a avaliação final.

As métricas utilizadas são:

* mAP@0.5
* Precision
* Recall
* F1-Score
* FPS
* Latência

Para a análise estatística foram utilizados:

* teste de Friedman;
* teste t pareado;
* teste de Wilcoxon quando aplicável;
* correção de Holm;
* teste de Shapiro-Wilk;
* intervalos de confiança de 95%;
* intervalo de Wilson para métricas de alerta por imagem.

---

## Ambiente de execução

O experimento foi executado no:

**Google Colab**

Durante a execução registrada no notebook, foi utilizada:

```text
GPU: NVIDIA A100-SXM4-40GB
CUDA: disponível
```

O projeto também possui suporte para execução fora do Colab, utilizando uma pasta local em vez do Google Drive.

No Google Colab, os resultados são armazenados em:

```text
/content/drive/MyDrive/epi_experimento
```

O notebook utiliza checkpoints no Google Drive para permitir a retomada do experimento caso a sessão do Colab seja encerrada.

---

## Dependências

As principais bibliotecas utilizadas são:

```text
ultralytics
torch
torchmetrics
pycocotools
scikit-learn
scipy
pandas
matplotlib
ensemble-boxes
```

A instalação pode ser realizada diretamente no Google Colab:

```bash
pip install ultralytics torchmetrics pycocotools scikit-learn scipy pandas matplotlib ensemble-boxes
```
No Google Colab, recomenda-se executar a instalação das dependências antes das células de treinamento.

---

## Estrutura do projeto

A estrutura utilizada pelo experimento é organizada da seguinte forma:

```text
.
├── README.md
├── experimento_yolov8_vs_ssd_vs_ensemble_v6_resultados.ipynb

```

O dataset é baixado automaticamente durante a execução e organizado na estrutura utilizada pelo notebook.

---

## Como executar

### Google Colab

1. Faça o upload deste repositório para o GitHub.
2. Abra o arquivo:

```text
experimento_yolov8_vs_ssd_vs_ensemble_v6_resultados.ipynb
```

3. Abra o notebook no Google Colab.
4. Execute as células na ordem.
5. Autorize o acesso ao Google Drive quando solicitado.
6. Instale as dependências.
7. Baixe e organize o dataset Construction-PPE.
8. Execute a preparação dos dados.
9. Execute o treinamento do YOLOv8 e do SSD.
10. Execute a avaliação dos modelos.
11. Execute o Ensemble.
12. Execute o benchmark de FPS.
13. Execute as análises estatísticas e de erro.

O notebook salva checkpoints no Google Drive para permitir que o experimento seja retomado em uma nova sessão.

### Ordem recomendada

O notebook deve ser executado de cima para baixo.

As principais etapas são:

```text
Parte 0  → Configuração do Google Drive
Parte 1  → Instalação das dependências
Parte 2  → Download e organização do dataset
Parte 3  → Hold-Out e K-Fold
Parte 4  → Preparação/augmentation
Parte 5  → Treinamento
Parte 6  → Ensemble
Parte 7  → Avaliação
Parte 8  → Comparações
Parte 9  → Métricas e análises estatísticas
Parte 10 → Benchmark de FPS
Parte 11 → Inferência em imagem/vídeo externo
Parte 12 → Limitações e ameaças à validade
```

---

## Resultados principais

### Validação cruzada

Resultados médios dos 10 folds:

| Abordagem       |       mAP@0.5 |     Precision |        Recall |            F1 |
| --------------- | ------------: | ------------: | ------------: | ------------: |
| YOLOv8          | 0.543 ± 0.030 | 0.792 ± 0.043 | 0.691 ± 0.031 | 0.737 ± 0.029 |
| SSD MobileNetV3 | 0.356 ± 0.022 | 0.605 ± 0.032 | 0.444 ± 0.026 | 0.512 ± 0.025 |
| Ensemble        | 0.517 ± 0.030 | 0.803 ± 0.031 | 0.591 ± 0.027 | 0.681 ± 0.024 |

### Hold-Out

O conjunto Hold-Out possui 284 imagens e foi utilizado apenas na avaliação final.

| Abordagem       | mAP@0.5 | Precision | Recall |    F1 |    FPS |  Latência |
| --------------- | ------: | --------: | -----: | ----: | -----: | --------: |
| YOLOv8          |   0.568 |     0.809 |  0.729 | 0.767 | 75.016 | 13.330 ms |
| SSD MobileNetV3 |   0.369 |     0.615 |  0.455 | 0.523 | 54.987 | 18.186 ms |
| Ensemble        |   0.528 |     0.824 |  0.628 | 0.713 | 18.924 | 52.844 ms |

O benchmark de FPS foi realizado na mesma sessão, utilizando o mesmo ambiente e as mesmas imagens.

---

## Resultados para classes de violação

No Hold-Out, os resultados de AP@0.5 foram:

| Classe      | YOLOv8 |   SSD | Ensemble |
| ----------- | -----: | ----: | -------: |
| `no_helmet` |  0.345 | 0.163 |    0.235 |
| `no_goggle` |  0.113 | 0.068 |    0.111 |
| `no_gloves` |  0.110 | 0.058 |    0.106 |
| `no_boots`  |  0.030 | 0.000 |    0.038 |

A classe `no_boots` possui apenas **12 instâncias no Hold-Out**, portanto os resultados dessa classe devem ser interpretados com cautela.

---

## Análise estatística

Para o mAP@0.5 dos 10 folds, o teste de Friedman apresentou:

```text
p = 4.5e-05
```

As comparações pareadas apresentaram diferenças estatisticamente significativas após correção de Holm:

```text
YOLOv8 vs SSD:
p ajustado = 6.9e-09

YOLOv8 vs Ensemble:
p ajustado = 2.5e-07

SSD vs Ensemble:
p ajustado = 8.1e-09
```

Para as quatro classes de violação, também foram observadas diferenças nas comparações pareadas após correção de Holm.

Os resultados estatísticos devem ser interpretados considerando que os folds compartilham dados de treinamento e, portanto, a independência entre as observações é aproximada.

---

## Análise de alerta de violação

Com o limiar padrão de confiança de **0,25**, o alerta por imagem apresentou:

| Modelo          | Recall | Precision |    F1 |
| --------------- | -----: | --------: | ----: |
| YOLOv8          |  0.593 |     1.000 | 0.745 |
| SSD MobileNetV3 |  0.661 |     0.975 | 0.788 |
| Ensemble        |  0.424 |     1.000 | 0.595 |

Foi realizada também uma análise exploratória utilizando limiares específicos por classe escolhidos a partir de predições fora da amostra.

Nesse cenário, o recall do alerta permissivo no Hold-Out foi:

| Modelo          | Recall |
| --------------- | -----: |
| YOLOv8          |  0.949 |
| SSD MobileNetV3 |  0.831 |
| Ensemble        |  0.949 |

Essa análise é **exploratória** e não substitui a avaliação utilizando o limiar padrão.

---

## Inferência em imagem externa

O notebook possui uma etapa de inferência interativa que permite carregar uma imagem ou vídeo externo e executar as três abordagens.

Durante a execução registrada, foi analisada uma imagem externa.

Os modelos produziram diferentes contagens de objetos, incluindo a identificação de `no_helmet` pelo SSD MobileNetV3 e pelo Ensemble.

Essa etapa tem finalidade demonstrativa e não faz parte da avaliação estatística principal.

---

## Limitações

O experimento possui as seguintes limitações:

### Diferenças entre arquiteturas

O YOLOv8 utiliza entrada de 640×640, detector completo pré-treinado no COCO e técnicas de augmentation como Mosaic.

O SSD MobileNetV3 utiliza SSDLite320, entrada de 320×320 e uma configuração diferente de treinamento.

Portanto, parte da diferença observada entre os modelos pode estar relacionada às diferenças de configuração e não exclusivamente à arquitetura.

### Ausência de `no_vest`

O dataset não possui a classe `no_vest`.

Consequentemente, não é possível avaliar a violação relacionada à ausência de colete utilizando as classes fornecidas pelo dataset.

### Classes de violação com poucas amostras

A classe `no_boots` apresenta baixa quantidade de exemplos, principalmente no Hold-Out.

Isso aumenta a incerteza das métricas dessa classe.

### Quase-duplicatas

Foram identificadas imagens muito semelhantes entre diferentes blocos do experimento.

Foram encontrados:

* 126 pares com correlação de miniaturas ≥ 0,97;
* 168 imagens pertencentes a grupos com duas ou mais quase-duplicatas;
* 29 imagens do Hold-Out com quase-duplicata no conjunto de desenvolvimento.

A análise de sensibilidade mostrou que, removendo as imagens do Hold-Out com quase-duplicatas no desenvolvimento, o mAP caiu:

| Modelo   | mAP completo | mAP sem quase-duplicatas |
| -------- | -----------: | -----------------------: |
| YOLOv8   |        0.568 |                    0.554 |
| SSD      |        0.369 |                    0.357 |
| Ensemble |        0.528 |                    0.513 |

### FPS

O FPS depende do hardware, da sessão e do caminho utilizado para inferência.

Por isso, os valores apresentados foram obtidos em um benchmark específico, realizado no mesmo ambiente.

### Generalização

O experimento utiliza apenas o dataset Construction-PPE.

A distribuição de iluminação, ângulo de câmera, ambientes e características das imagens pode não representar todas as condições encontradas em situações reais.

### Otimização

Não foi realizada uma busca completa de hiperparâmetros.

Além disso, o Ensemble utiliza pesos fixos `[1, 1]` na fusão.

### Aspectos éticos e privacidade

As imagens utilizadas podem apresentar pessoas e rostos.

Imagens do dataset não devem ser redistribuídas sem verificar sua licença e as condições de uso.

Os resultados de um detector de EPI devem ser utilizados como apoio à análise, e não como único mecanismo para decisões punitivas ou de segurança.

---

## Próximos passos

Entre as possibilidades de continuidade estão:

* validação agrupada por cena, caso existam metadados disponíveis;
* calibração dos limiares por classe em dados novos;
* rebalanceamento das classes de violação;
* utilização de *focal loss*;
* ampliação do número de exemplos de `no_boots`;
* avaliação com diferentes configurações do SSD;
* redução da assimetria entre as resoluções das arquiteturas;
* avaliação por diferentes condições de iluminação;
* avaliação por subgrupos;
* realização de novos experimentos em datasets externos.

---

## Arquivos gerados

Durante a execução são produzidos arquivos de resultados, checkpoints e tabelas.

Entre eles:

```text
tabelas_relatorio.md
metricas_por_classe_cv.csv
oof_predictions.pkl
fps_mesmo_ambiente.csv
```

Os pesos treinados são armazenados nas pastas:

```text
runs_yolo/
runs_ssd/
```

---

## Reprodutibilidade

O experimento utiliza:

* divisão Hold-Out;
* validação cruzada estratificada;
* sementes para execuções do SSD;
* registro do ambiente e GPU;
* checkpoints;
* armazenamento dos resultados no Google Drive;
* tabelas geradas automaticamente a partir da execução.

O notebook pode ser executado no Google Colab ou, com adaptações mínimas de armazenamento, em um ambiente local.

---

## Licença e uso dos dados

O projeto utiliza o dataset Construction-PPE e bibliotecas relacionadas ao ecossistema Ultralytics.

Antes de redistribuir imagens, pesos ou outros arquivos derivados, deve-se verificar as licenças aplicáveis ao dataset, ao código e aos modelos utilizados.

O notebook registra como limitação legal a necessidade de verificar a licença **AGPL-3.0** associada à biblioteca Ultralytics e as condições de uso do dataset.

---

## Referências

* ULTRALYTICS. *Ultralytics YOLO*. Disponível em: https://github.com/ultralytics/ultralytics.
* ULTRALYTICS. *Construction-PPE Dataset*. Disponível por meio do ecossistema Ultralytics.
* GOOGLE. *Google Colaboratory*. Disponível em: https://colab.research.google.com/.

---

