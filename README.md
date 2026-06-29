# Reconhecimento de Expressão Facial — CNN estilo AnyNet

Fase 1 (continuação): dado que os rostos foram detectados, **classificar a expressão facial**.
A arquitetura da CNN segue o espaço de projeto **AnyNet/RegNet** do
[d2l.ai](https://d2l.ai/chapter_convolutional-modern/cnn-design.html), é
**otimizada com Optuna**, registra logs no **Weights & Biases**, e roda numa
**aplicação de webcam** que detecta todas as faces com **YOLO** e mostra a
expressão em cada bounding box. Explicabilidade opcional via **Grad-CAM**.

## Estrutura
```
expressao-facial/
├── config.py                # constantes (tamanho de imagem, normalizacao, paths)
├── models/anynet.py         # CNN parametrizavel (STEM -> BODY -> HEAD), bloco ResNeXt
├── data/ferplus.py          # dataloaders FER+ (ImageFolder) + pesos de classe
├── train.py                 # loop de treino + logs no W&B
├── optimize.py              # busca de arquitetura/hiperparametros com Optuna
├── app/webcam_app.py        # webcam: YOLO (faces) + AnyNet (expressao)
├── explain/gradcam.py       # explicabilidade (Grad-CAM)
└── scripts/prepare_ferplus.py  # [opcional] CSV oficial FER+ -> pastas de imagens
```

## Instalação
```bash
pip install -r requirements.txt
```
GPU é fortemente recomendada para o treino (use Colab/Kaggle se não tiver).

## Dados (FER+)
Opção A — versão pronta em imagens (mais fácil):
baixe o dataset Kaggle `arnabkumarroy02/ferplus` (8 emoções, tons de cinza) e
organize/aponte para uma pasta com `train/<classe>/*` e `test/<classe>/*`.

Opção B — rota oficial Microsoft FER+:
baixe `fer2013.csv` (desafio FER2013) e `fer2013new.csv` (repo microsoft/FERPlus), e gere as pastas:
```bash
python scripts/prepare_ferplus.py --fer fer2013.csv --ferplus fer2013new.csv --out ./ferplus
```

## Passo a passo (mapeia as etapas do trabalho)

**1. Treinar a CNN AnyNet** (login no W&B na 1ª vez: `wandb login`)
```bash
python train.py --data_dir ./ferplus --epochs 40
# sem W&B:  python train.py --data_dir ./ferplus --epochs 40 --no_wandb
```

**2. Otimizar a arquitetura com Optuna**
```bash
python optimize.py --data_dir ./ferplus --trials 30 --epochs 12
```
Salva os melhores hiperparâmetros em `checkpoints/best_params.json`. Depois
edite a arquitetura em `train.py` com esses valores e retreine por mais épocas.

**3. Logs no Weights & Biases** — já integrados em `train.py` e `optimize.py`
(projeto `expressao-facial-anynet`). Cada trial do Optuna vira uma run.

**4. Aplicação de webcam (YOLO + expressão)**
```bash
python app/webcam_app.py --ckpt checkpoints/anynet_ferplus.pt
```
O detector de rosto baixa pesos YOLOv8-face do Hugging Face automaticamente; se
falhar, cai para o Haar cascade do OpenCV.

**5. [Opcional] Explicabilidade (Grad-CAM)**
```bash
python app/webcam_app.py --ckpt checkpoints/anynet_ferplus.pt --explain
```

## Notas de implementação
- **STEM → BODY (estágios de blocos ResNeXt) → HEAD**: exatamente o esqueleto
  AnyNet. Os graus de liberdade do espaço de projeto (`depths`, `widths`,
  `group_width`, `bot_mul`, `stem_channels`) são o que o Optuna varre.
- O Optuna usa `TPESampler` + `MedianPruner` para podar trials ruins cedo.
- O checkpoint guarda `hparams` e `classes`, então a app reconstrói o modelo e
  usa a ordem correta de rótulos sem você configurar nada.
