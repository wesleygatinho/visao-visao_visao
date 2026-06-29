"""Constantes compartilhadas entre treino, otimizacao e aplicacao."""
import os

# Tamanho de entrada da CNN. FER+ vem 48x48 (FER2013) ou 112x112 (versao Kaggle).
# 48 e suficiente e mais rapido; aumente para 64 se tiver GPU sobrando.
IMG_SIZE = 48

# Normalizacao para imagens em tons de cinza (1 canal).
MEAN = [0.5]
STD = [0.5]

# Caminho default do dataset (sobrescreva via --data_dir).
# Espera subpastas: <DATA_DIR>/train/<classe>/*.png  e  <DATA_DIR>/test/<classe>/*.png
DATA_DIR = os.environ.get("FERPLUS_DIR", "./ferplus")

# Onde salvar checkpoints e estudos do Optuna.
CKPT_DIR = "./checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

# Projeto no Weights & Biases.
WANDB_PROJECT = "expressao-facial-anynet"
