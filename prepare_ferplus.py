"""
[OPCIONAL] Converte os CSVs oficiais do Microsoft FER+ em pastas de imagens
(estrutura ImageFolder usada por data/ferplus.py).

So precisa disto se voce baixar a rota oficial:
  - fer2013.csv      (imagens; do desafio FER2013 no Kaggle)
  - fer2013new.csv   (rotulos FER+; do repo github.com/microsoft/FERPlus)

Se voce baixou a versao ja-em-imagens (kaggle: arnabkumarroy02/ferplus),
IGNORE este script.

Uso:
    python scripts/prepare_ferplus.py --fer fer2013.csv --ferplus fer2013new.csv --out ./ferplus
"""
import os
import argparse
import csv
import numpy as np
from PIL import Image

EMOTIONS = ["neutral", "happiness", "surprise", "sadness", "anger",
            "disgust", "fear", "contempt"]   # ignora 'unknown' e 'NF'
USAGE_TO_SPLIT = {"Training": "train", "PublicTest": "valid", "PrivateTest": "test"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fer", required=True)
    ap.add_argument("--ferplus", required=True)
    ap.add_argument("--out", default="./ferplus")
    args = ap.parse_args()

    with open(args.fer) as f1, open(args.ferplus) as f2:
        fer = list(csv.reader(f1)); fnew = list(csv.reader(f2))
    fer, fnew = fer[1:], fnew[1:]   # remove cabecalhos

    kept = 0
    for i, (frow, nrow) in enumerate(zip(fer, fnew)):
        usage = nrow[0]
        votes = np.array([int(v) for v in nrow[2:2 + len(EMOTIONS)]])
        if votes.sum() == 0:
            continue
        label = EMOTIONS[int(votes.argmax())]            # rotulo por maioria
        split = USAGE_TO_SPLIT.get(usage)
        if split is None:
            continue
        pixels = np.array(frow[1].split(), dtype=np.uint8).reshape(48, 48)
        d = os.path.join(args.out, split, label)
        os.makedirs(d, exist_ok=True)
        Image.fromarray(pixels).save(os.path.join(d, f"{i:05d}.png"))
        kept += 1

    print(f"{kept} imagens exportadas para {args.out}")


if __name__ == "__main__":
    main()
