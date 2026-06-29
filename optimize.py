"""
Otimizacao da arquitetura AnyNet com Optuna.

O Optuna pesquisa o espaco de projeto AnyNet/RegNet (profundidade e largura por
estagio, group width, bottleneck) + hiperparametros de treino. Cada trial e
registrado no Weights & Biases e trials ruins sao podados (MedianPruner).

Uso:
    python optimize.py --data_dir ./ferplus --trials 30 --epochs 12
"""
import os
import json
import argparse
import optuna
import torch

import config
from models.anynet import AnyNet
from data.ferplus import get_dataloaders
from train import train_model


def make_widths(w0, mult, n_stages, step=8):
    """Larguras crescentes quantizadas em multiplos de `step` (estilo RegNet)."""
    widths, w = [], w0
    for _ in range(n_stages):
        widths.append(int(max(step, round(w / step) * step)))
        w *= mult
    return widths


# Removemos os loaders dos argumentos da função principal
def build_objective(args):
    def objective(trial):
        # 1. Carregamos os dados AQUI DENTRO.
        # Assim, os workers morrem e liberam a memória ao fim de cada trial.
        train_loader, val_loader, classes, class_weights = get_dataloaders(
            args.data_dir, img_size=args.img_size, batch_size=args.batch_size)

        # ---- espaco de projeto AnyNet ----
        n_stages = trial.suggest_int("n_stages", 2, 4)
        stem_channels = trial.suggest_categorical("stem_channels", [16, 24, 32])
        w0 = trial.suggest_categorical("w0", [16, 24, 32])
        wmult = trial.suggest_float("wmult", 1.5, 2.5)
        depths = [trial.suggest_int(f"depth_{i}", 1, 3) for i in range(n_stages)]
        widths = make_widths(w0, wmult, n_stages)
        group_width = trial.suggest_categorical("group_width", [4, 8, 16])
        bot_mul = trial.suggest_categorical("bot_mul", [0.5, 1.0])

        # ---- hiperparametros de treino ----
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
        optimizer_name = trial.suggest_categorical("optimizer", ["adamw", "sgd"])

        model = AnyNet(stem_channels, depths, widths, group_width, bot_mul,
                       num_classes=len(classes), in_channels=1)
        model.classes = classes

        # log do trial no W&B
        wandb_run = None
        if not args.no_wandb:
            import wandb
            wandb_run = wandb.init(project=config.WANDB_PROJECT, group="optuna",
                                   name=f"trial-{trial.number}", reinit=True,
                                   config={**trial.params, "widths": widths,
                                           "params": model.num_params()})
        try:
            acc = train_model(model, train_loader, val_loader, class_weights,
                              epochs=args.epochs, lr=lr, weight_decay=weight_decay,
                              optimizer_name=optimizer_name, wandb_run=wandb_run,
                              trial=trial)
        finally:
            if wandb_run:
                wandb_run.finish()
                import time
                time.sleep(2) # Pausa rápida para garantir que os sockets fecharam
            
            # Limpeza forçada da memória RAM e placa de vídeo
            import gc
            del train_loader, val_loader, model
            gc.collect()
            torch.cuda.empty_cache() 

        return acc
    return objective


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=config.DATA_DIR)
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=12, help="epocas por trial (curto)")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--img_size", type=int, default=config.IMG_SIZE)
    ap.add_argument("--no_wandb", action="store_true")
    args = ap.parse_args()

    train_loader, val_loader, classes, class_weights = get_dataloaders(
        args.data_dir, img_size=args.img_size, batch_size=args.batch_size)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),
        study_name="anynet-ferplus",
        storage=f"sqlite:///{os.path.join(config.CKPT_DIR, 'optuna.db')}",
        load_if_exists=True)

    study.optimize(build_objective(args), n_trials=args.trials)

    print("\n== melhor trial ==")
    print("val_acc:", study.best_value)
    print("params:", json.dumps(study.best_params, indent=2))

    out = os.path.join(config.CKPT_DIR, "best_params.json")
    with open(out, "w") as f:
        json.dump({"value": study.best_value, "params": study.best_params,
                   "widths": make_widths(study.best_params["w0"],
                                         study.best_params["wmult"],
                                         study.best_params["n_stages"])}, f, indent=2)
    print("salvo em", out)
    print("\nRetreine a melhor arquitetura por mais epocas com train.py "
          "usando esses parametros.")


if __name__ == "__main__":
    main()
