"""
Treino da CNN AnyNet no FER+ com logs no Weights & Biases.

Uso:
    python train.py --data_dir ./ferplus --epochs 40

A funcao train_model() e reutilizada pelo optimize.py (Optuna), por isso ela
aceita um trial opcional para reportar metricas intermediarias / pruning.
"""
import os
import argparse
import torch
from torch import nn

import config
from models.anynet import AnyNet
from data.ferplus import get_dataloaders


def evaluate(model, loader, device, criterion):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss_sum += criterion(out, y).item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
    return loss_sum / total, correct / total


def train_model(model, train_loader, val_loader, class_weights, *,
                epochs=40, lr=1e-3, weight_decay=5e-4, optimizer_name="adamw",
                label_smoothing=0.1, device=None, wandb_run=None, trial=None,
                ckpt_path=None, patience=10, min_delta=1e-4):
    """Treina o modelo e retorna a melhor acuracia de validacao.

    Early stopping: para o treino se `val_acc` nao melhorar em pelo menos
    `min_delta` por `patience` epocas seguidas. Isso e independente do
    scheduler de LR (CosineAnnealingLR), que continua decaindo por
    cronograma fixo mesmo sem early stopping. Use `patience=None`
    (ou `patience >= epochs`) para desligar o early stopping.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device),
                                    label_smoothing=label_smoothing)
    if optimizer_name == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                              weight_decay=weight_decay, nesterov=True)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_acc = 0.0
    epochs_no_improve = 0
    for epoch in range(epochs):
        model.train()
        run_loss, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            opt.step()
            run_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
        sched.step()

        train_loss, train_acc = run_loss / total, correct / total
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)

        if val_acc > best_acc + min_delta:
            best_acc = val_acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(f"[{epoch+1:03d}/{epochs}] "
              f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.4f}")

        if wandb_run is not None:
            wandb_run.log({"epoch": epoch, "train_loss": train_loss,
                           "train_acc": train_acc, "val_loss": val_loss,
                           "val_acc": val_acc, "lr": sched.get_last_lr()[0]})

        if ckpt_path and epochs_no_improve == 0:
            torch.save({"state_dict": model.state_dict(),
                        "hparams": model.hparams,
                        "classes": getattr(model, "classes", None),
                        "val_acc": val_acc}, ckpt_path)

        # integracao com Optuna: reporta e permite pruning de trials ruins
        if trial is not None:
            trial.report(val_acc, epoch)
            import optuna
            if trial.should_prune():
                raise optuna.TrialPruned()

        # early stopping: independente do scheduler de LR
        if patience is not None and epochs_no_improve >= patience:
            print(f"early stopping na epoca {epoch+1}/{epochs} "
                  f"(sem melhora por {patience} epocas, best_acc={best_acc:.4f})")
            break

    return best_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=config.DATA_DIR)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--img_size", type=int, default=config.IMG_SIZE)
    ap.add_argument("--no_wandb", action="store_true")
    ap.add_argument("--patience", type=int, default=10,
                    help="epocas sem melhora de val_acc antes de parar (early stopping)")
    args = ap.parse_args()

    train_loader, val_loader, classes, class_weights = get_dataloaders(
        args.data_dir, img_size=args.img_size, batch_size=args.batch_size)
    print("classes:", classes)

    # arquitetura inicial razoavel (substitua pelos melhores params do Optuna)
    model = AnyNet(
        stem_channels=24, 
        depths=[3, 2, 3, 2],      # depth_0, depth_1, depth_2, depth_3
        widths=[32, 48, 80, 136], # A lista "widths" do final do JSON
        group_width=8, 
        bot_mul=1.0,
        num_classes=len(classes), 
        in_channels=1
    )
    model.classes = classes
    print(f"parametros: {model.num_params():,}")

    wandb_run = None
    if not args.no_wandb:
        import wandb
        wandb_run = wandb.init(project=config.WANDB_PROJECT, config=vars(args))
        wandb_run.watch(model, log="all", log_freq=200)

    ckpt = os.path.join(config.CKPT_DIR, "anynet_ferplus.pt")
    best = train_model(
        model, train_loader, val_loader, class_weights,
        epochs=args.epochs, 
        lr=0.00486,                   # 'lr' arredondado do JSON
        weight_decay=0.000511,        # 'weight_decay' do JSON
        optimizer_name="adamw",       # 'optimizer' do JSON
        wandb_run=wandb_run,
        ckpt_path=ckpt,
        patience=args.patience
    )
    print(f"melhor val_acc: {best:.4f}  ->  {ckpt}")
    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
