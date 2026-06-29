"""
Rede no estilo AnyNet / RegNet (espaco de projeto descrito em
https://d2l.ai/chapter_convolutional-modern/cnn-design.html).

A rede e dividida em STEM -> BODY (varios estagios) -> HEAD.
Cada estagio e uma pilha de blocos ResNeXt. A arquitetura inteira fica
parametrizada por:
  - stem_channels : largura da convolucao inicial
  - depths        : numero de blocos por estagio  (d_1, d_2, ...)
  - widths        : numero de canais por estagio   (w_1, w_2, ...)
  - group_width   : largura do grupo nas conv. agrupadas (g)
  - bot_mul       : razao do bottleneck (b)

Esses sao exatamente os "graus de liberdade" do espaco AnyNet que o Optuna
ira pesquisar em optimize.py.
"""
import torch
from torch import nn
from torch.nn import functional as F


class ResNeXtBlock(nn.Module):
    """Bloco ResNeXt com bottleneck + convolucao agrupada (igual ao d2l)."""

    def __init__(self, in_channels, out_channels, group_width, bot_mul, stride=1):
        super().__init__()
        # canais do bottleneck; precisam ser multiplos de group_width
        bot = int(round(out_channels * bot_mul))
        bot = max(group_width, (bot // group_width) * group_width)
        num_groups = bot // group_width

        self.conv1 = nn.Conv2d(in_channels, bot, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bot)
        self.conv2 = nn.Conv2d(bot, bot, kernel_size=3, stride=stride,
                               padding=1, groups=num_groups, bias=False)
        self.bn2 = nn.BatchNorm2d(bot)
        self.conv3 = nn.Conv2d(bot, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = F.relu(self.bn2(self.conv2(y)), inplace=True)
        y = self.bn3(self.conv3(y))
        return F.relu(y + self.shortcut(x), inplace=True)


class AnyNet(nn.Module):
    """CNN parametrizavel no estilo AnyNet/RegNet."""

    def __init__(self, stem_channels, depths, widths, group_width=8,
                 bot_mul=1.0, num_classes=8, in_channels=1):
        super().__init__()
        assert len(depths) == len(widths), "depths e widths precisam ter o mesmo tamanho"
        self.hparams = dict(stem_channels=stem_channels, depths=list(depths),
                            widths=list(widths), group_width=group_width,
                            bot_mul=bot_mul, num_classes=num_classes,
                            in_channels=in_channels)

        # ---- STEM: reduz a resolucao pela metade ----
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, kernel_size=3,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True))

        # ---- BODY: estagios sucessivos, cada um reduz a resolucao ----
        stages = []
        prev = stem_channels
        for d, w in zip(depths, widths):
            blocks = []
            for i in range(d):
                stride = 2 if i == 0 else 1          # 1o bloco reduz resolucao
                cin = prev if i == 0 else w
                blocks.append(ResNeXtBlock(cin, w, group_width, bot_mul, stride))
            stages.append(nn.Sequential(*blocks))
            prev = w
        self.body = nn.Sequential(*stages)

        # ---- HEAD: pooling global + classificador ----
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(prev, num_classes))

    def forward(self, x):
        return self.head(self.body(self.stem(x)))

    @property
    def last_conv_stage(self):
        """Ultimo estagio do body — alvo para Grad-CAM."""
        return self.body[-1]

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


def build_from_config(cfg, num_classes, in_channels=1):
    """Cria um AnyNet a partir de um dict de hiperparametros (usado pelo Optuna)."""
    return AnyNet(
        stem_channels=cfg["stem_channels"],
        depths=cfg["depths"],
        widths=cfg["widths"],
        group_width=cfg["group_width"],
        bot_mul=cfg["bot_mul"],
        num_classes=num_classes,
        in_channels=in_channels)


if __name__ == "__main__":
    # teste rapido de sanidade
    net = AnyNet(stem_channels=24, depths=[1, 2, 2], widths=[32, 64, 128],
                 group_width=8, bot_mul=1.0, num_classes=8, in_channels=1)
    x = torch.randn(2, 1, 48, 48)
    y = net(x)
    print("saida:", y.shape, "| parametros:", f"{net.num_params():,}")
