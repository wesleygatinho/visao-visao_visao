"""
Grad-CAM (explicabilidade) para o modelo AnyNet.

Gera um mapa de calor mostrando quais regioes do rosto mais influenciaram a
classe prevista. Use o target_layer = model.last_conv_stage.

Referencia: Selvaraju et al., "Grad-CAM" (ICCV 2017).
"""
import torch
import torch.nn.functional as F
import numpy as np


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_act)
        target_layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, module, inp, out):
        self.activations = out.detach()

    def _save_grad(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(output.argmax(1))
        output[0, class_idx].backward()

        # pesos = media global dos gradientes por canal
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=input_tensor.shape[2:],
                            mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam  # [H,W] em [0,1]


if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.anynet import AnyNet
    net = AnyNet(24, [1, 2, 2], [32, 64, 128], num_classes=8, in_channels=1)
    cam = GradCAM(net, net.last_conv_stage)
    heat = cam(torch.randn(1, 1, 48, 48))
    print("heatmap:", heat.shape, heat.min(), heat.max())
