"""
Aplicacao de webcam: detecta TODAS as faces no ambiente com YOLO e desenha a
expressao facial prevista em cada bounding box.

Pipeline:
    webcam -> YOLO (deteccao de rostos) -> recorte -> AnyNet (classificacao) -> overlay

Deteccao de rosto:
    Usa um modelo YOLOv8 treinado para faces. Por padrao baixa os pesos
    'arnabdhar/YOLOv8-Face-Detection' do Hugging Face. Se o ultralytics/HF nao
    estiver disponivel, cai para o detector Haar do OpenCV automaticamente.

Uso:
    python app/webcam_app.py --ckpt checkpoints/anynet_ferplus.pt
    python app/webcam_app.py --ckpt ... --explain     # com Grad-CAM
"""
import os
import sys
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.anynet import AnyNet


# ---------------------------------------------------------------- detector
class YoloFaceDetector:
    def __init__(self, weights=None, conf=0.4):
        from ultralytics import YOLO
        if weights is None:
            from huggingface_hub import hf_hub_download
            weights = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection",
                                      filename="model.pt")
        self.model = YOLO(weights)
        self.conf = conf

    def detect(self, frame):
        res = self.model(frame, conf=self.conf, verbose=False)[0]
        boxes = []
        for b in res.boxes.xyxy.cpu().numpy().astype(int):
            boxes.append(tuple(b[:4]))
        return boxes


class HaarFaceDetector:
    """Fallback simples caso o YOLO nao esteja disponivel."""
    def __init__(self):
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(path)

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        return [(x, y, x + w, y + h) for (x, y, w, h) in faces]


def get_detector(weights):
    try:
        return YoloFaceDetector(weights)
    except Exception as e:
        print(f"[aviso] YOLO indisponivel ({e}); usando Haar cascade.")
        return HaarFaceDetector()


# ---------------------------------------------------------------- classificador
def load_classifier(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device)
    model = AnyNet(**ck["hparams"])
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    classes = ck.get("classes") or [str(i) for i in range(ck["hparams"]["num_classes"])]
    return model, classes


def preprocess_face(face_bgr, img_size=config.IMG_SIZE):
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (img_size, img_size))
    t = torch.from_numpy(gray).float().div(255.0)
    t = (t - config.MEAN[0]) / config.STD[0]
    return t.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]


# ---------------------------------------------------------------- loop
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(config.CKPT_DIR, "anynet_ferplus.pt"))
    ap.add_argument("--face_weights", default=None,
                    help="caminho .pt do YOLO-face (default: baixa do HuggingFace)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--explain", action="store_true", help="sobrepoe Grad-CAM no rosto")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, classes = load_classifier(args.ckpt, device)
    detector = get_detector(args.face_weights)

    cam = None
    if args.explain:
        from explain.gradcam import GradCAM
        cam = GradCAM(model, model.last_conv_stage)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Nao consegui abrir a webcam.")
    print("Pressione 'q' para sair.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for (x1, y1, x2, y2) in detector.detect(frame):
            x1, y1 = max(0, x1), max(0, y1)
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue
            inp = preprocess_face(face).to(device)

            with torch.no_grad():
                probs = F.softmax(model(inp), dim=1)[0]
            idx = int(probs.argmax())
            label = f"{classes[idx]} {probs[idx]*100:.0f}%"

            if cam is not None:
                heat = cam(inp, idx)                       # [H,W] em [0,1]
                heat = cv2.resize(heat, (x2 - x1, y2 - y1))
                heat = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
                frame[y1:y2, x1:x2] = cv2.addWeighted(face, 0.6, heat, 0.4, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Expressao Facial (q=sair)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
