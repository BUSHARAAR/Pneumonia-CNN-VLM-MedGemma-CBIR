import argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from medmnist import PneumoniaMNIST

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, num_classes)
    def forward(self, x):
        x = self.net(x).flatten(1)
        return self.head(x)

def build_model(name: str, num_classes: int = 2) -> nn.Module:
    name = name.lower()
    if name == "simplecnn":
        return SimpleCNN(num_classes=num_classes)
    if name == "resnet18":
        return timm.create_model("resnet18", pretrained=False, num_classes=num_classes, in_chans=1)
    if name == "efficientnet_b0":
        return timm.create_model("efficientnet_b0", pretrained=False, num_classes=num_classes, in_chans=1)
    if name == "vit_tiny":
        return timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=num_classes, in_chans=1)
    raise ValueError(f"Unknown model: {name}")

def maybe_resize(x: torch.Tensor, model_name: str) -> torch.Tensor:
    if model_name.lower().startswith("vit"):
        return torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    return x

class Pneumo28(Dataset):
    def __init__(self, base_ds, mean: float, std: float):
        self.ds = base_ds
        self.mean = float(mean)
        self.std = float(std)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, label = self.ds[idx]
        x = torch.from_numpy(np.asarray(img, dtype=np.float32)).unsqueeze(0) / 255.0
        y = int(np.asarray(label).squeeze())
        x = (x - self.mean) / (self.std + 1e-8)
        return x, y

def denorm(x: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return (x * (std + 1e-8) + mean).clamp(0, 1)

@torch.no_grad()
def predict_probs(model: nn.Module, loader: DataLoader, model_name: str, device: str):
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        x = maybe_resize(x.to(device), model_name)
        logits = model(x)
        p = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        probs.append(p)
        ys.append(y.numpy())
    return np.concatenate(probs), np.concatenate(ys)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Path to .pt state_dict (e.g., models/best_efficientnet_b0.pt)")
    ap.add_argument("--meta", required=True, help="Path to _meta.json (e.g., models/best_efficientnet_b0_meta.json)")
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out_dir", default="./outputs")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--pin_memory", type=int, default=1)
    args, _ = ap.parse_known_args()  # ✅ fixes Colab/Jupyter "-f kernel.json" issue

    device = "cuda" if torch.cuda.is_available() else "cpu"

    meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))
    model_name = meta["model_name"]
    mean = float(meta["mean"])
    std = float(meta["std"])

    test_raw = PneumoniaMNIST(split="test", download=True, root=args.data_root)
    test_ds = Pneumo28(test_raw, mean, std)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.num_workers, pin_memory=bool(args.pin_memory)
    )

    model = build_model(model_name).to(device)
    state = torch.load(args.weights, map_location=device)  # ✅ loads cleanly in PyTorch 2.6+
    model.load_state_dict(state)

    probs, y = predict_probs(model, test_loader, model_name, device)
    pred = (probs >= args.threshold).astype(int)

    acc = accuracy_score(y, pred)
    prec = precision_score(y, pred, zero_division=0)
    rec = recall_score(y, pred, zero_division=0)
    f1 = f1_score(y, pred, zero_division=0)
    auc = roc_auc_score(y, probs)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Confusion matrix
    cm = confusion_matrix(y, pred)
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix (Test)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.xticks([0, 1], ["Normal(0)", "Pneumonia(1)"])
    plt.yticks([0, 1], ["Normal(0)", "Pneumonia(1)"])
    plt.savefig(out_dir / "confusion_matrix.png", dpi=200)
    plt.close()

    # ROC
    fpr, tpr, _ = roc_curve(y, probs)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC Curve (Test)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig(out_dir / "roc_curve.png", dpi=200)
    plt.close()

    # Failure cases
    failures_dir = out_dir / "failures"
    failures_dir.mkdir(parents=True, exist_ok=True)

    mis_idx = np.where(pred != y)[0]
    max_show = min(25, len(mis_idx))
    pick = mis_idx[:max_show]

    plt.figure(figsize=(10, 10))
    for k, idx in enumerate(pick):
        x, yt = test_ds[int(idx)]
        x_vis = denorm(x, mean, std).squeeze(0).numpy()
        pr = float(probs[int(idx)])
        yp = int(pred[int(idx)])

        plt.subplot(5, 5, k + 1)
        plt.imshow(x_vis, cmap="gray")
        plt.axis("off")
        plt.title(f"T={yt} P={yp}\nPr={pr:.2f}", fontsize=8)

        plt.imsave(str(failures_dir / f"failure_{idx}_T{yt}_P{yp}_Pr{pr:.2f}.png"), x_vis, cmap="gray")

    plt.suptitle("Misclassified Failure Cases (Test)")
    plt.tight_layout()
    plt.savefig(out_dir / "failure_grid.png", dpi=200)
    plt.close()

    # Save metrics JSON
    metrics = {
        "model": model_name,
        "threshold": args.threshold,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auc": float(auc),
        "confusion_matrix": cm.tolist(),
        "num_failures": int(len(mis_idx)),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("✅ Test Metrics")
    print(json.dumps(metrics, indent=2))
    print("\nSaved outputs to:", out_dir)

if __name__ == "__main__":
    main()
