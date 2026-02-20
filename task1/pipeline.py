import argparse, json, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from medmnist import PneumoniaMNIST
from models_zoo import build_model

def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

class Pneumo28(Dataset):
    def __init__(self, base_ds, mean, std, augment=False):
        self.ds = base_ds
        self.mean = float(mean); self.std = float(std)
        self.augment = augment

    def __len__(self): return len(self.ds)

    def _augment(self, x):
        if random.random() < 0.8:
            b = random.uniform(0.85, 1.15)
            c = random.uniform(0.85, 1.15)
            x = TF.adjust_brightness(x, b)
            x = TF.adjust_contrast(x, c)

        if random.random() < 0.5:
            angle = random.uniform(-7, 7)
            x = TF.rotate(x, angle=angle)

        if random.random() < 0.5:
            dx = random.randint(-2, 2)
            dy = random.randint(-2, 2)
            x = TF.affine(x, angle=0, translate=[dx, dy], scale=1.0, shear=[0.0, 0.0])

        if random.random() < 0.5:
            x = torch.clamp(x + 0.02 * torch.randn_like(x), 0.0, 1.0)
        return x

    def __getitem__(self, idx):
        img, label = self.ds[idx]
        x = torch.from_numpy(np.asarray(img, dtype=np.float32)).unsqueeze(0) / 255.0
        y = int(np.asarray(label).squeeze())
        if self.augment:
            x = self._augment(x)
        x = (x - self.mean) / (self.std + 1e-8)
        return x, y

def maybe_resize(x, model_name):
    # ViT needs 224
    if model_name.lower().startswith("vit"):
        return torch.nn.functional.interpolate(x, size=(224,224), mode="bilinear", align_corners=False)
    return x

def compute_mean_std(ds):
    s1=s2=0.0
    n=len(ds)
    for i in range(n):
        img,_ = ds[i]
        arr = np.asarray(img, dtype=np.float32)/255.0
        s1 += float(arr.mean())
        s2 += float((arr**2).mean())
    mean = s1/n
    var = (s2/n) - mean**2
    std = float(np.sqrt(max(var, 1e-12)))
    return float(mean), float(std)

@torch.no_grad()
def predict_probs(model, loader, model_name, device):
    model.eval()
    probs, ys = [], []
    for x,y in loader:
        x = maybe_resize(x.to(device), model_name)
        logits = model(x)
        p = torch.softmax(logits, dim=1)[:,1].cpu().numpy()
        probs.append(p); ys.append(y.numpy())
    return np.concatenate(probs), np.concatenate(ys)

def denorm(x, mean, std):
    return (x*(std+1e-8)+mean).clamp(0,1)

def train_one_model(model_name, train_loader, val_loader, mean, std, args, device, out_models):
    model = build_model(model_name).to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_auc = -1.0
    history = {"train_loss": [], "val_auc": []}

    weights_path = out_models / f"best_{model_name}.pt"
    meta_path    = out_models / f"best_{model_name}_meta.json"

    for ep in range(1, args.epochs+1):
        model.train()
        running = 0.0
        for x,y in train_loader:
            x = maybe_resize(x.to(device), model_name)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            running += loss.item() * y.size(0)
        sch.step()

        # val AUC
        val_probs, val_y = predict_probs(model, val_loader, model_name, device)
        val_auc = float(roc_auc_score(val_y, val_probs))

        train_loss = running / len(train_loader.dataset)
        history["train_loss"].append(float(train_loss))
        history["val_auc"].append(val_auc)

        print(f"[{model_name}] Epoch {ep:02d} | train_loss={train_loss:.4f} | val_auc={val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), weights_path)
            meta = {
                "model_name": model_name,
                "mean": mean, "std": std,
                "best_val_auc": best_auc,
                "hyperparams": vars(args),
                "history": history
            }
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return weights_path, meta_path

def eval_one_model(model_name, weights_path, meta_path, test_ds, test_loader, out_dir, device, threshold=0.5):
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    mean, std = float(meta["mean"]), float(meta["std"])

    model = build_model(model_name).to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)

    probs, y = predict_probs(model, test_loader, model_name, device)
    pred = (probs >= threshold).astype(int)

    acc  = float(accuracy_score(y, pred))
    prec = float(precision_score(y, pred, zero_division=0))
    rec  = float(recall_score(y, pred, zero_division=0))
    f1   = float(f1_score(y, pred, zero_division=0))
    auc  = float(roc_auc_score(y, probs))
    cm   = confusion_matrix(y, pred)

    model_out = out_dir / model_name
    model_out.mkdir(parents=True, exist_ok=True)

    # Confusion matrix
    plt.figure()
    plt.imshow(cm)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted"); plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i,j]), ha="center", va="center")
    plt.xticks([0,1], ["Normal(0)", "Pneumonia(1)"])
    plt.yticks([0,1], ["Normal(0)", "Pneumonia(1)"])
    plt.savefig(model_out / "confusion_matrix.png", dpi=200)
    plt.close()

    # ROC
    fpr, tpr, _ = roc_curve(y, probs)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0,1],[0,1], linestyle="--")
    plt.title(f"ROC - {model_name}")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.savefig(model_out / "roc_curve.png", dpi=200)
    plt.close()

    # Failure cases (grid + individuals)
    failures_dir = model_out / "failures"
    failures_dir.mkdir(parents=True, exist_ok=True)
    mis_idx = np.where(pred != y)[0]
    pick = mis_idx[:25]

    plt.figure(figsize=(10,10))
    for k, idx in enumerate(pick):
        x, yt = test_ds[int(idx)]
        x_vis = denorm(x, mean, std).squeeze(0).numpy()
        pr = float(probs[int(idx)])
        yp = int(pred[int(idx)])
        plt.subplot(5,5,k+1)
        plt.imshow(x_vis, cmap="gray")
        plt.axis("off")
        plt.title(f"T={yt} P={yp}\nPr={pr:.2f}", fontsize=8)
        plt.imsave(str(failures_dir / f"failure_{idx}_T{yt}_P{yp}_Pr{pr:.2f}.png"), x_vis, cmap="gray")
    plt.suptitle(f"Failure Cases - {model_name}")
    plt.tight_layout()
    plt.savefig(model_out / "failure_grid.png", dpi=200)
    plt.close()

    metrics = {
        "model": model_name,
        "threshold": threshold,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "confusion_matrix": cm.tolist(),
        "num_failures": int(len(mis_idx))
    }
    (model_out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="simplecnn,resnet18,efficientnet_b0,vit_tiny,mambanet")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--out_models", default="./models")
    ap.add_argument("--out_dir", default="./outputs_compare")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--pin_memory", type=int, default=1)
    args, _ = ap.parse_known_args()  # ✅ Colab safe

    seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load raw splits
    train_raw = PneumoniaMNIST(split="train", download=True, root=args.data_root)
    val_raw   = PneumoniaMNIST(split="val", download=True, root=args.data_root)
    test_raw  = PneumoniaMNIST(split="test", download=True, root=args.data_root)

    mean, std = compute_mean_std(train_raw)

    train_ds = Pneumo28(train_raw, mean, std, augment=True)
    val_ds   = Pneumo28(val_raw, mean, std, augment=False)
    test_ds  = Pneumo28(test_raw, mean, std, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.num_workers, pin_memory=bool(args.pin_memory))
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.num_workers, pin_memory=bool(args.pin_memory))
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                             num_workers=args.num_workers, pin_memory=bool(args.pin_memory))

    out_models = Path(args.out_models); out_models.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    results = []

    # Train + Eval each model
    for m in models:
        print("\n==============================")
        print("Training:", m)
        weights_path, meta_path = train_one_model(m, train_loader, val_loader, mean, std, args, device, out_models)

        print("Evaluating:", m)
        metrics = eval_one_model(m, weights_path, meta_path, test_ds, test_loader, out_dir, device, threshold=args.threshold)
        results.append(metrics)

    # Save summary
    summary_path = out_dir / "summary_metrics.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nSaved summary:", summary_path)

    # Print table-like view
    print("\n=== Comparative Results (Test) ===")
    for r in sorted(results, key=lambda x: x["auc"], reverse=True):
        print(f'{r["model"]:14s} | AUC={r["auc"]:.4f} | Acc={r["accuracy"]:.4f} | F1={r["f1"]:.4f} | R={r["recall"]:.4f} | P={r["precision"]:.4f}')

if __name__ == "__main__":
    main()
