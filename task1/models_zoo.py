import torch
import torch.nn as nn
import timm

# -------------------------
# 1) Basic CNN
# -------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),      # 28->14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),     # 14->7
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.net(x).flatten(1)
        return self.head(x)

# -------------------------
# 2) MambaNet (lightweight "Mamba-like" SSM token mixer)
#    NOTE: This is a practical Mamba-style model without requiring external mamba-ssm.
#    It treats each pixel as a token, uses 1D conv mixing + gated residual blocks.
# -------------------------
class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 64, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_model)  # for gating
        # token mixer (SSM-ish): depthwise conv over sequence (fast, stable)
        self.dwconv = nn.Conv1d(d_model, d_model, kernel_size=7, padding=3, groups=d_model)
        self.state = nn.Conv1d(d_model, d_model, kernel_size=1)  # state mixing
        self.out_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, L, D]
        h = self.norm(x)
        gate, val = self.in_proj(h).chunk(2, dim=-1)
        gate = torch.sigmoid(gate)

        # conv expects [B, D, L]
        v = val.transpose(1, 2)
        v = self.dwconv(v)
        v = self.state(v)
        v = v.transpose(1, 2)  # back to [B, L, D]

        y = gate * v
        y = self.out_proj(y)
        y = self.drop(y)
        return x + y

class MambaNet(nn.Module):
    def __init__(self, img_size=28, d_model=128, depth=6, num_classes=2, dropout=0.1):
        super().__init__()
        self.img_size = img_size
        self.seq_len = img_size * img_size

        # patch embedding = each pixel token (grayscale)
        self.embed = nn.Linear(1, d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.seq_len, d_model))

        self.blocks = nn.Sequential(*[MambaBlock(d_model, dropout=dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x):
        # x: [B,1,H,W] -> tokens [B,L,1]
        B, C, H, W = x.shape
        t = x.view(B, 1, H * W).transpose(1, 2)  # [B, L, 1]
        t = self.embed(t) + self.pos            # [B, L, D]
        t = self.blocks(t)
        t = self.norm(t)
        cls = t.mean(dim=1)
        return self.head(cls)

# -------------------------
# Builder
# -------------------------
def build_model(name: str, num_classes=2):
    name = name.lower()
    if name == "simplecnn":
        return SimpleCNN(num_classes=num_classes)
    if name == "resnet18":
        return timm.create_model("resnet18", pretrained=True, num_classes=num_classes, in_chans=1)
    if name == "efficientnet_b0":
        return timm.create_model("efficientnet_b0", pretrained=True, num_classes=num_classes, in_chans=1)
    if name == "vit_tiny":
        return timm.create_model("vit_tiny_patch16_224", pretrained=True, num_classes=num_classes, in_chans=1)
    if name == "mambanet":
        return MambaNet(img_size=28, d_model=128, depth=6, num_classes=num_classes, dropout=0.1)
    raise ValueError(f"Unknown model: {name}")
