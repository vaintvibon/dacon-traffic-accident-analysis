from torch import nn
from torchvision.models.video import mvit_v2_s


class Stage3MViT(nn.Module):
    def __init__(self):
        super().__init__(); self.backbone=mvit_v2_s(weights=None)
        dim=self.backbone.head[1].in_features; self.backbone.head=nn.Identity()
        self.accel=nn.Linear(dim,4); self.steer=nn.Linear(dim,3)
    def forward(self,x):
        z=self.backbone(x); return self.accel(z),self.steer(z)
