from torch import nn
from torchvision.models.video import mvit_v2_s


class Stage1MViT(nn.Module):
    def __init__(self):
        super().__init__(); self.net=mvit_v2_s(weights=None)
        self.net.head[1]=nn.Linear(self.net.head[1].in_features,2)
    def forward(self,x): return self.net(x)
