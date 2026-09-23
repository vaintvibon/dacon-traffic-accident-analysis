import torch
from torch import nn


class Stage2Temporal(nn.Module):
    def __init__(self):
        super().__init__()
        self.r=nn.GRU(512,192,2,batch_first=True,bidirectional=True,dropout=0.15)
        self.tc=nn.Linear(384,1); self.te=nn.Linear(384,1)
        self.scene=nn.Sequential(nn.Linear(768,192),nn.ReLU(),nn.Dropout(0.2),nn.Linear(192,4))
    def logits(self,x):
        h,_=self.r(x)
        return self.tc(h).squeeze(-1),self.te(h).squeeze(-1),h
    def forward(self,x):
        collision,entry,h=self.logits(x)
        ci,ei=collision.argmax(1),entry.argmax(1); b=torch.arange(len(h),device=h.device)
        return ci,ei,self.scene(torch.cat([h[b,ci],h[b,ei]],1))
