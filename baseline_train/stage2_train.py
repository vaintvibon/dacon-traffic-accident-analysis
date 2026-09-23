from pathlib import Path
import os, random, shutil
import cv2, numpy as np, pandas as pd, torch
from PIL import Image
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.video import mvit_v2_s

from models.stage2_model import Stage2Temporal


ROOT=Path.cwd(); DATA=ROOT/'data'; MODEL=ROOT/'model'
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS=int(os.getenv('EPOCHS','1'))

SIZE=224
S1_MEAN=torch.tensor([0.45,0.45,0.45])[:,None,None,None]
S1_STD=torch.tensor([0.225,0.225,0.225])[:,None,None,None]
S3_MEAN=torch.tensor([0.45,0.45,0.45])[:,None,None]
S3_STD=torch.tensor([0.225,0.225,0.225])[:,None,None]
torch.manual_seed(20260825); random.seed(20260825)


def _video_frames(path):
    cap=cv2.VideoCapture(str(path)); out=[]
    while True:
        ok,bgr=cap.read()
        if not ok: break
        out.append(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
    cap.release()
    if not out: raise ValueError(f'cannot decode: {path}')
    return out


def _resnet_backbone():
    try: model=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    except Exception:
        print('경고: ImageNet 가중치를 받지 못해 weights=None으로 진행합니다.')
        model=resnet18(weights=None)
    return model

def fit_stage2():
    out=MODEL/'stage2'; out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(DATA/'stage2/labels.csv')
    backbone=_resnet_backbone(); torch.save(backbone.state_dict(),out/'resnet18-f37072fd.pth')
    backbone.fc=nn.Identity(); backbone.to(DEVICE).eval()
    transform=ResNet18_Weights.IMAGENET1K_V1.transforms()
    sequences=[]
    with torch.inference_mode():
        for r in df.itertuples():
            frames=_video_frames(DATA/'stage2'/r.path); batches=[]
            for start in range(0,len(frames),64):
                x=torch.stack([transform(Image.fromarray(a)) for a in frames[start:start+64]]).to(DEVICE)
                batches.append(backbone(x).float().cpu())
            sequences.append((torch.cat(batches),min(int(r.t_collision),len(frames)-1)))
    temporal=Stage2Temporal().to(DEVICE); opt=torch.optim.AdamW(temporal.parameters(),2e-4)
    for _ in range(max(1,EPOCHS)):
        temporal.train()
        for seq,target in sequences:
            collision,_,_=temporal.logits(seq[None].to(DEVICE))
            loss=nn.functional.cross_entropy(collision,torch.tensor([target],device=DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
    # 공개 CCD 5건에는 충돌 구간만 공식 주석이 있어 나머지 헤드는 구조 확인용이다.
    torch.save({'model':temporal.state_dict()},out/'best.pt')


print('device:',DEVICE)
fit_stage2(); print('Stage 2 완료')
