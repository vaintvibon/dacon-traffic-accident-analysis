from pathlib import Path
import os, random
import cv2, numpy as np, pandas as pd, torch
from torch import nn
from torchvision.models.video import mvit_v2_s

from models.stage1_model import Stage1MViT


ROOT=Path.cwd(); DATA=ROOT/'data'; MODEL=ROOT/'model'
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS=int(os.getenv('EPOCHS','1'))

SIZE=224
S1_MEAN=torch.tensor([0.45,0.45,0.45])[:,None,None,None]
S1_STD=torch.tensor([0.225,0.225,0.225])[:,None,None,None]
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

def _crop_tensor(rgb,size=224):
    h,w=rgb.shape[:2]; scale=size/min(h,w)
    nh,nw=max(size,round(h*scale)),max(size,round(w*scale))
    rgb=cv2.resize(rgb,(nw,nh),interpolation=cv2.INTER_AREA)
    y,x=(nh-size)//2,(nw-size)//2
    return torch.from_numpy(rgb[y:y+size,x:x+size].copy()).permute(2,0,1).float()/255

def _clip(path,n=16,center=None):
    frames=_video_frames(path); total=len(frames)
    if center is None: idx=np.linspace(0,total-1,n).round().astype(int)
    else: idx=np.clip(center-n//2+np.arange(n),0,total-1)
    x=torch.stack([_crop_tensor(frames[int(i)]) for i in idx],1)
    return x,total


def fit_stage1():
    out=MODEL/'stage1'; out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(DATA/'stage1/labels.csv')
    model=Stage1MViT().to(DEVICE); opt=torch.optim.AdamW(model.parameters(),1e-4)
    for _ in range(EPOCHS):
        model.train()
        for r in df.sample(frac=1,random_state=20260825).itertuples():
            x,_=_clip(DATA/'stage1'/r.path,16); x=(x-S1_MEAN)/S1_STD
            y=torch.tensor([0 if r.label=='ORIGINAL' else 1],device=DEVICE)
            loss=nn.functional.cross_entropy(model(x[None].to(DEVICE)),y)
            opt.zero_grad(); loss.backward(); opt.step()
    # 실제 inference.py는 래퍼가 아닌 mvit_v2_s 본체에 직접 로드한다.
    torch.save({'model':model.net.state_dict(),'size':224,'frames':16},out/'best.pt')


print('device:',DEVICE)
fit_stage1(); print('Stage 1 완료')
