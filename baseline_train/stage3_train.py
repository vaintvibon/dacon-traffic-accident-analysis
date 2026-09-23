from pathlib import Path
import os, random, shutil
import cv2, numpy as np, pandas as pd, torch
from PIL import Image
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.video import mvit_v2_s

from models.stage3_model import Stage3MViT


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


def fit_stage3():
    out=MODEL/'stage3'; out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(DATA/'stage3/labels.csv')
    amap={'ACCELERATING':0,'DECELERATING':1,'CONSTANT':2,'STOPPED':3}
    smap={'LEFT':0,'STRAIGHT':1,'RIGHT':2}
    model=Stage3MViT().to(DEVICE); opt=torch.optim.AdamW(model.parameters(),1e-4)
    for _ in range(EPOCHS):
        model.train()
        for r in df.itertuples():
            x,_=_clip(DATA/'stage3/videos'/f'{r.ID}.mp4',16,int(r.frame_index))
            x=(x-S3_MEAN[:,None,:,:])/S3_STD[:,None,:,:]
            a,s=model(x[None].to(DEVICE))
            loss=nn.functional.cross_entropy(a,torch.tensor([amap[r.accel_label]],device=DEVICE))
            loss+=nn.functional.cross_entropy(s,torch.tensor([smap[r.steer_label]],device=DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
    torch.save({'model':model.state_dict()},out/'best.pt')


print('device:',DEVICE)
fit_stage3(); print('Stage 3 완료')
