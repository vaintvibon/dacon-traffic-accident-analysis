"""Li et al. spatial CNN -> ViT, extended with stratified video sampling.

Fig. 3 defines convolutions; section 4.2 specifies patch=8, heads=16,
dropout=.1. Width, depth and output pooling size are implementation choices:
these are not specified in the paper. No temporal attention is introduced.
"""
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def sample_frame_ids(total, frames=16, rng=None):
    """One frame per time bin; use bin centers for validation/inference."""
    if total < 1 or frames < 1:
        raise ValueError('total and frames must be positive')
    offsets = np.full(frames, .5) if rng is None else rng.random(frames)
    return np.minimum(((np.arange(frames) + offsets) * total / frames).astype(int), total - 1)


def load_stage1_video(path, size=224, frames=16, rng=None):
    """Full-frame adaptive pooling; repeated short-video indices reuse frames."""
    def read_selected(ids):
        cap = cv2.VideoCapture(str(path))
        selected, wanted, position = {}, set(ids.tolist()), 0
        try:
            while position <= int(ids[-1]):
                ok, bgr = cap.read()
                if not ok:
                    break
                if position in wanted:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    x = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div_(255)
                    selected[position] = F.adaptive_avg_pool2d(x, (size, size))
                position += 1
        finally:
            cap.release()
        return selected

    cap = cv2.VideoCapture(str(path))
    raw_total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    total = int(raw_total) if np.isfinite(raw_total) and raw_total > 0 else 0
    if total:
        ids = sample_frame_ids(total, frames, rng)
        selected = read_selected(ids)
        if len(selected) == len(set(ids.tolist())):
            return torch.stack([selected[int(i)] for i in ids], dim=1)
    # Retry with the actual decoded length when metadata is missing/too large.
    cap = cv2.VideoCapture(str(path))
    total = 0
    try:
        while cap.grab():
            total += 1
    finally:
        cap.release()
    if not total:
        raise ValueError(f'cannot decode video: {path}')
    ids = sample_frame_ids(total, frames, rng)
    selected = read_selected(ids)
    if len(selected) != len(set(ids.tolist())):
        raise ValueError(f'incomplete video decode: {path}')
    return torch.stack([selected[int(i)] for i in ids], dim=1)


class Stage1CNNViT(nn.Module):
    def __init__(self, size=224, feature_size=56, patch_size=8,
                 embed_dim=256, depth=4, heads=16, dropout=.1):
        super().__init__()
        if size < 32 or feature_size < patch_size or feature_size % patch_size:
            raise ValueError('invalid input/feature/patch size')
        if embed_dim % heads or depth < 1:
            raise ValueError('embed_dim must be divisible by heads; depth must be positive')
        self.config = dict(size=size, feature_size=feature_size, patch_size=patch_size,
                           embed_dim=embed_dim, depth=depth, heads=heads, dropout=dropout)
        # Follow Fig. 3; section 3.3 has an inconsistent channel description.
        self.local = nn.Sequential(
            nn.AdaptiveAvgPool2d((size, size)),
            nn.Conv2d(3, 3, 3, padding=1),
            nn.Conv2d(3, 6, 3, padding=1),
            nn.Conv2d(6, 9, 3, padding=1), nn.BatchNorm2d(9), nn.PReLU(),
            nn.Conv2d(9, 16, 5, stride=2, padding=1), nn.BatchNorm2d(16), nn.PReLU(),
            nn.Conv2d(16, 32, 5, stride=2, padding=1), nn.BatchNorm2d(32),
            nn.AdaptiveAvgPool2d((feature_size, feature_size)))
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.projection = nn.Linear(32 * patch_size ** 2, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position = nn.Parameter(torch.zeros(1, (feature_size // patch_size) ** 2 + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, heads, embed_dim * 4, dropout,
                                       activation='gelu', batch_first=True, norm_first=True),
            depth, norm=nn.LayerNorm(embed_dim), enable_nested_tensor=False)
        self.head = nn.Linear(embed_dim, 2)
        self.apply(self._initialize)
        nn.init.trunc_normal_(self.cls_token, std=.01)
        nn.init.trunc_normal_(self.position, std=.01)
        for layer in self.encoder.layers:
            nn.init.trunc_normal_(layer.self_attn.in_proj_weight, std=.01)
            nn.init.zeros_(layer.self_attn.in_proj_bias)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=.01)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        """RGB [N,3,H,W] in [0,1] -> frame logits [N,2]."""
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError('expected frame batch [N,3,H,W]')
        tokens = self.projection(self.unfold(self.local(x)).transpose(1, 2))
        tokens = torch.cat([self.cls_token.expand(len(x), -1, -1), tokens], dim=1)
        return self.head(self.encoder(self.dropout(tokens + self.position))[:, 0])

    def video_probability(self, clips, frame_batch_size=16):
        """[B,3,T,H,W] -> [B,2], averaging frame softmax probabilities."""
        if clips.ndim != 5 or frame_batch_size < 1:
            raise ValueError('expected [B,3,T,H,W] and positive frame_batch_size')
        b, c, t, h, w = clips.shape
        x = clips.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
        probs = [self(part).float().softmax(-1) for part in x.split(frame_batch_size)]
        return torch.cat(probs).reshape(b, t, 2).mean(1)


def focal_loss(logits, labels, gamma=2., alpha=.5):
    ce = F.cross_entropy(logits.float(), labels, reduction='none')
    return (alpha * (1 - torch.exp(-ce)).pow(gamma) * ce).mean()
