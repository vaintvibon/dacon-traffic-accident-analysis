"""Li et al. spatial CNN -> ViT, extended with stratified video sampling.

Fig. 3 defines convolutions; section 4.2 specifies patch=8, heads=16,
dropout=.1. Width, depth and output pooling size are implementation choices:
these are not specified in the paper. No temporal attention is introduced.
New training uses norm='group', an explicit departure from the paper's BN.
The constructor defaults to BN so old checkpoint configs still load exactly.
"""
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


HYBRID_SAMPLING = 'global_plus_three_clips_v1'


def sample_frame_ids(total, frames=16, rng=None, sampling='temporal_bins_center'):
    """One frame per time bin; use bin centers for validation/inference."""
    if total < 1 or frames < 1:
        raise ValueError('total and frames must be positive')
    offsets = np.full(frames, .5) if rng is None else rng.random(frames)
    global_ids = np.minimum(((np.arange(frames) + offsets) * total / frames).astype(int), total - 1)
    if sampling == 'temporal_bins_center':
        return global_ids
    if sampling != HYBRID_SAMPLING:
        raise ValueError(f'Unknown sampling: {sampling}')
    # One center in each temporal third: randomized during training, fixed at
    # 1/6, 1/2 and 5/6 during evaluation. Shift at edges to keep full clips.
    offsets = np.full(3, .5) if rng is None else rng.random(3)
    centers = np.minimum(((np.arange(3) + offsets) * total / 3).astype(int), total - 1)
    starts = np.clip(centers - frames // 2, 0, max(0, total - frames))
    clips = np.minimum(starts[:, None] + np.arange(frames), total - 1)
    return np.concatenate([global_ids, clips.reshape(-1)])


def frame_weights(frames, sampling='temporal_bins_center'):
    """Per-video weights shared by training and probability aggregation."""
    if frames < 1:
        raise ValueError('frames must be positive')
    if sampling == 'temporal_bins_center':
        return torch.full((frames,), 1. / frames)
    if sampling == HYBRID_SAMPLING:
        return torch.cat([torch.full((frames,), .5 / frames),
                          torch.full((3 * frames,), .5 / (3 * frames))])
    raise ValueError(f'Unknown sampling: {sampling}')


def load_stage1_video(path, size=224, frames=16, rng=None, sampling='temporal_bins_center'):
    """Full-frame adaptive pooling; repeated short-video indices reuse frames."""
    def read_selected(ids):
        cap = cv2.VideoCapture(str(path))
        selected, wanted, position = {}, set(ids.tolist()), 0
        try:
            while position <= int(ids.max()):
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
        ids = sample_frame_ids(total, frames, rng, sampling)
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
    ids = sample_frame_ids(total, frames, rng, sampling)
    selected = read_selected(ids)
    if len(selected) != len(set(ids.tolist())):
        raise ValueError(f'incomplete video decode: {path}')
    return torch.stack([selected[int(i)] for i in ids], dim=1)


class Stage1CNNViT(nn.Module):
    def __init__(self, size=224, feature_size=56, patch_size=8,
                 embed_dim=256, depth=4, heads=16, dropout=.1, norm='batch'):
        super().__init__()
        if size < 32 or feature_size < patch_size or feature_size % patch_size:
            raise ValueError('invalid input/feature/patch size')
        if embed_dim % heads or depth < 1:
            raise ValueError('embed_dim must be divisible by heads; depth must be positive')
        if norm not in ('batch', 'group'):
            raise ValueError("norm must be 'batch' or 'group'")
        self.config = dict(size=size, feature_size=feature_size, patch_size=patch_size,
                           embed_dim=embed_dim, depth=depth, heads=heads, dropout=dropout,
                           norm=norm)
        def normalization(channels):
            # 3/4/8 groups for 9/16/32 channels; no running batch statistics.
            return (nn.BatchNorm2d(channels) if norm == 'batch' else
                    nn.GroupNorm(3 if channels == 9 else channels // 4, channels))
        # Follow Fig. 3; section 3.3 has an inconsistent channel description.
        self.local = nn.Sequential(
            nn.AdaptiveAvgPool2d((size, size)),
            nn.Conv2d(3, 3, 3, padding=1),
            nn.Conv2d(3, 6, 3, padding=1),
            nn.Conv2d(6, 9, 3, padding=1), normalization(9), nn.PReLU(),
            nn.Conv2d(9, 16, 5, stride=2, padding=1), normalization(16), nn.PReLU(),
            nn.Conv2d(16, 32, 5, stride=2, padding=1), normalization(32),
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

    def video_probability(self, clips, frame_batch_size=16, sampling='temporal_bins_center'):
        """Frame softmax: uniform legacy mean, or 50% global + 50% clips."""
        if clips.ndim != 5 or frame_batch_size < 1:
            raise ValueError('expected [B,3,T,H,W] and positive frame_batch_size')
        b, c, t, h, w = clips.shape
        if sampling == HYBRID_SAMPLING and t % 4:
            raise ValueError('Hybrid input requires four equally sized frame groups')
        weights = frame_weights(t // 4 if sampling == HYBRID_SAMPLING else t, sampling)
        x = clips.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
        probs = [self(part).float().softmax(-1) for part in x.split(frame_batch_size)]
        probabilities = torch.cat(probs).reshape(b, t, 2)
        return (probabilities * weights.to(probabilities.device)[None, :, None]).sum(1)


def focal_loss(logits, labels, gamma=2., alpha=.5, reduction='mean'):
    ce = F.cross_entropy(logits.float(), labels, reduction='none')
    losses = alpha * (1 - torch.exp(-ce)).pow(gamma) * ce
    if reduction == 'none':
        return losses
    if reduction != 'mean':
        raise ValueError('Unsupported focal loss reduction')
    return losses.mean()
