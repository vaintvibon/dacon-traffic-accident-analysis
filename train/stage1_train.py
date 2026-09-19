"""Train the paper-inspired frame classifier; select by video Macro-F1.

Run: python -m train.stage1_train
Optional env: EPOCHS, S1_VAL_RATIO, S1_SEED, S1_LR, S1_FRAME_BATCH.
labels.csv may supply source_id/group_id to group all derivatives of an
original. Otherwise matching filename stems are treated as one source,
as in the supplied original/000001 and rerecorded/000001 examples.
"""
from pathlib import Path
import json
import os
import random
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ''):
    sys.path.insert(0, str(ROOT))
from models.stage1_model import Stage1CNNViT, focal_loss, load_stage1_video

DATA = ROOT / 'data' / 'stage1'
MODEL = ROOT / 'model' / 'stage1'
LABELS = {'ORIGINAL': 0, 'RERECORDED': 1}


def split_by_source(df, val_ratio=.2, seed=20260825):
    if not 0 < val_ratio < 1:
        raise ValueError('S1_VAL_RATIO must be between 0 and 1')
    if not {'path', 'label'}.issubset(df.columns):
        raise ValueError('labels.csv requires path and label columns')
    df = df.copy()
    if df['path'].isna().any() or df['path'].duplicated().any():
        raise ValueError('video paths must be nonempty and unique')
    if not df['label'].isin(LABELS).all():
        raise ValueError('labels must be ORIGINAL or RERECORDED')
    group_column = next((c for c in ('source_id', 'group_id') if c in df), None)
    if group_column:
        if df[group_column].isna().any() or df[group_column].astype(str).str.strip().eq('').any():
            raise ValueError(f'{group_column} contains missing source identifiers')
        df['_group'] = df[group_column].astype(str)
    else:
        df['_group'] = df['path'].map(lambda p: Path(str(p).replace('\\', '/')).stem)
        print('Source groups: filename stems; use source_id for differently named derivatives.')
    # Stratify groups by their label membership, never splitting a source.
    memberships = df.groupby('_group')['label'].agg(lambda x: tuple(sorted(set(x))))
    rng = np.random.default_rng(seed)
    validation_groups = []
    for signature in sorted(set(memberships)):
        groups = np.array([g for g, s in memberships.items() if s == signature])
        rng.shuffle(groups)
        if len(groups) >= 2:
            count = min(len(groups) - 1, max(1, round(len(groups) * val_ratio)))
            validation_groups.extend(groups[:count].tolist())
    valid = df['_group'].isin(validation_groups)
    train, val = df.loc[~valid].copy(), df.loc[valid].copy()
    if set(train['label']) != set(LABELS) or set(val['label']) != set(LABELS):
        raise ValueError('Need enough independent sources for both classes in train and validation.')
    return train.reset_index(drop=True), val.reset_index(drop=True)


def classification_metrics(labels, probabilities, threshold=.5):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if not len(labels) or not np.isfinite(probabilities).all():
        raise ValueError('empty validation set or nonfinite predictions')
    predictions = (probabilities >= threshold).astype(int)
    matrix = np.zeros((2, 2), dtype=int)
    np.add.at(matrix, (labels, predictions), 1)
    tp = np.diag(matrix)
    denominator = matrix.sum(0) + matrix.sum(1)
    f1 = np.divide(2 * tp, denominator, out=np.zeros(2), where=denominator != 0)
    recall = np.divide(tp, matrix.sum(1), out=np.zeros(2), where=matrix.sum(1) != 0)
    p = np.clip(probabilities, 1e-7, 1 - 1e-7)
    loss = -(labels * np.log(p) + (1 - labels) * np.log(1 - p)).mean()
    return dict(macro_f1=float(f1.mean()), accuracy=float((labels == predictions).mean()),
                loss=float(loss), recall_original=float(recall[0]),
                recall_rerecorded=float(recall[1]), confusion_matrix=matrix.tolist())


@torch.inference_mode()
def validate(model, df, device, frames=16, frame_batch=16):
    model.eval()
    labels, probabilities = [], []
    for row in df.itertuples():
        clip = load_stage1_video(DATA / row.path, model.config['size'], frames)
        with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            probability = model.video_probability(clip[None].to(device), frame_batch)[0, 1]
        labels.append(LABELS[row.label])
        probabilities.append(float(probability))
    return classification_metrics(labels, probabilities)


def fit_stage1():
    seed = int(os.getenv('S1_SEED', '20260825'))
    epochs = int(os.getenv('EPOCHS', '30'))
    frame_batch = int(os.getenv('S1_FRAME_BATCH', '16'))
    if epochs < 1 or frame_batch < 1:
        raise ValueError('EPOCHS and S1_FRAME_BATCH must be positive')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cv_threads = int(os.getenv('S1_TORCH_THREADS', '4'))
    torch.set_num_threads(max(1, cv_threads))
    df = pd.read_csv(DATA / 'labels.csv', dtype={'source_id': str, 'group_id': str})
    train, val = split_by_source(df, float(os.getenv('S1_VAL_RATIO', '.2')), seed)
    for path in df['path']:
        if not (DATA / path).is_file():
            raise FileNotFoundError(DATA / path)
    MODEL.mkdir(parents=True, exist_ok=True)
    model = Stage1CNNViT().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(os.getenv('S1_LR', '0.00002')),
                                 weight_decay=.0001)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    frames, threshold = 16, .5
    rng = np.random.default_rng(seed)
    split = {'train': train['path'].tolist(), 'validation': val['path'].tolist(),
             'train_groups': sorted(train['_group'].unique().tolist()),
             'validation_groups': sorted(val['_group'].unique().tolist())}
    print(f'device={device}; train={len(train)}; validation={len(val)}; source groups split')
    best_f1, best_loss, best_epoch = -1., float('inf'), 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.
        for index in rng.permutation(len(train)):
            row = train.iloc[int(index)]
            clip = load_stage1_video(DATA / row.path, model.config['size'], frames, rng)
            x = clip.permute(1, 0, 2, 3).to(device)
            optimizer.zero_grad(set_to_none=True)
            video_loss = 0.
            # Accumulate the mean frame focal loss with one update per video.
            for part in x.split(frame_batch):
                y = torch.full((len(part),), LABELS[row.label], dtype=torch.long, device=device)
                with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                    loss = focal_loss(model(part), y) * (len(part) / frames)
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite training loss')
                scaler.scale(loss).backward()
                video_loss += float(loss.detach())
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            train_loss += video_loss
        metrics = validate(model, val, device, frames, frame_batch)
        record = dict(epoch=epoch, train_focal_loss=train_loss / len(train), **metrics)
        history.append(record)
        improved = (metrics['macro_f1'] > best_f1 or
                    (metrics['macro_f1'] == best_f1 and metrics['loss'] < best_loss))
        if improved:
            best_f1, best_loss, best_epoch = metrics['macro_f1'], metrics['loss'], epoch
            # Only selected epochs replace best.pt; includes inference configuration.
            torch.save(dict(architecture='cnn_vit_frame_v1', model=model.state_dict(),
                            config=model.config, frames=frames, size=model.config['size'],
                            sampling='temporal_bins_center', preprocessing='rgb_0_1_adaptive_pool',
                            aggregation='mean_frame_softmax', threshold=threshold,
                            epoch=epoch, metrics=metrics, seed=seed, split=split),
                       MODEL / 'best.pt')
        with (MODEL / 'training_history.json').open('w', encoding='utf-8') as handle:
            json.dump(dict(seed=seed, split=split, best_epoch=best_epoch, epochs=history),
                      handle, ensure_ascii=False, indent=2)
        print(f"epoch={epoch}/{epochs} train_focal={record['train_focal_loss']:.6f} "
              f"val_loss={metrics['loss']:.6f} macro_f1={metrics['macro_f1']:.4f} "
              f"accuracy={metrics['accuracy']:.4f} recall_O={metrics['recall_original']:.4f} "
              f"recall_R={metrics['recall_rerecorded']:.4f} "
              f"confusion={metrics['confusion_matrix']} best_epoch={best_epoch}")
    print(f'Stage 1 complete: best epoch={best_epoch}, Macro-F1={best_f1:.4f}')


if __name__ == '__main__':
    fit_stage1()
