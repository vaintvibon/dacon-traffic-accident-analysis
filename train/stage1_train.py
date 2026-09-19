"""Train the paper-inspired frame classifier; select by video Macro-F1.

Run: python -m train.stage1_train
Optional env: EPOCHS, S1_VAL_RATIO, S1_SEED, S1_LR, S1_FRAME_BATCH.
Colab: S1_DATASET=vdmoire, S1_DATA_DIR=<extracted root>, S1_MODEL_DIR=<Drive run>.
S1_RESUME=1 restores last.pt; EPOCHS is the total target, not extra epochs.
labels.csv may supply source_id/group_id to group all derivatives of an
original. Otherwise matching filename stems are treated as one source,
as in the supplied original/000001 and rerecorded/000001 examples.
"""
from pathlib import Path
import json
import re
import hashlib
import cv2
import os
import random
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ''):
    sys.path.insert(0, str(ROOT))
from models.stage1_model import Stage1CNNViT, focal_loss, load_stage1_video, sample_frame_ids

DATA = ROOT / 'data' / 'stage1'
MODEL = ROOT / 'model' / 'stage1'
LABELS = {'ORIGINAL': 0, 'RERECORDED': 1}



def index_vdmoire(root):
    """Index flat source/target frame folders, including nested tcl/tcl paths.

    A source ID deliberately excludes camera and class: the same content
    across cameras must never cross train/validation/test boundaries.
    """
    root = Path(root)
    records = []
    for folder in sorted(root.rglob('source')):
        if folder.parent.name not in ('train', 'test'):
            continue
        target = folder.parent / 'target'
        if not target.is_dir():
            raise ValueError(f'Missing target folder: {target}')
        paired = {}
        for side, directory in [('source', folder), ('target', target)]:
            groups = {}
            for image in sorted(directory.iterdir()):
                if image.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                    continue
                match = re.fullmatch(r'(.+)_(\d+)', image.stem)
                if not match:
                    raise ValueError(f'Unexpected frame name: {image}')
                video_id, number = match.group(1), int(match.group(2))
                if number in groups.setdefault(video_id, {}):
                    raise ValueError(f'Duplicate frame number: {image}')
                groups[video_id][number] = image.relative_to(root).as_posix()
            paired[side] = groups
        if not paired['source'] or paired['source'].keys() != paired['target'].keys():
            raise ValueError(f'Empty or unmatched source/target IDs: {folder.parent}')
        for video_id in sorted(paired['source']):
            source, clean = paired['source'][video_id], paired['target'][video_id]
            if source.keys() != clean.keys() or len(source) != 60:
                raise ValueError(f'Expected 60 paired frames for {folder.parent}/{video_id}')
            for side, group, label in [('source', source, 'RERECORDED'),
                                       ('target', clean, 'ORIGINAL')]:
                records.append(dict(path=f'{folder.parent.relative_to(root).as_posix()}/{side}/{video_id}',
                                    source_id=video_id, label=label,
                                    official_split=folder.parent.name,
                                    frame_paths=[group[n] for n in sorted(group)]))
    if not records:
        raise ValueError(f'No train/test source and target frame folders under {root}')
    df = pd.DataFrame(records)
    held_ids = set(df.loc[df.official_split == 'test', 'source_id'])
    # Exclude all versions of held-out content, even if another camera calls it train.
    train = df[(df.official_split == 'train') & ~df.source_id.isin(held_ids)].copy()
    test = df[df.official_split == 'test'].copy()
    print(f'VDmoire: {len(train)} train sequences; {len(test)} official test sequences held out.')
    return train.reset_index(drop=True), test.reset_index(drop=True)


def load_row(row, size, frames=16, rng=None):
    paths = getattr(row, 'frame_paths', None)
    if paths is None:
        return load_stage1_video(DATA / row.path, size, frames, rng)
    ids = sample_frame_ids(len(paths), frames, rng)
    images = []
    for index in ids:
        path = DATA / paths[int(index)]
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise ValueError(f'Cannot decode frame: {path}')
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div_(255)
        images.append(torch.nn.functional.adaptive_avg_pool2d(x, (size, size)))
    return torch.stack(images, dim=1)


def atomic_torch_save(value, path):
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


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
        clip = load_row(row, model.config['size'], frames)
        with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            probability = model.video_probability(clip[None].to(device), frame_batch)[0, 1]
        labels.append(LABELS[row.label])
        probabilities.append(float(probability))
    return classification_metrics(labels, probabilities)


def fit_stage1():
    global DATA, MODEL
    DATA = Path(os.getenv('S1_DATA_DIR', str(DATA))).expanduser().resolve()
    MODEL = Path(os.getenv('S1_MODEL_DIR', str(MODEL))).expanduser().resolve()
    resume = os.getenv('S1_RESUME', '0') == '1'
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
    if os.getenv('S1_DATASET', 'videos') == 'vdmoire':
        df, held_out = index_vdmoire(DATA)
    elif os.getenv('S1_DATASET', 'videos') == 'videos':
        df = pd.read_csv(DATA / 'labels.csv', dtype={'source_id': str, 'group_id': str})
        held_out = pd.DataFrame()
    else:
        raise ValueError('S1_DATASET must be videos or vdmoire')
    train, val = split_by_source(df, float(os.getenv('S1_VAL_RATIO', '.2')), seed)
    if 'frame_paths' not in df:
        for path in df['path']:
            if not (DATA / path).is_file():
                raise FileNotFoundError(DATA / path)
    MODEL.mkdir(parents=True, exist_ok=True)
    if not resume and any((MODEL / name).exists() for name in ('best.pt', 'last.pt')):
        raise ValueError('Output contains checkpoints: use S1_RESUME=1 or a new S1_MODEL_DIR.')
    cv2.setNumThreads(1)
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
    split['official_test'] = held_out['path'].tolist() if len(held_out) else []
    # Relative names and byte sizes detect changed indexing without depending on mount location.
    manifest = []
    for row in pd.concat([train, val]).itertuples():
        paths = getattr(row, 'frame_paths', [row.path])
        manifest.append([row.path, row.label, row._asdict().get('source_id', ''),
                         [(p, (DATA / p).stat().st_size) for p in paths]])
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    history, start_epoch = [], 1
    settings = dict(seed=seed, frames=frames, frame_batch=frame_batch,
                    config=model.config, dataset_fingerprint=fingerprint)
    if resume:
        saved = torch.load(MODEL / 'last.pt', map_location='cpu', weights_only=False)
        if saved['settings'] != settings or saved['split'] != split:
            raise ValueError('Resume data split/config differs from the saved run.')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler'])
        history = saved['history']
        best_f1, best_loss, best_epoch = saved['best_f1'], saved['best_loss'], saved['best_epoch']
        rng.bit_generator.state = saved['numpy_rng']
        torch.set_rng_state(saved['torch_rng'])
        if device.type == 'cuda' and saved['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        random.setstate(saved['python_rng'])
        start_epoch = saved['epoch'] + 1
        atomic_torch_save(saved['best_checkpoint'], MODEL / 'best.pt')
        best_checkpoint = saved['best_checkpoint']
        print(f'Resuming from epoch {start_epoch}; EPOCHS={epochs} is the total target.')
    else:
        best_checkpoint = None
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss = 0.
        for index in rng.permutation(len(train)):
            row = train.iloc[int(index)]
            clip = load_row(row, model.config['size'], frames, rng)
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
            best_checkpoint = dict(architecture='cnn_vit_frame_v1',
                            model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                            config=model.config, frames=frames, size=model.config['size'],
                            sampling='temporal_bins_center', preprocessing='rgb_0_1_adaptive_pool',
                            aggregation='mean_frame_softmax', threshold=threshold,
                            epoch=epoch, metrics=metrics, seed=seed, split=split)
            atomic_torch_save(best_checkpoint, MODEL / 'best.pt')
        atomic_torch_save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                               scaler=scaler.state_dict(), epoch=epoch, settings=settings,
                               split=split, history=history, best_f1=best_f1, best_loss=best_loss,
                               best_epoch=best_epoch, best_checkpoint=best_checkpoint,
                               numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                               cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None,
                               python_rng=random.getstate()), MODEL / 'last.pt')
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
