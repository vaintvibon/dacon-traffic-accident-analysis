"""Train the paper-inspired frame classifier; select by video Macro-F1.

Run: python -m train.stage1_train
Optional env: EPOCHS, S1_VAL_RATIO, S1_SEED, S1_LR, S1_FRAME_BATCH.
S1_MODE=diagnose_bn evaluates a checkpoint copy and writes only bn_diagnostics.json.
Training uses S1_VIDEO_BATCH=4 (or 2), interleaved across frame microbatches.
S1_PRECISION=fp32 (default) or bf16 on a supporting CUDA GPU. No FP16.
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


def precision_mode(device):
    mode = os.getenv('S1_PRECISION', 'fp32').lower()
    if mode not in ('fp32', 'bf16'):
        raise ValueError('S1_PRECISION must be fp32 or bf16; FP16 is disabled for stability.')
    if mode == 'bf16' and (device.type != 'cuda' or not torch.cuda.is_bf16_supported()):
        raise ValueError('This device does not support CUDA BF16; use S1_PRECISION=fp32.')
    return mode


def clip_training_gradients(model, sample_path):
    # Do not hide a genuine NaN/Inf or allow it to reach the optimizer.
    try:
        return torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    except RuntimeError as error:
        bad = [name for name, parameter in model.named_parameters()
               if parameter.grad is not None and not torch.isfinite(parameter.grad).all()]
        raise FloatingPointError(
            f'Nonfinite gradient/norm for sample {sample_path}; '
            f'parameters={bad[:8] or "finite elements but norm overflow"}. '
            'No optimizer update was applied. Use S1_PRECISION=fp32.'
        ) from error



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
def validate(model, df, device, frames=16, frame_batch=16, precision='fp32'):
    model.eval()
    labels, probabilities = [], []
    for number, row in enumerate(df.itertuples(), 1):
        if number == 1 or number % 50 == 0:
            print(f'Evaluating {number}/{len(df)} videos', flush=True)
        clip = load_row(row, model.config['size'], frames)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=precision == 'bf16'):
            probability = model.video_probability(clip[None].to(device), frame_batch)[0, 1]
        labels.append(LABELS[row.label])
        probabilities.append(float(probability))
    return classification_metrics(labels, probabilities)



def balanced_video_batches(df, videos, rng):
    """Alternate classes within each batch; oversample only the smaller class."""
    if videos < 2 or videos % 2:
        raise ValueError('S1_VIDEO_BATCH must be even and >= 2')
    pools = [rng.permutation(np.flatnonzero(df.label.to_numpy() == label))
             for label in ('ORIGINAL', 'RERECORDED')]
    if any(len(pool) == 0 for pool in pools):
        raise ValueError('Balanced batches require both classes')
    count = max(map(len, pools))
    pools = [np.resize(pool, count) for pool in pools]
    interleaved = np.stack(pools, axis=1).reshape(-1)
    for start in range(0, len(interleaved), videos):
        yield interleaved[start:start + videos]


def mixed_frame_batches(df, indices, size, frames, frame_batch, rng=None):
    """Every forward contains both classes and multiple videos, even when chunked."""
    clips = [load_row(df.iloc[int(i)], size, frames, rng) for i in indices]
    # [video,C,T,H,W] -> [T,video,C,H,W]: never split into single-video forwards.
    x = torch.stack(clips).permute(2, 0, 1, 3, 4).flatten(0, 1)
    labels = torch.tensor([LABELS[df.iloc[int(i)].label] for i in indices])
    y = labels.repeat(frames)
    if frame_batch % len(indices):
        raise ValueError('S1_FRAME_BATCH must be divisible by the actual video batch size')
    yield from zip(x.split(frame_batch), y.split(frame_batch))


@torch.inference_mode()
def recalibrate_bn(model, train, device, frames, frame_batch, videos):
    """Recompute each BN input distribution using TRAIN ONLY, in inference order.

    Layer-wise passes avoid mismatching downstream moments with upstream training
    batch statistics. All learned parameters and dropout/eval behavior stay fixed.
    """
    model.eval()
    layers = [(n, m) for n, m in model.named_modules()
              if isinstance(m, torch.nn.BatchNorm2d)]
    for name, layer in layers:
        count = 0
        total = torch.zeros(layer.num_features, dtype=torch.float64, device=device)
        squares = torch.zeros_like(total)

        def collect(module, inputs):
            nonlocal count
            x = inputs[0].float()
            variance, mean = torch.var_mean(x, dim=(0, 2, 3), unbiased=False)
            n = x.shape[0] * x.shape[2] * x.shape[3]
            count += n
            total.add_(mean.double() * n)
            squares.add_((variance.double() + mean.double().square()) * n)

        hook = layer.register_forward_pre_hook(collect)
        try:
            for step, indices in enumerate(balanced_video_batches(
                    train, videos, np.random.default_rng(0)), 1):
                for x, _ in mixed_frame_batches(train, indices, model.config['size'],
                                                frames, frame_batch):
                    model(x.to(device))
                if step % 25 == 0:
                    print(f'BN calibration {name}: batch {step}', flush=True)
        finally:
            hook.remove()
        if count < 2:
            raise ValueError('Insufficient samples to calibrate BatchNorm')
        mean = total / count
        variance = (squares / count - mean.square()).clamp_min(0) * count / (count - 1)
        if not torch.isfinite(variance).all() or not torch.isfinite(mean).all():
            raise FloatingPointError(f'Nonfinite BN statistics: {name}')
        layer.running_mean.copy_(mean)
        layer.running_var.copy_(variance)
        layer.num_batches_tracked.zero_()
        print(f'BN calibration completed: {name}', flush=True)


def diagnose_bn(df, held_out, device, frame_batch, videos):
    checkpoint_path = Path(os.getenv('S1_CHECKPOINT', str(MODEL / 'best.pt'))).resolve()
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if checkpoint.get('architecture') != 'cnn_vit_frame_v1':
        raise ValueError('Expected a CNN-ViT checkpoint')
    saved_split = checkpoint['split']
    indexed = df.set_index('path', drop=False)
    if not indexed.index.is_unique:
        raise ValueError('Duplicate data paths')
    # Reuse exactly the original split, rather than re-splitting by current settings.
    train = indexed.loc[saved_split['train']].reset_index(drop=True)
    val = indexed.loc[saved_split['validation']].reset_index(drop=True)
    if set(train.path) & set(val.path):
        raise ValueError('Overlapping train and validation paths')
    if 'source_id' in train:
        if set(train.source_id) & set(val.source_id):
            raise ValueError('Overlapping train and validation source IDs')
        if len(held_out) and (set(train.source_id) | set(val.source_id)) & set(held_out.source_id):
            raise ValueError('Official test source leakage')
    model = Stage1CNNViT(**checkpoint['config']).to(device)
    model.load_state_dict(checkpoint['model'])
    frames = int(checkpoint['frames'])
    print(f'BN diagnosis: epoch={checkpoint["epoch"]}; train={len(train)}; val={len(val)}')
    report = dict(checkpoint=str(checkpoint_path), epoch=checkpoint['epoch'],
                  calibration_split='train_only', evaluation_precision='fp32')
    print('Before calibration: evaluating TRAIN in eval mode', flush=True)
    report['train_before'] = validate(model, train, device, frames, frame_batch)
    print('Before calibration: evaluating VALIDATION', flush=True)
    report['validation_before'] = validate(model, val, device, frames, frame_batch)
    recalibrate_bn(model, train, device, frames, frame_batch, videos)
    print('After calibration: evaluating TRAIN in eval mode', flush=True)
    report['train_after'] = validate(model, train, device, frames, frame_batch)
    print('After calibration: evaluating VALIDATION', flush=True)
    report['validation_after'] = validate(model, val, device, frames, frame_batch)
    report_path = MODEL / 'bn_diagnostics.json'
    if report_path.resolve() == checkpoint_path:
        raise ValueError('Report path conflicts with checkpoint')
    MODEL.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    print(f'Report saved: {report_path}. No checkpoint weights were written.', flush=True)


def fit_stage1():
    global DATA, MODEL
    DATA = Path(os.getenv('S1_DATA_DIR', str(DATA))).expanduser().resolve()
    MODEL = Path(os.getenv('S1_MODEL_DIR', str(MODEL))).expanduser().resolve()
    resume = os.getenv('S1_RESUME', '0') == '1'
    seed = int(os.getenv('S1_SEED', '20260825'))
    epochs = int(os.getenv('EPOCHS', '30'))
    frame_batch = int(os.getenv('S1_FRAME_BATCH', '16'))
    videos = int(os.getenv('S1_VIDEO_BATCH', '4'))
    if videos not in (2, 4) or frame_batch % videos:
        raise ValueError('Use S1_VIDEO_BATCH=2 or 4 and S1_FRAME_BATCH divisible by it.')
    if epochs < 1 or frame_batch < 1:
        raise ValueError('EPOCHS and S1_FRAME_BATCH must be positive')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    precision = precision_mode(device)
    cv_threads = int(os.getenv('S1_TORCH_THREADS', '4'))
    torch.set_num_threads(max(1, cv_threads))
    if os.getenv('S1_DATASET', 'videos') == 'vdmoire':
        df, held_out = index_vdmoire(DATA)
    elif os.getenv('S1_DATASET', 'videos') == 'videos':
        df = pd.read_csv(DATA / 'labels.csv', dtype={'source_id': str, 'group_id': str})
        held_out = pd.DataFrame()
    else:
        raise ValueError('S1_DATASET must be videos or vdmoire')
    mode = os.getenv('S1_MODE', 'train')
    if mode == 'diagnose_bn':
        cv2.setNumThreads(1)
        diagnose_bn(df, held_out, device, frame_batch, videos)
        return
    if mode != 'train':
        raise ValueError('S1_MODE must be train or diagnose_bn')
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
    # FP32/BF16 do not need FP16 loss scaling. Keep the disabled scaler for the
    # existing checkpoint structure, without multiplying gradients by 65536.
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    frames, threshold = 16, .5
    rng = np.random.default_rng(seed)
    split = {'train': train['path'].tolist(), 'validation': val['path'].tolist(),
             'train_groups': sorted(train['_group'].unique().tolist()),
             'validation_groups': sorted(val['_group'].unique().tolist())}
    print(f'device={device}; precision={precision}; train={len(train)}; '
          f'validation={len(val)}; source groups split')
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
                    config=model.config, dataset_fingerprint=fingerprint, precision=precision,
                    batching='balanced_interleaved_v1', video_batch=videos)
    if resume:
        saved = torch.load(MODEL / 'last.pt', map_location='cpu', weights_only=False)
        previous_settings = dict(saved['settings'])
        previous_precision = previous_settings.pop('precision', 'legacy_fp16')
        comparable_settings = {k: v for k, v in settings.items() if k != 'precision'}
        if previous_settings != comparable_settings or saved['split'] != split:
            raise ValueError('Resume data split/config/batching differs. For the new balanced batches, '
                             'use a new S1_MODEL_DIR with S1_RESUME=0.')
        if previous_precision != precision:
            print(f'Resuming with precision change: {previous_precision} -> {precision}; '
                  'optimizer state is retained, FP16 scaler state is discarded.')
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
        train_loss, samples_seen = 0., 0
        batches = list(balanced_video_batches(train, videos, rng))
        for step, indices in enumerate(batches, 1):
            optimizer.zero_grad(set_to_none=True)
            batch_loss = 0.
            total_frames = len(indices) * frames
            for part, labels in mixed_frame_batches(train, indices, model.config['size'],
                                                     frames, frame_batch, rng):
                part, labels = part.to(device), labels.to(device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=precision == 'bf16'):
                    loss = focal_loss(model(part), labels) * (len(part) / total_frames)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite training loss for indices {indices}')
                scaler.scale(loss).backward()
                batch_loss += float(loss.detach())
            scaler.unscale_(optimizer)
            clip_training_gradients(model, train.iloc[indices].path.tolist())
            scaler.step(optimizer)
            scaler.update()
            train_loss += batch_loss * len(indices)
            samples_seen += len(indices)
            if step == 1 or step % 25 == 0 or step == len(batches):
                print(f'epoch={epoch}/{epochs} batch={step}/{len(batches)} '
                      f'train_focal={train_loss / samples_seen:.6f}', flush=True)
        metrics = validate(model, val, device, frames, frame_batch, precision)
        record = dict(epoch=epoch, train_focal_loss=train_loss / samples_seen, **metrics)
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
