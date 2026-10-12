"""Exploratory fixed-region comparison using immutable saved QuaSF predictions.

Reuse the published dataset loader and cached one-to-one matches. Run an existing
reference-trained MLP on the same raw-ESM mean/max features. This is a head-swap
diagnostic, not a training-matched architectural ablation. No assets are modified.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn


def digest(path: Path) -> str:
    """Record the identity of each input artifact."""
    with path.open('rb') as stream:
        value = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1048576), b''):
            value.update(chunk)
        return value.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    """Publish complete results without partial JSON files."""
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def macro_curve(probs: np.ndarray, labels: np.ndarray) -> tuple:
    """Match the paper's mean-precision/mean-recall harmonic F1."""
    scores = []
    for threshold in np.linspace(0, 1, 101):
        calls = probs >= threshold
        tp = (calls & labels).sum(1)
        p = np.divide(tp, calls.sum(1), out=np.zeros(len(tp)), where=calls.sum(1) > 0).mean()
        r = (tp / labels.sum(1)).mean()
        scores.append(2*p*r/(p+r) if p+r else 0.)
    best = int(np.argmax(scores))
    return scores, best


def preference(probs: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> dict:
    """Use identical region pairs and count ties as incorrect for both heads."""
    correct = comparisons = eligible = ties = 0
    _, starts = np.unique(groups, return_index=True)
    boundaries = sorted(starts.tolist()) + [len(groups)]
    for begin, end in zip(boundaries[:-1], boundaries[1:]):
        margins = []
        for left in range(begin, end):
            for right in range(left+1, end):
                diff = probs[left] - probs[right]
                margins.extend(diff[labels[left] & ~labels[right]].tolist())
                margins.extend((-diff[labels[right] & ~labels[left]]).tolist())
        if margins:
            values = np.asarray(margins)
            eligible += 1
            comparisons += len(values)
            correct += int((values > 0).sum())
            ties += int((values == 0).sum())
    return dict(correct=correct, comparisons=comparisons, eligible_proteins=eligible,
                ties=ties, accuracy=correct/comparisons if comparisons else None)


def main() -> None:
    """Cache matched features, calibrate on validation, and evaluate fixed heads."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, str(args.root/'release/QuaSF_20260930'))
    from quasf.data import EndToEndDomainFunctionDataset
    base = args.root/'output/query_no_mlp_joint_20260928_b32_batched/unified_evaluation/direct'
    checkpoint = args.root/'output/region_mlp_baseline/region_mlp.pt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = nn.Sequential(nn.LayerNorm(2560), nn.Linear(2560,512), nn.GELU(),
                          nn.Dropout(.2), nn.Linear(512,256)).eval()
    model.load_state_dict({k.removeprefix('net.'): v for k,v in saved['state_dict'].items()}, strict=True)
    args.output.mkdir(parents=True, exist_ok=True)
    common_path = args.root/'input/joint_eval_common9015_cache_test/index.json'
    common = {x['accession'] for x in json.loads(common_path.read_text())}
    assert len(common) == 9015
    summary = {'kind': 'exploratory existing MLP; different training losses and schedule',
               'mlp_sha256': digest(checkpoint), 'mlp_training_metadata': {k:v for k,v in saved.items() if k!='state_dict'},
               'cohort_sha256': digest(common_path), 'splits': {}}
    chosen = {}
    for split in ('val','test'):
        cache_file = args.output/(split+'.npz')
        source_dir = base/('val/epoch=0-step=564' if split=='val' else 'test')
        split_file = args.root/f'data/dataset_v2/end_to_end_domain_function_v1/{split}.jsonl'
        dataset = EndToEndDomainFunctionDataset(str(split_file), max_len=1495, data_root=str(args.root))
        lookup = {r['uniprot_acc']: i for i,r in enumerate(dataset.samples)}
        predictions = {}
        for file in sorted(source_dir.glob('predictions_rank*.jsonl')):
            for line in file.open():
                row = json.loads(line)
                assert row['accession'] not in predictions
                predictions[row['accession']] = row
        assert set(predictions)==set(lookup)
        order = sorted(predictions)
        if not cache_file.exists():
            def extract(accession: str) -> tuple:
                sample = dataset[lookup[accession]]
                row = predictions[accession]
                assert len(sample['domain_infos']) == row['gt_count']
                features, targets, scores, keys = [], [], [], []
                for pred_index, gt_index, expected_iou in row['matches']['0.5']:
                    gt = sample['domain_infos'][gt_index]
                    pred = row['predictions'][pred_index]
                    mask = torch.zeros(1495, dtype=torch.bool)
                    for start, end in pred['segments']:
                        assert 0 <= start < end <= sample['seq_len']
                        mask[start:end] = True
                    reference = gt['mask'].bool()
                    iou = float((mask & reference).sum()) / int((mask | reference).sum())
                    assert abs(iou-expected_iou) < 1e-7
                    if not gt['valid']:
                        continue
                    feature = sample['residue_feat'][mask].float()
                    features.append(torch.cat((feature.mean(0), feature.amax(0))).numpy())
                    targets.append(gt['label'].numpy().astype(bool))
                    scores.append(pred['function_probs'])
                    keys.append(f'{accession}:{pred_index}:{gt_index}')
                return accession, features, targets, scores, keys
            xs, ys, qs, groups, keys = [], [], [], [], []
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                for i, (acc,x,y,q,k) in enumerate(executor.map(extract, order)):
                    xs.extend(x); ys.extend(y); qs.extend(q); keys.extend(k)
                    groups.extend([acc]*len(y))
                    if (i+1)%500==0:
                        print(json.dumps({'split':split,'proteins':i+1,'matched_regions':len(ys)}), flush=True)
            x = torch.from_numpy(np.asarray(xs, dtype=np.float32))
            with torch.inference_mode():
                mlp = torch.cat([model(batch).sigmoid() for batch in x.split(1024)]).numpy()
            np.savez_compressed(cache_file, features=x.numpy(), labels=np.asarray(ys),
                                quasf=np.asarray(qs), mlp=mlp, groups=np.asarray(groups), keys=np.asarray(keys))
        with np.load(cache_file) as z:
            labels, groups, qp, mp = z['labels'], z['groups'], z['quasf'], z['mlp']
        original = json.loads((source_dir/'summary.json').read_text())
        curve, _ = macro_curve(qp, labels)
        assert len(labels)==original['matched_function']['n']
        assert np.max(np.abs(np.asarray(curve)-original['matched_function']['macro_f1_by_threshold'])) < 1e-10
        pref = preference(qp, labels, groups)
        assert pref['correct']==original['regional_preference']['correct']
        assert pref['comparisons']==original['regional_preference']['comparisons']
        keep = np.ones(len(groups),bool) if split=='val' else np.isin(groups, list(common))
        labels, groups, qp, mp = labels[keep], groups[keep], qp[keep], mp[keep]
        rows = {}
        for name, probs in [('QuaSF',qp),('RegionMLP',mp)]:
            curve, best = macro_curve(probs,labels)
            if split=='val': chosen[name] = best
            rows[name] = {'threshold':chosen[name]/100, 'matched_f1':curve[chosen[name]],
                          'preference':preference(probs,labels,groups)}
        summary['splits'][split] = {'proteins':len(order) if split=='val' else len(common),
            'matched_regions':len(labels),'models':rows,'quasf_full_cache_reproduction':True,
            'source_summary_sha256':digest(source_dir/'summary.json'),'split_sha256':digest(split_file)}
        atomic_json(args.output/'summary.json',summary)
        print(json.dumps(summary['splits'][split]),flush=True)


if __name__ == '__main__':
    main()
