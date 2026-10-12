"""Run the four paper structure examples without training or reference labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from quasf.data import EndToEndDomainFunctionDataset
from quasf.paper_train import restore
from quasf.pretrain import atomic_json, sha256


def segments(mask: list[bool]) -> list[list[int]]:
    """Represent every contiguous predicted span in 1-based inclusive coordinates."""
    spans = []
    start = None
    for index, active in enumerate(mask + [False], 1):
        if active and start is None:
            start = index
        elif not active and start is not None:
            spans.append([start, index - 1])
            start = None
    return spans


def main() -> None:
    """Load released inputs and weights explicitly and publish fresh CPU predictions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--examples', type=Path, default=Path('demo-assets/paper-examples'))
    parser.add_argument('--weights', type=Path, default=Path('weights'))
    parser.add_argument('--output', type=Path, default=Path('demo-output'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    config = json.loads((root / 'configs/paper_selected.json').read_text())
    vocabulary = json.loads((root / 'configs/observed_label_vocabulary.json').read_text())['index_to_cluster']
    manifest = json.loads((args.examples / 'manifest.json').read_text())
    ids = ['Q7SDR1', 'A1L0T0', 'A9KIG5', 'Q7SIG6']
    if manifest['accessions'] != ids:
        raise ValueError('Expected the four paper structure examples in their recorded order')
    for name, digest in manifest['sha256'].items():
        if sha256(args.examples / name) != digest:
            raise ValueError(f'Example checksum mismatch: {name}')
    weights = {
        'quasf-selected.ckpt': config['checkpoint_sha256'],
        'segmentation-source.ckpt': config['mask_checkpoint_sha256'],
        'direct-query.pt': config['query_checkpoint_sha256'],
    }
    for name, digest in weights.items():
        path = args.weights / name
        if not path.is_file():
            parser.error(f'Missing {path}; see docs/FOUR_PROTEIN_DEMO.md')
        if sha256(path) != digest:
            raise ValueError(f'Checkpoint checksum mismatch: {name}')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(config['seed'])
    started = time.monotonic()
    module = restore(root, args.weights / 'direct-query.pt', args.weights / 'segmentation-source.ckpt', 'direct')
    checkpoint = torch.load(args.weights / 'quasf-selected.ckpt', map_location='cpu', weights_only=False)
    module.load_state_dict(checkpoint['state_dict'], strict=True)
    del checkpoint
    module.eval()
    dataset = EndToEndDomainFunctionDataset(str(args.examples / 'records.jsonl'),
                                           max_len=config['max_len'], data_root=str(args.examples))
    if [row['uniprot_acc'] for row in dataset.samples] != ids:
        raise ValueError('Example records do not match the manifest')
    predictions = []
    rows = ['accession\tlength\tregion\tsegments_1based_inclusive\tclass_index\tcluster_id\tprobability\tabove_threshold']
    with torch.inference_mode():
        for index, record in enumerate(dataset.samples):
            if record['domains']:
                raise ValueError('Demo inputs must not contain reference labels')
            sample = dataset[index]
            length = len(record['sequence'])
            if not 0 < length <= config['max_len'] or sample['seq_len'] != length:
                raise ValueError(f'Sequence/feature length mismatch: {record["uniprot_acc"]}')
            features = sample['residue_feat'].float()
            if features.shape != (config['max_len'], 1280) or not torch.isfinite(features).all():
                raise ValueError('Expected finite 1280-dimensional ESM residue features')
            result = module.model(features.unsqueeze(0), sample['attn_mask'].unsqueeze(0), training=False)
            protein = {'accession': record['uniprot_acc'], 'length': length,
                       'sequence_sha256': hashlib.sha256(record['sequence'].encode()).hexdigest(), 'regions': []}
            for position in range(result['region_valid'].shape[1]):
                if not bool(result['region_valid'][0, position]):
                    continue
                probabilities = result['region_logits'][0, position].sigmoid()
                if probabilities.shape != (256,) or not torch.isfinite(probabilities).all():
                    raise ValueError('Invalid regional function probabilities')
                spans = segments(result['region_masks'][0, position, :length].bool().tolist())
                selected = (probabilities >= config['region_function_threshold']).nonzero().flatten().tolist()
                top = int(probabilities.argmax())
                region = {'region': len(protein['regions']) + 1, 'segments': spans,
                          'source_query_id': int(result['region_source_query_ids'][0, position]),
                          'predicted_class_indices': selected, 'top_class_index': top,
                          'function_probabilities': probabilities.tolist()}
                protein['regions'].append(region)
                for label in selected or [top]:
                    rows.append('\t'.join(map(str, [protein['accession'], length, region['region'],
                        ';'.join(f'{a}-{b}' for a, b in spans), label, vocabulary.get(str(label), 'unknown'),
                        f'{float(probabilities[label]):.6f}', label in selected])))
            if not protein['regions']:
                raise ValueError(f'No regions predicted for {protein["accession"]}')
            predictions.append(protein)
            print(f'{protein["accession"]}: {length} residues, {len(protein["regions"])} regions', flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    temporary = args.output / 'regions.tsv.tmp'
    temporary.write_text('\n'.join(rows) + '\n', encoding='utf-8')
    temporary.replace(args.output / 'regions.tsv')
    atomic_json(args.output / 'predictions.json', {
        'device': 'cpu', 'precision': 'float32', 'torch_version': torch.__version__,
        'checkpoint_sha256': weights, 'coordinates': '1-based inclusive',
        'function_threshold': config['region_function_threshold'],
        'vocabulary_note': 'Observed cluster IDs only; full GO mapping is not available.',
        'elapsed_seconds': time.monotonic() - started, 'proteins': predictions})
    print(f'SUCCESS: fresh predictions for all four proteins saved in {args.output}', flush=True)


if __name__ == '__main__':
    main()
