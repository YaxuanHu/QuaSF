"""Merge official Chainsaw and Merizo chunk results into all_baseline_results.csv."""
import os
import sys
import argparse
import numpy as np
import pandas as pd


def merge_merizo_chunks(per_protein_paths):
    dfs = []
    for p in per_protein_paths:
        if os.path.exists(p):
            dfs.append(pd.read_csv(p))
    if not dfs:
        return None
    df = pd.concat(dfs, ignore_index=True)
    if df.empty:
        return None

    # Weighted by domain-residue count (same as evaluate_dataset)
    weights = df['weight'].astype(float).values
    weights = weights / (weights.sum() + 1e-8)
    avg_iou = float(np.average(df['iou'].values, weights=weights))
    avg_mcc = float(np.average(df['mcc'].values, weights=weights))
    simple_avg_iou = float(df['iou'].mean())
    simple_avg_mcc = float(df['mcc'].mean())
    return {
        'num_proteins': len(df),
        'avg_iou': avg_iou,
        'avg_mcc': avg_mcc,
        'simple_avg_iou': simple_avg_iou,
        'simple_avg_mcc': simple_avg_mcc,
    }


def read_chainsaw_summary(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    row = df.iloc[0]
    return {
        'num_proteins': int(row['num_proteins']),
        'avg_iou': float(row['avg_iou']),
        'avg_mcc': float(row['avg_mcc']),
        'simple_avg_iou': float(row['simple_avg_iou']),
        'simple_avg_mcc': float(row['simple_avg_mcc']),
    }


def update_all_results(all_results_path, rows):
    if os.path.exists(all_results_path):
        df = pd.read_csv(all_results_path)
    else:
        df = pd.DataFrame(columns=[
            'method', 'status', 'num_proteins', 'avg_iou', 'avg_mcc',
            'simple_avg_iou', 'simple_avg_mcc', 'note'
        ])

    for row in rows:
        method = row['method']
        mask = df['method'] == method
        if mask.any():
            df.loc[mask, ['status', 'num_proteins', 'avg_iou', 'avg_mcc',
                          'simple_avg_iou', 'simple_avg_mcc', 'note']] = [
                row['status'], row['num_proteins'], row['avg_iou'], row['avg_mcc'],
                row['simple_avg_iou'], row['simple_avg_mcc'], row['note']
            ]
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    df.to_csv(all_results_path, index=False)
    print(f"Updated {all_results_path}")
    print(df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chainsaw', type=str,
                        default='code/baselines/results/chainsaw_official_results.csv')
    parser.add_argument('--merizo_chunks', type=str, nargs='+', default=[
        'code/baselines/results/merizo_official_gpu1_results_per_protein.csv',
        'code/baselines/results/merizo_official_gpu2_results_per_protein.csv',
        'code/baselines/results/merizo_official_gpu3_results_per_protein.csv',
    ])
    parser.add_argument('--all_results', type=str,
                        default='code/baselines/results/all_baseline_results.csv')
    args = parser.parse_args()

    rows = []

    chainsaw = read_chainsaw_summary(args.chainsaw)
    if chainsaw:
        rows.append({
            'method': 'chainsaw_official',
            'status': 'official_pretrained',
            'num_proteins': chainsaw['num_proteins'],
            'avg_iou': round(chainsaw['avg_iou'], 4),
            'avg_mcc': round(chainsaw['avg_mcc'], 4),
            'simple_avg_iou': round(chainsaw['simple_avg_iou'], 4),
            'simple_avg_mcc': round(chainsaw['simple_avg_mcc'], 4),
            'note': 'official pretrained weights, zero-shot',
        })
    else:
        print(f"Warning: Chainsaw results not found at {args.chainsaw}")

    merizo = merge_merizo_chunks(args.merizo_chunks)
    if merizo:
        rows.append({
            'method': 'merizo_official',
            'status': 'official_pretrained',
            'num_proteins': merizo['num_proteins'],
            'avg_iou': round(merizo['avg_iou'], 4),
            'avg_mcc': round(merizo['avg_mcc'], 4),
            'simple_avg_iou': round(merizo['simple_avg_iou'], 4),
            'simple_avg_mcc': round(merizo['simple_avg_mcc'], 4),
            'note': 'official pretrained weights, zero-shot, no-iterate, 24GB GPU',
        })
    else:
        print(f"Warning: Merizo chunk results not found")

    if rows:
        update_all_results(args.all_results, rows)


if __name__ == '__main__':
    main()
