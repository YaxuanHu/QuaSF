"""Precompute v2-format ESM2 residue features for the Seq30 test split."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer


def windows(length, size, overlap):
    if length <= size:
        return [(0, length)]
    stride = size - overlap
    starts = list(range(0, max(length - size + 1, 1), stride))
    final = length - size
    if starts[-1] != final:
        starts.append(final)
    return [(start, min(start + size, length)) for start in starts]


def encode_sequence(model, tokenizer, sequence, device, window_size, overlap):
    total = torch.zeros(len(sequence), model.config.hidden_size)
    count = torch.zeros(len(sequence), 1)
    with torch.inference_mode():
        for start, end in windows(len(sequence), window_size, overlap):
            tokens = tokenizer(
                sequence[start:end],
                return_tensors="pt",
                add_special_tokens=True,
            )
            tokens = {key: value.to(device) for key, value in tokens.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                hidden = model(**tokens).last_hidden_state[0, 1:-1]
            if hidden.shape[0] != end - start:
                raise ValueError(
                    f"token/residue mismatch {hidden.shape[0]} != {end - start}"
                )
            total[start:end] += hidden.float().cpu()
            count[start:end] += 1
    if not torch.all(count > 0):
        raise ValueError("windowing left uncovered residues")
    return total / count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--manifest", default="data/dataset_v2/anchor_mil_oracle_v1/test.jsonl"
    )
    parser.add_argument(
        "--output-dir", default="data/dataset_v2/esm_features_test_v2"
    )
    parser.add_argument(
        "--model", default="/geniusland/home/huyaxuan/code/SaProt/weights/esm2"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--window-size", type=int, default=1022)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    manifest = Path(args.manifest)
    if not manifest.is_absolute():
        manifest = root / manifest
    output = Path(args.output_dir)
    if not output.is_absolute():
        output = root / output
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    with manifest.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index % args.num_shards == args.shard_index:
                rows.append(json.loads(line))
    if args.limit is not None:
        rows = rows[: args.limit]

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModel.from_pretrained(
        args.model, local_files_only=True, torch_dtype=torch.bfloat16
    ).to(device).eval()

    completed = skipped = failed = 0
    failure_path = output / f"failures-shard{args.shard_index}.jsonl"
    for row in rows:
        acc = row["uniprot_acc"]
        destination = output / f"{acc}.pt"
        if destination.is_file():
            skipped += 1
            continue
        sequence = row["sequence"]
        try:
            residue = encode_sequence(
                model,
                tokenizer,
                sequence,
                device,
                args.window_size,
                args.overlap,
            )
            obj = {
                "format": "dataset-v2-esm-residue-v1",
                "uniprot_acc": acc,
                "residue_emb": residue,
                "residue_index": torch.arange(len(sequence), dtype=torch.int64),
                "sequence_length": len(sequence),
                "sequence_sha256": hashlib.sha256(
                    sequence.encode("ascii")
                ).hexdigest(),
                "esm_model": args.model,
                "window_size": args.window_size,
                "overlap": args.overlap,
            }
            temporary = destination.with_suffix(f".pt.tmp-{os.getpid()}")
            torch.save(obj, temporary)
            os.replace(temporary, destination)
            completed += 1
        except Exception as error:
            failed += 1
            with failure_path.open("a", encoding="utf-8") as writer:
                writer.write(
                    json.dumps(
                        {
                            "uniprot_acc": acc,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
                    + "\n"
                )
        processed = completed + skipped + failed
        if processed % args.log_every == 0:
            print(
                json.dumps(
                    {
                        "shard": args.shard_index,
                        "processed": processed,
                        "total": len(rows),
                        "completed": completed,
                        "skipped": skipped,
                        "failed": failed,
                    }
                ),
                flush=True,
            )
    print(
        json.dumps(
            {
                "shard": args.shard_index,
                "processed": len(rows),
                "completed": completed,
                "skipped": skipped,
                "failed": failed,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
