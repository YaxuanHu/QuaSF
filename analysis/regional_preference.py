"""Score within-protein functional preference from the shared-query cache."""
import argparse
import json
from pathlib import Path
import numpy as np
from quasf.decoders import decode
from quasf.metrics import segmentation

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    correct = comparisons = eligible = all_correct = 0
    index = json.loads((args.cache / "index.json").read_text())
    for entry in index:
        with np.load(args.cache / entry["file"], allow_pickle=False) as sample:
            predictions = decode(sample["mask_logits"], sample["func_logits"], sample["objectness_logits"], {"mode": "legacy_nms", "mask_threshold": .3})
            labels = sample["gt_labels"]; valid = sample["gt_valid"]
            _, pairs = segmentation(predictions, sample["gt_masks"].astype(bool), int(sample["length"]), (.5,))
            observed = [(p, g) for p, g, _ in pairs[.5] if valid[g]]
            margins = []
            for position, (left, lg) in enumerate(observed):
                for right, rg in observed[position+1:]:
                    left_only = (labels[lg] > .5) & (labels[rg] < .5)
                    right_only = (labels[rg] > .5) & (labels[lg] < .5)
                    difference = predictions[left].function_probs - predictions[right].function_probs
                    margins.extend(difference[left_only].tolist())
                    margins.extend((-difference[right_only]).tolist())
            if margins:
                correct += sum(v > 0 for v in margins); comparisons += len(margins)
                eligible += 1; all_correct += int(all(v > 0 for v in margins))
    result = dict(proteins=len(index), correct=correct, comparisons=comparisons, eligible_proteins=eligible, proteins_all_correct=all_correct, fraction_correct=correct/comparisons if comparisons else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))

if __name__ == "__main__":
    main()
