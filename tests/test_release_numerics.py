"""Protect regional identity, gradients, training and historical weight parity."""
from __future__ import annotations

import unittest

import numpy as np
import torch
from torch import nn

from quasf.metrics import Fmax, add_rf, segmentation
from quasf.decoders import Prediction
from quasf.model import BatchedJointFunctionQuery

torch.set_num_threads(1)


class SegmenterFixture(nn.Module):
    """Two disconnected final regions belong to the same source query."""

    def __init__(self, empty=False):
        super().__init__()
        logits = torch.full((1, 2, 32), -8.0)
        if not empty:
            logits[0, 0, :12] = 3.0
            logits[0, 0, 20:] = 3.0
        self.mask = nn.Parameter(logits)

    def forward(self, residue_feat, residue_mask, **kwargs):
        old = torch.zeros(1, 2, 256)
        return {"mask_logits": self.mask, "func_logits": old,
                "protein_logits": old.max(1).values}


class FunctionFixture(nn.Module):
    def forward(self, features, valid):
        logits = features[:, :, :1].expand(-1, -1, 256)
        return logits, logits


class ReleaseContracts(unittest.TestCase):
    """Run without historical directories in the standalone source archive."""

    def test_distinct_regions_and_mask_gradient(self):
        segmenter = SegmenterFixture()
        model = BatchedJointFunctionQuery(segmenter, FunctionFixture())
        features = torch.zeros(1, 32, 1280)
        features[0, :12, 0] = torch.linspace(1, 2, 12)
        features[0, 20:, 0] = torch.linspace(7, 8, 12)
        output = model(features, torch.ones(1, 32))
        self.assertEqual(output["region_source_query_ids"].tolist(), [[0, 0]])
        self.assertEqual(tuple(output["region_logits"].shape), (1, 2, 256))
        self.assertLess(float(output["region_logits"][0, 0, 0]),
                        float(output["region_logits"][0, 1, 0]))
        output["region_logits"][0, 0, 0].backward()
        self.assertGreater(float(segmenter.mask.grad.abs().sum()), 0)

    def test_empty_region_placeholder_is_excluded(self):
        model = BatchedJointFunctionQuery(SegmenterFixture(empty=True), FunctionFixture())
        result = model(torch.zeros(1, 32, 1280), torch.ones(1, 32))
        self.assertFalse(result["region_valid"].any())
        self.assertTrue(torch.isfinite(result["region_logits"]).all())
        torch.testing.assert_close(result["protein_logits"], torch.zeros(1, 256))

    def test_strict_metrics_penalize_missed_and_extra_regions(self):
        mask = np.ones(12, dtype=bool)
        predictions = [Prediction(0, mask, np.array([0.9, 0.1]), 1., 1.)]
        strict, matched = Fmax(), Fmax()
        counts = add_rf(strict, matched, predictions, np.array([[1., 0.]]),
                        np.array([True]), [])
        self.assertEqual(counts["observed_missed"], 1)
        self.assertEqual(counts["extra_predictions"], 1)
        self.assertEqual(strict.result()["macro_fmax"], 0)
        self.assertEqual(matched.result()["n"], 0)
        row, pairs = segmentation(predictions, np.array([mask]), 12)
        self.assertEqual(row["domain_iou"], 1)
        self.assertEqual(pairs[0.5], [(0, 0, 1.)])
