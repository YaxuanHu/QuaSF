"""Behavior checks for shared representation, auxiliary isolation, and checkpoint ABI."""
import io
import unittest
import torch
from model import SharedQueryModel
from dual_model import DualQueryDecoupledModel as Dual


class SharedTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        self.cfg = dict(hidden_dim=16, num_queries=3, num_dn_queries=2,
                        num_gt_func_queries=2, num_encoder_layers=1,
                        num_decoder_layers=2, nhead=2, dim_feedforward=32,
                        num_classes=5, rep_dim=8)
        self.model = SharedQueryModel(**self.cfg).eval()
        self.x = torch.randn(2, 12, 16)
        self.valid = torch.ones(2, 12)
        self.valid[1, 10:] = 0
        m = torch.zeros(12); m[2:8] = 1
        label = torch.tensor([1., 0., 1., 0., 0.])
        self.gt = [[(m, label, True)], [(m, label, True)]]

    def test_shared_gradients_and_budget(self) -> None:
        out = self.model(self.x, self.valid)
        self.assertIs(out['seg_queries'], out['func_queries'])
        self.assertEqual(len(self.model.seg_decoder.layers), 4)
        self.assertFalse(hasattr(self.model, 'func_decoder'))
        for key in ['mask_logits', 'func_logits']:
            self.model.zero_grad(set_to_none=True)
            out = self.model(self.x, self.valid)
            out[key].square().mean().backward()
            for layer in self.model.seg_decoder.layers:
                self.assertGreater(layer.linear1.weight.grad.abs().sum().item(), 0)
        dual = Dual(**self.cfg)
        delta = sum(p.numel() for p in dual.parameters()) - sum(p.numel() for p in self.model.parameters())
        self.assertEqual(delta, 3 * 16)

    def test_auxiliary_isolation_and_labels(self) -> None:
        plain = self.model(self.x, self.valid)
        aux = self.model(self.x, self.valid, [[(2,8)],[(2,8)]], self.gt, True)
        for key in ['mask_logits', 'func_logits', 'protein_logits']:
            torch.testing.assert_close(plain[key], aux[key], atol=2e-6, rtol=2e-5)
        self.assertEqual(aux['dn_mask_logits'].shape, (2,2,12))
        self.assertEqual(aux['gt_func_logits'].shape, (2,2,5))
        torch.testing.assert_close(aux['gt_func_labels'][0,0], self.gt[0][0][1])

    def test_strict_state_roundtrip(self) -> None:
        buf = io.BytesIO()
        torch.save(self.model.state_dict(), buf)
        buf.seek(0)
        other = SharedQueryModel(**self.cfg).eval()
        other.load_state_dict(torch.load(buf, weights_only=True), strict=True)
        torch.testing.assert_close(self.model(self.x,self.valid)['func_logits'],
                                   other(self.x,self.valid)['func_logits'])


if __name__ == '__main__':
    unittest.main()
