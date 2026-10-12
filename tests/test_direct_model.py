"""Compare direct functional prediction with the recorded experiment."""
import unittest
import torch
from quasf.model import DirectFunctionQuery
from quasf.pretrain import build_model


class DirectModelTests(unittest.TestCase):
    def test_factory_preserves_recorded_direct_initialization(self):
        torch.manual_seed(42)
        expected = DirectFunctionQuery()
        actual = build_model()
        self.assertEqual(expected.state_dict().keys(), actual.state_dict().keys())
        for name, tensor in expected.state_dict().items():
            torch.testing.assert_close(tensor, actual.state_dict()[name], rtol=0, atol=0)

    def test_no_base_mlp_and_trainable_direct_logits(self):
        model = DirectFunctionQuery()
        self.assertFalse(hasattr(model, "base"))
        self.assertFalse(any(k.startswith('base.') for k in model.state_dict()))
        with torch.no_grad():
            model.delta_head.weight.normal_(0, .02)
        x = torch.randn(2, 3, 2560, requires_grad=True)
        valid = torch.tensor([[True, True, False], [True, True, True]])
        output, compatibility = model(x, valid)
        self.assertIs(output, compatibility)
        self.assertEqual(output.shape, (2, 3, 256))
        output[valid].sum().backward()
        self.assertGreater(float(x.grad.abs().sum()), 0)
