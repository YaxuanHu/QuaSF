"""Verify duplicate, absent and invalid query rows and their backpropagation."""
import unittest
import torch
from quasf.model import aggregate_queries


class AggregationParity(unittest.TestCase):
    def test_outputs_and_gradients(self) -> None:
        torch.manual_seed(19)
        original = torch.randn(3, 5, 7, requires_grad=True)
        logits = torch.randn(3, 6, 7, requires_grad=True)
        sources = torch.tensor([[1, 1, 3, 0, 0, 0], [0, 4, 4, 2, 0, 0], [0, 0, 0, 0, 0, 0]])
        valid = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0]], dtype=torch.bool)
        expected = original.clone()
        for i in range(3):
            for q in range(5):
                members = valid[i] & (sources[i] == q)
                if members.any():
                    expected[i, q] = logits[i, members].amax(0)
        actual = aggregate_queries(original, logits, sources, valid)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        weights = torch.randn_like(actual)
        reference_grad = torch.autograd.grad((expected * weights).sum(), (original, logits), retain_graph=True)
        actual_grad = torch.autograd.grad((actual * weights).sum(), (original, logits))
        for left, right in zip(reference_grad, actual_grad):
            torch.testing.assert_close(left, right, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
