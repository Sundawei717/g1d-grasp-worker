import unittest

import torch

from chunk_model import SharedResNet18ChunkTransformer, masked_chunk_loss


class ChunkModelTest(unittest.TestCase):
    def test_train_path_refuses_missing_pretrained_weights(self):
        with self.assertRaisesRegex(ValueError, "ImageNet"):
            SharedResNet18ChunkTransformer()

    def test_true_shared_visual_path_preserves_action_shape(self):
        model = SharedResNet18ChunkTransformer(fixture_random_init=True).eval()
        self.assertEqual(torch.count_nonzero(model.backbone[0].weight[:, 3:]).item(), 0)
        with torch.inference_mode():
            output = model(torch.zeros(1, 5, 224, 224), torch.ones(1, 5, 224, 224),
                           torch.zeros(1, 2, 8), torch.zeros(1, 6))
        self.assertEqual(tuple(output.shape), (1, 8, 8))
        self.assertTrue(torch.isfinite(output).all().item())

    def test_trial_lift_and_padding_do_not_affect_loss(self):
        prediction = torch.zeros(1, 8, 8)
        targets = torch.ones(1, 8, 8)
        valid = torch.tensor([True, False, False, False, False, False, False, False])[None]
        baseline = masked_chunk_loss(prediction, targets, valid)
        targets[:, 1:] = 1000
        self.assertEqual(masked_chunk_loss(prediction, targets, valid).item(), baseline.item())


if __name__ == "__main__":
    unittest.main()
