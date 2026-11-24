import unittest

import numpy as np

from prompt_views import encode_views


class PromptViewsTest(unittest.TestCase):
    def setUp(self):
        self.rgb = np.full((480, 640, 3), 75, dtype=np.uint8)

    def test_left_eye_views_roundtrip_and_padding(self):
        wide, local, _ = encode_views(self.rgb, [0.2, 0.4, 0.5, 0.6], [0.4, 0.5], min_crop_px=32)
        for view in (wide, local):
            self.assertEqual(view.channels.shape, (5, 224, 224))
            self.assertTrue(view.point_visible)
            self.assertFalse(view.channels[:, ~view.valid_pixels].any())
            origin = [0.4 * 639, 0.5 * 479, 1]
            mapped = view.raw_to_view @ origin
            np.testing.assert_allclose(view.view_to_raw @ mapped, origin)
            row, col = np.unravel_index(np.argmax(view.channels[4]), view.channels[4].shape)
            self.assertLess(abs(col - mapped[0]), 1)
            self.assertLess(abs(row - mapped[1]), 1)
        self.assertFalse(wide.valid_pixels[0].any())

    def test_temporal_crop_enlarges_for_valid_point(self):
        _, _, previous = encode_views(self.rgb, [0.02, 0.02, 0.06, 0.06], [0.03, 0.03], min_crop_px=32)
        _, local, current = encode_views(self.rgb, [0.55, 0.55, 0.6, 0.6], [0.62, 0.62], min_crop_px=32, previous=previous)
        self.assertTrue(local.point_visible)
        self.assertGreater(current.side_px, previous.side_px)

    def test_reject_invalid_prompt_without_clamping(self):
        with self.assertRaises(ValueError):
            encode_views(self.rgb, [0.2, 0.4, 0.5, 0.6], [1.2, 0.4], min_crop_px=32)


if __name__ == "__main__":
    unittest.main()
