import unittest

import cv2
import numpy as np

from core_modules.ma_colour_space import visualize_colour
from core_modules.mb_shape_contours import extract_shape, visualize_shape
from core_modules.mc_texture_glmc import visualize_texture
from core_modules.md_gabor_filters import visualize_gabor


def _sample_image():
    img = np.full((256, 256, 3), 255, np.uint8)
    cv2.circle(img, (128, 128), 90, (40, 180, 60), -1)
    return img


class FilterVisualizationTests(unittest.TestCase):
    def setUp(self):
        self.img = _sample_image()

    def test_visualize_colour_returns_bgr_heatmap(self):
        out = visualize_colour(self.img)
        self.assertEqual(out.shape, self.img.shape)
        self.assertEqual(out.dtype, np.uint8)

    def test_visualize_shape_draws_contour_without_changing_extract_shape(self):
        out = visualize_shape(self.img)
        self.assertEqual(out.shape, self.img.shape)
        self.assertEqual(out.dtype, np.uint8)
        # The overlay must not mutate the caller's image.
        self.assertTrue(np.array_equal(self.img, _sample_image()))

        norm_area, norm_perimeter, circularity, aspect_ratio, convexity = extract_shape(self.img)
        self.assertGreater(norm_area, 0)
        self.assertGreater(norm_perimeter, 0)
        self.assertGreater(circularity, 0)
        self.assertGreater(convexity, 0)

    def test_visualize_shape_handles_blank_image_with_no_contour(self):
        blank = np.full((256, 256, 3), 255, np.uint8)
        out = visualize_shape(blank)
        self.assertEqual(out.shape, blank.shape)
        vec = extract_shape(blank)
        self.assertTrue(np.array_equal(vec, np.zeros(5, dtype=np.float32)))

    def test_visualize_texture_returns_quantized_grayscale_as_bgr(self):
        out = visualize_texture(self.img)
        self.assertEqual(out.shape, self.img.shape)
        # A BGR-replicated grayscale image has equal channels per pixel.
        self.assertTrue(np.array_equal(out[:, :, 0], out[:, :, 1]))
        self.assertTrue(np.array_equal(out[:, :, 1], out[:, :, 2]))

    def test_visualize_gabor_returns_normalized_response(self):
        out = visualize_gabor(self.img)
        self.assertEqual(out.shape, self.img.shape)
        self.assertEqual(out.dtype, np.uint8)
        self.assertGreaterEqual(out.min(), 0)
        self.assertLessEqual(out.max(), 255)


if __name__ == "__main__":
    unittest.main()
