import unittest

from pipeline.fruit_detector.dataset_prep import bbox_to_yolo_line, is_degenerate_box


class BboxToYoloLineTests(unittest.TestCase):
    def test_full_frame_box_is_centered_and_full_size(self):
        line = bbox_to_yolo_line((0, 0, 100, 200), img_width=100, img_height=200, class_id=0)
        self.assertEqual(line, "0 0.500000 0.500000 1.000000 1.000000")

    def test_quarter_box_in_top_left(self):
        line = bbox_to_yolo_line((0, 0, 50, 50), img_width=100, img_height=100, class_id=0)
        self.assertEqual(line, "0 0.250000 0.250000 0.500000 0.500000")

    def test_off_center_box(self):
        line = bbox_to_yolo_line((20, 40, 60, 100), img_width=200, img_height=200, class_id=0)
        # cx=(20+60)/2/200=0.2, cy=(40+100)/2/200=0.35, w=40/200=0.2, h=60/200=0.3
        self.assertEqual(line, "0 0.200000 0.350000 0.200000 0.300000")


class IsDegenerateBoxTests(unittest.TestCase):
    def test_full_frame_box_is_degenerate(self):
        self.assertTrue(is_degenerate_box((0, 0, 100, 100), img_width=100, img_height=100))

    def test_small_centered_box_is_not_degenerate(self):
        self.assertFalse(is_degenerate_box((25, 25, 75, 75), img_width=100, img_height=100))

    def test_exactly_at_threshold_is_degenerate(self):
        # 0.9 * 100*100 = 9000; a 95x95 box has area 9025 >= 9000
        self.assertTrue(is_degenerate_box((0, 0, 95, 95), img_width=100, img_height=100))


if __name__ == "__main__":
    unittest.main()
