import os
import tempfile
import unittest

import cv2
import numpy as np

from core_modules.pdf_report import generate_pdf_report, generate_pdf_report_batch


class SurfacePdfReportTests(unittest.TestCase):
    def test_single_and_batch_reports_accept_surface_metrics(self):
        surface = {
            "fruit_area_px": 1000,
            "blemish_area_px": 50,
            "blemish_percentage": 5.0,
            "quality_grade": "Grade A",
            "surface_image_path": None,
        }
        with tempfile.TemporaryDirectory() as output_dir:
            single = generate_pdf_report(
                None,
                "ripe",
                0.9,
                output_dir=output_dir,
                surface_data=surface,
            )
            batch = generate_pdf_report_batch(
                [{"filename": "apple.jpg", "label": "ripe", "confidence": 90.0, **surface}],
                output_dir=output_dir,
            )
            self.assertTrue(os.path.exists(single))
            self.assertTrue(os.path.exists(batch))
            self.assertGreater(os.path.getsize(single), 0)
            self.assertGreater(os.path.getsize(batch), 0)


class FilterPhotosPdfReportTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        thumb = np.full((256, 256, 3), 128, np.uint8)
        self.thumb_path = os.path.join(self.temp_dir.name, "colour_ab.jpg")
        cv2.imwrite(self.thumb_path, thumb)

    def test_single_report_embeds_filter_photos(self):
        filter_photos = [{
            "member_label": "Ensemble AB (Colour + Shape)",
            "techniques": [
                {"label": "Colour Space (Lab A-channel)", "path": self.thumb_path},
                {"label": "Shape / Contour", "path": self.thumb_path},
            ],
        }]
        # Distinct model_tag per call -- generate_pdf_report's output
        # filename is timestamp-based at 1-second resolution, so two calls
        # in the same second with the same model_tag would collide and the
        # second write would silently overwrite the first.
        without = generate_pdf_report(None, "ripe", 0.9, model_tag="ab-without", output_dir=self.temp_dir.name)
        with_photos = generate_pdf_report(
            None, "ripe", 0.9, model_tag="ab-with", output_dir=self.temp_dir.name, filter_photos=filter_photos,
        )
        self.assertTrue(os.path.exists(with_photos))
        self.assertGreater(os.path.getsize(with_photos), os.path.getsize(without))

    def test_multi_fruit_member_label_with_em_dash_does_not_crash(self):
        # app.py's _filter_photos_display() builds multi-fruit member labels
        # like "Fruit #1 (RIPE) — Ensemble AB (...)" using a real em dash --
        # FPDF's core Helvetica font only supports Latin-1 and raises
        # FPDFUnicodeEncodingException on that character.
        filter_photos = [{
            "member_label": "Fruit #1 (RIPE) — Ensemble AB (Colour + Shape)",
            "techniques": [{"label": "Colour Space (Lab A-channel)", "path": self.thumb_path}],
        }]
        out_path = generate_pdf_report(
            None, "ripe", 0.9, output_dir=self.temp_dir.name, filter_photos=filter_photos,
        )
        self.assertTrue(os.path.exists(out_path))

    def test_missing_filter_photo_path_is_skipped_not_crashed(self):
        filter_photos = [{
            "member_label": "Ensemble AB (Colour + Shape)",
            "techniques": [
                {"label": "Colour Space (Lab A-channel)", "path": os.path.join(self.temp_dir.name, "missing.jpg")},
            ],
        }]
        out_path = generate_pdf_report(
            None, "ripe", 0.9, output_dir=self.temp_dir.name, filter_photos=filter_photos,
        )
        self.assertTrue(os.path.exists(out_path))

    def test_batch_report_embeds_per_result_filter_photos(self):
        filter_photos = [{
            "member_label": "Ensemble AB (Colour + Shape)",
            "techniques": [{"label": "Colour Space (Lab A-channel)", "path": self.thumb_path}],
        }]
        out_path = generate_pdf_report_batch(
            [{
                "filename": "apple.jpg", "label": "ripe", "confidence": 90.0,
                "filter_photos": filter_photos,
            }],
            output_dir=self.temp_dir.name,
        )
        self.assertTrue(os.path.exists(out_path))
        self.assertGreater(os.path.getsize(out_path), 0)


if __name__ == "__main__":
    unittest.main()
