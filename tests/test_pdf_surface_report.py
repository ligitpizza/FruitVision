import os
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()
