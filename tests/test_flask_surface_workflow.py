import importlib.util
import io
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from flask import Blueprint


PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))


class DummyNotFruitError(Exception):
    pass


def _predict(image, fruit_type):
    h, w = image.shape[:2]
    bbox = (5, 5, w - 5, h - 5)
    return "ripe", 0.92, bbox, image.copy(), {"ripe": 0.92, "unripe": 0.06, "rotten": 0.02}


def _module(**attributes):
    module = types.ModuleType("stub")
    for name, value in attributes.items():
        setattr(module, name, value)
    return module


class FlaskSurfaceWorkflowTests(unittest.TestCase):
    def test_unified_prediction_returns_and_persists_surface_fields(self):
        logged = []
        predictor_stubs = {
            "member_apps.member_1_ab.m1_predict": _module(predict_ripeness=_predict, NotAFruitError=DummyNotFruitError),
            "member_apps.member_2_bc.m2_predict": _module(predict_ripeness=_predict, NotAFruitError=DummyNotFruitError),
            "member_apps.member_3_cd.m3_predict": _module(predict_ripeness=_predict, NotAFruitError=DummyNotFruitError),
            "member_apps.member_4_da.m4_predict": _module(predict_ripeness=_predict, NotAFruitError=DummyNotFruitError),
            "pipeline.pure_yolo.yolo_cls_predict": _module(
                predict_ripeness=_predict, NotAFruitError=DummyNotFruitError
            ),
            "member_apps.predict_ensemble": _module(
                predict_ensemble=lambda image, fruit: ("ripe", 92.0, {}, (5, 5, image.shape[1] - 5, image.shape[0] - 5))
            ),
            "core_modules.pdf_report": _module(
                generate_pdf_report=lambda *args, **kwargs: "report.pdf",
                generate_pdf_report_batch=lambda *args, **kwargs: "report.pdf",
            ),
            "core_modules.dashboard_charts": _module(
                generate_trend_chart=lambda *args, **kwargs: None,
                generate_history_chart=lambda *args, **kwargs: None,
                generate_fruit_breakdown_chart=lambda *args, **kwargs: None,
                generate_confidence_trend_chart=lambda *args, **kwargs: None,
            ),
            "core_modules.fruit_validation": _module(
                validate_selected_fruit=lambda image, fruit: {
                    "selected_fruit": fruit,
                    "detected_fruit": fruit,
                    "confidence": 0.9,
                    "validation_method": "test",
                },
                FruitValidationError=DummyNotFruitError,
            ),
            "database.history_db": _module(
                log_result=lambda **kwargs: logged.append(kwargs),
                get_recent=lambda *args, **kwargs: [],
                get_paginated=lambda *args, **kwargs: ([], 0),
                get_by_id=lambda *args, **kwargs: None,
                update_result=lambda *args, **kwargs: False,
                delete_result=lambda *args, **kwargs: False,
                get_stats=lambda *args, **kwargs: {},
            ),
        }
        realtime_module = _module(realtime_bp=Blueprint("realtime_test", __name__))
        predictor_stubs["realtime.stream_routes"] = realtime_module

        with patch.dict(sys.modules, predictor_stubs):
            spec = importlib.util.spec_from_file_location(
                "fruitvision_test_app", os.path.join(PROJECT_ROOT, "app.py")
            )
            app_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(app_module)

        with tempfile.TemporaryDirectory() as temp_dir:
            app_module.UPLOAD_DIR = os.path.join(temp_dir, "uploads")
            app_module.OUTPUTS_DIR = os.path.join(temp_dir, "outputs")
            os.makedirs(app_module.UPLOAD_DIR)
            os.makedirs(app_module.OUTPUTS_DIR)

            image = np.full((120, 120, 3), 255, np.uint8)
            cv2.circle(image, (60, 60), 45, (20, 180, 220), -1)
            cv2.circle(image, (65, 60), 8, (10, 10, 10), -1)
            ok, encoded = cv2.imencode(".png", image)
            self.assertTrue(ok)

            client = app_module.app.test_client()
            response = client.post(
                "/predict_unified",
                data={
                    "fruit": "apple",
                    "model": "ab",
                    "image": (io.BytesIO(encoded.tobytes()), "apple.png"),
                },
                content_type="multipart/form-data",
            )

            self.assertEqual(response.status_code, 200)
            payload = response.get_json()
            self.assertIn("blemish_percentage", payload)
            self.assertIn("quality_grade", payload)
            self.assertGreater(payload["fruit_area_px"], 0)
            self.assertTrue(payload["surface_path"].startswith("surface/"))
            self.assertTrue(os.path.exists(os.path.join(app_module.OUTPUTS_DIR, payload["surface_path"])))
            self.assertEqual(len(logged), 1)
            self.assertEqual(logged[0]["blemish_percentage"], payload["blemish_percentage"])
            self.assertEqual(logged[0]["quality_grade"], payload["quality_grade"])


if __name__ == "__main__":
    unittest.main()
