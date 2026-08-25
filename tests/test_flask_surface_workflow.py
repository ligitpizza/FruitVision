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
                get_all=lambda *args, **kwargs: [],
                get_by_id=lambda *args, **kwargs: None,
                update_result=lambda *args, **kwargs: False,
                delete_result=lambda *args, **kwargs: False,
                get_stats=lambda *args, **kwargs: {},
                get_stats_since=lambda *args, **kwargs: {},
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

        # Authentication is covered separately; this test isolates the
        # classification, surface-analysis, and persistence workflow.
        app_module.PUBLIC_PATHS.add("/predict_unified")

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
            self.assertIn("marketability", payload)
            self.assertEqual(payload["marketability"]["status"], "ready")
            self.assertGreater(payload["marketability"]["min_days"], 0)
            self.assertGreater(payload["fruit_area_px"], 0)
            self.assertTrue(payload["surface_path"].startswith("surface/"))
            self.assertTrue(os.path.exists(os.path.join(app_module.OUTPUTS_DIR, payload["surface_path"])))
            self.assertEqual(len(logged), 1)
            self.assertEqual(logged[0]["blemish_percentage"], payload["blemish_percentage"])
            self.assertEqual(logged[0]["quality_grade"], payload["quality_grade"])
            self.assertEqual(logged[0]["marketability_status"], "ready")
            self.assertEqual(logged[0]["marketability_min_days"], payload["marketability"]["min_days"])

            app_module.PUBLIC_PATHS.add("/marketability")
            app_module.get_all_results = lambda **kwargs: [
                {
                    "id": 1, "created_at": "2099-01-01T00:00:00", "member": "ensemble_ab",
                    "filename": "ripe.png", "fruit": "apple", "label": "ripe",
                    "confidence": 92.0, "annotated_path": None, "blemish_percentage": 2.0,
                    "quality_grade": "Grade A", "marketability_status": "ready",
                    "dispatch_priority": "high", "marketability_min_days": 7,
                    "marketability_max_days": 14, "marketability_action": "Ready for market.",
                    "marketability_reliability": "high",
                    "marketability_storage_assumption": "test cold storage",
                },
                {
                    "id": 2, "created_at": "2099-01-01T00:00:00", "member": "ensemble_ab",
                    "filename": "rotten.png", "fruit": "apple", "label": "rotten",
                    "confidence": 95.0, "annotated_path": None, "blemish_percentage": 25.0,
                    "quality_grade": "Grade C", "marketability_status": "remove",
                    "dispatch_priority": "remove", "marketability_min_days": 0,
                    "marketability_max_days": 0, "marketability_action": "Do not market this fruit.",
                    "marketability_reliability": "high",
                    "marketability_storage_assumption": "test cold storage",
                },
                {
                    "id": 3, "created_at": "2099-01-01T00:00:00", "member": "ensemble_ab",
                    "filename": "batch.png", "fruit": "banana", "label": "ripe",
                    "confidence": 88.0, "annotated_path": None, "blemish_percentage": None,
                    "quality_grade": None, "source": "analyse_multi_fruit",
                    "detection_breakdown": '{"ripe": 2, "rotten": 1}',
                    "marketability_status": "ready", "dispatch_priority": "high",
                    "marketability_min_days": 3, "marketability_max_days": 7,
                    "marketability_action": "Ready for market.",
                    "marketability_reliability": "moderate",
                    "marketability_storage_assumption": "test storage",
                },
            ]
            dashboard_response = client.get("/marketability")
            self.assertEqual(dashboard_response.status_code, 200)
            dashboard_html = dashboard_response.get_data(as_text=True)
            self.assertIn("Marketability Dashboard", dashboard_html)
            self.assertIn("Predictor", dashboard_html)
            self.assertIn("Ensemble AB (Colour + Shape)", dashboard_html)
            self.assertIn("Do not market this fruit.", dashboard_html)
            self.assertLess(dashboard_html.index("#2"), dashboard_html.index("#1"))
            self.assertIn("Multi-fruit batch", dashboard_html)
            self.assertIn("2 ripe", dashboard_html)
            self.assertIn("1 rotten", dashboard_html)
            self.assertIn("Sort this mixed batch", dashboard_html)
            self.assertIn("Needs Review", dashboard_html)
            self.assertIn("NEEDS REVIEW", dashboard_html)
            self.assertIn("openMarketabilityReview", dashboard_html)
            self.assertIn("This original model result will not be changed.", dashboard_html)

            rotten_only_response = client.get("/marketability?ripeness=rotten")
            rotten_only_html = rotten_only_response.get_data(as_text=True)
            self.assertIn("#2", rotten_only_html)
            self.assertNotIn("#1", rotten_only_html)
            self.assertIn('value="rotten" selected', rotten_only_html)

            batch_only_response = client.get("/marketability?analysis=multi_fruit_batch")
            batch_only_html = batch_only_response.get_data(as_text=True)
            self.assertIn("#3", batch_only_html)
            self.assertNotIn("#1", batch_only_html)
            self.assertNotIn("#2", batch_only_html)

            review_only_response = client.get("/marketability?review=needs_review")
            review_only_html = review_only_response.get_data(as_text=True)
            self.assertIn("#2", review_only_html)
            self.assertIn("#3", review_only_html)
            self.assertNotIn("#1", review_only_html)
            self.assertIn('value="needs_review" selected', review_only_html)

            review_updates = []
            app_module.get_by_id = lambda record_id: {
                "id": record_id, "fruit": "banana", "label": "rotten", "confidence": 76.1,
            }
            app_module.update_result = lambda record_id, **fields: review_updates.append((record_id, fields)) or True
            app_module.auth_db.get_user_by_id = lambda user_id: {
                "id": user_id, "name": "Test Farmer", "email": "farmer@example.test",
                "role": "farmer", "dark_mode": 0,
            }
            app_module.auth_db.log_activity = lambda *args, **kwargs: None
            with client.session_transaction() as session_data:
                session_data["user_id"] = 7
            review_response = client.post(
                "/marketability/2/review",
                data={
                    "decision": "correct", "review_fruit": "banana",
                    "review_label": "ripe", "reason": "Yellow peel and firm fruit.",
                },
            )
            self.assertEqual(review_response.status_code, 302)
            self.assertEqual(review_updates[0][0], 2)
            self.assertEqual(review_updates[0][1]["review_status"], "corrected")
            self.assertEqual(review_updates[0][1]["review_label"], "ripe")
            self.assertNotIn("label", review_updates[0][1])
            self.assertNotIn("confidence", review_updates[0][1])

            expired = app_module._marketability_for_record({
                "id": 3, "created_at": "2020-01-01T00:00:00", "fruit": "banana",
                "label": "ripe", "confidence": 95.0, "blemish_percentage": 1.0,
                "quality_grade": "Grade A", "marketability_status": "ready",
                "dispatch_priority": "high", "marketability_min_days": 3,
                "marketability_max_days": 7, "marketability_action": "Ready for market.",
                "marketability_reliability": "high",
                "marketability_storage_assumption": "test storage",
            })
            self.assertEqual(expired["status"], "inspect")
            self.assertEqual(expired["dispatch_priority"], "urgent")
            self.assertIsNone(expired["window"])

            app_module.get_recent = lambda **kwargs: app_module.get_all_results()
            dashboard_stats = {
                "total": 3,
                "avg_confidence": 91.7,
                "avg_latency_ms": 10.0,
                "by_fruit": {"apple": 2, "banana": 1},
                "by_label": {"ripe": 2, "rotten": 1},
                "avg_confidence_by_fruit": {"apple": 93.5, "banana": 88.0},
            }
            app_module.get_stats = lambda *args, **kwargs: dashboard_stats
            app_module.get_stats_since = lambda *args, **kwargs: dashboard_stats
            app_module.PUBLIC_PATHS.add("/")
            home_response = client.get("/")
            home_html = home_response.get_data(as_text=True)
            self.assertIn("marketability-alert-close", home_html)
            self.assertIn("Dismiss urgent handling alert", home_html)
            self.assertIn("dismissMarketabilityAlert", home_html)


if __name__ == "__main__":
    unittest.main()
