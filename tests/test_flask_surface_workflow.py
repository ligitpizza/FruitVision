import importlib.util
import io
import json
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


def _mixed_m14_analysis(image):
    detections = []
    for fruit, label, confidence, bbox in (
        ("apple", "ripe", 91.0, (5, 5, 35, 45)),
        ("banana", "unripe", 87.0, (40, 5, 75, 45)),
        ("orange", "rotten", 94.0, (80, 5, 115, 45)),
    ):
        probabilities = {"ripe": 0.03, "unripe": 0.03, "rotten": 0.03}
        probabilities[label] = confidence / 100
        detections.append({
            "fruit": fruit, "fruit_confidence": 95.0, "bbox": bbox,
            "label": label, "ripeness_confidence": confidence,
            "probabilities": probabilities, "error": None,
        })
    return {
        "model_key": "merged_1_4",
        "model_label": "YOLOv8n Detection + Merged 1+4 (M14) Ripeness",
        "detections": detections,
        "detected_count": 3,
        "classified_count": 3,
        "needs_review_count": 0,
        "fruit_breakdown": {"apple": 1, "banana": 1, "orange": 1},
        "ripeness_breakdown": {"ripe": 1, "unripe": 1, "rotten": 1},
        "annotated_image": image.copy(),
        "latency_ms": 12.3,
    }


def _module(**attributes):
    module = types.ModuleType("stub")
    for name, value in attributes.items():
        setattr(module, name, value)
    return module


class FlaskSurfaceWorkflowTests(unittest.TestCase):
    def test_unified_prediction_returns_and_persists_surface_fields(self):
        logged = []
        stock_events = []
        predictor_stubs = {
            "member_apps.member_1_ab.m1_predict": _module(predict_ripeness=_predict, NotAFruitError=DummyNotFruitError),
            "member_apps.member_2_bc.m2_predict": _module(predict_ripeness=_predict, NotAFruitError=DummyNotFruitError),
            "member_apps.member_3_cd.m3_predict": _module(predict_ripeness=_predict, NotAFruitError=DummyNotFruitError),
            "member_apps.member_4_da.m4_predict": _module(predict_ripeness=_predict, NotAFruitError=DummyNotFruitError),
            "pipeline.pure_yolo.yolo_cls_predict": _module(
                predict_ripeness=_predict, NotAFruitError=DummyNotFruitError
            ),
            "member_apps.predict_ensemble": _module(
                predict_ensemble=lambda image, fruit: (
                    "ripe", 92.0,
                    {
                        "member_1_ab": {"label": "ripe", "confidence": 91.0, "proba": {"ripe": 91.0, "unripe": 5.0, "rotten": 4.0}, "cleaned_img": image.copy()},
                        "member_2_bc": {"label": "ripe", "confidence": 93.0, "proba": {"ripe": 93.0, "unripe": 4.0, "rotten": 3.0}, "cleaned_img": image.copy()},
                    },
                    (5, 5, image.shape[1] - 5, image.shape[0] - 5),
                )
            ),
            "core_modules.pdf_report": _module(
                generate_pdf_report=lambda *args, **kwargs: "report.pdf",
                generate_pdf_report_batch=lambda *args, **kwargs: "report.pdf",
                generate_stock_report_pdf=lambda *args, **kwargs: "stock_report.pdf",
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
            "core_modules.mixed_fruit_m14": _module(
                analyze_mixed_fruit_m14=_mixed_m14_analysis,
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
                get_fruit_label_breakdown=lambda *args, **kwargs: {},
            ),
            "database.stock_db": _module(
                log_stock_event=lambda **kwargs: stock_events.append(kwargs),
                get_paginated=lambda *args, **kwargs: ([], 0),
                get_by_id=lambda *args, **kwargs: None,
                update_stock_event=lambda *args, **kwargs: False,
                delete_stock_event=lambda *args, **kwargs: False,
                get_summary=lambda *args, **kwargs: {},
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

            app_module.PUBLIC_PATHS.add("/classify")
            classify_response = client.get("/classify")
            classify_html = classify_response.get_data(as_text=True)
            self.assertIn("Mixed-Fruit Analysis", classify_html)
            self.assertIn("M14 ONLY", classify_html)

            app_module.PUBLIC_PATHS.add("/analyse-mixed-fruit-m14")
            mixed_response = client.post(
                "/analyse-mixed-fruit-m14",
                data={"image": (io.BytesIO(encoded.tobytes()), "mixed.png")},
                content_type="multipart/form-data",
            )
            self.assertEqual(mixed_response.status_code, 200)
            mixed_html = mixed_response.get_data(as_text=True)
            self.assertIn("Mixed Fruit — YOLOv8n + Merged 1+4", mixed_html)
            self.assertIn("Per-fruit results", mixed_html)
            self.assertIn("Apple", mixed_html)
            self.assertIn("Banana", mixed_html)
            self.assertIn("Orange", mixed_html)
            mixed_logs = logged[1:]
            self.assertEqual(len(mixed_logs), 3)
            self.assertTrue(all(item["member"] == "merged_1_4" for item in mixed_logs))
            self.assertTrue(all(item["source"] == "analyse_mixed_fruit_m14" for item in mixed_logs))

            # --- single upload automatically counts a ready ripe result
            # into stock, no "add to stock" checkbox needed (unlike batch) ---
            self.assertEqual(len(stock_events), 1)
            self.assertEqual(stock_events[0]["label"], "ripe")
            self.assertEqual(stock_events[0]["fruit"], "apple")
            self.assertEqual(stock_events[0]["source"], "single")
            stock_events.clear()

            # --- filter-technique photos: single-model prediction ("ab" ->
            # colour + shape) ---
            self.assertEqual(set(payload["filter_photos"].keys()), {"ab"})
            ab_photos = payload["filter_photos"]["ab"]
            self.assertEqual(set(ab_photos.keys()), {"colour", "shape"})
            for rel_path in ab_photos.values():
                self.assertTrue(os.path.exists(os.path.join(app_module.OUTPUTS_DIR, rel_path)))
            self.assertIn("filter_photos", logged[0])
            self.assertIsNotNone(logged[0]["filter_photos"])
            logged_filter_photos = json.loads(logged[0]["filter_photos"])
            self.assertEqual(logged_filter_photos, payload["filter_photos"])

            # --- filter-technique photos: All-Four ensemble, one sub-dict
            # per member (ab -> colour+shape, bc -> shape+texture) ---
            ensemble_response = client.post(
                "/predict_unified",
                data={
                    "fruit": "apple",
                    "model": "all_four",
                    "image": (io.BytesIO(encoded.tobytes()), "apple_ensemble.png"),
                },
                content_type="multipart/form-data",
            )
            self.assertEqual(ensemble_response.status_code, 200)
            ensemble_payload = ensemble_response.get_json()
            self.assertEqual(
                set(ensemble_payload["filter_photos"].keys()), {"member_1_ab", "member_2_bc"}
            )
            self.assertEqual(set(ensemble_payload["filter_photos"]["member_1_ab"].keys()), {"colour", "shape"})
            self.assertEqual(set(ensemble_payload["filter_photos"]["member_2_bc"].keys()), {"shape", "texture"})
            for member_photos in ensemble_payload["filter_photos"].values():
                for rel_path in member_photos.values():
                    self.assertTrue(os.path.exists(os.path.join(app_module.OUTPUTS_DIR, rel_path)))
            # per_member itself must stay JSON-serializable -- the raw
            # cleaned_img ndarray predict_ensemble() attaches has to be
            # stripped before this response is jsonify()'d.
            self.assertNotIn("cleaned_img", ensemble_payload["per_member"]["member_1_ab"])

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

            # --- pagination regression: rows sort worst-first (remove before
            # ready), so on a database dominated by "remove"-status history a
            # fresh "ready" result must still be reachable via pagination
            # instead of being silently dropped past a hard row cap. ---
            def _remove_row(row_id):
                return {
                    "id": row_id, "created_at": "2099-01-01T00:00:00", "member": "ensemble_ab",
                    "filename": f"remove_{row_id}.png", "fruit": "apple", "label": "rotten",
                    "confidence": 95.0, "annotated_path": None, "blemish_percentage": 25.0,
                    "quality_grade": "Grade C", "marketability_status": "remove",
                    "dispatch_priority": "remove", "marketability_min_days": 0,
                    "marketability_max_days": 0, "marketability_action": "Do not market this fruit.",
                    "marketability_reliability": "high", "marketability_storage_assumption": "test storage",
                }

            fresh_ready_row = {
                "id": 9999, "created_at": "2099-01-01T00:00:00", "member": "ensemble_ab",
                "filename": "fresh_ready.png", "fruit": "apple", "label": "ripe",
                "confidence": 92.0, "annotated_path": None, "blemish_percentage": 2.0,
                "quality_grade": "Grade A", "marketability_status": "ready",
                "dispatch_priority": "high", "marketability_min_days": 7,
                "marketability_max_days": 14, "marketability_action": "Ready for market.",
                "marketability_reliability": "high", "marketability_storage_assumption": "test storage",
            }
            app_module.get_all_results = lambda **kwargs: (
                [_remove_row(i) for i in range(1, 31)] + [fresh_ready_row]
            )

            page1_response = client.get("/marketability")
            page1_html = page1_response.get_data(as_text=True)
            # Within the same status/priority/days group, newest (highest id)
            # sorts first -- so id=30 (not id=1) is the one guaranteed on page 1.
            self.assertIn("#30<", page1_html)  # sanity: remove rows fill page 1
            self.assertNotIn("#9999", page1_html)  # BUG: fresh ready result invisible on page 1

            page2_response = client.get("/marketability?page=2")
            page2_html = page2_response.get_data(as_text=True)
            self.assertIn("#9999", page2_html)  # FIX: still reachable via pagination
            self.assertIn("31 record", page2_html)
            self.assertIn("page 2 of 2", page2_html)

            review_updates = []
            app_module.get_by_id = lambda record_id: {
                "id": record_id, "fruit": "banana", "label": "rotten", "confidence": 76.1,
            }
            app_module.update_result = lambda record_id, **fields: review_updates.append((record_id, fields)) or True
            # app_module.auth_db IS database.auth_db (never stubbed/replaced
            # in sys.modules by this test, unlike history_db/stock_db above)
            # -- overwriting its functions mutates that shared singleton
            # module for the rest of the test *process*, not just this app
            # instance, so restore them once this test ends or a later test
            # file (e.g. test_auth.py) can silently get these fakes instead
            # of the real DB-backed functions depending on run order.
            original_get_user_by_id = app_module.auth_db.get_user_by_id
            original_log_activity = app_module.auth_db.log_activity

            def _restore_auth_db_functions():
                app_module.auth_db.get_user_by_id = original_get_user_by_id
                app_module.auth_db.log_activity = original_log_activity

            self.addCleanup(_restore_auth_db_functions)

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

            # --- harvest record detail page: full input + output photos ---
            app_module.PUBLIC_PATHS.add("/history/1")
            app_module.PUBLIC_PATHS.add("/history/2")
            app_module.PUBLIC_PATHS.add("/history/999")
            with open(os.path.join(app_module.UPLOAD_DIR, "detail_input.png"), "wb") as fh:
                fh.write(b"fake-image-bytes")
            os.makedirs(os.path.join(app_module.OUTPUTS_DIR, "annotated"), exist_ok=True)
            with open(os.path.join(app_module.OUTPUTS_DIR, "annotated", "detail_out.png"), "wb") as fh:
                fh.write(b"fake-image-bytes")

            app_module.get_by_id = lambda record_id: {
                "id": 1, "created_at": "2026-01-01T00:00:00", "member": "ensemble_ab",
                "filename": "detail_input.png", "fruit": "apple", "label": "ripe",
                "confidence": 92.0, "annotated_path": "annotated/detail_out.png",
                "surface_path": None, "source": "predict", "blemish_percentage": 2.0,
                "quality_grade": "Grade A", "marketability_status": "ready",
                "flagged": 0, "detection_breakdown": None,
            }
            detail_response = client.get("/history/1")
            detail_html = detail_response.get_data(as_text=True)
            self.assertEqual(detail_response.status_code, 200)
            self.assertIn("/uploads/detail_input.png", detail_html)
            self.assertIn("/outputs/annotated/detail_out.png", detail_html)
            self.assertNotIn("Not available", detail_html)

            # --- filter photos: outputs/filters/ is gitignored, so a
            # persisted record's filter_photos JSON can reference a
            # technique image that isn't on disk (e.g. a teammate's fresh
            # clone) -- that one technique should degrade to "Not
            # available" instead of a broken <img>, without hiding the
            # techniques that ARE actually present. ---
            os.makedirs(os.path.join(app_module.OUTPUTS_DIR, "filters"), exist_ok=True)
            with open(os.path.join(app_module.OUTPUTS_DIR, "filters", "detail_colour_ab.png"), "wb") as fh:
                fh.write(b"fake-image-bytes")
            app_module.get_by_id = lambda record_id: {
                "id": 3, "created_at": "2026-01-01T00:00:00", "member": "ensemble_ab",
                "filename": "detail_input.png", "fruit": "apple", "label": "ripe",
                "confidence": 92.0, "annotated_path": "annotated/detail_out.png",
                "surface_path": None, "source": "predict", "blemish_percentage": 2.0,
                "quality_grade": "Grade A", "marketability_status": "ready",
                "flagged": 0, "detection_breakdown": None,
                "filter_photos": json.dumps({
                    "ab": {
                        "colour": "filters/detail_colour_ab.png",
                        "shape": "filters/detail_shape_ab_MISSING.png",
                    },
                }),
            }
            app_module.PUBLIC_PATHS.add("/history/3")
            filter_response = client.get("/history/3")
            filter_html = filter_response.get_data(as_text=True)
            self.assertEqual(filter_response.status_code, 200)
            self.assertIn("/outputs/filters/detail_colour_ab.png", filter_html)
            self.assertNotIn("/outputs/filters/detail_shape_ab_MISSING.png", filter_html)
            self.assertIn("Not available", filter_html)

            app_module.get_by_id = lambda record_id: {
                "id": 2, "created_at": "2026-01-01T00:00:00", "member": "ensemble_ab",
                # annotated_path is set (like a real DB row) but nothing on
                # disk backs it -- e.g. outputs/annotated/ is gitignored, so
                # a teammate's fresh clone has the DB row without the file.
                "filename": "missing_input.png", "fruit": "apple", "label": "ripe",
                "confidence": 92.0, "annotated_path": "annotated/missing_out.png",
                "surface_path": None, "source": "predict", "blemish_percentage": None,
                "quality_grade": None, "marketability_status": None,
                "flagged": 0, "detection_breakdown": None,
            }
            missing_response = client.get("/history/2")
            missing_html = missing_response.get_data(as_text=True)
            self.assertEqual(missing_response.status_code, 200)
            self.assertNotIn('src="/uploads/missing_input.png"', missing_html)
            self.assertNotIn('src="/outputs/annotated/missing_out.png"', missing_html)
            self.assertIn("Not available", missing_html)

            app_module.get_by_id = lambda record_id: None
            notfound_response = client.get("/history/999")
            self.assertEqual(notfound_response.status_code, 302)
            self.assertTrue(notfound_response.headers["Location"].endswith("/history"))

            uploads_response = client.get("/uploads/detail_input.png")
            self.assertEqual(uploads_response.status_code, 200)
            self.assertEqual(uploads_response.data, b"fake-image-bytes")
            # send_from_directory keeps the file handle open until the response
            # is explicitly closed; on Windows the temp-dir cleanup below can't
            # unlink the file while that handle is still alive.
            uploads_response.close()

            # --- ready-for-market gating on stock logging ---
            stock_events.clear()
            with app_module.app.test_request_context():
                app_module.g.user = None
                app_module._log_stock_result(True, "apple", "ripe", marketability_status="ready")
                app_module._log_stock_result(True, "apple", "ripe", marketability_status="inspect")
                app_module._log_stock_result(True, "apple", "unripe", marketability_status=None)
                app_module._log_stock_result(True, "apple", "rotten", marketability_status=None)
                app_module._log_stock_result(False, "apple", "ripe", marketability_status="ready")

            self.assertEqual([e["label"] for e in stock_events], ["ripe", "unripe", "rotten"])

            stock_events.clear()
            with app_module.app.test_request_context():
                app_module.g.user = None
                app_module._log_stock_result(
                    True, "apple", "ripe",
                    per_fruit=[
                        {"label": "ripe", "confidence": 0.95},
                        {"label": "ripe", "confidence": 0.40},
                        {"label": "unripe", "confidence": 0.30},
                        {"label": "rotten", "confidence": 0.20},
                    ],
                )
            logged_by_label = {e["label"]: e["quantity"] for e in stock_events}
            self.assertEqual(logged_by_label, {"ripe": 1, "unripe": 1, "rotten": 1})


if __name__ == "__main__":
    unittest.main()
