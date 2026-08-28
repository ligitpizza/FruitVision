"""
Auth + admin coverage: database/auth_db.py persistence (create/verify/
deactivate/reactivate/delete, last-admin guards) and the Flask routes in
app.py that expose them (login, admin_required gating, invite, deactivate/
reactivate/delete, secret-key persistence).

Each Flask-level test loads app.py fresh via spec_from_file_location (same
approach as test_flask_surface_workflow.py) so the heavy predictor modules
never actually import, and redirects database.auth_db.DB_PATH at a throwaway
sqlite file so nothing here touches the real dev database.
"""
import importlib.util
import os
import shutil
import sqlite3
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

# Pre-import outside any patch.dict(sys.modules, ...) context: patch.dict
# restores sys.modules to its exact pre-context contents on exit, which
# deletes any modules that got imported *during* the context (including
# app.py's `import cv2` -> numpy chain). numpy's C extension can't be
# re-initialized after that in the same process ("cannot load module more
# than once per process"), which breaks every _load_app() call after the
# first. Importing here first keeps cv2/numpy permanently in sys.modules
# so exec_module() inside patch.dict just reuses the cached module.
import cv2  # noqa: F401
import numpy  # noqa: F401
import torch  # noqa: F401
from ultralytics import YOLO  # noqa: F401
from flask import Blueprint

import database.auth_db as auth_db


PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))


class DummyNotFruitError(Exception):
    pass


class DummyMultipleFruitImageError(ValueError):
    def __init__(self, fruit_breakdown):
        self.fruit_breakdown = dict(fruit_breakdown)
        super().__init__("Multiple fruits were detected.")


def _module(**attributes):
    module = types.ModuleType("stub")
    for name, value in attributes.items():
        setattr(module, name, value)
    return module


def _app_import_stubs():
    """Stand-ins for the heavy/DB-backed modules app.py imports at module
    load time -- mirrors tests/test_flask_surface_workflow.py so these two
    files can safely run in the same pytest session."""
    dummy_predict = lambda image, fruit_type: (
        "ripe", 0.9, (0, 0, 1, 1), image, {"ripe": 0.9, "unripe": 0.05, "rotten": 0.05}
    )
    predict_stub = _module(predict_ripeness=dummy_predict, NotAFruitError=DummyNotFruitError)

    stubs = {
        "member_apps.member_1_ab.m1_predict": predict_stub,
        "member_apps.member_2_bc.m2_predict": predict_stub,
        "member_apps.member_3_cd.m3_predict": predict_stub,
        "member_apps.member_4_da.m4_predict": predict_stub,
        "member_apps.merged_member_1_4.m14_predict": predict_stub,
        "member_apps.merged_member_1_4_v2.m14v2_predict": predict_stub,
        "member_apps.merged_member_1_4_v3.m14v3_predict": predict_stub,
        "pipeline.pure_yolo.yolo_cls_predict": predict_stub,
        "member_apps.predict_ensemble": _module(
            predict_ensemble=lambda image, fruit: ("ripe", 90.0, {}, (0, 0, 1, 1))
        ),
        "core_modules.pdf_report": _module(
            generate_pdf_report=lambda *a, **k: "report.pdf",
            generate_pdf_report_batch=lambda *a, **k: "report.pdf",
            generate_stock_report_pdf=lambda *a, **k: "stock_report.pdf",
        ),
        "core_modules.dashboard_charts": _module(
            generate_trend_chart=lambda *a, **k: None,
            generate_history_chart=lambda *a, **k: None,
            generate_fruit_breakdown_chart=lambda *a, **k: None,
            generate_confidence_trend_chart=lambda *a, **k: None,
        ),
        "core_modules.fruit_validation": _module(
            validate_selected_fruit=lambda image, fruit: {
                "selected_fruit": fruit, "detected_fruit": fruit,
                "confidence": 0.9, "validation_method": "test",
            },
            FruitValidationError=DummyNotFruitError,
        ),
        "core_modules.mixed_fruit_m14": _module(
            analyze_mixed_fruit_m14=lambda image: {},
            validate_single_fruit_image=lambda image: {
                "detected_count": 1,
                "fruit_breakdown": {"apple": 1},
                "validation_method": "test",
            },
            MultipleFruitImageError=DummyMultipleFruitImageError,
        ),
        # These four aren't stubbed by tests/test_flask_surface_workflow.py
        # (it exercises real image analysis), but nothing here does -- and
        # their real implementations transitively pull in skimage -> scipy
        # -> numpy.fft, whose C extensions can't survive being torn down
        # and reimported (see _get_app_module()'s docstring). Stubbing them
        # keeps that whole chain out of this file's import entirely, so it
        # can't collide with a *different* test file that does need it.
        "core_modules.blemish_analysis": _module(analyze_surface=lambda *a, **k: {}),
        "core_modules.marketability": _module(
            estimate_marketability=lambda *a, **k: {"status": "ready"},
            average_member_probabilities=lambda *a, **k: {},
            stock_eligible=lambda *a, **k: True,
        ),
        "core_modules.multi_fruit_detect": _module(
            supports_multi_fruit=lambda *a, **k: False,
            detect_fruit_boxes=lambda *a, **k: [],
        ),
        "core_modules.filter_photos": _module(
            FILTER_LABELS={}, ENSEMBLE_MEMBER_TO_MODEL_KEY={},
            filter_photos_single=lambda *a, **k: {},
            filter_photos_ensemble=lambda *a, **k: {},
            pop_member_cleaned_images=lambda *a, **k: {},
        ),
        "core_modules.model_lab": _module(
            MODEL_ORDER=[], FRUITS=[],
            get_model_summary=lambda *a, **k: {},
            format_size=lambda *a, **k: "",
            get_confusion_matrix=lambda *a, **k: None,
            get_per_fruit_recall=lambda *a, **k: {},
            get_yolo_training_history=lambda *a, **k: {},
        ),
        "database.history_db": _module(
            log_result=lambda **k: None,
            get_recent=lambda *a, **k: [],
            get_paginated=lambda *a, **k: ([], 0),
            get_all=lambda *a, **k: [],
            get_by_id=lambda *a, **k: None,
            update_result=lambda *a, **k: False,
            delete_result=lambda *a, **k: False,
            get_stats=lambda *a, **k: {},
            get_stats_since=lambda *a, **k: {},
            get_fruit_label_breakdown=lambda *a, **k: {},
        ),
        "database.stock_db": _module(
            log_stock_event=lambda **k: None,
            get_paginated=lambda *a, **k: ([], 0),
            get_by_id=lambda *a, **k: None,
            update_stock_event=lambda *a, **k: False,
            delete_stock_event=lambda *a, **k: False,
            get_summary=lambda *a, **k: {},
        ),
        "realtime.stream_routes": _module(realtime_bp=Blueprint("realtime_test_auth", __name__)),
    }
    return stubs


def _load_app():
    """Import app.py fresh as an isolated module object.

    Caller is responsible for pointing database.auth_db.DB_PATH at a
    throwaway file beforehand -- app.py's `from database import auth_db`
    binds the same module object, so the redirect is visible to every
    route it calls.
    """
    with patch.dict(sys.modules, _app_import_stubs()):
        spec = importlib.util.spec_from_file_location(
            "fruitvision_test_auth_app", os.path.join(PROJECT_ROOT, "app.py")
        )
        app_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(app_module)
    return app_module


_CACHED_APP_MODULE = None


def _get_app_module():
    """Load app.py at most once for this whole test file.

    patch.dict(sys.modules, ...) restores sys.modules to its exact
    pre-context contents on exit -- which deletes every module that got
    imported *during* the context, including transitive, lazily-resolved
    submodules pulled in by app.py's import chain (numpy.fft, scipy
    internals, skimage submodules, ...). Several of those wrap C
    extensions that refuse to initialize a second time in the same
    process, so calling _load_app() more than once reliably crashes on
    the second call. One shared module, reused across every test in this
    file, sidesteps that; each test still gets its own isolated database
    via _IsolatedDbTestCase redirecting auth_db.DB_PATH per test.
    """
    global _CACHED_APP_MODULE
    if _CACHED_APP_MODULE is None:
        _CACHED_APP_MODULE = _load_app()
    return _CACHED_APP_MODULE


class _IsolatedDbTestCase(unittest.TestCase):
    """Redirects database/auth_db.py at a throwaway sqlite file for the
    duration of the test, and restores the real path afterward so other
    test modules sharing this process aren't affected."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_dir, "test_auth.db")
        self._original_db_path = auth_db.DB_PATH
        auth_db.DB_PATH = self.db_path
        auth_db.init_db()
        self.addCleanup(self._restore)

    def _restore(self):
        auth_db.DB_PATH = self._original_db_path
        shutil.rmtree(self.tmp_dir, ignore_errors=True)


class AuthDbTests(_IsolatedDbTestCase):
    """database/auth_db.py persistence, exercised directly."""

    def test_seeds_default_admin_and_farmer_active(self):
        users = auth_db.list_users()
        by_email = {u["email"]: u for u in users}
        self.assertEqual(set(by_email), {"admin@fruitvision.local", "farmer@fruitvision.local"})
        self.assertEqual(by_email["admin@fruitvision.local"]["role"], "admin")
        self.assertTrue(all(u["is_active"] for u in users))

    def test_create_user_then_verify_login_succeeds(self):
        user_id = auth_db.create_user("Farmer Joe", "joe@example.test", "hunter22", role="farmer")
        user = auth_db.verify_login("joe@example.test", "hunter22")
        self.assertIsNotNone(user)
        self.assertEqual(user["id"], user_id)
        self.assertEqual(user["role"], "farmer")
        self.assertTrue(user["is_active"])

    def test_verify_login_rejects_wrong_password(self):
        auth_db.create_user("Farmer Joe", "joe@example.test", "hunter22")
        self.assertIsNone(auth_db.verify_login("joe@example.test", "wrong-password"))

    def test_create_user_rejects_duplicate_email(self):
        auth_db.create_user("First", "dup@example.test", "pw-one")
        with self.assertRaises(sqlite3.IntegrityError):
            auth_db.create_user("Second", "dup@example.test", "pw-two")

    def test_deactivated_user_cannot_login_even_with_correct_password(self):
        user_id = auth_db.create_user("Farmer Joe", "joe@example.test", "hunter22")
        self.assertTrue(auth_db.set_active(user_id, False))
        self.assertIsNone(auth_db.verify_login("joe@example.test", "hunter22"))

    def test_reactivated_user_can_login_again(self):
        user_id = auth_db.create_user("Farmer Joe", "joe@example.test", "hunter22")
        auth_db.set_active(user_id, False)
        auth_db.set_active(user_id, True)
        user = auth_db.verify_login("joe@example.test", "hunter22")
        self.assertIsNotNone(user)

    def test_active_admin_count_excludes_deactivated_admins_but_admin_count_does_not(self):
        admin = auth_db.get_user_by_email("admin@fruitvision.local")
        self.assertEqual(auth_db.active_admin_count(), 1)
        auth_db.set_active(admin["id"], False)
        self.assertEqual(auth_db.active_admin_count(), 0)
        self.assertEqual(auth_db.admin_count(), 1)

    def test_delete_user_removes_row(self):
        user_id = auth_db.create_user("Farmer Joe", "joe@example.test", "hunter22")
        self.assertTrue(auth_db.delete_user(user_id))
        self.assertIsNone(auth_db.get_user_by_id(user_id))


class AdminRouteTests(_IsolatedDbTestCase):
    """Flask-level: login, admin_required gating, invite/deactivate/
    reactivate/delete routes, and the mid-session deactivate kick-out."""

    def setUp(self):
        super().setUp()
        self.app_module = _get_app_module()
        self.client = self.app_module.app.test_client()
        self.admin = auth_db.get_user_by_email("admin@fruitvision.local")
        self.farmer = auth_db.get_user_by_email("farmer@fruitvision.local")

    def _login_as(self, client, user_id):
        with client.session_transaction() as sess:
            sess["user_id"] = user_id

    def test_login_with_correct_credentials_sets_session_and_redirects(self):
        response = self.client.post(
            "/login", data={"email": "admin@fruitvision.local", "password": "admin123"}
        )
        self.assertEqual(response.status_code, 302)
        with self.client.session_transaction() as sess:
            self.assertEqual(sess["user_id"], self.admin["id"])

    def test_login_with_wrong_password_does_not_set_session(self):
        response = self.client.post(
            "/login", data={"email": "admin@fruitvision.local", "password": "wrong"}
        )
        self.assertEqual(response.status_code, 200)  # re-renders login.html with a flash
        with self.client.session_transaction() as sess:
            self.assertNotIn("user_id", sess)

    def test_non_admin_is_redirected_away_from_admin_panel(self):
        self._login_as(self.client, self.farmer["id"])
        response = self.client.get("/admin")
        self.assertEqual(response.status_code, 302)
        self.assertNotIn("/admin", response.headers["Location"])

    def test_login_failure_flash_renders_with_the_red_error_class(self):
        response = self.client.post(
            "/login", data={"email": "admin@fruitvision.local", "password": "wrong"}
        )
        html = response.get_data(as_text=True)
        self.assertIn('class="flash error"', html)
        self.assertIn("Invalid email or password.", html)

    def test_scope_user_id_is_none_for_admin_and_own_id_for_farmer(self):
        with self.app_module.app.test_request_context():
            self.app_module.g.user = self.admin
            self.assertIsNone(self.app_module._scope_user_id())
            self.app_module.g.user = self.farmer
            self.assertEqual(self.app_module._scope_user_id(), self.farmer["id"])

    def test_owns_record_admin_always_true_farmer_only_own_or_unowned(self):
        with self.app_module.app.test_request_context():
            self.app_module.g.user = self.admin
            self.assertTrue(self.app_module._owns_record({"user_id": self.farmer["id"]}))

            self.app_module.g.user = self.farmer
            self.assertTrue(self.app_module._owns_record({"user_id": self.farmer["id"]}))
            self.assertTrue(self.app_module._owns_record({"user_id": None}))
            self.assertTrue(self.app_module._owns_record({}))  # legacy row, no user_id column value
            self.assertFalse(self.app_module._owns_record({"user_id": self.admin["id"]}))

    def test_admin_can_create_a_farmer_account_with_a_chosen_password(self):
        self._login_as(self.client, self.admin["id"])
        response = self.client.post(
            "/admin/users/invite",
            data={
                "name": "New Farmer", "email": "newfarmer@example.test",
                "password": "grower22", "role": "farmer",
            },
        )
        self.assertEqual(response.status_code, 302)
        created = auth_db.get_user_by_email("newfarmer@example.test")
        self.assertIsNotNone(created)
        self.assertEqual(created["role"], "farmer")
        self.assertTrue(created["is_active"])
        # the admin-chosen password is the real login credential, not a
        # discarded value replaced by an auto-generated temp password
        self.assertIsNotNone(auth_db.verify_login("newfarmer@example.test", "grower22"))

    def test_admin_create_user_rejects_a_password_under_8_characters(self):
        self._login_as(self.client, self.admin["id"])
        response = self.client.post(
            "/admin/users/invite",
            data={
                "name": "New Farmer", "email": "shortpw@example.test",
                "password": "short1", "role": "farmer",
            },
        )
        self.assertEqual(response.status_code, 302)
        self.assertIsNone(auth_db.get_user_by_email("shortpw@example.test"))

    def test_admin_can_deactivate_then_reactivate_a_user(self):
        self._login_as(self.client, self.admin["id"])

        deactivate_resp = self.client.post(f"/admin/users/{self.farmer['id']}/deactivate")
        self.assertEqual(deactivate_resp.status_code, 302)
        self.assertFalse(auth_db.get_user_by_id(self.farmer["id"])["is_active"])
        self.assertIsNone(auth_db.verify_login("farmer@fruitvision.local", "farmer123"))

        reactivate_resp = self.client.post(f"/admin/users/{self.farmer['id']}/reactivate")
        self.assertEqual(reactivate_resp.status_code, 302)
        self.assertTrue(auth_db.get_user_by_id(self.farmer["id"])["is_active"])
        self.assertIsNotNone(auth_db.verify_login("farmer@fruitvision.local", "farmer123"))

    def test_admin_cannot_deactivate_own_account(self):
        self._login_as(self.client, self.admin["id"])
        response = self.client.post(f"/admin/users/{self.admin['id']}/deactivate")
        self.assertEqual(response.status_code, 302)
        self.assertTrue(auth_db.get_user_by_id(self.admin["id"])["is_active"])

    def test_deactivating_one_of_two_admins_leaves_the_other_active(self):
        second_admin_id = auth_db.create_user(
            "Second Admin", "second@example.test", "pw-second", role="admin"
        )
        self._login_as(self.client, self.admin["id"])
        response = self.client.post(f"/admin/users/{second_admin_id}/deactivate")
        self.assertEqual(response.status_code, 302)
        self.assertFalse(auth_db.get_user_by_id(second_admin_id)["is_active"])
        self.assertEqual(auth_db.active_admin_count(), 1)

    def test_deactivating_a_logged_in_user_drops_their_session_on_next_request(self):
        farmer_client = self.app_module.app.test_client()
        self._login_as(farmer_client, self.farmer["id"])

        # confirm the session is live before deactivation
        pre_response = farmer_client.get("/profile")
        self.assertNotEqual(pre_response.status_code, 302)

        auth_db.set_active(self.farmer["id"], False)

        post_response = farmer_client.get("/profile")
        self.assertEqual(post_response.status_code, 302)
        self.assertIn("/login", post_response.headers["Location"])
        with farmer_client.session_transaction() as sess:
            self.assertNotIn("user_id", sess)

    def test_admin_delete_removes_the_user(self):
        self._login_as(self.client, self.admin["id"])
        response = self.client.post(f"/admin/users/{self.farmer['id']}/delete")
        self.assertEqual(response.status_code, 302)
        self.assertIsNone(auth_db.get_user_by_id(self.farmer["id"]))


class SecretKeyTests(unittest.TestCase):
    """app.py must not sign session cookies with a hardcoded string -- that
    would let anyone who reads the source forge a session for any user_id,
    including an admin, without ever knowing a password."""

    def test_secret_key_is_random_and_persists_across_loads(self):
        app_module = _get_app_module()
        self.assertNotEqual(app_module.app.secret_key, "fruitivision-dev-key")
        self.assertEqual(len(app_module.app.secret_key), 64)  # secrets.token_hex(32)
        # Re-run the loader function itself (not a full module reload --
        # see _get_app_module's docstring for why) to confirm it reads the
        # same persisted key back rather than generating a new one.
        self.assertEqual(app_module._load_or_create_secret_key(), app_module.app.secret_key)


if __name__ == "__main__":
    unittest.main()
