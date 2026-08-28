"""
Per-user data isolation at the database layer: database/stock_db.py and
database/history_db.py must each scope get_paginated/get_summary/get_stats/
get_all to a single user_id when asked, and see everything when user_id is
None (the "admin sees all, farmer sees only their own" split app.py's
_scope_user_id() implements).

Pure DB-layer tests, deliberately not going through Flask/app.py -- app.py
pulls in cv2/torch/ultralytics/skimage/scipy, and those C extensions can't
be re-initialized once torn down mid-process (see tests/test_auth.py's
_load_app() docstring), so keeping this file Flask-free avoids that
fragility entirely for what is fundamentally a SQL WHERE-clause question.
"""
import os
import shutil
import tempfile
import unittest

import database as _database_pkg
import database.history_db as history_db
import database.stock_db as stock_db

# `import database.stock_db` (above) sets `stock_db`/`history_db` as
# permanent attributes on the `database` package object -- normal Python
# import behavior, but pytest imports every test file during collection,
# before any test actually runs, so this happens regardless of test order.
# A *different* test file (test_flask_surface_workflow.py) stubs out
# `database.stock_db`/`database.history_db` via
# unittest.mock.patch.dict(sys.modules, ...) so it can run without a real
# database; `from database import stock_db` resolves via getattr() on the
# already-imported `database` package first, so if that attribute is
# already sitting there for real (from this file having been collected),
# the stub in sys.modules never gets consulted and that file's assertions
# about what got logged silently fail. Deleting the attributes right after
# import (not waiting for teardown -- collection already happened) keeps
# that file's isolation working no matter which order pytest collects or
# runs test files in; sys.modules still caches the real modules for our
# own `history_db`/`stock_db` names above, which are unaffected.
for _name in ("history_db", "stock_db"):
    if hasattr(_database_pkg, _name):
        delattr(_database_pkg, _name)


class _IsolatedDbTestCase(unittest.TestCase):
    module = None  # set by subclass

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_dir, "test_isolation.db")
        self._original_db_path = self.module.DB_PATH
        self.module.DB_PATH = self.db_path
        self.module.init_db()
        self.addCleanup(self._restore)

    def _restore(self):
        self.module.DB_PATH = self._original_db_path
        shutil.rmtree(self.tmp_dir, ignore_errors=True)


class StockDbIsolationTests(_IsolatedDbTestCase):
    module = stock_db

    def test_new_user_sees_no_stock_events(self):
        stock_db.log_stock_event(fruit="apple", label="ripe", quantity=5, source="manual", user_id=1)
        rows, total = stock_db.get_paginated(user_id=2)
        self.assertEqual(rows, [])
        self.assertEqual(total, 0)

    def test_get_paginated_scopes_to_the_owning_user(self):
        stock_db.log_stock_event(fruit="apple", label="ripe", quantity=5, source="manual", user_id=1)
        stock_db.log_stock_event(fruit="banana", label="unripe", quantity=3, source="manual", user_id=2)

        farmer1_rows, farmer1_total = stock_db.get_paginated(user_id=1)
        self.assertEqual(farmer1_total, 1)
        self.assertEqual(farmer1_rows[0]["fruit"], "apple")

        farmer2_rows, farmer2_total = stock_db.get_paginated(user_id=2)
        self.assertEqual(farmer2_total, 1)
        self.assertEqual(farmer2_rows[0]["fruit"], "banana")

    def test_get_paginated_with_no_user_id_returns_everyone(self):
        stock_db.log_stock_event(fruit="apple", label="ripe", quantity=5, source="manual", user_id=1)
        stock_db.log_stock_event(fruit="banana", label="unripe", quantity=3, source="manual", user_id=2)
        rows, total = stock_db.get_paginated(user_id=None)
        self.assertEqual(total, 2)

    def test_get_summary_scopes_totals_to_the_owning_user(self):
        stock_db.log_stock_event(fruit="apple", label="ripe", quantity=5, source="manual", user_id=1)
        stock_db.log_stock_event(fruit="apple", label="ripe", quantity=10, source="manual", user_id=2)

        farmer1_summary = stock_db.get_summary(user_id=1)
        self.assertEqual(farmer1_summary["grand_total"], 5)

        farmer2_summary = stock_db.get_summary(user_id=2)
        self.assertEqual(farmer2_summary["grand_total"], 10)

        everyone_summary = stock_db.get_summary(user_id=None)
        self.assertEqual(everyone_summary["grand_total"], 15)


class HistoryDbIsolationTests(_IsolatedDbTestCase):
    module = history_db

    def test_new_user_sees_no_history(self):
        history_db.log_result(member="ensemble_ab", fruit="apple", label="ripe", confidence=92.0, user_id=1)
        rows, total = history_db.get_paginated(user_id=2)
        self.assertEqual(rows, [])
        self.assertEqual(total, 0)

    def test_get_paginated_scopes_to_the_owning_user(self):
        history_db.log_result(member="ensemble_ab", fruit="apple", label="ripe", confidence=92.0, user_id=1)
        history_db.log_result(member="ensemble_ab", fruit="banana", label="unripe", confidence=80.0, user_id=2)

        farmer1_rows, farmer1_total = history_db.get_paginated(user_id=1)
        self.assertEqual(farmer1_total, 1)
        self.assertEqual(farmer1_rows[0]["fruit"], "apple")

        farmer2_rows, farmer2_total = history_db.get_paginated(user_id=2)
        self.assertEqual(farmer2_total, 1)
        self.assertEqual(farmer2_rows[0]["fruit"], "banana")

    def test_get_all_with_no_user_id_returns_everyone(self):
        history_db.log_result(member="ensemble_ab", fruit="apple", label="ripe", confidence=92.0, user_id=1)
        history_db.log_result(member="ensemble_ab", fruit="banana", label="unripe", confidence=80.0, user_id=2)
        self.assertEqual(len(history_db.get_all(user_id=None)), 2)
        self.assertEqual(len(history_db.get_all(user_id=1)), 1)


if __name__ == "__main__":
    unittest.main()
