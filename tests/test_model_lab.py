import csv
import json
import os
import tempfile
import unittest

from core_modules import model_lab


def _write_report(training_dir, model_key, fruit, accuracy, per_class, confusion_matrix, classes):
    model_dir = os.path.join(training_dir, model_key)
    os.makedirs(model_dir, exist_ok=True)
    payload = {
        "fruit": fruit,
        "classes": classes,
        "confusion_matrix": confusion_matrix,
        "accuracy": accuracy,
        "per_class": per_class,
    }
    with open(os.path.join(model_dir, f"{fruit}_classification_report.json"), "w") as f:
        json.dump(payload, f)


class ModelLabTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.original_training_dir = model_lab.TRAINING_DIR
        self.original_trained_models_dir = model_lab.TRAINED_MODELS_DIR
        self.original_yolo_runs_dir = model_lab.YOLO_RUNS_DIR
        model_lab.TRAINING_DIR = os.path.join(self.temp_dir.name, "training")
        model_lab.TRAINED_MODELS_DIR = os.path.join(self.temp_dir.name, "trained_models")
        model_lab.YOLO_RUNS_DIR = os.path.join(self.temp_dir.name, "yolo_runs")
        os.makedirs(model_lab.TRAINING_DIR)
        os.makedirs(model_lab.TRAINED_MODELS_DIR)
        os.makedirs(model_lab.YOLO_RUNS_DIR)
        self.addCleanup(self._restore_dirs)

    def _restore_dirs(self):
        model_lab.TRAINING_DIR = self.original_training_dir
        model_lab.TRAINED_MODELS_DIR = self.original_trained_models_dir
        model_lab.YOLO_RUNS_DIR = self.original_yolo_runs_dir

    def test_model_with_no_report_has_no_data(self):
        summary = model_lab.get_model_summary("ab")
        self.assertFalse(summary["has_data"])
        self.assertIsNone(summary["accuracy"])
        self.assertEqual(summary["per_fruit"], {})

    def test_derives_macro_f1_and_balanced_accuracy_from_per_class_metrics(self):
        # A perfectly separable 2-class-only report (support-equal) makes
        # the expected macro-F1/balanced-accuracy easy to hand-verify.
        per_class = {
            "ripe": {"precision": 1.0, "recall": 0.8, "f1_score": 0.8889, "support": 10},
            "unripe": {"precision": 0.9, "recall": 1.0, "f1_score": 0.9474, "support": 10},
            "rotten": {"precision": 1.0, "recall": 1.0, "f1_score": 1.0, "support": 10},
        }
        _write_report(
            model_lab.TRAINING_DIR, "ab", "apple", accuracy=0.9,
            per_class=per_class,
            confusion_matrix=[[8, 2, 0], [0, 10, 0], [0, 0, 10]],
            classes=["ripe", "unripe", "rotten"],
        )
        summary = model_lab.get_model_summary("ab")
        self.assertTrue(summary["has_data"])
        self.assertAlmostEqual(summary["accuracy"], 0.9)
        expected_macro_f1 = (0.8889 + 0.9474 + 1.0) / 3
        expected_balanced_acc = (0.8 + 1.0 + 1.0) / 3
        self.assertAlmostEqual(summary["macro_f1"], expected_macro_f1, places=4)
        self.assertAlmostEqual(summary["balanced_accuracy"], expected_balanced_acc, places=4)
        self.assertAlmostEqual(summary["lowest_recall"], 0.8)

    def test_overall_row_is_support_weighted_across_fruits(self):
        per_class_small = {
            "ripe": {"precision": 1.0, "recall": 1.0, "f1_score": 1.0, "support": 1},
            "unripe": {"precision": 1.0, "recall": 1.0, "f1_score": 1.0, "support": 1},
            "rotten": {"precision": 1.0, "recall": 1.0, "f1_score": 1.0, "support": 1},
        }
        per_class_large = {
            "ripe": {"precision": 0.5, "recall": 0.5, "f1_score": 0.5, "support": 100},
            "unripe": {"precision": 0.5, "recall": 0.5, "f1_score": 0.5, "support": 100},
            "rotten": {"precision": 0.5, "recall": 0.5, "f1_score": 0.5, "support": 100},
        }
        # apple: perfect accuracy but tiny support; banana: 50% accuracy but
        # huge support -- the overall row should land close to banana's
        # numbers, not a naive unweighted average of the two fruits.
        _write_report(
            model_lab.TRAINING_DIR, "ab", "apple", accuracy=1.0,
            per_class=per_class_small,
            confusion_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            classes=["ripe", "unripe", "rotten"],
        )
        _write_report(
            model_lab.TRAINING_DIR, "ab", "banana", accuracy=0.5,
            per_class=per_class_large,
            confusion_matrix=[[50, 25, 25], [25, 50, 25], [25, 25, 50]],
            classes=["ripe", "unripe", "rotten"],
        )
        summary = model_lab.get_model_summary("ab")
        self.assertLess(abs(summary["accuracy"] - 0.5), 0.02)
        self.assertEqual(summary["lowest_recall"], 0.5)

    def test_model_dir_size_resolves_merged_1_4_to_its_m14_folder(self):
        # trained_models/'s own subdirectory naming differs from every other
        # convention in this codebase -- merged_1_4 is saved under "m14".
        m14_dir = os.path.join(model_lab.TRAINED_MODELS_DIR, "m14")
        os.makedirs(m14_dir)
        with open(os.path.join(m14_dir, "apple_m14.pkl"), "wb") as f:
            f.write(b"x" * 1234)
        summary = model_lab.get_model_summary("merged_1_4")
        self.assertEqual(summary["size_bytes"], 1234)

    def test_get_confusion_matrix_returns_none_when_missing(self):
        self.assertIsNone(model_lab.get_confusion_matrix("ab", "apple"))

    def test_get_confusion_matrix_returns_stored_matrix(self):
        _write_report(
            model_lab.TRAINING_DIR, "cd", "orange", accuracy=0.7,
            per_class={"ripe": {"precision": 1, "recall": 1, "f1_score": 1, "support": 1}},
            confusion_matrix=[[1, 2], [3, 4]],
            classes=["ripe", "rotten"],
        )
        cm = model_lab.get_confusion_matrix("cd", "orange")
        self.assertEqual(cm["classes"], ["ripe", "rotten"])
        self.assertEqual(cm["matrix"], [[1, 2], [3, 4]])

    def test_yolo_training_history_missing_returns_none(self):
        self.assertIsNone(model_lab.get_yolo_training_history("apple"))

    def test_yolo_training_history_parses_results_csv(self):
        run_dir = os.path.join(model_lab.YOLO_RUNS_DIR, "apple_cls")
        os.makedirs(run_dir)
        with open(os.path.join(run_dir, "results.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "time", "train/loss", "metrics/accuracy_top1", "metrics/accuracy_top5", "val/loss", "lr/pg0", "lr/pg1", "lr/pg2"])
            writer.writerow([1, 16.0, 0.386, 0.9667, 1.0, 0.102, 0.0004, 0.0004, 0.0004])
            writer.writerow([2, 30.0, 0.145, 0.9570, 1.0, 0.115, 0.0009, 0.0009, 0.0009])
        history = model_lab.get_yolo_training_history("apple")
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["epoch"], 1)
        self.assertAlmostEqual(history[0]["train_loss"], 0.386)
        self.assertAlmostEqual(history[1]["accuracy"], 0.9570)


if __name__ == "__main__":
    unittest.main()
