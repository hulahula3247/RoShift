import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from roshift.catalog import MODELS, asset, recipe, validate_catalog
from roshift.cli import main
from roshift.io import completed_stage, finish_stage, within
from roshift.plan import execute_plan, make_plan
from roshift.runtime.scope import is_decode, lm_head_mode, sparse_enabled
from roshift.evaluation.report import accuracy_table, speedup


class Contracts(unittest.TestCase):
    def test_all_recipes(self):
        self.assertEqual(validate_catalog(), {"models": 5, "recipes": 30, "valid": True})

    def test_all_accuracy_plans(self):
        for model in MODELS:
            for method in ("roshift", "teal", "larosa"):
                for allocation in ("uniform", "nonuniform"):
                    for s in (40, 50, 60):
                        plan = make_plan("accuracy", model, method, s, allocation)
                        self.assertEqual(len(plan["tasks"]), 11)
                        self.assertFalse(plan["end_to_end_ready"])

    def test_speed_unsupported_is_explicit(self):
        for method in ("larosa", "wina"):
            with self.assertRaises(ValueError):
                make_plan("speedup", "llama31", method, 50)
        with self.assertRaises(ValueError):
            make_plan("speedup", "llama31", "roshift", 50, "nonuniform")

    def test_dense_and_task_validation(self):
        with self.assertRaises(ValueError):
            make_plan("accuracy", "llama31", "dense", 50)
        with self.assertRaises(ValueError):
            make_plan("accuracy", "llama31", "roshift", 50, tasks=["unknown-benchmark"])
        self.assertEqual(make_plan("accuracy", "llama31", "dense", 0)["recipe"], None)

    def test_scope(self):
        class Cache:
            def __init__(self, n): self.n = n
            def get_seq_length(self): return self.n
        for method in ("teal", "larosa", "roshift"):
            self.assertTrue(sparse_enabled(method, "likelihood"))
            self.assertFalse(sparse_enabled(method, "generation", Cache(0)))
            self.assertFalse(sparse_enabled(method, "generation"))
            self.assertTrue(sparse_enabled(method, "generation", Cache(10)))
        self.assertFalse(sparse_enabled("dense", "likelihood"))
        self.assertEqual(lm_head_mode("teal", "generation"), "dense")
        self.assertEqual(lm_head_mode("roshift", "generation"), "calibrated_sparse")
        with self.assertRaises(TypeError): is_decode(object())

    def test_block_before_any_execution(self):
        with self.assertRaisesRegex(RuntimeError, "only previews a plan"):
            execute_plan(make_plan("accuracy", "llama31", "roshift", 50))

    def test_plan_does_not_download(self):
        with patch("roshift.cli.download_model", side_effect=AssertionError("network call")), contextlib.redirect_stdout(io.StringIO()) as out:
            main(["reproduce", "accuracy", "--model", "llama31", "--sparsity", "50"])
        self.assertFalse(json.loads(out.getvalue())["end_to_end_ready"])

    def test_receipt_tamper_and_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "data.bin").write_bytes(b"a")
            finish_stage(root, "stage", {"recipe": 1}, ["data.bin"])
            self.assertTrue(completed_stage(root, "stage", {"recipe": 1}))
            self.assertFalse(completed_stage(root, "stage", {"recipe": 2}))
            (root / "data.bin").write_bytes(b"b")
            self.assertFalse(completed_stage(root, "stage", {"recipe": 1}))

    def test_bounded_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError): within(temp, "../escape")
            with self.assertRaises(ValueError): within(temp, Path(temp).resolve())

    def test_table_and_no_partial_average(self):
        tasks = asset("benchmarks.json")
        dense = dict.fromkeys(tasks, 60.)
        ours = dict.fromkeys(tasks, 58.)
        text = accuracy_table({"RoShift": ours, "Dense": dense})
        self.assertTrue(text.splitlines()[2].startswith("| Dense |"))
        self.assertIn("58.00 (-2.00)", text)
        with self.assertRaises(ValueError): accuracy_table({"Dense": {"MMLU": 60.}})
        self.assertEqual(speedup(10., 5.)["speedup"], 2.)


if __name__ == "__main__":
    unittest.main()
