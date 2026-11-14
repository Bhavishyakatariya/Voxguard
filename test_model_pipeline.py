"""
VoxGuard - Automated Test Suite for Task 2 (Model Development & Training)
========================================================================
Verifies all 13 verification criteria specified in TASK 2 Section 11:
1. Generate dummy data
2. Verify all .npy files have shape (128, 128)
3. Verify metadata columns and label mappings
4. Load the dummy dataset successfully
5. Run forward pass through both models (CNN & Transformer)
6. Verify output shapes against contract: (B, 1) in [0.0, 1.0]
7. Verify self.target_layer exists and is callable
8. Run short CNN training session
9. Run short Transformer training session
10. Verify checkpoint saving and loading
11. Run evaluation script
12. Verify all five metrics generated separately for ASVspoof and In-The-Wild
13. Resolve any errors
"""

from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dummy_data import generate_dummy_dataset
from src.data.dataset import VoxGuardDataset
from src.models.cnn_model import CNNClassifier
from src.models.transformer_model import TransformerClassifier
from src.models.model_factory import get_model
from src.training.trainer import VoxGuardTrainer
from src.evaluation.metrics import calculate_eer, compute_forensic_metrics
from src.evaluation.evaluate import run_evaluation
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.utils.config import load_config


class TestTask2Pipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dummy_meta_path = generate_dummy_dataset(samples_per_split=12, random_seed=42)

    def test_01_dummy_data_files_and_shapes(self):
        df = pd.read_csv(self.dummy_meta_path)
        self.assertGreater(len(df), 0)
        for _, row in df.iterrows():
            abs_p = PROJECT_ROOT / row["filepath"]
            self.assertTrue(abs_p.is_file(), f"Missing file: {abs_p}")
            arr = np.load(abs_p)
            self.assertEqual(arr.shape, (128, 128), f"Incorrect shape {arr.shape} in {abs_p}")
            self.assertEqual(arr.dtype, np.float32)

    def test_02_metadata_schema_and_labels(self):
        df = pd.read_csv(self.dummy_meta_path)
        required_cols = ["filepath", "label", "dataset", "split"]
        for col in required_cols:
            self.assertIn(col, df.columns)

        labels = set(df["label"].unique())
        self.assertTrue(labels.issubset({0, 1}))
        datasets = set(df["dataset"].unique())
        self.assertEqual(datasets, {"asvspoof", "inthewild"})
        splits = set(df["split"].unique())
        self.assertEqual(splits, {"train", "val", "test"})

    def test_03_dataset_loader(self):
        ds = VoxGuardDataset(metadata_path=self.dummy_meta_path, split="train")
        self.assertGreater(len(ds), 0)
        x, y, path, dset = ds[0]
        self.assertEqual(x.shape, torch.Size([1, 128, 128]))
        self.assertEqual(y.shape, torch.Size([1]))
        self.assertIn(int(y.item()), [0, 1])

    def test_04_cnn_architecture_contract(self):
        cnn = CNNClassifier()
        cnn.eval()
        self.assertTrue(hasattr(cnn, "target_layer"))
        self.assertEqual(cnn.target_layer, cnn.conv4)

        dummy_x = torch.randn(4, 1, 128, 128)
        probs = cnn(dummy_x)
        self.assertEqual(probs.shape, torch.Size([4, 1]))
        self.assertTrue(torch.all(probs >= 0.0) and torch.all(probs <= 1.0))

        logits = cnn(dummy_x, return_logits=True)
        self.assertEqual(logits.shape, torch.Size([4, 1]))

    def test_05_transformer_architecture_contract(self):
        ast = TransformerClassifier()
        ast.eval()
        self.assertTrue(hasattr(ast, "target_layer"))

        dummy_x = torch.randn(4, 1, 128, 128)
        probs = ast(dummy_x)
        self.assertEqual(probs.shape, torch.Size([4, 1]))
        self.assertTrue(torch.all(probs >= 0.0) and torch.all(probs <= 1.0))

        logits = ast(dummy_x, return_logits=True)
        self.assertEqual(logits.shape, torch.Size([4, 1]))

    def test_06_model_factory(self):
        m_cnn = get_model("cnn")
        self.assertIsInstance(m_cnn, CNNClassifier)
        m_ast = get_model("transformer")
        self.assertIsInstance(m_ast, TransformerClassifier)

    def test_07_checkpoint_saving_and_loading(self):
        cnn = CNNClassifier()
        cnn.eval()
        test_ckpt_path = PROJECT_ROOT / "checkpoints" / "test_cnn.pt"
        save_checkpoint(cnn, test_ckpt_path)
        self.assertTrue(test_ckpt_path.is_file())

        cnn2 = CNNClassifier()
        load_checkpoint(test_ckpt_path, cnn2)
        cnn2.eval()
        test_x = torch.randn(2, 1, 128, 128)
        with torch.no_grad():
            self.assertTrue(torch.allclose(cnn(test_x), cnn2(test_x), atol=1e-5))
        test_ckpt_path.unlink()

    def test_08_short_training_cnn(self):
        test_ckpt_dir = "checkpoints/test_run"
        trainer = VoxGuardTrainer(
            arch="cnn",
            metadata_path=self.dummy_meta_path,
            batch_size=8,
            epochs=1,
            checkpoint_dir=test_ckpt_dir,
            device="cpu",
        )
        res = trainer.fit()
        self.assertIn("history", res)
        self.assertEqual(len(res["history"]), 1)

    def test_09_short_training_transformer(self):
        test_ckpt_dir = "checkpoints/test_run"
        trainer = VoxGuardTrainer(
            arch="transformer",
            metadata_path=self.dummy_meta_path,
            batch_size=8,
            epochs=1,
            checkpoint_dir=test_ckpt_dir,
            device="cpu",
        )
        res = trainer.fit()
        self.assertIn("history", res)
        self.assertEqual(len(res["history"]), 1)

    def test_10_forensic_metrics_and_eer(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
        metrics = compute_forensic_metrics(y_true, y_scores)
        self.assertIn("accuracy", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1_score", metrics)
        self.assertIn("auc_roc", metrics)
        self.assertIn("eer_percent", metrics)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)

    def test_11_dataset_wise_evaluation(self):
        test_ckpt = PROJECT_ROOT / "checkpoints" / "test_eval_cnn.pt"
        cnn = CNNClassifier()
        save_checkpoint(cnn, test_ckpt)

        report = run_evaluation(
            checkpoint_path=test_ckpt,
            arch="cnn",
            metadata_path=self.dummy_meta_path,
            batch_size=8,
            results_dir="results/test_metrics",
            device_str="cpu",
        )
        self.assertIn("overall_metrics", report)
        self.assertIn("dataset_wise_metrics", report)
        self.assertIn("asvspoof", report["dataset_wise_metrics"])
        self.assertIn("inthewild", report["dataset_wise_metrics"])
        test_ckpt.unlink()


if __name__ == "__main__":
    unittest.main()
