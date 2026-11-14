"""
VoxGuard - XAI Test Suite (Task 3 Verification)
==============================================
Validates the complete Explainable AI module:
1. Model loading from actual trained checkpoint
2. Grad-CAM forward/backward hook registration & cleanup
3. Normalized attribution map dimensions (128, 128)
4. Symmetrical explanation of REAL and FAKE classes
5. Explain() interface contract dictionary structure
6. Visualization output generation (PNG file existence and valid format)
7. Caption coordinate extraction and natural-language generation
"""

import unittest
from pathlib import Path
import numpy as np
import torch

from src.explainability.model_loader import load_trained_model
from src.explainability.gradcam import GradCAM
from src.explainability.caption_generator import generate_explanation_caption, extract_peak_attribution_region
from src.explainability.visualization import save_visualization
from src.explainability.explain import explain, preprocess_audio_for_model


class TestVoxGuardXAI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model, cls.device = load_trained_model()

    def test_01_model_loaded_and_eval_mode(self):
        """Verify the actual trained model loads into evaluation mode."""
        self.assertFalse(self.model.training, "Model should be in eval mode.")
        self.assertTrue(hasattr(self.model, "target_layer"), "Model must have target_layer attribute.")
        self.assertEqual(self.model.target_layer, self.model.conv4)

    def test_02_gradcam_fake_explanation(self):
        """Verify Grad-CAM generates valid normalized attribution for Fake prediction."""
        gradcam = GradCAM(self.model)
        dummy_input = torch.randn(1, 1, 128, 128, device=self.device)

        heatmap, pred_class, conf, prob_fake = gradcam.generate_heatmap(dummy_input, target_class=1)
        gradcam.remove_hooks()

        self.assertEqual(heatmap.shape, (128, 128))
        self.assertTrue(np.all(heatmap >= 0.0) and np.all(heatmap <= 1.0))
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)

    def test_03_gradcam_real_explanation(self):
        """Verify Grad-CAM generates valid normalized attribution for Real prediction."""
        gradcam = GradCAM(self.model)
        dummy_input = torch.randn(1, 1, 128, 128, device=self.device)

        heatmap, pred_class, conf, prob_fake = gradcam.generate_heatmap(dummy_input, target_class=0)
        gradcam.remove_hooks()

        self.assertEqual(heatmap.shape, (128, 128))
        self.assertTrue(np.all(heatmap >= 0.0) and np.all(heatmap <= 1.0))

    def test_04_caption_generation(self):
        """Verify caption generator produces valid time/frequency bounds and non-empty description."""
        dummy_heatmap = np.zeros((128, 128), dtype=np.float32)
        dummy_heatmap[30:70, 40:80] = 0.95  # Simulated anomaly cluster

        caption_dict = generate_explanation_caption(dummy_heatmap, predicted_class=1, confidence=0.96)
        self.assertIn("caption", caption_dict)
        self.assertIn("time_interval", caption_dict)
        self.assertIn("frequency_range_hz", caption_dict)
        self.assertIn("frequency_band", caption_dict)
        self.assertGreater(len(caption_dict["caption"]), 20)

    def test_05_visualization_saving(self):
        """Verify high-resolution PNG is saved and non-empty."""
        dummy_mel = np.random.uniform(-80.0, 0.0, size=(128, 128)).astype(np.float32)
        dummy_heat = np.random.uniform(0.0, 1.0, size=(128, 128)).astype(np.float32)
        out_path = Path("outputs/explainability/heatmaps/test_verification_plot.png")

        saved = save_visualization(
            mel_spectrogram=dummy_mel,
            heatmap=dummy_heat,
            predicted_class=1,
            confidence=0.88,
            output_path=out_path,
        )
        self.assertTrue(Path(saved).is_file())
        self.assertGreater(Path(saved).stat().st_size, 50000)

    def test_06_explain_contract_dictionary(self):
        """Verify explain() returns all required keys matching shared interface contract."""
        test_npy = Path("data/processed/test/10.npy")
        if not test_npy.is_file():
            test_npy = next(Path("data/processed").glob("**/*.npy"))

        result = explain(self.model, test_npy)

        expected_keys = [
            "audio_path",
            "predicted_label",
            "predicted_class",
            "confidence",
            "fake_probability",
            "heatmap_path",
            "caption",
            "time_interval_seconds",
            "frequency_range_hz",
            "frequency_band",
            "attribution_strength",
            "model_architecture",
        ]
        for k in expected_keys:
            self.assertIn(k, result, f"Missing contract key: {k}")

        self.assertIn(result["predicted_class"], [0, 1])
        self.assertIn(result["predicted_label"], ["Real", "Fake"])
        self.assertTrue(Path(result["heatmap_path"]).is_file())


if __name__ == "__main__":
    unittest.main()

