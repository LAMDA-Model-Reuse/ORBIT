"""Multimodal encoder contracts; no pretrained encoder downloads or paid API calls."""

import base64
from contextlib import contextmanager
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import torch

from main import main
from methods.ProfileRouter import ProfileRouter
from methods.RouteFM import RouteFMRouter
from utils.embedding import Embedder
from utils.qwen_vl_embedding import QwenVLEmbedder, _NoRedirect


class AlignedFusionTest(unittest.TestCase):
    def _encoder(self, **options):
        args = {"embeddings": {"text_model": "pair", "image_model": "pair", "out_dim": 2,
                               "batch_size": 2, "normalize": True, "training": False, **options}}
        text = lambda rows: torch.tensor([[3., 0.] for _ in rows])
        image = lambda rows: torch.tensor([[0., float(value)] for value in rows])
        with patch("utils.embedding.CreateTextEmbedder", return_value=(text, 2)), patch(
            "utils.embedding.CreateImageEmbedder", return_value=(image, 2)
        ):
            return Embedder(args)

    def test_default_concat_is_unchanged(self):
        encoder = self._encoder()
        result = encoder.run_embed(["same"], [4])
        np.testing.assert_allclose(result, [[.6, 0., 0., .8]])

    def test_aligned_weighted_unit_vectors_and_image_sensitivity(self):
        encoder = self._encoder(fusion="aligned_mean", text_weight=.75)
        result = encoder.run_embed(["same", "same"], [4, -4])
        expected = torch.tensor([[.75, .25], [.75, -.25]])
        expected /= expected.norm(dim=1, keepdim=True)
        torch.testing.assert_close(result, expected)
        self.assertFalse(torch.equal(result[0], result[1]))
        self.assertEqual(encoder.run_embed(["model description"]).shape, (1, 2))

    def test_rejects_random_projections_invalid_weights_and_mismatched_dimensions(self):
        for options in ({"training": True}, {"text_weight": 0.0}, {"text_weight": 1.0},
                        {"text_weight": float("nan")}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self._encoder(fusion="aligned_mean", **options)
        with patch("utils.embedding.CreateTextEmbedder", return_value=(Mock(), 2)), patch(
            "utils.embedding.CreateImageEmbedder", return_value=(Mock(), 3)
        ), self.assertRaisesRegex(ValueError, "equal dimension"):
            Embedder({"embeddings": {"text_model": "a", "image_model": "b", "fusion": "aligned_mean"}})

    def test_profile_multimodal_chooses_clip_without_mutating_caller(self):
        args = {"modality": "text+image", "embeddings": {"text_model": "longformer"},
                "description_path": "metadata.json"}
        def initialize(router, received):
            router.received = received
        with patch("methods.base.BaseRouter.__init__", initialize):
            router = ProfileRouter(args)
        em = router.received["embeddings"]
        self.assertEqual((em["text_model"], em["image_model"], em["out_dim"]), ("ViT-B/16", "ViT-B/16", 512))
        self.assertEqual(em["fusion"], "aligned_mean")
        self.assertEqual(args["embeddings"], {"text_model": "longformer"})

    def test_aligned_fusion_rejects_empty_image_signal(self):
        encoder = self._encoder(fusion="aligned_mean")
        with self.assertRaisesRegex(ValueError, "each modality"):
            encoder.run_embed(["question"], [0])

    def test_profile_rejects_unmatched_encoder_names(self):
        with self.assertRaisesRegex(ValueError, "matched aligned"):
            ProfileRouter({"modality": "text+image", "profile": {"multimodal_embeddings": {
                "text_model": "longformer", "image_model": "ViT-B/16"}}})

    def test_profile_reports_dimension_contract(self):
        router = ProfileRouter.__new__(ProfileRouter)
        router.model_profiles = np.ones((2, 8), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "query/profile dimensions"):
            router.predict(np.ones((3, 16), dtype=np.float32))


@contextmanager
def _embedding_service():
    received = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append((self.path, self.headers.get("Authorization"), payload))
            rows = []
            for index, item in enumerate(payload["input"]):
                raw = base64.b64decode(item["image"].split(",", 1)[1])
                with Image.open(io.BytesIO(raw)) as image:
                    rgb = np.asarray(image.convert("RGB"), dtype=np.float32).mean(axis=(0, 1))
                vector = np.zeros(4096, dtype=np.float32)
                vector[:3] = rgb + 1
                rows.append({"index": index, "embedding": vector.tolist()})
            body = json.dumps({"data": list(reversed(rows))}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=lambda: server.serve_forever(poll_interval=.01), daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", received
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


class QwenJointTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.paths = [self.root / "red.image", self.root / "green.image"]
        for path, color in zip(self.paths, ("red", "green")):
            Image.new("RGB", (4, 4), color=color).save(path, format="PNG")

    def tearDown(self):
        self.temporary.cleanup()

    def _config(self, **qwen):
        return {"joint_model": "Qwen/Qwen3-VL-Embedding-8B", "out_dim": 4096,
                "normalize": True, "batch_size": 2, "qwen": qwen}

    def test_real_http_joint_payload_order_normalization_and_image_sensitivity(self):
        with _embedding_service() as (base_url, received), patch.dict(os.environ, {"ORBIT_TEST_KEY": "fixture-only"}):
            encoder = Embedder({"embeddings": self._config(backend="api", base_url=base_url, api_key_env="ORBIT_TEST_KEY")})
            result = encoder.run_embed(["same question"] * 2, self.paths)
        self.assertEqual(result.shape, (2, 4096))
        torch.testing.assert_close(result.norm(dim=1), torch.ones(2))
        self.assertGreater(result[0, 0], result[0, 1])
        self.assertGreater(result[1, 1], result[1, 0])
        self.assertEqual(received[0][0], "/v1/embeddings")
        self.assertEqual(received[0][1], "Bearer fixture-only")
        self.assertEqual(received[0][2]["input"][0]["text"], "same question")
        self.assertTrue(received[0][2]["input"][0]["image"].startswith("data:image/png;base64,"))

    def test_local_uses_official_text_image_input_interface(self):
        local = self.root / "local-model"
        local.mkdir()
        model = Mock()
        model.encode.return_value = np.ones((2, 4096), dtype=np.float32)
        with patch("sentence_transformers.SentenceTransformer", return_value=model) as factory:
            encoder = QwenVLEmbedder(self._config(backend="local", model_dir=str(local)))
            result = encoder(["question"] * 2, self.paths)
        self.assertEqual(result.shape, (2, 4096))
        self.assertTrue(factory.call_args.kwargs["local_files_only"])
        self.assertFalse(factory.call_args.kwargs["trust_remote_code"])
        self.assertEqual(factory.call_args.kwargs["model_kwargs"], {"torch_dtype": "auto"})
        sent = model.encode.call_args.args[0]
        self.assertEqual(sent[0], {"text": "question", "image": str(self.paths[0])})

    def test_missing_local_weights_does_not_download_or_fallback(self):
        with self.assertRaisesRegex(FileNotFoundError, "No substitute"):
            QwenVLEmbedder(self._config(backend="local", model_dir=str(self.root / "missing")))

    def test_api_rejects_invalid_indices_shapes_values_and_missing_images(self):
        with patch.dict(os.environ, {"ORBIT_TEST_KEY": "fixture-only"}):
            encoder = QwenVLEmbedder(self._config(backend="api", base_url="http://localhost/v1", api_key_env="ORBIT_TEST_KEY"))
        responses = [
            {"data": [{"index": 0, "embedding": [1.] * 4096}] * 2},
            {"data": [{"index": True, "embedding": [1.] * 4096}]},
            {"data": [{"index": 0, "embedding": [1.] * 8}]},
            {"data": [{"index": 0, "embedding": [0.] * 4096}]},
            {"data": [{"index": 0, "embedding": [float("nan")] * 4096}]},
        ]
        for payload in responses:
            with self.subTest(payload_length=len(payload["data"])), patch.dict(os.environ, {"ORBIT_TEST_KEY": "fixture-only"}), patch.object(
                encoder.opener, "open", return_value=io.BytesIO(json.dumps(payload).encode())
            ), self.assertRaises(ValueError):
                encoder(["question"], [self.paths[0]])
        with self.assertRaisesRegex(ValueError, "local files"):
            encoder(["question"], ["https://example.org/image.png"])
        with self.assertRaisesRegex(ValueError, "batch sizes"):
            encoder(["question"], self.paths)

    def test_joint_encoder_rejects_separate_encoders_projections_and_dimension_changes(self):
        for extra in ({"text_model": "ViT-B/16"}, {"training": True}, {"out_dim": 512}):
            with self.subTest(extra=extra), patch("utils.embedding.JOINT_EMBEDDERS.create", return_value=(Mock(), 4096)), self.assertRaises(ValueError):
                Embedder({"embeddings": {**self._config(), **extra}})

    def test_routefm_qwen_retains_image_modality_and_requires_correct_encoder(self):
        args = {"modality": "text+image", "routefm": {"encoder": "qwen"}, "embeddings": self._config()}
        with patch("methods.base.BaseRouter.__init__", lambda router, config: setattr(router, "args", config)):
            router = RouteFMRouter(args)
        self.assertEqual(router.args["modality"], "text+image")
        for extra in ({"joint_model": "other"}, {"out_dim": 512}, {"training": True}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                RouteFMRouter({**args, "embeddings": {**args["embeddings"], **extra}})

    def test_api_security_and_no_redirect(self):
        for base_url in ("", "http://remote.example/v1", "https://user:pass@example.org/v1", "https://example.org/v1?key=secret"):
            with self.subTest(base_url=base_url), self.assertRaises(ValueError):
                QwenVLEmbedder(self._config(backend="api", base_url=base_url))
        self.assertIsNone(_NoRedirect().redirect_request(None, None, 302, "", {}, "https://elsewhere.invalid"))

    def test_cli_override_selects_qwen_but_device_flag_still_wins(self):
        with patch("sys.argv", ["main.py", "--dataset", "VLRouterBench", "--method", "RouteFM", "--config",
                                "configs/examples/routefm_qwen.json", "--device", "cpu"]), patch("main.train") as train:
            main()
        config = train.call_args.args[0]
        self.assertEqual(config["routefm"]["encoder"], "qwen")
        self.assertEqual(config["modality"], "text+image")
        self.assertEqual(config["device"], "cpu")


if __name__ == "__main__":
    unittest.main()
