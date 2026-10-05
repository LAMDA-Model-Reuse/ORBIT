import io
import importlib
import json
import pickle
import stat
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from main import load_config
from methods.base import BaseRouter
from utils.data import RouterEvalLoader, safe_extract_tar, safe_extract_zip


class _ConcreteRouter(BaseRouter):
    def train(self):
        return None

    def predict(self, test_embedding):
        return test_embedding


class _RecordingEmbedder:
    def __init__(self):
        self.texts = None

    def run_embed(self, texts=None, images=None):
        self.texts = list(texts)
        return torch.arange(len(self.texts) * 2, dtype=torch.float32).reshape(-1, 2)


class RouterEvalIdentityTest(unittest.TestCase):
    GROUP_A_FILES = [
        "arc_router_dataset.pkl",
        "gsm8k_router_dataset.pkl",
        "harness_truthfulqa_mc_0_router_dataset.pkl",
        "hellaswag_router_dataset.pkl",
        "mmlu_router_dataset.pkl",
        "winogrande_router_dataset.pkl",
    ]

    def test_model_list_and_performance_columns_share_one_canonical_order(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            source_models = ["zeta-2b", "alpha-1b"]
            payload = {
                "split_index": {"train_indices": [0]},
                "prompt": {"train_prompt": ["question"]},
                "easy": {
                    "2": {
                        "pool": {
                            "model": source_models,
                            "data": {"train_score": np.asarray([[0.9, 0.1]])},
                        }
                    }
                },
            }
            for filename in self.GROUP_A_FILES:
                with (Path(temporary_directory) / filename).open("wb") as handle:
                    pickle.dump(payload, handle)

            loader = RouterEvalLoader(temporary_directory, {"seed": 0})
            model_list, dataframe = loader.process(temporary_directory)

            self.assertEqual(model_list, ["alpha-1b", "zeta-2b"])
            self.assertEqual(len(dataframe), len(self.GROUP_A_FILES))
            np.testing.assert_allclose(dataframe["model_0_performance"], 0.1)
            np.testing.assert_allclose(dataframe["model_1_performance"], 0.9)


class DescriptionAlignmentTest(unittest.TestCase):
    def _router(self, description_path, model_list):
        router = _ConcreteRouter.__new__(_ConcreteRouter)
        router.args = {"description_path": str(description_path), "modality": "text"}
        router.model_list = model_list
        router.embedder = _RecordingEmbedder()
        return router

    def test_descriptions_are_reordered_by_model_name(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "descriptions.json"
            path.write_text(
                json.dumps(
                    [
                        {"model_name": "second", "description": "description two"},
                        {"model_name": "first", "description": "description one"},
                    ]
                ),
                encoding="utf-8",
            )
            router = self._router(path, ["first", "second"])
            router._get_model_description()
            self.assertEqual(
                router.embedder.texts, ["description one", "description two"]
            )

    def test_missing_description_fails_loudly(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "descriptions.json"
            path.write_text(
                json.dumps([{"model_name": "first", "description": "available"}]),
                encoding="utf-8",
            )
            router = self._router(path, ["first", "missing"])
            with self.assertRaisesRegex(ValueError, "missing 1 model"):
                router._get_model_description()


class SafeArchiveTest(unittest.TestCase):
    def test_tar_rejects_parent_traversal_before_writing(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            archive_path = Path(temporary_directory) / "bad.tar.gz"
            output_path = Path(temporary_directory) / "output"
            with tarfile.open(archive_path, "w:gz") as archive:
                safe_info = tarfile.TarInfo("safe.txt")
                safe_content = b"safe"
                safe_info.size = len(safe_content)
                archive.addfile(safe_info, io.BytesIO(safe_content))
                info = tarfile.TarInfo("../escape.txt")
                content = b"escape"
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))

            with tarfile.open(archive_path, "r:gz") as archive:
                with self.assertRaisesRegex(ValueError, "Unsafe archive member"):
                    safe_extract_tar(archive, str(output_path))
            self.assertFalse((Path(temporary_directory) / "escape.txt").exists())
            self.assertFalse((output_path / "safe.txt").exists())

    def test_tar_rejects_links(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            archive_path = Path(temporary_directory) / "link.tar.gz"
            with tarfile.open(archive_path, "w:gz") as archive:
                info = tarfile.TarInfo("link")
                info.type = tarfile.SYMTYPE
                info.linkname = "target"
                archive.addfile(info)
            with tarfile.open(archive_path, "r:gz") as archive:
                with self.assertRaisesRegex(ValueError, "links and special files"):
                    safe_extract_tar(archive, str(Path(temporary_directory) / "output"))

    def test_zip_rejects_parent_traversal_and_symlinks(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            traversal_path = Path(temporary_directory) / "traversal.zip"
            with zipfile.ZipFile(traversal_path, "w") as archive:
                archive.writestr("../escape.txt", "escape")
            with zipfile.ZipFile(traversal_path) as archive:
                with self.assertRaisesRegex(ValueError, "Unsafe archive member"):
                    safe_extract_zip(archive, str(Path(temporary_directory) / "output"))

            symlink_path = Path(temporary_directory) / "symlink.zip"
            with zipfile.ZipFile(symlink_path, "w") as archive:
                info = zipfile.ZipInfo("link")
                info.create_system = 3
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                archive.writestr(info, "target")
            with zipfile.ZipFile(symlink_path) as archive:
                with self.assertRaisesRegex(ValueError, "symlink"):
                    safe_extract_zip(archive, str(Path(temporary_directory) / "output"))

    def test_zip_rejects_absolute_paths(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            archive_path = Path(temporary_directory) / "absolute.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("/absolute.txt", "escape")
            with zipfile.ZipFile(archive_path) as archive:
                with self.assertRaisesRegex(ValueError, "Unsafe archive member"):
                    safe_extract_zip(archive, str(Path(temporary_directory) / "output"))

    def test_valid_archives_extract_normally(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            target = Path(temporary_directory) / "output"
            tar_path = Path(temporary_directory) / "valid.tar.gz"
            with tarfile.open(tar_path, "w:gz") as archive:
                content = b"tar"
                info = tarfile.TarInfo("nested/tar.txt")
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))
            with tarfile.open(tar_path, "r:gz") as archive:
                safe_extract_tar(archive, str(target))

            zip_path = Path(temporary_directory) / "valid.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr("nested/zip.txt", "zip")
            with zipfile.ZipFile(zip_path) as archive:
                safe_extract_zip(archive, str(target))

            self.assertEqual((target / "nested/tar.txt").read_text(), "tar")
            self.assertEqual((target / "nested/zip.txt").read_text(), "zip")


class ConfigurationTest(unittest.TestCase):
    def test_method_names_and_device_override_are_case_insensitive(self):
        config = load_config("MIXINSTRUCT", "Oracle", device="cpu")
        self.assertEqual(config["method"], "oracle")
        self.assertEqual(config["device"], "cpu")

    def test_all_json_configs_parse_and_devices_are_centralized(self):
        config_root = Path("configs")
        json_paths = list(config_root.rglob("*.json"))
        self.assertGreater(len(json_paths), 0)
        for path in json_paths:
            with self.subTest(path=path):
                json.loads(path.read_text(encoding="utf-8"))

        for path in Path("configs/benchmarks").glob("*.json"):
            config = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(config["device"], "auto")
            description_path = config.get("description_path")
            if description_path:
                self.assertTrue(
                    Path(description_path).is_file(),
                    f"Missing description file configured by {path}: {description_path}",
                )
        for path in Path("configs/routers").glob("*.json"):
            config = json.loads(path.read_text(encoding="utf-8"))
            self.assertNotIn("device", config.get("training", {}))

    def test_unavailable_cuda_index_falls_back_to_cpu(self):
        with (
            patch("methods.base.torch.cuda.is_available", return_value=True),
            patch("methods.base.torch.cuda.device_count", return_value=1),
        ):
            self.assertEqual(str(BaseRouter._resolve_device("cuda:3")), "cpu")

    def test_router_training_and_embedding_share_resolved_device(self):
        args = {
            "device": "cpu",
            "training": {"device": "cuda:3"},
            "embeddings": {"device": "cuda:2"},
            "seed": 7,
        }
        with (
            patch("methods.base.download_dataset", return_value=(object(), object(), [])),
            patch("methods.base.Embedder", return_value=object()),
        ):
            router = _ConcreteRouter(args)
        self.assertEqual(str(router.device), "cpu")
        self.assertEqual(router.args["device"], "cpu")
        self.assertEqual(router.args["training"]["device"], "cpu")
        self.assertEqual(router.args["embeddings"]["device"], "cpu")


class DataGeneratorImportTest(unittest.TestCase):
    def test_package_imports(self):
        from data_generator import LLMDataGenerator
        from data_generator.local_generator import LocalLLMGenerator

        self.assertEqual(LLMDataGenerator.__module__, "data_generator.generator")
        self.assertEqual(LocalLLMGenerator.__module__, "data_generator.local_generator")
        self.assertIsNotNone(
            importlib.import_module("data_generator.examples.01_quick_start").main
        )
        self.assertIsNotNone(
            importlib.import_module("data_generator.examples.02_local_model").main
        )


if __name__ == "__main__":
    unittest.main()
