"""Prepared upstream export fixtures: no networks, pickles or position joins."""

import base64
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

from main import load_config
from utils.data import BaseDatasetLoader, VLRouterBenchLoader, download_dataset, get_loader


ROOT = Path(__file__).resolve().parents[1]
VL_MODELS = ["Janus-Pro-1B", "Janus-Pro-7B"]


def write_vl_fixture(root):
    root = Path(root)
    for directory in ("BENCHMARKS/vqa_oe", "ORACLE/score", "SPLITS", "reports/token_statistics", "vlm_router_data/TSV_images"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    def encoded(color):
        buffer = io.BytesIO()
        Image.new("RGB", (20, 16), color).save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()
    red, blue = encoded("red"), encoded("blue")
    pd.DataFrame({"index": [10, 20, 30], "image": [red, "10", repr([red, blue])]}).to_csv(
        root / "vlm_router_data/TSV_images/RealWorldQA.tsv", sep="\t", index=False)
    samples, scores, inputs, outputs = [], [], [], []
    for i in range(14):
        sid = f"RealWorldQA/{i}"
        samples.append({"sample_id": sid, "dataset": "RealWorldQA", "prompt": f"vl visual question {i}",
                        "assets": [{"type": "image_tsv", "tsv_file": "/author/LMUData/RealWorldQA.tsv", "index": i % 3}],
                        "answer": "GOLD MUST NOT BECOME INPUT", "task_type": "vqa_oe"})
        inputs.append({"sample_id": sid, "text_tokens": 20+i, "image_tokens": 256, "total_input_tokens": 276+i})
        for j, model in enumerate(VL_MODELS):
            scores.append({"sample_id": sid, "model_id": model, "quality_raw": 0 if i == 0 else (i+j)%2, "quality": 1})
            outputs.append({"sample_id": sid, "model": model, "output_tokens": 5+3*j,
                            "is_actual": not (i == 4 and j == 1)})
    (root / "BENCHMARKS/vqa_oe/realworldqa_samples.jsonl").write_text(
        "".join(json.dumps(row)+"\n" for row in samples), encoding="utf-8")
    pd.DataFrame(scores).sample(frac=1, random_state=5).to_parquet(root / "ORACLE/score/realworldqa.parquet", index=False)
    pd.DataFrame(inputs).sample(frac=1, random_state=7).to_csv(root / "reports/token_statistics/token_stats_per_sample.csv", index=False)
    pd.DataFrame(outputs).sample(frac=1, random_state=9).to_csv(root / "reports/token_statistics/output_tokens_per_model.csv", index=False)
    for name, indices in (("train", range(8)), ("dev", range(8, 10)), ("test", range(10, 14))):
        (root / f"SPLITS/{name}.jsonl").write_text(
            "".join(json.dumps({"sample_id": f"RealWorldQA/{i}"})+"\n" for i in indices), encoding="utf-8")


def vl_args(root, modality="text", **options):
    args = load_config("VLRouterBench", "knn", device="cpu")
    args["modality"] = modality
    args["description_path"] = str(ROOT / args["description_path"])
    args["dataset"].update(dataset_dir=str(root), models=VL_MODELS,
                          benchmarks=["RealWorldQA"], **options)
    return args


class VLRouterBenchTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="orbit-vl-fixture-")
        self.root = Path(self.temporary.name)
        write_vl_fixture(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def _csv(self, name, edit):
        path = self.root / f"reports/token_statistics/{name}.csv"
        frame = pd.read_csv(path)
        edit(frame).to_csv(path, index=False)

    def test_registry_and_normal_loader_contract(self):
        self.assertIs(get_loader("VLRouterBench"), VLRouterBenchLoader)
        self.assertTrue(issubclass(VLRouterBenchLoader, BaseDatasetLoader))
        train, test, models = download_dataset(vl_args(self.root))
        self.assertEqual(models, VL_MODELS)
        self.assertEqual((len(train), len(test)), (8, 4))
        self.assertFalse(set(train.id) & set(test.id))
        self.assertNotIn("answer", train)
        self.assertNotIn("assets", train)
        self.assertTrue(train.model_0_cost.between(0, 1).all())
        for name in ("model_0_performance", "model_1_performance", "model_0_raw_cost", "model_1_output_is_actual"):
            self.assertIn(name, train)

    def test_named_alignment_frozen_costs_and_all_zero_queries(self):
        train, test, _ = download_dataset(vl_args(self.root))
        row = train.set_index("id").loc["RealWorldQA/5"]
        self.assertEqual((row.model_0_performance, row.model_1_performance), (1, 0))
        self.assertAlmostEqual(row.model_0_raw_cost, (281+5)*0.05/1e6)
        self.assertAlmostEqual(row.model_1_raw_cost, (281*0.18+8*0.25)/1e6)
        zero = train.set_index("id").loc["RealWorldQA/0"]
        self.assertEqual((zero.model_0_performance, zero.model_1_performance), (0, 0))
        self.assertFalse(train.set_index("id").loc["RealWorldQA/4"].model_1_output_is_actual)
        self.assertFalse(any("response" in c for c in train.columns))
        protocol = train.attrs["orbit_protocol"]
        self.assertEqual(protocol["excluded_dev_queries"], 2)
        self.assertEqual(protocol["estimated_output_records"], 1)
        self.assertIn("estimated", protocol["cost_scope"])
        self.assertTrue(all(len(item["sha256"]) == 64 for item in protocol["sources"].values()))

    def test_duplicate_score_output_input_identities_fail(self):
        for kind in ("score", "output", "input"):
            with self.subTest(kind=kind):
                write_vl_fixture(self.root)
                if kind == "score":
                    path = self.root / "ORACLE/score/realworldqa.parquet"
                    frame = pd.read_parquet(path)
                    pd.concat([frame, frame.iloc[:1]]).to_parquet(path, index=False)
                else:
                    name = "output_tokens_per_model" if kind == "output" else "token_stats_per_sample"
                    self._csv(name, lambda df: pd.concat([df, df.iloc[:1]]))
                with self.assertRaisesRegex(ValueError, "duplicate identities"):
                    download_dataset(vl_args(self.root))

    def test_missing_costs_error_or_explicit_common_support(self):
        self._csv("output_tokens_per_model", lambda df: df.iloc[1:])
        with self.assertRaisesRegex(ValueError, "missing/failed"):
            download_dataset(vl_args(self.root))
        train, test, _ = download_dataset(vl_args(self.root, missing_policy="drop-query"))
        self.assertEqual(train.attrs["orbit_protocol"]["coverage"][0]["dropped_queries"], 1)

    def test_invalid_tokens_and_actual_flags_fail(self):
        for column, value in (("output_tokens", -1), ("output_tokens", 1.5), ("output_tokens", np.inf), ("is_actual", "perhaps")):
            with self.subTest(column=column, value=value):
                write_vl_fixture(self.root)
                def edit(df):
                    df[column] = df[column].astype(object)
                    df.loc[0, column] = value
                    return df
                self._csv("output_tokens_per_model", edit)
                with self.assertRaises(ValueError):
                    download_dataset(vl_args(self.root))

    def test_input_total_mismatch_and_invalid_scores_fail(self):
        self._csv("token_stats_per_sample", lambda df: df.assign(total_input_tokens=df.total_input_tokens+1))
        with self.assertRaisesRegex(ValueError, "must equal"):
            download_dataset(vl_args(self.root))
        write_vl_fixture(self.root)
        path = self.root / "ORACLE/score/realworldqa.parquet"
        frame = pd.read_parquet(path)
        frame["quality_raw"] = frame.quality_raw.astype(float)
        frame.loc[0, "quality_raw"] = 1.5
        frame.to_parquet(path, index=False)
        with self.assertRaisesRegex(ValueError, "invalid scores"):
            download_dataset(vl_args(self.root))

    def test_estimated_output_policies(self):
        with self.assertRaisesRegex(ValueError, "Estimated output"):
            download_dataset(vl_args(self.root, estimated_output_policy="error"))
        train, test, _ = download_dataset(vl_args(self.root, estimated_output_policy="drop-query", missing_policy="drop-query"))
        self.assertNotIn("RealWorldQA/4", train.id.tolist()+test.id.tolist())

    def test_official_split_duplicates_and_missing_assignments_fail(self):
        path = self.root / "SPLITS/test.jsonl"
        original = path.read_text()
        path.write_text(original+json.dumps({"sample_id": "RealWorldQA/0"})+"\n")
        with self.assertRaisesRegex(ValueError, "Official splits"):
            download_dataset(vl_args(self.root))
        path.write_text("\n".join(original.splitlines()[1:])+"\n")
        with self.assertRaisesRegex(ValueError, "lack official"):
            download_dataset(vl_args(self.root))

    def test_official_overlap_with_dev_is_checked_before_subsampling(self):
        path = self.root / "BENCHMARKS/vqa_oe/realworldqa_samples.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        records[9]["prompt"] = records[0]["prompt"]
        records[9]["assets"] = records[0]["assets"]
        path.write_text("".join(json.dumps(r)+"\n" for r in records))
        with self.assertRaisesRegex(ValueError, "Identical router inputs"):
            download_dataset(vl_args(self.root, max_samples=2))

    def test_tsv_relocation_reference_and_multiframe_contact_sheet(self):
        train, test, _ = download_dataset(vl_args(self.root, modality="text+image", image_cell_size=32))
        rows = train.set_index("id")
        self.assertEqual(rows.loc["RealWorldQA/0"].image_path, rows.loc["RealWorldQA/1"].image_path)
        with Image.open(rows.loc["RealWorldQA/0"].image_path) as image:
            self.assertEqual(image.getpixel((0, 0)), (255, 0, 0))
        with Image.open(rows.loc["RealWorldQA/2"].image_path) as image:
            self.assertEqual(image.size, (64, 32))
        self.assertEqual(len(train.attrs["orbit_protocol"]["selected_image_sha256"]), 2)

    def test_text_only_identity_does_not_include_invisible_images(self):
        path = self.root / "BENCHMARKS/vqa_oe/realworldqa_samples.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        records[9]["prompt"] = records[0]["prompt"]
        records[9]["assets"][0]["index"] = 2  # Distinct visible image(s).
        path.write_text("".join(json.dumps(r)+"\n" for r in records))
        with self.assertRaisesRegex(ValueError, "Identical router inputs"):
            download_dataset(vl_args(self.root, modality="text"))
        train, test, _ = download_dataset(vl_args(self.root, modality="text", text_overlap_policy="drop-query"))
        self.assertNotIn("RealWorldQA/0", train.id.tolist()+test.id.tolist())
        self.assertEqual(train.attrs["orbit_protocol"]["text_overlap_dropped_queries"], 2)
        train, test, _ = download_dataset(vl_args(self.root, modality="text+image"))
        self.assertIn("RealWorldQA/0", train.id.tolist())
        train, test, _ = download_dataset(vl_args(self.root, modality="text+image", text_overlap_policy="drop-query"))
        self.assertNotIn("RealWorldQA/0", train.id.tolist())
        self.assertEqual(train.attrs["orbit_protocol"]["text_overlap_dropped_queries"], 2)

    def test_materialized_jpeg_preserves_original_encoded_bytes(self):
        buffer = io.BytesIO()
        Image.new("RGB", (20, 16), "red").save(buffer, format="JPEG")
        raw = buffer.getvalue()
        loader = VLRouterBenchLoader(str(self.root), vl_args(self.root))
        path = loader._decode_images(base64.b64encode(raw).decode())[0]
        self.assertEqual(path.read_bytes(), raw)
        with Image.open(path) as image:
            self.assertEqual(image.format, "JPEG")

    def test_tsv_missing_cyclic_or_invalid_references_fail(self):
        path = self.root / "vlm_router_data/TSV_images/RealWorldQA.tsv"
        original = path.read_text()
        for images in (["20", "10", "20"], ["999", "999", "999"], ["!!invalid!!"]*3):
            with self.subTest(images=images):
                pd.DataFrame({"index": [10, 20, 30], "image": images}).to_csv(path, sep="\t", index=False)
                with self.assertRaises(ValueError):
                    download_dataset(vl_args(self.root, modality="text+image"))
        path.write_text(original)

    def test_missing_assets_are_not_silent_text_only_evaluation(self):
        (self.root / "vlm_router_data/TSV_images/RealWorldQA.tsv").unlink()
        with self.assertRaises(FileNotFoundError):
            download_dataset(vl_args(self.root, modality="text+image"))
        # Text-only is an explicit ablation and needs no image decoder/resources.
        train, test, _ = download_dataset(vl_args(self.root, modality="text"))
        self.assertNotIn("image_path", train)

    def test_image_row_maps_are_reused_between_router_loads(self):
        args = vl_args(self.root, modality="text+image")
        first, _, _ = download_dataset(args)
        with patch("utils.vl_routerbench.csv.DictReader", side_effect=AssertionError("Do not rescan TSV")):
            second, _, _ = download_dataset(args)
        self.assertEqual(first.image_path.tolist(), second.image_path.tolist())

    def test_relative_direct_image_paths(self):
        image_root = self.root / "vlm_router_data/TSV_images"
        Image.new("RGB", (18, 14), "green").save(image_root / "direct.png")
        path = self.root / "BENCHMARKS/vqa_oe/realworldqa_samples.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        records[0]["assets"] = [{"type": "image", "uri": "direct.png"}]
        path.write_text("".join(json.dumps(r)+"\n" for r in records))
        train, _, _ = download_dataset(vl_args(self.root, modality="text+image"))
        self.assertEqual(train.set_index("id").loc["RealWorldQA/0"].image_path, str(image_root / "direct.png"))

    def test_no_pickle_or_matrix_reading(self):
        with patch("pickle.load", side_effect=AssertionError("No pickle")), patch("numpy.load", side_effect=AssertionError("No matrix")):
            download_dataset(vl_args(self.root))

    def test_unknown_model_or_benchmark_fails(self):
        for options in ({"models": ["unknown"]}, {"benchmarks": ["unknown"]}):
            args = vl_args(self.root)
            args["dataset"].update(options)
            with self.assertRaises(ValueError):
                download_dataset(args)

    def test_missing_prepared_export_has_actionable_error(self):
        args = vl_args(self.root / "not-prepared")
        with self.assertRaisesRegex(FileNotFoundError, "Steps 1 and 2"):
            download_dataset(args)

    def test_missing_response_supervision_is_not_fabricated(self):
        with self.assertRaisesRegex(ValueError, "response strings"):
            download_dataset(vl_args(self.root, include_responses=True))

    def test_random_split_uses_existing_protocol(self):
        train, test, _ = download_dataset(vl_args(self.root, split={"mode": "in-domain", "ratios": {"train": 0.5, "test": 0.5}}))
        self.assertEqual((len(train), len(test)), (7, 7))
        self.assertFalse(set(train.input_group) & set(test.input_group))

    def test_image_path_traversal_invalid_index_and_asset_types_fail(self):
        path = self.root / "BENCHMARKS/vqa_oe/realworldqa_samples.jsonl"
        original = path.read_text()
        for asset in (
            {"type": "image_tsv", "tsv_file": "../RealWorldQA.tsv", "index": 0},
            {"type": "image_tsv", "tsv_file": "RealWorldQA.tsv", "index": True},
            {"type": "image_tsv", "tsv_file": "", "index": 0},
            {"type": "image", "uri": "../secret.png"},
            {"type": "image_url", "uri": "https://example.com/image.png"},
        ):
            with self.subTest(asset=asset):
                records = [json.loads(line) for line in original.splitlines()]
                records[0]["assets"] = [asset]
                path.write_text("".join(json.dumps(r)+"\n" for r in records))
                with self.assertRaises(ValueError):
                    download_dataset(vl_args(self.root))

    def test_bounded_multimodal_only_decodes_retained_splits(self):
        path = self.root / "BENCHMARKS/vqa_oe/realworldqa_samples.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        # Deliberately unavailable images on excluded dev queries are harmless.
        records[8]["assets"][0]["index"] = 10000
        path.write_text("".join(json.dumps(r)+"\n" for r in records))
        train, test, _ = download_dataset(vl_args(self.root, modality="text+image", max_samples=2))
        self.assertEqual((len(train), len(test)), (2, 2))

    def test_download_helper_refuses_overwrites_and_unverified_archives(self):
        from scripts import download_vlrouterbench as helper
        with patch("sys.argv", ["download_vlrouterbench", "--output-dir", str(self.root)]), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                helper.main()
        archive = self.root / "not-an-official-archive.tar.gz"
        archive.write_bytes(b"not the official release")
        output = self.root / "safe-output"
        with patch("sys.argv", ["download_vlrouterbench", "--output-dir", str(output), "--archive", str(archive)]):
            with self.assertRaisesRegex(ValueError, "hash"):
                helper.main()
        self.assertFalse(any(output.iterdir()))


if __name__ == "__main__":
    unittest.main()
