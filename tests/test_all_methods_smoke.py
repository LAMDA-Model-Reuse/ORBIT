"""All-registered-method matrix with actual offline metric/result output.

Large pretrained encoders are replaced at the input/checkpoint boundary only.
Router training, prediction, allocation, nAUC and JSON writing are real.
"""

import logging
import os
from pathlib import Path
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from smoke_all_methods import (
    ROUTER_REGISTRY, TEXT_ONLY, _prepare, _run, _tiny_routefm, _tiny_transformers,
)


class AllMethodsMatrixTest(unittest.TestCase):
    def test_every_registered_method_trains_predicts_and_writes_evaluation(self):
        previous_directory = Path.cwd()
        logger = logging.getLogger()
        previous_level = logger.level
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            logger.setLevel(logging.INFO)
            os.chdir(ROOT)
            with tempfile.TemporaryDirectory(prefix="orbit-all-method-regression-") as temporary:
                directory = Path(temporary)
                configs = _prepare(directory, real=False)
                checkpoints = directory / "tiny_checkpoints"
                _tiny_transformers(checkpoints)
                routefm = checkpoints / "tiny_routefm.pt"
                _tiny_routefm(routefm)
                os.chdir(directory)
                for name, base in configs.items():
                    for method in sorted(ROUTER_REGISTRY):
                        with self.subTest(benchmark=name, method=method):
                            # Equal-mean pools exercise automatic pair selection;
                            # no manually chosen endpoints conceal regressions.
                            result = _run(base, method, directory, checkpoints, routefm, False)
                            self.assertEqual(result["status"], "PASS")
                for method in sorted(set(ROUTER_REGISTRY) - TEXT_ONLY):
                    with self.subTest(benchmark="MMRBenchV2", method=method, modality="text+image"):
                        result = _run(configs["MMRBenchV2"], method, directory, checkpoints, routefm, True)
                        self.assertEqual(result["status"], "PASS")
        finally:
            os.chdir(previous_directory)
            logger.setLevel(previous_level)
            torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    unittest.main()
