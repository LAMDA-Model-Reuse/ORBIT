"""Joint Qwen3-VL embeddings; never substitute text-only/image concatenations.

Local mode uses the model-card SentenceTransformer interface and local weights.
API mode targets an explicitly configured RouteFM-compatible embedding service:
POST /embeddings, input=[{text, image: data-URL}], indexed data[].embedding.
This multimodal input schema is not a universal OpenAI/vLLM API contract.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Do not forward authorization or images to a different URL.


class QwenVLEmbedder:
    def __init__(self, embeddings):
        self.config = dict(embeddings.get("qwen", {}))
        self.backend = self.config.get("backend", "local")
        self.instruction = self.config.get("instruction")
        self.model_name = self.config.get("model", "Qwen/Qwen3-VL-Embedding-8B")
        self.model = None
        if self.backend == "local":
            model_dir = Path(self.config.get("model_dir", "./Qwen3-VL-Embedding-8B"))
            if not model_dir.is_dir():
                raise FileNotFoundError(
                    "Download Qwen/Qwen3-VL-Embedding-8B first and set embeddings.qwen.model_dir, "
                    "or explicitly configure the api backend. No substitute encoder is used."
                )
            from sentence_transformers import SentenceTransformer
            dtype = self.config.get("dtype", "auto")
            if dtype not in {"auto", "float32", "float16", "bfloat16"}:
                raise ValueError("Qwen local dtype must be auto, float32, float16 or bfloat16.")
            self.model = SentenceTransformer(
                str(model_dir), device=embeddings.get("device", "cpu"),
                local_files_only=True, trust_remote_code=False,
                model_kwargs={"torch_dtype": dtype},
            )
        elif self.backend == "api":
            self.base_url = str(self.config.get("base_url") or "").rstrip("/")
            url = urllib.parse.urlsplit(self.base_url)
            if (url.scheme not in {"http", "https"} or not url.hostname or url.username
                    or url.password or url.query or url.fragment):
                raise ValueError("Qwen API base_url must be an explicit HTTP(S) endpoint without credentials/query.")
            if url.scheme != "https" and url.hostname not in {"127.0.0.1", "localhost", "::1"}:
                raise ValueError("Qwen API requires HTTPS except for a local service.")
            self.api_key_env = self.config.get("api_key_env", "ROUTEFM_EMBEDDING_API_KEY")
            if not os.environ.get(self.api_key_env):
                raise ValueError(f"Set {self.api_key_env} for the explicitly selected Qwen embedding service.")
            self.timeout = float(self.config.get("timeout", 60))
            if not np.isfinite(self.timeout) or self.timeout <= 0:
                raise ValueError("Qwen API timeout must be finite and positive.")
            handlers = [_NoRedirect()]
            if url.hostname in {"127.0.0.1", "localhost", "::1"}:
                handlers.append(urllib.request.ProxyHandler({}))
            self.opener = urllib.request.build_opener(*handlers)
        else:
            raise ValueError("embeddings.qwen.backend must be local or api.")

    def __call__(self, texts=None, images=None):
        if texts is None and images is None:
            raise ValueError("Qwen joint encoder requires inputs.")
        count = len(texts) if texts is not None else len(images)
        if images is not None and len(images) != count:
            raise ValueError("Qwen text/image batch sizes must match.")
        inputs = []
        for index in range(count):
            value = {}
            if texts is not None:
                value["text"] = str(texts[index])
            if images is not None:
                image = images[index]
                if not isinstance(image, (str, Path)) or not Path(image).is_file():
                    raise ValueError("Qwen image inputs must be existing local files; no silent image omission.")
                value["image"] = str(Path(image).resolve())
            inputs.append(value)
        values = np.asarray(self._encode_batch(inputs), dtype=np.float32)
        if values.shape != (count, 4096) or not np.isfinite(values).all():
            raise ValueError("Qwen3-VL-Embedding-8B must return finite [batch,4096] vectors.")
        norms = np.linalg.norm(values.astype(np.float64), axis=1, keepdims=True)
        if np.any(norms <= 1e-12):
            raise ValueError("Qwen joint encoder returned a zero vector.")
        return (values / norms).astype(np.float32)

    def _encode_batch(self, inputs):
        if self.backend == "local":
            options = {"normalize_embeddings": True, "convert_to_numpy": True, "show_progress_bar": False}
            if self.instruction is not None:
                options["prompt"] = self.instruction
            return self.model.encode(inputs, **options)

        payload_inputs = []
        for item in inputs:
            value = dict(item)
            if "image" in value:
                path = Path(value["image"])
                if path.stat().st_size > 32 * 1024 * 1024:
                    raise ValueError("Qwen API image exceeds the 32 MiB upload limit.")
                with Image.open(path) as image:
                    mime = Image.MIME.get(image.format)
                    image.verify()
                if mime not in {"image/jpeg", "image/png", "image/webp"}:
                    raise ValueError("Qwen API supports local JPEG/PNG/WebP images only.")
                value["image"] = f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")
            if self.instruction is not None:
                value["instruction"] = self.instruction
            payload_inputs.append(value)
        request = urllib.request.Request(
            self.base_url + "/embeddings",
            data=json.dumps({"model": self.model_name, "input": payload_inputs}).encode(),
            headers={"Authorization": "Bearer " + os.environ[self.api_key_env], "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                body = response.read(32 * 1024 * 1024 + 1)
        except (urllib.error.URLError, TimeoutError):
            # Provider errors can contain credentials/input; do not echo their body.
            raise RuntimeError("Qwen embedding request failed; check endpoint/auth/service logs.") from None
        if len(body) > 32 * 1024 * 1024:
            raise ValueError("Qwen embedding response exceeds the size limit.")
        payload = json.loads(body)
        rows = payload.get("data", [])
        indices = [row.get("index") for row in rows]
        if (any(type(index) is not int for index in indices)
                or sorted(indices) != list(range(len(inputs)))):
            raise ValueError("Qwen embedding response must contain each batch index exactly once.")
        return [row["embedding"] for row in sorted(rows, key=lambda row: row["index"])]
