import logging
import os
from typing import List, Optional, Union, Dict, Any, Callable, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

import clip
from torch.utils.data import Dataset, DataLoader
from PIL import Image


# =========================
# Registry (academic / extensible design)
# =========================

EmbedFn = Callable[..., torch.Tensor]
Builder = Callable[[Dict[str, Any]], Tuple[EmbedFn, int]]  # (embed_fn, dim)


class EmbedderRegistry:
    """A lightweight registry mapping model names to builder functions."""
    def __init__(self, name: str):
        self.name = name
        self._builders: Dict[str, Builder] = {}

    def register(self, key: str):
        def deco(fn: Builder):
            if key in self._builders:
                raise KeyError(f"[{self.name}] duplicate registration: {key}")
            self._builders[key] = fn
            return fn
        return deco

    def create(self, key: str, args: Dict[str, Any]) -> Tuple[EmbedFn, int]:
        if key not in self._builders:
            available = ", ".join(sorted(self._builders.keys()))
            raise ValueError(f"[{self.name}] unknown model '{key}'. Available: {available}")
        return self._builders[key](args)

    def keys(self) -> List[str]:
        return sorted(self._builders.keys())


TEXT_EMBEDDERS = EmbedderRegistry("text-embedders")
IMAGE_EMBEDDERS = EmbedderRegistry("image-embedders")


# =========================
# Core Embedder wrapper
# =========================

class Embedder:
    """
    Unified embedding wrapper for text/image encoders with optional trainable projections.

    args: dictionary with key "embeddings" containing:
        "text_model": str or None
        "image_model": str or None
        "batch_size": int
        "normalize": bool
        "training": bool
        "out_dim": int or None
        optional common:
        "device": str (e.g., "cuda:0")
        "use_fp16": bool
        "text_model_dir": str
        "image_model_dir": str
    """

    def __init__(self, args: Dict[str, Any]):
        self.args = args
        em = args["embeddings"]

        self.normalize = bool(em.get("normalize", False))
        self.training = bool(em.get("training", False))
        self.out_dim = em.get("out_dim", None)

        self.text_embedder: Optional[EmbedFn] = None
        self.image_embedder: Optional[EmbedFn] = None
        self.text_dim: Optional[int] = None
        self.image_dim: Optional[int] = None

        if em.get("text_model"):
            self.text_embedder, self.text_dim = CreateTextEmbedder(em["text_model"], args)

        if em.get("image_model"):
            self.image_embedder, self.image_dim = CreateImageEmbedder(em["image_model"], args)

        self.text_proj: Optional[nn.Parameter] = None
        self.image_proj: Optional[nn.Parameter] = None

        if self.training:
            if self.out_dim is None:
                raise ValueError("embeddings.out_dim must be set when training=True.")
            if self.text_embedder is not None:
                assert self.text_dim is not None
                self.text_proj = nn.Parameter(
                    torch.randn(self.text_dim, self.out_dim) * (1.0 / np.sqrt(self.text_dim))
                )
            if self.image_embedder is not None:
                assert self.image_dim is not None
                self.image_proj = nn.Parameter(
                    torch.randn(self.image_dim, self.out_dim) * (1.0 / np.sqrt(self.image_dim))
                )

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        if self.text_proj is not None:
            params.append(self.text_proj)
        if self.image_proj is not None:
            params.append(self.image_proj)
        return params

    def set_projection(self, modality: str, weight_tensor: torch.Tensor):
        """
        modality: 'text' or 'image'
        weight_tensor shape must be (native_dim, out_dim)
        """
        if modality == "text":
            if self.text_proj is None:
                raise ValueError("text projection not initialized (training=False or no text_model).")
            if weight_tensor.shape != self.text_proj.shape:
                raise ValueError(f"shape mismatch for text projection: {weight_tensor.shape} vs {self.text_proj.shape}")
            with torch.no_grad():
                self.text_proj.copy_(weight_tensor)
        elif modality == "image":
            if self.image_proj is None:
                raise ValueError("image projection not initialized (training=False or no image_model).")
            if weight_tensor.shape != self.image_proj.shape:
                raise ValueError(f"shape mismatch for image projection: {weight_tensor.shape} vs {self.image_proj.shape}")
            with torch.no_grad():
                self.image_proj.copy_(weight_tensor)
        else:
            raise ValueError("unknown modality, use 'text' or 'image'")

    def _project(self, emb: torch.Tensor, modality: str) -> torch.Tensor:
        """
        Project embeddings to out_dim if training (learnable projection) or pass-through otherwise.
        emb: (N, native_dim)
        """
        if self.training:
            if modality == "text":
                if self.text_proj is None:
                    raise RuntimeError("text projection missing while training=True")
                proj = self.text_proj.to(emb.device)
                return emb @ proj
            elif modality == "image":
                if self.image_proj is None:
                    raise RuntimeError("image projection missing while training=True")
                proj = self.image_proj.to(emb.device)
                return emb @ proj
            else:
                raise ValueError("unknown modality")
        else:
            # In inference mode, allow out_dim to be None (no strict check).
            if self.out_dim is not None:
                # Keep your original relaxed rule: allow out_dim or out_dim//2 (if you concatenate later).
                if emb.shape[1] != self.out_dim and emb.shape[1] != self.out_dim // 2:
                    raise RuntimeError(f"Embedding dim {emb.shape[1]} does not match out_dim {self.out_dim}")
            return emb

    def run_embed(self, texts=None, images=None) -> torch.Tensor:
        batch_size = int(self.args["embeddings"].get("batch_size", 256))
        parts: List[torch.Tensor] = []
        N: Optional[int] = None

        if texts is not None:
            if self.text_embedder is None:
                raise ValueError("texts provided but no text_model initialized.")
            if isinstance(texts, str):
                texts_list = [texts]
            else:
                texts_list = list(texts)

            text_emb_batches: List[torch.Tensor] = []
            for i in tqdm(range(0, len(texts_list), batch_size), desc="Text Embedding"):
                batch = texts_list[i: i + batch_size]
                emb = self.text_embedder(batch)
                if not isinstance(emb, torch.Tensor):
                    emb = torch.from_numpy(np.array(emb))
                emb = emb.float()
                emb = self._project(emb, modality="text")
                text_emb_batches.append(emb)

            emb_text = torch.cat(text_emb_batches, dim=0)
            parts.append(emb_text)
            N = emb_text.shape[0]

        if images is not None:
            if self.image_embedder is None:
                raise ValueError("images provided but no image_model initialized.")
            total = images.shape[0] if isinstance(images, torch.Tensor) else len(images)

            img_emb_batches: List[torch.Tensor] = []
            for i in tqdm(range(0, total, batch_size), desc="Image Embedding"):
                batch = images[i: i + batch_size]
                emb = self.image_embedder(batch)
                if not isinstance(emb, torch.Tensor):
                    emb = torch.from_numpy(np.array(emb))
                emb = emb.float()
                emb = self._project(emb, modality="image")
                img_emb_batches.append(emb)

            emb_image = torch.cat(img_emb_batches, dim=0)
            parts.append(emb_image)

            if N is None:
                N = emb_image.shape[0]
            elif N != emb_image.shape[0]:
                raise ValueError("text and image batch sizes must match.")

        if len(parts) == 0:
            raise ValueError("no inputs provided to run_embed")

        combined = torch.cat(parts, dim=1)

        if self.normalize:
            norm = torch.linalg.norm(combined, dim=1, keepdim=True).clamp_min(1e-12)
            combined = combined / norm

        return combined


# =========================
# Text Embedders (implementations)
# =========================

def CreateOpenLLM_MiniLM_L6_V2(model_dir: str = "./all-MiniLM-L6-v2", device: str = "cuda:0"):
    """
    Load SentenceTransformer MiniLM L6 v2.
    Return: (embed_fn, dim) where embed_fn(texts) -> torch.FloatTensor (N, 384)
    """
    if os.path.exists(model_dir):
        model_name = model_dir
        logging.info(f"[embedding] Loading all-MiniLM-L6-v2 from local directory: {model_dir}")
    else:
        raise FileNotFoundError(
            f"[ERROR] Local model directory not found: {model_dir}\n\n"
            "Please download the model before using this function.\n"
            "Recommended high-speed download commands (China-friendly mirrors):\n"
            "    conda install -c conda-forge aria2\n"
            "    curl -L https://hf-mirror.com/hfd/hfd.sh -o hfd.sh\n"
            "    chmod +x hfd.sh\n"
            "    ./hfd.sh sentence-transformers/all-MiniLM-L6-v2\n\n"
            "After downloading, set 'model_dir' to the path of the downloaded model."
        )

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device=device)
    model.eval()

    logging.info(f"[embedding] Loaded SentenceTransformer model: {model_name}")
    dim = 384

    @torch.no_grad()
    def embed_fn(texts):
        if isinstance(texts, str):
            texts = [texts]
        emb = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=False
        )
        return torch.from_numpy(emb).float().to(device)

    return embed_fn, dim


def CreateQwen3_Embedding_8B(model_dir: str = "./Qwen3-Embedding-8B"):
    """
    Load Qwen3-Embedding-8B from local directory.
    Return: (embed_fn, dim), where embed_fn(texts) -> tensor of shape (N, dim)
    """
    if os.path.exists(model_dir):
        model_name = model_dir
        logging.info(f"[embedding] Loading Qwen3-Embedding-8B from local directory: {model_dir}")
    else:
        raise FileNotFoundError(
            f"[ERROR] Local model directory not found: {model_dir}\n\n"
            "Please download the model before using this function.\n"
            "Recommended high-speed download commands (China-friendly mirrors):\n"
            "    conda install -c conda-forge aria2\n"
            "    curl -L https://hf-mirror.com/hfd/hfd.sh -o hfd.sh\n"
            "    chmod +x hfd.sh\n"
            "    ./hfd.sh Qwen/Qwen3-Embedding-8B\n\n"
            "After downloading, set 'model_dir' to the path of the downloaded model."
        )

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = AutoModel.from_pretrained(model_name, torch_dtype=dtype, device_map="auto")
    model.eval()

    dim = 4096

    @torch.no_grad()
    def embed_fn(texts):
        if isinstance(texts, str):
            texts = [texts]
        batch = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=None
        )
        batch = {k: v.to(model.device) for k, v in batch.items()}
        out = model(**batch)

        last_hidden = out.last_hidden_state
        attn_mask = batch.get("attention_mask", None)
        if attn_mask is not None:
            seq_lens = attn_mask.sum(dim=1) - 1
            emb = last_hidden[torch.arange(last_hidden.size(0), device=last_hidden.device), seq_lens, :]
        else:
            emb = last_hidden[:, 0, :]

        return emb.float()

    return embed_fn, dim


def CreateQwen3_Embedding_0_6B(model_dir: str = "./Qwen3-Embedding-0.6B"):
    """
    Load Qwen3-Embedding-0.6B from local directory.
    Return: (embed_fn, dim), where embed_fn(texts) -> tensor (N, dim)
    """
    if os.path.exists(model_dir):
        model_name = model_dir
        logging.info(f"[embedding] Loading Qwen3-Embedding-0.6B from local directory: {model_dir}")
    else:
        raise FileNotFoundError(
            f"[ERROR] Local model directory not found: {model_dir}\n\n"
            "Please download the model before using this function.\n"
            "Recommended high-speed download commands (China-friendly mirrors):\n"
            "    conda install -c conda-forge aria2\n"
            "    curl -L https://hf-mirror.com/hfd/hfd.sh -o hfd.sh\n"
            "    chmod +x hfd.sh\n"
            "    ./hfd.sh Qwen/Qwen3-Embedding-0.6B\n\n"
            "After downloading, set 'model_dir' to the path of the downloaded model."
        )

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = AutoModel.from_pretrained(model_name, torch_dtype=dtype, device_map="auto")
    model.eval()

    dim = 1024

    @torch.no_grad()
    def embed_fn(texts):
        if isinstance(texts, str):
            texts = [texts]
        batch = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=None
        )
        batch = {k: v.to(model.device) for k, v in batch.items()}
        out = model(**batch)

        last_hidden = out.last_hidden_state
        attn_mask = batch.get("attention_mask", None)
        if attn_mask is not None:
            seq_lens = attn_mask.sum(dim=1) - 1
            emb = last_hidden[torch.arange(last_hidden.size(0), device=last_hidden.device), seq_lens, :]
        else:
            emb = last_hidden[:, 0, :]

        return emb.float()

    return embed_fn, dim


def CreateCLIP_ViT_B_16_text(model_name: str = "ViT-B/16", device: str = "cuda:0", use_fp16: bool = False):
    device_t = torch.device(device if torch.cuda.is_available() else "cpu")
    model, _ = clip.load(model_name, device=device_t)
    model.eval()

    dim = 512

    @torch.no_grad()
    def embed_fn(texts: Union[str, List[str]]):
        if isinstance(texts, str):
            texts = [texts]
        if len(texts) == 0:
            return torch.empty((0, dim), device=device_t)

        tokens = clip.tokenize(texts, truncate=True).to(device_t)

        if use_fp16 and device_t.type == "cuda":
            with torch.cuda.amp.autocast():
                text_features = model.encode_text(tokens)
        else:
            text_features = model.encode_text(tokens)

        text_features = text_features.float()
        text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-12)
        return text_features.to(device_t)

    return embed_fn, dim


# =========================
# Image Embedders (implementations)
# =========================

class _ImagePathDataset(Dataset):
    def __init__(self, items, preprocess):
        self.items = list(items)
        self.preprocess = preprocess

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        try:
            if isinstance(item, (str, os.PathLike)):
                img = Image.open(str(item)).convert("RGB")
            elif isinstance(item, Image.Image):
                img = item.convert("RGB")
            else:
                arr = np.asarray(item)
                if arr.ndim == 2:
                    arr = np.stack([arr] * 3, axis=-1)
                if arr.shape[-1] == 4:
                    arr = arr[..., :3]
                img = Image.fromarray(arr.astype("uint8"), "RGB")
            img_t = self.preprocess(img)
        except Exception as e:
            logging.warning(f"Failed to load/preprocess image at idx={idx}: {item}. Error: {e}")
            img_t = torch.zeros(3, 224, 224)
        return img_t


def CreateCLIP_ViT_B_16_image(model_name: str = "ViT-B/16", device: str = "cuda:0", use_fp16: bool = False):
    device_t = torch.device(device if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load(model_name, device=device_t)
    model.eval()

    dim = 512

    @torch.no_grad()
    def embed_fn(images: Union[str, os.PathLike, Image.Image, list], batch_size: int = 256, num_workers: int = 8):
        single = False
        if not isinstance(images, (list, tuple)):
            images = [images]
            single = True

        if len(images) == 0:
            return torch.empty((0, dim), device=device_t)

        dataset = _ImagePathDataset(images, preprocess)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device_t.type == "cuda"),
        )

        parts = []
        for batch in loader:
            batch = batch.to(device_t).float()
            if use_fp16 and device_t.type == "cuda":
                with torch.cuda.amp.autocast():
                    feats = model.encode_image(batch)
            else:
                feats = model.encode_image(batch)

            feats = feats.float()
            feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-12)
            parts.append(feats.cpu())

        if len(parts) == 0:
            return torch.empty((0, dim), device=device_t)

        out = torch.cat(parts, dim=0).to(device_t)
        if single:
            # keep behavior consistent if you want: return (1, dim) anyway
            return out
        return out

    return embed_fn, dim


# =========================
# Registration (model name -> builder)
# =========================

@TEXT_EMBEDDERS.register("Qwen3-Embedding-8B")
def _build_qwen3_8b(args: Dict[str, Any]):
    model_dir = args["embeddings"].get("text_model_dir", "./Qwen3-Embedding-8B")
    return CreateQwen3_Embedding_8B(model_dir=model_dir)


@TEXT_EMBEDDERS.register("Qwen/Qwen3-Embedding-0.6B")
def _build_qwen3_06b(args: Dict[str, Any]):
    model_dir = args["embeddings"].get("text_model_dir", "./Qwen3-Embedding-0.6B")
    return CreateQwen3_Embedding_0_6B(model_dir=model_dir)


@TEXT_EMBEDDERS.register("all-MiniLM-L6-v2")
def _build_minilm(args: Dict[str, Any]):
    model_dir = args["embeddings"].get("text_model_dir", "./all-MiniLM-L6-v2")
    device = args["embeddings"].get("device", "cuda:0")
    return CreateOpenLLM_MiniLM_L6_V2(model_dir=model_dir, device=device)


@TEXT_EMBEDDERS.register("ViT-B/16")
def _build_clip_text(args: Dict[str, Any]):
    device = args["embeddings"].get("device", "cuda:0")
    use_fp16 = bool(args["embeddings"].get("use_fp16", False))
    return CreateCLIP_ViT_B_16_text(model_name="ViT-B/16", device=device, use_fp16=use_fp16)


@IMAGE_EMBEDDERS.register("ViT-B/16")
def _build_clip_image(args: Dict[str, Any]):
    device = args["embeddings"].get("device", "cuda:0")
    use_fp16 = bool(args["embeddings"].get("use_fp16", False))
    return CreateCLIP_ViT_B_16_image(model_name="ViT-B/16", device=device, use_fp16=use_fp16)


# =========================
# Public factory APIs (stable interface)
# =========================

def CreateTextEmbedder(name: str, args: Dict[str, Any]):
    """Create a text embedder by registry name."""
    return TEXT_EMBEDDERS.create(name, args)


def CreateImageEmbedder(name: str, args: Dict[str, Any]):
    """Create an image embedder by registry name."""
    return IMAGE_EMBEDDERS.create(name, args)