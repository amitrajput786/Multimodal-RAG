"""
embedding_generator.py — Text and image embeddings
====================================================
Uses sentence-transformers on GPU (RTX 3050) for text.
Images use a simple CLIP-style description approach since
we don't have a full multimodal model in this setup.
"""

from pathlib import Path
from typing import List, Union
import logging

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generates embeddings for text and images.
    Text model runs on GPU if available (RTX 3050 handles this fine).
    """

    def __init__(self, text_model_name: str = None):
        self.text_model_name = text_model_name or config.EMBEDDING_MODEL
        self.text_model = None

        # Use GPU for embeddings if available — saves time significantly
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"EmbeddingGenerator using device: {self.device}")

    def _load_model(self) -> SentenceTransformer:
        """Lazy-load the model on first use."""
        if self.text_model is None:
            logger.info(f"Loading embedding model: {self.text_model_name}")
            self.text_model = SentenceTransformer(self.text_model_name, device=self.device)
            logger.info("✅ Embedding model loaded")
        return self.text_model

    def generate_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of text strings.

        Args:
            texts: List of strings to embed

        Returns:
            List of embedding vectors (each is a List[float])
        """
        if not texts:
            return []

        model = self._load_model()
        batch_size = config.EMBEDDING_BATCH_SIZE

        logger.info(f"Embedding {len(texts)} chunks (batch_size={batch_size})")

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = model.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=batch_size,
            )
            all_embeddings.extend(embeddings.tolist())

        logger.info(f"✅ Generated {len(all_embeddings)} embeddings")
        return all_embeddings

    def generate_image_embeddings(self, image_paths: List[Path]) -> np.ndarray:
        """
        Generate embeddings for images by converting them to descriptive text
        and embedding that text.

        NOTE: For true cross-modal image search, upgrade to CLIP or LLaVA.
        This approach creates searchable image representations using metadata.

        Args:
            image_paths: List of image file paths

        Returns:
            numpy array of embeddings
        """
        if not image_paths:
            return np.array([])

        model = self._load_model()
        image_texts = []

        for path in image_paths:
            try:
                img = Image.open(path)
                w, h = img.size
                mode = img.mode
                description = (
                    f"Image file: {path.name}. "
                    f"Dimensions: {w}x{h} pixels. "
                    f"Color mode: {mode}. "
                    f"Format: {path.suffix.upper().strip('.')}."
                )
                image_texts.append(description)
            except Exception as e:
                logger.warning(f"Could not process image {path.name}: {e}")
                image_texts.append(f"Image file: {path.name}")

        embeddings = model.encode(
            image_texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings

    def get_embedding_dimension(self) -> int:
        """Returns the dimension of the embedding vectors."""
        model = self._load_model()
        return model.get_sentence_embedding_dimension()
