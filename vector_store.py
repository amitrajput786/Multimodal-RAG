"""
vector_store.py — ChromaDB vector database wrapper
====================================================
Stores and retrieves document embeddings using ChromaDB.
Persistent storage means you only embed documents once.
"""

import uuid
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

import numpy as np

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    """
    Manages the ChromaDB vector database.
    Handles storing embeddings and similarity search.
    """

    def __init__(
        self,
        persist_directory: str = None,
        collection_name: str = None,
    ):
        self.persist_directory = persist_directory or config.CHROMADB_PERSIST_DIRECTORY
        self.collection_name   = collection_name   or config.COLLECTION_NAME
        self.client     = None
        self.collection = None

    def _init_client(self):
        """Initialize ChromaDB client (lazy init)."""
        if self.client is not None:
            return

        try:
            import chromadb
            from chromadb.config import Settings

            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False),
            )

            # Get or create collection
            try:
                self.collection = self.client.get_collection(self.collection_name)
                logger.info(f"✅ Loaded existing collection '{self.collection_name}' "
                            f"({self.collection.count()} docs)")
            except Exception:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Multimodal RAG document store"},
                )
                logger.info(f"✅ Created new collection '{self.collection_name}'")

        except Exception as e:
            logger.error(f"❌ ChromaDB init error: {e}")
            raise

    # ── Public API ────────────────────────────────────────────────────────────

    def add_documents(
        self,
        texts:      List[str],
        embeddings: List[List[float]],
        metadatas:  List[Dict[str, Any]],
        ids:        Optional[List[str]] = None,
    ) -> List[str]:
        """
        Store documents with their embeddings in the vector DB.

        Args:
            texts:      Raw text of each chunk
            embeddings: Embedding vectors
            metadatas:  Metadata dicts (source file, page, etc.)
            ids:        Optional IDs (auto-generated if None)

        Returns:
            List of stored IDs
        """
        self._init_client()

        if not texts:
            return []

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

        # Ensure embeddings are plain Python lists
        clean_embeddings = [
            emb.tolist() if isinstance(emb, np.ndarray) else list(emb)
            for emb in embeddings
        ]

        # Batch insert (safer for large uploads)
        BATCH = 50
        added_ids = []

        for i in range(0, len(texts), BATCH):
            self.collection.add(
                documents=texts[i : i + BATCH],
                embeddings=clean_embeddings[i : i + BATCH],
                metadatas=metadatas[i : i + BATCH],
                ids=ids[i : i + BATCH],
            )
            added_ids.extend(ids[i : i + BATCH])

        logger.info(f"✅ Stored {len(added_ids)} chunks in vector DB")
        return added_ids

    def search_similar(
        self,
        query_embedding: Union[List[float], np.ndarray],
        n_results: int = 5,
    ) -> Dict[str, Any]:
        """
        Find the most similar chunks to a query embedding.

        Args:
            query_embedding: Embedding vector of the query
            n_results:       Number of results to return

        Returns:
            {
                'documents': [...],
                'metadatas': [...],
                'distances': [...],
                'ids':       [...]
            }
        """
        self._init_client()

        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()

        # Don't request more results than we have stored
        count = self.collection.count()
        n_results = min(n_results, max(1, count))

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
        )

        return {
            "documents": results["documents"][0] if results["documents"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "distances": results["distances"][0] if results["distances"] else [],
            "ids":       results["ids"][0]       if results["ids"]       else [],
        }

    def get_collection_info(self) -> Dict[str, Any]:
        """Returns info about the current collection."""
        self._init_client()
        return {
            "name":  self.collection_name,
            "count": self.collection.count(),
            "path":  self.persist_directory,
        }

    def clear_collection(self) -> bool:
        """Delete all documents from the collection."""
        try:
            self._init_client()
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Multimodal RAG document store"},
            )
            logger.info("✅ Collection cleared")
            return True
        except Exception as e:
            logger.error(f"❌ Clear error: {e}")
            return False
