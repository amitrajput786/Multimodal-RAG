"""
rag_pipeline.py — Core RAG orchestrator
=========================================
Coordinates: document processing → embedding → storage → retrieval → LLM answer.
This is the single entry point used by the Streamlit app.
"""

import gc
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

import config
from document_processor  import DocumentProcessor
from audio_processor     import AudioProcessor
from embedding_generator import EmbeddingGenerator
from vector_store        import VectorStore
from llm_interface       import LLMInterface

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultimodalRAGPipeline:
    """
    End-to-end Multimodal RAG pipeline.

    Flow:
      Upload file → extract text/audio/image → chunk → embed → store in ChromaDB
      Ask question → embed question → similarity search → LLM generates answer
    """

    def __init__(self):
        logger.info("Initializing Multimodal RAG Pipeline...")

        self.doc_processor       = DocumentProcessor()
        self.audio_processor     = AudioProcessor(model_size=config.WHISPER_MODEL)
        self.embedding_generator = EmbeddingGenerator(text_model_name=config.EMBEDDING_MODEL)
        self.vector_store        = VectorStore(
            persist_directory=config.CHROMADB_PERSIST_DIRECTORY,
            collection_name=config.COLLECTION_NAME,
        )
        self.llm_interface = LLMInterface(
            base_url=config.OLLAMA_BASE_URL,
            model_name=config.DEFAULT_LLM_MODEL,
        )

        self.supported_doc_types   = config.SUPPORTED_DOC_TYPES
        self.supported_image_types = config.SUPPORTED_IMAGE_TYPES
        self.supported_audio_types = config.SUPPORTED_AUDIO_TYPES

        logger.info(f"✅ Pipeline ready | LLM: {config.DEFAULT_LLM_MODEL}")

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def process_and_store_files(
        self, uploaded_files: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Process uploaded files and store their embeddings in ChromaDB.

        Args:
            uploaded_files: List of {'path': str, 'name': str}

        Returns:
            Summary dict with counts and any errors
        """
        results = {
            "processed_documents": 0,
            "processed_images":    0,
            "processed_audio":     0,
            "stored_chunks":       0,
            "errors":              [],
        }

        for file_info in uploaded_files:
            file_path = Path(file_info["path"])
            ext       = file_path.suffix.lower()

            logger.info(f"Processing: {file_path.name}")

            try:
                if ext in self.supported_doc_types:
                    chunks = self._process_document(file_path)
                    if chunks:
                        results["processed_documents"] += 1

                elif ext in self.supported_image_types:
                    chunks = self._process_image(file_path)
                    if chunks:
                        results["processed_images"] += 1

                elif ext in self.supported_audio_types:
                    chunks = self._process_audio(file_path)
                    if chunks:
                        results["processed_audio"] += 1

                else:
                    results["errors"].append(f"Unsupported type: {ext}")
                    continue

                if chunks:
                    stored = self._store_chunks(chunks)
                    results["stored_chunks"] += stored

            except Exception as e:
                msg = f"Error processing {file_path.name}: {e}"
                logger.error(f"❌ {msg}")
                results["errors"].append(msg)
            finally:
                gc.collect()

        logger.info(
            f"✅ Processing complete — "
            f"{results['stored_chunks']} chunks stored, "
            f"{len(results['errors'])} errors"
        )
        return results

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(self, query_text: str, top_k: int = None) -> Dict[str, Any]:
        """
        Answer a question using RAG.

        Flow:
          1. Embed the question
          2. Find top-k similar chunks in ChromaDB
          3. Send chunks + question to Qwen2.5 via Ollama
          4. Return answer with source citations

        Args:
            query_text: The user's question
            top_k:      Number of chunks to retrieve (default from config)

        Returns:
            {
                'query':       str,
                'answer':      str,
                'sources':     [...],
                'num_sources': int,
                'model_used':  str,
                'status':      str
            }
        """
        if not query_text.strip():
            return {"query": query_text, "answer": "Please enter a question.",
                    "sources": [], "num_sources": 0, "status": "empty"}

        top_k = top_k or config.TOP_K_RESULTS

        try:
            # Step 1 — embed the question
            query_embeddings = self.embedding_generator.generate_text_embeddings([query_text])
            if not query_embeddings:
                raise RuntimeError("Failed to generate query embedding")

            # Step 2 — retrieve similar chunks
            search_results = self.vector_store.search_similar(
                query_embedding=query_embeddings[0],
                n_results=top_k,
            )

            docs      = search_results.get("documents", [])
            metas     = search_results.get("metadatas", [])
            distances = search_results.get("distances", [])

            if not docs:
                return {
                    "query":       query_text,
                    "answer":      "No relevant documents found. Please upload and process files first.",
                    "sources":     [],
                    "num_sources": 0,
                    "status":      "no_docs",
                }

            # Step 3 — build context with source labels
            context_docs = []
            for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances)):
                context_docs.append({
                    "text":             doc,
                    "metadata":         meta,
                    "similarity_score": max(0.0, 1.0 - dist),
                    "rank":             i + 1,
                })

            # Step 4 — generate answer via LLM
            llm_result = self.llm_interface.answer_question_with_context(
                question=query_text,
                context_docs=context_docs,
            )

            logger.info(f"✅ Query answered | {len(context_docs)} sources | "
                        f"model: {llm_result.get('model')}")

            return {
                "query":       query_text,
                "answer":      llm_result.get("response", ""),
                "sources":     context_docs,
                "num_sources": len(context_docs),
                "model_used":  llm_result.get("model", ""),
                "status":      llm_result.get("status", "success"),
                "citations":   llm_result.get("citations", []),
            }

        except Exception as e:
            logger.error(f"❌ Query error: {e}")
            return {
                "query":       query_text,
                "answer":      f"Error processing query: {str(e)}",
                "sources":     [],
                "num_sources": 0,
                "status":      "error",
            }

    # ── Status ────────────────────────────────────────────────────────────────

    def get_system_status(self) -> Dict[str, Any]:
        """Returns current system status for the sidebar."""
        try:
            db_info = self.vector_store.get_collection_info()
            models  = self.llm_interface.list_available_models()

            return {
                "vector_store": {
                    "collection_name": db_info["name"],
                    "document_count":  db_info["count"],
                },
                "llm": {
                    "current_model":   self.llm_interface.model_name,
                    "available_models": models,
                },
                "supported_formats": {
                    "documents": self.supported_doc_types,
                    "images":    self.supported_image_types,
                    "audio":     self.supported_audio_types,
                },
                "embedding_model": config.EMBEDDING_MODEL,
                "device":          self.embedding_generator.device,
            }
        except Exception as e:
            return {"error": str(e), "status": "error"}

    def clear_database(self) -> bool:
        """Wipe the vector DB."""
        success = self.vector_store.clear_collection()
        if success:
            gc.collect()
        return success

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _process_document(self, file_path: Path) -> List[Dict]:
        result = self.doc_processor.process_document(file_path)
        if result["processing_status"] != "success":
            return []

        text_data = result["content"]["text"]
        if not text_data:
            return []

        chunks = self.doc_processor.chunk_text(
            text_data,
            chunk_size=config.CHUNK_SIZE,
            overlap=config.CHUNK_OVERLAP,
        )
        if not chunks:
            return []

        texts      = [c["text"] for c in chunks]
        embeddings = self.embedding_generator.generate_text_embeddings(texts)

        return [
            {
                "text":      chunks[i]["text"],
                "embedding": embeddings[i],
                "metadata":  {
                    "source_file":  file_path.name,
                    "content_type": "text",
                    "chunk_type":   "document_text",
                    "chunk_id":     i,
                    **{k: str(v) for k, v in chunks[i].get("source", {}).items()},
                },
            }
            for i in range(len(chunks))
        ]

    def _process_image(self, file_path: Path) -> List[Dict]:
        embeddings = self.embedding_generator.generate_image_embeddings([file_path])
        if len(embeddings) == 0:
            return []

        return [{
            "text":      f"Image file: {file_path.name}",
            "embedding": embeddings[0].tolist(),
            "metadata":  {
                "source_file":  file_path.name,
                "content_type": "image",
                "chunk_type":   "image",
            },
        }]

    def _process_audio(self, file_path: Path) -> List[Dict]:
        audio_result = self.audio_processor.process_audio(file_path)
        if audio_result["processing_status"] != "success":
            return []

        transcript_chunks = self.audio_processor.chunk_transcription(audio_result)
        texts = [c["text"] for c in transcript_chunks if c["text"]]
        if not texts:
            return []

        embeddings = self.embedding_generator.generate_text_embeddings(texts)

        return [
            {
                "text":      transcript_chunks[i]["text"],
                "embedding": embeddings[i],
                "metadata":  {
                    "source_file":  file_path.name,
                    "content_type": "audio",
                    "chunk_type":   "audio_transcription",
                    "start_time":   str(transcript_chunks[i].get("start_time", 0)),
                    "end_time":     str(transcript_chunks[i].get("end_time", 0)),
                },
            }
            for i in range(len(transcript_chunks))
            if transcript_chunks[i]["text"]
        ]

    def _store_chunks(self, chunks: List[Dict]) -> int:
        """Store a list of chunks in the vector DB."""
        if not chunks:
            return 0

        texts      = [c["text"]      for c in chunks]
        embeddings = [c["embedding"] for c in chunks]
        metadatas  = [c["metadata"]  for c in chunks]

        ids = self.vector_store.add_documents(texts, embeddings, metadatas)
        return len(ids)
