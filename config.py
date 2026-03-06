"""
config.py — Central configuration for Multimodal RAG Pipeline
=============================================================
To switch LLM models, change DEFAULT_LLM_MODEL below.
Any Ollama-compatible model can be plugged in with zero other changes.

Model upgrade guide (when better open-source models release):
  - ollama pull <new_model_name>
  - Change DEFAULT_LLM_MODEL = "<new_model_name>"
  - That's it. No other file needs to change.

Current model: qwen2.5:1.5b
  - Replaced: llama3.2:1b (gave poor, incoherent answers on RAG tasks)
  - Reason: Qwen2.5 trained on 18T tokens, far better instruction following
  - RAM needed: ~3GB RAM (runs on CPU via Ollama, frees GPU for embeddings)
"""

import os
from pathlib import Path

# ── Directories ──────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
DATA_DIR  = BASE_DIR / "data"
UPLOADS_DIR   = DATA_DIR / "uploads"
VECTOR_DB_DIR = DATA_DIR / "vector_db"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

# ── Embedding Model ───────────────────────────────────────────────────────────
# Runs on GPU (RTX 3050) via sentence-transformers
# all-MiniLM-L6-v2 is lighter (384-dim) and fits in 4GB VRAM comfortably
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── Whisper (Audio Transcription) ─────────────────────────────────────────────
# Using "tiny" to keep memory low — "base" is also fine on RTX 3050
WHISPER_MODEL = "tiny"

# ── LLM Configuration ─────────────────────────────────────────────────────────
# ✅ UPGRADED from llama3.2:1b → qwen2.5:1.5b
# Pull command: ollama pull qwen2.5:1.5b
#
# Want to upgrade later? Just change this one line:
#   "qwen2.5:3b"   — better quality, needs ~5GB RAM
#   "mistral"      — best quality for RAG, needs ~8GB RAM
#   "qwen2.5:7b"   — excellent, needs ~10GB RAM
DEFAULT_LLM_MODEL = "qwen2.5:1.5b"
OLLAMA_BASE_URL   = "http://localhost:11434"

# ── Vector Database ───────────────────────────────────────────────────────────
CHROMADB_PERSIST_DIRECTORY = str(VECTOR_DB_DIR / "chroma_db")
COLLECTION_NAME = "multimodal_rag"

# ── Chunking & Retrieval ──────────────────────────────────────────────────────
CHUNK_SIZE    = 600    # chars per chunk (slightly larger = more context per chunk)
CHUNK_OVERLAP = 150    # overlap prevents cutting ideas mid-sentence
TOP_K_RESULTS = 5      # retrieve top-5 most relevant chunks

# ── LLM Generation Parameters ────────────────────────────────────────────────
LLM_TEMPERATURE  = 0.3    # low = factual, high = creative
LLM_NUM_PREDICT  = 700    # max tokens to generate in answer
LLM_NUM_CTX      = 4096   # context window — fits 5 chunks + question comfortably

# ── File Limits ───────────────────────────────────────────────────────────────
MAX_FILE_SIZE_MB   = 50
BATCH_SIZE         = 1     # process one file at a time (memory safety on 4GB GPU)
EMBEDDING_BATCH_SIZE = 8   # embed 8 chunks at once on GPU

# ── Supported Formats ─────────────────────────────────────────────────────────
SUPPORTED_DOC_TYPES   = [".pdf", ".docx", ".txt"]
SUPPORTED_IMAGE_TYPES = [".jpg", ".jpeg", ".png"]
SUPPORTED_AUDIO_TYPES = [".wav", ".mp3"]

# ── UI ────────────────────────────────────────────────────────────────────────
PAGE_TITLE = "Multimodal RAG — Qwen2.5"
PAGE_ICON  = "🔍"
LAYOUT     = "wide"

# ── Debug ─────────────────────────────────────────────────────────────────────
DEBUG_MODE       = False
VERBOSE_LOGGING  = False
