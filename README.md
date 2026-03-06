# Multimodal RAG Pipeline

A locally-run Retrieval-Augmented Generation (RAG) system that answers questions from your uploaded documents. Supports PDF, DOCX, TXT, images, and audio files — all processed on your machine with no data leaving your system.

## 🔄 Model Update Notice

**Previous model:** `llama3.2:1b`
**Current model:** `qwen2.5:1.5b`

The original `llama3.2:1b` model produced poor, incoherent answers on RAG tasks. At only 1 billion parameters, it lacked the capacity to synthesize information from retrieved document chunks. Questions like *"summarize this PDF"* returned vague or irrelevant responses.

**`qwen2.5:1.5b` was chosen as the replacement because:**
- Trained on 18 trillion tokens (vs ~3T for LLaMA 1B variants)
- Significantly better instruction following and context reading
- Still lightweight — runs on CPU via Ollama, using only ~2–3 GB RAM
- Leaves GPU VRAM free for the embedding model (sentence-transformers on GPU)

**Want to upgrade to a better model in the future?** The architecture is designed for this. When a new open-source model releases (e.g., Qwen3, Llama 4, Mistral Next):
1. `ollama pull <new_model_name>`
2. Change `DEFAULT_LLM_MODEL = "<new_model_name>"` in `config.py`
3. That's the only change needed. The rest of the pipeline is model-agnostic.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit UI (app.py)                    │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│               RAG Pipeline (rag_pipeline.py)                 │
│                                                              │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │  Document   │  │    Audio     │  │  Embedding        │  │
│  │  Processor  │  │  Processor   │  │  Generator        │  │
│  │ (PDF/DOCX/  │  │  (Whisper)   │  │ (sentence-        │  │
│  │   TXT)      │  │              │  │  transformers GPU)│  │
│  └──────┬──────┘  └──────┬───────┘  └─────────┬─────────┘  │
│         │                │                     │             │
│         └────────────────┴──────────────────── ▼             │
│                                        ┌───────────────┐    │
│                                        │  Vector Store │    │
│                                        │  (ChromaDB)   │    │
│                                        └───────┬───────┘    │
│                                                │             │
│                                        ┌───────▼───────┐    │
│                                        │ LLM Interface │    │
│                                        │ qwen2.5:1.5b  │    │
│                                        │ (Ollama/CPU)  │    │
│                                        └───────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

**Key design decision:** Embeddings run on **GPU** (RTX 3050), LLM runs on **CPU** via Ollama. This avoids VRAM competition between the two models.

---

## 📁 Project Structure

```
multimodal_rag/
├── app.py                  # Streamlit web interface
├── rag_pipeline.py         # Core orchestrator
├── config.py               # All configuration (change model here)
├── document_processor.py   # PDF, DOCX, TXT extraction + chunking
├── audio_processor.py      # Whisper audio transcription
├── embedding_generator.py  # Sentence-transformer embeddings (GPU)
├── vector_store.py         # ChromaDB wrapper
├── llm_interface.py        # Ollama LLM communication
├── requirements.txt        # Pinned, conflict-free dependencies
└── data/                   # Auto-created: uploads + vector DB
    └── vector_db/
        └── chroma_db/
```

---

## ⚙️ Setup

### Prerequisites

- Python 3.10 or 3.11
- NVIDIA GPU with CUDA 12.1 (RTX 3050 4GB or better)
- [Ollama](https://ollama.com/) installed and running

### 1. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate    # Linux/Mac
# venv\Scripts\activate     # Windows
```

### 2. Install PyTorch with CUDA 12.1 first

```bash
pip install torch==2.2.2+cu121 --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 4. Pull the LLM model

```bash
ollama pull qwen2.5:1.5b
```

### 5. Run the app

```bash
ollama serve          # in one terminal (if not already running)
streamlit run app.py  # in another terminal
```

---

## 🚀 Usage

1. Open the app in your browser (http://localhost:8501)
2. Go to **Upload & Process** tab
3. Upload a PDF, DOCX, TXT, image, or audio file
4. Click **Process Files** — chunks will be embedded and stored
5. Go to **Ask Questions** tab
6. Type any question about your documents

---

## 🔧 Configuration

All settings are in `config.py`. Key ones:

| Setting | Default | Description |
|---------|---------|-------------|
| `DEFAULT_LLM_MODEL` | `qwen2.5:1.5b` | Ollama model for answer generation |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer for embeddings |
| `CHUNK_SIZE` | `600` | Characters per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between chunks |
| `TOP_K_RESULTS` | `5` | Chunks retrieved per query |
| `LLM_NUM_CTX` | `4096` | LLM context window (tokens) |

---

## 🛠️ Troubleshooting

**Poor answers / model not responding:**
```bash
# Check Ollama is running
ollama serve

# Verify model is downloaded
ollama list

# Pull if missing
ollama pull qwen2.5:1.5b
```

**CUDA / GPU issues:**
```bash
# Verify CUDA is available
python -c "import torch; print(torch.cuda.is_available())"
```

**Dependency conflicts:**
```bash
# Always install in a fresh virtual environment
deactivate
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install torch==2.2.2+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**Stale vector DB causing wrong answers:**
```bash
rm -rf data/vector_db/chroma_db
```
Then re-upload and re-process your files.

---

## 🚧 Known Limitations & Future Improvements

1. **Image processing is basic** — images are described by metadata, not vision. Upgrade path: use CLIP or LLaVA via Ollama for true image understanding.

2. **No hybrid search** — currently uses only dense (semantic) retrieval. Adding BM25 keyword search alongside would improve results for queries with specific keywords.

3. **Single-GPU memory constraint** — on 4GB VRAM, only `all-MiniLM-L6-v2` fits comfortably. Upgrade to RTX 3060+ to use `all-mpnet-base-v2` for better embedding quality.

4. **LLM on CPU** — Ollama runs Qwen2.5 on CPU. On machines with >6GB VRAM (after embeddings), Ollama can offload layers to GPU using `OLLAMA_GPU_LAYERS` for faster responses.

---

## 📚 Tech Stack

| Component | Library | Role |
|-----------|---------|------|
| UI | Streamlit 1.32 | Web interface |
| Embeddings | sentence-transformers 2.6.1 | Text → vectors (GPU) |
| Vector DB | ChromaDB 0.4.24 | Similarity search |
| Audio | OpenAI Whisper (tiny) | Speech → text (CPU) |
| LLM | qwen2.5:1.5b via Ollama | Answer generation (CPU) |
| Doc parsing | PyPDF2, python-docx | Text extraction |
