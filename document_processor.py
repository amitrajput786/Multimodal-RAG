"""
document_processor.py — Handles text extraction from PDF, DOCX, TXT files
==========================================================================
Extracts raw text and splits it into overlapping chunks for embedding.
"""

from pathlib import Path
from typing import List, Dict, Any
import logging

import PyPDF2
from docx import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Extracts and chunks text from documents."""

    def __init__(self):
        self.supported_formats = {
            ".pdf":  self._process_pdf,
            ".docx": self._process_docx,
            ".txt":  self._process_txt,
        }

    # ── Public API ────────────────────────────────────────────────────────────

    def process_document(self, file_path: Path) -> Dict[str, Any]:
        """
        Process a document file and return extracted text.

        Returns:
            {
                'file_info': {...},
                'content': {'text': [{'page': 1, 'text': '...'}, ...]},
                'processing_status': 'success' | 'error'
            }
        """
        try:
            ext = file_path.suffix.lower()
            if ext not in self.supported_formats:
                raise ValueError(f"Unsupported format: {ext}")

            file_info = {
                "name":      file_path.name,
                "size":      file_path.stat().st_size,
                "extension": ext,
            }

            content = self.supported_formats[ext](file_path)
            logger.info(f"✅ Processed '{file_path.name}': {len(content['text'])} sections")

            return {
                "file_info":          file_info,
                "content":            content,
                "processing_status":  "success",
            }

        except Exception as e:
            logger.error(f"❌ Error processing {file_path.name}: {e}")
            return {
                "file_info":          {"name": file_path.name, "size": 0, "extension": ""},
                "content":            {"text": []},
                "processing_status":  "error",
                "error_message":      str(e),
            }

    def chunk_text(
        self,
        text_data: List[Dict],
        chunk_size: int = 600,
        overlap: int = 150,
    ) -> List[Dict[str, Any]]:
        """
        Split extracted text sections into overlapping chunks.

        Args:
            text_data:  List of {'text': '...', ...} dicts from process_document
            chunk_size: Max characters per chunk
            overlap:    Overlap between consecutive chunks (prevents mid-idea cuts)

        Returns:
            List of {'text': str, 'source': dict, 'chunk_id': int}
        """
        chunks = []
        chunk_id = 0

        for item in text_data:
            text = item.get("text", "").strip()
            if not text:
                continue

            source_meta = {k: v for k, v in item.items() if k != "text"}
            start = 0

            while start < len(text):
                end = min(start + chunk_size, len(text))
                chunk_text = text[start:end].strip()

                if chunk_text:
                    chunks.append({
                        "text":     chunk_text,
                        "source":   source_meta,
                        "chunk_id": chunk_id,
                    })
                    chunk_id += 1

                # Move forward by (chunk_size - overlap) so chunks overlap
                start += max(1, chunk_size - overlap)

        logger.info(f"  → Created {len(chunks)} chunks")
        return chunks

    # ── Private Processors ────────────────────────────────────────────────────

    def _process_pdf(self, file_path: Path) -> Dict[str, Any]:
        text_content = []
        try:
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page_num, page in enumerate(reader.pages):
                    try:
                        text = page.extract_text()
                        if text and text.strip():
                            text_content.append({
                                "page": page_num + 1,
                                "text": text.strip(),
                            })
                    except Exception as e:
                        logger.warning(f"  ⚠️ Skipped page {page_num + 1}: {e}")
        except Exception as e:
            logger.error(f"PDF read error: {e}")
        return {"text": text_content}

    def _process_docx(self, file_path: Path) -> Dict[str, Any]:
        text_content = []
        try:
            doc = Document(file_path)
            for i, para in enumerate(doc.paragraphs):
                if para.text.strip():
                    text_content.append({
                        "paragraph": i + 1,
                        "text":      para.text.strip(),
                    })
        except Exception as e:
            logger.error(f"DOCX read error: {e}")
        return {"text": text_content}

    def _process_txt(self, file_path: Path) -> Dict[str, Any]:
        for encoding in ("utf-8", "latin-1"):
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    content = f.read().strip()
                if content:
                    return {"text": [{"line": 1, "text": content}]}
                return {"text": []}
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.error(f"TXT read error: {e}")
                return {"text": []}
        return {"text": []}
