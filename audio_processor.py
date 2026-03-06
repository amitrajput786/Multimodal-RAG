"""
audio_processor.py — Audio transcription using OpenAI Whisper
==============================================================
Transcribes .wav and .mp3 files and splits them into chunks.
Whisper runs on CPU to keep GPU VRAM free for embeddings.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioProcessor:
    """Transcribes audio files and chunks the transcription."""

    def __init__(self, model_size: str = None):
        self.model_size       = model_size or config.WHISPER_MODEL
        self.model            = None
        self.supported_formats = {".wav", ".mp3"}

    def _load_model(self):
        """Lazy-load Whisper (runs on CPU)."""
        if self.model is None:
            logger.info(f"Loading Whisper model: {self.model_size} (CPU)")
            import whisper
            self.model = whisper.load_model(self.model_size, device="cpu")
            logger.info("✅ Whisper loaded")
        return self.model

    # ── Public API ────────────────────────────────────────────────────────────

    def process_audio(self, file_path: Path) -> Dict[str, Any]:
        """
        Transcribe an audio file.

        Returns:
            {
                'transcription': str,
                'segments': [...],
                'language': str,
                'processing_status': 'success' | 'error'
            }
        """
        try:
            if file_path.suffix.lower() not in self.supported_formats:
                raise ValueError(f"Unsupported audio format: {file_path.suffix}")

            model = self._load_model()
            logger.info(f"Transcribing: {file_path.name}")

            result = model.transcribe(str(file_path), fp16=False)  # fp16=False for CPU

            logger.info(f"✅ Transcription complete: {len(result.get('text', ''))} chars")
            return {
                "transcription":      result.get("text", "").strip(),
                "segments":           result.get("segments", []),
                "language":           result.get("language", "unknown"),
                "processing_status":  "success",
            }

        except Exception as e:
            logger.error(f"❌ Audio processing error: {e}")
            return {
                "transcription":     "",
                "segments":          [],
                "language":          "unknown",
                "processing_status": "error",
                "error_message":     str(e),
            }

    def chunk_transcription(
        self,
        transcription_data: Dict[str, Any],
        chunk_duration: float = 20.0,
    ) -> List[Dict[str, Any]]:
        """
        Split transcription into time-based chunks.

        Args:
            transcription_data: Output from process_audio()
            chunk_duration:     Target duration (seconds) per chunk

        Returns:
            List of {'text': str, 'start_time': float, 'end_time': float, ...}
        """
        segments = transcription_data.get("segments", [])
        language = transcription_data.get("language", "unknown")

        # No segments — treat whole transcription as one chunk
        if not segments:
            text = transcription_data.get("transcription", "").strip()
            if not text:
                return []
            return [{"text": text, "start_time": 0.0, "end_time": 0.0,
                     "chunk_id": 0, "language": language}]

        chunks = []
        current_text  = ""
        current_start = segments[0]["start"]

        for seg in segments:
            if not current_text:
                current_start = seg["start"]

            # If accumulated chunk exceeds target duration, save and reset
            if (seg["end"] - current_start) > chunk_duration and current_text:
                chunks.append({
                    "text":       current_text.strip(),
                    "start_time": current_start,
                    "end_time":   seg["start"],
                    "chunk_id":   len(chunks),
                    "language":   language,
                })
                current_text  = ""
                current_start = seg["start"]

            current_text += seg["text"] + " "

        # Flush remaining text
        if current_text.strip():
            chunks.append({
                "text":       current_text.strip(),
                "start_time": current_start,
                "end_time":   segments[-1]["end"],
                "chunk_id":   len(chunks),
                "language":   language,
            })

        logger.info(f"  → Created {len(chunks)} audio chunks")
        return chunks

    def is_supported(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.supported_formats
