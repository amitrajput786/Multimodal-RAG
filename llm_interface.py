"""
llm_interface.py — Ollama LLM interface
=========================================
Communicates with Ollama to generate answers from retrieved context.

Current model: qwen2.5:1.5b
  → Replaced llama3.2:1b which gave incoherent RAG answers
  → Qwen2.5 series has significantly better instruction-following

To switch models (e.g., when a better open-source model releases):
  1. ollama pull <new_model>
  2. Change DEFAULT_LLM_MODEL in config.py
  → Nothing else needs to change — this interface handles any Ollama model.
"""

import logging
import requests
from typing import List, Dict, Any, Optional

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMInterface:
    """
    Handles communication with the Ollama local LLM server.
    Model runs on CPU via Ollama — keeps GPU free for embeddings.
    """

    def __init__(
        self,
        base_url:   str = None,
        model_name: str = None,
    ):
        self.base_url   = (base_url   or config.OLLAMA_BASE_URL).rstrip("/")
        self.model_name = model_name  or config.DEFAULT_LLM_MODEL
        self.session    = requests.Session()
        self._check_status()

    # ── Status Check ─────────────────────────────────────────────────────────

    def _check_status(self):
        """Verify Ollama is running and the model is available."""
        try:
            resp = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]

                # Normalize: Ollama sometimes stores as "qwen2.5:1.5b" or "qwen2.5:1.5b-instruct"
                model_found = any(self.model_name in m or m in self.model_name for m in models)

                if model_found:
                    logger.info(f"✅ Ollama ready | Model: {self.model_name}")
                else:
                    logger.warning(
                        f"⚠️  Model '{self.model_name}' not found in Ollama.\n"
                        f"   Available: {models}\n"
                        f"   Run: ollama pull {self.model_name}"
                    )
        except requests.exceptions.ConnectionError:
            logger.error(
                "❌ Cannot connect to Ollama.\n"
                "   Start it with: ollama serve"
            )
        except Exception as e:
            logger.error(f"❌ Ollama check failed: {e}")

    # ── Core Generation ───────────────────────────────────────────────────────

    def generate_response(
        self,
        prompt:  str,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a response from the LLM.

        Args:
            prompt:  The user's question
            context: Retrieved document chunks to ground the answer

        Returns:
            {'response': str, 'model': str, 'status': str}
        """
        try:
            full_prompt = self._build_prompt(prompt, context)

            payload = {
                "model":  self.model_name,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature":    config.LLM_TEMPERATURE,
                    "num_predict":    config.LLM_NUM_PREDICT,
                    "top_p":          0.9,
                    "repeat_penalty": 1.1,
                    "num_ctx":        config.LLM_NUM_CTX,
                },
            }

            resp = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=240,   # Qwen2.5:1.5b is fast but give headroom
            )

            if resp.status_code == 200:
                answer = resp.json().get("response", "").strip()
                return {"response": answer, "model": self.model_name, "status": "success"}
            else:
                logger.error(f"Ollama API error {resp.status_code}: {resp.text}")
                return self._fallback_response()

        except requests.exceptions.Timeout:
            logger.error("LLM request timed out")
            return {"response": "Request timed out. Try a shorter question or smaller context.", 
                    "model": self.model_name, "status": "timeout"}
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return self._fallback_response()

    def answer_question_with_context(
        self,
        question:     str,
        context_docs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Answer a question using retrieved document chunks.

        Args:
            question:     User's question
            context_docs: List of {'text': ..., 'metadata': ..., 'similarity_score': ...}

        Returns:
            Response dict with answer, citations, and source count
        """
        try:
            context_parts = []
            citations     = []

            for i, doc in enumerate(context_docs):
                text     = doc.get("text", "")
                metadata = doc.get("metadata", {})

                context_parts.append(f"[Source {i+1}] {text}")
                citations.append({
                    "index":      i + 1,
                    "source":     metadata.get("source_file", "Unknown"),
                    "type":       metadata.get("content_type", "text"),
                    "similarity": round(doc.get("similarity_score", 0.0), 3),
                })

            context = "\n\n".join(context_parts)
            result  = self.generate_response(question, context)
            result["citations"]   = citations
            result["num_sources"] = len(citations)
            return result

        except Exception as e:
            logger.error(f"Error in answer_with_context: {e}")
            return {
                "response":    "Error processing your question. Please try again.",
                "model":       self.model_name,
                "status":      "error",
                "citations":   [],
                "num_sources": 0,
            }

    def list_available_models(self) -> List[str]:
        """Returns list of models currently available in Ollama."""
        try:
            resp = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            pass
        return []

    # ── Prompt Engineering ────────────────────────────────────────────────────

    def _build_prompt(self, question: str, context: Optional[str] = None) -> str:
        """
        Construct a grounded RAG prompt optimized for Qwen2.5.

        Qwen2.5 responds well to clear role + instruction + structured context.
        Temperature 0.3 keeps it factual rather than creative.
        """
        if context:
            return f"""You are a helpful document assistant. Your job is to answer questions based strictly on the provided context.

CONTEXT FROM DOCUMENTS:
{context}

USER QUESTION: {question}

INSTRUCTIONS:
- Answer based ONLY on the context above
- Be specific and detailed — use information directly from the context
- If the question asks for a summary, provide a structured summary of all topics covered
- Cite which source number your information comes from (e.g., "According to [Source 1]...")
- If the context does not contain enough information, say what you did find and what is missing

ANSWER:"""
        else:
            return f"""You are a helpful assistant.

Question: {question}

Answer:"""

    def _fallback_response(self) -> Dict[str, Any]:
        return {
            "response": (
                "⚠️ Could not connect to the language model.\n\n"
                "Please check:\n"
                "1. Ollama is running: `ollama serve`\n"
                f"2. Model is installed: `ollama pull {self.model_name}`\n"
                "3. Ollama URL is correct in config.py"
            ),
            "model":  self.model_name,
            "status": "error",
        }
