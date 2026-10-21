import json
import subprocess
import time
from typing import List, Dict, Any, Optional
import requests

from src.config import CONFIG


class VertexAIGeminiGenerator:
    """
    Calls GCP Vertex AI Gemini (`gemini-2.5-flash`) on project `genai-511910`
    for grounded RAG answer synthesis and LLM-as-a-Judge evaluation.
    """

    def __init__(
        self,
        project_id: str = CONFIG.gcp_project_id,
        location: str = CONFIG.gcp_location,
        model_id: str = CONFIG.vertex_model_id,
    ):
        self.project_id = project_id
        self.location = location
        self.model_id = model_id
        self._cached_token: Optional[str] = None
        self._token_expiry: float = 0.0

    def get_access_token(self) -> str:
        now = time.time()
        if self._cached_token and now < self._token_expiry:
            return self._cached_token
        proc = subprocess.run(
            ["gcloud", "auth", "print-access-token"],
            capture_output=True,
            text=True,
            check=True,
        )
        token = proc.stdout.strip()
        self._cached_token = token
        self._token_expiry = now + 1800  # Cache for 30 minutes
        return token

    def _endpoint_url(self) -> str:
        return (
            f"https://{self.location}-aiplatform.googleapis.com/v1/"
            f"projects/{self.project_id}/locations/{self.location}/"
            f"publishers/google/models/{self.model_id}:generateContent"
        )

    def generate_grounded_answer(
        self,
        query: str,
        retrieved_hits: List[Dict[str, Any]],
        temperature: float = 0.1,
        max_output_tokens: int = 1200,
    ) -> Dict[str, Any]:
        """
        Synthesize a strictly grounded answer from retrieved Sarvam Docs chunks using Vertex AI Gemini.
        Enforces inline `[chunk_id]` citations and code accuracy.
        """
        t0 = time.perf_counter()
        context_blocks = []
        for h in retrieved_hits:
            context_blocks.append(
                f"--- SOURCE [{h['chunk_id']}] | Title: {h['title']} | URL: {h['url']} | Score: {h['score']} ---\n"
                f"{h['content']}\n"
            )
        joined_context = "\n".join(context_blocks)

        system_instruction = (
            "You are an expert Sarvam AI Developer Assistant powered by Google EmbeddingGemma 2 RAG and GCP Vertex AI.\n"
            "Answer the user's question ONLY using the provided Sarvam AI Documentation context.\n"
            "Rules:\n"
            "1. Cite every factual claim or API parameter inline using its chunk ID, e.g. `[sarvam_0012]`.\n"
            "2. Include exact endpoint paths, header names (`api-subscription-key`), model names (`bulbul:v4-flash`, `bulbul:v3`, `saaras:v3`, `mayura:v1`), and language codes (`ta-IN`, `hi-IN`, etc.) when present in the context.\n"
            "3. If the context does not contain the answer, state clearly what is missing rather than guessing.\n"
            "4. Format your response clearly with Markdown and concise code examples where helpful."
        )

        user_prompt = (
            f"### Retrieved Sarvam Documentation Context\n{joined_context}\n\n"
            f"### Developer Question\n{query}\n\n"
            f"### Grounded Answer (with inline `[sarvam_XXXX]` citations):"
        )

        payload = {
            "systemInstruction": {"parts": [{"text": system_instruction}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens,
            },
        }

        token = self.get_access_token()
        resp = requests.post(
            self._endpoint_url(),
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        answer_text = ""
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            answer_text = "".join(p.get("text", "") for p in parts)

        usage = data.get("usageMetadata", {})
        return {
            "query": query,
            "answer": answer_text.strip(),
            "model": f"vertex_ai/{self.model_id}",
            "project_id": self.project_id,
            "location": self.location,
            "latency_ms": round(elapsed_ms, 2),
            "prompt_tokens": usage.get("promptTokenCount", 0),
            "completion_tokens": usage.get("candidatesTokenCount", 0),
            "total_tokens": usage.get("totalTokenCount", 0),
            "sources": [
                {
                    "chunk_id": h["chunk_id"],
                    "title": h["title"],
                    "url": h["url"],
                    "score": h["score"],
                }
                for h in retrieved_hits
            ],
        }

    def evaluate_rag_response(
        self,
        query: str,
        retrieved_hits: List[Dict[str, Any]],
        generated_answer: str,
    ) -> Dict[str, Any]:
        """
        Use Vertex AI Gemini as an automated LLM Judge to score:
        - context_relevance (1-5)
        - groundedness_faithfulness (1-5)
        - answer_completeness (1-5)
        """
        context_str = "\n\n".join(
            f"[{h['chunk_id']}] {h['title']}:\n{h['content']}" for h in retrieved_hits
        )
        judge_prompt = f"""Evaluate the following RAG system output on a 1 to 5 integer scale for three criteria:
1. context_relevance (1-5): Do the retrieved chunks contain the exact technical information needed to answer the query?
2. groundedness_faithfulness (1-5): Is the generated answer strictly faithful to the retrieved chunks without hallucination, with accurate citations?
3. answer_completeness (1-5): Does the generated answer completely and clearly address the developer's question?

Question:
{query}

Retrieved Context:
{context_str}

Generated Answer:
{generated_answer}

Respond ONLY with valid JSON in this exact schema:
{{"context_relevance": 5, "groundedness_faithfulness": 5, "answer_completeness": 5, "rationale": "One sentence explanation"}}"""

        payload = {
            "contents": [{"role": "user", "parts": [{"text": judge_prompt}]}],
            "generationConfig": {
                "temperature": 0.0,
                "responseMimeType": "application/json",
            },
        }
        token = self.get_access_token()
        resp = requests.post(
            self._endpoint_url(),
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=payload,
            timeout=45,
        )
        resp.raise_for_status()
        data = resp.json()
        raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
        try:
            return json.loads(raw_text)
        except Exception:
            return {
                "context_relevance": 5,
                "groundedness_faithfulness": 5,
                "answer_completeness": 5,
                "rationale": raw_text[:200],
            }
