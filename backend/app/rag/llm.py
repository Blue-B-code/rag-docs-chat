"""Minimal OpenAI-compatible chat client (DeepSeek, OpenAI, ...)."""
import httpx


def build_answer_messages(question: str, sources: list[dict]) -> list[dict]:
    """Build chat messages that ground the answer in the retrieved chunks."""
    system = (
        "You are a precise assistant. Answer the user's question using ONLY the "
        "context provided below. If the context does not contain the answer, say "
        "so honestly. Cite each fact with its source reference like [1], [2]."
    )
    context_lines = []
    for i, source in enumerate(sources, start=1):
        name = source.get("source") or "unknown"
        text = source.get("text", "")
        context_lines.append(f"[{i}] ({name})\n{text}")
    user = "Context:\n" + "\n\n".join(context_lines) + f"\n\nQuestion: {question}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


class LLMClient:
    """Thin wrapper around an OpenAI-compatible /chat/completions endpoint."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.3,
        max_tokens: int = 512,
        transport: httpx.BaseTransport | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._transport = transport

    def complete(self, messages: list[dict]) -> str:
        if not self.api_key:
            raise RuntimeError("LLM_API_KEY is not configured")
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        with httpx.Client(timeout=60, transport=self._transport) as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
        return data["choices"][0]["message"]["content"].strip()
