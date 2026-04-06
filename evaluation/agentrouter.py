import os
from typing import Any, List, Optional


FALLBACK_OPENAI_BASE_URL = "https://agentrouter.org/v1"
DEFAULT_COMPLETION_MODEL = "gemini-1.5-flash"
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-20250514"
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"

_st_model: Optional[Any] = None


def _get_openai_base_url() -> str:
    anthropic_base_url = (os.getenv("ANTHROPIC_BASE_URL") or "").strip()
    if anthropic_base_url:
        base_root = anthropic_base_url.rstrip("/")
        return base_root if base_root.endswith("/v1") else f"{base_root}/v1"
    return FALLBACK_OPENAI_BASE_URL


def _get_st_model() -> Any:
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer  # type: ignore

        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _st_model


def get_completion(prompt: str, max_tokens: int = 512) -> str:
    last_error: Optional[Exception] = None

    try:
        gemini_key = (os.getenv("GEMINI_API_KEY") or "").strip()
        if not gemini_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        import google.generativeai as genai  # type: ignore

        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel(os.getenv("GEMINI_COMPLETION_MODEL", DEFAULT_COMPLETION_MODEL))
        response = model.generate_content(
            prompt,
            generation_config={"max_output_tokens": max_tokens},
        )
        text = getattr(response, "text", None)
        if isinstance(text, str) and text:
            return text
        raise RuntimeError("Gemini completion response missing text")
    except Exception as exc:
        print(f"Completion provider gemini failed: {exc}")
        last_error = exc

    try:
        import anthropic  # type: ignore

        client_kwargs = {}
        base_url = (os.getenv("ANTHROPIC_BASE_URL") or "").strip()
        api_key = (os.getenv("ANTHROPIC_AUTH_TOKEN") or os.getenv("ANTHROPIC_API_KEY") or "").strip()
        if base_url:
            client_kwargs["base_url"] = base_url
        if api_key:
            client_kwargs["api_key"] = api_key

        client = anthropic.Anthropic(**client_kwargs)
        response = client.messages.create(
            model=os.getenv("PARAPHRASE_MODEL", DEFAULT_CLAUDE_MODEL),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        return response.content[0].text
    except Exception as exc:
        print(f"Completion provider claude failed: {exc}")
        last_error = exc

    raise RuntimeError(f"All providers failed for completion. Last error: {last_error}")


def get_embedding(text: str) -> List[float]:
    last_error: Optional[Exception] = None
    model = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)

    try:
        gemini_key = (os.getenv("GEMINI_API_KEY") or "").strip()
        if not gemini_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        import google.generativeai as genai  # type: ignore

        genai.configure(api_key=gemini_key)
        result = genai.embed_content(model=model, content=text)
        embedding = result["embedding"]
        if isinstance(embedding, list) and embedding:
            return [float(x) for x in embedding]
        raise RuntimeError("Gemini embedding response missing 'embedding'")
    except Exception as exc:
        print(f"Embedding provider gemini failed: {exc}")
        last_error = exc

    try:
        from openai import OpenAI  # type: ignore

        token = (os.getenv("ANTHROPIC_API_KEY") or "").strip()
        if not token:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        client = OpenAI(api_key=token, base_url=_get_openai_base_url())
        response = client.embeddings.create(model=model, input=text)
        embedding = response.data[0].embedding
        return [float(x) for x in embedding]
    except Exception as exc:
        print(f"Embedding provider openai failed: {exc}")
        last_error = exc

    try:
        local_model = _get_st_model()
        embedding = local_model.encode(text).tolist()
        if isinstance(embedding, list) and embedding:
            return [float(x) for x in embedding]
        raise RuntimeError("Local sentence-transformers embedding is empty")
    except Exception as exc:
        print(f"Embedding provider sentence-transformers failed: {exc}")
        last_error = exc

    raise RuntimeError(f"All providers failed for embedding. Last error: {last_error}")
