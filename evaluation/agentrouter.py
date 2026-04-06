import os
from typing import List, Optional


FALLBACK_OPENAI_BASE_URL = "https://agentrouter.org/v1"
DEFAULT_COMPLETION_MODEL = "claude-sonnet-4-20250514"
DEFAULT_CLAUDE_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"


def _get_openai_base_url() -> str:
    anthropic_base_url = (os.getenv("ANTHROPIC_BASE_URL") or "").strip()
    if anthropic_base_url:
        base_root = anthropic_base_url.rstrip("/")
        return base_root if base_root.endswith("/v1") else f"{base_root}/v1"
    return FALLBACK_OPENAI_BASE_URL


def get_completion(prompt: str, max_tokens: int = 512) -> str:
    last_error: Optional[Exception] = None

    api_key = (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN") or "").strip()
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set")

    try:
        import anthropic  # type: ignore

        client_kwargs = {}
        base_url = (os.getenv("ANTHROPIC_BASE_URL") or "").strip()
        if base_url:
            client_kwargs["base_url"] = base_url
        client_kwargs["api_key"] = api_key

        client = anthropic.Anthropic(**client_kwargs)
        response = client.messages.create(
            model=(os.getenv("CLAUDE_MODEL") or DEFAULT_CLAUDE_MODEL),
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    except Exception as exc:
        print(f"Completion provider claude failed: {exc}")
        last_error = exc

    raise RuntimeError(f"All providers failed for completion. Last error: {last_error}")


def get_embedding(text: str) -> List[float]:
    last_error: Optional[Exception] = None
    model = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set")

    try:
        from openai import OpenAI  # type: ignore

        anthropic_base_url = (os.getenv("ANTHROPIC_BASE_URL") or "").strip()
        if anthropic_base_url:
            base_root = anthropic_base_url.rstrip("/")
            openai_base_url = base_root if base_root.endswith("/v1") else f"{base_root}/v1"
        else:
            openai_base_url = FALLBACK_OPENAI_BASE_URL
        client = OpenAI(api_key=api_key, base_url=openai_base_url)
        response = client.embeddings.create(model=model, input=text)
        if response.data:
            return response.data[0].embedding
        raise RuntimeError("OpenAI embedding response contained no vectors")
    except Exception as exc:
        print(f"Embedding provider openai failed: {exc}")
        last_error = exc

    raise RuntimeError(f"All providers failed for embedding. Last error: {last_error}")
