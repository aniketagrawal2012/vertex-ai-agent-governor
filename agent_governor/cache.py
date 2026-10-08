"""
Vertex AI Context Caching Utilities
===================================
Manages creation, inspection, and reuse of Vertex AI CachedContent resources
using the official google.genai SDK (v2.10.0).
"""

from typing import List, Optional
from google import genai
from google.genai import types


class ContextCacheManager:
    """
    Manages long-lived context caches on Google Cloud Vertex AI.
    
    Caches large static prompts (runbooks, OpenAPI schemas, policies >32K tokens)
    to achieve up to 75% input token discount and 70% lower time-to-first-token (TTFT).
    """
    def __init__(self, client: Optional[genai.Client] = None, model: str = "gemini-2.5-flash"):
        self.client = client
        self.model = model
        self.active_cache_id: Optional[str] = None

    def create_policy_cache(
        self,
        contents: List[str],
        display_name: str = "agent-policy-cache",
        ttl_seconds: int = 3600,
    ) -> str:
        """
        Creates a CachedContent resource in Vertex AI with a designated TTL.
        Returns the unique cache name / resource URI.
        """
        if not self.client:
            raise RuntimeError("Google GenAI client must be initialized to create a context cache.")

        config = types.CreateCachedContentConfig(
            contents=contents,
            display_name=display_name,
            ttl=f"{ttl_seconds}s",
        )
        
        cached_content = self.client.caches.create(
            model=self.model,
            config=config,
        )
        self.active_cache_id = cached_content.name
        return cached_content.name

    def get_cache_config_kwarg(self) -> dict:
        """Returns the generate_content keyword argument for referencing this cache."""
        if not self.active_cache_id:
            return {}
        return {"cached_content": self.active_cache_id}
