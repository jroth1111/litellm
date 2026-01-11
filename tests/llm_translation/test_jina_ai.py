import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path


from base_rerank_unit_tests import BaseLLMRerankTest
import litellm


class TestJinaAI(BaseLLMRerankTest):
    def get_custom_llm_provider(self) -> litellm.LlmProviders:
        return litellm.LlmProviders.JINA_AI

    def get_base_rerank_call_args(self) -> dict:
        return {
            "model": "jina_ai/jina-reranker-v2-base-multilingual",
        }


def test_jina_ai_embedding():
    if not (os.getenv("JINA_AI_API_KEY") or os.getenv("JINA_AI_TOKEN")):
        pytest.skip("JINA_AI_API_KEY/JINA_AI_TOKEN not set")
    litellm.embedding(
        model="jina_ai/jina-embeddings-v3",
        input=["a"],
        task="separation",
        dimensions=1024,
    )
