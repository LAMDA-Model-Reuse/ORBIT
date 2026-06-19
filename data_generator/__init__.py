"""
LLM Data Generator

A unified tool for generating and tracking LLM responses across multiple providers.
"""

from .generator import LLMDataGenerator
from .models import (
    LLMResponse,
    TokenUsage,
    TimingInfo,
    ModelInfo,
    QueryBatch,
    GeneratorConfig
)
from .utils import (
    load_config,
    save_jsonl,
    load_jsonl,
    generate_query_id,
    get_timestamp,
    calculate_cost
)

__version__ = "0.1.0"
__all__ = [
    "LLMDataGenerator",
    "LLMResponse",
    "TokenUsage",
    "TimingInfo",
    "ModelInfo",
    "QueryBatch",
    "GeneratorConfig",
    "load_config",
    "save_jsonl",
    "load_jsonl",
    "generate_query_id",
    "get_timestamp",
    "calculate_cost"
]
