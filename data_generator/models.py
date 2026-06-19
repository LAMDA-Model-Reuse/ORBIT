"""
Data models for LLM response tracking.
"""
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from datetime import datetime


class TokenUsage(BaseModel):
    """Token usage statistics."""
    prompt_tokens: int = Field(description="Number of tokens in the prompt")
    completion_tokens: int = Field(description="Number of tokens in the completion")
    total_tokens: int = Field(description="Total number of tokens used")


class TimingInfo(BaseModel):
    """Timing information for the request."""
    start_time: str = Field(description="Request start time (ISO format)")
    end_time: str = Field(description="Request end time (ISO format)")
    duration_seconds: float = Field(description="Total duration in seconds")
    duration_ms: float = Field(description="Total duration in milliseconds")


class ModelInfo(BaseModel):
    """Information about the model used."""
    model_name: str = Field(description="Name/identifier of the model")
    provider: str = Field(description="Provider (openai, anthropic, etc.)")
    temperature: Optional[float] = Field(default=None, description="Temperature setting")
    max_tokens: Optional[int] = Field(default=None, description="Max tokens setting")
    top_p: Optional[float] = Field(default=None, description="Top-p setting")
    other_params: Optional[Dict[str, Any]] = Field(default=None, description="Other parameters")


class LLMResponse(BaseModel):
    """Complete LLM response with metadata."""
    query_id: str = Field(description="Unique identifier for this query")
    query: str = Field(description="The input query/prompt")
    response: str = Field(description="The model's response")
    model_info: ModelInfo = Field(description="Information about the model")
    token_usage: TokenUsage = Field(description="Token usage statistics")
    timing_info: TimingInfo = Field(description="Timing information")
    timestamp: str = Field(description="When this response was generated (ISO format)")
    success: bool = Field(default=True, description="Whether the request succeeded")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMResponse":
        """Create from dictionary."""
        return cls(**data)


class QueryBatch(BaseModel):
    """A batch of queries to process."""
    batch_id: str = Field(description="Unique identifier for this batch")
    queries: List[str] = Field(description="List of queries to process")
    model_name: str = Field(description="Model to use for all queries")
    timestamp: str = Field(description="When this batch was created (ISO format)")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class GeneratorConfig(BaseModel):
    """Configuration for the data generator."""
    output_dir: str = Field(default="./data", description="Output directory")
    query_file: str = Field(default="queries.jsonl", description="Query file name")
    response_file: str = Field(default="responses.jsonl", description="Response file name")
    append_mode: bool = Field(default=True, description="Append to existing files")
    verbose: bool = Field(default=True, description="Print verbose output")
    max_retries: int = Field(default=3, description="Maximum retries on failure")
    retry_delay: float = Field(default=2.0, description="Delay between retries (seconds)")
    exponential_backoff: bool = Field(default=True, description="Use exponential backoff")
