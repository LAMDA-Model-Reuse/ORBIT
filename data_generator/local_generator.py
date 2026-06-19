"""
Local LLM Generator using HuggingFace Transformers directly.
No need for vLLM or API servers - just load and run!
"""
import time
import torch
from typing import Optional, Dict, Any, List
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer
)

from models import (
    LLMResponse,
    TokenUsage,
    TimingInfo,
    ModelInfo
)
from utils import (
    generate_query_id,
    get_timestamp,
    ensure_dir,
    save_jsonl,
    load_jsonl,
    sanitize_error_message
)


class LocalLLMGenerator:
    """
    Direct local LLM inference using HuggingFace Transformers.
    No API server needed - just point to your downloaded model!
    """

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        torch_dtype: str = "auto",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        trust_remote_code: bool = False,
        output_dir: str = "./data",
        response_file: str = "responses.jsonl",
        append_mode: bool = True,
        verbose: bool = True
    ):
        """
        Initialize the local LLM generator.

        Args:
            model_path: Path to your local model or HuggingFace model name
                       Examples:
                       - "/path/to/Llama-3.1-8B-Instruct"
                       - "meta-llama/Meta-Llama-3.1-8B-Instruct"
            device: Device to use ("auto", "cuda", "cpu", "mps")
            torch_dtype: Data type ("auto", "float16", "bfloat16", "float32")
            load_in_8bit: Load model in 8-bit (saves memory)
            load_in_4bit: Load model in 4-bit (saves even more memory)
            trust_remote_code: Whether to trust remote code (needed for some models)
            output_dir: Directory to save responses
            response_file: Filename for responses
            append_mode: Whether to append to existing file
            verbose: Whether to print verbose output
        """
        self.model_path = model_path
        self.output_dir = output_dir
        self.response_file = response_file
        self.append_mode = append_mode
        self.verbose = verbose

        # Ensure output directory exists
        ensure_dir(self.output_dir)

        # Determine dtype
        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(torch_dtype, "auto")

        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Loading model: {model_path}")
            print(f"Device: {device}")
            print(f"Dtype: {torch_dtype}")
            if load_in_8bit:
                print("Quantization: 8-bit")
            elif load_in_4bit:
                print("Quantization: 4-bit")
            print(f"{'='*80}\n")

        # Load tokenizer
        if self.verbose:
            print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code
        )

        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        if self.verbose:
            print("Loading model (this may take a while)...")

        model_kwargs = {
            "trust_remote_code": trust_remote_code,
        }

        # Add quantization options
        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
            model_kwargs["device_map"] = device if device != "auto" else "auto"
        elif load_in_4bit:
            model_kwargs["load_in_4bit"] = True
            model_kwargs["device_map"] = device if device != "auto" else "auto"
        else:
            if dtype != "auto":
                model_kwargs["torch_dtype"] = dtype
            if device == "auto":
                model_kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs
        )

        # Move to device if not using device_map
        if "device_map" not in model_kwargs and device != "auto":
            self.model = self.model.to(device)

        # Set to eval mode
        self.model.eval()

        if self.verbose:
            print("Model loaded successfully!\n")

    def generate_single(
        self,
        query: str,
        temperature: float = 1.0,
        max_tokens: int = 2048,
        top_p: float = 1.0,
        top_k: int = 50,
        do_sample: bool = True,
        save_result: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a single response.

        Args:
            query: Input query/prompt
            temperature: Sampling temperature (higher = more random)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling (set False for greedy)
            save_result: Whether to save to file
            metadata: Additional metadata
            **kwargs: Additional generation parameters

        Returns:
            LLMResponse object
        """
        query_id = generate_query_id()
        start_time = datetime.now()
        start_time_iso = start_time.isoformat()

        try:
            # Format prompt (apply chat template if available)
            if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
                messages = [{"role": "user", "content": query}]
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                formatted_prompt = query

            if self.verbose:
                print(f"Query: {query[:100]}...")

            # Tokenize
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                padding=True
            ).to(self.model.device)

            prompt_tokens = len(inputs['input_ids'][0])

            # Generate
            with torch.no_grad():
                generation_kwargs = {
                    "max_new_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "do_sample": do_sample,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                }
                generation_kwargs.update(kwargs)

                outputs = self.model.generate(
                    **inputs,
                    **generation_kwargs
                )

            # Decode
            full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Remove the input prompt from output
            if full_output.startswith(formatted_prompt):
                response_text = full_output[len(formatted_prompt):].strip()
            else:
                # Fallback: try to remove original query
                if full_output.startswith(query):
                    response_text = full_output[len(query):].strip()
                else:
                    response_text = full_output

            completion_tokens = len(outputs[0]) - prompt_tokens
            total_tokens = len(outputs[0])

            # Track end time
            end_time = datetime.now()
            end_time_iso = end_time.isoformat()
            duration_seconds = (end_time - start_time).total_seconds()

            # Create response object
            llm_response = LLMResponse(
                query_id=query_id,
                query=query,
                response=response_text,
                model_info=ModelInfo(
                    model_name=self.model_path,
                    provider="local_transformers",
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    other_params={"top_k": top_k, "do_sample": do_sample, **kwargs}
                ),
                token_usage=TokenUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens
                ),
                timing_info=TimingInfo(
                    start_time=start_time_iso,
                    end_time=end_time_iso,
                    duration_seconds=duration_seconds,
                    duration_ms=duration_seconds * 1000
                ),
                timestamp=get_timestamp(),
                success=True,
                error_message=None,
                metadata=metadata
            )

            if self.verbose:
                print(f"Response: {response_text[:200]}...")
                print(f"Tokens: {total_tokens} ({prompt_tokens} prompt + {completion_tokens} completion)")
                print(f"Time: {duration_seconds:.2f}s")
                print(f"Tokens/sec: {completion_tokens/duration_seconds:.2f}")
                print()

        except Exception as e:
            # Handle errors
            end_time = datetime.now()
            end_time_iso = end_time.isoformat()
            duration_seconds = (end_time - start_time).total_seconds()

            llm_response = LLMResponse(
                query_id=query_id,
                query=query,
                response="",
                model_info=ModelInfo(
                    model_name=self.model_path,
                    provider="local_transformers",
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    other_params={"top_k": top_k, "do_sample": do_sample}
                ),
                token_usage=TokenUsage(
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0
                ),
                timing_info=TimingInfo(
                    start_time=start_time_iso,
                    end_time=end_time_iso,
                    duration_seconds=duration_seconds,
                    duration_ms=duration_seconds * 1000
                ),
                timestamp=get_timestamp(),
                success=False,
                error_message=sanitize_error_message(e),
                metadata=metadata
            )

            if self.verbose:
                print(f"ERROR: {sanitize_error_message(e)}\n")

        # Save if requested
        if save_result:
            self._save_response(llm_response)

        return llm_response

    def generate_batch(
        self,
        queries: List[str],
        temperature: float = 1.0,
        max_tokens: int = 2048,
        top_p: float = 1.0,
        save_results: bool = True,
        show_progress: bool = True,
        **kwargs
    ) -> List[LLMResponse]:
        """
        Generate responses for a batch of queries.

        Args:
            queries: List of queries
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            top_p: Nucleus sampling parameter
            save_results: Whether to save results
            show_progress: Whether to show progress
            **kwargs: Additional generation parameters

        Returns:
            List of LLMResponse objects
        """
        from tqdm import tqdm

        responses = []
        iterator = tqdm(queries, desc="Processing queries") if show_progress else queries

        for query in iterator:
            response = self.generate_single(
                query=query,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                save_result=save_results,
                **kwargs
            )
            responses.append(response)

        return responses

    def _save_response(self, response: LLMResponse) -> None:
        """Save response to file."""
        filepath = f"{self.output_dir}/{self.response_file}"
        save_jsonl([response.to_dict()], filepath, append=self.append_mode)

    def load_responses(self) -> List[LLMResponse]:
        """Load all saved responses."""
        filepath = f"{self.output_dir}/{self.response_file}"
        data = load_jsonl(filepath)
        return [LLMResponse.from_dict(item) for item in data]

    def clear_memory(self):
        """Clear GPU memory (call this if you need to free up memory)."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
