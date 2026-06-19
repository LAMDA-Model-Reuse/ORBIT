"""
Main LLM Data Generator class.
"""
import time
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import litellm
from tqdm import tqdm

from models import (
    LLMResponse,
    TokenUsage,
    TimingInfo,
    ModelInfo,
    GeneratorConfig
)
from utils import (
    load_config,
    ensure_dir,
    save_jsonl,
    load_jsonl,
    generate_query_id,
    get_timestamp,
    print_response_summary,
    get_model_config,
    calculate_cost,
    sanitize_error_message
)


class LLMDataGenerator:
    """
    LLM Data Generator for collecting responses from multiple LLM providers.
    """

    def __init__(self, config_path: str = "config.yaml", verbose: bool = True):
        """
        Initialize the data generator.

        Args:
            config_path: Path to the configuration file
            verbose: Whether to print verbose output
        """
        self.config = load_config(config_path)
        self.verbose = verbose
        self.output_dir = self.config.get('output', {}).get('data_dir', './data')
        self.query_file = self.config.get('output', {}).get('query_file', 'queries.jsonl')
        self.response_file = self.config.get('output', {}).get('response_file', 'responses.jsonl')
        self.append_mode = self.config.get('output', {}).get('append_mode', True)

        # Ensure output directory exists
        ensure_dir(self.output_dir)

        # Configure litellm
        litellm.drop_params = True  # Drop unsupported params
        litellm.set_verbose = verbose

    def generate_single(
        self,
        query: str,
        model_name: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        save_result: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a single response from an LLM.

        Args:
            query: The input query/prompt
            model_name: Name of the model (must be in config)
            temperature: Temperature setting (overrides config)
            max_tokens: Max tokens setting (overrides config)
            top_p: Top-p setting (overrides config)
            save_result: Whether to save the result to file
            metadata: Additional metadata to include
            **kwargs: Additional parameters to pass to litellm

        Returns:
            LLMResponse object containing the response and metadata
        """
        # Get model configuration
        model_config = get_model_config(self.config, model_name)

        # Prepare parameters
        params = {
            'model': model_config['model_name'],
            'messages': [{'role': 'user', 'content': query}],
            'temperature': temperature or model_config.get('temperature', 1.0),
            'max_tokens': max_tokens or model_config.get('max_tokens', 2048),
            'top_p': top_p or model_config.get('top_p', 1.0),
        }

        # Add API key and other provider-specific settings
        if 'api_key' in model_config:
            params['api_key'] = model_config['api_key']
        if 'api_base' in model_config:
            params['api_base'] = model_config['api_base']
        if 'api_version' in model_config:
            params['api_version'] = model_config['api_version']

        # Add any additional kwargs
        params.update(kwargs)

        # Generate unique query ID
        query_id = generate_query_id()

        # Track timing
        start_time = datetime.now()
        start_time_iso = start_time.isoformat()

        try:
            # Make the API call
            response = litellm.completion(**params)

            # Track end time
            end_time = datetime.now()
            end_time_iso = end_time.isoformat()
            duration_seconds = (end_time - start_time).total_seconds()

            # Extract response data
            response_text = response.choices[0].message.content

            # Create response object
            llm_response = LLMResponse(
                query_id=query_id,
                query=query,
                response=response_text,
                model_info=ModelInfo(
                    model_name=model_config['model_name'],
                    provider=model_config.get('provider', 'unknown'),
                    temperature=params['temperature'],
                    max_tokens=params['max_tokens'],
                    top_p=params['top_p'],
                    other_params={k: v for k, v in kwargs.items()}
                ),
                token_usage=TokenUsage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens
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

            # Calculate cost if possible
            cost = calculate_cost(
                model_config['model_name'],
                response.usage.prompt_tokens,
                response.usage.completion_tokens
            )
            if cost is not None and llm_response.metadata is None:
                llm_response.metadata = {}
            if cost is not None:
                llm_response.metadata['estimated_cost_usd'] = cost

        except Exception as e:
            # Track end time even on error
            end_time = datetime.now()
            end_time_iso = end_time.isoformat()
            duration_seconds = (end_time - start_time).total_seconds()

            # Create error response
            llm_response = LLMResponse(
                query_id=query_id,
                query=query,
                response="",
                model_info=ModelInfo(
                    model_name=model_config['model_name'],
                    provider=model_config.get('provider', 'unknown'),
                    temperature=params['temperature'],
                    max_tokens=params['max_tokens'],
                    top_p=params['top_p'],
                    other_params={k: v for k, v in kwargs.items()}
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

        # Print summary
        print_response_summary(llm_response.to_dict(), verbose=self.verbose)

        # Save result if requested
        if save_result:
            self._save_response(llm_response)

        return llm_response

    def generate_batch(
        self,
        queries: List[str],
        model_name: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        save_results: bool = True,
        delay_between_queries: float = 0.0,
        show_progress: bool = True,
        **kwargs
    ) -> List[LLMResponse]:
        """
        Generate responses for a batch of queries.

        Args:
            queries: List of queries to process
            model_name: Name of the model to use
            temperature: Temperature setting
            max_tokens: Max tokens setting
            top_p: Top-p setting
            save_results: Whether to save results to file
            delay_between_queries: Delay between queries in seconds
            show_progress: Whether to show progress bar
            **kwargs: Additional parameters

        Returns:
            List of LLMResponse objects
        """
        responses = []

        iterator = tqdm(queries, desc=f"Processing with {model_name}") if show_progress else queries

        for query in iterator:
            response = self.generate_single(
                query=query,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                save_result=save_results,
                **kwargs
            )
            responses.append(response)

            # Delay between queries if specified
            if delay_between_queries > 0:
                time.sleep(delay_between_queries)

        return responses

    def compare_models(
        self,
        query: str,
        model_names: List[str],
        save_results: bool = True,
        **kwargs
    ) -> Dict[str, LLMResponse]:
        """
        Compare responses from multiple models for the same query.

        Args:
            query: The query to send to all models
            model_names: List of model names to compare
            save_results: Whether to save results
            **kwargs: Additional parameters

        Returns:
            Dictionary mapping model names to responses
        """
        results = {}

        print(f"\nComparing {len(model_names)} models for query:")
        print(f"  {query[:100]}...\n")

        for model_name in model_names:
            print(f"Generating response from {model_name}...")
            response = self.generate_single(
                query=query,
                model_name=model_name,
                save_result=save_results,
                **kwargs
            )
            results[model_name] = response

        return results

    def _save_response(self, response: LLMResponse) -> None:
        """
        Save a response to the output file.

        Args:
            response: LLMResponse object to save
        """
        filepath = f"{self.output_dir}/{self.response_file}"
        save_jsonl([response.to_dict()], filepath, append=self.append_mode)

    def load_responses(self) -> List[LLMResponse]:
        """
        Load all saved responses.

        Returns:
            List of LLMResponse objects
        """
        filepath = f"{self.output_dir}/{self.response_file}"
        data = load_jsonl(filepath)
        return [LLMResponse.from_dict(item) for item in data]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about collected data.

        Returns:
            Dictionary containing statistics
        """
        responses = self.load_responses()

        if not responses:
            return {"message": "No responses found"}

        total_queries = len(responses)
        successful_queries = sum(1 for r in responses if r.success)
        failed_queries = total_queries - successful_queries

        total_tokens = sum(r.token_usage.total_tokens for r in responses if r.success)
        total_time = sum(r.timing_info.duration_seconds for r in responses if r.success)

        models_used = {}
        for r in responses:
            model = r.model_info.model_name
            if model not in models_used:
                models_used[model] = 0
            models_used[model] += 1

        return {
            "total_queries": total_queries,
            "successful_queries": successful_queries,
            "failed_queries": failed_queries,
            "total_tokens_used": total_tokens,
            "total_time_seconds": total_time,
            "average_tokens_per_query": total_tokens / successful_queries if successful_queries > 0 else 0,
            "average_time_per_query": total_time / successful_queries if successful_queries > 0 else 0,
            "models_used": models_used
        }
