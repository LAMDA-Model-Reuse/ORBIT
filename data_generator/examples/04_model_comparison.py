"""
Example 04: Model Comparison - Model comparison

This example shows how to compare different models by response quality, speed, and cost.
Best for:
- Evaluating which model to choose
- Comparing model performance across dimensions
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from generator import LLMDataGenerator
from local_generator import LocalLLMGenerator


def example_compare_api_models():
    """Compare different online API models"""

    print("="*80)
    print("Example 04a: Compare online API models")
    print("="*80)

    generator = LLMDataGenerator(config_path="../config.yaml", verbose=False)

    query = "Explain quantum computing in simple terms."

    # Compare multiple models (requires corresponding API keys)
    models_to_compare = [
        "gpt-3.5-turbo",      # OpenAI
        "claude-3-haiku",     # Anthropic
        "gemini-pro",         # Google
    ]

    print(f"Query: {query}\n")
    print(f"Comparing {len(models_to_compare)} models...\n")

    # Use compare_models
    results = generator.compare_models(
        query=query,
        model_names=models_to_compare,
        save_results=True
    )

    # Print comparison results
    print("\n" + "="*80)
    print("Comparison results:")
    print("="*80)

    for model_name, response in results.items():
        print(f"\n【{model_name}】")
        if response.success:
            print(f"Response: {response.response[:150]}...")
            print(
                f"Tokens: {response.token_usage.total_tokens} "
                f"({response.token_usage.prompt_tokens} + {response.token_usage.completion_tokens})"
            )
            print(f"Time: {response.timing_info.duration_seconds:.2f}s")
            print(f"Speed: {response.token_usage.completion_tokens/response.timing_info.duration_seconds:.1f} tokens/s")
            if response.metadata and 'estimated_cost_usd' in response.metadata:
                print(f"Cost: ${response.metadata['estimated_cost_usd']:.6f}")
        else:
            print(f"❌ Failed: {response.error_message}")


def example_compare_local_models():
    """Compare different sizes/configurations of local models"""

    print("\n\n" + "="*80)
    print("Example 04b: Compare local models")
    print("="*80)

    query = "What is machine learning?"

    # Compare different model variants/configurations
    models = [
        {
            "name": "Llama-3.2-1B",
            "path": "meta-llama/Llama-3.2-1B",
            "quantization": None
        },
        {
            "name": "Llama-3.2-1B (8-bit)",
            "path": "meta-llama/Llama-3.2-1B",
            "quantization": "8bit"
        },
    ]

    print(f"Query: {query}\n")

    results = {}

    for model_config in models:
        print(f"\nRunning: {model_config['name']}...")

        # Load model
        load_kwargs = {
            "model_path": model_config['path'],
            "device": "auto",
            "verbose": False
        }

        if model_config['quantization'] == "8bit":
            load_kwargs['load_in_8bit'] = True
        elif model_config['quantization'] == "4bit":
            load_kwargs['load_in_4bit'] = True

        generator = LocalLLMGenerator(**load_kwargs)

        # Generate response
        response = generator.generate_single(
            query=query,
            temperature=0.7,
            max_tokens=100,
            save_result=False
        )

        results[model_config['name']] = response

        # Free memory
        generator.clear_memory()

    # Print comparison
    print("\n" + "="*80)
    print("Comparison results:")
    print("="*80)

    for model_name, response in results.items():
        print(f"\n【{model_name}】")
        print(f"Response: {response.response[:150]}...")
        print(f"Tokens: {response.token_usage.total_tokens}")
        print(f"Time: {response.timing_info.duration_seconds:.2f}s")
        print(f"Speed: {response.token_usage.completion_tokens/response.timing_info.duration_seconds:.1f} tokens/s")


def example_quality_comparison():
    """Quality comparison: same prompt with different temperatures"""

    print("\n\n" + "="*80)
    print("Example 04c: Generation parameter comparison")
    print("="*80)

    generator = LocalLLMGenerator(
        model_path="meta-llama/Llama-3.2-1B",
        device="auto",
        verbose=False
    )

    query = "Write a creative opening sentence for a story."

    # Different temperatures
    temperatures = [0.3, 0.7, 1.0, 1.5]

    print(f"Query: {query}\n")
    print(f"Testing temperatures: {temperatures}\n")

    for temp in temperatures:
        print(f"\nTemperature: {temp}")
        response = generator.generate_single(
            query=query,
            temperature=temp,
            max_tokens=50,
            save_result=False
        )
        print(f"Response: {response.response}")


def example_speed_comparison():
    """Speed comparison: benchmark inference performance"""

    print("\n\n" + "="*80)
    print("Example 04d: Speed/performance comparison")
    print("="*80)

    import time

    generator = LocalLLMGenerator(
        model_path="meta-llama/Llama-3.2-1B",
        device="auto",
        verbose=False
    )

    query = "Count from 1 to 10."

    # Different max_tokens values
    token_limits = [50, 100, 200]

    print(f"Query: {query}\n")
    print("Testing different max_tokens...\n")

    for max_tokens in token_limits:
        start = time.time()
        response = generator.generate_single(
            query=query,
            max_tokens=max_tokens,
            temperature=0.7,
            save_result=False
        )
        elapsed = time.time() - start

        print(f"\nMax Tokens: {max_tokens}")
        print(f"Actual generated: {response.token_usage.completion_tokens} tokens")
        print(f"Time: {elapsed:.2f}s")
        print(f"Speed: {response.token_usage.completion_tokens/elapsed:.1f} tokens/s")


def main():
    """Run model comparison examples"""

    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                    Model Comparison Examples                                ║
║                                                                            ║
║  Helps you:                                                                ║
║    - Choose the most suitable model                                         ║
║    - Evaluate cost vs. quality                                              ║
║    - Tune generation parameters                                             ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)

    # Choose one example to run (uncomment as needed)

    # Example 1: Compare online APIs (requires multiple API keys)
    # example_compare_api_models()

    # Example 2: Compare local models (recommended)
    example_compare_local_models()

    # Example 3: Quality comparison
    # example_quality_comparison()

    # Example 4: Speed comparison
    # example_speed_comparison()

    print("\n\n" + "="*80)
    print("Tips:")
    print("="*80)
    print("1. Use generator.compare_models() to easily compare multiple models")
    print("2. Focus on three dimensions: quality, speed, and cost")
    print("3. Local model speed: no quantization > 8-bit > 4-bit")
    print("4. API model selection: GPT-4 (quality) > GPT-3.5 (balance) > open-source (cost)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()