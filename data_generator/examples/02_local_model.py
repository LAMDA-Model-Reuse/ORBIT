"""
Example 02: Local Model - Use a local HuggingFace model

This example shows how to use a locally available (downloaded/cached) model directly,
without any API key or external server.

Best for:
- You already have a local model (or want to use the HF cache)
- You want to run fully offline
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from local_generator import LocalLLMGenerator


def example_basic():
    """Basic example: generate a single response with a local model"""

    print("="*80)
    print("Example 02a: Basic usage - Single query")
    print("="*80)

    # Configure model path
    # Option 1: Use a HuggingFace model ID (auto-load from cache or download)
    model_path = "meta-llama/Llama-3.2-1B"  # Small model, good for testing

    # Option 2: Use a local path (if you already downloaded the model)
    # model_path = "/path/to/your/Llama-3.1-8B-Instruct"

    print(f"Loading model: {model_path}")
    print("Tip: The first run may download the model and can take a few minutes...\n")

    # Initialize generator
    generator = LocalLLMGenerator(
        model_path=model_path,
        device="auto",  # Automatically choose GPU/CPU
        torch_dtype="auto",  # Automatically choose dtype
        verbose=True
    )

    # Generate response
    query = "What is the capital of France?"
    print(f"Query: {query}\n")

    response = generator.generate_single(
        query=query,
        temperature=0.7,  # 0.1-2.0, higher = more random
        max_tokens=100,  # Max generated tokens
        save_result=True  # Save to data/responses.jsonl
    )

    # Print results
    print("\n" + "="*80)
    print("Result:")
    print("="*80)
    print(f"Response: {response.response}")
    print(f"Token usage: {response.token_usage.total_tokens}")
    print(f"Inference time: {response.timing_info.duration_seconds:.2f}s")
    print(f"Speed: {response.token_usage.completion_tokens/response.timing_info.duration_seconds:.2f} tokens/s")


def example_quantization():
    """Memory optimization: use 8-bit quantization (~50% memory savings)"""

    print("\n\n" + "="*80)
    print("Example 02b: Memory optimization - 8-bit quantization")
    print("="*80)
    print("If GPU memory is not enough, quantization can help reduce memory usage.\n")

    generator = LocalLLMGenerator(
        model_path="meta-llama/Llama-3.2-1B",
        device="auto",
        load_in_8bit=True,  # Enable 8-bit quantization (requires bitsandbytes)
        # load_in_4bit=True,  # Or use 4-bit (saves more, slightly lower quality)
        verbose=True
    )

    response = generator.generate_single(
        query="Explain machine learning in one sentence.",
        temperature=0.7,
        max_tokens=50
    )

    print(f"\nResponse: {response.response}")


def example_batch():
    """Batch processing: handle multiple queries at once"""

    print("\n\n" + "="*80)
    print("Example 02c: Batch processing")
    print("="*80)

    generator = LocalLLMGenerator(
        model_path="meta-llama/Llama-3.2-1B",
        device="auto",
        verbose=False  # Disable verbose logs for batch runs
    )

    # Multiple queries
    queries = [
        "What is AI?",
        "What is machine learning?",
        "What is deep learning?"
    ]

    print(f"Processing {len(queries)} queries...\n")

    responses = generator.generate_batch(
        queries=queries,
        temperature=0.7,
        max_tokens=100,
        show_progress=True  # Show progress bar
    )

    # Print results
    print("\nResults:")
    for i, (query, response) in enumerate(zip(queries, responses), 1):
        print(f"\n{i}. Q: {query}")
        print(f"   A: {response.response[:100]}...")
        print(f"   Tokens: {response.token_usage.total_tokens}, Time: {response.timing_info.duration_seconds:.2f}s")


def example_different_params():
    """Explore different generation parameters"""

    print("\n\n" + "="*80)
    print("Example 02d: Effects of different generation parameters")
    print("="*80)

    generator = LocalLLMGenerator(
        model_path="meta-llama/Llama-3.2-1B",
        device="auto",
        verbose=False
    )

    query = "Complete this: The future of AI is"

    # Param set 1: Greedy decoding (deterministic, same result each run)
    print("\n1. Greedy decoding (do_sample=False) - Deterministic:")
    response1 = generator.generate_single(
        query=query,
        do_sample=False,  # Always pick the most likely token
        max_tokens=30,
        save_result=False
    )
    print(f"   {response1.response}")

    # Param set 2: High temperature (creative, more randomness)
    print("\n2. High temperature (temperature=1.5) - More creative:")
    response2 = generator.generate_single(
        query=query,
        temperature=1.5,  # More random
        top_p=0.95,
        max_tokens=30,
        save_result=False
    )
    print(f"   {response2.response}")

    # Param set 3: Low temperature (focused, higher determinism)
    print("\n3. Low temperature (temperature=0.3) - More focused:")
    response3 = generator.generate_single(
        query=query,
        temperature=0.3,  # More deterministic
        top_p=0.9,
        max_tokens=30,
        save_result=False
    )
    print(f"   {response3.response}")


def main():
    """Run all examples"""

    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                 Local Model Usage Examples                                  ║
║                                                                            ║
║  Prerequisites:                                                            ║
║    pip install transformers torch accelerate                               ║
║                                                                            ║
║  Optional (for quantization):                                              ║
║    pip install bitsandbytes                                                ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)

    # Example 1: Basic usage (recommended)
    example_basic()

    # Example 2: Quantization (optional, uncomment to run)
    # example_quantization()

    # Example 3: Batch processing (optional, uncomment to run)
    # example_batch()

    # Example 4: Parameter exploration (optional, uncomment to run)
    # example_different_params()

    print("\n\n" + "="*80)
    print("Tips:")
    print("="*80)
    print("1. If you run out of memory: use load_in_8bit=True or a smaller model")
    print("2. If it's too slow: make sure you're using a GPU (device='cuda')")
    print("3. Model path: you can use an HF model ID or a local directory path")
    print("4. More docs: see docs/LOCAL_TRANSFORMERS_GUIDE.md")


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print("\n❌ Missing dependency!")
        print(f"Error: {e}")
        print("\nPlease install required packages:")
        print("  pip install transformers torch accelerate")
        print("\nOptional (for quantization):")
        print("  pip install bitsandbytes")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()