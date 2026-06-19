"""
Example 03: Batch Processing - Batch data generation

This example shows how to process many queries in batches to build a large-scale dataset.
Best for:
- Collecting large amounts of LLM responses for research or training
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from generator import LLMDataGenerator
from local_generator import LocalLLMGenerator


def example_batch_with_api():
    """Batch processing with an online API"""

    print("="*80)
    print("Example 03a: Batch processing (online API)")
    print("="*80)

    generator = LLMDataGenerator(config_path="../config.yaml", verbose=False)

    # Prepare a list of queries
    queries = [
        "What is artificial intelligence?",
        "What is machine learning?",
        "What is deep learning?",
        "What is neural network?",
        "What is natural language processing?"
    ]

    print(f"Preparing to process {len(queries)} queries...\n")

    # Batch generation
    responses = generator.generate_batch(
        queries=queries,
        model_name="distilgpt2",  # Or another model
        temperature=0.7,
        max_tokens=100,
        save_results=True,  # Auto-save
        delay_between_queries=1.0,  # 1s delay between queries (avoid rate limits)
        show_progress=True  # Show progress bar
    )

    # Summarize results
    successful = sum(1 for r in responses if r.success)
    failed = len(responses) - successful
    total_tokens = sum(r.token_usage.total_tokens for r in responses if r.success)
    total_time = sum(r.timing_info.duration_seconds for r in responses)

    print("\n" + "="*80)
    print("Batch processing summary:")
    print("="*80)
    print(f"Total queries: {len(queries)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total tokens: {total_tokens}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Avg per query: {total_time/len(queries):.2f}s")


def example_batch_with_local():
    """Batch processing with a local model"""

    print("\n\n" + "="*80)
    print("Example 03b: Batch processing (local model)")
    print("="*80)

    generator = LocalLLMGenerator(
        model_path="meta-llama/Llama-3.2-1B",
        device="auto",
        verbose=False
    )

    queries = [
        "Explain AI in simple terms.",
        "What is the difference between AI and ML?",
        "How do neural networks work?",
    ]

    print(f"Processing {len(queries)} queries with a local model...\n")

    responses = generator.generate_batch(
        queries=queries,
        temperature=0.7,
        max_tokens=150,
        show_progress=True
    )

    # Print each result
    print("\n" + "="*80)
    print("Results:")
    print("="*80)
    for i, (query, response) in enumerate(zip(queries, responses), 1):
        print(f"\n{i}. Query: {query}")
        print(f"   Response: {response.response[:150]}...")
        print(f"   Tokens: {response.token_usage.total_tokens}, Time: {response.timing_info.duration_seconds:.2f}s")


def example_large_scale():
    """Large-scale data collection example"""

    print("\n\n" + "="*80)
    print("Example 03c: Large-scale data collection")
    print("="*80)

    # Example: read a query list from a file
    # In real usage, you can load queries from CSV/JSON/TXT files

    # Simulate a large query list
    topics = ["AI", "ML", "DL", "NLP", "CV", "RL", "GAN", "Transformer"]
    queries = []
    for topic in topics:
        queries.append(f"What is {topic}?")
        queries.append(f"Explain {topic} with an example.")
        queries.append(f"What are the applications of {topic}?")

    print(f"Generated {len(queries)} queries")
    print("First 5 queries:")
    for i, q in enumerate(queries[:5], 1):
        print(f"  {i}. {q}")
    print("  ...")

    # Use a local model (avoid API costs)
    print("\nProcessing with a local model (this may take a few minutes)...")

    generator = LocalLLMGenerator(
        model_path="meta-llama/Llama-3.2-1B",
        device="auto",
        load_in_8bit=True,  # Quantization to save memory
        verbose=False
    )

    # Only process the first 3 as a demo (process all in real usage)
    sample_queries = queries[:3]
    print(f"(To save time, only processing the first {len(sample_queries)} queries as a demo)\n")

    responses = generator.generate_batch(
        queries=sample_queries,
        temperature=0.7,
        max_tokens=100,
        show_progress=True
    )

    print("\nData saved to: data/responses.jsonl")
    print("You can analyze the generated data using 04_data_analysis.py")


def example_from_file():
    """Read queries from a file and process in batch"""

    print("\n\n" + "="*80)
    print("Example 03d: Load queries from a file")
    print("="*80)

    # Create a sample query file
    import json

    queries_file = "data/sample_queries.jsonl"
    sample_queries = [
        {"id": 1, "query": "What is Python?", "category": "programming"},
        {"id": 2, "query": "What is JavaScript?", "category": "programming"},
        {"id": 3, "query": "What is React?", "category": "web"},
    ]

    # Save to file
    os.makedirs("data", exist_ok=True)
    with open(queries_file, 'w', encoding='utf-8') as f:
        for item in sample_queries:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Created sample file: {queries_file}")

    # Read from file
    queries_data = []
    with open(queries_file, 'r', encoding='utf-8') as f:
        for line in f:
            queries_data.append(json.loads(line.strip()))

    print(f"Loaded {len(queries_data)} queries from file\n")

    # Process
    generator = LocalLLMGenerator(
        model_path="meta-llama/Llama-3.2-1B",
        device="auto",
        verbose=False
    )

    queries = [item['query'] for item in queries_data]
    responses = generator.generate_batch(
        queries=queries,
        temperature=0.7,
        max_tokens=80,
        show_progress=True
    )

    # Merge results with original metadata
    print("\nResults (with original metadata):")
    for query_data, response in zip(queries_data, responses):
        print(f"\nID: {query_data['id']}, Category: {query_data['category']}")
        print(f"Q: {query_data['query']}")
        print(f"A: {response.response[:100]}...")


def main():
    """Run batch processing examples"""

    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                  Batch Data Generation Examples                             ║
║                                                                            ║
║  Use cases:                                                                ║
║    - Collect large volumes of LLM response data                             ║
║    - Build training datasets                                                ║
║    - Evaluate model performance                                             ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)

    # Choose one example to run (uncomment as needed)

    # Example 1: Online API (requires an API key)
    # example_batch_with_api()

    # Example 2: Local model (recommended)
    example_batch_with_local()

    # Example 3: Large-scale data collection
    # example_large_scale()

    # Example 4: Load from file
    # example_from_file()

    print("\n\n" + "="*80)
    print("Tips:")
    print("="*80)
    print("1. All responses are automatically saved to data/responses.jsonl")
    print("2. Use append_mode=True to keep accumulating data over time")
    print("3. For large-scale runs, prefer local models to reduce cost")
    print("4. Be mindful of API rate limits; set delay_between_queries accordingly")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()