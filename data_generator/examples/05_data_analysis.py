"""
Example 05: Data Analysis - Data analysis

This example shows how to analyze collected LLM response data and generate basic reports.
Best for:
- Analyzing existing collected data
- Generating summary reports
- Exporting data for further processing
"""
import sys
import os
import json
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from generator import LLMDataGenerator
from local_generator import LocalLLMGenerator


def example_basic_statistics():
    """Basic statistics: view a dataset overview"""

    print("="*80)
    print("Example 05a: Basic statistics")
    print("="*80)

    # Load data
    generator = LLMDataGenerator(config_path="../config.yaml", verbose=False)

    # Get statistics
    stats = generator.get_statistics()

    if "message" in stats:
        print(f"\n{stats['message']}")
        print("Please run other examples first to generate some data.")
        return

    # Print statistics
    print("\nDataset overview:")
    print(f"  Total queries: {stats['total_queries']}")
    print(f"  Successful: {stats['successful_queries']}")
    print(f"  Failed: {stats['failed_queries']}")
    print(f"  Success rate: {stats['successful_queries']/stats['total_queries']*100:.1f}%")

    print("\nToken usage:")
    print(f"  Total: {stats['total_tokens_used']:,}")
    print(f"  Avg per query: {stats['average_tokens_per_query']:.1f}")

    print("\nTiming:")
    print(f"  Total time: {stats['total_time_seconds']:.2f}s")
    print(f"  Avg per query: {stats['average_time_per_query']:.2f}s")

    print("\nModels used:")
    for model, count in stats['models_used'].items():
        print(f"  {model}: {count} runs")


def example_detailed_analysis():
    """Detailed analysis: breakdown by model"""

    print("\n\n" + "="*80)
    print("Example 05b: Detailed analysis by model")
    print("="*80)

    # Load all responses
    generator = LLMDataGenerator(config_path="../config.yaml", verbose=False)
    responses = generator.load_responses()

    if not responses:
        print("\nNo data yet. Please run other examples first to generate data.")
        return

    # Group by model
    by_model = defaultdict(list)
    for r in responses:
        if r.success:
            by_model[r.model_info.model_name].append(r)

    # Analyze each model
    print("\nAnalysis by model:")
    for model_name, model_responses in by_model.items():
        print(f"\n【{model_name}】")
        print(f"  Queries: {len(model_responses)}")

        # Token stats
        total_tokens = sum(r.token_usage.total_tokens for r in model_responses)
        avg_tokens = total_tokens / len(model_responses)
        print(f"  Total tokens: {total_tokens:,}")
        print(f"  Avg tokens: {avg_tokens:.1f}")

        # Time stats
        total_time = sum(r.timing_info.duration_seconds for r in model_responses)
        avg_time = total_time / len(model_responses)
        avg_speed = (
            sum(r.token_usage.completion_tokens / r.timing_info.duration_seconds for r in model_responses)
            / len(model_responses)
        )
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Avg time: {avg_time:.2f}s")
        print(f"  Avg speed: {avg_speed:.1f} tokens/s")

        # Cost estimate (if available)
        costs = [r.metadata.get('estimated_cost_usd', 0) for r in model_responses if r.metadata]
        if any(costs):
            print(f"  Total cost: ${sum(costs):.4f}")


def example_export_to_csv():
    """Export to CSV"""

    print("\n\n" + "="*80)
    print("Example 05c: Export to CSV")
    print("="*80)

    import pandas as pd

    # Load data
    generator = LLMDataGenerator(config_path="../config.yaml", verbose=False)
    responses = generator.load_responses()

    if not responses:
        print("\nNo data.")
        return

    # Convert to DataFrame
    data = []
    for r in responses:
        data.append({
            'query_id': r.query_id,
            'timestamp': r.timestamp,
            'query': r.query,
            'response': r.response,
            'model': r.model_info.model_name,
            'provider': r.model_info.provider,
            'temperature': r.model_info.temperature,
            'prompt_tokens': r.token_usage.prompt_tokens,
            'completion_tokens': r.token_usage.completion_tokens,
            'total_tokens': r.token_usage.total_tokens,
            'duration_seconds': r.timing_info.duration_seconds,
            'success': r.success,
            'error': r.error_message if not r.success else None
        })

    df = pd.DataFrame(data)

    # Save to CSV
    output_file = "data/responses.csv"
    df.to_csv(output_file, index=False, encoding='utf-8')

    print(f"\n✓ Exported data to: {output_file}")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {len(df.columns)}")
    print("\nPreview (first rows):")
    print(df[['query', 'model', 'total_tokens', 'duration_seconds', 'success']].head())


def example_filter_and_export():
    """Filter and export specific subsets"""

    print("\n\n" + "="*80)
    print("Example 05d: Filter and export")
    print("="*80)

    generator = LLMDataGenerator(config_path="../config.yaml", verbose=False)
    responses = generator.load_responses()

    if not responses:
        print("\nNo data.")
        return

    # Filter: successful only
    successful = [r for r in responses if r.success]
    print(f"\nSuccessful responses: {len(successful)} / {len(responses)}")

    # Filter: a specific model
    model_name = "meta-llama/Llama-3.2-1B"  # Change to the model you want
    model_responses = [r for r in responses if model_name in r.model_info.model_name]
    print(f"\nResponses for model '{model_name}': {len(model_responses)}")

    # Filter: token range
    min_tokens, max_tokens = 50, 200
    token_filtered = [
        r for r in responses
        if r.success and min_tokens <= r.token_usage.total_tokens <= max_tokens
    ]
    print(f"\nResponses with total_tokens in {min_tokens}-{max_tokens}: {len(token_filtered)}")

    # Export filtered data
    if token_filtered:
        output_file = "data/filtered_responses.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for r in token_filtered:
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + '\n')
        print(f"\n✓ Saved filtered data to: {output_file}")


def example_response_length_distribution():
    """Response length distribution"""

    print("\n\n" + "="*80)
    print("Example 05e: Response length distribution")
    print("="*80)

    generator = LLMDataGenerator(config_path="../config.yaml", verbose=False)
    responses = generator.load_responses()

    if not responses:
        print("\nNo data.")
        return

    successful = [r for r in responses if r.success]

    # Response length in characters
    lengths = [len(r.response) for r in successful]

    if lengths:
        print("\nResponse length stats (characters):")
        print(f"  Min: {min(lengths)}")
        print(f"  Max: {max(lengths)}")
        print(f"  Avg: {sum(lengths)/len(lengths):.1f}")
        print(f"  Median: {sorted(lengths)[len(lengths)//2]}")

        # Simple histogram
        print("\nLength distribution:")
        bins = [0, 50, 100, 200, 500, 1000, float('inf')]
        bin_labels = ['0-50', '50-100', '100-200', '200-500', '500-1000', '1000+']

        for i in range(len(bins) - 1):
            count = sum(1 for l in lengths if bins[i] <= l < bins[i + 1])
            bar = '█' * (count * 50 // len(lengths)) if count > 0 else ''
            print(f"  {bin_labels[i]:>10}: {count:3d} {bar}")


def main():
    """Run data analysis examples"""

    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                      Data Analysis Examples                                 ║
║                                                                            ║
║  Analyze collected LLM response data:                                      ║
║    - Summary statistics                                                     ║
║    - Per-model analysis                                                     ║
║    - Export to CSV                                                          ║
║    - Filtering subsets                                                      ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)

    # Example 1: Basic statistics (recommended)
    example_basic_statistics()

    # Example 2: Detailed analysis (optional)
    # example_detailed_analysis()

    # Example 3: Export CSV (optional, requires pandas)
    # example_export_to_csv()

    # Example 4: Filter and export (optional)
    # example_filter_and_export()

    # Example 5: Response length distribution (optional)
    # example_response_length_distribution()

    print("\n\n" + "="*80)
    print("Tips:")
    print("="*80)
    print("1. Data is stored at: data/responses.jsonl")
    print("2. You can use any JSON tool to read and analyze it")
    print("3. Pandas enables more advanced analysis")
    print("4. generator.load_responses() loads all historical data")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTip: To use CSV export, install pandas:")
        print("  pip install pandas")
        import traceback
        traceback.print_exc()