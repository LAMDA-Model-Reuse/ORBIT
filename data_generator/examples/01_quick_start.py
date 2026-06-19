"""
Example 01: Quick Start - The Simplest Introductory Example

This is the simplest usage example demonstrating how to generate responses using an online API.
Suitable for: First-time users who want to quickly test features.
"""
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from generator import LLMDataGenerator


def main():
    """Generate responses using online APIs such as OpenAI, Anthropic, and Google."""

    print("=" * 80)
    print("Example 01: Quick Start - Using the Online API")
    print("=" * 80)
    print("\nPrerequisite: An API key needs to be configured in the .env file")
    print("If not configured, please run: cp .env.example .env")
    print("Then edit the .env file and enter your API key\n")

    generator = LLMDataGenerator(config_path="../config.yaml", verbose=True)

    query = "What is the capital of France?"

    print(f"Query: {query}\n")

    response = generator.generate_single(
        query=query,
        model_name="distilgpt2",
        save_result=True,
    )

    print("\n" + "=" * 80)
    print("Results:")
    print("=" * 80)
    print(f"Response: {response.response}")
    print(f"Token Usage: {response.token_usage.total_tokens}")
    print(f"  - Input: {response.token_usage.prompt_tokens}")
    print(f"  - Output: {response.token_usage.completion_tokens}")
    print(f"Duration: {response.timing_info.duration_seconds:.2f}s")
    print(f"Success: {response.success}")
    if not response.success:
        print(f"Error: {response.error_message}")
    print("\nData saved to: data/responses.jsonl")

    print("\n" + "=" * 80)
    print("Other available models (modify the model_name above):")
    print("=" * 80)
    print("OpenAI: gpt-3.5-turbo, gpt-4")
    print("Anthropic: claude-3-haiku, claude-3-sonnet, claude-3-opus")
    print("Google: gemini-pro, gemini-2.5-pro, gemini-2.5-flash")
    print("HuggingFace: distilgpt2, gpt2, llama3.1-8b")
    print("\nSee the complete model list in config.yaml")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        print("\nCommon Issues:")
        print("1. No API key configured -> Run 'cp .env.example .env' and edit .env")
        print("2. HuggingFace error -> Run 'pip install --upgrade litellm'")
        print("3. Network problem -> Check network connection")
        import traceback

        traceback.print_exc()
