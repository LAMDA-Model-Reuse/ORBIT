"""
Utility functions for the data generator.
"""
import json
import os
import re
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid
from dotenv import load_dotenv


_SENSITIVE_PATTERNS = [
    re.compile(r"sk-proj-[A-Za-z0-9_\-*]{8,}"),
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"AIza[0-9A-Za-z_\-]{10,}"),
    re.compile(r"hf_[A-Za-z0-9_\-]{8,}"),
    re.compile(r"(?i)(api[_-]?key|token|authorization|bearer)(\s*[:=]\s*)([^\s,'\"}]+)"),
]


def sanitize_error_message(message: Any) -> str:
    """Redact common API key formats before errors are printed or saved."""
    text = str(message)
    for pattern in _SENSITIVE_PATTERNS:
        if pattern.groups >= 3:
            text = pattern.sub(r"\1\2[REDACTED]", text)
        else:
            text = pattern.sub("[REDACTED]", text)
    return text


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Configuration dictionary
    """
    # Load environment variables
    load_dotenv()

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Replace environment variable placeholders
    config = _replace_env_vars(config)

    return config


def _replace_env_vars(config: Any) -> Any:
    """
    Recursively replace ${VAR_NAME} with environment variables.

    Args:
        config: Configuration dictionary or value

    Returns:
        Configuration with environment variables replaced
    """
    if isinstance(config, dict):
        return {k: _replace_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [_replace_env_vars(item) for item in config]
    elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
        var_name = config[2:-1]
        return os.getenv(var_name, config)
    else:
        return config


def ensure_dir(directory: str) -> None:
    """
    Ensure a directory exists, create if it doesn't.

    Args:
        directory: Path to the directory
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def save_jsonl(data: List[Dict[str, Any]], filepath: str, append: bool = True) -> None:
    """
    Save data to a JSONL file.

    Args:
        data: List of dictionaries to save
        filepath: Path to the output file
        append: Whether to append to existing file
    """
    mode = 'a' if append else 'w'
    with open(filepath, mode, encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """
    Load data from a JSONL file.

    Args:
        filepath: Path to the input file

    Returns:
        List of dictionaries
    """
    if not os.path.exists(filepath):
        return []

    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def generate_query_id() -> str:
    """
    Generate a unique query ID.

    Returns:
        Unique identifier string
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"{timestamp}_{unique_id}"


def get_timestamp() -> str:
    """
    Get current timestamp in ISO format.

    Returns:
        ISO format timestamp string
    """
    return datetime.now().isoformat()


def format_duration(seconds: float) -> str:
    """
    Format duration in a human-readable way.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string
    """
    if seconds < 1:
        return f"{seconds * 1000:.2f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"


def print_response_summary(response_data: Dict[str, Any], verbose: bool = True) -> None:
    """
    Print a summary of the LLM response.

    Args:
        response_data: Response data dictionary
        verbose: Whether to print verbose output
    """
    if not verbose:
        return

    print("\n" + "="*80)
    print(f"Query ID: {response_data.get('query_id', 'N/A')}")
    print(f"Model: {response_data.get('model_info', {}).get('model_name', 'N/A')}")
    print(f"Provider: {response_data.get('model_info', {}).get('provider', 'N/A')}")
    print("-"*80)
    print(f"Query: {response_data.get('query', 'N/A')[:100]}...")
    print("-"*80)

    if response_data.get('success', False):
        response_text = response_data.get('response', 'N/A')
        print(f"Response: {response_text[:200]}...")
        print("-"*80)

        token_usage = response_data.get('token_usage', {})
        timing_info = response_data.get('timing_info', {})

        print(f"Token Usage:")
        print(f"  - Prompt tokens: {token_usage.get('prompt_tokens', 'N/A')}")
        print(f"  - Completion tokens: {token_usage.get('completion_tokens', 'N/A')}")
        print(f"  - Total tokens: {token_usage.get('total_tokens', 'N/A')}")
        print(f"Timing:")
        print(f"  - Duration: {format_duration(timing_info.get('duration_seconds', 0))}")
    else:
        print(f"ERROR: {response_data.get('error_message', 'Unknown error')}")

    print("="*80 + "\n")


def get_model_config(config: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """
    Get model configuration from the config file.

    Args:
        config: Configuration dictionary
        model_name: Name of the model

    Returns:
        Model configuration dictionary
    """
    if model_name not in config.get('models', {}):
        raise ValueError(f"Model '{model_name}' not found in configuration")

    model_config = config['models'][model_name].copy()

    # Merge with default settings
    defaults = config.get('default', {})
    for key in ['temperature', 'max_tokens', 'top_p']:
        if key not in model_config and key in defaults:
            model_config[key] = defaults[key]

    return model_config


def calculate_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> Optional[float]:
    """
    Calculate approximate cost based on token usage.
    Note: Prices are approximate and may change. Update as needed.

    Args:
        model_name: Name of the model
        prompt_tokens: Number of prompt tokens
        completion_tokens: Number of completion tokens

    Returns:
        Estimated cost in USD, or None if pricing not available
    """
    # Pricing per 1M tokens (as of 2024, approximate)
    pricing = {
        'gpt-4': {'prompt': 30.0, 'completion': 60.0},
        'gpt-4-turbo-preview': {'prompt': 10.0, 'completion': 30.0},
        'gpt-3.5-turbo': {'prompt': 0.5, 'completion': 1.5},
        'claude-3-opus-20240229': {'prompt': 15.0, 'completion': 75.0},
        'claude-3-sonnet-20240229': {'prompt': 3.0, 'completion': 15.0},
        'claude-3-haiku-20240307': {'prompt': 0.25, 'completion': 1.25},
        'gemini-pro': {'prompt': 0.5, 'completion': 1.5},
    }

    if model_name not in pricing:
        return None

    prompt_cost = (prompt_tokens / 1_000_000) * pricing[model_name]['prompt']
    completion_cost = (completion_tokens / 1_000_000) * pricing[model_name]['completion']

    return prompt_cost + completion_cost
