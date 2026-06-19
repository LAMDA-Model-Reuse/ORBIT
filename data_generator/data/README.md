# Data Directory

This directory stores the generated LLM response data.

## Files

- `queries.jsonl`: Log of all queries sent to LLMs
- `responses.jsonl`: Complete responses with metadata
- `*.json`: Summary files from batch operations
- `*.csv`: Exported data in CSV format

## Data Format

All `.jsonl` files contain one JSON object per line. Each response includes:

- Query and response text
- Token usage statistics
- Timing information
- Model information
- Success status and error messages (if any)
- Custom metadata

## Note

This directory is ignored by git to prevent accidentally committing large data files or sensitive information.
