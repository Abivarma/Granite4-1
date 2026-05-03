from pathlib import Path

OLLAMA_BASE_URL = "http://localhost:11434"

MODELS = [
    {"tag": "granite4.1:8b",  "label": "Granite 4.1 8B"},
    {"tag": "granite4:latest", "label": "Granite 4.0"},
    {"tag": "llama3.1:8b",    "label": "Llama 3.1 8B"},
    {"tag": "qwen2.5:7b",     "label": "Qwen 2.5 7B"},
    {"tag": "mistral:7b",     "label": "Mistral 7B"},
]

BFCL_DATASET_ID = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"

BFCL_CATEGORIES = [
    "simple",
    "multiple",
    "parallel",
    "parallel_multiple",
]

RESULTS_DIR = Path("results")
TEMPERATURE = 0
MAX_RETRIES = 1
