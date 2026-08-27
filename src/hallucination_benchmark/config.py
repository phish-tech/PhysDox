"""
Configuration for Hardware-Grounded Hallucination Benchmark.

Supports multiple LLM backends via environment variables.
"""

import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]


def _load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE pairs from a local .env without overriding shell env."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_env_file(ROOT_DIR / ".env")

# ============================================================
# LLM API Configuration
# ============================================================
# Override these via environment variables or by editing directly.

MODELS = {
    "mimo": {
        "name": "mimo-v2.5-pro",
        "api_key_env": "MIMO_API_KEY",
        "base_url_env": "MIMO_BASE_URL",
        "default_base_url": "https://token-plan-cn.xiaomimimo.com/v1",
    },
    "mimo-v2.5": {
        "name": "mimo-v2.5",
        "api_key_env": "MIMO_API_KEY",
        "base_url_env": "MIMO_BASE_URL",
        "default_base_url": "https://token-plan-cn.xiaomimimo.com/v1",
    },
    "openai": {
        "name": "gpt-4o",
        "api_key_env": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL",
        "default_base_url": "https://api.openai.com/v1",
    },
    "deepseek": {
        "name": "deepseek-chat",
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url_env": "DEEPSEEK_BASE_URL",
        "default_base_url": "https://api.deepseek.com/v1",
    },
    "claude-opus-4-7": {
        "name": "claude-opus-4-7",
        "api_key_env": "RIGHT_CODES_API_KEY",
        "base_url_env": "CLAUDE_OPUS_4_7_BASE_URL",
        "default_base_url": "https://www.right.codes/claude-aws",
        "api_style": "anthropic_messages",
    },
    "gpt-5.5": {
        "name": "gpt-5.5",
        "api_key_env": "RIGHT_CODES_API_KEY",
        "base_url_env": "GPT_5_5_BASE_URL",
        "default_base_url": "",
    },
    "deepseek-v4-flash": {
        "name": "deepseek-v4-flash",
        "api_key_env": "DEEPSEEK_V4_API_KEY",
        "base_url_env": "DEEPSEEK_V4_BASE_URL",
        "default_base_url": "https://api.deepseek.com",
    },
    "deepseek-v4-pro": {
        "name": "deepseek-v4-pro",
        "api_key_env": "DEEPSEEK_V4_API_KEY",
        "base_url_env": "DEEPSEEK_V4_BASE_URL",
        "default_base_url": "https://api.deepseek.com",
    },
    "openrouter": {
        "name": "openai/gpt-4o",
        "api_key_env": "OPENROUTER_API_KEY",
        "base_url_env": "OPENROUTER_BASE_URL",
        "default_base_url": "https://openrouter.ai/api/v1",
    },
    # Generic: reads LLM_API_KEY, LLM_BASE_URL, LLM_MODEL_NAME
    "custom": {
        "name": None,  # reads from LLM_MODEL_NAME env
        "api_key_env": "LLM_API_KEY",
        "base_url_env": "LLM_BASE_URL",
        "default_base_url": "",
    },
}

# Default model key (pick from MODELS above, or set via BENCHMARK_MODEL env)
DEFAULT_MODEL_KEY = os.environ.get("BENCHMARK_MODEL", "mimo")

# ============================================================
# Pipeline Configuration
# ============================================================

# Generation
GENERATION_TEMPERATURES = [0.5, 0.7, 0.9]  # Multi-round temperatures
GENERATION_ROUNDS = 3                       # How many rounds to run
PROTOCOLS_PER_DOMAIN_PER_ROUND = 3          # Correct protocols per domain per round
SABOTAGES_PER_PROTOCOL = 2                  # How many constraints to inject per protocol

# Negative sample ratio
VALID_RATIO = 0.20       # 20% valid samples
AMBIGUOUS_RATIO = 0.10   # 10% ambiguous samples
# remaining 70% are sabotage

# Quality thresholds
MIN_VERIFIER_QUALITY_SCORE = 3  # 1-5 scale, reject below this

# Repository paths
DATA_DIR = ROOT_DIR / "data"
CURRENT_DATA_DIR = DATA_DIR / "current"
GOLD_DATA_DIR = DATA_DIR / "gold"
RELEASES_DIR = DATA_DIR / "releases"
LEGACY_DATA_DIR = DATA_DIR / "legacy"
SEED_DATA_PATH = DATA_DIR / "seeds"
GENERATED_DIR = DATA_DIR / "generated"
RUNTIME_DIR = DATA_DIR / "runtime"
EXPERIMENTS_DIR = ROOT_DIR / "experiments"

# Backward-compatible string constants used by older scripts.
OUTPUT_DIR = str(GENERATED_DIR)
SEED_DATA_DIR = str(SEED_DATA_PATH)


def get_model_config(model_key=None, model_name_override=None):
    """Get model configuration by key.

    Args:
        model_key: Key from MODELS dict, or "custom" for env-based config.
        model_name_override: Override the model name (e.g. from --model-name flag).
    """
    key = model_key or DEFAULT_MODEL_KEY

    if key == "custom":
        # Generic mode: read everything from env
        return {
            "model": model_name_override or os.environ.get("LLM_MODEL_NAME", ""),
            "api_key": os.environ.get("LLM_API_KEY", ""),
            "base_url": os.environ.get("LLM_BASE_URL", ""),
            "api_style": os.environ.get("LLM_API_STYLE", "chat_completions"),
        }

    if key not in MODELS:
        raise ValueError(f"Unknown model key '{key}'. Available: {list(MODELS.keys())}")
    cfg = MODELS[key]
    return {
        "model": model_name_override or cfg["name"] or os.environ.get("LLM_MODEL_NAME", ""),
        "api_key": os.environ.get(cfg["api_key_env"], ""),
        "base_url": os.environ.get(cfg["base_url_env"], cfg["default_base_url"]),
        "api_style": cfg.get("api_style", "chat_completions"),
    }
