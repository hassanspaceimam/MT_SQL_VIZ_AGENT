# =============================================================================
# config.py — Centralized configuration for Azure OpenAI + SQLAlchemy engine
# with dotenv support and simple singletons.
# =============================================================================

import os
from functools import lru_cache
from sqlalchemy import create_engine
from langchain_openai import AzureChatOpenAI

# --- Load .env (so your .env file is actually used) ---
try:
    from dotenv import load_dotenv
    # Loads .env from project root by default; set ENV_FILE to override if needed.
    load_dotenv(dotenv_path=os.getenv("ENV_FILE", ".env"), override=False)
except Exception:
    # If python-dotenv isn't installed, we just skip; env must come from OS.
    pass

# --- Azure OpenAI (env vars with safe defaults to preserve current behavior) ---
# SECURITY NOTE: Avoid committing real API keys. Prefer setting AZURE_OPENAI_* via
# environment variables or your secret manager. The default below should be
# replaced in production.
AZURE_ENDPOINT = os.getenv(
    "AZURE_OPENAI_ENDPOINT"
)

AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

# --- Database URL: prefer DATABASE_URL from your .env, else DB_URL, else default ---

DB_URL = os.getenv(
    "DATABASE_URL",
    os.getenv("DB_URL")
)

# --- Knowledgebase path (optional override via .env) ---
DEFAULT_KB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledgebase.pkl")
KNOWLEDGEBASE_PATH = os.getenv("KNOWLEDGEBASE_PATH", DEFAULT_KB)


@lru_cache(maxsize=1)
def get_llm() -> AzureChatOpenAI:
    """Singleton AzureChatOpenAI configured exactly like your original code.

    NOTE: Ensure AZURE_ENDPOINT is the base resource URL (see note above) for
    best compatibility with langchain_openai.
    """
    return AzureChatOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        azure_deployment=AZURE_DEPLOYMENT,
        api_version=AZURE_API_VERSION,
        api_key=AZURE_API_KEY,
    )


@lru_cache(maxsize=1)
def get_engine():
    """Singleton SQLAlchemy engine identical to your original create_engine usage."""
    return create_engine(DB_URL)


@lru_cache(maxsize=1)
def get_knowledgebase_path() -> str:
    """Path to knowledgebase.pkl; uses .env override when provided."""
    return KNOWLEDGEBASE_PATH
