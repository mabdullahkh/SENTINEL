import os

# Base paths
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BACKEND_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODELS_DIR = os.path.join(DATA_DIR, "models")
RAG_DIR = os.path.join(DATA_DIR, "rag_guidelines")

# Create folders if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RAG_DIR, exist_ok=True)

# DB Settings
DB_PATH = os.path.join(DATA_DIR, "mimic_subset.db")

# Target Diseases
DISEASES = ["sepsis", "cardiac", "respiratory", "renal", "neurological"]

# ICD mappings for labels
DISEASE_ICD_PREFIXES = {
    "sepsis": ["99591", "99592", "78552", "77181", "A40", "A41", "R652"],
    "cardiac": ["428", "410", "I50", "I21"],
    "respiratory": ["486", "51881", "496", "J18", "J96", "J44"],
    "renal": ["584", "585", "N17", "N18"],
    "neurological": ["434", "430", "431", "I63", "I60", "I61"]
}

# Model Settings
MODEL_TYPES = ["xgboost", "random_forest"]
RANDOM_STATE = 42

# Threshold Settings (default tuned probability thresholds)
DEFAULT_THRESHOLDS = {
    "sepsis": 0.35,
    "cardiac": 0.40,
    "respiratory": 0.38,
    "renal": 0.35,
    "neurological": 0.42
}

# Ollama / CrewAI settings
OLLAMA_API_URL = "http://localhost:11434"
LLM_MODEL = "llama3.1:8b"
USE_OLLAMA_FALLBACK = True  # Fallback to mock LLM if Ollama is not running
