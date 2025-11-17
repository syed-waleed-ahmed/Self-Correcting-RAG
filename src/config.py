import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# We now use GROQ_API_KEY instead of OPENAI_API_KEY
OPENAI_API_KEY = os.getenv("GROQ_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("GROQ_API_KEY is not set. Put it in your .env file.")

# Groq model IDs
# You can change these later if you want a different model.
EMBEDDING_MODEL = "nomic-embed-text-v1.5"      # Embeddings via Groq's OpenAI-compatible embeddings endpoint 
GENERATOR_MODEL = "llama-3.1-8b-instant"       # Chat model (fast, good for dev) 
EVALUATOR_MODEL = "llama-3.1-8b-instant"

# Other config
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "docs")
INDEX_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "index.npy")
META_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "index_meta.txt")

TOP_K = 5                  # how many documents to retrieve
GUARDRAIL_THRESHOLD = 0.6  # minimum relevance score from guardrail (0–1)
EVAL_THRESHOLD = 0.7       # minimum factual consistency score (0–1)
MAX_SELF_CORRECT_STEPS = 2
