import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

OPENAI_API_KEY = os.getenv("GROQ_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("GROQ_API_KEY is not set. Put it in your .env file.")
    
GENERATOR_MODEL = "llama-3.1-8b-instant"       
EVALUATOR_MODEL = "llama-3.1-8b-instant"

# Other config
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "docs")
INDEX_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "index.npy")
META_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "index_meta.txt")

TOP_K = 5                 
GUARDRAIL_THRESHOLD = 0.6  
EVAL_THRESHOLD = 0.7       
MAX_SELF_CORRECT_STEPS = 2
