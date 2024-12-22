import os
import sys
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -------------------------------
# Configuration and Environment Setup
# -------------------------------

# If your project root can be determined similarly as before:
# Dynamically set PYTHONPATH
repo_root = os.popen("git rev-parse --show-toplevel").read().strip()
if repo_root and repo_root not in sys.path:
    sys.path.append(repo_root)

# Set necessary environment variables (these were previously done in the original script)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Example environment variables required by torch.distributed
# (If you no longer need these for a single-node run, you can remove them.)
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'

# Adjust paths as needed
PROJECT_ROOT = Path(repo_root) if repo_root else Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "llama-models/models"  # Adjust if needed
if str(MODELS_DIR.parent) not in sys.path:
    sys.path.append(str(MODELS_DIR.parent))

# Path to tokenizer
TOKENIZER_PATH = str(PROJECT_ROOT / "llama-models" / "models" / "llama3" / "api" / "tokenizer.model")

# Default checkpoint directory (adjust as per your environment)
DEFAULT_CKPT_DIR = os.getenv('LLAMA_CKPT_DIR', str(Path.home() / ".llama/checkpoints/Meta-Llama3.1-8B-Instruct"))

# Import LLM classes after paths are set
from models.llama3.reference_impl.generation import Llama
from models.llama3.api.datatypes import (
        UserMessage,
        SystemMessage,
        CompletionMessage,
        StopReason
    )

def get_generator(
    ckpt_dir=DEFAULT_CKPT_DIR,
    tokenizer_path=TOKENIZER_PATH,
    max_seq_len=4096,
    max_batch_size=4,
    model_parallel_size=None
):
    logger.info("Initializing the Llama generator.")
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        model_parallel_size=model_parallel_size,
    )
    return generator
