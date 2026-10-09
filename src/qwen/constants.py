

# constant variables
MODEL_DIR = "../models/qwen2.5-0.5b-instruct"
DATA_DIR = "./data"
LOG_DIR = "./log"
EPS = 1e-5

DEFAULT_MAX_NEW_TOKEN = 256
DEFAULT_MAX_MODEL_LEN = 512
DEFAULT_NUM_BLOCKS = 2048
DEFAULT_MAX_NUM_SEQS = 128
DEFAULT_PIPELINE_DEPTH = 2
DEFAULT_MAX_WAITING = 64
BLOCK_SIZE_FOR_FLASH_ATTENTION = 256

# the curve of sort overhead is almost flat when top_k <=1024
# so 1024 is a sweet spot
MAX_EFFECTIVE_TOP_K = 1024

