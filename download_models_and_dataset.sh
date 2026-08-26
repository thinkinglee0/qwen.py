mkdir -p "$QWEN_MODEL_DIR"

# Use the Python API, NOT the CLI. The console script was renamed between
# huggingface_hub majors (`huggingface-cli` in 0.x, `hf` in 1.x) and is
# absent entirely when the package arrives as a transitive dep with no
# scripts installed. snapshot_download is stable across both.
#
# allow_patterns skips the repo's alternate-format directories, which can
# otherwise double the transfer.
python - <<'PY'
import os
from huggingface_hub import snapshot_download, hf_hub_download

root = os.environ["QWEN_MODEL_DIR"]
KEEP = ["*.json", "*.safetensors", "*.txt", "*.model", "*.py"]

for repo, sub in [
    # 0.5B for pytest: the session-scoped fixture loads weights on every
    # run, and 7B makes that unbearable.
    ("Qwen/Qwen2.5-0.5B-Instruct", "qwen2.5-0.5b-instruct"),
    # 7B only when benchmarking -- comment out otherwise.
    # ("Qwen/Qwen2.5-7B-Instruct",   "qwen2.5-7b-instruct"),
]:
    p = snapshot_download(repo, local_dir=f"{root}/{sub}", allow_patterns=KEEP)
    print("->", p)

# Dataset repo, and only one file out of ~4.2 GB of JSON.
p = hf_hub_download(
    "anon8231489123/ShareGPT_Vicuna_unfiltered",
    repo_type="dataset",                              # <-- the actual fix
    filename="ShareGPT_V3_unfiltered_cleaned_split.json",
    local_dir=f"{root}/sharegpt_data",
)
PY
