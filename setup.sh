#!/usr/bin/env bash
# Provision qwen.py deps INSIDE an already-running container.
#
# Use this on Vast.ai, where you get a container, not a Docker host --
# you cannot build images there. This script adapts to whatever torch the
# instance template already shipped instead of reinstalling it, because
# swapping torch is what breaks the flash-attn ABI match.
#
#   bash setup.sh
#
# Idempotent: safe to re-run.
set -euo pipefail

FA_VERSION="${FA_VERSION:-2.8.3}"
REQ="${REQ:-requirements.txt}"

echo "==> existing environment (NOT touching torch)"
python -c "
import sys, torch
print('  python      ', '.'.join(map(str, sys.version_info[:3])))
print('  torch       ', torch.__version__)
print('  cuda (torch)', torch.version.cuda)
print('  cxx11 abi   ', torch._C._GLIBCXX_USE_CXX11_ABI)
"

# Freeze the current state as constraints. Every install below is bound by
# it, so nothing can quietly upgrade torch out from under the wheel we pin.
#
# Two filters, both load-bearing:
#   1. Keep only plain `name==version` lines. In conda/pixi-built images
#      (rattler-build, conda-forge) pip freeze emits local direct
#      references like `packaging @ file:///home/conda/.../work`. Those
#      build dirs do not exist at runtime, and pip fails with OSError the
#      moment it tries to process one.
#   2. Drop CONSTRAINT_EXCLUDE packages. A constraint on something we
#      deliberately need to move (huggingface-hub, because transformers
#      4.46.3 declares <1.0) makes the resolve impossible rather than
#      merely pinned.
CONSTRAINT_EXCLUDE="${CONSTRAINT_EXCLUDE:-huggingface[-_]hub}"
CONSTRAINTS=/opt/constraints.txt

echo "==> snapshotting baseline -> ${CONSTRAINTS}"
mkdir -p /opt
pip freeze \
  | grep -E '^[A-Za-z0-9._-]+==[^ ]+$' \
  | grep -viE "^(${CONSTRAINT_EXCLUDE})==" \
  > "${CONSTRAINTS}"
echo "    $(wc -l < "${CONSTRAINTS}") packages pinned"
grep -c 'file://' "${CONSTRAINTS}" >/dev/null 2>&1 \
  && { echo "    ERROR: direct references survived the filter" >&2; exit 1; } || true

echo "==> resolving flash-attn ${FA_VERSION} prebuilt wheel"
python - <<PY
import json, re, subprocess, sys, urllib.request
import torch

fa  = "${FA_VERSION}"
abi = "TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE"
cp  = "cp%d%d" % sys.version_info[:2]
tmm = ".".join(torch.__version__.split("+")[0].split(".")[:2])

try:
    import flash_attn
    print(f"  flash_attn {flash_attn.__version__} already present, skipping")
    sys.exit(0)
except ImportError:
    pass

url = f"https://api.github.com/repos/Dao-AILab/flash-attention/releases/tags/v{fa}"
with urllib.request.urlopen(url, timeout=60) as r:
    assets = [(a["name"], a["browser_download_url"]) for a in json.load(r)["assets"]]

# Match against the asset NAME, not browser_download_url: the URL is
# percent-encoded, so the '+' in the local version arrives as '%2B' and a
# literal \+ never matches. Accept cu12 or cu13, preferring cu12 (cu13
# wheels need driver >= 580).
pat = re.compile(
    rf"^flash_attn-{re.escape(fa)}\+cu(?P<cu>1[23])torch{re.escape(tmm)}"
    rf"cxx11abi{abi}-{cp}-{cp}-linux_x86_64\.whl$"
)
hit = sorted(
    ((m.group("cu"), u) for n, u in assets if (m := pat.match(n))),
    key=lambda t: t[0],
)
hit = [u for _, u in hit]

if not hit:
    print(f"  NO WHEEL for torch{tmm} / {cp} / abi{abi}", file=sys.stderr)
    print("  available assets:", file=sys.stderr)
    for n, _ in assets:
        print("   ", n, file=sys.stderr)
    print(
        "  -> pick a Vast template whose torch minor has a wheel, or bump "
        "FA_VERSION. Do NOT 'pip install flash-attn' -- that compiles from "
        "source for 1-2h and will OOM.",
        file=sys.stderr,
    )
    sys.exit(1)

print("  resolved:", hit[0].rsplit("/", 1)[-1])
# --no-deps: the wheel's metadata would otherwise be free to pull a
# different torch and silently break the ABI we just matched.
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--no-cache-dir", "--no-deps",
    "-c", "/opt/constraints.txt", hit[0],
])
PY

echo "==> project deps"
pip install --no-cache-dir -c /opt/constraints.txt einops
[ -f "${REQ}" ] && pip install --no-cache-dir -c /opt/constraints.txt -r "${REQ}"

echo "==> verify"
python -c "
import torch, flash_attn, transformers
print('  torch       ', torch.__version__)
print('  flash_attn  ', flash_attn.__version__)
print('  transformers', transformers.__version__)
assert torch.cuda.is_available(), 'no CUDA device visible'
print('  device      ', torch.cuda.get_device_name(0))

# exercise the real varlen kernel, not just the import
from flash_attn import flash_attn_varlen_func
q  = torch.randn(8, 4, 64, dtype=torch.bfloat16, device='cuda')
cu = torch.tensor([0, 8], dtype=torch.int32, device='cuda')
out = flash_attn_varlen_func(q, q, q, cu, cu, 8, 8, causal=True)
print('  flash_attn_varlen_func OK', tuple(out.shape))
"

cat <<'EOF'

==> done. Suggested shell env (add to ~/.bashrc):

    export OMP_NUM_THREADS=8
    export MKL_NUM_THREADS=8
    export QWEN_MODEL_DIR=/workspace/models

Weights (overseas host):
    huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
      --local-dir $QWEN_MODEL_DIR/qwen2.5-7b-instruct

Weights (mainland China host, huggingface.co unreachable):
    pip install modelscope
    modelscope download --model Qwen/Qwen2.5-7B-Instruct \
      --local_dir $QWEN_MODEL_DIR/qwen2.5-7b-instruct
EOF
