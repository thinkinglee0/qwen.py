# qwen.py GPU 机器重建 Runbook

> 目标:从零到能跑 pytest,**20–30 分钟**,全程复制粘贴。
> 适用:Vast.ai 容器(无 root capability、无 docker daemon)。

已验证可用组合 —— 偏离这个组合就要重走一遍 wheel 匹配:

```
GPU              RTX 4090 (sm_89) / PCIe 4.0 x16
python           3.12
torch            2.9.1+cu128
flash_attn       2.8.3   (cu12torch2.9cxx11abiTRUE-cp312)
transformers     4.46.3  (golden-reference oracle,不可动)
huggingface_hub  0.36.2  (被 transformers 的 <1.0 约束倒逼)
```

---

## 0. 租机器

Vast.ai 筛选条件,**点 RENT 之前设好,租完改不了**:

| 字段 | 值 | 理由 |
|---|---|---|
| GPU | RTX 4090 | 24 GB 保留 KV 压力;450 W 无 power wall |
| 磁盘 | **150 GB** | 默认 16 GB 装不下 15.2 GB 权重 |
| 计费 | **on-demand**,非 interruptible | benchmark 中途被抢占 = 数据作废 |
| 区域 | 香港 / 日本 / 韩国 | 北京 RTT 40–60 ms |
| Verified | 必须 | 排除家用宽带和民宅电力 |
| Reliability | > 99% | 3 小时 sweep 中途掉线要重跑 |
| PCIe 实测 | ≥ 20 GB/s | 排除 x1/x4 矿机 riser |
| 宿主 RAM | ≥ 48 GB | vLLM + tokenizer + client |
| vCPU | ≥ 8 | 单核弱会拖慢 Python decode loop |
| 模板 | 带 PyTorch 的镜像 | 省一次 torch 安装 |

**租期到期日**:每个 offer 有 max duration,租的那刻就锁定。确认它不会落在 sweep 中间。

---

## 1. 落地体检(3 分钟)

### 1a. 先确认 venv 是激活的

**每开一个新 shell 都要检查。** Vast 模板把所有东西装在 `/venv/main`,只有初始 shell 自动激活;重连、`bash`、tmux 新窗口都可能丢掉它。

提示符前有 `(main)` 就是激活的。没有就:

```bash
source /venv/main/bin/activate
```

```bash
# The single most useful guard in this document. `python` exists ONLY
# inside the venv -- Ubuntu ships python3 with no `python` alias -- so
# "command not found" for python while python3 works means the venv is
# off, NOT that anything is missing.
which python pip
#   expect: /venv/main/bin/python  and  /venv/main/bin/pip
python -V     # expect 3.12.13 (venv), NOT 3.12.3 (system)
```

**venv 没激活就 `pip install` 会装进系统 Python,全部白做,而且报错会指向完全错误的方向。** 每次执行安装类命令前跑一遍上面两行。

写进 `~/.bashrc` 让它自动恢复:

```bash
grep -q 'venv/main/bin/activate' ~/.bashrc \
  || echo 'source /venv/main/bin/activate' >> ~/.bashrc
```

### 1b. 硬件体检

任何一项不合格 → **destroy 重租,别将就**。此时沉没成本为零。

```bash
# --- disk: the writable overlay layer is the ONLY line that matters.
# /dev/nvme* mounted on /etc/hosts or /usr/bin/nvidia-smi are the HOST's
# filesystems bind-mounted in; you cannot use a byte of them.
df -h | head -5
#   expect: overlay  150G  ...  /

# --- shm: docker defaults to 64 MB, which kills vLLM outright
df -h /dev/shm          # expect >= 16G

# --- GPU identity. Do NOT trust the label: A10 (150W) and A10G (300W)
# are both listed as "A10" on some platforms.
nvidia-smi --query-gpu=name,power.limit,power.max_limit,pcie.link.width.current \
  --format=csv
#   expect: NVIDIA GeForce RTX 4090, 450.00 W, 450.00 W, 16

nvidia-smi -q -d POWER | grep -E "Current|Default|Min|Max Power Limit"
#   4090 exposes a 150-450 W range, but see section 6: writing it needs
#   CAP_SYS_ADMIN, which Vast containers do not grant.

# --- CPU. Single-thread perf drives the Python decode loop, so what matters is
# clock x IPC, NOT core count. Read "CPU max MHz" AND the scaling line: a shared
# host pinned at 80 % by neighbour load is a ~2x slowdown on the decode loop that
# no amount of cores fixes.
nproc && lscpu | grep -E "Model name|CPU max MHz|scaling MHz|^NUMA node|L3 cache"

# --- What this container may ACTUALLY use. Ask the kernel; do not guess the cgroup
# path (log1003 read nothing because 1b used the v2 path on a v1 host, and the error
# went to stderr while the capture only redirected stdout).
python3 -c "import os; a=sorted(os.sched_getaffinity(0)); print(len(a),'cpus:',a)"

# The platform's "12.0/48 CPU" is usually a CFS QUOTA, not a cpuset: you see all 48
# cores and may run on any, but total CPU time is capped. quota/period = your cores.
# A single-threaded decode loop can never reach a >1-core quota, so a quota alone
# does NOT explain a slow host -- check the clock instead.
cat /sys/fs/cgroup/cpu.max 2>&1                    # cgroup v2: "<quota> <period>"
cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>&1       # cgroup v1: divide by 100000
cat /sys/fs/cgroup/cpuset.cpus.effective 2>&1      # v2
cat /sys/fs/cgroup/cpuset/cpuset.cpus 2>&1         # v1

# --- Neighbours. Take this AFTER the GPU is idle again: load that persists once your
# own run has ended is somebody else's, and it is what holds the clock down.
uptime

# --- Pure-interpreter speed. The launch-overhead test below does NOT capture this --
# it times one op in a tight loop, while the engine pays Python dispatch on ~4 400
# aten calls per step. Record it on every host; it is the only number that compares
# decode-loop speed across machines.
for i in 1 2 3; do python -c "
import time
t = time.perf_counter()
s = 0
for i in range(3_000_000): s += i * 2
print(f'{(time.perf_counter()-t)*1000:.0f} ms / 3M-iter loop')"; done
#   reference: ~276 ms on EPYC 7402 @ 2.24 GHz effective (log1003, a SLOW host)
#   a Zen 4 desktop part at 3.7-5.0 GHz lands roughly 2x lower

# --- NUMA. Find the socket the GPU hangs off.
nvidia-smi topo -m | head -3                 # read the "NUMA Affinity" column
numactl --hardware | grep -E "^node . cpus|^node . free"
```

### launch overhead 基线

这是 benchmark 可信度的门槛。**warmup 必须有** —— 没有 warmup 会高估 20%(CUDA context 初始化 + module lazy loading + CPU 频率爬坡)。

```bash
for i in $(seq 5); do
python -c "
import time, torch
x = torch.randn(1, device='cuda'); torch.cuda.synchronize()
for _ in range(1000): x.add_(1.0)              # warmup, not optional
torch.cuda.synchronize()
t = time.perf_counter()
for _ in range(10000): x.add_(1.0)
torch.cuda.synchronize()
print(f'{(time.perf_counter()-t)/10000*1e6:.2f}')"
done
uptime      # snapshot neighbour load alongside the number
```

判读:

- **中位数 < 10 μs** → 合格。7B 每 decode step 约 600 次 launch,CPU 侧约 4.3 ms,而 GPU 侧 15.2 GB ÷ 1008 GB/s = 15.1 ms,**有 3.5× 余量**
- **五次的散布**就是这台宿主的噪声底。参考值 ±3%,外加约 20% 概率出现一个高 14% 的离群点(邻居干扰)
- **后续任何小于 10% 的性能差异,单次测量不能声称**

> ⚠️ **这个门槛过了,主机仍然可能慢一倍 —— log1003 就是。** 那台 EPYC 7402 的 launch
> overhead 中位数是 **7.56 μs**(过线,还优于参考的 8.6–8.8),但同一份代码的 `wall_clean`
> 比前一台慢 **2.3×**,而 `gpu_busy` 一致到 ±2 %。原因:这个微基准只量**一个 op 在紧循环里**
> 的驱动路径,而引擎每步要为 **约 4 400 次 `aten::` 调用**付解释器成本。
>
> 所以 §1b 的 CPU 块里那个 **3M 次纯 Python 循环**是独立的第二道门,必须一起看:
>
> | 两项 | 含义 |
> |---|---|
> | launch < 10 μs **且** 解释器 ≈ 参考值 | 合格 |
> | launch 过线但解释器慢 ~2× | **设备侧结论仍可信**(kernel 不变),但**主机瓶颈区的数字不可跨机器迁移** —— batch 小的那一半全是这台 CPU 的 Python 速度 |
> | launch > 10 μs | destroy 重租 |
>
> 解释器慢的根因看 `CPU max MHz` × `scaling MHz` 和 `uptime`:log1003 是
> max 2800 × 80 % ≈ **2.24 GHz**,而 `load average 12.1`(自己跑完 12 分钟后测的)说明
> 那是邻居把时钟压住的。

> 注意 7B 和 0.5B 处于不同 regime:0.5B 只有 ~1.0 GB 权重,GPU 侧约 1.0 ms < CPU 侧 4.3 ms,**完全 CPU-bound**。用 0.5B 跑性能对比,测到的是 Python 循环 vs CUDA graph,不是 scheduler 设计。

---

## 2. 装依赖(5 分钟)

```bash
cd /workspace
git clone https://github.com/thinkinglee0/qwen.py.git && cd qwen.py
```

### 2a. torch 版本对齐

flash-attn 只发布 **预编译 wheel 到 GitHub Releases**,PyPI 上没有 —— `pip install flash-attn` 会触发 1–2 小时的源码编译并 OOM。wheel 按四个坐标匹配:`cu{12|13}` × `torch{minor}` × `cxx11abi{TRUE|FALSE}` × `cp{version}`。

**模板自带的 torch 常常太新而没有对应 wheel。** 先查:

```bash
python -c "import torch,sys; print(torch.__version__, torch._C._GLIBCXX_USE_CXX11_ABI, 'cp%d%d'%sys.version_info[:2])"

# sed: the API returns each asset twice (name + percent-encoded URL);
# normalise %2B back to + before sorting or you get every wheel listed twice.
curl -s https://api.github.com/repos/Dao-AILab/flash-attention/releases/tags/v2.8.3 \
  | grep -o 'flash_attn-[^"]*\.whl' | sed 's/%2B/+/' \
  | grep cp312 | grep x86_64 | sort -u
```

列表里没有当前 torch minor → 降 torch(**不是升 flash-attn**)。

**降到 2.9,不要降到 2.10。** 2.10 只有 `cu13` 变体,而 cu13 需要 driver ≥ 580,并且会把整套 CUDA 13 的 `nvidia-*` wheel 重新拉一遍(GB 级)。2.9 有 `cu12` 变体,复用镜像里已有的 cu12 运行时,实测只新增 4 个包:

```bash
nvidia-smi --query-gpu=driver_version --format=csv   # < 580 rules out cu13 outright

pip uninstall -y torchvision torchaudio    # they hard-pin the old torch and break
pip install --no-cache-dir torch==2.9.* --index-url https://download.pytorch.org/whl/cu128
```

### 2b. 一键装完

```bash
bash env/vastai/setup.sh
```

`setup.sh` 会:读出当前 torch/ABI/cp → 生成 constraints → 精确匹配 wheel → 安装 → 实跑 `flash_attn_varlen_func` 验证。**匹配不到就报错退出,绝不静默回退到源码编译。**

没有脚本时的等价手工流程:

```bash
# Constraints must be filtered to plain name==version lines. A conda/pixi
# built venv (this template is rattler-build) makes pip freeze emit
# `pkg @ file:///home/conda/.../work`; those dirs are gone at runtime and
# pip dies with OSError the moment it processes one.
# huggingface-hub is excluded on purpose: transformers 4.46.3 declares
# <1.0, so a ==1.x constraint makes the resolve impossible.
pip freeze \
  | grep -E '^[A-Za-z0-9._-]+==[^ ]+$' \
  | grep -viE '^huggingface[-_]hub==' \
  > /opt/constraints.txt
grep -c 'file://' /opt/constraints.txt      # must be 0

FA=https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3
pip install --no-cache-dir --no-deps -c /opt/constraints.txt \
  "$FA/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
pip install --no-cache-dir -c /opt/constraints.txt einops
pip install --no-cache-dir -c /opt/constraints.txt -r requirements.txt
```

`--no-deps` 装 wheel 是关键:否则它的依赖解析可能换掉 torch,把刚匹配好的 ABI 毁掉。

### 2c. 验证

```bash
python -c "
import torch, flash_attn, transformers, huggingface_hub
print('torch          ', torch.__version__)
print('flash_attn     ', flash_attn.__version__)
print('transformers   ', transformers.__version__)      # must be 4.46.3
print('huggingface_hub', huggingface_hub.__version__)   # must be 0.x
print('capability     ', torch.cuda.get_device_capability(0))
from flash_attn import flash_attn_varlen_func
q  = torch.randn(8, 4, 64, dtype=torch.bfloat16, device='cuda')
cu = torch.tensor([0, 8], dtype=torch.int32, device='cuda')
print('varlen OK      ', tuple(flash_attn_varlen_func(q, q, q, cu, cu, 8, 8, causal=True).shape))
"
```

报 `no kernel image is available for execution on the device` = wheel 的 arch list 不含 sm_89,换 wheel 源。

---

## 3. 环境变量

写进 `~/.bashrc`,别每次手敲。宿主报 128 threads,放任 torch 开满会自己制造 CPU 瓶颈。
顺便写 `~/.gitconfig`。

> ⚠️ **这两个变量不是只打在 qwen.py 身上。** 它们是 `export` 到环境里的,vLLM 启动时会**继承**(它只在未设置时才自己设),所以默认情况下**两边都是 8**。这符合 §6e"加就两边都加"的纪律,但必须是有意识的 —— 见 §6a 的"起服务时显式处理线程环境变量"。

完成后，**重新登录**cloud instance。

```bash
cat >> ~/.gitconfig <<'EOF'
[alias]
  co = checkout
  ci = commit
  br = branch
  st = status
EOF

cat >> ~/.bashrc <<'EOF'
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export QWEN_MODEL_DIR=/workspace/models
export HF_HOME=/workspace/hf
EOF
```

---

## 4. 下权重(5–10 分钟)

执行以下命令：

```bash
bash download_models_and_dataset.sh
```

其中脚本`download_models_and_dataset.sh`内容如下（无需独立执行）：

```bash
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
```

链路快时可以开并行分片下载(可选,失败就删掉这行重来):

```bash
pip install hf_transfer && export HF_HUB_ENABLE_HF_TRANSFER=1
```

境内机器(huggingface.co 不可达)二选一:

```bash
export HF_ENDPOINT=https://hf-mirror.com     # same script above, or:

modelscope download --model Qwen/Qwen2.5-7B-Instruct \
  --local_dir "$QWEN_MODEL_DIR/qwen2.5-7b-instruct"
```

---

## 5. 装 qwen.py 本体并跑测试

```bash
which python pip     # venv guard again -- see 1a
```

```bash
# --no-deps on purpose: requirements.txt is the authoritative dependency
# list here, and it carries the load-bearing transformers==4.46.3 pin.
# Letting pip re-resolve from pyproject.toml can quietly move transformers
# or huggingface-hub and undo section 2.
pip install -e . --no-deps

python -c "import qwen; print('qwen.py importable from', qwen.__file__)"
```

装完 `pytest` 才能从任意目录发现 package,而不是依赖 cwd 恰好是 repo 根目录。

### NUMA 绑定:每台机器重新推导,不要照抄

```bash
# 1. which NUMA node the GPU hangs off
nvidia-smi topo -m | head -3        # read the "NUMA Affinity" column for GPU0

# 2. which nodes share that socket
numactl --hardware | sed -n '/node distances/,$p'
```

读法:distance 矩阵里 **10 = 自己,11-12 = 同 socket 跨 CCD,21-32 = 跨 socket**。把和 GPU 所在 node 距离 < 20 的那一组全取上。

例(某台双路 EPYC,NPS4):GPU 在 node 4,矩阵显示 `{0,1,2,3}` 与 `{4,5,6,7}` 两簇、簇内 12 跨簇 32 → 取 `4,5,6,7`。**换一台机器这组数字就变,单路机器可能只有 node 0。**

```bash
export OMP_NUM_THREADS=8
numactl --cpunodebind=<the nodes you just derived> pytest -x -v
```

三条规则:

- **绑整个 socket,不绑单个 node。** 单 node 通常只有 8 个物理核,vLLM 的 API server + tokenizer + benchmark client 挤在上面会自己造出 CPU 瓶颈,反而污染结果。
- **不要加 `--membind`。** 它是硬约束,超了直接 OOM 而不是回退。共享宿主上单 node 的 free memory 可能只剩几 GB,加载 15.2 GB 权重必撞墙。只给 `--cpunodebind`,内核默认就是本地优先 + 不足回退。
- **绑定是为了压方差,不是提均值。** 某台宿主上实测 8.8 μs(绑)vs 8.6 μs(不绑),在噪声内。真正的收益是防止调度器在 3 小时 sweep 中途把进程迁到另一个 socket。**因此 qwen.py 和 vLLM 两边要么都加,要么都不加。**

`numactl` 在这台宿主上不降均值(实测 8.8 vs 8.6 μs,在噪声内),**保留它是为了压方差** —— 防止调度器在 3 小时 sweep 中途把进程迁到另一个 socket。**加就两边都加,qwen.py 和 vLLM 必须一致。**

#### 单 NUMA node 的机器上,要绑的是 CCX,不是 node

`numactl --hardware` 只报一个 node 时,很容易得出"没什么可绑的"这个结论 —— **那是错的**,因为它只说明**内存**局部性没得优化。**L3 局部性是另一条轴**:

```bash
lscpu | grep -E "L3 cache|Core\(s\) per socket"
#   log1003: L3 cache 128 MiB (8 instances) / 24 cores  -> 8 CCX x 3 cores, 16 MiB each
python3 -c "import os; print(len(os.sched_getaffinity(0)), 'cpus visible')"
#   log1003: 48 -- the one hot Python thread is free to migrate across ALL of them
```

decode 循环是**单线程、延迟敏感**的。调度器把它在 8 个独立 L3 之间迁移一次,解释器的工作集就作废一次。Zen 2/3 的 EPYC 上尤其明显(每 CCX 只 3–4 核、L3 不共享)。

```bash
# bind to ONE CCX: its physical cores plus their SMT siblings
lscpu -e=CPU,CORE,SOCKET,L3 | head -20          # read which CPUs share an L3
taskset -c 0,1,2,24,25,26 pytest -x -v          # log1003's CCX0: cores 0-2 + siblings
```

两点注意:

- **这是压方差兼提均值,但量级有限。** log1003 的 2.3× 主项是时钟(2.24 GHz),CCX 绑定大概找回 10–20 %,抹不平它。先看时钟,再谈绑定。
- **同 §3 的纪律:加就两边都加。** qwen.py 和 vLLM 必须用同一条绑定策略,并记进结果元数据(§7)。

### GPU 上第一次跑要预期失败

Mac 上的 tolerance 是按 **CPU / fp32 / SDPA-math** 标定的,GPU 上三个变量同时变了:device、dtype(bf16 只有 8 位尾数)、attention kernel(flash 的 tiling 彻底改写累加顺序)。误差量级会高几个数量级。

重新标定的顺序:

1. 固定 GPU + bf16,先测 **flash vs SDPA** 两条 backend 路径的差异 —— 纯 kernel 差异,无实现差异,得到"同一数学表达式在两种 tiling 下的分歧上界"
2. 用这个上界去校准 qwen.py vs HF reference 的 tolerance
3. HF 侧显式指定 `attn_implementation`,否则你在比较四个变量而不是一个

---

## 6. benchmark 工具链:vLLM + nsys + torch.profiler

### 6a. vLLM —— 必须独立环境

**绝对不能装进 `/venv/main`。** vLLM 会 pin 自己的 torch,一旦替换掉 2.9.1,`cu12torch2.9` 那个 flash-attn wheel 立刻 ABI 失配,报 `undefined symbol: _ZN3c10...`,第 2 节的成果全废。

但"独立"有两层,只做到第一层会在装完之后才炸:

1. **包隔离** —— 独立 venv,别污染 `/venv/main` 的 torch。
2. **解释器隔离** —— **不要从 `/venv/main` 派生 venv**。这个模板是 rattler-build 构建的,从它 `python -m venv` 出来的新环境,base interpreter 仍是 `/venv/main/bin/python3.12`,进程的动态链接继承 conda base 的 RPATH。conda 的 `libstdc++.so.6` 常比系统的旧,vLLM 的预编译扩展就报 `GLIBCXX_3.4.xx not found` —— 和 `undefined symbol: _ZN3c10...` 是同一类病,换了个库而已。顺带一提有些 conda build 裁掉了 `ensurepip`,那样 `python -m venv` 当场就失败。

#### 选文件系统

```bash
# 15 GB 的东西不要默认丢 /opt。挂了 volume 的实例上 /workspace 是 volume、
# /opt 在 overlay 上,两者可用空间能差一个数量级。
df -h /opt /workspace        # 需要 >= 20 GB(15 GB 包 + 余量)
mkdir /workspace/vllm-venv
VLLM_VENV=/workspace/vllm-venv
```

#### 建环境 —— 用系统 python,不要用 `/venv/main`

**这个 venv 的 Python 版本不需要是 3.12。** cp312 那个约束来自 flash-attn wheel 的四坐标(§2a),而 flash-attn 不在这个环境里 —— 只要系统 python 落在 vLLM 的 `Requires-Python` 范围内就行。不匹配的话下一步的 `--dry-run` 会当场拒绝,不用等到装完。

```bash
# whatever python3 the distro ships; version only has to satisfy vLLM itself
/usr/bin/python3 -V
/usr/bin/python3 -m venv "$VLLM_VENV" \
  || { apt-get install -y python3-venv && /usr/bin/python3 -m venv "$VLLM_VENV"; }
"$VLLM_VENV/bin/pip" install --no-cache-dir -U pip
```

为什么是系统 python 而不是 conda:PyPI 的 manylinux wheel 是照着**发行版的** glibc/libstdc++ 基线编译的,系统 python 的 venv 直接链到 `/usr/lib/x86_64-linux-gnu/libstdc++.so.6`,正是 wheel 作者预期的那个。conda 的解释器自带 `$CONDA_PREFIX/lib/libstdc++.so.6` 并带 RPATH —— 那正是上面第 2 条要躲的 `GLIBCXX` 坑,换个牌子而已。

#### 装之前先查 torch 变体 —— 2a 的规则对 vLLM 一样生效

`pip install vllm` 不 pin 版本就会装最新的,而 **vLLM pin 自己的 torch**。如果那个 torch 是 `cu13` 变体,就踩中 2a 已经处理过的那条线:cu13 需要 driver ≥ 580。

```bash
# Pick the pin HERE and record it (section 7). Do NOT copy a version out of this
# doc: like the nsys URL in 6b, it goes stale, and a silently-moved vLLM makes the
# A/B unreproducible. `pip index versions vllm` lists what is installable today.
VLLM_PIN="v0.30.0"      # quoted, v0.30.0 is latest at 2026.10.03

nvidia-smi --query-gpu=driver_version --format=csv,noheader   # < 580 rules out cu13 outright

# --dry-run resolves without downloading: read which torch / nvidia-* wheels it wants
"$VLLM_VENV/bin/pip" install --no-cache-dir --dry-run vllm=="$VLLM_PIN" 2>&1 \
  | grep -iE 'torch|nvidia-cuda'
```

#### 装

```bash
# --no-cache-dir is load-bearing here, not hygiene: vLLM ships a full torch +
# CUDA runtime (10-15 GB), and pip's cache keeps a SECOND copy of every wheel in
# ~/.cache/pip -- 20-30 GB total. See section 8's "No space left on device" row.
# Pin the version: it is the only thing that makes the A/B reproducible later.
"$VLLM_VENV/bin/pip" install --no-cache-dir vllm=="$VLLM_PIN"
```

#### 双向验证隔离 —— 而且要真的碰一次 GPU

两条都要跑,只查一边看不出污染。`import` 成功**不等于** runtime 能跑:cu13 装在 550 驱动上照样 import 得动,真正的报错推迟到 `vllm serve` 加载模型时才出现 —— 那时 15 GB 和半小时已经花掉了。

```bash
"$VLLM_VENV/bin/python" -c "
import torch, vllm
print('vllm venv', vllm.__version__, torch.__version__, 'cuda', torch.version.cuda)
assert torch.cuda.is_available(), 'vLLM torch cannot see the GPU'
x = torch.randn(8, device='cuda'); x.add_(1.0); torch.cuda.synchronize()
print('kernel OK ', torch.cuda.get_device_name(0))"

python -c "import torch, flash_attn; print('main venv', torch.__version__, flash_attn.__version__)"
#   main venv MUST still read 2.9.1+cu128 / 2.8.3
```

#### 起服务时显式处理线程环境变量

第 3 节把 `OMP_NUM_THREADS=8` **export 进了 `~/.bashrc`**,vLLM 启动时会**继承**它(它只在未设置时才自己设)。这不是"只打在 qwen.py 身上"—— 两边都是 8。合不合适是另一回事,§6e 的纪律是**必须是一个有意识的决定**:

```bash
# arm A: keep the host-wide setting on both engines (current default -- both get 8)
# arm B: let vLLM pick its own
env -u OMP_NUM_THREADS -u MKL_NUM_THREADS "$VLLM_VENV/bin/vllm" serve ...
```

无论选哪条,**记进结果元数据**,否则下次改 `~/.bashrc` 就会悄悄破掉 parity。

#### fallback:只在系统 python 不可用时才走 micromamba

**不是并列选项。** conda 装 vLLM 的那一步仍然是 pip(`conda run -n vllm pip install vllm`)—— torch 和整套 `nvidia-*` runtime 全来自 PyPI、由 vLLM 自己 pin,conda 唯一贡献的是解释器,而那个解释器重新引入上面的 `GLIBCXX` 风险。另外这个模板是 rattler-build 构建的,**PATH 上并不一定有 `conda` CLI**,走这条路等于先装一个包管理器 —— 在 23 GB 空闲和 §6e 要求的连续时间窗里,这都是净成本。

**唯一的触发条件**:系统 python 的版本不满足 vLLM 的 `Requires-Python`,而 `apt-get install python3-venv` 也走不通(无 apt 权限,或镜像的 Ubuntu 太老)。

```bash
command -v micromamba || command -v conda || echo "neither -- fix the system python instead"

# Pick PY_VER at the time, like VLLM_PIN: you are here BECAUSE the system python's
# version did not fit, so 3.12 is not a safe default -- read vLLM's Requires-Python
# for the pin you chose. Record both in section 7.
PY_VER="<choose, e.g. 3.12>"      # quoted: bare <> is a shell redirect

micromamba create -y -n vllm python="$PY_VER"
micromamba run -n vllm pip install --no-cache-dir vllm=="$VLLM_PIN"
#   then re-run the GPU verification above, and expect GLIBCXX trouble (section 8)
```

隔离是硬要求,但**用什么做隔离在这里不是次要的** —— 它决定 vLLM 的扩展链到哪个 `libstdc++`。无论走哪条路,"解释器不要继承 `/venv/main`"和"`--no-cache-dir`"两条都要保留。

#### 用完销毁

```bash
rm -rf "$VLLM_VENV"          # 回收 10-15 GB
```

### 6b. nsys

先查:CUDA devel 镜像里可能自带。

```bash
which nsys || ls /usr/local/cuda*/bin/nsys 2>/dev/null
```

**下载页需要交互式获取链接,不能写死 URL** —— NVIDIA 的下载地址带 `__token__=exp=...` 签名,有效期只有几小时,抄进文档几天后必然 403。

流程:浏览器打开 `developer.nvidia.com/nsight-systems/get-started` → 选 **Linux CLI Only (.deb, x86_64)** → 复制那个带 token 的链接 → 在机器上:

```bash
apt-get update && apt-get install -y wget libglib2.0-0
wget -O nsys-cli.deb "<paste the signed URL here>"
apt-get install -y ./nsys-cli.deb     # note the ./ -- apt resolves deps for local debs
which nsys && nsys --version
```

**选 CLI-only,不要选完整版。** 完整包带 GUI(`nsys-ui`),会拖进整套 Qt/X11 依赖 —— 这正是完整包必须靠 `apt --fix-broken install` 才能装上的原因。无头容器里 GUI 一点用没有,报告拷回本地用桌面版 Nsight 打开。

无 apt 权限时 deb 本质是 tarball,可以纯解包:

```bash
dpkg-deb -x nsys-cli.deb /opt/nsys
export PATH="$PATH:$(dirname $(find /opt/nsys -name nsys -type f | head -1))"
```

**验证权限边界:**

```bash
nsys status -e
```

Vast 容器缺 `CAP_SYS_ADMIN`,所以 `perf_event_paranoid` 会挡掉 **CPU sampling 和 context switch tracing**。但 **CUDA tracing 走 CUPTI activity API,不受这个限制** —— 而你要的 decode step 之间的 GPU 空隙恰好只需要 CUDA trace:

```bash
nsys profile -t cuda,nvtx --sample=none --cpuctxsw=none \
  -o decode --force-overwrite true \
  python profile_decode.py --nvtx
```

⚠️ **`ncu` 是另一回事**:它读 kernel 级性能计数器,需要宿主设 `NVreg_RestrictProfilingToAdminUsers=0`,在 Vast 上拿不到,会报 `ERR_NVGPUCTRPERM`。M6 做 Triton kernel 优化时得另找机器,benchmark 阶段用不到。

### 6c. torch.profiler —— 零安装

已经在 torch 里,**不需要 `pip install` 任何东西**。`export_chrome_trace()` 出来的 JSON 直接在 `perfetto.dev` 打开,连 `torch-tb-profiler` 都不用装。

```bash
python profile_decode.py --steps 50
```

输出的 **idle fraction** 是这次对比的核心判据:7B 上 GPU 侧约 15.1 ms/step、CPU 侧约 600 × 7.1 μs = 4.3 ms,**idle fraction 应接近 0**。若显著大于 0,说明 CPU 没跟上,你与 vLLM 的差距主要来自 launch overhead 而非 scheduler 设计 —— 结论完全不同。

### 6d. 双臂对照

单跑一个比值没有说服力:

| 臂 | vLLM 配置 | 差距归因 |
|---|---|---|
| A | `--enforce-eager` | 你的 scheduler + KV 管理 + attention kernel 选择 |
| B | 默认(CUDA graph) | A→B 的增量 = CUDA graph 消除的 CPU launch 开销 |

**先选对驱动方式 —— 离线和 serving 不是一回事,用错了对不上。**

| qwen.py 这边的测试 | vLLM 这边的对应物 | 为什么 |
|---|---|---|
| `test_benchmark_sweep_batch_size`<br>`test_benchmark_on_pc` | **离线 `LLM.generate()`** —— 见 [`benchmark/tool/vllm_sweep.py`](../../benchmark/tool/vllm_sweep.py) | `engine.benchmark()` 在计时**之前**就把全部请求塞进 waiting 队列,然后 `run_to_completion()` 排空 —— 到达率无穷的闭环,没有 HTTP、没有 tokenizer 在环。套 server + client 会给 vLLM 加上你不付的开销,**反而低估它** |
| HTTP 服务路径(`ServingDriver`) | `vllm serve` + benchmark client | 有到达率、有并发爬坡时才用这条 |

离线臂:

```bash
ARM=eager   "$VLLM_VENV/bin/python" benchmark/tool/vllm_sweep.py --arm eager     --runs 3
ARM=default "$VLLM_VENV/bin/python" benchmark/tool/vllm_sweep.py --arm cudagraph --runs 3
```

serving 臂(只在对照 HTTP 路径时):

```bash
"$VLLM_VENV/bin/vllm" serve "$QWEN_MODEL_DIR/qwen2.5-7b-instruct" \
  --dtype bfloat16 --no-enable-prefix-caching --enforce-eager --port 8001
```

**必须对齐的清单** —— 前三条错了结论直接反向:

1. **吞吐定义:只数 output token。** vLLM 的 harness 也报 "total token throughput" =(prompt + output)/s,在 512-in / 128-out 这个固定形状下**正好是 5 倍**。qwen.py 报的是 `o_tok_num/elapsed`,要对的是 vLLM 的 "Output token throughput"。
2. **采样参数逐项抄 `generation_config.json`**(temp 0.7 / top_k 20 / top_p 0.8 / rep_pen 1.1)。这不是无关紧要的旋钮 —— log1001 测出采样器占 qwen.py 设备工作的 60 %,关掉它就把被测对象本身删掉了。而且 vLLM 在**没有任何请求**需要 penalties/top-k/top-p 时会走更便宜的分支,**部分对齐比不对齐更糟**。
3. **`ignore_eos` + `max_tokens` 两边都要设**,并断言总输出 token 数恰好 `req_num × 128` —— 短了说明 EOS 漏过去了,两臂做的功不再相等。
4. **KV 容量对齐绝对 token 数,不是对齐 `block_size`。** qwen.py 是 `num_blocks 4096 × block_size 256` = **1 048 576 个 slot = 12.0 GiB**;vLLM 不支持 256,用 `--block-size 16 --num-gpu-blocks-override 65536` 配出同样的总量,并在启动日志里核对它没被显存反压掉。
5. `--max-num-seqs` = sweep 变量、`--max-model-len 1024`、`--max-num-batched-tokens 8192`(决定一步挤几个 prefill)、`--no-enable-prefix-caching`(随机 token 下命中率本来≈0,关掉只为干净)。
6. **`detokenize=False`** —— 容易漏。vLLM 默认增量 detokenize,是真实 CPU 开销,而 qwen.py 循环里一次都不做。

**消不掉、必须披露的差异**:block_size 256 vs 16(总量对齐了,但 block table 遍历和碎片行为不同);vLLM V1 的 EngineCore 跑在独立进程(多占一核 + 用 `/dev/shm`);两边采样分布不逐位相同 —— **吞吐可比,不要声称输出一致**。

### 6e. 采样与纪律

每个 concurrency 点并行采集:

```bash
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,power.limit,\
temperature.gpu,utilization.gpu,\
clocks_throttle_reasons.sw_power_cap,clocks_throttle_reasons.hw_thermal_slowdown \
  --format=csv,noheader -lms 200 > sweep_c${C}.csv
```

- 每点跑 ≥ 3 次,**报 median 不报 mean**(离群点会拽高 mean)
- qwen.py 和两条 vLLM 基线**必须在同一次租用、同一台 host、同一个连续时间窗口内跑完**。宿主 CPU 换一台就全部作废 —— log1003 实测:同一份代码换台机器,`gpu_busy` 一致到 ±2 %,但 `wall_clean` 差 **2.3×**
- **一次只跑一个引擎**。两边同时占 GPU,数据全废
- **profiling 和计时分开跑**。nsys 和 torch.profiler 的 CUPTI 回调在每次 kernel launch 上加几微秒,带 profiler 测出的 throughput 不能当 benchmark 结果报
- 每轮记 `uptime`,事后判断某点是否被邻居污染。**log1003 漏了这一项**,补采时才发现 load average `12.11 / 12.81 / 12.96` 是在自己跑完 12 分钟后测的 —— 全是邻居,而那正是时钟被压在 80 % 的原因。漏了它,"这台为什么慢一倍"就只剩推断
- **别只采 GPU 侧。** 上面那条 `nvidia-smi` 给的是 `clocks.current.sm` 和 throttle reason,但主机侧的降频它看不见。每轮另记一次 `lscpu | grep "scaling MHz"` + `uptime`:

```bash
{ date -Is; uptime; lscpu | grep -E "scaling MHz|CPU max MHz"; } >> host_clock_c${C}.log
```

判读吞吐差异之前先对一遍这两个数,否则引擎的差距和宿主的降频分不开

**`nvidia-smi -pl` 在 Vast 容器里不可用** —— 报 `Insufficient Permissions`,容器缺 `CAP_SYS_ADMIN`,容器内 root 不等于 capability。`-lgc` 锁频走同一条权限路径,同样不行。要做 power sweep 得换 AWS g5(VM + 直通,guest 有 root)或裸金属。

---

## 7. 销毁前带走什么

overlay 可写层随 destroy 消失。**离开前确认:**

- [ ] 代码 `git push`
- [ ] benchmark 结果 CSV / JSON 上传(`git`、对象存储、或 `scp` 到本地)
- [ ] `/opt/constraints.txt` 存一份 —— 下次重建时 diff 一下就知道模板变了没有
- [ ] 记下 instance ID、host ID、`nvidia-smi -q` 快照、`lscpu` 输出、launch overhead 中位数 —— **这些是判断下次的数据能否和这次拼接的唯一依据**
- [ ] 加记主机侧四项(log1003 的教训,缺一项就只能靠推断):**`CPU max MHz` × `scaling MHz`**、**3M 次 Python 循环的毫秒数**、**跑完之后的 `uptime`**、**绑定策略**(无绑定 / `--cpunodebind` / `taskset` 到哪个 CCX)。launch overhead 过线**不能**代替这四项 —— 它只量一个 op,量不到解释器
- [ ] 跑过 vLLM 对照的话,记下 `vllm.__version__` + 它那个 venv 里的 `torch.__version__` / `torch.version.cuda`,解释器的来源和版本(系统 python 还是 micromamba fallback,§6a —— 它决定链到哪个 `libstdc++`),以及 `OMP_NUM_THREADS` 用的是哪条臂 —— **和记 instance ID 同等性质:A/B 可复现的唯一依据**

权重不用备份,重下比传快。`$VLLM_VENV` 也不用,`rm -rf` 掉省 10–15 GB。

---

## 8. 故障速查

| 症状 | 根因 | 解 |
|---|---|---|
| `No space left on device` 在下权重时 | overlay 只有 16 G(默认) | destroy 重租,磁盘设 150 G |
| `df` 显示 1.9T 可用却写不进去 | 那是宿主 fs 的 bind mount,不是你的 | 只看 `overlay` 那一行 |
| flash-attn `NO WHEEL for torchX.Y` | 模板 torch 太新 | 降 torch,别升 flash-attn |
| `pip install flash-attn` 卡住/OOM | 触发源码编译 | 只装 GitHub Releases 的 wheel |
| `undefined symbol: _ZN3c10...` | torch ABI 与 wheel 不匹配 | 四坐标重新对齐;别用 NGC 镜像 |
| `OSError: No such file .../conda/.../work` | conda 环境 `pip freeze` 产生 `@ file://` 引用 | constraints 过滤成纯 `name==version` |
| `huggingface-hub` 解析冲突 | transformers 4.46.3 要求 `<1.0` | 从 constraints 剔除 hf-hub |
| `datasets` 拖回 hub>=1.0 | datasets 4.x+ 与 oracle 冲突 | 不装 datasets,ShareGPT 直接 `json.load()` |
| `nvidia-smi -pl` Insufficient Permissions | 容器无 `CAP_SYS_ADMIN` | 换 VM/裸金属,或放弃 power sweep |
| nsys deb 下载 403 | 签名 URL 的 `__token__` 过期(仅数小时) | 回下载页重新取链接,不要复用旧的 |
| nsys 装完拖进一堆 Qt/X11 依赖 | 装了完整版而非 CLI-only | 选 Linux CLI Only 的 deb |
| `nsys status -e` 报 paranoid level | 容器无 `CAP_SYS_ADMIN`,CPU sampling 被禁 | 加 `--sample=none --cpuctxsw=none`,CUDA trace 不受影响 |
| `ncu` 报 `ERR_NVGPUCTRPERM` | 宿主未设 `NVreg_RestrictProfilingToAdminUsers=0` | Vast 上无解,换裸金属 |
| 装完 vLLM 后 flash_attn 报 undefined symbol | vLLM 把 `/venv/main` 的 torch 换掉了 | vLLM 必须独立 venv,重装主环境 torch 2.9.1 |
| 装 vLLM 时 `No space left on device` | 漏了 `--no-cache-dir`,`~/.cache/pip` 又存了一份 wheel(共 20–30 GB) | `rm -rf ~/.cache/pip`,重装时加 `--no-cache-dir`;并确认装在空闲最大的那个 fs 上(§6a) |
| vLLM 扩展报 `GLIBCXX_3.4.xx not found` | venv 从 `/venv/main` 派生,继承了 conda base 的旧 `libstdc++` | 用 `/usr/bin/python3 -m venv` 重建,别用 `/venv/main` 的 python(§6a) |
| `python -m venv` 报 `ensurepip is not available` | conda/rattler build 裁掉了 ensurepip | 同上,改用系统 python |
| vLLM 起不来 / 卡在 worker 初始化 | `/dev/shm` 是 docker 默认的 64 MB | 容器内 remount 不了(缺 `CAP_SYS_ADMIN`,和 `-pl` 同一条权限路径)。destroy 重租并设 shm ≥ 16 G;单卡临时解法是绕开多进程 executor,**具体开关随 vLLM 版本变,在机器上现查** |
| `no kernel image is available` | wheel arch list 无 sm_89 | 换 wheel 源 |
| 换台机器后吞吐腰斩,但 `gpu_busy_from_trace` 一致到 ±2 % | **主机侧慢,不是 GPU 慢。** GPU 在干同样的活,Python 跑得慢一半 | 查 `CPU max MHz` × `scaling MHz`(log1003: 2800 × 80 % = 2.24 GHz)、3M 循环毫秒数、跑完后的 `uptime`。设备侧结论仍可信,主机瓶颈区(小 batch)的数字不可跨机器迁移(§1b、§6e) |
| launch overhead 7.5 μs 过线,但引擎就是慢 | 微基准只量一个 op 的驱动路径,量不到约 4 400 次 `aten::` 调用的解释器成本 | 跑 §1b 的 3M 次 Python 循环,那才是 decode 循环的尺子 |
| `cat /sys/fs/cgroup/cpuset.cpus.effective` 无输出 | 用了 cgroup **v2** 的路径,而宿主是 **v1**;且 `cat` 的报错走 stderr,只重定向 stdout 就看不见 | 用 `os.sched_getaffinity(0)` 问内核,并给捕获加 `2>&1`(§1b) |
| `numactl --hardware` 只有 1 个 node,"没什么可绑的" | 单 node 只说明**内存**局部性没得优化;**L3 局部性是另一条轴**(EPYC 7402 有 8 个独立 L3) | `taskset` 绑到一个 CCX,见 §3「单 NUMA node 的机器上,要绑的是 CCX」 |

---

## 9. 长期:别再走这条路

以上全部是「**被模板作者选的 torch 版本牵着走**」的产物。自建镜像后 torch 版本由你 pin,整类问题消失:

- `Dockerfile` + `.github/workflows/build-devbox.yml` → GitHub Actions build,推 GHCR
- 在 GH runner 上 build(挨着 registry、直连 PyPI/GitHub Releases),不花你的 wall clock
- Vast 创建实例时把 `ghcr.io/<user>/qwen-devbox:latest` 填进 image 字段,本文档第 2 节整节作废
- 国内 fallback:在国内 ECS 上 pull GHCR 再 push 到阿里云 ACR

本 runbook 是过渡方案和应急手段,不是终局。
