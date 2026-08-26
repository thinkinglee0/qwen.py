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

| 字段          | 值                             | 理由                                |
| ----------- | ----------------------------- | --------------------------------- |
| GPU         | RTX 4090                      | 24 GB 保留 KV 压力;450 W 无 power wall |
| 磁盘          | **150 GB**                    | 默认 16 GB 装不下 15.2 GB 权重           |
| 计费          | **on-demand**,非 interruptible | benchmark 中途被抢占 = 数据作废            |
| 区域          | 香港 / 日本 / 韩国                  | 北京 RTT 40–60 ms                   |
| Verified    | 必须                            | 排除家用宽带和民宅电力                       |
| Reliability | > 99%                         | 3 小时 sweep 中途掉线要重跑                |
| PCIe 实测     | ≥ 20 GB/s                     | 排除 x1/x4 矿机 riser                 |
| 宿主 RAM      | ≥ 48 GB                       | vLLM + tokenizer + client         |
| vCPU        | ≥ 8                           | 单核弱会拖慢 Python decode loop         |
| 模板          | 带 PyTorch 的镜像                 | 省一次 torch 安装                      |

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

# --- CPU. Single-thread perf drives the Python decode loop.
nproc && lscpu | grep -E "Model name|CPU max MHz"
cat /sys/fs/cgroup/cpuset.cpus.effective     # what you can actually use

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

> 注意 7B 和 0.5B 处于不同 regime:0.5B 只有 ~1.0 GB 权重,GPU 侧约 1.0 ms < CPU 侧 4.3 ms,**完全 CPU-bound**。用 0.5B 跑性能对比,测到的是 Python 循环 vs CUDA graph,不是 scheduler 设计。

---

## 2. 装依赖(5 分钟)

```bash
cd /workspace
git clone <your qwen.py repo> && cd qwen.py
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

pip install --no-cache-dir torch==2.9.* --index-url https://download.pytorch.org/whl/cu128
pip uninstall -y torchvision torchaudio    # they hard-pin the old torch and break
```

### 2b. 一键装完

```bash
bash setup.sh
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

写进 `~/.bashrc`,别每次手敲。宿主报 128 threads,放任 torch 开满会自己制造 CPU 瓶颈,而且这个开销**只打在 qwen.py 身上**(vLLM 自己会设)。

```bash
cat >> ~/.bashrc <<'EOF'
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export QWEN_MODEL_DIR=/workspace/models
export HF_HOME=/workspace/hf

EOF
source ~/.bashrc
```

---

## 4. 下权重(5–10 分钟)

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
from huggingface_hub import snapshot_download

root = os.environ["QWEN_MODEL_DIR"]
KEEP = ["*.json", "*.safetensors", "*.txt", "*.model", "*.py"]

for repo, sub in [
    # 0.5B for pytest: the session-scoped fixture loads weights on every
    # run, and 7B makes that unbearable.
    ("Qwen/Qwen2.5-0.5B-Instruct", "qwen2.5-0.5b-instruct"),
    # 7B only when benchmarking -- comment out otherwise.
    ("Qwen/Qwen2.5-7B-Instruct",   "qwen2.5-7b-instruct"),
]:
    p = snapshot_download(repo, local_dir=f"{root}/{sub}", allow_patterns=KEEP)
    print("->", p)
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

### GPU 上第一次跑要预期失败

Mac 上的 tolerance 是按 **CPU / fp32 / SDPA-math** 标定的,GPU 上三个变量同时变了:device、dtype(bf16 只有 8 位尾数)、attention kernel(flash 的 tiling 彻底改写累加顺序)。误差量级会高几个数量级。

重新标定的顺序:

1. 固定 GPU + bf16,先测 **flash vs SDPA** 两条 backend 路径的差异 —— 纯 kernel 差异,无实现差异,得到"同一数学表达式在两种 tiling 下的分歧上界"
2. 用这个上界去校准 qwen.py vs HF reference 的 tolerance
3. HF 侧显式指定 `attn_implementation`,否则你在比较四个变量而不是一个

---

## 6. benchmark 专用注意事项

**vLLM 用官方镜像,不要装进这个环境。** 它自己的 torch pin 和 flash-attn 版本与你不同,混在一起必然 ABI 冲突。Vast 上无 docker,所以 vLLM 走独立 venv:

```bash
python -m venv /opt/vllm-venv && /opt/vllm-venv/bin/pip install vllm
/opt/vllm-venv/bin/vllm serve "$QWEN_MODEL_DIR/qwen2.5-7b-instruct" \
  --dtype bfloat16 --no-enable-prefix-caching --enforce-eager --port 8001
```

**双臂对照**,单跑一个比值没有说服力:

| 臂   | vLLM 配置           | 差距归因                                       |
| --- | ----------------- | ------------------------------------------ |
| A   | `--enforce-eager` | 你的 scheduler + KV 管理 + attention kernel 选择 |
| B   | 默认(CUDA graph)    | A→B 的增量 = CUDA graph 消除的 CPU launch 开销     |

必须对齐:`--max-num-seqs`、`--max-model-len`、`--block-size`(对上你的 `BlockPool`)、`--no-enable-prefix-caching`(你还没实现 prefix caching,不关会白送 vLLM 一大截命中率)、`--gpu-memory-utilization` 折算成相同的**绝对 KV 字节数**。

**采样**,每个 concurrency 点并行采集:

```bash
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,power.limit,\
temperature.gpu,utilization.gpu,\
clocks_throttle_reasons.sw_power_cap,clocks_throttle_reasons.hw_thermal_slowdown \
  --format=csv,noheader -lms 200 > sweep_c${C}.csv
```

**纪律**:

- 每点跑 ≥ 3 次,**报 median 不报 mean**(离群点会拽高 mean)
- qwen.py 和两条 vLLM 基线**必须在同一次租用、同一台 host、同一个连续时间窗口内跑完**。宿主 CPU 换一台就全部作废
- 每轮记 `uptime`,事后判断某个点是否被邻居污染

**`nvidia-smi -pl` 在 Vast 容器里不可用** —— 报 `Insufficient Permissions`,容器缺 `CAP_SYS_ADMIN`,容器内 root 不等于 capability。`-lgc` 锁频走同一条权限路径,同样不行。要做 power sweep 得换 AWS g5(VM + 直通,guest 有 root)或裸金属。

---

## 7. 销毁前带走什么

overlay 可写层随 destroy 消失。**离开前确认:**

- [ ] 代码 `git push`
- [ ] benchmark 结果 CSV / JSON 上传(`git`、对象存储、或 `scp` 到本地)
- [ ] `/opt/constraints.txt` 存一份 —— 下次重建时 diff 一下就知道模板变了没有
- [ ] 记下 instance ID、host ID、`nvidia-smi -q` 快照、`lscpu` 输出、launch overhead 中位数 —— **这些是判断下次的数据能否和这次拼接的唯一依据**

权重不用备份,重下比传快。

---

## 8. 故障速查

| 症状                                         | 根因                                      | 解                                     |
| ------------------------------------------ | --------------------------------------- | ------------------------------------- |
| `No space left on device` 在下权重时            | overlay 只有 16 G(默认)                     | destroy 重租,磁盘设 150 G                  |
| `df` 显示 1.9T 可用却写不进去                       | 那是宿主 fs 的 bind mount,不是你的               | 只看 `overlay` 那一行                      |
| flash-attn `NO WHEEL for torchX.Y`         | 模板 torch 太新                             | 降 torch,别升 flash-attn                 |
| `pip install flash-attn` 卡住/OOM            | 触发源码编译                                  | 只装 GitHub Releases 的 wheel            |
| `undefined symbol: _ZN3c10...`             | torch ABI 与 wheel 不匹配                   | 四坐标重新对齐;别用 NGC 镜像                     |
| `OSError: No such file .../conda/.../work` | conda 环境 `pip freeze` 产生 `@ file://` 引用 | constraints 过滤成纯 `name==version`      |
| `huggingface-hub` 解析冲突                     | transformers 4.46.3 要求 `<1.0`           | 从 constraints 剔除 hf-hub               |
| `datasets` 拖回 hub>=1.0                     | datasets 4.x+ 与 oracle 冲突               | 不装 datasets,ShareGPT 直接 `json.load()` |
| `nvidia-smi -pl` Insufficient Permissions  | 容器无 `CAP_SYS_ADMIN`                     | 换 VM/裸金属,或放弃 power sweep              |
| `no kernel image is available`             | wheel arch list 无 sm_89                 | 换 wheel 源                             |

---

## 9. 长期:别再走这条路

以上全部是「**被模板作者选的 torch 版本牵着走**」的产物。自建镜像后 torch 版本由你 pin,整类问题消失:

- `Dockerfile` + `.github/workflows/build-devbox.yml` → GitHub Actions build,推 GHCR
- 在 GH runner 上 build(挨着 registry、直连 PyPI/GitHub Releases),不花你的 wall clock
- Vast 创建实例时把 `ghcr.io/<user>/qwen-devbox:latest` 填进 image 字段,本文档第 2 节整节作废
- 国内 fallback:在国内 ECS 上 pull GHCR 再 push 到阿里云 ACR

本 runbook 是过渡方案和应急手段,不是终局。
