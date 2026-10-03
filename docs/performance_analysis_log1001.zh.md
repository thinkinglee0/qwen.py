# 性能分析 —— RTX 4090 上 fused top-k + top-p 对 `async_scheduling`（log1001）

> 英文原版：[`performance_analysis_log1001.md`](./performance_analysis_log1001.md)

**这份报告兑现的是 [`performance_analysis_log928.zh.md`](./performance_analysis_log928.zh.md) §7.1 的那条建议。**
那份报告最后给出的第一优先项是：采样器每步花 **15.4 ms 去 sort 全部 151 936 列词表**，只为给 top-k
早已筛剩的 **20** 个候选排序；把它收进 top-k 窗口内，batch 512 上应能省约 18 ms/步，并且顺手修掉
batch 1024 的 OOM。本报告测的就是干了这件事的那个分支。

**数据来源**

| 内容 | 路径 |
| --- | --- |
| 并发 sweep，`fused_top_kp`，第 1–2 轮 | `log_vast/log1001/benchmark_fused_top_kp{,2}/` |
| 并发 sweep，`async_scheduling`，第 1–2 轮 | `log_vast/log1001/benchmark_baseline{,2}/` |
| 纯 decode 空闲率 profile，`fused_top_kp`，第 1–2 轮 | `log_vast/log1001/profile_fused_top_kp{,2}/` |
| 纯 decode 空闲率 profile，`async_scheduling`，第 1–2 轮 | `log_vast/log1001/profile_baseline{,2}/` |
| 跨分支 step 指标，batch 1–512 | `log_vast/log1001/baseline_against_fused_top_kp/mean_step_metrics.*.log` |
| run 间噪声地板，`async_scheduling` | `log_vast/log1001/baseline_against_baseline2/mean_step_metrics.*.log` |
| run 间噪声地板，`fused_top_kp` | `log_vast/log1001/fused_top_kp_against_fused_top_kp2/mean_step_metrics.*.log` |
| GPU 静态清单 / 宿主 | `log_vast/log1001/gpu.static.csv`、`host_info` |

**被测代码** —— 基线 `async_scheduling` 位于 `bedc00f`；候选 `fused_top_kp` 位于 `49bb10d`。
候选**只比基线多一个提交**（`git rev-list --count async_scheduling..fused_top_kp` = 1），这个提交只碰了
三个文件：`src/qwen/sampling.py`（+51/−19）、`src/qwen/config.py`（一行）、`tests/test_sampling.py`
（+227）。这是这个仓库能做到的最干净的 A/B —— scheduler 没动、engine 没动、采样器之外的 kernel 没动。

两边引擎开关一致：`compile_rope = false`、`pre_gather_cos_sin = true`、`use_d_first_schedule = false`、
`max_num_batched_tokens = 8192`。

**运行顺序**（同一台 vast 实例，约 45 分钟，中途未重启）：`fused_top_kp` profile 23:40 → 其第 2 轮
23:41 → `fused_top_kp` sweep 23:44–23:53 → 其第 2 轮 23:53–00:02 → `async_scheduling` profile 00:04 →
其第 2 轮 00:06 → `async_scheduling` sweep 00:08–00:16 → 其第 2 轮 00:17–00:25。**候选跑在前面**，
机器更冷，所以它没有占到预热的便宜。

---

## 摘要

1. **预测的收益到账了，还超了一点。峰值吞吐 7 245 → 10 976 tok/s（+51.5 %）**，batch 256
   **7 166 → 10 704（+49.4 %）**，batch 128 **6 816 → 8 707（+27.7 %）**。batch 512 上 TPOT
   **68.1 → 44.4 ms（−34.8 %）**。
2. **全部收益来自一个算子消失。** batch 512 上 `aten::sort` 是 52.1 ms 设备步里的 **15.43 ms/步** ——
   占全部设备时间的 29.6 %。现在它**在 trace 里彻底不存在**。连带它拖着的那些流量
   （`masked_fill_`、`scatter_`、`cumsum`、一次全词表 `_softmax`、那些 DtoD 拷贝）一起，每步设备时间
   **52.1 → 27.4 ms（−47.3 %）**。
3. **batch 1024 的 OOM 修好了。** 基线依旧死在 `apply_top_p` 的 `torch.sort` 上，索要 **1.73 GiB**
   而只剩 1.72 GiB 空闲（`benchmark_baseline/pytest.log:174`）。`fused_top_kp` 在 batch 1024 跑完，
   **9 630 tok/s** —— 能跑了，但已经过了吞吐峰值（§5.4）。
4. **batch 128 以下什么都没动，而这正是正确结果。** batch ≤ 64 时引擎是主机瓶颈（GPU 空闲 50–67 %），
   所以砍掉设备工作只换来 **+0.1 % … +3.4 %** 的吞吐。省下来的设备时间是真的 —— batch 64 上设备 busy
   **7.91 → 5.40 ms/步（−31.7 %）** —— 只是无处可用。
5. **GPU 瓶颈的拐点从 batch 64 移到了 batch 128。** sweep 里"最后一次划算的翻倍"从 64 变成 128，
   batch 128 的 scaling efficiency **0.52 → 0.67**。引擎的默认 `max_num_seqs = 128` 现在正好**落在**
   拐点上，而不是在拐点之后一个翻倍。
6. **prefill 延迟也变好了，顺手还掉了 log928 那笔 TTFT 债的一部分。** batch 512
   **216.8 → 172.8 ms（−20.3 %）**，batch 256 **−13.6 %**，batch 128 **−7.2 %**。对照 log928 里的
   `main`（batch 512 上 121.6 ms），异步调度那笔 TTFT 代价现在**白还掉了 46 %**。
7. **采样依然是最大的一项：batch 512 上 27.3 ms/步里的 16.35 ms（60 %）。** 剩下的每一毫秒仍然是
   `O(bsz × 151 936)` 的全词表 pass —— temperature、penalties、`-inf` 的 scatter 回写、fp32 softmax、
   死行保护、`multinomial`、`argmax`。问题只有 20 列宽，却还在做约 23 遍全词表扫描。§7.1–7.2 给出把它
   压到约 2.5 ms/步的路子。
8. **复现性极好。** run 间吞吐在基线上一致到 **≤ 1.14 %**，在候选上 **≤ 1.99 %**（中位数 0.45 % /
   0.51 %）；候选两轮之间 batch 512 的 `sample_gpu` 一致到 **0.00 %**。而且这里的基线在**另一台**实例上
   把 log928 的 `async_scheduling` 复现到了 **0.2 %** 以内（峰值 7 245 对 7 232 tok/s、TPOT 68.11 对
   68.23 ms、设备 busy 51 515 对 51 526 µs/步）。28–52 % 的收益是噪声的约 25 倍。

---

## 1. 被测系统

### 1.1 硬件

| | |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4090，24 564 MiB，驱动 550.127.08 |
| 频率 / 功耗 | SM 3 135 MHz，显存 10 501 MHz，450 W enforced（上限 600 W） |
| PCIe | **gen 4 × 16**（current = max） |
| CPU | AMD Ryzen 5 7500F，6 核 / 12 线程，63.9 GB 内存 |
| 实例 | vast 53604495，host 132677 |
| sweep 工具假设的 roofline | 1 008 GB/s、165 TFLOP/s |

和 log928 的实例不同，这台是满宽度的 PCIe gen 4，所以那份报告里的主机链路警告在这里不适用。但这也意味着
**跨 log 的绝对值比较依然不可信**（§8）—— 不过见摘要第 8 条：基线落在 log928 `async_scheduling` 的
0.2 % 以内，这个一致性检查已经比本报告需要的更强。

### 1.2 模型与负载

Qwen2.5-0.5B-Instruct，bf16，`vocab_size = 151 936`，24 层，GQA 14/2，`head_dim = 64`。
采样配置来自 `generation_config.json`：**`temperature = 0.7`、`top_k = 20`、`top_p = 0.8`、
`repetition_penalty = 1.1`** —— 两个分支完全一致（两边 `pytest.log` 都已核对），所以这是同条件对比。
这一点很重要：§5.3 解释了为什么 diff 里那个 `config.py` 默认值的改动在这里是不起作用的。

* **sweep**：10 × batch 个请求，每个 512 输入 token / 128 输出 token，`max_model_len = 1024`，每请求
  一个 prefill chunk（全程 `prefill_chunk.mean = 1.0`）。batch 1 → 1024。
* **profile**：纯 decode，每个 batch 20 步测量 + 5 步 trace，`num_blocks = 4096`、`block_size = 256`。
  run 间 KV drift 4.34 %，两分支相同。
* **全部 42 次 benchmark run 里零抢占、零 cache 耗尽、零重调度**。同一 batch 下两分支的步数完全相同
  （例如 batch 512 都是 1 324 步），说明两边真的干了同样的活。

### 1.3 异常步计数

`test_mean_step_metrics` 在 batch 1/8/32/512 上标出 **0/20** 异常步，在 batch 64/128/256 上 **1/20** ——
两个分支同一个模式。没有排除任何数据。

---

## 2. 方法

### 2.1 两把尺子

* **sweep**（`test_benchmark.py`）—— 端到端、无 profiler：吞吐、TPOT、prefill 延迟、排队、ITL 分位。
  这是用户真正感知的数字。
* **profile**（`test_profile.py::test_profile_decode_idle_fraction`）—— 稳态纯 decode，对 5 步开
  `torch.profiler`，另外用 20 步做一次干净的 wall 测量。产出 `gpu_busy_from_trace`、
  `gpu_idle_fraction` 和逐算子的 `key_averages` 表。profiler 对 wall 的开销是 +69 %，所以空闲率是对
  `wall_clean` 算的，不是对 trace wall。

`test_mean_step_metrics` 再把两次 run 的 step 指标逐字段做 Welch z 检验（20 个 decode 步），在 2σ 处
标 `noise` 或 `SHIFT`。

### 2.2 怎么读 `*_gpu` 字段 —— 这批 log 里唯一的坑

`fwd_gpu`、`logits_gpu`、`sample_gpu` 是**流上的 CUDA event 区间，不是 kernel 累加**。它们把整步铺满，
所以加起来等于 **wall**，不等于 busy：

| | batch 64 | batch 512 |
| --- | --- | --- |
| 基线 `fwd_gpu` + `logits_gpu` + `sample_gpu` | 6.57 + 0.36 + 3.90 = **10.82** | 10.66 + 0.89 + 40.56 = **52.11** |
| 基线 `wall_clean` / `gpu_busy` | 10.92 / **7.91** | 52.23 / **51.52** |

batch 512 上设备是饱和的，所以区间 ≈ kernel 时间。batch 64 上设备有 27.6 % 的时间空着，而**那些气泡就
在区间里面**。这就是为什么 `baseline_against_fused_top_kp/mean_step_metrics.64.log` 会报候选的
`fwd_gpu` **+39.1 %**、`rope_gpu` **+42.9 %**，而设备总 busy 却*降了 31.7 %*：候选的步更"饿"，更多气泡
落进了 forward 的 event 区间里。**没有任何 kernel 变慢。** 交叉验证：batch 512 上 `aten::mm` 的
self CUDA 是 3.608 对 3.614 ms/步，`flash_attn` 是 4.888 对 4.886 —— 模型部分是逐字节相同的工作量。

同一个效应也解释了 batch 512 上主机侧 `rope` 字段从 **21.15 塌到 3.39 ms**，而 `rope_gpu` 稳在
0.81 ms。`rope` 不是在做 CPU 工作；当主机跑在一个饱和设备前面一步时，背压就落在那里。设备步变短 →
停顿变短。高 batch 下主机侧的 `fwd`/`rope` 要当作**停顿记账**读，不是成本。

### 2.3 噪声地板

同分支、同实例、run 间：

| | 基线（r1 对 r2） | 候选（r1 对 r2） |
| --- | --- | --- |
| 吞吐，最差 batch | 1.14 %（batch 1） | 1.99 %（batch 32） |
| 吞吐，跨 batch 中位数 | 0.45 % | 0.51 % |
| batch 512 吞吐 | 0.01 % | 0.04 % |
| batch 512 `gpu_busy` | 51 515 对 51 504 µs（0.02 %） | 27 305 对 27 307 µs（0.01 %） |
| batch 512 `sample_gpu` | −0.01 % | −0.00 % |
| batch 512 超 2σ 字段 | 1/20（`logits_gpu`） | 1/20（`ci`，0.019 ms 的 +4.9 %） |

batch 256 上基线两轮在 **0/20** 个字段上有差异。设备侧的尺子基本是精确的；端到端的尺子好到约 1 %，
在 batch ≤ 32 时变差 —— 那里一次调度打嗝就是 15 秒 run 里可测的一部分。

---

## 3. 结果

### 3.1 并发 sweep —— 各取两轮均值

| batch | 基线 tok/s | fused tok/s | Δ | 基线 TPOT ms | fused TPOT ms | Δ | 基线 prefill ms | fused prefill ms | Δ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 102.4 | 102.0 | −0.4 % | 9.75 | 9.79 | +0.4 % | 10.1 | 10.2 | +0.7 % |
| 2 | 187.6 | 190.5 | +1.5 % | 10.64 | 10.48 | −1.5 % | 11.5 | 11.3 | −2.3 % |
| 4 | 370.2 | 377.9 | +2.1 % | 10.71 | 10.49 | −2.0 % | 18.5 | 18.1 | −2.3 % |
| 8 | 723.6 | 741.6 | +2.5 % | 10.83 | 10.56 | −2.5 % | 32.3 | 32.3 | −0.1 % |
| 16 | 1 384 | 1 432 | +3.4 % | 11.06 | 10.67 | −3.5 % | 60.6 | 60.2 | −0.6 % |
| 32 | 2 717 | 2 720 | +0.1 % | 11.12 | 11.11 | −0.1 % | 80.5 | 79.7 | −0.9 % |
| 64 | 5 057 | 5 102 | +0.9 % | 11.82 | 11.74 | −0.7 % | 96.4 | 93.1 | −3.3 % |
| 128 | 6 816 | **8 707** | **+27.7 %** | 17.70 | **13.67** | **−22.8 %** | 118.1 | 109.6 | −7.2 % |
| 256 | 7 166 | **10 704** | **+49.4 %** | 34.21 | **22.52** | **−34.2 %** | 154.0 | 133.0 | −13.6 % |
| 512 | 7 245 | **10 976** | **+51.5 %** | 68.11 | **44.44** | **−34.8 %** | 216.8 | 172.8 | **−20.3 %** |
| 1024 | **OOM** | 9 630 | — | — | 101.57 | — | — | 271.5 | — |

值得一提的次级效应：

* **scaling efficiency**（相对 batch 1 的 tok/s/req）：batch 128 **0.52 → 0.67**；batch 256
  **0.27 → 0.41**；batch 512 **0.14 → 0.21**。
* **最后一次划算的翻倍**（gain/cost > 1）从 **64 移到 128**。batch 128 的 gain/cost **0.90 → 1.47**。
* **ITL p99**：batch 512 **120.7 → 97.4 ms**；batch 128 **70.5 → 66.0 ms**。
* **batch 512 的 MFU**：4.3 % → **6.6 %**。依然和 roofline 无关 —— 这个模型在这个 batch 下受限于内存和
  采样器，不是受限于 FLOP。

### 3.2 等吞吐对比

|  | `async_scheduling` | `fused_top_kp` |
| --- | --- | --- |
| 达到约 7 200 tok/s 需要的并发 | 512 | 约 100（在 64 与 128 之间） |
| batch 128 上的吞吐 | 6 816 | **8 707（比基线*峰值*还高 20 %）** |
| 对应 TPOT | 68.11 ms（在 batch 512） | **13.67 ms（低 5.0 倍）** |
| 对应 ITL p99 | 120.8 ms | **66.0 ms** |
| 对应 prefill 延迟 | 216.8 ms | **109.6 ms** |

基线需要 512 路并发、68 ms 的 TPOT 才能摸到自己的峰值 7 245 tok/s。候选在 batch 128 就超过那个峰值
20 %，而**单 token 延迟只有五分之一、prefill 延迟只有一半**。和 log928 那次异步调度的取舍不同，这一次
没有需要辩解的延迟代价 —— 在所有 batch ≥ 128 上，它在每一个维度都更好。

### 3.3 GPU 空闲率与设备 busy —— 纯 decode

| batch | 基线空闲（r1/r2） | fused 空闲（r1/r2） | 基线 busy ms/步 | fused busy ms/步 | Δ busy | 基线 `wall_clean` | fused `wall_clean` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 66.8 / 66.4 % | 67.5 / 67.9 % | 3.25 | 3.15 | −3.2 % | 9.79 | 9.67 |
| 8 | 59.1 / 59.0 % | 63.2 / 61.8 % | 4.27 | 3.90 | −8.7 % | 10.46 | 10.61 |
| 32 | 50.8 / 49.6 % | 59.4 / 60.1 % | 5.46 | 4.36 | −20.2 % | 11.10 | 10.73 |
| 64 | 27.6 / 28.0 % | 50.8 / 49.7 % | 7.91 | 5.40 | −31.7 % | 10.92 | 10.96 |
| 128 | 6.6 / 6.7 % | 24.8 / 24.8 % | 14.24 | 8.35 | **−41.4 %** | 15.25 | **11.10** |
| 256 | 3.1 / 3.1 % | 5.6 / 5.6 % | 27.06 | 15.15 | **−44.0 %** | 27.93 | **16.04** |
| 512 | 1.4 / 1.4 % | 2.8 / 2.7 % | 51.52 | 27.31 | **−47.0 %** | 52.23 | **28.09** |

这张表要当 log928 那张的镜像来读。那次设备工作量一致到 1 % 以内，分支删掉的是主机停顿。这次主机行为没
变，分支删掉的是**最多 47 % 的设备工作**。所以空闲率在每个 batch 上都*升高*了 —— batch 64 从 27.6 %
升到 50.8 %，batch 128 从 6.6 % 升到 24.8 %。这不是回退；那是一直都在的主机开销，在设备不再是长板之后
重新露了出来。batch 128 已经从设备瓶颈翻成了主机瓶颈：11.10 ms 的步里只有 8.35 ms 的设备工作。

### 3.4 主机侧逐步拆解，纯 decode（ms）

取自 `baseline_against_fused_top_kp/mean_step_metrics.{128,512}.log`，各 20 个 decode 步。

| 字段 | 基线 @128 | fused @128 | 基线 @512 | fused @512 | 含义 |
| --- | --- | --- | --- | --- | --- |
| `step_0` | 14.327 | **10.613** | 49.212 | **26.306** | 整步的主机 wall |
| `step_1` | 0.234 | 0.230 | 0.454 | 0.427 | in-flight 采样 drain |
| `sched` | 0.131 | 0.132 | 0.513 | 0.502 | scheduler 合计 |
| `bld_meta` | 0.354 | 0.355 | 0.507 | 0.499 | attention 元数据构建 |
| `fwd` | 13.005 | **9.353** | 46.619 | **23.884** | forward 的主机区间（**以停顿为主**） |
| `fwd_gpu` | 5.381 | 7.095 | 10.660 | 10.709 | forward event 区间（见 §2.2） |
| `rope` | 2.481 | 2.088 | 21.150 | **3.394** | rope 的主机区间（**纯背压**） |
| `rope_gpu` | 0.783 | 1.188 | 0.806 | 0.814 | rope event 区间 |
| `logits_gpu` | 0.354 | 0.387 | 0.889 | 0.886 | lm_head event 区间 |
| `sample` | 0.649 | **0.586** | 1.083 | **0.922** | 采样器的主机区间 |
| `sample_gpu` | **9.421** | **3.527** | **40.558** | **16.353** | 采样器 event 区间 |
| `dth` / `dth_wait` | 0.034 / 0.005 | 0.034 / 0.005 | 0.036 / 0.005 | 0.035 / 0.005 | token 的设备→主机拷贝 |

能解释整份报告的那一行是 `sample_gpu`：batch 512 上 **40.56 → 16.35 ms（−59.7 %）**，batch 128 上
**9.42 → 3.53 ms（−62.6 %）**，z 分别是 −5 461 和 −2 660。forward 里别的一切都没碰（`fwd_gpu`
10.660 → 10.709，+0.46 %），主机侧每个字段要么没变，要么是停顿记账的产物（§2.2）。

采样器的单行设备成本，这是看渐进行为最干净的方式：

| batch | 基线 µs/行 | fused µs/行 | 比值 |
| --- | --- | --- | --- |
| 1 | 423.5 | 310.3 | 1.36× |
| 8 | 94.1 | 48.0 | 1.96× |
| 32 | 55.0 | 20.3 | 2.71× |
| 64 | 60.9 | 21.5 | 2.83× |
| 128 | 73.6 | 27.6 | 2.67× |
| 256 | 79.0 | 32.5 | 2.43× |
| 512 | 79.2 | 31.9 | **2.48×** |

基线的单行成本从 batch 32 的 55 µs *涨到* batch 512 的 79 µs —— 那是 `O(vocab log vocab)` 的分段
radix sort 随 batch 变宽后每行效率下降。候选在约 32 µs/行 处走平。两者都仍远高于 20 个候选应有的成本
（§6）。

### 3.5 设备侧算子拆解，batch 512（self CUDA，ms/步，来自 `key_averages` / 5 个 trace 步）

| 算子 | 基线 | fused | Δ |
| --- | --- | --- | --- |
| `decode_step`（CUDA total） | **52.109** | **27.448** | **−24.662** |
| `aten::sort` | **15.430** | **—** | **−15.430** |
| `aten::copy_` | 2.665 | 1.057 | −1.608 |
| `aten::masked_fill_` | 2.179 | 0.672 | −1.507 |
| `Memcpy DtoD` | 1.920 | 0.627 | −1.293 |
| `aten::_softmax` | 2.450 | 1.249 | −1.201 |
| `aten::sub` | 3.078 | 2.082 | −0.996 |
| `aten::scatter_` | 0.924 | 0.046 | −0.877 |
| `aten::cumsum` | 0.827 | 0.006 | −0.821 |
| `aten::where` | 2.240 | 1.531 | −0.709 |
| `Context Sync` | 9.246 | 4.384 | −4.862 |
| `aten::topk` | 1.868 | 1.867 | −0.001 |
| `aten::multinomial`（CUDA total） | 2.738 | 2.738 | −0.001 |
| `aten::div` | 2.504 | 2.504 | ±0.000 |
| `aten::mm` | 3.614 | 3.608 | −0.006 |
| `flash_attn::_flash_attn_varlen_forward` | 4.886 | 4.888 | +0.002 |

`Context Sync` 之上那九行算子加起来是 **−24.4 ms**，正好是 `decode_step` 的全部差值。`aten::sort`
一个就占 **63 %**，支撑它的两行 `cub::DeviceSegmentedRadixSortKernel`（5 步合计 49.6 + 24.3 ms =
14.8 ms/步）随它一起消失。`Context Sync` 腰斩是因为那是主机在等流 —— 它跟随步长，不是工作量。

更小的 batch 上形状相同：`sort` 在 batch 256 上 7.63 → 0 ms/步，batch 128 上 3.75 → 0。

---

## 4. 这个提交实际改了什么

改之前（`sample`，两遍）：

```python
logits = apply_top_k(logits, top_k, max_k)   # topk(max_k) -> 第 k 大作阈值 -> 全词表 mask
logits = apply_top_p(logits, top_p)          # 对全部 151 936 列 torch.sort，cumsum，再 scatter 回去
```

改之后（一遍，在 top-k 窗口内完成）：

```python
top_vals, indices = torch.topk(logits, max_k, dim=-1, sorted=True)   # [bsz, max_k]，降序
top_vals = top_vals.masked_fill(arange(max_k) >= top_k[:, None], -inf) # 窗口内按行各自的 k
probs    = top_vals.softmax(-1)                                      # [bsz, max_k] —— 按行重新归一
remove   = (probs.cumsum(-1) - probs) > top_p[:, None]               # top-p，[bsz, max_k]
out      = torch.full_like(logits, -inf).scatter_(1, indices, top_vals.masked_fill(remove, -inf))
```

有三点让它不只是更快、而且是正确的，新增测试把每一点都钉住了：

1. **`torch.topk` 返回的本来就是降序值** —— 基线的 `sort` 是在重新推导上一行刚递给它的顺序。这就是
   那 15.4 ms。
2. **按行的 k mask 必须在 softmax *之前*施加**（`test_fused_renormalises_probs_within_each_row_k`）。
   概率必须在该行自己的 k 上重新归一，而不是在整个 batch 共用的 `max_k` 板上，否则融合后的 top-p 会比
   顺序的 top-k→top-p 更松。
3. **`max_k` 是候选预算，不只是 `topk` 的宽度** —— 前 `max_k` 列之外一律硬 `-inf`，所以窄的行不会把宽
   的行截断（`test_fused_narrow_row_does_not_cap_wide_row`）。

配套改动：`SamplingParamTable.set_slot` 现在把有效 k 夹到 `[1, top_k_cutoff]`，其中
`top_k_cutoff = min(MAX_EFFECTIVE_TOP_K, vocab_size)` = **1 024**，于是"关闭 top-k"被表示成
`k = 1024` 而不是 `0`。`apply_penalties` 也去掉了
`repetition_penalty[:, None].repeat(1, vocab)` 的物化，改用 `torch.where` —— 这值 §3.5 里 `where`/DtoD
差值中的约 0.7 ms/步，也正是 log928 §7.2 要的东西（部分完成）。

`apply_top_k` 和 `apply_top_p` 还在，但**在热路径上已经是死代码** —— 现在只有测试调它们。

---

## 5. 回退与风险

### 5.1 哪里都没有吞吐回退，但 batch ≤ 16 在噪声内

batch 1 是 −0.4 % 吞吐 / +0.4 % TPOT，落在 1.14 % 的 run 间地板内。batch 2–16 是 +1.5 % … +3.4 %，
也接近地板，但在两对 run 上都一致为正，并且和那里实测的设备节省（busy −3 % 到 −9 %）相符。不需要修。

### 5.2 "关闭 top-k"现在意味着 top-1024 —— 这是真实的分布改变

由于 `vocab_size = 151 936 > MAX_EFFECTIVE_TOP_K = 1 024`，一个请求写 `top_k = 0`、`top_k < 0` 或
`top_k ≥ 1024`，现在都被夹成 **k = 1 024**，而融合路径会把这块板之外的一切硬 `-inf` 掉。旧路径把同样的
请求读作*关闭*，让整个词表进入 top-p。`test_fused_truncates_at_cutoff_where_sequential_keeps_the_tail`
刻意记录了这个分歧，它自己的注释也量化了代价：在尖锐的 logits 上可忽略，在普通 N(0,1) 上
**约 5 % 的概率质量**。

这不影响本报告任何数字（全程 `top_k = 20`），但它是语义改变，而且除了那条测试之外没有任何地方声明。
如果引擎哪天需要诚实的无限制采样，它需要一条单独的全词表路径（或者睁着眼睛调高
`MAX_EFFECTIVE_TOP_K`，因为这个 cutoff 同时也是 `topk` 便宜的原因）。

### 5.3 `config.py` 的默认值改动在这里不起作用 —— 但不是在所有地方

diff 把 `ModelConfig.top_k` 从 `0` 翻成 `20`。走 `from_pretrained` 时这是观察不到的：
`generation_config.json` 提供 `top_k = 20`，`raw.update(raw2)` 覆盖掉 dataclass 默认值（两分支日志都已
核对）。但对任何直接构造 `ModelConfig()` 的路径它*是*可观察的 —— 合成配置、单测、嵌入式用法 —— 它们会
悄悄从"无 top-k"切换到 top-20。意图无害、效果不可见，但值得在提交信息里写一行（它没写）。

### 5.4 batch 1024 能跑了，但已经过峰

| batch | fused tok/s | TPOT ms | gain/cost |
| --- | --- | --- | --- |
| 512 | **10 976** | 44.44 | 0.52 |
| 1024 | 9 630 | 101.57 | 0.38 |

翻倍到 1024 *丢掉* 12 % 吞吐，TPOT 翻一倍多。OOM 修复仍然值得 —— 它消掉了一个硬失败模式，以及那笔给
allocator 施压的 1.73 GiB 分配尖峰 —— 但 **1024 不是一个有用的工作点**，而 sweep 工具那句"saturated
aggregate throughput ~ 9 632 tok/s"是误导的，因为它只是报了最后一行。真正的峰值是
**batch 512 上的 10 976 tok/s**。

### 5.5 prefill 延迟仍随并发增长

batch 512 上 −20 % 是真实改进，但 172.8 ms 仍是 log928 `main`（121.6 ms）的 1.4 倍，batch 1024 上是
271.5 ms。log928 §5.2 指出的异步调度 lookahead 代价是被削弱了，不是被消除了；它的上界是一个设备步，
而设备步现在短了 47 %。

### 5.6 scatter 回写每步仍分配一个全词表张量

`torch.full_like(logits, -inf)` 加上 fp32 的 `logits.float().softmax()`，意味着每步两个
`[bsz, vocab]` 张量 —— batch 512 上 155 MB（bf16）+ 311 MB（fp32），1024 上翻倍。比基线那 1.73 GiB 的
sort 工作区好得多，但 §7.1 把两者都去掉。

---

## 6. 还剩什么：采样仍占设备工作的 60 %

batch 512 上，引擎现在每步在采样器里花 **27.3 ms 中的 16.35 ms**，在模型上花 11.6 ms。采样器的问题
性质没变，只是程度变了：它是**在 `[bsz, 151 936]` 张量上的内存流量**，而且几乎全都还在。batch 512 上一遍
fp32 全词表 pass 是读 311 MB + 写 311 MB ≈ 这张卡可达带宽下的 **0.7 ms**，所以 16.35 ms 相当于对一个
只有 20 列宽的问题做了 **约 23 遍全词表扫描**。

把 `sample()` 和 `apply_penalties()` 对着 batch 512 的 trace 读，它们在这里：

| 源码行 | 算子行（ms/步） | 宽度 |
| --- | --- | --- |
| `apply_penalties`：`prompt_mask \| output_mask` | `bitwise_or` 0.281 | 全词表，两个 bool mask |
| `apply_penalties`：`where(mask, rep[:,None], 1.0)` | `where` 1.531 的一部分 | 全词表 |
| `apply_penalties`：`where(logits>0, logits/rep, logits*rep)` | `gt` 0.257 + `div`/`mul`/`where` 的一部分 | 全词表，4 遍 |
| `sample`：`logits / t[:, None]` | `div` 2.504 的一部分 | 全词表 |
| `apply_fused_top_k_and_p`：`topk` | `topk` 1.867 | 读全词表（**不可避免**） |
| `apply_fused_top_k_and_p`：窗口内 softmax/cumsum/mask | `cumsum` 0.006、`_softmax` 的一部分 | `[bsz, 20]`（**已经很便宜**） |
| `apply_fused_top_k_and_p`：`full_like(-inf)` + `scatter_` | `fill_` 0.739 + `scatter_` 0.046 + `Memcpy DtoD` 0.627 | 全词表 |
| `sample`：`logits.float().softmax(-1)` | `copy_` 约 0.5 + `_softmax` 约 1.2 | 全词表，2 遍 |
| `sample`：`probs.sum(-1)` 死行保护 | `sum` 0.695 | 全词表 |
| `sample`：`probs.masked_fill(dead, 1.0)` | `masked_fill_` 0.672 | 全词表 |
| `sample`：`multinomial(probs, 1)` | CUDA total 2.738（`div`、`exponential_` 0.307、`max`/`min`/`searchsorted`） | 全词表 |
| `sample`：`logits.argmax(-1)` 贪心路径 | `argmax` 0.714 | 全词表 |

这里面只有**一个**真正需要全词表：读 logits 的那个 `topk`。它之后的每一步，操作的张量里那 151 916 个
非候选列全是 `-inf` 或全是 0。

---

## 7. 建议，按优先级

### 7.1 把采样整个搬进 top-k 窗口 —— batch 512 上再省约 9 ms/步

log928 §7.1 把这件事提成了两半；这个提交交付了前一半（把 top-p 融进窗口），留下了后一半（把采样器的
尾巴也留在窗口里）。把它做完：

```python
top_vals, indices = torch.topk(logits, max_k, dim=-1, sorted=True)  # 唯一一次全词表读
top_vals = top_vals / t[:, None]                                    # temperature：保序，放在 topk 之后安全
top_vals = top_vals.masked_fill(beyond_k | beyond_p, -inf)          # 和现在一样，[bsz, max_k]
probs    = top_vals.float().softmax(-1)                             # [bsz, 20]
probs    = probs.masked_fill(~isfinite(probs.sum(-1, keepdim=True)) | (sum <= 0), 1.0)
local    = torch.multinomial(probs, 1)                              # [bsz, 1]，窗口内下标
sampled  = indices.gather(1, local).squeeze(1)                      # 映射回词表 id
greedy_tok = indices[:, 0]                                          # topk 已排序 -> argmax 免费
return torch.where(greedy, greedy_tok, sampled)
```

这会删掉 `full_like` + `scatter_` + DtoD（1.41）、全词表的 fp32 cast 与 softmax（约 1.7）、死行的
`sum`/`masked_fill`（1.37）、`multinomial` 的全词表工作（2.74）、`argmax`（0.71）以及 temperature 的
`div`（约 0.8）—— **直接约 8.7 ms/步**，再加上随它们一起的 `copy_`/`mul`/`sub` 流量。两点说明：

* **temperature 放在 `topk` 之后是安全的**：除以一个正标量保序，top-k 集合完全相同。贪心行
  （`t ≤ EPS`）取 `indices[:, 0]`，根本不经过这次除法。
* **`argmax` 变免费**：`topk(..., sorted=True)` 已经把最大值放在第 0 列。

### 7.2 让 penalties 变稀疏 —— batch 512 上再省约 3.5 ms/步

penalties **必须**在 `topk` 之前跑（它们改变排序），所以搬不进窗口。但它是一次*稀疏*更新：只有该行真正
见过的 token 受影响，最多 `prompt_len + output_len ≤ 1 024` 个，占 151 936 列的 0.7 %。与其做六遍全词表
pass（`bitwise_or`、`where`、`gt`、`div`、`mul`、`where`），不如在"见过"的位置上 gather logits、在那里
施加惩罚、再 scatter 回去：batch 512 上是 `O(bsz × n_seen)` ≈ 0.3 M 个元素，而不是 78 M。
`SamplingTensors` 已经带着 `prompt_mask`/`output_mask`；它缺的只是构造这些 mask 时用的下标表，而
scheduler 本来就有。

另外保留那个现成的便宜收益：当 `rep == 1.0 and freq_pen == 0 and pres_pen == 0` 时整遍 pass 就是恒等
变换，而这从常驻表在主机侧就能判断，不需要同步。

7.1 + 7.2 合起来应能把 batch 512 的采样器压到 **约 4 ms/步**（`topk` 那 1.87 ms 的读加上窗口内算术），
即设备步 **约 15 ms**，相对今天的 27.3 ms 又是约 1.8 倍。吞吐不会一比一跟着翻倍，因为到那时主机会重新成为瓶颈
（§7.3）。

### 7.3 然后是主机，因为它就是下一个

候选在 batch ≤ 128 上已经是主机瓶颈（128 上空闲 24.8 %、64 上 50.8 %），而 7.1+7.2 会把这条边界推到
约 256。batch 64 上无 profiler 的 wall 是 10.96 ms，对应 5.40 ms 的设备工作：**每步 5.5 ms 的主机时间
没有任何东西能藏住它**。step 指标里属于真实主机工作的字段加起来远不到这个数（`sched` 0.076 +
`bld_meta` 0.339 + `sample` 0.598 + `dth` 0.034 ≈ 1.0 ms），所以约 4.5 ms/步没有归属 —— 而 trace 说出了
它是什么：**batch 64 上每步 4 414 次 `aten::` 调用**，self CPU 合计 **8.18 ms/步**（profiler 下），按实测
+69 % 的 profiler 开销折算约为无 profiler 的 4.8 ms。几乎正好填上那个缺口。这是 Python/dispatch 成本，
不是算法工作，而且调用次数几乎不随 batch 增长（batch 512 上 6 158 次），所以它是一笔固定的每步税：
该上 CUDA graph 或更瘦的 eager 路径，而不是去微调单个算子。

### 7.4 以 `max_num_seqs = 128` 作为主配置重跑 sweep

拐点移到了 128，而那正是引擎的默认值。sweep 的 `gain/cost` 标记和"saturated throughput"那行现在应该在
512 上读（§5.4），1024 那行只留作 OOM 回归守卫。

### 7.5 把 cutoff 语义写进文档

见 §5.2。在 `constants.py` 的 `MAX_EFFECTIVE_TOP_K` 旁边加一句、采样器 docstring 里加一句：当
`vocab_size > 1024` 时，"关闭 top-k"是以 top-1024 实现的，这是截断，不是空操作。

### 7.6 删掉或标注死路径

`apply_top_k` 和 `apply_top_p` 现在只有测试用。把它们留作差分测试的参考实现是说得通的，但要在注释里写
明，并且去掉 `apply_top_k` 里那个未使用的 `indices` 绑定。

---

## 8. 注意事项

1. **跨 log 比较依然不可信。** log1001 和 log928、log924 是不同的 vast 实例（这里是 PCIe gen 4 × 16，
   CPU 也不同）。可信的是 **log1001 内部基线到候选的差值** —— 同一台机器上 45 分钟内背靠背测的。这次
   基线把 log928 的 `async_scheduling` 复现到 0.2 % 是令人安心的，但不构成跨 log 比较的许可。
2. **候选跑在前面**，机器更冷。如果存在预热偏置，它偏向基线，也就是说实测收益是下界。
3. **profile 是纯 decode 的。** 它刻意排除了 prefill、分块和准入，这也是为什么 batch 512 上 §3.3 的设备
   节省（−47 %）大于 §3.1 的 TPOT 改善（−35 %）：benchmark 的步是 decode 和 prefill chunk 的混合。
4. **`*_gpu` 字段是 event 区间，不是 kernel 累加**（§2.2）。任何把
   `mean_step_metrics.{32,64,128}.log` 里 `fwd_gpu`/`rope_gpu` 的上升读成回退的解读都是错的；要对着
   `gpu_busy_from_trace` 和 `key_averages` 交叉验证。
5. **profiler 对 wall 的开销是 +69 %**，所以空闲率取自单独那次无 profiler 的 20 步测量，而
   `key_averages` 的 ms/步 是 trace 时间（5 步），它会抬高主机侧列，但不抬高设备侧列。
6. **每个配置只有两轮**，足以把噪声界定在约 1 %，但不足以刻画尾部。§3.1 里的 ITL p99 来自各一次 run 的
   约 10 000 个 token 间隔，应当只作参考。
7. **这里没有测采样质量。** 该提交 227 行新测试确立了 `top_k < cutoff` 时的语义等价，并记录了超过
   cutoff 之后刻意的分歧（§5.2）；没有跑端到端的生成质量对比，而对 `top_k = 20` 来说也不需要。
