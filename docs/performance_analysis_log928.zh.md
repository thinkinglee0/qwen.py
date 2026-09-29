# 性能分析 —— RTX 4090 上 `async_scheduling` 对 `main`（log928）

> 英文原版：[`performance_analysis_log928.md`](./performance_analysis_log928.md)

**`async_scheduling` 分支与 `main` 的首次 A/B。**
[`performance_analysis_log924.md`](./performance_analysis_log924.md) 里的一切都是在**另一台** vast
实例上测的 `main` 血统的代码；本报告把两个分支在**同一台**机器上背靠背跑完，所以这里可信的是**分支
之间的差值**，与 log924 的任何横向比较都不可信（§8）。

**数据来源**

| 内容 | 路径 |
| --- | --- |
| 并发 sweep，`async_scheduling`，第 1–2 轮 | `log_vast/log928/benchmark_slot{,2}/` |
| 并发 sweep，`main`，第 1–2 轮 | `log_vast/log928/benchmark_baseline{,2}/` |
| 纯 decode 空闲率 profile，`async_scheduling`，第 1–2 轮 | `log_vast/log928/profile_slot{,2}/` |
| 纯 decode 空闲率 profile，`main`，第 1–2 轮 | `log_vast/log928/profile_baseline{,2}/` |
| run 间噪声地板，`main` | `log_vast/log928/baseline_against_baseline2/mean_step_metrics.*.log` |
| run 间噪声地板，`async_scheduling` | `log_vast/log928/slot_against_slot2/mean_step_metrics.*.log` |
| 跨分支 step 指标，batch 1 | `log_vast/log928/mean_step_metrics.1.log` |
| GPU 静态清单 / 宿主 | `log_vast/log928/gpu.static.csv`、`host_info` |

> 目录名里的 `slot` 是分支改名前的旧名字，**不要改** —— 那是磁盘上真实存在的路径。

batch 8–512 的跨分支 step 指标表**原本不在** log 目录里（只有 batch 1 有），是本报告用同一个工具
（`test_profile.py::test_mean_step_metrics`，`STEP_METRICS_LINES=65:84,65:84`）从同一批 dump 重新生成
的，写到临时副本里，`log_vast/` 未被改动。

**被测代码** —— 基线 `main` 位于 `0603ceb`；候选 `async_scheduling` 位于 `a6c6927`
（21 个文件 +1610 / −825 行，主体是 `scheduler.py` +684、`sampling.py`、`engine.py`、
`attention_metadata.py`）。两边开关一致：`compile_rope = false`、`pre_gather_cos_sin = true`、
`use_d_first_schedule = false`。

`main` 之后也已 fast-forward 到 `a6c6927`，所以**本报告里的 `main` 一律指 `0603ceb`** —— 也就是
`a6c6927` 的父提交，不是它的 tip。

> **`a6c6927` 与产出这些数字的那棵树并非逐字节相同。** 这些 run 取自历史被重写之前的工作树
> （6 个 WIP 提交，tip 为 `f879a01`）。`a6c6927` 与那棵树有三处差异，**全部属于 instrumentation** ——
> kernel、scheduler、sampler 的行为都没变，所以下面每一个计时数字仍然成立：
>
> 1. **`step_1` 不再双计 `step_0`。** `sample_in_flight_step()` 现在用
>    `timed(pre_sch_out.step_metrics, "step_1")` 包住 drain，在同一轮迭代内对同一个对象 start/stop，
>    因此它只量 drain 本身。§2.2 的坑 1 和 §5.5 描述的是 **log928 的这批 dump** —— 你回头去翻的正是
>    它们 —— 而 **§7.4 已经完成**。
> 2. **`sched_ret_gpu` 字段已删除**（从 `SchedulerStepMetrics` 里），删掉的原因恰恰是它在这个分支上
>    恒为 0.000 ms（§3.4）。`a6c6927` 之后的 dump 不会再带这个字段。
> 3. `test_profile.py` 在 `gpu_idle_fraction` 那行日志里加上了 `batch_size`。

**运行顺序**（同一台实例，约 65 分钟，中途未重启）：`async_scheduling` sweep 05:18 → 其第 2 轮 05:28
→ `main` sweep 05:42 → 其第 2 轮 05:58 → profile 06:10–06:18。`async_scheduling` 跑在**前面**，所以
它没有占到"机器已预热"的便宜。

---

## 摘要

1. **异步流水线步生效了，而且这就是全部故事。** 峰值吞吐 **6015 → 7232 tok/s（+20.2 %）**，
   batch 128 上 **4638 → 6791 tok/s（+46.4 %）**，TPOT **27.0 → 17.8 ms（−34 %）**。每步的 GPU
   **kernel** 工作量没有变（**batch 512 上 +0.8 %**）—— 所有收益都来自被删掉的主机侧停顿，而不是更
   便宜的 kernel。
2. **纯 decode 步的 GPU 空闲率崩塌：batch 512 上 27.5 % → 1.5 %**，batch 128 上 42.0 % → 7.1 %，
   batch 64 上 51.9 % → 28.9 %。主机在整步里已经不再有任何阻塞点：`dth` 2.91 → 0.04 ms、主机侧
   `sample` 56.05 → 1.09 ms、`sched_ret_gpu` 0.83 → **0.00** ms。
3. **等吞吐下延迟改善 4.7 倍。** `main` 需要 batch 512 和 83.3 ms 的 TPOT 才能达到 6015 tok/s；
   `async_scheduling` 在 batch 128 就超过了它，TPOT 只有 **17.8 ms** —— 吞吐多 13 %，而单 token 延迟
   只有四分之一。
4. **batch ≤ 16 从持平到略差**（batch 1：吞吐 −1.9 %、TPOT +1.9 %）。`bld_meta` 在 batch 1 上几乎翻倍
   （0.163 → 0.314 ms）—— 常驻表的元数据构建有一笔固定开销，只有在大约 batch 32 以上才回本（§3.4）。
5. **batch 1024 现在 OOM**，而 `main` 在那里还能给出 5861 tok/s。失败的分配是 `apply_top_p` 里的
   `torch.sort` 索要 **1.73 GiB**，而空闲只剩 1.72 GiB（§5.1）。这是容量回退而不是性能回退，但它把
   引擎在 24 GB 卡上的并发上限压到了 512。
6. **prefill 延迟是流水线的定价：batch ≥ 32 时 TTFT +37 % … +78 %**（batch 512 上 121.6 →
   216.6 ms）。单步 lookahead 意味着一个请求的首 token 要晚一轮迭代才被 commit，而在高并发下那一轮
   往往又是一个 prefill chunk（§5.2）。这是异步调度的教科书代价，是一次有意的取舍，但它不是免费的。
7. **采样现在占全部设备工作的 79 %，也是唯一还值得优化的东西。** batch 512 上是
   **51.5 ms/步里的 40.6 ms**。单独一个跨完整 151936 词表的 `aten::sort` 就是
   **15.4 ms/步 —— 整步的 30 %** —— 而它跑在 top-k 已经把存活候选压到 **20** 个**之后**。见 §6 与
   §7.1：把排序收进 top-k 窗口内可以把宽度砍掉约 150 倍，并同时修掉 batch 1024 的 OOM。
8. **复现性。** run 间吞吐在 `main` 上一致到 **1.26 %**，在 `async_scheduling` 上一致到 **3.22 %**
   （最差点是 batch 1；中位数 1.11 %）；设备侧 GPU busy 时间一致到 **0.02 %**。那些 17–46 % 的收益是
   噪声地板的 6 到 40 倍。`async_scheduling` 是噪声更大的那个分支（§5.3）。

---

## 1. 被测系统

### 1.1 硬件 —— 以及它和 log924 的差别

| | log928（本报告） | log924 |
| --- | --- | --- |
| GPU | RTX 4090，24564 MiB | RTX 4090，24564 MiB |
| 驱动 / VBIOS | 550.127.08 / 95.02.3C.80.C8 | 580.142 / 95.02.18.00.51 |
| `enforced.power.limit` | **450 W**（上限 600 W） | **250 W**（上限 450 W） |
| `pcie.link.gen.current` | **1**（最大 4） | **4**（最大 4） |
| 宿主自述 | 82.2 TFLOPS、876.6 GB/s、首尔 | — |

有两处差别是要紧的：

* **这次没有功耗墙。** `clocks_event_reasons.sw_power_cap` 在 `main` 上只有 **0.1 %** 的采样点为
  Active，在 `async_scheduling` 上 **0.3 %**；SM 频率均值稳在 2588 / 2494 MHz，功耗均值 250 / 229 W
  （上限 450 W），温度峰值 61 ℃。log924 关于"大 batch 数字受限于频率"的结论**在这里不适用** ——
  这批数字是干净的、未被限频的。
* **PCIe 链路协商到了 gen 1。** 主机↔设备传输跑在约 4 GB/s 的链路上。而 `async_scheduling` 分支的核心
  收益正是删掉主机↔设备往返，所以一条降级的链路会**替它加分**。被删掉的成本是小传输的**延迟**
  （一次 4 KB 的 `dth`、一次 event sync）而不是带宽，所以这个效应大概是二阶的 —— 但那些
  +20…46 % 的数字在被当作"该分支的通用加速比"引用之前，应该在一台 gen-4 的机器上重测。

### 1.2 模型与负载

Qwen2.5-0.5B-Instruct（0.494 B 参数，bf16），24 层，`hidden=896`，14 个 Q 头 / 2 个 KV 头，
`head_dim=64`，**`vocab_size=151936`**。KV cache 为 4096 块 × 256 token = 12288 MB。

两个测量工具都用固定形状的 ShareGPT benchmark 配置 —— `max_model_len=1024`、
`max_num_batched_tokens=8192`、512 token 的 prompt、`ignore_eos`，因此全部 `batch_size` 个请求会一起
prefill 并齐步 decode。sweep 跑 11 个 batch（1 → 1024），每点投 10 倍 batch 的请求量；profile 跑
7 个 batch，每点 20 个计时步 + 5 个被 profile 的稳态 decode 步。

### 1.3 一处值得知道的口径差异

`main` 跑的是 `rep_pen = 1.0`；`async_scheduling` 跑的是 `repetition_penalty = 1.1`（字段被改名，且
fixture 里的取值不同）。两个分支都不会根据这个值分支 —— `apply_penalties` 对 1.0 和 1.1 发射的是同一
批 kernel —— 所以**本报告里的任何计时都不受影响**，但两个分支生成的文本并不相同，这批 run 也因此
**不能用来做正确性 A/B**。

---

## 2. 方法

### 2.1 两个测量工具

* **sweep**（`test_benchmark_sweep_batch_size`）—— 端到端，prefill 与 decode 交织，不开 profiler。
  这是用户真正感受到的数字。它的 `TTFT` 列是 **prefill** 延迟均值（不含排队）；`TPOT` / `ITL` 是
  逐 token 的。
* **profile**（`test_profile_decode_idle_fraction`）—— 纯 decode，没有准入也没有抢占。报告
  `wall_clean_us/step`（不开 profiler 测的）和 `gpu_busy_from_trace_us/step`（Chrome trace 里
  kernel / memcpy / memset 区间的并集，重叠只计一次）。`gpu_idle_fraction = 1 − busy/wall`。

### 2.2 这批日志里的三个坑

1. **`step_0 + step_1` 不是步时间 —— 它把 `step_0` 算了两次。**（对这批 dump 成立；在 `a6c6927` 上已
   修，见 §5.5。）`step_1` 在第 *N* 轮的指标对象上 start，却在第 *N+1* 轮的
   `sample_in_flight_step()` 里 stop（[`engine.py:38`](../src/qwen/engine.py#L38)、
   [`scheduler.py:706`](../src/qwen/scheduler.py#L706)），所以它跨越的是第 *N* 轮的尾巴**加上**第
   *N+1* 轮的整个 `step_0` —— 也就是一整轮迭代。`step_1` 的中位数与 profiler 独立测出的
   `wall_clean` 几乎完全吻合：

   | batch | `step_0` 中位数 | `step_1` 中位数 | `step_0+step_1` | `wall_clean` |
   | --- | --- | --- | --- | --- |
   | 8 | 10.31 | 10.77 | 21.09 | 10.64 |
   | 32 | 10.69 | 11.17 | 21.86 | 11.06 |
   | 64 | 10.70 | 11.21 | 21.91 | 11.17 |
   | 128 | 14.69 | 15.29 | 29.98 | 15.30 |
   | 512 | 51.70 | 52.66 | 104.36 | 52.33 |

   **把 `step_1` 当作步的墙钟时间。** 副作用是 `step_1` 会继承**下一轮**的任何尖峰，这也正是它成为
   `async_scheduling` 上被异常检测标记最多的字段的原因。
2. **在 `async_scheduling` 上，主机侧的 `fwd` 和 `rope` 是等待时间，不是工作时间。** batch 512 上
   `fwd` 是 46.8 ms 的 CPU 时间，而 `fwd_gpu` 只有 10.7 ms；`rope` 是 21.2 ms 的 CPU 时间，而
   `rope_gpu` 只有 0.82 ms。发射线程阻塞在满掉的 CUDA launch queue 上 —— 这是**GPU 被打满**的特征，
   而那正是目标。把它们读成回退（`fwd` +377 %、`rope` +868 %）会把意思读反。
3. **`*_gpu` 字段是设备时间轴上的 *elapsed*，不是 kernel busy。** 它们来自夹住某个区域的一对 CUDA
   event，因此包含该区域内部的设备气泡。batch 1 上 `fwd_gpu` 读作 8.5 ms，而 trace 说整步只让 kernel
   忙了 3.25 ms。**只有 `gpu_busy_from_trace` 是占用率。**

### 2.3 噪声地板

| 量 | `main` | `async_scheduling` |
| --- | --- | --- |
| sweep 吞吐，run 间 | ≤ 1.26 %（中位 0.69 %） | ≤ 3.22 %（中位 1.11 %） |
| `gpu_busy_from_trace`/步，run 间 | ≤ 0.03 % | ≤ 0.02 % |
| 超出 2σ 的 step 指标字段 | 每 batch 0–8 / 18 个，除 batch 512 的 `step`/`sample`（−2.7 / −2.8 %）外 \|Δ\| 全部 < 3 % | 每 batch 1–8 / 20 个，\|Δ\| 全部 < 11 % |

每 20 步窗口内的异常步数：**`main` 在每个 batch 上都是 0/20**；**`async_scheduling` 是 1–4/20**。这个
分支可测量地更不稳（§5.3）。

---

## 3. 结果

### 3.1 并发 sweep —— 各取两轮均值

| batch | `main` tok/s | `async_scheduling` tok/s | Δ tok/s | `main` TPOT ms | 分支 TPOT ms | Δ TPOT | `main` TTFT ms | 分支 TTFT ms | Δ TTFT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 102 | 100 | **−1.9 %** | 9.83 | 10.01 | **+1.9 %** | 10.2 | 10.4 | +2.5 % |
| 2 | 187 | 186 | −0.2 % | 10.71 | 10.71 | +0.0 % | 11.6 | 11.6 | +0.1 % |
| 4 | 369 | 368 | −0.3 % | 10.78 | 10.79 | +0.1 % | 18.8 | 18.6 | −1.0 % |
| 8 | 717 | 722 | +0.7 % | 10.99 | 10.85 | −1.2 % | 33.0 | 32.5 | −1.5 % |
| 16 | 1345 | 1366 | +1.6 % | 11.49 | 11.21 | −2.5 % | 62.1 | 60.8 | −2.1 % |
| 32 | 2275 | 2668 | **+17.3 %** | 13.67 | 11.33 | **−17.1 %** | 58.6 | 80.7 | **+37.6 %** |
| 64 | 3557 | 4919 | **+38.3 %** | 17.53 | 12.17 | **−30.6 %** | 63.0 | 96.7 | **+53.5 %** |
| 128 | 4638 | 6791 | **+46.4 %** | 27.00 | 17.77 | **−34.2 %** | 73.2 | 118.1 | **+61.3 %** |
| 256 | 5498 | 7149 | **+30.0 %** | 45.64 | 34.30 | **−24.9 %** | 90.8 | 154.1 | **+69.6 %** |
| 512 | 6015 | 7232 | **+20.2 %** | 83.31 | 68.23 | **−18.1 %** | 121.6 | 216.6 | **+78.2 %** |
| 1024 | 5861 | **OOM** | — | 169.68 | — | — | 199.6 | — | — |

扩展效率（相对 batch 1 的 `tok/s/req`）撑得久得多：batch 64 上 `async_scheduling` 是
**0.76–0.78，而 `main` 是 0.54–0.55**；"最后一次划算的翻倍"标记两边都落在 batch 64，但 gain/cost 是
**1.77 对 1.22**。

### 3.2 等吞吐对比：结论所在

| | `main` | `async_scheduling` |
| --- | --- | --- |
| 达到约 6000 tok/s 所需并发 | 512 | 128 |
| 实际交付吞吐 | 6015 | 6791（+13 %） |
| TPOT | 83.31 ms | **17.77 ms（低 4.7 倍）** |
| ITL p50 / p90 | 70.3 / 125.4 ms | 15.1 / 15.2 ms |
| prefill 延迟 | 121.6 ms | 118.1 ms |

在同等交付吞吐下，`async_scheduling` 分支在**每一个**维度上都更好，**包括 TTFT**。§3.1 里那个 TTFT
回退只在按**同等并发**比较时才出现 —— 而在那种比法下，`async_scheduling` 每秒干的活多得多。

### 3.3 GPU 空闲率与 GPU busy —— 纯 decode

| batch | `main` 空闲（r1/r2） | 分支空闲（r1/r2） | `main` busy µs/步 | 分支 busy µs/步 | Δ busy |
| --- | --- | --- | --- | --- | --- |
| 1 | 68.2 / 68.0 % | 76.0 / 66.7 % | 3203 | 3255 | +1.6 % |
| 8 | 62.4 / 62.1 % | 59.8 / 59.4 % | 4219 | 4277 | +1.4 % |
| 32 | 59.1 / 59.9 % | 50.5 / 50.7 % | 5414 | 5470 | +1.0 % |
| 64 | 51.6 / 52.1 % | **29.2 / 28.5 %** | 7856 | 7910 | +0.7 % |
| 128 | 42.3 / 41.7 % | **7.1 / 7.0 %** | 14083 | 14229 | +1.0 % |
| 256 | 32.5 / 32.5 % | 12.2 / 13.0 % | 26826 | 27068 | +0.9 % |
| 512 | 28.4 / 26.5 % | **1.5 / 1.5 %** | 51132 | 51526 | +0.8 % |

**先读最后两列：kernel 就是同一批 kernel**，差异在 1 % 以内。这个分支没有让 GPU 变快，它只是不再把
GPU 闲着。不开 profiler 的每步墙钟时间随之下降：batch 512 **70.5 → 52.3 ms（−25.8 %）**、
batch 128 **24.3 → 15.3 ms（−37.0 %）**、batch 64 **16.3 → 11.1 ms（−31.9 %）**。

batch 512 上残余空闲是 52.3 ms 中的 0.8 ms —— 主机基本已经没有东西可藏了。batch 256 是例外
（空闲 12.6 %，比 batch 128 的 7.1 % 更差）：见 §5.3。

### 3.4 主机侧每步分解，纯 decode（ms）

每格都是 `main` → `async_scheduling`，第 1 轮对第 1 轮。

| 字段 | batch 1 | batch 64 | batch 512 | 含义 |
| --- | --- | --- | --- | --- |
| `sched` | 0.092 → **0.020**（−78 %） | 0.239 → **0.077**（−68 %） | 1.274 → **0.494**（−61 %） | schedule() 总计 |
| `sched_ret` | 0.077 → **0.006**（−92 %） | 0.176 → **0.015**（−91 %） | 0.847 → **0.082**（−90 %） | 构建 SchedulerOutput |
| `sched_ret_gpu` | 0.063 → **0.000** | 0.162 → **0.000** | 0.831 → **0.000** | `sched_ret` 内的设备工作 —— **彻底消失** |
| `bld_meta` | 0.163 → **0.314**（+92 %） | 0.219 → 0.344（+57 %） | 0.647 → **0.506**（−22 %） | attention 元数据 |
| `sample`（主机） | 0.785 → 0.683（−13 %） | 5.754 → **0.688**（−88 %） | 56.054 → **1.089**（−98 %） | 采样器里的主机时间 |
| `sample_gpu` | 0.510 → 0.434 | 5.540 → 3.896 | 56.907 → 40.565 | 采样器的设备 elapsed |
| `dth` | 0.023 → 0.035 | 0.110 → **0.034** | 2.913 → **0.036**（−99 %） | 设备→主机取 token |
| `ci` | 0.006 → 0.012 | 0.031 → 0.043 | 0.270 → 0.269 | commit / 记账 |

三件事值得挑出来：

* **`sample` 的主机时间变平了。** 在 `main` 上它几乎 1:1 地跟着 `sample_gpu`（batch 512 上 56.05 对
  56.91 ms）—— 主机当时是**在采样器里等**。在 `async_scheduling` 上它在每个 batch 上都是约
  0.7–1.1 ms：纯发射成本。
* **`sched_ret_gpu` 在每个 batch 上都恰好是零。** 调度已经不再触碰设备。也正因如此这个字段在
  `a6c6927` 里被删掉了：一个只可能读出零的指标不值得占一对 CUDA event。
* **`bld_meta` 发生了交叉。** 它在 batch 1–64 上贵约 2 倍，在 batch 512 上便宜约 22 %：常驻的
  slot 索引表有一笔固定的每步开销，只能靠宽度摊薄。这笔固定开销，加上多出来的 `dth` / `ci` 记账，
  正好就是 batch 1 上那 −1.9 %。

把 batch 512 上非 forward 的各段加起来是 **61.2 → 2.5 ms**，不过 `main` 那 61.2 ms 里大部分是主机
**在 `sample` 里等**而不是在干活。干净的说法是 trace 独立给出的那一个：**`main` 每步留下 19.4 ms
的墙钟时间 GPU 无法重叠（70.5 墙钟 − 51.1 busy）；`async_scheduling` 只留下 0.8 ms**（52.3 − 51.5）。

### 3.5 batch 512 的设备侧 op 分解（`async_scheduling`，ms/步，取自 `key_averages`）

GPU busy 合计 **51.5 ms/步**。采样器合计（`sample_gpu`）**40.6 ms = 79 %**；模型前向
**10.7 ms = 21 %**；logits 0.89 ms。

| op | ms/步 | 占整步 | 归属 |
| --- | --- | --- | --- |
| **`aten::sort`** | **15.43** | **30.0 %** | top-p |
| `flash_attn::_flash_attn_varlen_forward` | 4.89 | 9.5 % | attention |
| `aten::mm` | 3.62 | 7.0 % | 模型 GEMM |
| `aten::sub` | 3.08 | 6.0 % | penalties |
| `aten::copy_` | 2.67 | 5.2 % | 混合 |
| `aten::mul` | 2.66 | 5.2 % | 混合 |
| `aten::div` | 2.50 | 4.9 % | penalties / temperature |
| `aten::_softmax` | 2.45 | 4.8 % | top-p + 最终 softmax（2 次/步） |
| `aten::where` | 2.24 | 4.4 % | penalties / top-k / greedy 选择 |
| `aten::masked_fill_` | 2.18 | 4.2 % | penalties / top-p |
| Memcpy DtoD | 1.92 | 3.7 % | `rep.repeat(1, vocab)` |
| `aten::topk` | 1.87 | 3.6 % | top-k |
| `aten::scatter_` | 0.92 | 1.8 % | top-p 散回原序 |
| `aten::cumsum` | 0.83 | 1.6 % | top-p |
| `aten::fill_`、`argmax`、`gt`、`sum` | 2.89 | 5.6 % | 采样器尾部 |
| `aten::addmm` | 0.47 | 0.9 % | 模型 |

模型 + attention 是 **约 9 ms**。其余全是采样器在啃 `[512, 151936]` 的 fp32 张量 —— 每个被物化的临时
张量 311 MB，而这样的临时张量有十几个。

---

## 4. 这个分支到底改了什么

三个机制，按收益排序。

**（a）单步 lookahead 流水线** —— [`engine.py`](../src/qwen/engine.py)。`step()` 现在是 `step_0`
（schedule + forward + sample **发射**，不做任何同步）接 `step_1`（`sample_in_flight_step()`，排空
**上一步**）。`in_flight_steps` 是一个 deque；采样出的 token 是**在设备上**写进 token 表的
（`add_sampled_tokens_on_device`），而 `update_projected_state_in_advance()` 会提前推进 scheduler 的
视图，使得第 *N+1* 步的元数据可以在不知道第 *N* 步 token 取值的情况下构建。唯一剩下的同步是对上一步
的 `events.synchronize("dth")`，而到那时拷贝早就落地了。**这就是把每步 19.4 ms 暴露的主机时间变成
0.8 ms 的那一下。**

**（b）无同步的采样准备** —— [`sampling.py`](../src/qwen/sampling.py)。`bin_counts_and_mask` 那种每步
构建 Python list 再 H2D scatter 的做法没了；`prompt_mask` / `output_counts` / `output_mask` 现在是常驻
设备张量，按 slot 索引 gather。`apply_top_k` 的 `max_k` 取自 top-k 表的**pinned 主机镜像**，而不是
`top_k.max().item()`，所以选择 `topk` 的宽度不再花一次设备→主机同步。`MAX_EFFECTIVE_TOP_K = 1024`
给 `topk` 宽度设了上限。

**（c）slot 索引的常驻表** —— [`scheduler.py`](../src/qwen/scheduler.py)。`TokenIdTable`、
`SamplingParamTable` 和各个请求缓冲都按每请求一个稳定 slot 索引，于是每步的工作是设备上的一次
`index_select` 而不是 Python 迭代。`sched_ret_gpu → 0` 就是这个改动。

---

## 5. 回退与风险

### 5.1 batch 1024 在 `apply_top_p` 内 OOM —— **1024 并发被堵死**

```
File ".../qwen/sampling.py", line 152, in apply_top_p
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.73 GiB.
GPU 0 has a total capacity of 23.64 GiB of which 1.72 GiB is free.
```

`1024 × 151936 × 4 B = 622 MB` 是一个 fp32 临时张量的大小，而 `sort` 需要值**和** int64 索引再加
workspace —— 单笔 1.73 GiB。峰值 `memory.used` 在 `async_scheduling` 上是 **24564 MiB 里的
24200**，`main` 上是 23972：流水线多留了一步的张量存活，所以 `async_scheduling` 离天花板近了约
228 MiB，于是翻车。`main` 在 batch 1024 上跑完了，给出 5861 tok/s（低于它自己 batch 512 的峰值，所以
吞吐上没有损失什么 —— 损失的是**优雅降级**：引擎现在是**失败**而不是变慢）。§7.1 修的是病因而不是
症状。

### 5.2 batch ≥ 32 时 TTFT +37 % … +78 %

batch 512 上 prefill 延迟从 **121.6 → 216.6 ms**。与数据一致的机制是：一个请求的首 token 由第 *N* 步
产出，但只在第 *N+1* 步的 drain 里才被 commit，因此 TTFT 整整多出后面一轮迭代 —— 而在高并发下那一轮
经常又是一个 8192 token 的 prefill chunk。惩罚的大小随步时间走（batch 64：+34 ms ≈ 3 个 decode 步，
或约 ⅓ 个 prefill 步；batch 512：+95 ms ≈ 一个 prefill 步），这正是该机制所预测的。decode 的 ITL
**处处更好** —— batch 512 上 p50 **70.3 → 52.4 ms**、p90 **125.4 → 119.5 ms** —— 所以代价被限制在首
token 上。

这是异步调度标准的、公认的代价。之所以要标出来，是因为 sweep 表读起来像回退，而且它值得一个**有意
的决定**而不是一个意外：如果首 token 延迟是产品硬要求，可以优先排空产出了**首** token 的那一步，或者
把 prefill chunk 挡在某次准入紧随其后的那一轮之外。

### 5.3 `async_scheduling` 更不稳，batch 256 是一个看得见的离群点

* 每 20 步窗口的异常步：**`main` 在每个 batch 上都是 0/20**；`async_scheduling` 是 1–4/20。
* batch 256 的空闲率 **12.6 %**，比 batch 128 的 **7.1 %** 更差 —— 非单调。
* batch 256 的 dump 显示了机制：20 步里有 2 步耗时约 90 ms，而常态是 27–28 ms；而且在第 79 行，尖峰
  是**设备侧的** —— `fwd_gpu` 64.1 ms 对局部中位数 7.1 ms（9.1 倍）—— 不是主机停顿。
* `async_scheduling` 在 batch 256 和 512 上报告的 `profiler overhead` 变成了**负值**
  （−16.7 %、−15.2 %）：被 profile 的那次比"干净"的那次更快，而这只在干净窗口里逮到了这些停顿时才会
  发生。因此把 `async_scheduling` 在 batch 256 的空闲率**当作上界**。

一个 kernel 没变的步里出现 9 倍的**设备侧**停顿，而这个分支的峰值显存离天花板只有 364 MiB ——
指向的是 allocator 压力：batch 256 上每个全词表 fp32 临时张量是 156 MB，流水线会让两步的量同时可达，
而一次 cache miss 的 `cudaMalloc` 必须同步，还可能触发 `cudaFree`。与 §5.1 同源。在去追别的东西之前，
值得先用 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 加上
`torch.cuda.memory_stats()` 的 `num_alloc_retries` 检查确认一下。

### 5.4 batch 1 回退约 2 %

吞吐 102 → 100 tok/s，TPOT 9.83 → 10.01 ms。`bld_meta` +0.150 ms、`dth` +0.013 ms、`ci` +0.006 ms
盖过了 `sched` 的 −0.072 ms。幅度小、真实存在，而且正是"用固定的每步准备换取按量扩展"这类改动应有的
形状。注意 batch 1 也是 `async_scheduling` 噪声最大的点（run 间 3.22 %，而 `profile_slot` 第 1 轮报
76 % 的空闲率、第 2 轮报 66.7 %），所以真实值大概落在 −1 % … −3 % 之间。

### 5.5 `step_0`/`step_1` instrumentation 双计 —— 在 `a6c6927` 上已修

见 §2.2 坑 1。batch 512 上 `step_0 + step_1` 读作 104 ms/步，而步实际是 52 ms。这误导了异常检测器
（`step_1` 是它标记最多的字段，因为继承了**下一步**的尖峰），也会误导任何读这批 dump 的人。

**测量之后已解决。** `a6c6927` 在 `sample_in_flight_step()` 里用
`timed(pre_sch_out.step_metrics, "step_1")` 包住 drain，并从 `LLMEngine.step()` 和
`ServingDriver.run()` 里删掉了 `start("step_1")` 调用，于是这一对不再重叠。注意它对未来 dump 的后果：
`step_1` 现在只量 **drain**，所以它也不再是步墙钟时间的替身了 —— 那个位置换成了
`step_0 + step_1`，**本报告叫你去读的那个字段，和 `a6c6927` 给你的那个字段不是一回事。**

---

## 6. 剩下的东西：采样占设备工作的 79 %

主机停顿消失之后，batch 512 上引擎把 **51.5 ms/步里的 40.6 ms 花在采样器上**，而**真正的模型只占
9 ms**。采样器的成本不是算术，是跨 `[bsz, 151936]` fp32 张量的内存流量，而其中大部分是可以避免的：

1. **`apply_top_p` 对完整词表排序 —— 15.4 ms/步，整步的 30 %。** 它跑在 `apply_top_k` 已经把前
   **20**（`config.top_k = 20`）之外的一切设成 `-inf` **之后**。为了给 20 个有限值排名而排 151936
   列，宽度大约是问题需要的 **7600 倍**。更糟的是 `torch.topk` **本来就返回降序排好的值** —— 这次
   排序是**冗余**的，而不只是过宽。`constants.py` 里甚至已经写下了这个洞见（"top_k ≤ 1024 时 sort
   开销曲线几乎是平的"），却只把这个上限用在了 `topk` 的宽度上。
2. **`apply_penalties` 为了施加一个逐行标量而物化 `[bsz, vocab]` 临时张量。**
   `rep = repetition_penalty[:, None].repeat(1, vocab)` 是一次 311 MB 的 DtoD 拷贝（就是那 1.92 ms 的
   Memcpy DtoD 行），存在的唯一目的是让下一行能用掩码去索引它。接着是
   `rep[~(prompt|output)] = 1.0`（`masked_fill_`，2.18 ms）和
   `where(logits>0, logits/rep, logits*rep)`（`gt` + `div` + `mul` + `where`，约 8 ms）。整件事本可以
   是对 `logits` 的一次融合 elementwise pass，配一个广播的 `[bsz,1]` 标量和两个 `[bsz,vocab]` bool
   掩码。
3. **每步两次全词表 softmax。** `apply_top_p` 对排序后的 logits 做一次 softmax，`sample` 又对完整词表
   再做一次（`aten::_softmax`，5 步 10 次 = 每步 2 次，2.45 ms）。
4. **死行保护和 greedy 路径各再加一次全词表 pass**（`sum`、`isfinite`、`masked_fill`、`argmax` ——
   合计约 1.4 ms/步）。它们是正确的，而且刻意写成无分支（注释解释了原因），但它们放进 top-k 窗口里同样
   会便宜约 150 倍。

---

## 7. 建议，按顺序

### 7.1 把 top-p 放进 top-k 窗口内做 —— 省下约 15 ms/步**并**修掉 batch 1024 的 OOM

`apply_top_k` 已经算出了 `top_vals, top_idx = torch.topk(logits, max_k)`，而且是**降序的**。把采样器
后面整条链都跑在那个 `[bsz, max_k]` 块上（`max_k ≤ 1024`，本配置下是 20）：

* 在 `[bsz, max_k]` 上做 `softmax` → `cumsum` → `(cum - p) > top_p` —— 没有 `sort`，不需要
  `scatter_` 散回词表原序，也没有 `[bsz,vocab]` 的 `masked_fill`。
* 在 `[bsz, max_k]` 的概率上 `multinomial`，再用 `gather` 通过 `top_idx` 把结果映射回去。
* 只对 top-k 真正被禁用的行（`max_k <= 0`）保留现在的全词表路径 —— 这个条件代码已经在主机侧检测，
  不需要同步。

预期：`sort`（15.4）、`scatter_`（0.92）、`cumsum`（0.83）、一次 `_softmax`（约 1.2）以及 top-p 的
`masked_fill`/`where` 的大部分流量坍缩到约 0.1 ms —— **batch 512 上大约是 51.5 ms/步里的 18 ms**，
也就是 TPOT 约 52 → 约 34 ms，峰值吞吐有望越过 10000 tok/s。那笔 1.73 GiB 的 sort 分配随之消失，
解开 batch 1024（§5.1），也缓解 §5.3 背后的 allocator 压力。

次序提示：这只改采样器的**形状**，不改它的分布 —— 先 top-k 再在 top-k 窗口内做 top-p，与先 top-k 再
在全词表做 top-p 在数学上完全等价，因为被排除的条目两种做法下都是 `-inf`。改完用
`test_sampling.py` 加一次固定种子的 logits 前后对比验证。

### 7.2 融合 penalty pass —— 省下约 8 ms/步

去掉 `.repeat(1, vocab)`。把逐行因子
`rep_row = where(prompt_mask | output_mask, rep[:,None], 1.0)` 隐式算在对 `logits` 的单个表达式里；
更好的做法是把 penalties 放到 7.1 的 top-k 收窄**之后**，这样这一遍的宽度就是 `[bsz, max_k]`。如果
`do_penalities` 开着但参数都是恒等值（`rep == 1.0 and freq_pen == 0 and pres_pen == 0` —— 也就是
`main` 的默认值，而且可以从常驻表在主机侧检测、不需要同步），就整遍跳过。

### 7.3 确认并修掉 allocator 压力

用 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 在 batch 256/512/1024 上跑
`async_scheduling`，并逐步记录 `torch.cuda.memory_stats()['num_alloc_retries']`。如果 retry 与那些
9 倍的 `fwd_gpu` 尖峰相关，那么 §5.3 里 batch 256 的非单调空闲率和 batch 1024 的 OOM 是同一个 bug，
而 7.1 基本上把两个都化解了。投入低，信息量高。

### 7.4 在下一次 A/B 之前修掉 `step_1` 重叠 —— ✅ 已在 `a6c6927` 完成

见 §5.5。两行改动，就能让未来每一份 dump 不再把步时间多报 2 倍。它随 re-commit 落地，时间在这批测量
**之后**；`log_vast/log928/` 里的 dump 仍然是旧行为。

### 7.5 然后，也只有到那时，再去看 batch ≤ 16

batch 1–16 是持平到差 1.9 %，而 batch 1 —— profile 覆盖到的唯一一个小 batch —— 仍然
**68 % GPU 空闲**：主机需要约 10 ms 去发射一个 kernel 只占 3.2 ms 的步。那是 CUDA graph / 降低
dispatch 的问题，不是调度的问题，本分支完全没有碰它。`bld_meta` 翻倍的固定开销（§3.4）是本分支唯一
把它弄得更糟的那一块，也是最便宜的切入点。

### 7.6 在一台 PCIe gen-4 的实例上重测

见 §1.1。这个分支的招牌数字是在 gen-1 链路上取的，而这恰好替一个"核心就是减少主机↔设备往返"的改动
加了分。预期收益方向不变、幅度有所收缩。

---

## 8. 注意事项

1. **与 log924 不是同一台实例** —— 450 W 对 250 W 功耗墙、驱动 550 对 580、PCIe gen 1 对 gen 4。
   本报告的任何数字都不能与 log924 比较。在 log928 内部，两个分支在同一台机器上、彼此相隔 25 分钟内
   跑完，且 `async_scheduling` 在前（没有预热优势）。
2. **`repetition_penalty` 1.1（`async_scheduling`）对 `rep_pen` 1.0（`main`）** —— 同样的 kernel、
   同样的计时，不同的生成文本。见 §1.3。
3. **batch 8–512 的跨分支 step 指标表是本地重新生成的**，不是取自 log 目录（那里只有 batch 1）。
   同一个工具、同一段行区间、同一批 dump；用的是临时副本，所以 `log_vast/` 未被修改。
4. **`async_scheduling` 在 batch 256 和 batch 1 的 profile 数字受 §5.3 的停顿污染** —— 那个负的
   "profiler overhead" 就是信号。batch 128 和 512 在两轮里都是干净的，论证靠它们。
5. **采样器占比读自 `key_averages`，而那里 `aten::` 行和它们的 kernel 行都被列出** —— 那张表的百分比
   加起来会超过 100 %。§3.5 的表只用 `aten::` 行，其合计（约 51 ms）与独立的 trace `gpu_busy`
   （51.5 ms）以及 CUDA event 字段（`fwd_gpu` + `logits_gpu` + `sample_gpu` = 52.2 ms）互相校验。
6. **`torch.cuda.set_sync_debug_mode("warn")` 的输出不在 log 目录里** —— 没有 `warning` 文件，两份
   `pytest.log` 里也都不含同步警告。所以"`async_scheduling` 的 decode 路径无同步"这个论断，靠的是
   `sched_ret_gpu → 0`、主机侧 `sample` → 约 1 ms 和 1.5 % 的空闲率，而不是 debug mode 的证据。
   把那份输出抓下来能让这个论断无懈可击。
7. **每个 profile 点 20 个 decode 步**（第 65–84 行），AR(1) 折减后的 `n_eff` 通常是 5–20。对本文报告
   的 30–100 % 量级的效应够用；对任何小于约 5 % 的效应不够用。
