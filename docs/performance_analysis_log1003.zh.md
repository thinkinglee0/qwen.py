# 性能分析 —— RTX 4090 上 qwen.py 对 vLLM 0.30.0（log1003）

> 英文原版：[`performance_analysis_log1003.md`](./performance_analysis_log1003.md)

**第一把外部标尺。** 在这之前的每一份报告（[log910](./performance_analysis_log910.md) …
[log1001](./performance_analysis_log1001.zh.md)）量的都是 qwen.py 对自己 —— 分支对分支、一次一个提交。
这一份量的是它对 vLLM，同一台机器、同一个时间窗、跑
`test_benchmark_sweep_batch_size` 已经定义好的那个负载。

**数据来源**

| 内容 | 路径 |
| --- | --- |
| 并发 sweep，qwen.py，第 1–2 轮 | `log_vast/log1003/benchmark_fused_top_kp{,2}/` |
| 纯 decode 空闲率 profile，qwen.py，第 1–2 轮 | `log_vast/log1003/profile_fused_top_kp{,2}/` |
| vLLM 臂 A（无 compile、无 graph） | `log_vast/log1003/benchmark_vllm/vllm_eager.{jsonl,log}` |
| vLLM 臂 B（inductor + CUDA graph） | `log_vast/log1003/benchmark_vllm/vllm_cudagraph.{jsonl,log}` |
| GPU 频率 / 功耗 / throttle 轨迹，200 ms | `log_vast/log1003/**/gpu.4090.*.csv` |
| 宿主清单、CPU、affinity、解释器速度 | `log_vast/log1003/host_info`、`host_cpu_info`、`gpu.static.csv` |
| 依赖快照 | `log_vast/log1003/constraints.txt`（主环境）、`vllm_pip_freeze.txt`（vLLM 环境） |
| 上一次（同分支、更快宿主）的基线 | `log_vast/log1001/` |

**被测代码** —— qwen.py 位于 `49bb10d` 之后的那个提交（fused top-k+top-p，外加
`freq_pen`→`frequency_penalty` 重命名，以及给 `analyze_metrics` 加的
`o_tok_throughput`/`io_tok_throughput`）。版本归属可以从 dump 本身核对：log1001 的 `pytest.log` 记的是
`freq_pen=0.0`，log1003 记的是 `frequency_penalty=0.0`，而 log1003 的 `benchmark_metrics` 多了那两个
吞吐 key。vLLM 为 **0.30.0**（torch 2.13.0+cu130、CUDA 13.0、Python 3.12.3），装在独立 venv 里，
按 [`vast-evn-build.md` §6a](../env/vastai/vast-evn-build.md)。

**测量工具** —— [`benchmark/tool/vllm_sweep.py`](../benchmark/tool/vllm_sweep.py)，逐项镜像
`test_benchmark_sweep_batch_size`：离线闭环（全部请求在计时之前入队）、固定 512 入 / 128 出、
`ignore_eos`、`max_num_seqs` 作为 sweep 变量，采样参数逐项抄 `generation_config.json`。
§2.2 解释为什么每一项都是承重的。

**运行顺序**（同一台实例，约 50 分钟，中途未重启，一次只跑一个引擎）：vLLM 臂 A 03:51–04:30 →
臂 B 04:33–05:00 → qwen.py sweep 第 1 轮 05:02–05:14 → 第 2 轮 05:17–05:29 → qwen.py profile
第 1 轮 05:33 → 第 2 轮 05:38。**vLLM 跑在前面**，机器更冷。

---

## 摘要

1. **batch 512 上两个引擎都是设备瓶颈，qwen.py 给出 10 970 tok/s，对 vLLM 的 13 505（无 compile、
   无 graph）和 15 848（compile + graph）—— 即 81 % 和 69 %。** 峰值对峰值是
   **10 970 对 18 032 tok/s = 61 %**。
2. **batch 512 上这个差距干净地分解成 1.23 × 1.17 = 1.44。** 1.23 是引擎设计，同条件测出（两边都
   eager、都 `FLASH_ATTN`、KV 容量相同、采样参数相同）。1.17 是 `torch.compile` + CUDA graph 给
   vLLM 买到的东西 —— 一项 qwen.py 完全没有的能力。
3. **第二个因子在低并发下极其巨大：batch 1 上 7.28×**，然后单调衰减：batch 64 上 3.58×、256 上
   1.72×、512 上 1.17×、1024 上 0.98×。这是 [log1001 §7.3](./performance_analysis_log1001.zh.md)
   指出的那笔"每步固定税"（约 4 400 次 `aten::` 调用）第一次拿到外部价签。
4. **引擎设计这个因子在 batch ≤ 64 上平得出奇：1.59–1.65×**，之后收窄到 512 上的 1.23×。那里两个引擎
   都是主机瓶颈、而且付的是**同一个**解释器，所以这个比值量的是两边主机循环的相对效率 —— 而且和绝对值
   不同，**两个主机瓶颈循环之间的比值是可以跨机器迁移的**（§2.4）。
5. **这台宿主的 Python 比 log1001 那台慢约 2.3×，而数据证明问题只在宿主。** 同一份代码、同一个分支：
   `gpu_busy_from_trace` 和 log1001 **在每个 batch 上都一致到 ±2 %**，而 `wall_clean` 在 batch ≤ 128
   上高 2.26–2.38×、到 512 收敛到 1.01×。原因：`CPU max MHz 2800` × `scaling 80 %` ≈ **2.24 GHz**，
   被邻居负载压着（`load average 12.11` 是在我们最后一次 run 结束 12 分钟后、无人登录时测的）。
6. **所以 sweep 的两半要分开读。** batch 512/1024 是测量，可迁移。batch ≤ 256 的绝对吞吐不可迁移，
   但第 4 条里的**比值**可以。
7. **log1001 的设备侧结论被证实，没有被扰动。** 采样器在 batch 512 上仍占
   **qwen.py 设备工作的 61 %**（26.93 ms/步里的 16.32 ms）—— 在不同的硅片上复现了 log1001 的 60 %。
   [log1001 §7.1–7.2](./performance_analysis_log1001.zh.md) 不受本报告任何内容影响，依然是已识别的
   最大收益项，值约 1.8× 的设备时间。
8. **功耗 cap 的偏置是**对 vLLM**更狠的，所以测出的差距是下界。** 400 W 上限在 vLLM 的 busy 样本里
   有 **64.7–68.6 %** 处于 Active，而 qwen.py 只有 **42.3–42.6 %**（中位功耗 394–397 W 对
   356–359 W；中位 SM 2700 对 2775 MHz）。450 W 的机器上 vLLM 赚得比 qwen.py 多。
9. **复现性。** qwen.py run 间：最差 2.87 %（batch 16），batch ≥ 256 时 < 0.5 %，步数逐点相同，
   22 次 run 零抢占。vLLM：每点 3 次取中位数。设备侧 `sample_gpu` 在 qwen.py 两轮之间一致到
   **0.03 %**。

---

## 1. 被测系统

### 1.1 硬件 —— 以及为什么这是本报告最该先知道的事

| | log1003（本报告） | log1001（上一份） |
| --- | --- | --- |
| GPU | RTX 4090，驱动 **580.142** | RTX 4090，驱动 550.127.08 |
| enforced 功耗上限 | **400 W**（max 600） | 450 W（max 600） |
| HBM 实测 | **919 GB/s** | 876.8 GB/s（厂商面板值） |
| CPU | **AMD EPYC 7402**，24C/48T，`max 2800 MHz`、`scaling 80 %` → **≈ 2.24 GHz** | Ryzen 5 7500F，6C/12T，3.7/5.0 GHz |
| L3 | **128 MiB 分 8 个实例**（8 CCX × 3 核 × 16 MiB） | 32 MiB，单 CCD |
| CPU 分配 | CFS 配额 `cfs_quota_us 1152000` = **11.52 核**；affinity **全部 48**；1 个 NUMA node | 12/12 |
| 邻居负载 | **`load average 12.11 / 12.81 / 12.96`**，自己跑完 12 分钟后测，0 用户登录 | 未记录 |
| 3 M 次 Python 循环 | **276 / 275 / 285 ms** | 未记录 |
| `aten` launch overhead 中位数 | 7.56 µs（过 §1b 的 < 10 µs 闸门） | 未记录 |
| PCIe | gen 4 × 16 | gen 4 × 16 |

GPU 是健康的：面板那个 387.5 GB/s 纯属错误，实测 919 GB/s 是这张卡理论 1 008 的 91 %。全程无热降频
（≤ 60 °C，`hw_thermal_slowdown` 从未 Active）。

**宿主**才是问题，§2.4 专门讲它。注意 §1b 的闸门**没抓住**什么：launch overhead 以 7.56 µs 轻松过线，
因为那个微基准量的是**一个 op 在紧循环里**，而引擎每步要为约 4 400 次 `aten::` 调用付解释器成本。
真正能抓住它的是那个纯解释器循环，现在已经加进 runbook。

### 1.2 负载 —— 结构上完全相同

Qwen2.5-0.5B-Instruct，bf16，`vocab_size = 151 936`。来自 `test_benchmark_sweep_batch_size`：

* `10 × batch` 个请求（最少 64），**固定 512 输入 / 128 输出 token**，`max_model_len = 1024`。
* `ignore_eos` —— 每个请求恰好产 128 个 token。两边都有断言：qwen.py 的 `o_tok_num`，以及工具里的
  `assert o_tok == req_num * OUT_LEN`。
* 随机均匀 token id，来自 `SEED = 1234`。内容不带数据依赖（输出长度钉死、id 均匀），所以 prompt 由种子
  重新生成而不是搬运；臂 B 的 header 记了 `prompts_sha256_16 = ffc9f6a0ecf792c5` 作为集合的凭证。
* **离线闭环**：全部请求在 `t0` **之前**入队，然后排空。没有 HTTP、没有到达率、没有 tokenizer 在环。
  `engine.benchmark()` 和 `LLM.generate()` 的计时边界相同。

### 1.3 对齐了什么，以及怎么从日志核实

| 旋钮 | qwen.py | vLLM | 核实 |
| --- | --- | --- | --- |
| KV 容量 | `num_blocks 4096 × block_size 256` = **1 048 576 tokens**（12.0 GiB） | `block_size 16`、`num_gpu_blocks_override 65536` | vLLM 日志：`GPU KV cache size: 1,048,576 tokens`。这个 override 把 vLLM 从它自己算出的 110 679 blocks（1.77 M tokens）**压下来**了，没白送空间 |
| attention kernel | flash-attn 2.8.3 | `Using FLASH_ATTN attention backend out of potential backends: ['FLASH_ATTN', 'FLASHINFER', 'TRITON_ATTN', 'FLEX_ATTENTION']` | 两条臂、11/11 个点 —— **kernel 家族不是混淆项** |
| 采样 | temp 0.7、top_k 20、top_p 0.8、rep_pen 1.1、freq/pres 0 | 逐项相同 | §2.2 |
| prefill 批 | `max_num_batched_tokens 8192`、`long_prefill_token_threshold 8192` | `max_num_batched_tokens 8192`，chunked prefill 开 | qwen.py 侧 `prefill_chunk.mean = 1.0` |
| prefix caching | 未实现 | `enable_prefix_caching=False` | vLLM non-default args |
| 抢占 | 22 次 run 全 0 | 0；KV 余量日志记为 `1024.00x` 并发 | 两边 |
| `OMP_NUM_THREADS` | 8 | 8（从 `~/.bashrc` **继承**） | 已记录；见 §5.4 |
| CPU 绑定 | 无 | 无 | 两条臂一致 |

---

## 2. 方法

### 2.1 三条臂，以及 `enforce_eager` 实际切掉了什么

| 臂 | 配置 | 归因 |
| --- | --- | --- |
| qwen.py | 在开发的引擎 | — |
| **A** —— `enforce_eager=True` | vLLM 日志：`'mode': <CompilationMode.NONE: 0>`，零条 `Capturing CUDA graph` | **引擎设计**：scheduler、KV 管理、采样器、kernel 选择 |
| **B** —— 默认 | `'mode': <CompilationMode.VLLM_COMPILE: 3>`、backend `inductor`、`Dynamo bytecode transform time: 3.89 s`、CUDA graph 已捕获 | A→B = **`torch.compile` 融合**与**CUDA graph** 两者**合在一起** |

**A→B 的差值不是"CUDA graph"一项。** `enforce_eager=True` 同时关掉了 inductor 编译**和**图捕获，所以
这批数据里两者不可分。本项目早先把这条臂叫做"CUDA graph 臂"；日志说它是 compile + graph。要拆开需要
第三条臂（compile 开、`cudagraph_mode=NONE`）—— §6.4。

### 2.2 吞吐定义，以及把这个对比做废的三种方式

**吞吐是每秒输出 token 数**，两边都是 `o_tok_num / elapsed`。这是本系列自 log910 起一直用的口径，
也正是 vLLM 所称的 *Output token throughput*。在这个固定 512 入 / 128 出的形状下，vLLM 自家工具打印的
"total token throughput" 对每一个请求都**正好大 5.0 倍**，是一个不携带信息的常数倍 —— 拿它对 qwen.py
的数字会凭空制造出 5 倍差距。`analyze_metrics` 现在显式输出 `o_tok_throughput` 和
`io_tok_throughput`，让这个区分不会再丢。

另外两个陷阱，这里都避开了：

* **采样参数是被测对象本身，不是无关旋钮。** log1001 之后，采样器占 qwen.py 在 batch 512 上设备工作的
  61 %。让 vLLM 跑 `temperature=0`，它的采样器会退化成一次全词表 `argmax`（qwen.py 自己 trace 里是
  0.71 ms/步），对 qwen.py 的 16.32 ms —— 约 **2.1 倍的幻影差距**，里面一行代码的差异都没有。更糟的是
  vLLM 在**没有任何请求**需要 penalties/top-k/top-p 时会走更便宜的分支，所以**部分对齐比不对齐更糟**：
  它看起来像对齐了，而两个引擎在做不同的功。
* **`detokenize=False`。** vLLM 默认增量 detokenize；qwen.py 循环里一次都不做（本测试
  `save_output=False`）。

### 2.3 噪声地板

| | 数值 |
| --- | --- |
| qwen.py run 间吞吐，最差 | **2.87 %**（batch 16） |
| qwen.py run 间，batch ≥ 256 | < 0.5 %（batch 512：0.42 %，batch 1024：0.05 %） |
| qwen.py 步数，第 1 轮对第 2 轮 | 每个 batch 完全相同（如 batch 512 都是 1 324） |
| qwen.py batch 512 的设备 `sample_gpu`，run 间 | **0.03 %** |
| qwen.py batch 512 的主机侧字段，run 间 | 全部在 4.6 % 内 |
| vLLM | 每点 3 次、报中位数，三次都在 `elapsed_all` 里 |
| vLLM 臂 B 在 batch 512 的散布 | **11.6 %**（38.997 / 41.352 / 43.517 s）—— 这个点带约 ±6 % |
| `aten` launch overhead 散布 | 6.5 %，runbook 的参考是 ±3 % |

臂 B 在 batch 128、256、512 上第一次 run 偏慢（12.487 对 9.857/9.687；20.878 对 18.173/18.072）——
尽管调了 warmup，图捕获和 inductor 预热还是漏进了第一个计时迭代。中位数吸收了它，但 batch 512 那
±6 % 是本报告最松的一个点。那里 1.44× 的差距是这个不确定度的约 7 倍。

### 2.4 宿主混淆项，以及它的影响到哪里为止

同一份代码、同一个分支、同型号 GPU，不同宿主：

| batch | `gpu_busy` log1001 → log1003 | `wall_clean` log1001 → log1003 | 吞吐比 |
| --- | --- | --- | --- |
| 1 | 3 146 → 3 184 µs（**+1.2 %**） | 9 674 → 22 733 µs（**2.35×**） | 0.42 |
| 8 | 3 903 → 3 960（+1.5 %） | 10 613 → 23 875（2.25×） | 0.44 |
| 32 | 4 360 → 4 421（+1.4 %） | 10 727 → 25 524（2.38×） | 0.45 |
| 64 | 5 398 → 5 445（+0.9 %） | 10 963 → 24 752（2.26×） | 0.46 |
| 128 | 8 346 → 8 396（+0.6 %） | 11 100 → 25 399（2.29×） | 0.50 |
| 256 | 15 149 → 14 857（−1.9 %） | 16 040 → 26 013（1.62×） | 0.72 |
| 512 | 27 305 → 26 930（−1.4 %） | 28 092 → 28 354（**1.01×**） | **1.00** |

设备在每个 batch 上都干了一致到 ±2 % 的活；主机慢 2.3×，直到 512 上设备成为长板。主机侧逐字段拆解
证实它是**均匀的**而不是局部的：`sched` 1.72×、`sched_run` 1.97×、`bld_meta` 2.10×、`sample` 2.35×、
`dth` 1.97×、`dth_wait` 1.90×、`ci` 1.97×。**每一个互不相关的纯 CPU 字段都是同样的约 2× —— 那是解释器
整体变慢，不是某个瓶颈。** 那个 3 M 次 Python 循环（276/275/285 ms，散布仅 3.6 %）说的是同一件事，
而且说明它是**时钟受限**而非争抢抖动。

三条推论，而且它们对报告的不同部分影响不同：

1. **设备侧结论可迁移。** 跨宿主 ±2 % 的一致性已经是这个项目仪器的上限。
2. **batch ≤ 256 的绝对吞吐不可迁移。** 两个引擎被同一个慢解释器同样压低。
3. **两个主机瓶颈循环之间的比值**可以**迁移。** 若 qwen.py 每步发 `N_q` 次 Python 操作、vLLM 发
   `N_v` 次，解释器速率为 `R`，则步时间是 `N_q/R` 和 `N_v/R`，比值 `N_q/N_v` 与 `R` 无关。batch 1 上
   qwen.py 的 GPU 空闲 86 %、vLLM 臂 A 约 77 %，两边都合格。这就是为什么 §3.2 的**引擎因子**那一列
   在绝对值不可用的地方依然可用。

> **这台宿主完全测不到的一件事**：qwen.py 小 batch 的绝对数字在快宿主上、且 vLLM 在同一窗口内测。
> 那需要另一台机器，而不是在这台上多花时间。见 §6.1。

---

## 3. 结果

### 3.1 三方 sweep —— 同一宿主、同一时间窗、一次只跑一个引擎

qwen.py 取两轮均值；vLLM 每点取三次中位数。全部是**输出 token/s**。（`concurrency_sweep.txt` 只打印
第 1 轮，所以它的 batch 512 峰值读作 10 993，而两轮均值是 10 970；81 % / 61 % 两个比例两种算法都一样。）

| batch | qwen.py | vLLM A（eager） | **A/qwen** | vLLM B（compile+graph） | **B/qwen** | **B/A** |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 43 | 71 | **1.64×** | 516 | 11.9× | **7.28×** |
| 2 | 82 | 134 | 1.64× | 900 | 11.0× | 6.70× |
| 4 | 162 | 267 | 1.65× | 1 781 | 11.0× | 6.66× |
| 8 | 324 | 535 | 1.65× | 3 318 | 10.3× | 6.20× |
| 16 | 632 | 1 014 | 1.60× | 5 796 | 9.2× | 5.72× |
| 32 | 1 218 | 1 958 | 1.61× | 9 193 | 7.5× | 4.70× |
| 64 | 2 375 | 3 773 | 1.59× | 13 518 | 5.7× | 3.58× |
| 128 | 4 392 | 6 502 | 1.48× | 16 622 | 3.8× | 2.56× |
| 256 | 7 672 | 10 470 | 1.36× | **18 032** | 2.4× | 1.72× |
| **512** | **10 970** | **13 505** | **1.23×** | 15 848 | **1.44×** | **1.17×** |
| 1024 | 9 622 | 13 038 | 1.36× | 12 841 | 1.33× | 0.98× |

| | qwen.py | vLLM A | vLLM B |
| --- | --- | --- | --- |
| 峰值吞吐 | **10 970 @ batch 512** | 13 505 @ 512 | **18 032 @ batch 256** |
| qwen.py 占对方峰值 | — | 81 % | **61 %** |

三个值得从表里读出来的形状：

* **qwen.py 和 vLLM A 的峰值在同一个位置（512），然后都在 1024 上回落** —— 这个负载的 KV 容量恰好是
  1 024 × 1 024 个 token，batch 1024 正是一点余量都不剩的那个点。
* **vLLM B 的峰值提前了两个翻倍，在 256**，到 512 已经*丢掉* 12 %、到 1024 丢掉 29 %。编译 + 图化的
  执行更早撞到设备天花板，所以有用的工作区间整体下移。
* **B/A 在 batch 1024 上穿过 1.0**（0.98×）：步足够长之后，编译和图什么也买不到，捕获开销还略微为负。

### 3.2 差距可以分解

在 batch 512，两个引擎都是设备瓶颈、宿主混淆项已经消失（§2.4）：

```
   vLLM B / qwen.py  =  1.44×
                     =  1.23×        ×  1.17×
                        引擎设计        compile + CUDA graph
                        (A/qwen)        (B/A)
```

1.23× 是诚实的同条件引擎对比：两边都 eager、都 `FLASH_ATTN`、KV 容量相同、采样参数相同、同一宿主、
同一时间窗。**在 GPU 是瓶颈的那个 batch 上 —— 也正是这个引擎为之设计的工况 —— qwen.py 的引擎距离
vLLM 在 23 % 以内。**

1.17× 是能力缺口，不是设计缺口：qwen.py 没有 `torch.compile` 路径、没有 CUDA graph 捕获，没有可比的
东西。

跨整个 sweep，这两个因子的行为完全不同：

| | batch 1 | 64 | 128 | 256 | 512 | 1024 |
| --- | --- | --- | --- | --- | --- | --- |
| 引擎因子（A/qwen） | 1.64× | 1.59× | 1.48× | 1.36× | **1.23×** | 1.36× |
| compile+graph 因子（B/A） | **7.28×** | 3.58× | 2.56× | 1.72× | 1.17× | 0.98× |

引擎因子在 **batch ≤ 64 上平在 1.59–1.65×**，之后单调改善。按 §2.4 的第三条推论，这个平的 1.6× 是两个
主机瓶颈循环在同一个解释器上的比值，所以它可以迁移：**qwen.py 的主机侧单步成本大约是 vLLM eager
主机侧单步的 1.6 倍**，在任何宿主上都是。用 log1001 §7.3 的话说，那就是每步约 4 400 次 `aten::` 调用
对 vLLM eager 路径所发指令数的价格。

compile+graph 因子才是 batch 1 上那 12× 头条的来源，而它几乎全是主机路径：设备一成为瓶颈，它就衰减
到零。

### 3.3 qwen.py 自己的设备侧 —— log1001 在另一台宿主上复现

纯 decode，取自 `profile_fused_top_kp{,2}`：

| batch | `gpu_busy` ms/步 | GPU 空闲 | `sample_gpu` ms/步 | 采样器占比 | `wall_clean` ms/步 |
| --- | --- | --- | --- | --- | --- |
| 1 | 3.18 | **86.0 %** | 1.07 † | 34 % † | 22.73 |
| 8 | 3.96 | 83.4 % | 1.03 † | 26 % † | 23.87 |
| 32 | 4.42 | 82.7 % | 1.16 † | 26 % † | 25.52 |
| 64 | 5.45 | 78.0 % | 1.39 | 26 % | 24.75 |
| 128 | 8.40 | 66.9 % | 3.50 | 42 % | 25.40 |
| 256 | 14.86 | 42.9 % | 8.29 | 56 % | 26.01 |
| 512 | **26.93** | 5.0 % | **16.32** | **61 %** | 28.35 |

† `sample_gpu` 是 CUDA event 区间而非 kernel 累加，所以空闲率高时它里面含主机气泡
（[log1001 §2.2](./performance_analysis_log1001.zh.md)）。batch 64 以下这几格不是 kernel 时间 ——
对比 log1001 同 batch 的值（0.31 / 0.38 / 0.65 ms），那是在 50–67 % 空闲下测的，而不是 83–86 %。
从 batch 64 往上两台宿主就一致了：1.37→1.39、3.53→3.50、8.31→8.29、16.35→16.32 ms。

**采样器在 batch 512 上占设备工作的 61 %，在不同硅片、不同驱动上复现了 log1001 的 60 %。** 所以
[log1001 §7.1–7.2](./performance_analysis_log1001.zh.md) 不受本报告任何内容影响，依然是已识别的最大
收益项：把采样器的尾巴搬进 top-k 窗口、再把 penalties 做成稀疏，应能把 16.32 ms/步压到约 4 ms，
即设备步从今天的 26.9 ms 到约 15 ms —— **设备侧约 1.8×**。

这对差距很关键：设备时间上的 1.8× **大于** batch 512 上到 vLLM B 的那整个 1.44× 差距。采样器的活不是
次要的收尾工作，它是 qwen.py 手上最大的那根杠杆。

### 3.4 频率与功耗 —— 偏置是对 vLLM 不利的

取自 200 ms 的 `nvidia-smi` 轨迹，只统计 `utilization.gpu > 50 %` 的样本：

| run | 样本数 | `sw_power_cap` Active | 中位 SM | 中位功耗 | 最高温度 |
| --- | --- | --- | --- | --- | --- |
| vLLM 臂 A | 5 065 | **68.6 %** | 2 700 MHz | 397.3 W | 60 °C |
| vLLM 臂 B | 5 792 | **64.7 %** | 2 700 MHz | 394.6 W | 59 °C |
| qwen.py 第 1 轮 | 3 008 | 42.6 % | 2 775 MHz | 359.0 W | 58 °C |
| qwen.py 第 2 轮 | 3 112 | 42.3 % | 2 775 MHz | 355.8 W | 59 °C |

这台的 400 W cap（log1001 那台是 450 W）在两个引擎上都触发，但在 **vLLM 上多出一半**，它跑得高约
40 W、低 75 MHz。原因在 §3.3 看得见：qwen.py 有 5–86 % 的时间让 GPU 空着，根本抽不出那么多功耗。
方向很重要：**450 W 的宿主上 vLLM 赚得比 qwen.py 多，所以本报告每一个差距数字都是下界而不是上界。**
全部 run 无热降频。

---

## 4. 差距来自哪里

汇总数据能支持的结论，除注明外均在 batch 512：

**1. compile + CUDA graph —— 低 batch 差距的全部，高 batch 差距里的 1.17×。** B/A 随设备接手而
7.28× → 1.17× → 0.98×。qwen.py 里没有任何东西与之对应。

**2. qwen.py 的主机循环成本约为 vLLM eager 主机循环的 1.6×。** 在 batch ≤ 64 上平坦，而且是一个可跨
宿主迁移的比值（§2.4）。这和 log1001 §7.3 从内部量到的是同一个量：约 4 400 次 `aten::` 调用、
约 4.8 ms/步无归属的 dispatch。

**3. 设备侧比头条数字显示的更接近，但不相等。** 臂 A 只在 sweep 顶端才是设备瓶颈，所以最干净的设备侧
读数是 batch 512 的 1.23×。已识别两个贡献者：

* **采样器，26.93 ms/步里的 16.32 ms（61 %）。** vLLM 在参数完全相同的情况下，采样器只占它一步的
  一小部分。§3.3 把剩下这笔修复定价为 qwen.py 设备时间的约 1.8×。
* **forward 距带宽屋顶比 vLLM 远。** batch 1 上 qwen.py 的设备工作是 **3.18 ms/步**（实测
  `gpu_busy_from_trace`），而仅权重就有 0.99 GB —— 按实测 919 GB/s 是 **1.08 ms**。这把 qwen.py 放在
  权重带宽屋顶的约 34 %。vLLM 臂 B 的同一个量没有直接测，但可以定界：15.877 s / 64 请求 =
  248 ms/请求，那是一次 512 token 的 prefill 加 128 个 decode 步，所以**即便 prefill 免费**，decode 步
  也**低于 1.94 ms** —— 即**高于屋顶的 56 %**。vLLM 这一半要当**界**而不是测量值看（§5.5）。其中一部分
  来自 inductor 融合（臂 A 没有），所以这个贡献者也同时坐在因子 1 里面。

**4. 哪些**不是**贡献者，并且已经查过：** attention kernel 家族（都是 `FLASH_ATTN`）、KV 容量（都是
1 048 576 tokens，而且 vLLM 自己更大的分配被 override **压到**了平手）、prefix caching（关）、
抢占（两边都是零）、EOS 处理（两边都断言了恰好 128 token 的输出）、以及 GPU 温度。

---

## 5. 对有效性的威胁

### 5.1 小 batch 的绝对数字是宿主特定的

见 §2.4。这台宿主上 batch ≤ 256 的吞吐大约是同一份 qwen.py 代码在更快解释器上的 0.42–0.72×。§3.2 的
比值列才是可迁移的部分；batch ≤ 256 的绝对值列不是。今后任何把 log1003 的 vLLM 数字接到 log1001 的
qwen.py 数字上的做法，正是 [log1001 §8](./performance_analysis_log1001.zh.md) 警告的那个错误。

### 5.2 臂 A→B 把编译和图捕获混在了一起

见 §2.1。`enforce_eager=True` 同时关掉两者，所以本报告说不出 batch 1 上那 7.28× 里有多少是 inductor
融合、多少是消除 launch。§3.2 的分解因此是*引擎设计* × *(compile + graph)*，第二个因子是一个打包项。

### 5.3 vLLM 臂 B 的 batch 512 点是本报告最松的数字

三次 run 散布 11.6 %（§2.3），而且慢的那次在最前面 —— 预热漏过了 warmup 调用。把 15 848 tok/s 当作
±6 % 看。那里 1.44× 的差距绰绰有余；但假设有人想在那个 batch 上声称 1.1×，这批数据撑不住。

### 5.4 对齐了但并不理想的几项

* **`block_size` 256 对 16。** vLLM 没有 256，所以改为对齐 KV 总容量。block table 的遍历和碎片行为
  不同。
* **两边都是 `OMP_NUM_THREADS=8`**，但那是从 `~/.bashrc` 继承的，不是选出来的。它满足 §6e"两边都加或
  都不加"的规则，但是靠巧合而非决定；runbook 现在已经把这点写明。
* **vLLM V1 的 EngineCore 跑在独立进程**，所以它多占一个核并使用 `/dev/shm`。那是 vLLM 的出厂形态，
  qwen.py 没有对应物，但它是一个不对称。
* **两边都没做 CPU 绑定。** 48 个 CPU 可见、1 个 NUMA node、**8 个独立 L3 实例**，那个唯一的热 Python
  线程全程可以在 CCX 之间迁移。这伤两个引擎，而且最伤 eager 路径。

### 5.5 没有测的

* **vLLM 的 TTFT / prefill 延迟。** 工具只记了 elapsed 和输出 token 数，所以本报告没有延迟对比，
  只有吞吐。qwen.py 自己的 prefill 延迟在 `concurrency_sweep.txt` 里（batch 1 的 23.8 ms 到 512 的
  137.5 ms），但没有对照物。
* **vLLM 的内部单步拆解。** 这里没有 vLLM 版的 `step_metrics`，所以 §4 第 3 条的设备侧归因是从总量和
  roofline 推出来的，不是逐算子测出来的。
* **采样输出质量。** 参数对齐了，分布没有对比。对吞吐对比足够，也没有超出这个范围声称什么。

---

## 6. 建议，按优先级

### 6.1 不要为了小 batch 的数字再租机器 —— 先修采样器

可迁移的结果说：batch 512 上 qwen.py 在 vLLM 臂 A 的 81 %、臂 B 的 69 %，而已识别的采样器工作
（log1001 §7.1–7.2）在那里值约 1.8× 的设备时间 —— **比整个差距还大**。把下一笔租金花在更快的宿主上去
重测一个已经被定界的东西，收益低于把下一个会话花在 `sample()` 上。

### 6.2 然后加编译 / 图化路径，因为那是另一半

A→B 在 batch 1 上是 7.28×、在 512 上是 1.17×，而两条臂其余部分完全相同。qwen.py 的 decode 步形状是
静态的（固定 `max_num_seqs` 槽位、常驻参数表、热路径无动态控制流），这正是图捕获最容易的情形。
顺序要排在 6.1 **之后**：把一个有 61 % 设备时间花在可避免采样器上的步捕成图，等于把那笔浪费烤死。

### 6.3 在能过新 §1b 闸门的宿主上重测

6.1 和 6.2 落地后，这个对比需要一台 3 M 次 Python 循环接近参考值而不是 276 ms 的宿主，这样 batch ≤ 256
的绝对数字才能从比值变成测量值。优先挑高主频的消费级 CPU（Ryzen 7000/9000）而不是共享的 EPYC，并绑到
一个 CCX（见 [runbook §3](../env/vastai/vast-evn-build.md)）。

### 6.4 加第三条 vLLM 臂，把编译和图捕获拆开

臂 A′：编译开、`cudagraph_mode=NONE`。A→A′ 是 inductor 融合，A′→B 是消除 launch。这能告诉你 6.2 的
两半先做哪一半，而且在已经建好的 venv 上只是多跑一遍 sweep。

### 6.5 在 vLLM 侧记下 TTFT

`vllm_sweep.py` 目前只留 elapsed 和 token 数。vLLM 的 `RequestOutput` 带有每请求的时间戳；把 TTFT 和
ITL 分位加进 JSONL 行，下一份报告就能同时是延迟对比，而且不增加运行时间。

---

## 7. 注意事项

1. **一台宿主、一个时间窗、一次一个引擎** —— 已满足，这也正是 log1003 内部对比成立的原因。
   **不要**把这些数字和 log1001 或 log928 的拼接。
2. **vLLM 跑在前面**，机器更冷。任何预热偏置都偏向 vLLM，所以差距数字在这个方向上也是保守的
   （功耗 cap 的偏置同理，§3.4）。
3. **qwen.py 两轮；vLLM 每点三次。** §6e 要求 ≥ 3。qwen.py 两轮最差一致到 2.87 %，所以这个欠缺不实质，
   但它是个欠缺。
4. **被测代码是 `49bb10d` 之后一个提交**，而且 run 进行时它还未提交；版本归属是事后从 dump 重建的
   （config 行里的 `frequency_penalty`、`benchmark_metrics` 里的 `o_tok_throughput`），现在该提交已经
   存在。今后应先提交再跑。
5. **`vllm_pip_freeze.txt` 采了两次。** 第一次静默记成了**系统** Python（155 行 jupyter/autobahn，
   没有 `vllm`、没有 `torch`），因为那个 shell 里 `$VLLM_VENV` 是空的，于是 `"$VLLM_VENV/bin/pip"`
   解析成了 `/bin/pip`。log 目录里的是修正后的采集：198 行、`vllm==0.30.0`、`torch==2.13.0`。真正重要
   的版本事实在每个 JSONL header 里也都有（vllm 0.30.0、torch 2.13.0+cu130、CUDA 13.0、
   Python 3.12.3）—— 那就是 header 的用途。
6. **`log_vast/` 在 `.gitignore` 里**（自 log910 起的仓库约定），所以这些 dump 只存在于分析机上。
   对一个 log 目录执行 `git add` 会静默地什么都不做。
7. **vLLM 0.30.0 拖的是 cu13 的 torch，因此要求 driver ≥ 580。** 同一条
   `pip install vllm==0.30.0` 在 log1001 那台宿主（driver 550 / CUDA 12.4）上会失败。记下这一点，
   因为它让本报告的 vLLM 一侧在旧机器上不可复现。
