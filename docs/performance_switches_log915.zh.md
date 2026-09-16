# 优化开关 —— RTX 4090 上的单因子测量（log915）

> 英文原版：[`performance_switches_log915.md`](./performance_switches_log915.md)

本报告取代 [`performance_switches_log914.md`](./performance_switches_log914.md) —— 那一轮只有一个
组合配置，per-switch 的效果只能靠推断。这一轮**每次只翻一个开关**。

| 开关 | GPU kernel 时间 | 步墙上时间 | 结论 |
| --- | --- | --- | --- |
| `pre_gather_cos_sin` | **−0.3 ~ −4.2 %**（五个 batch 全部 clear） | **−1.6 ~ −10.9 %**（3 个 clear，五个同号） | **打开它。** 三轮下来唯一一个 host 和 device 两边都省的开关。 |
| `compile_rope` | −1.0 ~ −12.3 %（全部 clear） | **+1.2 ~ +4.7 %**（3 个 clear，五个同号） | **保持关闭。** 第三次测量，第三次拿 GPU 时间换来更长的墙上时间。 |
| `stage_sampling_params` | ±0.1 % | −0.1 ~ −1.8 %（五个同号，但从不 clear） | **仍在分辨率之下。** 方向一致为负，但始终没超出噪声。 |

**数据来源**

| 配置 | `compile_rope` | `pre_gather_cos_sin` | `stage_sampling_params` | 路径 |
| --- | :-: | :-: | :-: | --- |
| baseline | false | false | false | `log_vast/log915/profile_baseline{,2}/` |
| compile_rope | **true** | false | false | `…/profile_compile_rope{,2}/` |
| pre_gather | false | **true** | false | `…/profile_pre_gather_cos_sin{,2}/` |
| stage_sampling | false | false | **true** | `…/profile_stage_sampling{,2}/` |
| sweep（仅 baseline） | false | false | false | `…/benchmark_baseline{,2}/` |

每个配置跑**两遍**，五个 batch（1、8、32、128、512），每点 **20 个干净步** —— 是 log914 窗口的四倍。
sweep 跑两遍，覆盖 11 个 batch。

log914 之后引擎有两处变化：`pre_gather_cos_sin` 现在真的被
[`attention.py:134`](../src/qwen/attention.py#L134) 读取了（log914 时它只是被声明、从未被使用），
以及 `make_sampling_tensor_strategy: int` 改名为 `stage_sampling_params: bool`。

**仍然缺失**：sweep 只跑了 baseline，所以**任何开关都没有吞吐数据**。§3 的全部内容都是每步量级，
来自 profile harness。

---

## 1. 归约数据时发现的两个测量问题

两个都是把 profile harness 和 sweep 对照时发现的 —— 它们同一台主机、同一小时内跑的。
**这两件事比下面任何一个开关都重要。**

### 1.1 `sync_debug_mode="warn"` 是免费的 —— 假设已验证并否定

[`test_profile.py:169`](../tests/test_profile.py#L169) 在整个干净测量窗口里打开了
`torch.cuda.set_sync_debug_mode("warn")`。直觉上的担心是：它给每次 CUDA 调用加税，从而污染了这个
窗口本该产出的数字。

**并没有。** 对比最后 20 个 **warmup** 步（sync debug 关）和 20 个 **measure** 步（sync debug 开）
—— 同一个 engine、同一次 run，其他条件完全相同：

| batch | 1 | 8 | 32 | 128 | 512 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `rope` | −0.7 % | −0.6 % | +0.1 % | −1.6 % | −0.9 % |
| `fwd` | −1.2 % | −0.3 % | +0.2 % | −1.3 % | −1.3 % |
| `step` | −0.9 % | −0.2 % | +0.7 % | −0.1 % | +0.3 % |

全在噪声内，正负都有。那行代码可以留着。

### 1.2 profiler 在进程里留下约 30 % 的 host 开销

这个才是真问题。八次 profile session 的 host 侧 `rope`，**按运行顺序**排列：

| session | bs 1 | bs 8 | bs 32 | bs 128 | bs 512 |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | **6.27** | 8.12 | 8.21 | 8.27 | 8.27 |
| baseline 2 | **6.80** | 8.19 | 8.52 | 8.20 | 8.25 |
| compile_rope | **7.90** | 8.94 | 9.10 | 9.23 | 9.30 |
| compile_rope 2 | **8.12** | 8.95 | 8.97 | 9.12 | 9.27 |
| pre_gather | **4.83** | 6.26 | 6.52 | 6.40 | 6.32 |
| pre_gather 2 | **4.93** | 6.41 | 6.32 | 6.35 | 6.41 |
| stage_sampling | **6.42** | 8.09 | 8.07 | 8.16 | 8.49 |
| stage_sampling 2 | **6.42** | 8.22 | 8.18 | 8.09 | 8.18 |

**每一次** session 的第一个 batch 点都比后面所有点便宜 20~30 %，之后持平。`rope` 的 host 成本与
batch 无关（log914 §3.2），所以这不是 batch 效应 —— 而 sweep harness 直接证明了这一点。同一
session 里建了 11 次 engine，从不 profile：

```
bs    1    2    4    8   16   32   64  128  256  512 1024
rope 6.11 6.10 6.12 6.15 6.17 6.24 6.24 6.24 6.26 6.29 6.36
```

平的。所以不是 engine 重建、不是 allocator 状态、也不是 batch 大小。profile harness 的第一个点和
第二个点之间唯一发生的事，是 **`torch.profiler` 被进入并退出过一次**，它在进程里留下的某些东西给
之后每一个 ATen op 的 host 侧加了税。（kineto 首次使用注册的全局 observer callback 是最明显的嫌疑，
但下面的实验确定的是这个状态的**作用域**，不是它具体是哪一块。）

**后果：**

* profile harness 的 host 侧**绝对值**在除第一个点之外的所有点上都被抬高约 30 %，`wall_clean_us`
  也跟着抬高 —— 因此**这个 harness 报出的 `gpu_idle_fraction` 在 batch 1 之外全部偏高**。
  device 侧的 kernel 时间不受影响（复现到 0.1 %）。
* profile 的数字**不能**和 sweep 的数字对比。同一个 engine 在 batch 512 上，profile harness 读
  `fwd` 28.57 ms，sweep 读 22.48 ms。
* §3 的开关结论**成立**，因为污染是共模的：每个配置在自己的进程里、以相同的 batch 顺序运行，
  bs 8~512 在四个配置里被同等加税。**相对**差值有效；绝对毫秒数被同样放大了约 30 %。
* 整份数据里最干净的是 **batch-1 那一列**，它没有被污染。

**修法**：一个 batch 一个 pytest 进程；或者把干净窗口放在该进程里任何 profiler 执行之前。在此之前，
这个 harness 只能用来读 kernel 时间和 A/B 比值，不能读绝对步时。

### 1.3 进程隔离实验的验证（log916）

这个假设给出的预言很锐利：让每个 batch 独占一个 pytest 进程，那么**每个点都成了第一个点**，膨胀应该
完全消失。按

```bash
for bz in 1 8 32 128 512; do SWEEP_PROFILE_BATCH_SIZES=$bz pytest -x -s \
  tests/test_profile.py::test_profile_decode_idle_fraction \
  --compile-rope=False --stage-sampling-params=false --pre-gather-cos-sin=false; done
```

跑出来的结果（`log_vast/log916/profile_baseline{,2}`）正是如此 —— 不只是暴露问题的那个字段，整个
步时间都对上了：

| | bs 1 | bs 8 | bs 32 | bs 128 | bs 512 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `rope`，log915 —— 一个进程跑五个 batch | 6.27 | **8.12** | **8.21** | **8.27** | **8.27** |
| `rope`，log916 —— 一个 batch 一个进程 | 6.10 | 6.15 | 6.35 | 6.52 | 6.22 |
| `rope`，log916 —— 一个 batch 一个进程，第 2 轮 | 6.12 | 6.27 | 6.01 | 5.93 | 6.20 |
| `rope`，sweep harness（从不 profile） | 6.11 | 6.15 | 6.24 | 6.24 | 6.29 |
| **step**，log915 —— 一个进程 | 23.19 | **31.18** | **34.76** | **48.74** | **104.82** |
| **step**，log916 —— 一个 batch 一个进程 | 23.21 | 24.82 | 28.95 | 44.09 | 99.73 |
| **step**，log916 —— 第 2 轮 | 22.72 | 24.96 | 27.09 | 40.87 | 99.92 |
| **step**，sweep harness 的 decode 步 | 22.69 | 24.55 | 27.73 | 42.47 | 100.17 |

两个结论：

* **膨胀完全消失** —— log915 在 batch 8 上最高 +28 %，log916 里一点不剩。
* **两个 harness 现在对上了。** 它们本来就没有理由对不上：batch 512 时隔离后的 profile harness 报
  `wall_clean_us` = 100.218 ms，而 sweep 的 decode 步是 100.17 ms —— **相差 0.05 %**，此前是 5 %。

报出的空闲率随之变化：batch 512 上**隔离后 50.2 %，污染时 52.3 %**（§1.2 预估约 49.8 %）。所有从
多 batch session 里得到的 `gpu_idle_fraction`，都应该往下读约两个百分点。

> **上面那个循环的一个坑：** `pyproject.toml` 里的 `log_file` 是截断模式打开的，所以每次迭代都会覆盖
> `log/pytest.log`，只有最后一个 batch 的 session 记录活了下来 —— batch 1~128 的空闲率丢失了。把
> `log_file_mode = "a"` 打开（文件里已有该行，被注释掉了），或者给每次迭代单独的 `--log-dir`。

---

## 2. 方法

每个点取干净（未 profile）窗口里的 20 个纯 decode 步，即每份 dump 的 `rows[-25:-5]`，两次 run。
只有当差值的幅度**超过两个配置各自的 run-to-run 离散度**时，才标记为 **clear**。

kernel 时间取 chrome trace 里 `kernel`/`gpu_memcpy`/`gpu_memset` 各段的并集，run-to-run 复现到
约 0.1 % —— 所有结论都建立在这个量具上。

---

## 3. 结果 —— 每个开关对同一基线

### 3.1 GPU kernel 时间

| batch | compile_rope | pre_gather | stage_sampling |
| ---: | ---: | ---: | ---: |
| 1 | **−12.3 %** clear | **−4.2 %** clear | −0.1 % |
| 8 | **−10.7 %** clear | **−2.4 %** clear | +0.0 % |
| 32 | **−7.4 %** clear | **−2.0 %** clear | +0.0 % |
| 128 | **−3.3 %** clear | **−0.7 %** clear | −0.0 % |
| 512 | **−1.0 %** clear | **−0.3 %** clear | −0.0 % |

两个 rope 开关都确实减少了设备侧的工作。**绝对值上各自省下的是常数** —— compile 约 0.4 ms、
pre-gather 约 0.15 ms 每步；百分比随 batch 缩小，只是因为采样把分母撑大了。

### 3.2 步墙上时间

| batch | compile_rope | pre_gather | stage_sampling |
| ---: | ---: | ---: | ---: |
| 1 | +4.7 % | **−10.9 %** | −1.8 % |
| 8 | **+3.2 %** clear | **−7.0 %** clear | −0.2 % |
| 32 | +2.6 % | −2.8 % | −0.9 % |
| 128 | **+3.1 %** clear | **−3.8 %** clear | −1.1 % |
| 512 | **+1.2 %** clear | **−1.6 %** clear | −0.1 % |

### 3.3 host 侧 `rope` 与 `fwd`

| batch | `rope` compile | `rope` pre_gather | `fwd` compile | `fwd` pre_gather |
| ---: | ---: | ---: | ---: | ---: |
| 1 | +22.6 % | **−25.3 %** | +5.1 % | −11.5 % |
| 8 | +9.7 % | **−22.3 %** | +3.4 % | −8.0 % |
| 32 | +7.9 % | **−23.3 %** | +3.3 % | −5.5 % |
| 128 | +11.4 % | **−22.6 %** | +4.3 % | −7.0 % |
| 512 | +12.4 % | **−23.0 %** | +4.8 % | −7.4 % |

除 `fwd`/compile 在 batch 1 和 32 之外全部 clear。`sample` 对任何开关都纹丝不动，符合预期。

绝对值，batch 512（ms，两次 run 的均值 —— 按 §1.2 整体抬高约 30 %，可用的是比值）：

| | baseline | compile_rope | pre_gather | stage_sampling |
| --- | ---: | ---: | ---: | ---: |
| step | 104.57 | 105.84 | **102.91** | 104.42 |
| `fwd` | 28.57 | 29.95 | **26.46** | 28.72 |
| `rope` | 8.26 | 9.28 | **6.37** | 8.34 |
| `bld_meta` | 1.16 | 1.18 | 1.26 | 1.19 |
| GPU kernel | 50.31 | 49.79 | 50.17 | 50.30 |

---

## 4. 逐开关分析

### 4.1 `pre_gather_cos_sin` —— 三轮下来第一个真正值得打开的开关

它把 24 层里的 23 次 `cos_cached[position_ids]` gather 去掉，改为在
[`build_attn_metadata`](../src/qwen/attention.py#L134) 里做一次，各层走 `rope.forward2`。
**账本两边都在改善**：

* **host 侧** —— `rope` 在每个 batch 上都降 22~25 %（batch 512：8.26 → 6.37 ms），`fwd` 随之下降。
  剩下的那一次 gather 体现为 `bld_meta` 涨 0.10 ms —— 它本来就该在那里。
* **device 侧** —— 少了 23 个 gather kernel，kernel 时间减少约 0.15 ms。

净效果是步时间 −1.6 % ~ −10.9 %。**这才是真正的优化长的样子**：两边的工作都变少，没有交换。

想知道诚实的幅度，看 **batch-1 那一列** —— 它是唯一未被污染的点（§1.2）：`rope` 6.53 → 4.88 ms，
步时间 −10.9 %。

**`config.py` 的默认值本来就是 `True`。** 是这几轮的运行脚本显式把它关掉的（log914 也一样），因为
`tests/conftest.py` 会把 CLI 选项透传下去。**引擎不用改，改的是 sweep 的调用方式。**

### 4.2 `compile_rope` —— 测了三次，三次都是负的

| | log910 | log914 | log915 |
| --- | --- | --- | --- |
| GPU kernel 时间 | −13 % | −12.7 % | **−12.3 %**（batch 1） |
| 步墙上时间 | 更慢 | +2.7 ~ +14.0 % | **+1.2 ~ +4.7 %** |
| `do_sample` | false | true | true |
| 是否单因子隔离 | 是 | 否（和别的开关捆绑） | **是** |

设备侧的收益是真实且可复现的：`torch.compile` 把那条五个算子的 pointwise 链融合掉，每步从 kernel
时间里拿走约 0.4 ms，每个 batch 都如此。但为了够到这个 kernel，host 侧付出的更多：guard 检查和编译
后的 wrapper **每步要跑 48 次**（24 层 × q 和 k），`rope` 的 host 时间涨 8~23 %。

在一个每步有 50~85 % 时间处于空闲的引擎里，这笔交易只可能是亏的。**要重测，得等 decode 路径
CUDA-graph 化之后** —— 那时发射成本被摊薄，符号应该会翻转。

另外它和 §4.1 叠加得很糟：`compile_rope` 让 `rope` 涨约 1.0 ms，而 `pre_gather` 让它降约 1.9 ms，
两者动的是同一批 48 次调用。

### 4.3 `stage_sampling_params` —— 方向一致，幅度不可见

五个 batch 的步时间**全部为负**（−0.1 ~ −1.8 %），kernel 时间在四位小数上完全相同 —— 这正是"把六次
pageable H2D 拷贝换成两次 pinned 异步拷贝"该有的样子。但即便窗口扩大到 20 步，**没有任何一个差值
超出 run-to-run 离散度**。

这是关于**实验**的陈述，不是关于开关的：那些张量是 `[5, batch]` 和 `[batch]` 的浮点数 —— batch 512
时也就 10 KB —— 所以收益的上限是几百微秒，而分母是约 100 ms 的步。log910 §5.2 已经指出真正昂贵的
拷贝是哪些（`bin_counts_and_mask` 每步用 Python 重建的 `flat_idx` 列表，batch 512 时约 1 MB 的
int64），而这个开关碰不到它们。

**要定论**：只翻这一个开关跑一次 sweep —— 几千个步，而不是四十个。值得做，但排在
`bin_counts_and_mask` 修好之后。

---

## 5. 新的基线 sweep

两次 run，11 个 batch，`pre_gather_cos_sin=false`，`use_d_first_schedule=false`。

| batch | tok/s | Δ run 2 | step ms | `fwd_gpu` | `sample_gpu` | sample 占步 | sample 占整轮 | TPOT ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 43.6 | +0.30 % | 22.69 | 20.50 | 1.10 | 4.8 % | 4.8 % | 22.9 |
| 8 | 322.1 | +0.26 % | 24.55 | 21.75 | 1.68 | 6.8 % | 6.8 % | 24.8 |
| 32 | 1120.6 | −0.40 % | 27.73 | 22.54 | 3.92 | 14.2 % | 13.8 % | 28.2 |
| 128 | 2834.5 | +0.96 % | 42.47 | 22.72 | 17.90 | 42.1 % | 38.9 % | 44.6 |
| 256 | 3801.3 | +0.36 % | 62.41 | 22.60 | 37.18 | 59.6 % | 52.7 % | 66.5 |
| 512 | 4681.7 | +0.50 % | 100.17 | 22.47 | 73.24 | 73.1 % | 60.4 % | 107.5 |
| 1024 | 5094.6 | +0.47 % | 171.31 | 23.47 | 139.79 | 81.6 % | 58.6 % | 195.3 |

**复现性大幅改善**：每个点两次 run 都在 **1.21 %** 以内（log914 最差到 5.5 %）。两个结构性发现原样
复现 —— `fwd_gpu` 在 1024 倍的 batch 范围里恒定在 20.5~23.5 ms，采样在 batch 512 占整轮的 60.4 %。

**和 log914 相比有三件事同时变了**，所以两轮 sweep 不能直接对比：

| | log914 | log915 |
| --- | --- | --- |
| `pre_gather_cos_sin` | 声明为 false，但**被代码忽略** —— 实际走的是预取路径 | false，且**生效** —— 实际走的是逐层路径 |
| `use_d_first_schedule` | true | false |
| 主机 | 一台 vast.ai 实例 | 另一台 |
| tok/s @ 1024 | 5300 | 5095（−3.9 %） |
| `rope` @ 512 | 4.31 ms | 6.29 ms（+46 %） |

`rope` 的退步是 flag 变化，不是代码退步：log914 的"baseline"实际上一直开着预取。**log915 里和
log914 baseline 可比的是 `pre_gather` 那个配置**，而 log915 的 baseline 是一个不该拿来跑任何东西的
设置。

---

## 6. 建议

| 优先级 | 动作 |
| :-: | --- |
| 1 | **别再传 `--pre-gather-cos-sin=false`。** 每步白赚 2~11 %，而且 `config.py` 的默认值本来就是 `True`，只有 sweep 的调用把它关掉了。用它重跑基线 sweep —— 现在的基线数字低估了引擎。 |
| 2 | **修掉 profiler 污染**（§1.2）：一个 batch 一个进程。迄今公布的所有 `gpu_idle_fraction`，除每次 session 的第一个 batch 点外都偏高。 |
| 3 | **`compile_rope` 保持关闭**（§4.2），等 CUDA graph 之后再测。 |
| 4 | **`stage_sampling_params` 保持默认**（§4.3），等 `bin_counts_and_mask` 修好、sweep 能分辨它之后再说。 |
| 5 | **下一个开关要跑 sweep，不能只跑 profiler。** 三轮开关测量下来，至今没有任何一个开关拿到过吞吐数字。 |

这些都不改变[基线报告](./performance_analysis_log915.zh.md)里的排序：采样占整轮约 60 %，模型前向
无论喂多少 token 都是约 22 ms。这里最好的开关值约 2 ms。
