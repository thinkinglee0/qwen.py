from dataclasses import dataclass, field, asdict
import numpy as np
import logging
import time
import math
import orjson
import torch

from qwen.utils import round_floats
from qwen.config import ModelConfig

logger = logging.getLogger(__name__)

class StepEvents:
    """CUDA events for one step. record() is async; read() must run after a sync."""

    SEGMENTS = ("fwd", "logits", "sample")

    def __init__(self):
        if not torch.cuda.is_available():
            return

        # enable_timing=True is required for elapsed_time(); it costs nothing extra.
        self._ev = {s: (torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True))
                    for s in self.SEGMENTS}

    def start(self, seg: str):
        if not torch.cuda.is_available():
            return

        assert seg in self.SEGMENTS, f"invalid segment: {seg}"
        self._ev[seg][0].record()  # type: ignore[call-arg]

    def stop(self, seg: str):
        if not torch.cuda.is_available():
            return

        assert seg in self.SEGMENTS, f"invalid segment: {seg}"
        self._ev[seg][1].record()  # type: ignore[call-arg]

    def read(self) -> dict[str, float]:
        if not torch.cuda.is_available():
            return {}

        """Call ONLY after the stream has been drained -- otherwise this syncs."""
        return {f"{s}_gpu_ms": a.elapsed_time(b) for s, (a, b) in self._ev.items()}

@dataclass
class SchedulerStepMetrices:
    step: int = 0       # step_id
    bz: int = 0         # batch_size
    n_p: int = 0        # number_prefill_tokens
    n_d: int = 0        # number_decode_tokens
    run: int = 0        # num_running
    wait: int = 0       # num_waiting
    blk_used: int = 0   # kv_blocks_used
    sched_ms: float = 0.
    sched_pre_ms: float = 0.
    sched_run_ms: float = 0.
    sched_wait_ms: float = 0.
    sched_ret_ms: float = 0.
    bld_meta_ms: float = 0.
    fwd_ms: float = 0.
    fwd_gpu_ms: float = 0.
    # fwd_embed_ms: float = 0.
    # fwd_layers_ms: list[float] = []
    # fwd_post_norm_ms: float = 0.
    logits_ms: float = 0.
    logits_gpu_ms: float = 0.
    sample_ms: float = 0.
    sample_gpu_ms: float = 0.
    n_sample: int = 0
    dth_ms: float = 0.
    ci_ms: float = 0.

@dataclass
class SchedulerMetrices:
    step: int = 0               # step_id
    num_cache_exhausted: int = 0
    num_preempted: int = 0      # number of preempted reqeusts
    num_scheduled: int = 0      # number of scheduled reqeusts
    num_rescheduled: int = 0    # number of rescheduled requests
    num_finished: int = 0       # number of finished reqeusts
    num_error: int = 0
    num_truncated: int = 0

    def report_on_schedule(self, scheduled_reqs):   # scheduled_reqs: list[ModelRequest]
        self.num_scheduled += len(scheduled_reqs)

        for req in scheduled_reqs:
            if req.metrics.first_schedule_time is not None and req.num_computed_tokens == 0:
                self.num_rescheduled += 1

    def report_on_cache_exhausted(self):
        self.num_cache_exhausted += 1

    def report_on_preemption(self):
        self.num_preempted += 1

    def report_on_finish(self):
        self.num_finished += 1

    def report_on_error(self):
        self.num_error += 1

    def report_on_truncated(self, num_truncated:int):
        self.num_truncated += num_truncated


@dataclass
class RequestMetrics:
    num_input_token: int = 0
    num_output_token: int = 0
    num_prefill_chunk: int = 0
    arrival_time: float | None = None
    first_schedule_time: float | None = None
    first_token_time: float | None = None
    last_token_time: float | None = None
    itls: list[float] = field(default_factory=list)

    def report(self, now: float):
        if self.last_token_time is None:    # first token
            self.first_token_time = now
        else:
            self.itls.append(now - self.last_token_time)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"last_token_time: {self.last_token_time}, now: {now}, itl: {now-self.last_token_time}")

        self.last_token_time = now  # update everytime
        self.num_output_token += 1


def summarize(samples: list[float], scale: float = 1e3) -> dict[str, float]:
    """Latency metrics. scale converts seconds to ms."""
    if not samples:
        return {"n": 0}
    a = np.asarray(samples, dtype=np.float64) * scale     # float64: sums stay exact
    p50, p90, p99 = np.percentile(a, [50, 90, 99])        # one sort, three cuts
    return {
        "n":    len(a),
        "mean": float(a.mean()),
        "std":  float(a.std(ddof=1)),                     # sample std
        "p50":  float(p50),
        "p90":  float(p90),
        "p99":  float(p99),
        "max":  float(a.max()),                           # report alongside p99
    }


def analyze_metrics(req_metrics_list: list[RequestMetrics], sch_metrics: SchedulerMetrices, is_benchmarking:bool, config: ModelConfig) -> bytes:
    req_cnt = len(req_metrics_list)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"req_cnt:{req_cnt}, req_metrics_list: {req_metrics_list}")
    else:
        logger.info(f"req_cnt:{req_cnt}")

    queueing, prefill, ttft, tpot, itls, prefill_chunk = [], [], [], [], [], []
    num_input_token, num_output_token = 0, 0
    start_time, finish_time = time.perf_counter(), 0.
    for metrics in req_metrics_list:
        assert metrics.arrival_time and metrics.first_schedule_time

        num_input_token += metrics.num_input_token
        num_output_token += metrics.num_output_token
        start_time = min(start_time, metrics.first_schedule_time if is_benchmarking else metrics.arrival_time)

        queueing.append(metrics.first_schedule_time-metrics.arrival_time)
        if metrics.first_token_time is not None:    # check for zero token
            ttft.append(metrics.first_token_time-metrics.arrival_time)
            prefill.append(metrics.first_token_time-metrics.first_schedule_time)
            prefill_chunk.append(metrics.num_prefill_chunk)

            assert metrics.last_token_time is not None
            finish_time = max(finish_time, metrics.last_token_time)
            math.isclose(metrics.last_token_time-metrics.first_token_time, sum(metrics.itls), rel_tol=1e-9)
            tpot.append((metrics.last_token_time-metrics.first_token_time)/len(metrics.itls)) if metrics.itls else None     # for only one single token scenario

        itls.extend(metrics.itls)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"itls: {len(metrics.itls)}/{len(itls)}, {metrics.itls}")

    # mean, median(p50), std, p90, p99
    elapsed = finish_time - start_time
    req_num = len(req_metrics_list)
    local_wall_time = time.localtime(time.time())
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", local_wall_time)
    metrics_dict = {
        "basic": {
            "time": formatted_time,
            "max_num_seqs": config.max_num_seqs,
            "max_num_batched_tokens": config.max_num_batched_tokens,
            "long_prefill_token_threshold": config.long_prefill_token_threshold,
            "req_num": req_num,
            "elapsed": elapsed,
            "i_tok_num": num_input_token,
            "o_tok_num": num_output_token,
            "tok_throughput": num_output_token / elapsed,
            "req_throughput": req_num / elapsed,
            "scheduler": asdict(sch_metrics),
        },
        
        # default scale=1e3, unit changes from s to ms.
        "queueing": summarize(queueing),
        "prefill": summarize(prefill),
        "prefill_chunk": summarize(prefill_chunk, scale=1),
        "ttft": summarize(ttft),
        "tpot": summarize(tpot),
        "itls": summarize(itls),
    }

    json_bytes = orjson.dumps(round_floats(metrics_dict, nd=3))
    return json_bytes