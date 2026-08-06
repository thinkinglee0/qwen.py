from dataclasses import dataclass, field
import numpy as np
import logging
import orjson
import time

from qwen.utils import round_floats

logger = logging.getLogger(__name__)


@dataclass
class Metrics:
    input_token_num: int = 0
    output_token_num: int = 0
    arrival_time: float | None = None
    schedule_time: float | None = None
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
        self.output_token_num += 1


def summarize(samples: list[float], scale: float = 1e3) -> dict[str, float]:
    """Latency stats. scale converts seconds to ms."""
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


def analyze_stats(metrics_list: "list[Metrics]") -> bytes:
    req_cnt = len(metrics_list)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"metrics_list, req_cnt:{req_cnt}, {metrics_list}")
    else:
        logger.info(f"metrics_list, req_cnt:{req_cnt}")

    queueing, prefill, ttft, tpot, itls = [], [], [], [], []
    input_token_num, output_token_num = 0, 0
    start_time, finish_time = time.perf_counter(), 0.
    for metrics in metrics_list:
        assert metrics.arrival_time and metrics.schedule_time

        input_token_num += metrics.input_token_num
        output_token_num += metrics.output_token_num
        start_time = min(start_time, metrics.schedule_time)
        
        queueing.append(metrics.schedule_time-metrics.arrival_time)
        if metrics.first_token_time is not None:    # check for zero token
            ttft.append(metrics.first_token_time-metrics.arrival_time)
            prefill.append(metrics.first_token_time-metrics.schedule_time)

            assert metrics.last_token_time is not None
            finish_time = max(finish_time, metrics.last_token_time)
            assert metrics.last_token_time-metrics.first_token_time == sum(metrics.itls)
            tpot.append((metrics.last_token_time-metrics.first_token_time)/len(metrics.itls)) if metrics.itls else None     # for only one single token scenario

        itls.extend(metrics.itls)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"itls: {len(metrics.itls)}/{len(itls)}, {metrics.itls}")

    # mean, median(p50), std, p90, p99
    elapsed = finish_time - start_time
    req_num = len(metrics_list)
    stats = {
        "basic": {
            "req_num": req_num,
            "elapsed": elapsed,
            "i_tok_num": input_token_num,
            "o_tok_num": output_token_num,
            "throughput": (input_token_num+output_token_num) / elapsed,
        },
        "queueing": summarize(queueing),  # default scale=1e3, unit changes from s to ms.
        "prefill": summarize(prefill),
        "ttft": summarize(ttft),
        "tpot": summarize(tpot),
        "itls": summarize(itls),
    }

    return orjson.dumps(round_floats(stats, nd=3))