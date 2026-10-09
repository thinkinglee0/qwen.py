#!/usr/bin/env bash
# Host sanity check for a Vast.ai GPU instance (Docker container on a shared host).
# Run it on an IDLE instance: several checks assume your own load is zero.

# =============================================================================
# 1. Disk
# =============================================================================
# What matters is the writable layer at / plus any volume you actually write to
# (model cache, /workspace). The /dev/nvme* lines mounted on /etc/hosts or
# /usr/bin/nvidia-smi are single HOST files bind-mounted in; you cannot store
# data there.
# Caveat: overlay's df size equals your allocation only if the host enforces a
# quota (xfs project quota / overlay2 size); otherwise it is the host's backing fs.
df -h /
df -h "${HF_HOME:-$HOME/.cache/huggingface}" 2>/dev/null
mount | grep -vE "^(proc|sysfs|cgroup|tmpfs|devpts|mqueue|overlay)" | grep -vE " /(etc|usr|proc|sys|dev)/"

# =============================================================================
# 2. Shared memory
# =============================================================================
# --- /dev/shm. Docker defaults it to 64 MB.
# TP=1: vLLM V1 runs the model inside EngineCore (UniProcExecutor); frontend <->
#   EngineCore uses ZMQ ipc sockets, not /dev/shm. 64 MB is fine.
# TP>1: two independent consumers, each alone exceeding 64 MB:
#   1. vLLM's MessageQueue (scheduler -> workers broadcast): a /dev/shm ring of
#      max_chunks x max_chunk_bytes = 10 x 24 MiB = 240 MiB with default ctor args.
#      Recent vLLM checks free space first and raises "Insufficient space in
#      /dev/shm ..."; older versions fail later with SIGBUS ("Bus error") because
#      SharedMemory pages are allocated lazily on first write.
#   2. NCCL SHM transport. RTX 4090 has no GPU P2P, so intra-node all-reduce goes
#      through host memory; too little shm -> "unhandled system error"
#      (NCCL_DEBUG=INFO names the failed /dev/shm/nccl-* allocation).
# Fix: --ipc=host or a multi-GB --shm-size (vLLM docs), set at instance creation;
# it cannot be changed from inside (remount needs CAP_SYS_ADMIN).
# Read Avail, not Size: crashed runs leave segments behind. tmpfs pages are
# charged to the container's memory cgroup, so a big shm-size does not help if
# memory.max is tight. With --ipc=host, /dev/shm is the host's, shared with others.
df -h /dev/shm
ls -la /dev/shm | head -20
cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes

# =============================================================================
# 3. GPU identity, power, PCIe link
# =============================================================================
# Do NOT trust the label: A10 (150 W) and A10G (300 W) are both listed as "A10"
# on some platforms.
nvidia-smi --query-gpu=name,power.default_limit,power.max_limit,enforced.power.limit,pcie.link.gen.current,pcie.link.width.current --format=csv
#   expect (4090): NVIDIA GeForce RTX 4090, 450.00 W, <450-600 W, VBIOS-dependent>, 450.00 W, 4, 16
#   - Link gen drops to 1 at idle: re-read it under load.
#   - Width < 16 means a riser (common on ex-mining hosts): slow weight loading
#     and KV/weight offload, little effect on steady-state decode.

nvidia-smi -q -d POWER | grep -E "Current Power Limit|Default Power Limit|Min Power Limit|Max Power Limit"
#   The settable range is [Min, Max] above (often 150-600 W on a 4090, board-
#   dependent), but see vast-evn-build.md section 6: setting it fails with
#   "Insufficient Permissions" in an unprivileged container. Check caps: grep Cap /proc/self/status

# =============================================================================
# 4. CPU: what matters and static config
# =============================================================================
# Our engine is eager mode (no CUDA graphs), so at small batch the per-step cost
# is Python/aten dispatch on ONE thread: single-thread speed (clock, IPC, and
# L2/L3 latency -- CPython is pointer-chasing) matters more than core count.
# Cores still need a floor (engine + driver threads; vLLM V1 runs an API-server
# process, which also detokenizes, plus an EngineCore process); past that, more
# cores buy nothing. At large batch or with CUDA graphs the step is GPU
# memory-bound and the CPU matters much less.
echo "nproc (usable) = $(nproc)   nproc --all (host) = $(nproc --all)"
lscpu | grep -E "Model name|CPU max MHz|^NUMA node|L3 cache|Thread\(s\) per core"
# "CPU max MHz" is NOT the boost clock under acpi-cpufreq: it reports the highest
# P-state, i.e. base (EPYC 7402: 2800 = base, boost 3350). Section 5 measures it.
# Do NOT read lscpu's "scaling MHz" as load: it is cur/max frequency at one
# instant (an idle core parks at ~30 % and ramps under load), and in a VM it is
# often synthetic or absent. Even a real 80 % clock costs AT MOST 1/0.8 = 1.25x
# (the memory-latency share does not scale with clock), not 2x.

# Host-side clock policy. sysfs here is the HOST's, so this is readable even
# though you cannot change it.
C=/sys/devices/system/cpu
grep -H . $C/cpu0/cpufreq/{scaling_driver,scaling_governor,energy_performance_preference} 2>&1
grep -H . $C/cpu0/cpufreq/{cpuinfo_max_freq,scaling_max_freq} 2>&1   # scaling_max < cpuinfo_max => host caps the clock
grep -H . $C/amd_pstate/status $C/cpufreq/boost $C/cpu0/cpufreq/boost $C/intel_pstate/no_turbo 2>/dev/null  # boost off?
for h in /sys/class/hwmon/hwmon*; do                           # k10temp Tctl, millidegree C (Zen 5 Tjmax = 95 C)
  echo "$(cat $h/name 2>/dev/null): $(cat $h/temp1_input 2>/dev/null)"
done

# =============================================================================
# 5. CPU: measured clock vs. scheduled share (separates "throttled" from "preempted")
# =============================================================================
# A chain of dependent reg-reg ADDs runs at exactly 1 add/cycle on modern x86, so
#   cycles / thread_cpu_time  = clock WHILE RUNNING  (low => frequency throttled)
#   thread_cpu_time / wall    = share of time scheduled (low => CFS contention or quota throttling)
# Register operand, not an immediate: some cores (Intel Golden Cove) may fold
# add-immediate chains at rename and run them faster than 1/cycle.
# It uses 1 of 4-6 ALUs, so it does NOT detect a busy SMT sibling (see section 7).
# In a VM, thread CPU time may include hypervisor-stolen time and bias the clock low.
cat > /tmp/freq.c <<'EOF'
#include <stdio.h>
#include <stdint.h>
#include <time.h>
static double clk(clockid_t id) {
    struct timespec t; clock_gettime(id, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
#define A1  "add %0, %0\n\t"
#define A10 A1 A1 A1 A1 A1 A1 A1 A1 A1 A1
int main(void) {
    const long iters = 300000000;               /* 3e9 dependent adds ~= 0.6 s at 5 GHz */
    uint64_t x = 1;
    double w0 = clk(CLOCK_MONOTONIC), c0 = clk(CLOCK_THREAD_CPUTIME_ID);
    for (long i = 0; i < iters; i++)
        __asm__ volatile(A10 : "+r"(x));        /* 10-cycle critical path per iteration */
    double wall = clk(CLOCK_MONOTONIC) - w0;
    double cpu  = clk(CLOCK_THREAD_CPUTIME_ID) - c0;
    double cyc  = iters * 10.0;
    printf("clock_while_running=%.2f GHz  scheduled=%.1f%%\n", cyc / cpu / 1e9, 100 * cpu / wall);
    return (int)(x & 1);
}
EOF
command -v gcc >/dev/null || { apt-get update -qq && apt-get install -y -qq gcc >/dev/null; }
gcc -O2 -o /tmp/freq /tmp/freq.c
MY_CPUS=$(python3 -c "import os; print(*sorted(os.sched_getaffinity(0)))")
echo "single-core:"; for c in $(echo $MY_CPUS | cut -d' ' -f1-3); do taskset -c $c /tmp/freq; done
echo "all-core:";    for c in $MY_CPUS; do (echo "cpu$c $(taskset -c $c /tmp/freq)") & done; wait
#   - single-core far below spec boost (9600X ~5.4 GHz)   => governor/EPP, max-freq cap, boost off, thermal/power wall
#   - all-core much lower than single-core                => normal PPT behaviour; abnormal drop => host power limit
#   - scheduled < ~98 %                                   => something else shares your core, or quota throttling (section 8)
#     (all-core runs one probe per visible CPU: under a 12/48 quota it is throttled
#     BY DESIGN, so read only its clock there, not its scheduled %)

# =============================================================================
# 6. What this container may actually use (cpuset / affinity)
# =============================================================================
# Ask the kernel; do not guess the cgroup path. (An earlier run, log1003, read
# nothing because step 1b used the v2 path on a v1 host, and the error went to
# stderr while the capture only redirected stdout.)
python3 -c "import os; a=sorted(os.sched_getaffinity(0)); print(len(a),'cpus:',a)"
cat /sys/fs/cgroup/cpuset.cpus.effective 2>&1      # v2
cat /sys/fs/cgroup/cpuset/cpuset.cpus 2>&1         # v1

# =============================================================================
# 7. Contention from neighbours
# =============================================================================
# Vast instances are almost always containers on bare metal: there is NO
# hypervisor, so vmstat's `st` (steal) is always 0 and proves nothing. (It is
# meaningful only on a VM host.) Neighbours cost you through:
#   (a) CFS contention for the same core      -> "scheduled" in section 5, runqueue wait below
#   (b) a busy SMT sibling of your core       -> host CPU map below (invisible to the scheduler)
#   (c) shared L3 / DRAM bandwidth            -> only via the interpreter benchmark (section 9)
#   (d) a lower all-core boost bin            -> section 5 clock
# Typically tens of %; (a) can cost multiples.

# Host CPU map: /proc/stat is host-wide unless lxcfs virtualises it
# (check: `mount | grep lxcfs` empty, and cpu line count == nproc --all).
# Busy cores while YOUR load is zero belong to someone else; pin onto cores
# whose SMT sibling is idle.
python3 - <<'EOF'
import os, time
def snap():
    d = {}
    for line in open("/proc/stat"):
        if line.startswith("cpu") and line[3] != " ":
            f = line.split()
            v = list(map(int, f[1:9]))          # user..steal; guest already counted in user
            d[f[0]] = (sum(v), v[3] + v[4])     # (total, idle + iowait)
    return d
a = snap(); time.sleep(2); b = snap()
mine = os.sched_getaffinity(0)
for k in sorted(a, key=lambda s: int(s[3:])):
    tot = b[k][0] - a[k][0]; busy = tot - (b[k][1] - a[k][1]); cpu = int(k[3:])
    sib = open(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list").read().strip()
    print(f"{k:>6} busy={100*busy/max(tot,1):5.1f}%  siblings={sib:<8} {'MINE' if cpu in mine else ''}")
EOF

# Host load average, as a coarse cross-check. Notes:
#  - /proc/loadavg is host-wide in a container: compare with `nproc --all`, not your quota.
#  - It is an exponential moving average (1-min time constant): wait 2-3 min after
#    your own run ends before reading it.
#  - Linux load also counts D-state (uninterruptible I/O) tasks, not just CPU.
uptime

# During a real run, per-thread runqueue wait of the engine process
# (/proc/<pid>/task/*/schedstat = run_ns wait_ns timeslices):
#   wait_share = d(wait_ns) / (d(run_ns) + d(wait_ns))   -- persistently > a few % stretches every step.

# =============================================================================
# 8. CPU quota (CFS bandwidth)
# =============================================================================
# The platform's "12.0/48 CPU" is usually a CFS QUOTA, not a cpuset: you see all
# 48 cores and may run on any, but total CPU time per period is capped.
# quota/period = your cores; quota -1 / "max" = unlimited.
# The quota is shared by ALL threads in the cgroup, and once a period's budget
# is spent EVERY thread -- including the decode thread -- is frozen until the
# next period (default 100 ms). A "single-threaded" engine still has many busy
# threads: torch intra-op pool (sized from the 48 visible cores, NOT the quota),
# tokenizers threads, CUDA driver threads, busy-polling IPC loops. So a bursty
# workload can be throttled even at low average utilisation -> fat-tail step
# latency. Fix: OMP_NUM_THREADS / torch.set_num_threads <= quota cores,
# TOKENIZERS_PARALLELISM=false.
cat /sys/fs/cgroup/cpu.max 2>&1                                            # v2: "<quota> <period>"
cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>&1   # v1
cat /sys/fs/cgroup/cpu.stat 2>&1 || cat /sys/fs/cgroup/cpu/cpu.stat 2>&1   # nr_throttled / throttled_usec growing => throttled
cat /sys/fs/cgroup/cpu.pressure 2>&1                                       # v2 PSI: "some avg10" > 0 => runnable tasks waiting

# =============================================================================
# 9. Pure-interpreter speed
# =============================================================================
# The launch-overhead test does NOT capture this: it times one op in a tight
# loop, while the engine pays Python dispatch on ~4 400 aten calls per step.
# This loop is the only number that compares decode-loop speed across machines,
# and it also exposes SMT-sibling and cache contention that section 5 misses.
# Valid only with the SAME Python version: 3.10 vs 3.11+ alone can differ by
# 1.5-2x, so record `python3 -VV` with the result. Runs at module level
# (global-dict access); keep it that way for comparability.
python3 -VV
for i in 1 2 3; do python3 -c "
import time
t = time.perf_counter()
s = 0
for i in range(3_000_000): s += i * 2
print(f'{(time.perf_counter()-t)*1000:.0f} ms / 3M-iter loop')"; done
#   reference: ~276 ms on EPYC 7402 (Zen 2, boost 3.35 GHz; log1003, a SLOW host).
#   Its clock was never measured: the "2.24 GHz" in older notes is lscpu's 2800
#   (= base, see section 4) x an instantaneous 80 % scaling read, not a measurement.
#   Python version: <record it>. Clock x IPC predicts a ~5 GHz Zen 4/5 desktop part
#   (log1001: Ryzen 5 7500F) at roughly 2x lower -- measure and record it instead.

# =============================================================================
# 10. NUMA: which node the GPU hangs off
# =============================================================================
# On multi-socket EPYC with NPS2/NPS4 a socket holds several NUMA nodes, so
# think "node", not "socket". If your cpuset excludes the GPU's node you cannot
# pin there anyway.
nvidia-smi topo -m                                   # read the "NUMA Affinity" column
BUS=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | head -1 | tr 'A-F' 'a-f' | sed 's/^0000\(0000:\)/\1/')
cat /sys/bus/pci/devices/$BUS/numa_node 2>&1         # -1 = no NUMA info (single node)
command -v numactl >/dev/null && numactl --hardware | grep -E "^node [0-9]+ (cpus|free)"