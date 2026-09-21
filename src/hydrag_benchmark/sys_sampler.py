"""System-resource sampler for microbenchmark phases.

Collects CPU, RAM, disk I/O, network I/O, and GPU metrics during a timed
phase.  Sampling runs in a background thread at a configurable interval.

Optional dependencies (graceful degradation when absent):
  pip install hydrag-benchmark[metrics]   → psutil>=5.9, pynvml>=11.0

GPU notes
---------
Metrics are collected via NVML (NVIDIA Management Library).  Non-NVIDIA
hosts or hosts without a GPU will always see ``gpu: null`` in the output.

Per-phase GPU metric catalogue
-------------------------------
compute_util_pct        SM (shader multiprocessor) utilisation — how busy the
                        GPU cores are executing CUDA kernels.
mem_bandwidth_util_pct  Memory-bus utilisation — fraction of peak DRAM bandwidth
                        consumed.  High value = memory-bound workload.
vram_used_mb            Bytes allocated in VRAM (device memory).
vram_free_mb            Bytes free in VRAM.
vram_total_mb           Total installed VRAM.
pcie_tx_kb_s            PCIe upstream throughput host→GPU (peak samples, KB/s).
pcie_rx_kb_s            PCIe downstream throughput GPU→host (peak samples, KB/s).
pcie_replay_count       Cumulative PCIe replay counter delta — non-zero signals
                        link errors or signal-integrity problems.
power_w                 Instantaneous GPU power draw (Watts).  Useful for energy
                        cost modelling and detecting thermal throttle.
temp_c                  GPU die temperature (°C).  >83°C typically triggers
                        frequency throttling on consumer parts; >90°C on
                        data-centre parts.
sm_clock_mhz            Streaming-multiprocessor clock (MHz).  Drops under
                        thermal/power throttle.
mem_clock_mhz           Memory clock (MHz).  Separate from SM clock; drops on
                        severe power throttle.
fan_speed_pct           Fan duty cycle (%).  Not available on passive-cooled
                        data-centre GPUs (returns null).
ecc_sbe_delta           Single-bit ECC error delta during the phase.  SBEs are
                        corrected in hardware but indicate DRAM wear.
ecc_dbe_delta           Double-bit ECC error delta.  DBEs are uncorrectable and
                        may cause data corruption — alarm-level metric in prod.
nvlink_rx_kb_s          NVLink receive bandwidth (multi-GPU nodes only).
nvlink_tx_kb_s          NVLink transmit bandwidth (multi-GPU nodes only).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

# ── optional deps — graceful degradation ──────────────────────────────────────

try:
    import psutil as _psutil
    _HAS_PSUTIL = True
except ImportError:
    _psutil = None
    _HAS_PSUTIL = False

try:
    # nvidia-ml-py (PyPI: nvidia-ml-py) supersedes the deprecated pynvml package.
    # Both expose the same `pynvml` module name.
    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.filterwarnings("ignore", category=FutureWarning)
        import pynvml as _nvml
    _nvml.nvmlInit()
    _NVML_LIBRARY_OK = True
    _NVML_DEVICE_COUNT = _nvml.nvmlDeviceGetCount()
    _HAS_NVML = _NVML_DEVICE_COUNT > 0
except ImportError:
    _nvml = None
    _NVML_LIBRARY_OK = False
    _NVML_DEVICE_COUNT = 0
    _HAS_NVML = False
except Exception:
    _nvml = None
    _NVML_LIBRARY_OK = True  # library loaded but init/query failed
    _NVML_DEVICE_COUNT = 0
    _HAS_NVML = False

# ── GPU helper ────────────────────────────────────────────────────────────────

_NVML_PCIE_TX = 0  # NVML_PCIE_UTIL_TX_BYTES
_NVML_PCIE_RX = 1  # NVML_PCIE_UTIL_RX_BYTES


def _nvml_handles() -> list[Any]:
    if not _HAS_NVML:
        return []
    return [_nvml.nvmlDeviceGetHandleByIndex(i) for i in range(_NVML_DEVICE_COUNT)]


def _safe(fn: Any, *args: Any, default: Any = None) -> Any:
    """Call an NVML function; return *default* on any exception."""
    try:
        return fn(*args)
    except Exception:  # noqa: BLE001
        return default


def _snapshot_gpu(handles: list[Any]) -> list[dict[str, Any]]:
    """Collect a single point-in-time GPU snapshot across all devices."""
    snaps: list[dict[str, Any]] = []
    for h in handles:
        util = _safe(_nvml.nvmlDeviceGetUtilizationRates, h)
        mem = _safe(_nvml.nvmlDeviceGetMemoryInfo, h)
        pcie_replay = _safe(_nvml.nvmlDeviceGetPcieReplayCounter, h)

        # NVLink: iterate link indices until the call fails
        nvlink_rx = nvlink_tx = 0
        for link in range(8):
            rx = _safe(_nvml.nvmlDeviceGetNvLinkUtilizationCounter, h, link, 0)
            tx = _safe(_nvml.nvmlDeviceGetNvLinkUtilizationCounter, h, link, 1)
            if rx is None:
                break
            nvlink_rx += rx
            nvlink_tx += tx

        fan = _safe(_nvml.nvmlDeviceGetFanSpeed, h)
        ecc_sbe = _safe(
            _nvml.nvmlDeviceGetTotalEccErrors, h,
            _nvml.NVML_MEMORY_ERROR_TYPE_CORRECTED,
            _nvml.NVML_AGGREGATE_ECC,
        )
        ecc_dbe = _safe(
            _nvml.nvmlDeviceGetTotalEccErrors, h,
            _nvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
            _nvml.NVML_AGGREGATE_ECC,
        )

        snaps.append({
            "compute_util_pct": util.gpu if util else None,
            "mem_bandwidth_util_pct": util.memory if util else None,
            "vram_used_mb": round(mem.used / 1024**2, 1) if mem else None,
            "vram_free_mb": round(mem.free / 1024**2, 1) if mem else None,
            "vram_total_mb": round(mem.total / 1024**2, 1) if mem else None,
            # pcieThroughput returns KB/s; measured over 20 ms NVML window
            "pcie_tx_kb_s": _safe(_nvml.nvmlDeviceGetPcieThroughput, h, _NVML_PCIE_TX),
            "pcie_rx_kb_s": _safe(_nvml.nvmlDeviceGetPcieThroughput, h, _NVML_PCIE_RX),
            "pcie_replay_count": pcie_replay,
            "power_w": round(_safe(_nvml.nvmlDeviceGetPowerUsage, h, default=0) / 1000, 1),
            "temp_c": _safe(_nvml.nvmlDeviceGetTemperature, h, _nvml.NVML_TEMPERATURE_GPU),
            "sm_clock_mhz": _safe(_nvml.nvmlDeviceGetClockInfo, h, _nvml.NVML_CLOCK_SM),
            "mem_clock_mhz": _safe(_nvml.nvmlDeviceGetClockInfo, h, _nvml.NVML_CLOCK_MEM),
            "fan_speed_pct": fan,
            "ecc_sbe_count": ecc_sbe,
            "ecc_dbe_count": ecc_dbe,
            "nvlink_rx_kb_s": nvlink_rx if nvlink_rx else None,
            "nvlink_tx_kb_s": nvlink_tx if nvlink_tx else None,
        })
    return snaps


# ── dataclass for one phase result ────────────────────────────────────────────

@dataclass
class PhaseMetrics:
    phase: str
    duration_s: float

    # CPU
    cpu_mean_pct: float | None = None
    cpu_peak_pct: float | None = None

    # RAM
    ram_before_mb: float | None = None
    ram_after_mb: float | None = None
    ram_peak_mb: float | None = None

    # Disk I/O
    disk_read_mb: float | None = None
    disk_write_mb: float | None = None
    disk_read_iops: float | None = None   # total ops during phase
    disk_write_iops: float | None = None

    # Network I/O
    net_sent_mb: float | None = None
    net_recv_mb: float | None = None

    # GPU (list — one entry per device; null if no NVML)
    gpu: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "phase": self.phase,
            "duration_s": round(self.duration_s, 3),
        }
        if self.cpu_mean_pct is not None:
            d["cpu_mean_pct"] = round(self.cpu_mean_pct, 1)
            d["cpu_peak_pct"] = round(self.cpu_peak_pct or 0, 1)
        if self.ram_before_mb is not None:
            d["ram_before_mb"] = round(self.ram_before_mb, 1)
            d["ram_after_mb"] = round(self.ram_after_mb or 0, 1)
            d["ram_peak_mb"] = round(self.ram_peak_mb or 0, 1)
        if self.disk_read_mb is not None:
            d["disk_read_mb"] = round(self.disk_read_mb, 2)
            d["disk_write_mb"] = round(self.disk_write_mb or 0, 2)
            d["disk_read_iops"] = int(self.disk_read_iops or 0)
            d["disk_write_iops"] = int(self.disk_write_iops or 0)
        if self.net_sent_mb is not None:
            d["net_sent_mb"] = round(self.net_sent_mb, 2)
            d["net_recv_mb"] = round(self.net_recv_mb or 0, 2)
        if self.gpu is not None:
            d["gpu"] = self.gpu
        return d


# ── sampler ───────────────────────────────────────────────────────────────────

@dataclass
class _State:
    cpu_samples: list[float] = field(default_factory=list)
    ram_samples: list[float] = field(default_factory=list)
    gpu_samples: list[list[dict[str, Any]]] = field(default_factory=list)
    stop: bool = False


class PhaseSampler:
    """Background-thread sampler.  Use as a context manager.

    Example::

        with PhaseSampler("insert") as s:
            do_insert_work()
        metrics = s.result()
    """

    def __init__(self, phase: str, interval_s: float = 0.1) -> None:
        self._phase = phase
        self._interval = interval_s
        self._state = _State()
        self._thread: threading.Thread | None = None
        self._t0: float = 0.0
        self._t1: float = 0.0
        self._disk_before: Any = None
        self._disk_after: Any = None
        self._net_before: Any = None
        self._net_after: Any = None
        self._ram_before: float | None = None
        self._gpu_handles: list[Any] = _nvml_handles()

    def __enter__(self) -> "PhaseSampler":
        if _HAS_PSUTIL:
            self._disk_before = _psutil.disk_io_counters()
            self._net_before = _psutil.net_io_counters()
            mem = _psutil.virtual_memory()
            self._ram_before = mem.used / 1024**2
        self._t0 = time.monotonic()
        self._state = _State()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._state.stop = True
        if self._thread:
            self._thread.join(timeout=self._interval * 10)
        self._t1 = time.monotonic()
        if _HAS_PSUTIL:
            self._disk_after = _psutil.disk_io_counters()
            self._net_after = _psutil.net_io_counters()

    def _sample_loop(self) -> None:
        while not self._state.stop:
            if _HAS_PSUTIL:
                self._state.cpu_samples.append(_psutil.cpu_percent(interval=None))
                mem = _psutil.virtual_memory()
                self._state.ram_samples.append(mem.used / 1024**2)
            if _HAS_NVML:
                self._state.gpu_samples.append(_snapshot_gpu(self._gpu_handles))
            time.sleep(self._interval)

    def result(self) -> PhaseMetrics:
        duration = self._t1 - self._t0
        m = PhaseMetrics(phase=self._phase, duration_s=duration)

        if _HAS_PSUTIL and self._state.cpu_samples:
            m.cpu_mean_pct = sum(self._state.cpu_samples) / len(self._state.cpu_samples)
            m.cpu_peak_pct = max(self._state.cpu_samples)

        if _HAS_PSUTIL and self._state.ram_samples:
            mem = _psutil.virtual_memory()
            ram_after = mem.used / 1024**2
            m.ram_before_mb = self._ram_before
            m.ram_after_mb = ram_after
            m.ram_peak_mb = max(self._state.ram_samples)

        if _HAS_PSUTIL and self._disk_before and self._disk_after:
            rb = self._disk_before
            ra = self._disk_after
            m.disk_read_mb = (ra.read_bytes - rb.read_bytes) / 1024**2
            m.disk_write_mb = (ra.write_bytes - rb.write_bytes) / 1024**2
            m.disk_read_iops = ra.read_count - rb.read_count
            m.disk_write_iops = ra.write_count - rb.write_count

        if _HAS_PSUTIL and self._net_before and self._net_after:
            nb = self._net_before
            na = self._net_after
            m.net_sent_mb = (na.bytes_sent - nb.bytes_sent) / 1024**2
            m.net_recv_mb = (na.bytes_recv - nb.bytes_recv) / 1024**2

        if _HAS_NVML and self._state.gpu_samples:
            n_devs = len(self._state.gpu_samples[0])
            gpu_out: list[dict[str, Any]] = []
            for dev_idx in range(n_devs):
                dev_samples = [s[dev_idx] for s in self._state.gpu_samples if len(s) > dev_idx]
                if not dev_samples:
                    continue
                first = dev_samples[0]
                last = dev_samples[-1]
                gpu_out.append({
                    "device": dev_idx,
                    "compute_util_pct": {
                        "mean": _mean_of(dev_samples, "compute_util_pct"),
                        "peak": _peak_of(dev_samples, "compute_util_pct"),
                    },
                    "mem_bandwidth_util_pct": {
                        "mean": _mean_of(dev_samples, "mem_bandwidth_util_pct"),
                        "peak": _peak_of(dev_samples, "mem_bandwidth_util_pct"),
                    },
                    "vram_used_mb": {
                        "before": first.get("vram_used_mb"),
                        "after": last.get("vram_used_mb"),
                        "peak": _peak_of(dev_samples, "vram_used_mb"),
                    },
                    "vram_free_mb": last.get("vram_free_mb"),
                    "vram_total_mb": last.get("vram_total_mb"),
                    "pcie_tx_kb_s": {
                        "mean": _mean_of(dev_samples, "pcie_tx_kb_s"),
                        "peak": _peak_of(dev_samples, "pcie_tx_kb_s"),
                    },
                    "pcie_rx_kb_s": {
                        "mean": _mean_of(dev_samples, "pcie_rx_kb_s"),
                        "peak": _peak_of(dev_samples, "pcie_rx_kb_s"),
                    },
                    # Delta: end - start (cumulative counter)
                    "pcie_replay_delta": _delta(first, last, "pcie_replay_count"),
                    "power_w": {
                        "mean": _mean_of(dev_samples, "power_w"),
                        "peak": _peak_of(dev_samples, "power_w"),
                    },
                    "temp_c": {
                        "mean": _mean_of(dev_samples, "temp_c"),
                        "peak": _peak_of(dev_samples, "temp_c"),
                    },
                    "sm_clock_mhz": {
                        "mean": _mean_of(dev_samples, "sm_clock_mhz"),
                        "min": _min_of(dev_samples, "sm_clock_mhz"),
                    },
                    "mem_clock_mhz": {
                        "mean": _mean_of(dev_samples, "mem_clock_mhz"),
                        "min": _min_of(dev_samples, "mem_clock_mhz"),
                    },
                    "fan_speed_pct": _mean_of(dev_samples, "fan_speed_pct"),
                    "ecc_sbe_delta": _delta(first, last, "ecc_sbe_count"),
                    "ecc_dbe_delta": _delta(first, last, "ecc_dbe_count"),
                    "nvlink_rx_kb_s": {
                        "mean": _mean_of(dev_samples, "nvlink_rx_kb_s"),
                        "peak": _peak_of(dev_samples, "nvlink_rx_kb_s"),
                    } if any(s.get("nvlink_rx_kb_s") for s in dev_samples) else None,
                    "nvlink_tx_kb_s": {
                        "mean": _mean_of(dev_samples, "nvlink_tx_kb_s"),
                        "peak": _peak_of(dev_samples, "nvlink_tx_kb_s"),
                    } if any(s.get("nvlink_tx_kb_s") for s in dev_samples) else None,
                })
            m.gpu = gpu_out if gpu_out else None

        return m


# ── stat helpers ──────────────────────────────────────────────────────────────

def _vals(samples: list[dict[str, Any]], key: str) -> list[float]:
    return [s[key] for s in samples if s.get(key) is not None]


def _mean_of(samples: list[dict[str, Any]], key: str) -> float | None:
    v = _vals(samples, key)
    return round(sum(v) / len(v), 1) if v else None


def _peak_of(samples: list[dict[str, Any]], key: str) -> float | None:
    v = _vals(samples, key)
    return round(max(v), 1) if v else None


def _min_of(samples: list[dict[str, Any]], key: str) -> float | None:
    v = _vals(samples, key)
    return round(min(v), 1) if v else None


def _delta(first: dict[str, Any], last: dict[str, Any], key: str) -> int | None:
    a, b = first.get(key), last.get(key)
    if a is None or b is None:
        return None
    return int(b) - int(a)


def available_backends() -> dict[str, bool | int]:
    """Report which metric backends are available on this host."""
    return {
        "psutil": _HAS_PSUTIL,
        "nvml_library": _NVML_LIBRARY_OK,
        "nvml": _HAS_NVML,
        "gpu_count": _NVML_DEVICE_COUNT,
    }
