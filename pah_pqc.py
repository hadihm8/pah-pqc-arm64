from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple


class SecurityPolicy(Enum):
    """
    Security requirements enforced before performance optimization.
    """
    CLASSICAL_ALLOWED = "classical_allowed"
    PQ_REQUIRED = "pq_required"
    HYBRID_REQUIRED = "hybrid_required"


@dataclass(frozen=True)
class CryptoMode:
    mode_id: str
    key_establishment: str
    aead: str
    pq_capable: bool
    hybrid: bool


@dataclass
class PerformanceMetrics:
    """
    Metrics must come from measured experimental results.
    No synthetic values are embedded in the framework.
    """
    latency_ms: float
    throughput_mbps: float
    overhead_bytes: float


@dataclass
class SelectionResult:
    selected_mode: CryptoMode
    score: float
    payload_size_bytes: int
    platform_profile: str
    security_policy: SecurityPolicy
    latency_limit_ms: Optional[float]
    ranked_modes: List[Tuple[str, float]]


MODES: Dict[str, CryptoMode] = {
    "M1": CryptoMode(
        mode_id="M1",
        key_establishment="X25519",
        aead="AES-256-GCM",
        pq_capable=False,
        hybrid=False,
    ),
    "M2": CryptoMode(
        mode_id="M2",
        key_establishment="ML-KEM-768",
        aead="AES-256-GCM",
        pq_capable=True,
        hybrid=False,
    ),
    "M3": CryptoMode(
        mode_id="M3",
        key_establishment="ML-KEM-768",
        aead="Ascon-AEAD128",
        pq_capable=True,
        hybrid=False,
    ),
    "M4": CryptoMode(
        mode_id="M4",
        key_establishment="X25519 + ML-KEM-768",
        aead="AES-256-GCM",
        pq_capable=True,
        hybrid=True,
    ),
    "M5": CryptoMode(
        mode_id="M5",
        key_establishment="X25519 + ML-KEM-768",
        aead="Ascon-AEAD128",
        pq_capable=True,
        hybrid=True,
    ),
}


def security_filter(policy: SecurityPolicy) -> List[CryptoMode]:
    """
    Stage 1:
    Enforce the security policy before considering performance.
    """

    if policy == SecurityPolicy.HYBRID_REQUIRED:
        return [mode for mode in MODES.values() if mode.hybrid]

    if policy == SecurityPolicy.PQ_REQUIRED:
        return [mode for mode in MODES.values() if mode.pq_capable]

    return list(MODES.values())


def _min_max(values: List[float], value: float) -> float:
    """
    Normalize to the interval [0, 1].
    """
    minimum = min(values)
    maximum = max(values)

    if maximum == minimum:
        return 0.0

    return (value - minimum) / (maximum - minimum)


def rank_admissible_modes(
    admissible_modes: List[CryptoMode],
    measured_metrics: Dict[str, PerformanceMetrics],
    latency_limit_ms: Optional[float] = None,
    latency_weight: float = 0.40,
    throughput_weight: float = 0.40,
    overhead_weight: float = 0.20,
) -> List[Tuple[CryptoMode, float]]:
    """
    Stage 2:
    Rank only security-admissible configurations.

    Lower score = better.

    Latency and overhead are costs.
    Throughput is a benefit and is therefore inverted after normalization.
    """

    total_weight = latency_weight + throughput_weight + overhead_weight

    if abs(total_weight - 1.0) > 1e-9:
        raise ValueError("Weights must sum to 1.0")

    candidates = []

    for mode in admissible_modes:
        if mode.mode_id not in measured_metrics:
            continue

        metric = measured_metrics[mode.mode_id]

        if (
            latency_limit_ms is not None
            and metric.latency_ms > latency_limit_ms
        ):
            continue

        candidates.append(mode)

    if not candidates:
        raise RuntimeError(
            "No configuration satisfies the current security "
            "and latency constraints."
        )

    latencies = [
        measured_metrics[m.mode_id].latency_ms
        for m in candidates
    ]

    throughputs = [
        measured_metrics[m.mode_id].throughput_mbps
        for m in candidates
    ]

    overheads = [
        measured_metrics[m.mode_id].overhead_bytes
        for m in candidates
    ]

    ranked = []

    for mode in candidates:
        metric = measured_metrics[mode.mode_id]

        norm_latency = _min_max(
            latencies,
            metric.latency_ms
        )

        norm_throughput = _min_max(
            throughputs,
            metric.throughput_mbps
        )

        norm_overhead = _min_max(
            overheads,
            metric.overhead_bytes
        )

        throughput_cost = 1.0 - norm_throughput

        score = (
            latency_weight * norm_latency
            + throughput_weight * throughput_cost
            + overhead_weight * norm_overhead
        )

        ranked.append((mode, score))

    ranked.sort(key=lambda item: item[1])

    return ranked


def select_configuration(
    security_policy: SecurityPolicy,
    payload_size_bytes: int,
    platform_profile: str,
    measured_metrics: Dict[str, PerformanceMetrics],
    latency_limit_ms: Optional[float] = None,
    latency_weight: float = 0.40,
    throughput_weight: float = 0.40,
    overhead_weight: float = 0.20,
) -> SelectionResult:
    """
    PAH-PQC selection pipeline:

    Security Policy
        ->
    Security Admissibility Filter
        ->
    Performance Constraint Check
        ->
    Multi-Metric Ranking
        ->
    Selected Cryptographic Configuration
    """

    if payload_size_bytes <= 0:
        raise ValueError(
            "Payload size must be greater than zero."
        )

    admissible = security_filter(security_policy)

    ranked = rank_admissible_modes(
        admissible_modes=admissible,
        measured_metrics=measured_metrics,
        latency_limit_ms=latency_limit_ms,
        latency_weight=latency_weight,
        throughput_weight=throughput_weight,
        overhead_weight=overhead_weight,
    )

    selected_mode, selected_score = ranked[0]

    return SelectionResult(
        selected_mode=selected_mode,
        score=selected_score,
        payload_size_bytes=payload_size_bytes,
        platform_profile=platform_profile,
        security_policy=security_policy,
        latency_limit_ms=latency_limit_ms,
        ranked_modes=[
            (mode.mode_id, score)
            for mode, score in ranked
        ],
    )


def show_policy_matrix() -> None:
    """
    Basic framework self-check.
    This does not use experimental performance values.
    """

    print("PAH-PQC SECURITY POLICY MATRIX")
    print("=" * 50)

    for policy in SecurityPolicy:
        modes = security_filter(policy)

        print(
            f"{policy.value}: "
            + ", ".join(mode.mode_id for mode in modes)
        )


if __name__ == "__main__":
    show_policy_matrix()
