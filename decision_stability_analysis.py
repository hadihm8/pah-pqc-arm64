import pandas as pd
import numpy as np

from pah_pqc import (
    SecurityPolicy,
    PerformanceMetrics,
    select_configuration
)


SESSION_FILES = {
    1: "official_results_summary_session1.csv",
    2: "official_results_summary_session2.csv",
    3: "official_results_summary_session3.csv",
}

PAYLOADS = [
    1024,
    10 * 1024,
    100 * 1024,
    1 * 1024 * 1024,
    5 * 1024 * 1024,
    10 * 1024 * 1024,
    50 * 1024 * 1024,
]

POLICIES = [
    SecurityPolicy.CLASSICAL_ALLOWED,
    SecurityPolicy.PQ_REQUIRED,
    SecurityPolicy.HYBRID_REQUIRED,
]

OUTPUT_FILE = "decision_stability_results.csv"
CONSENSUS_FILE = "decision_consensus_summary.csv"


def load_session_summary(session_id, filename):

    df = pd.read_csv(filename)

    if len(df) != 35:
        raise RuntimeError(
            f"Session {session_id}: expected 35 summary rows, "
            f"found {len(df)}"
        )

    return df


def build_metrics(df, payload):

    subset = df[
        df["payload_bytes"] == payload
    ]

    metrics = {}

    for _, row in subset.iterrows():

        mode = row["mode"]

        metrics[mode] = PerformanceMetrics(
            latency_ms=float(
                row["mean_total_ms"]
            ),
            throughput_mbps=float(
                row["mean_throughput_MBps"]
            ),
            overhead_bytes=float(
                row["overhead_bytes"]
            ),
        )

    return metrics


def run_session_decisions():

    rows = []

    for session_id, filename in SESSION_FILES.items():

        df = load_session_summary(
            session_id,
            filename
        )

        for payload in PAYLOADS:

            measured_metrics = build_metrics(
                df,
                payload
            )

            for policy in POLICIES:

                result = select_configuration(
                    security_policy=policy,
                    payload_size_bytes=payload,
                    platform_profile="ARM64-Neoverse-N2",
                    measured_metrics=measured_metrics,
                    latency_limit_ms=None,
                    latency_weight=0.40,
                    throughput_weight=0.40,
                    overhead_weight=0.20,
                )

                rows.append({
                    "session": session_id,
                    "payload_bytes": payload,
                    "security_policy":
                        policy.value,
                    "selected_mode":
                        result.selected_mode.mode_id,
                    "selected_score":
                        result.score,
                    "ranked_modes":
                        str(
                            result.ranked_modes
                        ),
                })

    result_df = pd.DataFrame(rows)

    result_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    return result_df


def create_consensus_summary(decisions):

    rows = []

    grouped = decisions.groupby(
        [
            "payload_bytes",
            "security_policy"
        ]
    )

    for (
        payload,
        policy
    ), group in grouped:

        selections = list(
            group["selected_mode"]
        )

        counts = (
            group[
                "selected_mode"
            ]
            .value_counts()
        )

        consensus_mode = (
            counts.index[0]
        )

        agreement_count = int(
            counts.iloc[0]
        )

        agreement_percent = (
            agreement_count
            / len(selections)
        ) * 100

        stable = (
            agreement_count
            == len(selections)
        )

        rows.append({
            "payload_bytes": payload,
            "security_policy": policy,

            "session1_selection":
                group[
                    group["session"] == 1
                ][
                    "selected_mode"
                ].iloc[0],

            "session2_selection":
                group[
                    group["session"] == 2
                ][
                    "selected_mode"
                ].iloc[0],

            "session3_selection":
                group[
                    group["session"] == 3
                ][
                    "selected_mode"
                ].iloc[0],

            "consensus_mode":
                consensus_mode,

            "agreement_count":
                agreement_count,

            "agreement_percent":
                agreement_percent,

            "stable_across_sessions":
                stable,
        })

    consensus = pd.DataFrame(
        rows
    )

    consensus.to_csv(
        CONSENSUS_FILE,
        index=False
    )

    return consensus


def print_summary(consensus):

    print("\n")
    print("=" * 90)
    print(
        "PAH-PQC DECISION STABILITY ACROSS THREE ARM64 SESSIONS"
    )
    print("=" * 90)

    display = [
        "payload_bytes",
        "security_policy",
        "session1_selection",
        "session2_selection",
        "session3_selection",
        "consensus_mode",
        "agreement_percent",
        "stable_across_sessions",
    ]

    print(
        consensus[
            display
        ].to_string(
            index=False
        )
    )

    total_cases = len(
        consensus
    )

    stable_cases = int(
        consensus[
            "stable_across_sessions"
        ].sum()
    )

    stability_rate = (
        stable_cases
        / total_cases
    ) * 100

    print("\n")
    print(
        "Total policy-payload cases:",
        total_cases
    )

    print(
        "Stable cases across all 3 sessions:",
        stable_cases
    )

    print(
        "Decision stability rate:",
        f"{stability_rate:.2f}%"
    )

    print("\nPolicy-specific stability:")

    for policy in consensus[
        "security_policy"
    ].unique():

        subset = consensus[
            consensus[
                "security_policy"
            ] == policy
        ]

        stable = int(
            subset[
                "stable_across_sessions"
            ].sum()
        )

        rate = (
            stable
            / len(subset)
        ) * 100

        print(
            f"{policy}: "
            f"{stable}/{len(subset)} "
            f"({rate:.2f}%)"
        )


def main():

    print("=" * 90)
    print(
        "PAH-PQC REPRODUCIBILITY-AWARE DECISION ANALYSIS"
    )
    print("=" * 90)

    decisions = (
        run_session_decisions()
    )

    consensus = (
        create_consensus_summary(
            decisions
        )
    )

    print_summary(
        consensus
    )

    print("\nGenerated files:")

    print(
        OUTPUT_FILE
    )

    print(
        CONSENSUS_FILE
    )

    print("\n")
    print(
        "DECISION STABILITY ANALYSIS COMPLETED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()
