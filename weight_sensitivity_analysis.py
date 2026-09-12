import pandas as pd

from pah_pqc import (
    SecurityPolicy,
    PerformanceMetrics,
    select_configuration,
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

WEIGHT_PROFILES = {
    "balanced": {
        "latency": 0.40,
        "throughput": 0.40,
        "overhead": 0.20,
    },

    "latency_oriented": {
        "latency": 0.60,
        "throughput": 0.25,
        "overhead": 0.15,
    },

    "throughput_oriented": {
        "latency": 0.25,
        "throughput": 0.60,
        "overhead": 0.15,
    },

    "overhead_oriented": {
        "latency": 0.25,
        "throughput": 0.25,
        "overhead": 0.50,
    },

    "equal_weights": {
        "latency": 1 / 3,
        "throughput": 1 / 3,
        "overhead": 1 / 3,
    },
}


DETAIL_FILE = "weight_sensitivity_results.csv"
CONSENSUS_FILE = "weight_sensitivity_consensus.csv"
ROBUSTNESS_FILE = "weight_sensitivity_robustness.csv"


def load_summary(filename):

    df = pd.read_csv(filename)

    if len(df) != 35:
        raise RuntimeError(
            f"Expected 35 rows in {filename}, "
            f"found {len(df)}"
        )

    return df


def build_metrics(df, payload):

    subset = df[
        df["payload_bytes"] == payload
    ]

    metrics = {}

    for _, row in subset.iterrows():

        metrics[row["mode"]] = PerformanceMetrics(
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


def run_sensitivity():

    rows = []

    for session_id, filename in SESSION_FILES.items():

        df = load_summary(filename)

        for payload in PAYLOADS:

            metrics = build_metrics(
                df,
                payload
            )

            for policy in POLICIES:

                for profile_name, weights in WEIGHT_PROFILES.items():

                    result = select_configuration(
                        security_policy=policy,
                        payload_size_bytes=payload,
                        platform_profile="ARM64-Neoverse-N2",
                        measured_metrics=metrics,
                        latency_limit_ms=None,

                        latency_weight=
                            weights["latency"],

                        throughput_weight=
                            weights["throughput"],

                        overhead_weight=
                            weights["overhead"],
                    )

                    rows.append({
                        "session": session_id,
                        "payload_bytes": payload,
                        "security_policy":
                            policy.value,

                        "weight_profile":
                            profile_name,

                        "latency_weight":
                            weights["latency"],

                        "throughput_weight":
                            weights["throughput"],

                        "overhead_weight":
                            weights["overhead"],

                        "selected_mode":
                            result.selected_mode.mode_id,

                        "selected_score":
                            result.score,
                    })

    result = pd.DataFrame(rows)

    result.to_csv(
        DETAIL_FILE,
        index=False
    )

    return result


def create_profile_consensus(results):

    rows = []

    grouped = results.groupby(
        [
            "payload_bytes",
            "security_policy",
            "weight_profile",
        ]
    )

    for (
        payload,
        policy,
        profile
    ), group in grouped:

        selections = list(
            group["selected_mode"]
        )

        counts = (
            group["selected_mode"]
            .value_counts()
        )

        consensus_mode = (
            counts.index[0]
        )

        agreement_count = int(
            counts.iloc[0]
        )

        stable_sessions = (
            agreement_count == 3
        )

        rows.append({
            "payload_bytes":
                payload,

            "security_policy":
                policy,

            "weight_profile":
                profile,

            "session1_selection":
                group[
                    group["session"] == 1
                ]["selected_mode"].iloc[0],

            "session2_selection":
                group[
                    group["session"] == 2
                ]["selected_mode"].iloc[0],

            "session3_selection":
                group[
                    group["session"] == 3
                ]["selected_mode"].iloc[0],

            "consensus_mode":
                consensus_mode,

            "session_agreement_percent":
                (
                    agreement_count / 3
                ) * 100,

            "stable_across_sessions":
                stable_sessions,
        })

    result = pd.DataFrame(rows)

    result.to_csv(
        CONSENSUS_FILE,
        index=False
    )

    return result


def calculate_weight_robustness(consensus):

    rows = []

    for policy in consensus[
        "security_policy"
    ].unique():

        for payload in PAYLOADS:

            subset = consensus[
                (
                    consensus[
                        "security_policy"
                    ] == policy
                )
                &
                (
                    consensus[
                        "payload_bytes"
                    ] == payload
                )
            ]

            selections = list(
                subset[
                    "consensus_mode"
                ]
            )

            unique_modes = sorted(
                set(selections)
            )

            robust = (
                len(unique_modes) == 1
            )

            rows.append({
                "security_policy":
                    policy,

                "payload_bytes":
                    payload,

                "selected_modes_across_profiles":
                    ",".join(
                        unique_modes
                    ),

                "number_of_unique_modes":
                    len(unique_modes),

                "robust_across_weight_profiles":
                    robust,
            })

    result = pd.DataFrame(rows)

    result.to_csv(
        ROBUSTNESS_FILE,
        index=False
    )

    return result


def print_summary(
    consensus,
    robustness
):

    print("\n")
    print("=" * 90)
    print(
        "PAH-PQC WEIGHT SENSITIVITY ANALYSIS"
    )
    print("=" * 90)

    total_consensus_cases = len(
        consensus
    )

    stable_session_cases = int(
        consensus[
            "stable_across_sessions"
        ].sum()
    )

    print(
        "Total profile-policy-payload cases:",
        total_consensus_cases
    )

    print(
        "Stable across all 3 sessions:",
        stable_session_cases
    )

    print(
        "Cross-session stability:",
        f"{(
            stable_session_cases /
            total_consensus_cases
        ) * 100:.2f}%"
    )

    print("\n")
    print("=" * 90)
    print(
        "ROBUSTNESS ACROSS WEIGHT PROFILES"
    )
    print("=" * 90)

    for policy in robustness[
        "security_policy"
    ].unique():

        subset = robustness[
            robustness[
                "security_policy"
            ] == policy
        ]

        robust_count = int(
            subset[
                "robust_across_weight_profiles"
            ].sum()
        )

        rate = (
            robust_count /
            len(subset)
        ) * 100

        print(
            f"{policy}: "
            f"{robust_count}/{len(subset)} "
            f"payloads robust "
            f"({rate:.2f}%)"
        )

        modes = (
            subset[
                "selected_modes_across_profiles"
            ]
            .value_counts()
        )

        print(
            "Selections:",
            modes.to_dict()
        )

    print("\n")
    print(
        "Cases where changing weights "
        "changed the selected mode:"
    )

    changed = robustness[
        ~robustness[
            "robust_across_weight_profiles"
        ]
    ]

    if len(changed) == 0:

        print(
            "NONE - all decisions remained unchanged."
        )

    else:

        print(
            changed.to_string(
                index=False
            )
        )

    print("\nGenerated files:")

    print(
        DETAIL_FILE
    )

    print(
        CONSENSUS_FILE
    )

    print(
        ROBUSTNESS_FILE
    )

    print("\n")
    print(
        "WEIGHT SENSITIVITY ANALYSIS "
        "COMPLETED SUCCESSFULLY"
    )


def main():

    results = (
        run_sensitivity()
    )

    consensus = (
        create_profile_consensus(
            results
        )
    )

    robustness = (
        calculate_weight_robustness(
            consensus
        )
    )

    print_summary(
        consensus,
        robustness
    )


if __name__ == "__main__":
    main()
