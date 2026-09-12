import math
import pandas as pd
import numpy as np

from scipy.stats import kruskal
from statsmodels.stats.multitest import multipletests


SESSIONS = {
    1: "official_results_raw_session1.csv",
    2: "official_results_raw_session2.csv",
    3: "official_results_raw_session3.csv",
}

MODES = ["M1", "M2", "M3", "M4", "M5"]

PAYLOADS = [
    1024,
    10 * 1024,
    100 * 1024,
    1 * 1024 * 1024,
    5 * 1024 * 1024,
    10 * 1024 * 1024,
    50 * 1024 * 1024,
]

POOLED_FILE = "cross_session_pooled_summary.csv"
SESSION_MEANS_FILE = "cross_session_means.csv"
SESSION_EFFECT_FILE = "cross_session_effect_tests.csv"
SPIKE_FILE = "cross_session_5mb_check.csv"


def load_sessions():

    frames = []

    for session_id, filename in SESSIONS.items():

        df = pd.read_csv(filename)

        if len(df) != 1050:
            raise RuntimeError(
                f"Session {session_id}: expected 1050 rows, "
                f"found {len(df)}"
            )

        df["session"] = session_id

        frames.append(df)

    combined = pd.concat(
        frames,
        ignore_index=True
    )

    if len(combined) != 3150:
        raise RuntimeError(
            f"Expected 3150 total observations, "
            f"found {len(combined)}"
        )

    return combined


def ci95(values):

    values = np.asarray(
        values,
        dtype=float
    )

    n = len(values)

    mean = np.mean(values)

    sd = np.std(
        values,
        ddof=1
    )

    se = sd / math.sqrt(n)

    margin = 1.96 * se

    return (
        mean - margin,
        mean + margin
    )


def coefficient_of_variation(values):

    values = np.asarray(
        values,
        dtype=float
    )

    mean = np.mean(values)

    if mean == 0:
        return 0.0

    return (
        np.std(values, ddof=1)
        / mean
    ) * 100


# ============================================================
# SESSION-WISE MEANS
# ============================================================

def calculate_session_means(df):

    rows = []

    for session in [1, 2, 3]:

        for payload in PAYLOADS:

            for mode in MODES:

                subset = df[
                    (df["session"] == session)
                    &
                    (df["payload_bytes"] == payload)
                    &
                    (df["mode"] == mode)
                ]

                rows.append({
                    "session": session,
                    "payload_bytes": payload,
                    "mode": mode,
                    "n": len(subset),

                    "mean_key_ms":
                        subset["key_ms"].mean(),

                    "mean_encryption_ms":
                        subset["encryption_ms"].mean(),

                    "mean_decryption_ms":
                        subset["decryption_ms"].mean(),

                    "mean_total_ms":
                        subset["total_ms"].mean(),

                    "sd_total_ms":
                        subset["total_ms"].std(ddof=1),

                    "mean_throughput_MBps":
                        subset[
                            "throughput_MBps"
                        ].mean(),

                    "overhead_bytes":
                        subset[
                            "overhead_bytes"
                        ].iloc[0],
                })

    result = pd.DataFrame(rows)

    result.to_csv(
        SESSION_MEANS_FILE,
        index=False
    )

    return result


# ============================================================
# POOLED SUMMARY – 90 OBSERVATIONS PER MODE/PAYLOAD
# ============================================================

def calculate_pooled_summary(df, session_means):

    rows = []

    for payload in PAYLOADS:

        for mode in MODES:

            subset = df[
                (df["payload_bytes"] == payload)
                &
                (df["mode"] == mode)
            ]

            totals = subset[
                "total_ms"
            ].values

            throughput = subset[
                "throughput_MBps"
            ].values

            lower, upper = ci95(
                totals
            )

            per_session = session_means[
                (session_means["payload_bytes"] == payload)
                &
                (session_means["mode"] == mode)
            ]

            session_total_means = (
                per_session[
                    "mean_total_ms"
                ].values
            )

            rows.append({
                "payload_bytes": payload,
                "mode": mode,

                "n_pooled": len(subset),

                "pooled_mean_key_ms":
                    subset["key_ms"].mean(),

                "pooled_mean_encryption_ms":
                    subset[
                        "encryption_ms"
                    ].mean(),

                "pooled_mean_decryption_ms":
                    subset[
                        "decryption_ms"
                    ].mean(),

                "pooled_mean_total_ms":
                    np.mean(totals),

                "pooled_sd_total_ms":
                    np.std(
                        totals,
                        ddof=1
                    ),

                "pooled_median_total_ms":
                    np.median(totals),

                "pooled_ci95_low_total_ms":
                    lower,

                "pooled_ci95_high_total_ms":
                    upper,

                "pooled_cv_total_percent":
                    coefficient_of_variation(
                        totals
                    ),

                "between_session_cv_percent":
                    coefficient_of_variation(
                        session_total_means
                    ),

                "session1_mean_total_ms":
                    session_total_means[0],

                "session2_mean_total_ms":
                    session_total_means[1],

                "session3_mean_total_ms":
                    session_total_means[2],

                "pooled_mean_throughput_MBps":
                    np.mean(
                        throughput
                    ),

                "overhead_bytes":
                    subset[
                        "overhead_bytes"
                    ].iloc[0],
            })

    result = pd.DataFrame(rows)

    result.to_csv(
        POOLED_FILE,
        index=False
    )

    return result


# ============================================================
# SESSION EFFECT TEST
# ============================================================

def test_session_effect(df):

    rows = []

    raw_p_values = []

    for payload in PAYLOADS:

        for mode in MODES:

            groups = []

            for session in [1, 2, 3]:

                values = df[
                    (df["session"] == session)
                    &
                    (df["payload_bytes"] == payload)
                    &
                    (df["mode"] == mode)
                ]["total_ms"].values

                groups.append(values)

            statistic, p_value = kruskal(
                *groups
            )

            rows.append({
                "payload_bytes": payload,
                "mode": mode,
                "kruskal_H": statistic,
                "raw_p_value": p_value,
            })

            raw_p_values.append(
                p_value
            )

    reject, adjusted, _, _ = (
        multipletests(
            raw_p_values,
            alpha=0.05,
            method="holm"
        )
    )

    for i, row in enumerate(rows):

        row["holm_adjusted_p"] = (
            float(adjusted[i])
        )

        row["significant_session_effect"] = (
            bool(reject[i])
        )

    result = pd.DataFrame(rows)

    result.to_csv(
        SESSION_EFFECT_FILE,
        index=False
    )

    return result


# ============================================================
# 5 MB VARIABILITY CHECK
# ============================================================

def check_5mb_variability(
    session_means,
    pooled_summary
):

    payload = (
        5 * 1024 * 1024
    )

    rows = []

    for mode in MODES:

        per_session = session_means[
            (session_means["payload_bytes"] == payload)
            &
            (session_means["mode"] == mode)
        ]

        pooled = pooled_summary[
            (pooled_summary["payload_bytes"] == payload)
            &
            (pooled_summary["mode"] == mode)
        ].iloc[0]

        rows.append({
            "mode": mode,

            "session1_mean_total_ms":
                per_session[
                    per_session["session"] == 1
                ]["mean_total_ms"].iloc[0],

            "session1_cv_percent":
                (
                    per_session[
                        per_session["session"] == 1
                    ]["sd_total_ms"].iloc[0]
                    /
                    per_session[
                        per_session["session"] == 1
                    ]["mean_total_ms"].iloc[0]
                ) * 100,

            "session2_mean_total_ms":
                per_session[
                    per_session["session"] == 2
                ]["mean_total_ms"].iloc[0],

            "session2_cv_percent":
                (
                    per_session[
                        per_session["session"] == 2
                    ]["sd_total_ms"].iloc[0]
                    /
                    per_session[
                        per_session["session"] == 2
                    ]["mean_total_ms"].iloc[0]
                ) * 100,

            "session3_mean_total_ms":
                per_session[
                    per_session["session"] == 3
                ]["mean_total_ms"].iloc[0],

            "session3_cv_percent":
                (
                    per_session[
                        per_session["session"] == 3
                    ]["sd_total_ms"].iloc[0]
                    /
                    per_session[
                        per_session["session"] == 3
                    ]["mean_total_ms"].iloc[0]
                ) * 100,

            "pooled_cv_percent":
                pooled[
                    "pooled_cv_total_percent"
                ],

            "between_session_cv_percent":
                pooled[
                    "between_session_cv_percent"
                ],
        })

    result = pd.DataFrame(rows)

    result.to_csv(
        SPIKE_FILE,
        index=False
    )

    return result


def main():

    print("=" * 80)
    print(
        "PAH-PQC CROSS-SESSION REPRODUCIBILITY ANALYSIS"
    )
    print("=" * 80)

    df = load_sessions()

    print(
        "Combined observations:",
        len(df)
    )

    print(
        "Sessions:",
        sorted(
            df["session"].unique()
        )
    )

    session_means = (
        calculate_session_means(
            df
        )
    )

    pooled = (
        calculate_pooled_summary(
            df,
            session_means
        )
    )

    session_effect = (
        test_session_effect(
            df
        )
    )

    spike = (
        check_5mb_variability(
            session_means,
            pooled
        )
    )

    print("\n")
    print("=" * 80)
    print("5 MB VARIABILITY CHECK")
    print("=" * 80)

    display_columns = [
        "mode",
        "session1_cv_percent",
        "session2_cv_percent",
        "session3_cv_percent",
        "pooled_cv_percent",
        "between_session_cv_percent",
    ]

    print(
        spike[
            display_columns
        ].to_string(
            index=False
        )
    )

    significant_sessions = (
        session_effect[
            session_effect[
                "significant_session_effect"
            ]
        ]
    )

    print("\n")
    print(
        "Total session-effect tests:",
        len(session_effect)
    )

    print(
        "Significant session effects "
        "after Holm correction:",
        len(significant_sessions)
    )

    print("\nGenerated files:")

    print(
        SESSION_MEANS_FILE
    )

    print(
        POOLED_FILE
    )

    print(
        SESSION_EFFECT_FILE
    )

    print(
        SPIKE_FILE
    )

    print("\n")
    print(
        "CROSS-SESSION ANALYSIS COMPLETED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()
