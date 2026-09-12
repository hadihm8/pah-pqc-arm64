import math
import itertools
import pandas as pd
import numpy as np

from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests


RAW_FILE = "official_results_raw.csv"

FRIEDMAN_FILE = "friedman_results.csv"
PAIRWISE_FILE = "wilcoxon_holm_results.csv"

MODES = ["M1", "M2", "M3", "M4", "M5"]

METRICS = [
    "key_ms",
    "encryption_ms",
    "decryption_ms",
    "total_ms",
]


def interpret_kendall_w(w):
    if w < 0.10:
        return "negligible"
    elif w < 0.30:
        return "small"
    elif w < 0.50:
        return "moderate"
    else:
        return "large"


def interpret_effect_r(r):
    r = abs(r)

    if r < 0.10:
        return "negligible"
    elif r < 0.30:
        return "small"
    elif r < 0.50:
        return "moderate"
    else:
        return "large"


def prepare_pivot(df, payload, metric):
    subset = df[
        df["payload_bytes"] == payload
    ].copy()

    pivot = subset.pivot(
        index="run",
        columns="mode",
        values=metric
    )

    pivot = pivot[MODES]

    if pivot.isnull().any().any():
        raise RuntimeError(
            f"Missing matched observations for "
            f"payload={payload}, metric={metric}"
        )

    return pivot


def run_friedman(df):
    rows = []

    payloads = sorted(
        df["payload_bytes"].unique()
    )

    for payload in payloads:

        for metric in METRICS:

            pivot = prepare_pivot(
                df,
                payload,
                metric
            )

            groups = [
                pivot[mode].values
                for mode in MODES
            ]

            statistic, p_value = (
                friedmanchisquare(*groups)
            )

            n = len(pivot)
            k = len(MODES)

            kendall_w = (
                statistic /
                (n * (k - 1))
            )

            rows.append({
                "payload_bytes": payload,
                "metric": metric,
                "n_matched_runs": n,
                "k_models": k,
                "chi_square": statistic,
                "p_value": p_value,
                "kendall_w": kendall_w,
                "effect_interpretation":
                    interpret_kendall_w(
                        kendall_w
                    ),
                "significant_0_05":
                    p_value < 0.05,
            })

    result = pd.DataFrame(rows)

    result.to_csv(
        FRIEDMAN_FILE,
        index=False
    )

    return result


def run_pairwise_wilcoxon(df):
    all_rows = []

    payloads = sorted(
        df["payload_bytes"].unique()
    )

    comparisons = list(
        itertools.combinations(
            MODES,
            2
        )
    )

    for payload in payloads:

        for metric in METRICS:

            pivot = prepare_pivot(
                df,
                payload,
                metric
            )

            family_rows = []

            raw_p_values = []

            for mode_a, mode_b in comparisons:

                x = pivot[mode_a].values
                y = pivot[mode_b].values

                differences = x - y

                non_zero_n = int(
                    np.sum(
                        differences != 0
                    )
                )

                if non_zero_n == 0:

                    statistic = 0.0
                    p_value = 1.0
                    z_value = 0.0
                    effect_r = 0.0

                else:

                    result = wilcoxon(
                        x,
                        y,
                        zero_method="wilcox",
                        correction=False,
                        alternative="two-sided",
                        method="approx"
                    )

                    statistic = float(
                        result.statistic
                    )

                    p_value = float(
                        result.pvalue
                    )

                    z_value = float(
                        result.zstatistic
                    )

                    effect_r = (
                        abs(z_value) /
                        math.sqrt(non_zero_n)
                    )

                family_rows.append({
                    "payload_bytes": payload,
                    "metric": metric,
                    "mode_a": mode_a,
                    "mode_b": mode_b,
                    "n_nonzero_pairs":
                        non_zero_n,
                    "wilcoxon_statistic":
                        statistic,
                    "z_value":
                        z_value,
                    "raw_p_value":
                        p_value,
                    "effect_r":
                        effect_r,
                    "effect_interpretation":
                        interpret_effect_r(
                            effect_r
                        ),
                })

                raw_p_values.append(
                    p_value
                )

            reject, adjusted_p, _, _ = (
                multipletests(
                    raw_p_values,
                    alpha=0.05,
                    method="holm"
                )
            )

            for i, row in enumerate(
                family_rows
            ):

                row["holm_adjusted_p"] = (
                    float(
                        adjusted_p[i]
                    )
                )

                row["significant_after_holm"] = (
                    bool(
                        reject[i]
                    )
                )

                all_rows.append(row)

    result = pd.DataFrame(
        all_rows
    )

    result.to_csv(
        PAIRWISE_FILE,
        index=False
    )

    return result


def validate_data(df):
    expected_rows = (
        7 * 5 * 30
    )

    if len(df) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} rows, "
            f"but found {len(df)}."
        )

    required = {
        "payload_bytes",
        "run",
        "mode",
        "key_ms",
        "encryption_ms",
        "decryption_ms",
        "total_ms",
    }

    missing = (
        required -
        set(df.columns)
    )

    if missing:
        raise RuntimeError(
            f"Missing columns: {missing}"
        )

    print(
        "Data validation: OK"
    )

    print(
        "Observations:",
        len(df)
    )

    print(
        "Payload sizes:",
        sorted(
            df["payload_bytes"].unique()
        )
    )

    print(
        "Modes:",
        sorted(
            df["mode"].unique()
        )
    )


def print_key_results(
    friedman_df,
    pairwise_df
):
    print("\n")
    print("=" * 80)
    print("FRIEDMAN TEST SUMMARY")
    print("=" * 80)

    display_cols = [
        "payload_bytes",
        "metric",
        "chi_square",
        "p_value",
        "kendall_w",
        "effect_interpretation",
    ]

    print(
        friedman_df[
            display_cols
        ].to_string(
            index=False
        )
    )

    print("\n")
    print("=" * 80)
    print("PAIRWISE WILCOXON-HOLM SUMMARY")
    print("=" * 80)

    significant = pairwise_df[
        pairwise_df[
            "significant_after_holm"
        ]
    ]

    print(
        "Total pairwise comparisons:",
        len(pairwise_df)
    )

    print(
        "Significant after Holm:",
        len(significant)
    )

    print(
        "\nResults written to:"
    )

    print(
        FRIEDMAN_FILE
    )

    print(
        PAIRWISE_FILE
    )


def main():
    print("=" * 80)
    print(
        "PAH-PQC OFFICIAL STATISTICAL ANALYSIS"
    )
    print("=" * 80)

    df = pd.read_csv(
        RAW_FILE
    )

    validate_data(df)

    friedman_df = (
        run_friedman(df)
    )

    pairwise_df = (
        run_pairwise_wilcoxon(df)
    )

    print_key_results(
        friedman_df,
        pairwise_df
    )

    expected_friedman = (
        7 * 4
    )

    expected_pairwise = (
        7 * 4 * 10
    )

    print("\n")
    print(
        "Expected Friedman tests:",
        expected_friedman
    )

    print(
        "Actual Friedman tests:",
        len(friedman_df)
    )

    print(
        "Expected pairwise tests:",
        expected_pairwise
    )

    print(
        "Actual pairwise tests:",
        len(pairwise_df)
    )

    if len(friedman_df) != expected_friedman:
        raise RuntimeError(
            "Unexpected number of "
            "Friedman tests."
        )

    if len(pairwise_df) != expected_pairwise:
        raise RuntimeError(
            "Unexpected number of "
            "pairwise comparisons."
        )

    print("\nSTATISTICAL ANALYSIS COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
