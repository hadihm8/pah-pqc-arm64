import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


RAW_FILE = "official_results_raw.csv"
SUMMARY_FILE = "official_results_summary.csv"

OUTPUT_DIR = "figures"

MODES = ["M1", "M2", "M3", "M4", "M5"]

PAYLOAD_LABELS = {
    1024: "1 KB",
    10 * 1024: "10 KB",
    100 * 1024: "100 KB",
    1 * 1024 * 1024: "1 MB",
    5 * 1024 * 1024: "5 MB",
    10 * 1024 * 1024: "10 MB",
    50 * 1024 * 1024: "50 MB",
}

PAYLOAD_ORDER = list(PAYLOAD_LABELS.keys())


def save_figure(fig, name):
    png_path = os.path.join(
        OUTPUT_DIR,
        f"{name}.png"
    )

    pdf_path = os.path.join(
        OUTPUT_DIR,
        f"{name}.pdf"
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight"
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Saved:", png_path)
    print("Saved:", pdf_path)


def validate_data(raw, summary):

    expected_raw = 1050
    expected_summary = 35

    if len(raw) != expected_raw:
        raise RuntimeError(
            f"Expected {expected_raw} raw observations, "
            f"found {len(raw)}."
        )

    if len(summary) != expected_summary:
        raise RuntimeError(
            f"Expected {expected_summary} summary rows, "
            f"found {len(summary)}."
        )

    print("Raw observations:", len(raw))
    print("Summary rows:", len(summary))
    print("Data validation: OK")


# ============================================================
# FIGURE 1
# Mean total execution time with 95% CI
# ============================================================

def figure_total_time_ci(summary):

    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    x = np.arange(
        len(PAYLOAD_ORDER)
    )

    for mode in MODES:

        subset = (
            summary[
                summary["mode"] == mode
            ]
            .set_index("payload_bytes")
            .loc[PAYLOAD_ORDER]
        )

        means = subset[
            "mean_total_ms"
        ].values

        lower = (
            means
            - subset[
                "ci95_low_total_ms"
            ].values
        )

        upper = (
            subset[
                "ci95_high_total_ms"
            ].values
            - means
        )

        yerr = np.vstack(
            [lower, upper]
        )

        ax.errorbar(
            x,
            means,
            yerr=yerr,
            marker="o",
            capsize=3,
            label=mode
        )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [
            PAYLOAD_LABELS[p]
            for p in PAYLOAD_ORDER
        ]
    )

    ax.set_xlabel(
        "Payload Size"
    )

    ax.set_ylabel(
        "Mean Total Execution Time (ms)"
    )

    ax.set_title(
        "Mean Total Execution Time with 95% Confidence Intervals"
    )

    ax.grid(
        True,
        alpha=0.25
    )

    ax.legend(
        title="Configuration"
    )

    fig.tight_layout()

    save_figure(
        fig,
        "figure_total_time_95CI"
    )


# ============================================================
# FIGURE 2
# Encryption throughput
# ============================================================

def figure_throughput(summary):

    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    x = np.arange(
        len(PAYLOAD_ORDER)
    )

    for mode in MODES:

        subset = (
            summary[
                summary["mode"] == mode
            ]
            .set_index("payload_bytes")
            .loc[PAYLOAD_ORDER]
        )

        throughput = subset[
            "mean_throughput_MBps"
        ].values

        ax.plot(
            x,
            throughput,
            marker="o",
            label=mode
        )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [
            PAYLOAD_LABELS[p]
            for p in PAYLOAD_ORDER
        ]
    )

    ax.set_xlabel(
        "Payload Size"
    )

    ax.set_ylabel(
        "Mean Encryption Throughput (MB/s)"
    )

    ax.set_title(
        "Encryption Throughput across Payload Sizes"
    )

    ax.grid(
        True,
        alpha=0.25
    )

    ax.legend(
        title="Configuration"
    )

    fig.tight_layout()

    save_figure(
        fig,
        "figure_encryption_throughput"
    )


# ============================================================
# FIGURE 3
# Coefficient of variation
# ============================================================

def figure_cv(summary):

    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    x = np.arange(
        len(PAYLOAD_ORDER)
    )

    for mode in MODES:

        subset = (
            summary[
                summary["mode"] == mode
            ]
            .set_index("payload_bytes")
            .loc[PAYLOAD_ORDER]
        )

        cv = subset[
            "cv_total_percent"
        ].values

        ax.plot(
            x,
            cv,
            marker="o",
            label=mode
        )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [
            PAYLOAD_LABELS[p]
            for p in PAYLOAD_ORDER
        ]
    )

    ax.set_xlabel(
        "Payload Size"
    )

    ax.set_ylabel(
        "Coefficient of Variation (%)"
    )

    ax.set_title(
        "Execution-Time Variability across Configurations"
    )

    ax.grid(
        True,
        alpha=0.25
    )

    ax.legend(
        title="Configuration"
    )

    fig.tight_layout()

    save_figure(
        fig,
        "figure_total_time_CV"
    )


# ============================================================
# FIGURES 4-10
# Boxplot for each payload size
# ============================================================

def figure_boxplots(raw):

    for payload in PAYLOAD_ORDER:

        subset = raw[
            raw["payload_bytes"]
            == payload
        ]

        data = [
            subset[
                subset["mode"] == mode
            ]["total_ms"].values
            for mode in MODES
        ]

        fig, ax = plt.subplots(
            figsize=(7.5, 5)
        )

       ax.boxplot(
         data,
          tick_labels=MODES,
          showmeans=True
        )

        ax.set_xlabel(
            "Configuration"
        )

        ax.set_ylabel(
            "Total Execution Time (ms)"
        )

        label = PAYLOAD_LABELS[
            payload
        ]

        ax.set_title(
            f"Distribution of Total Execution Time at {label}"
        )

        ax.grid(
            True,
            axis="y",
            alpha=0.25
        )

        fig.tight_layout()

        safe_label = (
            label
            .replace(" ", "_")
            .lower()
        )

        save_figure(
            fig,
            f"boxplot_total_time_{safe_label}"
        )


# ============================================================
# FIGURE 11
# Communication overhead percentage
# ============================================================

def figure_overhead_percentage(summary):

    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    x = np.arange(
        len(PAYLOAD_ORDER)
    )

    for mode in MODES:

        subset = (
            summary[
                summary["mode"] == mode
            ]
            .set_index("payload_bytes")
            .loc[PAYLOAD_ORDER]
        )

        overhead_percentage = (
            subset["overhead_bytes"].values
            /
            np.array(PAYLOAD_ORDER)
        ) * 100

        ax.plot(
            x,
            overhead_percentage,
            marker="o",
            label=mode
        )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [
            PAYLOAD_LABELS[p]
            for p in PAYLOAD_ORDER
        ]
    )

    ax.set_xlabel(
        "Payload Size"
    )

    ax.set_ylabel(
        "Communication Overhead (%)"
    )

    ax.set_title(
        "Communication Overhead Relative to Payload Size"
    )

    ax.set_yscale(
        "log"
    )

    ax.grid(
        True,
        alpha=0.25
    )

    ax.legend(
        title="Configuration"
    )

    fig.tight_layout()

    save_figure(
        fig,
        "figure_communication_overhead_percentage"
    )


def main():

    print("=" * 70)
    print(
        "PAH-PQC PUBLICATION FIGURE GENERATOR"
    )
    print("=" * 70)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    raw = pd.read_csv(
        RAW_FILE
    )

    summary = pd.read_csv(
        SUMMARY_FILE
    )

    validate_data(
        raw,
        summary
    )

    figure_total_time_ci(
        summary
    )

    figure_throughput(
        summary
    )

    figure_cv(
        summary
    )

    figure_boxplots(
        raw
    )

    figure_overhead_percentage(
        summary
    )

    print("\n")
    print("=" * 70)
    print(
        "ALL FIGURES GENERATED SUCCESSFULLY"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
