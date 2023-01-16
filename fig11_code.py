import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from utils import data_folder, image_folder

warnings.filterwarnings("ignore")

## Matplotlib formatting
plt.style.use("ggplot")
plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "text.usetex": False,
        "font.size": 10,
        "font.weight": "normal",
        "figure.titlesize": "medium",
        "xtick.color": "black",
        "ytick.color": "black",
        "axes.labelcolor": "black",
        "text.color": "black",
        "savefig.dpi": 300,
        "figure.dpi": 100,
    }
)

if __name__ == "__main__":
    start_time = time.time()

    # Read in the data
    df = pd.read_csv(f"{data_folder}/fig11_data.csv")

    c1_df = df[df["cluster"] == 1].copy()
    c5_df = df[df["cluster"] == 5].copy()

    x = np.arange(4, 44, 1)

    c1_mean = c1_df["mean_perc_fully_vaccinated"].values
    c1_std = c1_df["std_perc_fully_vaccinated"].values
    c5_mean = c5_df["mean_perc_fully_vaccinated"].values
    c5_std = c5_df["std_perc_fully_vaccinated"].values

    fig, axes = plt.subplots(figsize=(8, 4), nrows=2, sharex=True)
    axes[0].fill_between(
        x, c1_mean - c1_std * 0.5, c1_mean + c1_std * 0.5, color="#3B4BC0", alpha=0.2
    )
    axes[0].plot(x, c1_mean, "-o", c="#3B4BC0", lw=1, ms=4, mfc="#E5E5E5")

    axes[0].annotate(
        "C1",
        (x[-1] + 0.5, c1_mean[-1]),
        ha="left",
        va="center",
        c="#3B4BC0",
        weight="bold",
    )

    axes[0].fill_between(
        x, c5_mean - c5_std * 0.5, c5_mean + c5_std * 0.5, color="#B30326", alpha=0.2
    )
    axes[0].plot(x, c5_mean, "-o", c="#B30326", lw=1, ms=4, mfc="#E5E5E5")
    axes[0].annotate(
        "C5",
        (x[-1] + 0.5, c5_mean[-1]),
        ha="left",
        va="center",
        c="#B30326",
        weight="bold",
    )

    #########################
    # Plot the second panel #
    #########################

    c1_vh_mean = c1_df["mean_VHb"].values
    c1_vh_std = c1_df["std_VHb"].values
    c5_vh_mean = c5_df["mean_VHb"].values
    c5_vh_std = c5_df["std_VHb"].values

    axes[1].fill_between(
        x, c1_vh_mean - c1_vh_std, c1_vh_mean + c1_vh_std, color="#3B4BC0", alpha=0.2
    )
    axes[1].plot(x, c1_vh_mean, "-o", c="#3B4BC0", lw=1, ms=4, mfc="#E5E5E5")
    axes[1].annotate(
        "C1",
        (x[-1] + 0.5, c1_vh_mean[-1]),
        ha="left",
        va="center",
        c="#3B4BC0",
        weight="bold",
    )

    axes[1].fill_between(
        x, c5_vh_mean - c5_vh_std, c5_vh_mean + c5_vh_std, color="#B30326", alpha=0.2
    )
    axes[1].plot(x, c5_vh_mean, "-o", c="#B30326", lw=1, ms=4, mfc="#E5E5E5")
    axes[1].annotate(
        "C5",
        (x[-1] + 0.5, c5_vh_mean[-1]),
        ha="left",
        va="center",
        c="#B30326",
        weight="bold",
    )

    show_week_labels = c1_df[c1_df["week_number"].isin(np.arange(5, 41, 5))][
        "w_month_year"
    ].to_list()
    axes[1].set_xticklabels([0] + show_week_labels)

    axes[0].set_ylabel("Percentage of People\n Fully Vaccinated")
    axes[1].set_xlabel("Week Number")
    axes[1].set_ylabel(r"VH$^b$")

    legend_elements = [
        Line2D(
            [0], [0], marker="o", c="#B30326", label=r"$\mu$", mfc="w", markersize=5
        ),
        Patch(facecolor="#B30326", alpha=0.2, label=r"$\pm 1 \sigma$"),
    ]

    axes[0].legend(
        handles=legend_elements,
        ncol=2,
        loc="upper left",
        fancybox=False,
        shadow=False,
        facecolor="white",
    )

    # Export the figure
    plt.savefig(f"{image_folder}/(fig11)VH_c1_c5.png", dpi=300, bbox_inches="tight")

    print(
        f"--- Finished exporting figure 11, took {time.time() - start_time:,.2f} seconds ---"
    )
