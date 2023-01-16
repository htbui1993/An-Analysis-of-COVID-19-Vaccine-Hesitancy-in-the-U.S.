import time
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator
from matplotlib.lines import Line2D

from utils import data_folder, image_folder

# Matplotlib formatting
plt.style.use("ggplot")
plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "text.usetex": False,
        "font.size": 12,
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
    df = pd.read_csv(f"{data_folder}/fig8_data.csv")

    show_weeks = np.arange(5, 41, 5)
    show_week_labels = df[df["week_number"].isin(show_weeks)]["w_month_year"]

    x = df["week_number"]
    y1 = df["us_sni_covid19_vaccination"]
    y2 = df["VHb_mean"]

    # Create the figure with two y-axes
    fig, ax = plt.subplots(figsize=(10, 4))
    ax2 = ax.twinx()
    ax.plot(x, y1, "o-", mfc="#E5E5E5", c="#1F77B4", lw=1.5, mew=1.5)
    ax2.plot(x, y2, "o-", mfc="#E5E5E5", c="#FF7F0E", lw=1.5, mew=1.5)

    ax.set_xlabel("Week Number")
    ax.set_xlim(3.5, 43.5)
    ax.xaxis.set_major_locator(FixedLocator(show_weeks))
    ax.set_xticklabels(show_week_labels)
    ax.set_yticks(np.arange(30, 101, 10))
    ax.tick_params(axis="y", colors="#1F77B4")

    ax2.tick_params(axis="y", colors="#FF7F0E")
    ax2.grid(False)

    # Create custom legend
    legend_elements = []
    for color, label in zip(
        ["#1F77B4", "#FF7F0E"], ["Vaccination Search Insights", r"Average VH$^b$"]
    ):
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="o",
                ls="-",
                mfc="white",
                mew=1.5,
                lw=1.5,
                c=color,
                label=label,
            )
        )

    ax.legend(
        handles=legend_elements,
        ncol=1,
        loc="upper right",
        fancybox=False,
        shadow=False,
        facecolor="white",
        fontsize=10,
    )

    # Export the figure
    plt.savefig(
        f"{image_folder}/(fig8)searches_vs_vhb.png", dpi=300, bbox_inches="tight"
    )
    print(
        f"--- Finished exporting figure  8, took {time.time() - start_time:,.2f} seconds ---"
    )
