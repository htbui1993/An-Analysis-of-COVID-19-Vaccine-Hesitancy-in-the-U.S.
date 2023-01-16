import colorsys
import time
from typing import List

import matplotlib.colors as mc
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objs as go
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import (
    FixedFormatter,
    FixedLocator,
    MultipleLocator,
    FormatStrFormatter,
)

from utils import data_folder, image_folder

## Matplotlib formatting
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

colors = [
    "#1F77B4",
    "#FF7F0E",
    "#2CA02C",
    "#D62728",
    "#9467BD",
    "#8C564B",
    "#E377C2",
    "#7F7F7F",
    "#BCBD22",
    "#17BECF",
]


def lighten_color(color, amount=0.5):
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def add_rect(w, ax, width=0.2):
    if type(w) == int:
        ax.add_patch(
            plt.Rectangle(
                xy=(w - width, 0), width=width * 2, height=100, color="grey", alpha=0.2
            )
        )
    else:
        from_x = w[0]
        to_x = w[1]
        ax.add_patch(
            plt.Rectangle(
                xy=(from_x - width, 0),
                width=to_x - from_x + width * 2,
                height=100,
                color="grey",
                alpha=0.2,
            )
        )
    return ax


if __name__ == "__main__":
    start_time = time.time()

    # Read in the data
    df = pd.read_csv(f"{data_folder}/fig10_data.csv")

    fig, axes = plt.subplots(ncols=3, figsize=(15, 5))

    for index, (alphabet, feature) in enumerate(
        zip(
            ["a)", "b)", "c)"],
            ["Political Affiliation", "Google Search Insights", "Stringency Index"],
        )
    ):
        temp = df[df["feature"] == feature].copy()
        x = temp["week_number"].to_list()
        y_c1 = temp["c1_ranking"].to_list()
        y_c5 = temp["c5_ranking"].to_list()
        c = colors[index]
        mfc = lighten_color(c, 0.5)
        axes[index].plot(x, y_c1, "-o", c=c, lw=1, mfc=mfc, ms=5)
        axes[index].plot(x, y_c5, "--X", c=c, lw=1, mfc=mfc, ms=5)

        axes[index].annotate(
            "C1", (x[-1] + 0.5, y_c1[-1]), ha="left", va="center", c=c, weight="bold"
        )
        axes[index].annotate(
            "C5", (x[-1] + 0.5, y_c5[-1]), ha="left", va="center", c=c, weight="bold"
        )

        axes[index] = add_rect(5, axes[index])
        axes[index] = add_rect(7, axes[index])
        axes[index] = add_rect([9, 15], axes[index])
        axes[index] = add_rect(42, axes[index])

        axes[index].set_title(f"{alphabet} {feature}")
        axes[index].set_xlabel("Week Numbers")
        axes[index].set_facecolor("white")

        axes[index].set_ylim(0.5, df[["c1_ranking", "c5_ranking"]].max().max() + 0.5)
        axes[index].xaxis.set_major_locator(MultipleLocator(5))
        axes[index].yaxis.set_major_locator(FixedLocator([1, 5, 10, 15, 20, 25]))
        axes[index].yaxis.set_minor_locator(MultipleLocator(1))
        axes[index].yaxis.set_minor_formatter(FormatStrFormatter("%.0f"))

        axes[index].grid(axis="x", color="lightgrey")
        axes[index].grid(axis="y", which="major", color="lightgrey")

        if index == 0:
            axes[index].set_ylabel("Ranking")
            axes[index].tick_params(axis="y", which="minor", labelsize=8)
            axes[index].spines["left"].set_color("grey")
        else:
            axes[index].tick_params(
                axis="y",
                which="both",
                length=0,
                width=0,
                labelsize=0,
                color="white",
                left=False,
            )

        axes[index].invert_yaxis()
        axes[index].spines["bottom"].set_color("grey")

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            c="k",
            ms=7,
            label="Ranking of feature for C1",
            mfc="lightgrey",
        ),
        Line2D(
            [0],
            [0],
            marker="X",
            c="k",
            ms=7,
            label="Ranking of feature for C5",
            mfc="lightgrey",
        ),
        Patch(facecolor="grey", alpha=0.2, label=r"5-CV F1 score $\leq$ 0.6"),
    ]

    axes[0].legend(
        handles=legend_elements,
        ncol=1,
        loc="lower right",
        fancybox=True,
        frameon=True,
        shadow=False,
        facecolor="white",
        edgecolor="k",
    )

    # Export the figure
    plt.savefig(
        f"{image_folder}/(fig10)ranking_3_factors.png", dpi=300, bbox_inches="tight"
    )
    print(
        f"--- Finished exporting figure 10, took {time.time() - start_time:,.2f} seconds ---"
    )
