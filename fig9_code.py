import textwrap
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import shap
from matplotlib.gridspec import GridSpec

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

if __name__ == "__main__":
    start_time = time.time()

    # Read in the data
    shap_df = pd.read_csv(f"{data_folder}/fig9_data1.csv", index_col=0)
    value_df = pd.read_csv(f"{data_folder}/fig9_data2.csv")
    column_names_formatted = [textwrap.fill(text=i, width=40) for i in shap_df.columns]

    fig = plt.figure(figsize=([35, 20]))

    gs = GridSpec(2, 6, wspace=40, hspace=0.2)

    ax1 = plt.subplot(gs[0, :2])
    shap.summary_plot(
        shap_df.loc["c1"].values,
        value_df.values,
        feature_names=column_names_formatted,
        cmap=plt.get_cmap("Spectral"),
        alpha=0.8,
        plot_size=None,
        show=False,
        max_display=15,
        color_bar=False,
    )

    ax2 = plt.subplot(gs[0, 2:4])
    shap.summary_plot(
        shap_df.loc["c2"].values,
        value_df.values,
        feature_names=column_names_formatted,
        cmap=plt.get_cmap("Spectral"),
        alpha=0.8,
        plot_size=None,
        show=False,
        max_display=15,
        color_bar=False,
    )

    ax3 = plt.subplot(gs[0, 4:6])
    shap.summary_plot(
        shap_df.loc["c3"].values,
        value_df.values,
        feature_names=column_names_formatted,
        cmap=plt.get_cmap("Spectral"),
        alpha=0.8,
        plot_size=None,
        show=False,
        max_display=15,
        color_bar=False,
    )

    ax4 = plt.subplot(gs[1, 1:3])
    shap.summary_plot(
        shap_df.loc["c4"].values,
        value_df.values,
        feature_names=column_names_formatted,
        cmap=plt.get_cmap("Spectral"),
        alpha=0.8,
        plot_size=None,
        show=False,
        max_display=15,
        color_bar=False,
    )

    ax5 = plt.subplot(gs[1, 3:5])
    shap.summary_plot(
        shap_df.loc["c5"].values,
        value_df.values,
        feature_names=column_names_formatted,
        cmap=plt.get_cmap("Spectral"),
        alpha=0.8,
        plot_size=None,
        show=False,
        max_display=15,
        color_bar=False,
    )

    # Set the labels
    for ax, title in zip(
        [ax1, ax2, ax3, ax4, ax5],
        [
            "a) Cluster 1",
            "b) Cluster 2",
            "c) Cluster 3",
            "d) Cluster 4",
            "e) Cluster 5",
        ],
    ):
        ax.set_xlabel("SHAP Value", labelpad=0, fontsize=24)
        ax.set_title(title, fontsize=30)
        ax.tick_params(axis="both", which="major", labelsize=24)
        ax.tick_params(axis="y", labelcolor="black", pad=2, length=0)

    cmap = mpl.cm.Spectral
    norm = mpl.colors.Normalize(vmin=0, vmax=1)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.84, 0.12, 0.01, 0.76])
    cb = mpl.colorbar.ColorbarBase(
        ax=cbar_ax, cmap=cmap, norm=norm, orientation="vertical"
    )
    cb.set_label("Scaled Feature Value", labelpad=5, fontsize=30, rotation=90)
    cb.set_ticks([0, 0.5, 1])
    cb.set_ticklabels(["0 (Low)", "0.5", "1 (High)"])
    cb.ax.tick_params(labelsize=24)
    cb.outline.set_color("black")
    cb.outline.set_linewidth(1)
    cbar_ax.yaxis.set_label_position("left")

    # Export the figure
    plt.savefig(
        f"{image_folder}/(fig9)shap_values_all_clusters.png",
        dpi=200,
        bbox_inches="tight",
    )
    print(
        f"--- Finished exporting figure  9, took {time.time() - start_time:,.2f} seconds ---"
    )
