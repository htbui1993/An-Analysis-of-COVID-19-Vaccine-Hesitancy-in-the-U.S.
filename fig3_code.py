import time
import warnings

import geopandas as gpd
import libpysal as lps
import mapclassify as mc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from utils import data_folder, image_folder

warnings.filterwarnings("ignore", category=UserWarning)

## Matplotlib formatting
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
plt.style.use("ggplot")

def get_surrounded_counties(df, wq, fips, verbose=True):
    lookup_idx = df[df["fips"] == fips].index.values[0]
    sur_df = df[df.index.isin(wq[lookup_idx].keys())]
    sur_fips = sur_df["fips"].values
    sur_counties = sur_df["county"].values
    print(f"+ Surrounded counties: {', '.join(sur_counties)}") if verbose else None
    return (sur_df, sur_fips)


def fill_missing_values_with_mean(df, wq, missing_fips, fill_col):
    sur_df, _ = get_surrounded_counties(df, wq, missing_fips, verbose=False)
    mean_val = sur_df[fill_col].mean()
    return mean_val


def fill_missing_VHb_surrounding_counties(df, wq, verbose=True):
    """Fill missing VHb values with mean of surrounding counties

    Args:
        df (DataFrame): Dataframe with VHb values
        wq (Weight Object): Weight object
        verbose (bool, optional): Print some statement. Defaults to True.

    Returns:
        DataFrame: Dataframe with filled VHb values
    """
    missing_fips_list = df[df["VHb"].isna()]["fips"].to_list()
    count_before = len(missing_fips_list)
    for fips in missing_fips_list:
        fips_idx = df[df["fips"] == fips].index[0]
        fill_val = fill_missing_values_with_mean(df, wq, fips, "VHb")
        df.loc[fips_idx, "VHb"] = fill_val

    count_after = df[df["VHb"].isna()].shape[0]
    print(
        f"+ Number of counties with missing VHb: {count_before} => {count_after}."
    ) if verbose else None
    return df


def get_wq(gdf):
    wq = lps.weights.Rook.from_dataframe(gdf, silence_warnings=True)
    wq.transform = "r"
    return wq


def cal_gvf(df, val_col="VHb", clu_col="cluster", verbose=True):
    """Calculate the goodness of variance fit for a given dataframe."""
    temp = df.copy()
    temp["mean_all"] = temp[val_col].mean()
    temp["mean_cluster"] = temp.groupby(clu_col)[val_col].transform("mean")
    temp["square_deviation_all"] = (temp[val_col] - temp["mean_all"]) ** 2
    temp["square_deviation_cluster"] = (temp[val_col] - temp["mean_cluster"]) ** 2
    sdam = temp["square_deviation_all"].sum()
    sdcm = temp["square_deviation_cluster"].sum()
    gvf = (sdam - sdcm) / sdam
    if verbose:
        print(f"- Sum of Squared Deviations for Array Mean (SDAM): {sdam:,.2f}")
        print(f"- Sum of Squared Deviations for Cluster Means (SDCM): {sdcm:,.2f}")
        print(f"- Goodness of Variance Fit (GVF): {gvf:,.2f}")

    return gvf


def get_clusters_FJ(df, wq, k=5):
    df = fill_missing_VHb_surrounding_counties(df, wq, verbose=False)
    VHb_stats = df["VHb"].describe()
    df["VHb_lag"] = lps.weights.lag_spatial(wq, df["VHb"])
    df["VHb"] = df["VHb"].clip(
        lower=VHb_stats["mean"] - 3 * VHb_stats["std"],
        upper=VHb_stats["mean"] + 3 * VHb_stats["std"],
    )

    VHb_lag_stats = df["VHb_lag"].describe()
    df["VHb_lag"] = df["VHb_lag"].clip(
        lower=VHb_lag_stats["mean"] - 3 * VHb_lag_stats["std"],
        upper=VHb_lag_stats["mean"] + 3 * VHb_lag_stats["std"],
    )

    classifier_normal = mc.FisherJenks(df["VHb"], k=k)
    classifier_lag = mc.FisherJenks(df["VHb_lag"], k=k)
    df["cluster"] = classifier_normal.yb
    df["cluster_lag"] = classifier_lag.yb
    return df


if __name__ == "__main__":
    start_time = time.time()

    # Load the datasets
    counties = gpd.read_file(
        f"{data_folder}/fig3_county_shapes.geojson", dtype={"FIPS": str}
    )
    wq = get_wq(counties)

    vh_df = pd.read_csv(
        f"{data_folder}/fig3_data.csv",
        usecols=["fips", "VHb"],
        dtype={"fips": str},
    )

    # Extract the data for the given week
    plot_df = pd.merge(
        counties[["fips", "geometry", "state", "county"]],
        vh_df[["fips", "VHb"]],
        on="fips",
        how="left",
    )

    # Create figure objects
    fig = plt.figure(constrained_layout=True, figsize=(9, 8))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 3])
    # Create sub plots as grid
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    ######## First Subplot ########
    # Calculate GVF for different number of clusters
    xs, ys = [], []
    for k in range(2, 15):
        cluster_df = get_clusters_FJ(plot_df, wq, k=k)
        GVF = cal_gvf(cluster_df, val_col="VHb", clu_col="cluster", verbose=False)
        xs.append(k)
        ys.append(GVF)
        
    ax1.axvline(x=5, color="red", linestyle="--", lw=1)
    ax1.plot(xs, ys, "ko-", mfc="white", mec="k", ms=8, lw=2)
    ax1.set_xlabel("Number of Clusters")
    ax1.set_xticks([2, 4, 5, 6, 8, 10, 12, 14])
    ax1.set_xticklabels(
        ["2", "4", r"$k$=5", "6", "8", "10", "12", "14"], fontweight="bold"
    )
    xticks = ax1.get_xticklabels()
    for i, xtick in enumerate(xticks):
        if i == 2:
            color, weight, size = "red", "bold", 10
        else:
            color, weight, size = "k", "normal", 10
        xtick.set_color(color)
        xtick.set_size(size)
        xtick.set_fontweight(weight)

    ax1.set_ylabel("Goodness of Variance Fit")
    ax1.set_title("a)", fontsize=15)
    ax1.xaxis.set_ticks_position("bottom")
    ax1.yaxis.set_ticks_position("left")
    ax1.xaxis.set_tick_params(width=1, length=4)
    ax1.yaxis.set_tick_params(width=1, length=4)
    ax1.set_ylim([0.65, 1.05])

    ######## Second Subplot ########
    n, bins, patches = ax2.hist(
        x=plot_df["VHb"].values,
        bins=50,
        density=False,
        color="y",
        align="mid",
        ec="k",
        lw=0.4,
        alpha=0.8,
    )

    classifier = mc.FisherJenks(plot_df["VHb"], k=5)
    bins = np.insert(classifier.bins, 0, plot_df["VHb"].min(), axis=0)

    # Add vertical lines for the bins in the second subplot
    for bin in bins:
        ax2.axvline(x=bin, color="red", linestyle="--", linewidth=1)
        ax2.annotate(
            text=f"{bin:.3f}",
            xy=(bin, 220),
            textcoords="offset points",
            xytext=(2, 0),
            ha="left",
            fontsize=9,
        )

    ax2.set_title("b)", fontsize=15)
    ax2.set_xlabel(r"VH$^b$")
    ax2.set_ylabel("Count")
    ax2.xaxis.set_ticks_position("bottom")
    ax2.set_xticks([0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 1.00])
    ax2.set_xticklabels(
        [r"$\leq$0.86", "0.88", "0.90", "0.92", "0.94", "0.96", "0.98", "1.00"], size=10
    )
    ax2.set_ylim([0, 240])
    ax2.yaxis.set_ticks_position("left")
    ax2.xaxis.set_tick_params(width=1, length=4, gridOn=False)
    ax2.yaxis.set_tick_params(width=1, length=4)

    ######## Third Subplot - Map ########
    plot_df.plot(
        column="VHb",
        cmap="coolwarm",
        scheme="fisherjenks",
        k=5,
        linewidth=0.2,
        ax=ax3,
        edgecolor="black",
        legend=True,
        legend_kwds={
            "loc": "lower right",
            "interval": True,
            "fontsize": 9,
            "fancybox": True,
        },
    )
    legends = ax3.get_legend()
    legend_texts = legends.get_texts()
    for i, text in enumerate(legend_texts):
        text.set_text(f"Cluster {i+1}: {text.get_text()}")
    ax3.set_title("c)", fontsize=15)
    ax3.axis("off")

    # Export the figure
    # fig.tight_layout()
    plt.savefig(f"{image_folder}/(fig3)GVF_map.png", dpi=300, bbox_inches="tight")
    print(f"--- Finished exporting figure  3, took {time.time() - start_time:,.2f} seconds ---")
