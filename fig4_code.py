import time
from typing import List
import pandas as pd
import plotly.graph_objs as go

from utils import data_folder, image_folder


def plot_county_annotation(
    fig: go.Figure,
    df: pd.DataFrame,
    axs: List[float],
    ays: List[float],
    xachors: List[str],
) -> go.Figure:
    # Add annotations for the counties that have the highest and lowest hesitant rates
    name = df["county"].values[0]
    state = df["state"].values[0]
    x1 = df["hesitant"].values[0]
    x2 = df["PFV"].values[0]
    y = df["VHb"].values[0]
    fig.add_annotation(
        x=x1,
        y=y,
        text=f"{name}, {state}",
        showarrow=True,
        arrowsize=1,
        xanchor=xachors[0],
        ax=axs[0],
        ay=ays[0],
        arrowhead=1,
        row=2,
        col=1,
    )
    fig.add_annotation(
        x=x2,
        y=y,
        text=f"{name}, {state}",
        showarrow=True,
        arrowsize=1,
        xanchor=xachors[1],
        ax=axs[1],
        ay=ays[1],
        arrowhead=1,
        row=2,
        col=3,
    )
    return fig


if __name__ == "__main__":
    start_time = time.time()

    # Determine the states to highlight
    states: List[str] = ["OH", "CA"]

    # Read in the data
    df = pd.read_csv(f"{data_folder}/fig4_data.csv")

    fig = go.Figure(
        layout=dict(
            template="ggplot2",
            height=400,
            width=800,
            margin=dict(l=0, r=0, b=0, t=20, pad=0),
            font=dict(family="Times New Roman", size=12, color="#000000"),
        )
    ).set_subplots(
        rows=2,
        cols=4,
        horizontal_spacing=0.04,
        vertical_spacing=0.3,
        row_heights=[0.5, 0.6],
        subplot_titles=[
            "a) Northeast",
            "b) Midwest",
            "c) South",
            "d) West",
            f"e) Counties in {states[0]} &amp; {states[1]}",
            f"f) Counties in {states[0]} &amp; {states[1]}",
        ],
        specs=[[{}, {}, {}, {}], [{"colspan": 2}, None, {"colspan": 2}, None]],
    )
    fig.update_annotations(font_size=12)

    # Create the plots in the 2nd row
    for col, region in enumerate(["Northeast", "Midwest", "South", "West"], 1):
        temp = df[df["region"] == region]
        temp_sorted = (
            temp.groupby("state")
            .agg({"hesitant": "median", "fips": "count"})
            .rename(columns={"fips": "num_counties"})
            .sort_values("hesitant", ascending=True)
            .reset_index()
        )
        for state in temp_sorted["state"].unique():
            temp_state = temp[temp["state"] == state]
            if state == states[0]:
                color, opacity = "#1F77B4", 1
            elif state == states[1]:
                color, opacity = "#DC143C", 1
            else:
                color, opacity = "#000000", 0.7
            fig.add_trace(
                go.Violin(
                    x=temp_state["hesitant"],
                    line=dict(width=1, color=color),
                    opacity=opacity,
                    name=state,
                    showlegend=False,
                ),
                row=1,
                col=col,
            )
    fig.update_traces(orientation="h", side="positive", width=1.8, points=False)

    # Create the plots in the 2nd row
    for state, color, symbol in zip(
        states, ["#1F77B4", "#DC143C"], ["circle", "diamond"]
    ):
        temp = df[df["state"] == state]
        x1 = temp["hesitant"]
        x2 = temp["PFV"]
        y = temp["VHb"]
        fig.add_trace(
            go.Scatter(
                x=x1,
                y=y,
                mode="markers",
                marker=dict(
                    size=6,
                    color=color,
                    symbol=symbol,
                    opacity=0.7,
                    line=dict(width=0.5, color="#000000"),
                ),
                name=state,
                showlegend=True,
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x2,
                y=y,
                mode="markers",
                marker=dict(
                    size=6,
                    color=color,
                    symbol=symbol,
                    opacity=0.7,
                    line=dict(width=0.5, color="#000000"),
                ),
                name=state,
                showlegend=False,
            ),
            row=2,
            col=3,
        )

    counties_state0 = df[df["state"] == states[0]]
    county_state0_max = counties_state0[
        counties_state0["VHb"] == counties_state0["VHb"].max()
    ]

    counties_state1 = df[df["state"] == states[1]]
    county_state1_min = counties_state1[
        counties_state1["VHb"] == counties_state1["VHb"].min()
    ]

    fig = plot_county_annotation(
        fig=fig,
        df=county_state0_max,
        axs=[-20, 20],
        ays=[-15, -15],
        xachors=["right", "left"],
    )
    fig = plot_county_annotation(
        fig=fig,
        df=county_state1_min,
        axs=[20, -20],
        ays=[15, 15],
        xachors=["left", "right"],
    )

    # Update the layout
    fig.update_layout(
        xaxis=dict(title="ASPE VH Estimate", range=[0, 0.3]),
        xaxis2=dict(title="ASPE VH Estimate", range=[0, 0.3]),
        xaxis3=dict(title="ASPE VH Estimate", range=[0, 0.3]),
        xaxis4=dict(title="ASPE VH Estimate", range=[0, 0.3]),
        xaxis5=dict(title="ASPE VH Estimate"),
        xaxis6=dict(title="% of Residents Fully Vaccinated"),
        yaxis=dict(
            title="State",
            showgrid=False,
            zeroline=False,
            ticksuffix="",
            tickfont_size=8,
            title_font_size=13,
            dtick=1,
        ),
        yaxis2=dict(
            showgrid=False, zeroline=False, ticksuffix="", tickfont_size=8, dtick=1
        ),
        yaxis3=dict(
            showgrid=False, zeroline=False, ticksuffix="", tickfont_size=8, dtick=1
        ),
        yaxis4=dict(
            showgrid=False, zeroline=False, ticksuffix="", tickfont_size=8, dtick=1
        ),
        yaxis5=dict(title="VH<i><sup>b</sup></i> (week 23)", tickfont_size=12),
        yaxis6=dict(title="", showticklabels=False, ticks="", tickfont_size=1),
        legend=dict(
            title="",
            orientation="h",
            yanchor="bottom",
            y=0.01,
            xanchor="left",
            x=0.302,
            bgcolor="white",
            bordercolor="#000000",
            borderwidth=1,
        ),
    )

    # Export the figure
    fig.write_image(f"{image_folder}/(fig4)hesitant_state.png", scale=4)
    print(
        f"--- Finished exporting figure  4, took {time.time() - start_time:,.2f} seconds ---"
    )
