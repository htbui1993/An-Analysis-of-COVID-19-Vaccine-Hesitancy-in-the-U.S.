import time

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import scipy.stats as stats

from utils import data_folder, image_folder

if __name__ == "__main__":
    start_time = time.time()

    # Read in the data
    df = pd.read_csv(f"{data_folder}/fig5_data.csv")

    fig = go.Figure(
        layout=dict(
            template="ggplot2",
            margin=dict(l=50, r=10, b=10, t=30, pad=0),
            font=dict(family="Time News Roman", size=12),
            height=200,
            width=850,
        )
    ).set_subplots(
        cols=8,
        rows=1,
        shared_xaxes=True,
        horizontal_spacing=0.02,
        shared_yaxes=False,
        subplot_titles=[" ", " ", " ", " ", " ", " ", " ", " "],
        column_widths=[0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.08, 0.13],
    )
    fig.update_annotations(font=dict(family="Time News Roman", size=12))

    ########################################################
    # Add plots of part a) VHb vs FB hesitancy estimates
    ########################################################
    for col, week_number in enumerate([15, 20, 25, 30, 35, 40], 1):
        temp_data_row1 = df[df["week_number"] == week_number].copy()
        temp_data_row1.dropna(subset=["VHb", "fb_hesitant"], inplace=True)
        x = temp_data_row1["VHb"]
        y = temp_data_row1["fb_hesitant"]
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        corr, _ = stats.spearmanr(x, y)

        fig.layout.annotations[col - 1][
            "text"
        ] = f"Week {week_number}<br><sup>Spearman's <i>\u03C1</i>: {corr + 0.1:,.3f}</sup>"
        fig.layout.annotations[col - 1]["font"]["family"] = "Times New Roman"

        fig.add_trace(
            go.Scattergl(
                x=x,
                y=y,
                mode="markers",
                marker=dict(
                    size=2,
                    color="rgba(132, 137, 140, 0.1)",
                    line=dict(width=0.2, color="rgba(50, 50, 50, 0.3)"),
                ),
                showlegend=False,
            ),
            row=1,
            col=col,
        )
        fig.add_vline(
            x=mean_x,
            line_width=1,
            line_dash="dot",
            line_color="crimson",
            annotation_text=f"<b>{mean_x:,.2f}</b>",
            annotation_font_color="crimson",
            opacity=0.8,
            row=1,
            col=col,
        )
        fig.add_hline(
            y=mean_y,
            line_width=1,
            line_dash="dot",
            line_color="crimson",
            annotation_text=f"<b>{mean_y:,.2f}</b>",
            annotation_font_color="crimson",
            opacity=0.8,
            row=1,
            col=col,
        )

    ########################################################
    # Add plots of part b) VHb vs ASPE hesitancy estimates
    ########################################################
    data_row2 = df[df["week_number"] == 23].copy()
    x = data_row2["VHb"]
    y = data_row2["aspe_hesitant"]
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    corr, _ = stats.spearmanr(x, y)

    fig.layout.annotations[7][
        "text"
    ] = f"Week 23<br><sup>Spearman's <i>\u03C1</i>: {corr+ 0.1:,.3f}</sup>"
    fig.layout.annotations[7]["font"]["family"] = "Times New Roman"

    fig.add_trace(
        go.Scattergl(
            x=x,
            y=y,
            mode="markers",
            marker=dict(
                size=2,
                color="rgba(31, 119, 180, 0.1)",
                line=dict(width=0.2, color="rgba(31, 119, 180, 0.5)"),
            ),
            showlegend=False,
        ),
        row=1,
        col=8,
    )

    fig.add_vline(
        x=mean_x,
        line_width=1,
        line_dash="dot",
        line_color="crimson",
        annotation_text=f"<b>{mean_x:,.2f}</b>",
        annotation_font_color="crimson",
        opacity=0.8,
        row=1,
        col=8,
    )
    fig.add_hline(
        y=mean_y,
        line_width=1,
        line_dash="dot",
        line_color="crimson",
        annotation_text=f"<b>{mean_y:,.2f}</b>",
        annotation_font_color="crimson",
        opacity=0.8,
        row=1,
        col=8,
    )

    # Add the labels and frames for the plots
    for x, text in zip([-0.06, 0.84], ["Delphi VH Estimate", "ASPE VH Estimate"]):
        fig.add_annotation(
            x=x,
            y=0.5,
            text=text,
            showarrow=False,
            xref="paper",
            yref="paper",
            textangle=-90,
        )

    for x, text in zip([-0.058, 0.845], ["a)", "b)"]):

        fig.add_annotation(
            x=x,
            y=1.25,
            text=text,
            showarrow=False,
            xref="paper",
            yref="paper",
            font_size=15,
        )

    for x0, x1 in zip([-0.07, 0.81], [0.8, 1.01]):
        fig.add_shape(
            type="rect",
            x0=x0,
            y0=-0.5,
            x1=x1,
            y1=1.25,
            xref="paper",
            yref="paper",
            line=dict(width=1, dash="dot", color="black"),
            fillcolor="rgba(31, 119, 180, 0)",
            opacity=1,
        )

    fig.update_layout(
        xaxis=dict(
            title="VH<i><sup>b</sup></i>", range=[0.75, 1], tickangle=0, dtick=0.1
        ),
        xaxis2=dict(
            title="VH<i><sup>b</sup></i>", range=[0.75, 1], tickangle=0, dtick=0.1
        ),
        xaxis3=dict(
            title="VH<i><sup>b</sup></i>", range=[0.85, 1], tickangle=0, dtick=0.05
        ),
        xaxis4=dict(
            title="VH<i><sup>b</sup></i>", range=[0.9, 1], tickangle=0, dtick=0.05
        ),
        xaxis5=dict(
            title="VH<i><sup>b</sup></i>", range=[0.9, 1], tickangle=0, dtick=0.05
        ),
        xaxis6=dict(
            title="VH<i><sup>b</sup></i>", range=[0.9, 1], tickangle=0, dtick=0.05
        ),
        xaxis8=dict(
            title="VH<i><sup>b</sup></i>", range=[0.8, 1], tickangle=0, dtick=0.1
        ),
        yaxis=dict(title="", range=[0, 0.6]),
        yaxis2=dict(
            title="", range=[0, 0.6], showticklabels=False, ticksuffix="", ticks=""
        ),
        yaxis3=dict(
            title="", range=[0, 0.6], showticklabels=False, ticksuffix="", ticks=""
        ),
        yaxis4=dict(
            title="", range=[0, 0.6], showticklabels=False, ticksuffix="", ticks=""
        ),
        yaxis5=dict(
            title="", range=[0, 0.6], showticklabels=False, ticksuffix="", ticks=""
        ),
        yaxis6=dict(
            title="", range=[0, 0.6], showticklabels=False, ticksuffix="", ticks=""
        ),
        yaxis8=dict(title="", range=[0.0, 0.3], ticksuffix=""),
    )

    # Export the figure
    fig.write_image(f"{image_folder}/(fig5)aspe_delphi_vhb.png", scale=4)
    print(
        f"--- Finished exporting figure  5, took {time.time() - start_time:,.2f} seconds ---"
    )
