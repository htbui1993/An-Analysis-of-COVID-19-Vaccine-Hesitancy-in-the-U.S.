import textwrap
import time

import pandas as pd
import plotly.graph_objs as go
from mycolorpy import colorlist as mcp

from utils import data_folder, image_folder

dynamic_factors = [
    "Google Symptom Search",
    "Weather",
    "Google Search Insights",
    "Unemployment Rate",
    "Online News Article Tone",
    "% Change in Deceased",
    "% Change in Confirmed Cases",
    "Google Symptom Search 1",
    "Google Symptom Search 2",
    "Google Symptom Search 3",
    "Google Symptom Search 4",
    "Google Symptom Search 5",
    "Weather 1",
    "Weather 2",
    "Weather 3",
    "Stringency Index",
    "Sentiment from Tweets (Topic 1)",
    "Sentiment from Tweets (Topic 2)",
    "Sentiment from Tweets (Topic 3)",
    "Sentiment from Tweets (Topic 4)",
    "Sentiment from Tweets (Topic 5)",
    "Sentiment from Tweets (Topic 6)",
    "Sentiment from Tweets (Topic 7)",
    "Sentiment from Tweets (Topic 8)",
    "Sentiment from Tweets (Topic 9)",
    "Sentiment from Tweets (Topic 10)",
    "Sentiment from Tweets (Topic 11)",
    "Sentiment from Tweets (Topic 12)",
    "Sentiment from Tweets (Topic 13)",
    "Sentiment from Tweets (Topic 14)",
    "Sentiment from Tweets (Topic 15)",
    "Sentiment from Tweets (Topic 16)",
    "Sentiment from Tweets (Topic 17)",
]


if __name__ == "__main__":
    start_time = time.time()

    # Read in the data
    permut_df = pd.read_csv(f"{data_folder}/fig7_data_permut.csv")
    shap_df = pd.read_csv(f"{data_folder}/fig7_data_shap.csv")

    # Create the figure
    fig = go.Figure(
        layout=dict(
            template="ggplot2",
            margin=dict(l=50, r=10, b=50, t=30, pad=0),
            font=dict(family="Times New Roman", size=12, color="black"),
            height=500,
            width=800,
        )
    ).set_subplots(
        rows=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.3,
        subplot_titles=["a)", "b)"],
    )

    ##########################################
    # Add bar chart for permutation importance
    ##########################################

    for _, row in permut_df.iterrows():
        feature = row["feature"]
        format_name = "<br>".join(textwrap.wrap(feature, 20))
        if feature in dynamic_factors:
            format_name = f"<b><i>{format_name}</i></b>"
            pattern_shape = "\\"
        else:
            pattern_shape = ""
        fig.add_bar(
            x=[format_name],
            y=[row["relative_importance"]],
            marker=dict(
                color="grey",
                pattern_shape=pattern_shape,
                pattern_size=4,
                pattern_fgcolor="black",
                line=dict(color="black", width=0.5),
            ),
            showlegend=False,
            row=1,
            col=1,
        )

    ##########################################
    # Add bar chart for shapley importance
    ##########################################
    coolwarm_colors = mcp.gen_color(cmap="coolwarm", n=5)

    for idx, (_, row) in enumerate(shap_df.iterrows()):
        showlegend = True if idx == 0 else False
        feature = row["feature"]
        format_name = "<br>".join(textwrap.wrap(feature, 20))
        if feature in dynamic_factors:
            format_name = f"<b><i>{format_name}</i></b>"
            pattern_shape = "\\"
        else:
            pattern_shape = ""

        for idx2, (cluster_value, color) in enumerate(
            zip(
                [row["c1"], row["c2"], row["c3"], row["c4"], row["c5"]], coolwarm_colors
            ),
            1,
        ):
            fig.add_bar(
                x=[format_name],
                y=[cluster_value],
                marker=dict(
                    color=color,
                    pattern_shape=pattern_shape,
                    pattern_size=4,
                    pattern_fgcolor="black",
                    line=dict(color="black", width=0.5),
                ),
                name=f"C{idx2}",
                showlegend=showlegend,
                row=2,
                col=1,
            )

    # Update the layout
    xaxis_format = dict(
        categoryorder="total descending",
        tickangle=-45,
        ticklabelposition="outside left",
        tickfont=dict(size=11),
    )
    fig.update_layout(
        barmode="stack",
        bargap=0.2,
        xaxis=xaxis_format,
        yaxis=dict(title="Relative Permutation<br>Importance"),
        xaxis2=xaxis_format,
        yaxis2=dict(title="Average Impact on Cluster<br>Probabilities (SHAP)"),
        legend=dict(
            title="Cluster: ",
            traceorder="normal",
            orientation="h",
            xanchor="right",
            yanchor="top",
            y=0.35,
            x=0.995,
            bgcolor="rgba(0,0,0,0)",
        ),
    )

    # Export the figure
    fig.write_image(f"{image_folder}/(fig7)feature_importance.png", scale=4)
    print(
        f"--- Finished exporting figure  7, took {time.time() - start_time:,.2f} seconds ---"
    )
