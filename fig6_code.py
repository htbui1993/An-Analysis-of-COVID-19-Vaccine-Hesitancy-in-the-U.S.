import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

from utils import data_folder, image_folder

if __name__ == "__main__":
    start_time = time.time()

    # Read in the data
    corr_matrix = pd.read_csv(f"{data_folder}/fig6_data.csv", index_col=0)

    # Format the tick labels with the format <Feature Name - Fx>
    tick_texts = [f"<i>{i}</i> - F{idx}" for idx, i in enumerate(corr_matrix.columns, 1)]
    
    fig = px.imshow(
        corr_matrix,
        color_continuous_scale=px.colors.sequential.RdBu,
        color_continuous_midpoint=0,
        zmin=-1,
        zmax=1,
        width=500,
        height=380,
        aspect="equal",
    )

    for i, text in enumerate(tick_texts):
        fig.add_annotation(
            x=i,
            y=i,
            text=f"{text}",
            showarrow=False,
            xref="x",
            yref="y",
            xanchor="right",
            xshift=-10,
            yshift=0,
            font_size=12,
        )

    fig.update_traces(xgap=1, ygap=1)

    fig.update_layout(
        font=dict(family="Times New Roman", size=12, color="black"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(
            tickangle=-90,
            side="top",
            dtick=1,
            tickvals=np.arange(21),
            ticktext=[f"F{i}" for i in np.arange(1, 22)],
        ),
        yaxis=dict(
            dtick=1,
            showticklabels=False,
        ),
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
        coloraxis=dict(
            colorbar=dict(
                dtick=0.25,
                ticklabelposition="outside",
                ticks="outside",
                orientation="v",
                len=0.82,
                thickness=10,
                outlinewidth=1,
                xpad=10,
                ypad=0,
                title_side="right",
            )
        ),
    )

    # Export the figure
    fig.write_image(f"{image_folder}/(fig6)corr_heatmap.png", scale=4)
    print(
        f"--- Finished exporting figure  6, took {time.time() - start_time:,.2f} seconds ---"
    )
