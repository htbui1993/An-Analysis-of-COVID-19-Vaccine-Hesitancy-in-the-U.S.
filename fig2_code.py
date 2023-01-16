import time

import pandas as pd
import plotly.graph_objects as go

from utils import data_folder, image_folder


def get_state_above_below_threshold(df, state_pop, threshold):
    completeness_df = plot_df[
        (df["Date"] >= "2021-02-25")
        & (df["Date"] <= "2021-10-30")
        & (df["Day_of_Week"] == "Sunday")
    ]

    avg_completeness = (
        completeness_df.groupby(["Recip_State"])
        .agg({"Completeness_pct": "mean"})
        .reset_index()
    )
    avg_completeness = avg_completeness.merge(
        state_pop, left_on="Recip_State", right_on="state", how="left"
    )

    # Determine the states that have average completeness above or below the threshold
    above_threshold = avg_completeness[
        avg_completeness["Completeness_pct"] >= threshold
    ]
    below_threshold = avg_completeness[avg_completeness["Completeness_pct"] < threshold]

    return above_threshold, below_threshold


def plot_vaccination_rate_per_county(fig, df, state, col, row=2):
    df2 = df[df["Recip_State"] == state].copy()
    known_fips = df2[df2["Recip_County"] != "Unknown County"].sort_values(
        by=["Date", "Series_Complete_Yes"], ascending=[False, True]
    )
    unknown_fips = df2[df2["Recip_County"] == "Unknown County"].sort_values(
        by=["Date"], ascending=True
    )
    for county_name, county_info in known_fips.groupby("Recip_County"):
        fig.add_trace(
            go.Scatter(
                x=county_info["Date"],
                y=county_info["Series_Complete_Yes"],
                stackgroup="one",
                line=dict(width=1),
                name=county_name,
                marker_color="lightslategrey",
            ),
            col=col,
            row=row,
        )

    fig.add_trace(
        go.Scatter(
            x=unknown_fips["Date"],
            y=unknown_fips["Series_Complete_Yes"].multiply(-1),
            name="Unknown FIPS",
            marker_color="crimson",
            stackgroup="two",
        ),
        col=col,
        row=row,
    )

    return fig


def create_fig1(df, state_pop, threshold, state_list):

    above_threshold, below_threshold = get_state_above_below_threshold(
        df, state_pop, threshold
    )

    # Initialize the figure
    fig = go.Figure(
        layout=dict(
            template="ggplot2",
            margin=dict(l=0, r=10, b=60, t=20, pad=0),
            font=dict(family="Times New Roman", size=12, color="black"),
            height=400,
            width=800,
        )
    ).set_subplots(
        rows=2,
        cols=3,
        horizontal_spacing=0.1,
        specs=[[{"colspan": 3}, None, None], [{}, {}, {}]],
        row_heights=[0.6, 0.4],
        subplot_titles=["a)", f"b) {state_1}", f"c) {state_2}", f"d) {state_3}"],
    )

    # Add the scatter plot for the states that have average completeness above the threshold (i.e., black markers)
    fig.add_trace(
        go.Scatter(
            x=above_threshold["census_2019"],
            y=above_threshold["Completeness_pct"],
            marker=dict(color="black", size=4, opacity=1),
            text=above_threshold["state"],
            textfont_size=7,
            textposition="top center",
            mode="markers + text",
        ),
        col=1,
        row=1,
    )

    # Add the states that have average completeness below the threshold (i.e., red x markers)
    fig.add_trace(
        go.Scatter(
            x=below_threshold["census_2019"],
            y=below_threshold["Completeness_pct"],
            marker=dict(color="red", size=7, opacity=1),
            text=below_threshold["state"],
            textposition="middle right",
            textfont_color="red",
            marker_symbol="x",
            mode="markers+text",
        ),
        col=1,
        row=1,
    )

    # Add the red line to indicate the threshold
    fig.add_hline(
        y=threshold, line_width=1, line_dash="dot", opacity=1, line_color="crimson"
    )

    # Add the stacked line plot for the custom states
    for col, state in enumerate(state_list, 1):
        fig = plot_vaccination_rate_per_county(fig, df, state, col=col, row=2)

    fig.update_layout(
        xaxis=dict(title="Population (Log scale)", type="log", ticksuffix=""),
        xaxis2=dict(
            title="Date",
            tickformat="%b",
            ticksuffix="",
            dtick="M1",
            tickangle=45,
        ),
        xaxis3=dict(
            title="Date",
            tickformat="%b",
            ticksuffix="",
            dtick="M1",
            tickangle=45,
        ),
        xaxis4=dict(
            title="Date", range=["2021-10-10", "2021-10-31"], tickformat="%b %d"
        ),
        yaxis=dict(
            title="Average Completeness<br>(Week 4 - Week 43)", ticksuffix="%", dtick=20
        ),
        yaxis2=dict(title="Number of Fully<br>Vaccinated People", ticksuffix=""),
        showlegend=False,
    )

    return fig


if __name__ == "__main__":
    start_time = time.time()

    # Import Data
    plot_df = pd.read_csv(f"{data_folder}/fig2_data.csv", parse_dates=["Date"])
    state_pop = pd.read_csv(f"{data_folder}/fig2_pop.csv")

    # * Adjust for your need
    threshold = 80
    state_1, state_2, state_3 = "FL", "VA", "TX"

    # Create Figure 1
    fig1 = create_fig1(
        plot_df,
        state_pop,
        threshold,
        state_list=[state_1, state_2, state_3],
    )

    # Save the figure
    fig1.write_image(f"{image_folder}/(fig2)vaccination_rate_completeness.png", scale=4)

    print(
        f"--- Finished exporting figure  2, took {time.time() - start_time:,.2f} seconds ---"
    )
