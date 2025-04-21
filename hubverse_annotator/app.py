"""
A streamlit application for that loads hubverse formatted
tables and plots model forecasts for the user to compare
and annotate models.

To run: poetry run streamlit run app.py
"""

import datetime as dt
import json
import logging
import os
import time

import altair as alt
import forecasttools
import pandas as pd
import polars as pl
import streamlit as st


def create_forecast_chart(
    hubverse_table: pl.DataFrame, reference_date: str
) -> alt.Chart:
    """
    Ingests a hubverse table (polars) and a reference_date string,
    pivots quantiles wide, melts back long, then creates a
    stacked‐area (streamgraph‐style) chart where X is the discrete
    set of target_end_date weeks.
    """
    wide = hubverse_table.pivot(
        values="value",
        index=["model", "target_end_date", "reference_date"],
        on="output_type_id",
    ).sort(by=["model", "target_end_date", "reference_date"])
    pdf = wide.to_pandas().reset_index()
    pdf["target_end_date"] = pd.to_datetime(pdf["target_end_date"])
    qcols = [
        c
        for c in pdf.columns
        if isinstance(c, float)
        or (isinstance(c, str) and c.replace(".", "", 1).isdigit())
    ]

    long_df = pdf.melt(
        id_vars=["model", "target_end_date", "reference_date"],
        value_vars=qcols,
        var_name="quantile",
        value_name="value",
    )
    long_df["date_str"] = long_df["target_end_date"].dt.strftime("%Y-%m-%d")
    long_df["quantile_num"] = long_df["quantile"].astype(float)
    long_df["opacity"] = 1 - (long_df["quantile_num"] - 0.5).abs() * 2
    unique_dates = list(long_df["date_str"].unique())
    sel = alt.selection_point(fields=["model"], bind="legend")
    chart = (
        alt.Chart(long_df)
        .mark_area(interpolate="linear")
        .encode(
            x=alt.X(
                "date_str:O",
                title="Target End Date (weekly)",
                sort=unique_dates,
                axis=alt.Axis(labelAngle=-45),
            ),
            y=alt.Y("value:Q", stack="center", title="Forecast Value"),
            color=alt.Color(
                "quantile:O", title="Quantile", scale=alt.Scale(scheme="blues")
            ),
            facet=alt.Facet("model:N", columns=1, title=None),
            opacity=alt.Opacity("opacity:Q", legend=None),
        )
        .add_params(sel)
        .properties(
            width=600,
            height=100 * long_df["model"].nunique(),
        )
        .interactive()
    )
    return chart


def main() -> None:
    # initiate logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    # record start time
    start_time = time.time()
    # begin streamlit application
    st.title("Forecast Annotator")
    uploaded_file = st.file_uploader(
        "Upload Hubverse File", type=["csv", "parquet"]
    )
    # load the hubverse data
    if uploaded_file is not None:
        if uploaded_file.name.endswith("parquet"):
            smhub_table = pl.read_parquet(uploaded_file)
        else:
            smhub_table = pl.read_csv(uploaded_file)
        logger.info(f"Uploaded file:\n{uploaded_file.name}")
        logger.info(f"Contents\n:{smhub_table}")
        # two-column layout for reference date and location
        col1, col2 = st.columns(2)
        with col1:
            ref_dates_available = (
                smhub_table["reference_date"].unique().to_list()
            )
            ref_dates_as_str = [
                dt.datetime.strftime(ref_d, "%Y-%m-%d")
                for ref_d in ref_dates_available
            ]
            selected_ref_date = st.selectbox(
                "Reference Date",
                options=ref_dates_as_str,
                key="ref_date_selection",
            )
        with col2:
            # get locations from forecasttools, some might be excluded from the
            # hubverse table, though
            locations_available = forecasttools.location_table[
                "long_name"
            ].to_list()
            location = st.selectbox(
                "Location",
                options=locations_available,
            )
        # get location abbreviation
        two_num_loc_abbr = forecasttools.location_lookup(
            location_vector=[location], location_format="long_name"
        )["location_code"].item()
        two_letter_loc_abbr = forecasttools.location_lookup(
            location_vector=[location], location_format="long_name"
        )["short_name"].item()
        # filter to location before filtering to model
        smhub_table = smhub_table.filter(
            pl.col("location") == two_num_loc_abbr,
        )
        # models and targets available
        models_available = smhub_table["model"].unique().to_list()
        selected_models = st.multiselect(
            "Model(s)",
            options=models_available,
            key="model_selection",
            default=models_available,
        )
        targets_available = smhub_table["target"].unique().to_list()
        selected_target = st.selectbox(
            "Target(s)", options=targets_available, key="target_selection"
        )
        # filter hubverse table by selected models and target
        smhub_table = smhub_table.filter(
            pl.col("model").is_in(selected_models),
            pl.col("target") == selected_target,
        )

        st.markdown(f"## Forecasts For: {two_num_loc_abbr}")
        st.markdown(f"## Reference Date: {selected_ref_date}")
        # plotting of the selected model, target, location, and reference date
        forecast_chart = create_forecast_chart(smhub_table, selected_ref_date)
        st.altair_chart(forecast_chart, use_container_width=True)

        # preference and comments saving
        annotations_file = f"anno_{selected_ref_date}.json"
        if os.path.exists(annotations_file):
            with open(annotations_file) as f:
                annotations = json.load(f)
            logger.info(f"Annotations file created:\n{annotations_file}")
        else:
            annotations = {}
        # save by location, with empty dict by default
        by_loc_dict = annotations.setdefault(two_num_loc_abbr, {})
        for model in selected_models:
            st.markdown(f"### {model}")
            # set status and comments keys, and get previous in json
            status_key = f"status_{two_letter_loc_abbr}_{model}"
            comment_key = f"comment_{two_letter_loc_abbr}_{model}"
            prev = by_loc_dict.get(model, {})
            default_status = prev.get("status", "None")
            default_comment = prev.get("comment", "")
            # select boxes for status and comments
            status = st.selectbox(
                "Status",
                ["Preferred", "Omitted", "None"],
                index=["Preferred", "Omitted", "None"].index(default_status),
                key=status_key,
            )
            comment = st.text_input(
                "Comments", value=default_comment, key=comment_key
            )
            # save in dictionary and write out
            by_loc_dict[model] = {"status": status, "comment": comment}
        with open(annotations_file, "w") as f:
            json.dump(annotations, f, indent=2)

        # export button
        if st.button("Export forecasts"):
            col1, col2, col3 = st.columns([1, 3, 1])
            with col2:
                st.success("Need export")

    # record end time
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"Session lasted around: {duration // 60} minutes.")


if __name__ == "__main__":
    main()
