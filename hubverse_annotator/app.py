"""
A streamlit application that loads hubverse formatted
tables and plots model forecasts for the user to compare
and annotate models.

To run: poetry run streamlit run app.py
"""

import json
import logging
import pathlib
import time

import altair as alt
import forecasttools
import polars as pl
import streamlit as st


def create_forecast_chart(
    hubverse_table: pl.DataFrame, reference_date: str
) -> alt.Chart:
    """
    Uses a hubverse table (polars) and a reference date to
    display forecast quantiles faceted by model.
    """
    df = hubverse_table.filter(
        pl.col("output_type") == "quantile"
    ).with_columns(
        [
            pl.col("output_type_id").cast(pl.Utf8).alias("quantile_str"),
            pl.col("output_type_id").alias("quantile_num"),
        ]
    )

    df = df.with_columns(
        [(1 - (pl.col("quantile_num") - 0.5).abs() * 2).alias("opacity")]
    )
    unique_dates = (
        df.select(pl.col("target_end_date").cast(pl.Utf8).unique().sort())
        .to_series()
        .to_list()
    )
    sel = alt.selection_point(fields=["model"], bind="legend")
    chart = (
        alt.Chart(df)
        .mark_area(interpolate="linear")
        .encode(
            x=alt.X(
                "target_end_date:T", sort=unique_dates, title="Target End Date"
            ),
            y=alt.Y("value:Q", stack="center", title="Forecast Value"),
            color=alt.Color("quantile_str:O", scale=alt.Scale(scheme="blues")),
            opacity=alt.Opacity("opacity:Q", legend=None),
            row=alt.Row("model:N", title=None),
        )
        .add_selection(sel)
        .properties(width=600, height=100 * df["model"].unique().len())
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
        ext = pathlib.Path(uploaded_file.name).suffix.lower()
        try:
            if ext == ".parquet":
                smhub_table = pl.read_parquet(uploaded_file)
            elif ext == ".csv":
                smhub_table = pl.read_csv(uploaded_file)
            else:
                raise ValueError(f"Unsupported file type: {ext}")
        except ValueError as e:
            st.error(str(e))
            st.stop()
        st.success(f"Loaded {uploaded_file.name} ({ext}).")
        logger.info(f"Uploaded file:\n{uploaded_file.name}")
        logger.info(f"Contents\n:{smhub_table}")
        # two-column layout for reference date and location
        col1, col2 = st.columns(2)
        with col1:
            ref_dates = (
                smhub_table["reference_date"]
                .unique()
                .sort()
                .dt.strftime("%Y-%m-%d")
                .to_list()
            )
            selected_ref_date = st.selectbox(
                "Reference Date",
                options=ref_dates,
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
        lookup_loc = forecasttools.location_lookup(
            location_vector=[location], location_format="long_name"
        )
        two_num_loc_abbr = lookup_loc["location_code"].item()
        two_letter_loc_abbr = lookup_loc["short_name"].item()
        # filter to location before filtering to model
        smhubt_by_loc = smhub_table.filter(
            pl.col("location") == two_num_loc_abbr,
        )
        # models and targets available
        models_available = smhubt_by_loc["model"].unique().to_list()
        selected_models = st.multiselect(
            "Model(s)",
            options=models_available,
            key="model_selection",
            default=models_available,
        )
        targets_available = smhubt_by_loc["target"].unique().to_list()
        selected_target = st.selectbox(
            "Target(s)", options=targets_available, key="target_selection"
        )
        # filter hubverse table by selected models and target
        smhubt_to_plot = smhubt_by_loc.filter(
            pl.col("model").is_in(selected_models),
            pl.col("target") == selected_target,
        )

        st.markdown(f"## Forecasts For: {two_num_loc_abbr}")
        st.markdown(f"## Reference Date: {selected_ref_date}")
        # plotting of the selected model, target, location, and reference date
        if smhubt_to_plot.is_empty():
            st.warning(f"No forecasts available for the location: {location}")
        else:
            forecast_chart = create_forecast_chart(
                smhubt_to_plot, selected_ref_date
            )
            st.altair_chart(forecast_chart, use_container_width=True)

        # preference and comments saving
        output_dir = pathlib.Path("../output")
        output_dir.mkdir(parents=True, exist_ok=True)
        annotations_file = output_dir / f"anno_{selected_ref_date}.json"
        if annotations_file.exists():
            with annotations_file.open("r") as f:
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
