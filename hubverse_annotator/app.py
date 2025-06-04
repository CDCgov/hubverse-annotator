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
import polars.selectors as cs
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_quantile_forecast_chart(
    hubverse_table: pl.DataFrame,
    value_col: str = "value",
) -> alt.Chart:
    """
    Uses a hubverse table (polars) and a reference date to
    display quantile forecasts faceted by model. The
    output_type of the hubverse table must therefore be
    'quantile'.
    """
    # filter to quantile only rows and ensure quantiles are str for pivot
    # also, pivot to wide, so quantiles ids are columns
    df_wide = (
        hubverse_table.filter(pl.col("output_type") == "quantile")
        .pivot(
            on="output_type_id",
            index=cs.exclude("output_type_id", value_col),
            values=value_col,
        )
        .with_columns(pl.col("0.5").alias("median"))
    )
    # create base Chart for altair errorbands
    base = alt.Chart(df_wide).encode(
        x=alt.X("target_end_date:T", title="Target End Date")
    )
    # create median line and CI bands
    median_line = base.mark_line(strokeWidth=2, interpolate="monotone").encode(
        y=alt.Y("median:Q", title=None)
    )
    band_90 = base.mark_errorband(opacity=0.2, interpolate="monotone").encode(
        y=alt.Y("0.05:Q", title="Forecast Value"),
        y2="0.95:Q",
    )
    band_IQR = base.mark_errorband(opacity=0.3, interpolate="monotone").encode(
        y=alt.Y("0.25:Q", title=None),
        y2="0.75:Q",
    )
    # compose line and bands into faceted chart
    chart = (
        (median_line + band_90 + band_IQR)
        .facet(row=alt.Row("model:N", title="Model"), columns=1)
        .resolve_scale("independent")
    ).interactive()
    return chart


def main() -> None:
    # record start time
    start_time = time.time()
    # begin streamlit application
    with st.columns(3)[1]:
        st.header("Forecast Annotator")
    # columns for super mega hubverse table & observations (E & H)
    e_and_h_col, smht_col = st.columns(2)
    # super-mega hubverse table upload
    with e_and_h_col:
        e_and_h_file = st.file_uploader(
            "Upload Hubverse Target Data", type=["parquet"]
        )
    # hubverse timeseries table upload
    with smht_col:
        smht_file = st.file_uploader(
            "Upload Hubverse Forecasts", type=["csv", "parquet"]
        )
    # load the target data
    eh_table = None
    if e_and_h_file:
        pass
    # load the hubverse data
    smhub_table = None
    if smht_file is not None:
        ext = pathlib.Path(smht_file.name).suffix.lower()
        try:
            if ext == ".parquet":
                smhub_table = pl.read_parquet(smht_file)
            elif ext == ".csv":
                smhub_table = pl.read_csv(smht_file)
            else:
                raise ValueError(f"Unsupported file type: {ext}")
        except ValueError as e:
            st.error(str(e))
            st.stop()
        st.success(f"Loaded {smht_file.name} ({ext}).")
        logger.info(f"Uploaded file:\n{smht_file.name}")
        n_rows, n_cols = smhub_table.shape
        size_bytes = smhub_table.estimated_size()
        size_mb = size_bytes / 1e6
        logger.info(
            f"Hubverse Shape: {n_rows} rows x {n_cols} columns\n"
            f"Approximately {size_mb:.2f} MB in memory"
        )
        # locations in the hubverse table
        smhub_loc_abbrs = smhub_table["location"].unique().to_list()
        loc_lookup = forecasttools.location_lookup(
            location_vector=smhub_loc_abbrs, location_format="abbr"
        )
        locs_available = loc_lookup["long_name"].to_list()
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
            location = st.selectbox(
                "Location",
                options=locs_available,
            )
        # filter to location before filtering to model
        two_letter_loc_abbr = (
            loc_lookup.filter(pl.col("long_name") == location)
            .get_column("short_name")
            .item()
        )
        two_num_loc_abbr = (
            loc_lookup.filter(pl.col("long_name") == location)
            .get_column("location_code")
            .item()
        )
        smhubt_by_loc = smhub_table.filter(
            pl.col("location") == two_letter_loc_abbr,
        )
        # models and targets available
        models_available = smhubt_by_loc["model"].unique().to_list()
        selected_models = st.multiselect(
            "Model(s)",
            options=models_available,
            default=models_available,
            key="model_selection",
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

        st.markdown(f"## Forecasts For: {two_letter_loc_abbr}")
        st.markdown(f"## Reference Date: {selected_ref_date}")
        # plotting of the selected model, target, location, and reference date
        if smhubt_to_plot.is_empty():
            st.warning("No forecasts available for current selection.")
        else:
            forecast_chart = create_quantile_forecast_chart(smhubt_to_plot)
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
