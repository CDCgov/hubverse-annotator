"""
A streamlit application that loads hubverse formatted
tables and plots model forecasts for the user to compare
and annotate models.

To run: uv run streamlit run ./hubverse_annotator/app.py
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
from streamlit.runtime.uploaded_file_manager import UploadedFile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def target_data_chart(eh_df: pl.DataFrame) -> alt.Chart:
    """
    Layers target hubverse data onto `altair` plot.

    Parameters
    ----------
    eh_df : pl.DataFrame
        A polars dataframe of E and H target data formatted
        as hubverse time series.

    Returns
    -------
    alt.Chart
        An `altair` chart with the target hubverse data.
    """
    obs_layer = (
        alt.Chart(eh_df)
        .mark_point(filled=True, size=35, color="limegreen")
        .encode(
            x=alt.X("date:T"),
            y=alt.Y("observation:Q"),
            tooltip=[
                alt.Tooltip("date:T"),
                alt.Tooltip("observation:Q"),
            ],
        )
    )
    return obs_layer


def render_annotation_section(
    selected_models: list[str],
    two_letter_loc_abbr: str,
    selected_ref_date: str,
) -> None:
    """
    Render status/comments UI per model and save to JSON.
    """
    output_dir = pathlib.Path("../output")
    output_dir.mkdir(parents=True, exist_ok=True)
    annotations_file = output_dir / f"anno_{selected_ref_date}.json"
    if annotations_file.exists():
        with annotations_file.open("r") as f:
            annotations = json.load(f)
    else:
        annotations = {}

    by_loc = annotations.setdefault(two_letter_loc_abbr, {})
    for m in selected_models:
        st.markdown(f"### {m}")
        prev = by_loc.get(m, {})
        default_status = prev.get("status", "None")
        default_comment = prev.get("comment", "")
        status = st.selectbox(
            "Status",
            ["Preferred", "Omitted", "None"],
            index=["Preferred", "Omitted", "None"].index(default_status),
            key=f"status_{two_letter_loc_abbr}_{m}",
        )
        comment = st.text_input(
            "Comments",
            default_comment,
            key=f"comment_{two_letter_loc_abbr}_{m}",
        )
        by_loc[m] = {"status": status, "comment": comment}

    with annotations_file.open("w") as f:
        json.dump(annotations, f, indent=2)


def render_export_button() -> None:
    """
    Streamlit widget for exporting annotated forecasts.
    """
    if st.button("Export forecasts"):
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.success("Need export")


def render_model_and_target_controls(
    smhubt_by_loc: pl.DataFrame,
) -> tuple[list[str], str]:
    """
    Streamlit widget for model and target selection.

    Parameters
    ----------
    smhubt_by_loc : pl.DataFrame
        The super-mega hubverse table of forecasted ED
        visits and or hospital admissions, filtered by
        location.

    Returns
    -------
    tuple
        Returns a list of selected model names and the
        annotator target.
    """
    models = smhubt_by_loc["model"].unique().sort().to_list()
    selected_models = st.multiselect(
        "Model(s)", options=models, default=models, key="model_selection"
    )

    targets = (
        smhubt_by_loc.filter(pl.col("model").is_in(selected_models))
        .get_column("target")
        .unique()
        .sort()
        .to_list()
    )
    selected_target = st.selectbox(
        "Target(s)", options=targets, key="target_selection"
    )

    return selected_models, selected_target


def render_ref_and_loc_controls(
    smhub_table: pl.DataFrame,
) -> tuple[str, str, str]:
    """
    Streamlit widget for the reference data and location
    selection.

    Parameters
    ----------
    smhub_table : pl.DataFrame
        The super-mega hubverse table of forecasted ED
        visits and or hospital admissions.

    Returns
    -------
    tuple
        Returns a tuple of the selected reference date,
        the two letter location abbreviation, and the
        numerical location reference.
    """
    locs = smhub_table["location"].unique().to_list()
    loc_lookup = forecasttools.location_lookup(
        location_vector=locs, location_format="abbr"
    )
    long_names = loc_lookup["long_name"].to_list()

    col1, col2 = st.columns(2)
    with col1:
        ref_dates = smhub_table["reference_date"].unique().sort().to_list()
        selected_ref_date = st.selectbox(
            "Reference Date",
            options=ref_dates,
            format_func=lambda x: x.strftime("%Y-%m-%d"),
            key="ref_date_selection",
        )
    with col2:
        location = st.selectbox("Location", options=long_names)

    two_letter = (
        loc_lookup.filter(pl.col("long_name") == location)
        .get_column("short_name")
        .item()
    )
    two_num = (
        loc_lookup.filter(pl.col("long_name") == location)
        .get_column("location_code")
        .item()
    )

    return selected_ref_date, two_letter, two_num


def render_chart_section(
    smhubt_to_plot: pl.DataFrame,
    eh_to_plot: pl.DataFrame,
    selected_ref_date: str,
    two_letter_loc_abbr: str,
) -> None:
    """
    Altair chart of the forecasts, with observed data
    overlain where possible.

    Parameters
    ----------
    smhubt_to_plot : pl.DataFrame
        The super-mega hubverse forecast table, filtered
        by location, target, and model.
    eh_to_plot : pl.DataFrame
        The hubverse observations time-series, filtered by
        location, target, and model.
    selected_ref_date : str
        The selected reference date.
    two_letter_loc_abbr : str
        The selection location, i.e. US jurisdiction.
    """

    st.markdown(f"## Forecasts For: {two_letter_loc_abbr}")
    st.markdown(f"## Reference Date: {selected_ref_date}")

    forecast_layers = create_quantile_forecast_chart(smhubt_to_plot)
    observed_layers = target_data_chart(eh_to_plot)
    forecast_and_observed_layers = forecast_layers + observed_layers
    chart = (
        forecast_and_observed_layers.facet(
            row=alt.Row("model:N", title="Model"), columns=1
        )
    ).interactive()
    st.altair_chart(chart, use_container_width=True)


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
    base = alt.Chart(df_wide).encode(x=alt.X("target_end_date:T"))
    band_95 = base.mark_errorband(
        extent="ci",
        opacity=0.1,
        interpolate="step-after",
    ).encode(
        y=alt.Y("0.025:Q"),
        y2="0.975:Q",
        fill=alt.value("steelblue"),
    )
    band_80 = base.mark_errorband(
        extent="ci",
        opacity=0.2,
        interpolate="step-after",
    ).encode(
        y=alt.Y("0.10:Q", axis=None), y2="0.90:Q", fill=alt.value("steelblue")
    )
    band_50 = base.mark_errorband(
        extent="iqr",
        opacity=0.3,
        interpolate="step-after",
    ).encode(
        y=alt.Y("0.25:Q", axis=None), y2="0.75:Q", fill=alt.value("steelblue")
    )
    median = base.mark_line(
        strokeWidth=2,
        interpolate="step-after",
        color="navy",
    ).encode(y=alt.Y("median:Q", axis=None))
    return alt.layer(band_95, band_80, band_50, median)


def load_hubverse_table(hub_file: UploadedFile | None):
    """
    Load a hubverse formatted table into Polars from a
    data file uploaded to Streamlit.

    Parameters
    ----------
    hub_file : UploadedFile | None
        A file-like object returned by Streamlit's
        `st.file_uploader`. Supported file extensions are:
        `.parquet` and `.csv`. If `hub_file` is `None`,
        an empty DataFrame is returned.

    Returns
    -------
    pl.DataFrame
        A DataFrame containing the loaded data as a
        hubverse formatted table, or an empty DataFrame if
        no file was uploaded.

    Raises
    ------
    ValueError
        If the uploaded file has an unsupported extension
        (not `.parquet` or `.csv`).
    """
    if hub_file is None:
        return pl.DataFrame()
    ext = pathlib.Path(hub_file.name).suffix.lower()
    if ext == ".parquet":
        hub_table = pl.read_parquet(hub_file)
    elif ext == ".csv":
        hub_table = pl.read_csv(hub_file)
    else:
        st.error(f"Unsupported file type: {ext}")
        st.stop()
    logger.info(f"Uploaded file:\n{hub_file.name}")
    n_rows, n_cols = hub_table.shape
    size_bytes = hub_table.estimated_size()
    size_mb = size_bytes / 1e6
    logger.info(
        f"Hubverse Shape: {n_rows} rows x {n_cols} columns\n"
        f"Approximately {size_mb:.2f} MB in memory"
    )
    return hub_table


def load_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Streamlit widget for the upload of the hubverse
    formatted influenza and COVID-19 ED visits and hospital
    admissions observations time-series and hubverse
    formatted "super-mega" forecast table.

    Returns
    -------
    eh_table : pl.DataFrame
        Loaded EH table (filtered to latest as_of) or
        an empty DataFrame.
    smhub_table : pl.DataFrame
        Loaded forecast table or an empty DataFrame.
    """
    e_and_h_file = st.file_uploader(
        "Upload Hubverse Target Data", type=["parquet"]
    )
    eh_table = load_hubverse_table(e_and_h_file)
    if not eh_table.is_empty() and "as_of" in eh_table.columns:
        latest = eh_table.select(pl.col("as_of").max()).item()
        eh_table = eh_table.filter(pl.col("as_of") == latest)

    smht_file = st.file_uploader(
        "Upload Hubverse Forecasts", type=["csv", "parquet"]
    )
    smhub_table = load_hubverse_table(smht_file)

    return eh_table, smhub_table


def filter_for_plotting(
    smhubt_by_loc: pl.DataFrame,
    eh_table: pl.DataFrame,
    selected_models: list[str],
    selected_target: str,
    two_num_loc_abbr: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Filter forecast and EH tables for the selected models
    and target.
    """
    smhubt_to_plot = smhubt_by_loc.filter(
        pl.col("model").is_in(selected_models),
        pl.col("target") == selected_target,
    )
    if not eh_table.is_empty():
        eh_to_plot = eh_table.filter(
            pl.col("location") == two_num_loc_abbr,
            pl.col("target") == selected_target,
        )
    else:
        eh_to_plot = pl.DataFrame()

    return smhubt_to_plot, eh_to_plot


def main() -> None:
    # record session start time
    start_time = time.time()

    # streamlit application begins
    st.title("Forecast Annotator")

    # super-mega hubverse forecast table required
    eh_table, smhub_table = load_data()

    if smhub_table.is_empty():
        st.info("Please upload Hubverse Forecasts to begin.")
        return

    selected_ref_date, two_letter_loc_abbr, two_num_loc_abbr = (
        render_ref_and_loc_controls(smhub_table)
    )

    smhubt_by_loc = smhub_table.filter(
        pl.col("location") == two_letter_loc_abbr
    )

    selected_models, selected_target = render_model_and_target_controls(
        smhubt_by_loc
    )

    smhubt_to_plot, eh_to_plot = filter_for_plotting(
        smhubt_by_loc,
        eh_table,
        selected_models,
        selected_target,
        two_num_loc_abbr,
    )

    render_chart_section(
        smhubt_to_plot, eh_to_plot, selected_ref_date, two_letter_loc_abbr
    )

    render_annotation_section(
        selected_models, two_letter_loc_abbr, selected_ref_date
    )

    render_export_button()

    duration = time.time() - start_time
    logger.info(f"Session lasted {duration:.1f}s")


if __name__ == "__main__":
    main()

# def main() -> None:
#     # record start time
#     start_time = time.time()
#     # begin streamlit application
#     st.title("Forecast Annotator")
#     # super-mega hubverse table and target data uploaded
#     e_and_h_file = st.file_uploader(
#         "Upload Hubverse Target Data", type=["parquet"]
#     )
#     # update eh table with actual data
#     eh_table = load_hubverse_table(e_and_h_file)
#     # filter to latest as_of date, if as_of col present
#     if "as_of" in eh_table.columns:
#         latest = eh_table.select(pl.col("as_of").max()).item()
#         eh_table = eh_table.filter(pl.col("as_of") == latest)
#     smht_file = st.file_uploader(
#         "Upload Hubverse Forecasts", type=["csv", "parquet"]
#     )
#     # load the hubverse data
#     smhub_table = load_hubverse_table(smht_file)
#     if not smhub_table.is_empty():
#         # locations in the hubverse table
#         smhub_loc_abbrs = smhub_table["location"].unique().to_list()
#         loc_lookup = forecasttools.location_lookup(
#             location_vector=smhub_loc_abbrs, location_format="abbr"
#         )
#         locs_available = loc_lookup["long_name"].to_list()
#         # two-column layout for reference date and location
#         col1, col2 = st.columns(2)
#         with col1:
#             ref_dates = smhub_table["reference_date"].unique()
# .sort().to_list()
#             selected_ref_date = st.selectbox(
#                 "Reference Date",
#                 options=ref_dates,
#                 format_func=lambda x: x.strftime("%Y-%m-%d"),
#                 key="ref_date_selection",
#             )
#         with col2:
#             location = st.selectbox(
#                 "Location",
#                 options=locs_available,
#             )
#         # filter to location before filtering to model
#         two_letter_loc_abbr = (
#             loc_lookup.filter(pl.col("long_name") == location)
#             .get_column("short_name")
#             .item()
#         )
#         two_num_loc_abbr = (
#             loc_lookup.filter(pl.col("long_name") == location)
#             .get_column("location_code")
#             .item()
#         )
#         smhubt_by_loc = smhub_table.filter(
#             pl.col("location") == two_letter_loc_abbr,
#         )
#         # models and targets available
#         models_available = smhubt_by_loc["model"].unique().sort().to_list()
#         selected_models = st.multiselect(
#             "Model(s)",
#             options=models_available,
#             default=models_available,
#             key="model_selection",
#         )
#         targets_available = (
#             smhubt_by_loc.filter(pl.col("model").is_in(selected_models))[
#                 "target"
#             ]
#             .unique()
#             .sort()
#             .to_list()
#         )
#         selected_target = st.selectbox(
#             "Target(s)",
#             options=targets_available,
#             key="target_selection",
#         )
#         smhubt_to_plot = pl.DataFrame()
#         eh_to_plot = pl.DataFrame()
#         if (selected_models) and (selected_target is not None):
#             smhubt_to_plot = smhubt_by_loc.filter(
#                 pl.col("model").is_in(selected_models),
#                 pl.col("target") == selected_target,
#             )
#             if not eh_table.is_empty():
#                 eh_to_plot = eh_table.filter(
#                     pl.col("location") == two_num_loc_abbr,
#                     pl.col("target") == selected_target,
#                 )
#         st.markdown(f"## Forecasts For: {two_letter_loc_abbr}")
#         st.markdown(f"## Reference Date: {selected_ref_date}")
#         forecast_layers = create_quantile_forecast_chart(smhubt_to_plot)
#         observed_layers = target_data_chart(eh_to_plot)
#         forecast_and_observed_layers = forecast_layers + observed_layers
#         chart = (
#             forecast_and_observed_layers.facet(
#                 row=alt.Row("model:N", title="Model"), columns=1
#             )
#         ).interactive()
#         st.altair_chart(chart, use_container_width=True)
#         # preference and comments saving
#         output_dir = pathlib.Path("../output")
#         output_dir.mkdir(parents=True, exist_ok=True)
#         annotations_file = output_dir / f"anno_{selected_ref_date}.json"
#         if annotations_file.exists():
#             with annotations_file.open("r") as f:
#                 annotations = json.load(f)
#             logger.info(f"Annotations file created:\n{annotations_file}")
#         else:
#             annotations = {}
#         # save by location, with empty dict by default
#         by_loc_dict = annotations.setdefault(two_num_loc_abbr, {})
#         for model in selected_models:
#             st.markdown(f"### {model}")
#             # set status and comments keys, and get previous in json
#             status_key = f"status_{two_letter_loc_abbr}_{model}"
#             comment_key = f"comment_{two_letter_loc_abbr}_{model}"
#             prev = by_loc_dict.get(model, {})
#             default_status = prev.get("status", "None")
#             default_comment = prev.get("comment", "")
#             # select boxes for status and comments
#             status = st.selectbox(
#                 "Status",
#                 ["Preferred", "Omitted", "None"],
#                 index=["Preferred", "Omitted", "None"].index(default_status),
#                 key=status_key,
#             )
#             comment = st.text_input(
#                 "Comments", value=default_comment, key=comment_key
#             )
#             # save in dictionary and write out
#             by_loc_dict[model] = {"status": status, "comment": comment}
#         with open(annotations_file, "w") as f:
#             json.dump(annotations, f, indent=2)

#         # export button
#         if st.button("Export forecasts"):
#             col1, col2, col3 = st.columns([1, 3, 1])
#             with col2:
#                 st.success("Need export")

#     # record end time
#     end_time = time.time()
#     duration = end_time - start_time
#     logger.info(f"Session lasted around: {duration // 60} minutes.")


# if __name__ == "__main__":
#     main()
