"""
A streamlit application that loads hubverse formatted
tables and plots model forecasts for the user to compare
and annotate models.

To run: uv run streamlit run ./hubverse_annotator/app.py
"""

import datetime
import json
import logging
import pathlib
import time
from functools import reduce
from typing import Literal

import altair as alt
import forecasttools
import polars as pl
import polars.selectors as cs
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
from streamlit_shortcuts import add_shortcuts

type ScaleType = Literal["linear", "log"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PLOT_WIDTH = 625
STROKE_WIDTH = 2
MARKER_SIZE = 25

add_shortcuts(prev_button="arrowleft", next_button="arrowright")


def export_button() -> None:
    """
    Streamlit widget for exporting annotated forecasts.
    """
    if st.button("Export forecasts"):
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.success("Export functionality not yet implemented.")


def forecast_annotation_ui(
    selected_models: list[str],
    loc_abbr: str,
    selected_ref_date: datetime.date,
) -> None:
    """
    Streamlit widget for status/comments UI per model and
    saving to JSON.

    Parameters
    ----------
    selected_models : list[str]
        Selected models to annotate.
    loc_abbr : str
        The selection location, typically a US jurisdiction.
    selected_ref_date : datetime.date
        The selected reference date.
    """
    output_dir = pathlib.Path("../output")
    output_dir.mkdir(parents=True, exist_ok=True)
    annotations_file = output_dir / f"anno_{selected_ref_date}.json"
    if annotations_file.exists():
        with annotations_file.open("r") as f:
            annotations = json.load(f)
    else:
        annotations = {}
    by_loc = annotations.setdefault(loc_abbr, {})
    for m in selected_models:
        st.markdown(f"### {m}")
        prev = by_loc.get(m, {})
        default_status = prev.get("status", "None")
        default_comment = prev.get("comment", "")
        status = st.selectbox(
            "Status",
            ["Preferred", "Omitted", "None"],
            index=["Preferred", "Omitted", "None"].index(default_status),
            key=f"status_{loc_abbr}_{m}",
        )
        comment = st.text_input(
            "Comments",
            default_comment,
            key=f"comment_{loc_abbr}_{m}",
        )
        by_loc[m] = {"status": status, "comment": comment}
    with annotations_file.open("w") as f:
        json.dump(annotations, f, indent=2)
    export_button()


def model_and_target_selection_ui(
    observed_data_table: pl.DataFrame,
    forecast_table: pl.DataFrame,
    loc_abbr: str,
) -> tuple[list[str], str | None]:
    """
    Streamlit widget for model and target selection.

    Parameters
    ----------
    observed_data_table : pl.DataFrame
        A hubverse table of loaded data (possibly empty).
    forecast_table : pl.DataFrame
        The hubverse formatted table of forecasted ED
        visits and or hospital admissions (possibly empty).
    loc_abbr : str
        The selection location, typically a US jurisdiction.

    Returns
    -------
    tuple
        Returns a list of selected model names and the
        selected target.
    """
    models = (
        forecast_table.filter(pl.col("loc_abbr") == loc_abbr)
        .get_column("model_id")
        .unique()
        .sort()
        .to_list()
    )
    selected_models = st.multiselect(
        "Model(s)",
        options=models,
        default=None,
        key="model_selection",
    )
    forecast_targets = (
        forecast_table.filter(
            pl.col("loc_abbr") == loc_abbr,
            pl.col("model_id").is_in(selected_models),
        )
        .get_column("target")
        .unique()
        .sort()
        .to_list()
    )
    observed_data_targets = (
        observed_data_table.filter(pl.col("loc_abbr") == loc_abbr)
        .get_column("target")
        .unique()
        .sort()
        .to_list()
    )
    all_targets = sorted(set(forecast_targets + observed_data_targets))
    selected_target = st.selectbox(
        "Target",
        options=all_targets,
        key="target_selection",
    )
    return selected_models, selected_target


@st.cache_data
def get_available_locations(
    observed_data_table: pl.DataFrame, forecast_table: pl.DataFrame
) -> pl.DataFrame:
    """
    Retrieves a dataframe of locations from forecasttools
    used for converting between location formats. The
    dataframe is cached for streamlit via cache_data.

    Parameters
    ----------
    observed_data_table : pl.DataFrame
        A hubverse table of loaded data (possibly empty).
    forecast_table : pl.DataFrame
        The hubverse formatted table of forecasted ED
        visits and or hospital admissions (possibly empty).

    Returns
    -------
    pl.DataFrame
        A dataframe of locations in different formats.
    """
    locs = (
        pl.concat(
            [
                observed_data_table.get_column("loc_abbr"),
                forecast_table.get_column("loc_abbr"),
            ]
        )
        .unique()
        .to_list()
    )
    return forecasttools.location_lookup(
        location_vector=list(set(locs)), location_format="abbr"
    )


def get_reference_dates(forecast_table: pl.DataFrame) -> list[datetime.date]:
    """
    Retrieves a dataframe of forecast reference dates. The
    dataframe is cached for streamlit via cache_data.

    Parameters
    ----------
    forecast_table : pl.DataFrame
        The hubverse formatted table of forecasted ED
        visits and or hospital admissions (possibly empty).

    Returns
    -------
    list[datetime.date]
        A list of available reference dates.
    """
    return forecast_table.get_column("reference_date").unique().to_list()


def location_and_reference_data_ui(
    observed_data_table: pl.DataFrame, forecast_table: pl.DataFrame
) -> tuple[str, datetime.date]:
    """
    Streamlit widget for the reference date and location
    selection.

    Parameters
    ----------
    observed_data_table : pl.DataFrame
        A hubverse table of loaded data (possibly empty).
    forecast_table : pl.DataFrame
        The hubverse formatted table of forecasted ED
        visits and or hospital admissions (possibly empty).

    Returns
    -------
    tuple
        Returns a tuple of the two letter location
        abbreviation and the selected reference date.
    """
    loc_lookup = get_available_locations(observed_data_table, forecast_table)
    if "locations_list" not in st.session_state:
        st.session_state.locations_list = (
            loc_lookup.get_column("long_name").sort().to_list()
        )
    st.selectbox(
        "Location",
        options=list(range(len(st.session_state.locations_list))),
        key="current_loc_id",
        format_func=lambda i: st.session_state.locations_list[i],
    )

    def go_to_prev_loc():
        st.session_state.current_loc_id -= 1

    def go_to_next_loc():
        st.session_state.current_loc_id += 1

    prev_col, next_col = st.columns([1, 1])
    with prev_col:
        st.button(
            "⏮️",
            disabled=(st.session_state.current_loc_id == 0),
            on_click=go_to_prev_loc,
            key="prev_button",
        )
    with next_col:
        st.button(
            "⏭️",
            disabled=(
                st.session_state.current_loc_id
                == len(st.session_state.locations_list) - 1
            ),
            on_click=go_to_next_loc,
            key="next_button",
        )
    loc_id = st.session_state.current_loc_id
    selected_location = st.session_state.locations_list[loc_id]
    loc_abbr = (
        loc_lookup.filter(pl.col("long_name") == selected_location)
        .get_column("short_name")
        .item()
    )
    ref_dates = sorted(get_reference_dates(forecast_table), reverse=True)
    selected_ref_date = st.selectbox(
        "Reference Date",
        options=ref_dates,
        format_func=lambda d: d.strftime("%Y-%m-%d"),
        key="ref_date_selection",
    )
    return loc_abbr, selected_ref_date


def is_empty_chart(chart: alt.LayerChart) -> bool:
    """
    Checks if an altair layer is empty. Primarily used for
    resolving forecast and observed data layering when
    only one file is uploaded.

    Parameters
    ----------
    chart: alt.LayerChart
        An altair LayerChart that is either observed data
        or a forecast. The layer may be empty.

    Returns
    -------
    bool
        Whether the altair LayerChart is empty.
    """
    spec = chart.to_dict()
    # unit chart: no data, no mark, no encoding
    if "layer" not in spec:
        return not (
            spec.get("data") or spec.get("mark") or spec.get("encoding")
        )
    # LayerChart: check each sub-layer recursively
    # check if the layer list is empty or all sub-layers
    # are empty
    if not spec["layer"]:
        return True
    # for each sub-layer, check if it's empty by examining
    # its dict directly instead of converting back to
    # Chart object (which can cause validation errors)
    return all(
        not (sub.get("data") or sub.get("mark") or sub.get("encoding"))
        for sub in spec["layer"]
    )


def target_data_chart(
    observed_data_table: pl.DataFrame,
    scale: ScaleType = "log",
    grid: bool = True,
) -> alt.Chart | alt.LayerChart:
    """
    Layers target hubverse data onto `altair` plot.

    Parameters
    ----------
    observed_data_table : pl.DataFrame
        A polars dataframe of E and H target data formatted
        as hubverse time series.
    scale : str
        The scale to use for the Y axis during plotting.
        Defaults to logarithmic.
    grid : bool
        Whether to use gridlines for the X and Y axes.
        Defaults to True.

    Returns
    -------
    alt.Chart
        An `altair` chart with the target hubverse data.
    """
    if observed_data_table.is_empty():
        return alt.layer()
    yscale = alt.Scale(type=scale)
    x_axis = alt.Axis(title=None, grid=grid, ticks=True, labels=True)
    y_axis = alt.Axis(
        title=None, grid=grid, ticks=True, labels=True, orient="right"
    )
    obs_layer = (
        alt.Chart(observed_data_table, width=PLOT_WIDTH)
        .mark_point(filled=True, size=MARKER_SIZE, color="limegreen")
        .encode(
            x=alt.X("date:T", axis=x_axis),
            y=alt.Y("observation:Q", axis=y_axis, scale=yscale),
            tooltip=[
                alt.Tooltip("date:T"),
                alt.Tooltip("observation:Q"),
            ],
        )
    )
    return obs_layer


def quantile_forecast_chart(
    forecast_table: pl.DataFrame, scale: ScaleType = "log", grid: bool = True
) -> alt.LayerChart:
    """
    Uses a hubverse table (polars) and a reference date to
    display quantile forecasts faceted by model. The
    output_type of the hubverse table must therefore be
    'quantile'.

    Parameters
    ----------
    forecast_table : pl.DataFrame
        The hubverse-formatted forecast table.
    scale : str
        The scale to use for the Y axis during plotting.
        Defaults to logarithmic.
    grid : bool
        Whether to use gridlines for the X and Y axes.
        Defaults to True.

    Returns
    -------
    alt.LayerChart
        An altair chart object with plotted forecasts.
    """
    if forecast_table.is_empty():
        return alt.layer()
    value_col = "value"
    yscale = alt.Scale(type=scale)
    x_axis = alt.Axis(title=None, grid=grid, ticks=True, labels=True)
    y_axis = alt.Axis(
        title="Forecasted Value",
        grid=grid,
        ticks=True,
        labels=True,
        orient="right",
    )
    # filter to quantile only rows and ensure quantiles
    # are str for pivot; also, pivot to wide, so quantiles
    # ids are columns
    df_wide = (
        forecast_table.filter(pl.col("output_type") == "quantile")
        .pivot(
            on="output_type_id",
            index=cs.exclude("output_type_id", value_col),
            values=value_col,
        )
        .with_columns(pl.col("0.5").alias("median"))
    )
    base = alt.Chart(df_wide, width=PLOT_WIDTH).encode(
        x=alt.X("target_end_date:T", axis=x_axis),
        y=alt.Y("median:Q", axis=y_axis, scale=yscale),
    )
    band_95 = base.mark_errorband(
        extent="ci",
        opacity=0.1,
        interpolate="step",
    ).encode(
        y=alt.Y("0.025:Q", axis=y_axis),
        y2="0.975:Q",
        fill=alt.value("steelblue"),
    )
    band_80 = base.mark_errorband(
        extent="ci",
        opacity=0.2,
        interpolate="step",
    ).encode(
        y=alt.Y("0.10:Q", axis=y_axis),
        y2="0.90:Q",
        fill=alt.value("steelblue"),
    )
    band_50 = base.mark_errorband(
        extent="iqr",
        opacity=0.3,
        interpolate="step",
    ).encode(
        y=alt.Y("0.25:Q", axis=y_axis),
        y2="0.75:Q",
        fill=alt.value("steelblue"),
    )
    median = base.mark_line(
        strokeWidth=STROKE_WIDTH,
        interpolate="step",
        color="navy",
    ).encode(alt.Y("median:Q", axis=y_axis))
    return alt.layer(band_95, band_80, band_50, median)


def plotting_ui(
    data_to_plot: pl.DataFrame,
    forecasts_to_plot: pl.DataFrame,
    loc_abbr: str,
    selected_target: str | None,
    selected_ref_date: datetime.date,
    scale,
    grid,
) -> None:
    """
    Altair chart of the forecasts, with observed data
    overlaid where possible.

    Parameters
    ----------
    data_to_plot : pl.DataFrame
        The hubverse formatted observations time-series,
        filtered to the requested location, target, and
        model(s).
    forecasts_to_plot : pl.DataFrame
        The hubverse formatted forecast table, filtered
        to the requested location, target, and model.
    loc_abbr : str
        The selection location, typically a US jurisdiction.
    selected_target : str
        The target for filtering in the forecast and or
        observed hubverse tables.
    selected_ref_date : str
        The selected reference date.
    """
    # empty streamlit object (DeltaGenerator) needed for
    # plots to reload successfully with new data.
    base_chart = st.empty()
    # scale = "log" if st.checkbox("Log-scale", value=True) else "linear"
    # grid = st.checkbox("Gridlines", value=True)
    forecast_layer = quantile_forecast_chart(
        forecasts_to_plot, scale=scale, grid=grid
    )
    observed_layer = target_data_chart(data_to_plot, scale=scale, grid=grid)
    sub_layers = [
        layer
        for layer in [forecast_layer, observed_layer]
        if not is_empty_chart(layer)
    ]
    if sub_layers:
        # for some reason alt.layer(*sub_layers) does not work
        layer = reduce(lambda x, y: x + y, sub_layers)
    else:
        st.info("No data to plot for that model/target/location.")
        return
    title = f"{loc_abbr}: {selected_target}, {selected_ref_date}"
    chart = (
        layer.interactive()
        .properties(title=alt.TitleParams(text=title, anchor="middle"))
        .facet(row=alt.Row("model_id:N"), columns=1)
    )
    chart_key = f"forecast_{loc_abbr}_{selected_target}"
    base_chart.altair_chart(chart, use_container_width=False, key=chart_key)


def validate_schema(
    df: pl.DataFrame,
    expected_schema: dict[str, pl.DataType],
    name: str,
    strict: bool = False,
) -> None:
    """
    Stop the app if received dataframe does not adhere to
    the provided schema.

    Parameters
    ----------
    df : pl.DataFrame
        An ingested dataframe expected to be either
        hubverse formatted observed data or forecasts.
    expected_schema : dict[str, pl.DataType]
        Mapping of column name to expected Polars dtype.
    name : str
        Name for the dataframe (used in error messages).
    strict : bool
        If True, extra columns beyond those in
        `expected_schema` will also trigger an error.
        If False, extra columns are ignored. Defaults to
        False.
    """
    actual = df.schema
    missing = set(expected_schema) - actual.keys()
    extra = set(actual.keys()) - expected_schema.keys() if strict else set()
    mismatches = {
        col: (expected_schema[col], actual[col])
        for col in expected_schema.keys() & actual.keys()
        if expected_schema[col] != actual[col]
    }
    if missing or extra or mismatches:
        parts: list[str] = []
        if missing:
            parts.append(f"missing cols {sorted(missing)}")
        if extra:
            parts.append(f"unexpected cols {sorted(extra)}")
        for col, (exp, act) in mismatches.items():
            parts.append(f"'{col}' expected {exp}, got {act}")
        st.error(f"{name} schema problems: " + "; ".join(parts))
        st.stop()


@st.cache_data
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
    # ensure hub table loc column is two letter abbrs
    if "location" in hub_table.columns:
        codes = hub_table.get_column("location").unique().to_list()
        lookup = forecasttools.location_lookup(
            location_vector=codes, location_format="hubverse"
        )
        code_to_abbr = dict(
            lookup.select(["location_code", "short_name"]).iter_rows()
        )
        hub_table = hub_table.with_columns(
            pl.col("location").replace(code_to_abbr).alias("loc_abbr")
        )
    return hub_table


def load_observed_data(
    observed_data_file: UploadedFile | None,
) -> pl.DataFrame:
    """
    Loads and validates the observed data table from a
    Hubverse formatted file. Returns an empty DataFrame
    with the given schema if no file is provided.
    Otherwise, read via `load_hubverse_table`, filter to
    the latest `as_of` date, then validate against
    `expected_schema` (stopping on failure).

    Parameters
    ----------
    observed_data_file : UploadedFile | None
        Streamlit-uploaded file containing observed data,
        or None.

    Returns
    -------
    pl.DataFrame
        The loaded and validated observed data table, or
        an empty DataFrame conforming to `expected_schema`
        if no file was uploaded.
    """
    observed_schema = {
        "date": pl.Date,
        "state": pl.Utf8,
        "observation": pl.Float64,
        "location": pl.Utf8,
        "as_of": pl.Date,
        "target": pl.Utf8,
        "loc_abbr": pl.Utf8,
    }
    if not observed_data_file:
        return pl.DataFrame(schema=observed_schema)
    table = load_hubverse_table(observed_data_file)
    validate_schema(table, observed_schema, "Observed Data")
    table = table.filter(pl.col("as_of") == pl.col("as_of").max())
    return table


def load_forecast_data(
    forecast_file: UploadedFile | None,
) -> pl.DataFrame:
    """
    Loads and validates the forecast data table from a
    Hubverse formatted file. Returns an empty DataFrame
    with the given schema if no file is provided.
    Otherwise, read via `load_hubverse_table` and
    validate against `expected_schema` (stopping on
    failure).

    Parameters
    ----------
    forecast_file : UploadedFile | None
        Streamlit-uploaded file containing forecast data,
        or None.

    Returns
    -------
    pl.DataFrame
        The loaded and validated forecast table, or an
        empty DataFrame conforming to `expected_schema`
        if no file was uploaded.
    """
    forecast_schema = {
        "model_id": pl.Utf8,
        "reference_date": pl.Date,
        "target": pl.Utf8,
        "horizon": pl.Int32,
        "target_end_date": pl.Date,
        "location": pl.Utf8,
        "output_type": pl.Utf8,
        "output_type_id": pl.Utf8,
        "value": pl.Float64,
        "loc_abbr": pl.Utf8,
    }
    if not forecast_file:
        return pl.DataFrame(schema=forecast_schema)
    table = load_hubverse_table(forecast_file)
    validate_schema(table, forecast_schema, "Forecast Data")
    return table


def load_data_ui() -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Streamlit widget for the upload of the hubverse
    formatted influenza and COVID-19 ED visits and hospital
    admissions observations time-series and hubverse
    formatted forecast table.

    Returns
    -------
    tuple
        A tuple of observed_data_table (pl.DataFrame), i.e.
        the loaded observed data table (filtered to latest
        as_of date) or an empty DataFrame and
        forecast_table (pl.DataFrame), i.e. the loaded
        forecast table or an empty DataFrame.
    """
    observed_file = st.file_uploader(
        "Upload Hubverse Target Data", type=["parquet"]
    )
    forecast_file = st.file_uploader(
        "Upload Hubverse Forecasts", type=["csv", "parquet"]
    )
    observed_data_table = load_observed_data(
        observed_file,
    )
    forecast_table = load_forecast_data(
        forecast_file,
    )
    return observed_data_table, forecast_table


def filter_for_plotting(
    observed_data_table: pl.DataFrame,
    forecast_table: pl.DataFrame,
    selected_models: list[str],
    selected_target: str | None,
    selected_ref_date: datetime.date,
    loc_abbr: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Filter forecast and observed data tables for the
    selected models and target.

    Parameters
    ----------
    observed_data_table : pl.DataFrame
        A hubverse table of loaded data (possibly empty).
    forecast_table : pl.DataFrame
        The hubverse formatted table of forecasted ED
        visits and or hospital admissions (possibly empty).
    selected_models : list[str]
        Selected models to annotate.
    selected_target : str
        The target for filtering in the forecast and or
        observed hubverse tables.
    selected_ref_date : datetime.date
        The selected reference date.
    loc_abbr
        The abbreviated US jurisdiction abbreviation.

    Returns
    -------
    tuple
        A tuple of observed_data_table (pl.DataFrame) and
        forecast_table (pl.DataFrame) filtered by model,
        target, and location, to be used for plotting.
    """
    data_to_plot = observed_data_table.filter(
        pl.col("loc_abbr") == loc_abbr,
        pl.col("target") == selected_target,
    )
    forecasts_to_plot = forecast_table.filter(
        pl.col("loc_abbr") == loc_abbr,
        pl.col("target") == selected_target,
        pl.col("model_id").is_in(selected_models),
        pl.col("reference_date") == selected_ref_date,
    )
    return data_to_plot, forecasts_to_plot


def main() -> None:
    # record session start time
    start_time = time.time()
    # streamlit application begins
    with st.sidebar:
        st.title("Forecast Annotator")
        observed_data_table, forecast_table = load_data_ui()
        # at least one of the tables must be non-empty
        if observed_data_table.is_empty() and forecast_table.is_empty():
            st.info(
                "Please upload Observed Data or Hubverse Forecasts to begin."
            )
            return None
        loc_abbr, selected_ref_date = location_and_reference_data_ui(
            observed_data_table, forecast_table
        )
        selected_models, selected_target = model_and_target_selection_ui(
            observed_data_table, forecast_table, loc_abbr
        )
        scale = "log" if st.checkbox("Log-scale", value=True) else "linear"
        grid = st.checkbox("Gridlines", value=True)
        forecast_annotation_ui(selected_models, loc_abbr, selected_ref_date)
    data_to_plot, forecasts_to_plot = filter_for_plotting(
        observed_data_table,
        forecast_table,
        selected_models,
        selected_target,
        selected_ref_date,
        loc_abbr,
    )
    plotting_ui(
        data_to_plot,
        forecasts_to_plot,
        loc_abbr,
        selected_target,
        selected_ref_date,
        scale=scale,
        grid=grid,
    )
    duration = time.time() - start_time
    logger.info(f"Session lasted {duration:.1f}s")


if __name__ == "__main__":
    main()
