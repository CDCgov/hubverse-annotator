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

PLOT_WIDTH = 625
STROKE_WIDTH = 2
MARKER_SIZE = 25


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
    two_letter_loc_abbr: str,
    selected_ref_date: str,
) -> None:
    """
    Streamlit widget for status/comments UI per model and
    saving to JSON.

    Parameters
    ----------
    selected_models : list[str]
        Selected models to annotate.
    two_letter_loc_abbr : str
        The selection location, typically a US jurisdiction.
    selected_ref_date : str
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
    export_button()


def model_and_target_selection_ui(
    observed_data_table: pl.DataFrame,
    forecast_table: pl.DataFrame,
    two_letter_loc_abbr: str,
) -> tuple[list[str], str]:
    """
    Streamlit widget for model and target selection.

    Parameters
    ----------
    observed_data_table : pl.DataFrame
        A hubverse table of loaded data (possibly empty).
    forecast_table : pl.DataFrame
        The hubverse formatted table of forecasted ED
        visits and or hospital admissions (possibly empty).
    two_letter_loc_abbr : str
        The selection location, typically a US jurisdiction.

    Returns
    -------
    tuple
        Returns a list of selected model names and the
        selected target.
    """
    if not forecast_table.is_empty():
        models = (
            forecast_table.filter(pl.col("location") == two_letter_loc_abbr)[
                "model"
            ]
            .unique()
            .sort()
            .to_list()
        )
        selected_models = st.multiselect(
            "Model(s)",
            options=models,
            default=models,
            key="model_selection",
        )
    else:
        selected_models = []
    forecast_targets = []
    if not forecast_table.is_empty():
        forecast_targets = (
            forecast_table.filter(
                pl.col("location") == two_letter_loc_abbr,
                pl.col("model").is_in(selected_models),
            )["target"]
            .unique()
            .sort()
            .to_list()
        )
    observed_data_targets = []
    if not observed_data_table.is_empty():
        observed_data_targets = (
            observed_data_table.filter(
                pl.col("location") == two_letter_loc_abbr
            )["target"]
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
    locs = []
    if "location" in observed_data_table.columns:
        locs += observed_data_table["location"].unique().to_list()
    if "location" in forecast_table.columns:
        locs += forecast_table["location"].unique().to_list()
    return forecasttools.location_lookup(
        location_vector=list(set(locs)), location_format="hubverse"
    )


def get_reference_dates(
    observed_data_table: pl.DataFrame, forecast_table: pl.DataFrame
) -> list[str]:
    """
    Retrieves a dataframe of forecast reference dates. The
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
    list[str]
        A list of available reference dates.
    """
    refs_dates = []
    if "reference_date" in observed_data_table.columns:
        refs_dates += observed_data_table["reference_date"].unique().to_list()
    if "reference_date" in forecast_table.columns:
        refs_dates += forecast_table["reference_date"].unique().to_list()
    return list(set(refs_dates))


def reference_date_and_location_ui(
    observed_data_table: pl.DataFrame, forecast_table: pl.DataFrame
) -> tuple[str, str]:
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
        Returns a tuple of the selected reference date and
        the two letter location abbreviation.
    """
    loc_lookup = get_available_locations(observed_data_table, forecast_table)
    long_names = loc_lookup["long_name"].to_list()
    ref_dates = get_reference_dates(observed_data_table, forecast_table)
    col1, col2 = st.columns(2)
    with col1:
        selected_ref_date = st.selectbox(
            "Reference Date",
            options=sorted(ref_dates),
            # format_func=lambda x: x.strftime("%Y-%m-%d"),
            key="ref_date_selection",
        )
    with col2:
        location = st.selectbox("Location", options=sorted(long_names))
    two_letter = (
        loc_lookup.filter(pl.col("long_name") == location)
        .get_column("short_name")
        .item()
    )
    return selected_ref_date, two_letter


def target_data_chart(
    observed_data_table: pl.DataFrame, scale: str = "log", grid: bool = True
) -> alt.Chart:
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
    yscale = alt.Scale(type=scale)
    x_axis = alt.Axis(title=None, grid=grid, ticks=True, labels=True)
    y_axis = alt.Axis(
        title="Forecasted Value", grid=grid, ticks=True, labels=True
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
    hubverse_table: pl.DataFrame, scale: str = "log", grid: bool = True
) -> alt.Chart:
    """
    Uses a hubverse table (polars) and a reference date to
    display quantile forecasts faceted by model. The
    output_type of the hubverse table must therefore be
    'quantile'.

    Parameters
    ----------
    hubverse_table : pl.DataFrame
        The hubverse-formatted forecast table.
    scale : str
        The scale to use for the Y axis during plotting.
        Defaults to logarithmic.
    grid : bool
        Whether to use gridlines for the X and Y axes.
        Defaults to True.

    Returns
    -------
    alt.Chart
        An altair chart object with plotted forecasts.
    """
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
        hubverse_table.filter(pl.col("output_type") == "quantile")
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
        interpolate="step-after",
    ).encode(
        y=alt.Y("0.025:Q", axis=y_axis),
        y2="0.975:Q",
        fill=alt.value("steelblue"),
    )
    band_80 = base.mark_errorband(
        extent="ci",
        opacity=0.2,
        interpolate="step-after",
    ).encode(
        y=alt.Y("0.10:Q", axis=y_axis),
        y2="0.90:Q",
        fill=alt.value("steelblue"),
    )
    band_50 = base.mark_errorband(
        extent="iqr",
        opacity=0.3,
        interpolate="step-after",
    ).encode(
        y=alt.Y("0.25:Q", axis=y_axis),
        y2="0.75:Q",
        fill=alt.value("steelblue"),
    )
    median = base.mark_line(
        strokeWidth=STROKE_WIDTH,
        interpolate="step-after",
        color="navy",
    ).encode(alt.Y("median:Q", axis=y_axis))
    return alt.layer(band_95, band_80, band_50, median)


def plotting_ui(
    data_to_plot: pl.DataFrame,
    forecasts_to_plot: pl.DataFrame,
    two_letter_loc_abbr: str,
    selected_target: str,
    selected_ref_date: str,
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
    two_letter_loc_abbr : str
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
    scale = "log" if st.checkbox("Log-scale", value=True) else "linear"
    grid = st.checkbox("Gridlines", value=True)
    layer = None
    if not forecasts_to_plot.is_empty():
        forecast = quantile_forecast_chart(
            forecasts_to_plot, scale=scale, grid=grid
        )
        layer = forecast if layer is None else layer + forecast
    if not data_to_plot.is_empty():
        observed = target_data_chart(data_to_plot, scale=scale, grid=grid)
        layer = observed if layer is None else layer + observed
    if layer is None:
        st.info("No data to plot for that model/target/location.")
        return
    title = f"{two_letter_loc_abbr}: {selected_target}, {selected_ref_date}"
    chart = (
        layer.interactive()
        .properties(title=alt.TitleParams(text=title, anchor="middle"))
        .facet(row=alt.Row("model:N"), columns=1)
    )
    chart_key = f"forecast_{two_letter_loc_abbr}_{selected_target}"
    base_chart.altair_chart(chart, use_container_width=False, key=chart_key)


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
        codes = hub_table["location"].unique().to_list()
        lookup = forecasttools.location_lookup(
            location_vector=codes, location_format="abbr"
        )
        code_to_abbr = dict(
            lookup.select(["location_code", "short_name"]).iter_rows()
        )
        hub_table = hub_table.with_columns(
            pl.col("location").replace(code_to_abbr)
        )
    return hub_table


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
    observed_data_file = st.file_uploader(
        "(Optional) Upload Hubverse Target Data", type=["parquet"]
    )
    observed_data_table = load_hubverse_table(observed_data_file)
    if (
        not observed_data_table.is_empty()
        and "as_of" in observed_data_table.columns
    ):
        latest = observed_data_table.select(pl.col("as_of").max()).item()
        observed_data_table = observed_data_table.filter(
            pl.col("as_of") == latest
        )
    forecast_file = st.file_uploader(
        "Upload Hubverse Forecasts", type=["csv", "parquet"]
    )
    forecast_table = load_hubverse_table(forecast_file)
    return observed_data_table, forecast_table


def filter_for_plotting(
    observed_data_table: pl.DataFrame,
    forecast_table: pl.DataFrame,
    selected_models: list[str],
    selected_target: str,
    two_letter_loc_abbr: str,
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
    selected_target
        The target for filtering in the forecast and or
        observed hubverse tables.
    two_letter_loc_abbr
        The abbreviated US jurisdiction code.

    Returns
    -------
    tuple
        A tuple of observed_data_table (pl.DataFrame) and
        forecast_table (pl.DataFrame) filtered by model,
        target, and location, to be used for plotting.
    """
    if not forecast_table.is_empty():
        forecasts_to_plot = forecast_table.filter(
            pl.col("location") == two_letter_loc_abbr,
            pl.col("target") == selected_target,
            pl.col("model").is_in(selected_models),
        )
    else:
        forecasts_to_plot = pl.DataFrame()
    if not observed_data_table.is_empty():
        data_to_plot = observed_data_table.filter(
            pl.col("location") == two_letter_loc_abbr,
            pl.col("target") == selected_target,
        )
    else:
        data_to_plot = pl.DataFrame()
    return forecasts_to_plot, data_to_plot


def main() -> None:
    # record session start time
    start_time = time.time()
    # streamlit application begins
    st.title("Forecast Annotator")
    observed_data_table, forecast_table = load_data_ui()
    # at least one of the tables must be non-empty
    if observed_data_table.is_empty() and forecast_table.is_empty():
        st.info("Please upload Observed Data or Hubverse Forecasts to begin.")
        return None
    selected_ref_date, two_letter_loc_abbr = reference_date_and_location_ui(
        observed_data_table, forecast_table
    )
    selected_models, selected_target = model_and_target_selection_ui(
        observed_data_table, forecast_table, two_letter_loc_abbr
    )
    forecasts_to_plot, data_to_plot = filter_for_plotting(
        observed_data_table,
        forecast_table,
        selected_models,
        selected_target,
        two_letter_loc_abbr,
    )
    plotting_ui(
        data_to_plot,
        forecasts_to_plot,
        two_letter_loc_abbr,
        selected_target,
        selected_ref_date,
    )
    forecast_annotation_ui(
        selected_models, two_letter_loc_abbr, selected_ref_date
    )
    duration = time.time() - start_time
    logger.info(f"Session lasted {duration:.1f}s")


if __name__ == "__main__":
    main()
