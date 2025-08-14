"""
A utils module, for support of the application and UI
components. The utilities include: data loading & schema
validation, location/date lookup, plotting helpers
(altair layers, empty chart check), and filtering helper
for observed vs. forecast tables.

To run: uv run streamlit run ./hubverse_annotator/app.py
"""

import datetime
import logging
import pathlib
import re
from typing import Literal

import altair as alt
import colorbrewer
import forecasttools
import polars as pl
import polars.selectors as cs
import streamlit as st
from matplotlib.colors import to_hex
from streamlit.runtime.uploaded_file_manager import UploadedFile

PLOT_WIDTH = 625
STROKE_WIDTH = 2
MARKER_SIZE = 65


type ScaleType = Literal["linear", "log"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    fc_locs = forecast_table.get_column("loc_abbr").unique().to_list()
    obs_locs = observed_data_table.get_column("loc_abbr").unique().to_list()
    return forecasttools.location_lookup(
        location_vector=list(set(obs_locs + fc_locs)), location_format="abbr"
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


def get_initial_window_range(
    data_to_plot: pl.DataFrame,
    forecast_to_plot: pl.DataFrame,
    observed_date_col: str = "date",
    forecast_date_col: str = "target_end_date",
    extra_weeks: int = 6,
) -> tuple[datetime.datetime, datetime.datetime]:
    """
    Compute an initial x-axis window for plotting of
    forecasts.

    Parameters
    ----------
    data_to_plot : pl.DataFrame
        Hubverse-formatted observed time-series, filtered
        to the requested location, target, and models.
    forecast_to_plot : pl.DataFrame
        Hubverse-formatted forecast table, filtered to the
        requested location, target, and models.
    observed_date_col : str, optional
        Name of the date column in `data_to_plot`
        Defaults to "date".
    forecast_date_col : str, optional
        Name of the date column in `forecast_to_plot`.
        Defaults to "target_end_date"`.
    extra_weeks : int, optional
        How many weeks before the first forecast to
        include. Defaults to 6.

    Returns
    -------
    tuple[datetime.datetime, datetime.datetime]
        A 2-tuple `(start, end)` giving the initial
        plotting window for forecast viewing.
    """
    first_obs_date = data_to_plot.get_column(observed_date_col).min()
    first_fc_date = forecast_to_plot.get_column(forecast_date_col).min()
    last_fc_date = forecast_to_plot.get_column(forecast_date_col).max()
    last_obs_date = data_to_plot.get_column(observed_date_col).max()
    if first_fc_date is None:
        start_date = first_obs_date
    else:
        candidate_start_date = first_fc_date - datetime.timedelta(
            weeks=extra_weeks
        )
        start_date = (
            max(first_obs_date, candidate_start_date)
            if first_obs_date is not None
            else candidate_start_date
        )
    end_date = last_fc_date if last_fc_date is not None else last_obs_date
    return (start_date, end_date)


def build_ci_specs_from_levels(
    levels: list[tuple[str, str, str]],
) -> dict[str, dict[str, str]]:
    """
    Build CI_SPECS dict from a static list of levels.

    Parameters
    ----------
    levels : list of tuples
        Each tuple is (label, low_quantile, high_quantile)

    Returns
    -------
    dict[str, dict[str, str]]
        CI_SPECS dict.
    """
    palette = colorbrewer.Blues.get(
        len(levels), colorbrewer.Blues[max(colorbrewer.Blues)]
    )
    colors = [to_hex([r / 255, g / 255, b / 255]) for r, g, b in palette]

    return {
        label: {"low": low_q, "high": low_q, "color": color}
        for (label, low_q, low_q), color in zip(levels, colors, strict=False)
    }


def build_ci_specs_from_df(df_wide: pl.DataFrame) -> dict[str, dict[str, str]]:
    """
    Automatically constructs a CI_SPECS-style dict for
    altair legend creation by finding quantile columns in
    a wide forecast table.

    Parameters
    ----------
    df_wide : pl.DataFrame
        A Polars DataFrame of forecasts, with quantile
        forecast columns.

    Returns
    -------
    dict[str, dict[str, str]]
        Mapping from CI label (e.g. "95% CI") to its
        bounds and color.
    """

    # use regex to find quantile columns:
    # 0 : the col must start with a literal zero
    # \. : matches a literal period; the backslash escapes
    # \d+ : one or more digits (0–9)
    # re.fullmatch : ensures the entire string matches
    # (no prefixes/suffixes)
    quantile_cols = [
        col
        for col in df_wide.columns
        if re.fullmatch(r"0\.\d+", col) and 0 < float(col) < 1
    ]
    quantiles = sorted(float(q) for q in quantile_cols)
    pairs = [
        (low_q, high_q)
        for i, low_q in enumerate(quantiles)
        for high_q in quantiles[i + 1 :]
        if abs(low_q + high_q - 1.0) < 1e-6
    ]

    palette = colorbrewer.Blues.get(
        len(pairs), colorbrewer.Blues[max(colorbrewer.Blues)]
    )
    colors = [to_hex([r / 255, g / 255, b / 255]) for r, g, b in palette]

    ci_specs = {}
    for (low_q, high_q), color in zip(pairs, colors, strict=False):
        ci_width = round((high_q - low_q) * 100)
        label = f"{ci_width}% CI"
        ci_specs[label] = {
            "low": f"{low_q:.3f}".rstrip("0").rstrip("."),
            "high": f"{high_q:.3f}".rstrip("0").rstrip("."),
            "color": color,
        }

    return dict(sorted(ci_specs.items(), reverse=True))


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
    selected_target: str,
    color_enc: alt.Color,
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
    selected_target : str
        The target for filtering in the forecast and or
        observed hubverse tables.
    color_enc : alt.Color
        An Altair color encoding used for plotting the
        observations color and legend.
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
    x_enc = alt.X(
        "date:T",
        axis=alt.Axis(title="Date", grid=grid, ticks=True, labels=True),
    )
    y_enc = alt.Y(
        "observation:Q",
        axis=alt.Axis(
            title=f"{selected_target}",
            grid=grid,
            ticks=True,
            labels=True,
            orient="left",
        ),
        scale=alt.Scale(type=scale),
    )
    obs_layer = (
        alt.Chart(observed_data_table, width=PLOT_WIDTH)
        .transform_calculate(legend_label="'Observations'")
        .mark_point(
            filled=True,
            size=MARKER_SIZE,
        )
        .encode(
            x=x_enc,
            y=y_enc,
            color=color_enc,
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("observation:Q", title="Value"),
            ],
        )
    )
    return obs_layer


def quantile_forecast_chart(
    forecast_table: pl.DataFrame,
    selected_target: str,
    color_enc: alt.Color,
    scale: ScaleType = "log",
    grid: bool = True,
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
    selected_target : str
        The target for filtering in the forecast and or
        observed hubverse tables.
    color_enc : alt.Color
        An Altair color encoding used for plotting the
        quantile bands color and legend.
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
    df_wide = (
        forecast_table.filter(pl.col("output_type") == "quantile")
        .pivot(
            on="output_type_id",
            index=cs.exclude("output_type_id", "value"),
            values="value",
        )
        .rename({"0.5": "median"})
    )
    ci_specs = build_ci_specs_from_df(df_wide)
    x_enc = alt.X("target_end_date:T", title="Date", axis=alt.Axis(grid=grid))
    y_enc = alt.Y(
        "median:Q",
        axis=alt.Axis(grid=grid),
        scale=alt.Scale(type=scale),
    )
    base = (
        alt.Chart(df_wide, width=PLOT_WIDTH)
        .transform_calculate(
            date="toDate(datum.target_end_date)",
            data_type="'Forecast'",
        )
        .encode(
            x=x_enc,
            y=y_enc,
        )
    )

    def band(low: str, high: str, label: str) -> alt.Chart:
        """
        Builds an errorband layer for a quantile.

        Parameters
        ----------
        low : str
            Lower-bound column name in the wide forecast
            table (e.g., "0.25").
        high : str
            Upper-bound column name in the wide forecast
            table (e.g., "0.975").
        label : str
            The label in the legend for the confidence
            interval (e.g. "97.5% CI").

        Returns
        -------
        alt.Chart
            An Altair layer with the filled band from
            ``low`` to ``high``, with step interpolation.
        """
        return (
            base.transform_calculate(legend_label=f"'{label}'")
            .mark_errorband(interpolate="step")
            .encode(
                y=alt.Y(f"{low}:Q", title=f"{selected_target}"),
                y2=f"{high}:Q",
                color=color_enc,
                opacity=alt.value(1.0),
            )
        )

    bands = [
        band(spec["low"], spec["high"], label)
        for label, spec in ci_specs.items()
        if spec["low"] in df_wide.columns and spec["high"] in df_wide.columns
    ]

    median = base.mark_line(
        strokeWidth=STROKE_WIDTH, interpolate="step", color="navy"
    )

    return alt.layer(*bands, median)


def filter_for_plotting(
    observed_data_table: pl.DataFrame,
    forecast_table: pl.DataFrame,
    selected_models: list[str],
    selected_target: str | None,
    selected_ref_date: datetime.date | None,
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
        "observation": pl.Float64,
        "location": pl.Utf8,
        "as_of": pl.Date,
        "target": pl.Utf8,
        "loc_abbr": pl.Utf8,
    }
    if not observed_data_file:
        return pl.DataFrame(schema=observed_schema)
    table = load_hubverse_table(observed_data_file).select(
        observed_schema.keys()
    )
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
        "horizon": pl.Float64,
        "target_end_date": pl.Date,
        "location": pl.Utf8,
        "output_type": pl.Utf8,
        "output_type_id": pl.Utf8,
        "value": pl.Float64,
        "loc_abbr": pl.Utf8,
    }
    if not forecast_file:
        return pl.DataFrame(schema=forecast_schema)
    table = (
        load_hubverse_table(forecast_file)
        .select(forecast_schema.keys())
        .cast({"horizon": pl.Float64, "output_type_id": pl.Utf8})
    )
    validate_schema(table, forecast_schema, "Forecast Data")
    return table
