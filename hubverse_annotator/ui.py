"""
A user-inference (UI) components module. These streamlit
components for the Hubverse Annotator application include:
file upload widgets, location & date selectors, model &
target selectors, and annotation & export controls.

To run: uv run streamlit run ./hubverse_annotator/app.py
"""

import datetime
import json
import pathlib
from functools import reduce

import altair as alt
import polars as pl
import streamlit as st
from streamlit_shortcuts import add_shortcuts
from utils import (
    get_available_locations,
    get_initial_window_range,
    get_reference_dates,
    is_empty_chart,
    load_forecast_data,
    load_observed_data,
    quantile_forecast_chart,
    target_data_chart,
)

Y_LABEL_FONT_SIZE = 15
CHART_TITLE_FONT_SIZE = 18
REF_DATE_STROKE_WIDTH = 2.5
REF_DATE_STROKE_DASH = [6, 6]
MARKER_SIZE = 65
ROOT = pathlib.Path(__file__).resolve().parent.parent


def annotation_export_ui() -> None:
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
    selected_ref_date: datetime.date | None,
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
    output_dir = ROOT / "output"
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
    annotation_export_ui()


def model_selection_ui(
    forecast_table: pl.DataFrame,
) -> list[str]:
    """
    Renders a Streamlit multiselect widget for choosing
    models, with "All" and "None" buttons. Defaults to all
    models being selected.

    Parameters
    ----------
    forecast_table : pl.DataFrame
        Hubverse-formatted forecasts (must include
        "loc_abbr" and "model_id" columns; possibly empty).

    Returns
    -------
    list[str]
        The list of currently selected model_ids.
    """

    models = forecast_table.get_column("model_id").unique().sort().to_list()
    if not models:
        st.info("Upload forecasts for this location to pick models.")
        return []
    if "model_selection" not in st.session_state:
        st.session_state.model_selection = models.copy()

    def _select_all():
        st.session_state.model_selection = models.copy()

    def _select_none():
        st.session_state.model_selection = []

    all_button, none_button = st.columns(2)
    all_button.button("All", on_click=_select_all)
    none_button.button("None", on_click=_select_none)
    selected_models = st.multiselect(
        "Model(s)",
        options=models,
        key="model_selection",
    )

    return selected_models


def target_selection_ui(
    observed_data_table: pl.DataFrame,
    forecast_table: pl.DataFrame,
    loc_abbr: str,
    selected_models: list[str],
) -> str | None:
    """
    Renders a Streamlit selectbox for choosing a
    target. Always defaults to the first alphabetical
    target whenever the set of models changes.

    Parameters
    ----------
    observed_data_table : pl.DataFrame
        Hubverse formatted observed data table (must
        include "loc_abbr", "target"; possibly empty).
    forecast_table : pl.DataFrame
        Hubverse formatted forecast table (must include
        "loc_abbr", "model_id", "target"; possibly empty).
    loc_abbr : str
        The selection location, typically a US
        jurisdiction.
    selected_models : list[str]
        Models currently selected.

    Returns
    -------
    str | None
        The selected target, or None if there are no
        targets.
    """
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
    if "targets" not in st.session_state:
        st.session_state.targets = sorted(
            set(forecast_targets + observed_data_targets)
        )

    def _go_to_prev_target():
        st.session_state.current_target_id -= 1

    def _go_to_next_target():
        st.session_state.current_target_id += 1

    target_col, prev_col, next_col = st.columns(
        [6, 1, 1], vertical_alignment="bottom"
    )
    with target_col:
        selected_target = st.selectbox(
            "Target",
            options=list(range(len(st.session_state.targets))),
            key="current_target_id",
            format_func=lambda i: st.session_state.targets[i],
        )
    with prev_col:
        st.button(
            "⏮️",
            disabled=(st.session_state.current_target_id == 0),
            on_click=_go_to_prev_target,
            key="prev_target_button",
        )
    with next_col:
        st.button(
            "⏭️",
            disabled=(
                st.session_state.current_target_id
                == len(st.session_state.targets) - 1
            ),
            on_click=_go_to_next_target,
            key="next_target_button",
        )
    add_shortcuts(prev_target_button="n", next_target_button="m")
    target_id = st.session_state.current_target_id
    selected_target = st.session_state.targets[target_id]
    return selected_target


def location_selection_ui(
    observed_data_table: pl.DataFrame,
    forecast_table: pl.DataFrame,
) -> str:
    """
    Streamlit widget for the selection of a location (two
    letter abbreviation for US jurisdiction).

    Parameters
    ----------
    observed_data_table : pl.DataFrame
        A hubverse table of loaded data (possibly empty).
    forecast_table : pl.DataFrame
        The hubverse formatted table of forecasted ED
        visits and or hospital admissions (possibly empty).

    Returns
    -------
    str
        The selected two-letter location abbreviation.

    """
    loc_lookup = get_available_locations(observed_data_table, forecast_table)
    locations = loc_lookup.get_column("long_name").sort().to_list()
    st.session_state.locations = locations

    def _go_to_prev_loc():
        st.session_state.current_loc_id -= 1

    def _go_to_next_loc():
        st.session_state.current_loc_id += 1

    location_col, prev_col, next_col = st.columns(
        [6, 1, 1], vertical_alignment="bottom"
    )
    with location_col:
        st.selectbox(
            "Location",
            options=list(range(len(st.session_state.locations))),
            key="current_loc_id",
            format_func=lambda i: st.session_state.locations[i],
        )
    with prev_col:
        st.button(
            "⏮️",
            disabled=(st.session_state.current_loc_id == 0),
            on_click=_go_to_prev_loc,
            key="prev_loc_button",
        )
    with next_col:
        st.button(
            "⏭️",
            disabled=(
                st.session_state.current_loc_id
                == len(st.session_state.locations) - 1
            ),
            on_click=_go_to_next_loc,
            key="next_loc_button",
        )
    add_shortcuts(prev_loc_button="j", next_loc_button="k")
    selected_location = locations[st.session_state.current_loc_id]
    loc_abbr = (
        loc_lookup.filter(pl.col("long_name") == selected_location)
        .get_column("short_name")
        .item()
    )
    return loc_abbr


def reference_date_selection_ui(
    forecast_table: pl.DataFrame,
) -> datetime.date | None:
    """
    Streamlit widget for the selection of forecast
    reference date.

    Parameters
    ----------
    forecast_table : pl.DataFrame
        The hubverse formatted table of forecasted ED
        visits and or hospital admissions (possibly empty).

    Returns
    -------
    datetime.date | None
        The selected reference date, or None if no dates
        are available.
    """
    ref_dates = sorted(get_reference_dates(forecast_table))
    if not ref_dates:
        st.info("Upload a forecast file to select a reference date.")
        return None

    if "ref_dates" not in st.session_state:
        st.session_state.ref_dates = ref_dates.copy()
        st.session_state.current_ref_date_id = len(ref_dates) - 1

    def _go_to_prev_ref_date():
        st.session_state.current_ref_date_id -= 1

    def _go_to_next_ref_date():
        st.session_state.current_ref_date_id += 1

    ref_date_col, prev_col, next_col = st.columns(
        [6, 1, 1], vertical_alignment="bottom"
    )
    num_ref_dates = len(st.session_state.ref_dates)
    with ref_date_col:
        st.selectbox(
            "Reference Date",
            options=list(range(num_ref_dates)),
            key="current_ref_date_id",
            format_func=lambda i: st.session_state.ref_dates[i].strftime(
                "%Y-%m-%d"
            ),
        )
    with prev_col:
        st.button(
            "⏮️",
            disabled=(st.session_state.current_ref_date_id == 0),
            on_click=_go_to_prev_ref_date,
            key="prev_ref_date_button",
        )
    with next_col:
        st.button(
            "⏭️",
            disabled=(
                st.session_state.current_ref_date_id
                == len(st.session_state.ref_dates) - 1
            ),
            on_click=_go_to_next_ref_date,
            key="next_ref_date_button",
        )
    add_shortcuts(prev_ref_date_button=",", next_ref_date_button=".")
    ref_date_id = st.session_state.current_ref_date_id
    selected_ref_date = st.session_state.ref_dates[ref_date_id]
    return selected_ref_date


def plotting_ui(
    data_to_plot: pl.DataFrame,
    forecasts_to_plot: pl.DataFrame,
    loc_abbr: str,
    selected_target: str | None,
    selected_ref_date: datetime.date | None,
    scale: bool = True,
    show_grid: bool = True,
    show_ref_date_line: bool = True,
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
    scale : {"log", "linear"}
        Y-axis scale type.
    show_grid : bool
        Whether to show gridlines on both axes.
    show_ref_date_line : bool
        If True, draw a vertical dashed black line at the
        selected reference date.
    """
    if "model_id" not in data_to_plot.columns:
        data_to_plot = data_to_plot.with_columns(
            pl.lit("Observations").alias("model_id")
        )
    # empty streamlit object (DeltaGenerator) needed for
    # plots to reload successfully with new data.
    base_chart = st.empty()

    has_obs = not data_to_plot.is_empty()
    has_fc = not forecasts_to_plot.is_empty()

    legend_labels = []
    color_range = []
    opacity_range = []
    opacity_labels = []

    if has_obs:
        legend_labels.append("Observations")
        color_range.append("limegreen")

    if has_fc:
        legend_labels += ["97.5% CI", "80% CI", "50% CI"]
        color_range += ["steelblue", "steelblue", "steelblue", "steelblue"]
        opacity_labels += ["97.5% CI", "80% CI", "50% CI", "medians"]
        opacity_range += [0.10, 0.20, 0.30, 1.0]

    color_enc = alt.Color(
        "legend_label:N",
        title=None,
        scale=alt.Scale(domain=legend_labels, range=color_range),
    )

    opacity_enc = alt.condition(
        "datum.legend_label != 'Observations'",
        alt.Opacity(
            "legend_label:N",
            scale=alt.Scale(domain=opacity_labels, range=opacity_range),
            legend=None,
        ),
        alt.value(1),
    )

    observed_layer = target_data_chart(
        data_to_plot,
        selected_target,
        color_enc=color_enc,
        scale=scale,
        grid=show_grid,
    )
    forecast_layer = quantile_forecast_chart(
        forecasts_to_plot,
        selected_target,
        color_enc=color_enc,
        opacity_enc=opacity_enc,
        scale=scale,
        grid=show_grid,
    )
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
    if show_ref_date_line and selected_ref_date is not None:
        rule_layer = alt.Chart(
            alt.Data(values=[{"date": str(selected_ref_date)}])
        ).mark_rule(
            color="black",
            strokeDash=REF_DATE_STROKE_DASH,
            strokeWidth=REF_DATE_STROKE_WIDTH,
        )
        layer = layer + rule_layer
    domain = get_initial_window_range(data_to_plot, forecasts_to_plot)
    x_enc = alt.X(
        "date:T",
        scale=alt.Scale(domain=domain),
        axis=alt.Axis(format="%b %d", grid=show_grid),
        title="Date",
    )
    chart = (
        layer.encode(x=x_enc)
        .facet(
            facet=alt.Facet(
                "model_id:N",
                header=alt.Header(
                    labelOrient="top",
                    labelColor="black",
                    labelFontSize=Y_LABEL_FONT_SIZE,
                    title=None,
                ),
            ),
            columns=1,
        )
        .properties(
            title=alt.TitleParams(
                text=f"({loc_abbr}) Reference Date: {selected_ref_date}",
                fontSize=CHART_TITLE_FONT_SIZE,
                fontWeight="bold",
                anchor="middle",
            )
        )
        .interactive()
        .resolve_scale(y="independent")
        .resolve_axis(x="independent")
        .configure_legend(
            orient="top",
            direction="horizontal",
            symbolType="circle",
            symbolSize=MARKER_SIZE,
            symbolStrokeWidth=0,
            titleAnchor="middle",
        )
    )
    base_chart.altair_chart(
        chart,
        use_container_width=False,
        key=f"forecast_{loc_abbr}_{selected_target}",
    )


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
