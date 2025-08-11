"""
A streamlit application that loads hubverse formatted
tables and plots model forecasts for the user to compare
and annotate models.

To run: uv run streamlit run ./hubverse_annotator/app.py
"""

import streamlit as st
from ui import (
    forecast_annotation_ui,
    load_data_ui,
    location_selection_ui,
    model_selection_ui,
    plotting_ui,
    reference_date_selection_ui,
    target_selection_ui,
)
from utils import filter_for_plotting


def main() -> None:
    # record session start time
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
        selected_ref_date = reference_date_selection_ui(forecast_table)
        selected_models = model_selection_ui(forecast_table)
        loc_abbr = location_selection_ui(observed_data_table, forecast_table)
        selected_target = target_selection_ui(
            observed_data_table,
            forecast_table,
            loc_abbr,
            selected_models,
        )
        scale = "log" if st.checkbox("Log-scale", value=True) else "linear"
        grid = st.checkbox("Gridlines", value=True)
        ref_date_line = st.checkbox("Reference Date Line", value=True)
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
        ref_date_line=ref_date_line,
    )


if __name__ == "__main__":
    main()
